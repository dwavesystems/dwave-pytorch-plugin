# Copyright 2026 D-Wave
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests of maximum pseudolikelihood estimation.

The negative log-pseudolikelihood is checked against the conditional probabilities of single
spins obtained by enumeration; the temperature fit against dwave-system's estimator, an analytic
single-spin case and the planted temperature of exactly sampled data; the joint fit against
dwave-system's general estimator with one binary quadratic model per bias, and against the
planted biases."""
import itertools
import math
import unittest
import unittest.mock

import dimod
import torch
from dwave.system.temperatures import maximum_pseudolikelihood as mpl
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple

from dwave.plugins.torch import pseudolikelihood
from dwave.plugins.torch.graph import GraphIndex
from dwave.plugins.torch.pseudolikelihood import (
    InverseTemperatureFit,
    PseudolikelihoodFit,
    fit_inverse_temperature,
    fit_pseudolikelihood,
)
from dwave.plugins.torch.pseudolikelihood import negative_log_pseudolikelihood as nlpl
from dwave.plugins.torch.utils import to_bqm
from tests.helper_functions import randspins

# A triangle with a dangling node, in the deliberately scrambled node order of the model tests
GRAPH = GraphIndex("dbac", [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c")])
STATES = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=4)), dtype=torch.float64)
SINGLE = GraphIndex([0], [])  # one spin, no edges


def random_biases(graph: GraphIndex, scale: float = 0.5, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    linear = scale * torch.randn(graph.n_nodes, generator=generator, dtype=torch.float64)
    quadratic = scale * torch.randn(graph.n_edges, generator=generator, dtype=torch.float64)
    return linear, quadratic


def boltzmann_samples(linear, quadratic, num_samples: int, beta: float = 1.0, seed: int = 0):
    """Exact samples of the model on ``GRAPH``, by enumeration of its 16 configurations."""
    probabilities = torch.softmax(-beta * GRAPH.energy(STATES, linear, quadratic), 0)
    generator = torch.Generator().manual_seed(seed)
    rows = torch.multinomial(probabilities, num_samples, replacement=True, generator=generator)
    return STATES[rows]


def enumerated_nlpl(graph, linear, quadratic, spins):
    """The negative log-pseudolikelihood of configurations from the conditional probability of
    each spin: the Boltzmann weight of the configuration, normalized against the configuration
    with that spin flipped."""
    energies = graph.energy(spins, linear, quadratic)
    total = torch.zeros_like(energies)
    for k in range(graph.n_nodes):
        flipped = spins.clone()
        flipped[..., k] = -flipped[..., k]
        flipped_energies = graph.energy(flipped, linear, quadratic)
        total = total - (-energies - torch.logaddexp(-energies, -flipped_energies))
    return total


def ocean_biases(graph, spins):
    """dwave-system's maximum pseudolikelihood estimate of every bias, from one binary quadratic
    model per bias; it fits the exponent of ``exp(x H)``, so the biases are ``-x``."""
    bqms = [dimod.BQM.from_ising({node: 1.0}, {}) for node in graph.nodes]
    bqms += [dimod.BQM.from_ising({}, {edge: 1.0}) for edge in graph.edges]
    for bqm in bqms:
        for node in graph.nodes:
            bqm.add_variable(node, 0.0)
    x, _ = mpl(bqms=bqms, sampleset=(spins.numpy(), list(graph.nodes)))
    x = -torch.tensor(x, dtype=torch.float64)
    return x[:graph.n_nodes], x[graph.n_nodes:]


def ocean_beta(graph, linear, quadratic, spins) -> float:
    """dwave-system's maximum pseudolikelihood estimate of the inverse temperature."""
    bqm = to_bqm(graph, linear, quadratic)
    return 1.0 / mple(bqm, (spins.numpy(), list(graph.nodes)))[0]


class TestNegativeLogPseudolikelihood(unittest.TestCase):
    def test_matches_enumerated_conditionals(self):
        linear, quadratic = random_biases(GRAPH, seed=1)
        spins = randspins(7, 4, seed=2).double()
        expected = enumerated_nlpl(GRAPH, linear, quadratic, spins)
        torch.testing.assert_close(nlpl(GRAPH, linear, quadratic, spins, reduction="none"), expected)
        torch.testing.assert_close(nlpl(GRAPH, linear, quadratic, spins), expected.mean())
        torch.testing.assert_close(nlpl(GRAPH, linear, quadratic, spins, reduction="sum"), expected.sum())
        with self.subTest("The value is in the dtype of the biases, whatever the spins' dtype"):
            value = nlpl(GRAPH, linear.float(), quadratic.float(), spins.long())
            self.assertEqual(torch.float32, value.dtype)
            torch.testing.assert_close(value, expected.mean().float())

    def test_batched_models(self):
        linear = torch.stack([random_biases(GRAPH, seed=s)[0] for s in (1, 2)])
        quadratic = torch.stack([random_biases(GRAPH, seed=s)[1] for s in (1, 2)])
        spins = randspins(2, 5, 4, seed=3).double()
        values = nlpl(GRAPH, linear, quadratic, spins, reduction="none")
        self.assertEqual((2, 5), tuple(values.shape))
        for b in range(2):
            expected = enumerated_nlpl(GRAPH, linear[b], quadratic[b], spins[b])
            torch.testing.assert_close(values[b], expected)
        self.assertEqual((2,), tuple(nlpl(GRAPH, linear, quadratic, spins).shape))
        with self.subTest("Unbatched biases apply to every leading index of the spins"):
            values = nlpl(GRAPH, linear[0], quadratic[0], spins)
            self.assertEqual((2,), tuple(values.shape))
            expected = enumerated_nlpl(GRAPH, linear[0], quadratic[0], spins[1]).mean()
            torch.testing.assert_close(values[1], expected)

    def test_gradients(self):
        # The objective is differentiable in the biases on both evaluation paths
        linear, quadratic = random_biases(GRAPH, seed=4)
        spins = randspins(6, 4, seed=5).double()
        for dense in (True, False):
            with self.subTest(dense=dense), unittest.mock.patch.object(
                pseudolikelihood, "_use_dense_fields", return_value=dense
            ):
                biases = linear.clone().requires_grad_(), quadratic.clone().requires_grad_()
                self.assertTrue(
                    torch.autograd.gradcheck(lambda h, J: nlpl(GRAPH, h, J, spins), biases)
                )
        with self.subTest("The scatter path agrees with the dense path, batched too"):
            batched = torch.stack([linear, 2 * linear]), torch.stack([quadratic, -quadratic])
            batched_spins = randspins(2, 3, 4, seed=6).double()
            expected = nlpl(GRAPH, *batched, batched_spins, reduction="none")
            with unittest.mock.patch.object(pseudolikelihood, "_use_dense_fields", return_value=False):
                torch.testing.assert_close(
                    nlpl(GRAPH, *batched, batched_spins, reduction="none"), expected
                )

    def test_validation(self):
        linear, quadratic = random_biases(GRAPH)
        spins = randspins(3, 4).double()
        with self.assertRaisesRegex(ValueError, "reduction"):
            nlpl(GRAPH, linear, quadratic, spins, reduction="max")
        with self.assertRaisesRegex(ValueError, "one bias per node"):
            nlpl(GRAPH, linear[:3], quadratic, spins)
        with self.assertRaisesRegex(ValueError, "one bias per edge"):
            nlpl(GRAPH, linear, quadratic[:3], spins)
        with self.assertRaisesRegex(ValueError, "same batch of models"):
            nlpl(GRAPH, linear.expand(2, 4), quadratic, spins)
        with self.assertRaisesRegex(ValueError, r"spins must have shape \(\.\.\., M, 4\)"):
            nlpl(GRAPH, linear, quadratic, spins[:, :3])
        with self.assertRaisesRegex(ValueError, r"spins must have shape \(\.\.\., M, 4\)"):
            nlpl(GRAPH, linear, quadratic, spins[0])
        with self.assertRaisesRegex(ValueError, "batched like the biases"):
            nlpl(GRAPH, linear.expand(2, 4), quadratic.expand(2, 4), randspins(3, 5, 4).double())
        with self.assertRaisesRegex(ValueError, "At least one sample"):
            nlpl(GRAPH, linear, quadratic, spins[:0])
        with self.assertRaisesRegex(ValueError, "±1 values"):
            nlpl(GRAPH, linear, quadratic, spins * 0.5)
        with self.assertRaisesRegex(ValueError, "±1 values"):
            nlpl(GRAPH, linear, quadratic, spins.clone().fill_(torch.nan))


class TestFitInverseTemperature(unittest.TestCase):
    def test_matches_dwave_system(self):
        for seed in range(3):
            with self.subTest(seed=seed):
                linear, quadratic = random_biases(GRAPH, seed=seed)
                spins = randspins(50, 4, seed=10 + seed).double()
                fit = fit_inverse_temperature(GRAPH, linear, quadratic, spins)
                self.assertIsInstance(fit, InverseTemperatureFit)
                self.assertEqual((), tuple(fit.beta.shape))
                self.assertEqual(torch.float64, fit.beta.dtype)
                self.assertTrue(bool(fit.converged))
                expected = ocean_beta(GRAPH, linear, quadratic, spins)
                self.assertAlmostEqual(expected, float(fit.beta), places=10)
                self.assertTrue(0 < float(fit.beta_stderr) < math.inf)
        with self.subTest("float32 biases and integer spins give the same estimate"):
            fit32 = fit_inverse_temperature(GRAPH, linear.float(), quadratic.float(), spins.long())
            self.assertEqual(torch.float64, fit32.beta.dtype)
            torch.testing.assert_close(fit32.beta, fit.beta, rtol=1e-5, atol=1e-6)

    def test_single_spin_analytic(self):
        # One spin under the field 0.5, observed down three times and up once: the derivative of
        # the NLPL vanishes where sigmoid(beta) = 3 sigmoid(-beta), i.e. at beta = log(3). The
        # scores are then -1/4 (down) and 3/4 (up), of variance 1/4, and the curvature is
        # sigmoid(log 3) sigmoid(-log 3) = 3/16, so the standard error is (1/4) / (3/16) = 4/3.
        spins = torch.tensor([[-1.0], [-1.0], [-1.0], [1.0]])
        fit = fit_inverse_temperature(SINGLE, torch.tensor([0.5]), torch.zeros(0), spins)
        torch.testing.assert_close(fit.beta, torch.tensor(math.log(3.0), dtype=torch.float64))
        torch.testing.assert_close(fit.beta_stderr, torch.tensor(4 / 3, dtype=torch.float64))
        self.assertTrue(bool(fit.converged))

    def test_recovers_planted_temperature(self):
        linear, quadratic = random_biases(GRAPH, scale=0.6, seed=7)
        spins = boltzmann_samples(linear, quadratic, 20000, beta=1.7, seed=8)
        fit = fit_inverse_temperature(GRAPH, linear, quadratic, spins)
        self.assertLess(float(fit.beta_stderr), 0.1)
        self.assertLess(abs(float(fit.beta) - 1.7), 4 * float(fit.beta_stderr))

    def test_batched_models(self):
        biases = [random_biases(GRAPH, seed=s) for s in (1, 2, 3)]
        linear = torch.stack([h for h, _ in biases])
        quadratic = torch.stack([J for _, J in biases])
        spins = randspins(3, 40, 4, seed=9).double()
        fit = fit_inverse_temperature(GRAPH, linear, quadratic, spins)
        for field in (fit.beta, fit.beta_stderr, fit.converged):
            self.assertEqual((3,), tuple(field.shape))
        for b in range(3):
            single = fit_inverse_temperature(GRAPH, linear[b], quadratic[b], spins[b])
            torch.testing.assert_close(fit.beta[b], single.beta)
            torch.testing.assert_close(fit.beta_stderr[b], single.beta_stderr)
            self.assertTrue(bool(fit.converged[b]))
        with self.subTest("Any batch shape"):
            fit = fit_inverse_temperature(GRAPH, linear[:, None], quadratic[:, None], spins[:, None])
            self.assertEqual((3, 1), tuple(fit.beta.shape))
        with self.subTest("Unbatched biases give one estimate per leading index of the spins"):
            fit = fit_inverse_temperature(GRAPH, linear[0], quadratic[0], spins)
            self.assertEqual((3,), tuple(fit.beta.shape))
            single = fit_inverse_temperature(GRAPH, linear[0], quadratic[0], spins[2])
            torch.testing.assert_close(fit.beta[2], single.beta)

    def test_limits(self):
        # Under the field 0.5, a sample of down spins has no excitation: the zero-temperature
        # limit. Up spins are all excited, the limit of the opposite sign; and no field at all
        # leaves beta unidentifiable.
        linear, quadratic = torch.tensor([0.5]), torch.zeros(0)
        fit = fit_inverse_temperature(SINGLE, linear, quadratic, -torch.ones(3, 1))
        self.assertEqual(math.inf, float(fit.beta))
        self.assertTrue(math.isnan(float(fit.beta_stderr)))
        self.assertTrue(bool(fit.converged))
        fit = fit_inverse_temperature(SINGLE, linear, quadratic, torch.ones(3, 1))
        self.assertEqual(-math.inf, float(fit.beta))
        with self.assertRaisesRegex(ValueError, "unidentifiable"):
            fit_inverse_temperature(SINGLE, torch.zeros(1), quadratic, torch.ones(3, 1))
        with self.subTest("Limits and finite estimates coexist in a batch"):
            spins = torch.tensor([[[-1.0], [-1.0]], [[-1.0], [1.0]]])
            fit = fit_inverse_temperature(SINGLE, linear.expand(2, 1), quadratic.expand(2, 0), spins)
            torch.testing.assert_close(fit.beta, torch.tensor([math.inf, 0.0], dtype=torch.float64))
            # The scores -1/2 and 1/2 at beta = 0 have variance 1/2 and the curvature is 1/4
            torch.testing.assert_close(
                fit.beta_stderr, torch.tensor([math.nan, 2.0], dtype=torch.float64), equal_nan=True
            )
            self.assertEqual([True, True], fit.converged.tolist())

    def test_options(self):
        linear, quadratic = random_biases(GRAPH, seed=11)
        spins = randspins(30, 4, seed=12).double()
        fit = fit_inverse_temperature(GRAPH, linear, quadratic, spins)
        with self.subTest("Chunking does not change the estimate"):
            for sample_chunk in (7, None):
                chunked = fit_inverse_temperature(GRAPH, linear, quadratic, spins, sample_chunk=sample_chunk)
                torch.testing.assert_close(chunked.beta, fit.beta)
                torch.testing.assert_close(chunked.beta_stderr, fit.beta_stderr)
        with self.subTest("A single Newton step is the moment estimate, and is not converged"):
            energies = 2 * spins * GRAPH.effective_field(spins, linear=linear, quadratic=quadratic)
            moment = -2 * energies.sum() / energies.square().sum()
            one_step = fit_inverse_temperature(GRAPH, linear, quadratic, spins, max_iter=1)
            torch.testing.assert_close(one_step.beta, moment)
            self.assertFalse(bool(one_step.converged))
        with self.subTest("A single configuration has no standard error"):
            single = fit_inverse_temperature(GRAPH, linear, quadratic, spins[:1])
            self.assertTrue(math.isnan(float(single.beta_stderr)))
        for kwargs in (dict(max_iter=0), dict(tolerance=0.0), dict(tolerance=math.inf), dict(sample_chunk=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_inverse_temperature(GRAPH, linear, quadratic, spins, **kwargs)


class TestFitPseudolikelihood(unittest.TestCase):
    def setUp(self):
        self.linear, self.quadratic = random_biases(GRAPH, scale=0.5, seed=20)
        self.spins = boltzmann_samples(self.linear, self.quadratic, 3000, seed=21)

    def test_matches_dwave_system(self):
        fit = fit_pseudolikelihood(GRAPH, self.spins)
        self.assertIsInstance(fit, PseudolikelihoodFit)
        expected_linear, expected_quadratic = ocean_biases(GRAPH, self.spins)
        torch.testing.assert_close(fit.linear, expected_linear, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(fit.quadratic, expected_quadratic, rtol=1e-5, atol=1e-6)
        with self.subTest("Diagnostics"):
            self.assertTrue(fit.converged)
            self.assertLessEqual(fit.grad_inf_norm, 1e-7)
            self.assertGreater(fit.n_iterations, 0)
            self.assertGreaterEqual(fit.n_function_evals, fit.n_iterations)
            value = nlpl(GRAPH, fit.linear, fit.quadratic, self.spins)
            self.assertAlmostEqual(float(value), fit.nlpl, places=12)
            self.assertEqual(torch.float64, fit.linear.dtype)
            self.assertEqual(torch.float64, fit.quadratic.dtype)
        with self.subTest("The estimate is stationary along its own ray"):
            # Up to the precision of the joint fit: a gradient of 1e-7 moves the root by ~1e-7
            own = fit_inverse_temperature(GRAPH, fit.linear, fit.quadratic, self.spins)
            self.assertAlmostEqual(1.0, float(own.beta), places=6)
            tight = fit_pseudolikelihood(GRAPH, self.spins, tolerance_grad=1e-10)
            own = fit_inverse_temperature(GRAPH, tight.linear, tight.quadratic, self.spins)
            self.assertAlmostEqual(1.0, float(own.beta), places=7)

    def test_recovers_planted_biases(self):
        spins = boltzmann_samples(self.linear, self.quadratic, 50000, seed=22)
        fit = fit_pseudolikelihood(GRAPH, spins)
        torch.testing.assert_close(fit.linear, self.linear, rtol=0, atol=0.05)
        torch.testing.assert_close(fit.quadratic, self.quadratic, rtol=0, atol=0.05)
        with self.subTest("Data drawn at inverse temperature 2 are at inverse temperature 2 under "
                          "the planted biases, and at 1 under their own joint fit"):
            cold = boltzmann_samples(self.linear, self.quadratic, 50000, beta=2.0, seed=23)
            against_truth = fit_inverse_temperature(GRAPH, self.linear, self.quadratic, cold)
            self.assertLess(abs(float(against_truth.beta) - 2.0), 4 * float(against_truth.beta_stderr))
            fit = fit_pseudolikelihood(GRAPH, cold)
            self.assertTrue(fit.converged)
            own = fit_inverse_temperature(GRAPH, fit.linear, fit.quadratic, cold)
            self.assertAlmostEqual(1.0, float(own.beta), places=6)

    def test_degenerate_data(self):
        spins = self.spins.clone()
        spins[:, 0] = 1.0  # the spin of node "d" is pinned
        with self.assertRaisesRegex(ValueError, r"degenerate.*nodes \['d'\]"):
            fit_pseudolikelihood(GRAPH, spins)
        with self.subTest("A penalty bounds the estimate"):
            fit = fit_pseudolikelihood(GRAPH, spins, l2=1e-3)
            self.assertTrue(fit.converged)
            self.assertTrue(bool(torch.isfinite(fit.linear).all()))
        with self.subTest("A constant endpoint product is degenerate too"):
            spins = self.spins.clone()
            spins[:, 1] = spins[:, 2]  # nodes "b" and "a" always agree: the edge ("a", "b")
            with self.assertRaisesRegex(ValueError, r"edges \[\('a', 'b'\)\]"):
                fit_pseudolikelihood(GRAPH, spins)

    def test_penalties(self):
        pure = fit_pseudolikelihood(GRAPH, self.spins)
        pure_norm = float(torch.cat([pure.linear, pure.quadratic]).norm())
        for kwargs in (dict(l2=0.05), dict(l1=0.05), dict(l1=0.02, l2=0.02)):
            with self.subTest(**kwargs):
                fit = fit_pseudolikelihood(GRAPH, self.spins, **kwargs)
                self.assertLess(float(torch.cat([fit.linear, fit.quadratic]).norm()), pure_norm)
                # The reported NLPL is the unpenalized one, which the pure fit minimizes
                self.assertGreater(fit.nlpl, pure.nlpl)

    def test_options_and_graphs(self):
        fit = fit_pseudolikelihood(GRAPH, self.spins)
        with self.subTest("Chunking is exact"):
            for sample_chunk in (7, None):
                chunked = fit_pseudolikelihood(GRAPH, self.spins, sample_chunk=sample_chunk)
                torch.testing.assert_close(chunked.linear, fit.linear, rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(chunked.quadratic, fit.quadratic, rtol=1e-5, atol=1e-6)
        with self.subTest("The scatter path gives the same estimate"):
            with unittest.mock.patch.object(pseudolikelihood, "_use_dense_fields", return_value=False):
                sparse = fit_pseudolikelihood(GRAPH, self.spins)
            torch.testing.assert_close(sparse.linear, fit.linear, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(sparse.quadratic, fit.quadratic, rtol=1e-5, atol=1e-6)
        with self.subTest("A graph without edges fits independent spins by their log-odds"):
            graph = GraphIndex("dbac", [])
            fit = fit_pseudolikelihood(graph, self.spins)
            torch.testing.assert_close(fit.linear, -torch.atanh(self.spins.mean(0)), rtol=1e-5, atol=1e-6)
            self.assertEqual((0,), tuple(fit.quadratic.shape))
            self.assertTrue(fit.converged)
        with self.subTest("Invalid options and data"):
            for kwargs in (dict(sample_chunk=0), dict(l1=-1.0), dict(l2=math.inf), dict(l2=math.nan)):
                with self.assertRaises(ValueError):
                    fit_pseudolikelihood(GRAPH, self.spins, **kwargs)
            with self.assertRaisesRegex(ValueError, "single model"):
                fit_pseudolikelihood(GRAPH, self.spins[None])
            with self.assertRaisesRegex(ValueError, "±1"):
                fit_pseudolikelihood(GRAPH, self.spins * 2)
            with self.assertRaisesRegex(ValueError, r"\(\.\.\., M, 4\)"):
                fit_pseudolikelihood(GRAPH, self.spins[:, :3])


if __name__ == "__main__":
    unittest.main()
