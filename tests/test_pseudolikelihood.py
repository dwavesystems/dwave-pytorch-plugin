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

Nothing is drawn at random. The negative log-pseudolikelihood is checked against values worked out
by hand on a fixed model and fixed configurations; the temperature fit against dwave-system's
estimator on fixed configurations, against an analytic single-spin case and against the planted
temperature of a deterministic sample; the joint fit against dwave-system's general estimator with
one binary quadratic model per bias, and against the planted biases of a deterministic sample. The
deterministic sample of a model holds every configuration as often as its Boltzmann probability
says, rounded to a count, so its statistics are those of the model up to the rounding."""
import itertools
import math
import unittest

import dimod
import torch
import torch.nn.functional as F
from dwave.system.temperatures import maximum_pseudolikelihood as mpl
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple

from dwave.plugins.torch.graph import GraphIndex
from dwave.plugins.torch.pseudolikelihood import (
    InverseTemperatureFit,
    PseudolikelihoodFit,
    fit_inverse_temperature,
    fit_pseudolikelihood,
)
from dwave.plugins.torch.pseudolikelihood import negative_log_pseudolikelihood as nlpl
from dwave.plugins.torch.utils import to_bqm

# The triangle a, b, c with the dangling node d in the node order "dbac" (d=0, b=1, a=2, c=3), the
# fixture of the graph tests, with the same biases and configurations
GRAPH = GraphIndex("dbac", [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c")])
STATES = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=4)), dtype=torch.float64)
LINEAR = torch.tensor([1.0, -2.0, 3.0, 0.0], dtype=torch.float64)  # h_d, h_b, h_a, h_c
QUADRATIC = torch.tensor([1.0, 2.0, 3.0, -4.0], dtype=torch.float64)  # J_ab, J_ac, J_ad, J_bc
SPINS = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                      [-1.0, -1.0, -1.0, -1.0],
                      [1.0, -1.0, 1.0, -1.0],
                      [1.0, 1.0, -1.0, -1.0]], dtype=torch.float64)
# The excitation energies 2 s_k theta_k of the four configurations, theta_k = h_k + sum_l J_kl s_l
# being the effective fields (4, -5, 9, -2), (-2, 1, -3, 2), (4, 3, 3, 6) and (-2, 1, 5, -6)
EXCITATIONS = torch.tensor([[8.0, -10.0, 18.0, -4.0],
                            [4.0, -2.0, 6.0, -4.0],
                            [8.0, -6.0, 6.0, -12.0],
                            [-4.0, 2.0, -10.0, 12.0]], dtype=torch.float64)
# The negative log-pseudolikelihood of a configuration is the sum over its spins of the softplus
# of their excitation energies
NLPL = F.softplus(EXCITATIONS).sum(-1)
# A second model with the biases negated and the couplings doubled, and the excitation energies
# of the third and the fourth configuration under it
LINEAR_2 = torch.stack([LINEAR, -LINEAR])
QUADRATIC_2 = torch.stack([QUADRATIC, 2 * QUADRATIC])
EXCITATIONS_2 = torch.tensor([[10.0, -24.0, -6.0, -24.0],
                              [-14.0, 16.0, -2.0, 24.0]], dtype=torch.float64)
NLPL_2 = F.softplus(EXCITATIONS_2).sum(-1)
# Milder planted biases for the fits, whose Boltzmann distribution spreads over every
# configuration, with the energies of the 16 configurations in the order of STATES, all spins
# down first: for that one, -(0.25 - 0.5 + 0.75) + (0.5 - 0.25 + 0.75 - 1) = -0.5
PLANTED_LINEAR = torch.tensor([0.25, -0.5, 0.75, 0.0], dtype=torch.float64)
PLANTED_QUADRATIC = torch.tensor([0.5, -0.25, 0.75, -1.0], dtype=torch.float64)
PLANTED_ENERGIES = torch.tensor([-0.5, 2.0, -1.0, 0.5, -0.5, -2.0, 1.0, -1.5,
                                 -1.5, 1.0, 1.0, 2.5, -1.5, -3.0, 3.0, 0.5], dtype=torch.float64)
# Twelve configurations, with repeats, standing in for a sample in comparisons with dwave-system
SAMPLE = STATES[[0, 3, 5, 5, 6, 9, 10, 12, 15, 15, 1, 7]]
SINGLE = GraphIndex([0], [])  # one spin, no edges


def planted_sample(num_samples: int, beta: float = 1.0) -> torch.Tensor:
    """The sample of the planted model at inverse temperature ``beta`` in which every configuration
    appears as often as its Boltzmann probability says, rounded to a count."""
    probabilities = torch.softmax(-beta * PLANTED_ENERGIES, 0)
    return STATES.repeat_interleave(torch.round(probabilities * num_samples).long(), 0)


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
    def test_hand_worked_values(self):
        torch.testing.assert_close(nlpl(GRAPH, LINEAR, QUADRATIC, SPINS, reduction="none"), NLPL)
        torch.testing.assert_close(nlpl(GRAPH, LINEAR, QUADRATIC, SPINS), NLPL.mean())
        torch.testing.assert_close(nlpl(GRAPH, LINEAR, QUADRATIC, SPINS, reduction="sum"), NLPL.sum())
        with self.subTest("The value is in the dtype of the biases, whatever the spins' dtype"):
            value = nlpl(GRAPH, LINEAR.float(), QUADRATIC.float(), SPINS.long())
            self.assertEqual(torch.float32, value.dtype)
            torch.testing.assert_close(value, NLPL.mean().float())

    def test_batched_models(self):
        # The first two configurations under the first model, the last two under the second
        spins = torch.stack([SPINS[:2], SPINS[2:]])
        values = nlpl(GRAPH, LINEAR_2, QUADRATIC_2, spins, reduction="none")
        torch.testing.assert_close(values, torch.stack([NLPL[:2], NLPL_2]))
        means = nlpl(GRAPH, LINEAR_2, QUADRATIC_2, spins)
        torch.testing.assert_close(means, torch.stack([NLPL[:2].mean(), NLPL_2.mean()]))
        with self.subTest("Unbatched biases apply to every leading index of the spins"):
            values = nlpl(GRAPH, LINEAR, QUADRATIC, SPINS.reshape(2, 2, 4))
            torch.testing.assert_close(values, NLPL.reshape(2, 2).mean(-1))

    def test_gradients(self):
        # The objective is differentiable in the biases on both evaluation paths
        for dense in (True, False):
            with self.subTest(dense=dense):
                biases = LINEAR.clone().requires_grad_(), QUADRATIC.clone().requires_grad_()
                self.assertTrue(torch.autograd.gradcheck(
                    lambda h, J: nlpl(GRAPH, h, J, SPINS, dense=dense), biases
                ))
        with self.subTest("The scatter path gives the hand-worked values, batched too"):
            spins = torch.stack([SPINS[:2], SPINS[2:]])
            values = nlpl(GRAPH, LINEAR_2, QUADRATIC_2, spins, reduction="none", dense=False)
            torch.testing.assert_close(values, torch.stack([NLPL[:2], NLPL_2]))

    def test_validation(self):
        spins = SPINS[:3]
        with self.assertRaisesRegex(ValueError, "reduction"):
            nlpl(GRAPH, LINEAR, QUADRATIC, spins, reduction="max")
        with self.assertRaisesRegex(ValueError, "one per node"):
            nlpl(GRAPH, LINEAR[:3], QUADRATIC, spins)
        with self.assertRaisesRegex(ValueError, "one per edge"):
            nlpl(GRAPH, LINEAR, QUADRATIC[:3], spins)
        with self.assertRaisesRegex(ValueError, "same batch of models"):
            nlpl(GRAPH, LINEAR_2, QUADRATIC, spins)
        with self.assertRaisesRegex(ValueError, r"spins must have shape \(\.\.\., M, 4\)"):
            nlpl(GRAPH, LINEAR, QUADRATIC, spins[:, :3])
        with self.assertRaisesRegex(ValueError, r"spins must have shape \(\.\.\., M, 4\)"):
            nlpl(GRAPH, LINEAR, QUADRATIC, spins[0])
        with self.assertRaisesRegex(ValueError, "batched like the biases"):
            nlpl(GRAPH, LINEAR_2, QUADRATIC_2, SPINS.expand(3, 4, 4))
        with self.assertRaisesRegex(ValueError, "At least one sample"):
            nlpl(GRAPH, LINEAR, QUADRATIC, spins[:0])
        with self.assertRaisesRegex(ValueError, "±1 values"):
            nlpl(GRAPH, LINEAR, QUADRATIC, spins * 0.5)
        with self.assertRaisesRegex(ValueError, "±1 values"):
            nlpl(GRAPH, LINEAR, QUADRATIC, spins.clone().fill_(torch.nan))


class TestFitInverseTemperature(unittest.TestCase):
    def test_matches_dwave_system(self):
        cases = {
            "The fixture": (LINEAR, QUADRATIC, SAMPLE),
            "The planted model": (PLANTED_LINEAR, PLANTED_QUADRATIC, SAMPLE),
            "The second model": (-LINEAR, 2 * QUADRATIC, SAMPLE[:7]),
        }
        for name, (linear, quadratic, spins) in cases.items():
            with self.subTest(name):
                fit = fit_inverse_temperature(GRAPH, linear, quadratic, spins)
                self.assertIsInstance(fit, InverseTemperatureFit)
                self.assertEqual((), tuple(fit.beta.shape))
                self.assertEqual(torch.float64, fit.beta.dtype)
                self.assertTrue(bool(fit.converged))
                expected = ocean_beta(GRAPH, linear, quadratic, spins)
                self.assertAlmostEqual(expected, float(fit.beta), places=10)
                self.assertTrue(0 < float(fit.beta_stderr) < math.inf)
        with self.subTest("float32 biases and integer spins give the same estimate"):
            fit = fit_inverse_temperature(GRAPH, LINEAR, QUADRATIC, SAMPLE)
            fit32 = fit_inverse_temperature(GRAPH, LINEAR.float(), QUADRATIC.float(), SAMPLE.long())
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
        # The deterministic sample at inverse temperature 1.7 is at 1.7 up to the rounding of its
        # counts; the standard error says what a random sample of that size would add
        cold = planted_sample(20000, beta=1.7)
        fit = fit_inverse_temperature(GRAPH, PLANTED_LINEAR, PLANTED_QUADRATIC, cold)
        torch.testing.assert_close(fit.beta, torch.tensor(1.7, dtype=torch.float64), rtol=0, atol=1e-3)
        self.assertLess(float(fit.beta_stderr), 0.05)
        with self.subTest("The sample at unit inverse temperature"):
            fit = fit_inverse_temperature(GRAPH, PLANTED_LINEAR, PLANTED_QUADRATIC, planted_sample(20000))
            torch.testing.assert_close(fit.beta, torch.tensor(1.0, dtype=torch.float64), rtol=0, atol=1e-3)

    def test_batched_models(self):
        linear = torch.stack([LINEAR, -LINEAR, PLANTED_LINEAR])
        quadratic = torch.stack([QUADRATIC, 2 * QUADRATIC, PLANTED_QUADRATIC])
        spins = torch.stack([SAMPLE, -SAMPLE, SAMPLE.flip(0)])
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
            fit = fit_inverse_temperature(GRAPH, LINEAR, QUADRATIC, spins)
            self.assertEqual((3,), tuple(fit.beta.shape))
            single = fit_inverse_temperature(GRAPH, LINEAR, QUADRATIC, spins[2])
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
        fit = fit_inverse_temperature(GRAPH, LINEAR, QUADRATIC, SAMPLE)
        with self.subTest("Chunking does not change the estimate"):
            for sample_chunk in (7, None):
                chunked = fit_inverse_temperature(GRAPH, LINEAR, QUADRATIC, SAMPLE, sample_chunk=sample_chunk)
                torch.testing.assert_close(chunked.beta, fit.beta)
                torch.testing.assert_close(chunked.beta_stderr, fit.beta_stderr)
        with self.subTest("The scatter path gives the same estimate"):
            scattered = fit_inverse_temperature(GRAPH, LINEAR, QUADRATIC, SAMPLE, dense=False)
            torch.testing.assert_close(scattered.beta, fit.beta)
            torch.testing.assert_close(scattered.beta_stderr, fit.beta_stderr)
        with self.subTest("A single Newton step is the moment estimate, and is not converged"):
            # From zero, the first step is -2 sum(eps) / sum(eps ** 2) over the excitation energies
            moment = -2 * EXCITATIONS.sum() / EXCITATIONS.square().sum()
            one_step = fit_inverse_temperature(GRAPH, LINEAR, QUADRATIC, SPINS, max_iter=1)
            torch.testing.assert_close(one_step.beta, moment)
            self.assertFalse(bool(one_step.converged))
        with self.subTest("A single configuration has no standard error"):
            single = fit_inverse_temperature(GRAPH, LINEAR, QUADRATIC, SPINS[:1])
            self.assertTrue(math.isnan(float(single.beta_stderr)))
        for kwargs in (dict(max_iter=0), dict(tolerance=0.0), dict(tolerance=math.inf), dict(sample_chunk=0)):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                fit_inverse_temperature(GRAPH, LINEAR, QUADRATIC, SAMPLE, **kwargs)


class TestFitPseudolikelihood(unittest.TestCase):
    def setUp(self):
        self.spins = planted_sample(2000)

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
        # The deterministic sample of 20000 is the planted model up to the rounding of its counts
        fit = fit_pseudolikelihood(GRAPH, planted_sample(20000))
        torch.testing.assert_close(fit.linear, PLANTED_LINEAR, rtol=0, atol=2e-3)
        torch.testing.assert_close(fit.quadratic, PLANTED_QUADRATIC, rtol=0, atol=2e-3)
        with self.subTest("Data at inverse temperature 2 are fitted by twice the biases, are at "
                          "inverse temperature 2 under the planted ones and at 1 under the fit"):
            cold = planted_sample(20000, beta=2.0)
            fit = fit_pseudolikelihood(GRAPH, cold)
            torch.testing.assert_close(fit.linear, 2 * PLANTED_LINEAR, rtol=0, atol=1e-2)
            torch.testing.assert_close(fit.quadratic, 2 * PLANTED_QUADRATIC, rtol=0, atol=1e-2)
            against_planted = fit_inverse_temperature(GRAPH, PLANTED_LINEAR, PLANTED_QUADRATIC, cold)
            torch.testing.assert_close(
                against_planted.beta, torch.tensor(2.0, dtype=torch.float64), rtol=0, atol=1e-3
            )
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
            sparse = fit_pseudolikelihood(GRAPH, self.spins, dense=False)
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
