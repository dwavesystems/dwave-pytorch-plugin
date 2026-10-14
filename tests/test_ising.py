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
"""Tests of the Ising layer, its statistics and the gradient estimator behind them.

The estimator is tested exactly: on a three-node model every set of ``M`` samples is enumerated,
so the expectation of the estimated gradient is computed exactly and compared with the gradient of
the exact expectation, obtained by autograd through the enumerated probabilities."""

import itertools
import unittest

import torch
from dimod import BQM, ExactSolver

from dwave.plugins.torch.graph import GraphIndex
from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.nn import Ising, Mean, SquaredMMD
from dwave.plugins.torch.nn.functional import expectation, leave_one_out
from dwave.plugins.torch.nn.functional import maximum_mean_discrepancy_loss as mmd_loss
from dwave.plugins.torch.samplers import BlockSampler, DimodSampler
from dwave.plugins.torch.utils import to_bqm
from dwave.samplers import SimulatedAnnealingSampler as Neal
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple
from tests.helper_functions import randspins

# ---------------------------------------------------------------- exact enumeration ------------

GRAPH = GraphIndex("abc", [("a", "b"), ("b", "c"), ("a", "c")])
STATES = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=3)), dtype=torch.float64)


def biases(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    linear = (0.7 * torch.randn(3, generator=generator, dtype=torch.float64)).requires_grad_()
    quadratic = (0.7 * torch.randn(3, generator=generator, dtype=torch.float64)).requires_grad_()
    return linear, quadratic


def sample_sets(n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
    """All sets of ``n_samples`` states, and the indices of their states."""
    idx = torch.tensor(list(itertools.product(range(len(STATES)), repeat=n_samples)))
    return STATES[idx], idx


def exact_gradient(linear, quadratic, statistic, n_samples, weights, *inputs):
    """The gradient of the exact expectation of ``weights @ statistic`` over sets of ``n_samples``
    samples, by autograd through the enumerated probabilities (the statistic is constant)."""
    sets, idx = sample_sets(n_samples)
    probabilities = torch.softmax(-GRAPH.energy(STATES, linear, quadratic), 0)[idx].prod(-1)
    with torch.no_grad():
        values = statistic(sets, *inputs) @ weights
    return torch.autograd.grad((probabilities * values).sum(), (linear, quadratic))


def mean_estimated_gradient(linear, quadratic, statistic, n_samples, weights, *inputs):
    """The exact expectation of the gradient estimated by ``expectation``, over all sample sets."""
    sets, idx = sample_sets(n_samples)
    with torch.no_grad():
        probabilities = torch.softmax(-GRAPH.energy(STATES, linear, quadratic), 0)[idx].prod(-1)
    values = expectation(GRAPH, linear, quadratic, sets, statistic, *inputs) @ weights
    return torch.autograd.grad((probabilities * values).sum(), (linear, quadratic))


def relative_error(estimate, exact) -> float:
    return max((a - b).norm().item() / b.norm().item() for a, b in zip(estimate, exact))


def kappa3(x, y, z):
    """A symmetric kernel of three arguments."""
    return torch.exp(-(((x - y) ** 2).sum(-1) + ((y - z) ** 2).sum(-1) + ((z - x) ** 2).sum(-1)) / 8)


class DegreeThree(torch.nn.Module):
    """A U-statistic of degree three: the mean of ``kappa3`` over all triples of samples."""

    def forward(self, spins):
        triples = itertools.combinations(range(spins.shape[-2]), 3)
        values = [kappa3(spins[..., a, :], spins[..., b, :], spins[..., c, :]) for a, b, c in triples]
        return torch.stack(values, -1).mean(-1, keepdim=True)


def gaussian_kernel(x, y):
    """The Gaussian kernel of the battery, exp(-d / sqrt(N)) at the real distance d."""
    n = x.shape[-1]
    return torch.exp(-((n - x @ y.mT) / 2) / n ** 0.5)


class BiasedMMD(torch.nn.Module):
    """The biased (V-statistic) squared MMD, which is symmetric but not a U-statistic."""

    def forward(self, spins, reference):
        reference = reference.expand(*spins.shape[:-2], *reference.shape[-2:])
        return (gaussian_kernel(spins, spins).mean((-2, -1))
                - 2 * gaussian_kernel(spins, reference).mean((-2, -1))
                + gaussian_kernel(reference, reference).mean((-2, -1))).unsqueeze(-1)


class ExpMean(torch.nn.Module):
    """A nonlinear function of a sample mean, symmetric but not a U-statistic."""

    def forward(self, spins):
        return torch.exp(spins[..., 0].mean(-1)).unsqueeze(-1)


class Scale(torch.nn.Module):
    """A per-sample transform with a parameter."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))

    def forward(self, spins):
        return self.scale * spins


REFERENCE = torch.tensor([[0.8, -0.9, 1.0], [-1.0, 1.0, 0.7]], dtype=torch.float64)


class TestMean(unittest.TestCase):
    def test_value_and_leave_one_out(self):
        spins = randspins(2, 5, 3, seed=1)
        mean = Mean()
        torch.testing.assert_close(mean(spins), spins.mean(-2))
        with self.subTest("The closed-form leave-one-out values are the means of the other samples"):
            generic = leave_one_out(lambda s: s.mean(-2), spins)
            torch.testing.assert_close(mean.leave_one_out(spins), generic)
            self.assertEqual((2, 5, 3), tuple(generic.shape))

    def test_transform(self):
        spins = randspins(2, 5, 3, seed=2)
        with self.subTest("A bound method of a graph is a transform"):
            mean = Mean(GRAPH.statistics)
            torch.testing.assert_close(mean(spins), GRAPH.statistics(spins).mean(-2))
            self.assertEqual((2, 6), tuple(mean(spins).shape))
        with self.subTest("A module is a submodule whose parameters are exposed"):
            scale = Scale()
            mean = Mean(scale)
            self.assertIs(scale, dict(mean.named_children())["transform"])
            self.assertEqual(1, len(list(mean.parameters())))
        with self.subTest("Further inputs reach the transform"):
            mean = Mean(lambda s, shift: s + shift)
            torch.testing.assert_close(mean(spins, torch.tensor(1.0)), spins.mean(-2) + 1)

    def test_statistics_of_graph(self):
        spins = randspins(4, 3, seed=3)
        expected = torch.cat([spins, spins[:, [0, 1, 0]] * spins[:, [1, 2, 2]]], -1)
        torch.testing.assert_close(GRAPH.statistics(spins), expected)
        linear, quadratic = torch.randn(3), torch.randn(3)
        torch.testing.assert_close(
            GRAPH.statistics(spins) @ torch.cat([linear, quadratic]),
            GRAPH.energy(spins, linear, quadratic),
        )


class TestSquaredMMD(unittest.TestCase):
    def test_matches_functional(self):
        # One value per profile: [0] the Gaussian headline, [k] the order-k kernel
        spins = randspins(5, 3, seed=4).double()
        statistic = SquaredMMD()
        self.assertEqual((4,), tuple(statistic(spins, REFERENCE).shape))
        torch.testing.assert_close(statistic(spins, REFERENCE), mmd_loss(spins, REFERENCE))
        with self.subTest("Batches of sample sets, one reference sample for all"):
            batch = randspins(2, 5, 3, seed=5).double()
            values = statistic(batch, REFERENCE)
            self.assertEqual((2, 4), tuple(values.shape))
            for b in range(2):
                torch.testing.assert_close(values[b], mmd_loss(batch[b], REFERENCE))
        with self.subTest("One reference sample per set"):
            references = torch.stack([REFERENCE, -REFERENCE])
            values = statistic(batch, references)
            for b in range(2):
                torch.testing.assert_close(values[b], mmd_loss(batch[b], references[b]))
        with self.subTest("Kernel weights make the headline the alpha row"):
            alpha = torch.tensor([0.5, 0.2, 0.3], dtype=torch.float64)
            weighted = SquaredMMD(alpha)(batch, REFERENCE)
            torch.testing.assert_close(weighted[:, 1:], values[:, 1:] * 0 + statistic(batch, REFERENCE)[:, 1:])
            torch.testing.assert_close(weighted[:, 0], statistic(batch, REFERENCE)[:, 1:] @ alpha)

    def test_leave_one_out(self):
        spins = randspins(2, 6, 3, seed=6).double()
        for statistic in (SquaredMMD(), SquaredMMD(torch.tensor([0.5, 0.2, 0.3], dtype=torch.float64))):
            with self.subTest(alpha=statistic.alpha):
                generic = leave_one_out(lambda s, z: statistic(s, z), spins, REFERENCE)
                self.assertEqual((2, 6, 4), tuple(generic.shape))
                torch.testing.assert_close(statistic.leave_one_out(spins, REFERENCE), generic)

    def test_validation(self):
        statistic = SquaredMMD()
        with self.assertRaisesRegex(ValueError, "At least 2 samples"):
            statistic(randspins(1, 3).double(), REFERENCE)
        with self.assertRaisesRegex(ValueError, "At least 3 samples"):
            statistic.leave_one_out(randspins(2, 3).double(), REFERENCE)
        with self.assertRaisesRegex(ValueError, "At least two reference items"):
            statistic(randspins(4, 3).double(), REFERENCE[:1])
        with self.assertRaisesRegex(ValueError, "spins per item"):
            statistic(randspins(4, 3).double(), REFERENCE[:, :2])
        with self.assertRaisesRegex(ValueError, r"soft spins in \[-1, 1\]"):
            statistic(randspins(4, 3).double(), 2 * REFERENCE)
        with self.assertRaisesRegex(ValueError, "floating dtype"):
            statistic(randspins(4, 3).long(), REFERENCE)
        with self.assertRaisesRegex(ValueError, "one weight per order"):
            SquaredMMD(torch.ones(2))(randspins(4, 3).double(), REFERENCE)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            SquaredMMD(-torch.ones(3))


class TestExpectation(unittest.TestCase):
    """Exact tests of the gradient estimator by enumeration of all sample sets."""

    def test_unbiased_for_means_and_u_statistics(self):
        # For a sample mean and for U-statistics the expectation over M samples equals the one over
        # M - 1 samples, and the estimator is unbiased for the gradient of both
        linear, quadratic = biases()
        cases = [
            ("mean spins", Mean(), 3, torch.tensor([0.3, -1.2, 0.5], dtype=torch.float64), ()),
            ("mean sufficient statistics", Mean(GRAPH.statistics), 3,
             torch.arange(1.0, 7.0, dtype=torch.float64), ()),
            ("squared MMD (degree 2, every profile)", SquaredMMD(), 3,
             torch.tensor([1.0, 0.3, -0.7, 0.2], dtype=torch.float64), (REFERENCE,)),
            ("degree 3", DegreeThree(), 4, torch.ones(1, dtype=torch.float64), ()),
        ]
        for name, statistic, n_samples, weights, inputs in cases:
            with self.subTest(name):
                estimate = mean_estimated_gradient(linear, quadratic, statistic, n_samples, weights, *inputs)
                for m in (n_samples, n_samples - 1):
                    exact = exact_gradient(linear, quadratic, statistic, m, weights, *inputs)
                    self.assertLess(relative_error(estimate, exact), 1e-9, f"{name}, {m} samples")

    def test_symmetric_statistics_target_one_sample_fewer(self):
        # For any symmetric statistic the estimator is unbiased for the gradient of the expectation
        # over M - 1 samples, which differs from the one over M samples by the statistic's own bias
        linear, quadratic = biases(1)
        weights = torch.ones(1, dtype=torch.float64)
        for name, statistic, inputs in [("V-statistic", BiasedMMD(), (REFERENCE,)),
                                        ("nonlinear in a mean", ExpMean(), ())]:
            with self.subTest(name):
                estimate = mean_estimated_gradient(linear, quadratic, statistic, 3, weights, *inputs)
                self.assertLess(
                    relative_error(estimate, exact_gradient(linear, quadratic, statistic, 2, weights, *inputs)),
                    1e-9,
                )
                self.assertGreater(
                    relative_error(estimate, exact_gradient(linear, quadratic, statistic, 3, weights, *inputs)),
                    1e-3,
                )

    def test_covariance_form(self):
        # For a sample mean the gradient is the unbiased sample covariance of the transform with
        # the sufficient statistics, which the layer of an earlier release computed by hand
        spins = torch.tensor([[[-1, -1, -1], [-1, 1, 1], [1, 1, 1]],
                              [[1, -1, 1], [-1, -1, 1], [-1, -1, 1]]]).float().requires_grad_()
        transform = lambda s: torch.cat([s[..., [1]], s[..., [0, 1]] * s[..., [1, 2]]], -1)
        linear = torch.zeros((2, 3), requires_grad=True)
        quadratic = torch.zeros((2, 3), requires_grad=True)
        y = expectation(GRAPH, linear, quadratic, spins, Mean(transform))
        torch.testing.assert_close(y, torch.tensor([[1 / 3, 1 / 3, 1], [-1, 1 / 3, -1]]))
        (y ** 2).sum().backward()
        grads = []
        for b in range(2):
            t = spins[b].detach()
            stat = torch.hstack([t[..., [1]], t[..., [0, 1]] * t[..., [1, 2]]])
            sufficient = torch.hstack([t, t[..., [0, 1, 0]] * t[..., [1, 2, 2]]])
            grads.append(2 * y[b] @ (-torch.cat([stat, sufficient], -1).mT.cov()[:3, 3:]))
        grad = torch.vstack(grads)
        torch.testing.assert_close(grad[:, :3], linear.grad)
        torch.testing.assert_close(grad[:, 3:], quadratic.grad)
        with self.subTest("Spins receive no gradient"):
            self.assertIsNone(spins.grad)

    def test_gradients_of_statistic_parameters_and_inputs(self):
        linear, quadratic = biases(2)
        spins = randspins(4, 7, 3, seed=7).double()
        with self.subTest("A parameter of the transform receives the mean pathwise derivative"):
            scale = Scale()
            upstream = torch.randn(4, 3, dtype=torch.float64)
            y = expectation(GRAPH, linear.detach().expand(4, 3), quadratic.detach().expand(4, 3),
                            spins, Mean(scale))
            grad, = torch.autograd.grad((y * upstream).sum(), scale.scale)
            torch.testing.assert_close(grad, (spins.mean(1) * upstream).sum())
        with self.subTest("The reference sample of the MMD receives its pathwise gradient"):
            reference = REFERENCE.clone().requires_grad_()
            statistic = SquaredMMD()
            y = expectation(GRAPH, linear, quadratic, spins, statistic, reference)
            through_layer, = torch.autograd.grad(y[..., 0].sum(), reference)
            direct, = torch.autograd.grad(statistic(spins, reference)[..., 0].sum(), reference)
            torch.testing.assert_close(through_layer, direct)
            self.assertIsNone(spins.grad)

    def test_boltzmann_machine_prior(self):
        # The functional takes a Boltzmann machine: its parameters receive the MMD gradient
        prior = GRBM("abc", [("a", "b"), ("b", "c")], linear={"a": 0.2}, quadratic={("a", "b"): -0.5})
        samples = BlockSampler(prior, num_chains=8, seed=0).sample()
        reference = randspins(5, 3, seed=8)
        loss = expectation(prior, prior.linear, prior.quadratic, samples, SquaredMMD(), reference)
        self.assertEqual((4,), tuple(loss.shape))
        loss[0].backward()
        self.assertTrue(prior.linear.grad.isfinite().all() and prior.quadratic.grad.isfinite().all())

    def test_batch_dimensions(self):
        linear = torch.randn(2, 3, 3, requires_grad=True)
        quadratic = torch.randn(2, 3, 3, requires_grad=True)
        spins = randspins(2, 3, 6, 3, seed=9)
        y = expectation(GRAPH, linear, quadratic, spins, Mean())
        self.assertEqual((2, 3, 3), tuple(y.shape))
        y.sum().backward()
        with self.subTest("Every model of the batch is treated on its own"):
            for b, c in itertools.product(range(2), range(3)):
                h = linear[b, c].detach().requires_grad_()
                J = quadratic[b, c].detach().requires_grad_()
                single = expectation(GRAPH, h, J, spins[b, c], Mean())
                self.assertEqual((3,), tuple(single.shape))
                torch.testing.assert_close(single, y[b, c])
                single.sum().backward()
                torch.testing.assert_close(h.grad, linear.grad[b, c])
                torch.testing.assert_close(J.grad, quadratic.grad[b, c])
        with self.subTest("Unbatched biases apply to a batch of sample sets"):
            h, J = torch.randn(3), torch.randn(3)
            self.assertEqual((2, 3, 3), tuple(expectation(GRAPH, h, J, spins, Mean()).shape))

    def test_validation(self):
        h, J, spins = torch.zeros(3), torch.zeros(3), randspins(2, 4, 3)
        with self.assertRaisesRegex(ValueError, r"spins must have shape \(\.\.\., M, 3\)"):
            expectation(GRAPH, h, J, randspins(2, 4, 2), Mean())
        with self.assertRaisesRegex(ValueError, "At least two samples"):
            expectation(GRAPH, h, J, randspins(2, 1, 3), Mean())
        with self.assertRaisesRegex(ValueError, "batched like spins"):
            expectation(GRAPH, torch.zeros(3, 3), torch.zeros(3, 3), spins, Mean())
        with self.assertRaisesRegex(ValueError, "one bias per edge|edge biases"):
            expectation(GRAPH, h, torch.zeros(2), spins, Mean())
        with self.assertRaisesRegex(ValueError, "one row of values per model"):
            expectation(GRAPH, h, J, spins, lambda s: s.mean((-2, -1)))
        with self.assertRaisesRegex(ValueError, "leave-one-out statistics must have shape"):
            class BadLeaveOneOut(Mean):
                def leave_one_out(self, spins, *inputs):
                    return super().leave_one_out(spins, *inputs)[..., :-1, :]
            expectation(GRAPH, h, J, spins, BadLeaveOneOut())


class TestIsing(unittest.TestCase):
    NODES = "abc"
    EDGES = [("a", "b"), ("a", "c"), ("b", "c")]

    def test_has_properties(self):
        statistic = SquaredMMD()
        ising = Ising(self.NODES, self.EDGES, statistic=statistic)
        self.assertTupleEqual(("a", "b", "c"), ising.nodes)
        self.assertTupleEqual((("a", "b"), ("a", "c"), ("b", "c")), ising.edges)
        self.assertDictEqual({"a": 0, "b": 1, "c": 2}, ising.node_to_idx)
        self.assertEqual(3, ising.n_nodes)
        self.assertEqual(3, ising.n_edges)
        self.assertIs(statistic, dict(ising.named_children())["statistic"])
        self.assertEqual(0, len(list(ising.parameters())))
        self.assertIn("n_nodes=3, n_edges=3", repr(ising))

        with self.subTest("The default statistic is the mean of the spins"):
            ising = Ising(self.NODES, self.EDGES)
            self.assertIsInstance(ising.statistic, Mean)
            self.assertIsNone(ising.statistic.transform)

    def test_correct_node_indices_of_edges(self):
        ising = Ising("abc", [("b", "a"), ("a", "c"), ("b", "c")])
        # Canonical (upper-triangular) orientation regardless of the given orientation
        self.assertListEqual([0, 0, 1], ising.edge_idx_i.tolist())
        self.assertListEqual([1, 2, 2], ising.edge_idx_j.tolist())
        self.assertDictEqual(
            {("b", "a"): 0, ("a", "b"): 0, ("a", "c"): 1, ("c", "a"): 1, ("b", "c"): 2, ("c", "b"): 2},
            ising.edge_to_idx,
        )

    def test_dense_quadratic(self):
        ising = Ising("abc", [("b", "a"), ("a", "c"), ("b", "c")])
        per_edge = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        dense = ising.dense_quadratic(per_edge)
        self.assertEqual((2, 3, 3), tuple(dense.shape))
        torch.testing.assert_close(dense[0], torch.tensor([[0.0, 1.0, 2.0],
                                                           [0.0, 0.0, 3.0],
                                                           [0.0, 0.0, 0.0]]))
        torch.testing.assert_close(dense[..., ising.edge_idx_i, ising.edge_idx_j], per_edge)
        with self.assertRaisesRegex(ValueError, "Expected 3 edge biases"):
            ising.dense_quadratic(torch.zeros(2, 2))

    def test_invalid_graph(self):
        with self.assertRaisesRegex(ValueError, "Self-loops"):
            Ising("abc", [("a", "a")])
        with self.assertRaisesRegex(ValueError, "Duplicate edges"):
            Ising("abc", [("a", "b"), ("b", "a")])

    def test_forward_is_the_functional(self):
        ising = Ising(self.NODES, self.EDGES, statistic=Mean(GRAPH.statistics))
        linear, quadratic = torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        spins = randspins(2, 5, 3, seed=10)
        y = ising(linear, quadratic, spins)
        expected = expectation(ising, linear, quadratic, spins, ising.statistic)
        torch.testing.assert_close(y, expected)
        upstream = torch.randn(2, 6)
        grads = torch.autograd.grad((y * upstream).sum(), (linear, quadratic))
        expected_grads = torch.autograd.grad((expected * upstream).sum(), (linear, quadratic))
        for grad, expected_grad in zip(grads, expected_grads):
            torch.testing.assert_close(grad, expected_grad)
        with self.subTest("Further inputs are passed to the statistic"):
            ising = Ising(self.NODES, self.EDGES, statistic=SquaredMMD())
            reference = randspins(4, 3, seed=11)
            self.assertEqual((2, 4), tuple(ising(linear, quadratic, spins, reference).shape))

    def test_input_validation(self):
        ising = Ising("abc", [("a", "b")])
        linear, quadratic, spins = torch.zeros(2, 3), torch.zeros(2, 1), torch.ones(2, 5, 3)
        with self.assertRaisesRegex(ValueError, "linear"):
            ising(torch.zeros(2, 4), quadratic, spins)
        with self.assertRaisesRegex(ValueError, "edge biases"):
            ising(linear, torch.zeros(2, 2), spins)
        with self.assertRaisesRegex(ValueError, "batched like spins"):
            ising(linear, quadratic, torch.ones(3, 5, 3))
        with self.assertRaisesRegex(ValueError, r"spins must have shape \(\.\.\., M, 3\)"):
            ising(linear, quadratic, torch.ones(2, 5, 2))
        with self.assertRaisesRegex(ValueError, "batched like spins"):
            ising.estimate_betas(linear, quadratic, torch.ones(3, 5, 3))

    def test_energy(self):
        # The layer inherits the batched energies of its graph
        ising = Ising(self.NODES, self.EDGES)
        linear = torch.tensor([[0.1, 0.2, 0.4], [-0.9, -0.8, -0.6]])
        edge_biases = torch.tensor([[1.0, 9.0, 4.0], [0.9, 8.0, -12.0]])
        spins = randspins(2, 6, 3, seed=1)
        energies = ising.energy(spins, linear, edge_biases)
        self.assertEqual((2, 6), tuple(energies.shape))
        for b in range(2):
            bqm = to_bqm(ising, linear[b], edge_biases[b])
            expected = bqm.energies((spins[b].numpy(), list(ising.nodes)))
            torch.testing.assert_close(energies[b], torch.tensor(expected, dtype=torch.float32))

    def test_estimate_betas(self):
        s1 = [[-1, -1, -1],
              [-1,  1,  1],
              [1,  1,  1]]
        s2 = [[1,  -1,  1],
              [-1, -1, 1],
              [-1, -1, 1]]
        linear = torch.tensor([[0.1, 0.2, 0.4],
                               [-0.9, -0.8, -0.6]])
        quadratic = torch.tensor([[1.0, 9.0, 4.0],
                                  [0.9, 8.0, -12.0]])
        ising = Ising(self.NODES, self.EDGES)
        spins = torch.tensor([s1, s2]).float()

        estimated_betas = ising.estimate_betas(linear, quadratic, spins)

        dimod_betas = []
        for h, J, s in zip(linear, quadratic, (s1, s2)):
            bqm = BQM.from_ising(dict(zip("abc", h.tolist())), dict(zip(self.EDGES, J.tolist())))
            dimod_betas.append(1 / float(mple(bqm, (s, list("abc")))[0]))
        torch.testing.assert_close(torch.tensor(dimod_betas), estimated_betas)

        with self.subTest("Any batch shape, and unbatched biases"):
            self.assertEqual((2, 1), tuple(ising.estimate_betas(linear[:, None], quadratic[:, None], spins[:, None]).shape))
            self.assertEqual((), tuple(ising.estimate_betas(linear[0], quadratic[0], spins[0]).shape))
            torch.testing.assert_close(ising.estimate_betas(linear[0], quadratic[0], spins), estimated_betas.new_tensor(
                [dimod_betas[0], 1 / float(mple(BQM.from_ising(dict(zip("abc", linear[0].tolist())),
                                                              dict(zip(self.EDGES, quadratic[0].tolist()))), (s2, list("abc")))[0])]))

    def test_sampling_with_block_sampler(self):
        # The layer is sampled on its own graph by a sampler bound to it; strong fields make the
        # Gibbs draws deterministic (a spin flips against a field of 10 with probability e^-20)
        ising = Ising("abc", [("a", "b"), ("a", "c")])
        sampler = BlockSampler(ising, schedule=[1.0] * 5)
        linear = torch.tensor([[10.0] * 3, [-10.0] * 3])
        quadratic = torch.zeros(2, 2)
        spins = sampler.sample_biases(linear, quadratic, num_samples=100)
        self.assertEqual((2, 100, 3), tuple(spins.shape))
        torch.testing.assert_close(
            ising(linear, quadratic, spins), torch.tensor([[-1.0] * 3, [1.0] * 3])
        )

        with self.subTest("The layer has no parameters of its own to sample"):
            with self.assertRaisesRegex(TypeError, "GraphRestrictedBoltzmannMachine"):
                sampler.sample()

    def test_sampling_with_dimod_sampler(self):
        ising = Ising("abc", [("a", "b"), ("a", "c")])
        bs = 4
        quadratic = torch.zeros((bs, 2))

        with self.subTest("An exactly enumerated uniform prior gives exactly zero statistics"):
            # ExactSolver returns every state once, which is the Boltzmann distribution of zero
            # biases, so the layer's averages vanish exactly
            sampler = DimodSampler(ising, ExactSolver())
            linear = torch.zeros(bs, 3)
            spins = sampler.sample_biases(linear, quadratic)
            self.assertEqual((bs, 8, 3), tuple(spins.shape))
            torch.testing.assert_close(ising(linear, quadratic, spins), torch.zeros(bs, 3))

        with self.subTest("Biases are scaled by the prefactor of a sampler at another temperature"):
            # Scaled to 1e14, the fields pin every spin of every read of simulated annealing
            sampler = DimodSampler(ising, Neal(), prefactor=1e20,
                                   sample_kwargs=dict(num_sweeps=1, num_reads=10, beta_range=[1, 1]))
            linear = torch.ones((bs, 3)) * 1e-6
            spins = sampler.sample_biases(linear, quadratic)
            self.assertEqual((bs, 10, 3), tuple(spins.shape))
            torch.testing.assert_close(ising(linear, quadratic, spins), -torch.ones(bs, 3))

    def test_statistic_moves_with_layer(self):
        ising = Ising("abc", [("a", "b")], statistic=SquaredMMD(torch.ones(3))).to("meta")
        self.assertEqual("meta", ising.statistic.alpha.device.type)
        self.assertIn("statistic.alpha", ising.state_dict())


if __name__ == "__main__":
    unittest.main()
