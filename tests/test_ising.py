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

import unittest

import torch
from dimod import BQM, NullSampler, SampleSet

from dwave.plugins.torch.nn.modules.ising import (IdentityStatistic, Ising, IsingExpectation,
                                                  IsingStatistic, SpinStatistic)
from dwave.samplers import SimulatedAnnealingSampler as Neal
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple


class _ConcreteStatistic(SpinStatistic):
    """Minimal concrete implementation for testing the ABC."""

    def _transform(self, x):
        return x[..., :self.dim_out]


class TestStatistic(unittest.TestCase):
    """Verify the Statistic ABC enforces its contract:
    - inputs must be 3D (batch, samples, features)
    - outputs must be 3D with last dim == dim_out
    """

    def test_dim_out(self):
        # The dim_out property should reflect what was passed at construction
        stat = _ConcreteStatistic(dim_out=3)
        self.assertEqual(stat.dim_out, 3)

    def test_rejects_2d_input(self):
        # 2D tensors lack the sample dimension
        stat = _ConcreteStatistic(dim_out=4)
        x = torch.randn(2, 4)
        with self.assertRaisesRegex(ValueError, r"Input tensor.*ndim == 3"):
            stat(x)

    def test_rejects_4d_input(self):
        # 4D tensors have an extra spatial dim
        stat = _ConcreteStatistic(dim_out=4)
        x = torch.randn(2, 3, 4, 5)
        with self.assertRaisesRegex(ValueError, r"Input tensor.*ndim == 3"):
            stat(x)

    def test_rejects_wrong_output_dim(self):
        # If _transform returns a tensor whose last dim != dim_out, catch the bug early
        class BadStatistic(SpinStatistic):
            def _transform(self, x):
                return x[..., :2]

        stat = BadStatistic(dim_out=5)
        x = torch.randn(2, 3, 4)
        with self.assertRaisesRegex(ValueError, r"Output dimension.*does not match"):
            stat(x)

    def test_rejects_wrong_output_ndim(self):
        # If _transform collapses a dimension (e.g. returns 2D), catch it
        class SquashStatistic(SpinStatistic):
            def _transform(self, x):
                return x.mean(dim=1)  # (batch, features) — loses sample dim

        stat = SquashStatistic(dim_out=4)
        x = torch.randn(2, 3, 4)
        with self.assertRaisesRegex(ValueError, r"Output tensor.*ndim == 3"):
            stat(x)

    def test_valid_call(self):
        # Happy path: correct shape in, correct shape out
        stat = _ConcreteStatistic(dim_out=3)
        x = torch.randn(2, 4, 5)
        result = stat(x)
        self.assertEqual(result.shape, (2, 4, 3))


class TestIdentityStatistic(unittest.TestCase):
    """Verify IdentityStatistic passes input through unchanged."""

    def test_dim_out_matches_dim_in(self):
        stat = IdentityStatistic(dim_in=5)
        self.assertEqual(stat.dim_out, 5)

    def test_output_equals_input(self):
        stat = IdentityStatistic(dim_in=4)
        x = torch.randn(2, 3, 4)
        result = stat(x)
        torch.testing.assert_close(result, x)

    def test_output_shape(self):
        stat = IdentityStatistic(dim_in=7)
        x = torch.randn(5, 10, 7)
        result = stat(x)
        self.assertEqual(result.shape, (5, 10, 7))


class TestIsingStatistic(unittest.TestCase):
    """Verify Ising correctly computes [x[indices], x[indices_j]*x[indices_i]]
    under various edge cases encountered in practice."""

    def test_transform(self):
        # Core behaviour: picks nodes and computes pairwise products
        stat = IsingStatistic(
            node_indices=[0, 2],
            endpoints_1=[1],
            endpoints_2=[3],
        )
        # shape (batch=1, samples=2, nodes=4)
        x = torch.tensor([[[1.0, -1.0, 1.0, -1.0],
                           [1.0,  1.0, -1.0, 1.0]]])
        result = stat(x)
        # expected: [x[...,0], x[...,2], x[...,3]*x[...,1]]
        expected = torch.tensor([[[1.0, 1.0, (-1.0)*(-1.0)],
                                  [1.0, -1.0, 1.0*1.0]]])
        torch.testing.assert_close(result, expected)

    def test_output_shape(self):
        # Verify shape is (batch, samples, dim_out) for arbitrary input sizes
        stat = IsingStatistic(
            node_indices=[0, 1],
            endpoints_1=[0, 2],
            endpoints_2=[1, 3],
        )
        x = torch.randn(3, 5, 7)
        result = stat(x)
        self.assertEqual(result.shape, (3, 5, 4))

    def test_no_interactions(self):
        # Edge case: only node indices, no interaction terms
        # (e.g. an Ising model with no input_edges)
        stat = IsingStatistic(node_indices=[0, 3], endpoints_1=[], endpoints_2=[])
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]],
                          [[0.1, 2.0, 3.0, 5.0]]])
        result = stat(x)
        torch.testing.assert_close(result, torch.tensor([[[1.0, 4.0]], [[0.1, 5.0]]]))

    def test_no_nodes(self):
        # Edge case: only interaction terms, no direct node indices
        stat = IsingStatistic(node_indices=[], endpoints_1=[0], endpoints_2=[1])
        x = torch.tensor([[[3.0, -2.0]]])
        result = stat(x)
        torch.testing.assert_close(result, torch.tensor([[[-6.0]]]))

    def test_bad_edge_indices(self):
        # Edge indices have different length
        with self.assertRaisesRegex(ValueError, "Interaction indices should be of the same length, got"):
            IsingStatistic(node_indices=[], endpoints_1=[0], endpoints_2=[1, 2])

    def test_tensor_indices(self):
        # Ising internally passes nn.Parameter tensors as indices;
        # verify IsingStatistic handles them the same as lists
        stat = IsingStatistic(
            node_indices=torch.tensor([1]),
            endpoints_1=torch.tensor([0]),
            endpoints_2=torch.tensor([2]),
        )
        x = torch.nn.Parameter(torch.tensor([[[2.0, 3.0, 4.0]]]))
        result = stat(x)
        torch.testing.assert_close(result, torch.tensor([[[3.0, 8.0]]]))


class TestIsingExpectation(unittest.TestCase):

    def test_forward_backward(self):
        """Test forward and backward evaluations are as expected."""
        spins = torch.tensor([[[-1, -1, -1],
                               [-1,  1,  1],
                               [1,  1,  1]],
                              [[1, -1,  1],
                               [-1, -1,  1],
                               [-1, -1,  1]]]).float()
        # Interaction terms are col0*col1, col1*col2
        interactions = spins[..., [0, 1]] * spins[..., [1, 2]]
        sufficient_stats = torch.cat([spins, interactions], dim=-1)

        # Statistic is col0 + col1, col1*col2
        output_stats = torch.cat([spins[..., [0]] + spins[..., [1]],
                                  spins[..., [1]] * spins[..., [2]]],
                                 dim=-1)

        linear = torch.tensor([-0.1, -0.2, -0.3], requires_grad=True)
        quadratic = torch.tensor([0.0, 1.0], requires_grad=True)

        y = IsingExpectation.apply(spins, interactions, output_stats, linear, quadratic)

        with self.subTest("Ising aggregation layer produced unexpected output values"):
            with torch.no_grad():
                torch.testing.assert_close(y, output_stats.mean(1))

        # Gradient amounts to summing over gradients per obs
        loss = (y**2).sum()
        loss.backward()

        with torch.no_grad():
            # Manually compute the gradients
            dloss_dy = 2*y
            dy_dhJ = -torch.stack([torch.cat([o, i], -1).mT.cov()[:2, 2:]
                                   for i, o in zip(sufficient_stats, output_stats)])
            # hJ = torch.cat([linear, quadratic]).repeat(2, 1)
            dloss_dhJ = torch.einsum("bi, bij -> bj", dloss_dy, dy_dhJ).sum(0)

        with self.subTest("Linear gradients should match"):
            torch.testing.assert_close(dloss_dhJ[:3], linear.grad)

        with self.subTest("Quadratic gradients should match"):
            torch.testing.assert_close(dloss_dhJ[3:], quadratic.grad)


class TestIsing(unittest.TestCase):

    def test_has_properties(self):
        ising = Ising(
            nodes="abc",
            edges=[("a", "b"), ("a", "c"), ("b", "c")],
            beta=1.0,
            sampler=NullSampler(),
            statistic=IsingStatistic([1], [0, 1], [1, 2]),
            sample_params=dict(a=1),
        )
        self.assertEqual(1.0, ising.beta)
        self.assertListEqual(["a", "b", "c"], ising.nodes)
        self.assertListEqual([("a", "b"), ("a", "c"), ("b", "c")], ising.edges)
        self.assertEqual(NullSampler, ising.sampler.__class__)
        self.assertDictEqual(dict(a=1), ising.sample_params)

    def test_setters(self):
        ising = Ising(
            nodes="abc",
            edges=[("a", "b"), ("a", "c"), ("b", "c")],
            beta=1.0,
            sampler=NullSampler(),
            statistic=IsingStatistic([1], [0, 1], [1, 2]),
            sample_params=dict(a=1),
        )

        with self.subTest("Set beta"):
            ising.set_beta(342.0)
            self.assertEqual(342.0, ising.beta)

        with self.subTest("Set sampling parameters"):
            ising.set_sample_params(dict(b=2))
            self.assertDictEqual(dict(b=2), ising.sample_params)

        with self.subTest("Set sampler"):
            ising.set_sampler(Neal())
            self.assertEqual(Neal, ising.sampler.__class__)

    def test_correct_node_indices_of_edges(self):
        ising = Ising(
            nodes="abc",
            edges=[("a", "b"), ("a", "c"), ("b", "c")],
            beta=1.0,
            sampler=NullSampler(),
            statistic=IsingStatistic([1], [0, 1], [1, 2]),
            sample_params=dict(a=1),
        )
        self.assertListEqual([0, 0, 1], ising.node_idx_of_edges_1.tolist())
        self.assertListEqual([1, 2, 2], ising.node_idx_of_edges_2.tolist())

    def test_sampling_with_beta(self):
        sampler = Neal()
        bs = 10
        with self.subTest("Spins should be pinned"):
            sample_params = dict(num_sweeps=1, num_reads=10000, beta_range=[1, 1])
            model = Ising("abc", [("a", "b"), ("a", "c")],
                          sampler, sample_params, 1e-20)
            linear = torch.ones((bs, 3))*1e-6
            quadratic = torch.zeros((bs, 2))
            y = model(linear, quadratic)
            torch.testing.assert_close(y.mean(0), -torch.ones(3))

        with self.subTest("Average should be 0"):
            sample_params = dict(num_sweeps=2, num_reads=100000, beta_range=[1, 1])
            model = Ising("abc", [("a", "b"), ("a", "c")],
                          sampler, sample_params, 1e30)
            linear = torch.ones((bs, 3))
            quadratic = torch.zeros((bs, 2))
            y = model(linear, quadratic)
            torch.testing.assert_close(y.mean(0), torch.zeros(3), rtol=0.001, atol=0.01)

    def test_beta(self):
        sampler = NullSampler()
        with self.subTest("Invalid beta at initialization"):
            with self.assertRaisesRegex(ValueError, "Effective inverse temperature beta must be positive."):
                Ising("abc", [], sampler, dict(), 0.0)
            with self.assertRaisesRegex(ValueError, "Effective inverse temperature beta must be positive."):
                Ising("abc", [], sampler, dict(), -0.1)

        with self.subTest("Set invalid beta"):
            ising = Ising("abc", [], sampler, dict(), 1.0)
            with self.assertRaisesRegex(ValueError, "Effective inverse temperature beta must be positive."):
                ising.set_beta(0.0)
            with self.assertRaisesRegex(ValueError, "Effective inverse temperature beta must be positive."):
                ising.set_beta(-0.1)

    def test_estimate_beta(self):
        # Here we construct a dummy sampler to output a sample of size 3. The sampler produces these
        # two samples and returns them in a first-in-first-out manner.
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

        h0 = dict(zip("abc", linear[0].numpy()))
        J0 = dict(zip([("a", "b"), ("a", "c"), ("b", "c")], quadratic[0].numpy()))
        bqm0 = BQM.from_ising(h0, J0)

        h1 = dict(zip("abc", linear[1].numpy()))
        J1 = dict(zip([("a", "b"), ("a", "c"), ("b", "c")], quadratic[1].numpy()))
        bqm1 = BQM.from_ising(h1, J1)

        class FIFOSampler:
            def __init__(self):
                self.samples = [
                    SampleSet.from_samples_bqm((s1, list("abc")), bqm0),
                    SampleSet.from_samples_bqm((s2, list("abc")), bqm1)
                ]

            def sample(self, *args, **kwargs):
                return self.samples.pop(0)

        sampler = FIFOSampler()
        sample_params = dict()
        model = Ising("abc", [("a", "b"), ("a", "c"), ("b", "c")],
                      sampler, sample_params, 9999.0)

        estimated_betas = model.estimate_betas(linear, quadratic)
        dimod_betas = [
            1 / float(mple(bqm0, (s1, list("abc")))[0].item()),
            1 / float(mple(bqm1, (s2, list("abc")))[0].item())
        ]
        torch.testing.assert_close(torch.tensor(dimod_betas), estimated_betas)

    def test_forward_backward(self):

        # Here we construct a dummy sampler to output a sample of size 3. The sampler produces these
        # two samples and returns them in a first-in-first-out manner.
        s1 = [[-1, -1, -1],
              [-1,  1,  1],
              [1,  1,  1]]
        s2 = [[1,  -1,  1],
              [-1, -1, 1],
              [-1, -1, 1]]

        class FIFOSampler:
            def __init__(self):
                self.samples = [
                    SampleSet.from_samples((s1, list("abc")), "SPIN", [-1, 0, -2]),
                    SampleSet.from_samples((s2, list("abc")), "SPIN", [-1, -2, 0])
                ]

            def sample_ising(self, *args, **kwargs):
                print(len(self.samples))
                return self.samples.pop(0)

        ising = Ising(
            nodes="abc",
            edges=[("a", "b"), ("a", "c"), ("b", "c")],
            beta=1.0,
            sampler=FIFOSampler(),
            statistic=IsingStatistic([1], [0, 1], [1, 2]),
            sample_params=dict(),
        )

        # linear and quadratic values don't matter here because we're using a dummy sampler
        linear = torch.zeros((2, 3), requires_grad=True)
        quadratic = torch.zeros((2, 3), requires_grad=True)
        with self.subTest("Ising layer produced unexpected output values"):
            y = ising(linear, quadratic)
            torch.testing.assert_close(y, torch.tensor([[1/3, 1/3, 1],
                                                        [-1, 1/3, -1]]))

        # Gradient amounts to summing over gradients per obs
        (y**2).sum().backward()

        # Manually compute the gradients for first observation
        t1 = torch.tensor(s1).float()
        stat_1 = torch.hstack([t1[..., [1]], t1[..., [0, 1]] * t1[..., [1, 2]]])
        input_1 = torch.hstack([t1, t1[..., [0, 0, 1]] * t1[..., [1, 2, 2]]])
        grad1 = 2*y[0]@(-torch.cat([stat_1, input_1], -1).mT.cov()[:3, 3:])

        # Manually compute the gradients for second observation
        t2 = torch.tensor(s2).float()
        stat_2 = torch.hstack([t2[..., [1]], t2[..., [0, 1]] * t2[..., [1, 2]]])
        input_2 = torch.hstack([t2, t2[..., [0, 0, 1]] * t2[..., [1, 2, 2]]])
        grad2 = 2*y[1]@(-torch.cat([stat_2, input_2], -1).mT.cov()[:3, 3:])

        grad = torch.vstack([grad1, grad2])

        with self.subTest("Linear gradients should match"):
            torch.testing.assert_close(grad[:, :3], linear.grad)

        with self.subTest("Quadratic gradients should match"):
            torch.testing.assert_close(grad[:, 3:], quadratic.grad)


if __name__ == "__main__":
    unittest.main()
