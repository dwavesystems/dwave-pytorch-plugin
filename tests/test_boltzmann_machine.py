# Copyright 2025 D-Wave
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
from dimod import SPIN, BinaryQuadraticModel, IdentitySampler, SampleSet
from parameterized import parameterized

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple

from dwave.plugins.torch.models.boltzmann_machine import RestrictedBoltzmannMachine as RBM

class TestGraphRestrictedBoltzmannMachine(unittest.TestCase):
    def setUp(self) -> None:
        # Create a triangle graph with an additional dangling vertex
        #       a
        #     / | \
        #    b--c  d
        # Note the node order is deliberately "dbac" in order to test variable orderings
        self.nodes = list("dbac")
        self.edges = [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]
        self.n = 4

        # Manually set the parameter weights for testing
        dtype = torch.float32
        h = [0.0, 1, 2, 3]

        bm = GRBM(self.nodes, self.edges)
        bm._linear.data = torch.tensor(h, dtype=dtype)
        bm._quadratic.data = torch.tensor([1, 2, 3, 6], dtype=dtype)

        self.bm = bm

        self.ones = torch.ones(4).unsqueeze(0)
        self.mones = -torch.ones(4).unsqueeze(0)
        self.pmones = torch.tensor([[1, -1, 1, -1]], dtype=dtype)
        self.mpones = torch.tensor([[-1, 1, -1, 1]], dtype=dtype)

        self.sample_1 = torch.vstack([self.ones, self.ones, self.ones, self.pmones])
        self.sample_2 = torch.vstack([self.ones, self.ones, self.ones, self.mpones])
        return super().setUp()

    def test_constructor(self):
        self.assertListEqual(list("dbac"), self.bm._nodes)
        self.assertListEqual(
            [self.bm._idx_to_node[i] for i in range(self.bm._n_nodes)], self.bm._nodes
        )
        self.assertRaises(NotImplementedError, GRBM, [0, 1, 2], [[0, 1]], [0, 1])
        # Create a triangle graph with an additional dangling vertex
        #       a
        #     / | \
        #    b--c  d
        # Note the node order is deliberately "dbac" in order to test variable orderings
        self.nodes = list("dbac")
        self.edges = [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]
        w1 = 13337.14
        w2 = 4812.23
        bm = GRBM(self.nodes, self.edges, None, {"a": w1}, {("b", "c"): w2})
        self.assertAlmostEqual(bm.linear[2].item(), w1, 2)
        self.assertAlmostEqual(bm.quadratic[3].item(), w2, 2)

    def test_quadratic(self):
        self.bm.set_quadratic({("d", "b"): 999})
        self.assertEqual(999, self.bm.quadratic[0])
        self.bm.set_quadratic({})

    def test_set_linear(self):
        self.bm.set_linear({"d": 999})
        self.assertEqual(999, self.bm.linear[0])
        self.bm.set_linear({})

    def test_forward(self):
        # Model for reference:
        #       a
        #     / | \
        #    b--c  d
        # Linear biases for reference:
        # 0 1 2 3
        # d b a c
        # Edge list and weights for reference:
        # [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]
        #       1           2           3           6
        with self.subTest("Manually-computed energies"):
            self.assertEqual(18, self.bm(self.ones).item())
            self.assertEqual(6, self.bm(self.mones).item())
            # 1 -1 1 -1
            # d  b a  c
            self.assertEqual(4, self.bm(self.pmones).item())
            # -1 1 -1 1
            #  d b  a c
            self.assertEqual(8, self.bm(self.mpones).item())
            self.assertListEqual([18, 18, 18, 4], self.bm(self.sample_1).tolist())

        with self.subTest(
            "Arbitrary-valued weights and spins should match dimod.BQM energy"
        ):
            self.bm._linear.data = torch.linspace(-412, 23, 4)
            new_J = torch.linspace(-0.4, 4, 4**2)
            self.bm._quadratic.data = new_J[: len(self.bm._quadratic)]

            bqm = BinaryQuadraticModel.from_ising(*self.bm.to_ising(1))

            fake_spins = 1.0 * torch.arange(1, 5).unsqueeze(0)

            en_bqm = bqm.energies((fake_spins.numpy(), "dbac")).item()
            en_boltz = self.bm(fake_spins).item()
            self.assertAlmostEqual(en_bqm, en_boltz, 4)

    def test_estimate_beta(self):
        spins = torch.tensor(
            [[1, -1, 1, 1], [-1, -1, 1, 1], [1, -1, -1, 1], [1, 1, 1, -1]]
        )
        bqm = BinaryQuadraticModel.from_ising(*self.bm.to_ising(1))
        self.assertEqual(
            1.0 / mple(bqm, (spins.numpy(), "dbac"))[0],
            self.bm.estimate_beta(spins),
        )

    def test_pad(self):
        grbm = GRBM([0, 1, 2], [(0, 1), (0, 2), (1, 2)], [1])
        x = torch.zeros((99, 2))
        padded = grbm._pad(x)
        self.assertTrue(padded[:, 1].isnan().all())

    def test_compute_effective_field(self):
        grbm = GRBM([0, 1, 2], [(0, 1), (0, 2), (1, 2)], [2])
        # Note : In the diagram below linear biases are shown using  <>
        #        quadratic biases using (), and spin value of visibles using []
        #               (0.13)
        # Model: 2 <.4> ------- 0 [-1]
        #         \           /
        #   (-0.17)\         /(-0.7)
        #           \ 1 [1] /
        # effective field = quadratic(0,2) * [-1] + quadratic(1,2) * [1]+ linear(2)
        #                 = 0.13 * [-1] - 0.17 * [1] + 0.4 = 0.1
        grbm._linear.data = torch.tensor([-0.1, -0.2, 0.4])
        grbm._quadratic.data = torch.tensor([-0.7, 0.13, -0.17])
        padded = torch.tensor([[-1.0, 1.0, float("nan")]])
        h_eff = grbm._compute_effective_field(padded)
        self.assertAlmostEqual(h_eff.item(), 0.1)

    def test_compute_effective_field_unordered(self):
        grbm = GRBM([0, 3, 2, 1], [(1, 3), (0, 1), (0, 3), (0, 2), (1, 2)], [3, 2])
        # Note : In the diagram below linear biases are shown using  <>
        #        quadratic biases using (), and spin value of visibles using []
        #              (0.13)         (0.15)
        # Model: 2 <.4> ----- 0 [-1] -------- 3 <-0.2>
        #         \           |             /
        #   (-0.17)\          |(-0.7)      / -(0.15)
        #           \         |           /
        #            -------  1 [1] ---  /

        # effective field [3] = quadratic(0,3) * [-1] + quadratic(1,3) * [1] + linear(3)
        #                 =  0.15 * [-1] - 0.15 * [1] - 0.2 = -.5

        # effective field [2] = quadratic(0,2) * [-1] + quadratic(1,2) * [1] + linear(2)
        #                 =  0.13 * [-1] - 0.17 * [1] + 0.4 = .1

        grbm._linear.data = torch.tensor([-0.1, -0.2, 0.4, 0.2])
        grbm._quadratic.data = torch.tensor([-.15, -0.7, 0.15, 0.13, -0.17])
        padded = torch.tensor([[-1.0, float("nan"), float("nan"), 1.0]])
        h_eff = grbm._compute_effective_field(padded)
        self.assertTrue(torch.allclose(h_eff.data, torch.tensor([-0.5000, 0.1000]), atol=1e-6))

    def test_compute_expectation_disconnected(self):
        grbm = GRBM(list("acb"), [("a", "b"), ("a", "c"), ("b", "c")], ["c"])
        #         (0.13)
        # Model: c ----- a
        #         \      |
        #  (-0.17) \     |  (-0.7)
        #           \ b /
        grbm._linear.data = torch.tensor([-0.1, 0.4, -0.2])
        grbm._quadratic.data = torch.tensor([-0.7, 0.13, -0.17])
        obs = torch.tensor([[-1.0, 1.0]])
        expected = grbm._compute_expectation_disconnected(obs).tolist()
        # effective field = -quadratic(a,c) + quadratic(b,c) + linear(c)
        #                 = -0.13 - 0.17 + 0.4 = 0.1
        # expectation = tanh(effective field) = tanh(0.1)
        torch.testing.assert_close(expected, [[-1.0, torch.tanh(torch.tensor(-0.1)).item(), 1.0]])

    @parameterized.expand([
        (
            torch.ones(1, 4),
            [[1]*8]
        ),
        (  # Same values as above, but with padded dimensions
            torch.ones(1, 1, 1, 4),
            [[[[1]*8]]]
        ),
        (
            torch.vstack([torch.ones(4), -torch.ones(4)]),
            [[1]*8, [-1]*4+[1]*4]
        ),
        (
            torch.tensor([[1, -1, 1, -1]]),
            [[1, -1, 1, -1, -1, -1, 1, 1]]
        )
    ])
    def test_sufficient_statistics(self, x, answer):
        # Model for reference:
        #       a
        #     / | \
        #    b--c  d
        # Edge list for reference:
        # [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]
        t0 = self.bm.sufficient_statistics(x)
        self.assertListEqual(t0.tolist(), answer)

    def test_interactions(self):
        # Model for reference:
        #       a
        #     / | \
        #    b--c  d
        # Edge list for reference:
        # [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]
        self.assertListEqual(
            #                                   d    b    a    c
            self.bm.interactions(torch.tensor([[0.0, 3.0, 2.0, 1.0]])).tolist(),
            [[6.0, 2.0, 0, 3.0]],
        )
        all_ones = [[1, 1, 1, 1]]
        self.assertListEqual(self.bm.interactions(self.ones).tolist(), all_ones)
        self.assertListEqual(self.bm.interactions(self.ones).tolist(), all_ones)
        self.assertListEqual(self.bm.interactions(self.mones).tolist(), all_ones)
        # d  b a  c
        # 1 -1 1 -1
        mmpp = [[-1.0, -1, 1, 1]]
        self.assertListEqual(self.bm.interactions(self.pmones).tolist(), mmpp)
        #  d b  a c
        # -1 1 -1 1
        self.assertListEqual(self.bm.interactions(self.mpones).tolist(), mmpp)

    def test_to_ising(self):
        h_true = torch.tensor([-3, 0, 1, 3.0])
        J_true = torch.tensor([-1, 1, 2.0, 0])
        self.bm._linear.data = h_true
        self.bm._quadratic.data = J_true

        with self.subTest("Ising dictionaries without unbounded bias ranges"):
            h, J = self.bm.to_ising(1)
            h_list = list(h.values())
            J_list = [J[a, b] for a, b in self.edges]

            self.assertListEqual(h_list, h_true.tolist())
            self.assertListEqual(J_list, J_true.tolist())

        with self.subTest("Ising dictionaries with bounded bias ranges"):
            h, J = self.bm.to_ising(1, [-0.1, 1.5], [-0.05, 3])
            h_list = list(h.values())
            J_list = [J[a, b] for a, b in self.edges]

            for x_true, x_observed in zip([-0.1, 0, 1, 1.5], h_list):
                self.assertAlmostEqual(x_true, x_observed)
            for x_true, x_observed in zip([-0.05, 1, 2, 0], J_list):
                self.assertAlmostEqual(x_true, x_observed)

    def test_approximate_expectation_sampling(self):
        grbm = GRBM(list("acb"), [("a", "b"), ("a", "c"), ("b", "c")], ["c"])
        #         (0.13)
        # Model: c ----- a
        #         \      |
        #  (-0.17) \     |  (-0.7)
        #           \ b /
        grbm._linear.data = torch.tensor([-0.1, 0.4, -0.2])
        grbm._quadratic.data = torch.tensor([-0.7, 0.13, -0.17])
        obs = torch.tensor([[-1.0, 1.0], [-1.0, -1.0]])
        sampler = IdentitySampler()
        prefactor = 999

        fake_samples = ([[-1], [1]], ["c"])
        expectation = grbm._approximate_expectation_sampling(
            obs, sampler, prefactor, sample_kwargs=dict(initial_states=fake_samples)).tolist()
        self.assertListEqual(expectation, [[-1, 0.0, 1], [-1, 0.0, -1]])

        fake_samples = ([[1], [1]], ["c"])
        expectation = grbm._approximate_expectation_sampling(
            obs, sampler, prefactor, sample_kwargs=dict(initial_states=fake_samples)).tolist()
        self.assertListEqual(expectation, [[-1, 1, 1.0], [-1, 1, -1.0]])

    def test_sampleset_to_tensor(self):
        grbm = GRBM(list("cabd"), ["ab", "ac", "bc"])
        bogus_energy = [999] * 3
        spins_in = [[1, -1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ss = SampleSet.from_samples((spins_in, list("dbca")), SPIN, bogus_energy)
        spins = grbm.sampleset_to_tensor(ss)
        self.assertTupleEqual((3, 4), tuple(spins.shape))
        self.assertIsInstance(spins, torch.Tensor)
        # Test variable ordering is respected
        self.assertListEqual(
            spins.tolist(), [[1, 1, -1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )

    def test_sample(self):
        grbm = GRBM(list("abcd"), [("a", "b")])
        spins = grbm.sample(
            IdentitySampler(),
            prefactor=1,
            linear_range=None, quadratic_range=None,
            sample_params=dict(
                initial_states=([[1, 1, 1, 1], [1, 1, 1, 1], [-1, -1, 1, -1]], "abcd")
            ),
            as_tensor=True,
        )
        self.assertTupleEqual((3, 4), tuple(spins.shape))
        self.assertIsInstance(spins, torch.Tensor)

    def test_sample_return_sampleset(self):
        grbm = GRBM(list("abcd"), [("a", "b")])
        sampleset = grbm.sample(
            IdentitySampler(),
            prefactor=1,
            linear_range=None, quadratic_range=None,
            sample_params=dict(
                initial_states=([[1, 1, 1, 1], [1, 1, 1, 1], [-1, -1, 1, -1]], "abcd")
            ),
            as_tensor=False,
        )
        self.assertIsInstance(sampleset, SampleSet)

        self.assertEqual(3, len(sampleset.samples()))
        self.assertEqual(4, len(sampleset.variables))
        self.assertEqual(set(grbm.nodes), set(sampleset.variables))

    def test_quasi_objective(self):
        # Create a triangle graph with an additional dangling vertex
        self.nodes = list("abcd")
        self.edges = [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]
        self.n = 4

        # Manually set the parameter weights for testing
        dtype = torch.float32
        h = [0.0, 1, 2, 3]

        grbm = GRBM(self.nodes, self.edges)
        grbm._linear.data = torch.tensor(h, dtype=dtype)
        grbm._quadratic.data = torch.tensor([1, 2, 3, 6], dtype=dtype)

        # Test the gradient matches
        ones = torch.ones((1, 4))
        mones = -ones
        with self.subTest("Test gradients"):
            obj = grbm.quasi_objective(ones, mones)
            obj.backward()
            t1 = grbm.sufficient_statistics(ones)
            t2 = grbm.sufficient_statistics(mones)
            grad_auto = grbm._linear.grad.tolist() + grbm._quadratic.grad.tolist()
            self.assertListEqual(grad_auto, (t1.mean(0) - t2.mean(0)).tolist())

        pmones = torch.tensor([[1, -1, 1, -1]], dtype=dtype)
        mpones = torch.tensor([[-1, 1, -1, 1]], dtype=dtype)
        with self.subTest("Test objective value matches"):
            s1 = torch.vstack([ones, ones, ones, pmones])
            s2 = torch.vstack([ones, ones, ones, mpones])
            s3 = torch.vstack([s2, s2])
            self.assertEqual(-1, grbm.quasi_objective(s1, s2).item())
            self.assertEqual(-1, grbm.quasi_objective(s1, s3))

    def test_quasi_objective_gradient_hidden_units(self):
        grbm = GRBM([1, 2, 3],
                    [(1, 2), (1, 3), (2, 3)],
                    [1],
                    {1: 0.2, 2: 0.2, 3: 0.3},
                    {(1, 2): 0.2, (1, 3): 0.3, (2, 3): 0.6})
        # Note : In the digram bellow linear biases are shown using  <>
        #        quadratic biases using ()
        #                 (0.2)
        # Model:  v1 <0.2> ----- v2  <0.2>
        #           \           /
        #     (0.3)  \         / (0.6)
        #             \       /
        #                v3 <0.3>
        s_observed = torch.tensor([[1.0, -1.0]])
        s_model = torch.tensor([[1.0, -1.0, 1.0]])
        quasi = grbm.quasi_objective(s_observed, s_model, "exact-disc")
        quasi.backward()
        # Compute gradients manually
        # Compute unnormalized density
        #                            h1v1 + h2v2 + h3v3 + J23v2v3 + J12v1v2 + J13v1v3
        q_plus = torch.exp(-torch.tensor(0.2 + 0.2 - 0.3 - 0.6 + 0.2 - 0.3))
        q_minus = torch.exp(-torch.tensor(-0.2 - 0.2 + 0.3 - 0.6 + 0.2 - 0.3))
        # Normalize it
        z_cond = q_plus + q_minus
        p_plus = q_plus / z_cond
        p_minus = q_minus / z_cond
        # t = sufficient statistics = (v1 v2 v3 v1v2 v1v3 v2v3)
        t_plus = torch.tensor([1,  1, -1,  1, -1, -1]).float()
        t_minus = torch.tensor([-1, 1, -1, -1,  1, -1]).float()
        t_model = torch.tensor([1, -1,  1, -1,  1, -1]).float()
        # Compute expected stat
        t_cond = t_plus*p_plus + t_minus*p_minus
        grad = t_cond - t_model
        grad_auto = torch.cat([grbm.linear.grad, grbm.quadratic.grad])
        # NOTE: this test relied on the hidden units being disconnected. This assumption gives rise
        # to linearity in expectation of sufficient statistics, i.e., average spin, then calculating
        # the sufficient statistics of the average spins.
        torch.testing.assert_close(grad, grad_auto)

class TestRBM(unittest.TestCase):

    def setUp(self):
        # Small RBM for testing
        self.rbm = RBM(n_visible=4, n_hidden=3)

        # Common input data for CD tests
        self.batch = torch.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]])

        # Shared CD kwargs
        self.cd_kwargs = dict(
            epoch=0,
            n_gibbs_steps=1,
            learning_rate=0.1,
            momentum_coefficient=0.5,
            weight_decay=0.0,
            n_epochs=10,
        )

    def test_sample_hidden_shape(self):
        visible = torch.randint(0, 2, (5, self.rbm.n_visible)).float()
        hidden = self.rbm.sample_hidden(visible)
        # Ensure shape is correct
        self.assertEqual(hidden.shape, (5, self.rbm.n_hidden))

    def test_sample_hidden_binary(self):
        visible = torch.randint(0, 2, (5, self.rbm.n_visible)).float()
        hidden = self.rbm.sample_hidden(visible)

        # Ensure output is binary
        self.assertTrue(torch.all((hidden == 0) | (hidden == 1)))

    @parameterized.expand(
        [
            ("all_ones", 1000.0, 1),
            ("all_zeroes", -1000.0, 0),
        ]
    )
    def test_sample_hidden_saturation(self, name, bias_value, expected_value):
        """
        Test that sample_hidden saturates correctly when the hidden biases
        are set to very large positive or negative values.

        If hidden_bias[j] → +∞
            sigmoid(hidden_bias + visible @ weights) → 1
            bernoulli(1) → always 1

        If hidden_bias[j] → -∞
            sigmoid(hidden_bias + visible @ weights) → 0
            bernoulli(0) → always 0
        """

        # Set all hidden biases to an extreme constant
        with torch.no_grad():
            self.rbm._hidden_biases.fill_(bias_value)

        # The visible input does not matter in saturation conditions
        visible = torch.zeros(5, self.rbm.n_visible)

        # Sample hidden units
        hidden = self.rbm.sample_hidden(visible)

        # Assert that all hidden units match the expected saturated value
        self.assertTrue(torch.all(hidden == expected_value))

    def test_sample_visible_shape(self):
        hidden = torch.randint(0, 2, (5, self.rbm.n_hidden)).float()
        visible = self.rbm.sample_visible(hidden)

        # Ensure shape is correct
        self.assertEqual(visible.shape, (5, self.rbm.n_visible))

    def test_sample_visible_binary(self):
        hidden = torch.randint(0, 2, (5, self.rbm.n_hidden)).float()
        visible = self.rbm.sample_visible(hidden)

        # Ensure output is binary
        self.assertTrue(torch.all((visible == 0) | (visible == 1)).item())

    @parameterized.expand(
        [
            ("all_ones", 1000.0, 1),
            ("all_zeroes", -1000.0, 0),
        ]
    )
    def test_sample_visible_saturation(self, name, bias_value, expected_value):
        """
        Test that sample_visible saturates correctly when the visible biases
        are set to very large positive or negative values.

        If visible_bias → +∞: sigmoid → 1 → bernoulli(1) → always 1
        If visible_bias → -∞: sigmoid → 0 → bernoulli(0) → always 0
        """

        # Large positive/negative bias makes sigmoid output deterministic
        with torch.no_grad():
            self.rbm._visible_biases.fill_(bias_value)

        # Hidden input doesn't matter when biases dominate
        hidden = torch.zeros(5, self.rbm.n_hidden)

        visible = self.rbm.sample_visible(hidden)

        self.assertTrue(torch.all(visible == expected_value).item())

    def test_forward_scalar_output(self):
        """Forward should return a scalar tensor."""
        batch = torch.randn(5, self.rbm.n_visible)
        out = self.rbm.forward(batch)
        self.assertIsInstance(out, torch.Tensor)
        self.assertEqual(out.ndim, 0)

    def test_forward_zero_weights_biases(self):
        """
        Check free energy when all weights and biases are zero.
        Analytic test: all weights & biases = 0
            Free energy becomes:
            F(v) = - sum_j softplus(0) = -n_hidden * log(2)
        """
        with torch.no_grad():
            self.rbm._weights[:] = 0
            self.rbm._visible_biases[:] = 0
            self.rbm._hidden_biases[:] = 0
        v = torch.tensor([[1.0, 0.0, 1.0, 1.0]])  # value doesn't matter
        out = self.rbm.forward(v)
        expected = -self.rbm.n_hidden * torch.log(torch.tensor(2.0))
        self.assertTrue(torch.allclose(out, expected))

    def test_forward_ordering_bias(self):
        """
        Free energy ordering test:
            If visible_bias is very positive, visible=1 must yield
            much lower free energy than visible=0.
        """
        # Create a tiny RBM for easy testing
        rbm = RBM(n_visible=1, n_hidden=1)
        with torch.no_grad():
            rbm._weights[:] = 0
            rbm._hidden_biases[:] = 0
            rbm._visible_biases[:] = 1000.0

        f1 = rbm.forward(torch.tensor([[1.0]]))
        f0 = rbm.forward(torch.tensor([[0.0]]))
        self.assertLess(f1, f0)

    def test_forward_small_numeric_case(self):
        """Check free energy against manual calculation for 1 visible and 1 hidden unit."""
        rbm = RBM(n_visible=1, n_hidden=1)
        with torch.no_grad():
            rbm._weights[:] = 2.0
            rbm._visible_biases[:] = 1.0
            rbm._hidden_biases[:] = -1.0

        v = torch.tensor([[1.0]])
        # Independent manual calculation
        expected = -1.0 - torch.nn.functional.softplus(torch.tensor(1.0))
        out = rbm.forward(v)
        self.assertTrue(torch.allclose(out, expected))
    
if __name__ == "__main__":
    unittest.main()
