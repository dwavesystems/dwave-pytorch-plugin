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

import copy
import itertools
import pickle
import unittest

import torch

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.utils import to_bqm
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple
from tests.helper_functions import model_to_bqm, randspins, set_weights


def random_model(n: int, p: float, n_hidden: int = 0, seed: int = 0, connect_hidden=False) -> GRBM:
    """A model on a random graph with random weights; the last ``n_hidden`` nodes are hidden."""
    generator = torch.Generator().manual_seed(seed)
    hidden = set(range(n - n_hidden, n))
    edges = [
        (i, j) for i in range(n) for j in range(i + 1, n)
        if torch.rand((), generator=generator) < p
        and (connect_hidden or not (i in hidden and j in hidden))
    ]
    model = GRBM(range(n), edges, sorted(hidden) or None)
    set_weights(
        model,
        torch.randn(n, generator=generator),
        torch.randn(len(edges), generator=generator),
    )
    return model


class TestGraphRestrictedBoltzmannMachine(unittest.TestCase):
    def setUp(self) -> None:
        # Create a triangle graph with an additional dangling vertex
        #       a
        #     / | \
        #    b--c  d
        # Note the node order is deliberately "dbac" in order to test variable orderings
        self.nodes = list("dbac")
        self.edges = [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]

        # Linear biases (d b a c) and quadratic biases (ab ac ad bc)
        self.bm = GRBM(
            self.nodes, self.edges,
            linear=dict(zip(self.nodes, [0.0, 1.0, 2.0, 3.0])),
            quadratic={("a", "b"): 1.0, ("a", "c"): 2.0, ("a", "d"): 3.0, ("b", "c"): 6.0},
        )

        self.ones = torch.ones(4).unsqueeze(0)
        self.mones = -torch.ones(4).unsqueeze(0)
        self.pmones = torch.tensor([[1, -1, 1, -1]], dtype=torch.float32)
        self.mpones = torch.tensor([[-1, 1, -1, 1]], dtype=torch.float32)

    # ------------------------------------------------------------------ construction ----------

    def test_constructor(self):
        bm = self.bm
        self.assertTupleEqual(bm.nodes, tuple(self.nodes))
        self.assertTupleEqual(bm.edges, tuple(tuple(e) for e in self.edges))
        self.assertDictEqual(dict(bm.node_to_idx), {"d": 0, "b": 1, "a": 2, "c": 3})
        self.assertListEqual([bm.nodes[bm.node_to_idx[v]] for v in self.nodes], self.nodes)
        self.assertEqual((4,), tuple(bm.quadratic.shape))
        self.assertEqual((4,), tuple(bm.linear.shape))
        self.assertEqual(4, bm.n_nodes)
        self.assertEqual(4, bm.n_edges)
        self.assertEqual(4, bm.n_visible)
        self.assertEqual(0, bm.n_hidden)
        self.assertTupleEqual(bm.visible_nodes, tuple(self.nodes))
        self.assertFalse(bm.connected_hidden)
        self.assertIn("n_nodes=4, n_edges=4, n_hidden=0", repr(bm))

        with self.subTest("Quadratic biases are stored per edge, in the order of the edges"):
            torch.testing.assert_close(bm.quadratic, torch.tensor([1.0, 2.0, 3.0, 6.0]))
            # ("a", "d") has indices (2, 0); its endpoints are indexed in canonical order
            self.assertListEqual(bm.edge_idx_i.tolist(), [1, 2, 0, 1])
            self.assertListEqual(bm.edge_idx_j.tolist(), [2, 3, 2, 3])
            self.assertEqual(2, bm.edge_to_idx[("a", "d")])
            self.assertEqual(2, bm.edge_to_idx[("d", "a")])
            matrix = bm.quadratic_matrix(bm.quadratic)
            self.assertEqual((4, 4), tuple(matrix.shape))
            self.assertEqual(3.0, matrix[0, 2].item())
            self.assertEqual(3.0, matrix[2, 0].item())

        with self.subTest("Constructor weights"):
            w1, w2 = 13337.14, 4812.23
            bm = GRBM(self.nodes, self.edges, None, {"a": w1}, {("c", "b"): w2})
            self.assertAlmostEqual(bm.linear[2].item(), w1, 2)
            self.assertAlmostEqual(bm.quadratic[3].item(), w2, 2)

    def test_public_parameters_and_buffers(self):
        bm = GRBM([0, 1, 2], [(0, 1), (1, 2)], hidden_nodes=[2])
        self.assertListEqual(["linear", "quadratic"], [name for name, _ in bm.named_parameters()])
        self.assertListEqual(
            ["edge_idx_i", "edge_idx_j", "visible_idx", "hidden_idx"],
            [name for name, _ in bm.named_buffers()],
        )
        # The index buffers are derived from the constructor arguments and are not saved
        self.assertSetEqual({"linear", "quadratic"}, set(bm.state_dict()))
        with self.assertRaises(TypeError):
            bm.linear = torch.zeros(3)  # torch refuses to replace a parameter by a tensor

    def test_graph_attributes(self):
        bm = GRBM([0, 1, 2], [(0, 1), (1, 2)], hidden_nodes=[2])
        for name in ("nodes", "edges", "hidden_nodes", "visible_nodes"):
            with self.subTest(name):
                self.assertIsInstance(getattr(bm, name), tuple)
        self.assertDictEqual({0: 0, 1: 1, 2: 2}, bm.node_to_idx)
        self.assertDictEqual({(0, 1): 0, (1, 0): 0, (1, 2): 1, (2, 1): 1}, bm.edge_to_idx)
        self.assertFalse(hasattr(bm, "idx_to_node"))
        self.assertFalse(hasattr(bm, "adjacency"))

    def test_deepcopy_and_pickle(self):
        # Common PyTorch workflows (e.g. averaged models) deep-copy or pickle whole modules
        bm = GRBM([0, 1, 2], [(0, 1), (1, 2)], hidden_nodes=[2], linear={1: 0.5})
        for label, clone in (("deepcopy", copy.deepcopy(bm)),
                             ("pickle", pickle.loads(pickle.dumps(bm)))):
            with self.subTest(label):
                self.assertTupleEqual(bm.nodes, clone.nodes)
                self.assertTupleEqual(bm.hidden_nodes, clone.hidden_nodes)
                self.assertDictEqual(bm.node_to_idx, clone.node_to_idx)
                torch.testing.assert_close(bm.linear, clone.linear)
                torch.testing.assert_close(bm.quadratic, clone.quadratic)
                self.assertTrue(torch.equal(bm.edge_idx_i, clone.edge_idx_i))

    def test_default_quadratic_initialization_uses_connectivity(self):
        nodes = list("abcd")
        edges = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c")]
        degrees = torch.tensor([3.0, 2.0, 2.0, 1.0])
        edge_idx_i = torch.tensor([0, 0, 0, 1])
        edge_idx_j = torch.tensor([1, 2, 3, 2])
        # Standard deviation 1 / (T_0 (deg u deg v)^(1/4)) with the default T_0 = 4
        expected_std = 1 / (4 * (degrees[edge_idx_i] * degrees[edge_idx_j])**0.25)

        torch.manual_seed(1234)
        expected_quadratic = torch.randn(len(edges)) * expected_std

        torch.manual_seed(1234)
        bm = GRBM(nodes, edges)

        torch.testing.assert_close(bm.linear, torch.zeros(len(nodes)))
        torch.testing.assert_close(bm.quadratic, expected_quadratic)

        with self.subTest("The initial temperature scales the couplings inversely"):
            torch.manual_seed(1234)
            bm = GRBM(nodes, edges, init_temperature=2.0)
            torch.testing.assert_close(bm.quadratic, 2 * expected_quadratic)

        with self.subTest("The default reproduces Hinton's 0.01 on a 784-by-500 machine"):
            # Every visible unit has degree 500 and every hidden unit degree 784
            self.assertAlmostEqual(1 / (4 * (784 * 500) ** 0.25), 0.01, places=4)

        with self.assertRaisesRegex(ValueError, "must be positive"):
            GRBM(nodes, edges, init_temperature=0.0)

    def test_default_quadratic_initialization_edgeless(self):
        bm = GRBM([0, 1, 2], [])
        torch.testing.assert_close(bm.linear, torch.zeros(3))
        torch.testing.assert_close(bm.quadratic, torch.zeros(0))
        self.assertEqual(0, bm.n_edges)
        torch.testing.assert_close(bm(torch.ones(2, 3)), torch.zeros(2))

    def test_invalid_graphs(self):
        with self.assertRaisesRegex(ValueError, "Self-loops are not allowed"):
            GRBM(list("dbac"), [["a", "a"], ["a", "c"], ["a", "d"], ["b", "c"]])
        with self.assertRaisesRegex(ValueError, "Duplicate edges"):
            GRBM(list("abc"), [("a", "b"), ("b", "a")])
        with self.assertRaisesRegex(ValueError, "Duplicate edges"):
            GRBM(list("abc"), [("a", "b"), ("a", "b")])
        with self.assertRaisesRegex(ValueError, "duplicate entries"):
            GRBM(list("aab"), [("a", "b")])
        with self.assertRaisesRegex(ValueError, "not a node"):
            GRBM(list("ab"), [("a", "z")])
        with self.assertRaisesRegex(ValueError, "Hidden nodes .* are not nodes"):
            GRBM(list("ab"), [("a", "b")], hidden_nodes=["z"])
        with self.assertRaisesRegex(ValueError, "`hidden_nodes` contains duplicate"):
            GRBM(list("ab"), [("a", "b")], hidden_nodes=["a", "a"])

    def test_set_quadratic(self):
        # Reversed orientation of edge ("a", "b"); stored at [idx(b), idx(a)] = [1, 2]
        self.bm.set_quadratic({("b", "a"): 999})
        self.assertEqual(999, self.bm.quadratic[0].item())
        self.bm.set_quadratic({})
        with self.subTest("Several edges in mixed orientations"):
            self.bm.set_quadratic({("c", "a"): 7, ("a", "d"): 8, ("b", "c"): 9})
            torch.testing.assert_close(self.bm.quadratic, torch.tensor([999.0, 7.0, 8.0, 9.0]))

    def test_set_quadratic_unknown_edge(self):
        quadratic = self.bm.quadratic.detach().clone()
        with self.assertRaisesRegex(ValueError, r"Edge \('d', 'b'\) is not in the model"):
            self.bm.set_quadratic({("d", "b"): 999})
        with self.subTest("Self-loops and unknown nodes are not edges"):
            with self.assertRaisesRegex(ValueError, r"Edge \('a', 'a'\) is not in the model"):
                self.bm.set_quadratic({("a", "a"): 999})
            with self.assertRaisesRegex(ValueError, r"Edge \('a', 'z'\) is not in the model"):
                self.bm.set_quadratic({("a", "z"): 999})
        with self.subTest("A rejected call leaves every bias unchanged"):
            with self.assertRaisesRegex(ValueError, r"Edge \('d', 'c'\) is not in the model"):
                self.bm.set_quadratic({("a", "b"): 999, ("d", "c"): 999})
        torch.testing.assert_close(self.bm.quadratic, quadratic)

    def test_set_linear(self):
        self.bm.set_linear({"d": 999})
        self.assertEqual(999, self.bm.linear[0].item())
        with self.assertRaisesRegex(ValueError, "Node 'z' is not in the model"):
            self.bm.set_linear({"z": 1.0})
        self.bm.set_linear({})

    # ------------------------------------------------------------------ energies --------------

    def test_forward(self):
        # Linear biases for reference:
        # 0 1 2 3
        # d b a c
        # Edge list and weights for reference:
        # [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]
        #       1           2           3           6
        with self.subTest("Manually-computed energies"):
            self.assertEqual(18, self.bm(self.ones).item())
            self.assertEqual(6, self.bm(self.mones).item())
            self.assertEqual(4, self.bm(self.pmones).item())
            self.assertEqual(8, self.bm(self.mpones).item())
            batch = torch.vstack([self.ones, self.ones, self.ones, self.pmones])
            self.assertListEqual([18, 18, 18, 4], self.bm(batch).tolist())

        with self.subTest("Arbitrary leading dimensions"):
            energies = self.bm(batch.reshape(2, 2, 4))
            self.assertEqual((2, 2), tuple(energies.shape))
            self.assertListEqual([18, 18, 18, 4], energies.flatten().tolist())

        with self.subTest("Arbitrary-valued weights and spins should match dimod.BQM energy"):
            set_weights(self.bm, torch.linspace(-412, 23, 4), torch.linspace(-0.4, 4, 16)[:4])
            bqm = model_to_bqm(self.bm)
            fake_spins = 1.0 * torch.arange(1, 5).unsqueeze(0)
            en_bqm = bqm.energies((fake_spins.numpy(), "dbac")).item()
            self.assertAlmostEqual(en_bqm, self.bm(fake_spins).item(), 4)

    def test_forward_matches_dimod_random_graph(self):
        model = random_model(30, 0.3, seed=7)
        bqm = model_to_bqm(model)
        x = randspins(17, 30, seed=1)
        expected = torch.tensor(bqm.energies((x.numpy(), model.nodes)), dtype=torch.float32)
        torch.testing.assert_close(model(x), expected)

    def test_coupling(self):
        bm = self.bm
        matrix = bm.quadratic_matrix(bm.quadratic)
        torch.testing.assert_close(matrix, matrix.T)
        torch.testing.assert_close(matrix.diagonal(), torch.zeros(4))
        torch.testing.assert_close(matrix[bm.edge_idx_i, bm.edge_idx_j], bm.quadratic)
        torch.testing.assert_close(matrix[bm.edge_idx_j, bm.edge_idx_i], bm.quadratic)
        self.assertEqual(8, int((matrix != 0).sum()))

        with self.subTest("Couplings of other quadratic biases"):
            torch.testing.assert_close(bm.quadratic_matrix(2 * bm.quadratic), 2 * matrix)

        with self.subTest("Only the per-edge biases are stored"):
            self.assertEqual(bm.n_nodes + bm.n_edges, sum(p.numel() for p in bm.parameters()))
            self.assertFalse(hasattr(bm, "coupling"))

    def test_effective_field(self):
        # nodes d b a c; fields h_k + sum_l J_kl s_l
        spins = torch.tensor([[1.0, 1.0, -1.0, -1.0],
                              [-1.0, -1.0, 1.0, -1.0]])
        expected = torch.tensor([
            # d: 0 + J_ad s_a       b: 1 + J_ab s_a + J_bc s_c      a: 2 + J_ab s_b + J_ac s_c + J_ad s_d   c: 3 + J_ac s_a + J_bc s_b
            [0 + 3 * -1, 1 + 1 * -1 + 6 * -1, 2 + 1 * 1 + 2 * -1 + 3 * 1, 3 + 2 * -1 + 6 * 1],
            [0 + 3 * 1, 1 + 1 * 1 + 6 * -1, 2 + 1 * -1 + 2 * -1 + 3 * -1, 3 + 2 * 1 + 6 * -1],
        ], dtype=torch.float32)
        torch.testing.assert_close(self.bm.effective_field(spins), expected)

        with self.subTest("Subset of nodes"):
            idx = torch.tensor([2, 0])
            torch.testing.assert_close(self.bm.effective_field(spins, idx=idx), expected[:, [2, 0]])

        with self.subTest("Precomputed coupling matrix"):
            coupling = self.bm.quadratic_matrix(self.bm.quadratic)
            torch.testing.assert_close(self.bm.effective_field(spins, quadratic_matrix=coupling), expected)
            torch.testing.assert_close(
                self.bm.effective_field(spins, idx=idx, quadratic_matrix=coupling), expected[:, [2, 0]]
            )
            with self.assertRaisesRegex(ValueError, "not both"):
                self.bm.effective_field(spins, quadratic=self.bm.quadratic, quadratic_matrix=coupling)
            torch.testing.assert_close(self.bm.effective_field(spins, quadratic=self.bm.quadratic), expected)

        with self.subTest("Explicit biases override the parameters"):
            torch.testing.assert_close(
                self.bm.effective_field(spins, linear=torch.zeros(4)), expected - self.bm.linear
            )
            torch.testing.assert_close(
                self.bm.effective_field(spins, quadratic=torch.zeros(4)),
                self.bm.linear.expand_as(expected),
            )

        with self.subTest("Explicit biases must be batched like the model's own"):
            with self.assertRaisesRegex(ValueError, "same batch of models"):
                self.bm.effective_field(spins, linear=torch.zeros(2, 4))
            with self.assertRaisesRegex(ValueError, "same batch of models"):
                self.bm.effective_field(spins, quadratic=torch.zeros(2, 4))

        with self.subTest("NaN spins contribute nothing"):
            padded = spins.clone()
            padded[:, 2] = torch.nan  # unknown a
            expected_nan = expected.clone()
            expected_nan[:, 0] -= 3 * spins[:, 2]  # d loses J_ad s_a
            expected_nan[:, 1] -= 1 * spins[:, 2]
            expected_nan[:, 3] -= 2 * spins[:, 2]
            expected_nan[:, 2] = 2 + 1 * spins[:, 1] + 2 * spins[:, 3] + 3 * spins[:, 0]
            torch.testing.assert_close(self.bm.effective_field(padded), expected_nan)

        with self.subTest("Arbitrary leading dimensions"):
            fields = self.bm.effective_field(spins.reshape(2, 1, 4))
            self.assertEqual((2, 1, 4), tuple(fields.shape))

    def test_effective_field_is_energy_gradient(self):
        model = random_model(12, 0.5, seed=3)
        x = randspins(6, 12, seed=4).requires_grad_()
        grad, = torch.autograd.grad(model(x).sum(), x)
        torch.testing.assert_close(grad, model.effective_field(x.detach()))

    def test_sufficient_statistics(self):
        x = torch.vstack([self.ones, self.pmones, self.mpones])
        mean, second = self.bm.sufficient_statistics(x)
        torch.testing.assert_close(mean, x.mean(0))
        self.assertEqual((4,), tuple(second.shape))
        # Average products along the edges ab, ac, ad, bc (node order d b a c)
        torch.testing.assert_close(second, (x[:, [2, 2, 2, 1]] * x[:, [1, 3, 0, 3]]).mean(0))
        average_energy = mean @ self.bm.linear + second @ self.bm.quadratic
        torch.testing.assert_close(average_energy, self.bm(x).mean())

        with self.subTest("Arbitrary leading dimensions"):
            mean_3d, second_3d = self.bm.sufficient_statistics(x.reshape(3, 1, 4))
            torch.testing.assert_close(mean_3d, mean)
            torch.testing.assert_close(second_3d, second)

        with self.assertRaisesRegex(ValueError, "trailing dimension"):
            self.bm.sufficient_statistics(torch.ones(3, 5))

    # ------------------------------------------------------------------ temperature -----------

    def test_estimate_beta(self):
        spins = torch.tensor([[1, -1, 1, 1], [-1, -1, 1, 1], [1, -1, -1, 1], [1, 1, 1, -1]])
        bqm = model_to_bqm(self.bm)
        beta = self.bm.estimate_beta(spins)
        self.assertIsInstance(beta, float)
        self.assertAlmostEqual(1.0 / mple(bqm, (spins.numpy(), "dbac"))[0], beta, places=9)

    def test_fit_pseudolikelihood(self):
        # Exact samples of a mild model on the graph, by enumeration of its configurations
        truth = GRBM(
            self.nodes, self.edges,
            linear=dict(zip(self.nodes, [0.3, -0.2, 0.1, 0.4])),
            quadratic={("a", "b"): 0.5, ("a", "c"): -0.4, ("a", "d"): 0.2, ("b", "c"): -0.3},
        )
        states = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=4)))
        with torch.no_grad():
            probabilities = torch.softmax(-truth(states), 0)
        generator = torch.Generator().manual_seed(0)
        spins = states[torch.multinomial(probabilities, 20000, replacement=True, generator=generator)]

        model = GRBM(self.nodes, self.edges)
        fit = model.fit_pseudolikelihood(spins)
        self.assertTrue(fit.converged)
        self.assertEqual(torch.float64, fit.linear.dtype)
        with self.subTest("The estimate is written into the parameters, in their dtype"):
            self.assertEqual(torch.float32, model.linear.dtype)
            torch.testing.assert_close(model.linear.detach(), fit.linear.float())
            torch.testing.assert_close(model.quadratic.detach(), fit.quadratic.float())
            torch.testing.assert_close(model.linear.detach(), truth.linear.detach(), rtol=0, atol=0.08)
            torch.testing.assert_close(model.quadratic.detach(), truth.quadratic.detach(), rtol=0, atol=0.08)
        with self.subTest("The training data are at unit inverse temperature under the fit"):
            self.assertAlmostEqual(1.0, model.estimate_beta(spins), places=5)
        with self.subTest("Options are passed on"):
            with self.assertRaises(ValueError):
                model.fit_pseudolikelihood(spins, sample_chunk=0)
            ridge = model.fit_pseudolikelihood(spins, l2=0.1)
            self.assertLess(float(ridge.quadratic.norm()), float(fit.quadratic.norm()))
        with self.subTest("A model with hidden units cannot be fitted"):
            hidden = GRBM(self.nodes, self.edges, hidden_nodes=["c"])
            with self.assertRaisesRegex(ValueError, "hidden units"):
                hidden.fit_pseudolikelihood(spins)

    # ------------------------------------------------------------------ learning --------------

    def test_quasi_objective(self):
        ones = torch.ones((1, 4))
        mones = -ones
        with self.subTest("Test gradients"):
            objective = self.bm.quasi_objective(ones, mones)
            self.assertEqual((), tuple(objective.shape))
            objective.backward()
            # d/dh = <s>_data - <s>_model = 2; d/dJ = <s_i s_j>_data - <s_i s_j>_model = 0
            torch.testing.assert_close(self.bm.linear.grad, torch.full((4,), 2.0))
            torch.testing.assert_close(self.bm.quadratic.grad, torch.zeros(4))

        with self.subTest("Test objective value matches"):
            s1 = torch.vstack([ones, ones, ones, self.pmones])
            s2 = torch.vstack([ones, ones, ones, self.mpones])
            s3 = torch.vstack([s2, s2])
            self.assertEqual(-1, self.bm.quasi_objective(s1, s2).item())
            self.assertEqual(-1, self.bm.quasi_objective(s1, s3).item())
            self.assertEqual(-1, self.bm.quasi_objective(s1.reshape(2, 2, 4), s3).item())

    def test_quasi_objective_gradient_is_difference_of_statistics(self):
        model = random_model(25, 0.4, seed=5)
        s_observed = randspins(13, 25, seed=1)
        s_model = randspins(7, 25, seed=2)
        model.quasi_objective(s_observed, s_model).backward()

        mean_obs, second_obs = model.sufficient_statistics(s_observed)
        mean_model, second_model = model.sufficient_statistics(s_model)
        torch.testing.assert_close(model.linear.grad, mean_obs - mean_model)
        torch.testing.assert_close(model.quadratic.grad, second_obs - second_model)

    def test_quasi_objective_gradient_wrt_observations(self):
        # DVAE-style usage: three-dimensional observations that require gradients
        model = random_model(10, 0.5, seed=6)
        s_model = randspins(6, 10, seed=3)
        s_observed = randspins(3, 5, 10, seed=4).requires_grad_()
        model.quasi_objective(s_observed, s_model).backward()
        # d/ds of the average energy is the effective field divided by the number of observations
        torch.testing.assert_close(s_observed.grad, model.effective_field(s_observed.detach()) / 15)

    def test_quasi_objective_requires_complete_spins(self):
        # Data of a model with hidden units is completed (conditional_expectation or
        # TorchSampler.complete) before it is passed to the objective.
        bm = GRBM(self.nodes, self.edges, hidden_nodes=["d"])
        with self.assertRaisesRegex(ValueError, "trailing dimension 4"):
            bm.quasi_objective(torch.ones(1, 3), torch.ones(1, 4))
        with self.assertRaisesRegex(ValueError, "trailing dimension 4"):
            bm.quasi_objective(torch.ones(1, 4), torch.ones(1, 3))

    # ------------------------------------------------------------------ hidden units ----------

    def test_pad_visible(self):
        bm = GRBM([0, 1, 2], [(0, 1), (0, 2), (1, 2)], [1])
        self.assertEqual(2, bm.n_visible)
        self.assertEqual(1, bm.n_hidden)
        self.assertTupleEqual(bm.visible_nodes, (0, 2))
        padded = bm.pad_visible(torch.zeros((99, 2)))
        self.assertEqual((99, 3), tuple(padded.shape))
        self.assertTrue(padded[:, 1].isnan().all())
        self.assertTrue((padded[:, [0, 2]] == 0).all())
        self.assertEqual((4, 5, 3), tuple(bm.pad_visible(torch.zeros((4, 5, 2))).shape))
        with self.assertRaisesRegex(ValueError, "number of visible units"):
            bm.pad_visible(torch.zeros((99, 3)))

    def test_conditional_expectation(self):
        bm = GRBM([0, 1, 2], [(0, 1), (0, 2), (1, 2)], [2])
        # effective field = quadratic(0,2) * [-1] + quadratic(1,2) * [1] + linear(2)
        #                 = 0.13 * [-1] - 0.17 * [1] + 0.4 = 0.1
        set_weights(bm, [-0.1, -0.2, 0.4], [-0.7, 0.13, -0.17])
        expected = bm.conditional_expectation(bm.pad_visible(torch.tensor([[-1.0, 1.0]])))
        torch.testing.assert_close(
            expected, torch.tensor([[-1.0, 1.0, torch.tanh(torch.tensor(-0.1)).item()]])
        )

    def test_conditional_expectation_unordered(self):
        bm = GRBM([0, 3, 2, 1], [(1, 3), (0, 1), (0, 3), (0, 2), (1, 2)], [3, 2])
        # effective field [3] = 0.15 * [-1] - 0.15 * [1] - 0.2 = -0.5
        # effective field [2] = 0.13 * [-1] - 0.17 * [1] + 0.4 = 0.1
        set_weights(bm, [-0.1, -0.2, 0.4, 0.2], [-0.15, -0.7, 0.15, 0.13, -0.17])
        padded = bm.pad_visible(torch.tensor([[-1.0, 1.0]]))
        h_eff = bm.effective_field(padded, idx=bm.hidden_idx)
        torch.testing.assert_close(h_eff, torch.tensor([[-0.5, 0.1]]))
        expected = bm.conditional_expectation(padded)
        torch.testing.assert_close(expected[:, bm.visible_idx], torch.tensor([[-1.0, 1.0]]))
        torch.testing.assert_close(expected[:, bm.hidden_idx], -torch.tanh(h_eff))

    def test_conditional_expectation_mixed_edge_orientation(self):
        # The hidden node is the second endpoint of the first edge and the first endpoint of the
        # second edge; neighbours must be paired with the weights of their own edges.
        bm = GRBM(["a", "b", "h"], [("b", "h"), ("h", "a")], hidden_nodes=["h"],
                  quadratic={("b", "h"): 1.0, ("h", "a"): 0.1})
        padded = bm.pad_visible(torch.tensor([[1.0, -1.0]]))  # s_a = 1, s_b = -1
        self.assertAlmostEqual(
            bm.effective_field(padded, idx=bm.hidden_idx).item(), 0.1 - 1.0, places=6
        )

    def test_effective_field_ignores_hidden_couplings(self):
        bm = GRBM(
            ["v1", "v2", "h1", "h2"],
            [("v1", "h1"), ("v2", "h1"), ("v2", "h2"), ("h1", "h2")],
            hidden_nodes=["h1", "h2"],
            linear={"h1": 0.1, "h2": -0.4},
            quadratic={("v1", "h1"): 0.3, ("v2", "h1"): -0.2, ("v2", "h2"): 0.5, ("h1", "h2"): 0.7},
        )
        padded = bm.pad_visible(torch.tensor([[1.0, -1.0]]))
        h_eff = bm.effective_field(padded, idx=bm.hidden_idx)
        torch.testing.assert_close(h_eff, torch.tensor([[0.1 + 0.3 + 0.2, -0.4 - 0.5]]))
        self.assertTrue(bm.connected_hidden)
        with self.assertRaisesRegex(ValueError, "no two unknown spins are adjacent"):
            bm.conditional_expectation(padded)

    def test_conditional_expectation_arbitrary_pattern(self):
        # Any pattern of unknown spins is accepted as long as no two unknown spins are adjacent;
        # patterns may differ between rows. Nodes d and b, as well as d and c, are not adjacent.
        set_weights(self.bm, [0.3, -0.2, 0.5, 0.1], [0.8, -0.6, 0.4, -0.9])
        nan = float("nan")
        x = torch.tensor([[nan, nan, 1.0, -1.0],
                          [nan, 1.0, -1.0, nan],
                          [1.0, -1.0, 1.0, 1.0]], requires_grad=True)
        out = self.bm.conditional_expectation(x)

        with self.subTest("Observed spins are unchanged"):
            self.assertFalse(out.isnan().any())
            observed = ~x.isnan()
            torch.testing.assert_close(out[observed], x.detach()[observed])

        with self.subTest("Expectations match the enumeration of the unknown spins"):
            for row in range(2):
                unknown = x[row].isnan().nonzero().flatten()
                states = x[row].detach().clone().repeat(4, 1)
                states[:, unknown] = torch.tensor(
                    [[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]]
                )
                weights = torch.softmax(-self.bm(states), 0)
                torch.testing.assert_close(out[row, unknown], weights @ states[:, unknown])

        with self.subTest("Gradients flow to the observed spins but not to the parameters"):
            grad_x, = torch.autograd.grad(out.sum(), x, retain_graph=True)
            torch.testing.assert_close(grad_x, (~x.isnan()).float())
            grads = torch.autograd.grad(out.sum(), list(self.bm.parameters()), allow_unused=True)
            self.assertTrue(all(g is None for g in grads))

        with self.subTest("Adjacent unknown spins are rejected"):
            with self.assertRaisesRegex(ValueError, "no two unknown spins are adjacent"):
                self.bm.conditional_expectation(torch.tensor([[1.0, nan, nan, 1.0]]))  # b, a
            with self.assertRaisesRegex(ValueError, "no two unknown spins are adjacent"):
                self.bm.conditional_expectation(
                    torch.tensor([[1.0, 1.0, 1.0, 1.0], [nan, 1.0, nan, 1.0]])  # d, a in row 2
                )

    def test_quasi_objective_gradient_hidden_units(self):
        bm = GRBM([1, 2, 3],
                  [(1, 2), (1, 3), (2, 3)],
                  [1],
                  {1: 0.2, 2: 0.2, 3: 0.3},
                  {(1, 2): 0.2, (1, 3): 0.3, (2, 3): 0.6})
        # Note : In the diagram below linear biases are shown using  <>
        #        quadratic biases using ()
        #                 (0.2)
        # Model:  v1 <0.2> ----- v2  <0.2>
        #           \           /
        #     (0.3)  \         / (0.6)
        #             \       /
        #                v3 <0.3>
        s_observed = torch.tensor([[1.0, -1.0]])
        s_model = torch.tensor([[1.0, -1.0, 1.0]])
        s_data = bm.conditional_expectation(bm.pad_visible(s_observed))
        bm.quasi_objective(s_data, s_model).backward()
        # Exact conditional expectation of the sufficient statistics t = (v1 v2 v3 v1v2 v1v3 v2v3)
        q_plus = torch.exp(-torch.tensor(0.2 + 0.2 - 0.3 - 0.6 + 0.2 - 0.3))
        q_minus = torch.exp(-torch.tensor(-0.2 - 0.2 + 0.3 - 0.6 + 0.2 - 0.3))
        p_plus = q_plus / (q_plus + q_minus)
        p_minus = q_minus / (q_plus + q_minus)
        t_plus = torch.tensor([1, 1, -1, 1, -1, -1]).float()
        t_minus = torch.tensor([-1, 1, -1, -1, 1, -1]).float()
        t_model = torch.tensor([1, -1, 1, -1, 1, -1]).float()
        grad = t_plus * p_plus + t_minus * p_minus - t_model
        grad_auto = torch.cat([bm.linear.grad, bm.quadratic.grad])
        torch.testing.assert_close(grad, grad_auto)

    def test_quasi_objective_exact_disc_matches_enumeration(self):
        # Random RBM-like model: exact marginalization over hidden units by enumeration
        model = random_model(7, 0.6, n_hidden=3, seed=8)
        s_observed = randspins(5, 4, seed=9)
        s_model = randspins(6, 7, seed=10)
        s_data = model.conditional_expectation(model.pad_visible(s_observed))
        model.quasi_objective(s_data, s_model).backward()

        hidden_states = 1.0 - 2.0 * torch.tensor(
            [[(k >> i) & 1 for i in range(3)] for k in range(8)], dtype=torch.float32
        )
        expected_linear = torch.zeros(7)
        expected_second = torch.zeros(7, 7)
        with torch.no_grad():
            for obs in s_observed:
                full = model.pad_visible(obs.unsqueeze(0)).repeat(8, 1)
                full[:, model.hidden_idx] = hidden_states
                weights = torch.softmax(-model(full), 0)
                expected_linear += weights @ full / len(s_observed)
                expected_second += (full.T * weights) @ full / len(s_observed)
            mean_model, second_model = model.sufficient_statistics(s_model)
        torch.testing.assert_close(model.linear.grad, expected_linear - mean_model)
        torch.testing.assert_close(
            model.quadratic.grad,
            expected_second[model.edge_idx_i, model.edge_idx_j] - second_model,
        )

    def test_quasi_objective_gradient_connected_hidden_units(self):
        bm = GRBM([1, 2, 3],
                  [(1, 2), (1, 3), (2, 3)],
                  [1, 2],
                  {1: 0.2, 2: 0.2, 3: 0.3},
                  {(1, 2): 0.2, (1, 3): 0.3, (2, 3): 0.6})
        s_observed = torch.tensor([[1.0]])
        s_model = torch.tensor([[1.0, -1.0, 1.0]])
        # The hidden units are connected, so their expectations given the data are not exact:
        # fill them in with samples conditioned on v3=1 for (v1, v2), here a fixed set of three
        s_data = bm.pad_visible(s_observed).expand(3, -1).clone()
        s_data[:, bm.hidden_idx] = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0]])
        bm.quasi_objective(s_data, s_model).backward()

        t_cond_samples = torch.tensor(
            [
                [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            ]
        )
        t_model = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        grad = t_cond_samples.mean(0) - t_model
        grad_auto = torch.cat([bm.linear.grad, bm.quadratic.grad])
        torch.testing.assert_close(grad, grad_auto)

    # ------------------------------------------------------------------ misc ------------------

    def test_state_dict_roundtrip(self):
        model = random_model(15, 0.5, n_hidden=4, seed=14)
        other = GRBM(model.nodes, model.edges, model.hidden_nodes)
        other.load_state_dict(model.state_dict())
        torch.testing.assert_close(other.quadratic, model.quadratic)
        torch.testing.assert_close(other.linear, model.linear)
        self.assertTrue(torch.equal(other.edge_idx_i, model.edge_idx_i))
        self.assertTrue(torch.equal(other.hidden_idx, model.hidden_idx))

    def test_double(self):
        model = random_model(10, 0.5, seed=15).double()
        x = randspins(5, 10).double()
        self.assertEqual(torch.float64, model(x).dtype)
        self.assertEqual(torch.long, model.edge_idx_i.dtype)
        bqm = model_to_bqm(model)
        torch.testing.assert_close(
            model(x), torch.tensor(bqm.energies((x.numpy(), model.nodes)))
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda(self):
        model = random_model(20, 0.4, n_hidden=6, seed=16)
        s_observed = randspins(11, 14, seed=17)
        s_model = randspins(5, 20, seed=18)
        x = randspins(9, 20, seed=19)

        energies = model(x)
        s_data = model.conditional_expectation(model.pad_visible(s_observed))
        objective = model.quasi_objective(s_data, s_model)
        objective.backward()
        grads = [p.grad.clone() for p in model.parameters()]
        model.zero_grad()

        model = model.cuda()
        self.assertTrue(model.edge_idx_i.is_cuda and model.hidden_idx.is_cuda)
        torch.testing.assert_close(model(x.cuda()).cpu(), energies)
        s_data_cuda = model.conditional_expectation(model.pad_visible(s_observed.cuda()))
        self.assertTrue(s_data_cuda.is_cuda)
        torch.testing.assert_close(s_data_cuda.cpu(), s_data)
        objective_cuda = model.quasi_objective(s_data_cuda, s_model.cuda())
        objective_cuda.backward()
        torch.testing.assert_close(objective_cuda.cpu(), objective)
        for grad, p in zip(grads, model.parameters()):
            torch.testing.assert_close(p.grad.cpu(), grad)
        bqm_cuda = to_bqm(model, model.linear, model.quadratic)
        model.cpu()
        self.assertEqual(bqm_cuda, to_bqm(model, model.linear, model.quadratic))


if __name__ == "__main__":
    unittest.main()
