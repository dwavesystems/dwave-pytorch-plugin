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
import copy
import pickle
import unittest

import torch

from dwave.plugins.torch.graph import GraphIndex

# The triangle a, b, c with the dangling node d, in the node order "dbac", so that the node
# indices (d=0, b=1, a=2, c=3) differ from the order in which the nodes are written
NODES = "dbac"
EDGES = [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c")]
LINEAR = torch.tensor([1.0, -2.0, 3.0, 0.0])  # h_d, h_b, h_a, h_c
QUADRATIC = torch.tensor([1.0, 2.0, 3.0, -4.0])  # J_ab, J_ac, J_ad, J_bc
# Four configurations (s_d, s_b, s_a, s_c) and their energies sum_i h_i s_i + sum_ij J_ij s_i s_j;
# the third, for example, has energy (1 + 2 + 3 + 0) + (-1 - 2 + 3 - 4) = 6 - 4 = 2
SPINS = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                      [-1.0, -1.0, -1.0, -1.0],
                      [1.0, -1.0, 1.0, -1.0],
                      [1.0, 1.0, -1.0, -1.0]])
ENERGIES = torch.tensor([4.0, 0.0, 2.0, -2.0])
# The effective fields h_k + sum_l J_kl s_l of the first and the third configuration; for the
# third, theta_d = 1 + 3, theta_b = -2 + 1 + 4, theta_a = 3 - 1 - 2 + 3 and theta_c = 0 + 2 + 4
FIELDS_FIRST = torch.tensor([4.0, -5.0, 9.0, -2.0])
FIELDS_THIRD = torch.tensor([4.0, 3.0, 3.0, 6.0])
# A second model with the biases negated and the couplings doubled, for batches of models: its
# energies of the four configurations are -6 + 2 * (-4) = -14 for the third, for example
LINEAR_2 = torch.stack([LINEAR, -LINEAR])
QUADRATIC_2 = torch.stack([QUADRATIC, 2 * QUADRATIC])
ENERGIES_2 = torch.tensor([2.0, 6.0, -14.0, 8.0])
FIELDS_2_THIRD = torch.tensor([5.0, 12.0, -3.0, 12.0])  # -h + 2 * (3, 5, 0, 6)


class TestGraphIndex(unittest.TestCase):
    def setUp(self):
        self.graph = GraphIndex(NODES, EDGES)

    def test_index(self):
        graph = GraphIndex("dbac", [("a", "b"), ("a", "c"), ("d", "a"), ("b", "c")])
        self.assertTupleEqual(tuple("dbac"), graph.nodes)
        self.assertTupleEqual((("a", "b"), ("a", "c"), ("d", "a"), ("b", "c")), graph.edges)
        self.assertDictEqual({"d": 0, "b": 1, "a": 2, "c": 3}, graph.node_to_idx)
        self.assertEqual(4, graph.n_nodes)
        self.assertEqual(4, graph.n_edges)
        # canonical orientation: smaller index first
        self.assertListEqual([1, 2, 0, 1], graph.edge_idx_i.tolist())
        self.assertListEqual([2, 3, 2, 3], graph.edge_idx_j.tolist())
        self.assertListEqual([1, 2, 3, 2], graph.degrees().tolist())
        # every edge is found under both orientations
        self.assertEqual(8, len(graph.edge_to_idx))
        self.assertListEqual([0, 1, 2, 3], [graph.edge_to_idx[e] for e in graph.edges])
        self.assertListEqual([0, 1, 2, 3], [graph.edge_to_idx[e[::-1]] for e in graph.edges])
        self.assertIn("n_nodes=4, n_edges=4", repr(graph))

    def test_module(self):
        graph = GraphIndex("abc", [("a", "b")])
        self.assertIsInstance(graph, torch.nn.Module)
        self.assertEqual(0, len(list(graph.parameters())))
        self.assertSetEqual({"edge_idx_i", "edge_idx_j"}, {name for name, _ in graph.named_buffers()})
        self.assertDictEqual({}, dict(graph.state_dict()), "index buffers are derived, not saved")

        with self.subTest("Buffers move with the module"):
            graph.to("meta")
            self.assertEqual("meta", graph.edge_idx_i.device.type)
            self.assertEqual("meta", graph.edge_idx_j.device.type)

        with self.subTest("The module can be copied and pickled"):
            graph = GraphIndex("abc", [("a", "b")])
            clone = copy.deepcopy(graph)
            self.assertTupleEqual(graph.nodes, clone.nodes)
            self.assertDictEqual(graph.node_to_idx, clone.node_to_idx)
            self.assertTrue(torch.equal(graph.edge_idx_i, clone.edge_idx_i))
            self.assertTupleEqual(graph.nodes, pickle.loads(pickle.dumps(graph)).nodes)
            self.assertTupleEqual(graph.edges, pickle.loads(pickle.dumps(graph)).edges)

    def test_dense_quadratic(self):
        graph = GraphIndex("abc", [("b", "a"), ("a", "c"), ("b", "c")])
        per_edge = torch.tensor([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        dense = graph.dense_quadratic(per_edge)
        self.assertEqual((2, 3, 3), tuple(dense.shape))
        torch.testing.assert_close(dense[0], torch.tensor([[0.0, 1.0, 2.0],
                                                           [0.0, 0.0, 3.0],
                                                           [0.0, 0.0, 0.0]]))
        torch.testing.assert_close(dense[..., graph.edge_idx_i, graph.edge_idx_j], per_edge)
        with self.assertRaisesRegex(ValueError, "Expected 3 edge biases"):
            graph.dense_quadratic(torch.zeros(2, 2))

    def test_symmetric_quadratic(self):
        # Row k holds the couplings of node k, in the node order d, b, a, c
        expected = torch.tensor([[0.0, 0.0, 3.0, 0.0],
                                 [0.0, 0.0, 1.0, -4.0],
                                 [3.0, 1.0, 0.0, 2.0],
                                 [0.0, -4.0, 2.0, 0.0]])
        torch.testing.assert_close(self.graph.symmetric_quadratic(QUADRATIC), expected)
        batched = self.graph.symmetric_quadratic(QUADRATIC_2)
        self.assertEqual((2, 4, 4), tuple(batched.shape))
        torch.testing.assert_close(batched[1], 2 * expected)

    def test_energy(self):
        torch.testing.assert_close(self.graph.energy(SPINS, LINEAR, QUADRATIC), ENERGIES)

        with self.subTest("A single configuration"):
            torch.testing.assert_close(self.graph.energy(SPINS[2], LINEAR, QUADRATIC), ENERGIES[2])

        with self.subTest("Unbatched biases apply to spins of any leading shape"):
            energies = self.graph.energy(SPINS.reshape(2, 2, 4), LINEAR, QUADRATIC)
            torch.testing.assert_close(energies, ENERGIES.reshape(2, 2))

        with self.subTest("A batch of models with M configurations each"):
            spins = torch.stack([SPINS[:2], SPINS[2:]])  # the first two, and the last two
            energies = self.graph.energy(spins, LINEAR_2, QUADRATIC_2)
            torch.testing.assert_close(energies, torch.stack([ENERGIES[:2], ENERGIES_2[2:]]))

        with self.subTest("A batch of models with one configuration each"):
            energies = self.graph.energy(SPINS[[0, 2]], LINEAR_2, QUADRATIC_2)
            torch.testing.assert_close(energies, torch.stack([ENERGIES[0], ENERGIES_2[2]]))

    def test_energy_validation(self):
        spins = SPINS.reshape(2, 2, 4)
        # Broadcasting would silently return (2, 2, 2) energies
        for linear, quadratic in ((LINEAR, QUADRATIC_2), (LINEAR_2, QUADRATIC)):
            with self.assertRaisesRegex(ValueError, "same batch of models"):
                self.graph.energy(spins, linear, quadratic)
        with self.assertRaisesRegex(ValueError, "one per node"):
            self.graph.energy(spins, torch.zeros(2, 3), QUADRATIC_2)
        with self.assertRaisesRegex(ValueError, "Expected 4 edge biases"):
            self.graph.energy(spins, LINEAR_2, torch.zeros(2, 3))

    def test_effective_field(self):
        fields = self.graph.effective_field(SPINS, linear=LINEAR, quadratic=QUADRATIC)
        self.assertEqual((4, 4), tuple(fields.shape))
        torch.testing.assert_close(fields[0], FIELDS_FIRST)
        torch.testing.assert_close(fields[2], FIELDS_THIRD)

        with self.subTest("The fields are the gradients of the energy"):
            x = SPINS.clone().requires_grad_()
            grad, = torch.autograd.grad(self.graph.energy(x, LINEAR, QUADRATIC).sum(), x)
            torch.testing.assert_close(grad, fields)

        with self.subTest("A single configuration"):
            single = self.graph.effective_field(SPINS[2], linear=LINEAR, quadratic=QUADRATIC)
            torch.testing.assert_close(single, FIELDS_THIRD)

        with self.subTest("A subset of nodes"):
            idx = torch.tensor([2, 0])  # the fields of a and of d
            subset = self.graph.effective_field(SPINS[2], linear=LINEAR, quadratic=QUADRATIC, idx=idx)
            torch.testing.assert_close(subset, torch.tensor([3.0, 4.0]))

        with self.subTest("Unknown spins contribute nothing to the fields"):
            unknown = SPINS[2].clone()
            unknown[1] = torch.nan  # s_b is unknown: theta_a loses 1 * s_b, theta_c loses -4 * s_b
            fields_unknown = self.graph.effective_field(unknown, linear=LINEAR, quadratic=QUADRATIC)
            torch.testing.assert_close(fields_unknown, torch.tensor([4.0, 3.0, 4.0, 2.0]))

        with self.subTest("A precomputed coupling matrix"):
            coupling = self.graph.symmetric_quadratic(QUADRATIC)
            torch.testing.assert_close(
                self.graph.effective_field(SPINS, linear=LINEAR, coupling=coupling), fields
            )

        with self.subTest("A batch of models with M configurations each"):
            spins = SPINS[2].expand(2, 1, 4)  # the third configuration under both models
            batched = self.graph.effective_field(spins, linear=LINEAR_2, quadratic=QUADRATIC_2)
            torch.testing.assert_close(batched, torch.stack([FIELDS_THIRD, FIELDS_2_THIRD])[:, None])

        with self.subTest("A batch of models with one configuration each"):
            spins = SPINS[2].expand(2, 4)
            batched = self.graph.effective_field(spins, linear=LINEAR_2, quadratic=QUADRATIC_2)
            torch.testing.assert_close(batched, torch.stack([FIELDS_THIRD, FIELDS_2_THIRD]))

    def test_effective_field_validation(self):
        spins = SPINS.reshape(2, 2, 4)
        coupling = self.graph.symmetric_quadratic(QUADRATIC_2)
        # Broadcasting would silently return (2, 2, 2, 4) fields
        for linear, quadratic in ((LINEAR, QUADRATIC_2), (LINEAR_2, QUADRATIC)):
            with self.assertRaisesRegex(ValueError, "same batch of models"):
                self.graph.effective_field(spins, linear=linear, quadratic=quadratic)
        with self.assertRaisesRegex(ValueError, "same batch of models"):
            self.graph.effective_field(spins, linear=LINEAR, coupling=coupling)
        with self.assertRaisesRegex(ValueError, "one per node"):
            self.graph.effective_field(spins, linear=torch.zeros(3), quadratic=QUADRATIC)
        with self.assertRaisesRegex(ValueError, "`quadratic` or `coupling`"):
            self.graph.effective_field(spins, linear=LINEAR_2)
        with self.assertRaisesRegex(ValueError, "not both"):
            self.graph.effective_field(spins, linear=LINEAR_2, quadratic=QUADRATIC_2, coupling=coupling)

    def test_edgeless(self):
        graph = GraphIndex([0, 1], [])
        self.assertEqual(0, graph.n_edges)
        self.assertEqual((0,), tuple(graph.edge_idx_i.shape))
        self.assertListEqual([0, 0], graph.degrees().tolist())
        self.assertDictEqual({}, graph.edge_to_idx)

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "duplicate entries"):
            GraphIndex("aab", [])
        with self.assertRaisesRegex(ValueError, "not a node"):
            GraphIndex("ab", [("a", "c")])
        with self.assertRaisesRegex(ValueError, r"Self-loops.*\('a', 'a'\)"):
            GraphIndex("ab", [("a", "a")])
        with self.assertRaisesRegex(ValueError, "Duplicate edges"):
            GraphIndex("ab", [("a", "b"), ("b", "a")])
        with self.assertRaisesRegex(TypeError, "Nodes must be hashable"):
            GraphIndex([[0], [1]], [])
        with self.assertRaisesRegex(TypeError, "Edge endpoints must be hashable"):
            GraphIndex([0, 1], [([0], 1)])


if __name__ == "__main__":
    unittest.main()
