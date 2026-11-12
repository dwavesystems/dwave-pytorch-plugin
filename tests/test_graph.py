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
import pickle
import unittest

import torch

from dwave.plugins.torch.graph import GraphIndex, randspin
from dwave.plugins.torch.utils import to_bqm
from tests.helper_functions import randspins


class TestRandspin(unittest.TestCase):
    def test_randspin(self):
        spins = randspin((2000,), generator=torch.Generator().manual_seed(0))
        self.assertEqual(torch.int64, spins.dtype)
        self.assertSetEqual({-1, 1}, set(spins.unique().tolist()))

        with self.subTest("Keyword arguments reach torch.randint"):
            spins = randspin((3, 4), dtype=torch.float32, device="meta")
            self.assertEqual((3, 4), tuple(spins.shape))
            self.assertEqual(torch.float32, spins.dtype)
            self.assertEqual("meta", spins.device.type)

        with self.subTest("Seeded draws are reproducible"):
            first = randspin((5, 5), generator=torch.Generator().manual_seed(3))
            second = randspin((5, 5), generator=torch.Generator().manual_seed(3))
            self.assertTrue(torch.equal(first, second))


class TestGraphIndex(unittest.TestCase):
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

    def test_energy_and_effective_field(self):
        graph = GraphIndex("dbac", [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c")])
        linear = torch.tensor([[0.0, 1.0, 2.0, 3.0], [0.5, -1.0, 0.0, 2.0]])
        edge_biases = torch.tensor([[1.0, 2.0, 3.0, 6.0], [-1.0, 0.5, 0.0, 2.0]])
        quadratic = edge_biases  # per-edge biases are what energy and effective_field take
        spins = randspins(2, 7, 4, seed=3)
        energies = graph.energy(spins, linear, quadratic)
        self.assertEqual((2, 7), tuple(energies.shape))

        with self.subTest("Batched biases match dimod energies model by model"):
            for b in range(2):
                bqm = to_bqm(graph, linear[b], edge_biases[b])
                expected = bqm.energies((spins[b].numpy(), list(graph.nodes)))
                torch.testing.assert_close(energies[b], torch.tensor(expected, dtype=torch.float32))

        with self.subTest("Unbatched biases apply to spins of any leading shape"):
            unbatched = graph.energy(spins, linear[0], quadratic[0])
            self.assertEqual((2, 7), tuple(unbatched.shape))
            torch.testing.assert_close(unbatched[0], energies[0])
            torch.testing.assert_close(graph.energy(spins[1, 0], linear[0], quadratic[0]), unbatched[1, 0])

        with self.subTest("One configuration per batched model"):
            torch.testing.assert_close(graph.energy(spins[:, 0], linear, quadratic), energies[:, 0])

        with self.subTest("Effective fields are the gradients of the energy"):
            x = spins.clone().requires_grad_()
            grad, = torch.autograd.grad(graph.energy(x, linear, quadratic).sum(), x)
            fields = graph.effective_field(spins, linear=linear, quadratic=quadratic)
            torch.testing.assert_close(grad, fields)
            torch.testing.assert_close(
                graph.effective_field(spins[:, 0], linear=linear, quadratic=quadratic), fields[:, 0]
            )

        with self.subTest("Subsets of nodes and precomputed couplings"):
            idx = torch.tensor([2, 0])
            coupling = graph.symmetric_coupling(quadratic)
            torch.testing.assert_close(
                graph.effective_field(spins, linear=linear, coupling=coupling, idx=idx), fields[..., idx]
            )

        with self.subTest("Unknown spins contribute nothing"):
            x = spins.clone()
            x[..., 1] = torch.nan
            zeroed = spins.clone()
            zeroed[..., 1] = 0.0
            torch.testing.assert_close(
                graph.effective_field(x, linear=linear, quadratic=quadratic),
                graph.effective_field(zeroed, linear=linear, quadratic=quadratic),
            )

        with self.subTest("Biases that are not batched alike are rejected"):
            # Broadcasting would silently return (2, 2, 7) energies and (2, 2, 7, 4) fields
            for h, J in ((linear[0], quadratic), (linear, quadratic[0])):
                with self.assertRaisesRegex(ValueError, "same batch of models"):
                    graph.energy(spins, h, J)
                with self.assertRaisesRegex(ValueError, "same batch of models"):
                    graph.effective_field(spins, linear=h, quadratic=J)
            with self.assertRaisesRegex(ValueError, "same batch of models"):
                graph.effective_field(spins, linear=linear[0], coupling=coupling)
            with self.assertRaisesRegex(ValueError, "one per node"):
                graph.energy(spins, torch.zeros(2, 3), quadratic)
            with self.assertRaisesRegex(ValueError, "one per node"):
                graph.effective_field(spins, linear=torch.zeros(3), quadratic=quadratic[0])

        with self.subTest("Couplings are required"):
            with self.assertRaisesRegex(ValueError, "`quadratic` or `coupling`"):
                graph.effective_field(spins, linear=linear)

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


if __name__ == "__main__":
    unittest.main()
