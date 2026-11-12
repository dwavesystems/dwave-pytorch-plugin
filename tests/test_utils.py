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
from dimod import SPIN, BinaryQuadraticModel, SampleSet
from torch import Tensor

from dwave.plugins.torch.graph import GraphIndex
from dwave.plugins.torch.utils import estimate_beta, sampleset_to_tensor, to_bqm
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple


class TestUtils(unittest.TestCase):
    def test_sample_to_tensor(self):
        bogus_energy = [999] * 3
        spins_in = [[1, -1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ss = SampleSet.from_samples((spins_in, list("dbca")), SPIN, bogus_energy)
        spins = sampleset_to_tensor(list("cabd"), ss)
        self.assertTupleEqual((3, 4), tuple(spins.shape))
        self.assertIsInstance(spins, Tensor)
        self.assertEqual(torch.float32, spins.dtype)
        # Test variable ordering is respected
        self.assertListEqual(
            spins.tolist(), [[1, 1, -1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )

        with self.subTest("Aggregated samples are repeated according to their occurrences"):
            ss = SampleSet.from_samples(([[1, -1], [-1, 1]], "ab"), SPIN, [0, 0],
                                        num_occurrences=[3, 1])
            spins = sampleset_to_tensor("ba", ss)
            self.assertListEqual(spins.tolist(), [[-1, 1]] * 3 + [[1, -1]])

        with self.subTest("Empty sample sets"):
            ss = SampleSet.from_samples(([], "ab"), SPIN, [])
            self.assertEqual((0, 2), tuple(sampleset_to_tensor("ab", ss).shape))


class TestToBqm(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = GraphIndex("dbac", [["a", "b"], ("a", "c"), ("a", "d"), ("b", "c")])
        self.linear = torch.tensor([-3.0, 0.0, 1.0, 3.0], requires_grad=True)
        self.quadratic = torch.tensor([-1.0, 1.0, 2.0, 0.0], requires_grad=True)

    def test_to_bqm(self):
        graph, linear, quadratic = self.graph, self.linear, self.quadratic

        with self.subTest("Variables in node order, biases keyed by nodes and edges"):
            bqm = to_bqm(graph, linear, quadratic)
            self.assertIsInstance(bqm, BinaryQuadraticModel)
            self.assertEqual(SPIN, bqm.vartype)
            self.assertEqual(0.0, bqm.offset)
            self.assertListEqual(list("dbac"), list(bqm.variables))
            self.assertDictEqual({"d": -3.0, "b": 0.0, "a": 1.0, "c": 3.0}, dict(bqm.linear))
            self.assertDictEqual(
                {("a", "b"): -1.0, ("a", "c"): 1.0, ("a", "d"): 2.0, ("b", "c"): 0.0},
                {tuple(sorted(e)): b for e, b in bqm.quadratic.items()},
            )
            self.assertIsInstance(bqm.get_linear("d"), float)

        with self.subTest("Energies agree with those of the graph"):
            spins = torch.tensor([[1.0, -1.0, 1.0, 1.0], [-1.0, -1.0, 1.0, -1.0]])
            torch.testing.assert_close(
                torch.tensor(bqm.energies((spins.numpy(), list(graph.nodes))), dtype=torch.float32),
                graph.energy(spins, linear.detach(), quadratic.detach()),
            )

        with self.subTest("Prefactor"):
            bqm = to_bqm(graph, linear, quadratic, prefactor=2.0)
            self.assertListEqual(
                [bqm.get_linear(v) for v in graph.nodes], (2 * linear).detach().tolist()
            )
            self.assertListEqual(
                [bqm.get_quadratic(u, v) for u, v in graph.edges], (2 * quadratic).detach().tolist()
            )

        with self.subTest("Clipping after scaling"):
            bqm = to_bqm(graph, linear, quadratic, 1, [-0.1, 1.5], [-0.05, 3])
            for expected, node in zip([-0.1, 0, 1, 1.5], graph.nodes):
                self.assertAlmostEqual(expected, bqm.get_linear(node), places=6)
            for expected, (u, v) in zip([-0.05, 1, 2, 0], graph.edges):
                self.assertAlmostEqual(expected, bqm.get_quadratic(u, v), places=6)

        with self.subTest("Edgeless"):
            bqm = to_bqm(GraphIndex([0, 1], []), torch.zeros(2), torch.zeros(0))
            self.assertDictEqual({0: 0.0, 1: 0.0}, dict(bqm.linear))
            self.assertEqual(0, bqm.num_interactions)

        with self.subTest("Shape validation"):
            with self.assertRaisesRegex(ValueError, "Expected 4 linear biases"):
                to_bqm(graph, torch.zeros(3), quadratic)
            with self.assertRaisesRegex(ValueError, "Expected 4 quadratic biases"):
                to_bqm(graph, linear, torch.zeros(4, 4))


class TestEstimateBeta(unittest.TestCase):
    def test_estimate_beta(self):
        graph = GraphIndex("dbac", [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c")])
        linear = torch.tensor([0.0, 1.0, 2.0, 3.0])
        quadratic = torch.tensor([1.0, 2.0, 3.0, 6.0])
        spins = torch.tensor([[1, -1, 1, 1], [-1, -1, 1, 1], [1, -1, -1, 1], [1, 1, 1, -1]])
        beta = estimate_beta(graph, linear, quadratic, spins)
        self.assertIsInstance(beta, float)
        bqm = to_bqm(graph, linear, quadratic)
        self.assertEqual(1.0 / mple(bqm, (spins.numpy(), list(graph.nodes)))[0], beta)

if __name__ == "__main__":
    unittest.main()
