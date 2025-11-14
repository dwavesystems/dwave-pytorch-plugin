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

import dwave_networkx as dnx
import networkx as nx
import torch
from parameterized import parameterized

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.samplers.block_spin_sampler import BlockSampler


class TestBlockSampler(unittest.TestCase):
    ZEPHYR = dnx.zephyr_graph(1, coordinates=True)
    GRBM_ZEPHYR = GRBM(ZEPHYR.nodes, ZEPHYR.edges)
    CRAYON_ZEPHYR = dnx.zephyr_four_color

    BIPARTITE = nx.complete_bipartite_graph(5, 3)
    GRBM_BIPARTITE = GRBM(BIPARTITE.nodes, BIPARTITE.edges)
    def CRAYON_BIPARTITE(b): return b < 5

    GRBM_SINGLE = GRBM([0], [])
    def CRAYON_SINGLE(s): 0

    GRBM_CRAYON_TEST_CASES = [(GRBM_ZEPHYR, CRAYON_ZEPHYR),
                              (GRBM_BIPARTITE, CRAYON_BIPARTITE),
                              (GRBM_SINGLE, CRAYON_SINGLE)]

    def setUp(self) -> None:
        self.crayon_veqa = lambda v: v == "a"
        return super().setUp()

    @parameterized.expand(GRBM_CRAYON_TEST_CASES)
    def test_sample(self, grbm, crayon):
        for pac in "Metropolis", "Gibbs":
            schedule = [0.0, 1.0, 2.0]
            bss1 = BlockSampler(grbm, crayon, 10, schedule, pac, seed=1)
            bss1.sample()

            bss2 = BlockSampler(grbm, crayon, 10, [1.0], pac, seed=1)
            for beta in schedule:
                bss2._step(beta)

            self.assertListEqual(bss1._x.tolist(), bss2._x.tolist())

    def test_device(self):
        grbm = GRBM(list("ab"), [["a", "b"]])

        crayon = self.crayon_veqa
        sample_size = 1_000_000
        bss = BlockSampler(grbm, crayon, sample_size, [1.0], "Gibbs", seed=2)
        bss.to('meta')
        self.assertEqual("cpu", bss._grbm.linear.device.type)
        self.assertEqual("cpu", bss._grbm.quadratic.device.type)
        self.assertEqual("meta", bss._x.device.type)
        self.assertEqual("meta", bss._padded_adjacencies.device.type)
        self.assertEqual("meta", bss._padded_adjacencies_weight.device.type)
        self.assertEqual("meta", bss._zeros.device.type)
        self.assertEqual("meta", bss._schedule.device.type)
        self.assertEqual("meta", bss._partition[0].device.type)
        self.assertEqual("meta", bss._partition[1].device.type)
        # NOTE: "meta" device is not supported for torch.Generator
        self.assertEqual("cpu", bss._rng.device.type)

    def test_gibbs_update(self):
        grbm = GRBM(list("ab"), [["a", "b"]])

        crayon = self.crayon_veqa
        sample_size = 1_000_000
        bss = BlockSampler(grbm, crayon, sample_size, [1.0], "Gibbs", seed=2)
        bss._x.data[:] = 1
        zero = torch.tensor(0.0)
        ones = torch.ones((sample_size, 1))
        bss._gibbs_update(0.0, bss._partition[0], ones*zero)
        torch.testing.assert_close(torch.tensor(0.5), bss._x.mean(), atol=1e-3, rtol=1e-3)
        bss._gibbs_update(0.0, bss._partition[1], ones*zero)
        torch.testing.assert_close(torch.tensor(0.0), bss._x.mean(), atol=1e-3, rtol=1e-3)

        effective_field = torch.tensor(1.2)
        bss._gibbs_update(1.0, bss._partition[0], effective_field*ones)
        bss._gibbs_update(1.0, bss._partition[1], effective_field*ones)
        torch.testing.assert_close(
            torch.tanh(-effective_field),
            bss._x.mean(),
            atol=1e-3, rtol=1e-3)

    def test_initial_states_respected(self):
        grbm = GRBM(list("ab"), [["a", "b"]])

        crayon = self.crayon_veqa
        initial_states = torch.tensor([[-1, 1], [1, 1], [-1, -1], [1, 1], [-1, 1], [-1, 1], [1, 1]])

        bss = BlockSampler(grbm, crayon, len(initial_states), [1.0], "Metropolis",
                               initial_states, 2)
        self.assertListEqual(bss._x.tolist(), initial_states.tolist())

    def test_metropolis_update_average(self):
        grbm = GRBM(list("ab"), [["a", "b"]])

        crayon = self.crayon_veqa
        sample_size = 1_000_000
        bss = BlockSampler(grbm, crayon, sample_size, [1.0], "Metropolis", seed=2)
        bss._x.data[:] = 1
        ones = torch.ones((sample_size, 1))
        effective_field = torch.tensor(1.2)
        for i in range(10):
            bss._metropolis_update(1.0, bss._partition[0], effective_field*ones)
            bss._metropolis_update(1.0, bss._partition[1], effective_field*ones)
        torch.testing.assert_close(
            torch.tanh(-effective_field),
            bss._x.mean(),
            atol=1e-3, rtol=1e-3)

    def test_metropolis_update_oscillates(self):
        grbm = GRBM(list("ab"), [["a", "b"]])

        crayon = self.crayon_veqa
        sample_size = 1_00
        bss = BlockSampler(grbm, crayon, sample_size, [1.0], "Metropolis", seed=2)
        bss._x.data[:] = 1
        zero_effective_field = torch.zeros((sample_size, 1))
        bss._metropolis_update(0.0, bss._partition[0], zero_effective_field)
        self.assertTrue((bss._x[:, 1] == -1).all())
        bss._metropolis_update(0.0, bss._partition[1], zero_effective_field)
        self.assertTrue((bss._x == -1).all())

    def test_effective_field(self):
        # Create a triangle graph with an additional dangling vertex
        #       a
        #     / | \
        #    b--c  d
        self.nodes = list("abcd")
        self.edges = [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]

        # Manually set the parameter weights for testing
        dtype = torch.float32
        grbm = GRBM(self.nodes, self.edges)
        grbm._linear.data = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=dtype)
        grbm._quadratic.data = torch.tensor([1.1, 2.2, 3.3, 6.6], dtype=dtype)

        def crayon(v):
            if v == "a":
                return 0
            if v == "b":
                return 1
            if v == "c":
                return 2
            if v == "d":
                return 1
        bss = BlockSampler(grbm, crayon, 3, [1.0], seed=3)
        bss._x.data[:] = torch.tensor([[1, 1, -1, -1],
                                      [-1, -1, 1, -1],
                                      [1, 1, 1, -1]])
        # effective field for a
        effective_field_a = bss._compute_effective_field(bss._partition[0])
        torch.testing.assert_close(
            effective_field_a,
            torch.tensor([[0.0 + 1.1 - 2.2 - 3.3],
                          [0.0 - 1.1 + 2.2 - 3.3],
                          [0.0 + 1.1 + 2.2 - 3.3]])
        )
        # effective field for b, d
        effective_field_bd = bss._compute_effective_field(bss._partition[1])
        torch.testing.assert_close(effective_field_bd,
                                   torch.tensor([[1.0 + 1.1 - 6.6, 3.0 + 3.3],
                                                 [1.0 - 1.1 + 6.6, 3.0 - 3.3],
                                                 [1.0 + 1.1 + 6.6, 3.0 + 3.3]]))
        # effective field for c
        effective_field_c = bss._compute_effective_field(bss._partition[2])
        torch.testing.assert_close(effective_field_c,
                                   torch.tensor([[2.0 + 2.2 + 6.6],
                                                 [2.0 - 2.2 - 6.6],
                                                 [2.0 + 2.2 + 6.6]]))

    def test_get_adjacencies(self):
        # Create a triangle graph with an additional dangling vertex
        #       a
        #     / | \
        #    b--c  d
        self.nodes = list("abcd")
        self.edges = [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]

        # Manually set the parameter weights for testing
        dtype = torch.float32
        grbm = GRBM(self.nodes, self.edges)
        grbm._linear.data = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=dtype)
        grbm._quadratic.data = torch.tensor([1.1, 2.2, 3.3, 6.6], dtype=dtype)

        def crayon(v):
            if v == "a":
                return 0
            if v == "b":
                return 1
            if v == "c":
                return 2
            if v == "d":
                return 1
        bss = BlockSampler(grbm, crayon, 10, [1.0], seed=4)
        padded_adj, padded_adj_weights = bss._get_adjacencies()

        # First, check the neighbour indices are correct
        # a has neighbours b, c, d in that order, so 2, 3, 4
        self.assertListEqual(padded_adj[0].tolist(), [1, 2, 3])
        # b has neighbours a, c, in that order, so 0, 2, and padded -1
        self.assertListEqual(padded_adj[1].tolist(), [0, 2, -1])
        # c has neighbours a, b, in that order, so 0, 1, and padded -1
        self.assertListEqual(padded_adj[2].tolist(), [0, 1, -1])
        # d has neighbour a, so 0, and two padded -1
        self.assertListEqual(padded_adj[3].tolist(), [0, -1, -1])

        # Next, check weights are correct
        # a has edges 0, 1, 2
        self.assertListEqual(padded_adj_weights[0].tolist(), [0, 1, 2])
        # b has edges 0, 3,
        self.assertListEqual(padded_adj_weights[1].tolist(), [0, 3, -1])
        # c has edges 0, 3,
        self.assertListEqual(padded_adj_weights[2].tolist(), [1, 3, -1])
        # d has edges 2
        self.assertListEqual(padded_adj_weights[3].tolist(), [2, -1, -1])

    @parameterized.expand(GRBM_CRAYON_TEST_CASES)
    def test_get_partition(self, grbm: GRBM, crayon):
        bss = BlockSampler(grbm, crayon, 10, [1.0], seed=5)
        # Check every block is indeed coloured correctly
        for block in bss._partition:
            self.assertEqual(1, len({crayon(grbm.idx_to_node[bidx]) for bidx in block.tolist()}))
        # Check every node has been included
        self.assertSetEqual({idx for block in bss._partition for idx in block.tolist()},
                            {bss._grbm.node_to_idx[node] for node in bss._grbm.nodes})

    def test_invalid_crayon(self):
        grbm = GRBM([0, 1], [(0, 1)])
        def crayon(n): return 1
        self.assertRaisesRegex(ValueError, "not a valid colouring", BlockSampler, grbm, crayon, 10, [1.0])

    def test_invalid_proposal(self):
        grbm = GRBM([0, 1], [(0, 1)])
        def crayon(n): return 1
        self.assertRaisesRegex(ValueError, "Proposal acceptance criterion should be one of", BlockSampler,
                          grbm, crayon, 10, [1.0], "abc")

    def test_prepare_initial_states(self):
        grbm = GRBM([0, 1, 2], [(0, 1)])
        def crayon(n): return n
        bss = BlockSampler(grbm, crayon, 1, [1.0],)

        with self.subTest("Nonspin initial states."):
            self.assertRaisesRegex(ValueError, "contain nonspin values", bss._prepare_initial_states,
                              initial_states=torch.tensor([[0, 1, -1]]), num_chains=1)

        with self.subTest("Testing initial states with incorrect shape."):
            self.assertRaisesRegex(ValueError, "Initial states should be of shape", bss._prepare_initial_states,
                              num_chains=10, initial_states=torch.tensor([[-1, 1, 1, 1, -1]]))

    @parameterized.expand(GRBM_CRAYON_TEST_CASES)
    def test_invalid_num_reads(self, grbm, crayon):
        self.assertRaisesRegex(ValueError, "should be a positive integer", BlockSampler, grbm, crayon, 0, [1.0])


if __name__ == "__main__":
    unittest.main()
