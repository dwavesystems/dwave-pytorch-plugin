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
from dwave.plugins.torch.samplers.block_spin_sampler import BlockSpinSampler


class TestBlockSpinSampler(unittest.TestCase):
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

    @parameterized.expand(GRBM_CRAYON_TEST_CASES)
    def test_sample(self, grbm, crayon):
        for pac in "Metropolis", "Gibbs":
            schedule = [0.0, 1.0, 2.0]
            bss1 = BlockSpinSampler(grbm, crayon, 10, pac, seed=1)
            bss1.sample(schedule)

            bss2 = BlockSpinSampler(grbm, crayon, 10, pac, seed=1)
            for beta in schedule:
                bss2.step_(beta)

            self.assertListEqual(bss1.x.tolist(), bss2.x.tolist())

    def test_gibbs_update(self):
        grbm = GRBM(list("ab"), [["a", "b"]])

        def crayon(v):
            return v == "a"
        sample_size = 1_000_000
        bss = BlockSpinSampler(grbm, crayon, sample_size, "Gibbs", seed=2)
        bss.x.data[:] = 1
        zero = torch.tensor(0.0)
        ones = torch.ones((sample_size, 1))
        bss._gibbs_update(0.0, bss._partition[0], ones*zero)
        torch.testing.assert_close(torch.tensor(0.5), bss.x.mean(), atol=1e-3, rtol=1e-3)
        bss._gibbs_update(0.0, bss._partition[1], ones*zero)
        torch.testing.assert_close(torch.tensor(0.0), bss.x.mean(), atol=1e-3, rtol=1e-3)

        effective_field = torch.tensor(1.2)
        bss._gibbs_update(1.0, bss._partition[0], effective_field*ones)
        bss._gibbs_update(1.0, bss._partition[1], effective_field*ones)
        torch.testing.assert_close(torch.tanh(-effective_field), bss.x.mean(), atol=1e-3, rtol=1e-3)

    def test_metropolis_update_average(self):
        grbm = GRBM(list("ab"), [["a", "b"]])

        def crayon(v):
            return v == "a"
        sample_size = 1_000_000
        bss = BlockSpinSampler(grbm, crayon, sample_size, "Metropolis", seed=2)
        bss.x.data[:] = 1
        ones = torch.ones((sample_size, 1))
        effective_field = torch.tensor(1.2)
        for i in range(10):
            bss._metropolis_update(1.0, bss._partition[0], effective_field*ones)
            bss._metropolis_update(1.0, bss._partition[1], effective_field*ones)
        torch.testing.assert_close(torch.tanh(-effective_field), bss.x.mean(), atol=1e-3, rtol=1e-3)

    def test_metropolis_update_oscillates(self):
        grbm = GRBM(list("ab"), [["a", "b"]])

        def crayon(v):
            return v == "a"
        sample_size = 1_00
        bss = BlockSpinSampler(grbm, crayon, sample_size, "Metropolis", seed=2)
        bss.x.data[:] = 1
        zero_effective_field = torch.zeros((sample_size, 1))
        bss._metropolis_update(0.0, bss._partition[0], zero_effective_field)
        self.assertTrue((bss.x[:, 1] == -1).all())
        bss._metropolis_update(0.0, bss._partition[1], zero_effective_field)
        self.assertTrue((bss.x == -1).all())

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
        bss = BlockSpinSampler(grbm, crayon, 3, seed=3)
        bss.x.data[:] = torch.tensor([[1, 1, -1, -1],
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
        bss = BlockSpinSampler(grbm, crayon, 10, seed=4)
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
        bss = BlockSpinSampler(grbm, crayon, 10, seed=5)
        # Check every block is indeed coloured correctly
        for block in bss._partition:
            self.assertEqual(1, len({crayon(grbm.idx_to_node[bidx]) for bidx in block.tolist()}))
        # Check every node has been included
        self.assertSetEqual({idx for block in bss._partition for idx in block.tolist()},
                            {bss._grbm.node_to_idx[node] for node in bss._grbm.nodes})

    def test_invalid_crayon(self):
        grbm = GRBM([0, 1], [(0, 1)])
        def crayon(n): return 1
        self.assertRaises(ValueError, BlockSpinSampler, grbm, crayon, 10)

    @parameterized.expand(GRBM_CRAYON_TEST_CASES)
    def test_invalid_num_reads(self, grbm, crayon):
        self.assertRaises(ValueError, BlockSpinSampler, grbm, crayon, 0)


if __name__ == "__main__":
    unittest.main()
