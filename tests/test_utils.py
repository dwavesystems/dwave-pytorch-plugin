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

import networkx as nx

from dimod import SPIN, SampleSet
from torch import Tensor

from dwave.plugins.torch.utils import make_sampler_and_graph, sample_to_tensor
from dwave.system.testing import MockDWaveSampler
from dwave.embedding import is_valid_embedding


class TestUtils(unittest.TestCase):

    def test_make_sampler_and_graph(self):
        qpu = MockDWaveSampler(topology_type="zephyr", topology_shape=(1, 4))
        qpu.nodelist = [1, 2, 34, 35]
        qpu.edgelist = [(1, 34), (2, 34), (34, 35)]
        T = nx.from_edgelist(qpu.edgelist)
        sampler, S = make_sampler_and_graph(qpu)
        self.assertSetEqual(set(S.nodes), {0, 1, 2, 3})
        self.assertTrue(nx.algorithms.is_isomorphic(S, T))
        self.assertTrue(is_valid_embedding(sampler.embedding, S, T))

    def test_sample_to_tensor(self):
        ss = SampleSet.from_samples([[1, -1], [1, 1], [1, 1]], SPIN, [-1, 2, 2])
        spins = sample_to_tensor(ss)
        self.assertTupleEqual((3, 2), tuple(spins.shape))
        self.assertIsInstance(spins, Tensor)


if __name__ == "__main__":
    unittest.main()
