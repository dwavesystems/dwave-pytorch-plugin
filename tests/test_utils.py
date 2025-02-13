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

from dimod import SPIN, SampleSet
from torch import Tensor

from dwave.plugins.torch.utils import make_sampler_and_graph, sample_to_tensor


class TestUtils(unittest.TestCase):

    @unittest.expectedFailure
    def test_make_sampler_and_graph(self):
        raise NotImplementedError("TODO")

    def test_sample_to_tensor(self):
        ss = SampleSet.from_samples([[1, -1], [1, 1], [1, 1]], SPIN, [-1, 2, 2])
        spins = sample_to_tensor(ss)
        self.assertTupleEqual((3, 2), tuple(spins.shape))
        self.assertIsInstance(spins, Tensor)
