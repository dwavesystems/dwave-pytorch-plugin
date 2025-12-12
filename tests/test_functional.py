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
from parameterized import parameterized

from dwave.plugins.torch.functional import bit2spin_soft, spin2bit_soft


class TestFunctional(unittest.TestCase):

    def test_spin2bit_soft(self):
        self.assertListEqual(spin2bit_soft(torch.tensor([-1.0, 1.0, 0.5])).tolist(), [0, 1, 0.75])

    @parameterized.expand([([-1.1, 1.0],), ([-0.5, 1.1],)])
    def test_spin2bit_raises(self, input):
        self.assertRaises(ValueError, spin2bit_soft, torch.tensor(input))

    def test_bit2spin_soft(self):
        self.assertListEqual(bit2spin_soft(torch.tensor([0.0, 1.0, 0.5])).tolist(), [-1, 1, 0])

    @parameterized.expand([([-0.1, 1.0],), ([0.1, 1.1],)])
    def test_bit2spin_soft_raises(self, input):
        self.assertRaises(ValueError, bit2spin_soft, torch.tensor(input))


if __name__ == "__main__":
    unittest.main()
