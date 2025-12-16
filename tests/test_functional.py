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

from dwave.plugins.torch.nn.functional import SampleSizeError
from dwave.plugins.torch.nn.functional import maximum_mean_discrepancy_loss as mmd_loss
from dwave.plugins.torch.nn.modules.kernels import DimensionMismatchError, Kernel


class TestMaximumMeanDiscrepancyLoss(unittest.TestCase):
    def test_mmd_loss_constant(self):
        x = torch.tensor([[1.2], [4.1]])
        y = torch.tensor([[0.3], [0.5]])

        class Constant(Kernel):
            def __init__(self):
                super().__init__()
                self.k = torch.tensor([[10, 4, 0, 1],
                                       [4, 10, 4, 2],
                                       [0, 4, 10, 3],
                                       [1, 2, 3, 10]]).float()

            def _kernel(self, x, y):
                return self.k
        # The resulting kernel matrix will be constant, so (averages) KXX = KYY = 2KXY
        kernel = Constant()
        # kxx = (4 + 4)/2
        # kyy = (3 + 3)/2
        # kxy = (0 + 1 + 4 + 2)/4
        # kxx + kyy -2kxy = 4 + 3 - 3.5 = 3.5
        self.assertEqual(3.5, mmd_loss(x, y, kernel))

    def test_sample_size_error(self):
        x = torch.tensor([[1.2], [4.1]])
        y = torch.tensor([[0.3]])
        self.assertRaises(SampleSizeError, mmd_loss, x, y, None)

    def test_mmd_loss_dim_mismatch(self):
        x = torch.tensor([[1], [4]], dtype=torch.float32)
        y = torch.tensor([[0.1, 0.2, 0.3],
                          [0.4, 0.5, 0.6]])
        self.assertRaises(DimensionMismatchError, mmd_loss, x, y, None)

    def test_mmd_loss_arange(self):
        x = torch.tensor([[1.0], [4.0], [5.0]])
        y = torch.tensor([[0.3], [0.4]])

        class Constant(Kernel):
            def _kernel(self, x, y):
                return torch.tensor([[150, 22, 39, 34, 28],
                                     [22, 630, 98, 56, 44],
                                     [39, 98, 560, 78, 33],
                                     [-99, -99, -99, 299, 13],
                                     [-99, -99, -99, 13, 970]], dtype=torch.float32)

        mmd_loss(x, y, Constant())
        # NOTE: calculation takes kxy = upper-right corner; no PSD assumption
        # kxx = (22+39+98)/3
        # kyy = 13
        # kxy = (34+28+56+44+78+33)/6
        # kxx + kyy - 2*kxy
        # kxx + kyy - 2*kxy = -25.0
        self.assertEqual(-25, mmd_loss(x, y, Constant()))


if __name__ == "__main__":
    unittest.main()
