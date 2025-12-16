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

from dwave.plugins.torch.nn.functional import maximum_mean_discrepancy_loss as mmd_loss
from dwave.plugins.torch.nn.modules.kernels import Kernel
from dwave.plugins.torch.nn.modules.loss import MaximumMeanDiscrepancyLoss as MMDLoss


class TestMaximumMeanDiscrepancyLoss(unittest.TestCase):
    @parameterized.expand([
        (torch.tensor([[1.2], [4.1]]), torch.tensor([[0.3], [0.5]])),
        (torch.randn((123, 4, 3, 2)), torch.rand(100, 4, 3, 2)),
    ])
    def test_mmd_loss(self, x, y):
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
        compute_mmd = MMDLoss(kernel)
        torch.testing.assert_close(mmd_loss(x, y, kernel), compute_mmd(x, y))


if __name__ == "__main__":
    unittest.main()
