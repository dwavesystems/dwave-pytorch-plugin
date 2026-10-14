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

from dwave.plugins.torch.nn.functional import maximum_mean_discrepancy_loss as mmd_loss
from dwave.plugins.torch.nn.modules.loss import MaximumMeanDiscrepancyLoss as MMDLoss
from tests.helper_functions import randspins


class TestMaximumMeanDiscrepancyLoss(unittest.TestCase):
    def test_module_is_the_functional(self):
        x, y = randspins(8, 5, seed=1).double(), torch.tanh(randspins(8, 5, seed=2).double() * 0.7)
        torch.testing.assert_close(MMDLoss()(x, y), mmd_loss(x, y))
        alpha = torch.tensor([0.1, 0.2, 0.3, 0.0, 0.4], dtype=torch.float64)
        torch.testing.assert_close(MMDLoss(alpha, block_size=4)(x, y), mmd_loss(x, y, alpha, 4))
        self.assertEqual((6,), tuple(MMDLoss()(x, y).shape))

    def test_kernel_weights_are_a_buffer(self):
        alpha = torch.tensor([0.5, 0.5])
        loss = MMDLoss(alpha)
        self.assertIn("alpha", loss.state_dict())
        self.assertEqual(0, len(list(loss.parameters())))
        alpha.zero_()  # held as a copy
        torch.testing.assert_close(loss.alpha, torch.tensor([0.5, 0.5]))
        self.assertEqual("meta", loss.to("meta").alpha.device.type)
        self.assertNotIn("alpha", MMDLoss().state_dict())
        self.assertIn("alpha of 2 orders", repr(loss))
        self.assertIn("gaussian", repr(MMDLoss()))
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            MMDLoss(-alpha)


if __name__ == "__main__":
    unittest.main()
