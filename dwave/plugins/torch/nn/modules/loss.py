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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from dwave.plugins.torch.nn.functional import maximum_mean_discrepancy_loss as mmd_loss
from dwave.plugins.torch.nn.modules.utils import store_config

if TYPE_CHECKING:
    from dwave.plugins.torch.nn.modules.kernels import Kernel

__all__ = ["MaximumMeanDiscrepancyLoss"]


class MaximumMeanDiscrepancyLoss(nn.Module):
    """An unbiased estimator for the squared maximum mean discrepancy as a loss function.

    This uses the ``dwave.plugins.torch.nn.functional.maximum_mean_discrepancy_loss`` function to
    compute the loss.

    Args:
        kernel (Kernel): A kernel function object.
    """

    @store_config
    def __init__(self, kernel: Kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the MMD loss between two sets of samples x and y.

        Args:
            x (torch.Tensor): A (n_x, f1, f2, ...) tensor of samples from distribution p.
            y (torch.Tensor): A (n_y, f1, f2, ...) tensor of samples from distribution q.

        Returns:
            torch.Tensor: The computed MMD loss.
        """
        return mmd_loss(x, y, self.kernel)
