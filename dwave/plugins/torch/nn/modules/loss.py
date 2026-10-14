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

import torch
import torch.nn as nn

from dwave.plugins.torch.kernels import _validate_alpha
from dwave.plugins.torch.nn.functional import maximum_mean_discrepancy_loss as mmd_loss

__all__ = ["MaximumMeanDiscrepancyLoss"]


class MaximumMeanDiscrepancyLoss(nn.Module):
    """The unbiased squared maximum mean discrepancy (MMD) between two samples of spins, under
    every kernel of the battery, as a loss module.

    A module wrapper of :func:`~dwave.plugins.torch.nn.functional.maximum_mean_discrepancy_loss`:
    the forward pass returns one estimate per profile, entry 0 the headline kernel (the Gaussian
    kernel, or the alpha row when kernel weights are held) and entry ``k`` the order-``k``
    kernel, differentiable in both samples, which may be hard or soft spins in ``[-1, 1]``.
    ``loss(x, y)[0].backward()`` trains on the headline. The kernel weights are a buffer, so they
    move with :meth:`~torch.nn.Module.to` and are saved in :meth:`~torch.nn.Module.state_dict`.

    Args:
        alpha (torch.Tensor, optional): Kernel weights over orders ``1..dim``, a 1-D tensor of
            finite nonnegative entries with at least one positive, for example the weights an
            :class:`~dwave.plugins.torch.evaluation.Evaluator` calibrated. Defaults to ``None``,
            the Gaussian headline.
        block_size (int, optional): Rows per block of the paired blocked estimate; ``>= 2`` and a
            divisor of the samples' common row count. Defaults to ``None``, one block.

    Attributes:
        alpha (torch.Tensor | None): The kernel weights, a buffer, or ``None``.
        block_size (int | None): Rows per block.

    Raises:
        ValueError: If ``alpha`` is not a 1-D tensor of finite nonnegative entries with at least
            one positive.
    """

    alpha: torch.Tensor | None

    def __init__(self, alpha: torch.Tensor | None = None, block_size: int | None = None) -> None:
        super().__init__()
        if alpha is not None:
            _validate_alpha(alpha)
            alpha = alpha.clone()
        self.register_buffer("alpha", alpha)
        self.block_size = block_size

    def extra_repr(self) -> str:
        head = "gaussian" if self.alpha is None else f"alpha of {self.alpha.shape[0]} orders"
        return f"headline={head}, block_size={self.block_size}"

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes the per-profile MMD estimates between two samples ``x`` and ``y``.

        Args:
            x (torch.Tensor): A ``(n_x, dim)`` tensor of hard or soft spins from distribution p.
            y (torch.Tensor): A ``(n_y, dim)`` tensor of hard or soft spins from distribution q.

        Returns:
            torch.Tensor: A ``(dim + 1,)`` tensor of unbiased squared MMD estimates, entry 0 the
            headline kernel and entry ``k`` the order-``k`` kernel.
        """
        return mmd_loss(x, y, self.alpha, self.block_size)
