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
"""Functional interface."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dwave.plugins.torch.nn.modules.kernels import Kernel

import torch

__all__ = ["maximum_mean_discrepancy_loss"]


class SampleSizeError(ValueError):
    pass


class DimensionMismatchError(ValueError):
    pass


def maximum_mean_discrepancy_loss(x: torch.Tensor, y: torch.Tensor, kernel: Kernel) -> torch.Tensor:
    """Estimates the squared maximum mean discrepancy (MMD) given two samples ``x`` and ``y``.

    The `squared MMD <https://dl.acm.org/doi/abs/10.5555/2188385.2188410>`_ is defined as

    .. math::
        MMD^2(X, Y) = |E_{x\sim p}[\varphi(x)] - E_{y\sim q}[\varphi(y)] |^2,

    where :math:`\varphi` is a feature map associated with the kernel function
    :math:`k(x, y) = \langle \varphi(x), \varphi(y) \rangle`, and :math:`p` and :math:`q` are the
    distributions of the samples. It follows that, in terms of the kernel function, the squared MMD
    can be computed as

    .. math::
        E_{x, x'\sim p}[k(x, x')] + E_{y, y'\sim q}[k(y, y')] - 2E_{x\sim p, y\sim q}[k(x, y)].

    If :math:`p = q`, then :math:`MMD^2(X, Y) = 0`. This motivates the squared MMD as a loss
    function for minimizing the distance between the model distribution and data distribution.

    For more information, see
    Gretton, A., Borgwardt, K. M., Rasch, M. J., Schölkopf, B., & Smola, A. (2012).
    A kernel two-sample test. The journal of machine learning research, 13(1), 723-773.

    Args:
        x (torch.Tensor): A (n_x, f1, f2, ..., fk) tensor of samples from distribution p.
        y (torch.Tensor): A (n_y, f1, f2, ..., fk) tensor of samples from distribution q.
        kernel (Kernel): A kernel function object.

    Raises:
        SampleSizeError: If the sample size of ``x`` or ``y`` is less than two.
        DimensionMismatchError: If shape of ``x`` and ``y`` mismatch (excluding batch size)

    Returns:
        torch.Tensor: The squared maximum mean discrepancy estimate.
    """
    num_x = x.shape[0]
    num_y = y.shape[0]
    if num_x < 2 or num_y < 2:
        raise SampleSizeError(
            "Sample size of ``x`` and ``y`` must be at least two. "
            f"Got, respectively, {x.shape} and {y.shape}."
        )
    if x.shape[1:] != y.shape[1:]:
        raise DimensionMismatchError(
            "Input dimensions must match. You are trying to compute "
            f"the kernel between tensors of shape {x.shape} and {y.shape}."
        )
    xy = torch.cat([x, y], dim=0)
    kernel_matrix = kernel(xy, xy)
    kernel_xx = kernel_matrix[:num_x, :num_x]
    kernel_yy = kernel_matrix[num_x:, num_x:]
    kernel_xy = kernel_matrix[:num_x, num_x:]
    xx = (kernel_xx.sum() - kernel_xx.trace()) / (num_x * (num_x - 1))
    yy = (kernel_yy.sum() - kernel_yy.trace()) / (num_y * (num_y - 1))
    xy = kernel_xy.sum() / (num_x * num_y)
    return xx + yy - 2 * xy
