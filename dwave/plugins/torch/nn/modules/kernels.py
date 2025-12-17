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
"""Kernel functions."""

from abc import abstractmethod

import torch
import torch.nn as nn

from dwave.plugins.torch.nn.functional import DimensionMismatchError
from dwave.plugins.torch.nn.modules.utils import store_config

__all__ = ["Kernel", "GaussianKernel"]


class Kernel(nn.Module):
    """Base class for kernels.

    `Kernels <https://en.wikipedia.org/wiki/Kernel_method>`_ are functions that compute a similarity
    measure between data points. Any ``Kernel`` subclass must implement the ``_kernel`` method,
    which computes the kernel matrix for a given input multi-dimensional tensor with shape
    (n, f1, f2, ...), where n is the number of items and f1, f2, ... are feature dimensions, so that
    the output is a tensor of shape (n, n) containing the pairwise kernel values.
    """
    @abstractmethod
    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Perform a pairwise kernel evaluation over samples.

        Computes the kernel matrix for an input of shape (n, f1, f2, ...), whose shape is (n, n)
        containing the pairwise kernel values.

        Args:
            x (torch.Tensor): A (nx, f1, f2, ..., fk) tensor.
            y (torch.Tensor): A (ny, f1, f2, ..., fk) tensor.

        Returns:
            torch.Tensor: A (nx, ny) tensor.
        """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes kernels for all pairs between and within ``x`` and ``y``.

        In general, ``x`` and ``y`` are (n_x, f1, f2, ..., fk)- and (n_y, f1, f2, ..., fk)-shaped
        tensors, and the output is a (n_x + n_y, n_x + n_y)-shaped tensor containing pairwise kernel
        evaluations.

        Args:
            x (torch.Tensor): A (n_x, f1, f2, ..., fk) tensor.
            y (torch.Tensor): A (n_y, f1, f2, ..., fk) tensor.

        Raises:
            DimensionMismatchError: If shape of ``x`` and ``y`` mismatch (excluding batch size)

        Returns:
            torch.Tensor: A (n_x + n_y, n_x + n_y) tensor.
        """
        if x.shape[1:] != y.shape[1:]:
            raise DimensionMismatchError(
                "Input dimensions must match. You are trying to compute "
                f"the kernel between tensors of shape {x.shape} and {y.shape}."
            )
        return self._kernel(x, y)


class GaussianKernel(Kernel):
    """The Gaussian kernel.

    This kernel between two data points x and y is defined as
    :math:`k(x, y) = exp(-||x-y||^2 / (2 * \sigma))`, where :math:`\sigma` is the bandwidth
    parameter.

    This implementation considers aggregating multiple Gaussian kernels with different
    bandwidths. The bandwidths are determined by multiplying a base bandwidth with a set of
    multipliers. The base bandwidth can be provided directly or estimated from the data using the
    average distance between samples.

    Args:
        n_kernels (int): Number of kernel bandwidths to use.
        factor (int | float): Multiplicative factor to generate bandwidths. The bandwidths are
            computed as :math:`\sigma_i = \sigma * factor^{i - n\_kernels // 2}` for
            :math:`i` in ``[0, n\_kernels - 1]``. Defaults to 2.0.
        bandwidth (float | None): Base bandwidth parameter. If ``None``, the bandwidth is computed
            from the data (without gradients). Defaults to ``None``.
    """

    @store_config
    def __init__(
        self, n_kernels: int, factor: int | float = 2.0, bandwidth: float | None = None
    ):
        super().__init__()
        factors = factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.register_buffer("factors", factors)
        self.bandwidth = bandwidth

    @torch.no_grad()
    def _get_bandwidth(self, distance_matrix: torch.Tensor) -> torch.Tensor | float:
        """Heuristically determine a bandwidth parameter as the average distance between samples.

        Computes the base bandwidth parameter as the average distance between samples if the
        bandwidth is not provided during initialization. Otherwise, returns the provided bandwidth.
        See https://arxiv.org/abs/1707.07269 for more details about the motivation behind taking
        the average distance as the bandwidth.

        Args:
            distance_matrix (torch.Tensor): A (n, n) tensor representing the pairwise
                L2 distances between samples. If it is ``None`` and the bandwidth is not provided,
                an error will be raised. Defaults to ``None``.

        Returns:
            torch.Tensor | float: The base bandwidth parameter.
        """
        if self.bandwidth is None:
            num_samples = distance_matrix.shape[0]
            return distance_matrix.sum() / (num_samples**2 - num_samples)
        return self.bandwidth

    def _kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the Gaussian kernel between ``x`` and ``y``.

        .. math::
            k(x, y) = \sum_{i=1}^{num\_features} exp(-||x-y||^2 / (2 * \sigma_i)),

        where :math:`\sigma_i` are the bandwidths.

        Args:
            x (torch.Tensor): A (nx, f1, f2, ..., fk) tensor.
            y (torch.Tensor): A (ny, f1, f2, ..., fk) tensor.

        Returns:
            torch.Tensor: A (nx, ny) tensor representing the kernel matrix.
        """
        distance_matrix = torch.cdist(x.flatten(1), y.flatten(1), p=2)
        bandwidth = self._get_bandwidth(distance_matrix.detach()) * self.factors
        return torch.exp(-distance_matrix.unsqueeze(0) / bandwidth.reshape(-1, 1, 1)).sum(dim=0)
