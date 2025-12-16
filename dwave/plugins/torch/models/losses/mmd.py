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

from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from dwave.plugins.torch.nn.modules.utils import store_config

__all__ = ["Kernel", "RadialBasisFunction", "maximum_mean_discrepancy", "MaximumMeanDiscrepancy"]


class Kernel(nn.Module):
    """Base class for kernels.

    Kernels are functions that compute a similarity measure between data points. Any ``Kernel``
    subclass must implement the ``_kernel`` method, which computes the kernel matrix for a given
    input multi-dimensional tensor with shape (n, f1, f2, ...), where n is the number of items
    and f1, f2, ... are feature dimensions, so that the output is a tensor of shape (n, n)
    containing the pairwise kernel values.
    """

    @abstractmethod
    def _kernel(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a pairwise kernel evaluation over samples.

        Computes the kernel matrix for an input of shape (n, f1, f2, ...), whose shape is (n, n)
        containing the pairwise kernel values.

        Args:
            x (torch.Tensor): A (n, f1, f2, ..., fk) tensor.

        Returns:
            torch.Tensor: A (n, n) tensor.
        """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Computes kernels for all pairs between and within ``x`` and ``y``.

        In general, ``x`` and ``y`` are (n_x, f1, f2, ..., fk) and (n_y, f1, f2, ..., fk)-shaped
        tensors, and the output is a (n_x + n_y, n_x + n_y)-shaped tensor containing the pairwise
        kernel values.

        Args:
            x (torch.Tensor): A (n_x, f1, f2, ..., fk) tensor.
            y (torch.Tensor): A (n_y, f1, f2, ..., fk) tensor.

        Returns:
            torch.Tensor: A (n_x + n_y, n_x + n_y) tensor.
        """
        if x.shape[1:] != y.shape[1:]:
            raise ValueError(
                "Input dimensions must match. You are trying to compute "
                f"the kernel between tensors of shape {x.shape} and {y.shape}."
            )
        xy = torch.cat([x, y], dim=0)
        return self._kernel(xy)


class RadialBasisFunction(Kernel):
    """The radial basis function kernel.

    This kernel between two data points x and y is defined as
    :math:`k(x, y) = exp(-||x-y||^2 / (2 * \sigma))`, where :math:`\sigma` is the bandwidth
    parameter.

    This implementation considers aggregating multiple radial basis function kernels with different
    bandwidths. The bandwidths are determined by multiplying a base bandwidth with a set of
    multipliers. The base bandwidth can be provided directly or estimated from the data using the
    average distance between samples.

    Args:
        num_features (int): Number of kernel bandwidths to use.
        mul_factor (int | float): Multiplicative factor to generate bandwidths. The bandwidths are
            computed as :math:`\sigma_i = \sigma * mul\_factor^{i - num\_features // 2}` for
            :math:`i` in ``[0, num_features - 1]``. Defaults to 2.0.
        bandwidth (float | None): Base bandwidth parameter. If None, the bandwidth is estimated
            from the data. Defaults to None.
    """

    @store_config
    def __init__(
        self, num_features: int, mul_factor: int | float = 2.0, bandwidth: Optional[float] = None
    ):
        super().__init__()
        bandwidth_multipliers = mul_factor ** (torch.arange(num_features) - num_features // 2)
        self.register_buffer("bandwidth_multipliers", bandwidth_multipliers)
        self.bandwidth = bandwidth

    @torch.no_grad()
    def _get_bandwidth(self, l2_distance_matrix: torch.Tensor) -> torch.Tensor | float:
        """Heuristically determine a bandwidth parameter as the average distance between samples.

        Computes the base bandwidth parameter as the average distance between samples if the
        bandwidth is not provided during initialization. Otherwise, returns the provided bandwidth.
        See https://arxiv.org/abs/1707.07269 for more details about the motivation behind taking
        the average distance as the bandwidth.

        Args:
            l2_distance_matrix (torch.Tensor): A (n, n) tensor representing the pairwise
                L2 distances between samples. If it is None and the bandwidth is not provided, an
                error will be raised. Defaults to None.

        Returns:
            torch.Tensor | float: The base bandwidth parameter.
        """
        if self.bandwidth is None:
            num_samples = l2_distance_matrix.shape[0]
            return l2_distance_matrix.sum() / (num_samples**2 - num_samples)
        return self.bandwidth

    def _kernel(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the radial basis function kernel as

        .. math::
            k(x, y) = \sum_{i=1}^{num\_features} exp(-||x-y||^2 / (2 * \sigma_i)),

        where :math:`\sigma_i` are the bandwidths.

        Args:
            x (torch.Tensor): A (n, f1, f2, ...) tensor.

        Returns:
            torch.Tensor: A (n, n) tensor representing the kernel matrix.
        """
        distance_matrix = torch.cdist(x, x, p=2)
        bandwidth = self._get_bandwidth(distance_matrix.detach()) * self.bandwidth_multipliers
        return torch.exp(-distance_matrix.unsqueeze(0) / bandwidth.reshape(-1, 1, 1)).sum(dim=0)


def maximum_mean_discrepancy(x: torch.Tensor, y: torch.Tensor, kernel: Kernel) -> torch.Tensor:
    """Computes the maximum mean discrepancy (MMD) loss between two sets of samples ``x`` and ``y``.

    This is a two-sample test to test the null hypothesis that the two samples are drawn from the
    same distribution (https://dl.acm.org/doi/abs/10.5555/2188385.2188410). The squared MMD is
    defined as

    .. math::
        MMD^2(X, Y) = \|E_{x\sim p}[\varphi(x)] - E_{y\sim q}[\varphi(y)] \|^2,

    where :math:`\varphi` is a feature map associated with the kernel function
    :math:`k(x, y) = \langle \varphi(x), \varphi(y) \rangle`, and :math:`p` and :math:`q` are the
    distributions of the samples. It follows that, in terms of the kernel function, the squared MMD
    can be computed as

    .. math::
        E_{x, x'\sim p}[k(x, x')] + E_{y, y'\sim q}[k(y, y')] - 2E_{x\sim p, y\sim q}[k(x, y)].

    If :math:`p = q`, then :math:`MMD^2(X, Y) = 0`. In machine learning applications, the MMD can be
    used as a loss function to compare the distribution of model-generated samples to the
    distribution of real data samples to force model-generated samples to match the real data
    distribution.

    Args:
        x (torch.Tensor): A (n_x, f1, f2, ...) tensor of samples from distribution p.
        y (torch.Tensor): A (n_y, f1, f2, ...) tensor of samples from distribution q.
        kernel (Kernel): A kernel function object.

    Returns:
        torch.Tensor: The computed MMD loss.
    """
    num_x = x.shape[0]
    num_y = y.shape[0]
    kernel_matrix = kernel(x, y)
    kernel_xx = kernel_matrix[:num_x, :num_x]
    kernel_yy = kernel_matrix[num_x:, num_x:]
    kernel_xy = kernel_matrix[:num_x, num_x:]
    xx = (kernel_xx.sum() - kernel_xx.trace()) / (num_x * (num_x - 1))
    yy = (kernel_yy.sum() - kernel_yy.trace()) / (num_y * (num_y - 1))
    xy = kernel_xy.sum() / (num_x * num_y)
    return xx + yy - 2 * xy


class MaximumMeanDiscrepancy(nn.Module):
    """Creates a module that computes the maximum mean discrepancy (MMD) loss between two sets of
    samples.

    This uses the `mmd_loss` function to compute the loss.

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
        return maximum_mean_discrepancy(x, y, self.kernel)
