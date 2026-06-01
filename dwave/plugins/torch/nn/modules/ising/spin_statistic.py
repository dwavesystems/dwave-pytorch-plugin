# Copyright 2026 D-Wave
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

import abc
from collections.abc import Iterable

import torch

__all__ = ["SpinStatistic", "IdentityStatistic", "IsingStatistic"]


class SpinStatistic(abc.ABC):
    """Untrainable spin statistics for Ising output statistics.

    .. caution:: These transformations should not contain any trainable parameters. While possible,
    current implementation does not accumulate gradients for parameters used in such functions.

    Args:
        dim_out: The output dimension of the statistic.
    """

    def __init__(self, dim_out: int) -> None:
        self._dim_out = dim_out

    @property
    def dim_out(self) -> int:
        """Output dimension of statistic."""
        return self._dim_out

    @abc.abstractmethod
    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """The main function that will be invoked by ``__call__``.

        Args:
            x (torch.Tensor): Input spins of shape (B, N, D) where B is batch size, N is sample size,
                and D is the number of spins.

        Returns:
            torch.Tensor: Output statistic applied to the third dimension of ``x``.
        """

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        """Applies the defined transformation.

        Args:
            x: Input spins of shape (B, N, D) where B is batch size, N is sample size, and D is the
                number of spins.

        Raises:
            ValueError: If input does not have 3 dimensions.
            ValueError: If output does not have 3 dimensions.
            ValueError: Output dimension does not match that of promised.

        Returns:
            torch.Tensor: Output statistic applied to the third dimension of ``x``.
        """
        if tensor.ndim != 3:
            raise ValueError("Input tensor should have `ndim == 3`.")

        output = self._transform(tensor)

        if output.ndim != 3:
            raise ValueError("Output tensor should have `ndim == 3`.")

        if output.shape[-1] != self.dim_out:
            raise ValueError(f"Output dimension ({output.shape[2]}) does not match that "
                             f"promised ({self.dim_out}).")
        return output


class IdentityStatistic(SpinStatistic):
    """Identity function of statistics.

    Args:
        input_dim: Dimension of inputs.
    """

    def __init__(self, dim_in: int) -> None:
        super().__init__(dim_in)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """Identity function."""
        return x


class IsingStatistic(SpinStatistic):
    """Sufficient statistics of Ising models; a concatenation of node values and pairwise products.

    Computes the concatenation of selected node spins and element-wise
    products of spin pairs, i.e. [x[..., indices], x[..., indices_j] * x[..., indices_i]].

    Args:
        node_indices: Indices of nodes to include directly.
        endpoints_1: First indices for pairwise interaction terms.
        endpoints_2: Second indices for pairwise interaction terms.
    """

    def __init__(
        self,
        node_indices: Iterable[int],
        endpoints_1: Iterable[int],
        endpoints_2: Iterable[int]
    ) -> None:
        self.node_indices = list(node_indices)
        self.endpoints_1 = list(endpoints_1)
        self.endpoints_2 = list(endpoints_2)

        if len(self.endpoints_1) != len(self.endpoints_2):
            raise ValueError(
                "Interaction indices should be of the same length, got "
                f"length {len(self.endpoints_1)} for i and length {len(self.endpoints_2)} for j"
            )

        dim_out = len(self.node_indices) + len(self.endpoints_1)
        super().__init__(dim_out)

    def _transform(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` and pairwise products defined by the given indices."""
        return torch.cat([x[..., self.node_indices], x[..., self.endpoints_2] * x[..., self.endpoints_1]], -1)
