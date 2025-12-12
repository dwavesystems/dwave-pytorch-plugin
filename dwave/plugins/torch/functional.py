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

import torch


def bit2spin_soft(b: torch.Tensor) -> torch.Tensor:
    """Maps input `b` to `2b-1`.

    The mapping does not require ``b`` to be binary, only that it is in the interval `[0, 1]`.

    Args:
        b (torch.Tensor): Input tensor of values in `[0, 1]`.

    Raises:
        ValueError: If not all ``b`` values are in `[0, 1]`.

    Returns:
        torch.Tensor: A tensor with values `2b-1`.
    """
    if not ((b >= 0) & (b <= 1)).all():
        raise ValueError(f"Not all inputs are in [0, 1]: {b}")
    return b * 2 - 1


def spin2bit_soft(s: torch.Tensor) -> torch.Tensor:
    """Maps input `s` to `(s+1)/2`.

    The mapping does not require ``s`` to be spin-valued, only that it is in the interval `[-1, 1]`.

    Args:
        s (torch.Tensor): Input tensor of values in `[-1, 1]`.

    Raises:
        ValueError: If not all ``s`` values are in `[-1, 1]`.

    Returns:
        torch.Tensor: A tensor with values `(s+1)/2`.
    """
    if (s.abs() > 1).any():
        raise ValueError(f"Not all inputs are in [-1, 1]: {s}")
    return (s + 1) / 2
