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

if TYPE_CHECKING:
    from dimod import SampleSet


def sampleset_to_tensor(
    ordered_vars: list, sample_set: SampleSet, device: Optional[torch.device] = None
) -> torch.Tensor:
    """Converts a ``dimod.SampleSet`` to a ``torch.Tensor``.

    Args:
        ordered_vars: list[Literal]: The desired order of sample set variables.
        sample_set (dimod.SampleSet): A sample set.
        device (torch.device, optional): The device of the constructed tensor.
            If ``None`` and data is a tensor then the device of data is used.
            If ``None`` and data is not a tensor then the result tensor is constructed
            on the current device.

    Returns:
        torch.Tensor: The sample set as a ``torch.Tensor``.
    """
    var_to_sample_i = {v: i for i, v in enumerate(sample_set.variables)}
    permutation = [var_to_sample_i[v] for v in ordered_vars]
    sample = sample_set.record.sample[:, permutation]
    return torch.tensor(sample, dtype=torch.float32, device=device)


def straight_through_bitrounding(fuzzy_bits):
    if not ((fuzzy_bits >= 0) & (fuzzy_bits <= 1)).all():
        raise ValueError(f"Inputs should be in [0, 1]: {fuzzy_bits}")
    bits = fuzzy_bits + (fuzzy_bits.round() - fuzzy_bits).detach()
    return bits


def bit2spin_soft(b):
    if not ((b >= 0) & (b <= 1)).all():
        raise ValueError(f"Not all inputs are in [0, 1]: {b}")
    return b * 2.0 - 1.0


def spin2bit_soft(s):
    if (s.abs() > 1).any():
        raise ValueError(f"Not all inputs are in [-1, 1]: {s}")
    return (s + 1.0) / 2.0


def rands_like(x):
    return rands(x.shape, device=x.device)


def randb_like(x):
    return randb(x.shape, device=x.device)


def randb(shape, device=None):
    return torch.randint(0, 2, shape, device=device)


def rands(shape, device=None):
    if isinstance(shape, int):
        shape = (shape,)
    return bit2spin_soft(torch.randint(0, 2, shape, device=device))
