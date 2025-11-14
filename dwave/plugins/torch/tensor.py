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

from typing import TYPE_CHECKING, Optional

import torch

if TYPE_CHECKING:
    from torch import Generator
    from torch._prims_common import DeviceLikeType
    from torch.types import _bool, _dtype, _size

__all__ = ["randspin"]


def randspin(size: _size, **kwargs) -> torch.Tensor:
    """Wrapper for ``torch.randint`` restricted to spin outputs (+/-1 values).

    Args:
        size (torch.types._size): Shape of the output tensor.
        **kwargs: Keyword arguments of ``torch.randint``.

    Raises:
        ValueError: If ``low`` is supplied as a keyword argument.
        ValueError: If ``high`` is supplied as a keyword argument.

    Returns:
        torch.Tensor: A tensor of +/-1 values.
    """
    if "low" in kwargs:
        raise ValueError("Invalid keyword argument `low`.")
    if "high" in kwargs:
        raise ValueError("Invalid keyword argument `high`.")
    b = torch.randint(0, 2, size, **kwargs)
    return 2 * b - 1
