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

import inspect
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from functools import partial

from functools import wraps
from types import MappingProxyType

import torch
from torch import nn

__all__ = ["store_config", "SkipLinear", "LinearBlock"]


def store_config(fn: Callable) -> partial:
    """A decorator that tracks and stores arguments of methods (excluding ``self``).

    Args:
        fn (Callable[object, ...]): A method whose arguments will be stored in ``self.config``.

    Returns:
        partial: Wrapper function that stores argument of method.
    """
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        """Store ``args``, ``kwargs``, and ``{"module_name": self.__class__.__name__}`` as a dictionary in ``self.config``.
        """
        sig = inspect.signature(fn)
        bound = sig.bind(self, *args, **kwargs)
        bound.apply_defaults()

        config = {k: v for k, v in bound.arguments.items() if v != self}
        config['module_name'] = self.__class__.__name__
        self.config = MappingProxyType(config)

        fn(self, *args, **kwargs)
    return wrapper


class SkipLinear(nn.Module):
    """Applies a linear transformation to the incoming data: :math:`y = xA^T`.

    This module is identity when ``din == dout``, otherwise, it is a linear transformation, i.e.,
    no bias term.

    Args:
        din (int): Size of each input sample.
        dout (int): Size of each output sample.
    """

    @store_config
    def __init__(self, din: int, dout: int) -> None:
        super().__init__()
        if din == dout:
            self.linear = nn.Identity()
        else:
            self.linear = nn.Linear(din, dout, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a linear transformation to the input variable ``x``.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The linearly-transformed tensor of ``x``.
        """
        return self.linear(x)


class LinearBlock(nn.Module):
    """A linear block consisting of normalizations, linear transformations, dropout, relu, and a skip connection.

    The module is composed of (in order):
    1. a first layer norm,
    2. a first linear transformation,
    3. a dropout,
    4. a relu activation,
    5. a second layer norm,
    6. a second linear layer, and, finally,
    7. a skip connection from initial input to output.

    Args:
        din (int): Size of each input sample.
        dout (int): Size of each output sample.
        p (float): Dropout probability.
    """
    @store_config
    def __init__(self, din: int, dout: int, p: float) -> None:
        super().__init__()
        self._skip = SkipLinear(din, dout)
        self.block = nn.Sequential(
            nn.LayerNorm(din),
            nn.Linear(din, dout),
            nn.Dropout(p),
            nn.ReLU(),
            nn.LayerNorm(dout),
            nn.Linear(dout, dout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms the input `x` with the modules.

        Args:
            x (torch.Tensor): An input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.block(x) + self._skip(x)
