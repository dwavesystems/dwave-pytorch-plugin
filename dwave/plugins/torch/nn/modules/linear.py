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
#
import torch
from torch import nn

from dwave.plugins.torch.nn.modules.utils import store_config

__all__ = ["SkipLinear", "LinearBlock"]


class SkipLinear(nn.Module):
    """A linear transformation or the identity depending on whether input/output dimensions match.

    This module is identity when ``din == dout``, otherwise, it is a linear transformation (no bias
    term).

    This is based on the `ResNet paper <https://arxiv.org/abs/1512.03385>`.

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

    This is based on the `ResNet paper <https://arxiv.org/abs/1512.03385>`_.

    Args:
        din (int): Size of each input sample.
        dout (int): Size of each output sample.
        p (float): Dropout probability.
    """

    @store_config
    def __init__(self, din: int, dout: int, p: float) -> None:
        super().__init__()
        self._skip = SkipLinear(din, dout)
        dhid = max(din, dout)
        self._block = nn.Sequential(
            nn.LayerNorm(din),
            nn.Linear(din, dhid),
            nn.Dropout(p),
            nn.ReLU(),
            nn.LayerNorm(dhid),
            nn.Linear(dhid, dout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms the input ``x`` with the modules.

        Args:
            x (torch.Tensor): An input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self._block(x) + self._skip(x)
