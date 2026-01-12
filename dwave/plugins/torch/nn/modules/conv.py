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

__all__ = ["SkipConv2d", "ConvolutionBlock"]

class SkipConv2d(nn.Module):
    """A 2D convolution or the identity depending on whether input/output channels match.

    This module is identity when ``cin == cout``, otherwise it applies a 1×1 convolution
    without bias to match channel dimensions. This is used for residual (skip) connections
    as described in the ResNet architecture.

    Args:
        cin (int): Number of input channels.
        cout (int): Number of output channels.
    """

    @store_config
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        if cin == cout:
            self.conv = nn.Identity()
        else:
            self.conv = nn.Conv2d(cin, cout, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the skip connection transformation.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, cin, H, W)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(N, cout, H, W)``.
        """
        return self.conv(x)

class ConvolutionBlock(nn.Module):
    """A residual convolutional block with normalization, convolutions, and a skip connection.

    The block consists of:

    1. Layer normalization over the input,
    2. a 3×3 convolution,
    3. a ReLU activation,
    4. a second layer normalization,
    5. a second 3×3 convolution, and
    6. a skip connection from input to output.

    This block preserves spatial resolution and follows the residual learning
    principle introduced in the ResNet paper.

    Args:
        input_shape (tuple[int, int, int]): Input shape ``(channels, height, width)``.
        cout (int): Number of output channels.

    Raises:
        NotImplementedError: If input height and width are not equal.
    """
    
    @store_config
    def __init__(self, input_shape: tuple[int, int, int], cout: int) -> None:
        super().__init__()

        cin, hx, wx = tuple(input_shape)
        if hx != wx:
            raise NotImplementedError("Only square inputs are currently supported.")

        self._block = nn.Sequential(
            nn.LayerNorm((cin, hx, wx)),
            nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.LayerNorm((cout, hx, wx)),
            nn.Conv2d(cout, cout, kernel_size=3, stride=1, padding=1),
        )
        self._skip = SkipConv2d(cin, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolutional block and skip connection.

        Args:
            x (torch.Tensor): Input tensor of shape ``(N, cin, H, W)``.

        Returns:
            torch.Tensor: Output tensor of shape ``(N, cout, H, W)``.
        """
        return self._block(x) + self._skip(x)