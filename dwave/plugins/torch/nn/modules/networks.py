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

from dwave.plugins.torch.nn.modules.conv import ConvolutionBlock, SkipConv2d
from dwave.plugins.torch.nn.modules.linear import LinearBlock, SkipLinear
from dwave.plugins.torch.nn.modules.vision import CACBlock


class FullyConnectedNetwork(nn.Module):
    def __init__(self, din, dout, depth, sn, p) -> None:
        super().__init__(self, vars())
        if depth == 1:
            raise ValueError("Depth must be at least 2.")
        self.skip = SkipLinear(din, dout)
        big_d = max(din, dout)
        dims = [big_d]*(depth-1) + [dout]
        self.blocks = nn.Sequential()
        for d_in, d_out in zip([din]+dims[:-1], dims):
            self.blocks.append(LinearBlock(d_in, d_out, sn, p))
            self.blocks.append(nn.Dropout(p))
            self.blocks.append(nn.ReLU())
        # Remove the last ReLU and Dropout
        self.blocks.pop(-1)
        self.blocks.pop(-1)

    def forward(self, x):
        return self.blocks(x) + self.skip(x)


class ConvolutionNetwork(nn.Module):
    def __init__(
            self, channels: list[int], input_shape: tuple[int, int, int]
    ):
        super().__init__(self, vars())
        channels = channels.copy()
        input_shape = tuple(input_shape)
        cx, hx, wx = input_shape
        if hx != wx:
            raise NotImplementedError("TODO")
        self.channels = channels
        self.cin = cx
        self.cout = self.channels[-1]
        self.input_shape = input_shape

        channels_in = [cx] + channels[:-1]
        self.blocks = nn.Sequential()
        for cin, cout in zip(channels_in, channels):
            self.blocks.append(ConvolutionBlock((cin, hx, wx), cout))
            self.blocks.append(nn.ReLU())
        self.blocks.pop(-1)
        self.skip = SkipConv2d(cx, cout)

    def forward(self, x):
        x = self.blocks(x) + self.skip(x)
        return x


class CACNetwork(nn.Module):
    def __init__(
            self, channels: list[int], input_shape: tuple[int, int, int], ps: int, heads: int
    ):
        super().__init__(self, vars())
        channels = channels.copy()
        input_shape = tuple(input_shape)
        cx, hx, wx = input_shape
        if hx != wx:
            raise NotImplementedError("TODO")
        self.channels = channels
        self.cin = cx
        self.cout = self.channels[-1]
        self.input_shape = input_shape

        channels_in = [cx] + channels[:-1]
        self.activations = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for cin, cout in zip(channels_in, channels):
            self.blocks.append(CACBlock((cin, hx, wx), cout, ps, heads))
            self.skips.append(SkipConv2d(cin, cout))
            self.activations.append(nn.ReLU())
        self.activations[-1] = torch.nn.Identity()

    def forward(self, x):
        for block, skip, act in zip(self.blocks, self.skips, self.activations):
            x = act(block(x) + skip(x))
        return x
