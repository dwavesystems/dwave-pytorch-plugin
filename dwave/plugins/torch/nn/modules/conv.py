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
from torch import nn


class SkipConv2d(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__(self, vars())
        self.skip = nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        return self.skip(x)


class ConvolutionBlock(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], cout: int):
        super().__init__(self, vars())
        input_shape = tuple(input_shape)
        cin, hx, wx = input_shape
        if hx != wx:
            raise NotImplementedError("TODO")

        self.input_shape = tuple(input_shape)
        self.cin = cin
        self.cout = cout

        self.block = nn.Sequential(
            nn.LayerNorm(input_shape),
            nn.Conv2d(cin, cout, 3, 1, 1),
            nn.ReLU(),
            nn.LayerNorm((cout, hx, wx)),
            nn.Conv2d(cout, cout, 3, 1, 1),
        )
        self.skip = SkipConv2d(cin, cout)

    def forward(self, x):
        return self.block(x) + self.skip(x)
