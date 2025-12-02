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
from einops import rearrange
from torch import nn


class ReshapeTokens2Image(nn.Module):
    def __init__(self, im_shape, patch_size):
        super().__init__(self, vars())
        self.c, self.h, self.w = im_shape
        if self.h != self.w:
            raise NotImplementedError("TODO")
        self.patch_size = patch_size

        if self.h % patch_size:
            raise NotImplementedError("image shape must be divisible by patch size.")
        self.n_rows = self.h // patch_size
        self.n_cols = self.w // patch_size

    def forward(self, tokens):
        im = rearrange(tokens, "b (nrows ncols) (p1 p2 c) -> b c (nrows p1) (ncols p2)",
                       nrows=self.n_rows, ncols=self.n_cols,
                       c=self.c, p1=self.patch_size, p2=self.patch_size)
        return im


class ReshapeImage2Tokens(nn.Module):
    def __init__(self, patch_size):
        super().__init__(self, vars())
        self.patch_size = patch_size

    def forward(self, x):
        patches = rearrange(x, 'b c (nrows p1) (ncols p2) -> b (nrows ncols) (p1 p2 c)',
                            p1=self.patch_size, p2=self.patch_size)
        return patches

class PatchAttentionBlock(nn.Module):
    def __init__(self, input_shape, ps, heads):
        super().__init__(self, vars())

        cx, hx, wx = input_shape
        if hx != wx:
            raise NotImplementedError("TODO")
        if hx % ps or (wx % ps):
            raise ValueError("Height and width must be divisible by patch size")
        dim = int(cx * ps**2)
        n_rows = hx // ps
        n_cols = wx // ps
        seq_len = n_rows * n_cols
        self.input_shape = input_shape
        attn_shape = (seq_len, dim)

        self.block = nn.Sequential(
            ReshapeImage2Tokens(ps),
            nn.LayerNorm(attn_shape),
            Attention(dim=dim, heads=heads, flash=True),
            nn.ReLU(),
            nn.LayerNorm(attn_shape),
            Attention(dim=dim, heads=heads, flash=True),
            ReshapeTokens2Image(input_shape, ps),
        )
        self.skip = SkipConv2d(cx, cx)

    def forward(self, x):
        return self.block(x) + self.skip(x)


class CACBlock(nn.Module):
    def __init__(self, input_shape, cout, ps, heads):
        super().__init__(self, vars())
        cin, hx, wx = input_shape
        self.output_shape = (cout, hx, wx)
        self.input_shape = input_shape
        self.n_channel_out = cout
        self.block = nn.Sequential(
            ConvolutionBlock(input_shape, cout),
            nn.ReLU(),
            PatchAttentionBlock(self.output_shape, ps, heads),
            nn.ReLU(),
            ConvolutionBlock(self.output_shape, cout),
        )
        self.skip = SkipConv2d(cin, cout)

    def forward(self, x):
        return self.block(x) + self.skip(x)
