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


class BinCat(nn.Module):
    def __init__(self, length, dim):
        super().__init__()
        self.strings = torch.cartesian_prod(*torch.tensor([[0, 1],]*length))
        self.indices = torch.arange(length)
        self.cats = torch.nn.Parameter(torch.randn(2**length, dim))

    def forward(self, x):
        assert x.shape[-1] == self.indices.shape[0]
        assert set(x.unique().tolist()).issubset({0, 1})
        xcomp = torch.stack([x, 1-x], dim=-1)
        xhot = xcomp[..., self.indices, self.strings].prod(-1)
        assert (xhot.sum(-1) == 1).all()
        cats = torch.einsum("b ... i k, k d -> b i d", xhot, self.cats)
        return cats
