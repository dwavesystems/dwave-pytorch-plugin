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

from dwave.plugins.torch.utils import bit2spin_soft, spin2bit_soft, straight_through_bitrounding


class StraightThroughTanh(nn.Module):
    def __init__(self):
        super().__init__(self, vars())
        self.hth = nn.Tanh()

    def forward(self, x):
        fuzzy_spins = self.hth(x)
        fuzzy_bits = spin2bit_soft(fuzzy_spins)
        bits = straight_through_bitrounding(fuzzy_bits)
        spins = bit2spin_soft(bits)
        return spins


class StraightThroughHardTanh(nn.Module):
    def __init__(self):
        super().__init__(self, vars())
        self.hth = nn.Hardtanh()

    def forward(self, x):
        fuzzy_spins = self.hth(x)
        fuzzy_bits = spin2bit_soft(fuzzy_spins)
        bits = straight_through_bitrounding(fuzzy_bits)
        spins = bit2spin_soft(bits)
        return spins


class Bit2SpinSoft(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return bit2spin_soft(x)


class Spin2BitSoft(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return spin2bit_soft(x)
