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

import networkx as nx
import numpy as np
import torch
from dimod import SampleSet
from hybrid.composers import AggregatedSamples
from dwave.system import FixedEmbeddingComposite

if TYPE_CHECKING:
    from dwave.system import DWaveSampler

spread = AggregatedSamples.spread


# TODO: docstrings and typhinting
def make_sampler_and_graph(qpu: DWaveSampler) -> tuple:
    """TODO"""
    G = qpu.to_networkx_graph()
    mapping = {physical: logical for physical, logical in zip(G, range(len(G)))}
    inverse = {logical: physical for physical, logical in mapping.items()}

    G = nx.relabel_nodes(G, mapping)
    sampler = FixedEmbeddingComposite(qpu, {l_: [p] for p, l_ in mapping.items()})
    return sampler, G, inverse


def sample_to_tensor(sample_set: SampleSet, device=None):
    """Converts a ``dimod.SampleSet`` to a ``torch.Tensor``.
    TODO

    Args:
        sample_set (dimod.SampleSet): TODO
        obs (torch.Tensor, optional): TODO. Defaults to None.

    Returns:
        _type_: TODO
    """
    # Need to sort first because this module assumes variables
    # labelled by integers and ordered as such
    indices = np.argsort(sample_set.variables)
    sample = sample_set.record.sample[:, indices]

    # TODO: dtype as arg
    z = torch.tensor(sample, dtype=torch.float, device=device)
    return z
