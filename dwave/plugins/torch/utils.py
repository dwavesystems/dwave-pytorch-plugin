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


def make_sampler_and_graph(
    qpu: DWaveSampler,
) -> tuple[FixedEmbeddingComposite, nx.Graph, dict]:
    """A helper function that maps a QPU's variables to contiguous nonnegative integers.

    Mapping variables to contiguous nonnegative integers is a requirement motivated by
    convenience and efficiency for applying tensor operations downstream.

    Args:
        qpu (DWaveSampler): The ``dwave.system.DWaveSampler`` QPU for which an embedded
            composite sampler with linear variables is created.

    Returns:
        tuple[FixedEmbeddingComposite, nx.Graph]: The sampler with linear
            variables and its corresponding graph.
    """
    G = qpu.to_networkx_graph()
    mapping = {physical: logical for physical, logical in zip(G, range(len(G)))}

    G = nx.relabel_nodes(G, mapping)
    sampler = FixedEmbeddingComposite(qpu, {l_: [p] for p, l_ in mapping.items()})
    return sampler, G


def sample_to_tensor(
    sample_set: SampleSet, device: torch.device = None
) -> torch.Tensor:
    """Converts a ``dimod.SampleSet`` to a ``torch.Tensor``.

    Args:
        sample_set (dimod.SampleSet): A sample set.
        device (torch.device, optional): The device of the constructed tensor.
            If ``None`` and data is a tensor then the device of data is used.
            If ``None`` and data is not a tensor then the result tensor is constructed
            on the current device.

    Returns:
        torch.Tensor: The sample set as a ``torch.Tensor``.
    """
    # Need to sort first because this module assumes variables are labelled by integers
    # and ordered as such
    indices = np.argsort(sample_set.variables)
    sample = sample_set.record.sample[:, indices]

    return torch.tensor(sample, dtype=torch.float32, device=device)
