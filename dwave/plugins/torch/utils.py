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
"""Conversions between tensors and dimod objects."""
from __future__ import annotations

from typing import TYPE_CHECKING, Hashable, Sequence

import torch
from dimod import BinaryQuadraticModel, SampleSet

if TYPE_CHECKING:
    from dwave.plugins.torch.graph import GraphIndex

__all__ = ["sampleset_to_tensor", "to_bqm"]


def sampleset_to_tensor(
    ordered_vars: Sequence[Hashable], sample_set: SampleSet, device: torch.device | None = None
) -> torch.Tensor:
    """Converts a ``dimod.SampleSet`` to a ``torch.Tensor`` with one row per read.

    Aggregated samples, i.e. those with ``num_occurrences > 1``, are repeated accordingly, so
    that every row of the result has equal weight and statistics of the rows are statistics of
    the reads.

    Args:
        ordered_vars (Sequence[Hashable]): The desired order of the columns.
        sample_set (dimod.SampleSet): A sample set.
        device (torch.device, optional): The device of the constructed tensor. If ``None``, the
            tensor is constructed on the current device.

    Returns:
        torch.Tensor: The samples as a ``(num_reads, len(ordered_vars))`` tensor of
        ``torch.float32``.
    """
    var_to_sample_i = {v: i for i, v in enumerate(sample_set.variables)}
    permutation = [var_to_sample_i[v] for v in ordered_vars]
    record = sample_set.record
    # The samples are repeated on the target device, where the repetition is cheap
    sample = torch.as_tensor(record.sample[:, permutation], dtype=torch.float32, device=device)
    # astype copies the field of the record array, which torch cannot view
    repeats = torch.as_tensor(record.num_occurrences.astype("int64"), device=device)
    return torch.repeat_interleave(sample, repeats, dim=0)


def _scale_and_clip(
    biases: torch.Tensor, scalar: float, bounds: tuple[float, float] | None
) -> torch.Tensor:
    """Scales ``biases`` by ``scalar`` and then clips them to ``bounds``, if given."""
    biases = scalar * biases
    return biases if bounds is None else biases.clip(*bounds)


def to_bqm(
    graph: GraphIndex,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    scalar: float = 1.0,
    linear_range: tuple[float, float] | None = None,
    quadratic_range: tuple[float, float] | None = None,
) -> BinaryQuadraticModel:
    """Converts the biases of an Ising model on a graph to a ``dimod.BinaryQuadraticModel``.

    The biases are scaled by ``scalar`` and then clipped to ``linear_range`` and
    ``quadratic_range`` (if given), which is how a Hamiltonian is prepared for a sampler that
    operates at a fixed temperature and with bounded biases, e.g. a quantum annealer. The model is
    built from the tensors directly; ``bqm.to_ising()`` gives dimod's Ising dictionaries if they
    are needed.

    Args:
        graph (GraphIndex): The graph of the model, for example a
            :class:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine` or an
            :class:`~dwave.plugins.torch.nn.Ising` layer. Its nodes label the variables and its
            edges are the interactions.
        linear (torch.Tensor): Linear biases of shape ``(graph.n_nodes,)``, in the order of
            ``graph.nodes``.
        quadratic (torch.Tensor): Quadratic biases of shape ``(graph.n_edges,)``, in the order of
            ``graph.edges``.
        scalar (float): Scaling applied to all biases prior to clipping. Defaults to 1.
        linear_range (tuple[float, float], optional): Minimum and maximum of the linear biases.
        quadratic_range (tuple[float, float], optional): Minimum and maximum of the quadratic
            biases.

    Raises:
        ValueError: If the number of biases does not match the number of nodes or edges.

    Returns:
        dimod.BinaryQuadraticModel: The (scaled and clipped) model in the ``SPIN`` vartype, with
        its variables in the order of ``graph.nodes``.
    """
    linear = linear.detach()
    quadratic = quadratic.detach()
    if tuple(linear.shape) != (graph.n_nodes,):
        raise ValueError(
            f"Expected {graph.n_nodes} linear biases (one per node), got shape "
            f"{tuple(linear.shape)}."
        )
    if tuple(quadratic.shape) != (graph.n_edges,):
        raise ValueError(
            f"Expected {graph.n_edges} quadratic biases (one per edge), got shape "
            f"{tuple(quadratic.shape)}."
        )
    linear = _scale_and_clip(linear, scalar, linear_range).cpu().numpy()
    quadratic = _scale_and_clip(quadratic, scalar, quadratic_range).cpu().numpy()
    interactions = (graph.edge_idx_i.cpu().numpy(), graph.edge_idx_j.cpu().numpy(), quadratic)
    return BinaryQuadraticModel.from_numpy_vectors(
        linear, interactions, 0.0, "SPIN", variable_order=list(graph.nodes)
    )

