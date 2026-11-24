# Copyright 2026 D-Wave
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
"""Integer indexing of graphs and evaluation of Ising energies given biases: the base of the
models, layers and samplers."""
from __future__ import annotations

from typing import Hashable, Iterable

import torch

__all__ = ["GraphIndex"]


class GraphIndex(torch.nn.Module):
    """Integer indexing of the nodes and edges of a simple graph.

    Nodes are indexed by their position in ``nodes`` and edges by their position in ``edges``.
    Quadratic biases are given *per edge*, as tensors of shape ``(..., n_edges)`` in the order of
    :attr:`edges`. Where a dense matrix product is the efficient computation, as in :meth:`energy`
    and :meth:`effective_field`, the coupling matrix is built on the fly by
    :meth:`quadratic_matrix` and discarded afterwards, so the per-edge biases are the
    only representation that is stored, optimized and exchanged. The index tensors are
    non-persistent buffers: they move with
    :meth:`~torch.nn.Module.to` but, being derived from ``nodes`` and ``edges``, are not saved
    in :meth:`~torch.nn.Module.state_dict`, which holds parameters only. Modules defined on a
    graph, such as
    :class:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine` and
    :class:`~dwave.plugins.torch.nn.Ising`, subclass it.

    Args:
        nodes (Iterable[Hashable]): Nodes of the graph.
        edges (Iterable[tuple[Hashable, Hashable]]): Edges of the graph.

    Raises:
        TypeError: If a node or an edge endpoint is not hashable.
        ValueError: If ``nodes`` contains duplicates, an edge references an unknown node, an
            edge is a self-loop, or an edge is duplicated (in either orientation).

    The module also evaluates the Ising :meth:`energy` and :meth:`effective_field` of spins
    under biases that are given explicitly, for a single model or a batch of models on the graph.
    Subclasses that own parameters, such as the Boltzmann machine, wrap these methods with their
    own biases.

    Attributes:
        nodes (tuple[Hashable, ...]): The nodes, in index order.
        edges (tuple[tuple[Hashable, Hashable], ...]): The edges, in the orientation given.
        node_to_idx (dict[Hashable, int]): Mapping from node to index. It is derived from
            ``nodes`` and should not be modified.
        edge_to_idx (dict[tuple[Hashable, Hashable], int]): Mapping from an edge, in either
            orientation, to its index in :attr:`edges`. It is derived from ``edges`` and should
            not be modified.
        edge_idx_i (torch.Tensor): Smaller node index of each edge, shape ``(n_edges,)``.
        edge_idx_j (torch.Tensor): Larger node index of each edge, shape ``(n_edges,)``.
    """

    def __init__(
        self, nodes: Iterable[Hashable], edges: Iterable[tuple[Hashable, Hashable]]
    ) -> None:
        super().__init__()
        self.nodes = tuple(nodes)
        self.edges = tuple(tuple(edge) for edge in edges)
        try:
            self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        except TypeError:
            raise TypeError("Nodes must be hashable.") from None
        if len(self.node_to_idx) != self.n_nodes:
            raise ValueError("`nodes` contains duplicate entries.")
        try:
            # Reshaped to two columns, which the tensor of an edgeless graph would lack
            endpoints = torch.tensor(
                [[self.node_to_idx[u], self.node_to_idx[v]] for u, v in self.edges],
                dtype=torch.long,
            ).reshape(-1, 2)
        except TypeError:
            raise TypeError("Edge endpoints must be hashable.") from None
        except KeyError as err:
            # The lookup is an implementation detail: `from None` keeps the KeyError out of
            # the traceback
            raise ValueError(f"Edge endpoint {err.args[0]!r} is not a node.") from None

        loops = endpoints[:, 0] == endpoints[:, 1]
        if loops.any():
            raise ValueError(
                f"Self-loops are not allowed. Edges with self-loops: "
                f"{[self.edges[k] for k in loops.nonzero().flatten().tolist()]}"
            )
        # Both orientations of every edge are keys
        self.edge_to_idx = {
            edge: k for k, (u, v) in enumerate(self.edges) for edge in ((u, v), (v, u))
        }
        # Twice the number of edges, as both orientations are keys, unless an edge is repeated
        if len(self.edge_to_idx) != 2 * self.n_edges:
            raise ValueError("Duplicate edges are not allowed.")

        self.register_buffer("edge_idx_i", endpoints.min(1).values, persistent=False)
        self.register_buffer("edge_idx_j", endpoints.max(1).values, persistent=False)

    def extra_repr(self) -> str:
        """The numbers of nodes and edges, shown in the representation of the module."""
        return f"n_nodes={self.n_nodes}, n_edges={self.n_edges}"

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return len(self.edges)

    def degrees(self) -> torch.Tensor:
        """Degree of every node, shape ``(n_nodes,)``."""
        return torch.bincount(
            torch.cat([self.edge_idx_i, self.edge_idx_j]), minlength=self.n_nodes
        )

    # ------------------------------------------------------------------ coupling matrices --------

    def quadratic_matrix(self, quadratic: torch.Tensor) -> torch.Tensor:
        """The symmetric coupling matrix of per-edge quadratic biases, holding the bias of every
        edge at both of its positions, so that row ``k`` holds the couplings of node ``k`` to
        every other node.

        This is the single place where a dense matrix is made from the per-edge biases; callers
        build it right before a matrix product and let it go afterwards.

        Args:
            quadratic (torch.Tensor): Quadratic biases of the edges, of shape ``(..., n_edges)``
                and in the order of :attr:`edges`.

        Raises:
            ValueError: If the trailing dimension of ``quadratic`` is not the number of edges.

        Returns:
            torch.Tensor: A symmetric tensor of shape ``(..., n_nodes, n_nodes)`` with zero
            diagonal.
        """
        if quadratic.shape[-1] != self.n_edges:
            raise ValueError(
                f"Expected {self.n_edges} edge biases, got {quadratic.shape[-1]}."
            )
        matrix = quadratic.new_zeros(*quadratic.shape[:-1], self.n_nodes, self.n_nodes)
        matrix[..., self.edge_idx_i, self.edge_idx_j] = quadratic
        matrix[..., self.edge_idx_j, self.edge_idx_i] = quadratic
        return matrix

    # ------------------------------------------------------------------ Ising energies -----------

    def energy(
        self, x: torch.Tensor, linear: torch.Tensor, quadratic: torch.Tensor
    ) -> torch.Tensor:
        r"""Energies :math:`\sum_i h_i s_i + \sum_{(i, j)} J_{ij} s_i s_j` of spins under given
        biases.

        The biases define one Ising model or a batch of them. Unbatched biases, of shapes
        ``(n_nodes,)`` and ``(n_edges,)``, apply to spins of any shape ``(..., n_nodes)``. Batched
        biases, of shapes ``(*batch, n_nodes)`` and ``(*batch, n_edges)``, define one model per
        batch element and apply to spins of shape ``(*batch, M, n_nodes)``, i.e. ``M``
        configurations per model, or ``(*batch, n_nodes)``, one configuration per model. The
        coupling matrices are built on the fly with :meth:`quadratic_matrix`.

        Args:
            x (torch.Tensor): Spins.
            linear (torch.Tensor): Linear biases.
            quadratic (torch.Tensor): Quadratic biases of the edges, in the order of
                :attr:`edges`.

        Raises:
            ValueError: If ``linear`` does not hold one bias per node, ``quadratic`` does not hold
                one bias per edge, or the two are not batched alike.

        Returns:
            torch.Tensor: Energies of shape ``x.shape[:-1]``.
        """
        self._check_linear(linear, quadratic.shape[:-1])
        # x: (*batch, M, n_nodes) for batched biases, with the sample dimension inserted when
        # the spins hold one configuration per model, or (..., n_nodes) for unbatched biases;
        # matrix: (*batch, n_nodes, n_nodes), symmetric, so that the quadratic form counts every
        # edge twice and is halved
        x, squeeze = self._align_shapes(x, linear)
        matrix = self.quadratic_matrix(quadratic)
        energy = (x @ linear.unsqueeze(-1)).squeeze(-1) + 0.5 * ((x @ matrix) * x).sum(-1)
        return energy.squeeze(-1) if squeeze else energy

    def effective_field(
        self,
        x: torch.Tensor,
        *,
        linear: torch.Tensor,
        quadratic: torch.Tensor | None = None,
        idx: torch.Tensor | None = None,
        quadratic_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Effective fields :math:`h_k + \sum_{l} J_{kl} s_l` acting on nodes under given biases.

        The effective field of node :math:`k` is the derivative of the energy with respect to
        :math:`s_k`; the conditional distribution of :math:`s_k` given all other spins is
        :math:`P(s_k = \pm 1) = \sigma(\mp 2 h^{\text{eff}}_k)`. Entries of ``x`` that are
        ``torch.nan`` denote unknown spins and contribute nothing to the fields. Biases are batched
        as in :meth:`energy`.

        Args:
            x (torch.Tensor): Spins of shape ``(..., n_nodes)``; ``torch.nan`` marks unknown spins.
            linear (torch.Tensor): Linear biases.
            quadratic (torch.Tensor, optional): Quadratic biases of the edges, in the order of
                :attr:`edges`. Required unless ``quadratic_matrix`` is given.
            idx (torch.Tensor, optional): Indices of the nodes whose fields are returned. If
                ``None``, the fields of all nodes are returned. Defaults to ``None``.
            quadratic_matrix (torch.Tensor, optional): The :meth:`quadratic_matrix` of
                ``quadratic``, given in its place. A caller that evaluates the fields repeatedly
                for the same biases, such as for several blocks of nodes, should build it once
                and pass it rather than have it rebuilt on every call. Defaults to ``None``,
                i.e. it is built from ``quadratic``.

        Raises:
            ValueError: If neither or both of ``quadratic`` and ``quadratic_matrix`` are given,
                ``linear`` does not hold one bias per node, or the biases are not batched alike.

        Returns:
            torch.Tensor: Effective fields of shape ``(..., n_nodes)`` or ``(..., len(idx))``.
        """
        if quadratic_matrix is None:
            if quadratic is None:
                raise ValueError("Either `quadratic` or `quadratic_matrix` is required.")
            quadratic_matrix = self.quadratic_matrix(quadratic)
        elif quadratic is not None:
            raise ValueError("Give either `quadratic` or `quadratic_matrix`, not both.")
        self._check_linear(linear, quadratic_matrix.shape[:-2])
        x, squeeze = self._align_shapes(x, linear)
        spins = torch.nan_to_num(x, nan=0.0)
        fields = linear if linear.ndim == 1 else linear.unsqueeze(-2)
        if idx is not None:
            fields, quadratic_matrix = fields[..., idx], quadratic_matrix[..., :, idx]
        fields = fields + spins @ quadratic_matrix
        return fields.squeeze(-2) if squeeze else fields

    def _check_linear(self, linear: torch.Tensor, batch_shape: torch.Size) -> None:
        """Raise unless ``linear`` holds one bias per node for every model of a batch.

        Args:
            linear (torch.Tensor): Linear biases, expected to be of shape
                ``(*batch_shape, n_nodes)``.
            batch_shape (torch.Size): Batch shape of the quadratic biases or of the couplings,
                i.e. ``quadratic.shape[:-1]`` or ``quadratic_matrix.shape[:-2]``.

        Raises:
            ValueError: If ``linear`` does not have shape ``(*batch_shape, n_nodes)``, in
                particular when only one of the biases is batched, which would otherwise
                broadcast into results of the wrong shape.
        """
        if linear.ndim < 1 or linear.shape[-1] != self.n_nodes:
            raise ValueError(
                f"Expected {self.n_nodes} linear biases (one per node), got shape "
                f"{tuple(linear.shape)}."
            )
        if linear.shape[:-1] != batch_shape:
            raise ValueError(
                "`linear` and the quadratic biases must define the same batch of models, got "
                f"batch shapes {tuple(linear.shape[:-1])} and {tuple(batch_shape)}."
            )

    @staticmethod
    def _align_shapes(x: torch.Tensor, linear: torch.Tensor) -> tuple[torch.Tensor, bool]:
        """Insert a sample dimension into ``x`` when it holds one configuration per batched model.

        This handles the case of the docstring of :meth:`energy` where the ``M`` dimension is
        missing, i.e. ``M = 1`` with spins of shape ``(*batch, n_nodes)``.

        Returns:
            tuple[torch.Tensor, bool]: The spins with a sample dimension, and whether one was
            inserted (and should be squeezed out of the results again).
        """
        if linear.ndim > 1 and x.ndim == linear.ndim:
            return x.unsqueeze(-2), True
        return x, False
