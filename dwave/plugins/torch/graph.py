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
    and :meth:`effective_field` given a :meth:`quadratic_matrix`, the coupling matrix is built on
    the fly and discarded afterwards, so the per-edge biases are the only representation that is
    stored, optimized and exchanged; given per-edge biases, :meth:`effective_field` scatters over
    the edges instead, which is cheaper on large sparse graphs. The index tensors are
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

    The module also evaluates the Ising :meth:`energy`, :meth:`effective_field` and
    :meth:`excitation_energies` of spins under biases that are given explicitly, for a single
    model or a batch of models on the graph, and checks such biases and spins against the graph
    with :meth:`validate_biases` and :meth:`validate_spins`. Subclasses that own parameters, such
    as the Boltzmann machine, wrap these methods with their own biases.

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
                :attr:`edges`, evaluated by two scatters over the edges, which touch every
                (configuration, edge) pair once. Required unless ``quadratic_matrix`` is given.
            idx (torch.Tensor, optional): Indices of the nodes whose fields are returned. If
                ``None``, the fields of all nodes are returned. Defaults to ``None``.
            quadratic_matrix (torch.Tensor, optional): The :meth:`quadratic_matrix` of the
                biases, given in place of ``quadratic`` and evaluated by the dense product, which
                costs ``n_nodes ** 2`` multiply-adds per configuration and is the cheaper path on
                all but large sparse graphs. A caller that evaluates the fields repeatedly for the
                same biases, such as for several blocks of nodes, should build it once and pass
                it. Defaults to ``None``.

        Raises:
            ValueError: If neither or both of ``quadratic`` and ``quadratic_matrix`` are given,
                ``quadratic`` does not hold one bias per edge, ``linear`` does not hold one bias
                per node, or the biases are not batched alike.

        Returns:
            torch.Tensor: Effective fields of shape ``(..., n_nodes)`` or ``(..., len(idx))``.
        """
        if quadratic_matrix is not None:
            if quadratic is not None:
                raise ValueError("Give either `quadratic` or `quadratic_matrix`, not both.")
            self._check_linear(linear, quadratic_matrix.shape[:-2])
        else:
            if quadratic is None:
                raise ValueError("Either `quadratic` or `quadratic_matrix` is required.")
            self.validate_biases(linear, quadratic)

        x, squeeze = self._align_shapes(x, linear)
        spins = torch.nan_to_num(x, nan=0.0)  # unknown spins contribute nothing
        fields = _per_model(linear) if idx is None else _per_model(linear)[..., idx]
        if quadratic_matrix is not None:
            fields = fields + self._dense_interaction(spins, quadratic_matrix, idx)
        else:
            fields = fields + self._scattered_interaction(spins, quadratic, idx)
        return fields.squeeze(-2) if squeeze else fields

    @staticmethod
    def _dense_interaction(
        spins: torch.Tensor, quadratic_matrix: torch.Tensor, idx: torch.Tensor | None
    ) -> torch.Tensor:
        """The interaction part of the fields of the nodes ``idx`` (all nodes if ``None``) as the
        product of the spins with the columns of the coupling matrix selected beforehand."""
        return spins @ (quadratic_matrix if idx is None else quadratic_matrix[..., :, idx])

    def _scattered_interaction(
        self, spins: torch.Tensor, quadratic: torch.Tensor, idx: torch.Tensor | None
    ) -> torch.Tensor:
        """The interaction part of the fields by two scatters over the edges: every edge adds its
        bias times the spin at one endpoint to the field at the other. The fields of all nodes
        are computed and those of ``idx`` (all nodes if ``None``) returned."""
        edge_biases = _per_model(quadratic)
        fields = torch.zeros_like(spins)
        for i, j in ((self.edge_idx_i, self.edge_idx_j), (self.edge_idx_j, self.edge_idx_i)):
            fields = fields.index_add(-1, i, edge_biases * spins[..., j])
        return fields if idx is None else fields[..., idx]

    def excitation_energies(
        self,
        x: torch.Tensor,
        *,
        linear: torch.Tensor,
        quadratic: torch.Tensor | None = None,
        idx: torch.Tensor | None = None,
        quadratic_matrix: torch.Tensor | None = None,
    ) -> torch.Tensor:
        r"""Energies released by flipping single spins,
        :math:`\epsilon_k(s) = E(s) - E(s^{(k)}) = 2 s_k h^{\text{eff}}_k(s)`, :math:`s^{(k)}`
        being :math:`s` with spin :math:`k` flipped: positive where the flip would lower the
        energy, i.e. the spin is excited against its effective field, and negative where the spin
        is relaxed.

        The arguments are those of :meth:`effective_field` and the energies have the shape of its
        result; the energy of an unknown spin (``torch.nan``) is ``torch.nan``.
        """
        fields = self.effective_field(
            x,
            linear=linear,
            quadratic=quadratic,
            idx=idx,
            quadratic_matrix=quadratic_matrix,
        )
        return 2.0 * (x if idx is None else x[..., idx]) * fields

    # ------------------------------------------------------------------ validation ---------------

    def validate_biases(self, linear: torch.Tensor, quadratic: torch.Tensor) -> torch.Size:
        """Check that biases define one model or a batch of models on the graph.

        Args:
            linear (torch.Tensor): Linear biases of shape ``(*batch, n_nodes)``.
            quadratic (torch.Tensor): Quadratic biases of the edges of shape ``(*batch, n_edges)``.

        Raises:
            ValueError: If ``linear`` does not hold one bias per node, ``quadratic`` does not hold
                one bias per edge, or the two are not batched alike.

        Returns:
            torch.Size: The batch shape ``(*batch)``, empty for a single model.
        """
        if quadratic.ndim < 1 or quadratic.shape[-1] != self.n_edges:
            raise ValueError(
                f"Expected {self.n_edges} quadratic biases (one per edge), got shape "
                f"{tuple(quadratic.shape)}."
            )
        self._check_linear(linear, quadratic.shape[:-1])
        return linear.shape[:-1]

    def validate_spins(
        self, spins: torch.Tensor, batch_shape: torch.Size | None = None, *, values: bool = True
    ) -> None:
        """Check that spins are configurations of the graph, batched like given biases.

        Args:
            spins (torch.Tensor): Spins expected to have shape ``(..., M, n_nodes)``, ``M``
                configurations with one column per node.
            batch_shape (torch.Size, optional): The batch shape of the biases the spins go with,
                as returned by :meth:`validate_biases`; when it is not empty, the spins must have
                shape ``(*batch_shape, M, n_nodes)``. Defaults to ``None``, any leading shape.
            values (bool): Whether to check that every entry is ``±1``. Defaults to ``True``.

        Raises:
            ValueError: If ``spins`` do not have the expected shape, hold no configuration, or,
                with ``values``, hold a value other than ``±1``.
        """
        if spins.ndim < 2 or spins.shape[-1] != self.n_nodes:
            raise ValueError(
                f"spins must have shape (..., M, {self.n_nodes}), got {tuple(spins.shape)}."
            )
        if batch_shape and spins.shape[:-2] != batch_shape:
            raise ValueError(
                f"spins must have shape {(*batch_shape, 'M', self.n_nodes)}, batched like the "
                f"biases, got {tuple(spins.shape)}."
            )
        if spins.shape[-2] == 0:
            raise ValueError("At least one sample per model is required, got none.")
        if values and not bool((spins.abs() == 1).all()):
            raise ValueError("spins must hold ±1 values only.")

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


def _per_model(biases: torch.Tensor) -> torch.Tensor:
    """Biases of one model as they are; biases of a batch of models with a sample dimension
    inserted after the batch dimensions, so that they broadcast against spins of shape
    ``(*batch, M, n)``."""
    return biases if biases.ndim == 1 else biases.unsqueeze(-2)
