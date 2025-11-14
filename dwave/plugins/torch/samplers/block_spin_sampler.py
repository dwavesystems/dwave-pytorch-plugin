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

from collections import defaultdict
from collections.abc import Iterable
from typing import TYPE_CHECKING, Callable, Hashable, Literal

import torch
from torch import nn

if TYPE_CHECKING:
    from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
    from torch._prims_common import DeviceLikeType

from dwave.plugins.torch.nn.functional import bit2spin_soft
from dwave.plugins.torch.tensor import randspin

__all__ = ["BlockSampler"]


class BlockSampler:
    """A block-spin update sampler for graph-restricted Boltzmann machines.

    Due to the sparse definition of GRBMs, some tedious indexing tricks are required to
    efficiently sample in blocks of spins. Ideally, an adjacency list can be used, however,
    adjacencies are ragged, making vectorization inapplicable.

    Block-Gibbs and Block-Metropolis obey detailed balance and are ergodic methods at finite nonzero
    temperature which, at fixed parameters, converge upon Boltzmann distributions. Block-Metropolis
    allows higher acceptance rates for proposals (faster single-step mixing), but is non-ergodic in
    the limit of zero or infinite temperature. Decorrelation from an initial condition can be slower.
    Block-Gibbs represents best practice for independent sampling.

    Args:
        grbm (GRBM): The Graph-Restricted Boltzmann Machine to sample from.
        crayon (Callable[Hashable, Hashable]): A colouring function that maps a single
            node of the ``grbm`` to its colour.
        num_chains (int): Number of Markov chains to run in parallel.
        initial_states (torch.Tensor | None): A tensor of +/-1 values of shape
            (``num_chains``, ``grbm.n_nodes``) representing the initial states of the Markov chains.
            If None, initial states will be uniformly randomized with number of chains equal to
            ``num_chains``. Defaults to None.
        schedule (Iterable[Float]): The inverse temperature schedule.
        proposal_acceptance_criteria (Literal["Gibbs", "Metropolis"]): The proposal acceptance
            criterion used to accept or reject states in the Markov chain. Defaults to "Gibbs".
        seed (int | None): Random seed. Defaults to None.

    Raises:
        InvalidProposalAcceptanceCriteriaError: If the proposal acceptance criteria is not one of
            "Gibbs" or "Metropolis".
    """

    def __init__(self, grbm: GRBM, crayon: Callable[[Hashable], Hashable], num_chains: int,
                 schedule: Iterable[float],
                 proposal_acceptance_criteria: Literal["Gibbs", "Metropolis"] = "Gibbs",
                 initial_states: torch.Tensor | None = None,
                 seed: int | None = None):
        super().__init__()

        if num_chains < 1:
            raise ValueError("Number of reads should be a positive integer.")

        self._proposal_acceptance_criteria = proposal_acceptance_criteria.title()
        if self._proposal_acceptance_criteria not in {"Gibbs", "Metropolis"}:
            raise ValueError(
                'Proposal acceptance criterion should be one of "Gibbs" or "Metropolis"'
            )

        self._grbm: GRBM = grbm
        self._crayon: Callable[[Hashable], Hashable] = crayon
        if not self._valid_crayon():
            raise ValueError(
                "crayon is not a valid colouring of grbm. "
                + "At least one edge has vertices of the same colour."
            )

        self._partition = self._get_partition()
        self._padded_adjacencies, self._padded_adjacencies_weight = self._get_adjacencies()

        self._rng = torch.Generator()
        if seed is not None:
            self._rng = self._rng.manual_seed(seed)

        initial_states = self._prepare_initial_states(num_chains, initial_states, self._rng)
        self._schedule = nn.Parameter(torch.tensor(list(schedule)), requires_grad=False)
        self._x = nn.Parameter(initial_states.float(), requires_grad=False)
        self._zeros = nn.Parameter(torch.zeros((num_chains, 1)), requires_grad=False)

    def to(self, device: DeviceLikeType) -> BlockSampler:
        """Moves sampler components to the target device.

        If the device is "meta", then the random number generator (RNG)
        will not be modified at all. For all other devices, all attributes used for performing
        block-spin updates will be moved to the target device. Importantly, the RNG's device is
        relayed by the following procedure:
        1. Draw a random integer between 0 (inclusive) and 2**60 (exclusive) with the current
           generator as a new seed ``s``.
        2. Create a new generator on the target device.
        3. Set the new generator's seed as ``s``.

        Developer-note: Not sure the above constitutes a good practice, but I not aware of any
        obvious solution for moving generators across devices.

        Args:
            device (DeviceLikeType): The target device.
        """
        self._x = self._x.to(device)
        self._zeros = self._zeros.to(device)
        self._schedule = self._schedule.to(device)
        self._partition = self._partition.to(device)
        self._padded_adjacencies = self._padded_adjacencies.to(device)
        self._padded_adjacencies_weight = self._padded_adjacencies_weight.to(device)
        if device != "meta":
            rng = torch.Generator(device)
            rng.manual_seed(torch.randint(0, 2**60, (1,), generator=self._rng).item())
            self._rng = rng
        return self

    def _prepare_initial_states(
            self, num_chains: int, initial_states: torch.Tensor | None = None,
            generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Convert initial states to tensor or sample uniformly random spins as initial states.

        Args:
            num_chains (int): Number of initial states.
            initial_states (torch.Tensor | None): A tensor of shape
                (``num_chains``, ``self._grbm.n_nodes``) representing the initial states of the
                sampler's Markov chains. If None, then initial states are sampled uniformly from
                +/-1 values. Defaults to None.
            generator (torch.Generator | None): A random number generator.

        Raises:
            ShapeMismatchError: If the shape of initial states do not match that of the expected
                (``num_chains``, ``self._grbm.n_nodes``).
            NonSpinError: If the provided initial states have nonspin-valued entries.

        Returns:
            torch.Tensor: The initial states of the sampler's Markov chain.
        """
        if initial_states is None:
            initial_states = randspin((num_chains, self._grbm.n_nodes), generator=generator)

        if initial_states.shape != (num_chains, self._grbm.n_nodes):
            raise ValueError(
                "Initial states should be of shape ``num_chains, grbm.n_nodes`` "
                f"{(num_chains, self._grbm.n_nodes)}, but got {tuple(initial_states.shape)} instead."
            )

        if not set(initial_states.unique().tolist()).issubset({-1, 1}):
            raise ValueError("Initial states contain nonspin values.")

        return initial_states

    def _valid_crayon(self) -> bool:
        """Determines whether ``crayon`` is a valid colouring of the graph-restricted Boltzmann machine.

        Returns:
            bool: True if the colouring is valid and False otherwise.
        """
        for u, v in self._grbm.edges:
            if self._crayon(u) == self._crayon(v):
                return False
        return True

    def _get_partition(self) -> nn.ParameterList:
        """Computes the vertex partition induced by the colouring function.

        Returns:
            nn.ParameterList: The partition induced by the colouring.
        """
        partition = defaultdict(list)
        for node in self._grbm.nodes:
            idx = self._grbm.node_to_idx[node]
            c = self._crayon(node)
            partition[c].append(idx)
        partition = nn.ParameterList([
            nn.Parameter(torch.tensor(partition[k], requires_grad=False), requires_grad=False)
            for k in sorted(partition)
        ])
        return partition

    def _get_adjacencies(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Create two adjacency matrices, one for neighbouring indices and another for the
        corresponding edge weights' indices.

        The issue begins with the adjacency lists being ragged. To address this, we pad adjacencies
        with ``-1`` values. The exact values do not matter, as the way these adjacencies will be used
        is by padding an input state with 0s, so when accessing ``-1``, the output will be masked out.

        For example, consider the returned adjacency matrices ``padded_adjacencies`` and
        ``padded_adjacencies_weight``.

        In the first adjacency matrix, ``padded_adjacencies[0]`` is a
        ``torch.Tensor`` consisting of indices of neighbouring vertices of vertex ``0``. Values of
        ``-1`` in this tensor indicates no neighbour.

        In the second adjacency matrix, ``padded_adjacencies_weight[0]`` is a ``torch.Tensor``
        consisting of indices of edge weight indices corresponding to edges of vertex ``0``.
        Similarly, ``-1`` values in this tensor indicates no neighbour.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: The first output is a padded adjacency
            matrix, the second output is an adjacency matrix of edge weight indices.
        """
        max_degree = 0
        if self._grbm.n_edges:
            max_degree = torch.unique(torch.cat([self._grbm.edge_idx_i, self._grbm.edge_idx_j]),
                                      return_counts=True)[1].max().item()
        adjacency = nn.Parameter(
            -torch.ones(self._grbm.n_nodes, max_degree, dtype=int), requires_grad=False
        )
        adjacency_weight = nn.Parameter(
            -torch.ones(self._grbm.n_nodes, max_degree, dtype=int), requires_grad=False
        )

        adjacency_dict = defaultdict(list)
        edge_to_idx = dict()
        for idx, (u, v) in enumerate(
            zip(self._grbm.edge_idx_i.tolist(),
                self._grbm.edge_idx_j.tolist())):
            adjacency_dict[v].append(u)
            adjacency_dict[u].append(v)
            edge_to_idx[u, v] = idx
            edge_to_idx[v, u] = idx
        for u in self._grbm.idx_to_node:
            neighbours = adjacency_dict[u]
            adj_weight_idxs = [edge_to_idx[u, v] for v in neighbours]
            num_neighbours = len(neighbours)
            adjacency[u][:num_neighbours] = torch.tensor(neighbours)
            adjacency_weight[u][:num_neighbours] = torch.tensor(adj_weight_idxs)
        return adjacency, adjacency_weight

    @torch.no_grad
    def _compute_effective_field(self, block) -> torch.Tensor:
        """Computes the effective field for all vertices in ``block``.

        Args:
            block (nn.ParameterList): A list of integers (indices) corresponding to the vertices of
                a colour.

        Returns:
            torch.Tensor: The effective fields of each vertex in ``block``.
        """
        xnbr = torch.hstack([self._x, self._zeros])[:, self._padded_adjacencies[block]]
        h = self._grbm.linear[block]
        J = self._grbm.quadratic[self._padded_adjacencies_weight[block]]
        return (xnbr * J.unsqueeze(0)).sum(2) + h

    @torch.no_grad
    def _metropolis_update(self, beta: float, block: nn.ParameterList,
                           effective_field: torch.Tensor) -> None:
        """Performs a Metropolis update in-place.

        Args:
            beta (float): The inverse temperature to sample at.
            block (nn.ParameterList): A list of integers (indices) corresponding to the vertices of
                a colour.
            effective_field (torch.Tensor): Effective fields of each spin corresponding to indices
                of the block.
        """
        delta = -2 * self._x[:, block] * effective_field
        prob = (-delta * beta).exp().clip(0, 1)

        # if the delta field is negative, then flipping the spin will improve the energy
        prob[delta <= 0] = 1
        flip = -bit2spin_soft(prob.bernoulli(generator=self._rng))
        self._x[:, block] = flip * self._x[:, block]

    @torch.no_grad
    def _gibbs_update(self, beta: torch.Tensor, block: torch.nn.ParameterList, effective_field: torch.Tensor) -> None:
        """Performs a Gibbs update in-place.

        Args:
            beta (torch.Tensor): The (scalar) inverse temperature to sample at.
            block (nn.ParameterList): A list of integers (indices) corresponding to the vertices of
                a colour.
            effective_field (torch.Tensor): Effective fields of each spin corresponding to indices
                of the block.
        """
        prob = 1 / (1 + torch.exp(2 * beta * effective_field))
        spins = bit2spin_soft(prob.bernoulli(generator=self._rng))
        self._x[:, block] = spins

    @torch.no_grad
    def _step(self, beta: torch.Tensor) -> None:
        """Performs a block-spin update in-place.

        Args:
            beta (torch.Tensor): Inverse temperature to sample at.
        """
        for block in self._partition:
            effective_field = self._compute_effective_field(block)
            if self._proposal_acceptance_criteria == "Metropolis":
                self._metropolis_update(beta, block, effective_field)
            elif self._proposal_acceptance_criteria == "Gibbs":
                self._gibbs_update(beta, block, effective_field)
            else:
                # NOTE: This line should never be reached because acceptance proposal criterion
                # should've been checked on instantiation
                raise ValueError(f"Invalid proposal acceptance criterion.")

    @torch.no_grad
    def sample(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Performs block updates.

        Args:
            x (torch.Tensor): A tensor of shape (``batch_size``, ``dim``) or (``batch_size``, ``n_nodes``)
                interpreted as a batch of partially-observed spins. Entries marked with ``torch.nan`` will
                be sampled; entries with +/-1 values will remain constant.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, dim) of +/-1 values sampled from the model.
        """
        if x is not None:
            raise NotImplementedError("Support for conditional sampling has not been implemented.")
        for beta in self._schedule:
            self._step(beta)
        return self._x
