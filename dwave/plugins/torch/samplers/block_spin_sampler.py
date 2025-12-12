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

from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Literal

import torch
from torch import nn

from dwave.plugins.torch.functional import bit2spin_soft
from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.tensor import rands


class BlockSpinSampler(nn.Module):
    """A block-spin update sampler for graph-restricted Boltzmann machines.

    Note this is a sampler and not a neural network. It extends `nn.Module` for the convenience of
    managing devices of indices (stored as parameters).

    Due to the sparse definition of GRBMs, some tedious, and ugly, indexing tricks are required for
    efficiently sampling in blocks of spins. Ideally, an adjacency list can be used, however,
    adjacencies are ragged, making vectorization inapplicable.

    Block-Gibbs and Block-Metropolis obey detailed balance and are ergodic methods at finite
    temperature which, at fixed parameters, converge upon Boltzmann distributions. Block-Metropolis
    allows higher acceptance rates for proposals (faster single-step mixing), but is non-ergodic in
    the limit of zero temperature. Decorrelation from an initial condition can be slower.
    Block-Gibbs represents best practice for independent sampling.

    Args:
        grbm (GRBM): The Graph-Restricted Boltzmann Machine to sample from.
        crayon (Callable): A colouring function; a function that maps a single node of the `grbm` to
            its colour.
        num_chains (int): Number of Markov chains to run in parallel.
        proposal_acceptance_criteria (Literal["Gibbs", "Metropolis"]): The proposal acceptance
            criterion used to accept or reject states in the Markov chain. Defaults to "Gibbs".
        seed (Optional[int]): Random seed. Defaults to None.
    """

    def __init__(self, grbm: GRBM, crayon: Callable, num_chains: int,
                 proposal_acceptance_criteria: Literal["Gibbs", "Metropolis"] = "Gibbs", seed=None):
        super().__init__()
        if num_chains < 1 or not isinstance(num_chains, int):
            raise ValueError("Number of reads should be a positive integer.")
        self._proposal_acceptance_criteria = proposal_acceptance_criteria.title()
        if self._proposal_acceptance_criteria not in {"Gibbs", "Metropolis"}:
            raise ValueError(
                f'Proposal acceptance criterion should be one of "Gibbs" or "Metropolis"'
            )
        self._grbm = grbm
        self._crayon = crayon
        if not self._valid_crayon():
            raise ValueError("`crayon` is not a valid colouring of `grbm`")
        self._partition = self._get_partition()
        self._padded_adjacencies, self._padded_adjacencies_weight = self._get_adjacencies()
        self.rng = None
        if seed is not None:
            self.rng = torch.Generator()
            self.rng.manual_seed(seed)
        self.x = nn.Parameter(
            rands((num_chains, grbm.n_nodes), generator=self.rng).float(),
            requires_grad=False
        )
        self.zeros = nn.Parameter(torch.zeros((num_chains, 1)), requires_grad=False)

    def _valid_crayon(self) -> bool:
        """Determines whether `crayon` is a valid colouring of `grbm`.

        Returns:
            bool: True if the colouring is valid and False otherwise.
        """
        for u, v in self._grbm.edges:
            if self._crayon(u) == self._crayon(v):
                return False
        return True

    def _get_partition(self) -> nn.ParameterList:
        """Computes the vertex partition induced by the colouring function `crayon`.

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

    def _get_adjacencies(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Create two lists of padded adjacency lists (tensors), one for neighbouring indices and
        another for the corresponding weight indices.

        The issue begins with the adjacency lists being ragged. To address this, we pad adjacencies
        with -1 values. The exact values do not matter, as the way these adjacencies will be used is
        by padding an input state with 0s, so when accessing `-1`, the output will be masked out.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: The first output is a padded adjacency
            list of neighbouring indices where -1 indicates empty. The second output is an adjacency
            list of weight indices (in sparse indexing) used to retrieve the corresponding edge
            weight.
        """
        max_degree = 0
        if self._grbm.n_edges:
            max_degree = torch.unique(torch.cat([self._grbm.edge_idx_i, self._grbm.edge_idx_j]),
                                      return_counts=True)[1].max().item()
        padded_adjacencies = nn.Parameter(
            -torch.ones(self._grbm.n_nodes, max_degree, dtype=int), requires_grad=False
        )
        padded_adjacencies_weight = nn.Parameter(
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
            padded_adjacencies[u][:num_neighbours] = torch.tensor(neighbours)
            padded_adjacencies_weight[u][:num_neighbours] = torch.tensor(adj_weight_idxs)
        return padded_adjacencies, padded_adjacencies_weight

    @torch.no_grad
    def _compute_effective_field(self, block) -> torch.Tensor:
        """Computes the effective field for all vertices in `block`.

        Args:
            block (nn.ParameterList): A list of integers (indices) corresponding to the vertices of
                a colour.

        Returns:
            torch.Tensor: The effective fields of each vertex in `block`.
        """
        xnbr = torch.hstack([self.x, self.zeros])[:, self._padded_adjacencies[block]]
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
        delta = -2*self.x[:, block]*effective_field
        prob = (-delta*beta).exp().clip(0, 1)
        # if the delta field is negative, then flipping the spin will improve the energy
        prob[delta <= 0] = 1
        flip = -bit2spin_soft(prob.bernoulli(generator=self.rng))
        self.x[:, block] = self.x[:, block]*flip

    @torch.no_grad
    def _gibbs_update(self, beta, block, effective_field):
        """Performs a Gibbs update in-place.

        Args:
            beta (float): The inverse temperature to sample at.
            block (nn.ParameterList): A list of integers (indices) corresponding to the vertices of
                a colour.
            effective_field (torch.Tensor): Effective fields of each spin corresponding to indices
                of the block.
        """
        prob = 1/(1+torch.exp(2*effective_field*beta))
        spins = bit2spin_soft(prob.bernoulli(generator=self.rng))
        self.x[:, block] = spins

    @torch.no_grad
    def step_(self, beta: torch.Tensor) -> None:
        """Performs a block-spin update in-place.

        Args:
            beta (float): Inverse temperature to sample at.
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
    def sample(self, schedule: torch.Tensor) -> torch.Tensor:
        """Performs block-spin updates in-place as prescribed by the inverse temperature schedule.

        Args:
            schedule (torch.Tensor): The inverse temperature schedule.
        """
        for beta in schedule:
            self.step_(beta)
        return self.x
