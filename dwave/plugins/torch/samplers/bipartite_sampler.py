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

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from dwave.plugins.torch.models.boltzmann_machine import (
        GraphRestrictedBoltzmannMachine as GRBM,
    )
    from torch._prims_common import DeviceLikeType

from dwave.plugins.torch.samplers.base import TorchSampler
from dwave.plugins.torch.nn.functional import bit2spin_soft
from dwave.plugins.torch.tensor import randspin


__all__ = ["BipartiteGibbsSampler"]


class BipartiteGibbsSampler(TorchSampler):
    """A block-Gibbs sampler specialized for bipartite graph-restricted Boltzmann machines.

    This sampler exploits the bipartite structure of the underlying GRBM, in which
    nodes are partitioned into visible and hidden sets and there are no connections
    within the same set. Under this assumption, all spins in one layer are conditionally
    independent given the spins in the other layer. This allows the sampler to update
    layers simultaneously using block Gibbs updates.

    Each Gibbs step alternates between:
        1. Sampling visible spins conditioned on the hidden spins.
        2. Sampling hidden spins conditioned on the visible spins.

    The sampler maintains persistent Markov chains that are updated
    in-place whenever :meth:`sample` is called. These chains can be used both for
    unconditional sampling from the model and for conditional sampling by clamping
    a subset of spins.

    Args:
        grbm (GRBM): The Graph-Restricted Boltzmann Machine to sample from.
        num_chains (int): Number of Markov chains to run in parallel.
        schedule (Iterable[Float]): The inverse temperature schedule.
        initial_states (torch.Tensor | None): A tensor of +/-1 values of shape
            (``num_chains``, ``grbm.n_nodes``) representing the initial states of the Markov chains.
            If None, initial states will be uniformly randomized with number of chains equal to
            ``num_chains``. Defaults to None.
        seed (int | None): Random seed. Defaults to None.
    """

    def __init__(
        self,
        grbm: GRBM,
        num_chains: int,
        schedule: Iterable[float],
        initial_states: torch.Tensor | None = None,
        seed: int | None = None,
    ):
        if grbm._connected_hidden:
            raise ValueError("BipartiteGibbsSampler requires no hidden-hidden connections.")
        
        visible_nodes = set(grbm.nodes) - set(grbm.hidden_nodes)
        connected_visible = self._connected_hidden = any(
            a in visible_nodes and b in visible_nodes for a, b in grbm.edges
        )
        if connected_visible:
            raise ValueError("BipartiteGibbsSampler requires no visible-visible connections.")
        
        self._grbm = grbm
        self._num_chains = num_chains

        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)

        initial_states = self._prepare_initial_states(
            num_chains, initial_states
        )
        self._schedule = nn.Parameter(torch.tensor(list(schedule)), requires_grad=False)
        self._x = nn.Parameter(initial_states.float(), requires_grad=False)

        # call base sampler after setting parameters for correctly identifying them
        # in super methods 'properties' and 'modules'
        super().__init__()

    def to(self, device: DeviceLikeType) -> BipartiteGibbsSampler:
        """Creates a sampler copy with components moved to the target device.

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
        sampler = super().to(device=device)

        if device != "meta":
            rng = torch.Generator(device)
            rand_tensor = torch.randint(0, 2**60, (1,), generator=sampler._rng)
            rng.manual_seed(int(rand_tensor.item()))
            sampler._rng = rng

        return sampler

    def _prepare_initial_states(
        self,
        num_chains: int,
        initial_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert initial states to tensor or sample uniformly random spins as initial states.

        Args:
            num_chains (int): Number of initial states.
            initial_states (torch.Tensor | None): A tensor of shape
                (``num_chains``, ``self._grbm.n_nodes``) representing the initial states of the
                sampler's Markov chains. If None, then initial states are sampled uniformly from
                +/-1 values. Defaults to None.

        Raises:
            ValueError: If the shape of initial states does not match that of the expected
                (``num_chains``, ``self._grbm.n_nodes``) or if the provided initial states 
                have nonspin-valued entries.

        Returns:
            torch.Tensor: The initial states of the sampler's Markov chain.
        """
        if initial_states is None:
            initial_states = randspin(
                (num_chains, self._grbm.n_nodes), generator=self._rng
            )

        if initial_states.shape != (num_chains, self._grbm.n_nodes):
            raise ValueError(
                "Initial states should be of shape ``num_chains, grbm.n_nodes`` "
                f"{(num_chains, self._grbm.n_nodes)}, but got {tuple(initial_states.shape)} instead."
            )

        if not set(initial_states.unique().tolist()).issubset({-1, 1}):
            raise ValueError("Initial states contain nonspin values.")

        return initial_states

    @torch.no_grad
    def _compute_effective_field(self, block: torch.nn.ParameterList) -> torch.Tensor:
        """
        Computes the effective field for all vertices in ``block``.

        Args:
            block (nn.ParameterList): A list of integers (indices) corresponding to the vertices of
                a block. An example could be ``self._grbm.visible_idx`` or ``self._grbm.hidden_idx``.

        Returns:
            torch.Tensor: The effective fields of each vertex in ``block``.
        """

        linear = self._grbm.linear
        quadratic = self._grbm.quadratic
        
       # Edge endpoints: for each edge index k, the edge connects nodes (i[k], j[k])
        i = self._grbm.edge_idx_i
        j = self._grbm.edge_idx_j

        # For each edge (i[k],j[k]), compute its contribution to both endpoints.
        contrib_i = self._x[:, j] * quadratic 
        contrib_j = self._x[:, i] * quadratic

        # Initialize the field accumulator for every node
        field = torch.zeros_like(self._x)

        # Accumulate neighbor contributions using scatter-add.
        field.index_add_(1, i, contrib_i)  
        field.index_add_(1, j, contrib_j)
    
        # Add the linear bias term to every node
        field += linear

        return field[:, block]

    @torch.no_grad
    def _gibbs_update(
        self,
        beta: torch.Tensor,
        block: torch.nn.ParameterList,
        effective_field: torch.Tensor,
    ) -> None:
        """Performs a Gibbs update in-place.

        Args:
            beta (torch.Tensor): The (scalar) inverse temperature to sample at.
            block (torch.nn.ParameterList): A list of integers (indices) corresponding to the vertices of
                a block. An example could be ``self._grbm.visible_idx`` or ``self._grbm.hidden_idx``.
            effective_field (torch.Tensor): Effective fields of each spin corresponding to indices
                of the block.
        """
        prob = 1 / (1 + torch.exp(2 * beta * effective_field))
        spins = bit2spin_soft(prob.bernoulli(generator=self._rng))
        self._x[:, block] = spins

    @torch.no_grad
    def _step(
        self,
        beta: torch.Tensor,
        mask: torch.Tensor | None = None,
        x: torch.Tensor | None = None,
    ):
        """Performs a block-spin update in-place.

        The sampler state ``self._x`` is updated **in-place** by sequentially
        sampling visible and hidden nodes conditioned on each other. If a mask
        is provided, selected variables are re-clamped after each block update.

        Args:
            beta (torch.Tensor): The (scalar) inverse temperature to sample at.
            mask (torch.Tensor, optional): Boolean tensor of shape 
                ``(num_chains, n_nodes)`` indicating which variables are clamped. 
                Entries set to ``True`` will keep their values during sampling.
            x (torch.Tensor, optional): Tensor of shape ``(num_chains, n_nodes)``
                containing the values assigned to clamped variables. Only used
                where ``mask`` is ``True``.
        """
        effective_field = self._compute_effective_field(self._grbm.visible_idx)
        self._gibbs_update(beta, self._grbm.visible_idx, effective_field)
        # Re-clamp visible variables (if they were fixed)
        if mask is not None:
            v = self._grbm.visible_idx
            self._x[:, v] = torch.where(mask[:, v], x[:, v], self._x[:, v])

        effective_field = self._compute_effective_field(self._grbm.hidden_idx)
        self._gibbs_update(beta, self._grbm.hidden_idx, effective_field)
        # Re-clamp hidden variables (if they were fixed)
        if mask is not None:
            h = self._grbm.hidden_idx
            self._x[:, h] = torch.where(mask[:, h], x[:, h], self._x[:, h])

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate conditional sampling input.

        This function checks that the provided tensor ``x`` is a valid
        partially observed state for conditional sampling. Observed variables
        must take values in ``{-1, +1}``, while unobserved variables must be
        represented using ``NaN``.

        The function converts this representation into a boolean mask
        indicating which variables are clamped.

        For each chain the function enforces that only one
        layer of the bipartite graph may contain unclamped variables.
        In other words, either the visible nodes or the hidden nodes may
        contain ``NaN`` values, but not both.

        Args:
            x (torch.Tensor): A tensor of shape (``num_chains``, ``dim``)
                or (``num_chains``, ``n_nodes``) interpreted as a batch of
                partially observed spins. Entries marked with ``torch.nan``
                will be sampled; entries with +/-1 values will remain constant.

        Raises:
            ValueError: If ``x`` does not match the sampler state shape
            ``(num_chains, n_nodes)``, contains values other than ``±1``
            or ``NaN``, or if both visible and hidden variables are
            simultaneously unclamped within the same chain.
        """
        if x.shape != self._x.shape:
            raise ValueError(
                "x should be of shape ``num_chains, grbm.n_nodes`` "
                f"{self._x.shape}, but got {tuple(x.shape)} instead."
            )

        mask = ~torch.isnan(x)

        # Ensure values are ±1 or NaN
        if not torch.all((x[mask] == 1) | (x[mask] == -1)):
            raise ValueError("x contains values other than ±1 or NaN")

        visible_unclamped = (~mask[:, self._grbm.visible_idx]).any(dim=1)
        hidden_unclamped = (~mask[:, self._grbm.hidden_idx]).any(dim=1)

        if (visible_unclamped & hidden_unclamped).any():
            raise ValueError(
                "The input must be unclamped for visible or hidden but not both."
            )

    @torch.no_grad
    def sample(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Draw samples from the model using Gibbs sampling.
        
        If no input ``x`` is provided, he sampler performs an unconditional 
        Gibbs update of its internal chains and returns samples from the joint
        model distribution over visible and hidden variables.

        If a partially observed state ``x`` is provided, conditional sampling
        is performed: variables with specified values are clamped while the
        remaining variables are sampled.

        Args:
            x (torch.Tensor): A tensor of shape (``num_chains``, ``dim``) or (``num_chains``, ``n_nodes``)
                interpreted as a batch of partially observed spins. Entries marked with ``torch.nan`` will
                be sampled; entries with +/-1 values will remain constant. For each chain, either visible
                nodes or hidden nodes may contain ``NaN`` values, but not both.

        Returns:
            torch.Tensor: A tensor of shape (num_chains, n_nodes) of +/-1 values sampled from the model.
        """
        if x is not None:
            mask = ~torch.isnan(x)
            self._validate_input(x)

            # Initialize state respecting clamped spins
            self._x.data[:] = torch.where(mask, x, self._x)
        else:
            mask = None
        for beta in self._schedule:
            self._step(beta, mask=mask, x=x)
        return self._x.clone()