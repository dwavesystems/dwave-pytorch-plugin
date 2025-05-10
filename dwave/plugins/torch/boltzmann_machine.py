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
# The use of the Boltzmann Machine implementations below (including the
# GraphRestrictedBoltzmannMachine) with a quantum computing system is
# protected by the intellectual property rights of D-Wave Quantum Inc.
# and its affiliates.
#
# The use of the Boltzmann Machine implementations below (including the
# GraphRestrictedBoltzmannMachine) with D-Wave's quantum computing
# system will require access to D-Wave’s LeapTM quantum cloud service and
# will be governed by the Leap Cloud Subscription Agreement available at:
# https://cloud.dwavesys.com/leap/legal/cloud_subscription_agreement/
#

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from dimod import Sampler

from dimod import BinaryQuadraticModel, SampleSet
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple
from hybrid.composers import AggregatedSamples

spread = AggregatedSamples.spread

__all__ = ["GraphRestrictedBoltzmannMachine", "GRBM"]


class GraphRestrictedBoltzmannMachine(torch.nn.Module):
    """Creates a graph-restricted Boltzmann machine.

    Args:
        nodes (list): List of nodes.
        edges (list): List of edges.
        hiddens (list, optional): List of hidden nodes. Each hidden node should also be
            listed in the input ``nodes``.
    """

    def __init__(
        self,
        nodes: list,
        edges: list,
        hiddens: list = None,
        h_range: tuple[float, float] = None,
        j_range: tuple[float, float] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n = len(nodes)
        self.idx_to_var = {i: v for i, v in enumerate(nodes)}
        self.var_to_idx = {v: i for i, v in self.idx_to_var.items()}
        self.ordered_vars = [self.idx_to_var[i] for i in range(self.n)]

        self.m = len(edges)
        edge_idx_i = torch.tensor([self.var_to_idx[i] for i, j in edges])
        edge_idx_j = torch.tensor([self.var_to_idx[j] for i, j in edges])

        self.h = torch.nn.Parameter(0.05 * (2 * torch.rand(self.n) - 1))
        self.J = torch.nn.Parameter(5.0 * (2 * torch.rand(self.m) - 1))

        self.register_buffer("edge_idx_i", edge_idx_i)
        self.register_buffer("edge_idx_j", edge_idx_j)

        infs = [-torch.inf, torch.inf]
        self.register_buffer(
            "h_range", torch.tensor(h_range if h_range is not None else infs)
        )
        self.register_buffer(
            "j_range", torch.tensor(j_range if j_range is not None else infs)
        )

        self.fully_visible = hiddens is None
        # TODO check hidden units are conditionally independent
        if not self.fully_visible:
            vidx = torch.tensor([self.var_to_idx[v] for v in nodes if v not in hiddens])
            hidx = torch.tensor([i for i in torch.arange(self.n) if i not in vidx])
            self.register_buffer("vidx", vidx)
            self.register_buffer("hidx", hidx)

            # If hidden units are present, we need to keep track of several sets of
            # indices in order to vectorize computations. These indices will be used in
            # the :meth:`GraphRestrictedBoltzmannMachine._compute_effective_field` and
            # details are described there.
            adj = list()
            bidx = list()
            bin_pointer = -1
            J_indices = torch.arange(self.m)
            jidx = list()
            for idx in self.hidx.tolist():
                mask_i = self.edge_idx_i == idx
                mask_j = self.edge_idx_j == idx
                edges = torch.cat([self.edge_idx_j[mask_i], self.edge_idx_i[mask_j]])
                jidx.extend(J_indices[mask_i + mask_j].tolist())
                bin_pointer += edges.shape[0]
                bidx.append(bin_pointer)
                adj.extend(edges.tolist())
            # ``self.flat_adj`` is a flattened adjacency list. It is flattened because
            # it would otherwise be a ragged tensor.
            self.register_buffer("flat_adj", torch.tensor(adj))
            # ``self.jidx`` is used to track the corresponding edge weights of the
            # flattened adjacency.
            self.register_buffer("jidx", torch.tensor(jidx))
            # Because the adjacency list has been flattened, we need to track the
            # bin indices for each hidden unit.
            self.register_buffer("bidx", torch.tensor(bidx))
            # Visually, this is the data structure we want to track.
            # [0 1 4 5 | 0 | 0 | 1 3 4 | ... ]
            # The bin indices denoted by pipes |.
            # Each bin corresponds to edges of a single hidden unit.
            # For example, the sequence 0 1 4 5 corresponds to the adjacency of the
            # first hidden unit.

    @property
    def theta(self) -> torch.Tensor:
        """Parameters of the model---linear and quadratic biases---as a one-dimensional
        tensor. The linear and quadratic biases are concatenated in the order as defined
        by the model's input ``nodes`` and ``edges``."""
        return torch.cat([self.h, self.J])

    def sample(
        self,
        sampler: Sampler,
        prefactor: float,
        *,
        device: torch.device = None,
        sample_params: dict = None,
    ) -> torch.Tensor:
        """Sample from the Boltzmann machine.

        This method samples and converts a sample of spins to tensors and ensures they
        are not aggregated---provided the aggregation information is retained in the
        sample set.

        Args:
            sampler (Sampler): The sampler used to sample from the model.
            prefactor (float): The prefactor for which the Hamiltonian is scaled by.
                This quantity is typically the temperature at which the sampler operates
                at. Standard CPU-based samplers such as Metropolis- or Gibbs-based
                samplers will often default to sampling at an unit temperature, thus a
                unit prefactor should be used. In the case of a quantum annealer, a
                reasonable choice of a prefactor is 1/beta where beta is the effective
                inverse temperature and can be estimated using
                :meth:`GraphRestrictedBoltzmannMachine.estimate_beta`.
            device (torch.device, optional): The device of the constructed tensor.
                If ``None`` and data is a tensor then the device of data is used.
                If ``None`` and data is not a tensor then the result tensor is
                constructed on the current device.
            sampler_params (dict): Parameters of the `sampler.sample` method.

        Returns:
            torch.Tensor: Spins sampled from the model
                (shape prescribed by ``sampler`` and ``sample_params``).
        """
        if sample_params is None:
            sample_params = dict()
        h, J = self._ising(prefactor, clip=True)
        ss = spread(sampler.sample_ising(h, J, **sample_params))
        spins = self.sample_to_tensor(ss, device=device)
        return spins

    def sample_to_tensor(
        self, sample_set: SampleSet, device: torch.device = None
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
        var_to_sample_i = {v: i for i, v in enumerate(sample_set.variables)}
        permutation = [var_to_sample_i[v] for v in self.ordered_vars]
        sample = sample_set.record.sample[:, permutation]
        return torch.tensor(sample, dtype=torch.float32, device=device)

    def objective(
        self, s_observed: torch.Tensor, s_model: torch.Tensor
    ) -> torch.Tensor:
        """An objective function with gradients equivalent to the gradients of the
        negative log likelihood.

        Args:
            s_observed (torch.Tensor): Tensor of observed spins (data) with shape
                (b1, N) where b1 denotes the batch size and N denotes the number of
                variables in the model.
            s_model (torch.Tensor): Tensor of spins drawn from the model with shape
                (b2, N) where b2 denotes the batch size and N denotse the number of
                variables in the model.

        Returns:
            torch.Tensor: Scalar difference of the average energy of data and model.
        """
        if self.fully_visible:
            obs = s_observed
        else:
            obs = self._compute_expectation_disconnected(s_observed)

        return (
            self.sufficient_statistics(obs).mean(0, True)
            - self.sufficient_statistics(s_model).mean(0, True)
        ) @ self.theta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the Hamiltonian.

        Args:
            x (torch.tensor): A tensor of shape (B, N) where B denotes batch size and
                N denotes the number of variables in the model.

        Returns:
            torch.tensor: Hamiltonians of shape (B,).
        """
        return self.sufficient_statistics(x) @ self.theta

    def interactions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute interactions prescribed by the model's edges.

        Args:
            x (torch.tensor): Tensor of shape (..., N) where N denotes the number of
                variables in the model.

        Returns:
            torch.tensor: Tensor of interaction terms of shape (..., M) where M denotes
                the number of edges in the model.
        """
        return x[..., self.edge_idx_i] * x[..., self.edge_idx_j]

    def sufficient_statistics(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the sufficient statistics of a Boltzmann machine, i.e., spins
        concatenated with interaction values (per edge) of ``x``.

        Args:
            x (torch.Tensor): A tensor of shape (..., N) where N denotes the number of
                variables in the model.

        Returns:
            torch.Tensor: The sufficient statistics of ``x``.
        """
        interactions = self.interactions(x)
        return torch.cat([x, interactions], 1)

    def _ising(self, prefactor: float, clip: bool) -> tuple[dict, dict]:
        """Convert the model to Ising format"""
        h = prefactor * self.h.detach()
        J = prefactor * self.J.detach()
        if clip:
            h = h.clip(*self.h_range)
            J = J.clip(*self.j_range)

        edge_idx_i = self.edge_idx_i.detach().cpu().tolist()
        edge_idx_j = self.edge_idx_j.detach().cpu().tolist()
        h = {self.idx_to_var[i]: b for i, b in enumerate(h.cpu().tolist())}
        J = {
            (self.idx_to_var[a], self.idx_to_var[b]): w
            for a, b, w in zip(edge_idx_i, edge_idx_j, J)
        }
        return h, J

    @property
    def ising(self):
        return self._ising(1, False)

    @property
    def bqm(self):
        bqm = BinaryQuadraticModel.from_ising(*self.ising)
        return bqm

    def _pad(self, x: torch.Tensor) -> torch.Tensor:
        """Pads the observed spins with ``torch.nan``s at ``self.hidx`` to mark them as
        hidden units.

        Args:
            x (torch.Tensor): Partially-observed spins of shape (b, N) where b is the
                batch size and N is the number of visible units in the model.

        Raises:
            ValueError: Fully-visible models should not ``_pad`` data.

        Returns:
            torch.Tensor: A (b, N) tensor of spin variables where N is the total number
                of variables, i.e., number of visible and hidden units.
        """
        if self.fully_visible:
            raise ValueError("Fully-visible models should not `_pad` data.")
        bs = x.shape[0]
        padded = torch.ones((bs, self.n), device=x.device) * torch.nan
        padded[:, self.vidx] = x
        return padded

    def estimate_beta(self, spins: torch.Tensor) -> float:
        """Estimate the maximum pseudolikelihood temperature using
        ``dwave.system.temperatures.maximum_pseudolikelihood_temperature``.

        Args:
            spins (torch.Tensor): A tensor of shape (b, N) where b is the sample size,
                and N denotes the number of variables in the model.

        Returns:
            float: The estimated inverse temperature of the model.
        """
        bqm = self.bqm
        beta = 1 / mple(bqm, (spins.detach().cpu().numpy(), self.ordered_vars))[0]
        return beta

    def _compute_effective_field(self, padded: torch.Tensor) -> torch.Tensor:
        """Compute the effective field of disconnected hidden units.

        Args:
            padded (torch.tensor): Tensor of shape (..., N) where N denotes the total
                number of variables in the model, i.e., number of visible and hidden
                units.

        Returns:
            torch.Tensor: effective field of hidden units
        """
        bs = padded.shape[0]

        # Ideally, we can apply a scatter-add here for fast vectorized computation.
        # An optimized implementation of scatter-add is available in the pip package
        # ``torch-scatter`` but is unsupported on MacOS as of 2025-05.
        # The following is a work-around.

        # Extract the spins prescribed by a flattened adjacency list and multiply them
        # by the corresponding edges. Transforming this contribution vector by a
        # cumulative sum yields cumulative contributions to effective fields.
        # Differencing removes the extra gobbledygook.
        contribution = padded[:, self.flat_adj] * self.J[self.jidx]
        cumulative_contribution = contribution.cumsum(1)
        # Don't forget to add the linear fields!
        h_eff = self.h[self.hidx] + cumulative_contribution[:, self.bidx].diff(
            dim=1, prepend=torch.zeros(bs).unsqueeze(1)
        )

        return h_eff

    def _compute_expectation_disconnected(self, obs: torch.Tensor) -> torch.Tensor:
        """Compute and return the conditional expectation of spins including observed
        spins.

        Args:
            obs (torch.Tensor): A tensor of spins with shape (b, N) where b is the
                sample size and N is the number of visible units in the model.
        Returns:
            torch.Tensor: A (b, N)-shaped tensor of expected spins conditioned on
                ``obs`` where b is the sample size and N is the total number of
                variables in the model, i.e., number of hidden and visible units.
        """
        m = self._pad(obs)
        h_eff = self._compute_effective_field(m)
        m[:, self.hidx] = -torch.tanh(h_eff)
        return m


GRBM = GraphRestrictedBoltzmannMachine
