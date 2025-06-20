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

import warnings
from typing import TYPE_CHECKING, Hashable, Iterable, Literal, Optional

import torch

if TYPE_CHECKING:
    from dimod import Sampler, SampleSet

from dimod import BinaryQuadraticModel
from hybrid.composers import AggregatedSamples

from dwave.plugins.torch.utils import sampleset_to_tensor
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple

spread = AggregatedSamples.spread


__all__ = ["GraphRestrictedBoltzmannMachine"]


class GraphRestrictedBoltzmannMachine(torch.nn.Module):
    """Creates a graph-restricted Boltzmann machine.

    Args:
        nodes (Iterable[Hashable]): List of nodes.
        edges (Iterable[tuple[Hashable, Hashable]]): List of edges.
        hidden_nodes (Iterable[Hashable], optional): List of hidden nodes. Each hidden node should
            also be listed in the input ``nodes``.
        linear (dict[tuple[Hashable], float], optional): A dictionary mapping from nodes of the
            model to its corresponding linear bias.
        quadratic (dict[tuple[Hashable, Hashable], float]): A dictionary mapping from edges of the
            model to its corresponding quadratic bias.
    """

    def __init__(
        self,
        nodes: Iterable[Hashable],
        edges: Iterable[tuple[Hashable, Hashable]],
        hidden_nodes: Optional[Iterable[Hashable]] = None,
        linear: Optional[dict[Hashable, float]] = None,
        quadratic: Optional[dict[tuple[Hashable, Hashable], float]] = None,
    ) -> None:
        super().__init__()

        self._nodes = list(nodes)
        self._edges = list(map(tuple, edges))

        self._n_nodes = len(self._nodes)
        self._n_edges = len(self._edges)

        self._idx_to_node = {i: v for i, v in enumerate(self._nodes)}
        self._node_to_idx = {v: i for i, v in self._idx_to_node.items()}

        self._idx_to_edge = {i: e for i, e in enumerate(self._edges)}
        self._edge_to_idx = {e: i for i, e in self._idx_to_edge.items()}

        self._linear = torch.nn.Parameter(0.05 * (2 * torch.rand(self._n_nodes) - 1))
        self._quadratic = torch.nn.Parameter(5.0 * (2 * torch.rand(self._n_edges) - 1))

        edge_idx_i = torch.tensor([self._node_to_idx[i] for i, _ in self._edges])
        edge_idx_j = torch.tensor([self._node_to_idx[j] for _, j in self._edges])

        self.register_buffer("_edge_idx_i", edge_idx_i)
        self.register_buffer("_edge_idx_j", edge_idx_j)

        # Use initial weights if provided
        if linear is not None:
            self.set_linear(linear)
        if quadratic is not None:
            self.set_quadratic(quadratic)

        if hidden_nodes is None:
            self._hidden_nodes = []
        else:
            self._hidden_nodes = list(hidden_nodes)
        # NOTE: `_setup_hidden` must be invoked as the last step as it depends on properties
        #     defined above
        self._setup_hidden()

    def set_linear(self, linear: dict[tuple[Hashable], float]):
        """Set linear biases of the model.

        Args:
            linear (dict[tuple[Hashable], float]): A dictionary mapping from nodes of the model to
                its corresponding linear bias.
        """
        for node, bias in linear.items():
            idx = self.node_to_idx[node]
            self._linear.data[idx] = bias

    def set_quadratic(self, quadratic: dict[tuple[Hashable, Hashable], float]):
        """Set quadratic biases of the model.

        Args:
            quadratic (dict[tuple[Hashable, Hashable], float]): A dictionary mapping from edges of
                the model to its corresponding quadratic bias.
        """
        for (u, v), bias in quadratic.items():
            idx = self._edge_to_idx.get((u, v), self._edge_to_idx.get((v, u)))
            self._quadratic.data[idx] = bias

    def _setup_hidden(self):
        """Preprocess some indexes to enable vectorized computation of effective fields of hidden
        units."""
        self._connected_hidden = any(
            a in self.hidden_nodes and b in self.hidden_nodes for a, b in self.edges)
        if self._connected_hidden:
            err_message = "Current implementation does not support intrahidden-unit connections."
            raise NotImplementedError(err_message)

        visible_idx = torch.tensor([self._node_to_idx[v]
                                    for v in self._nodes if v not in self.hidden_nodes], dtype=int)
        hidden_idx = torch.tensor(
            [i for i in torch.arange(self._n_nodes) if i not in visible_idx], dtype=int)
        self.register_buffer("_visible_idx", visible_idx)
        self.register_buffer("_hidden_idx", hidden_idx)

        # If hidden units are present, we need to keep track of several sets of
        # indices in order to vectorize computations. These indices will be used in
        # the :meth:`GraphRestrictedBoltzmannMachine._compute_effective_field` and
        # details are described there.
        flat_adj = []
        bin_idx = []
        bin_pointer = -1
        quadratic_idx = torch.arange(self._n_edges)
        flat_j_idx = []
        for idx in self.hidden_idx.tolist():
            mask_i = self._edge_idx_i == idx
            mask_j = self._edge_idx_j == idx
            edges = torch.cat([self.edge_idx_j[mask_i], self._edge_idx_i[mask_j]])
            flat_j_idx.extend(quadratic_idx[mask_i + mask_j].tolist())
            bin_pointer += edges.shape[0]
            bin_idx.append(bin_pointer)
            flat_adj.extend(edges.tolist())
        # ``self.flat_adj`` is a flattened adjacency list. It is flattened because
        # it would otherwise be a ragged tensor.
        self.register_buffer("_flat_adj", torch.tensor(flat_adj, dtype=int))
        # ``self.jidx`` is used to track the corresponding edge weights of the
        # flattened adjacency.
        self.register_buffer("_flat_j_idx", torch.tensor(flat_j_idx, dtype=int))
        # Because the adjacency list has been flattened, we need to track the
        # bin indices for each hidden unit.
        self.register_buffer("_bin_idx", torch.tensor(bin_idx, dtype=int))
        # Visually, this is the data structure we want to track.
        # [0 1 4 5 | 0 | 0 | 1 3 4 | ... ]
        # The bin indices denoted by pipes |.
        # Each bin corresponds to edges of a single hidden unit.
        # For example, the sequence 0 1 4 5 corresponds to the adjacency of the
        # first hidden unit.

    @property
    def linear(self):
        """The linear biases of the model."""
        return self._linear

    @property
    def quadratic(self):
        """The quadratic biases of the model."""
        return self._quadratic

    @property
    def nodes(self):
        """List of nodes in the model. This list includes both visible and hidden nodes."""
        return self._nodes

    @property
    def hidden_nodes(self):
        """List of hidden nodes in the model."""
        return self._hidden_nodes

    @property
    def edges(self):
        """List of nodes in the model."""
        return self._edges

    @property
    def node_to_idx(self):
        """A dictionary mapping from node to index of model variables."""
        return self._node_to_idx

    @property
    def idx_to_node(self):
        """A dictionary mapping from index of model variables to nodes."""
        return self._idx_to_node

    @property
    def n_nodes(self):
        """Total number of model variables or graph nodes (including hidden units)."""
        return self._n_nodes

    @property
    def n_edges(self):
        """Total number of edges in the model or graph."""
        return self._n_edges

    @property
    def visible_idx(self):
        """A ``torch.Tensor`` of model variable indices corresponding to visible units."""
        return self._visible_idx

    @property
    def hidden_idx(self):
        """A ``torch.Tensor`` of model variable indices corresponding to hidden units."""
        return self._hidden_idx

    @property
    def edge_idx_i(self):
        """A ``torch.Tensor`` of model variable indices corresponding to one endpoints of edges.

        The other endpoint of edges is stored in :attr:`~edge_idx_j`."""
        return self._edge_idx_i

    @property
    def edge_idx_j(self):
        """A ``torch.Tensor`` of model variable indices corresponding to one endpoints of edges.

        The other endpoint of edges is stored in :attr:`~edge_idx_i`."""
        return self._edge_idx_j

    @property
    def theta(self) -> torch.Tensor:
        """Parameters of the model---linear and quadratic biases---as a one-dimensional tensor.

        The linear and quadratic biases are concatenated in the order as defined
        by the model's input ``nodes`` and ``edges``."""
        return torch.cat([self._linear, self._quadratic])

    def sample(
        self,
        sampler: Sampler,
        *,
        prefactor: float,
        linear_range: Optional[tuple[float, float]] = None,
        quadratic_range: Optional[tuple[float, float]] = None,
        device: Optional[torch.device] = None,
        sample_params: Optional[dict] = None,
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
            linear_range (tuple[float, float], optional): Linear weights are clipped to
                ``linear_range`` prior to sampling. This clipping occurs after the ``prefactor``
                scaling has been applied. When None, no clipping is applied. Defaults to None.
            quadratic_range (tuple[float, float], optional): Quadratic weights are clipped to
                ``quadratic_range`` prior to sampling. This clipping occurs after the ``prefactor``
                scaling has been applied. When None, no clipping is applied.Defaults to None.
            device (torch.device, optional): The device of the constructed tensor.
                If ``None`` and data is a tensor then the device of data is used.
                If ``None`` and data is not a tensor then the result tensor is
                constructed on the current device.
            sample_params (dict, optional): Parameters of the `sampler.sample` method.

        Returns:
            torch.Tensor: Spins sampled from the model
            (shape prescribed by ``sampler`` and ``sample_params``).
        """
        if sample_params is None:
            sample_params = dict()
        h, J = self.to_ising(prefactor, linear_range, quadratic_range)
        ss = spread(sampler.sample_ising(h, J, **sample_params))
        spins = self.sampleset_to_tensor(ss, device=device)
        return spins

    def sampleset_to_tensor(
            self, sample_set: SampleSet, device: Optional[torch.device] = None) -> torch.Tensor:
        """Converts a ``dimod.SampleSet`` to a ``torch.Tensor`` using the node order of the class.

        Args:
            sample_set (dimod.SampleSet): A sample set.
            device (torch.device, optional): The device of the constructed tensor.
                If ``None`` and data is a tensor then the device of data is used.
                If ``None`` and data is not a tensor then the result tensor is constructed
                on the current device.

        Returns:
            torch.Tensor: The sample set as a ``torch.Tensor``.
        """
        return sampleset_to_tensor(self._nodes, sample_set, device)

    def quasi_objective(
        self,
        s_observed: torch.Tensor,
        s_model: torch.Tensor,
        kind: Optional[Literal["sampling", "exact-disc"]] = None,
        *,
        prefactor: Optional[float] = None,
        linear_range: Optional[tuple[float, float]] = None,
        quadratic_range: Optional[tuple[float, float]] = None,
        sampler: Optional[Sampler] = None,
        sample_kwargs: Optional[dict] = None
    ) -> torch.Tensor:
        """A quasi-objective function with gradients equivalent to the gradients of the
        negative log likelihood.

        Args:
            s_observed (torch.Tensor): Tensor of observed spins (data) with shape
                (b1, N) where b1 denotes the batch size and N denotes the number of
                variables in the model.
            s_model (torch.Tensor): Tensor of spins drawn from the model with shape
                (b2, N) where b2 denotes the batch size and N denotse the number of
                variables in the model.
            kind (Literal["sampling", "exact-disc"]): Method for computing, or approximating,
                marginal expectations given partial observations.
                The "sampling" method samples, conditionally, for each observation.
                The "exact-disc" method computes exact margnials for when hidden units are
                disconnected, i.e., no connections between hidden units.
            prefactor (float, optional): A scaling applied to the Hamiltonian weights (linear and
                quadratic weights). When None, no scaling is applied. Defaults to None.
            linear_range (tuple[float, float], optional): Linear weights are clipped to ``linear_range`` prior
                to sampling. This clipping occurs after the ``prefactor`` scaling has been applied.
                When None, no clipping is applied. Defaults to None.
            quadratic_range (tuple[float, float], optional): Quadratic weights are clipped to ``quadratic_range``
                prior to sampling. This clipping occurs after the ``prefactor`` scaling has been
                applied. When None, no clipping is applied.Defaults to None.
            sampler (Sampler, optional): The sampler used to sample the model and is only required
                when ``kind`` is "sampling". Defaults to None.
            sample_kwargs (dict, optional): Sample kwargs for ``sampler``. Defaults to None.

        Returns:
            torch.Tensor: Scalar difference of the average energy of data and model whose gradients
            are equivalent to the gradients of the negative log likelihood.
        """
        if self.hidden_nodes:
            if kind == "exact-disc":
                if sampler or sample_kwargs:
                    warning_msg = (
                        "`sampler` and `sample_kwargs` are not used "
                        f"({sampler}, {sample_kwargs})"
                    )
                    warnings.warn(warning_msg)
                if self._connected_hidden:
                    err_msg = (
                        'The "exact-disc" method requires hidden units to be disconnected from '
                        'each other.'
                    )
                    raise ValueError(err_msg)
                obs = self._compute_expectation_disconnected(s_observed)
            elif kind == "sampling":
                obs = self._approximate_expectation_sampling(
                    s_observed, sampler, prefactor, linear_range, quadratic_range, sample_kwargs
                )
            else:
                err_msg = f'Invalid kind ({kind}). Should be one of "sampling" or "exact-disc"'
                raise ValueError(err_msg)
        else:
            obs = s_observed
            if kind is not None:
                raise ValueError(
                    f"`kind` {kind} should not be specified if the model is fully visible.")
        return (
            self.sufficient_statistics(obs).mean(0, True)
            - self.sufficient_statistics(s_model).mean(0, True)
        ) @ self.theta

    def _compute_effective_field(self, padded: torch.Tensor) -> torch.Tensor:
        """Compute effective fields of hidden units.

        Args:
            padded (torch.tensor): Tensor of shape (..., N) where N denotes the total
                number of variables in the model, i.e., number of visible and hidden
                units.

        Returns:
            torch.Tensor: Effective field of hidden units.
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
        contribution = padded[:, self._flat_adj] * self._quadratic[self._flat_j_idx]
        cumulative_contribution = contribution.cumsum(1)
        # Don't forget to add the linear fields!
        h_eff = self._linear[self.hidden_idx] + cumulative_contribution[:, self._bin_idx].diff(
            dim=1, prepend=torch.zeros(bs, device=padded.device).unsqueeze(1)
        )

        return h_eff

    def _approximate_expectation_sampling(
            self,
            obs: torch.Tensor,
            sampler: Sampler,
            prefactor: float,
            linear_range: Optional[tuple[float, float]] = None,
            quadratic_range: Optional[tuple[float, float]] = None,
            sample_kwargs: Optional[dict] = None
    ) -> torch.Tensor:
        """Approximate expectation of hidden units via sampling.

        This is a computationally expensive method as it requires performing sampling for every
        observation in ``obs``.

        Args:
            obs (torch.Tensor): The partially-observed data corresponding to visible units of
                the model. It should have shape (b, N) where b is the batch size and N is the number
                of visible units in the model.
            sampler (Sampler): The sampler used to approximate expectations.
            prefactor (float): A scaling term applied to the linear and quadratic biases prior to,
                if applicable, clipping.
            linear_range (tuple[float, float], Optional): The minimum and maximum values to clip linear
                biases with.
            quadratic_range (tuple[float, float], Optional): The minimum and maximum values to clip
                quadratic biases with.
            sample_kwargs (dict, optional): Sample kwargs for ``sampler``. Defaults to None.

        Returns:
            torch.Tensor: A tensor of shape (b, N) where N is the total number of variables in the
            model, i.e., number of hidden and visible units and number of visible units.
        """
        # Create the BQM and remove visible units
        bqm = BinaryQuadraticModel.from_ising(
            *self.to_ising(prefactor, linear_range, quadratic_range))
        bqm.remove_variables_from([self.idx_to_node[vidx] for vidx in self.visible_idx.tolist()])

        # Compute the effective fields for hidden units
        padded = self._pad(obs)
        effective_fields = self._compute_effective_field(padded)

        # Clip linear biases if a range is provided
        if linear_range is not None:
            effective_fields.clip_(*linear_range)

        res = []
        # Iterate over each observation and do conditional sampling
        for spins, fields in zip(padded.tolist(), effective_fields.tolist()):
            # Set linear biases with effective fields
            for hidx, bias in zip(self.hidden_idx.tolist(), fields):
                hnode = self.idx_to_node[hidx]
                bqm.set_linear(hnode, bias)

            # Clip quadratic biases if a range is provided
            if quadratic_range is not None:
                lb, ub = quadratic_range
                for node_u, node_v, bias in bqm.iter_quadratic():
                    if bias > ub:
                        bqm.set_quadratic(node_u, node_v, ub)
                    if bias < lb:
                        bqm.set_quadratic(node_u, node_v, lb)

            # Sample from conditional distribution
            sample_set = sampler.sample(bqm, **sample_kwargs)
            # Populate the hidden indices with the average
            avg = torch.tensor(sample_set.record.sample).float().mean(0)
            for idx_avg, node in enumerate(sample_set.variables):
                idx = self.node_to_idx[node]
                spins[idx] = avg[idx_avg]
            res.append(spins)

        return torch.tensor(res, device=obs.device)

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
        if self._connected_hidden:
            err_msg = ("`_compute_expectation_disconnected` is not applicable when edges exist "
                       "between hidden units.")
            raise ValueError(err_msg)
        m = self._pad(obs)
        h_eff = self._compute_effective_field(m)
        m[:, self.hidden_idx] = -torch.tanh(h_eff)
        return m

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
        bs = x.shape[0]
        padded = torch.nan * torch.ones((bs, self._n_nodes), device=x.device)
        padded[:, self.visible_idx] = x
        return padded

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
        """Compute the sufficient statistics of a Boltzmann machine.

        Computes and concatenates spins and interactions (per edge) of ``x``.

        Args:
            x (torch.Tensor): A tensor of shape (..., N) where N denotes the number of
                variables in the model.

        Returns:
            torch.Tensor: The sufficient statistics of ``x``.
        """
        interactions = self.interactions(x)
        return torch.cat([x, interactions], 1)

    def to_ising(self, prefactor: float, linear_range: Optional[tuple[float, float]] = None,
                 quadratic_range: Optional[tuple[float, float]] = None) -> tuple[dict, dict]:
        """Convert the model to Ising format.

        Convert the model to Ising format with scaling (``prefactor``) followed by clipping (if
        ``linear_range`` and/or ``quadratic_range`` are supplied).

        Args:
            prefactor (float): A scaling term applied to the linear and quadratic biases prior to,
                if applicable, clipping.
            linear_range (tuple[float, float], Optional): The minimum and maximum values to clip linear
                biases with.
            quadratic_range (tuple[float, float], Optional): The minimum and maximum values to clip
                quadratic biases with.

        Returns:
            tuple[dict, dict]: The linear and quadratic biases in dictionary format compatible with
            `dimod.Sampler.sample_ising`.
        """
        linear = prefactor * self._linear.detach()
        quadratic = prefactor * self._quadratic.detach()
        if linear_range is not None:
            linear = linear.clip(*linear_range)
        if quadratic_range is not None:
            quadratic = quadratic.clip(*quadratic_range)

        edge_idx_i = self.edge_idx_i.detach().cpu().tolist()
        edge_idx_j = self.edge_idx_j.detach().cpu().tolist()
        h = {self._idx_to_node[i]: b for i, b in enumerate(linear.cpu().tolist())}
        J = {
            (self._idx_to_node[a], self._idx_to_node[b]): w
            for a, b, w in zip(edge_idx_i, edge_idx_j, quadratic.cpu().tolist())
        }
        return h, J

    def estimate_beta(self, spins: torch.Tensor) -> float:
        """Estimate the maximum pseudolikelihood temperature using
        ``dwave.system.temperatures.maximum_pseudolikelihood_temperature``.

        Args:
            spins (torch.Tensor): A tensor of shape (b, N) where b is the sample size,
                and N denotes the number of variables in the model.

        Returns:
            float: The estimated inverse temperature of the model.
        """
        bqm = BinaryQuadraticModel.from_ising(*self.to_ising(1))
        beta = 1 / mple(bqm, (spins.detach().cpu().numpy(), self._nodes))[0]
        return beta
