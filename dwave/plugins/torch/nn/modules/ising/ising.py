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
#
# The use of the Ising layer implementations below with a quantum computing system is
# protected by the intellectual property rights of D-Wave Quantum Inc. and its affiliates.
#
# The use of the Ising layer implementations below with D-Wave's quantum computing
# system will require access to D-Wave’s LeapTM quantum cloud service and
# will be governed by the Leap Cloud Subscription Agreement available at:
# https://cloud.dwavesys.com/leap/legal/cloud_subscription_agreement/

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dimod
    from dwave.plugins.torch.nn.modules.ising.spin_statistic import SpinStatistic

import torch
from dimod import BinaryQuadraticModel
from torch import nn

from dwave.plugins.torch.nn.modules.ising.spin_statistic import IdentityStatistic
from dwave.plugins.torch.utils import sampleset_to_tensor
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple

__all__ = ["Ising", "IsingExpectation"]


class IsingExpectation(torch.autograd.Function):
    """Computes the sample average statistic of the Ising model.

    This is a helper function to facilitate backpropagation through an Ising layer. Details of the
    Ising layer are in :class:`.Ising`.

    The gradient of the expected output statistic (``statistic`` or :math:`g` mapping from
    :math:`\{\pm 1\} ^N` to real numbers), is the negative covariance between the sufficient statistic
    and :math:`g`. See
    `Graphical Models, Exponential Families, and Variational Inference <https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf>`_
    for a more rigorous treatment.

    .. math::

        \nabla_\theta \mathbb{E} \left[ g(S)\right]  & = \sum_{s\in\{\pm 1\}^d} \nabla_\theta g(s) \frac{\exp \Big \{ -\langle T(s), \theta \rangle \Big \}}{Z(\theta)}

         & = \sum_{s\in\{\pm 1\}^d} \frac{g(s)}{Z(\theta)^2} \Bigg [ -T(s)\exp \Big \{ -\langle T(s), \theta \rangle  \Big\} Z(\theta) + \exp \Big \{ -\langle T(s), \theta\rangle  \Big\} \mathbb{E}\left[ T(S) \right]Z(\theta) \Bigg]

         & = \sum_{s\in\{\pm 1\}^d} \frac{g(s)}{Z(\theta)} \Bigg[ -T(s)\exp \Big \{ -\langle T(s),\theta\rangle  \Big\} + \exp \Big \{ -\langle T(s), \theta\rangle  \Big\} \mathbb{E}\left[ T(S) \right]\Bigg]

         & = -\mathbb{E}\left[g(S)T(S)\right] + \mathbb{E}\left[g(S)\right] \mathbb{E}\left[ T(S) \right]

         & = -\text{Covariance}\left[g(S), T(S)\right].
    """

    @staticmethod
    def forward(
        ctx,
        spins: torch.Tensor,
        interactions: torch.Tensor,
        stats_out: torch.Tensor,
        linear: torch.Tensor,
        quadratic: torch.Tensor
    ) -> torch.Tensor:
        """Computes the sample average of ``stats_out``.

        Args:
            spins: Spins of shape (B, N, D1) where B is a batch size, N is the sample size of spin
                vectors per observation in batch, and D1 is the number of nodes in the Ising model.
            interactions: Spin-spin interaction terms of shape (B, N, D2) where B is a batch size,
                N is the sample size of spin vectors per observation in batch, and D2 is the number
                of edges in the Ising model.
            stats_out: Output statistic of spins of (B, N, D3) where B is a batch size, N is the
                sample size of spin vectors per observation in batch, and D3 is the dimension of
                output statistics.
            linear: Linear biases of the Ising model with shape (B, D1) where B is batch size and D1
                is the number of nodes in the Ising model.
            quadratic: Quadratic biases of the Ising model with shape (B, D2) where B is batch size
                and D2 is the number of edges in the Ising model.

        Raises:
            ValueError: If ``stats_out.ndim`` != 3.

        Returns:
            torch.Tensor: Sample average of statistics with shape (B, D3).
        """
        if stats_out.ndim != 3:
            raise ValueError(f"stats.ndim should be 3. stats.ndim is {stats_out.ndim}")

        avg_stats = stats_out.mean(-2)
        ctx.save_for_backward(stats_out, spins, interactions)
        return avg_stats

    @staticmethod
    def backward(
        ctx,
        grad_dloss_dinput_i: torch.Tensor
    ) -> tuple[None, None, None, torch.Tensor, torch.Tensor]:
        """Backpropagation of gradients approximated via sample covariance.

        Args:
            grad_dloss_dinput_i: Gradients of the loss with respect to inputs y with
                shape (B, D3) where B is batch size and D3 is dimension of output statistic.

        Returns:
            tuple[None, None, None, torch.Tensor, torch.Tensor]: Gradients with respect to the
            linear and quadratic weights with shapes (B, D1) and (B, D2) respectively where B is
            batch size, D1 is number of nodes in model, and D2 is number of edges in model.
        """
        stats, spins, interactions = ctx.saved_tensors
        # grad_dloss_dyi = dloss / dyi
        # jacobian = -covariance = dyi / dthetaj, each row i corresponds to grad_thetaj <y(s)>_i
        stats_centred = stats - stats.mean(1, True)
        spins_centred = spins - spins.mean(1, True)
        intxn_centred = interactions - interactions.mean(1, True)
        sample_size = stats.shape[-2]
        dl_dlin = -torch.einsum("bmy, bmx, by -> bx",
                                stats_centred,
                                spins_centred,
                                grad_dloss_dinput_i) / (sample_size - 1)
        dl_dqdr = -torch.einsum("bmy, bmx, by -> bx",
                                stats_centred,
                                intxn_centred,
                                grad_dloss_dinput_i) / (sample_size - 1)
        return None, None, None, dl_dlin, dl_dqdr


class Ising(nn.Module):
    """An Ising layer in which inputs are interpreted as Hamiltonian parameters and outputs are
    expected statistics of the system.

    An Ising model is defined by a graph :math:`G = (V, E)` or, equivalently, a set of nodes and
    edges. In implementation, the model is defined by an ordered list of nodes and edges.
    Inputs ``x`` and ``y`` are interpreted as the linear and quadratic biases associated with
    :math:`V` and :math:`E` respectively in the given order. The output of the model is the expected
    output statistic (``statistic`` or :math:`g` below). That is,

    .. math::

        f(x, y) = \sum_{s\in\{\pm 1\}^d} g(s) \frac{\exp \Big \{ -\langle T(s), (x, y) \rangle \Big \}}{Z(x, y)}.

    where

    .. math::

        Z(x, y) = \sum_{s\in\{\pm 1\}^d} \exp \Big \{ -\langle T(s), (x, y) \rangle \Big \}

    is the partition function.

    Model outputs are, in theory, deterministic. In practice, in this implementation, model outputs
    are stochastic due to its computational intractability and thus the need to employ a Monte Carlo
    approximation.

    Inputs ``x`` and ``y`` should have shape ``(B, |V|)`` and ``(B, |E|)`` respectively where
    ``B`` indicates a batch size. Outputs have shape ``(B, D)`` where ``D`` is the output dimension
    of ``statistic``.

    In practice, when sampling using a quantum annealer, samples are not guaranteed to be
    Boltzmann---which is an assumption this implementation leans on. Furthermore, a quantum annealer
    samples at an effective inverse temperature (beta) that is not guaranteed to be 1. This poses a
    problem when the Ising module is used in conjunction with other modules, where---if beta is not
    accounted for---gradient estimates will be off by a factor equal to beta. In other words, the
    effective learning rate of this layer will differ from other parameters by a factor of beta.
    To account for beta, use the method ``set_beta``. To estimate beta, use the method
    ``estimate_betas``. See
    `Global Warming: Temperature Estimation in Annealers <https://doi.org/10.3389/fict.2016.00023>`_.
    for more on estimating beta.

    The gradient of the expected output statistic (``statistic`` or :math:`g` mapping from
    :math:`\{\pm 1\} ^N` to real numbers), is the negative covariance between the sufficient statistic
    and :math:`g`. See
    `Graphical Models, Exponential Families, and Variational Inference <https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf>`_
    for a more rigorous treatment.

    .. math::

        \nabla_\theta \mathbb{E} \left[ g(S)\right]  & = \sum_{s\in\{\pm 1\}^d} \nabla_\theta g(s) \frac{\exp \Big \{ -\langle T(s), \theta \rangle \Big \}}{Z(\theta)}

         & = \sum_{s\in\{\pm 1\}^d} \frac{g(s)}{Z(\theta)^2} \Bigg [ -T(s)\exp \Big \{ -\langle T(s), \theta \rangle  \Big\} Z(\theta) + \exp \Big \{ -\langle T(s), \theta\rangle  \Big\} \mathbb{E}\left[ T(S) \right]Z(\theta) \Bigg]

         & = \sum_{s\in\{\pm 1\}^d} \frac{g(s)}{Z(\theta)} \Bigg[ -T(s)\exp \Big \{ -\langle T(s),\theta\rangle  \Big\} + \exp \Big \{ -\langle T(s), \theta\rangle  \Big\} \mathbb{E}\left[ T(S) \right]\Bigg]

         & = -\mathbb{E}\left[g(S)T(S)\right] + \mathbb{E}\left[g(S)\right] \mathbb{E}\left[ T(S) \right]

         & = -\text{Covariance}\left[g(S), T(S)\right].

    Args:
        nodes: Nodes of the model.
        edges: Edges of the model.
        sampler: The sampler used to sample from the model.
        sample_params: Keyword arguments used in the ``sampler.sample`` method.
        beta: Effective inverse temperature of the sampler.
        statistic: Function mapping spins to statistics. If None, the statistic corresponds
            to the input nodes and input edges. Defaults to None.
    """

    def __init__(
        self,
        nodes: Iterable[Hashable],
        edges: Iterable[tuple[Hashable, Hashable]],
        sampler: dimod.Sampler,
        sample_params: dict,
        beta: float,
        statistic: SpinStatistic | None = None,
    ) -> None:
        super().__init__()
        if beta <= 0:
            raise ValueError(f"Effective inverse temperature beta must be positive. Got {beta}.")

        self._nodes = list(nodes)
        self._edges = list(edges)

        self._num_nodes = len(self.nodes)
        self._num_edges = len(self.edges)

        self._sampler = sampler
        self._sample_params = sample_params.copy()
        self._beta = nn.Parameter(torch.tensor(beta, dtype=torch.float32), False)

        # Some indexing
        self.register_buffer(
            "_node_idx_of_edges_1",
            torch.tensor([self.nodes.index(u) for u, _ in self.edges], dtype=int)
        )
        self.register_buffer(
            "_node_idx_of_edges_2",
            torch.tensor([self.nodes.index(v) for _, v in self.edges], dtype=int)
        )

        # Default to identity
        if statistic is None:
            statistic = IdentityStatistic(self.num_nodes)
        self.statistic = statistic
        self.dim_out = statistic.dim_out

        # Helper function for autograd
        self.ising_aggregation = IsingExpectation.apply

    def set_sampler(self, sampler: dimod.Sampler) -> None:
        """Set the sampler to ``sampler``.

        Args:
            sampler: The sampler used to sample from the model.
        """
        self._sampler = sampler

    @property
    def sampler(self) -> dimod.Sampler:
        """Sampler used to sample from the model."""
        return self._sampler

    def set_sample_params(self, sample_params: dict) -> None:
        """Set sampling parameters.

        Args:
            sample_params: Keyword arguments used in the ``sampler.sample`` method.
        """
        self._sample_params = sample_params.copy()

    @property
    def sample_params(self) -> dict:
        """Sampling parameters used to sample from the model."""
        return self._sample_params

    @property
    def nodes(self) -> list[Hashable]:
        """Edges of the model."""
        return self._nodes

    @property
    def edges(self) -> list[Hashable, Hashable]:
        """Nodes of the model."""
        return self._edges

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the model."""
        return self._num_nodes

    @property
    def num_edges(self) -> int:
        """Number of edges in the model."""
        return self._num_edges

    @property
    def beta(self) -> nn.Parameter[torch.FloatTensor]:
        """The effective inverse temperature of the sampler."""
        return self._beta

    def set_beta(self, beta: float) -> None:
        """Set the effective inverse temperature of the sampler.

        Args:
            beta: The effective inverse temperature of the sampler.
        """
        if beta <= 0:
            raise ValueError(f"Effective inverse temperature beta must be positive. Got {beta}.")
        self._beta.data = torch.tensor(beta, dtype=self._beta.dtype)

    @property
    def node_idx_of_edges_1(self) -> torch.Tensor:
        """Node indices of first endpoint of edges."""
        return self._node_idx_of_edges_1

    @property
    def node_idx_of_edges_2(self) -> torch.Tensor:
        """Node indices of second endpoint of edges."""
        return self._node_idx_of_edges_2

    def _interactions(self, x: torch.Tensor) -> torch.Tensor:
        """Interaction terms of the Ising model.

        Args:
            x: Spins of shape (..., D).

        Returns:
            Interaction terms ``x[..., self.edge_idx_u] * x[..., self.edge_idx_v]``
        """
        return x[..., self.node_idx_of_edges_1] * x[..., self.node_idx_of_edges_2]

    def _sample(
        self,
        linear_biases: torch.Tensor,
        quadratic_biases: torch.Tensor
    ) -> list[dimod.SampleSet]:
        """Sample from a batch of models defined by ``linear/self.beta`` and ``quadratic/self.beta`` biases.

        .. note:: Linear and quadratic biases are scaled by ``1/self.beta`` prior to sampling, thus extra
        caution should be taken when estimating beta.

        Args:
            linear_biases: Linear biases of shape (B, D1) where B is batch size and D1 is the number
                of nodes in the model.
            quadratic_biases: Quadratic biases of shape (B, D2) where B is batch size and D2 is the
                number of edges in the model.

        Returns:
            A corresponding list of B sample sets.
        """
        sample_sets = []
        for linear, quadratic in zip(linear_biases / self.beta, quadratic_biases / self.beta):
            sample_sets.append(self._sampler.sample_ising(
                dict(zip(self.nodes, linear.tolist())),
                dict(zip(self.edges, quadratic.tolist())),
                **self._sample_params
            ))
        return sample_sets

    def _to_tensor(
        self,
        sample_sets: Iterable[dimod.SampleSet],
        device: torch.device | None = None
    ) -> torch.Tensor:
        """Converts a list of sample sets to a tensor.

        Args:
            sample_sets: A list of sample sets.
            device: The device of the constructed tensor. If None, then the
                resulting tensor is constructed on self.beta's device. Defaults to None.

        Returns:
            A tensor of shape (B, ``len(sample_sets)``, D1) where B is batch size and D1 is the
            number of nodes in the model.
        """
        if device is None:
            device = self.beta.device
        spins = torch.stack([sampleset_to_tensor(self.nodes, ss, device) for ss in sample_sets])
        return spins

    def forward(self, linear: torch.Tensor, quadratic: torch.Tensor) -> torch.Tensor:
        """Approximate the expected output statistics of the Ising model.

        Args:
            linear: Input data with shape (B, D1) where D1 is the number of nodes in the model.
            linear: Input data with shape (B, D2) where D2 is the number of edges in the model.

        Raises:
            ValueError: If ``x.ndim != 2``.

        Returns:
            Sample-approximation of expected output statistics with shape (B, D3) where D3 is the
            output dimension of the output statistic (the ``statistic`` parameter in the constructor).
        """
        if linear.ndim != 2:
            raise ValueError(f"linear.ndim should be exactly 2. linear.ndim is {linear.ndim}")

        if quadratic.ndim != 2:
            raise ValueError(
                f"quadratic.ndim should be exactly 2. quadratic.ndim is {quadratic.ndim}"
            )

        sample_sets = self._sample(linear, quadratic)

        spins = self._to_tensor(sample_sets, linear.device)
        interactions = self._interactions(spins)
        stats_out = self.statistic(spins)

        return self.ising_aggregation(spins, interactions, stats_out, linear, quadratic)

    def estimate_betas(self, linear_biases, quadratic_biases) -> list[float]:
        """Estimate the maximum pseudolikelihood temperature using
        ``dwave.system.temperatures.maximum_pseudolikelihood_temperature``.

        See
        `Global Warming: Temperature Estimation in Annealers <https://doi.org/10.3389/fict.2016.00023>`_.
        for more on estimating beta.

        Args:
            linear_biases: Linear biases of shape (B, D1) where B is batch size and D1 is the number
                of nodes in the model.
            quadratic_biases: Quadratic biases of shape (B, D2) where B is batch size and D2 is the
                number of edges in the model.

        Returns:
            Tensor of length B estimates of inverse temperature of the model where B is batch size.
        """
        bqms = [BinaryQuadraticModel.from_ising(
                dict(zip(self.nodes, linear.tolist())),
                dict(zip(self.edges, quadratic.tolist())))
                # NOTE: Notice `self.beta` is not used to scale when sampling, c.f., `_sample`.
                for linear, quadratic in zip(linear_biases, quadratic_biases)]
        sample_sets = [self._sampler.sample(bqm, **self._sample_params) for bqm in bqms]
        betas = torch.tensor([1 / float(mple(bqm, ss)[0]) for bqm, ss in zip(bqms, sample_sets)])
        return betas
