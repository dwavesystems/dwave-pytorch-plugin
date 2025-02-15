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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch

from dwave.plugins.torch.utils import sample_to_tensor, spread

if TYPE_CHECKING:
    from dimod import Sampler

__all__ = [
    "GraphRestrictedBoltzmannMachine",
]


class AbstractBoltzmannMachine(ABC, torch.nn.Module):
    """Abstract class for Boltzmann machines.

    Args:
        h_range (tuple[float, float], optional): range of linear weights. Defaults to None.
        j_range (tuple[float, float], optional): range of quadratic weights. Defaults to None.
    """

    def __init__(
        self, h_range: tuple[float, float] = None, j_range: tuple[float, float] = None
    ) -> None:
        super().__init__()

        self.register_buffer(
            "h_range",
            torch.tensor(h_range if h_range is not None else [-torch.inf, torch.inf]),
        )
        self.register_buffer(
            "j_range",
            torch.tensor(j_range if j_range is not None else [-torch.inf, torch.inf]),
        )

        if (h_range and not j_range) or (j_range and not h_range):
            raise NotImplementedError(
                "Both or neither weight range should be specified."
            )

        self.register_forward_pre_hook(lambda *args: self.clip_parameters())

    @abstractmethod
    def sufficient_statistics(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the sufficient statistics of a Boltzmann machine, i.e., average spin
        and average interaction values (per edge) of ``x``.

        Args:
            x (torch.Tensor): a tensor of shape (..., N)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the average spin and average spin-spin of `x`.
        """

    def clip_parameters(self):
        """Clips linear and quadratic bias weights in-place."""
        self.get_parameter("h").data.clamp_(*self.h_range)
        self.get_parameter("J").data.clamp_(*self.j_range)

    @property
    def ising(self) -> tuple[dict, dict]:
        """Converts the model to Ising format."""
        self.clip_parameters()
        return self._ising

    @property
    @abstractmethod
    def _ising(self) -> tuple[dict, dict]:
        """Convert the model to Ising format"""

    @staticmethod
    def pairwise_matrix(x: torch.Tensor) -> torch.Tensor:
        """Computes a matrix whose off-diagonals are average spin-spin interactions and
        diagonal is the average spin.

        Args:
            x (torch.Tensor): A tensor with shape (batch, N)

        Returns:
            torch.Tensor: the pairwise matrix with shape (batch, N, N)
        """
        mtx = torch.bmm(x.unsqueeze(2), x.unsqueeze(1)).mean(0)
        mtx = mtx - torch.diag(mtx.diagonal()) + torch.diag(x.mean(0))
        return mtx

    def objective(
        self, s_observed: torch.Tensor, s_model: torch.Tensor
    ) -> torch.Tensor:
        """An objective function with gradients equivalent to the gradients of the
        negative log likelihood.

        Args:
            s_observed (torch.Tensor): tensor of observed (data) spins with shape (b1, N)
            s_model (torch.Tensor): tensor of spins drawn from the model with shape (b2, N)

        Returns:
            torch.Tensor: a scalar; difference of the average energies between the two samples
        """
        self.clip_parameters()
        return self(s_observed).mean() - self(s_model).mean()

    def sample(
        self, sampler: Sampler, device: str = None, **sample_params: dict
    ) -> torch.Tensor:
        """Sample from the Boltzmann machine.
        This method converts the sampled spins to tensors and ensures they are not
        aggregated.

        Args:
            sampler (Sampler): the sampler used to sample from the model
            sampler_params (dict): parameters of the `sampler.sample` method.

        Returns:
            torch.torch.Tensor: spins sampled from the model
            (shape prescribed by `sampler` and `sample_params`)
        """
        self.clip_parameters()
        h, J = self.ising
        ss = spread(sampler.sample_ising(h, J, **sample_params))
        spins = sample_to_tensor(ss, device=device)
        return spins


class GraphRestrictedBoltzmannMachine(AbstractBoltzmannMachine):
    """Creates a graph-restricted Boltzmann machine.

    Args:
        num_nodes (int): number of variables in the model.
        edge_idx_i (torch.Tensor): list of endpoints i of a list of edges.
        edge_idx_j (torch.Tensor): list of endpoints j of a list of edges.
        h_range (tuple[float, float], optional): range of linear weights. Defaults to None.
        j_range (tuple[float, float], optional): range of quadratic weights. Defaults to None.

    """

    def __init__(
        self,
        num_nodes,
        edge_idx_i,
        edge_idx_j,
        *,
        h_range: tuple = None,
        j_range: tuple = None,
    ):
        super().__init__(h_range=h_range, j_range=j_range)

        number_of_interactions = len(edge_idx_i)
        if edge_idx_i.size(0) != edge_idx_j.size(0):
            raise ValueError("Endpoints 'edge_idx_i' and 'edge_idx_j' are mismatched")

        if torch.unique(torch.cat([edge_idx_i, edge_idx_j])).size(0) > num_nodes:
            raise ValueError(
                "Vertices are assumed to be contiguous nonnegative integers starting from 0 (inclusive). The input edge set implies otherwise."
            )

        self.nodes = num_nodes

        self.h = torch.nn.Parameter(0.01 * (2 * torch.randint(0, 2, (num_nodes,)) - 1))
        self.J = torch.nn.Parameter(
            1.0 * (2 * torch.randint(0, 2, (number_of_interactions,)) - 1)
        )

        self.register_buffer("edge_idx_i", edge_idx_i)
        self.register_buffer("edge_idx_j", edge_idx_j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the Hamiltonian.

        Args:
            x (torch.tensor): a tensor of shape (B, N)

        Returns:
            torch.tensor: Hamiltonians of shape (B,)
        """
        return x @ self.h + self.interactions(x) @ self.J

    def interactions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute interactions prescribed by the model's edges.

        Args:
            x (torch.tensor): tensor of shape (..., number_of_variables)

        Returns:
            torch.tensor: tensor of interaction terms of shape (..., number_of_edges)
        """
        return x[..., self.edge_idx_i] * x[..., self.edge_idx_j]

    def sufficient_statistics(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the sufficient statistics of a Boltzmann machine, i.e., average spin
        and average interaction values (per edge) of `x`.

        Args:
            x (torch.Tensor): a tensor of shape (..., N)

        Returns:
            tuple[torch.Tensor, torch.Tensor]: the average spin and average spin-spin of `x`.
        """
        interactions = self.interactions(x)
        return x.mean(dim=0), interactions.mean(dim=0)

    @property
    def _ising(self) -> tuple[dict, dict]:
        """Convert the model to Ising format"""
        h = self.h.clip(*self.h_range).detach().cpu().tolist()
        edge_idx_i = self.edge_idx_i.detach().cpu().tolist()
        edge_idx_j = self.edge_idx_j.detach().cpu().tolist()
        J_list = self.J.clip(*self.j_range).detach().cpu().tolist()
        J = {(a, b): w for a, b, w in zip(edge_idx_i, edge_idx_j, J_list)}

        return h, J
