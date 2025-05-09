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

import torch


__all__ = ["GraphRestrictedBoltzmannMachine"]


class GraphRestrictedBoltzmannMachine(torch.nn.Module):
    """Creates a graph-restricted Boltzmann machine.

    Args:
        nodes (list): List of nodes.
        edges (list): List of edges.
    """

    def __init__(self, nodes: list, edges: list, *args, **kwargs):
        super().__init__(*args, **kwargs)

        n = len(nodes)
        self.idx_to_var = {i: v for i, v in enumerate(nodes)}
        var_to_idx = {v: i for i, v in self.idx_to_var.items()}

        m = len(edges)
        edge_idx_i = torch.tensor([var_to_idx[i] for i, j in edges])
        edge_idx_j = torch.tensor([var_to_idx[j] for i, j in edges])

        self.h = torch.nn.Parameter(0.05 * (2 * torch.randint(0, 2, (n,)) - 1))
        self.J = torch.nn.Parameter(5.0 * (2 * torch.randint(0, 2, (m,)) - 1))

        self.register_buffer("edge_idx_i", edge_idx_i)
        self.register_buffer("edge_idx_j", edge_idx_j)

    @property
    def theta(self) -> torch.Tensor:
        return torch.cat([self.h, self.J])

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
            tuple[torch.Tensor, torch.Tensor]: The sufficient statistics of ``x``.
        """
        interactions = self.interactions(x)
        return torch.cat([x, interactions], 1)

    def ising(self, beta) -> tuple[dict, dict]:
        """Convert the model to Ising format"""
        linear_bias_list = (beta * self.h.detach()).cpu().tolist()
        linear_biases = {self.idx_to_var[i]: b for i, b in enumerate(linear_bias_list)}
        edge_idx_i = self.edge_idx_i.detach().cpu().tolist()
        edge_idx_j = self.edge_idx_j.detach().cpu().tolist()
        quadratic_bias_list = (beta * self.J.detach()).cpu().tolist()
        quadratic_biases = {
            (self.idx_to_var[a], self.idx_to_var[b]): w
            for a, b, w in zip(edge_idx_i, edge_idx_j, quadratic_bias_list)
        }
        return linear_biases, quadratic_biases
