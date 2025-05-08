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

import unittest

import numpy as np
import torch

from dwave.plugins.torch.autoencoder import DiscreteAutoEncoder
from dwave.plugins.torch.autoencoder.losses.kl_divergence import (
    pseudo_kl_divergence_loss,
)
from dwave.plugins.torch.boltzmann_machine import GraphRestrictedBoltzmannMachine
from dwave.samplers import SimulatedAnnealingSampler


class TestDiscreteAutoEncoder(unittest.TestCase):
    """
    Tests the DiscreteAutoEncoder with dummy data
    """

    def setUp(self):
        torch.manual_seed(1234)
        input_features = 2
        latent_features = 2

        # Datapoints in corners of unit square:
        self.data = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])

        # The encoder maps input data to logits. We make this encoder without parameters
        # for simplicity. The encoder will map 1s to 10s and 0s to -10s, so that the
        # stochasticity from the Gumbel softmax will only change these logits to [11, 9]
        # and [-9, -11] respectively.
        # When generating the discrete representation, -1s and 1s will be sampled using
        # these logits, so, almost deterministically we will have that the encoder plus
        # the latent_to_discrete map of the autoencoder will perform the bits to spins
        # mapping, i.e., the datapoint [1, 0] will be mapped to the spin string
        # [1, -1].

        class Encoder(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x * 20 - 10

        decoder = torch.nn.Linear(latent_features, input_features)

        self.boltzmann_machine = GraphRestrictedBoltzmannMachine(
            num_nodes=2, edge_idx_i=torch.tensor([0]), edge_idx_j=torch.tensor([1])
        )
        self.autoencoder = DiscreteAutoEncoder(Encoder(), decoder)

        self.sampler = SimulatedAnnealingSampler()

    def test_train(self):
        optimiser = torch.optim.SGD(
            list(self.autoencoder.parameters())
            + list(self.boltzmann_machine.parameters()),
            lr=0.01,
            momentum=0.9,
        )
        n_samples = 1
        for _ in range(100):
            reconstructed_data, discretes, latents = self.autoencoder(
                self.data, n_samples=n_samples
            )
            true_data = self.data.unsqueeze(1).repeat(1, n_samples, 1)
            loss = torch.nn.functional.mse_loss(reconstructed_data, true_data)
            loss += 1e-1 * pseudo_kl_divergence_loss(
                discretes,
                latents,
                self.boltzmann_machine,
                self.sampler,
                dict(num_sweeps=10, seed=1234, num_reads=100),
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # We should reach almost perfect reconstruction of the data:
        np.testing.assert_almost_equal(
            true_data.detach().numpy(),
            reconstructed_data.detach().numpy(),
            decimal=2,
        )
        # Furthermore, the GRBM should learn that all spin strings of length 2 are
        # equally likely, so the h and J parameters should be close to 0:
        np.testing.assert_almost_equal(
            self.boltzmann_machine.h.detach().numpy(),
            np.zeros(2),
            decimal=1,
        )
        np.testing.assert_almost_equal(
            self.boltzmann_machine.J.detach().numpy(),
            np.zeros(1),
            decimal=1,
        )


if __name__ == "__main__":
    unittest.main()
