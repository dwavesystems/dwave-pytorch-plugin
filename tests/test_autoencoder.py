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
from parameterized import parameterized
import torch

from dwave.plugins.torch.autoencoder import DiscreteAutoEncoder
from dwave.plugins.torch.autoencoder.losses.kl_divergence import pseudo_kl_divergence_loss
from dwave.plugins.torch.boltzmann_machine import GraphRestrictedBoltzmannMachine
from dwave.samplers import SimulatedAnnealingSampler


class TestDiscreteAutoEncoder(unittest.TestCase):
    """Tests the DiscreteAutoEncoder with dummy data"""

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

        self.encoder = Encoder()
        self.decoder = torch.nn.Linear(latent_features, input_features)

        self.autoencoder = DiscreteAutoEncoder(self.encoder, self.decoder)

        self.boltzmann_machine = GraphRestrictedBoltzmannMachine(
            nodes=(0, 1), edges=((0, 1),)
        )

        self.sampler = SimulatedAnnealingSampler()


    def test_mappings(self):
        """Test the mapping between data and logits."""
        # Let's make sure that indeed the maps are correct:
        _, discretes, _ = self.autoencoder(self.data, n_samples=1)
        discretes = discretes.squeeze(1)
        # map [1, 1] to [1, 1]:
        np.testing.assert_array_almost_equal(np.array([1, 1]), discretes[0].numpy())
        # map [1, 0] to [1, -1]:
        np.testing.assert_array_almost_equal(np.array([1, -1]), discretes[1].numpy())
        # map [0, 0] to [-1, -1]:
        np.testing.assert_array_almost_equal(np.array([-1, -1]), discretes[2].numpy())
        # map [0, 1] to [-1, 1]:
        np.testing.assert_array_almost_equal(np.array([-1, 1]), discretes[3].numpy())


    def test_train(self):
        """Test training simple dataset."""
        optimiser = torch.optim.SGD(
            list(self.autoencoder.parameters())
            + list(self.boltzmann_machine.parameters()),
            lr=0.01,
            momentum=0.9,
        )
        n_samples = 1
        for _ in range(1000):
            reconstructed_data, discretes, latents = self.autoencoder(
                self.data, n_samples=n_samples
            )
            true_data = self.data.unsqueeze(1).repeat(1, n_samples, 1)

            # Measure the reconstruction loss
            loss = torch.nn.functional.mse_loss(reconstructed_data, true_data)

            discretes = discretes.reshape(discretes.shape[0], -1)
            latents = latents.reshape(latents.shape[0], -1)
            kl_loss = pseudo_kl_divergence_loss(
                discretes,
                latents,
                self.boltzmann_machine,
                self.sampler,
                dict(num_sweeps=10, seed=1234, num_reads=100),
            ).mean()  # the tensor is tensor([number]), so the mean extracts the number
            # TODO: discuss with Kevin, how do I get a number tensor, i.e.,
            # tensor(number) instead of the tensor([number])?
            loss += 1e-1 * kl_loss
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
            self.boltzmann_machine.linear.detach().numpy(),
            np.zeros(2),
            decimal=1,
        )
        np.testing.assert_almost_equal(
            self.boltzmann_machine.quadratic.detach().numpy(),
            np.zeros(1),
            decimal=1,
        )

    @parameterized.expand([
        (
            1,
            torch.tensor([[[ 1.,  1.]], [[ 1., -1.]], [[-1., -1.]], [[-1.,  1.]]])
        ),
        (
            5,
            torch.tensor([[[ 1.,  1.]] * 5, [[ 1., -1.]] * 5, [[-1., -1.]] * 5, [[-1.,  1.]] * 5])
        ),
    ])
    def test_latent_to_discrete(self, n_samples, expected):
        """Test the latent_to_discrete default method."""
        logits = self.encoder(self.data)
        discretes = self.autoencoder.latent_to_discrete(logits, n_samples)
        assert torch.equal(discretes, expected)

    @parameterized.expand([0, 1, 5, 1000])
    def test_forward(self, n_samples):
        """Test the forward method."""
        logits = self.encoder(self.data)
        discretes = self.autoencoder.latent_to_discrete(logits, n_samples)
        reconstructed_x = self.decoder(discretes)

        expected_reconstructed_x, expected_discretes, expected_logits = self.autoencoder.forward(
            x=self.data,
            n_samples=n_samples
        )

        assert torch.equal(reconstructed_x, expected_reconstructed_x)
        assert torch.equal(discretes, expected_discretes)
        assert torch.equal(logits, expected_logits)


if __name__ == "__main__":
    unittest.main()
