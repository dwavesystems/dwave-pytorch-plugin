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

import torch
from parameterized import parameterized

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine
from dwave.plugins.torch.models.discrete_variational_autoencoder import (
    DiscreteVariationalAutoencoder as DVAE,
)
from dwave.plugins.torch.models.losses.kl_divergence import pseudo_kl_divergence_loss
from dwave.samplers import SimulatedAnnealingSampler


class TestDiscreteVariationalAutoencoder(unittest.TestCase):
    """Tests the DiscreteVariationalAutoencoder with dummy data"""

    def setUp(self):
        torch.manual_seed(1234)
        input_features = 2
        latent_features = 2

        # Data in corners of unit square:
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
            def __init__(self, n_latent_dims: int):
                super().__init__()
                self.n_latent_dims = n_latent_dims

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x is always two-dimensional of shape (batch_size, features_size)
                dims_to_add = self.n_latent_dims - 1
                output = x * 20 - 10
                for _ in range(dims_to_add):
                    output = output.unsqueeze(-2)
                return output

        class Decoder(torch.nn.Module):
            def __init__(self, latent_features: int, input_features: int):
                super().__init__()
                self.linear = torch.nn.Linear(latent_features, input_features)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # x is of shape (batch_size, n_samples, l1, l2, ...)
                n_latent_dims_to_remove = x.ndim - 3
                for _ in range(n_latent_dims_to_remove):
                    x = x.squeeze(1)
                return self.linear(x)

        # self.encoders is a dict whose keys are the number of latent dims and the values
        # are the models themselves
        latent_dims_list = [1, 2]
        self.encoders = {i: Encoder(i) for i in latent_dims_list}
        # self.decoders is independent of number of latent dims, but we also create a dict to separate
        # them
        self.decoders = {i: Decoder(latent_features, input_features) for i in latent_dims_list}

        # self.dvaes is a dict whose keys are the numbers of latent dims and the values are the models
        # themselves

        self.dvaes = {i: DVAE(self.encoders[i], self.decoders[i]) for i in latent_dims_list}

        self.boltzmann_machine = GraphRestrictedBoltzmannMachine(
            nodes=(0, 1),
            edges=[(0, 1)],
            linear={0: 0.1, 1: -0.2},
            quadratic={(0, 1): -1.2},
        )

        self.sampler_sa = SimulatedAnnealingSampler()

    def test_mappings(self):
        """Test the mapping between data and logits."""
        # Let's make sure that indeed the maps are correct. For this, we use only the first
        # autoencoder, which is the one whose encoder maps data to a single feature dimension. The
        # second autoencoder maps data to two feature dimensions (the last one is a dummy dimension)
        _, discretes, _ = self.dvaes[1](self.data, n_samples=1)
        # squeeze the replica dimension:
        discretes = discretes.squeeze(1)
        # map [1, 1] to [1, 1]:
        torch.testing.assert_close(torch.tensor([1, 1]).float(), discretes[0])
        # map [1, 0] to [1, -1]:
        torch.testing.assert_close(torch.tensor([1, -1]).float(), discretes[1])
        # map [0, 0] to [-1, -1]:
        torch.testing.assert_close(torch.tensor([-1, -1]).float(), discretes[2])
        # map [0, 1] to [-1, 1]:
        torch.testing.assert_close(torch.tensor([-1, 1]).float(), discretes[3])

    @parameterized.expand([1, 2])
    def test_train(self, n_latent_dims):
        """Test training simple dataset."""
        dvae = self.dvaes[n_latent_dims]
        optimiser = torch.optim.SGD(
            list(dvae.parameters()) + list(self.boltzmann_machine.parameters()),
            lr=0.01,
            momentum=0.9,
        )
        N_SAMPLES = 1
        for _ in range(1000):
            latents, discretes, reconstructed_data = dvae(self.data, n_samples=N_SAMPLES)
            true_data = self.data.unsqueeze(1).repeat(1, N_SAMPLES, 1)

            # Measure the reconstruction loss
            loss = torch.nn.functional.mse_loss(reconstructed_data, true_data)

            discretes = discretes.reshape(discretes.shape[0], -1)
            latents = latents.reshape(latents.shape[0], -1)
            samples = self.boltzmann_machine.sample(
                self.sampler_sa,
                as_tensor=True,
                prefactor=1.0,
                sample_params=dict(num_sweeps=10, seed=1234, num_reads=100),
            )
            kl_loss = pseudo_kl_divergence_loss(
                discretes,
                latents,
                samples,
                self.boltzmann_machine,
            )
            loss = loss + 1e-1 * kl_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        # We should reach almost perfect reconstruction of the data:
        torch.testing.assert_close(true_data, reconstructed_data)
        # Furthermore, the GRBM should learn that all spin strings of length 2 are
        # equally likely, so the h and J parameters should be close to 0:
        torch.testing.assert_close(
            self.boltzmann_machine.linear, torch.zeros(2), rtol=1e-2, atol=1e-2
        )
        torch.testing.assert_close(
            self.boltzmann_machine.quadratic, torch.tensor([0.0]).float(), rtol=1e-2, atol=1e-2
        )

    @parameterized.expand(
        [
            (1, torch.tensor([[[1.0, 1.0]], [[1.0, -1.0]], [[-1.0, -1.0]], [[-1.0, 1.0]]])),
            (
                5,
                torch.tensor(
                    [[[1.0, 1.0]] * 5, [[1.0, -1.0]] * 5, [[-1.0, -1.0]] * 5, [[-1.0, 1.0]] * 5]
                ),
            ),
        ]
    )
    def test_latent_to_discrete(self, n_samples, expected):
        """Test the latent_to_discrete default method."""
        # All encoders and dvaes only differ in the number of dummy feature dimensions in the latent
        # space. For this reason, this test can only be done with the case of one feature dimension,
        # which corresponds to the first encoder and dvae.
        latents = self.encoders[1](self.data)
        discretes = self.dvaes[1].latent_to_discrete(latents, n_samples)
        assert torch.equal(discretes, expected)

    @parameterized.expand([(i, j) for i in range(1, 3) for j in [0, 1, 5, 1000]])
    def test_forward(self, n_latent_dims, n_samples):
        """Test the forward method."""
        expected_latents = self.encoders[n_latent_dims](self.data)
        expected_discretes = self.dvaes[n_latent_dims].latent_to_discrete(
            expected_latents, n_samples
        )
        expected_reconstructed_x = self.decoders[n_latent_dims](expected_discretes)

        latents, discretes, reconstructed_x = self.dvaes[n_latent_dims].forward(
            x=self.data, n_samples=n_samples
        )

        assert torch.equal(reconstructed_x, expected_reconstructed_x)
        assert torch.equal(discretes, expected_discretes)
        assert torch.equal(latents, expected_latents)


if __name__ == "__main__":
    unittest.main()
