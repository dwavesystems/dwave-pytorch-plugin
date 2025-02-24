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

import torch


def binary_to_spins(x: torch.Tensor) -> torch.Tensor:
    return 2 * x - 1  # Convert binary to {-1, 1}


class GumbelSoftmax(torch.nn.Module):
    def __init__(self, tau: int | float):
        """
        Model that performs the gumbel trick on a single spin variable. See Eq. 10 in
        "A path towards quantum advantage in training deep generative models with
        quantum annealers".

        Args:
            tau (int | float): Temperature parameter. Higher (lower) temperature makes
            the output be more (less) distributed according to a Bernoulli distribution.
        """
        super().__init__()
        self.tau = tau

    def forward(self, x: torch.Tensor):
        uniforms = torch.rand(x.shape, device=x.device)
        zeta = binary_to_spins(
            torch.sigmoid(
                (x - torch.log(uniforms) - torch.log(1 - uniforms)) / self.tau
            )
        )
        zeta_detached = zeta.detach()
        spins = binary_to_spins(torch.heaviside(zeta_detached, torch.tensor(0)))
        # The following ensures that the forward pass is spins, and the backward pass
        # uses zeta, as both spins and zeta_detached have gradients equal to zero.
        return spins - zeta_detached + zeta


class AutoEncoder(torch.nn.Module):
    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_to_spins: torch.nn.Module | None = None,
    ):
        """
        Autoencoder architecture amenable for training spin-variable models as priors.
        These include QPU models.

        Args:
            encoder (torch.nn.Module): The encoder must output latents that are later on
                passed to `latent_to_spins`. An encoder has signature (x) -> l
            decoder (torch.nn.Module): Decodes spin tensors into data tensors. A decoder
                has signature (s) -> x.
            latent_to_spins (torch.nn.Module | None): Module that maps the latent
                representation of data (obtained with the encoder) to a tensor of spins
                that is then consumed by the decoder. A latent_to_spins has signature
                (l) -> s. If None, a model that uses the gumbel trick is used. Defaults
                to None.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if latent_to_spins is None:
            # We use the tau value in "A path towards quantum advantage in training deep
            # generative models with quantum annealers":
            latent_to_spins = GumbelSoftmax(tau=1 / 7)
        self.latent_to_spins = latent_to_spins

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self.encoder(x)
        spins = self.latent_to_spins(latents)
        # TODO: discuss: most decoders work like decoder(spins); however, diffusion
        # models could also be captured in this architecture by allowing
        # decoder(x, spins). Should we include this here?
        reconstructed_x = self.decoder(spins)
        return reconstructed_x, spins, latents
