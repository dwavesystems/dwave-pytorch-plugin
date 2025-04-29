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

# The use of the auto encoder implementations below (including the
# AutoEncoder) with a quantum computing system is
# protected by the intellectual property rights of D-Wave Quantum Inc.
# and its affiliates.
#
# The use of the auto encoder implementations below (including the
# AutoEncoder) with D-Wave's quantum computing
# system will require access to D-Wave’s LeapTM quantum cloud service and
# will be governed by the Leap Cloud Subscription Agreement available at:
# https://cloud.dwavesys.com/leap/legal/cloud_subscription_agreement/
#

from collections.abc import Callable

import torch

__all__ = ["AutoEncoder"]


class AutoEncoder(torch.nn.Module):
    """Autoencoder architecture amenable for training discrete models as priors.

    Such discrete models include spin-variable models amenable for the QPU. This
    architecture is a modification of the standard autoencoder architecture, where
    the encoder outputs a latent representation of the data, and the decoder
    reconstructs the data from the latent representation. In our case, there is an
    additional step where the latent representation is mapped to a discrete
    representation, which is then passed to the decoder.

    Args:
        encoder (torch.nn.Module): The encoder must output latents that are later on
            passed to `latent_to_discrete`. An encoder has signature (x) -> l.
        decoder (torch.nn.Module): Decodes discrete tensors into data tensors. A decoder
            has signature (d) -> x', where x' might be the reconstructed data with the
            same shape as x, or might be another representation of the data (e.g. in a
            text-to-image model, x is a sequence of tokens, and x' is an image).
        latent_to_discrete (Callable[[torch.Tensor], torch.Tensor] | None): Maps the
            latent representation of data (obtained with the encoder) to a tensor of
            discretes that is then consumed by the decoder. A latent_to_discrete has
            signature (l) -> d. If None, a model that uses the gumbel trick is used.
            This assumes that the encoder output is of shape (batch_size, n_discrete)
            and that the decoder input is of the same shape. Defaults to None.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
        latent_to_discrete: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        if latent_to_discrete is None:

            def _latent_to_discrete(logits: torch.Tensor) -> torch.Tensor:
                # Logits is of shape (batch_size, n_discrete), we assume these logits
                # refer to the probability of each discrete variable being 1. To use the
                # gumbel softmax function we need to reshape the logits to (batch_size,
                # n_discrete, 1), and then stack the logits to a zeros tensor of the
                # same shape. This is done to ensure that the gumbel softmax function
                # works correctly.

                logits = logits.unsqueeze(-1)
                logits = torch.cat((logits, torch.zeros_like(logits)), dim=-1)
                one_hots = torch.nn.functional.gumbel_softmax(
                    logits, tau=1 / 7, hard=True
                )
                # one_hots is of shape (batch_size, n_discrete, 2), we need to take the
                # first element of the last dimension and convert it to spin variables
                # to make the latent space compatible with QPU models.
                return one_hots[..., 0] * 2 - 1

            latent_to_discrete = _latent_to_discrete

        self.latent_to_discrete = latent_to_discrete

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latents = self.encoder(x)
        discrete = self.latent_to_discrete(latents)
        reconstructed_x = self.decoder(discrete)
        return reconstructed_x, discrete, latents
