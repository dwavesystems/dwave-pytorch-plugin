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

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine

__all__ = ["pseudo_kl_divergence_loss"]


def pseudo_kl_divergence_loss(
    spins: torch.Tensor,
    logits: torch.Tensor,
    samples: torch.Tensor,
    boltzmann_machine: GraphRestrictedBoltzmannMachine,
):
    """A pseudo Kullback-Leibler divergence loss function for a discrete autoencoder with a
    Boltzmann machine prior.

    This is not the true KL divergence, but the gradient of this function is the same as
    the KL divergence gradient. See https://arxiv.org/abs/1609.02200 for more details.

    Args:
        spins (torch.Tensor): A tensor of spins of shape (batch_size, n_spins) or shape
            (batch_size, n_samples, n_spins) obtained from a stochastic function that
            maps the output of the encoder (logit representation) to a spin
            representation.
        logits (torch.Tensor): A tensor of logits of shape (batch_size, n_spins). These
            logits are the raw output of the encoder.
        boltzmann_machine (GraphRestrictedBoltzmannMachine): An instance of a Boltzmann
            machine.
        samples (torch.Tensor): A tensor of samples from the Boltzmann machine.

    Returns:
        torch.Tensor: The computed pseudo KL divergence loss.
    """
    probabilities = torch.sigmoid(logits)
    entropy = torch.nn.functional.binary_cross_entropy_with_logits(logits, probabilities)
    cross_entropy = boltzmann_machine.quasi_objective(spins, samples)
    pseudo_kl_divergence = cross_entropy - entropy
    return pseudo_kl_divergence
