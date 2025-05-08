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
from dimod import Sampler

from dwave.plugins.torch.boltzmann_machine import AbstractBoltzmannMachine

__all__ = ["pseudo_kl_divergence_loss"]


def pseudo_kl_divergence_loss(
    spins: torch.Tensor,
    logits: torch.Tensor,
    boltzmann_machine: AbstractBoltzmannMachine,
    sampler: Sampler,
    sampler_kwargs: dict,
):
    """
    A pseudo Kullback-Leibler divergence loss function for a discrete autoencoder with a
    Boltzmann machine prior.

    This is not the true KL divergence, but the gradient of
    this function is the same as the KL divergence gradient.

    Args:
        spins (torch.Tensor): A tensor of spins of shape (batch_size, n_spins) or shape
            (batch_size, n_samples, n_spins) obtained from a stochastic function that
            maps the output of the encoder (logit representation) to a spin
            representation.
        logits (torch.Tensor): A tensor of logits of shape (batch_size, n_spins). These
            logits are the raw output of the encoder.
        boltzmann_machine (AbstractBoltzmannMachine): An instance of a Boltzmann
            machine.
        sampler (Sampler): A sampler used for generating samples.
        sampler_kwargs (dict): Additional keyword arguments for the ``sampler.sample``
            method.

    Returns:
        torch.Tensor: The computed pseudo KL divergence loss.
    """
    samples = boltzmann_machine.sample(
        sampler=sampler, device=spins.device, **sampler_kwargs
    )
    probabilities = torch.sigmoid(logits)
    entropy_of_encoder = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, probabilities
    )
    cross_entropy = torch.mean(boltzmann_machine(spins)) - torch.mean(
        boltzmann_machine(samples)
    )
    pseudo_kl_divergence = cross_entropy - entropy_of_encoder
    return pseudo_kl_divergence
