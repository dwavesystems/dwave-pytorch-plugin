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

from typing import Optional

import torch
from dimod import Sampler

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine

__all__ = ["pseudo_kl_divergence_loss"]


def pseudo_kl_divergence_loss(
    spins: torch.Tensor,
    logits: torch.Tensor,
    boltzmann_machine: GraphRestrictedBoltzmannMachine,
    sampler: Sampler,
    sample_kwargs: dict,
    prefactor: Optional[float] = None,
    linear_range: Optional[tuple[float, float]] = None,
    quadratic_range: Optional[tuple[float, float]] = None,
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
        sampler (Sampler): A sampler used for generating samples.
        sample_kwargs (dict): Additional keyword arguments for the ``sampler.sample``
            method.
        prefactor (float, optional): A scaling applied to the Hamiltonian weights
            (linear and quadratic weights). When None, no scaling is applied. Defaults
            to None.
        linear_range (tuple[float, float], optional): Linear weights are clipped to
            ``linear_range`` prior to sampling. This clipping occurs after the
            ``prefactor`` scaling has been applied. When None, no clipping is applied.
            Defaults to None.
        quadratic_range (tuple[float, float], optional): Quadratic weights are clipped
            to ``quadratic_range`` prior to sampling. This clipping occurs after the
            ``prefactor`` scaling has been applied. When None, no clipping is applied.
            Defaults to None.

    Returns:
        torch.Tensor: The computed pseudo KL divergence loss.
    """
    samples = boltzmann_machine.sample(
        sampler=sampler,
        device=spins.device,
        prefactor=prefactor if prefactor is not None else 1.0,
        linear_range=linear_range,
        quadratic_range=quadratic_range,
        sample_params=sample_kwargs,
    )
    probabilities = torch.sigmoid(logits)
    entropy = torch.nn.functional.binary_cross_entropy_with_logits(logits, probabilities)
    cross_entropy = boltzmann_machine.quasi_objective(spins, samples)
    pseudo_kl_divergence = cross_entropy - entropy
    return pseudo_kl_divergence
