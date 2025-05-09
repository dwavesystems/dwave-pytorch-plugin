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

from typing import TYPE_CHECKING

import numpy as np
import torch

from dimod import SampleSet
from hybrid.composers import AggregatedSamples

if TYPE_CHECKING:
    from dimod import Sampler

from dwave.plugins.torch.boltzmann_machine import GraphRestrictedBoltzmannMachine

spread = AggregatedSamples.spread


def sample_to_tensor(
    sample_set: SampleSet, device: torch.device = None
) -> torch.Tensor:
    """Converts a ``dimod.SampleSet`` to a ``torch.Tensor``.

    Args:
        sample_set (dimod.SampleSet): A sample set.
        device (torch.device, optional): The device of the constructed tensor.
            If ``None`` and data is a tensor then the device of data is used.
            If ``None`` and data is not a tensor then the result tensor is constructed
            on the current device.

    Returns:
        torch.Tensor: The sample set as a ``torch.Tensor``.
    """
    # Need to sort first because this module assumes variables are labelled by integers
    # and ordered as such
    indices = np.argsort(sample_set.variables)
    sample = sample_set.record.sample[:, indices]

    return torch.tensor(sample, dtype=torch.float32, device=device)


def sample(
    grbm: GraphRestrictedBoltzmannMachine,
    sampler: Sampler,
    beta_correction: float,
    device: torch.device = None,
    **sample_params: dict,
) -> torch.Tensor:
    """Sample from the Boltzmann machine.

    This method samples and converts a sample of spins to tensors and ensures they
    are not aggregated---provided the aggregation information is retained in the
    sample set.

    Args:
        sampler (Sampler): The sampler used to sample from the model.
        sampler_params (dict): Parameters of the `sampler.sample` method.
        device (torch.device, optional): The device of the constructed tensor.
            If ``None`` and data is a tensor then the device of data is used.
            If ``None`` and data is not a tensor then the result tensor is
            constructed on the current device.

    Returns:
        torch.Tensor: Spins sampled from the model
            (shape prescribed by ``sampler`` and ``sample_params``).
    """
    h, J = grbm.ising(beta_correction)
    ss = spread(sampler.sample_ising(h, J, **sample_params))
    spins = sample_to_tensor(ss, device=device)
    return spins


def grbm_objective(
    grbm: GraphRestrictedBoltzmannMachine,
    s_observed: torch.Tensor,
    s_model: torch.Tensor,
) -> torch.Tensor:
    """An objective function with gradients equivalent to the gradients of the
    negative log likelihood.

    Args:
        s_observed (torch.Tensor): Tensor of observed spins (data) with shape
            (b1, N) where b1 denotes the batch size and N denotes the number of
            variables in the model.
        s_model (torch.Tensor): Tensor of spins drawn from the model with shape
            (b2, N) where b2 denotes the batch size and N denotse the number of
            variables in the model.

    Returns:
        torch.Tensor: Scalar difference of the average energy of data and model.
    """
    return (
        grbm.sufficient_statistics(s_observed).mean(0, True)
        - grbm.sufficient_statistics(s_model).mean(0, True)
    ) @ grbm.theta
