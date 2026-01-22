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

from typing import TYPE_CHECKING, Any

import torch
from dimod import Sampler
from hybrid.composers import AggregatedSamples

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine
from dwave.plugins.torch.samplers._base import TorchSampler
from dwave.plugins.torch.utils import sampleset_to_tensor

if TYPE_CHECKING:
    import dimod

    from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine


__all__ = ["DimodSampler"]


class DimodSampler(TorchSampler):
    """PyTorch plugin wrapper for a dimod sampler.

    Args:
        module (GraphRestrictedBoltzmannMachine): GraphRestrictedBoltzmannMachine module. Requires the
            methods ``to_ising`` and ``nodes``.
        sampler (dimod.Sampler): Dimod sampler.
        prefactor (float): The prefactor for which the Hamiltonian is scaled by.
            This quantity is typically the temperature at which the sampler operates
            at. Standard CPU-based samplers such as Metropolis- or Gibbs-based
            samplers will often default to sampling at an unit temperature, thus a
            unit prefactor should be used. In the case of a quantum annealer, a
            reasonable choice of a prefactor is 1/beta where beta is the effective
            inverse temperature and can be estimated using
            :meth:`GraphRestrictedBoltzmannMachine.estimate_beta`.
        linear_range (tuple[float, float], optional): Linear weights are clipped to
            ``linear_range`` prior to sampling. This clipping occurs after the ``prefactor``
            scaling has been applied. When None, no clipping is applied. Defaults to None.
        quadratic_range (tuple[float, float], optional): Quadratic weights are clipped to
            ``quadratic_range`` prior to sampling. This clipping occurs after the ``prefactor``
            scaling has been applied. When None, no clipping is applied.Defaults to None.
        sample_kwargs (dict[str, Any]): Dictionary containing optional arguments for the dimod sampler.
    """

    def __init__(
            self,
            grbm: GraphRestrictedBoltzmannMachine,
            sampler: dimod.Sampler,
            prefactor: float | None = None,
            linear_range: tuple[float, float] | None = None,
            quadratic_range: tuple[float, float] | None = None,
            sample_kwargs: dict[str, Any] | None = None
    ) -> None:
        self._grbm = grbm

        # use default prefactor value of 1
        self._prefactor = prefactor or 1

        self._linear_range = linear_range
        self._quadratic_range = quadratic_range

        self._sampler = sampler
        self._sampler_params = sample_kwargs or {}

        # cached sample_set from latest sample
        self._sample_set = None

        # adds all torch parameters to 'self._parameters' for automatic device/dtype
        # update support unless 'refresh_parameters = False'
        super().__init__()

    def sample(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Sample from the dimod sampler and return the corresponding tensor.

        The sample set returned from the latest sample call is stored in :func:`DimodSampler.sample_set`
        which is overwritten by subsequent calls.

        Args:
            x (torch.Tensor): A tensor of shape (``batch_size``, ``dim``) or (``batch_size``, ``n_nodes``)
                interpreted as a batch of partially-observed spins. Entries marked with ``torch.nan`` will
                be sampled; entries with +/-1 values will remain constant.
        """
        if x is not None:
            raise NotImplementedError("Support for conditional sampling has not been implemented.")
        h, J = self._grbm.to_ising(self._prefactor, self._linear_range, self._quadratic_range)
        self._sample_set = AggregatedSamples.spread(
            self._sampler.sample_ising(h, J, **self._sampler_params)
        )

        # use same device as modules linear
        device = self._grbm._linear.device
        return sampleset_to_tensor(self._grbm.nodes, self._sample_set, device)

    @property
    def sample_set(self) -> dimod.SampleSet:
        """The sample set returned from the latest sample call."""
        if self._sample_set is None:
            raise AttributeError("no samples found; call 'sample()' first")

        return self._sample_set
