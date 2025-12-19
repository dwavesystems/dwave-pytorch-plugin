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

import abc
import copy

import torch

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine


__all__ = ["TorchSampler"]


class TorchSampler(abc.ABC):
    """Base class for all PyTorch plugin samplers."""

    def __init__(self, refresh: bool = True) -> None:
        self._parameters = {}
        self._modules = {}

        if refresh:
            self.refresh_parameters()

    def parameters(self):
        """Parameters in the sampler."""
        for p in self._parameters.values():
            yield p

    def modules(self):
        """Modules in the sampler."""
        for m in self._modules.values():
            yield m

    @abc.abstractmethod
    def sample(self, x: torch.Tensor | None = None) -> torch.Tensor:
        """Abstract sample method."""

    def to(self, *args, **kwargs):
        """Performs Tensor dtype and/or device conversion on sampler parameters.

        See :meth:`torch.Tensor.to` for usage details."""
        # perform a shallow copy of the sampler to be returned
        sampler = copy.copy(self)
        parameters = {}
        modules = {}

        for name, p in self._parameters.items():
            new_p = p.to(*args, **kwargs)

            # set attribute and update parameters
            setattr(sampler, name, new_p)
            parameters[name] = new_p

        for name, m in self._modules.items():
            new_m = m.to(*args, **kwargs)

            # set attribute and update modules
            setattr(sampler, name, new_m)
            modules[name] = new_m

        sampler._parameters = parameters
        sampler._modules = modules

        return sampler

    def refresh_parameters(self, replace=True, clear=True):
        """Refreshes the parameters and modules attributes in-place.

        Searches the sampler for any initialized torch parameters and modules
        and adds them to the :attr:`TorchSampler_parameters` attribute, which
        is used to update device or dtype using the
        :meth:`TorchSampler.to` method.

        Args:
            replace: Replace any previous parameters with new values.
            clear: Clear the parameters attribute before adding new ones.
        """
        if clear:
            self._parameters.clear()
            self._modules.clear()

        for attr_, val in self.__dict__.items():
            # NOTE: Only refreshes torch parameters and modules, but _not_ any
            # GRBM models. Can be generalized if plugin gets a baseclass module.
            if replace or attr_ not in self._parameters:
                if isinstance(val, torch.Tensor):
                    self._parameters[attr_] = val
                elif (
                    isinstance(val, torch.nn.Module) and
                    not isinstance(val, GraphRestrictedBoltzmannMachine)
                ):
                    self._modules[attr_] = val
