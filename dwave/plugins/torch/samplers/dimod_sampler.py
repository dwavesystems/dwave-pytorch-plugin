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

from typing import TYPE_CHECKING, Any, Optional

import torch
import dimod
import warnings
from hybrid.composers import AggregatedSamples

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine
from dwave.plugins.torch.samplers.base import TorchSampler
from dwave.plugins.torch.utils import sampleset_to_tensor

if TYPE_CHECKING:
    import dimod
    from dimod import SampleSet
    from dwave.plugins.torch.models.boltzmann_machine import (
        GraphRestrictedBoltzmannMachine,
    )


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
        prefactor: float,
        linear_range: tuple[float, float] | None = None,
        quadratic_range: tuple[float, float] | None = None,
        sample_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._grbm = grbm

        self._prefactor = prefactor

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

        The sample set returned from the latest sample call is available via :attr:`DimodSampler.sample_set`
        which is overwritten by subsequent calls.

        Args:
            x (torch.Tensor): A tensor of shape (``batch_size``, ``dim``) or (``batch_size``, ``n_nodes``)
                interpreted as a batch of partially-observed spins. Entries marked with ``torch.nan`` will
                be sampled; entries with +/-1 values will remain constant.
        Raises:
            ValueError: If ``x`` has an invalid shape or contains values other than ±1 or NaN or if the
                sampler returns more than one sample per input row.

        Returns:
            torch.Tensor: Sampled spin configurations with entries in ``{-1, +1}``.
            If ``x is None`` the returned tensor has shape ``(num_reads, n_nodes)``.
            Otherwise, the returned tensor has shape ``(batch_size, num_reads, n_nodes)``.
        """
        device = self._grbm.linear.device
        n_nodes = self._grbm.n_nodes

        h, J = self._grbm.to_ising(self._prefactor, self._linear_range, self._quadratic_range)

        # Unconditional sampling
        if x is None:
            self._sample_set = AggregatedSamples.spread(
                self._sampler.sample_ising(h, J, **self._sampler_params)
            )
            return self._sampleset_to_tensor(self._sample_set, device)

        # Conditional sampling
        if x.shape[1] != n_nodes:
            raise ValueError(f"x must have shape (batch_size, {n_nodes})")

        mask = ~torch.isnan(x)
        if not torch.all(torch.isin(x[mask], torch.tensor([-1, 1], device=device))):
            raise ValueError("x must contain only ±1 or NaN")

        results = []
        for row, row_mask in zip(x, mask):
            # Fresh BQM
            bqm = dimod.BinaryQuadraticModel.from_ising(h, J)

            # Build conditioning dict
            conditioned = {
                node: int(val.item()) for node, val, m in zip(self._grbm.nodes, row, row_mask) if m
            }

            # Apply conditioning
            if conditioned:
                bqm.fix_variables(conditioned)

            # Handle fully clamped case
            if bqm.num_variables == 0:
                num_reads = self._sampler_params.get("num_reads", 1)
                full_read = torch.empty((num_reads, n_nodes), device=device)
                for node, idx in self._grbm.node_to_idx.items():
                    full_read[:, idx] = conditioned[node]
                results.append(full_read)
                continue

            # Clip linear biases for remaining free variables
            if self._linear_range is not None:
                lb, ub = self._linear_range
                for v, bias in bqm.iter_linear():
                    if bias > ub:
                        bqm.set_linear(v, ub)
                    elif bias < lb:
                        bqm.set_linear(v, lb)

            # Clip quadratic biases
            if self._quadratic_range is not None:
                lb, ub = self._quadratic_range
                for u, v, bias in bqm.iter_quadratic():
                    if bias > ub:
                        bqm.set_quadratic(u, v, ub)
                    elif bias < lb:
                        bqm.set_quadratic(u, v, lb)

            # Storing the latest samples
            self._sample_set = AggregatedSamples.spread(
                self._sampler.sample(bqm, **self._sampler_params)
            )
            sample_array = self._sample_set.record.sample

            num_reads = sample_array.shape[0]

            full_read = torch.empty((num_reads, n_nodes), device=device)
            var_to_idx = {v: i for i, v in enumerate(self._sample_set.variables)}

            for node, idx in self._grbm.node_to_idx.items():
                if node in conditioned:
                    full_read[:, idx] = conditioned[node]
                else:
                    full_read[:, idx] = torch.from_numpy(sample_array[:, var_to_idx[node]]).to(
                        device=device, dtype=torch.float
                    )
            results.append(full_read)

        reference_shape = results[0].shape
        if not all(result.shape == reference_shape for result in results):
            raise ValueError(f"Expected all samples to have shape {reference_shape}")
        # Stack to get (batch_size, num_reads, n_nodes)
        samples = torch.stack(results, dim=0)
        return samples

    def _sampleset_to_tensor(self, sample_set: SampleSet, device: Optional[torch.device] = None) -> torch.Tensor:
        """Converts a ``dimod.SampleSet`` to a ``torch.Tensor`` using GRBM node order.

        Args:
            sample_set (dimod.SampleSet): A sample set.
            device (torch.device, optional): The device of the constructed tensor.
                If ``None`` and data is a tensor then the device of data is used.
                If ``None`` and data is not a tensor then the result tensor is constructed
                on the current device.

        Returns:
            torch.Tensor: The sample set as a ``torch.Tensor``.
        """
        var_to_sample_i = {v: i for i, v in enumerate(sample_set.variables)}

        # Convert dict -> ordered list by index
        ordered_vars = [v for v, _ in sorted(self._grbm.node_to_idx.items(), key=lambda x: x[1])]

        permutation = [var_to_sample_i[v] for v in ordered_vars]

        sample = sample_set.record.sample[:, permutation]

        return torch.from_numpy(sample).to(device=device, dtype=torch.float32)

    @property
    def sample_set(self) -> dimod.SampleSet:
        """The sample set returned from the latest sample call."""
        if self._sample_set is None:
            raise AttributeError("no samples found; call 'sample()' first")

        return self._sample_set
