from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from dimod import SampleSet


def sampleset_to_tensor(
        ordered_vars: list, sample_set: SampleSet, device: torch.device = None) -> torch.Tensor:
    """Converts a ``dimod.SampleSet`` to a ``torch.Tensor``.

    Args:
        ordered_vars: list[Literal]: The desired order of sample set variables.
        sample_set (dimod.SampleSet): A sample set.
        device (torch.device, optional): The device of the constructed tensor.
            If ``None`` and data is a tensor then the device of data is used.
            If ``None`` and data is not a tensor then the result tensor is constructed
            on the current device.

    Returns:
        torch.Tensor: The sample set as a ``torch.Tensor``.
    """
    var_to_sample_i = {v: i for i, v in enumerate(sample_set.variables)}
    permutation = [var_to_sample_i[v] for v in ordered_vars]
    sample = sample_set.record.sample[:, permutation]
    return torch.tensor(sample, dtype=torch.float32, device=device)
