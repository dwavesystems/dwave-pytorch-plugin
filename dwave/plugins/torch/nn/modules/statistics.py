# Copyright 2026 D-Wave
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
"""Statistics of a set of samples, for the :class:`~dwave.plugins.torch.nn.Ising` layer.

A statistic is a :class:`torch.nn.Module` that maps a set of samples of shape ``(..., M, N)``, and
any further inputs, to values of shape ``(..., D)``, and is symmetric in the ``M`` samples. It may
provide ``leave_one_out(spins, *inputs)``, the values of the statistic on every set of all samples
but one, of shape ``(..., M, D)``; :func:`~dwave.plugins.torch.nn.functional.expectation` uses it
to attach an unbiased gradient of the statistic's expectation to its value, and evaluates the
statistic once per sample when it is absent.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from dwave.plugins.torch.kernels import (
    _check_soft_spins,
    _off_diagonal_distances,
    _validate_alpha,
    kernel_means,
)

__all__ = ["Mean", "SquaredMMD"]


class Mean(torch.nn.Module):
    r"""The sample mean of a per-sample transform,
    :math:`F(s_1, \dots, s_M) = \frac1M \sum_m g(s_m)`.

    With this statistic the :class:`~dwave.plugins.torch.nn.Ising` layer estimates the expectation
    :math:`\mathbb E[g(S)]` of the transform; with the identity transform, the default, it
    estimates the mean spins, and ``Mean(graph.statistics)`` estimates the sufficient statistics of
    the Ising model on ``graph``. The leave-one-out values are closed form, so the gradient of the
    layer costs nothing beyond the transform; the resulting weights are the centred transforms
    divided by ``M - 1``, i.e. the gradient is the unbiased sample covariance of the transform
    with the sufficient statistics.

    Args:
        transform (Callable, optional): Maps spins of shape ``(..., M, N)``, and any further inputs
            of the statistic, to values of shape ``(..., M, D)`` sample by sample. A
            :class:`torch.nn.Module` is registered as a submodule and its parameters receive
            gradients through the layer. If ``None``, the identity. Defaults to ``None``.

    Attributes:
        transform (Callable | None): The transform.
    """

    def __init__(self, transform: Callable[..., torch.Tensor] | None = None) -> None:
        super().__init__()
        self.transform = transform

    def per_sample(self, spins: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        """The transform of every sample, shape ``(..., M, D)``."""
        return spins if self.transform is None else self.transform(spins, *inputs)

    def forward(self, spins: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        """The mean over the samples of the transform, shape ``(..., D)``."""
        return self.per_sample(spins, *inputs).mean(-2)

    def leave_one_out(self, spins: torch.Tensor, *inputs: torch.Tensor) -> torch.Tensor:
        """The mean over all samples but one, for every sample, shape ``(..., M, D)``."""
        values = self.per_sample(spins, *inputs)
        return (values.sum(-2, keepdim=True) - values) / (values.shape[-2] - 1)


class SquaredMMD(torch.nn.Module):
    r"""The unbiased estimate of the squared maximum mean discrepancy between the samples and a
    reference sample, under every kernel of the battery.

    For samples :math:`s_1, \dots, s_M` and a reference sample :math:`z_1, \dots, z_R`,

    .. math::

        F(s; z) = \frac{1}{M(M-1)} \sum_{m \ne m'} k(s_m, s_{m'})
                - \frac{2}{MR} \sum_{m, r} k(s_m, z_r)
                + \frac{1}{R(R-1)} \sum_{r \ne r'} k(z_r, z_{r'}),

    the estimator of :func:`~dwave.plugins.torch.nn.functional.maximum_mean_discrepancy_loss`,
    here for batches of sample sets and for every kernel :math:`k` of the Hamming-isotropic
    battery of :mod:`dwave.plugins.torch.kernels`: the value has ``D = N + 1`` entries, entry 0
    the headline kernel (the Gaussian kernel :math:`e^{-d_H / \sqrt N}`, or the alpha row when
    kernel weights are held) and entry :math:`k = 1..N` the order-:math:`k` kernel. As a function
    of the samples each entry is a U-statistic of degree two with a fixed kernel, so the
    leave-one-out gradient of the :class:`~dwave.plugins.torch.nn.Ising` layer is unbiased for the
    gradient of its expectation, :math:`-2\,\mathrm{Cov}(w(S), T(S))` with the witness
    :math:`w(s) = \mathbb E_{s'}[k(s, s')] - \frac1R \sum_r k(s, z_r)` and the sufficient
    statistics :math:`T`. The reference sample receives its gradient through the kernel, which is
    evaluated at the real distance :math:`(N - \langle s, z \rangle) / 2` and is therefore smooth
    in a soft reference (an encoder's ``tanh`` output, or straight-through spins), so an encoder
    producing ``z`` and a Boltzmann machine producing the samples are trained by one loss, for
    example ``expectation(prior, h, J, samples, SquaredMMD(), z)[..., 0].sum().backward()``.

    The leave-one-out values are closed form in the row means of the kernel matrices and cost
    nothing beyond the value; the order kernels of all rows are summed in one sweep of
    :class:`~dwave.plugins.torch.kernels.KravchukSums`. The kernel weights are a buffer, so they
    move with :meth:`~torch.nn.Module.to` and are saved in :meth:`~torch.nn.Module.state_dict`.

    Args:
        alpha (torch.Tensor, optional): Kernel weights over orders ``1..N``, a 1-D tensor of
            finite nonnegative entries with at least one positive, for example the weights an
            :class:`~dwave.plugins.torch.evaluation.Evaluator` calibrated. Defaults to ``None``,
            the Gaussian headline.

    Attributes:
        alpha (torch.Tensor | None): The kernel weights, a buffer, or ``None``.

    Raises:
        ValueError: If ``alpha`` is not a 1-D tensor of finite nonnegative entries with at least
            one positive.
    """

    alpha: torch.Tensor | None

    def __init__(self, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        if alpha is not None:
            _validate_alpha(alpha)
            alpha = alpha.clone()
        self.register_buffer("alpha", alpha)

    def extra_repr(self) -> str:
        return "headline=gaussian" if self.alpha is None else f"alpha of {self.alpha.shape[0]} orders"

    def _row_means(
        self, spins: torch.Tensor, reference: torch.Tensor, min_samples: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Row means of the kernel matrices, per profile.

        Args:
            spins (torch.Tensor): Samples of shape ``(..., M, N)``.
            reference (torch.Tensor): Reference sample of shape ``(..., R, N)`` or ``(R, N)``.
            min_samples (int): The least number of samples ``M`` that the caller requires.

        Raises:
            ValueError: If there are fewer than ``min_samples`` samples or fewer than two reference
                items, the spin dimensions differ, or the inputs are not finite soft spins.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The means over the other samples of
            the kernel with every sample, shape ``(D, ..., M)``; the means over the reference of
            the kernel with every sample, shape ``(D, ..., M)``; and the mean kernel between
            distinct reference items, shape ``(D, ...)``; profile-indexed on the first axis.
        """
        _check_soft_spins(spins, "spins")
        _check_soft_spins(reference, "reference")
        n_samples, n_nodes = spins.shape[-2:]
        n_reference = reference.shape[-2]
        if reference.shape[-1] != n_nodes:
            raise ValueError(
                f"The reference sample must have {n_nodes} spins per item like the samples, got "
                f"shape {tuple(reference.shape)}."
            )
        if n_samples < min_samples:
            raise ValueError(f"At least {min_samples} samples are required, got {n_samples}.")
        if n_reference < 2:
            raise ValueError(f"At least two reference items are required, got {n_reference}.")
        if self.alpha is not None:
            _validate_alpha(self.alpha, n_nodes)
        reference = reference.expand(*spins.shape[:-2], *reference.shape[-2:])
        between = kernel_means(_off_diagonal_distances(spins, n_nodes), n_nodes, self.alpha)
        to_reference = kernel_means((n_nodes - spins @ reference.mT) / 2, n_nodes, self.alpha)
        within_reference = kernel_means(
            _off_diagonal_distances(reference, n_nodes).flatten(-2), n_nodes, self.alpha
        )
        return between, to_reference, within_reference

    def forward(self, spins: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """The squared maximum mean discrepancy estimates, shape ``(..., N + 1)``.

        Args:
            spins (torch.Tensor): Samples of shape ``(..., M, N)``, at least two per set.
            reference (torch.Tensor): Reference sample of shape ``(..., R, N)`` or ``(R, N)``, at
                least two items, hard or soft spins in ``[-1, 1]``.

        Returns:
            torch.Tensor: The estimates, entry 0 the headline kernel and entry ``k`` the
            order-``k`` kernel.
        """
        between, to_reference, within = self._row_means(spins, reference, min_samples=2)
        value = between.mean(-1) - 2 * to_reference.mean(-1) + within
        return value.movedim(0, -1)

    def leave_one_out(self, spins: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """The estimates on every set of all samples but one, shape ``(..., M, N + 1)``; requires
        at least three samples per set."""
        between, to_reference, within = self._row_means(spins, reference, min_samples=3)
        n_samples = spins.shape[-2]
        # The within-sample mean over the other M - 1 samples: all ordered pairs minus the pairs
        # involving the left-out sample, over (M - 1)(M - 2) pairs
        between_others = (between.sum(-1, keepdim=True) - 2 * between) / (n_samples - 2)
        to_reference_others = (
            n_samples * to_reference.mean(-1, keepdim=True) - to_reference
        ) / (n_samples - 1)
        values = between_others - 2 * to_reference_others + within.unsqueeze(-1)
        return values.movedim(0, -1)
