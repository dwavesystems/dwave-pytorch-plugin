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
"""Hamming-isotropic kernels on spins, as profile tables and as differentiable sums.

A Hamming-isotropic kernel :math:`\\kappa(x, y) = f(d_H(x, y))` depends on two spin vectors
only through their Hamming distance :math:`d_H` (the number of coordinates where they differ),
so one *profile* :math:`f`, a value per distance ``0..dim``, describes it completely. This module
builds the profiles of the kernel battery behind the maximum mean discrepancy (MMD) of this
package and evaluates them at real distances for differentiable losses:

- :func:`kernel_profiles` is the ``(dim + 1, dim + 1)`` table of the fixed battery: row 0 the
  Gaussian kernel :math:`e^{-d / \\sqrt{dim}}`, row :math:`k` the order-:math:`k` kernel
  :math:`K_k(d) / \\binom{dim}{k}`, with :math:`K_k` the degree-:math:`k` Kravchuk polynomial,
  i.e. the average parity agreement over the :math:`\\binom{dim}{k}` parities of order :math:`k`.
- :func:`uniform_kernel_mean` is the mean of a kernel against the uniform distribution, its
  order-0 Walsh eigenvalue :math:`\\lambda_0`, in closed form.
- :class:`KravchukSums` is a :class:`torch.autograd.Function` returning
  :math:`\\sum_e R_k(d_e)` for every order :math:`k` at real distances :math:`d_e`, with a
  hand-written backward pass whose memory does not grow with ``dim``, and
  :func:`kernel_means` the mean kernel value per profile over a set of real distances, the
  building block of :func:`~dwave.plugins.torch.nn.functional.maximum_mean_discrepancy_loss`
  and :class:`~dwave.plugins.torch.nn.SquaredMMD`.

Everything is computed in the dtype and on the device the caller asks for: float64 reproduces
each exact profile to about ``2e-14`` per entry at any dimension, float32 to about ``5e-7`` up
to ``dim = 200`` and about ``2e-5`` at ``dim = 4800``. The definitions, derivations and
references are in ``docs/evaluation.md``.
"""

from __future__ import annotations

import math
from typing import Any

import torch

__all__ = ["KravchukSums", "kernel_means", "kernel_profiles", "uniform_kernel_mean"]

# A torch device specification: a ``torch.device``, its name, or ``None`` for the default device.
Device = torch.device | str | None


def _log_binomials(
    dim: int, dtype: torch.dtype = torch.float64, device: Device = None
) -> torch.Tensor:
    """Natural logarithms of the binomial coefficients ``C(dim, k)`` for ``k = 0..dim``.

    Formed as ``lgamma(dim + 1) - lgamma(k + 1) - lgamma(dim - k + 1)`` in float64 and cast to
    ``dtype`` last: the three terms are of size ``dim log dim`` and cancel to a result of size at
    most ``dim log 2``, so a float32 sweep would lose about ``dim log dim * 2^-24`` absolute
    (``1e-3`` relative in the binomials at ``dim = 4800``), while float64 keeps every value
    within a few ``1e-12`` relative, and finite at any ``dim``, where ``C(dim, k)`` itself
    overflows past ``dim ~ 1030``.
    """
    k = torch.arange(dim + 1, dtype=torch.float64, device=device)
    logs = math.lgamma(dim + 1) - torch.lgamma(k + 1) - torch.lgamma(dim - k + 1)
    return logs.to(dtype)


def _binomial_pmf(
    dim: int, dtype: torch.dtype = torch.float64, device: Device = None
) -> torch.Tensor:
    """Binomial(``dim``, 1/2) probabilities of the distances ``d = 0..dim``, in ``dtype``.

    Exactly ``C(dim, d)`` of the ``2^dim`` spin vectors lie at Hamming distance ``d`` from any
    fixed one, so this is the distance law between a fixed point and a uniform draw, i.e. the
    weights of :func:`uniform_kernel_mean`. Formed in float64 log space as
    ``log C(dim, d) - dim log 2``, the logarithm of a probability and hence ``<= 0``: the
    exponential never overflows at any ``dim``, and tail entries below the float range
    underflow to exactly ``0``, which is what the weighted sum needs. Cast to ``dtype`` last, so
    a float32 table receives the correctly rounded law.
    """
    log_pmf = _log_binomials(dim, torch.float64, device) - dim * math.log(2.0)
    return log_pmf.exp().to(dtype)


def _normalized_kravchuk_matrix(
    dim: int, dtype: torch.dtype = torch.float64, device: Device = None
) -> torch.Tensor:
    """The ``(dim + 1, dim + 1)`` table ``R_k(d) = K_k(d) / C(dim, k)``, indexed ``[k, d]``.

    ``K_k(d) = sum_j (-1)^j C(d, j) C(dim - d, k - j)`` is the degree-``k`` Kravchuk
    polynomial: the sum of the parity agreements ``chi_S(x) chi_S(y)`` over the ``C(dim, k)``
    index sets ``S`` of size ``k`` at Hamming distance ``d``. ``R_k`` is their average, so every
    entry lies in ``[-1, 1]`` and no order overflows (the raw ``K_k`` exceed the float range
    past ``C(dim, k) ~ 1e308``). Rows ``0..dim // 2`` run the normalized three-term recurrence
    ``R_{k+1}(d) = ((dim - 2d) R_k(d) - k R_{k-1}(d)) / (dim - k)`` from ``R_0 = 1`` and
    ``R_1(d) = (dim - 2d) / dim``, which is numerically stable while ``C(dim, k)`` still grows.
    The rows above ``dim // 2`` are the exact reflection ``R_{dim-k}(d) = (-1)^d R_k(d)`` of the
    swept rows (complementing every index set), because the recurrence run past the middle
    divides rounding noise by a vanishing band size and diverges. Float64 entries are within
    about ``2e-14`` of the exact rationals; the sweep runs in ``dtype`` on ``device``.
    """
    distances = torch.arange(dim + 1, dtype=dtype, device=device)
    k1 = dim - 2.0 * distances  # (dim - 2d), the raw K_1
    mtx = torch.empty((dim + 1, dim + 1), dtype=dtype, device=device)
    half = dim // 2  # the sweep's last degree; every row above it is a reflection
    mtx[0] = 1.0  # R_0; replaced by the Gaussian profile in kernel_profiles
    if half >= 1:
        mtx[1] = k1 / dim
    for k in range(1, half):
        mtx[k + 1] = (k1 * mtx[k] - k * mtx[k - 1]) / (dim - k)
    parity_sign = 1.0 - 2.0 * (distances % 2)  # (-1)^d, the reflection twist
    mtx[half + 1 :] = parity_sign * mtx[: dim - half].flip(0)  # R_k = (-1)^d R_{dim-k}
    return mtx


def kernel_profiles(
    dim: int, dtype: torch.dtype = torch.float64, device: Device = None
) -> torch.Tensor:
    r"""The fixed kernel battery at dimension ``dim`` as a ``(dim + 1, dim + 1)`` profile table.

    Row ``j`` is the profile of battery entry ``j`` and column ``d`` its value at Hamming
    distance ``d``:

    - Row 0 is the Gaussian kernel :math:`e^{-d / \sqrt{dim}}`. On spins
      :math:`d_H = \|x - y\|^2 / 4`, so this is the Gaussian radial-basis-function kernel at
      bandwidth :math:`\sigma = \sqrt{dim}`, the scale on which the Hamming distance of
      independent coordinates concentrates (Hoeffding); a bandwidth that tracks it keeps the
      kernel non-degenerate as ``dim`` grows. It is strictly proper: its Walsh eigenvalues are
      positive at every order and decay geometrically with ratio
      :math:`\tanh(1 / (2 \sqrt{dim}))`.
    - Row ``k = 1..dim`` is the order-``k`` kernel :math:`K_k(d) / \binom{dim}{k}`: the average
      parity agreement over the order-``k`` band (unit diagonal, values in ``[-1, 1]``), whose
      squared maximum mean discrepancy (MMD) is the mean squared error per order-``k`` Walsh
      coefficient, the order-``k`` *Fourier weight* of the mismatch. Proper, but blind off its
      band.

    An :class:`~dwave.plugins.torch.evaluation.Evaluator` swaps its alpha row in at row 0 while
    it holds kernel weights; any ``(K, dim + 1)`` stack of rows (these, another bandwidth, a
    synthesized spectrum) can be handed to
    :func:`~dwave.plugins.torch.evaluation.blocked_scores`.

    Args:
        dim (int): The number of spins, ``>= 1``.
        dtype (torch.dtype): The floating dtype the table is swept and stored in. Defaults to
            ``torch.float64``, which is exact to about ``2e-14`` per entry at any dimension;
            float32 to about ``5e-7`` up to ``dim = 200`` and about ``2e-5`` at ``dim = 4800``.
        device (torch.device | str, optional): The device the table is allocated on. Defaults
            to ``None``, torch's default device.

    Raises:
        ValueError: If ``dim`` is below 1.

    Returns:
        torch.Tensor: The ``(dim + 1, dim + 1)`` profile table in ``dtype`` on ``device``.
    """
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim!r}")
    table = _normalized_kravchuk_matrix(dim, dtype, device)  # row k = the order-k profile
    distances = torch.arange(dim + 1, dtype=dtype, device=device)
    table[0] = torch.exp(-distances / math.sqrt(dim))  # the Gaussian replaces the constant R_0
    return table


def uniform_kernel_mean(profiles: torch.Tensor) -> torch.Tensor:
    r"""The kernel's mean against the uniform distribution,
    :math:`\sum_d \binom{dim}{d} 2^{-dim} f(d)`.

    For a Hamming-isotropic kernel with profile :math:`f` the mean
    :math:`\mathbb E\,\kappa(x, Y)` over a uniform :math:`Y` is the same number
    :math:`\lambda_0` for every :math:`x`, the kernel's order-0 Walsh eigenvalue, so it is also
    :math:`\mathbb E\,\kappa(Y, Y')` for two uniform draws, and the whole of the uniform
    distribution's contribution to :math:`\mathrm{MMD}^2(P, U) = \mathbb E_P\,\kappa(X, X') -
    \lambda_0`. The sum weights the profile by the Binomial(``dim``, 1/2) distance law, in the
    profile's dtype on its device. Closed forms it reproduces to float tolerance: ``0`` for every
    order-``k`` row (a band of order ``k >= 1`` has no order-0 component) and
    :math:`((1 + e^{-1 / \sqrt{dim}}) / 2)^{dim}` for the Gaussian row.

    Args:
        profiles (torch.Tensor): One profile (a 1-D tensor of ``dim + 1`` values) or a stack of
            them (a 2-D ``(K, dim + 1)`` table, one profile per row).

    Raises:
        ValueError: If ``profiles`` is not 1-D or 2-D with at least two columns.

    Returns:
        torch.Tensor: A 0-d tensor for one profile, a ``(K,)`` tensor for a stack.
    """
    if profiles.ndim not in (1, 2) or profiles.shape[-1] < 2:
        raise ValueError(
            "profiles must be a 1-D or 2-D table of dim + 1 >= 2 columns, got "
            f"{tuple(profiles.shape)}"
        )
    dim = profiles.shape[-1] - 1
    return profiles @ _binomial_pmf(dim, profiles.dtype, profiles.device)


def _validate_alpha(alpha: torch.Tensor, dim: int | None = None) -> None:
    """Raise unless ``alpha`` is a 1-D tensor of finite, nonnegative kernel weights with at least
    one positive entry, and one entry per order ``1..dim`` when ``dim`` is given."""
    if not isinstance(alpha, torch.Tensor) or alpha.ndim != 1:
        raise ValueError(
            "alpha must be a 1-D tensor of one weight per order, got "
            f"{tuple(alpha.shape) if isinstance(alpha, torch.Tensor) else type(alpha)}"
        )
    if dim is not None and alpha.shape[0] != dim:
        raise ValueError(f"alpha must hold one weight per order 1..{dim}, got {alpha.shape[0]}")
    if alpha.numel() == 0:
        raise ValueError("alpha must hold at least one weight")
    if not bool(torch.isfinite(alpha).all()):
        raise ValueError("alpha must be finite")
    if bool((alpha < 0).any()) or not bool((alpha > 0).any()):
        raise ValueError("alpha must be nonnegative with at least one positive entry")


# --------------------------------------------------------------------------------------
# Kernel profiles at real distances: the differentiable path behind the losses
# --------------------------------------------------------------------------------------


def _twist(d: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``cos(pi d)`` and ``-pi sin(pi d)``, the reflection's twist and its derivative, exact at
    integer distances in every dtype.

    The angle is reduced to the nearest integer first, ``cos(pi d) = (-1)^m cos(pi (d - m))`` and
    ``sin(pi d) = (-1)^m sin(pi (d - m))`` with ``m = round(d)``: ``d - m`` is exact in floating
    point, so at an integer distance the sine is exactly zero and the cosine exactly ``+-1``,
    where ``torch.sin(math.pi * d)`` in float32 would carry an error of about ``3e-7 d`` and
    contaminate the gradient of the reflected rows at hard spins.
    """
    nearest = d.round()
    sign = 1.0 - 2.0 * (nearest % 2)  # (-1)^m
    angle = math.pi * (d - nearest)
    return sign * torch.cos(angle), -math.pi * sign * torch.sin(angle)


class KravchukSums(torch.autograd.Function):
    r"""Sums of the order kernels over real distances, with a gradient in ``O(E)`` memory.

    ``KravchukSums.apply(d, dim)`` maps a tensor ``d`` of real distances, of shape ``(..., E)``,
    to the ``dim + 1`` sums over its last axis, :math:`S_k = \sum_e F_k(d_e)` for
    :math:`k = 0..dim`, of shape ``(dim + 1, ...)``. Here :math:`F_k` is the function the kernel
    table evaluates in row :math:`k`, extended to real distances: the normalized Kravchuk
    polynomial :math:`R_k` for :math:`k \le \lfloor dim / 2 \rfloor`, run by the same three-term
    recurrence as :func:`kernel_profiles`, and the smooth reflection
    :math:`\cos(\pi d)\, R_{dim-k}(d)` above it, which equals :math:`R_k` at every integer
    distance (:math:`\cos(\pi d)` is :math:`(-1)^d` there). At hard spins the sums therefore
    reproduce the table exactly; at soft spins, a real distance :math:`d = (dim - \langle x, y
    \rangle) / 2` from a Gram of entries in ``[-1, 1]``, they are a smooth surrogate.
    :math:`S_k / E` is the mean order-:math:`k` kernel value over the pairs, which
    :func:`kernel_means` returns.

    Plain autograd through the recurrence would keep one row of ``E`` entries per order alive
    between forward and backward, ``(dim + 1) E`` floats; this function saves only ``d`` and
    recomputes the rows in the backward sweep, carrying their derivatives alongside, so both
    passes cost ``O(E dim)`` time and a handful of tensors of ``d``'s size. The backward pass is
    the exact vector-Jacobian product of the functions the forward evaluates (``docs/evaluation.md``
    proves it; the tests check it against finite differences and against autograd through the
    unrolled recurrence). ``dim`` receives no gradient.
    """

    @staticmethod
    def forward(ctx: Any, d: torch.Tensor, dim: int) -> torch.Tensor:
        """Sweep the recurrence once, writing each row's sum and its reflection's as produced.

        Args:
            ctx: The autograd context; ``d`` is saved on it for the backward pass, and ``dim``.
            d (torch.Tensor): Real Hamming distances in ``[0, dim]`` of shape ``(..., E)``
                (flattened Gram entries, typically), in the batch's dtype on its device.
            dim (int): The number of spins.

        Returns:
            torch.Tensor: The sums ``S_0..S_dim`` over the last axis of ``d``, of shape
            ``(dim + 1, ...)`` in ``d``'s dtype on its device.
        """
        ctx.save_for_backward(d)
        ctx.dim = dim
        half = dim // 2
        n = float(dim)
        sums = d.new_empty((dim + 1, *d.shape[:-1]))
        twist, _ = _twist(d)  # the reflection's (-1)^d, at real d
        k1 = n - 2.0 * d
        prev = torch.ones_like(d)  # R_0
        sums[0] = prev.sum(-1)
        if dim > half:  # row dim is the reflection of row 0
            sums[dim] = (twist * prev).sum(-1)
        if half >= 1:
            cur = k1 / n  # R_1
            sums[1] = cur.sum(-1)
            if dim - 1 > half:  # false at dim = 2 only (not reached at dim = 1)
                sums[dim - 1] = (twist * cur).sum(-1)
            for k in range(1, half):  # R_{k+1} from R_k and R_{k-1}
                nxt = (k1 * cur - k * prev) / (n - k)
                sums[k + 1] = nxt.sum(-1)
                if dim - k - 1 > half:
                    sums[dim - k - 1] = (twist * nxt).sum(-1)
                prev, cur = cur, nxt
        return sums

    @staticmethod
    def backward(ctx: Any, grad_sums: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Accumulate ``sum_k g_k F_k'(d_e)`` by re-sweeping the recurrence with derivatives.

        The derivative recurrence is
        ``R'_{k+1} = ((dim - 2d) R'_k - 2 R_k - k R'_{k-1}) / (dim - k)`` from ``R'_0 = 0`` and
        ``R'_1 = -2 / dim``; a reflected row contributes ``cos(pi d) R'_k - pi sin(pi d) R_k``
        by the product rule.

        Args:
            ctx: The autograd context holding ``d`` and ``dim``.
            grad_sums (torch.Tensor): The upstream gradient ``g`` of shape ``(dim + 1, ...)``,
                one entry per sum.

        Returns:
            tuple[torch.Tensor, None]: The gradient with respect to ``d`` (its shape) and
            ``None`` for ``dim``.
        """
        (d,) = ctx.saved_tensors
        dim = ctx.dim
        half = dim // 2
        n = float(dim)
        twist, twist_d = _twist(d)
        k1 = n - 2.0 * d
        g = grad_sums.unsqueeze(-1)  # (dim + 1, ..., 1): one upstream scalar per sum
        grad_d = torch.zeros_like(d)

        def accumulate(k: int, row: torch.Tensor, row_d: torch.Tensor) -> None:
            """Add row ``k``'s term and, when it exists, its reflection's into the gradient."""
            nonlocal grad_d
            grad_d = grad_d + g[k] * row_d
            if dim - k > half:  # row dim - k is this row's reflection
                grad_d = grad_d + g[dim - k] * (twist * row_d + twist_d * row)

        prev, prev_d = torch.ones_like(d), torch.zeros_like(d)  # R_0, R'_0
        accumulate(0, prev, prev_d)
        if half >= 1:
            cur, cur_d = k1 / n, torch.full_like(d, -2.0 / n)  # R_1, R'_1
            accumulate(1, cur, cur_d)
            for k in range(1, half):
                nxt = (k1 * cur - k * prev) / (n - k)
                nxt_d = (k1 * cur_d - 2.0 * cur - k * prev_d) / (n - k)
                accumulate(k + 1, nxt, nxt_d)
                prev, prev_d, cur, cur_d = cur, cur_d, nxt, nxt_d
        return grad_d, None


def kernel_means(
    distances: torch.Tensor, dim: int, alpha: torch.Tensor | None = None
) -> torch.Tensor:
    r"""Mean kernel value over the last axis of real distances, for every profile of the battery.

    Entry ``k = 1..dim`` of the result is the mean of the order-``k`` kernel over the distances,
    :math:`\frac1E \sum_e F_k(d_e)` (see :class:`KravchukSums`), and entry 0 is the headline
    kernel: the mean of the Gaussian kernel :math:`e^{-d / \sqrt{dim}}`, or, when kernel weights
    ``alpha`` are given, the alpha row :math:`\sum_k \alpha_k \cdot` (order-``k`` mean). At
    integer distances every entry is the mean of the corresponding row of
    :func:`kernel_profiles`; at real distances, the smooth extension. Differentiable in
    ``distances``, in ``O(E)`` memory. Nothing is cast: ``alpha`` must share the dtype and device
    of ``distances``.

    Args:
        distances (torch.Tensor): Real Hamming distances in ``[0, dim]`` of shape ``(..., E)``
            with ``E >= 1``, for example the entries ``(dim - <x, y>) / 2`` of a Gram of soft
            spins.
        dim (int): The number of spins.
        alpha (torch.Tensor, optional): Kernel weights over orders ``1..dim``, a 1-D tensor of
            ``dim`` finite nonnegative entries with at least one positive. Defaults to ``None``,
            the Gaussian headline.

    Raises:
        ValueError: If ``alpha`` is given and invalid.

    Returns:
        torch.Tensor: The ``(dim + 1, ...)`` means, entry 0 the headline and entry ``k`` the
        order-``k`` kernel.
    """
    if alpha is not None:
        _validate_alpha(alpha, dim)
    count = distances.shape[-1]
    means = KravchukSums.apply(distances, dim) / count  # entry k; [0] is R_0 = 1
    if alpha is None:
        head = torch.exp(-distances / math.sqrt(dim)).sum(-1) / count
    else:
        head = torch.einsum("k,k...->...", alpha, means[1:])
    return torch.cat([head[None], means[1:]])


def _check_soft_spins(tensor: torch.Tensor, name: str) -> None:
    """Raise unless ``tensor`` is a floating tensor of at least two dimensions whose entries are
    finite and lie in ``[-1, 1]``: hard spins, or soft spins such as a ``tanh`` output."""
    if tensor.ndim < 2:
        raise ValueError(
            f"{name} must have shape (..., n, dim) with one row per item, got "
            f"{tuple(tensor.shape)}."
        )
    if not tensor.is_floating_point():
        raise ValueError(
            f"{name} must be soft spins in [-1, 1] of a floating dtype, got {tensor.dtype}."
        )
    values = tensor.detach()
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must hold finite soft spins in [-1, 1].")
    if bool((values.abs() > 1).any()):
        raise ValueError(f"{name} must hold soft spins in [-1, 1].")


def _off_diagonal_distances(z: torch.Tensor, dim: int) -> torch.Tensor:
    """The real Hamming distances ``(dim - <z_i, z_j>) / 2`` between distinct rows of ``z``.

    Args:
        z (torch.Tensor): Soft spins of shape ``(..., n, dim)``.
        dim (int): The number of spins.

    Returns:
        torch.Tensor: The distances of the ``n (n - 1)`` ordered off-diagonal pairs, of shape
        ``(..., n, n - 1)``, row ``i`` holding the distances of row ``i`` to every other row.
    """
    n = z.shape[-2]
    off_diagonal = ~torch.eye(n, dtype=torch.bool, device=z.device)
    distances = (dim - z @ z.mT) / 2
    return distances[..., off_diagonal].reshape(*z.shape[:-2], n, n - 1)
