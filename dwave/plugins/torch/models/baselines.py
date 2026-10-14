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
"""Moment-matched baseline models of spins: three models with one ``fit`` / ``sample``
interface, weakest first.

A kernel score (see :mod:`dwave.plugins.torch.evaluation`) measures how far a generative model's
samples sit from the data, but a distance alone cannot say how much of the structure a model
reproduces was *cheap*: the per-spin means and the pairwise correlations of any sample can be
matched by a classical model that knows nothing else. The baselines here are fit to an
``(N, dim)`` matrix of spins, rows whose every entry is ``-1`` or ``+1``, and draw new rows in
the same convention, so a candidate model is read against progressively stronger moment-matched
surrogates: how much of the data's structure survives once the per-spin means are matched, and
how much once the pairwise tables are matched too. Weakest first:

* :class:`UniformBernoulli`: fair coins on every spin, the null model; it ignores the data.
* :class:`IndependentBernoulli`: independent spins at the data's per-spin means (the order-1
  moments).
* :class:`DichotomizedGaussian`: the signs of a latent multivariate normal ``X ~ N(mu, R)``
  with ``R`` a correlation matrix, matched exactly to the data's means and pairwise ``2x2``
  tables (the order-1 and order-2 moments): ``mu_i = Phi^-1(p_i)`` from the smoothed
  frequencies of ``+1``, and per pair the latent correlation ``rho_ij`` that reproduces the
  smoothed table, ``Phi_2(mu_i, mu_j; rho_ij) = p_ij``, Pearson's *tetrachoric correlation*,
  with ``Phi`` the standard normal distribution function and ``Phi_2`` its bivariate
  counterpart; the pairwise solutions are then projected onto the nearest correlation matrix.

:class:`Baseline` is the structural interface (a :class:`typing.Protocol`) the three share, and
:data:`BASELINES` maps a short name to each constructor, weakest first.

Conventions: spins are ``+-1``, and a latent value of exactly zero maps to ``+1``
(``sign(0) = +1``); ``fit`` accepts a floating-point or integer ``(N, dim)`` spin matrix and
raises :class:`ValueError` on any other shape or on an entry that is not ``+-1``;
``sample(n, generator=None)`` returns ``float32`` spins of shape ``(n, dim)`` on the model's
device and accepts an optional :class:`torch.Generator`. The derivation of the tetrachoric
match, its figures and the references are in ``docs/baselines.md``.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from typing import Protocol

import numpy as np
import torch
from torch import nn
from torch.special import ndtr as _normal_cdf
from torch.special import ndtri as _normal_icdf

__all__ = [
    "BASELINES",
    "Baseline",
    "DichotomizedGaussian",
    "IndependentBernoulli",
    "UniformBernoulli",
]


def _check_fit_input(x: torch.Tensor, dim: int) -> None:
    """Raise :class:`ValueError` unless ``x`` is a non-empty ``(N, dim)`` matrix of ``+-1`` spins."""
    if x.ndim != 2 or x.shape[1] != dim:
        raise ValueError(f"expected an (N, {dim}) spin matrix, got shape {tuple(x.shape)}")
    if x.shape[0] < 1:
        raise ValueError("the spin matrix has no rows")
    if not bool((x.abs() == 1).all()):
        raise ValueError("expected a spin matrix: every entry must be -1 or +1")


def _nearest_semidefinite(matrix: torch.Tensor, floor: float) -> torch.Tensor:
    """Clip the eigenvalues of symmetric ``matrix`` to at least ``floor``: a spectral projection."""
    evals, evecs = torch.linalg.eigh(matrix)
    clipped = (evecs * evals.clamp(min=floor)) @ evecs.mT
    return 0.5 * (clipped + clipped.mT)


def _feasible_correlation(matrix: torch.Tensor, floor: float) -> torch.Tensor:
    """Eigenvalue floor at ``floor``, then unit-diagonal rescaling: a valid correlation matrix."""
    semidefinite = _nearest_semidefinite(matrix, floor)
    scale = semidefinite.diagonal().sqrt()
    return semidefinite / scale[:, None] / scale[None, :]


def _nearest_correlation(
    corr: torch.Tensor, min_eigenvalue: float, tol: float, max_iterations: int
) -> tuple[torch.Tensor, int, float]:
    """Project ``corr`` onto the nearest correlation matrix by Higham's alternating projections.

    Algorithm 3.3 of Higham (2002), *Computing the nearest correlation matrix: a problem from
    finance*: alternate the spectral projection onto the cone ``{lambda_min >= min_eigenvalue}``
    with the projection onto unit-diagonal matrices, with Dykstra's correction on the cone step,
    so the iteration converges to the nearest point of the intersection in the Frobenius norm and
    not to an arbitrary one. The loop stops at Higham's relative-change criterion (the largest
    of the relative changes of the two iterates and of their gap, all in the Frobenius norm) at
    or below ``tol``, or after ``max_iterations`` sweeps. A ``corr`` whose symmetrized,
    unit-diagonal form already meets the eigenvalue floor is returned after zero sweeps.

    Args:
        corr (torch.Tensor): A square matrix, typically the raw pairwise estimate, which need not
            be positive semidefinite; its diagonal is ignored (reset to one) and it is not
            modified.
        min_eigenvalue (float): Floor on the eigenvalues of the result; a strictly positive floor
            makes the result positive definite, so a Cholesky factorization of it succeeds.
        tol (float): Tolerance of the relative-change stopping criterion.
        max_iterations (int): Cap on the number of sweeps, at least one; the caller compares the
            returned count against it to detect an early stop.

    Returns:
        tuple[torch.Tensor, int, float]: The final iterate passed once more through
        :func:`_feasible_correlation`, so it is a valid correlation matrix (symmetric, unit
        diagonal, positive definite) even when the loop stopped at the cap; the number of sweeps
        taken, ``0`` when no projection was needed; and the last value of the stopping criterion,
        ``0.0`` when no projection was needed.
    """
    y = 0.5 * (corr + corr.mT)
    ones = torch.ones_like(y.diagonal())
    y.diagonal().copy_(ones)

    if float(torch.linalg.eigvalsh(y).min()) >= min_eigenvalue:
        return y, 0, 0.0

    correction = torch.zeros_like(y)
    xprev = yprev = y
    residual = math.inf
    iteration = 0  # bound even when ``max_iterations`` is zero
    for iteration in range(1, max_iterations + 1):
        r = y - correction
        x = _nearest_semidefinite(r, min_eigenvalue)
        correction = x - r
        y = x.clone()
        y.diagonal().copy_(ones)
        residual = max(
            float(torch.linalg.matrix_norm(x - xprev) / torch.linalg.matrix_norm(x)),
            float(torch.linalg.matrix_norm(y - yprev) / torch.linalg.matrix_norm(y)),
            float(torch.linalg.matrix_norm(y - x) / torch.linalg.matrix_norm(y)),
        )
        xprev, yprev = x, y
        if residual <= tol:
            break
    return _feasible_correlation(y, min_eigenvalue), iteration, residual


class _BivariateNormal:
    """The standard bivariate normal distribution function, its density, and its inverse in ``rho``.

    ``Phi_2(a, b; rho) = P(X <= a, Y <= b)`` for standard normal ``X`` and ``Y`` with correlation
    ``rho``. :meth:`cdf` evaluates it by Gauss-Legendre quadrature on the one-dimensional
    integral of Drezner and Wesolowsky (1990); :meth:`pdf` is the bivariate normal density, which
    by Plackett's identity is the derivative ``d Phi_2 / d rho``; :meth:`invert` solves
    ``Phi_2(a, b; rho) = target`` for ``rho`` element-wise, the tetrachoric inversion
    :class:`DichotomizedGaussian` needs. Every method is vectorized over tensors of one common
    shape.

    Args:
        order (int): Number of Gauss-Legendre nodes. Defaults to 96, the order calibrated for
            ``|rho| <= 0.9999``, the cap :meth:`invert` works within.
        dtype (torch.dtype): Floating-point dtype of the nodes and weights, hence of every
            evaluation.
        device (torch.device): Device the nodes and weights live on; inputs must live there too.
    """

    def __init__(self, order: int = 96, *, dtype: torch.dtype, device: torch.device) -> None:
        nodes, weights = np.polynomial.legendre.leggauss(order)  # on [-1, 1], weights summing to 2
        # Mapped onto [0, 1]: the integral below runs from 0 to arcsin(rho)
        self.nodes = torch.as_tensor(0.5 * (nodes + 1.0), dtype=dtype, device=device)
        self.weights = torch.as_tensor(0.5 * weights, dtype=dtype, device=device)

    def cdf(self, a: torch.Tensor, b: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Evaluate ``Phi_2(a, b; rho)``, the lower-orthant probability, element-wise.

        Uses the single-integral form of Drezner and Wesolowsky (1990), equation (3) of Genz
        (2004), *Numerical computation of rectangular bivariate and trivariate normal and t
        probabilities*::

            Phi_2(a, b; rho) = Phi(a) Phi(b)
                + (1 / 2 pi) int_0^alpha exp(-(a^2 - 2ab sin(t) + b^2) / (2 cos^2(t))) dt,
            alpha = arcsin(rho),

        with ``Phi`` the standard normal distribution function. The integral is taken with the
        ``order``-node Gauss-Legendre rule on ``[0, 1]`` scaled by ``arcsin(rho)``, so a negative
        ``rho`` integrates backwards as the formula requires, and the result is clamped to the
        Frechet-Hoeffding bounds ``max(0, Phi(a) + Phi(b) - 1) <= Phi_2 <= min(Phi(a), Phi(b))``,
        which every bivariate distribution function with these margins obeys.

        Args:
            a (torch.Tensor): Upper limit of the first coordinate; ``a``, ``b``, and ``rho``
                share one shape.
            b (torch.Tensor): Upper limit of the second coordinate.
            rho (torch.Tensor): Correlation in ``(-1, 1)``; at ``|rho| = 1`` the integrand is
                singular.

        Returns:
            torch.Tensor: ``Phi_2(a, b; rho)``, the shape of the inputs.
        """
        angle = torch.asin(rho)
        theta = angle[..., None] * self.nodes
        sine = torch.sin(theta)
        cosine_squared = torch.cos(theta).square()
        # (a - b sin t)^2 / cos^2 t + b^2 = (a^2 - 2ab sin t + b^2) / cos^2 t, the exponent of
        # equation (3) written for the lower tail
        standardized_square = (a[..., None] - sine * b[..., None]).square() / cosine_squared + b[
            ..., None
        ].square()
        integrand = torch.exp(-0.5 * standardized_square)
        correction = angle * (integrand * self.weights).sum(dim=-1) / (2.0 * math.pi)

        probability_a = _normal_cdf(a)
        probability_b = _normal_cdf(b)
        result = probability_a * probability_b + correction

        lower = torch.clamp(probability_a + probability_b - 1.0, min=0.0)
        upper = torch.minimum(probability_a, probability_b)
        return torch.maximum(lower, torch.minimum(result, upper))

    @staticmethod
    def pdf(a: torch.Tensor, b: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
        """Evaluate the standard bivariate normal density ``phi_2(a, b; rho)`` element-wise.

        ``phi_2(a, b; rho) = exp(-(a^2 - 2 rho a b + b^2) / (2 (1 - rho^2))) / (2 pi sqrt(1 -
        rho^2))``, computed from the equivalent factorization ``(a - rho b)^2 / (1 - rho^2) +
        b^2`` of the quadratic form. By Plackett's identity this density is the derivative
        ``d Phi_2(a, b; rho) / d rho`` of the distribution function, which makes it the Newton
        derivative in :meth:`invert`.

        Args:
            a (torch.Tensor): First coordinate; ``a``, ``b``, and ``rho`` share one shape.
            b (torch.Tensor): Second coordinate.
            rho (torch.Tensor): Correlation in ``(-1, 1)``; at ``|rho| = 1`` the result is not
                finite.

        Returns:
            torch.Tensor: The density at ``(a, b)``, the shape of the inputs.
        """
        q = 1.0 - rho.square()
        quadratic = (a - rho * b).square() / q + b.square()
        log_density = -math.log(2.0 * math.pi) - 0.5 * torch.log(q) - 0.5 * quadratic
        return torch.exp(log_density)

    def invert(
        self, a: torch.Tensor, b: torch.Tensor, target: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Solve ``Phi_2(a, b; rho) = target`` for ``rho`` element-wise: the tetrachoric inversion.

        A safeguarded Newton iteration: each round takes a Newton step with :meth:`pdf` as the
        derivative (Plackett's identity) when that step is finite and lands strictly inside the
        element's current bracket ``[lower, upper]``, and the bracket's midpoint otherwise. The
        bracket starts at ``[-0.9999, 0.9999]`` and shrinks by the sign of the residual, which is
        exact because ``Phi_2`` is strictly increasing in ``rho`` (its derivative is a positive
        density), so the root is unique. The initial guess is the zero-threshold closed form
        ``rho_0 = sin(pi phi / 2)`` applied to the phi coefficient ``phi`` of the target table,
        the Pearson correlation ``(target - Phi(a) Phi(b)) / sqrt(Phi(a) (1 - Phi(a)) Phi(b) (1
        - Phi(b)))`` of the two indicator variables, which is the exact solution at ``a = b = 0``
        by Sheppard's theorem ``r = (2 / pi) arcsin(rho)``. An element stops when ``|Phi_2 -
        target| <= 1e-12`` or when its bracket is at most ``1e-10`` wide (then at the midpoint),
        after at most 100 rounds.

        The magnitude of every solution is capped at ``0.9999``: the correlation the quadrature
        order is calibrated for, and a safe distance from the singular point ``|rho| = 1``. A
        ``target`` outside ``[Phi_2(a, b; -0.9999), Phi_2(a, b; 0.9999)]`` is clamped onto that
        range before the solve, and flagged when it lies outside by more than the probability
        tolerance (closer than that is rounding, not a boundary case). At zero thresholds the
        largest spin correlation a pair can then realize is ``(2 / pi) arcsin(0.9999) ~ 0.991``.

        Args:
            a (torch.Tensor): First threshold, any shape; ``a``, ``b``, and ``target`` hold the
                same number of elements and are matched element-wise.
            b (torch.Tensor): Second threshold.
            target (torch.Tensor): Wanted value of ``Phi_2(a, b; rho)``, a joint probability in
                ``[0, 1]``.

        Returns:
            dict[str, torch.Tensor]: Four tensors in ``target``'s shape: ``"rho"``, the solution;
            ``"residual"``, ``Phi_2(a, b; rho) - target`` against the (possibly clamped) target;
            ``"converged"``, whether the element met the probability tolerance or stopped at the
            bracket width; and ``"clamped"``, whether the target had to be moved onto the
            reachable range, in which case ``rho`` is the cap itself and the residual is zero.
        """
        shape = target.shape
        a, b, target = a.reshape(-1), b.reshape(-1), target.reshape(-1)
        # An element has converged when its residual |Phi_2 - target| falls below 1e-12.
        probability_tolerance = 1e-12
        # Cap 0.9999 on the magnitude of every solution. A target that asks for more is clamped to
        # the bound and flagged; the cap keeps the solve away from the singular point |rho| = 1
        # and is the correlation the quadrature order is calibrated for.
        max_correlation = 0.9999
        lower = torch.full_like(target, -max_correlation)
        upper = torch.full_like(target, max_correlation)

        cdf_lower = self.cdf(a, b, lower)
        cdf_upper = self.cdf(a, b, upper)
        # Flagged only beyond the tolerance: where the table is flat in rho, a reachable target
        # can exceed Phi_2 at the cap by rounding alone, and that is not a boundary case.
        below = target < cdf_lower - probability_tolerance
        above = target > cdf_upper + probability_tolerance
        clamped = below | above
        target = torch.maximum(cdf_lower, torch.minimum(target, cdf_upper))

        # Start at the sine match of the table's phi coefficient, exact for zero thresholds.
        marginal_a = _normal_cdf(a)
        marginal_b = _normal_cdf(b)
        observed_covariance = target - marginal_a * marginal_b
        observed_scale = torch.sqrt(
            (marginal_a * (1.0 - marginal_a) * marginal_b * (1.0 - marginal_b)).clamp_min(
                torch.finfo(target.dtype).tiny
            )
        )
        observed_correlation = (observed_covariance / observed_scale).clamp(-1.0, 1.0)
        rho = torch.sin(0.5 * math.pi * observed_correlation).clamp(
            min=-max_correlation, max=max_correlation
        )
        # A clamped target equals Phi_2 at the cap by construction, so the cap is its exact
        # solution; set it directly instead of bisecting toward a bracket end the Newton test
        # excludes.
        rho = torch.where(below, lower, torch.where(above, upper, rho))

        active = ~clamped
        for _ in range(100):
            active_index = torch.where(active)[0]
            if active_index.numel() == 0:
                break
            current = rho[active_index]
            residual = self.cdf(a[active_index], b[active_index], current) - target[active_index]

            converged = residual.abs() <= probability_tolerance
            active[active_index[converged]] = False
            remaining_index = active_index[~converged]
            if remaining_index.numel() == 0:
                continue
            remaining_residual = residual[~converged]
            remaining_rho = current[~converged]

            new_lower = torch.where(remaining_residual < 0.0, remaining_rho, lower[remaining_index])
            new_upper = torch.where(remaining_residual > 0.0, remaining_rho, upper[remaining_index])
            lower[remaining_index] = new_lower
            upper[remaining_index] = new_upper

            # Where the density has vanished the step is infinite or not a number, and where it
            # is tiny the step is wild; the finiteness and bracket checks hand both to bisection.
            derivative = self.pdf(a[remaining_index], b[remaining_index], remaining_rho)
            newton = remaining_rho - remaining_residual / derivative
            midpoint = 0.5 * (new_lower + new_upper)
            valid_newton = torch.isfinite(newton) & (newton > new_lower) & (newton < new_upper)
            rho[remaining_index] = torch.where(valid_newton, newton, midpoint)

            # ... or stop when the bracket narrows below 1e-10 in rho: five orders below the
            # resolution 1 / sqrt(N) of any sample under 1e10 rows.
            narrow = (new_upper - new_lower) <= 1e-10
            narrow_index = remaining_index[narrow]
            rho[narrow_index] = midpoint[narrow]
            active[narrow_index] = False

        # The residual at the returned value, including elements stopped by bracket width; an
        # element still active after the last round has not converged unless its residual says so.
        residual = self.cdf(a, b, rho) - target
        converged = ~active | (residual.abs() <= probability_tolerance)
        return {
            "rho": rho.reshape(shape),
            "residual": residual.reshape(shape),
            "converged": converged.reshape(shape),
            "clamped": clamped.reshape(shape),
        }


class UniformBernoulli(nn.Module):
    """Fair coins on every spin: the null baseline, which ignores the data it is fit to.

    Every row is ``dim`` independent spins, each ``+1`` with probability one half, so the model
    has no parameters and :meth:`fit` only validates its input. A stronger baseline read against
    this one shows how much of a sample set's structure it captured at all.

    Args:
        dim (int): Number of spins per row.

    Attributes:
        anchor (torch.Tensor): A zero-size buffer, so that ``.to(device)`` moves the model and
            :meth:`sample` follows it.
    """

    anchor: torch.Tensor

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.register_buffer("anchor", torch.zeros(0))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def fit(self, x: torch.Tensor) -> None:
        """Validate ``x`` as an ``(N, dim)`` spin matrix; the model has nothing to learn from it.

        Args:
            x (torch.Tensor): Spin matrix with every entry ``+-1``, shape ``(N, dim)``.

        Raises:
            ValueError: If ``x`` is not a non-empty ``(N, dim)`` matrix of ``+-1`` entries.
        """
        _check_fit_input(x, self.dim)

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``n`` rows of independent fair spins.

        Args:
            n (int): Number of rows.
            generator (torch.Generator, optional): Random-number generator for the draw.
                Defaults to ``None``, torch's global one.

        Returns:
            torch.Tensor: ``float32`` spins of shape ``(n, dim)`` on the model's device.
        """
        bits = torch.randint(0, 2, (n, self.dim), generator=generator, device=self.anchor.device)
        return bits.to(torch.float32) * 2 - 1


class IndependentBernoulli(nn.Module):
    """Independent spins at the data's per-spin means: the order-1 baseline.

    Spin ``i`` is ``+1`` with probability ``p_i`` independently of the others, and :meth:`fit`
    sets ``p_i = (1 + m_i) / 2`` from the data's spin mean ``m_i``, the per-spin frequency of
    ``+1`` (the *magnetization* in physics vocabulary). A sample set whose structure lies in the
    means alone is captured here in full; whatever is left is interaction between spins.

    Args:
        dim (int): Number of spins per row.

    Attributes:
        p (torch.nn.Parameter): The ``(dim,)`` probabilities of ``+1``, initialized uniformly at
            random and overwritten by :meth:`fit`. A parameter only so that it moves and persists
            with the module; nothing trains it.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.p = nn.Parameter(torch.rand(dim))

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    def fit(self, x: torch.Tensor) -> None:
        """Set every ``p_i`` to the frequency of ``+1`` in column ``i`` of ``x``.

        Args:
            x (torch.Tensor): Spin matrix with every entry ``+-1``, shape ``(N, dim)``.

        Raises:
            ValueError: If ``x`` is not a non-empty ``(N, dim)`` matrix of ``+-1`` entries.
        """
        _check_fit_input(x, self.dim)
        # Column sums of +-1 entries are integers, exact in float64 at any N; no full-size copy.
        mean = x.sum(dim=0, dtype=torch.float64) / x.shape[0]
        self.p.data[:] = (1.0 + mean) / 2.0

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``n`` rows of independent spins at the fitted probabilities.

        Args:
            n (int): Number of rows.
            generator (torch.Generator, optional): Random-number generator for the draw.
                Defaults to ``None``, torch's global one.

        Returns:
            torch.Tensor: ``float32`` spins of shape ``(n, dim)`` on the device of ``p``,
            detached from autograd.
        """
        # Detached: ``p`` is a Parameter only so that it moves and persists with the module, and a
        # draw through it would otherwise carry a (zero-gradient) autograd graph.
        pp = self.p.detach().unsqueeze(0).expand(n, -1)
        return (2.0 * pp.bernoulli(generator=generator) - 1.0).to(torch.float32)


class DichotomizedGaussian(nn.Module):
    """Signs of a latent multivariate normal, matched exactly to the means and pairwise tables.

    The model draws ``X ~ N(mu, R)`` with ``R`` a correlation matrix and emits ``s = sign(X)``
    (``sign(0) = +1``): the *dichotomized Gaussian* (Emrich and Piedmonte 1991; Macke et al.
    2009). Its one- and two-spin laws have closed forms in the standard normal distribution
    function ``Phi`` and its bivariate counterpart ``Phi_2``::

        P(s_i = +1) = Phi(mu_i),      P(s_i = +1, s_j = +1) = Phi_2(mu_i, mu_j; rho_ij),

    so :meth:`fit` matches the data's per-spin frequencies of ``+1`` and its pairwise ``2x2``
    tables exactly (the order-1 and order-2 moments): ``mu_i = Phi^-1(p_i)``, and ``rho_ij``
    solved per pair from ``Phi_2(mu_i, mu_j; rho_ij) = p_ij``, Pearson's *tetrachoric
    correlation*, the latent correlation a ``2x2`` table implies under a bivariate normal, by a
    vectorized safeguarded Newton solve. At zero means this is the closed form ``rho = sin(pi r /
    2)`` for a spin correlation ``r``, the inverse of Sheppard's ``r = (2 / pi) arcsin(rho)``.
    Every margin and table is smoothed with a pseudo-count of ``1/4`` per cell (margins ``(n_i +
    1/2) / (N + 1)``, the Jeffreys prior of each binomial margin, and joint cells ``(n_ij + 1/4)
    / (N + 1)``), so a table with an empty cell still inverts, and the four smoothed cells of a
    pair sum to one, which keeps every target inside the Frechet-Hoeffding bounds. The matrix of
    pairwise solutions need not be positive semidefinite, so it is projected onto the nearest
    correlation matrix (Higham's alternating projections) with an eigenvalue floor of ``1e-4``
    before it is stored. Nothing above order two is fitted: the higher-order structure is
    whatever the Gaussian family implies.

    Args:
        dim (int): Number of spins per row; must be positive.
        device (torch.device | str, optional): Device of the parameters. Defaults to ``None``,
            torch's current default device.

    Attributes:
        mu (torch.Tensor): The ``(dim,)`` float64 latent means, a buffer; zeros before
            :meth:`fit`.
        corr (torch.Tensor): The ``(dim, dim)`` float64 latent correlation matrix, a buffer; the
            identity before :meth:`fit`.
        fit_info (dict[str, float | int]): Diagnostics of the last :meth:`fit`, empty before the
            first one: ``boundary_pairs`` (pairs whose target lay outside the range reachable at
            ``|rho| <= 0.9999`` and was clamped onto it), ``nonconverged_pairs``,
            ``max_probability_residual`` (the largest ``|Phi_2 - p_ij|`` over the pairs),
            ``projection_iterations``, ``projection_residual`` (Higham's stopping criterion at
            exit), and ``projection_max_change`` (the largest entry-wise change the projection
            made to the raw pairwise solutions).

    Raises:
        ValueError: If ``dim`` is not positive.
    """

    mu: torch.Tensor
    corr: torch.Tensor

    def __init__(self, dim: int, device: torch.device | str | None = None) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}")
        self.dim = int(dim)
        self.register_buffer("mu", torch.zeros(dim, dtype=torch.float64, device=device))
        self.register_buffer("corr", torch.eye(dim, dtype=torch.float64, device=device))
        self.fit_info: dict[str, float | int] = {}

    def extra_repr(self) -> str:
        return f"dim={self.dim}"

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> None:
        """Match ``mu`` and ``corr`` to the spin means and pairwise tables of ``x``.

        Estimates the latent means and the raw tetrachoric correlations (:meth:`_estimate`),
        projects the raw matrix onto the nearest correlation matrix with eigenvalue floor
        ``1e-4`` (relative tolerance ``1e-8``, at most 2000 sweeps), stores both, and records
        the diagnostics in :attr:`fit_info`.

        Args:
            x (torch.Tensor): Spin matrix with every entry ``+-1``, shape ``(N, dim)``, in any
                floating-point or integer dtype and on any device; rows are moved to the model's
                device in batches.

        Raises:
            ValueError: If ``x`` is not a non-empty ``(N, dim)`` matrix of ``+-1`` entries.

        Warns:
            RuntimeWarning: If a column holds a single sign (its correlations are left at zero),
                if a pair's inversion was clamped at the correlation cap or did not converge, or
                if the projection stopped at its sweep cap short of the tolerance; each case is
                also counted in :attr:`fit_info`.
        """
        _check_fit_input(x, self.dim)
        mean, raw_correlation, diagnostics = self._estimate(x)
        # The pairwise solutions need not form a positive definite matrix, which a multivariate
        # normal requires: not every pairwise structure a binary distribution can carry is the
        # image of a correlation matrix under the family's link.
        tol, max_iterations = 1e-8, 2000
        correlation, iterations, residual = _nearest_correlation(
            raw_correlation, 1e-4, tol, max_iterations
        )
        if iterations >= max_iterations and residual > tol:
            warnings.warn(
                f"the correlation projection stopped after {iterations} iterations at relative "
                f"residual {residual:.2e} > {tol:g}; corr is a valid correlation "
                "matrix but not the nearest one (fit_info carries the residual)",
                RuntimeWarning,
                stacklevel=2,
            )
        diagnostics["projection_iterations"] = iterations
        diagnostics["projection_residual"] = residual
        diagnostics["projection_max_change"] = float((correlation - raw_correlation).abs().max())
        self.mu.copy_(mean)
        self.corr.copy_(correlation)
        self.fit_info = diagnostics

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``n`` rows of ``sign(mu + L eps)`` with ``L L^T = corr`` and ``eps ~ N(0, I)``.

        Args:
            n (int): Number of rows.
            generator (torch.Generator, optional): Random-number generator for ``eps``.
                Defaults to ``None``, torch's global one.

        Raises:
            RuntimeError: If ``corr`` is not positive definite (torch's Cholesky failure); a
                fitted matrix always is, by the projection's eigenvalue floor.

        Returns:
            torch.Tensor: ``float32`` spins of shape ``(n, dim)`` on the model's device; a latent
            value of exactly zero maps to ``+1``.
        """
        chol = torch.linalg.cholesky(self.corr)  # the floor keeps a fitted matrix positive definite
        eps = torch.randn(
            (n, self.dim), dtype=self.corr.dtype, device=self.corr.device, generator=generator
        )
        latent = eps @ chol.mT + self.mu
        return torch.where(latent >= 0, 1.0, -1.0).to(torch.float32)

    def _sign_statistics(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Column sums and the Gram matrix of ``x``, accumulated over row batches.

        These two statistics determine every ``2x2`` table: the count of ``(+1, +1)`` rows of a
        pair is ``n_ij = (N + sum_r s_ri + sum_r s_rj + sum_r s_ri s_rj) / 4``. Rows are moved to
        the device and dtype of ``mu`` 4096 at a time, so ``x`` may live elsewhere or in another
        dtype.

        Args:
            x (torch.Tensor): Spin matrix of shape ``(N, dim)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The ``(dim,)`` column sums and the ``(dim, dim)``
            Gram matrix ``x^T x``, both in the dtype of ``mu`` (float64 as constructed).
        """
        n, dim = x.shape
        column_sum = torch.zeros(dim, dtype=self.mu.dtype, device=self.mu.device)
        gram = torch.zeros((dim, dim), dtype=self.mu.dtype, device=self.mu.device)
        # Batches of 4096 rows: the pass moves rows x dim float64 values at a time, 0.3 GiB at
        # ten thousand spins, and its speed does not depend on the batch.
        batch = 4096
        for start in range(0, n, batch):
            chunk = x[start : start + batch].to(
                device=self.mu.device, dtype=self.mu.dtype, non_blocking=True
            )
            column_sum.add_(chunk.sum(dim=0))
            gram.addmm_(chunk.mT, chunk)
        return column_sum, gram

    def _estimate(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]:
        """Latent means and the raw, unprojected tetrachoric correlation matrix of ``x``.

        From the column sums and the Gram matrix, every margin and ``2x2`` table is smoothed
        with the ``1/4``-per-cell pseudo-count; the means are ``Phi^-1`` of the smoothed
        margins, and each pair's latent correlation solves ``Phi_2(mu_i, mu_j; rho) = p_ij`` by
        :meth:`_BivariateNormal.invert`, 32768 pairs at a time. A column holding a single sign
        is *degenerate*: its mean still comes from its smoothed margin, but every correlation
        involving it is left at zero, with a warning.

        Args:
            x (torch.Tensor): Spin matrix of shape ``(N, dim)``, already validated.

        Returns:
            tuple[torch.Tensor, torch.Tensor, dict[str, float | int]]: The ``(dim,)`` latent
            means; the ``(dim, dim)`` symmetric unit-diagonal matrix of pairwise solutions, not
            necessarily positive semidefinite; and the inversion counters ``boundary_pairs``,
            ``nonconverged_pairs``, and ``max_probability_residual`` that :meth:`fit` extends
            and stores.

        Warns:
            RuntimeWarning: On degenerate columns, and when any pair was clamped at the
                correlation cap or did not converge.
        """
        n, dim = x.shape
        device, dtype = self.mu.device, self.mu.dtype
        column_sum, gram = self._sign_statistics(x)
        solve_dtype = torch.float64
        positive_count = 0.5 * (float(n) + column_sum.to(dtype=solve_dtype))
        # Smoothing of every 2x2 table: a pseudo-count of 1/4 per cell, the amount that gives
        # each margin the Jeffreys prior of a binomial (1/2 per sign), so the margins become
        # (n_i + 1/2) / (N + 1) and the joint cells (n_ij + 1/4) / (N + 1).
        cell_prior = 0.25
        denominator = float(n) + 4.0 * cell_prior
        marginal = (positive_count + 2.0 * cell_prior) / denominator
        threshold = _normal_icdf(marginal)
        mean = threshold.to(dtype=dtype)

        correlation = torch.eye(dim, dtype=dtype, device=device)
        informative = (positive_count > 0.0) & (positive_count < float(n))
        degenerate_count = int((~informative).sum().item())
        if degenerate_count:
            warnings.warn(
                f"{degenerate_count} variables contain only one sign; their means are "
                "regularized by the Jeffreys prior and their correlations are set to zero",
                RuntimeWarning,
                stacklevel=3,
            )

        bivariate = _BivariateNormal(dtype=solve_dtype, device=device)
        diagnostics: dict[str, float | int] = {
            "boundary_pairs": 0,
            "nonconverged_pairs": 0,
            "max_probability_residual": 0.0,
        }
        pairs = torch.triu_indices(dim, dim, offset=1, device=device)
        # Chunks of 32768 pairs: about 0.15 GiB of intermediates at the quadrature order, whatever
        # N and dim; the solve takes a few percent of a fit at any chunk size.
        for chunk in pairs.split(32768, dim=1):
            usable = informative[chunk[0]] & informative[chunk[1]]
            if not bool(usable.any()):
                continue
            i, j = chunk[0][usable], chunk[1][usable]

            joint_count = 0.25 * (
                float(n)
                + column_sum[i].to(solve_dtype)
                + column_sum[j].to(solve_dtype)
                + gram[i, j].to(solve_dtype)
            )
            joint_probability = (joint_count + cell_prior) / denominator
            inversion = bivariate.invert(threshold[i], threshold[j], joint_probability)
            correlation[i, j] = correlation[j, i] = inversion["rho"].to(dtype)

            for key, flags in (
                ("boundary_pairs", inversion["clamped"]),
                ("nonconverged_pairs", ~inversion["converged"]),
            ):
                diagnostics[key] = int(diagnostics[key]) + int(flags.sum().item())
            diagnostics["max_probability_residual"] = max(
                float(diagnostics["max_probability_residual"]),
                float(inversion["residual"].abs().max().item()),
            )
        if diagnostics["boundary_pairs"] or diagnostics["nonconverged_pairs"]:
            warnings.warn(
                "tetrachoric inversion reached the correlation boundary for "
                f"{diagnostics['boundary_pairs']} pairs and did not converge for "
                f"{diagnostics['nonconverged_pairs']} pairs; inspect fit_info",
                RuntimeWarning,
                stacklevel=3,
            )
        return mean, correlation, diagnostics


class Baseline(Protocol):
    """The interface every baseline implements: ``fit`` a spin matrix, ``sample`` rows."""

    def fit(self, x: torch.Tensor) -> None:
        """Match the model to the ``(N, dim)`` spin matrix ``x``."""

    def sample(self, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
        """Draw ``n`` rows of float32 ``+-1`` spins, shape ``(n, dim)``."""


#: Baseline name -> constructor (takes the spin count), weakest baseline first.
BASELINES: dict[str, Callable[[int], Baseline]] = {
    "uniform": UniformBernoulli,
    "bernoulli": IndependentBernoulli,
    "dichotomized": DichotomizedGaussian,
}
