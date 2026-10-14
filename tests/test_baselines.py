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
"""Tests of the moment-matched baselines and their tetrachoric core.

The bivariate-normal quadrature is checked against independent oracles (closed forms, Owen's T
function of SciPy where available, and a finite difference of the distribution function), the
tetrachoric inversion by round trips, the correlation projection against Higham's published
example, and the three models against exact moment identities plus statistical recovery of a
known dichotomized Gaussian. Statistical assertions use sample sizes and bounds (at least eight
standard errors) that do not flake; the random draws are seeded for reproducibility.
"""

import itertools
import math
import unittest
import warnings

import numpy as np
import torch
from torch.special import ndtr as torch_ndtr
from torch.special import ndtri

from dwave.plugins.torch.models import (
    BASELINES,
    DichotomizedGaussian,
    IndependentBernoulli,
    UniformBernoulli,
)
from dwave.plugins.torch.models.baselines import (
    _BivariateNormal,
    _check_fit_input,
    _feasible_correlation,
    _nearest_correlation,
    _nearest_semidefinite,
)

try:
    from scipy.special import ndtr, owens_t
except ImportError:  # pragma: no cover
    owens_t = None

F64 = torch.float64
CAP = 0.9999  # the solver's cap on |rho|; the quadrature order is calibrated for it
PROBABILITY_TOL = 1e-12  # the solver's convergence tolerance on |Phi2 - target|
FLOOR = 1e-4  # the eigenvalue floor ``DichotomizedGaussian.fit`` projects with
CPU = torch.device("cpu")


def t64(values: object) -> torch.Tensor:
    """A float64 CPU tensor from anything array-like."""
    return torch.as_tensor(np.asarray(values, dtype=float), dtype=F64)


def owen_phi2(h: np.ndarray, k: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Oracle: ``Phi2(h, k; rho)`` through Owen's T function, for ``h, k != 0``.

    Owen (1956, Ann. Math. Statist. 27:1075), in the form of Patefield & Tandy (2000, J. Stat.
    Softw. 5(5)): ``Phi2 = Phi(h)/2 + Phi(k)/2 - T(h, (k - rho h)/(h s)) - T(k, (h - rho k)/(k s))
    - beta`` with ``s = sqrt(1 - rho^2)`` and ``beta = 1/2`` when ``h k < 0``, else ``0``.
    """
    h, k, rho = (np.asarray(v, dtype=float) for v in (h, k, rho))
    s = np.sqrt(1.0 - rho * rho)
    beta = np.where((h * k > 0) | ((h * k == 0) & (h + k >= 0)), 0.0, 0.5)
    return (
        0.5 * ndtr(h)
        + 0.5 * ndtr(k)
        - owens_t(h, (k - rho * h) / (h * s))
        - owens_t(k, (h - rho * k) / (k * s))
        - beta
    )


def away_from_zero(values: np.ndarray, least: float = 1e-3) -> np.ndarray:
    """Push entries within ``least`` of zero out to ``least`` (the oracle divides by them)."""
    small = np.abs(values) < least
    return np.where(small, least, values)


def sign_of_gaussian(mean: torch.Tensor, corr: torch.Tensor, n: int) -> torch.Tensor:
    """The family's definition: ``n`` rows of ``sign(mean + L eps)`` with ``L L^T = corr``."""
    chol = torch.linalg.cholesky(corr.to(F64))
    latent = torch.randn(n, mean.shape[0], dtype=F64) @ chol.mT + mean.to(F64)
    return torch.where(latent >= 0, 1.0, -1.0).to(torch.float32)


def random_spins(n: int, d: int, p_plus: float = 0.5) -> torch.Tensor:
    """``(n, d)`` float32 spins, each entry ``+1`` with probability ``p_plus``."""
    return torch.where(torch.rand(n, d) < p_plus, 1.0, -1.0)


def random_correlation(d: int, factors: int = 2, noise: float = 0.3) -> torch.Tensor:
    """A random positive-definite correlation matrix from a low-rank loading plus a diagonal."""
    loadings = torch.randn(d, factors, dtype=F64) / math.sqrt(factors)
    covariance = (1.0 - noise) * loadings @ loadings.mT + noise * torch.eye(d, dtype=F64)
    scale = covariance.diagonal().sqrt()
    return covariance / scale[:, None] / scale[None, :]


def is_spin_matrix(x: torch.Tensor, n: int, d: int) -> bool:
    """Shape ``(n, d)``, float32, every entry ±1."""
    return tuple(x.shape) == (n, d) and x.dtype == torch.float32 and bool((x.abs() == 1).all())


class SeededTestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        np.random.seed(0)


class TestBivariateNormalCdf(SeededTestCase):
    """``_BivariateNormal.cdf`` against closed forms and Owen's T, up to the cap on ``|rho|``."""

    def setUp(self):
        super().setUp()
        self.biv = _BivariateNormal(dtype=F64, device=CPU)

    @unittest.skipUnless(owens_t is not None, "SciPy not available")
    def test_oracle_reduces_to_the_product_at_zero_correlation(self):
        h = away_from_zero(np.random.uniform(-4, 4, 2000))
        k = away_from_zero(np.random.uniform(-4, 4, 2000))
        self.assertLess(np.abs(owen_phi2(h, k, 0.0) - ndtr(h) * ndtr(k)).max(), 1e-14)

    def test_nodes_and_weights_tile_the_unit_interval(self):
        nodes, weights = self.biv.nodes, self.biv.weights
        self.assertTrue(bool(((nodes > 0) & (nodes < 1)).all()))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=14)
        self.assertAlmostEqual(float((weights * nodes).sum()), 0.5, places=14)

    def test_zero_thresholds_closed_form(self):
        rho = torch.linspace(-CAP, CAP, 1001, dtype=F64)
        zero = torch.zeros_like(rho)
        expected = 0.25 + torch.asin(rho) / (2.0 * math.pi)
        self.assertLess(float((self.biv.cdf(zero, zero, rho) - expected).abs().max()), 1e-14)

    def test_independence_at_zero_correlation(self):
        a, b = torch.randn(500, dtype=F64) * 2, torch.randn(500, dtype=F64) * 2
        got = self.biv.cdf(a, b, torch.zeros_like(a))
        self.assertLess(float((got - torch_ndtr(a) * torch_ndtr(b)).abs().max()), 1e-15)

    def test_symmetric_in_the_thresholds(self):
        a, b = torch.randn(500, dtype=F64) * 2, torch.randn(500, dtype=F64) * 2
        rho = torch.rand(500, dtype=F64) * 2 * CAP - CAP
        self.assertLess(
            float((self.biv.cdf(a, b, rho) - self.biv.cdf(b, a, rho)).abs().max()), 1e-15
        )

    def test_complement_identity(self):
        # P(X <= a, Y <= b) + P(X <= a, Y > b) = Phi(a), and (X, -Y) has correlation -rho.
        a, b = torch.randn(500, dtype=F64) * 2, torch.randn(500, dtype=F64) * 2
        rho = torch.rand(500, dtype=F64) * 2 * CAP - CAP
        total = self.biv.cdf(a, b, rho) + self.biv.cdf(a, -b, -rho)
        self.assertLess(float((total - torch_ndtr(a)).abs().max()), 1e-14)

    def test_monotone_in_the_correlation(self):
        rho = torch.linspace(-CAP, CAP, 4001, dtype=F64)
        for a, b in ((0.0, 0.0), (1.0, 1.0), (1.957, -1.869), (-3.0, 3.0), (0.5, -0.4)):
            values = self.biv.cdf(torch.full_like(rho, a), torch.full_like(rho, b), rho)
            self.assertGreaterEqual(float((values[1:] - values[:-1]).min()), -1e-15, (a, b))

    def test_frechet_bounds_hold_at_and_beyond_the_cap(self):
        a, b = torch.randn(1000, dtype=F64) * 2, torch.randn(1000, dtype=F64) * 2
        pa, pb = torch_ndtr(a), torch_ndtr(b)
        lower, upper = torch.clamp(pa + pb - 1.0, min=0.0), torch.minimum(pa, pb)
        for rho in (-1.0, -CAP, CAP, 1.0):
            got = self.biv.cdf(a, b, torch.full_like(a, rho))
            self.assertTrue(bool(torch.isfinite(got).all()), rho)
            self.assertTrue(bool((got >= lower).all()) and bool((got <= upper).all()), rho)

    @unittest.skipUnless(owens_t is not None, "SciPy not available")
    def test_matches_owens_t_on_a_random_scan(self):
        a = away_from_zero(np.random.uniform(-4, 4, 20000))
        b = away_from_zero(np.random.uniform(-4, 4, 20000))
        rho = np.random.uniform(-CAP, CAP, 20000)
        got = self.biv.cdf(t64(a), t64(b), t64(rho)).numpy()
        self.assertLess(np.abs(got - owen_phi2(a, b, rho)).max(), 1e-13)  # measured 3e-16

    @unittest.skipUnless(owens_t is not None, "SciPy not available")
    def test_matches_owens_t_on_nearly_equal_and_opposite_thresholds_at_the_cap(self):
        # The theta-integrand has a boundary layer for a ~ ±b as |rho| -> 1: the hard case.
        base = np.linspace(-5.0, 5.0, 41)
        offsets = np.concatenate([[0.0], np.logspace(-4, 0.5, 15)])
        a = np.repeat(base, 2 * offsets.size)
        b = np.concatenate([[s * x + o for o in offsets for s in (1.0, -1.0)] for x in base])
        a, b = away_from_zero(a), away_from_zero(np.clip(b, -5.0, 5.0))
        for rho in (0.9, 0.999, CAP, -CAP):
            got = self.biv.cdf(t64(a), t64(b), t64(np.full(a.size, rho))).numpy()
            self.assertLess(np.abs(got - owen_phi2(a, b, rho)).max(), 1e-12, rho)  # measured 9e-15

    def test_pdf_is_the_correlation_derivative_of_the_cdf(self):
        # Plackett's identity d Phi2 / d rho = phi2, checked by a central difference.
        a, b = torch.randn(2000, dtype=F64), torch.randn(2000, dtype=F64)
        rho = torch.rand(2000, dtype=F64) * 1.8 - 0.9
        step = 1e-5
        difference = (self.biv.cdf(a, b, rho + step) - self.biv.cdf(a, b, rho - step)) / (2 * step)
        density = self.biv.pdf(a, b, rho)
        error = (difference - density).abs()
        self.assertLess(float(error.max()), 1e-8)
        resolved = density >= 1e-2
        self.assertLess(float((error[resolved] / density[resolved]).max()), 1e-7)

    def test_pdf_closed_form(self):
        a, b = torch.randn(500, dtype=F64), torch.randn(500, dtype=F64)
        rho = torch.rand(500, dtype=F64) * 1.8 - 0.9
        q = 1.0 - rho.square()
        expected = torch.exp(-0.5 * (a.square() - 2 * rho * a * b + b.square()) / q) / (
            2.0 * math.pi * q.sqrt()
        )
        self.assertTrue(torch.allclose(self.biv.pdf(a, b, rho), expected, rtol=1e-12, atol=0))


class TestBivariateNormalInvert(SeededTestCase):
    """``_BivariateNormal.invert``, the tetrachoric solve, by round trips and closed forms."""

    def setUp(self):
        super().setUp()
        self.biv = _BivariateNormal(dtype=F64, device=CPU)

    def test_round_trip_reproduces_the_target_and_the_correlation_where_conditioned(self):
        a, b = torch.rand(5000, dtype=F64) * 4 - 2, torch.rand(5000, dtype=F64) * 4 - 2
        rho = torch.rand(5000, dtype=F64) * 1.98 - 0.99
        target = self.biv.cdf(a, b, rho)
        out = self.biv.invert(a, b, target)
        self.assertTrue(bool(out["converged"].all()))
        self.assertLessEqual(float(out["residual"].abs().max()), PROBABILITY_TOL)
        self.assertLessEqual(
            float((self.biv.cdf(a, b, out["rho"]) - target).abs().max()), PROBABILITY_TOL
        )
        # |d rho| * phi2 ~ |d Phi2| <= tol: the correlation is recovered to tol / phi2 where the
        # density is not vanishing (a flat table cannot identify rho, and is not asked to).
        error = (out["rho"] - rho).abs()
        density = self.biv.pdf(a, b, rho)
        self.assertLess(float((error * density).max()), 4 * PROBABILITY_TOL)
        resolved = density >= 1e-3
        self.assertGreater(int(resolved.sum()), 3000)
        self.assertLess(float(error[resolved].max()), 1e-8)  # measured 8e-10

    def test_output_keys_shapes_and_dtypes(self):
        a, b = torch.randn(3, 4, dtype=F64), torch.randn(3, 4, dtype=F64)
        target = self.biv.cdf(a, b, torch.full_like(a, 0.3))
        out = self.biv.invert(a, b, target)
        self.assertEqual(set(out), {"rho", "residual", "converged", "clamped"})
        for value in out.values():
            self.assertEqual(tuple(value.shape), (3, 4))
        self.assertEqual(out["rho"].dtype, F64)
        self.assertEqual(out["residual"].dtype, F64)
        self.assertEqual(out["converged"].dtype, torch.bool)
        self.assertEqual(out["clamped"].dtype, torch.bool)
        self.assertTrue(bool(out["converged"].all()) and not bool(out["clamped"].any()))

    def test_zero_thresholds_recover_the_sine_closed_form(self):
        # Phi2(0, 0; rho) = 1/4 + r/4 with r = (2/pi) arcsin(rho): the inverse is rho = sin(pi r/2).
        r = torch.rand(2000, dtype=F64) * 1.96 - 0.98
        zero = torch.zeros_like(r)
        out = self.biv.invert(zero, zero, 0.25 + r / 4)
        self.assertLess(float((out["rho"] - torch.sin(0.5 * math.pi * r)).abs().max()), 1e-12)
        self.assertTrue(bool(out["converged"].all()) and not bool(out["clamped"].any()))

    def test_infeasible_targets_are_clamped_to_the_cap(self):
        a, b = t64([0.0, 1.0, 0.0]), t64([0.0, -1.0, 0.0])
        # 0.6 exceeds Phi2(0, 0; CAP) ~ 0.498; 0.0 is below Phi2(1, -1; -CAP); 0.05 is feasible.
        out = self.biv.invert(a, b, t64([0.6, 0.0, 0.05]))
        self.assertEqual(out["clamped"].tolist(), [True, True, False])
        self.assertTrue(bool(out["converged"].all()))
        self.assertLessEqual(float(out["rho"].abs().max()), CAP)
        self.assertAlmostEqual(float(out["rho"][0]), CAP, delta=1e-9)
        self.assertAlmostEqual(float(out["rho"][1]), -CAP, delta=1e-9)
        self.assertFalse(bool(out["clamped"][2]))
        self.assertLessEqual(float(out["residual"][2].abs()), PROBABILITY_TOL)

    def test_clamped_targets_are_solved_to_tolerance(self):
        # A clamped target equals Phi2(a, b; ±CAP) by construction, so the cap is its exact
        # solution and the residual is zero.
        a, b = t64([1.0]), t64([-1.0])
        out = self.biv.invert(a, b, t64([0.0]))
        self.assertTrue(bool(out["clamped"][0]))
        self.assertLessEqual(float(out["residual"][0].abs()), PROBABILITY_TOL)

    def test_feasible_targets_are_never_flagged_clamped(self):
        # A target Phi2(a, b; rho) with |rho| <= 0.99 lies inside the cap's range, but where the
        # table is flat in rho (a ~ -b, |rho| near 1) it can exceed Phi2(a, b; CAP) by rounding;
        # the flag allows the probability tolerance of slack, so none of the scan is flagged.
        a, b = torch.rand(20000, dtype=F64) * 4 - 2, torch.rand(20000, dtype=F64) * 4 - 2
        rho = torch.rand(20000, dtype=F64) * 1.98 - 0.99
        out = self.biv.invert(a, b, self.biv.cdf(a, b, rho))
        self.assertEqual(int(out["clamped"].sum()), 0)

    def test_moderate_correlations_are_never_flagged_clamped(self):
        # |a|, |b| <= 1.5 and |rho| <= 0.8: every target sits at least 5e-8 inside both bounds
        # (measured over a grid), ten orders of magnitude above the rounding that trips the flag.
        a, b = torch.rand(20000, dtype=F64) * 3 - 1.5, torch.rand(20000, dtype=F64) * 3 - 1.5
        rho = torch.rand(20000, dtype=F64) * 1.6 - 0.8
        out = self.biv.invert(a, b, self.biv.cdf(a, b, rho))
        self.assertEqual(int(out["clamped"].sum()), 0)
        self.assertTrue(bool(out["converged"].all()))


class TestNearestCorrelation(SeededTestCase):
    """Higham's alternating projections and its two building blocks."""

    def test_valid_correlation_matrix_is_returned_unchanged(self):
        corr = random_correlation(6)
        out, iterations, residual = _nearest_correlation(corr, FLOOR, 1e-8, 2000)
        self.assertEqual(iterations, 0)
        self.assertEqual(residual, 0.0)
        self.assertLess(float((out - corr).abs().max()), 1e-15)
        self.assertTrue(torch.equal(out.diagonal(), torch.ones(6, dtype=F64)))

    def test_higham_2002_example(self):
        # Higham (2002, IMA J. Numer. Anal. 22:329), section 4: the nearest correlation matrix to
        # [[1, 1, 0], [1, 1, 1], [0, 1, 1]] has off-diagonals 0.7607 (adjacent) and 0.1573
        # (corner); the eigenvalue floor of 1e-4 moves them by O(1e-4).
        corr = t64([[1, 1, 0], [1, 1, 1], [0, 1, 1]])
        out, iterations, residual = _nearest_correlation(corr, FLOOR, 1e-8, 2000)
        self.assertGreaterEqual(iterations, 1)
        self.assertLessEqual(residual, 1e-8)
        self.assertAlmostEqual(float(out[0, 1]), 0.7607, delta=1e-3)
        self.assertAlmostEqual(float(out[1, 2]), 0.7607, delta=1e-3)
        self.assertAlmostEqual(float(out[0, 2]), 0.1573, delta=1e-3)
        self.assertTrue(torch.allclose(out.diagonal(), torch.ones(3, dtype=F64), atol=1e-14))

    def test_projection_of_an_indefinite_matrix_is_a_correlation_matrix(self):
        d = 8
        raw = torch.rand(d, d, dtype=F64) * 2 - 1  # indefinite with probability ~1
        raw = 0.5 * (raw + raw.mT)
        raw.diagonal().fill_(1.0)
        out, iterations, residual = _nearest_correlation(raw, FLOOR, 1e-8, 2000)
        self.assertGreaterEqual(iterations, 1)
        self.assertLessEqual(residual, 1e-8)
        self.assertLess(float((out - out.mT).abs().max()), 1e-15)
        self.assertTrue(torch.allclose(out.diagonal(), torch.ones(d, dtype=F64), atol=1e-14))
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(out).min()), FLOOR - 1e-10)
        torch.linalg.cholesky(out)  # positive definite
        # Nearer than the trivial feasible point: the identity.
        self.assertLess(
            float(torch.linalg.matrix_norm(out - raw)),
            float(torch.linalg.matrix_norm(torch.eye(d, dtype=F64) - raw)),
        )

    def test_nearest_semidefinite_clips_the_spectrum(self):
        matrix = torch.diag(t64([-1.0, 0.5, 2.0]))
        out = _nearest_semidefinite(matrix, FLOOR)
        self.assertTrue(torch.allclose(out, torch.diag(t64([FLOOR, 0.5, 2.0])), atol=1e-14))
        definite = random_correlation(5)
        self.assertTrue(
            torch.allclose(_nearest_semidefinite(definite, FLOOR), definite, atol=1e-13)
        )

    def test_feasible_correlation_rescales_to_unit_diagonal(self):
        loadings = torch.randn(5, 3, dtype=F64)
        covariance = loadings @ loadings.mT + 0.1 * torch.eye(5, dtype=F64)
        out = _feasible_correlation(covariance, FLOOR)
        self.assertTrue(torch.allclose(out.diagonal(), torch.ones(5, dtype=F64), atol=1e-14))
        scale = covariance.diagonal().sqrt()
        self.assertTrue(
            torch.allclose(out, covariance / scale[:, None] / scale[None, :], atol=1e-13)
        )


class TestCheckFitInput(SeededTestCase):
    def test_rejects_malformed_input(self):
        with self.assertRaises(ValueError):
            _check_fit_input(torch.ones(5), 3)  # one-dimensional
        with self.assertRaises(ValueError):
            _check_fit_input(torch.ones(5, 4), 3)  # wrong width
        with self.assertRaises(ValueError):
            _check_fit_input(torch.ones(0, 3), 3)  # no rows
        with self.assertRaises(ValueError):
            _check_fit_input(torch.zeros(5, 3), 3)  # not spins
        with self.assertRaises(ValueError):
            _check_fit_input(torch.full((5, 3), 0.5), 3)  # soft spins are not spins

    def test_accepts_a_spin_matrix(self):
        _check_fit_input(random_spins(7, 3), 3)
        _check_fit_input(torch.ones(1, 3), 3)
        _check_fit_input(torch.ones(2, 3, dtype=torch.int8), 3)  # any dtype


class TestDichotomizedGaussian(SeededTestCase):
    FIT_INFO_KEYS = {
        "boundary_pairs",
        "nonconverged_pairs",
        "max_probability_residual",
        "projection_iterations",
        "projection_residual",
        "projection_max_change",
    }

    def test_construction(self):
        model = DichotomizedGaussian(4)
        self.assertEqual(model.dim, 4)
        self.assertTrue(torch.equal(model.mu, torch.zeros(4, dtype=F64)))
        self.assertTrue(torch.equal(model.corr, torch.eye(4, dtype=F64)))
        self.assertEqual(set(model.state_dict()), {"mu", "corr"})
        self.assertEqual(model.fit_info, {})
        self.assertIn("dim=4", repr(model))
        self.assertEqual("meta", DichotomizedGaussian(4).to("meta").corr.device.type)
        for bad in (0, -3):
            with self.assertRaises(ValueError):
                DichotomizedGaussian(bad)

    def test_fit_validates_input(self):
        model = DichotomizedGaussian(3)
        for bad in (torch.ones(5), torch.ones(5, 4), torch.ones(0, 3), torch.zeros(5, 3)):
            with self.assertRaises(ValueError):
                model.fit(bad)

    def test_estimate_on_balanced_columns_is_the_sine_closed_form(self):
        # Appending each row's negation zeroes every column sum and keeps every pairwise product:
        # the smoothed margins are exactly 1/2, the thresholds exactly 0, and the raw tetrachoric
        # correlation is exactly sin(pi/2 * N/(N+1) * r) with r the mean spin product.
        d = 5
        half = random_spins(400, d)
        x = torch.cat([half, -half])
        n = x.shape[0]
        mean, raw, diagnostics = DichotomizedGaussian(d)._estimate(x)
        self.assertTrue(torch.equal(mean, torch.zeros(d, dtype=F64)))
        xd = x.to(F64)
        r = xd.mT @ xd / n
        expected = torch.sin(0.5 * math.pi * (n / (n + 1)) * r)
        off = ~torch.eye(d, dtype=torch.bool)
        self.assertLess(float((raw - expected)[off].abs().max()), 1e-10)
        self.assertTrue(torch.equal(raw.diagonal(), torch.ones(d, dtype=F64)))
        self.assertEqual(diagnostics["boundary_pairs"], 0)
        self.assertEqual(diagnostics["nonconverged_pairs"], 0)

    def test_estimate_means_are_jeffreys_smoothed_probit_frequencies(self):
        x = random_spins(333, 3, p_plus=0.7)
        mean, _, _ = DichotomizedGaussian(3)._estimate(x)
        positive = (x > 0).sum(0).to(F64)
        expected = ndtri((positive + 0.5) / (x.shape[0] + 1))
        self.assertLess(float((mean - expected).abs().max()), 1e-12)

    def test_fit_stores_a_projected_correlation_matrix_and_diagnostics(self):
        d = 6
        x = sign_of_gaussian(torch.randn(d) * 0.5, random_correlation(d), 2000)
        model = DichotomizedGaussian(d)
        mean, raw, _ = model._estimate(x)
        model.fit(x)
        self.assertTrue(torch.equal(model.mu, mean))
        self.assertLess(float((model.corr - model.corr.mT).abs().max()), 1e-15)
        self.assertTrue(torch.allclose(model.corr.diagonal(), torch.ones(d, dtype=F64)))
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(model.corr).min()), FLOOR - 1e-10)
        torch.linalg.cholesky(model.corr)
        self.assertEqual(set(model.fit_info), self.FIT_INFO_KEYS)
        self.assertAlmostEqual(
            model.fit_info["projection_max_change"], float((model.corr - raw).abs().max())
        )
        self.assertLessEqual(model.fit_info["max_probability_residual"], PROBABILITY_TOL)
        if model.fit_info["projection_iterations"] == 0:
            self.assertTrue(torch.equal(model.corr, raw))

    def test_symmetric_two_plus_states_force_the_projection(self):
        # Four spins, the six states with exactly two +1's, k copies each: every column sum is
        # exactly zero and every pairwise product mean exactly -1/3, so the smoothed thresholds
        # are exactly 0 and the raw tetrachoric matrix is equicorrelated with off-diagonal
        # c = sin(pi/2 * N/(N+1) * (-1/3)) ~ -0.5, whose smallest eigenvalue 1 + 3c ~ -0.5 is
        # infeasible. The nearest correlation matrix with eigenvalues >= FLOOR is, by symmetry,
        # equicorrelated with 1 + 3c' = FLOOR, i.e. c' = -(1 - FLOOR)/3.
        d, copies = 4, 50
        states = [
            [1.0 if position in pair else -1.0 for position in range(d)]
            for pair in itertools.combinations(range(d), 2)
        ]
        x = torch.tensor(states, dtype=torch.float32).repeat(copies, 1)
        n = x.shape[0]
        model = DichotomizedGaussian(d)
        mean, raw, _ = model._estimate(x)
        off = ~torch.eye(d, dtype=torch.bool)
        c = math.sin(0.5 * math.pi * (n / (n + 1)) * (-1.0 / 3.0))
        self.assertTrue(torch.equal(mean, torch.zeros(d, dtype=F64)))
        self.assertLess(float((raw[off] - c).abs().max()), 1e-10)
        self.assertLess(float(torch.linalg.eigvalsh(raw).min()), -0.4)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model.fit(x)
        c_projected = -(1.0 - FLOOR) / 3.0
        self.assertGreater(model.fit_info["projection_iterations"], 0)
        self.assertLessEqual(model.fit_info["projection_residual"], 1e-8)
        self.assertTrue(torch.allclose(model.corr.diagonal(), torch.ones(d, dtype=F64)))
        self.assertGreaterEqual(float(torch.linalg.eigvalsh(model.corr).min()), FLOOR - 1e-10)
        self.assertLess(float((model.corr[off] - model.corr[off].mean()).abs().max()), 1e-8)
        self.assertLess(float((model.corr[off] - c_projected).abs().max()), 1e-3)
        self.assertAlmostEqual(model.fit_info["projection_max_change"], c_projected - c, delta=1e-3)
        torch.linalg.cholesky(model.corr)
        # The fitted model's spins: zero means and Sheppard's (2/pi) arcsin(c') ~ -0.216.
        y = model.sample(65536).to(F64)
        self.assertLess(float(y.mean(0).abs().max()), 0.04)  # se = 0.004
        spin_corr = torch.corrcoef(y.mT)[off]
        self.assertLess(float((spin_corr - 2 / math.pi * math.asin(c_projected)).abs().max()), 0.04)

    def test_degenerate_column_warns_and_decouples(self):
        x = random_spins(200, 3)
        x[:, 1] = 1.0
        model = DichotomizedGaussian(3)
        with self.assertWarnsRegex(RuntimeWarning, "only one sign"):
            model.fit(x)
        self.assertTrue(bool(torch.isfinite(model.mu).all()))
        self.assertAlmostEqual(float(model.mu[1]), float(ndtri(t64(200.5 / 201))), places=12)
        self.assertEqual(model.corr[1].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(model.corr[:, 1].tolist(), [0.0, 1.0, 0.0])
        model.sample(10)  # still a valid correlation matrix

    def test_recovers_a_known_dichotomized_gaussian(self):
        # n = 65536 rows: the threshold standard error is <= 0.007 for |mu| <= 0.8 and the
        # tetrachoric one ~0.01, so 0.06 / 0.08 are eight-plus standard errors.
        d, n = 5, 65536
        mu = torch.rand(d, dtype=F64) * 1.6 - 0.8
        corr = random_correlation(d)
        x = sign_of_gaussian(mu, corr, n)
        model = DichotomizedGaussian(d)
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            model.fit(x)
        self.assertLess(float((model.mu - mu).abs().max()), 0.06)
        self.assertLess(float((model.corr - corr).abs().max()), 0.08)
        self.assertEqual(model.fit_info["boundary_pairs"], 0)
        self.assertEqual(model.fit_info["nonconverged_pairs"], 0)

    def test_samples_reproduce_the_fitted_moments(self):
        d, n = 5, 65536
        x = sign_of_gaussian(torch.rand(d, dtype=F64) - 0.5, random_correlation(d), n)
        model = DichotomizedGaussian(d)
        model.fit(x)
        y = model.sample(n)
        self.assertTrue(is_spin_matrix(y, n, d))
        xd, yd = x.to(F64), y.to(F64)
        self.assertLess(float((xd.mean(0) - yd.mean(0)).abs().max()), 0.05)  # se ~0.006
        self.assertLess(
            float((torch.corrcoef(xd.mT) - torch.corrcoef(yd.mT)).abs().max()), 0.06
        )  # se ~0.007

    def test_sample_shape_dtype_device_and_generator(self):
        model = DichotomizedGaussian(4)
        model.fit(random_spins(64, 4))
        y = model.sample(17, generator=torch.Generator().manual_seed(1))
        self.assertTrue(is_spin_matrix(y, 17, 4))
        self.assertEqual(y.device, model.corr.device)
        self.assertTrue(is_spin_matrix(DichotomizedGaussian(3).sample(5), 5, 3))  # unfitted: coins

    def test_state_dict_round_trip_samples_without_refitting(self):
        d = 4
        fitted = DichotomizedGaussian(d)
        fitted.fit(sign_of_gaussian(torch.randn(d) * 0.3, random_correlation(d), 4096))
        fresh = DichotomizedGaussian(d)
        fresh.load_state_dict(fitted.state_dict())
        self.assertTrue(torch.equal(fresh.mu, fitted.mu) and torch.equal(fresh.corr, fitted.corr))
        self.assertTrue(is_spin_matrix(fresh.sample(9), 9, d))


class TestIndependentBernoulli(SeededTestCase):
    def test_fit_sets_the_plus_probabilities_exactly(self):
        x = random_spins(100, 4, p_plus=0.3)
        model = IndependentBernoulli(4)
        model.fit(x)
        exact = (x > 0).sum(0).to(F64) / x.shape[0]  # the frequency of +1, as a rational
        self.assertLess(float((model.p.detach().to(F64) - exact).abs().max()), 1e-6)  # float32
        self.assertEqual(set(model.state_dict()), {"p"})
        self.assertIn("dim=4", repr(model))

    def test_fit_validates_input(self):
        for bad in (torch.ones(5), torch.ones(5, 4), torch.ones(0, 3), torch.zeros(5, 3)):
            with self.assertRaises(ValueError):
                IndependentBernoulli(3).fit(bad)

    def test_samples_match_the_fitted_means(self):
        d, n = 6, 20000
        x = random_spins(500, d, p_plus=0.65)
        model = IndependentBernoulli(d)
        model.fit(x)
        y = model.sample(n, generator=torch.Generator().manual_seed(2))
        self.assertTrue(is_spin_matrix(y, n, d))
        # each spin mean has standard error <= 1/sqrt(n) = 0.007: 0.05 is seven of them
        self.assertLess(float((y.mean(0) - x.mean(0)).abs().max()), 0.05)

    def test_constant_columns_stay_constant(self):
        x = random_spins(50, 3)
        x[:, 0], x[:, 2] = 1.0, -1.0
        model = IndependentBernoulli(3)
        model.fit(x)
        y = model.sample(200)
        self.assertTrue(bool((y[:, 0] == 1).all()) and bool((y[:, 2] == -1).all()))
        self.assertTrue(bool((y[:, 1] == 1).any()) and bool((y[:, 1] == -1).any()))


class TestUniformBernoulli(SeededTestCase):
    def test_fit_validates_and_ignores_the_data(self):
        model = UniformBernoulli(3)
        model.fit(torch.ones(20, 3))  # all +1: a fitted model would be constant; this one is not
        y = model.sample(400)
        self.assertTrue(is_spin_matrix(y, 400, 3))
        self.assertTrue(bool((y == -1).any(dim=0).all()))
        for bad in (torch.ones(5), torch.ones(5, 4), torch.ones(0, 3), torch.zeros(5, 3)):
            with self.assertRaises(ValueError):
                model.fit(bad)

    def test_samples_are_fair_coins(self):
        d, n = 5, 20000
        y = UniformBernoulli(d).sample(n, generator=torch.Generator().manual_seed(3))
        self.assertTrue(is_spin_matrix(y, n, d))
        self.assertLess(float(y.mean(0).abs().max()), 0.05)  # se = 1/sqrt(n) = 0.007
        self.assertLess(float(torch.corrcoef(y.to(F64).mT).fill_diagonal_(0).abs().max()), 0.05)

    def test_anchor_buffer_carries_the_device(self):
        model = UniformBernoulli(2)
        self.assertEqual(set(model.state_dict()), {"anchor"})
        self.assertEqual(model.sample(3).device, model.anchor.device)
        self.assertIn("dim=2", repr(model))


class TestRegistry(SeededTestCase):
    def test_baselines_are_registered_weakest_first(self):
        self.assertEqual(list(BASELINES), ["uniform", "bernoulli", "dichotomized"])
        self.assertIs(BASELINES["uniform"], UniformBernoulli)
        self.assertIs(BASELINES["bernoulli"], IndependentBernoulli)
        self.assertIs(BASELINES["dichotomized"], DichotomizedGaussian)

    def test_every_baseline_implements_the_protocol(self):
        x = random_spins(128, 4)
        for name, cls in BASELINES.items():
            model = cls(4)
            self.assertTrue(callable(model.fit) and callable(model.sample), name)
            model.fit(x)
            self.assertTrue(is_spin_matrix(model.sample(6), 6, 4), name)
            self.assertTrue(
                is_spin_matrix(model.sample(6, generator=torch.Generator()), 6, 4), name
            )


if __name__ == "__main__":
    unittest.main()
