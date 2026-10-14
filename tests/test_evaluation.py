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
"""Tests of the kernel-score evaluation: the histogram estimator against its per-entry
reference and closed-form block variances, the evaluator against ``blocked_scores``, the
calibration rule, the loss against the estimator, the selection-bias measurement, the dtype and
device contract, and the plots. Assertions are structural, exact, or statistical with bounds
wide enough not to flake; the random fixtures are seeded."""

import math
import pathlib
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import dwave.plugins.torch.evaluation.estimator as estimator
import dwave.plugins.torch.kernels as kernels
from dwave.plugins.torch.evaluation import UNIFORM, Calibration, Evaluator, blocked_scores
from dwave.plugins.torch.kernels import kernel_profiles, uniform_kernel_mean
from tests.helper_functions import enumerate_walsh

try:
    from matplotlib.collections import PathCollection

    from dwave.plugins.torch.evaluation.visualize import plot_fourier_weights, plot_kernel_spectra
except ImportError:  # pragma: no cover
    PathCollection = None

N = 24
# block_size=32 gives B=32 blocks at the 1024-row test fixtures (and the minimum 8 at the
# 256-row calibration splits): the noise-level bounds below are ratios of degenerate
# U-statistics to their standard errors, heavier-tailed than a t-statistic on 31 degrees of
# freedom, and 7 se keeps 25 of them per test from flaking
BS = 32
NOISE = 7.0  # |mmd2 / se| bound for an entry whose population value is zero
GENERATOR = torch.Generator().manual_seed(2026)
CENTER = torch.randint(0, 2, (N,), dtype=torch.float64, generator=GENERATOR) * 2 - 1


def mixture(m: int, eps: float = 0.1) -> torch.Tensor:
    """Two-cluster spin mixture (zero means, strong pairwise structure) as float64 spins.

    Float64 is the dtype every fixture is built in: the package computes in the dtype it is
    given, and float64 is where the exact-arithmetic assertions below hold.
    """
    s = torch.randint(0, 2, (m, 1), dtype=torch.float64, generator=GENERATOR) * 2 - 1
    flips = torch.where(torch.rand(m, N, generator=GENERATOR) < eps, -1.0, 1.0)
    return s * CENTER * flips


def indep(m: int) -> torch.Tensor:
    """Independent fair bits (matches the mixture's zero means, no interactions)."""
    return torch.randint(0, 2, (m, N), dtype=torch.float64, generator=GENERATOR) * 2 - 1


def mixture_with_means(m: int, p_plus: float, eps: float = 0.1) -> torch.Tensor:
    """The mixture with unequal cluster weights: even orders exact, odd-order structure imposed."""
    s = torch.where(torch.rand(m, 1, dtype=torch.float64, generator=GENERATOR) < p_plus, 1.0, -1.0)
    flips = torch.where(torch.rand(m, N, generator=GENERATOR) < eps, -1.0, 1.0).to(torch.float64)
    return s.to(torch.float64) * CENTER * flips


class SeededTestCase(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)


class TestUniformArm(unittest.TestCase):
    """The analytic uniform model and the alpha row against direct enumeration at n = 8."""

    n = 8

    @classmethod
    def setUpClass(cls):
        cls.states, cls.orders, cls.chi = enumerate_walsh(cls.n)
        cls.table = kernel_profiles(cls.n).numpy()
        neg = ((1 - cls.states) // 2).sum(axis=1)
        lam_by_subset = np.stack([(cls.chi @ row[neg]) / (1 << cls.n) for row in cls.table])
        cls.lam = np.array(
            [[ls[cls.orders == j][0] for j in range(cls.n + 1)] for ls in lam_by_subset]
        )

    def test_uniform_arm_is_unbiased_by_enumeration(self):
        # the analytic uniform model: its per-block score is the within-block reference
        # U-statistic minus the closed-form uniform kernel mean, unbiased for the exact
        # MMD^2(P, U) = sum_{S != 0} lambda_|S| W_P[S]^2; at entry k that is the mean squared
        # order-k coefficient of P, the Fourier weight the module reports as uniform_score.
        # Monte Carlo, 400 blocks of 64.
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.full(1 << self.n, 0.3))
        wp = self.chi @ p
        num_blocks, r = 400, 64
        draws = torch.as_tensor(
            self.states[rng.choice(1 << self.n, size=num_blocks * r, p=p)], dtype=torch.float64
        )
        blocked = blocked_scores(draws, {"u": UNIFORM}, num_blocks, kernel_profiles(self.n))
        for j in range(self.n + 1):
            est = blocked.scores[j, 0].numpy()
            mean, se = float(est.mean()), float(est.std(ddof=1) / math.sqrt(num_blocks))
            exact = float((self.lam[j][self.orders] * wp**2)[1:].sum())  # S != empty set
            self.assertLess(abs(mean - exact), 5 * se, f"entry {j}: {mean} vs {exact} +- {se}")
            if j >= 1:
                self.assertAlmostEqual(exact, float(np.mean(wp[self.orders == j] ** 2)), places=12)

    def test_alpha_row_of_an_evaluator_by_enumeration(self):
        # holding kernel weights swaps the alpha row in at profile 0 of the evaluator's table:
        # its value at distance 0 is the weights' sum and its spectrum alpha_k / C(n, k)
        n = self.n
        neg = ((1 - self.states) // 2).sum(axis=1)
        binom = np.array([math.comb(n, k) for k in range(1, n + 1)], dtype=float)
        alpha = torch.tensor([0.4, 0.0, 0.16, 0.02, 0.07, 0.002, 0.03, 0.01], dtype=torch.float64)
        spins = torch.as_tensor(
            self.states[torch.randint(0, 1 << n, (512,), generator=GENERATOR).numpy()],
            dtype=torch.float64,
        )
        ev_ = Evaluator(spins, spins, spins, block_size=64, alpha=alpha)
        self.assertEqual(ev_.profile_names[0], "alpha")
        row = ev_.profiles[0].numpy()
        self.assertAlmostEqual(float(row[0]), float(alpha.sum()), places=12)
        by_subset = (self.chi @ row[neg]) / (1 << n)
        lam = np.array([by_subset[self.orders == j][0] for j in range(n + 1)])
        self.assertAlmostEqual(float(lam[0]), 0.0, places=12)
        np.testing.assert_allclose(lam[1:], alpha.numpy() / binom, rtol=1e-10, atol=1e-14)
        self.assertAlmostEqual(float(uniform_kernel_mean(ev_.profiles[0])), 0.0, places=12)


class TestEstimator(SeededTestCase):
    def test_histogram_scores_match_the_per_entry_reference(self):
        # blocked_scores reduces each block pair to a Hamming-distance histogram and reads
        # every kernel off it with one dot product; the per-entry table lookup is the
        # reference here, arm by arm (sampled models and the analytic uniform model)
        x = mixture(256)
        ys = {"gen": mixture(256, 0.16), "u": UNIFORM, "indep": indep(256)}
        num_blocks, r = 8, 32
        table = kernel_profiles(N)
        fast = blocked_scores(x, ys, num_blocks, table)
        self.assertEqual(tuple(fast.scores.shape), (N + 1, 3, num_blocks))
        xn, tn = x.numpy(), table.numpy()

        def ham(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return (N - a.astype(np.int32) @ b.astype(np.int32).T) >> 1

        def offdiag(fk: np.ndarray) -> float:
            return float((fk.sum() - np.trace(fk)) / (r * (r - 1)))

        for b in range(num_blocks):
            xb = xn[b * r : (b + 1) * r]
            dxx = ham(xb, xb)
            for i, (name, y) in enumerate(ys.items()):
                for j, prof in enumerate(tn):
                    if isinstance(y, torch.Tensor):
                        yb = y.numpy()[b * r : (b + 1) * r]
                        ref = offdiag(prof[dxx]) + offdiag(prof[ham(yb, yb)])
                        ref -= 2.0 * float(prof[ham(xb, yb)].mean())
                    else:
                        ref = offdiag(prof[dxx]) - float(uniform_kernel_mean(table[j]))
                    self.assertAlmostEqual(
                        float(fast.scores[j, i, b]), ref, places=12, msg=f"{name} {j} {b}"
                    )

    def test_block_variance_under_uniform_matches_closed_form(self):
        # the per-block variance of the blocked estimator under P = Q = uniform is known
        # exactly: two within-sample terms of variance 2/(r (r - 1)) each and a cross term of
        # variance 4/r^2, all in units of the kernel's uniform-reference variance 1/C(n, k);
        # the UNIFORM arm keeps only its one within-sample term, 2/(r (r - 1)). Monte Carlo
        # over 512 blocks; the sample variance's own noise is ~6%, tolerance 25%.
        n, r, num_blocks = 10, 16, 512
        x = torch.randint(0, 2, (r * num_blocks, n), dtype=torch.float64) * 2 - 1
        y = torch.randint(0, 2, (r * num_blocks, n), dtype=torch.float64) * 2 - 1
        blocked = blocked_scores(x, {"y": y, "u": UNIFORM}, num_blocks, kernel_profiles(n)[1:5])
        two_sample = 4 / (r * (r - 1)) + 4 / r**2
        one_within = 2 / (r * (r - 1))
        for i, k in enumerate(range(1, 5)):
            arr = blocked.scores[i].numpy()
            var_pair = float(arr[0].var(ddof=1)) * math.comb(n, k)
            var_uniform = float(arr[1].var(ddof=1)) * math.comb(n, k)
            self.assertLess(abs(var_pair / two_sample - 1.0), 0.25, f"k={k}")
            self.assertLess(abs(var_uniform / one_within - 1.0), 0.25, f"k={k}")
            for row in (0, 1):  # both arms estimate the population value 0 without bias
                se = float(arr[row].std(ddof=1) / math.sqrt(num_blocks))
                self.assertLess(abs(float(arr[row].mean())), 5 * se, f"k={k} row={row}")

    def test_point_mass_scores_exactly(self):
        # every block of a point mass has all distances 0, so the uniform arm reads exactly
        # f(0) - lambda_0 in every block: 1 for an order (unit diagonal, zero uniform mean),
        # 1 - lambda_0 for gaussian, with zero block spread
        n = 16
        x = torch.ones((64, n), dtype=torch.float64)
        table = kernel_profiles(n)
        got = blocked_scores(x, {"u": UNIFORM}, 8, table).scores[:, 0, :]
        np.testing.assert_allclose(got[1:].numpy(), 1.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(
            got[0].numpy(), 1.0 - float(uniform_kernel_mean(table[0])), rtol=1e-12
        )

    def test_uniform_marker_mixes_with_samples_and_custom_tables(self):
        x, y = indep(256), indep(256)
        custom = torch.exp(-torch.arange(N + 1, dtype=torch.float64) / 3.0)[None, :]
        got = blocked_scores(x, {"u": UNIFORM, "y": y}, 16, custom)
        self.assertEqual(got.model_names, ("u", "y"))
        self.assertEqual(tuple(got.scores.shape), (1, 2, 16))
        self.assertEqual(repr(UNIFORM), "UNIFORM")
        for row in (0, 1):  # for uniform data both arms estimate 0 within noise
            est = got.scores[0, row].numpy()
            self.assertLess(abs(est.mean()), 5 * est.std(ddof=1) / math.sqrt(16) + 1e-12)

    def test_profile_table_validated(self):
        x = torch.ones((8, 4), dtype=torch.float64)
        y = torch.randint(0, 2, (8, 4), dtype=torch.float64) * 2 - 1
        for label, bad in {
            "1-D": torch.ones(5),
            "short": torch.ones((2, 4)),
            "long": torch.ones((1, 6)),
            "empty": torch.ones((0, 5)),
            "scalar": torch.tensor(1.0),
        }.items():
            with self.assertRaisesRegex(ValueError, "profiles", msg=label):
                blocked_scores(x, {"m": y}, 2, bad)

    def test_block_count_validated(self):
        x, y = indep(64), indep(64)
        table = kernel_profiles(N)[:1]
        for bad in (0, -4):
            with self.assertRaisesRegex(ValueError, "num_blocks", msg=repr(bad)):
                blocked_scores(x, {"m": y}, bad, table)
        with self.assertRaisesRegex(ValueError, "divisible"):
            blocked_scores(x, {"m": y}, 7, table)
        with self.assertRaisesRegex(ValueError, "at least 2 rows"):
            blocked_scores(x, {"m": y}, 64, table)
        got = blocked_scores(x, {"m": y}, np.int64(8), table)
        self.assertEqual(got.num_blocks, 8)
        self.assertEqual(got.block_size, 8)

    def test_spin_validation_everywhere(self):
        table = kernel_profiles(N)[:1]
        good = indep(64)
        # shape: empty dimension or empty sample, model shape != data shape, no model at all
        for bad in (torch.ones((0, N)), torch.ones((64, 0))):
            with self.assertRaisesRegex(ValueError, "non-empty"):
                blocked_scores(bad, {"m": bad}, 1, table)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            Evaluator(good, good, torch.ones((32, 0)), block_size=4)
        with self.assertRaisesRegex(ValueError, "shape"):
            blocked_scores(good, {"m": good[:32]}, 8, table)
        with self.assertRaisesRegex(ValueError, "at least one model"):
            blocked_scores(good, {}, 8, table)
        # values: zeros, a bad entry hiding past the first rows, NaN
        with self.assertRaisesRegex(ValueError, "spins"):
            blocked_scores(torch.zeros((64, N), dtype=torch.float64), {"m": good}, 8, table)
        bad_tail = good.clone()
        bad_tail[40, 3] = 0
        with self.assertRaisesRegex(ValueError, "spins"):
            blocked_scores(good, {"m": bad_tail}, 8, table)
        nan = good.clone()
        nan[5, 5] = float("nan")
        with self.assertRaisesRegex(ValueError, "spins"):
            blocked_scores(nan, {"m": good}, 8, table)
        # dtype: integers and booleans are refused rather than cast (an int8 Gram wraps silently
        # on the CPU, 200 agreements read -56, and integer matmul is not implemented on CUDA)
        for bad in (
            good.to(torch.int8),
            torch.ones((64, N), dtype=torch.bool),
            torch.where(good > 0, 1, 255).to(torch.uint8),
        ):
            with self.assertRaisesRegex(ValueError, "floating dtype", msg=str(bad.dtype)):
                blocked_scores(bad, {"m": good}, 8, table)
        # a float32 copy with autograd history scores identically to its plain copy against a
        # float32 table, and no score carries a gradient; nothing is cast, so the float64 table
        # against float32 spins fails inside torch
        good32 = good.to(torch.float32)
        table32 = kernel_profiles(N, dtype=torch.float32)[:1]
        scores = blocked_scores(good32.clone().requires_grad_(), {"m": good32}, 8, table32).scores
        reference = blocked_scores(good32, {"m": good32}, 8, table32).scores
        self.assertTrue(torch.equal(scores, reference))
        self.assertFalse(scores.requires_grad)
        self.assertEqual(scores.dtype, torch.float32)
        with self.assertRaises(RuntimeError):
            blocked_scores(good32, {"m": good32}, 8, table)

    def test_exact_counting_guard(self):
        # the Gram and the histogram are exact integers only while they fit the dtype's mantissa:
        # float32 counts to 2^24, so a block pair of more than 4096^2 pairs is refused in float32
        # (before any Gram is formed) and accepted in float64, which counts to 2^53
        check = estimator._check_exact_counting
        check(torch.float32, 4, 4096)
        check(torch.float64, 4, 4097)
        check(torch.float64, 2**24 + 1, 2)
        with self.assertRaisesRegex(ValueError, "counts exactly"):
            check(torch.float32, 4, 4097)
        with self.assertRaisesRegex(ValueError, "counts exactly"):
            check(torch.float32, 2**24 + 1, 2)
        x32 = torch.ones((2 * 4097, 4), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "counts exactly"):
            blocked_scores(x32, {"m": x32}, 2, kernel_profiles(4, dtype=torch.float32))
        with self.assertRaisesRegex(ValueError, "counts exactly"):
            Evaluator(x32.repeat(4, 1), x32.repeat(4, 1), x32.repeat(4, 1), block_size=4097)

    def test_degenerate_zero_spread(self):
        # constant rows make every block identical, so se = 0 exactly, and the estimates are
        # exact too: x against -x flips every odd-order coefficient end to end (Delta W = +-2),
        # so the mean squared coefficient error is exactly 4 at odd k; a copy scores exactly 0
        x = torch.ones((1024, 16), dtype=torch.float64)
        ev_ = Evaluator(x, x, x, block_size=64)
        scores, se = ev_.score(-x, "test")
        self.assertEqual(float(scores[1]), 4.0)
        self.assertEqual(float(se[1]), 0.0)
        same, same_se = ev_.score(x.clone(), "test")
        self.assertEqual(float(same[1]), 0.0)
        self.assertEqual(float(same_se[1]), 0.0)
        # the score-side contract is depth-free: a deep explicit order reads exactly 4
        n = 4800
        deep = torch.ones((4, n), dtype=torch.float64)
        prof = kernels._normalized_kravchuk_matrix(n)[165][None, :]
        np.testing.assert_array_equal(blocked_scores(deep, {"neg": -deep}, 2, prof).scores.numpy(), 4.0)

    def test_deterministic_in_its_inputs(self):
        # no randomness anywhere in a score: rows are consumed in order, so two calls on the
        # same arrays agree bitwise, the same test rows give every candidate the same reference
        # values, and the calibration splits never enter a score
        x, g = mixture(512), mixture(512)
        ev_ = Evaluator(mixture(256), mixture(256), x, BS)
        a, b = ev_.score(g, "test"), ev_.score(g, "test")
        other = Evaluator(indep(256), indep(256), x, BS)
        for first, second in zip(a, b, strict=True):
            self.assertTrue(torch.equal(first, second))
        for first, second in zip(a, other.score(g, "test"), strict=True):
            self.assertTrue(torch.equal(first, second))
        for first, second in zip(ev_.uniform_score(), other.uniform_score(), strict=True):
            self.assertTrue(torch.equal(first, second))  # the shared arm, whatever the candidate
        self.assertTrue(
            torch.equal(ev_.uniform_score()[0], ev_.blocked({"u": UNIFORM}).scores[:, 0, :].mean(1))
        )


class TestEvaluator(SeededTestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = mixture(1024)
        cls.train = mixture(2048)
        cls.val = mixture(256)
        cls.gen_same = mixture(1024)
        cls.gen_indep = indep(1024)
        cls.evaluator = Evaluator(cls.train, cls.val, cls.x, block_size=BS)

    def test_structure_and_index_convention(self):
        ev_ = self.evaluator
        self.assertEqual((ev_.dim, ev_.block_size), (N, BS))
        self.assertEqual(
            (ev_.num_blocks("train"), ev_.num_blocks("val"), ev_.num_blocks("test")), (64, 8, 32)
        )
        self.assertIsNone(ev_.alpha)
        self.assertEqual(len(ev_.profile_names), N + 1)
        self.assertEqual(ev_.profile_names[:3], ("gaussian", "order 1", "order 2"))
        for name, rows in (("train", 2048), ("val", 256), ("test", 1024)):
            split = ev_.split(name)
            self.assertEqual((tuple(split.shape), split.dtype), ((rows, N), torch.float64))
            self.assertIs(split, getattr(ev_, name))
            self.assertIs(split, ev_.splits[name])
        self.assertTrue(torch.equal(ev_.test, self.x))
        with self.assertRaisesRegex(ValueError, "split must be one of"):
            ev_.split("holdout")
        # score: two tensors in the splits' dtype, one entry per profile ([0] gaussian, [k]
        # order k), bitwise the module-level blocked_scores on the same rows: one battery, one
        # code path
        scores, se = ev_.score(self.gen_same, "test")
        for values in (scores, se):
            self.assertEqual((tuple(values.shape), values.dtype), ((N + 1,), torch.float64))
        direct = blocked_scores(self.x, {"xgen": self.gen_same}, 32, kernel_profiles(N)).scores
        self.assertTrue(torch.equal(scores, direct[:, 0, :].mean(1)))
        self.assertTrue(torch.equal(se, direct[:, 0, :].std(1) / math.sqrt(32)))
        self.assertTrue(torch.equal(ev_.profiles, kernel_profiles(N)))
        self.assertIs(ev_.profiles, ev_.profiles)  # the evaluator's own table, not a copy
        # the reference: the split against the analytic uniform distribution, cached
        data, data_se = ev_.uniform_score()
        direct_u = blocked_scores(self.x, {"u": UNIFORM}, 32, kernel_profiles(N)).scores[:, 0, :]
        self.assertTrue(torch.equal(data, direct_u.mean(1)))
        self.assertTrue(torch.equal(data_se, direct_u.std(1) / math.sqrt(32)))
        self.assertTrue(torch.equal(ev_.uniform_score("test")[0], data))
        # any split can be scored against, at its own shape and block count
        sample = mixture(2048)
        on_train, _ = ev_.score(sample, "train")
        direct_t = blocked_scores(self.train, {"xgen": sample}, 64, kernel_profiles(N)).scores
        self.assertTrue(torch.equal(on_train, direct_t[:, 0, :].mean(1)))
        self.assertEqual(tuple(ev_.uniform_score("val")[0].shape), (N + 1,))
        with self.assertRaisesRegex(ValueError, "same shape as train"):
            ev_.score(self.gen_same, "train")
        with self.assertRaisesRegex(ValueError, "split must be one of"):
            ev_.score(self.gen_same, "holdout")

    def test_same_distribution_measures_zero(self):
        scores, se = self.evaluator.score(self.gen_same, "test")
        self.assertLess(float((scores / se).abs().max()), NOISE)

    def test_interaction_mismatch_measured_band_blind(self):
        # means match by construction AND the balanced mixture has zero odd-order structure,
        # so this mismatch is purely even-order: the odd orders are structurally blind to it,
        # while every entry that can see an even interaction order measures it
        scores, se = self.evaluator.score(self.gen_indep, "test")
        snr = scores / se
        for k in (1, 3):
            self.assertLess(abs(float(snr[k])), NOISE, k)
        for j in (0, 2, 4):
            self.assertGreater(float(snr[j]), 6.0, j)

    def test_uniform_score_is_the_data_spectrum(self):
        # the split against the analytic uniform distribution: the mixture's Fourier weight is
        # (1 - 2 eps)^(2k) at even k and exactly 0 at odd k; independent fair bits ARE the
        # uniform distribution, so their score estimates the same population values within noise
        data, data_se = self.evaluator.uniform_score()
        gamma2 = (1.0 - 2.0 * 0.1) ** 2
        snr = data / data_se
        for k in (2, 4):
            self.assertLess(abs(float(data[k]) - gamma2**k), 5 * float(data_se[k]), k)
            self.assertGreater(float(snr[k]), 10.0, k)
        for k in (1, 3):
            self.assertLess(abs(float(snr[k])), NOISE, k)
        scores, se = self.evaluator.score(self.gen_indep, "test")
        spread = torch.hypot(se, data_se)
        self.assertTrue(bool(((scores - data).abs() < 5 * spread).all()))

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "same shape"):  # fewer rows
            self.evaluator.score(self.gen_same[:512], "test")
        with self.assertRaisesRegex(ValueError, "same shape"):  # fewer columns
            self.evaluator.score(self.gen_same[:, :-1], "test")
        with self.assertRaisesRegex(ValueError, "spins"):
            self.evaluator.score(torch.zeros((1024, N), dtype=torch.float64), "test")
        with self.assertRaisesRegex(ValueError, "whole number of blocks"):
            Evaluator(self.train, self.val, self.x[:1000], BS)
        with self.assertRaisesRegex(ValueError, "too few blocks: test"):
            Evaluator(self.train, self.val, self.x[:128], BS)
        with self.assertRaisesRegex(ValueError, "too few blocks: val"):  # every split is blocked
            Evaluator(self.train, self.val[:64], self.x, BS)
        for bad in (0, 1):
            with self.assertRaisesRegex(ValueError, "block_size", msg=repr(bad)):
                Evaluator(self.train, self.val, self.x, bad)
        with self.assertRaisesRegex(ValueError, "test must be spins"):
            Evaluator(self.train, self.val, torch.zeros((1024, N), dtype=torch.float64), BS)
        with self.assertRaisesRegex(ValueError, "train must be spins"):
            Evaluator(torch.zeros((2048, N), dtype=torch.float64), self.val, self.x, BS)
        with self.assertRaisesRegex(ValueError, "val must be spins .* floating dtype"):
            Evaluator(self.train, self.val.to(torch.int8), self.x, BS)
        with self.assertRaisesRegex(ValueError, "val must have dim"):
            Evaluator(self.train, self.val[:, :-1], self.x, BS)
        with self.assertRaisesRegex(ValueError, "train must be a non-empty"):
            Evaluator(self.train[:0], self.val, self.x, BS)

    def test_alpha_row_is_profile_zero(self):
        # holding kernel weights swaps row 0 of the battery from gaussian to the alpha kernel
        # (alpha @ the order rows) in the table, in every score, and in the reference; its
        # per-block scores are, by linearity, the same combination of the order rows' per-block
        # scores, and agree with scoring the row directly; the order rows are untouched
        alpha = torch.rand(N, dtype=torch.float64) + 0.05
        ev_ = Evaluator(self.train, self.val, self.x, BS, alpha=alpha)
        self.assertTrue(torch.equal(ev_.alpha, alpha))  # as given: nothing is normalized
        given = alpha.clone()
        alpha.zero_()  # ...in a copy, so the caller's tensor cannot change the row afterwards
        self.assertTrue(torch.equal(ev_.alpha, given))
        alpha = given
        self.assertEqual(ev_.profile_names[0], "alpha")
        profiles = ev_.profiles
        self.assertEqual(tuple(profiles.shape), (N + 1, N + 1))
        self.assertTrue(torch.equal(profiles[1:], kernel_profiles(N)[1:]))
        torch.testing.assert_close(profiles[0], ev_.alpha @ profiles[1:], rtol=0, atol=0)
        # the value at distance 0 is the weights' sum, not 1
        self.assertAlmostEqual(float(profiles[0, 0]), float(alpha.sum()), places=12)
        scores, se = ev_.score(self.gen_indep, "test")
        blocked = ev_.blocked({"xgen": self.gen_indep}).scores
        self.assertEqual(tuple(blocked.shape), (N + 1, 1, 32))
        combined = torch.einsum("k,kmb->mb", ev_.alpha, blocked[1:])
        torch.testing.assert_close(blocked[0], combined, rtol=1e-12, atol=1e-15)
        direct = blocked_scores(self.x, {"xgen": self.gen_indep}, 32, profiles[:1]).scores[0, 0]
        torch.testing.assert_close(blocked[0, 0], direct, rtol=1e-12, atol=1e-13)
        self.assertAlmostEqual(float(scores[0]), float((ev_.alpha * scores[1:]).sum()), places=12)
        plain_scores, plain_se = self.evaluator.score(self.gen_indep, "test")
        self.assertTrue(torch.equal(scores[1:], plain_scores[1:]))
        self.assertTrue(torch.equal(se[1:], plain_se[1:]))
        data = ev_.uniform_score()[0]
        self.assertAlmostEqual(float(data[0]), float((ev_.alpha * data[1:]).sum()), places=12)
        self.assertTrue(torch.equal(data[1:], self.evaluator.uniform_score()[0][1:]))
        # the weights are a property: None returns row 0 to gaussian, re-setting holds the new
        # weights as given (unit weights make row 0 the sum of the order rows, not their mean)
        ev_.alpha = None
        self.assertEqual(ev_.profile_names[0], "gaussian")
        self.assertTrue(torch.equal(ev_.score(self.gen_indep, "test")[0], plain_scores))
        ev_.alpha = torch.ones(N, dtype=torch.float64)
        self.assertTrue(torch.equal(ev_.alpha, torch.ones(N, dtype=torch.float64)))
        flat = ev_.uniform_score()[0]
        self.assertAlmostEqual(float(flat[0]), float(flat[1:].sum()), places=12)
        # all the weight on one order: the alpha row is that order row, at the weight's scale
        one = torch.zeros(N, dtype=torch.float64)
        one[1] = 3.0
        ev_.alpha = one
        one_scores = ev_.score(self.gen_indep, "test")[0]
        torch.testing.assert_close(one_scores[0], 3.0 * one_scores[2], rtol=1e-15, atol=0)
        torch.testing.assert_close(ev_.profiles[0], 3.0 * ev_.profiles[2], rtol=1e-15, atol=0)

    def test_alpha_validation(self):
        ev_ = Evaluator(self.train, self.val, self.x, BS)
        for label, bad in {
            "2-D": torch.ones((2, N)),
            "empty": torch.ones(0),
            "scalar": torch.tensor(1.0),
            "short": torch.ones(N - 1),
        }.items():
            with self.assertRaisesRegex(ValueError, "alpha", msg=label):
                ev_.alpha = bad
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            ev_.alpha = -torch.ones(N)
        with self.assertRaisesRegex(ValueError, "positive"):
            ev_.alpha = torch.zeros(N)
        for value in (math.nan, math.inf):
            with self.assertRaisesRegex(ValueError, "finite", msg=repr(value)):
                ev_.alpha = torch.full((N,), value)
        self.assertIsNone(ev_.alpha)  # a rejected assignment leaves the weights unchanged
        with self.assertRaisesRegex(ValueError, "alpha"):
            Evaluator(self.train, self.val, self.x, BS, alpha=torch.ones(3))
        # array-likes are not converted: alpha must be a tensor, kept in its own dtype and device
        with self.assertRaisesRegex(ValueError, "alpha must be a 1-D tensor"):
            ev_.alpha = [1.0] * N
        ev_.alpha = torch.ones(N, dtype=torch.float64)
        self.assertEqual(ev_.alpha.dtype, torch.float64)
        # a float32 alpha is stored as given and fails inside torch when the alpha row forms
        ev_.alpha = torch.ones(N)
        self.assertEqual(ev_.alpha.dtype, torch.float32)
        with self.assertRaises(RuntimeError):
            _ = ev_.profiles

    def test_blocked_scores_many_models_against_a_split(self):
        # several models at once, the reference half cached: bitwise the module-level
        # blocked_scores on the same rows, and the split's shape is enforced
        ys = {"same": self.gen_same, "u": UNIFORM, "indep": self.gen_indep}
        got = self.evaluator.blocked(ys)
        self.assertEqual(got.model_names, ("same", "u", "indep"))
        self.assertEqual((got.num_blocks, got.block_size), (32, BS))
        direct = blocked_scores(self.x, ys, 32, kernel_profiles(N))
        self.assertTrue(torch.equal(got.scores, direct.scores))
        with self.assertRaisesRegex(ValueError, "same shape as test"):
            self.evaluator.blocked({"short": self.gen_same[:512]})
        on_val = self.evaluator.blocked({"u": UNIFORM, "m": mixture(256)}, "val")
        self.assertEqual((tuple(on_val.scores.shape), on_val.num_blocks), ((N + 1, 2, 8), 8))

    def test_high_dimension_stays_finite(self):
        # a uniform sample at 1202 spins with flat kernel weights at row 0: every entry finite
        # where the middle orders' estimates and errors underflow toward zero
        n = 1202
        uniform = torch.randint(0, 2, (1024, n), dtype=torch.float64) * 2 - 1
        alpha = torch.ones(n, dtype=torch.float64)
        ev_ = Evaluator(uniform, uniform, uniform, block_size=128, alpha=alpha)
        data, data_se = ev_.uniform_score()
        self.assertEqual(tuple(data.shape), (n + 1,))
        self.assertTrue(bool(torch.isfinite(data).all()))
        self.assertTrue(bool(torch.isfinite(data_se).all()))
        self.assertTrue(bool(torch.isfinite(ev_.profiles[0]).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_device_round_trip(self):
        # nothing is moved: tables, splits, scores, and losses live where the caller put them;
        # CPU and CUDA agree to float tolerance; a tensor on the other device fails inside torch
        x, g, cal = mixture(512), mixture(512, 0.16), mixture(256)
        on_cpu = Evaluator(cal, cal, x, block_size=BS)
        on_gpu = Evaluator(cal.cuda(), cal.cuda(), x.cuda(), block_size=BS)
        for split in ("train", "val", "test"):
            self.assertEqual(on_gpu.split(split).device.type, "cuda")
        a, b = on_cpu.score(g, "test"), on_gpu.score(g.cuda(), "test")
        self.assertEqual((b[0].device.type, b[1].device.type), ("cuda", "cuda"))
        torch.testing.assert_close(b[0].cpu(), a[0], rtol=1e-11, atol=1e-13)
        torch.testing.assert_close(
            on_gpu.uniform_score()[0].cpu(), on_cpu.uniform_score()[0], rtol=1e-11, atol=1e-13
        )
        soft = torch.tanh(torch.randn(64, N, device="cuda", dtype=torch.float64)).requires_grad_()
        loss = on_gpu.compute_loss(soft, 32)
        self.assertEqual(loss.device.type, "cuda")
        loss[0].backward()
        self.assertEqual(soft.grad.device.type, "cuda")
        self.assertEqual(on_gpu.compute_loss(g[:64].cuda(), x=x[:64].cuda()).device.type, "cuda")
        with self.assertRaises(RuntimeError):  # a CPU sample is not moved to the CUDA splits
            on_gpu.score(g, "test")
        with self.assertRaises(RuntimeError):  # a CPU batch against a CUDA reference draw
            on_gpu.compute_loss(g[:64], 32)
        mixed = Evaluator(cal, cal.cuda(), x, block_size=BS)  # constructs; fails on use
        with self.assertRaises(RuntimeError):
            mixed.uniform_score("val")


class TestCalibrate(SeededTestCase):
    """``Evaluator.calibrate``: the two-level alpha row on the zero-mean mixture.

    The mixture's spectrum is exact (``f_k = 0.64^k`` at even orders, ``0`` at odd orders), so
    the measurable set and the two weight levels have a closed form to be checked against; every
    statistical assertion has a tolerance that covers the sampling noise of the fixtures.
    """

    @classmethod
    def setUpClass(cls):
        cls.train = mixture(2048)
        cls.val = mixture(256)
        cls.x = mixture(1024)
        cls.evaluator = Evaluator(cls.train, cls.val, cls.x, block_size=BS)

    def test_rule_reproduced_by_hand_from_the_report(self):
        # the returned record is the rule's own input: the gate and the two weight levels
        # recompute from its pooled fourier weights, standard errors and floor exactly
        ev_ = self.evaluator
        cal = ev_.calibrate()
        self.assertIsInstance(cal, Calibration)
        self.assertEqual(cal.num_blocks, 64 + 8)  # train and val blocks, pooled
        self.assertEqual((cal.ratio, cal.test_blocks, cal.floor), (10.0, 32, 1e-3))
        for field in (cal.alpha, cal.fourier_weight, cal.se, cal.noise, cal.shares):
            self.assertEqual((tuple(field.shape), field.dtype), ((N,), torch.float64))
        self.assertEqual((tuple(cal.resolved.shape), cal.resolved.dtype), ((N,), torch.bool))
        # pooled fourier weights: the two splits' per-block uniform arms, concatenated
        arms = torch.cat([ev_._reference("train")[1][1:], ev_._reference("val")[1][1:]], dim=1)
        self.assertTrue(torch.equal(cal.fourier_weight, arms.mean(dim=1)))
        self.assertTrue(torch.equal(cal.se, arms.std(dim=1) / math.sqrt(72)))
        # the floor: the first val-sized rows of train against the val blocks, per order, scaled
        # to the test split's block count
        per_block = ev_.blocked({"real": ev_.train[: ev_.val.shape[0]]}, "val").scores[1:, 0, :]
        self.assertTrue(torch.equal(cal.noise, per_block.std(dim=1) / math.sqrt(32)))
        resolved = cal.fourier_weight - 3.0 * cal.se > 0
        gate = resolved & (cal.noise > 0) & (cal.fourier_weight >= 10.0 * cal.noise)
        self.assertTrue(torch.equal(cal.resolved, gate))
        m = int(gate.sum())
        self.assertTrue(0 < m < N)
        high, low = (1 - 1e-3) / m, 1e-3 / (N - m)
        levels = torch.tensor([low, high], dtype=torch.float64)
        expected = levels[gate.to(torch.long)]
        torch.testing.assert_close(cal.alpha, expected, rtol=0, atol=1e-15)
        self.assertEqual(cal.effective_floor, 1e-3)
        # the calibrated row sums to 1 (the rule's property; the setter normalizes nothing),
        # is positive, and the shares of the calibration baseline sum to 1
        self.assertAlmostEqual(float(cal.alpha.sum()), 1.0, places=12)
        self.assertGreater(float(cal.alpha.min()), 0.0)
        self.assertAlmostEqual(float(cal.shares.sum()), 1.0, places=12)
        torch.testing.assert_close(
            cal.shares,
            cal.alpha * cal.fourier_weight / (cal.alpha @ cal.fourier_weight),
            rtol=1e-12,
            atol=0,
        )
        # the weights are held: profile 0 is now the alpha row, bitwise the returned values, in
        # a tensor of the evaluator's own, so mutating the record cannot change the kernel
        self.assertEqual(ev_.profile_names[0], "alpha")
        self.assertTrue(torch.equal(ev_.alpha, cal.alpha))
        self.assertIsNot(ev_.alpha, cal.alpha)

    def test_mixture_gate_follows_the_exact_spectrum(self):
        # f_k = 0.64^k at even orders and 0 at odd orders: the low even orders carry structure
        # well above the floor and are measurable, every odd order carries none and is not
        cal = self.evaluator.calibrate()
        self.assertTrue(bool(cal.resolved[1]) and bool(cal.resolved[3]))  # orders 2 and 4
        self.assertFalse(bool(cal.resolved[0::2].any()))  # orders 1, 3, 5, ...
        m = int(cal.resolved.sum())
        on = cal.alpha[cal.resolved]
        torch.testing.assert_close(on, on[:1].expand_as(on), rtol=0, atol=1e-15)  # one level
        self.assertAlmostEqual(float(on[0]), (1 - cal.floor) / m, places=15)
        off = cal.alpha[~cal.resolved]
        torch.testing.assert_close(off, off[:1].expand_as(off), rtol=0, atol=1e-15)
        self.assertAlmostEqual(float(off[0]), cal.floor / (N - m), places=15)
        self.assertGreater(float(on[0]), float(off[0]))

    def test_forgiving_bound_and_the_two_readings(self):
        # a model that imposes odd-order structure on the mixture (cluster weights 0.65/0.35,
        # every even-order coefficient exact) pays at most 4 * floor at the unmeasurable orders
        # (exactly, per block, since every per-order score lies in [-4, 4]) and in practice at
        # most the floor, since an unmeasurable order's score is about the structure imposed
        # there, itself at most 1; its headline sits near the noise level, while fair bits (no
        # pairwise structure) read the full baseline
        ev_ = self.evaluator
        cal = ev_.calibrate()
        spurious = mixture_with_means(1024, p_plus=0.65)
        blocked = ev_.blocked({"spurious": spurious, "fair": indep(1024)}, "test")
        per_order = blocked.scores[1:]  # (N, 2, B)
        rejected = ~cal.resolved
        cost = torch.einsum("k,kb->b", cal.alpha[rejected], per_order[rejected, 0, :])
        self.assertLessEqual(float(cost.max()), 4 * cal.effective_floor + 1e-12)
        self.assertLessEqual(float(cost.mean()), cal.effective_floor)
        headline = blocked.scores[0].mean(dim=1)  # the alpha row, both models
        baseline, _ = ev_.uniform_score("test")
        self.assertLess(float(headline[0] / baseline[0]), 0.05)  # spurious means: forgiven
        self.assertAlmostEqual(float(headline[1] / baseline[0]), 1.0, delta=0.1)  # fair bits

    def test_a_real_sample_reads_at_the_floor(self):
        # real data of the test split's size against the test split: the headline is within a
        # few standard errors of zero; the measurable orders are exactly those where that
        # floor is far below the data's weight, so a sample that misses them reads the baseline
        ev_ = self.evaluator
        cal = ev_.calibrate()
        scores, se = ev_.score(ev_.train[: ev_.test.shape[0]], "test")
        self.assertLess(abs(float(scores[0])), 6.0 * float(se[0]))
        data, _ = ev_.uniform_score("test")
        self.assertGreater(float(data[0]), 10.0 * float(se[0]))
        self.assertTrue(
            bool((cal.fourier_weight[cal.resolved] >= 10 * cal.noise[cal.resolved]).all())
        )

    def test_weights_depend_on_the_calibration_splits_alone(self):
        # the same train and val with another test split of the same block count give bitwise
        # the same weights, and a second call reproduces the first (the references are cached,
        # the rule deterministic)
        first = self.evaluator.calibrate().alpha
        other = Evaluator(self.train, self.val, mixture(1024), block_size=BS)
        self.assertTrue(torch.equal(other.calibrate().alpha, first))
        self.assertTrue(torch.equal(self.evaluator.calibrate().alpha, first))

    def test_ratio_and_test_blocks_dials(self):
        # a stricter ratio keeps no more orders; the floor scales as 1 / sqrt(test_blocks), so
        # a test split four times as long halves it and can only widen the measurable set
        ev_ = self.evaluator
        counts = [int(ev_.calibrate(ratio=r).resolved.sum()) for r in (1.0, 10.0, 100.0)]
        self.assertEqual(counts, sorted(counts, reverse=True))
        base = ev_.calibrate(test_blocks=32)
        longer = ev_.calibrate(test_blocks=128)
        torch.testing.assert_close(longer.noise * 2.0, base.noise, rtol=1e-12, atol=0)
        self.assertGreaterEqual(int(longer.resolved.sum()), int(base.resolved.sum()))
        with self.assertWarnsRegex(RuntimeWarning, "no order is measurable"):
            flat = ev_.calibrate(ratio=1e12)
        torch.testing.assert_close(
            flat.alpha, torch.full((N,), 1.0 / N, dtype=torch.float64), rtol=0, atol=1e-15
        )
        self.assertEqual(flat.effective_floor, 1.0)

    def test_eps_sets_the_floor(self):
        cal = self.evaluator.calibrate(eps=0.1)
        self.assertEqual((cal.floor, cal.effective_floor), (0.1, 0.1))
        self.assertAlmostEqual(float(cal.alpha[~cal.resolved].sum()), 0.1, places=12)
        self.assertAlmostEqual(float(cal.alpha[cal.resolved].sum()), 0.9, places=12)
        self.assertGreaterEqual(float(cal.alpha.min()), 0.1 / N)
        # eps moves no gate
        self.assertTrue(torch.equal(cal.resolved, self.evaluator.calibrate().resolved))

    def test_validation_leaves_the_weights_untouched(self):
        ev_ = Evaluator(self.train, self.val, self.x, block_size=BS)
        for kwargs in (
            {"ratio": -0.5},
            {"ratio": 0.0},
            {"ratio": math.nan},
            {"ratio": math.inf},
            {"eps": 0.0},
            {"eps": 1.0},
            {"eps": -1.0},
            {"test_blocks": 0},
        ):
            with self.assertRaises(ValueError, msg=str(kwargs)):
                ev_.calibrate(**kwargs)
        self.assertIsNone(ev_.alpha)
        short = Evaluator(mixture(256), mixture(512), self.x, block_size=BS)  # train < val
        with self.assertRaises(ValueError):
            short.calibrate()
        self.assertIsNone(short.alpha)

    def test_nothing_resolved_falls_back_to_flat_with_a_warning(self):
        # every block a full enumeration of {-1, +1}^6: the within-block score of every order
        # kernel against uniform is exactly -1 / (2^6 - 1) in every block, so nothing clears the
        # resolution gate, the standard errors vanish, and the rule must fall back to flat
        # weights, with a warning, since flat weights here are a default, not evidence
        n = 6
        states = torch.tensor(
            [[1.0 if (i >> b) & 1 else -1.0 for b in range(n)] for i in range(1 << n)],
            dtype=torch.float64,
        )

        def enumerations(blocks: int) -> torch.Tensor:
            return torch.cat([states[torch.randperm(1 << n)] for _ in range(blocks)])

        ev_ = Evaluator(enumerations(8), enumerations(8), enumerations(8), block_size=1 << n)
        with self.assertWarnsRegex(RuntimeWarning, "no order is measurable"):
            cal = ev_.calibrate()
        self.assertFalse(bool(cal.resolved.any()))
        torch.testing.assert_close(
            cal.fourier_weight, torch.full((n,), -1.0 / 63, dtype=torch.float64), rtol=0, atol=1e-14
        )
        self.assertEqual(float(cal.se.max()), 0.0)
        torch.testing.assert_close(
            cal.alpha, torch.full((n,), 1.0 / n, dtype=torch.float64), rtol=0, atol=1e-15
        )
        self.assertEqual(cal.effective_floor, 1.0)
        self.assertTrue(bool(torch.isnan(cal.shares).all()))  # no positive baseline to share
        self.assertEqual(ev_.profile_names[0], "alpha")


class TestLoss(SeededTestCase):
    """``compute_loss``: the histogram estimator at hard spins, a smooth extension at soft ones."""

    @classmethod
    def setUpClass(cls):
        cls.x = mixture(1024)
        cls.evaluator = Evaluator(mixture(2048), mixture(256), cls.x, block_size=BS)

    def test_hard_spins_reproduce_the_histogram_estimator(self):
        # single block: the plain U-statistic over all pairs; block_size: the blocked paired
        # estimator; both equal blocked_scores on the same rows to rounding, profile by profile.
        # The loss runs in the batch's dtype: a float32 evaluator reads the same numbers to
        # float32 rounding, an integer batch is refused rather than cast, and a batch in another
        # dtype than the reference fails inside torch
        ev_ = self.evaluator
        y, xb = mixture(256), ev_.train[:256]
        loss = ev_.compute_loss(y, None, x=xb)
        self.assertEqual((tuple(loss.shape), loss.dtype), ((N + 1,), torch.float64))
        single = blocked_scores(xb, {"y": y}, 1, ev_.profiles).scores[:, 0, 0]
        torch.testing.assert_close(loss, single, rtol=0, atol=1e-8)
        blocked = ev_.compute_loss(y, 32, x=xb)
        eight = blocked_scores(xb, {"y": y}, 8, ev_.profiles).scores[:, 0, :].mean(1)
        torch.testing.assert_close(blocked, eight, rtol=0, atol=1e-8)
        self.assertFalse(loss.requires_grad)  # no autograd history in, none out
        ev32 = Evaluator(ev_.train.float(), ev_.val.float(), self.x.float(), BS)
        loss32 = ev32.compute_loss(y.float(), 32, x=xb.float())
        self.assertEqual(loss32.dtype, torch.float32)
        torch.testing.assert_close(loss32.double(), blocked, rtol=1e-4, atol=1e-5)
        with self.assertRaisesRegex(ValueError, "floating dtype"):
            ev_.compute_loss(y.to(torch.int8), 32, x=xb)
        with self.assertRaises(RuntimeError):
            ev_.compute_loss(y.float(), 32, x=xb)

    def test_alpha_row_is_profile_zero_of_the_loss(self):
        # with kernel weights held, loss[0] is alpha @ loss[1:] and equals the direct estimate of
        # the alpha row; without them, loss[0] is the Gaussian kernel's U-statistic computed
        # straight from the Grams
        ev_ = Evaluator(self.evaluator.train, self.evaluator.val, self.x, BS)
        y, xb = mixture(256), ev_.train[:256]
        loss = ev_.compute_loss(y, 64, x=xb)
        sigma = math.sqrt(N)

        def gaussian_mean(a: torch.Tensor, b: torch.Tensor, within: bool) -> torch.Tensor:
            kernel = torch.exp(-((N - a @ b.T) / 2) / sigma)
            if within:
                kernel = kernel - torch.diag(torch.diagonal(kernel))
                return kernel.sum() / (a.shape[0] * (a.shape[0] - 1))
            return kernel.mean()

        direct = torch.stack(
            [
                gaussian_mean(xb[b], xb[b], True)
                + gaussian_mean(y[b], y[b], True)
                - 2 * gaussian_mean(xb[b], y[b], False)
                for b in (slice(i * 64, (i + 1) * 64) for i in range(4))
            ]
        ).mean()
        torch.testing.assert_close(loss[0], direct, rtol=1e-10, atol=1e-12)
        alpha = torch.rand(N, dtype=torch.float64) + 0.1
        ev_.alpha = alpha
        loss_a = ev_.compute_loss(y, 64, x=xb)
        torch.testing.assert_close(loss_a[0], ev_.alpha @ loss_a[1:], rtol=1e-12, atol=1e-14)
        torch.testing.assert_close(loss_a[1:], loss[1:], rtol=0, atol=0)  # orders untouched
        row = blocked_scores(xb, {"y": y}, 4, ev_.profiles).scores[0, 0, :].mean()
        torch.testing.assert_close(loss_a[0], row, rtol=0, atol=1e-9)

    def test_soft_spins_are_differentiable(self):
        # a tanh output in (-1, 1) gets a finite per-profile loss with finite gradients; a
        # float64 batch gradchecks end to end through the Grams, the sweep, and the Gaussian
        ev_ = self.evaluator
        soft = torch.tanh(torch.randn(64, N, dtype=torch.float64)).requires_grad_()
        loss = ev_.compute_loss(soft, 32)
        self.assertEqual((tuple(loss.shape), loss.dtype), ((N + 1,), torch.float64))
        self.assertTrue(loss.requires_grad)
        self.assertTrue(bool(torch.isfinite(loss).all()))
        loss[2].backward()
        self.assertEqual(tuple(soft.grad.shape), (64, N))
        self.assertTrue(bool(torch.isfinite(soft.grad).all()))
        self.assertGreater(float(soft.grad.abs().max()), 0.0)
        soft64 = torch.tanh(torch.randn(12, N, dtype=torch.float64)).requires_grad_()
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda t: ev_.compute_loss(t, 6, x=ev_.train[:12]), (soft64,), eps=1e-6, atol=1e-6
            )
        )
        # a hard batch with autograd history gets a gradient too: the kernels are smooth in
        # the Gram, so the loss is differentiable at the corners of the cube
        hard = self.x[:64].clone().requires_grad_()
        ev_.compute_loss(hard, 32)[0].backward()
        self.assertTrue(bool(torch.isfinite(hard.grad).all()))

    def test_reference_draw_blocks_and_errors(self):
        ev_ = self.evaluator
        batch = mixture(64)
        # the default reference is a fresh draw of the split's rows: two calls differ, a given
        # x makes the call deterministic, and any split serves
        first, second = ev_.compute_loss(batch), ev_.compute_loss(batch)
        self.assertEqual(tuple(first.shape), (N + 1,))
        self.assertFalse(torch.equal(first, second))
        fixed = ev_.compute_loss(batch, 16, x=ev_.val[:64])
        self.assertTrue(torch.equal(fixed, ev_.compute_loss(batch, 16, x=ev_.val[:64])))
        self.assertEqual(tuple(ev_.compute_loss(batch, split="val").shape), (N + 1,))
        with self.assertRaisesRegex(ValueError, "pass x"):
            ev_.compute_loss(mixture(512), split="val")  # 512 > val's 256 rows
        with self.assertRaisesRegex(ValueError, "split must be one of"):
            ev_.compute_loss(batch, split="holdout")
        for bad in (7, 1, 0):
            with self.assertRaisesRegex(ValueError, "block_size", msg=repr(bad)):
                ev_.compute_loss(batch, bad)
        with self.assertRaisesRegex(ValueError, r"soft spins in \[-1, 1\]"):
            ev_.compute_loss(batch * 1.5)
        nan = batch.clone()
        nan[3, 3] = math.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            ev_.compute_loss(nan)
        with self.assertRaisesRegex(ValueError, "xquery must be"):
            ev_.compute_loss(batch[:, :-1])
        with self.assertRaisesRegex(ValueError, "xquery must be"):
            ev_.compute_loss(batch[:1])
        with self.assertRaisesRegex(ValueError, "generated batch's shape"):
            ev_.compute_loss(batch, x=ev_.train[:32])
        with self.assertRaisesRegex(ValueError, "x must be spins"):
            ev_.compute_loss(batch, x=torch.zeros((64, N), dtype=torch.float64))


class TestBatteryIndependence(SeededTestCase):
    def test_battery_chosen_from_the_scored_rows_biases_the_gap(self):
        # each arm's estimate is unbiased only for a battery fixed without looking at the rows
        # it scores: under IID uniform sampling with Q = P, picking the order on which one arm
        # ("base") looks best reads that arm low on the chosen entry and biases the paired
        # gap upward (a mean standardized gap near +0.7 noise units at B = 8 blocks), while
        # the same rule run on a disjoint calibration split and a fixed order both stay at
        # zero within noise. Monte Carlo.
        n, num_blocks, r, trials = 16, 8, 16, 600
        m = num_blocks * r
        table = kernel_profiles(n)[1:7]  # orders 1..6

        def uniform(rows: int) -> torch.Tensor:
            return torch.randint(0, 2, (rows, n), dtype=torch.float64) * 2 - 1

        z: dict[str, list[float]] = {"scored": [], "calibration": [], "fixed": []}
        for _ in range(trials):
            scored = blocked_scores(
                uniform(m), {"gen": uniform(m), "base": uniform(m)}, num_blocks, table
            )
            cal = blocked_scores(uniform(m), {"cal": uniform(m)}, num_blocks, table)
            by_base = int(scored.scores[:, 1, :].mean(dim=1).argmin())
            by_cal = int(cal.scores[:, 0, :].mean(dim=1).argmin())
            for rule, j in (("scored", by_base), ("calibration", by_cal), ("fixed", 0)):
                d = scored.scores[j, 0] - scored.scores[j, 1]  # generated minus base
                z[rule].append(float(d.mean() / (d.std() / math.sqrt(num_blocks))))
        mean = {rule: float(np.mean(v)) for rule, v in z.items()}
        se = {rule: float(np.std(v, ddof=1) / math.sqrt(trials)) for rule, v in z.items()}
        self.assertLess(abs(mean["fixed"]), 4 * se["fixed"], (mean, se))
        self.assertLess(abs(mean["calibration"]), 4 * se["calibration"], (mean, se))
        self.assertGreater(mean["scored"], 0.5, (mean, se))
        self.assertGreater(mean["scored"], 8 * se["scored"], (mean, se))


class _StubEvaluator:
    """The four things ``plot_fourier_weights`` reads off an evaluator, with chosen numbers."""

    def __init__(self, dim: int, reference: torch.Tensor, reference_se: torch.Tensor) -> None:
        self.dim = dim
        self._reference = (reference, reference_se)

    def uniform_score(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self._reference

    def split(self, name: str) -> torch.Tensor:
        return torch.zeros((16, self.dim))

    def num_blocks(self, name: str) -> int:
        return 8


@unittest.skipUnless(PathCollection is not None, "matplotlib not available")
class TestVisualize(SeededTestCase):
    def test_plot_fourier_weights_smoke(self):
        x = mixture(512)
        ev_ = Evaluator(mixture(256), mixture(256), x, BS)
        results = {"a": ev_.score(mixture(512, 0.16), "test"), "b": ev_.score(indep(512), "test")}
        fig = plot_fourier_weights(ev_, results)
        (ax,) = fig.axes
        labels = [ln.get_label() for ln in ax.lines]
        for series in ("data vs uniform", "a vs data", "b vs data"):
            self.assertIn(series, labels)
        self.assertEqual(ax.get_yscale(), "symlog")
        self.assertEqual(len(ax.get_legend().get_texts()), 3)
        self.assertIn("test split", fig.get_supxlabel())
        # no models: the reference alone
        alone = plot_fourier_weights(ev_, {})
        (ax2,) = alone.axes
        series = [ln.get_label() for ln in ax2.lines if not ln.get_label().startswith("_")]
        self.assertEqual(series, ["data vs uniform"])  # error-bar segments carry no label
        # max_order truncates the x axis to the orders kept; another split has its own reference
        cut = plot_fourier_weights(ev_, results, max_order=6)
        (line,) = [ln for ln in cut.axes[0].lines if ln.get_label() == "data vs uniform"]
        self.assertLessEqual(float(np.max(line.get_xdata())), 6.0)
        on_val = plot_fourier_weights(ev_, {"v": ev_.score(mixture(256), "val")}, "val")
        self.assertIn("val split", on_val.get_supxlabel())
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "fw.png"
            fig.savefig(out)
            self.assertGreater(out.stat().st_size, 10_000)
        # errors: too many models, a malformed result, no order kept, an unknown split
        five = {f"m{i}": results["a"] for i in range(5)}
        with self.assertRaisesRegex(ValueError, "at most 4"):
            plot_fourier_weights(ev_, five)
        with self.assertRaisesRegex(ValueError, "dim \\+ 1"):
            plot_fourier_weights(ev_, {"bad": (results["a"][0][:5], results["a"][1][:5])})
        with self.assertRaisesRegex(ValueError, "max_order"):
            plot_fourier_weights(ev_, results, max_order=0)
        with self.assertRaisesRegex(ValueError, "split must be one of"):
            plot_fourier_weights(ev_, results, "holdout")

    def test_negative_estimates_drawn_on_symlog_axis(self):
        # unbiased estimates can be <= 0, which a log axis cannot place; the axis is symmetric
        # log (linear within the smallest +-2 se half-width shown, logarithmic beyond), so a
        # negative point is drawn where it falls, with its bars, and the line is unbroken
        n = 4
        mmd2 = torch.tensor([1e-2, -1e-3, 2e-2, 1e-2, 5e-3], dtype=torch.float64)
        se = torch.full((n + 1,), 1e-3, dtype=torch.float64)
        uniform = torch.tensor([1e-1, -2e-4, 0.4, 0.2, 0.1], dtype=torch.float64)
        stub = _StubEvaluator(n, uniform, se)
        fig = plot_fourier_weights(stub, {"model": (mmd2, se)})
        (ax,) = fig.axes
        self.assertEqual(ax.get_yscale(), "symlog")
        self.assertAlmostEqual(ax.yaxis.get_transform().linthresh, 2e-3)  # 2 x the smallest se
        (model,) = [ln for ln in ax.lines if ln.get_label() == "model vs data"]
        np.testing.assert_array_equal(model.get_xdata(), [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(model.get_ydata(), mmd2[1:].numpy())  # as is, no gap
        self.assertFalse([c for c in ax.collections if isinstance(c, PathCollection)])  # no marks
        lo, hi = ax.get_ylim()
        self.assertAlmostEqual(lo, -2e-3)  # the linear band is the floor of the view
        self.assertGreater(hi, 0.4 + 2e-3)
        self.assertIn("0 estimate(s) below the band", fig.get_supxlabel())
        # below the band an estimate is clamped onto the floor, its lower bar cut there
        fixed = plot_fourier_weights(stub, {"model": (mmd2, se)}, linthresh=5e-4)
        (fax,) = fixed.axes
        self.assertAlmostEqual(fax.yaxis.get_transform().linthresh, 5e-4)
        (clamped,) = [ln for ln in fax.lines if ln.get_label() == "model vs data"]
        np.testing.assert_array_equal(clamped.get_ydata(), [-5e-4, 2e-2, 1e-2, 5e-3])
        self.assertAlmostEqual(fax.get_ylim()[0], -5e-4)
        self.assertIn("1 estimate(s) below the band", fixed.get_supxlabel())
        for bad in (0.0, -1.0, float("nan")):
            with self.assertRaisesRegex(ValueError, "linthresh", msg=repr(bad)):
                plot_fourier_weights(stub, {"model": (mmd2, se)}, linthresh=bad)

    def test_plot_kernel_spectra(self):
        fig = plot_kernel_spectra(dims=(12, 16))
        self.assertEqual(len(fig.axes), 2)
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "spec.png"
            fig.savefig(path)
            self.assertGreater(path.stat().st_size, 10_000)
        # the y axis names the normalized eigenvalue (the weight one unit of order-j mismatch
        # energy receives), never a "share" of spectral mass (which carries C(n, j) as well)
        n = 48
        ax = plot_kernel_spectra(dims=(n,)).axes[0]
        self.assertNotIn("share", ax.get_ylabel())
        self.assertIn("lambda", ax.get_ylabel())
        self.assertIn("per unit", ax.get_ylabel())
        (line,) = [ln for ln in ax.lines if ln.get_label() == "gaussian"]
        j = line.get_xdata()
        rho = math.tanh(1.0 / (2.0 * math.sqrt(n)))
        per_unit = rho**j / (rho ** np.arange(1, n + 1)).sum()
        np.testing.assert_allclose(line.get_ydata(), per_unit, rtol=1e-12)
        self.assertAlmostEqual(float(line.get_ydata()[0]), 0.927956, places=6)
        mass = np.array([math.comb(n, i) * rho**i for i in range(1, n + 1)])
        mass /= mass.sum()
        self.assertAlmostEqual(float(mass[0]), 0.127151, places=6)
        self.assertEqual(int(np.argmax(mass)) + 1, 3)  # the spectral-mass mode is order 3
        (ladder,) = [ln for ln in ax.lines if ln.get_label() == f"order k, k = 1..{n}"]
        np.testing.assert_array_equal(ladder.get_ydata(), 1.0)  # single bands: exactly 1
        np.testing.assert_array_equal(ladder.get_xdata(), np.arange(1, 33))  # up to max_order
        # symmetric-log axis whose linear band clamps numerically-zero weights (default
        # threshold: the double's relative precision): gaussian is drawn at every order, down
        # to underflow, and labeled at its knee, the last order above the threshold
        self.assertEqual(ax.get_yscale(), "symlog")
        self.assertAlmostEqual(ax.yaxis.get_transform().linthresh, 1e-16)
        self.assertEqual(len(line.get_xdata()), 32)
        big = plot_kernel_spectra(dims=(4800,), max_order=4800).axes[0]
        (gaussian,) = [ln for ln in big.lines if ln.get_label() == "gaussian"]
        self.assertEqual(float(gaussian.get_ydata()[-1]), 0.0)  # underflow, on the linear band
        rho_big = math.tanh(1.0 / (2.0 * math.sqrt(4800)))
        weights_big = rho_big ** np.arange(1, 4801, dtype=np.float64)
        knee = int(np.count_nonzero(weights_big / weights_big.sum() > 1e-16))
        (label,) = [t for t in big.texts if t.get_text() == "gaussian"]
        self.assertEqual(float(label.xy[0]), float(knee))
        with self.assertRaisesRegex(ValueError, "linthresh"):
            plot_kernel_spectra(dims=(8,), linthresh=0.0)
        # one panel per dimension, each carrying exactly the two fixed series: nothing on the
        # spectra plot depends on data
        two = plot_kernel_spectra(dims=(12, n))
        self.assertEqual(len(two.axes), 2)
        for ax, dim in zip(two.axes, (12, n), strict=True):
            labels = [text.get_text() for text in ax.get_legend().get_texts()]
            self.assertEqual(labels, ["gaussian", f"order k, k = 1..{dim}"])
        # the spectra are analytic: no table is built even at a dimension whose table never fits
        with mock.patch.object(
            kernels, "_normalized_kravchuk_matrix", side_effect=AssertionError("table built")
        ):
            self.assertEqual(len(plot_kernel_spectra(dims=(50_000,)).axes), 1)
        for bad in ((0,), (8.0,)):
            with self.assertRaisesRegex(ValueError, "dims"):
                plot_kernel_spectra(dims=bad)
        with self.assertRaisesRegex(ValueError, "max_order"):
            plot_kernel_spectra(dims=(8,), max_order=0)


if __name__ == "__main__":
    unittest.main()
