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
"""Tests of the Hamming-isotropic kernel battery: the profile tables against exact integer and
rational arithmetic, the spectra against direct enumeration, and the differentiable sums against
the tables and against finite differences."""

import math
import unittest
from decimal import Decimal, getcontext
from fractions import Fraction
from typing import Any

import numpy as np
import torch

import dwave.plugins.torch.kernels as kernels
from dwave.plugins.torch.kernels import (
    KravchukSums,
    kernel_means,
    kernel_profiles,
    uniform_kernel_mean,
)
from tests.helper_functions import enumerate_walsh


def exact_integer_rows(n: int, k_max: int) -> list[list[int]]:
    """K_k(d) for k = 0..k_max as exact integers (the raw three-term recurrence)."""
    rows = [[1] * (n + 1), [n - 2 * d for d in range(n + 1)]]
    for k in range(1, k_max):
        prev, cur = rows[k - 1], rows[k]
        rows.append(
            [((n - 2 * d) * cur[d] - (n - k + 1) * prev[d]) // (k + 1) for d in range(n + 1)]
        )
    return rows[: k_max + 1]


def exact_normalized_rows(n: int) -> np.ndarray:
    """Every R_k(d) = K_k(d) / C(n, k) from the exact integer rows, converted to float last."""
    rows = exact_integer_rows(n, n)
    return np.array([[v / math.comb(n, k) for v in row] for k, row in enumerate(rows)], dtype=float)


def exact_eigenvalue(table: Any, j: int, n: int, rows: list[list[int]]) -> Fraction:
    """Exact Walsh eigenvalue of a binary64 profile table at order j.

    lambda_j = 2^-n sum_d f(d) K_d(j), with K_d(j) = K_j(d) C(n, d) / C(n, j).
    """
    acc = sum(Fraction(float(table[d])) * rows[j][d] * math.comb(n, d) for d in range(n + 1))
    return Fraction(acc) / math.comb(n, j) / 2**n


class TestProfiles(unittest.TestCase):
    def test_table_layout_and_index_convention(self):
        # (n + 1, n + 1) float64: row 0 gaussian, row k the order-k profile, column d the value
        # at Hamming distance d; every order row has unit diagonal
        n = 8
        table = kernel_profiles(n)
        self.assertEqual(tuple(table.shape), (n + 1, n + 1))
        self.assertEqual(table.dtype, torch.float64)
        d = torch.arange(n + 1, dtype=torch.float64)
        torch.testing.assert_close(table[0], torch.exp(-d / math.sqrt(n)), rtol=1e-15, atol=0)
        np.testing.assert_array_equal(table[1:, 0].numpy(), 1.0)
        torch.testing.assert_close(kernel_profiles(np.int64(n)), table, rtol=0, atol=0)
        # dtype and device are allocation parameters: a float32 table is swept in float32
        table32 = kernel_profiles(n, dtype=torch.float32)
        self.assertEqual(table32.dtype, torch.float32)
        torch.testing.assert_close(table32.double(), table, rtol=0, atol=1e-6)
        for bad in (0, -1):
            with self.assertRaisesRegex(ValueError, "^dim must be", msg=repr(bad)):
                kernel_profiles(bad)

    def test_log_binomials_match_exact_binomials(self):
        # the lgamma route to C(dim, k) behind the uniform distance law: every binomial to
        # ~1e-12 relative where it is representable, finite past the float ceiling
        # (C(4800, 2400) ~ 1e1443)
        for dim in (1, 2, 8, 200):
            got = kernels._log_binomials(dim).exp().numpy()
            exact = np.array([math.comb(dim, k) for k in range(dim + 1)], dtype=float)
            np.testing.assert_allclose(got, exact, rtol=1e-12, err_msg=f"dim={dim}")
        big = kernels._log_binomials(4800)
        self.assertEqual(tuple(big.shape), (4801,))
        self.assertTrue(bool(torch.isfinite(big).all()))
        self.assertAlmostEqual(float(big[2400]), math.log(math.comb(4800, 2400)), delta=1e-9)

    def test_gaussian_bandwidth_is_sqrt_n(self):
        # the Gaussian kernel exp(-d_H / sigma) at sigma = sqrt(n): on spins d_H = ||x - y||^2 / 4,
        # so this is the Gaussian radial-basis-function kernel in Euclidean metric
        for n in (2, 16, 100):
            d = np.arange(n + 1, dtype=np.float64)
            np.testing.assert_allclose(
                kernel_profiles(n)[0].numpy(), np.exp(-d / math.sqrt(n)), rtol=1e-15
            )

    def test_all_degrees_all_distances_match_exact_integers(self):
        # the forward sweep run past n/2 returns rounding noise divided by a vanishing C(n, k)
        # (R_100(1) ~ 7e10 instead of -1 at n = 100); every degree at every distance matches the
        # exact big-integer values to ~1e-13 and stays inside [-1, 1]; the boundary rows have
        # elementary closed forms
        for n in (7, 8, 50, 100, 200):
            got = kernel_profiles(n).numpy()
            exact = exact_normalized_rows(n)
            np.testing.assert_allclose(got[1:], exact[1:], atol=2e-13, rtol=0, err_msg=f"n={n}")
            self.assertLessEqual(float(np.abs(got[1:]).max()), 1.0 + 1e-12)
            d = np.arange(n + 1)
            np.testing.assert_array_equal(got[n], (-1.0) ** d)  # R_n(d) = (-1)^d exactly
            np.testing.assert_allclose(got[n - 1], (-1.0) ** d * (n - 2 * d) / n, atol=1e-15)

    def test_deep_projections_no_float_ceiling(self):
        # the normalized recurrence keeps every order representable at any (n, k): these
        # degrees overflow the raw-row route (C(4800, 165) > 1.8e308)
        n = 4800
        rows = kernels._normalized_kravchuk_matrix(n)
        for k in (165, 300):
            prof = rows[k]
            self.assertEqual(float(prof[0]), 1.0)  # unit diagonal, exact
            self.assertLessEqual(float(prof.abs().max()), 1.0 + 1e-12)
            # exact big-integer cross-check at d = 1: K_k(1) = C(n-1, k) - C(n-1, k-1),
            # so R_k(1) = (n - 2k)/n
            exact = Fraction(math.comb(n - 1, k) - math.comb(n - 1, k - 1), math.comb(n, k))
            self.assertAlmostEqual(float(prof[1]), float(exact), places=12)

    def test_deep_degrees_near_n_at_large_n(self):
        # the reflection R_{n-k}(d) = (-1)^d R_k(d) serves the upper half at any n: the top
        # rows are exact, and a reflected row equals its swept partner to the sign
        n = 4800
        d = np.arange(n + 1)
        rows = kernels._normalized_kravchuk_matrix(n).numpy()
        np.testing.assert_array_equal(rows[4800], (-1.0) ** d)
        np.testing.assert_allclose(rows[4799], (-1.0) ** d * (n - 2 * d) / n, atol=1e-15)
        np.testing.assert_array_equal(rows[4635], (-1.0) ** d * rows[165])
        central = rows[2400]
        self.assertLessEqual(float(np.abs(central).max()), 1.0 + 1e-12)
        self.assertEqual(float(central[0]), 1.0)
        self.assertLess(float(np.abs(central[1::2]).max()), 1e-13)  # K_{n/2}(odd d) = 0
        k = 2400  # a central-row spot check against the exact integer at d = 2
        exact = (math.comb(n - 2, k) - 2 * math.comb(n - 2, k - 1) + math.comb(n - 2, k - 2)) / (
            math.comb(n, k)
        )
        self.assertAlmostEqual(float(central[2]), exact, places=13)

    def test_full_table_rows_are_reflections(self):
        n = 200
        table = kernel_profiles(n)
        self.assertEqual(table.numel() * table.element_size(), (n + 1) ** 2 * 8)
        d = torch.arange(n + 1, dtype=torch.float64)
        self.assertTrue(torch.equal(table[199], (-1.0) ** d * table[1]))
        self.assertLess(float(table[100][1::2].abs().max()), 1e-13)  # K_{n/2}(odd d) = 0

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_device(self):
        n = 16
        cpu, gpu = kernel_profiles(n), kernel_profiles(n, device="cuda")
        self.assertEqual(gpu.device.type, "cuda")
        torch.testing.assert_close(gpu.cpu(), cpu, rtol=1e-14, atol=1e-14)
        d = torch.rand(5, 7, dtype=torch.float64, device="cuda") * n
        self.assertEqual(kernel_means(d, n).device.type, "cuda")


class TestSpectra(unittest.TestCase):
    """Spectral identities against direct enumeration (n = 8: all 256 states and subsets)."""

    n = 8

    @classmethod
    def setUpClass(cls):
        n = cls.n
        cls.states, cls.orders, cls.chi = enumerate_walsh(n)
        cls.dh = (n - cls.states.astype(np.int32) @ cls.states.astype(np.int32).T) >> 1
        cls.table = kernel_profiles(n).numpy()  # (9, 9): row 0 gaussian, row k order k
        # order spectra by direct enumeration, the machinery-free oracle:
        # lambda_S = 2^-n sum_u f(#neg(u)) chi_S(u), one value per subset
        cls.neg = ((1 - cls.states) // 2).sum(axis=1)
        cls.lam_by_subset = np.stack([(cls.chi @ row[cls.neg]) / (1 << n) for row in cls.table])
        cls.lam = np.array(
            [[ls[cls.orders == j][0] for j in range(n + 1)] for ls in cls.lam_by_subset]
        )  # (9, 9): row j = kernel entry, column = interaction order

    def spectrum(self, profile: np.ndarray) -> np.ndarray:
        """The Walsh spectrum of a profile by enumeration, checked to be isotropic."""
        by_subset = (self.chi @ profile[self.neg]) / (1 << self.n)
        for order in range(self.n + 1):
            self.assertLess(float(np.ptp(by_subset[self.orders == order])), 1e-12)  # isotropic
        return np.array([by_subset[self.orders == j][0] for j in range(self.n + 1)])

    def test_spectra_by_enumeration_are_isotropic_and_rebuild(self):
        # lambda_S depends only on |S| (Hamming isotropy), and the kernel rebuilds from its
        # order spectrum: kappa(x, y) = sum_S lambda_{|S|} chi_S(x) chi_S(y)
        for j, row in enumerate(self.table):
            ls = self.lam_by_subset[j]
            for order in range(self.n + 1):
                self.assertLess(float(np.ptp(ls[self.orders == order])), 1e-10, j)
            rebuilt = self.chi.T @ (self.lam[j][self.orders][:, None] * self.chi)
            np.testing.assert_allclose(rebuilt, row[self.dh], atol=1e-8, err_msg=str(j))

    def test_gaussian_spectrum_is_geometric(self):
        # the closed form the spectra plot draws: lambda_j proportional to rho^j with
        # rho = tanh(1/(2 sqrt(n)))
        rho = math.tanh(1.0 / (2.0 * math.sqrt(self.n)))
        lam = self.lam[0]
        np.testing.assert_allclose(lam[2:] / lam[1:-1], rho, rtol=1e-9)

    def test_order_rows_are_explicit_kravchuk_sums_and_single_bands(self):
        n = self.n
        for k in range(1, n + 1):
            norm = math.comb(n, k)
            exact = np.array(
                [
                    sum(
                        (-1) ** i * math.comb(d, i) * math.comb(n - d, k - i)
                        for i in range(max(0, k - (n - d)), min(k, d) + 1)
                    )
                    for d in range(n + 1)
                ],
                dtype=float,
            )
            np.testing.assert_allclose(self.table[k] * norm, exact, rtol=1e-10)
            expected = np.zeros(n + 1)
            expected[k] = 1.0 / norm
            np.testing.assert_allclose(self.lam[k], expected, atol=1e-10)

    def test_propriety_dictionary(self):
        # proper <=> conditionally positive semi-definite (order >= 1 eigenvalues >= 0):
        # every entry; strictly proper <=> characteristic (all > 0): of the fixed rows gaussian
        # only, each order-k entry being deliberately blind off its band; gaussian's order-0
        # eigenvalue is positive too (universal)
        self.assertGreaterEqual(float(self.lam[:, 1:].min()), -1e-12)
        self.assertGreater(float(self.lam[0, 1:].min()), 0.0)  # characteristic
        self.assertGreater(float(self.lam[0].min()), 0.0)  # universal
        for k in range(1, self.n + 1):
            off_band = np.delete(self.lam[k], k)
            self.assertAlmostEqual(float(np.abs(off_band).max()), 0.0, places=12, msg=f"k={k}")

    def test_alpha_row_spectrum_by_enumeration(self):
        # the alpha row, sum_k alpha_k R_k, has Walsh eigenvalue alpha_k / C(n, k) at order k and
        # zero at order 0: proper for any nonnegative weights (a zero weight switches its order
        # off, blind there like the order row) and characteristic (strictly proper) exactly when
        # every weight is positive; its value at distance 0 is the weights' sum
        n = self.n
        binom = np.array([math.comb(n, k) for k in range(1, n + 1)], dtype=float)
        alpha = torch.tensor([0.4, 0.0, 0.16, 0.02, 0.07, 0.002, 0.03, 0.01], dtype=torch.float64)
        row = (alpha @ kernel_profiles(n)[1:]).numpy()
        self.assertAlmostEqual(float(row[0]), float(alpha.sum()), places=12)
        lam = self.spectrum(row)
        self.assertAlmostEqual(float(lam[0]), 0.0, places=12)  # no order-0 component
        np.testing.assert_allclose(lam[1:], alpha.numpy() / binom, rtol=1e-10, atol=1e-14)
        self.assertAlmostEqual(float(lam[2]), 0.0, places=14)  # order 2 switched off
        self.assertGreater(float(lam[1:][alpha.numpy() > 0].min()), 0.0)
        self.assertAlmostEqual(float(uniform_kernel_mean(torch.as_tensor(row))), 0.0, places=12)
        flat = (torch.ones(n, dtype=torch.float64) @ kernel_profiles(n)[1:]).numpy()
        lam_flat = self.spectrum(flat)
        np.testing.assert_allclose(lam_flat[1:], 1.0 / binom, rtol=1e-10, atol=1e-14)
        self.assertGreater(float(lam_flat[1:].min()), 0.0)

    def test_uniform_kernel_mean_is_the_order0_eigenvalue(self):
        got = uniform_kernel_mean(kernel_profiles(self.n))
        self.assertEqual(tuple(got.shape), (self.n + 1,))
        np.testing.assert_allclose(got.numpy(), self.lam[:, 0], atol=1e-14)
        single = uniform_kernel_mean(kernel_profiles(self.n)[0])
        self.assertEqual(single.ndim, 0)
        self.assertAlmostEqual(float(single), float(self.lam[0, 0]), places=14)
        self.assertAlmostEqual(
            float(single), ((1 + math.exp(-1 / math.sqrt(self.n))) / 2) ** self.n, places=14
        )
        for bad in (torch.ones((5, 1, 2)), torch.ones(1), torch.tensor(1.0)):
            with self.assertRaisesRegex(ValueError, "profiles"):
                uniform_kernel_mean(bad)


class TestExactTables(unittest.TestCase):
    """The binary64 tables' contract, pinned with exact rational arithmetic."""

    def test_binary64_table_spectrum_within_profile_error_of_single_band(self):
        # the shipped tables are NOT exactly single-band kernels in general: transformed
        # exactly, the central row at n = 200 has lambda_2 ~ -1.3e-48 and lambda_200 ~ 1.9e-46
        # against an intended band of 1/C(200, 100) = 1.1e-59, and at that degree no binary64
        # table is (the correctly rounded exact profile leaks ~2e-48 too). The contract is the
        # precision bound on _normalized_kravchuk_matrix: every entry within eps <= 2e-14 of the
        # exact rational, hence every eigenvalue within eps of the single band (lambda_j = 2^-n
        # sum_d f(d) K_d(j) with |K_d(j)| <= C(n, d)) and every score exact to 4 eps.
        for n, k in ((24, 12), (48, 24), (100, 50), (200, 100)):
            rows = exact_integer_rows(n, n)
            binom = [math.comb(n, d) for d in range(n + 1)]
            exact = [Fraction(rows[k][d], binom[k]) for d in range(n + 1)]
            table = kernel_profiles(n)[k]
            eps = max(abs(Fraction(float(v)) - e) for v, e in zip(table, exact, strict=True))
            self.assertLessEqual(eps, Fraction(2, 10**14), f"n={n} k={k}: eps={float(eps):.2e}")
            lam = [exact_eigenvalue(table, j, n, rows) for j in range(n + 1)]
            off_band = max(abs(v) for j, v in enumerate(lam) if j != k)
            self.assertLessEqual(off_band, eps, f"n={n} k={k}")  # the leakage bound
            self.assertLessEqual(abs(lam[k] - Fraction(1, binom[k])), eps, f"n={n} k={k}")
            if (n, k) == (200, 100):
                self.assertLess(min(lam), 0)  # the represented table is not PSD ...
                self.assertGreater(
                    off_band, 1e10 * Fraction(1, binom[k])
                )  # ... leaks past its band
                rounded = [float(e) for e in exact]  # the best table binary64 admits
                self.assertLess(min(exact_eigenvalue(rounded, j, n, rows) for j in (2, 200)), 0)

    def test_on_band_eigenvalue_survives_past_the_certificate(self):
        # an upper bound on the off-band error says nothing about the realized on-band error,
        # and the two have different mechanisms: by reciprocity the on-band eigenvalue is the
        # table's inner product with its own exact profile under the Binomial(n, 1/2) distance
        # law, so relative entry errors move it only relatively (exact to ~1e-16 relative at
        # every surveyed degree), while the off-band eigenvalues carry the absolute entry
        # errors and overtake the band weight only past n ~ 100 at central degrees
        two52 = Fraction(1, 2**52)
        rel: dict[tuple[int, int], Fraction] = {}
        leak: dict[tuple[int, int], Fraction] = {}
        for n, k in ((24, 12), (48, 24), (64, 32), (100, 50), (200, 50), (200, 100), (200, 150)):
            rows = exact_integer_rows(n, n)
            binom = [math.comb(n, d) for d in range(n + 1)]
            exact = [Fraction(rows[k][d], binom[k]) for d in range(n + 1)]
            table = kernel_profiles(n)[k]
            lam_k = exact_eigenvalue(table, k, n, rows)
            rel[(n, k)] = abs(lam_k * binom[k] - 1)
            self.assertLessEqual(rel[(n, k)], two52, f"n={n} k={k}: {float(rel[(n, k)]):.2e}")
            weighted = sum(
                Fraction(binom[d], 2**n) * abs(Fraction(float(table[d])) - exact[d]) * abs(exact[d])
                for d in range(n + 1)
            )
            self.assertLessEqual(abs(lam_k - Fraction(1, binom[k])), weighted, f"n={n} k={k}")
            off = max(abs(exact_eigenvalue(table, j, n, rows)) for j in range(n + 1) if j != k)
            leak[(n, k)] = off * binom[k]  # largest off-band eigenvalue, in units of the band
        self.assertAlmostEqual(float(rel[(200, 100)]) / 1.046e-16, 1.0, places=2)
        self.assertGreater(math.comb(100, 50), 5e13)  # past the certificate's threshold ...
        self.assertLess(leak[(100, 50)], Fraction(5, 100))  # ... yet the band still dominates
        self.assertGreater(leak[(100, 50)], Fraction(1, 100))
        self.assertGreater(leak[(200, 50)], 1e7)  # the crossover lies between n = 100 and 200
        self.assertGreater(leak[(200, 100)], 1e13)
        self.assertLess(leak[(24, 12)], 1e-13)

    def test_dyadic_profiles_are_exactly_single_band(self):
        # a profile whose exact values are dyadic rationals is stored exactly, and its exact
        # Walsh transform is then exactly the single band: order n (the alternating +-1 row) at
        # every n, and orders 1 / n-1 (entries in multiples of 1/n) when n is a power of two
        for n, k in ((8, 1), (8, 7), (8, 8), (16, 1), (16, 15), (5, 5), (12, 12), (2, 1)):
            rows = exact_integer_rows(n, n)
            binom = math.comb(n, k)
            table = kernel_profiles(n)[k]
            self.assertEqual(
                [Fraction(float(v)) for v in table],
                [Fraction(rows[k][d], binom) for d in range(n + 1)],
                f"n={n} k={k}: table is not the exact profile",
            )
            lam = [exact_eigenvalue(table, j, n, rows) for j in range(n + 1)]
            self.assertEqual([v for j, v in enumerate(lam) if j != k], [0] * n, f"n={n} k={k}")
            self.assertEqual(lam[k], Fraction(1, binom), f"n={n} k={k}")
        # a non-dyadic neighbour is not exact, even at tiny n: order 1 at n = 12
        rows = exact_integer_rows(12, 12)
        table = kernel_profiles(12)[1]
        self.assertNotEqual(Fraction(float(table[1])), Fraction(rows[1][1], 12))
        self.assertLess(min(exact_eigenvalue(table, j, 12, rows) for j in range(13)), 0)
        # the central degree is exact at n = 2 only: R_{n/2}(2) = -1/(n - 1) for even n and
        # R_{(n-1)/2}(1) = 1/n for odd n are non-dyadic rationals once n >= 3
        for n in range(3, 65):
            k = n // 2
            rows = exact_integer_rows(n, k)
            d, expected = (2, Fraction(-1, n - 1)) if n % 2 == 0 else (1, Fraction(1, n))
            self.assertEqual(Fraction(rows[k][d], math.comb(n, k)), expected, f"n={n}")
            self.assertNotEqual(Fraction(float(expected)), expected, f"n={n}: dyadic?")
            self.assertNotEqual(Fraction(float(kernel_profiles(n)[k][d])), expected, f"n={n}")

    def test_gaussian_table_spectrum_within_leakage(self):
        # the binary64 gaussian table is NOT positive semi-definite: transformed exactly, the
        # n = 24 table has lambda_22 = -1.28e-19 where the kernel has +1.42e-23. The table's
        # contract is the leakage bound |lambda~_j - lambda_j| <= L = 2^-n sum_d |delta_d|
        # C(n, d) <= u lambda_0 (delta_d the entry errors, u their largest relative size), so
        # scores are exact to 4 max|delta_d| and eigenvalues of either sign only past
        # j* = min{j : lambda_j < L}. 50-digit decimals for the kernel (Decimal's exp and
        # sqrt are correctly rounded).
        getcontext().prec = 50
        j_star_seen: dict[int, int] = {}
        for n in (8, 16, 24, 48, 100, 200):
            rows = exact_integer_rows(n, n)
            table = kernel_profiles(n)[0]
            sig = Decimal(n).sqrt()
            exact_f = [(-Decimal(d) / sig).exp() for d in range(n + 1)]
            delta = [abs(Decimal(float(table[d])) - f) for d, f in enumerate(exact_f)]
            u = max(dl / f for dl, f in zip(delta, exact_f, strict=True))
            self.assertLessEqual(max(delta), Decimal("1.1e-16"), f"n={n}")
            self.assertLessEqual(u, (sig + 4) * Decimal(2) ** -52, f"n={n}")  # argument rounding
            a = (-1 / sig).exp()
            lam0, rho = ((1 + a) / 2) ** n, (1 - a) / (1 + a)  # rho = tanh(1 / (2 sigma))
            leak = sum(dl * math.comb(n, d) for d, dl in enumerate(delta)) / Decimal(2) ** n
            self.assertLessEqual(leak, u * lam0, f"n={n}")
            j_star = next((j for j in range(n + 1) if lam0 * rho**j < leak), n + 1)
            j_star_seen[n] = j_star
            lam = [exact_eigenvalue(table, j, n, rows) for j in range(n + 1)]
            for j, v in enumerate(lam):
                dec = Decimal(v.numerator) / Decimal(v.denominator)
                self.assertLessEqual(abs(dec - lam0 * rho**j), leak, f"n={n} j={j}")
                if v < 0:
                    self.assertGreaterEqual(j, j_star, f"n={n}: negative eigenvalue below j*")
            if n <= 16:
                self.assertEqual(j_star, n + 1)  # every kernel eigenvalue clears the leakage
                self.assertGreater(min(lam), Fraction(0))  # ... so the table is exactly PSD
            if n == 24:
                self.assertLess(lam[22], Fraction(0))
                self.assertAlmostEqual(float(lam[22]) / -1.2805e-19, 1.0, places=3)
                self.assertGreater(lam0 * rho**22, 0)  # the kernel's eigenvalue is positive
                dec22 = Decimal(lam[22].numerator) / Decimal(lam[22].denominator)
                self.assertGreaterEqual(dec22, -4 * max(delta))  # inside the score tolerance
        # the documented leakage orders: 16 at n = 24, 14 at n = 48, 11 at n = 200
        self.assertEqual((j_star_seen[24], j_star_seen[48], j_star_seen[200]), (16, 14, 11))


class TestKravchukSums(unittest.TestCase):
    """The differentiable sweep: the tables at integer distances, a smooth extension between."""

    def setUp(self):
        torch.manual_seed(0)

    def test_sums_match_the_table_and_their_gradient(self):
        # at integer distances the differentiable sweep reproduces every table row (the
        # reflected rows through cos(pi d)), and its hand-written backward matches finite
        # differences at real distances, reflected rows included (odd and even n)
        for n in (1, 2, 3, 8, 9, 24):
            d_int = torch.arange(n + 1, dtype=torch.float64)
            sums = torch.stack([KravchukSums.apply(d_int[i : i + 1], n) for i in range(n + 1)], 1)
            torch.testing.assert_close(sums[1:], kernel_profiles(n)[1:], rtol=0, atol=1e-14)
            np.testing.assert_array_equal(sums[0].numpy(), 1.0)  # R_0 = 1
        for n in (8, 9):
            d = (torch.rand(15, dtype=torch.float64) * n).requires_grad_()
            self.assertTrue(
                torch.autograd.gradcheck(
                    lambda t, n=n: KravchukSums.apply(t, n),
                    (d,),
                    eps=1e-6,
                    atol=1e-6,
                    rtol=1e-4,
                )
            )

    def test_backward_is_autograd_through_the_unrolled_recurrence(self):
        # for an arbitrary upstream gradient the hand-written backward equals autograd run
        # through the same rows written out in plain tensor operations (the swept recurrence
        # and the cos(pi d) reflections) to rounding, sharper than the finite-difference
        # gradcheck above; both parities of n, and the partition of the row indices into swept
        # and reflected rows enumerated alongside
        def unrolled(d: torch.Tensor, n: int) -> torch.Tensor:
            half = n // 2
            rows = [torch.ones_like(d)]
            if half >= 1:
                rows.append((n - 2 * d) / n)
            for k in range(1, half):
                rows.append(((n - 2 * d) * rows[k] - k * rows[k - 1]) / (n - k))
            twist = torch.cos(math.pi * d)
            sums = torch.zeros(n + 1, dtype=torch.float64)
            for j, row in enumerate(rows):
                sums[j] = row.sum()
                if n - j > half:  # the reflection of every swept row above the middle
                    sums[n - j] = (twist * row).sum()
            return sums

        for n in (1, 2, 8, 9, 24, 25):
            half = n // 2
            written = list(range(half + 1)) + [n - j for j in range(half + 1) if n - j > half]
            self.assertEqual(sorted(written), list(range(n + 1)), n)  # every row exactly once
            d = torch.rand(97, dtype=torch.float64) * n
            g = torch.randn(n + 1, dtype=torch.float64)
            hand, plain = d.clone().requires_grad_(), d.clone().requires_grad_()
            (KravchukSums.apply(hand, n) @ g).backward()
            (unrolled(plain, n) @ g).backward()
            torch.testing.assert_close(hand.grad, plain.grad, rtol=0, atol=1e-13, msg=f"n={n}")

    def test_sums_over_the_last_axis_of_any_shape(self):
        # a (..., E) input sums over its last axis: the (n + 1, ...) result and its gradient
        # agree with the 1-D sweep row by row
        n = 9
        d = torch.rand(3, 4, 11, dtype=torch.float64) * n
        g = torch.randn(n + 1, 3, 4, dtype=torch.float64)
        batched = d.clone().requires_grad_()
        sums = KravchukSums.apply(batched, n)
        self.assertEqual((n + 1, 3, 4), tuple(sums.shape))
        (sums * g).sum().backward()
        for i in range(3):
            for j in range(4):
                single = d[i, j].clone().requires_grad_()
                values = KravchukSums.apply(single, n)
                torch.testing.assert_close(sums[:, i, j], values, rtol=0, atol=1e-13)
                (values @ g[:, i, j]).backward()
                torch.testing.assert_close(batched.grad[i, j], single.grad, rtol=0, atol=1e-13)

    def test_float32_gradient_of_reflected_rows_at_integer_distances(self):
        # the reflection's derivative cos(pi d) R'_k - pi sin(pi d) R_k must lose its sine term
        # exactly at integer d in every dtype: a float32 sin(pi * d) is off by ~3e-7 d, which at
        # n = 4800 turned the gradient of row n - 1 (exactly (-1)^d (-2 / n)) into a value of the
        # wrong sign and five times the magnitude; the angle is reduced to the nearest integer
        # first, so the sine is exactly zero there
        n = 4800
        d = torch.tensor([4704.0, 17.0, 0.0, n], dtype=torch.float32, requires_grad=True)
        sums = KravchukSums.apply(d, n)
        (sums[n - 1]).backward()
        expected = (-1.0) ** d.detach() * (-2.0 / n)
        torch.testing.assert_close(d.grad, expected, rtol=1e-6, atol=1e-9)
        d.grad = None
        KravchukSums.apply(d, n)[n].backward()  # the parity row: derivative exactly 0
        torch.testing.assert_close(d.grad, torch.zeros(4), rtol=0, atol=0)
        with self.subTest("Real distances keep the smooth extension and its gradient"):
            real = (torch.rand(9, dtype=torch.float64) * 9).requires_grad_()
            self.assertTrue(torch.autograd.gradcheck(lambda t: KravchukSums.apply(t, 9), (real,), eps=1e-6, atol=1e-6))
            unrolled = torch.cos(math.pi * real.detach()) * (9 - 2 * real.detach()) / 9  # row 8 = reflection of row 1
            torch.testing.assert_close(KravchukSums.apply(real, 9)[8], unrolled.sum(), rtol=1e-13, atol=1e-13)

    def test_kernel_means(self):
        # entry k is the mean order-k kernel over the last axis, entry 0 the Gaussian mean or
        # the alpha combination of the order means; at integer distances both read off the table
        n = 6
        table = kernel_profiles(n)
        distances = torch.tensor([[0.0, 2.0, 5.0], [6.0, 1.0, 1.0]], dtype=torch.float64)
        means = kernel_means(distances, n)
        self.assertEqual((n + 1, 2), tuple(means.shape))
        torch.testing.assert_close(means, table[:, distances.long()].mean(-1), rtol=0, atol=1e-14)
        alpha = torch.tensor([0.5, 0.0, 0.2, 0.1, 0.1, 0.1], dtype=torch.float64)
        weighted = kernel_means(distances, n, alpha)
        torch.testing.assert_close(weighted[1:], means[1:], rtol=0, atol=0)
        torch.testing.assert_close(weighted[0], alpha @ means[1:])
        with self.subTest("A real distance gives a smooth extension with a gradient"):
            d = (torch.rand(7, dtype=torch.float64) * n).requires_grad_()
            self.assertTrue(torch.autograd.gradcheck(lambda t: kernel_means(t, n, alpha), (d,), eps=1e-6, atol=1e-6))
        with self.subTest("Kernel weights are validated"):
            for bad in (torch.ones(5), torch.ones(2, 6), -torch.ones(6), torch.zeros(6), torch.full((6,), math.inf)):
                with self.assertRaisesRegex(ValueError, "alpha"):
                    kernel_means(distances, n, bad)
            with self.assertRaisesRegex(ValueError, "alpha"):
                kernel_means(distances, n, [1.0] * 6)


if __name__ == "__main__":
    unittest.main()
