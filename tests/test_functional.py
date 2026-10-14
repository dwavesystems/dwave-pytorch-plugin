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
import math
import unittest

import torch

from dwave.plugins.torch.kernels import kernel_profiles
from dwave.plugins.torch.nn.functional import gumbel_spins
from dwave.plugins.torch.nn.functional import maximum_mean_discrepancy_loss as mmd_loss
from tests.helper_functions import randspins


def table_mmd(x: torch.Tensor, y: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    """The per-profile MMD estimate of hard spins by table lookup at the Hamming distances."""
    n = x.shape[1]

    def hamming(a, b):
        return ((n - a @ b.T) / 2).long()

    values = []
    for profile in table:
        kxx, kyy, kxy = profile[hamming(x, x)], profile[hamming(y, y)], profile[hamming(x, y)]
        n_x, n_y = x.shape[0], y.shape[0]
        values.append(
            (kxx.sum() - kxx.trace()) / (n_x * (n_x - 1))
            + (kyy.sum() - kyy.trace()) / (n_y * (n_y - 1))
            - 2 * kxy.mean()
        )
    return torch.stack(values)


class TestMaximumMeanDiscrepancyLoss(unittest.TestCase):
    def test_hard_spins_match_the_profile_table(self):
        # One estimate per profile: [0] the Gaussian kernel exp(-d / sqrt(n)), [k] the order-k
        # kernel; on hard spins every entry is the plain U-statistic read off the profile table,
        # for any two sample sizes
        x, y = randspins(9, 7, seed=1).double(), randspins(5, 7, seed=2).double()
        loss = mmd_loss(x, y)
        self.assertEqual((8,), tuple(loss.shape))
        torch.testing.assert_close(loss, table_mmd(x, y, kernel_profiles(7)), rtol=0, atol=1e-12)
        with self.subTest("Symmetric in its arguments"):
            torch.testing.assert_close(mmd_loss(y, x), loss, rtol=0, atol=1e-12)
        with self.subTest("A point mass against its copy scores zero at every profile"):
            # (A copy of a spread-out sample does not: the unbiased estimate's cross term
            # includes the self-pairs that the within-sample terms exclude.)
            ones = torch.ones(4, 7, dtype=torch.float64)
            torch.testing.assert_close(mmd_loss(ones, ones.clone()), torch.zeros(8, dtype=torch.float64))

    def test_hand_computed(self):
        # Two spins: the order-1 kernel is the mean spin agreement s_i . s_j / 2, the order-2
        # kernel the product agreement s_1 s_2 s_1' s_2'
        x = torch.tensor([[1.0, 1.0], [1.0, -1.0], [-1.0, -1.0]])
        y = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])
        # order 1: within x: pairs (0,1): 0, (0,2): -1, (1,2): 0 -> mean 0 - 1/3 ... computed below
        order1 = lambda a, b: (a @ b.T) / 2
        order2 = lambda a, b: (a.prod(1)[:, None] * b.prod(1)[None, :])
        expected = []
        for kernel in (order1, order2):
            kxx, kyy, kxy = kernel(x, x), kernel(y, y), kernel(x, y)
            expected.append((kxx.sum() - kxx.trace()) / 6 + (kyy.sum() - kyy.trace()) / 2 - 2 * kxy.mean())
        loss = mmd_loss(x, y)
        torch.testing.assert_close(loss[1:], torch.stack(expected))
        gaussian = lambda a, b: torch.exp(-((2 - a @ b.T) / 2) / math.sqrt(2))
        kxx, kyy, kxy = gaussian(x, x), gaussian(y, y), gaussian(x, y)
        torch.testing.assert_close(
            loss[0], (kxx.sum() - kxx.trace()) / 6 + (kyy.sum() - kyy.trace()) / 2 - 2 * kxy.mean()
        )

    def test_kernel_weights_and_blocks(self):
        x, y = randspins(8, 6, seed=3).double(), randspins(8, 6, seed=4).double()
        loss = mmd_loss(x, y)
        with self.subTest("The alpha row is the weighted sum of the order entries"):
            alpha = torch.rand(6, dtype=torch.float64, generator=torch.Generator().manual_seed(5))
            weighted = mmd_loss(x, y, alpha)
            torch.testing.assert_close(weighted[1:], loss[1:], rtol=0, atol=0)
            torch.testing.assert_close(weighted[0], alpha @ loss[1:])
        with self.subTest("Blocks are scored block against block and averaged"):
            blocked = mmd_loss(x, y, block_size=4)
            expected = torch.stack([mmd_loss(x[:4], y[:4]), mmd_loss(x[4:], y[4:])]).mean(0)
            torch.testing.assert_close(blocked, expected)
            torch.testing.assert_close(mmd_loss(x, y, block_size=8), loss)

    def test_soft_spins_are_differentiable(self):
        # Soft spins give a smooth surrogate: finite values and gradients to both samples, which
        # agree with finite differences; hard spins with autograd history get gradients too
        generator = torch.Generator().manual_seed(6)
        x = torch.tanh(torch.randn(5, 4, dtype=torch.float64, generator=generator)).requires_grad_()
        y = torch.tanh(torch.randn(3, 4, dtype=torch.float64, generator=generator)).requires_grad_()
        loss = mmd_loss(x, y)
        self.assertTrue(loss.requires_grad and loss.isfinite().all())
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: mmd_loss(a, b), (x, y), eps=1e-6, atol=1e-6))
        alpha = torch.tensor([0.1, 0.5, 0.0, 0.4], dtype=torch.float64)
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: mmd_loss(a, b, alpha), (x, y), eps=1e-6, atol=1e-6))
        hard = randspins(6, 4, seed=7).double().requires_grad_()
        mmd_loss(hard, randspins(6, 4, seed=8).double(), block_size=3)[0].backward()
        self.assertTrue(hard.grad.isfinite().all())

    def test_validation(self):
        x = torch.tensor([[1.0, -1.0], [1.0, 1.0]])
        with self.assertRaisesRegex(ValueError, "at least two samples"):
            mmd_loss(x, x[:1])
        with self.assertRaisesRegex(ValueError, "Feature shapes of x and y must match"):
            mmd_loss(x, torch.ones(2, 3))
        with self.assertRaisesRegex(ValueError, "2-D"):
            mmd_loss(x.unsqueeze(0), x)
        with self.assertRaisesRegex(ValueError, "floating dtype"):
            mmd_loss(x.long(), x)
        with self.assertRaisesRegex(ValueError, "finite"):
            mmd_loss(x, torch.tensor([[1.0, math.nan], [1.0, 1.0]]))
        with self.assertRaisesRegex(ValueError, r"soft spins in \[-1, 1\]"):
            mmd_loss(x, 2 * x)
        with self.assertRaisesRegex(ValueError, "same number of rows"):
            mmd_loss(x, torch.ones(4, 2), block_size=2)
        for bad in (1, 3, 0):
            with self.assertRaisesRegex(ValueError, "block_size"):
                mmd_loss(x, x, block_size=bad)
        with self.assertRaisesRegex(ValueError, "one weight per order"):
            mmd_loss(x, x, torch.ones(3))
        with self.assertRaisesRegex(ValueError, "positive"):
            mmd_loss(x, x, torch.zeros(2))


class TestGumbelSpins(unittest.TestCase):
    def test_shape_and_values(self):
        logits = torch.linspace(-2, 2, 24).reshape(4, 3, 2)
        spins = gumbel_spins(logits, n_samples=5)
        self.assertEqual((4, 5, 3, 2), tuple(spins.shape))
        self.assertTrue(torch.all(spins.abs() == 1))
        self.assertEqual((4, 1, 3, 2), tuple(gumbel_spins(logits).shape))

    def test_strong_logits_are_deterministic(self):
        # A logit of 20 is flipped by the Gumbel noise with probability sigmoid(-20), i.e. never
        logits = torch.tensor([[20.0, -20.0], [-20.0, 20.0]])
        spins = gumbel_spins(logits, n_samples=100)
        expected = torch.tensor([[1.0, -1.0], [-1.0, 1.0]]).unsqueeze(1).expand(-1, 100, -1)
        torch.testing.assert_close(spins, expected)

    def test_straight_through_gradient(self):
        # At a high temperature the relaxation never saturates, so every logit receives a positive
        # gradient from every sample while the forward spins are still exactly ±1
        logits = torch.zeros(3, 4, requires_grad=True)
        spins = gumbel_spins(logits, n_samples=7, tau=100.0)
        self.assertTrue(torch.all(spins.abs() == 1))
        spins.sum().backward()
        self.assertTrue(torch.all(logits.grad > 0))

if __name__ == "__main__":
    unittest.main()
