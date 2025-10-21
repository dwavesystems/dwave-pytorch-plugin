import unittest

import torch

from tests import utils


class TestTestUtils(unittest.TestCase):

    def test_probably_unconstrained(self):
        x = torch.randn((1000, 10, 10))
        self.assertTrue(utils.probably_unconstrained(x))

        # Activate
        self.assertFalse(utils.probably_unconstrained(x.sigmoid()))
        self.assertFalse(utils.probably_unconstrained(x.relu()))
        self.assertFalse(utils.probably_unconstrained(x.tanh()))

    def test__are_all_spins(self):
        # Scalar case
        self.assertTrue(utils.are_all_spins(torch.tensor([1])))
        self.assertTrue(utils.are_all_spins(torch.tensor([-1])))
        self.assertFalse(utils.are_all_spins(torch.tensor([0])))

        # Zeros
        self.assertFalse(utils.are_all_spins(torch.tensor([0, 1])))
        self.assertFalse(utils.are_all_spins(torch.tensor([0, -1])))
        self.assertFalse(utils.are_all_spins(torch.tensor([0, 0])))
        # Nonzeros
        self.assertFalse(utils.are_all_spins(torch.tensor([1, 1.2])))
        self.assertFalse(utils.are_all_spins(-torch.tensor([1, 1.2])))

        # All spins
        self.assertTrue(utils.are_all_spins(torch.tensor([-1, 1])))
        self.assertTrue(utils.are_all_spins(torch.tensor([-1.0, 1.0])))

    def test__has_zeros(self):
        # Scalar
        self.assertFalse(utils._has_zeros(torch.tensor([1])))
        self.assertTrue(utils._has_zeros(torch.tensor([0])))
        self.assertTrue(utils._has_zeros(torch.tensor([-0])))

        # Tensor
        self.assertTrue(utils._has_zeros(torch.tensor([0, 1])))

    def test__has_mixed_signs(self):
        # Single entries cannot have mixed signs
        self.assertFalse(utils._has_mixed_signs(torch.tensor([-0])))
        self.assertFalse(utils._has_mixed_signs(torch.tensor([0])))
        self.assertFalse(utils._has_mixed_signs(torch.tensor([1])))
        self.assertFalse(utils._has_mixed_signs(torch.tensor([-1])))

        # Zeros are unsigned
        self.assertFalse(utils._has_mixed_signs(torch.tensor([0, 0])))
        self.assertFalse(utils._has_mixed_signs(torch.tensor([0, 1.2])))
        self.assertFalse(utils._has_mixed_signs(torch.tensor([0, -1.2])))

        # All entries have same sign
        self.assertFalse(utils._has_mixed_signs(torch.tensor([0.4, 1.2])))
        self.assertFalse(utils._has_mixed_signs(-torch.tensor([0.4, 1.2])))

        # Finally!
        self.assertTrue(utils._has_mixed_signs(torch.tensor([-0.1, 1.2])))

    def test__bounded_in_plus_minus_one(self):
        # Violation on one end
        self.assertFalse(utils._bounded_in_plus_minus_one(torch.tensor([1.2])))
        self.assertFalse(utils._bounded_in_plus_minus_one(torch.tensor([-1.2])))
        self.assertFalse(utils._bounded_in_plus_minus_one(torch.tensor([1.2, 0])))
        self.assertFalse(utils._bounded_in_plus_minus_one(torch.tensor([-1.2, 0])))

        # Boundary
        self.assertTrue(utils._bounded_in_plus_minus_one(torch.tensor([1])))
        self.assertTrue(utils._bounded_in_plus_minus_one(torch.tensor([-1])))
        self.assertTrue(utils._bounded_in_plus_minus_one(torch.tensor([1, -1])))
        self.assertTrue(utils._bounded_in_plus_minus_one(torch.tensor([1, 0])))
        self.assertTrue(utils._bounded_in_plus_minus_one(torch.tensor([0, 1])))

        # Correct
        self.assertTrue(utils._bounded_in_plus_minus_one(torch.tensor([0.5, 0.9, -0.2])))


if __name__ == "__main__":
    unittest.main()
