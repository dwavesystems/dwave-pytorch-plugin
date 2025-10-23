import unittest

import torch
from parameterized import parameterized

from dwave.plugins.torch.nn import store_config
from tests import utils


class TestUtils(unittest.TestCase):

    def test_probably_unconstrained(self):
        x = torch.randn((1000, 10, 10))
        self.assertTrue(utils.probably_unconstrained(x))

        # Activate
        self.assertFalse(utils.probably_unconstrained(x.sigmoid()))
        self.assertFalse(utils.probably_unconstrained(x.relu()))
        self.assertFalse(utils.probably_unconstrained(x.tanh()))

    def test_are_all_spins(self):
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

    def test_has_zeros(self):
        # Scalar
        self.assertFalse(utils.has_zeros(torch.tensor([1])))
        self.assertTrue(utils.has_zeros(torch.tensor([0])))
        self.assertTrue(utils.has_zeros(torch.tensor([-0])))

        # Tensor
        self.assertTrue(utils.has_zeros(torch.tensor([0, 1])))

    def test_has_mixed_signs(self):
        # Single entries cannot have mixed signs
        self.assertFalse(utils.has_mixed_signs(torch.tensor([-0])))
        self.assertFalse(utils.has_mixed_signs(torch.tensor([0])))
        self.assertFalse(utils.has_mixed_signs(torch.tensor([1])))
        self.assertFalse(utils.has_mixed_signs(torch.tensor([-1])))

        # Zeros are unsigned
        self.assertFalse(utils.has_mixed_signs(torch.tensor([0, 0])))
        self.assertFalse(utils.has_mixed_signs(torch.tensor([0, 1.2])))
        self.assertFalse(utils.has_mixed_signs(torch.tensor([0, -1.2])))

        # All entries have same sign
        self.assertFalse(utils.has_mixed_signs(torch.tensor([0.4, 1.2])))
        self.assertFalse(utils.has_mixed_signs(-torch.tensor([0.4, 1.2])))

        # Finally!
        self.assertTrue(utils.has_mixed_signs(torch.tensor([-0.1, 1.2])))

    def test_bounded_in_plus_minus_one(self):
        # Violation on one end
        self.assertFalse(utils.bounded_in_plus_minus_one(torch.tensor([1.2])))
        self.assertFalse(utils.bounded_in_plus_minus_one(torch.tensor([-1.2])))
        self.assertFalse(utils.bounded_in_plus_minus_one(torch.tensor([1.2, 0])))
        self.assertFalse(utils.bounded_in_plus_minus_one(torch.tensor([-1.2, 0])))

        # Boundary
        self.assertTrue(utils.bounded_in_plus_minus_one(torch.tensor([1])))
        self.assertTrue(utils.bounded_in_plus_minus_one(torch.tensor([-1])))
        self.assertTrue(utils.bounded_in_plus_minus_one(torch.tensor([1, -1])))
        self.assertTrue(utils.bounded_in_plus_minus_one(torch.tensor([1, 0])))
        self.assertTrue(utils.bounded_in_plus_minus_one(torch.tensor([0, 1])))

        # Correct
        self.assertTrue(utils.bounded_in_plus_minus_one(torch.tensor([0.5, 0.9, -0.2])))

    @parameterized.expand([[dict(a=1, x=4)], [dict(a="hello")]])
    def test_has_correct_config(self, kwargs):
        class MyModel(torch.nn.Module):
            @store_config
            def __init__(self, a, b=2, *, x=4, y=5):
                super().__init__()

            def forward(self, x):
                return torch.ones(5)
        model = MyModel(**kwargs)
        self.assertTrue(utils.has_correct_config(model))
        self.assertFalse(utils.has_correct_config(torch.nn.Linear(5, 3)))

    def test_shapes_match(self):
        shape = (123, 456)
        x = torch.randn(shape)
        self.assertTrue(utils.shapes_match(x, shape))
        self.assertFalse(utils.shapes_match(x, (1, 2, 3)))

    def test_model_probably_good(self):
        with self.subTest("Model should be good"):
            class MyModel(torch.nn.Module):
                @store_config
                def __init__(self, a, b=2, *, x=4, y=5):
                    super().__init__()

                def forward(self, x):
                    return 2*x
            self.assertTrue(utils.model_probably_good(MyModel("hello"), (500, ), (500,)))

        with self.subTest("Model should be bad: config not stored"):
            class MyModel(torch.nn.Module):
                def __init__(self, a, b=2, *, x=4, y=5):
                    super().__init__()

                def forward(self, x):
                    return 2*x
            self.assertFalse(utils.model_probably_good(MyModel("hello"), (500, ), (500,)))

        with self.subTest("Model should be bad: shape mismatch"):
            class MyModel(torch.nn.Module):
                def __init__(self, a, b=2, *, x=4, y=5):
                    super().__init__()

                def forward(self, x):
                    return torch.randn(500)
            self.assertFalse(utils.model_probably_good(MyModel("hello"), (123, ), (123,)))

        with self.subTest("Model should be bad: constrained output"):
            class MyModel(torch.nn.Module):
                def __init__(self, a, b=2, *, x=4, y=5):
                    super().__init__()

                def forward(self, x):
                    return torch.ones_like(x)
            self.assertFalse(utils.model_probably_good(MyModel("hello"), (123, ), (123,)))


if __name__ == "__main__":
    unittest.main()
