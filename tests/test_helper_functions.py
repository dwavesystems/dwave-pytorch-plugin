import unittest

import torch
from parameterized import parameterized

from dwave.plugins.torch.nn import store_config
from tests import helper_functions


class TestHelperFunctions(unittest.TestCase):

    def test_probably_unconstrained(self):
        x = torch.randn((1000, 10, 10))
        self.assertTrue(helper_functions.probably_unconstrained(x))

    @parameterized.expand([torch.nn.Sigmoid(), torch.nn.Tanh(), torch.nn.ReLU()])
    def test_probably_unconstrained_activated(self, activation):
        x = torch.randn((1000, 10, 10))
        self.assertFalse(helper_functions.probably_unconstrained(activation(x)),
                         f"Failed to flag {activation.__class__.__name__} as probably not good.")

    def test_are_all_spins(self):
        # Scalar case
        self.assertTrue(helper_functions.are_all_spins(torch.tensor([1])))
        self.assertTrue(helper_functions.are_all_spins(torch.tensor([-1])))
        self.assertFalse(helper_functions.are_all_spins(torch.tensor([0])))

        # Zeros
        self.assertFalse(helper_functions.are_all_spins(torch.tensor([0, 1])))
        self.assertFalse(helper_functions.are_all_spins(torch.tensor([0, -1])))
        self.assertFalse(helper_functions.are_all_spins(torch.tensor([0, 0])))
        # Nonzeros
        self.assertFalse(helper_functions.are_all_spins(torch.tensor([1, 1.2])))
        self.assertFalse(helper_functions.are_all_spins(-torch.tensor([1, 1.2])))

        # All spins
        self.assertTrue(helper_functions.are_all_spins(torch.tensor([-1, 1])))
        self.assertTrue(helper_functions.are_all_spins(torch.tensor([-1.0, 1.0])))

    def test_has_zeros(self):
        # Scalar
        self.assertFalse(helper_functions.has_zeros(torch.tensor([1])))
        self.assertTrue(helper_functions.has_zeros(torch.tensor([0])))
        self.assertTrue(helper_functions.has_zeros(torch.tensor([-0])))

        # Tensor
        self.assertTrue(helper_functions.has_zeros(torch.tensor([0, 1])))

    def test_has_mixed_signs(self):
        # Single entries cannot have mixed signs
        self.assertFalse(helper_functions.has_mixed_signs(torch.tensor([-0])))
        self.assertFalse(helper_functions.has_mixed_signs(torch.tensor([0])))
        self.assertFalse(helper_functions.has_mixed_signs(torch.tensor([1])))
        self.assertFalse(helper_functions.has_mixed_signs(torch.tensor([-1])))

        # Zeros are unsigned
        self.assertFalse(helper_functions.has_mixed_signs(torch.tensor([0, 0])))
        self.assertFalse(helper_functions.has_mixed_signs(torch.tensor([0, 1.2])))
        self.assertFalse(helper_functions.has_mixed_signs(torch.tensor([0, -1.2])))

        # All entries have same sign
        self.assertFalse(helper_functions.has_mixed_signs(torch.tensor([0.4, 1.2])))
        self.assertFalse(helper_functions.has_mixed_signs(-torch.tensor([0.4, 1.2])))

        # Finally!
        self.assertTrue(helper_functions.has_mixed_signs(torch.tensor([-0.1, 1.2])))

    def test_bounded_in_plus_minus_one(self):
        # Violation on one end
        self.assertFalse(helper_functions.bounded_in_plus_minus_one(torch.tensor([1.2])))
        self.assertFalse(helper_functions.bounded_in_plus_minus_one(torch.tensor([-1.2])))
        self.assertFalse(helper_functions.bounded_in_plus_minus_one(torch.tensor([1.2, 0])))
        self.assertFalse(helper_functions.bounded_in_plus_minus_one(torch.tensor([-1.2, 0])))

        # Boundary
        self.assertTrue(helper_functions.bounded_in_plus_minus_one(torch.tensor([1])))
        self.assertTrue(helper_functions.bounded_in_plus_minus_one(torch.tensor([-1])))
        self.assertTrue(helper_functions.bounded_in_plus_minus_one(torch.tensor([1, -1])))
        self.assertTrue(helper_functions.bounded_in_plus_minus_one(torch.tensor([1, 0])))
        self.assertTrue(helper_functions.bounded_in_plus_minus_one(torch.tensor([0, 1])))

        # Correct
        self.assertTrue(helper_functions.bounded_in_plus_minus_one(torch.tensor([0.5, 0.9, -0.2])))

    @parameterized.expand([[dict(a=1, x=4)], [dict(a="hello")]])
    def test_has_correct_config_common(self, kwargs):
        class MyModel(torch.nn.Module):
            @store_config
            def __init__(self, a, b=2, *, x=4, y=5):
                super().__init__()

            def forward(self, x):
                return torch.ones(5)
        model = MyModel(**kwargs)
        self.assertTrue(helper_functions.has_correct_config(model))

    def test_has_correct_config(self):
        self.assertFalse(helper_functions.has_correct_config(torch.nn.Linear(5, 3)))

    def test_shapes_match(self):
        shape = (123, 456)
        x = torch.randn(shape)
        self.assertTrue(helper_functions.shapes_match(x, shape))
        self.assertFalse(helper_functions.shapes_match(x, (1, 2, 3)))

    def test_model_probably_good(self):
        with self.subTest("Model should be good"):
            class MyModel(torch.nn.Module):
                @store_config
                def __init__(self, a, b=2, *, x=4, y=5):
                    super().__init__()

                def forward(self, x):
                    return 2*x
            self.assertTrue(helper_functions.model_probably_good(MyModel("hello"), (500, ), (500,)))

        with self.subTest("Model should be bad: config not stored"):
            class MyModel(torch.nn.Module):
                def __init__(self, a, b=2, *, x=4, y=5):
                    super().__init__()

                def forward(self, x):
                    return 2*x
            self.assertFalse(helper_functions.model_probably_good(
                MyModel("hello"), (500, ), (500,)))

        with self.subTest("Model should be bad: shape mismatch"):
            class MyModel(torch.nn.Module):
                def __init__(self, a, b=2, *, x=4, y=5):
                    super().__init__()

                def forward(self, x):
                    return torch.randn(500)
            self.assertFalse(helper_functions.model_probably_good(
                MyModel("hello"), (123, ), (123,)))

        with self.subTest("Model should be bad: constrained output"):
            class MyModel(torch.nn.Module):
                def __init__(self, a, b=2, *, x=4, y=5):
                    super().__init__()

                def forward(self, x):
                    return torch.ones_like(x)
            self.assertFalse(helper_functions.model_probably_good(
                MyModel("hello"), (123, ), (123,)))


if __name__ == "__main__":
    unittest.main()
