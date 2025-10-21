import unittest
from itertools import product

import torch
from parameterized import parameterized

from dwave.plugins.torch.nn import LinearBlock, Module, SkipLinear
from tests.utils import probably_unconstrained


class TestNN(unittest.TestCase):
    """The tests in this class is, generally, concerned with two characteristics of the output.
    1. Module outputs, probably, do not end with an activation function, and
    2. the output tensor shapes are as expected.
    """

    def test_Module(self):
        # Check the Module stores configs as expected.
        class MyModel(Module):
            def __init__(self, a, b, c):
                super().__init__(self, vars())
                self.a = a
                self.b = b
                self.c = c

            def forward(self, x):
                return x
        model = MyModel(a=123, b=2, c=3)
        self.assertDictEqual(model.config, {"a": 123, "b": 2, "c": 3})

        # This probably doesn't need to be a test but it should fail because `super().__init__()`
        # expects `self` and `vars()` as its two input args.
        class MyFailedModel(Module):
            def __init__(self, x, y, z):
                super().__init__()

            def forward(self, x):
                return x
        self.assertRaises(TypeError, MyFailedModel, (1, 2, 3))

    @parameterized.expand([0, 0.5, 1])
    def test_LinearBlock(self, p):
        din = 32
        dout = 177
        bs = 3
        model = LinearBlock(din, dout, p)
        x = torch.randn((bs, din))
        y = model(x)
        self.assertTupleEqual((bs, dout), tuple(y.shape))
        self.assertTrue(probably_unconstrained(y))

    def test_SkipLinear(self):
        model = SkipLinear(33, 99)
        x = torch.randn((5, 33))
        y = model(x)
        self.assertTupleEqual((5, 99), tuple(y.shape))
        self.assertTrue(probably_unconstrained(y))
        with self.subTest("Check identity for `din == dout`"):
            model = SkipLinear(123, 123)
            x = torch.randn((10, 123))
            y = model(x)
            self.assertTrue((x == y).all())


if __name__ == "__main__":
    unittest.main()
