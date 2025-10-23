import unittest

import torch
from parameterized import parameterized

from dwave.plugins.torch.nn import LinearBlock, SkipLinear, store_config
from tests.utils import model_probably_good


class TestNN(unittest.TestCase):
    """The tests in this class is, generally, concerned with two characteristics of the output.
    1. Module outputs, probably, do not end with an activation function, and
    2. the output tensor shapes are as expected.
    """

    def test_store_config(self):
        # Simple case
        class MyModel(torch.nn.Module):
            @store_config
            def __init__(self, a, b=1, *, x=4, y='hello'):
                super().__init__()

        model = MyModel(a=123, x=5)
        self.assertDictEqual(dict(model.config),
                             {"a": 123, "b": 1, "x": 5, "y": "hello", "module_name": "MyModel"})

        # Nested case
        class InnerModel(torch.nn.Module):
            @store_config
            def __init__(self, a, b=1, *, x=4, y='hello'):
                super().__init__()

        class OuterModel(torch.nn.Module):
            @store_config
            def __init__(self, module_1, module_2=None):
                super().__init__()

        module_1 = InnerModel(a=123, x=5)
        module_2 = InnerModel(a="second", y="lol")
        model = OuterModel(module_1, module_2)
        self.assertDictEqual(dict(model.config),
                             {"module_1": module_1.config,
                              "module_2": module_2.config,
                              "module_name": "OuterModel"})

    @parameterized.expand([0, 0.5, 1])
    def test_LinearBlock(self, p):
        din = 32
        dout = 177
        model = LinearBlock(din, dout, p)
        self.assertTrue(model_probably_good(model, (din,), (dout,)))

    def test_SkipLinear(self):
        din = 33
        dout = 99
        model = SkipLinear(din, dout)
        self.assertTrue(model_probably_good(model, (din,), (dout, )))
        with self.subTest("Check identity for `din == dout`"):
            dim = 123
            model = SkipLinear(dim, dim)
            x = torch.randn((dim,))
            y = model(x)
            self.assertTrue((x == y).all())
            self.assertTrue(model_probably_good(model, (dim,), (dim, )))


if __name__ == "__main__":
    unittest.main()
