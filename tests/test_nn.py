import unittest

import torch
from parameterized import parameterized

from dwave.plugins.torch.nn import LinearBlock, SkipLinear, store_config
from tests.helper_functions import model_probably_good


class TestUtils(unittest.TestCase):

    def test_store_config(self):
        with self.subTest("Simple case"):
            class MyModel(torch.nn.Module):
                @store_config
                def __init__(self, a, b=1, *, x=4, y='hello'):
                    super().__init__()

            model = MyModel(a=123, x=5)
            self.assertDictEqual(dict(model.config),
                                 {"a": 123, "b": 1, "x": 5, "y": "hello", "module_name": "MyModel"})

            model = MyModel(456)
            self.assertDictEqual(dict(model.config),
                                 {"a": 456, "b": 1, "x": 4, "y": "hello", "module_name": "MyModel"})
        with self.subTest("Case with default args"):
            class MyModel(torch.nn.Module):
                @store_config
                def __init__(self, b=1, x=4, y='hello'):
                    super().__init__()

            model = MyModel()
            self.assertDictEqual(dict(model.config),
                                 {"b": 1, "x": 4, "y": "hello", "module_name": "MyModel"})

        with self.subTest("Empty config case failed."):
            class MyModel(torch.nn.Module):
                @store_config
                def __init__(self):
                    super().__init__()

            model = MyModel()
            self.assertDictEqual(dict(model.config), {"module_name": "MyModel"})

    def test_store_config_nested(self):
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
        self.assertDictEqual(dict(model.config["module_1"]),
                             dict(a=123, b=1, x=5, y="hello", module_name="InnerModel"))
        self.assertDictEqual(dict(model.config["module_2"]),
                             dict(a="second", b=1, x=4, y="lol", module_name="InnerModel"))


class TestLinear(unittest.TestCase):
    """The tests in this class is, generally, concerned with two characteristics of the output.
    1. Module outputs, probably, do not end with an activation function, and
    2. the output tensor shapes are as expected.
    """

    @parameterized.expand([0, 0.5, 1])
    def test_LinearBlock(self, p):
        din = 32
        dout = 177
        model = LinearBlock(din, dout, p)
        self.assertTrue(model_probably_good(model, (din,), (dout,)))

    def test_SkipLinear_different_dim(self):
        din = 33
        dout = 99
        model = SkipLinear(din, dout)
        self.assertTrue(model_probably_good(model, (din,), (dout, )))

    def test_SkipLinear_identity(self):
        # The skip linear function behaves as an identity function when the input dimension and
        # output dimension are equal, and so we test for this.
        dim = 123
        model = SkipLinear(dim, dim)
        x = torch.randn((dim,))
        y = model(x)
        self.assertTrue((x == y).all())
        self.assertTrue(model_probably_good(model, (dim,), (dim, )))


if __name__ == "__main__":
    unittest.main()
