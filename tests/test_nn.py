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
import unittest

import torch
from parameterized import parameterized

from dwave.plugins.torch.nn import Affine, LinearBlock, SkipLinear, store_config
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


class TestAffine(unittest.TestCase):
    """Verify the Affine module correctly applies y = scale * x + shift
    and integrates properly with PyTorch's module system (state_dict, device)."""

    def test_forward(self):
        # Basic correctness: 2*x + 3
        affine = Affine(scale=2.0, shift=3.0)
        x = torch.tensor([1.0, -1.0, 0.0])
        result = affine(x)
        torch.testing.assert_close(result, torch.tensor([5.0, 1.0, 3.0]))

    def test_forward_batched(self):
        # Ensure broadcasting works over batch dimensions
        affine = Affine(scale=0.5, shift=-1.0)
        x = torch.tensor([[2.0, 4.0], [0.0, -2.0]])
        result = affine(x)
        torch.testing.assert_close(result, torch.tensor([[0.0, 1.0], [-1.0, -2.0]]))

    def test_identity(self):
        # scale=1, shift=0 should be a no-op
        affine = Affine(scale=1.0, shift=0.0)
        x = torch.tensor([1.2, 4.3, 5.0, 3.1415926535, -324234.93])
        torch.testing.assert_close(affine(x), x)

    def test_zero_scale(self):
        # scale=0 should collapse all inputs to the shift value
        x = torch.tensor([100.0, -100.0])
        with self.subTest("Zero-scale affine with offset should return offset"):
            torch.testing.assert_close(Affine(scale=0.0, shift=5.0)(x), torch.tensor([5.0, 5.0]))
        with self.subTest("Zero-scale affine with zero-offset should return 0"):
            torch.testing.assert_close(Affine(scale=0.0, shift=0.0)(x), torch.tensor([0.0, 0.0]))

    def test_state_dict(self):
        # scale and shift are registered buffers, so they must appear in state_dict
        # (critical for checkpointing and .to(device) to propagate correctly)
        affine = Affine(scale=2.0, shift=3.0)
        sd = affine.state_dict()
        self.assertIn("scale", sd)
        self.assertIn("shift", sd)
        torch.testing.assert_close(sd["scale"], torch.tensor(2.0))
        torch.testing.assert_close(sd["shift"], torch.tensor(3.0))


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
