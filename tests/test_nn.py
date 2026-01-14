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

from dwave.plugins.torch.nn import GivensRotation, LinearBlock, SkipLinear, store_config
from tests.helper_functions import model_probably_good
from tests.helper_models import NaiveGivensRotationLayer


class TestUtils(unittest.TestCase):

    def test_store_config(self):
        with self.subTest("Simple case"):

            class MyModel(torch.nn.Module):
                @store_config
                def __init__(self, a, b=1, *, x=4, y="hello"):
                    super().__init__()

            model = MyModel(a=123, x=5)
            self.assertDictEqual(
                dict(model.config),
                {"a": 123, "b": 1, "x": 5, "y": "hello", "module_name": "MyModel"},
            )

            model = MyModel(456)
            self.assertDictEqual(
                dict(model.config),
                {"a": 456, "b": 1, "x": 4, "y": "hello", "module_name": "MyModel"},
            )
        with self.subTest("Case with default args"):

            class MyModel(torch.nn.Module):
                @store_config
                def __init__(self, b=1, x=4, y="hello"):
                    super().__init__()

            model = MyModel()
            self.assertDictEqual(
                dict(model.config), {"b": 1, "x": 4, "y": "hello", "module_name": "MyModel"}
            )

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
            def __init__(self, a, b=1, *, x=4, y="hello"):
                super().__init__()

        class OuterModel(torch.nn.Module):
            @store_config
            def __init__(self, module_1, module_2=None):
                super().__init__()

        module_1 = InnerModel(a=123, x=5)
        module_2 = InnerModel(a="second", y="lol")
        model = OuterModel(module_1, module_2)
        self.assertDictEqual(
            dict(model.config),
            {"module_1": module_1.config, "module_2": module_2.config, "module_name": "OuterModel"},
        )
        self.assertDictEqual(
            dict(model.config["module_1"]),
            dict(a=123, b=1, x=5, y="hello", module_name="InnerModel"),
        )
        self.assertDictEqual(
            dict(model.config["module_2"]),
            dict(a="second", b=1, x=4, y="lol", module_name="InnerModel"),
        )


class TestOrthogonal(unittest.TestCase):

    def test_bad_initialization_parameters(self):
        with self.subTest("n less than 2"):
            with self.assertRaisesRegex(ValueError, "n must be an integer greater than 1"):
                GivensRotation(1)
            with self.assertRaisesRegex(ValueError, "n must be an integer greater than 1"):
                GivensRotation(0)
            with self.assertRaisesRegex(ValueError, "n must be an integer greater than 1"):
                GivensRotation(-5)
        with self.subTest("n not integer"):
            with self.assertRaisesRegex(ValueError, "n must be an integer greater than 1"):
                GivensRotation(3.5)
            with self.assertRaisesRegex(ValueError, "n must be an integer greater than 1"):
                GivensRotation("a string")
        with self.subTest("bias not boolean"):
            with self.assertRaisesRegex(ValueError, "bias must be a boolean"):
                GivensRotation(5, bias="another string")
            with self.assertRaisesRegex(ValueError, "bias must be a boolean"):
                GivensRotation(5, bias=1)

    @parameterized.expand([9, 10])
    def test_get_blocks_edges(self, n):
        blocks = GivensRotation._get_blocks_edges(n).tolist()
        # `blocks` is a list of lists of pairs, i.e., a list of blocks. Each block must contain
        # pairs of dimensions such that each dimension appears only once per block.
        # Also, across all blocks, each pair of dimensions must appear exactly once.
        appeared_pairs = set()
        for block in blocks:
            appeared_dims = set()
            for i, j in block:
                self.assertNotIn(i, appeared_dims)
                self.assertNotIn(j, appeared_dims)
                appeared_dims.add(i)
                appeared_dims.add(j)
                pair = (min(i, j), max(i, j))
                self.assertNotIn(pair, appeared_pairs)
                appeared_pairs.add(pair)
        # Check that all pairs appeared:
        for i in range(n):
            for j in range(i + 1, n):
                pair = (i, j)
                self.assertIn(pair, appeared_pairs)

    @parameterized.expand([(n, bias) for n in [9, 10] for bias in [True, False]])
    def test_GivensRotationLayer(self, n, bias):
        din = n
        dout = n
        model = GivensRotation(n, bias=bias)
        self.assertTrue(model_probably_good(model, (din,), (dout,)))

    @parameterized.expand([(n, bias) for n in [9, 10] for bias in [True, False]])
    def test_forward_agreement(self, n, bias):
        layer = GivensRotation(n, bias=bias).double()
        naive_layer = NaiveGivensRotationLayer(n, n, bias=bias).double()
        blocks = layer.blocks
        U_naive = naive_layer._create_rotation_matrix(layer.angles, blocks)
        U_parallel = layer._create_rotation_matrix()

        # Test that the matrices are close
        self.assertTrue(torch.allclose(U_naive, U_parallel, atol=1e-6))

        # Test orthogonality:
        I = torch.eye(n, dtype=U_parallel.dtype)
        UU_T = U_parallel @ U_parallel.T
        self.assertTrue(torch.allclose(I, UU_T, atol=1e-6))

        # Random input:
        with torch.no_grad():
            # forward pass will check consistency, so angles must be the same
            naive_layer.angles.copy_(layer.angles)
        x = torch.randn((7, n), dtype=U_parallel.dtype)  # batch size 7
        y_naive = naive_layer(x, blocks)
        y_parallel = layer(x)
        self.assertTrue(torch.allclose(y_naive, y_parallel, atol=1e-6))

    @parameterized.expand([(n, bias) for n in [9, 10] for bias in [True, False]])
    def test_backward_agreement(self, n, bias):
        layer = GivensRotation(n, bias=bias).double()
        naive_layer = NaiveGivensRotationLayer(n, n, bias=bias).double()
        blocks = layer.blocks

        with torch.no_grad():
            # forward and backward pass will check consistency, so angles must be the same
            naive_layer.angles.copy_(layer.angles)

        x = torch.randn((7, n), dtype=layer.angles.dtype)  # batch size 7

        y_naive = naive_layer(x, blocks)
        y_parallel = layer(x)

        # Define some dummy loss, e.g. closeness to the identity:
        loss_naive = torch.sum((y_naive - x) ** 2)
        loss_parallel = torch.sum((y_parallel - x) ** 2)
        loss_naive.backward()
        loss_parallel.backward()
        grad_parallel = layer.angles.grad
        grad_naive = naive_layer.angles.grad
        self.assertTrue(torch.allclose(grad_naive, grad_parallel, atol=1e-6))


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
        self.assertTrue(model_probably_good(model, (din,), (dout,)))

    def test_SkipLinear_identity(self):
        # The skip linear function behaves as an identity function when the input dimension and
        # output dimension are equal, and so we test for this.
        dim = 123
        model = SkipLinear(dim, dim)
        x = torch.randn((dim,))
        y = model(x)
        self.assertTrue((x == y).all())
        self.assertTrue(model_probably_good(model, (dim,), (dim,)))


if __name__ == "__main__":
    unittest.main()
