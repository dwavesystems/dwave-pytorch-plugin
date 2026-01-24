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

from dwave.plugins.torch.samplers.base import TorchSampler


class TestTorchSampler(unittest.TestCase):
    """Test TorchSampler base class."""

    def test_subclass_without_sample(self):
        """Test creating a new subclass."""

        class EmptySubClass(TorchSampler):
            """Empty subclass without any methods."""

            def something_else(self):
                pass

        with self.assertRaises(TypeError):
            EmptySubClass()  # type: ignore

    def test_simple_subclass(self):
        """Test creating a new subclass."""

        expected_sample = torch.Tensor([1, 2, 3])

        class SimpleSubClass(TorchSampler):
            """Simple subclass with a dummy sample method."""

            def sample(self, x: torch.Tensor | None = None):
                return x or expected_sample

        simple_obj = SimpleSubClass()

        torch.testing.assert_close(simple_obj.sample(), expected_sample)

    def test_parameters(self):
        """Test that parameters are correctly."""
        init_device = torch.device("cpu")

        parameters = {
            "param_0": torch.nn.Parameter(torch.Tensor([2, 4, 8], device=init_device)),
            "param_1": torch.Tensor([1, 1, 2], device=init_device),
        }

        class SubClassWithParameters(TorchSampler):
            """Simple subclass with a dummy sample method."""

            def __init__(self) -> None:
                self.param_0 = parameters["param_0"]
                self.param_1 = parameters["param_1"]

                # refresh parameters
                super().__init__(refresh=True)

            def sample(self, x: torch.Tensor | None = None):  # type: ignore
                pass

        params_obj = SubClassWithParameters()

        self.assertDictEqual(params_obj._parameters, parameters)

        with self.subTest("set device to meta"):
            params_obj = params_obj.to(torch.device("meta"))

            for p in params_obj.parameters():
                self.assertEqual(p.device, torch.device("meta"))

            self.assertIs(params_obj.param_0, list(params_obj.parameters())[0])
            self.assertIs(params_obj.param_1, list(params_obj.parameters())[1])

        with self.subTest("add new parameter on cpu"):
            cpu_param = torch.Tensor([4, 2, 0], device=torch.device("cpu"))
            setattr(params_obj, "cpu_param", cpu_param)

            # check that new param is _not_ part of parameters unless refreshed
            self.assertNotIn(cpu_param, list(params_obj.parameters()))

            # refresh parameters and check again
            params_obj.refresh_parameters()
            self.assertIn(cpu_param, list(params_obj.parameters()))

            # check that new param has different device
            self.assertEqual(params_obj._parameters["cpu_param"].device, torch.device("cpu"))
            self.assertEqual(params_obj._parameters["param_0"].device, torch.device("meta"))
            self.assertEqual(params_obj._parameters["param_1"].device, torch.device("meta"))

            # finally, check that setting 'params_obj' to meta device again works
            params_obj = params_obj.to(torch.device("meta"))
            for p in params_obj.parameters():
                self.assertEqual(p.device, torch.device("meta"))

    def test_module_parameters(self):
        """Test that modules are correctly set."""
        init_device = torch.device("cpu")

        param = torch.nn.Conv2d(1, 20, 5).to(init_device)

        class SubClassWithModule(TorchSampler):
            """Simple subclass with a dummy sample method."""

            def __init__(self) -> None:
                self.param = param

                # refresh parameters
                super().__init__(refresh=True)

            def sample(self, x: torch.Tensor | None = None):  # type: ignore
                pass

        module_obj = SubClassWithModule()

        self.assertEqual(type(next(module_obj.modules())), type(param))

        with self.subTest("set device to meta"):
            res_device = torch.device("meta")
            res_module = next(module_obj.modules())

            # set all modules to device, recursively setting all the
            # modules parameters to device
            module_obj = module_obj.to(res_device)

            # assert that modules parameters have set device (just check one param)
            self.assertEqual(next(res_module.parameters()).device, res_device)
