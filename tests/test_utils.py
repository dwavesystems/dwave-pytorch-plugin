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

from dimod import SPIN, SampleSet, IdentitySampler

from dwave.plugins.torch.boltzmann_machine import (
    GraphRestrictedBoltzmannMachine as GRBM,
)
from dwave.plugins.torch.utils import sample_to_tensor, grbm_objective, sample


class TestUtils(unittest.TestCase):

    def test_sample_to_tensor(self):
        ss = SampleSet.from_samples([[1, -1], [1, 1], [1, 1]], SPIN, [-1, 2, 2])
        spins = sample_to_tensor(ss)
        self.assertTupleEqual((3, 2), tuple(spins.shape))
        self.assertIsInstance(spins, torch.Tensor)

    def test_sample(self):
        grbm = GRBM(list("abcd"), [("a", "b")])
        spins = sample(
            grbm,
            IdentitySampler(),
            beta_correction=1,
            initial_states=([[1, 1, 1, 1], [1, 1, 1, 1], [-1, -1, 1, -1]], "abcd"),
        )
        self.assertTupleEqual((3, 4), tuple(spins.shape))
        self.assertIsInstance(spins, torch.Tensor)

    def test_grbm_objective(self):
        # Create a triangle graph with an additional dangling vertex
        self.nodes = list("abcd")
        self.edges = [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]
        self.n = 4

        # Manually set the parameter weights for testing
        dtype = torch.float32
        h = [0.0, 1, 2, 3]

        grbm = GRBM(self.nodes, self.edges)
        grbm.h.data = torch.tensor(h, dtype=dtype)
        grbm.J.data = torch.tensor([1, 2, 3, 6], dtype=dtype)

        # Test the gradient matches
        ones = torch.ones((1, 4))
        mones = -ones
        with self.subTest("Test gradients"):
            obj = grbm_objective(grbm, ones, mones)
            obj.backward()
            t1 = grbm.sufficient_statistics(ones)
            t2 = grbm.sufficient_statistics(mones)
            grad_auto = grbm.h.grad.tolist() + grbm.J.grad.tolist()
            self.assertListEqual(grad_auto, (t1.mean(0) - t2.mean(0)).tolist())

        pmones = torch.tensor([[1, -1, 1, -1]], dtype=dtype)
        mpones = torch.tensor([[-1, 1, -1, 1]], dtype=dtype)
        with self.subTest("Test objective value matches"):
            s1 = torch.vstack([ones, ones, ones, pmones])
            s2 = torch.vstack([ones, ones, ones, mpones])
            s3 = torch.vstack([s2, s2])
            self.assertEqual(-1, grbm_objective(grbm, s1, s2).item())
            self.assertEqual(-1, grbm_objective(grbm, s1, s3))


if __name__ == "__main__":
    unittest.main()
