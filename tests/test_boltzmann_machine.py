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

from dimod import BinaryQuadraticModel, SampleSet, SPIN, IdentitySampler
from dwave.system.temperatures import maximum_pseudolikelihood_temperature as mple

from dwave.plugins.torch.boltzmann_machine import GRBM


class TestGraphRestrictedBoltzmannMachine(unittest.TestCase):
    def setUp(self) -> None:
        # Create a triangle graph with an additional dangling vertex
        self.nodes = list("abcd")
        self.edges = [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]
        self.n = 4

        # Manually set the parameter weights for testing
        dtype = torch.float32
        h = [0.0, 1, 2, 3]

        bm = GRBM(self.nodes, self.edges)
        bm.h.data = torch.tensor(h, dtype=dtype)
        bm.J.data = torch.tensor([1, 2, 3, 6], dtype=dtype)

        self.bm = bm

        self.ones = torch.ones(4).unsqueeze(0)
        self.mones = -torch.ones(4).unsqueeze(0)
        self.pmones = torch.tensor([[1, -1, 1, -1]], dtype=dtype)
        self.mpones = torch.tensor([[-1, 1, -1, 1]], dtype=dtype)

        self.sample_1 = torch.vstack([self.ones, self.ones, self.ones, self.pmones])
        self.sample_2 = torch.vstack([self.ones, self.ones, self.ones, self.mpones])
        return super().setUp()

    def test_forward(self):
        with self.subTest("Manually-computed energies"):
            self.assertEqual(18, self.bm(self.ones).item())
            self.assertEqual(6, self.bm(self.mones).item())
            self.assertEqual(-10, self.bm(self.pmones).item())
            self.assertEqual(-6, self.bm(self.mpones).item())
            self.assertListEqual([18, 18, 18, -10], self.bm(self.sample_1).tolist())

        with self.subTest(
            "Arbitrary-valued weights and spins should match dimod.BQM energy"
        ):
            self.bm.h.data = torch.linspace(-412, 23, 4)
            new_J = torch.linspace(-0.4, 4, 4**2)
            self.bm.J.data = new_J[: len(self.bm.J)]

            bqm = BinaryQuadraticModel.from_ising(*self.bm.ising(1))

            fake_spins = 1.0 * torch.arange(1, 5).unsqueeze(0)

            en_bqm = bqm.energies((fake_spins.numpy(), bqm.variables)).item()
            en_boltz = self.bm(fake_spins).item()
            self.assertAlmostEqual(en_bqm, en_boltz, 4)

    def test_estimate_beta(self):
        s1 = self.sample_1
        s2 = self.sample_2
        s3 = torch.vstack([self.sample_2, self.sample_2])
        bqm = self.bm.bqm(1)
        self.assertEqual(
            1.0 / mple(bqm, (s1.numpy(), bqm.variables))[0],
            self.bm.estimate_beta(s1),
        )
        self.assertEqual(
            1.0 / mple(bqm, (s2.numpy(), bqm.variables))[0],
            self.bm.estimate_beta(s2),
        )
        self.assertEqual(
            1.0 / mple(bqm, (s3.numpy(), bqm.variables))[0],
            self.bm.estimate_beta(s3),
        )

        fake_spins = torch.randn_like(s3)
        self.assertEqual(
            1.0 / mple(bqm, (fake_spins.numpy(), bqm.variables))[0],
            self.bm.estimate_beta(fake_spins),
        )

    def test_pad(self):
        grbm = GRBM([0, 1, 2], [(0, 1), (0, 2), (1, 2)], [1])
        x = torch.zeros((99, 2))
        padded = grbm._pad(x)
        self.assertTrue(padded[:, 1].isnan().all())
        self.assertRaises(ValueError, self.bm._pad, x)

    def test_compute_effective_field(self):
        grbm = GRBM([0, 1, 2], [(0, 1), (0, 2), (1, 2)], [2])
        #         (0.13)
        # Model: 2 ----- 0
        #         \      |
        #  (-0.17) \     |  (-0.7)
        #           \ 1 /
        # effective field = quadratic(0,1) + quadratic(0,2) + linear(2)
        #                 = -0.13 - 0.17 + 0.4 = 0.1
        grbm.h.data = torch.tensor([-0.1, -0.2, 0.4])
        grbm.J.data = torch.tensor([-0.7, 0.13, -0.17])
        padded = torch.tensor([[-1.0, 1.0, float("nan")]])
        h_eff = grbm._compute_effective_field(padded)
        self.assertAlmostEqual(h_eff.item(), 0.1)

    def test_compute_expectation_disconnected(self):
        grbm = GRBM([0, 1, 2], [(0, 1), (0, 2), (1, 2)], [2])
        #         (0.13)
        # Model: 2 ----- 0
        #         \      |
        #  (-0.17) \     |  (-0.7)
        #           \ 1 /
        grbm.h.data = torch.tensor([-0.1, -0.2, 0.4])
        grbm.J.data = torch.tensor([-0.7, 0.13, -0.17])
        beta = 1.337
        obs = torch.tensor([[-1.0, 1.0]])
        expected = grbm._compute_expectation_disconnected(obs, beta)[0].tolist()
        self.assertListEqual(expected[:2], [-1, 1])
        # effective field = quadratic(0,1) + quadratic(0,2) + linear(2)
        #                 = -0.13 - 0.17 + 0.4 = 0.1
        # expectation = -tanh(beta*effective field) = tanh(0.1 * 1.337)
        # -tanh(0.1337) ~= -0.132909
        self.assertAlmostEqual(expected[-1], -0.132909)

    def test_sufficient_statistics(self):
        t0 = self.bm.sufficient_statistics(self.ones)
        self.assertListEqual(t0.tolist(), [[1] * 8])

        t1 = self.bm.sufficient_statistics(torch.vstack([self.ones, self.mones]))
        self.assertListEqual(t1.tolist(), [[1] * 8, [-1] * 4 + [1] * 4])

        t2 = self.bm.sufficient_statistics(self.pmones)
        self.assertEqual(t2.tolist(), [[1, -1, 1, -1, -1, 1, -1, -1]])

    def test_interactions(self):
        self.assertListEqual(
            self.bm.interactions(torch.tensor([[0.0, 3.0, 2.0, 1.0]])).tolist(),
            [[0, 0, 0, 6]],
        )
        all_ones = [[1, 1, 1, 1]]
        self.assertListEqual(self.bm.interactions(self.ones).tolist(), all_ones)
        self.assertListEqual(self.bm.interactions(self.ones).tolist(), all_ones)
        self.assertListEqual(self.bm.interactions(self.mones).tolist(), all_ones)
        mpmm = [[-1, 1, -1, -1]]
        self.assertListEqual(self.bm.interactions(self.pmones).tolist(), mpmm)
        self.assertListEqual(self.bm.interactions(self.mpones).tolist(), mpmm)

    def test_ising(self):
        h_true = torch.tensor([-3, 0, 1, 3.0])
        J_true = torch.tensor([-1, 1, 2.0, 0])

        self.bm.h.data = h_true
        self.bm.J.data = J_true

        h, J = self.bm.ising(1)
        h_list = list(h.values())
        J_list = [J[a, b] for a, b in self.edges]

        self.assertListEqual(h_list, h_true.tolist())
        self.assertListEqual(J_list, J_true.tolist())

    def test_sample_to_tensor(self):
        grbm = GRBM(list("cabd"), ["ab", "ac", "bc"])
        bogus_energy = [999] * 3
        spins_in = [[1, -1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        ss = SampleSet.from_samples((spins_in, list("dbca")), SPIN, bogus_energy)
        spins = grbm.sample_to_tensor(ss)
        self.assertTupleEqual((3, 4), tuple(spins.shape))
        self.assertIsInstance(spins, torch.Tensor)
        # Test variable ordering is respected
        self.assertListEqual(
            spins.tolist(), [[1, 1, -1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
        )

    def test_sample(self):
        grbm = GRBM(list("abcd"), [("a", "b")])
        spins = grbm.sample(
            IdentitySampler(),
            beta_correction=1,
            initial_states=([[1, 1, 1, 1], [1, 1, 1, 1], [-1, -1, 1, -1]], "abcd"),
        )
        self.assertTupleEqual((3, 4), tuple(spins.shape))
        self.assertIsInstance(spins, torch.Tensor)

    def test_objective(self):
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
            obj = grbm.objective(ones, mones, 1)
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
            self.assertEqual(-1, grbm.objective(s1, s2, 1).item())
            self.assertEqual(-1, grbm.objective(s1, s3, 1))


if __name__ == "__main__":
    unittest.main()
