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

import networkx as nx
import torch
from dimod import BinaryQuadraticModel, IdentitySampler
from dwave.system.temperatures import maximum_pseudolikelihood_temperature

from dwave.plugins.torch.boltzmann_machine import (
    GraphRestrictedBoltzmannMachine as GRBM,
)


class TestGraphRestrictedBoltzmannMachine(unittest.TestCase):
    def setUp(self) -> None:
        # Create a triangle graph with an additional dangling vertex
        self.edges = [[0, 1], [0, 2], [0, 3], [1, 2]]
        self.a, self.b = torch.tensor(self.edges).mT
        self.G = nx.from_edgelist(self.edges)
        self.n = self.G.number_of_nodes()

        # Manually set the parameter weights for testing
        dtype = torch.float32
        h = [0.0, 1, 2, 3]

        bm = GRBM(self.n, *torch.tensor(self.edges).mT)
        bm.h.data = torch.tensor(h, dtype=dtype)
        bm.J.data = torch.tensor([1, 2, 3, 6], dtype=dtype)

        self.bm = bm
        self.bqm = BinaryQuadraticModel.from_ising(
            h,
            {
                e: j
                for e, j in zip(
                    zip(self.a.tolist(), self.b.tolist()), bm.J.data.tolist()
                )
            },
        )

        self.ones = torch.ones(4).unsqueeze(0)
        self.mones = -torch.ones(4).unsqueeze(0)
        self.pmones = torch.tensor([[1, -1, 1, -1]], dtype=dtype)
        self.mpones = torch.tensor([[-1, 1, -1, 1]], dtype=dtype)

        self.sample_1 = torch.vstack([self.ones, self.ones, self.ones, self.pmones])
        self.sample_2 = torch.vstack([self.ones, self.ones, self.ones, self.mpones])
        return super().setUp()

    def test_register_forward_pre_hook(self):
        self.bm.h_range = torch.tensor([-0.1, 0.1])
        self.bm.j_range = torch.tensor([-0.1, 0.2])
        self.assertEqual(-0.1 * 3 + 4 * 0.2, self.bm(self.mones).item())

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

            bqm = BinaryQuadraticModel.from_ising(*self.bm.ising)

            fake_spins = 1.0 * torch.arange(1, 5).unsqueeze(0)

            en_bqm = bqm.energies(fake_spins.numpy()).item()
            en_boltz = self.bm(fake_spins).item()
            self.assertAlmostEqual(en_bqm, en_boltz, 4)

    def test_estimate_beta(self):
        s1 = self.sample_1
        s2 = self.sample_2
        s3 = torch.vstack([self.sample_2, self.sample_2])
        bqm = self.bqm
        self.assertEqual(
            1.0 / maximum_pseudolikelihood_temperature(bqm, s1.numpy())[0],
            self.bm.estimate_beta(s1),
        )
        self.assertEqual(
            1.0 / maximum_pseudolikelihood_temperature(bqm, s2.numpy())[0],
            self.bm.estimate_beta(s2),
        )
        self.assertEqual(
            1.0 / maximum_pseudolikelihood_temperature(bqm, s3.numpy())[0],
            self.bm.estimate_beta(s3),
        )

        fake_spins = torch.randn_like(s3)
        self.assertEqual(
            1.0 / maximum_pseudolikelihood_temperature(bqm, fake_spins.numpy())[0],
            self.bm.estimate_beta(fake_spins),
        )

    def test_pad(self):
        grbm = GRBM(
            3, torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2]), hidx=torch.tensor([1])
        )
        x = torch.zeros((99, 2))
        padded = grbm._pad(x)
        self.assertTrue(padded[:, 1].isnan().all())
        self.assertRaises(ValueError, self.bm._pad, x)

    def test_compute_effective_field(self):
        grbm = GRBM(
            3, torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2]), hidx=torch.tensor([2])
        )
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
        grbm = GRBM(
            3, torch.tensor([0, 0, 1]), torch.tensor([1, 2, 2]), hidx=torch.tensor([2])
        )
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

    def test_objective(self):
        s1 = self.sample_1
        s2 = self.sample_2
        s3 = torch.vstack([self.sample_2, self.sample_2])
        self.assertEqual(-1, self.bm.objective(s1, s2).item())
        self.assertEqual(-1, self.bm.objective(s1, s3))

    def test_sufficient_statistics(self):
        c_answer_0 = [1, 1, 1, 1]
        c_answer_1 = [1, 1, 1, 1]
        c_answer_2 = [-1, 1, -1, -1]

        m0, c0 = self.bm.sufficient_statistics(self.ones)
        self.assertListEqual(m0.tolist(), [1, 1, 1, 1])
        self.assertListEqual(c0.tolist(), c_answer_0)

        m1, c1 = self.bm.sufficient_statistics(torch.vstack([self.ones, self.mones]))
        self.assertListEqual(m1.tolist(), [0] * 4)
        self.assertListEqual(c1.tolist(), c_answer_1)

        m2, c2 = self.bm.sufficient_statistics(self.pmones)
        self.assertEqual(m2.tolist(), [1, -1, 1, -1])
        self.assertEqual(c2.tolist(), c_answer_2)

    def test_gradient(self):
        s1 = self.ones
        s2 = self.pmones
        kld = self.bm.objective(s1, s2)
        kld.backward()
        m_1, c_1 = self.bm.sufficient_statistics(s1)
        m_2, c_2 = self.bm.sufficient_statistics(s2)
        self.assertListEqual(self.bm.h.grad.tolist(), (m_1 - m_2).tolist())
        self.assertListEqual(self.bm.J.grad.tolist(), (c_1 - c_2).tolist())

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
        with self.subTest("Unbounded weight range"):
            h_true = torch.tensor([-3, 0, 1, 3.0])
            J_true = torch.tensor([-1, 1, 2.0, 0])

            self.bm.h.data = h_true
            self.bm.J.data = J_true
            h, J = self.bm.ising
            self.assertListEqual(h, h_true.tolist())

            self.assertListEqual([J[a, b] for a, b in self.edges], J_true.tolist())

        with self.subTest("sparse bounded weight range"):
            hr = [-1, 2.0]
            jr = [-3, 4.0]
            bm = GRBM(
                self.n,
                *torch.tensor(self.edges).mT,
                h_range=hr,
                j_range=jr,
            )
            bm.h.data = torch.tensor([-123, 0, 0, 567.0])
            bm.J.data = torch.tensor([5555, -333.3, 0, 0])
            h, J = bm.ising

            self.assertEqual(min(h), hr[0])
            self.assertEqual(max(h), hr[1])

            self.assertEqual(min(J.values()), jr[0])
            self.assertEqual(max(J.values()), jr[1])

    def test_constructor(self):
        self.assertRaises(ValueError, GRBM, 2, torch.tensor([0, 1]), torch.tensor([3]))
        self.assertRaises(
            ValueError, GRBM, 2, torch.tensor([0, 1]), torch.tensor([4, 5])
        )

    def test_sample(self):
        kwargs = dict(initial_states=[[1, 1, 1, 1], [1, 1, 1, 1], [-1, -1, 1, -1]])
        spins = self.bm.sample(IdentitySampler(), **kwargs)
        self.assertTupleEqual((3, 4), tuple(spins.shape))
        self.assertIsInstance(spins, torch.Tensor)


if __name__ == "__main__":
    unittest.main()
