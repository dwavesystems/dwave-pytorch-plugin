# Copyright 2026 D-Wave
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
from dimod import IdentitySampler, SampleSet, TrackingComposite

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.samplers.dimod_sampler import DimodSampler
from dwave.samplers import SteepestDescentSampler
from dwave.samplers import SimulatedAnnealingSampler


class TestDimodSampler(unittest.TestCase):
    def setUp(self) -> None:
        # Create a triangle graph with an additional dangling vertex
        #       a
        #     / | \
        #    b--c  d
        # Note the node order is deliberately "dbac" in order to test variable orderings
        self.nodes = list("dbac")
        self.edges = [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]
        self.n = 4

        # Manually set the parameter weights for testing
        dtype = torch.float32
        h = [0.0, 1, 2, 3]

        bm = GRBM(self.nodes, self.edges)
        bm._linear.data = torch.tensor(h, dtype=dtype)
        bm._quadratic.data = torch.tensor([1, 2, 3, 6], dtype=dtype)

        self.bm = bm

        self.ones = torch.ones(4).unsqueeze(0)
        self.mones = -torch.ones(4).unsqueeze(0)
        self.pmones = torch.tensor([[1, -1, 1, -1]], dtype=dtype)
        self.mpones = torch.tensor([[-1, 1, -1, 1]], dtype=dtype)

        self.sample_1 = torch.vstack([self.ones, self.ones, self.ones, self.pmones])
        self.sample_2 = torch.vstack([self.ones, self.ones, self.ones, self.mpones])
        return super().setUp()

    def test_sample(self):
        grbm = GRBM(list("abcd"), [("a", "b"), ("a", "c"), ("a", "d"), ("b", "c")])

        with self.subTest("Spins should be identical to input."):
            initial_states = [[1, 1, 1, 1],
                              [1, 1, 1, 1],
                              [-1, -1, 1, -1]]
            sampler = DimodSampler(grbm, IdentitySampler(),
                                   prefactor=1, linear_range=None, quadratic_range=None,
                                   sample_kwargs=dict(initial_states=(initial_states, "abcd")))
            spins = sampler.sample()
            self.assertIsInstance(spins, torch.Tensor)
            self.assertTupleEqual((3, 4), tuple(spins.shape))
            self.assertListEqual(initial_states, spins.tolist())

        with self.subTest("Prefactor should scale weights up."):
            grbm.linear.data[:] = 1
            grbm.quadratic.data[:] = -1
            prefactor = 12345
            tracker = TrackingComposite(SteepestDescentSampler())
            sampler = DimodSampler(grbm, tracker,
                                   prefactor=prefactor, linear_range=None, quadratic_range=None,
                                   sample_kwargs=dict())
            sampler.sample()
            self.assertDictEqual(tracker.input['h'], dict(zip(grbm.nodes, [prefactor]*4)))
            self.assertDictEqual(tracker.input['J'], dict(zip(grbm.edges, [-prefactor]*4)))

        with self.subTest("Linear weights should be clipped to be 0."):
            grbm.linear.data[:] = torch.tensor([-2, -0.002, 0.002, 3])
            tracker = TrackingComposite(SteepestDescentSampler())
            sampler = DimodSampler(grbm, tracker,
                                   prefactor=100, linear_range=[0, 0], quadratic_range=None,
                                   sample_kwargs=dict())
            sampler.sample()
            torch.testing.assert_close(
                torch.tensor(list(tracker.input['h'].values())),
                torch.tensor([0, 0, 0, 0.0])
            )

        with self.subTest("Linear weights should be clipped to be within range."):
            grbm.linear.data[:] = torch.tensor([-2, -0.002, 0.002, 3])
            tracker = TrackingComposite(SteepestDescentSampler())
            sampler = DimodSampler(grbm, tracker,
                                   prefactor=100, linear_range=[-1, 1], quadratic_range=None,
                                   sample_kwargs=dict())
            sampler.sample()
            torch.testing.assert_close(
                torch.tensor(list(tracker.input['h'].values())),
                torch.tensor([-1, -0.2, 0.2, 1])
            )

        with self.subTest("Quadratic weights should be clipped to be within range."):
            grbm.quadratic.data[:] = torch.tensor([-2, -0.002, 0.002, 3])
            tracker = TrackingComposite(SteepestDescentSampler())
            sampler = DimodSampler(grbm, tracker,
                                   prefactor=100, linear_range=None, quadratic_range=[-1, 1],
                                   sample_kwargs=dict())
            sampler.sample()
            torch.testing.assert_close(
                torch.tensor(list(tracker.input['J'].values())),
                torch.tensor([-1, -0.2, 0.2, 1])
            )

        with self.subTest("Quadratic weights should be clipped to be 0."):
            grbm.quadratic.data[:] = torch.tensor([-2, -0.002, 0.002, 3])
            tracker = TrackingComposite(SteepestDescentSampler())
            sampler = DimodSampler(grbm, tracker,
                                   prefactor=100, linear_range=None, quadratic_range=[0, 0],
                                   sample_kwargs=dict())
            sampler.sample()
            torch.testing.assert_close(
                torch.tensor(list(tracker.input['J'].values())),
                torch.tensor([0, 0, 0, 0.0])
            )

        sampler = DimodSampler(
            self.bm,
            SimulatedAnnealingSampler(),
            prefactor=1,
            sample_kwargs=dict(num_reads=1)
        )

        x = torch.tensor([
            [1.0, float("nan"), -1.0, float("nan")],
            [float("nan"), -1.0, float("nan"), 1.0],
        ])

        samples = sampler.sample(x)

        with self.subTest("Conditional sampling returns expected shape"):
            # Shape check
            self.assertTupleEqual(samples.shape, (2, 1, 4))
        samples = samples.squeeze()

        with self.subTest("Conditional sampling preserves clamped variables"):
            # Check clamped values unchanged
            mask = ~torch.isnan(x)
            self.assertTrue(torch.all(samples[mask] == x[mask]))

        with self.subTest("Conditional sampling samples free variables as ±1"):
            # Check free variables are ±1
            free_mask = torch.isnan(x)
            free_values = samples[free_mask]
            self.assertTrue(torch.all(torch.isin(free_values, torch.tensor([-1.0, 1.0]))),
                            "Free variables should be sampled as ±1")

        with self.subTest("Conditional sampling supports multiple reads."):
            num_reads = 5
            sampler = DimodSampler(
                self.bm,
                SimulatedAnnealingSampler(),
                prefactor=1,
                sample_kwargs=dict(num_reads=num_reads)
            )

            x = torch.tensor([
                [1.0, float("nan"), -1.0, float("nan")],
                [float("nan"), -1.0, float("nan"), 1.0],
            ])

            samples = sampler.sample(x)

            # Shape should be (batch_size, num_reads, n_nodes)
            self.assertTupleEqual(samples.shape, (2, 5, 4))

            # Check clamped values for every read
            mask = ~torch.isnan(x)
            free_mask = torch.isnan(x)
            for i in range(num_reads):
                self.assertTrue(torch.all(samples[:, i, :][mask] == x[mask]))

                free_values = samples[:, i, :][free_mask]
                self.assertTrue(torch.all(torch.isin(free_values, torch.tensor([-1.0, 1.0]))),
                                f"Free variables should be sampled as ±1 in read {i}")

        with self.subTest("Conditional sampling with all variables clamped returns input unchanged."):
            num_reads = 5
            sampler = DimodSampler(
                self.bm,
                SimulatedAnnealingSampler(),
                prefactor=1,
                sample_kwargs=dict(num_reads=num_reads)
            )

            x = torch.tensor([
                [+1.0, -1.0, -1.0, +1.0],
                [-1.0, +1.0, -1.0, -1.0],
            ])

            samples = sampler.sample(x)
            for i in range(num_reads):
                torch.testing.assert_close(
                    samples[:, i, :],
                    x,
                    msg=f"Fully clamped inputs should be preserved in read {i}"
                )

        with self.subTest("Conditional sampling supports mixed fully clamped and partially clamped rows."):
            num_reads = 5
            sampler = DimodSampler(
                self.bm,
                SimulatedAnnealingSampler(),
                prefactor=1,
                sample_kwargs=dict(num_reads=num_reads)
            )

            x = torch.tensor([
                [1.0, -1.0, -1.0,  1.0],          # fully clamped
                [-1.0, float("nan"), 1.0, -1.0],  # partially clamped
            ])

            samples = sampler.sample(x)

            # First row fully clamped, should be preserved
            for i in range(num_reads):
                torch.testing.assert_close(
                    samples[0, i, :],
                    x[0],
                    msg=f"Fully clamped row should be preserved in read {i}"
                )

            # Clamped values in the second row should be preserved
            self.assertTrue(torch.all(samples[1, :, 0] == -1))
            self.assertTrue(torch.all(samples[1, :, 2] == 1))
            self.assertTrue(torch.all(samples[1, :, 3] == -1))

    def test_sample_set(self):
        grbm = GRBM(list("abcd"), [("a", "b")])
        initial_states = [[1, 1, 1, 1],
                          [1, 1, 1, 1],
                          [-1, -1, 1, -1]]
        sampler = DimodSampler(grbm, IdentitySampler(),
                               prefactor=1, linear_range=None, quadratic_range=None,
                               sample_kwargs=dict(initial_states=(initial_states, "abcd")))
        with self.subTest("Accessing `sample_set` field before sampling should raise an error."):
            with self.assertRaisesRegex(AttributeError, "no samples found"):
                sampler.sample_set

        sampler.sample()
        with self.subTest("The `sample_set` attribute should be of type `dimod.SampleSet`."):
            self.assertTrue(isinstance(sampler.sample_set, SampleSet))


if __name__ == "__main__":
    unittest.main()
