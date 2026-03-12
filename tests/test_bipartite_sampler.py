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
from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.samplers.bipartite_sampler import BipartiteGibbsSampler


class TestBipartiteGibbsSampler(unittest.TestCase):

    def test_device(self):
        nodes = ["v1", "h1"]
        edges = [["v1", "h1"]]
        grbm = GRBM(nodes, edges, hidden_nodes=["h1"])

        sample_size = 10
        sampler = BipartiteGibbsSampler(
            grbm,
            num_chains=sample_size,
            schedule=[1.0],
            seed=2
        )

        sampler = sampler.to("meta")

        # GRBM parameters should remain on CPU
        self.assertEqual("cpu", sampler._grbm.linear.device.type)
        self.assertEqual("cpu", sampler._grbm.quadratic.device.type)

        # Sampler-owned tensors should move to meta
        self.assertEqual("meta", sampler._x.device.type)
        self.assertEqual("meta", sampler._schedule.device.type)

        # RNG should remain on CPU (meta not supported)
        self.assertEqual("cpu", sampler._rng.device.type)
    
    def test_prepare_initial_states(self):
        nodes = ["v1", "v2", "h1", "h2"]
        edges = [["v1", "h1"], ["v1", "h2"], ["v2", "h1"], ["v2", "h2"]]
        grbm = GRBM(nodes, edges, hidden_nodes=["h1", "h2"])

        sampler = BipartiteGibbsSampler(grbm, num_chains=2, schedule=[1.0],)
        # Invalid spins
        with self.subTest("Non-spin initial states."):
            self.assertRaisesRegex(ValueError, "contain nonspin values", sampler._prepare_initial_states,
                            initial_states=torch.tensor([[0, 1, -1, 1]]), num_chains=1)

        # Incorrect shape
        with self.subTest("Testing initial states with incorrect shape."):
            self.assertRaisesRegex(ValueError, "Initial states should be of shape", sampler._prepare_initial_states,
                              num_chains=2, initial_states=torch.tensor([[-1, 1, 1, 1, -1]]))
                
    def test_compute_effective_field_bipartite(self):
        # Define bipartite graph
        nodes = ["v1", "v2", "h1", "h2"]
        edges = [["v1", "h1"],["v1", "h2"],["v2", "h1"],["v2", "h2"]]
        grbm = GRBM(nodes, edges, hidden_nodes=["h1", "h2"])

        # Set parameters manually
        grbm._linear.data = torch.tensor([0.1, -0.2, 0.3, -0.4])
        grbm._quadratic.data = torch.tensor([0.5, 0.2, -0.7, 0.6])

        sampler = BipartiteGibbsSampler(grbm, num_chains=1, schedule=[1.0])

        # Force a known spin state
        spin_state = torch.tensor([[1., -1., 1., -1.]])
        sampler._x.data[:] = spin_state

        # Block indices
        visible_block = grbm.visible_idx
        hidden_block = grbm.hidden_idx

        # Expected fields computed manually
        expected_visible_field = torch.tensor([[0.1 + 0.5*1 + 0.2*(-1),  # v1
                                                -0.2 + (-0.7)*1 + 0.6*(-1)]])  # v2
        expected_hidden_field = torch.tensor([[0.3 + 0.5*1 + (-0.7)*(-1),  # h1
                                               -0.4 + 0.2*1 + 0.6*(-1)]])   # h2

        # Compute via sampler
        sampler_visible_field = sampler._compute_effective_field(visible_block)
        sampler_hidden_field = sampler._compute_effective_field(hidden_block)

        # Compare
        torch.testing.assert_close(expected_visible_field, sampler_visible_field)
        torch.testing.assert_close(expected_hidden_field, sampler_hidden_field)

    def test_gibbs_update(self):
        # Define bipartite graph
        nodes = ["v1", "v2", "h1", "h2"]
        edges = [["v1", "h1"], ["v1", "h2"], ["v2", "h1"], ["v2", "h2"]]
        grbm = GRBM(nodes, edges, hidden_nodes=["h1", "h2"])

        sample_size = 1_000_000
        sampler = BipartiteGibbsSampler(grbm, num_chains=sample_size, schedule=[1.0], seed=42)

        # Force all spins to +1
        sampler._x.data[:] = 1.0

        ones = torch.ones((sample_size, 1))
        zero_field = torch.tensor(0.0)
        
        visible_block = grbm.visible_idx
        hidden_block = grbm.hidden_idx
        
        # Gibbs update for visible block (block=0)
        with self.subTest("visible block Gibbs update"):
            sampler._gibbs_update(0.0, visible_block, ones*zero_field)
            torch.testing.assert_close(torch.tensor(0.5), sampler._x.mean(), atol=1e-3, rtol=1e-3)

        # Gibbs update for hidden block (block=1)
        with self.subTest("hidden block Gibbs update"):
            sampler._gibbs_update(0.0, hidden_block, ones*zero_field)
            torch.testing.assert_close(torch.tensor(0.0), sampler._x.mean(), atol=1e-2, rtol=1e-2)

        # Gibbs update with a nonzero effective field
        with self.subTest("Gibbs update with nonzero effective field"):
            effective_field = torch.tensor(1.2)
            sampler._x.data[:] = 1.0
            sampler._gibbs_update(1.0, visible_block, effective_field*ones)
            sampler._gibbs_update(1.0, hidden_block, effective_field*ones)
            torch.testing.assert_close(
                torch.tanh(-effective_field),
                sampler._x.mean(),
                atol=1e-3, rtol=1e-3)

    def test_sample(self):
        # Define bipartite graph
        nodes = ["v1", "v2", "h1", "h2"]
        edges = [["v1", "h1"], ["v1", "h2"], ["v2", "h1"], ["v2", "h2"]]
        grbm = GRBM(nodes, edges, hidden_nodes=["h1", "h2"])

        # Set parameters manually
        grbm._linear.data = torch.tensor([0.1, -0.2, 0.3, -0.4])
        grbm._quadratic.data = torch.tensor([0.5, 0.2, -0.7, 0.6])

        # Create two samplers with a fixed random seed
        sampler1 = BipartiteGibbsSampler(grbm, num_chains=5, schedule=[1.0, 2.0], seed=42)
        sampler2 = BipartiteGibbsSampler(grbm, num_chains=5, schedule=[1.0, 2.0], seed=42)

        # Sample spins from both samplers
        sampler1.sample()
        
        # Manually apply Gibbs updates
        for beta in sampler2._schedule:
            sampler2._step(beta)

        # Ensure the results are the same
        self.assertListEqual(sampler1._x.tolist(), sampler2._x.tolist())
    
    def test_validate_input_and_generate_mask(self):
        # Define bipartite graph
        nodes = ["v1", "v2", "h1", "h2"]
        edges = [["v1", "h1"], ["v1", "h2"], ["v2", "h1"], ["v2", "h2"]]
        grbm = GRBM(nodes, edges, hidden_nodes=["h1", "h2"])

        sampler = BipartiteGibbsSampler(grbm, num_chains=2, schedule=[1.0])

        # Unclamped visible only (valid)
        with self.subTest("Visible unclamped while hidden nodes are clamped"):
            x = torch.tensor([
                [float("nan"), float("nan"),  1., -1.],   # chain 0: visible unclamped
                [float("nan"), float("nan"), -1.,  1.]    # chain 1: visible unclamped
            ])

            mask = sampler._validate_input_and_generate_mask(x)
            self.assertTrue(mask.shape == x.shape)

        
        # Unclamped hidden only (valid)
        with self.subTest("Hidden nodes unclamped while visible nodes are clamped"):
            x = torch.tensor([
                [ 1., -1., float("nan"), float("nan")],
                [-1.,  1., float("nan"), float("nan")]
            ])

            mask = sampler._validate_input_and_generate_mask(x)
            self.assertTrue(mask.shape == x.shape)

        
        # Unclamped in both blocks (invalid)
        with self.subTest("Both visible and hidden nodes contain unclamped variables"):
            x_invalid = torch.tensor([
                [float("nan"), -1., float("nan"), 1.],  # visible + hidden unclamped
                [1., -1., 1., -1.]
            ])

            with self.assertRaisesRegex(ValueError, "unclamped for visible or hidden"):
                sampler._validate_input_and_generate_mask(x_invalid)

        
        # Invalid spin values
        with self.subTest("Input contains values other than ±1 or NaN"):
            x_invalid_spin = torch.tensor([
                [0., 1., float("nan"), float("nan")],
                [1., -1., 1., -1.]
            ])

            with self.assertRaisesRegex(ValueError, "x contains values other than ±1 or NaN"):
                sampler._validate_input_and_generate_mask(x_invalid_spin)

            
        # Wrong shape
        with self.subTest("Input has wrong shape"):
            x_wrong_shape = torch.tensor([[1., -1., 1.]])  # wrong dimension

            with self.assertRaisesRegex(ValueError, "x should be of shape"):
                sampler._validate_input_and_generate_mask(x_wrong_shape)

    def test_sample_conditional(self):
        nodes = ["v1", "v2", "h1", "h2"]
        edges = [["v1", "h1"], ["v1", "h2"], ["v2", "h1"], ["v2", "h2"]]
        grbm = GRBM(nodes, edges, hidden_nodes=["h1", "h2"])

        sampler = BipartiteGibbsSampler(grbm, num_chains=3, schedule=[1.0, 2.0], seed=123)

        visible = grbm.visible_idx
        hidden  = grbm.hidden_idx

        # Clamp visible, sample hidden
        x = torch.full((3, 4), float("nan"))
        x[:, visible] = torch.tensor([[1., -1.],
                                    [1.,  1.],
                                    [-1., -1.]])

        result = sampler.sample(x=x)
        
        # Visible must remain unchanged
        torch.testing.assert_close(result[:, visible], x[:, visible])

        # Hidden must be valid spins
        self.assertTrue(set(result[:, hidden].unique().tolist()).issubset([-1.0, 1.0]))

        # Clamp hidden, sample visible
        sampler = BipartiteGibbsSampler(grbm, num_chains=3, schedule=[1.0, 2.0], seed=123)

        x = torch.full((3, 4), float("nan"))
        x[:, hidden] = torch.tensor([[1., -1.],
                                    [-1.,  1.],
                                    [1.,  1.]])

        result = sampler.sample(x=x)

        # Hidden must remain unchanged
        torch.testing.assert_close(result[:, hidden], x[:, hidden])

        # Visible must be valid spins
        self.assertTrue(set(result[:, visible].unique().tolist()).issubset([-1.0, 1.0]))
if __name__ == "__main__":
    unittest.main()
