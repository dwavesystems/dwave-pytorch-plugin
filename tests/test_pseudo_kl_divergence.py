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
"""
Unit tests for pseudo_kl_divergence_loss.

These tests verify the *statistical structure* of the pseudo-KL divergence used
in the DVAE setting, not the correctness of the Boltzmann machine itself.

In particular, we test that:
1) The loss matches the reference decomposition:
       pseudo_KL = cross_entropy_with_prior - entropy_of_encoder
2) The function supports both documented spin shapes.
3) The gradient w.r.t. encoder logits behaves as expected.

The tests intentionally use deterministic dummy Boltzmann machines to isolate
and validate the behavior of pseudo_kl_divergence_loss in isolation.
"""
import unittest

import torch
import torch.nn.functional as F

from dwave.plugins.torch.models.losses.kl_divergence import pseudo_kl_divergence_loss


class DummyBoltzmannMachine:
    """A minimal and deterministic stand-in for GraphRestrictedBoltzmannMachine.

    The purpose of this class is NOT to model a real Boltzmann machine.
    Instead, it provides a simple, deterministic quasi_objective so that
    we can verify how pseudo_kl_divergence_loss combines its terms.
    """

    def quasi_objective(self, spins_data: torch.Tensor, spins_model: torch.Tensor) -> torch.Tensor:
        """Return a deterministic scalar representing a positive-minus-negative phase
        objective, independent of encoder logits.
        """
        return spins_data.float().mean() - spins_model.float().mean()

class TestPseudoKLDivergenceLoss(unittest.TestCase):
    """Unit tests for pseudo_kl_divergence_loss."""

    def test_matches_reference_2d(self):
        """Match explicit cross-entropy minus entropy reference for 2D spins."""

        bm = DummyBoltzmannMachine()

```suggestion
        spins_data = torch.tensor(
            [[-1, 1, -1, 1, -1, 1],
             [1, -1, 1, -1, 1, -1],
             [-1, -1, 1, 1, -1, 1],
             [1, 1, -1, -1, 1, -1]],
            dtype=torch.float32
        )

        batch_size, n_spins = spins_data.shape
        logits = torch.linspace(-2.0, 2.0, steps=batch_size * n_spins).reshape(batch_size, n_spins)

        spins_model = torch.ones(batch_size, n_spins, dtype=torch.float32)

        out = pseudo_kl_divergence_loss(
            spins=spins_data,
            logits=logits,
            samples=spins_model,
            boltzmann_machine=bm
        )

        probs = torch.sigmoid(logits)
        entropy = F.binary_cross_entropy_with_logits(logits, probs)
        cross_entropy = bm.quasi_objective(spins_data, spins_model)
        ref = cross_entropy - entropy

        torch.testing.assert_close(out, ref)


    def test_supports_3d_spins(self):
        """Support 3D spins of shape (batch_size, n_samples, n_spins) as documented."""
        bm = DummyBoltzmannMachine()

        batch_size, n_samples, n_spins = 3, 5, 4
        logits = torch.zeros(batch_size, n_spins)
        # Zero logits are used in the 3D shape test to keep the entropy term simple and stable (p = 0.5),
        # allowing the test to focus purely on documented shape support; nonzero values are covered in the
        # 2D numerical correctness test.

        # spins: (batch_size, n_samples, n_spins)
        spins_data = torch.ones(batch_size, n_samples, n_spins)
        spins_model = torch.zeros(batch_size, n_spins)

        out = pseudo_kl_divergence_loss(
            spins=spins_data, 
            logits=logits, 
            samples=spins_model, 
            boltzmann_machine=bm
        )

        probs = torch.sigmoid(logits)
        entropy = F.binary_cross_entropy_with_logits(logits, probs)
        cross_entropy = bm.quasi_objective(spins_data, spins_model)

        torch.testing.assert_close(out, cross_entropy - entropy)


    def test_gradient_from_entropy_only(self):
        """Verify gradient behavior of pseudo_kl_divergence_loss.

        If the Boltzmann machine quasi_objective returns a constant value,
        then the loss gradient w.r.t. logits must come entirely from the
        negative entropy term.

        This test ensures that pseudo_kl_divergence_loss applies the correct
        statistical pressure on encoder logits.
        """

        class ConstantObjectiveBM:
            def quasi_objective(self, spins_data: torch.Tensor, 
                                spins_model: torch.Tensor) -> torch.Tensor:
                # Constant => contributes no gradient wrt logits
                return torch.tensor(1.2345, dtype=spins_data.dtype, device=spins_data.device)
        
        bm = ConstantObjectiveBM()

        batch_size, n_spins = 2, 3

        logits = torch.randn(batch_size, n_spins, requires_grad=True)
        spins_data = torch.ones(batch_size, n_spins)
        spins_model = torch.zeros(batch_size, n_spins)

        out = pseudo_kl_divergence_loss(
            spins=spins_data, 
            logits=logits, 
            samples=spins_model, 
            boltzmann_machine=bm
        )

        out.backward()

        # reference gradient from -entropy only
        logits2 = logits.detach().clone().requires_grad_(True) 
        probs2 = torch.sigmoid(logits2)
        entropy2 = F.binary_cross_entropy_with_logits(logits2, probs2)
        (-entropy2).backward()

        torch.testing.assert_close(logits.grad, logits2.grad)

if __name__ == "__main__":
    unittest.main()
