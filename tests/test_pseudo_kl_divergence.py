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
from dimod import ExactSolver

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.nn.functional import gumbel_spins
from dwave.plugins.torch.nn.functional import maximum_mean_discrepancy_loss as mmd_loss
from dwave.plugins.torch.nn.functional import pseudo_kl_divergence_loss
from dwave.plugins.torch.samplers import DimodSampler


def encoder_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Average over the batch of the entropy of the factorized distribution of each data point."""
    probs = torch.sigmoid(logits)
    return F.binary_cross_entropy_with_logits(logits, probs, reduction="none").sum(-1).mean()


class UnitLinearBiasObjective:
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

        bm = UnitLinearBiasObjective()

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

        cross_entropy = bm.quasi_objective(spins_data, spins_model)
        ref = cross_entropy - encoder_entropy(logits)

        torch.testing.assert_close(out, ref)

    def test_supports_3d_spins(self):
        """Support 3D spins of shape (batch_size, n_samples, n_spins) as documented."""
        bm = UnitLinearBiasObjective()

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

        cross_entropy = bm.quasi_objective(spins_data, spins_model)

        torch.testing.assert_close(out, cross_entropy - encoder_entropy(logits))

    def test_gradient_matches_exact_kl_divergence(self):
        """The gradient with respect to the logits equals the gradient of the exact KL divergence
        between a factorized encoder distribution and the Boltzmann machine prior.

        With spins replaced by their expectations under the factorized encoder distribution,
        the energy term of the pseudo-KL divergence is the exact expected energy (the model has
        no self-couplings), so the pseudo-KL divergence and the exact KL divergence differ only by
        a term independent of the encoder.
        """
        torch.manual_seed(0)
        bm = GRBM([0, 1, 2], [(0, 1), (1, 2), (0, 2)],
                  linear={0: 0.3, 1: -0.2, 2: 0.1},
                  quadratic={(0, 1): 0.5, (1, 2): -0.4, (0, 2): 0.2})
        logits = torch.randn(4, 3, requires_grad=True)

        # P(s = +1) = sigmoid(logit) so that E[s] = 2 sigmoid(logit) - 1 = tanh(logit / 2)
        expected_spins = torch.tanh(logits / 2)
        samples = torch.ones(1, 3)  # the negative phase does not depend on the encoder
        loss = pseudo_kl_divergence_loss(expected_spins, logits, samples, bm)
        grad_pseudo, = torch.autograd.grad(loss, logits)

        states = 1.0 - 2.0 * torch.tensor(
            [[(k >> i) & 1 for i in range(3)] for k in range(8)], dtype=torch.float32
        )
        log_partition = torch.logsumexp(-bm(states), 0)
        log_probs = F.logsigmoid(logits)   # log P(s = +1)
        log_probs_minus = F.logsigmoid(-logits)  # log P(s = -1)
        plus = (states == 1).float()
        log_q = plus @ log_probs.T + (1 - plus) @ log_probs_minus.T  # (8, 4)
        kl = (log_q.exp() * (log_q + bm(states).unsqueeze(1))).sum(0) + log_partition
        grad_exact, = torch.autograd.grad(kl.mean(), logits)

        torch.testing.assert_close(grad_pseudo, grad_exact)


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

        torch.manual_seed(0)
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
        (-encoder_entropy(logits2)).backward()

        torch.testing.assert_close(logits.grad, logits2.grad)


# Data in the corners of the unit square and the spin strings they encode to
CORNERS = torch.tensor([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
SPINS = 2 * CORNERS - 1


def bit_encoder(x: torch.Tensor) -> torch.Tensor:
    """A parameter-free encoder mapping bits to logits of ±20, at which the Gumbel noise flips a
    spin with probability sigmoid(-20), i.e. never in practice."""
    return x * 40 - 20


def sign_map(logits: torch.Tensor, n_samples: int) -> torch.Tensor:
    """A deterministic straight-through discretisation: the signs of the logits, repeated."""
    spins = torch.sign(logits) - logits.detach() + logits
    return spins.unsqueeze(1).expand(-1, n_samples, *logits.shape[1:])


class TestDiscreteAutoencoderObjectives(unittest.TestCase):
    """The objectives of a discrete variational autoencoder with a Boltzmann machine prior, wired
    as in ``examples/discrete_variational_autoencoder.py``: encoder -> spins -> decoder."""

    def test_objectives_reach_every_parameter(self):
        # Gradients flow through the straight-through discretisation to the encoder, to the
        # decoder and to the prior
        encoder, decoder = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
        prior = GRBM((0, 1), [(0, 1)], linear={0: 0.1, 1: -0.2}, quadratic={(0, 1): -1.2})

        logits = encoder(CORNERS)
        spins = sign_map(logits, n_samples=3)
        reconstruction = F.mse_loss(decoder(spins), CORNERS.unsqueeze(1).expand(-1, 3, -1))
        kl = pseudo_kl_divergence_loss(spins, logits, SPINS, prior)
        (reconstruction + 0.1 * kl).backward()
        parameters = [*encoder.named_parameters(), *decoder.named_parameters(),
                      *prior.named_parameters()]
        for name, parameter in parameters:
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.any(parameter.grad != 0))

        with self.subTest("The maximum mean discrepancy also reaches the encoder"):
            encoder.zero_grad()
            spins = sign_map(encoder(CORNERS), n_samples=1)
            mmd_loss(spins.flatten(1), SPINS)[0].backward()
            self.assertTrue(torch.any(encoder.weight.grad != 0))

    def test_uniform_prior_is_a_fixed_point_of_uniform_data(self):
        # The corners encode to the four spin strings, which the zero-parameter prior generates
        # uniformly. ExactSolver enumerates every state once, i.e. it samples that prior exactly,
        # so the gradient of the pseudo-KL divergence with respect to the prior vanishes: training
        # the prior on this data leaves it uniform.
        prior = GRBM((0, 1), [(0, 1)], linear={0: 0.0, 1: 0.0}, quadratic={(0, 1): 0.0})
        logits = bit_encoder(CORNERS)
        spins = gumbel_spins(logits)  # strong logits: exactly the spin strings of the corners
        torch.testing.assert_close(spins, SPINS.unsqueeze(1))
        samples = DimodSampler(prior, ExactSolver()).sample()
        self.assertEqual((4, 2), tuple(samples.shape))
        loss = pseudo_kl_divergence_loss(spins, logits, samples, prior)
        for grad in torch.autograd.grad(loss, list(prior.parameters())):
            torch.testing.assert_close(grad, torch.zeros_like(grad))


if __name__ == "__main__":
    unittest.main()
