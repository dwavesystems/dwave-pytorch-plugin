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

import torch
import torch.nn.functional as F

from dwave.plugins.torch.models.losses.kl_divergence import pseudo_kl_divergence_loss

class DummyBoltzmannMachine:
    """
    Minimal deterministic stand-in for GraphRestrictedBoltzmannMachine.

    The purpose of this class is NOT to model a real Boltzmann machine.
    Instead, it provides a simple, deterministic quasi_objective so that
    we can verify how pseudo_kl_divergence_loss combines its terms.
    """

    def quasi_objective(self, spins: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        """
        Return a deterministic scalar depending on spins and samples.

        Using a simple mean ensures:
        - deterministic behavior
        - no dependency on logits
        - gradients w.r.t. logits come only from the entropy term
        """
        return spins.float().mean() + samples.float().mean()


def test_pseudo_kl_matches_reference_2d_spins():
    """
    Verify that pseudo_kl_divergence_loss matches the reference formula
    for 2D spins of shape (batch_size, n_spins).

    This test directly reconstructs the loss as:
        cross_entropy - entropy
    and checks numerical equality.
    """

    bm = DummyBoltzmannMachine()

    batch, n_spins = 4, 6
    logits = torch.linspace(-2.0, 2.0, steps=batch * n_spins).reshape(batch, n_spins)

    # spins: (batch_size, n_spins)
    spins = torch.tensor(
        [[-1,1,-1, 1,-1, 1],
         [1,-1, 1,-1, 1,-1],
         [-1,-1,1,1,-1,1],
         [1,1,-1,-1,1,-1]],
        dtype=torch.float32
        )
    
    samples = torch.ones(batch, n_spins, dtype=torch.float32)

    out = pseudo_kl_divergence_loss(spins=spins, logits=logits, samples=samples, boltzmann_machine=bm)

    probs = torch.sigmoid(logits)
    entropy = F.binary_cross_entropy_with_logits(logits, probs)
    cross_binary = bm.quasi_objective(spins, samples)
    ref = cross_binary - entropy

    torch.testing.assert_close(out, ref)

def test_pseudo_kl_works_with_3d_spins():
    """
    Verify that pseudo_kl_divergence_loss supports 3D spins of shape:
        (batch_size, n_samples, n_spins)

    as documented in the function docstring.
    """
    bm = DummyBoltzmannMachine()

    batch, n_samples, n_spins = 3, 5, 4
    logits = torch.zeros(batch, n_spins)

    # spins: (batch_size, n_samples, n_spins)
    spins = torch.ones(batch, n_samples, n_spins)
    samples = torch.zeros(batch, n_spins)

    out = pseudo_kl_divergence_loss(spins=spins, logits=logits, samples=samples, boltzmann_machine=bm)

    probs = torch.sigmoid(logits)
    entropy = F.binary_cross_entropy_with_logits(logits, probs)
    cross_binary = bm.quasi_objective(spins, samples)

    torch.testing.assert_close(out, cross_binary - entropy)

def test_pseudo_kl_gradient_matches_negative_entropy_when_cross_entropy_constant():
    """
    Verify gradient behavior of pseudo_kl_divergence_loss.

    If the Boltzmann machine quasi_objective returns a constant value,
    then the loss gradient w.r.t. logits must come entirely from the
    negative entropy term.

    This test ensures that pseudo_kl_divergence_loss applies the correct
    statistical pressure on encoder logits.
    """

    class ConstantObjectiveBM:
        def quasi_objective(self, spins: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
            # Constant => contributes no gradient wrt logits
            return torch.tensor(1.2345, dtype=spins.dtype, device=spins.device)
    
    bm = ConstantObjectiveBM()

    batch, n_spins = 2, 3

    logits = torch.randn(batch, n_spins, requires_grad = True)
    spins = torch.ones(batch, n_spins)
    samples = torch.zeros(batch, n_spins)

    out = pseudo_kl_divergence_loss(spins=spins, logits=logits, samples=samples, boltzmann_machine=bm)

    out.backward()

    # reference: gradient should be gradient of (-entropy)
    logits2 = logits.detach().clone().requires_grad_(True) 
    # note: require_grad is a property so require_grad_ is used to modify in place
    probs2 = torch.sigmoid(logits2)
    entropy2 = F.binary_cross_entropy_with_logits(logits2, probs2)
    (-entropy2).backward()

    torch.testing.assert_close(logits.grad, logits2.grad)
