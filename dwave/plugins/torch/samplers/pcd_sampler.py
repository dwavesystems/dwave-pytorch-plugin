import torch
from dwave.plugins.torch.models.boltzmann_machine import (
    RestrictedBoltzmannMachine as RBM,
)

class PCDSampler:
    """Persistent Contrastive Divergence (PCD) sampler for RBMs.

    This sampler maintains a persistent Markov chain of visible states 
    across minibatches and performs Gibbs sampling using the RBM’s 
    sampling functions.
    
    Args:
        rbm (RBM): The RBM model from which the sampler draws samples.
    """
    def __init__(self, rbm: RBM):
        self.rbm = rbm

        # Stores the last visible states to initialize the Markov chain in Persistent Contrastive Divergence (PCD)
        self.previous_visible_values = None

    def sample(
        self,
        batch_size: int,
        gibbs_steps: int,
        start_visible: torch.Tensor | None = None,
    ):
        """Generate a sample of visible and hidden units using gibbs sampling.

        Args:
            batch_size (int): Number of samples to generate.
            gibbs_steps (int): Number of Gibbs sampling steps to perform.
            start_visible (torch.Tensor | None, optional):  Initial visible states to
                start the Gibbs chain (shape: [batch_size, n_visible]). If None,
                a random Gaussian initialization is used.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of (visible, hidden) from the last Gibbs step:
                - visible: (batch_size, n_visible)
                - hidden:  (batch_size, n_hidden)
        """
        if start_visible is None:
            visible_values = torch.randn(
                batch_size, self.rbm.n_visible, device=self.rbm.weights.device
            )
        else:
            visible_values = start_visible

        hidden_values = None

        for _ in range(gibbs_steps):
            hidden_values = self.rbm.sample_hidden(visible_values)
            visible_values = self.rbm.sample_visible(hidden_values)

        # Store samples to initialize the next Markov chain with (PCD)
        self.previous_visible_values = visible_values.detach()

        return visible_values, hidden_values
