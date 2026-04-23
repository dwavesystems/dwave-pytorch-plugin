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

import os
import torch
from torch.utils.data import DataLoader
from dwave.plugins.torch.models.boltzmann_machine import (
    RestrictedBoltzmannMachine as RBM,
)
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from torch.optim import SGD
from dwave.plugins.torch.samplers.pcd_sampler import PCDSampler


def load_binarized_mnist(dataset_path: str = "data") -> datasets.MNIST:
    """Load the MNIST dataset and binarize it (pixels >= 0.5 become 1, else 0).

    Args:
        dataset_path (str): Path to download/store the MNIST dataset. Defaults to "data".

    Returns:
        datasets.MNIST: Binarized MNIST training dataset.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: (x >= 0.5).float())]
    )

    train_dataset = datasets.MNIST(
        root=dataset_path, train=True, transform=transform, download=True
    )
    return train_dataset


def contrastive_divergence(
    rbm: RBM,
    batch: torch.Tensor,
    n_gibbs_steps: int,
    sampler: PCDSampler,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    """Perform one step of Contrastive Divergence (CD-k).

    Uses Persistent Contrastive Divergence (PCD) by maintaining the last visible states
    for Gibbs sampling across batches.
    Gradients are applied via the provided PyTorch optimizer.

    Args:
        batch (torch.Tensor): A batch of input data of shape (batch_size, n_visible).
        n_gibbs_steps (int): Number of Gibbs sampling steps per epoch.
        sampler (PCDSampler): Sampler responsible for producing negative-phase samples.
        optimizer (torch.optim.Optimizer): PyTorch optimizer.

    Returns:
        torch.Tensor: The reconstruction error (L1 norm) for the batch.
    """
    # Positive phase (data-driven)
    hidden_probs = torch.sigmoid(batch @ rbm.weights + rbm.hidden_biases)
    
    weight_grads = torch.matmul(batch.t(), hidden_probs)
    visible_bias_grads = batch.clone()
    hidden_bias_grads = hidden_probs.clone()

    batch_size = batch.size(0)

    # Negative phase (model-driven)
    # Sample from the model using gibbs sampling
    visible_values, hidden_values = sampler.sample(
        batch_size,
        gibbs_steps=n_gibbs_steps,
        start_visible=sampler.previous_visible_values,
    )

    visible_values = visible_values.detach()
    hidden_values = hidden_values.detach()

    # Compute the gradients for negative phase
    weight_grads -= torch.matmul(visible_values.t(), hidden_values)
    visible_bias_grads -= visible_values
    hidden_bias_grads -= hidden_values

    # Average across the batch
    weight_grads /= batch_size
    visible_bias_grads = torch.mean(visible_bias_grads, dim=0)
    hidden_bias_grads = torch.mean(hidden_bias_grads, dim=0)

    # Apply gradients via optimizer
    rbm.weights.grad = -weight_grads
    rbm.visible_biases.grad = -visible_bias_grads
    rbm.hidden_biases.grad = -hidden_bias_grads

    optimizer.step()
    optimizer.zero_grad()

    # Compute reconstruction error (L1 norm)
    reconstruction = rbm.sample_visible(rbm.sample_hidden(batch))
    reconstruction = reconstruction.detach()
    reconstruction_error = torch.sum(torch.abs(batch - reconstruction))

    return reconstruction_error

def train_loop(
    train_loader: DataLoader,
    rbm: RBM,
    n_epochs: int,
    n_gibbs_steps: int,
    sampler: PCDSampler,
    optimizer: torch.optim.Optimizer,
) -> None:
    """Train the RBM using contrastive divergence with a given PCDSampler and optimizer.

    Args:
        train_loader (DataLoader): PyTorch DataLoader for training data.
        rbm (RBM): Restricted Boltzmann Machine instance.
        n_epochs (int): Number of training epochs.
        n_gibbs_steps (int): Number of Gibbs sampling steps per CD update.
        sampler (PCDSampler): sampler responsible for producing negative-phase samples.
        optimizer (torch.optim.Optimizer): PyTorch optimizer.
    """
    device = rbm._weights.device
    for epoch in range(n_epochs):
        total_error = 0
        num_examples = 0
        for batch, _ in train_loader:
            # flatten input data
            batch = batch.reshape(batch.size(0), rbm.n_visible).to(device)

            # Perform one step of contrastive divergence and accumulate error
            error = contrastive_divergence(
                rbm, batch, n_gibbs_steps, sampler, optimizer
            )
            total_error += error
            num_examples += batch.size(0)
        average_error = total_error / num_examples  # Average reconstruction error
        print(
            f"Epoch {epoch + 1}/{n_epochs} - Avg reconstruction error: {average_error:.4f}"
        )


def generate_and_save_images(
    sampler: PCDSampler,
    rows: int = 8,
    columns: int = 8,
    steps: int = 1000,
    output_dir: str = "samples",
    output_filename: str = "generated_images.png",
) -> PCDSampler:
    """Generate samples from a trained RBM and save them as a grid of images.

    Args:
        sampler (PCDSampler): sampler to generate samples from the trained RBM.
        rows (int): Number of rows in the output image grid. Defaults to 8.
        columns (int): Number of columns in the output image grid. Defaults to 8.
        steps (int): Number of Gibbs sampling steps for generation. Defaults to 1000.
        output_dir (str): Directory to save the generated images. Defaults to "samples".
        output_filename (str): File name for saving the generated image grid. Defaults to "generated_images.png".
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    num_images = rows * columns

    # Generate batch of images
    samples, _ = sampler.sample(num_images, gibbs_steps=steps)

    # for SpinRBM
    #samples = ((samples + 1) / 2).view(num_images, 28, 28).detach().cpu().numpy()  # convert -1/+1 → 0/1
    samples = samples.view(num_images, 28, 28).detach().cpu().numpy()

    # Plot grid of images
    fig, axs = plt.subplots(rows, columns, figsize=(columns, rows))

    idx = 0
    for r in range(rows):
        for c in range(columns):
            axs[r, c].imshow(samples[idx], cmap="gray")
            axs[r, c].axis("off")
            idx += 1

    fig.suptitle("Generated images from RBM trained on MNIST", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    plt.show()
    print(f"Generated {num_images} samples in {output_dir}/{output_filename}")


def train_rbm(
    n_visible: int,
    n_hidden: int,
    n_gibbs_steps: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    n_epochs: int,
    batch_size: int,
    dataset_path: str = "data",
) -> PCDSampler:
    """Train an RBM on MNIST and generate sample images.

    Args:
        n_visible (int, optional): Number of visible units.
        n_hidden (int, optional): Number of hidden units.
        n_gibbs_steps (int, optional): Number of Gibbs sampling steps per CD update.
        learning_rate (float, optional): Base learning rate for CD updates.
        momentum (float, optional): Momentum coefficient for CD updates.
        weight_decay (float, optional):  Weight decay (L2 regularization) coefficient.
        n_epochs (int, optional): Number of training epochs.
        batch_size (int, optional): Batch size for training.
        dataset_path (str, optional): Path to download/store the MNIST dataset. Defaults to "data".
    Returns:
        PCDSampler: The sampler used for training the RBM.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device")

    # Load MNIST data
    print("Loading MNIST dataset...")
    train_dataset = load_binarized_mnist(dataset_path)

    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Initialize RBM
    rbm = RBM(n_visible, n_hidden).to(device)

    # Initialize PCD Sampler
    sampler = PCDSampler(rbm)

    optimizer = SGD(
        [rbm.weights, rbm.visible_biases, rbm.hidden_biases],
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # Train RBM
    train_loop(train_loader, rbm, n_epochs, n_gibbs_steps, sampler, optimizer)

    return sampler


if __name__ == "__main__":
    # Run an example of fitting a Restricted Boltzmann Machine to the MNIST dataset
    sampler = train_rbm(
        n_visible=784,
        n_hidden=500,
        n_gibbs_steps=10,
        learning_rate=1e-3,
        momentum=0.5,
        weight_decay=1e-7,
        n_epochs=50,
        batch_size=64,
    )
    # Generate and save samples
    generate_and_save_images(sampler)
