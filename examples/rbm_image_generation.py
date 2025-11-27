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


def load_binarized_mnist(dataset_path: str = "data") -> datasets.MNIST:
    """
    Load the MNIST dataset and binarize it (pixels >= 0.5 become 1, else 0).

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


def train_loop(
    train_loader: DataLoader,
    rbm: RBM,
    n_epochs: int,
    n_gibbs_steps: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> None:
    """
    Train the RBM using contrastive divergence with momentum and weight decay.

    Args:
        train_loader (DataLoader): PyTorch DataLoader for training data.
        rbm (RBM): Restricted Boltzmann Machine instance.
        n_epochs (int): Number of training epochs.
        n_gibbs_steps (int): Number of Gibbs sampling steps per CD update.
        learning_rate (float): Base learning rate.
        momentum (float): Momentum coefficient for parameter updates.
        weight_decay (float): Weight decay (L2 regularization) coefficient.
    """
    device = rbm._weights.device
    for epoch in range(n_epochs):
        total_error = 0
        num_examples = 0
        for batch, _ in train_loader:
            # flatten input data
            batch = batch.reshape(batch.size(0), rbm.n_visible).to(device)

            # Perform one step of contrastive divergence and accumulate error
            error = rbm._contrastive_divergence(
                batch,
                epoch,
                n_gibbs_steps,
                learning_rate,
                momentum,
                weight_decay,
                n_epochs,
            )
            total_error += error
            num_examples += batch.size(0)
        average_error = total_error / num_examples  # Average reconstruction error
        print(
            f"Epoch {epoch + 1}/{n_epochs} - Avg reconstruction error: {average_error:.4f}"
        )


def generate_and_save_images(
    rbm: RBM,
    rows: int = 8,
    columns: int = 8,
    steps: int = 1000,
    output_dir: str = "samples",
    output_filename: str = "generated_images.png",
) -> None:
    """
    Generate samples from a trained RBM and save them as a grid of images.

    Args:
        rbm (RBM): Trained RBM instance.
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
    samples, _ = rbm.generate_sample(num_images, gibbs_steps=steps)
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
    n_visible: int = 784,
    n_hidden: int = 500,
    n_gibbs_steps: int = 10,
    learning_rate: float = 1e-3,
    momentum: float = 0.5,
    weight_decay: float = 1e-7,
    n_epochs: int = 20,
    batch_size: int = 32,
    dataset_path: str = "data",
    output_dir: str = "samples",
    output_filename: str = "generated_images.png",
) -> None:
    """Train an RBM on MNIST and generate sample images.

    Args:
        n_visible (int, optional): Number of visible units. Defaults to 784.
        n_hidden (int, optional): Number of hidden units. Defaults to 500.
        n_gibbs_steps (int, optional): Number of Gibbs sampling steps per CD update. Defaults to 10.
        learning_rate (float, optional): Base learning rate for CD updates. Defaults to 1e-3.
        momentum (float, optional): Momentum coefficient for CD updates. Defaults to 0.5.
        weight_decay (float, optional):  Weight decay (L2 regularization) coefficient. Defaults to 1e-7.
        n_epochs (int, optional): Number of training epochs. Defaults to 20.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        dataset_path (str, optional): Path to download/store the MNIST dataset. Defaults to "data".
        output_dir (str, optional): Directory to save the generated images. Defaults to "samples".
        output_filename (str, optional): File name for saving the generated image grid. Defaults to "generated_images.png".
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

    # Train RBM
    train_loop(
        train_loader,
        rbm,
        n_epochs=n_epochs,
        n_gibbs_steps=n_gibbs_steps,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    # Generate and save samples
    generate_and_save_images(
        rbm,
        rows=8,
        columns=8,
        steps=1000,
        output_dir=output_dir,
        output_filename=output_filename,
    )


if __name__ == "__main__":
    # Run an example of fitting a Restricted Boltzmann Machine to the MNIST dataset
    train_rbm(
        n_visible=784,
        n_hidden=500,
        n_gibbs_steps=10,
        learning_rate=1e-3,
        momentum=0.5,
        weight_decay=1e-7,
        n_epochs=50,
        batch_size=64,
    )
