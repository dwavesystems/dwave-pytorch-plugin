import time
from typing import Literal

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import make_grid

from dwave.plugins.torch.autoencoder import AutoEncoder
from dwave.plugins.torch.autoencoder.losses import pseudo_kl_divergence_loss
from dwave.plugins.torch.autoencoder.losses.mmd import RadialBasisFunction, mmd_loss
from dwave.plugins.torch.boltzmann_machine import GraphRestrictedBoltzmannMachine

USE_QPU = True
NUM_READS = 512
LOSS_FUNCTION: Literal["kl", "mmd"] = "mmd"


class Encoder(torch.nn.Module):
    def __init__(self, n_latents: int = 256):
        super().__init__()
        channels = [1, 32, 64, 128, n_latents]
        layers = []
        for i in range(len(channels) - 1):
            # A convolutional layer does not modify the image size
            layers.append(
                torch.nn.Conv2d(
                    channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1
                )
            )
            # Batch normalisation is used to stabilise the learning process
            layers.append(torch.nn.BatchNorm2d(channels[i + 1]))
            # We downsample the image size by 2
            layers.append(torch.nn.MaxPool2d(kernel_size=2, stride=2))
            # Finally, we apply a non-linearity
            layers.append(torch.nn.LeakyReLU())
        layers = layers[:-1]  # Remove the last LeakyReLU
        self.conv = torch.nn.Sequential(*layers)
        self.flatten_last_two_dims = torch.nn.Flatten(start_dim=-2, end_dim=-1)
        self.projection = torch.nn.Linear(2 * 2, 1)
        self.flatten = torch.nn.Flatten()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.flatten_last_two_dims(x)
        x = self.projection(x)
        return self.flatten(x)


class Decoder(torch.nn.Module):
    def __init__(self, n_latents: int = 256):
        super().__init__()
        channels = [n_latents, 128, 64, 32, 1]
        layers = []
        # The input will be of shape (batch_size, n_latents), we need to project it
        # to the shape (batch_size, n_latents, 2, 2)
        self.projection = torch.nn.Linear(n_latents, n_latents * 2 * 2)
        self.unflatten = torch.nn.Unflatten(1, (n_latents, 2, 2))
        for i in range(len(channels) - 1):
            # A transposed convolutional layer does not modify the image size
            layers.append(
                torch.nn.ConvTranspose2d(
                    channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1
                )
            )
            # Batch normalisation is used to stabilise the learning process
            layers.append(torch.nn.BatchNorm2d(channels[i + 1]))
            # We upsample the image size by 2
            layers.append(torch.nn.Upsample(scale_factor=2))
            # Finally, we apply a non-linearity
            layers.append(torch.nn.LeakyReLU())
        # We append a last convolutional transpose layer to obtain the final image
        layers.append(
            torch.nn.ConvTranspose2d(
                channels[-1], channels[-1], kernel_size=3, stride=1, padding=1
            )
        )
        self.convtrans = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.projection(x)
        x = self.unflatten(x)
        x = self.convtrans(x)
        return x


def get_sampler_and_sampler_kwargs():
    from dwave.plugins.torch.utils import make_sampler_and_graph
    from dwave.system import DWaveSampler

    qpu = DWaveSampler(solver="Advantage2_prototype2.6")
    sampler, graph = make_sampler_and_graph(qpu)
    if USE_QPU:
        h_range, j_range = qpu.properties["h_range"], qpu.properties["j_range"]
        sampler_kwargs = dict(
            num_reads=NUM_READS,
            # Set `answer_mode` to "raw" so no samples are aggregated
            answer_mode="raw",
            # Set `auto_scale`` to `False` so the sampler sample from the intended
            # distribution
            auto_scale=False,
            annealing_time=200,
        )
    else:

        from dwave.samplers import SimulatedAnnealingSampler

        sampler = SimulatedAnnealingSampler()
        h_range = j_range = None
        sampler_kwargs = dict(
            num_reads=NUM_READS,
            beta_range=[1, 1],
            proposal_acceptance_criterion="Gibbs",
            randomize_order=True,
        )
    return sampler, graph, h_range, j_range, sampler_kwargs


def train_discrete_vae_from_scratch(n_latents: int = 256):
    # Set the random seed for reproducibility
    torch.manual_seed(42975)

    # Load the dataset and create the dataloader
    dataset = MNIST(
        root="data",
        train=True,
        download=True,
        transform=Compose([Resize((32, 32)), ToTensor()]),
    )
    batch_size = 128
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Set the device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    autoencoder = AutoEncoder(
        encoder=Encoder(n_latents=256), decoder=Decoder(n_latents=256)
    )
    autoencoder.to(device)

    sampler, graph, h_range, j_range, sampler_kwargs = get_sampler_and_sampler_kwargs()
    # We use a subgraph of the QPU graph to train the model. A better (more connected)
    # subgraph could be obtained by using a heuristic to select the nodes of the
    # subgraph. For now, we just take the first n_latents nodes of the graph.
    graph = graph.subgraph(range(n_latents))

    grbm = GraphRestrictedBoltzmannMachine(
        n_latents,
        *torch.tensor(list(graph.edges)).mT,
        h_range=h_range,
        j_range=j_range,
    )

    optimizer = torch.optim.Adam(
        list(autoencoder.parameters()) + list(grbm.parameters()),
        lr=1e-3,
        weight_decay=1e-5,
    )

    mse_losses = []
    other_losses = []

    if LOSS_FUNCTION == "mmd":
        kernel = RadialBasisFunction(num_features=7)
        kernel.to(device)
        loss_name = "MMD"
        other_loss_constant = 1.0
    else:
        kernel = None
        loss_name = "Pseudo KL Divergence"
        other_loss_constant = 1e-6

    n_epochs = 15
    start_time = time.time()
    for epoch in range(n_epochs):
        for batch in dataloader:
            x, _ = batch
            x = x.to(device)
            x_hat, spins, logits = autoencoder(x)

            optimizer.zero_grad()
            mse_loss = torch.nn.functional.mse_loss(x_hat, x)
            mse_losses.append(mse_loss.item())
            if LOSS_FUNCTION == "mmd":
                other_loss = mmd_loss(
                    spins=spins,
                    kernel=kernel,
                    boltzmann_machine=grbm,
                    sampler=sampler,
                    sampler_kwargs=sampler_kwargs,
                )
            else:
                other_loss = pseudo_kl_divergence_loss(
                    spins=spins,
                    logits=logits,
                    boltzmann_machine=grbm,
                    sampler=sampler,
                    sampler_kwargs=sampler_kwargs,
                )
            other_losses.append(other_loss.item())
            # We multiply the pseudo KL divergence by a small constant to have both
            # losses in the same range.
            loss = mse_loss + other_loss_constant * other_loss
            loss.backward()
            optimizer.step()
        print(
            f"Epoch {epoch + 1}/{n_epochs} - MSE Loss: {mse_loss.item():.4f} - "
            f"{loss_name} Loss: {other_loss.item():.4f}. Time: "
            f"{(time.time() - start_time)/60:.2f} mins"
        )

        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        axes[0].plot(mse_losses)
        axes[0].set_title("MSE Loss")
        axes[0].set_xlabel("Batch")
        axes[0].set_ylabel("Loss")
        axes[1].plot(other_losses)
        axes[1].set_title(f"{loss_name} Loss")
        axes[1].set_xlabel("Batch")
        axes[1].set_ylabel("Loss")
        axes[1].set_xlabel("Batch")
        axes[1].set_ylabel("Loss")
        plt.tight_layout()
        # Save the plot
        plt.savefig(f"losses_qpu_epoch_{epoch + 1}.png", dpi=300)
        # Close the plot and figure:
        plt.close(fig)
        plt.close("all")

        # Now we use the trained autoencoder both to generate new samples as well as to
        # show the reconstruction of the input samples.
        batch = next(iter(dataloader))[0]
        reconstructed_batch, _, _ = autoencoder(batch.to(device))
        images_per_row = 16
        reconstruction_tensor_for_plot = make_grid(
            torch.cat(
                (
                    batch.cpu(),
                    torch.ones((images_per_row, 1, 32, 32)),
                    reconstructed_batch.cpu(),
                ),
                dim=0,
            ),
            nrow=images_per_row,
        )
        plt.imshow(reconstruction_tensor_for_plot.permute(1, 2, 0))
        plt.axis("off")
        plt.title("Reconstruction of the input samples")
        plt.savefig(f"reconstruction_qpu_epoch_{epoch + 1}.png", dpi=300)
        plt.close("all")

        # Now we generate new samples
        samples = grbm.sample(sampler, **sampler_kwargs)
        images = autoencoder.decoder(samples).cpu()
        generation_tensor_for_plot = make_grid(images, nrow=images_per_row)
        plt.imshow(generation_tensor_for_plot.permute(1, 2, 0))
        plt.axis("off")
        plt.title("Generated samples")
        plt.savefig(f"generated_samples_qpu_epoch_{epoch + 1}.png", dpi=300)
        plt.close("all")
        # Save the model
        torch.save(autoencoder.state_dict(), f"discrete_vae_qpu_epoch_{epoch + 1}.pth")
        # Save the RBM
        torch.save(grbm.state_dict(), f"rbm_qpu_epoch_{epoch + 1}.pth")


if __name__ == "__main__":
    train_discrete_vae_from_scratch()
