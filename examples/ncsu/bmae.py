from itertools import cycle

import numpy as np
import torch
from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.nn.functional import bit2spin_soft
from dwave.plugins.torch.samplers.dimod_sampler import DimodSampler
from dwave.samplers import SimulatedAnnealingSampler as Neal
from torch import nn
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.v2 import Compose, ToDtype, ToImage
from torchvision.utils import make_grid, save_image


class ResidualBlock(nn.Module):
    def __init__(self, din: int, dout: int) -> None:
        super().__init__()
        self.skip = nn.Linear(din, dout, False)
        linear_1 = nn.Linear(din, dout, True)
        linear_2 = nn.Linear(dout, dout, True)
        self.block = nn.Sequential(
            nn.LayerNorm(din),
            linear_1,
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.LayerNorm(dout),
            linear_2,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


def straight_through_bitrounding(fuzzy_bits: torch.Tensor) -> torch.Tensor:
    if not ((fuzzy_bits >= 0) & (fuzzy_bits <= 1)).all():
        raise ValueError(f"Inputs should be in [0, 1]: {fuzzy_bits}")
    bits = fuzzy_bits + (fuzzy_bits.round() - fuzzy_bits).detach()
    return bits


class Binarize(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.act = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fuzzy_bits = self.act(x)
        bits = straight_through_bitrounding(fuzzy_bits)
        spins = bit2spin_soft(bits)
        return spins


class Autoencoder(nn.Module):

    def __init__(self, n_bits: int) -> None:
        super().__init__()
        c, h, w = (1, 28, 28)
        dim = c * h * w
        self.encoder = nn.Sequential(
            nn.Flatten(),
            ResidualBlock(dim, n_bits),
        )
        self.binarizer = Binarize()
        self.decoder = nn.Sequential(
            ResidualBlock(n_bits, dim),
            nn.ReLU(),
            ResidualBlock(dim, dim),
            nn.Unflatten(1, (c, h, w)),
        )

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        spins = self.binarizer(z)
        logits = self.decoder(spins)
        return z, spins, logits


def get_dataset(bs: int) -> tuple[DataLoader, DataLoader]:
    transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    train_kwargs = dict(download=True)
    transforms = Compose([transforms, lambda x: 1 - x])
    data_train = MNIST("/tmp/", transform=transforms, **train_kwargs)
    train_loader = DataLoader(data_train, bs, True)
    data_test = MNIST("/tmp/", transform=transforms, **train_kwargs, train=False)
    test_loader = DataLoader(data_test, bs, True)
    return train_loader, test_loader


def fit_grbm(nbits, bs):
    nodes = list(range(nbits))
    edges = [(i, j) for i in range(nbits) for j in range(i, nbits)]
    linear = {v: 0.0 for v in nodes}
    quadratic = {e: np.random.standard_normal() * nbits**-0.5 for e in edges}
    grbm = GRBM(nodes, edges, linear=linear, quadratic=quadratic).to(DEVICE)

    model = Autoencoder(nbits).to(DEVICE)
    model.load_state_dict(torch.load("autoencoder.pt"))
    model.eval()

    sampler = DimodSampler(
        grbm, Neal(), 1.0,
        sample_kwargs=dict(
            num_reads=bs,
            beta_range=[0.01, 1.0],
            num_sweeps=100,
            proposal_acceptance_criteria="Gibbs"
        )
    )
    opt_grbm = SGD(grbm.parameters(), lr=1e-2, nesterov=True, momentum=0.8)

    train_loader, test_loader = get_dataset(bs=64)
    for step, (x, _) in enumerate(cycle(train_loader)):
        if step > 3000:
            break
        x = x.to(DEVICE).float()
        with torch.no_grad():
            z, s, logits = model(x)

        q = sampler.sample()

        quasi = grbm.quasi_objective(s, q)
        opt_grbm.zero_grad()
        quasi.backward()
        opt_grbm.step()

        if step % 100 == 0:
            q = sampler.sample()
            xgen = model.decoder(q).sigmoid()
            save_image(make_grid(xgen, 8, pad_value=1), "xgen.png")
            print(f"Step: {step}, quasi: {quasi.item()}")


def fit_autoencoder(nbits, bs):
    model = Autoencoder(nbits).to(DEVICE)
    opt_model = AdamW(model.parameters(), lr=1e-3)
    # Set up data
    train_loader, test_loader = get_dataset(bs=bs)
    for step, (x, _) in enumerate(cycle(train_loader), 1):
        if step > 3000:
            break
        # Send data to device
        x = x.to(DEVICE).float()

        z, s, logits = model(x)

        loss = nn.functional.binary_cross_entropy_with_logits(logits, x)
        opt_model.zero_grad()
        loss.backward()
        opt_model.step()

        if step % 100 == 0:
            model.eval()
            xtest = next(iter(test_loader))[0].to(DEVICE)
            _, _, logits_test = model(xtest)
            loss_test = nn.functional.binary_cross_entropy_with_logits(logits_test, xtest)
            save_image(make_grid(logits_test.sigmoid(), 8, pad_value=1), "xhat.png")
            print(f"Iteration {step}, Train: {loss.item():.2f}, Test: {loss_test.item():.2f}")
            model.train()
            torch.save(model.state_dict(), f"autoencoder.pt")


if __name__ == "__main__":
    DEVICE = "cuda"
    _nbits = 32
    _bs = 64
    fit_autoencoder(_nbits, _bs)
    fit_grbm(_nbits, _bs)
