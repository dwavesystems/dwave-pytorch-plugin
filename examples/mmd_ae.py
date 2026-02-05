from itertools import cycle
from math import prod
from os import makedirs

import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.v2 import Compose, ToDtype, ToImage
from torchvision.utils import make_grid, save_image

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.nn.modules import (ConvolutionNetwork, FullyConnectedNetwork,
                                            MaximumMeanDiscrepancy, RadialBasis,
                                            StraightThroughTanh, rands_like, zephyr_subgraph)
from dwave.system import DWaveSampler


@torch.compile
class Autoencoder(nn.Module):
    def __init__(self, shape, n_bits):
        super().__init__()
        dim = prod(shape)
        c, h, w = shape
        chidden = 1
        depth_fcnn = 3
        depth_cnn = 3
        dropout = 0.0
        self.encoder = nn.Sequential(
            ConvolutionNetwork([chidden]*depth_cnn, shape),
            nn.Flatten(),
            FullyConnectedNetwork(chidden*h*w, n_bits, depth_fcnn, False, dropout),
        )
        self.binarizer = StraightThroughTanh()
        self.decoder = nn.Sequential(
            FullyConnectedNetwork(n_bits, chidden*h*w, depth_fcnn, False, dropout),
            nn.Unflatten(1, (chidden, h, w)),
            ConvolutionNetwork([chidden]*(depth_cnn-1) + [1], (chidden, h, w))
        )

    def decode(self, q):
        xhat = self.decoder(q)
        return xhat

    def forward(self, x):
        spins = self.binarizer(self.encoder(x))
        xhat = self.decode(spins)
        return spins, xhat


def collect_stats(model, grbm, x, q, compute_mmd):
    s, xhat = model(x)
    stats = {
        "quasi": grbm.quasi_objective(s.detach(), q),
        "bce": nn.functional.binary_cross_entropy_with_logits(xhat, x),
        "mmd": compute_mmd(s, q),
    }
    return stats


def get_dataset(bs, data_dir="/tmp/"):
    transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    train_kwargs = dict(root=data_dir, download=True)
    transforms = Compose([transforms, lambda x: 1 - x])
    data_train = MNIST(transform=transforms, **train_kwargs)
    train_loader = DataLoader(data_train, bs, True)
    return train_loader


def round_graph_down(graph, group_size):
    n_in = graph.number_of_nodes()
    no = group_size*(n_in//group_size)
    return graph.subgraph(list(graph.nodes)[:no])


def run(*, num_steps):
    sampler = DWaveSampler(solver="Advantage2_system1.8")
    sample_params = dict(num_reads=500, annealing_time=0.5, answer_mode="raw", auto_scale=False)
    h_range, j_range = sampler.properties["h_range"], sampler.properties["j_range"]
    outdir = "output/example_mmd_ae/"
    makedirs(outdir, exist_ok=True)

    device = "cuda"

    # Setup data
    train_loader = get_dataset(500)

    # Instantiate model
    G = zephyr_subgraph(sampler.to_networkx_graph(), 4)
    nodes = list(G.nodes)
    edges = list(G.edges)
    grbm = GRBM(nodes, edges).to(device)
    model = Autoencoder((1, 28, 28), grbm.n_nodes).to(device)
    model.train()
    grbm.train()

    compute_mmd = MaximumMeanDiscrepancy(RadialBasis()).to(device)

    opt_grbm = SGD(grbm.parameters(), lr=1e-3)
    opt_ae = AdamW(model.parameters(), lr=1e-3)

    for step, (x, y) in enumerate(cycle(train_loader)):
        torch.cuda.empty_cache()
        if step > num_steps:
            break
        # Send data to device
        x = x.to(device).float()

        q = grbm.sample(sampler, prefactor=1, linear_range=h_range, quadratic_range=j_range,
                        device=device, sample_params=sample_params)

        # Train autoencoder
        stats = collect_stats(model, grbm, x, q, compute_mmd)
        opt_ae.zero_grad()
        (stats["bce"] + stats["mmd"]).backward()
        opt_ae.step()

        # Train GRBM
        if step < 1000:
            # NOTE: collecting stats because the autoencoder has been updated.
            stats = collect_stats(model, grbm, x, q, compute_mmd)
            opt_grbm.zero_grad()
            stats['quasi'].backward()
            opt_grbm.step()
        print(step, {k: v.item() for k, v in stats.items()})
        if step % 10 == 0:
            with torch.no_grad():
                grbm.eval()
                xgen = model.decode(q[:100])
                xuni = model.decode(rands_like(q[:100]))
                xhat = model(x[:100])[-1]
                save_image(make_grid(x[:100], 10, pad_value=1), outdir + "x.png")
                save_image(make_grid(xgen.sigmoid(), 10, pad_value=1), outdir + "xgen.png")
                save_image(make_grid(xhat.sigmoid(), 10, pad_value=1), outdir + "xhat.png")
                save_image(make_grid(xuni.sigmoid(), 10, pad_value=1), outdir + "xuni.png")
                grbm.train()


if __name__ == "__main__":
    run(num_steps=10_000)
