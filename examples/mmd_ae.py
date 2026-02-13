from itertools import cycle
from math import prod

import dwave_networkx as dnx
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import SGD, AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.v2 import Compose, ToDtype, ToImage
from torchvision.utils import make_grid, save_image

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.nn.functional import bit2spin_soft, spin2bit_soft
from dwave.system import DWaveSampler


class RadialBasisFunction(nn.Module):

    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.register_buffer("bandwidth_multipliers", bandwidth_multipliers)
        self.bandwidth = bandwidth

    def get_bandwidth(self, l2_dist):
        if self.bandwidth is None:
            n = l2_dist.shape[0]
            avg = l2_dist.sum() / (n**2 - n)  # (diagonal is zero)
            return avg

        return self.bandwidth

    def forward(self, X):
        l2 = torch.cdist(X, X) ** 2
        bandwidth = self.get_bandwidth(l2.detach()) * self.bandwidth_multipliers
        res = torch.exp(-l2.unsqueeze(0) / bandwidth.reshape(-1, 1, 1)).sum(dim=0)
        return res


class MMDLoss(nn.Module):
    def __init__(self, kernel):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X.flatten(1), Y.flatten(1)]))
        n = X.shape[0]
        m = Y.shape[0]
        XX = (K[:n, :n].sum() - K[:n, :n].trace()) / (n*(n-1))
        YY = (K[n:, n:].sum() - K[n:, n:].trace()) / (m*(m-1))
        XY = K[:n, n:].mean()
        mmd = XX - 2 * XY + YY
        return mmd


class SkipLinear(nn.Module):
    def __init__(self, din, dout) -> None:
        super().__init__()
        self.linear = nn.Linear(din, dout, bias=False)

    def forward(self, x):
        return self.linear(x)


class LinearBlock(nn.Module):
    def __init__(self, din, dout, sn, p, bias) -> None:
        super().__init__()
        self.skip = SkipLinear(din, dout)
        linear_1 = nn.Linear(din, dout, bias)
        linear_2 = nn.Linear(dout, dout, bias)
        self.block = nn.Sequential(
            nn.LayerNorm(din),
            linear_1,
            nn.Dropout(p),
            nn.ReLU(),
            nn.LayerNorm(dout),
            linear_2,
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)


class ConvolutionBlock(nn.Module):
    def __init__(self, input_shape: tuple[int, int, int], cout: int):
        super().__init__()
        input_shape = tuple(input_shape)
        cin, hx, wx = input_shape
        if hx != wx:
            raise NotImplementedError("TODO")

        self.input_shape = tuple(input_shape)
        self.cin = cin
        self.cout = cout

        self.block = nn.Sequential(
            nn.LayerNorm(input_shape),
            nn.Conv2d(cin, cout, 3, 1, 1),
            nn.ReLU(),
            nn.LayerNorm((cout, hx, wx)),
            nn.Conv2d(cout, cout, 3, 1, 1),
        )
        self.skip = SkipConv2d(cin, cout)

    def forward(self, x):
        return self.block(x) + self.skip(x)


class SkipConv2d(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.skip = nn.Conv2d(cin, cout, 1, bias=False)

    def forward(self, x):
        return self.skip(x)


class ConvolutionNetwork(nn.Module):
    def __init__(
            self, channels: list[int], input_shape: tuple[int, int, int]
    ):
        super().__init__()
        channels = channels.copy()
        input_shape = tuple(input_shape)
        cx, hx, wx = input_shape
        if hx != wx:
            raise NotImplementedError("TODO")
        self.channels = channels
        self.cin = cx
        self.cout = self.channels[-1]
        self.input_shape = input_shape

        channels_in = [cx] + channels[:-1]
        self.blocks = nn.Sequential()
        for cin, cout in zip(channels_in, channels):
            self.blocks.append(ConvolutionBlock((cin, hx, wx), cout))
            self.blocks.append(nn.ReLU())
        self.blocks.pop(-1)
        self.skip = SkipConv2d(cx, cout)

    def forward(self, x):
        x = self.blocks(x) + self.skip(x)
        return x


class FullyConnectedNetwork(nn.Module):
    def __init__(self, din, dout, depth, sn, p, bias=True) -> None:
        super().__init__()
        if depth == 1:
            raise ValueError("Depth must be at least 2.")
        self.skip = SkipLinear(din, dout)
        big_d = max(din, dout)
        dims = [big_d]*(depth-1) + [dout]
        self.blocks = nn.Sequential()
        for d_in, d_out in zip([din]+dims[:-1], dims):
            self.blocks.append(LinearBlock(d_in, d_out, sn, p, bias))
            self.blocks.append(nn.Dropout(p))
            self.blocks.append(nn.ReLU())
        # Remove the last ReLU and Dropout
        self.blocks.pop(-1)
        self.blocks.pop(-1)

    def forward(self, x):
        return self.blocks(x) + self.skip(x)


def straight_through_bitrounding(fuzzy_bits):
    if not ((fuzzy_bits >= 0) & (fuzzy_bits <= 1)).all():
        raise ValueError(f"Inputs should be in [0, 1]: {fuzzy_bits}")
    bits = fuzzy_bits + (fuzzy_bits.round() - fuzzy_bits).detach()
    return bits


class StraightThroughTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.hth = nn.Tanh()

    def forward(self, x):
        fuzzy_spins = self.hth(x)
        fuzzy_bits = spin2bit_soft(fuzzy_spins)
        bits = straight_through_bitrounding(fuzzy_bits)
        spins = bit2spin_soft(bits)
        return spins


def zephyr_subgraph(G, zephyr_m):
    Z_m = dnx.zephyr_graph(zephyr_m)
    zsm = next(dnx.zephyr_sublattice_mappings(Z_m, G))
    S = G.subgraph([zsm(z) for z in Z_m])
    original_m = S.graph['rows']
    if original_m == zephyr_m:
        return G.copy()
    S.graph = G.graph.copy()
    S.graph['rows'] = zephyr_m
    S.graph['columns'] = zephyr_m
    S.graph['name'] = S.graph['name'].replace(f"({original_m},", "("+str(zephyr_m)+",")
    S.graph['name'] = S.graph['name'] + "-subgraph of " + G.graph['name']
    return S


def subtile(G, num_tiles):
    zc = dnx.zephyr_coordinates(G.graph['rows'], 4)
    return G.subgraph([g for g in G if zc.linear_to_zephyr(g)[2] < num_tiles])


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
            ConvolutionNetwork([chidden]*(depth_cnn-1) + [1], (chidden, h, w)),
            # nn.Sigmoid()
        )

    def decode(self, q):
        xhat = self.decoder(q)
        return xhat

    def forward(self, x):
        z = self.encoder(x)
        spins = self.binarizer(z)
        xhat = self.decode(spins)
        return z, spins, xhat


def collect_stats(model, grbm, x, q, compute_mmd, compute_pkl):
    z, s, xhat = model(x)
    stats = {
        "quasi": grbm.quasi_objective(s.detach(), q),
        "mse": nn.functional.mse_loss(xhat.sigmoid(), x),
        "bce": nn.functional.binary_cross_entropy_with_logits(xhat, x),
        "mmd": compute_mmd(s, q),
        "pkl": compute_pkl(grbm, z, s, q),
    }
    return stats


def get_dataset(bs, data_dir="/tmp/"):
    transforms = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
    train_kwargs = dict(root=data_dir, download=True)
    transforms = Compose([transforms, lambda x: 1 - x])
    data_train = MNIST(transform=transforms, **train_kwargs)
    train_loader = DataLoader(data_train, bs, True)
    data_test = MNIST(transform=transforms, **train_kwargs, train=False)
    test_loader = DataLoader(data_test, bs, True)
    return train_loader, test_loader


def save_viz(step, grbm, model, x, q):
    bs = min(x.shape[0], 500)
    rows = int(bs**0.5)
    with torch.no_grad():
        # Save images
        xgen = model.decode(q[:bs]).sigmoid()
        xuni = model.decode(bit2spin_soft(torch.randint_like(q[:bs], 2))).sigmoid()
        z, s, xhat = model(x[:bs])
        xhat = xhat.sigmoid()
        xgrid = make_grid(x[:bs], rows, pad_value=1)
        xgengrid = make_grid(xgen, rows, pad_value=1)
        xunigrid = make_grid(xuni, rows, pad_value=1)
        xhatgrid = make_grid(xhat, rows, pad_value=1)
        save_image(xgrid, "x.png")
        save_image(xgengrid, "xgen.png")
        save_image(xunigrid, "xuni.png")
        save_image(xhatgrid, "xhat.png")


def get_qpu_model_grbm(solver, device):
    # Set up QPU and QPU parameters
    qpu = DWaveSampler(solver=solver)
    # Instantiate model
    # G = zephyr_subgraph(qpu.to_networkx_graph(), 4)
    G = subtile(zephyr_subgraph(qpu.to_networkx_graph(), 5), 3)
    nodes = list(G.nodes)
    edges = list(G.edges)
    grbm = GRBM(nodes, edges).to(device)
    # grbm.linear.data[:] = 0
    # grbm.quadratic.data[:] = 0
    model = Autoencoder((1, 28, 28), grbm.n_nodes).to(device)
    return qpu, model, grbm


def run(*, title, loss_fn, solver, stop_grbm, num_reads,
        annealing_time, alpha, num_steps, args):
    device = "cuda"
    qpu, model, grbm = get_qpu_model_grbm(solver, device)
    nprng = np.random.default_rng(8257213849)
    grbm.linear.data[:] = 0.1 * bit2spin_soft(torch.tensor(nprng.binomial(1, 0.5, grbm.n_nodes)))
    grbm.quadratic.data[:] = bit2spin_soft(torch.tensor(nprng.binomial(1, 0.5, grbm.n_edges)))
    sampler = qpu

    model.train()
    grbm.train()

    opt_grbm = SGD(grbm.parameters(), lr=1e-3)
    opt_model = AdamW(model.parameters(), lr=1e-3)

    sample_params = dict(num_reads=num_reads, annealing_time=annealing_time,
                         answer_mode="raw", auto_scale=False)
    h_range, j_range = qpu.properties["h_range"], qpu.properties["j_range"]

    # Set up data
    train_loader, test_loader = get_dataset(num_reads)

    compute_mmd = MMDLoss(RadialBasisFunction()).to(device)

    def compute_pkl(grbm: GRBM, logits_data: torch.Tensor, spins_data: torch.Tensor,
                    spins_model: torch.Tensor):
        probabilities = torch.sigmoid(logits_data)
        entropy = torch.nn.functional.binary_cross_entropy_with_logits(logits_data, probabilities)
        # bce = p(log(q)) + (1-p) log(1-q)
        cross_entropy = grbm.quasi_objective(spins_data, spins_model)
        pkl = cross_entropy - entropy
        return pkl

    for step, (x, _) in enumerate(cycle(train_loader), 1):
        torch.cuda.empty_cache()
        if step > num_steps:
            break
        # Send data to device
        x = x.to(device).float()
        q = grbm.sample(sampler, prefactor=1,
                        linear_range=h_range, quadratic_range=j_range,
                        device=device, sample_params=sample_params)

        # Train autoencoder
        stats = collect_stats(model, grbm, x, q, compute_mmd, compute_pkl)
        opt_model.zero_grad()
        (stats["bce"] + alpha*stats[loss_fn]).backward()
        # alpha ~ 1e-6
        opt_model.step()

        # Train GRBM
        if step < stop_grbm:
            # NOTE: collecting stats again because the autoencoder has been updated.
            stats = collect_stats(model, grbm, x, q, compute_mmd, compute_pkl)
            opt_grbm.zero_grad()
            stats['quasi'].backward()
            opt_grbm.step()

        print(title, step, {k: f"{v.item():.4f}"
                            if isinstance(v, torch.Tensor)
                            else f"{v:.4f}"
                            for k, v in stats.items()})

        if step % 10 == 0:
            model.eval()

            xtest = next(iter(test_loader))[0].to(device)
            q = grbm.sample(sampler, prefactor=1,
                            linear_range=h_range, quadratic_range=j_range,
                            device=device, sample_params=sample_params)
            stats = collect_stats(model, grbm, xtest, q, compute_mmd, compute_pkl)
            save_viz(step, grbm, model, x, q)

            model.train()


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--title", type=str, default="NoExperimentName")
    parser.add_argument("--annealing_time", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=1_000)
    parser.add_argument("--num_reads", type=int, default=1000)
    parser.add_argument("--stop_grbm", type=int, default=500)
    parser.add_argument("--loss_fn", type=str, default="mmd")
    parser.add_argument("--solver", type=str, default="Advantage2_system1.11")
    args_ = parser.parse_args()

    args_dict = vars(args_)
    run(**args_dict, args=args_)
    # postprocess(**args_dict, args=args_)
