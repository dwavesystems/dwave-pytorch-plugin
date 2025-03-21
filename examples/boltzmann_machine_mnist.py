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

from math import prod
from functools import reduce
from itertools import product
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import dwave_networkx as dnx
import networkx as nx
import numpy as np

from dwave.plugins.torch.boltzmann_machine import GraphRestrictedBoltzmannMachine
from dwave.plugins.torch.utils import make_sampler_and_graph
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dwave.samplers import SimulatedAnnealingSampler
from torch import nn
from torch.optim import SGD, AdamW
from torchvision.datasets import MNIST
from torchvision.transforms.v2 import Compose, Resize, ToImage, ToDtype
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torchvision.transforms import InterpolationMode
from hybrid.decomposers import make_origin_embeddings

mpl.use("agg")


def make_sampler_and_graph_visual(qpu: DWaveSampler, shape: tuple[int, int]):
    T = qpu.to_networkx_graph()
    n_pixels = prod(shape)
    assert n_pixels <= T.number_of_nodes()
    # Ideally you embed on full graph then drop unused vertices, but dropping vertices
    # first makes the code much simpler
    T.remove_nodes_from(
        np.random.choice(list(T), T.number_of_nodes() - n_pixels, replace=False)
    )
    qpu_to_linear = {v: lin for lin, v in enumerate(T)}
    linear_to_qpu = {lin: v for v, lin in qpu_to_linear.items()}
    # Get layout
    pos_qpu = dnx.pegasus_layout(T)
    # Shift to [0, 1]^2
    pos_qpu = {v: (a, b + 1) for v, (a, b) in pos_qpu.items()}
    # Relabel to use linear coordinates
    pos_qpu = {qpu_to_linear[v]: coord for v, coord in pos_qpu.items()}
    # Sort the order of dict for turning it into an array
    pos_qpu = {v: pos_qpu[v] for v in sorted(pos_qpu)}
    assert np.allclose(list(pos_qpu.keys()), sorted(pos_qpu.keys()))
    # Relabel nodes linearly
    T = nx.relabel_nodes(T, qpu_to_linear)

    # Create array
    pos_qpu = np.array(list(pos_qpu.values()))
    w, h = shape
    if w != h:
        raise NotImplementedError("TODO: check h, w are not swapped etc.")
    # Get image positions
    pos_im = np.transpose(
        [np.repeat(np.arange(h), w), np.tile(np.arange(w), h)]
    ).astype(float)
    # Rescale to [0, 1]^2
    pos_im[:, 0] = pos_im[:, 0] / (w - 1.0)
    pos_im[:, 1] = pos_im[:, 1] / (h - 1.0)
    # Compute the pairwise distance matrix
    pairwise_dist = cdist(pos_im, pos_qpu)
    # Find a matching
    a, b = linear_sum_assignment(pairwise_dist)
    assert np.allclose(a, sorted(a))
    permutation = {i: t for i, t in enumerate(b.tolist())}
    T = nx.relabel_nodes(T, permutation)
    embedding = {permutation[lin]: (q,) for lin, q in linear_to_qpu.items()}
    sampler = FixedEmbeddingComposite(qpu, embedding)
    return sampler, T


def make_sampler_graph_mapping_filled(qpu):
    T = qpu.to_networkx_graph()
    l2q = {lin: (q,) for lin, q in enumerate(T)}
    q2l = {qq[0]: lin for lin, qq in l2q.items()}
    T = nx.relabel_nodes(T, q2l)
    # Target graph is now linearly labelled

    emb_king = make_origin_embeddings(qpu, "kings")[0]
    emb_king = {g: tuple(map(q2l.get, c)) for g, c in emb_king.items()}
    # emb_king now maps grid to linear coordinates of qubit

    shape = [max(emb_king, key=lambda x: x[i])[i] + 1 for i in [0, 1]]
    linq2g = dict()
    for g, c in emb_king.items():
        for q in c:
            linq2g[q] = g
    nodes = list(T)
    np.random.shuffle(nodes)
    for linq in T:
        # If the linear qubit is already assigned, skip it
        if linq in linq2g:
            continue
        # Otherwise, let's assign it a neighbouring pixel
        neighbors = list(T.neighbors(linq))
        np.random.shuffle(neighbors)
        for nbr in neighbors:
            # If it's neighbour has a pixel, assign the same pixel.
            if nbr in linq2g:
                linq2g[linq] = linq2g[nbr]
                break
    assert len(T) == len(linq2g), "every linear qubit should be assigned to a pixel"
    used_pixels = set(list(linq2g.values()))
    g2lim = dict()
    lim = 0
    for g in product(range(shape[0]), range(shape[1])):
        if g in used_pixels:
            g2lim[g] = lim
            lim += 1
    mapping = torch.tensor([g2lim[linq2g[t]] for t in T])
    sampler = FixedEmbeddingComposite(qpu, l2q)
    return sampler, T, mapping


def make_sampler_graph_corrupted_king(qpu):
    T = qpu.to_networkx_graph()
    emb_king = make_origin_embeddings(qpu, "kings")[0]
    emb_king = {k: emb_king[k] for k in sorted(emb_king)}
    for chain in emb_king.values():
        nx.contracted_edge(T, chain, False, False)
    shape = [max(emb_king, key=lambda x: x[i])[i] + 1 for i in [0, 1]]
    plt.figure(figsize=(16, 16))
    plt.clf()
    dnx.draw_pegasus_embedding(qpu.to_networkx_graph(), emb=emb_king)
    plt.savefig("P16King.png")

    occupied_qubits = reduce(set.union, emb_king.values(), set())
    T.remove_nodes_from(set(T) - occupied_qubits)

    present = list()
    lim2qpu = dict()
    t2lim = dict()
    n_missing = 0
    lin = 0
    for g in product(range(shape[0]), range(shape[1])):
        if g in emb_king:
            chain = emb_king[g]
            lim2qpu[lin] = chain
            present.append(lin)
            t2lim[chain[0]] = lin
            lin += 1
        else:
            n_missing += 1
    sampler = FixedEmbeddingComposite(qpu, lim2qpu)
    T = nx.relabel_nodes(T, t2lim)
    return sampler, T, torch.tensor(present)


if __name__ == "__main__":
    USE_QPU = True
    NUM_READS = 500
    SAMPLE_SIZE = 17
    upsize = Resize((45, 45), interpolation=InterpolationMode.NEAREST)
    downsize = Resize((28, 28))

    mnist = MNIST(
        "/tmp/",
        transform=Compose([ToImage(), ToDtype(torch.float32, scale=True), upsize]),
        download=True,
    )
    train_loader = DataLoader(mnist, 50_000)
    x, y = next(iter(train_loader))
    x = x.flatten(1)
    x = (x > 0.5).float()
    x = 1 - 2 * x

    neal = SimulatedAnnealingSampler()
    qpu = DWaveSampler(solver="Advantage_system4.1")
    h_range, j_range = qpu.properties["h_range"], qpu.properties["j_range"]

    sampler, G, mapping = make_sampler_graph_mapping_filled(qpu)
    # sampler, G, mapping = make_sampler_graph_corrupted_king(qpu)
    # sampler, G = make_sampler_and_graph_visual(qpu, upsize.size)
    # plt.figure(figsize=(16, 16))
    # plt.clf()
    # idx = 9
    # x_draw = x[idx].tolist()
    # save_image(downsize(x[idx].reshape(upsize.size).unsqueeze(0)), "x_draw.png")
    # inverse = {qc[0]: logic for logic, qc in sampler.embedding.items()}
    # Q = qpu.to_networkx_graph()
    # linear_biases = {q: (1 - x_draw[inverse[q]]) / 2 if q in inverse else 0 for q in Q}
    # dnx.draw_pegasus(
    #     Q,
    #     linear_biases=linear_biases,
    #     vmin=-1,
    #     vmax=1,
    #     node_size=100,
    #     cmap=plt.cm.binary,
    # )
    # plt.savefig("G_emb")
    sample_kwargs = dict(
        num_reads=NUM_READS,
        annealing_time=1000,
        answer_mode="raw",
        auto_scale=False,
    )
    if not USE_QPU:
        sampler = neal
        sample_kwargs.pop("answer_mode")
        sample_kwargs.pop("auto_scale")
        sample_kwargs.pop("annealing_time")
        sample_kwargs["beta_range"] = [1, 1]
        sample_kwargs["num_sweeps"] = 500
        h_range = j_range = None

    # Instantiate the model
    grbm = GraphRestrictedBoltzmannMachine(
        len(mapping), *torch.tensor(list(G.edges)).mT, h_range=h_range, j_range=j_range
    )

    # Instantiate the optimizer
    opt_grbm = AdamW(grbm.parameters(), lr=0.01)

    with_neal = True
    for iteration in range(1000):
        s = grbm.sample(sampler, **sample_kwargs)
        if with_neal:
            s = torch.tensor(
                neal.sample_ising(
                    *grbm.ising,
                    num_sweeps=1,
                    initial_states=s.int().tolist(),
                    beta_range=[1.0, 1.0],
                    proposal_acceptance_criteria="Gibbs",
                    randomize_order=True,
                ).record.sample,
                dtype=torch.float32,
            )
        opt_grbm.zero_grad()
        objective = grbm.objective(x[:, mapping], s)
        objective.backward()
        opt_grbm.step()
        s_ = torch.zeros(
            [
                s.shape[0],
            ]
            + upsize.size
        ).flatten(1)
        s_[:, mapping] = s
        im = (1 + s_.unflatten(1, (1, *upsize.size))) / 2
        save_image(make_grid(downsize(im), 10), f"xgen-{with_neal}.png")
        print(iteration, objective.item())
    print("Ex.ted")
