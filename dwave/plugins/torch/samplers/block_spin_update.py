from collections import defaultdict
from time import perf_counter

import dwave_networkx as dnx
import torch
from networkx import Graph
from torch import nn

from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.utils import bit2spin_soft, rands


class BlockSpinUpdateSampler(nn.Module):
    """A block-spin update sampler for Chimera-, Pegasus-, and Zephyr-structured graph-restricted Boltzmann machines.

    Note this is a sampler and not a neural network. It extends `nn.Module` for the convenience of
    managing devices of indices (stored as parameters).

    Due to the sparse definition of GRBMs (for better or worse), some tedious, and ugly, indexing
    tricks were employed. Ideally, an adjacency list can be used, however, adjacencies are ragged,
    which makes vectorization inapplicable.

    Block-Gibbs and Block-Metropolis obey detailed balance and are ergodic methods at finite
    temperature, which at fixed parameters converge upon Boltzmann distributions. Block-Metropolis
    allows higher acceptance rates for proposals (faster single-step mixing), but is non-ergodic in
    the limit of zero temperature. Decorrelation from an initial condition can be slower.
    Block-Gibbs represents best practice for independent sampling.

    Args:
        G (Graph): A Chimera, Pegasus, or Zephyr graph.
        grbm (GRBM): The Graph-Restricted Boltzmann Machine to sample from.
    """

    def __init__(self, G: Graph, grbm: GRBM, num_reads: int, kind: str):
        super().__init__()
        topology = G.graph.get("family", None)
        if topology not in {"zephyr", "pegasus", "chimera"}:
            raise NotImplementedError("TODO")
        G = G.copy()
        self._kind = kind
        self.G = G
        self.grbm = grbm
        self.edge_idx_i = grbm.edge_idx_i
        self.edge_idx_j = grbm.edge_idx_j
        self.node_to_idx = grbm.node_to_idx
        self.idx_to_node = grbm.idx_to_node
        self.partition = self.get_partition(topology)
        self.padded_adjacencies, self.padded_adjacencies_weight = self.get_adjacencies()
        self.x = nn.Parameter(rands((num_reads, grbm.n_nodes)), requires_grad=False)
        self.linear = grbm._linear
        self.quadratic = grbm._quadratic

    @property
    def kind(self):
        return self._kind

    def get_adjacencies(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Create two lists of padded adjacency lists (tensors), one for neighbouring indices and
        another for the corresponding weight indices.

        The issue begins with the adjacency lists being ragged. To address this, we pad adjacencies
        with -1 values. The exact values do not matter, as the way these adjacencies will be used is
        by padding an input state with 0s, so when accessing `-1`, the output will be masked out.

        Returns:
            tuple[list[torch.Tensor], list[torch.Tensor]]: The first output is a padded adjacency
            list of neighbouring indices where -1 indicates empty. The second output is an adjacency
            list of weight indices (in sparse indexing) used to retrieve the corresponding edge
            weight.
        """
        max_degree = max([self.G.degree[v] for v in self.G])
        padded_adjacencies = nn.Parameter(
            -torch.ones(len(self.G), max_degree, dtype=int), requires_grad=False
        )
        padded_adjacencies_weight = nn.Parameter(
            -torch.ones(len(self.G), max_degree, dtype=int), requires_grad=False
        )

        adjacency_dict = defaultdict(list)
        edge_to_idx = dict()
        for idx, (u, v) in enumerate(
            zip(self.edge_idx_i.tolist(),
                self.edge_idx_j.tolist())):
            adjacency_dict[v].append(u)
            adjacency_dict[u].append(v)
            edge_to_idx[u, v] = idx
            edge_to_idx[v, u] = idx
        for u in self.idx_to_node:
            neighbours = adjacency_dict[u]
            adj_weight_idxs = [edge_to_idx[u, v] for v in neighbours]
            num_neighbours = len(neighbours)
            padded_adjacencies[u][:num_neighbours] = torch.tensor(neighbours)
            padded_adjacencies_weight[u][:num_neighbours] = torch.tensor(adj_weight_idxs)
        return padded_adjacencies, padded_adjacencies_weight

    def get_partition(self, topology):
        partition = defaultdict(list)
        if topology == "zephyr":
            lin2lattice = dnx.zephyr_coordinates(self.G.graph['rows']).linear_to_zephyr
            crayon = dnx.zephyr_four_color
        elif topology == "pegasus":
            lin2lattice = dnx.pegasus_coordinates(self.G.graph['rows']).linear_to_pegasus
            crayon = dnx.pegasus_four_color
        elif topology == "chimera":
            lin2lattice = dnx.chimera_coordinates(self.G.graph['rows']).linear_to_chimera
            crayon = dnx.chimera_two_color
        else:
            raise ValueError("Invalid topology.")

        for lin in self.G:
            idx = self.node_to_idx[lin]
            c = crayon(lin2lattice(lin))
            partition[c].append(idx)
        partition = nn.ParameterList([
            nn.Parameter(torch.tensor(partition[k], requires_grad=False), requires_grad=False)
            for k in sorted(partition)
        ])

        return partition

    @torch.compile
    @torch.no_grad
    def step_(self, beta: torch.Tensor):
        """Performs a block-Gibbs update in-place.

        Args:
            beta (float): Effective inverse temperature to sample at.
        """
        num_reads = self.x.shape[0]
        zeros = torch.zeros(num_reads, device=self.x.device).unsqueeze(1)
        for block in self.partition:
            xnbr = torch.hstack([self.x, zeros])[:, self.padded_adjacencies[block]]
            h = self.linear[block]
            J = self.quadratic[self.padded_adjacencies_weight[block]]
            effective_field = (xnbr * J.unsqueeze(0)).sum(2) + h
            if self.kind == "metropolis":
                delta = -2*self.x[:, block]*effective_field
                prob = (-delta*beta).exp().clip(0, 1)
                # if the delta field is negative, then flipping the spin will improve the energy
                prob[delta <= 0] = 1
                flip = -bit2spin_soft(prob.bernoulli())
                self.x[:, block] = self.x[:, block]*flip
            elif self.kind == "gibbs":
                prob = 1/(1+torch.exp(2*effective_field*beta))
                spins = bit2spin_soft(prob.bernoulli())
                self.x[:, block] = spins
            else:
                raise ValueError(f"Invalid kind: {self.kind}")

    @torch.no_grad
    def run_(self, schedule: torch.Tensor):
        # NOTE: this is a silly way to force compilation for native and tensor-valued scalars.
        # There is a further distinction between 0 and nonzero values.
        # TODO: investigate the cases to cover and find a better way to force compilation.
        self.step_(0)
        self.step_(0.0)
        self.step_(0.01)
        self.step_(torch.tensor(0))
        self.step_(torch.tensor(0.0))
        self.step_(torch.tensor(0.01))
        torch.cuda.empty_cache()
        t0 = perf_counter()
        for beta in schedule:
            self.step_(beta)
        return perf_counter() - t0
