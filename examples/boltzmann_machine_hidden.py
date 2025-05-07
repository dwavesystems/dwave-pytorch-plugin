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

import torch

from dwave.plugins.torch.boltzmann_machine import GraphRestrictedBoltzmannMachine
from dwave.plugins.torch.utils import make_sampler_and_graph
from dwave.system import DWaveSampler
from torch.optim import SGD
from dwave.samplers import SimulatedAnnealingSampler
from dwave_networkx import zephyr_graph, zephyr_four_color, zephyr_coordinates


if __name__ == "__main__":
    USE_QPU = False
    NUM_READS = 100
    SAMPLE_SIZE = 17

    if USE_QPU:
        qpu = DWaveSampler(solver="Advantage2_prototype2.6")
        h_range, j_range = qpu.properties["h_range"], qpu.properties["j_range"]
        # A helper function that wraps the QPU with a
        # `dwave.system.FixedEmbeddingComposite` so it can sample a model with
        # contiguous nonnegative integer variable names---this is an implementation
        # requirement of the graph-restricted Boltzmann machine
        sampler, G = make_sampler_and_graph(qpu)
        sample_kwargs = dict(
            num_reads=NUM_READS,
            # Set `answer_mode` to "raw" so no samples are aggregated
            answer_mode="raw",
            # Set `auto_scale`` to `False` so the sampler sample from the intended
            # distribution
            auto_scale=False,
        )
        m, t = qpu.properties["topology"]["shape"]
    else:
        # Use an MCMC sampler that can sample from the equilibrium distribution
        sampler = SimulatedAnnealingSampler()
        sample_kwargs = dict(
            num_reads=NUM_READS,
            beta_range=[1, 1],
            proposal_acceptance_criterion="Gibbs",
            randomize_order=True,
        )
        h_range = j_range = None
        m = 1
        t = 4
        G = zephyr_graph(m, t)

    num_nodes = G.number_of_nodes()

    # Use Zephyr-colours to identify a set of qubits for which we will use as hiddens.
    if USE_QPU:
        linear2qpu = lambda lin: sampler.embedding.get(lin)[0]
    else:
        linear2qpu = lambda lin: lin
    zcoord = zephyr_coordinates(m, t)
    qpu2zephyr = zcoord.linear_to_zephyr
    colours = torch.tensor([zephyr_four_color(qpu2zephyr(linear2qpu(g))) for g in G])
    mask_hid = colours == 0
    hidx = mask_hid.argwhere().flatten()

    # Generate fake data to fit the Boltzmann machine to
    # Make sure ``x`` is of type float
    n_vis = num_nodes - hidx.shape[0]
    x = 1 - 2.0 * torch.randint(0, 2, (SAMPLE_SIZE, n_vis))

    # Instantiate the model
    grbm = GraphRestrictedBoltzmannMachine(
        num_nodes,
        *torch.tensor(list(G.edges)).mT,
        h_range=h_range,
        j_range=j_range,
        hidx=hidx,
    )

    # Instantiate the optimizer
    opt_grbm = SGD(grbm.parameters())

    # Example of one iteration in a training loop
    # Generate a sample set from the model
    s = grbm.sample(sampler, **sample_kwargs)
    # Reset the gradients of the model weights
    opt_grbm.zero_grad()
    # Compute the objective---this objective yields the same gradient as the negative
    # log likelihood of the model
    objective = grbm.objective(x, s)
    # Backpropgate gradients
    objective.backward()
    # Update model weights with a step of stochastic gradient descent
    opt_grbm.step()
