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
from dwave_networkx import zephyr_coordinates, zephyr_four_color, zephyr_graph
from torch.optim import SGD

from dwave.plugins.torch.boltzmann_machine import GRBM
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler

if __name__ == "__main__":
    USE_QPU = True
    NUM_READS = 100
    BATCH_SIZE = 17
    N_ITERATIONS = 10
    FULLY_VISIBLE = True

    if USE_QPU:
        sampler = DWaveSampler(solver="Advantage2_prototype2.6")
        G = sampler.to_networkx_graph()
        sample_kwargs = dict(
            num_reads=NUM_READS,
            # Set `answer_mode` to "raw" so no samples are aggregated
            answer_mode="raw",
            # Set `auto_scale`` to `False` so the sampler sample from the intended
            # distribution
            auto_scale=False,
        )
        h_range = sampler.properties["h_range"]
        j_range = sampler.properties["j_range"]
    else:
        # Use an MCMC sampler that can sample from the equilibrium distribution
        sampler = SimulatedAnnealingSampler()
        sample_kwargs = dict(
            num_reads=NUM_READS,
            beta_range=[1, 1],
            proposal_acceptance_criterion="Gibbs",
            randomize_order=True,
        )
        G = zephyr_graph(6)
        h_range = j_range = None

    if FULLY_VISIBLE:
        hiddens = None
        n_vis = G.number_of_nodes()
    else:
        linear_to_zephyr = zephyr_coordinates(6).linear_to_zephyr
        qubit_colour = {g: zephyr_four_color(linear_to_zephyr(g)) for g in G}
        hiddens = [q for q, c in qubit_colour.items() if c == 0]
        n_hid = len(hiddens)
        n_vis = G.number_of_nodes() - n_hid

    # Generate fake data to fit the Boltzmann machine to
    # Make sure ``x`` is of type float
    X = 1 - 2.0 * torch.randint(0, 2, (N_ITERATIONS, BATCH_SIZE, n_vis))

    # Instantiate the model
    grbm = GRBM(G.nodes, G.edges, hiddens, h_range, j_range)

    # Instantiate the optimizer
    opt_grbm = SGD(grbm.parameters())

    # Example of one iteration in a training loop
    # Generate a sample set from the model
    for iteration, x in zip(range(N_ITERATIONS), X):
        s = grbm.sample(sampler, 1 / 6.6, sample_params=sample_kwargs)
        measured_beta = grbm.estimate_beta(s)
        # Reset the gradients of the model weights
        opt_grbm.zero_grad()
        # Compute the objective---this objective yields the same gradient as the negative
        # log likelihood of the model
        objective = grbm.objective(x, s)
        # Backpropgate gradients
        objective.backward()
        # Update model weights with a step of stochastic gradient descent
        opt_grbm.step()
        print(
            f"iteration: {iteration}, obj: {objective.item():.2f}, beta: {measured_beta:.4f}"
        )
