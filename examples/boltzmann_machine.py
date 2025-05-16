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

from dwave.plugins.torch.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import DWaveSampler

if __name__ == "__main__":
    USE_QPU = False
    NUM_READS = 100
    BATCH_SIZE = 100
    N_ITERATIONS = 10
    FULLY_VISIBLE = True

    if USE_QPU:
        sampler = DWaveSampler(solver="Advantage2_prototype2.6")
        zephyr_grid_size = sampler.properties['topology']['shape'][0]
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
        # A ball-park estimate used later to scale the Hamiltonian for the QPU such that it is
        # effectively sampling at, approximately, an effective inverse temperature of one.
        prefactor = 1.0/6.35
    else:
        # Use an MCMC sampler that can sample from the equilibrium distribution
        sampler = SimulatedAnnealingSampler()
        # Parameters chosen to reflect a valid MCMC sampler (despite the name "simulated annealing")
        sample_kwargs = dict(
            num_reads=NUM_READS,
            beta_range=[1, 1],
            proposal_acceptance_criterion="Gibbs",
            randomize_order=True,
        )
        zephyr_grid_size = 6
        G = zephyr_graph(zephyr_grid_size)
        h_range = j_range = None
        # In contrast to the prefactor for the QPU, the MCMC sampler can sample at a designated
        # inverse temperature or annealing parameter (one), so no scaling is required (one).
        prefactor = 1.0

    if FULLY_VISIBLE:
        hidden_nodes = None
        n_vis = G.number_of_nodes()
    else:
        # Use a four-colouring of the Zephyr graph to determine a set of conditionally-independent
        # nodes to define as hidden units.
        linear_to_zephyr = zephyr_coordinates(zephyr_grid_size).linear_to_zephyr
        qubit_colour = {g: zephyr_four_color(linear_to_zephyr(g)) for g in G}
        hidden_nodes = [q for q, c in qubit_colour.items() if c == 0]
        n_hid = len(hidden_nodes)
        n_vis = G.number_of_nodes() - n_hid

    # Generate fake data to fit the Boltzmann machine to
    # Make sure ``x`` is of type float
    X = 1 - 2.0 * torch.randint(0, 2, (N_ITERATIONS, BATCH_SIZE, n_vis))

    # Instantiate the model
    grbm = GRBM(list(G.nodes), list(G.edges), hidden_nodes)

    # Instantiate the optimizer
    opt_grbm = SGD(grbm.parameters(), 0.1)

    # Example of one iteration in a training loop
    # Generate a sample set from the model
    for iteration, x in enumerate(X):
        # Sample from the model
        s = grbm.sample(sampler, prefactor, h_range, j_range, sample_params=sample_kwargs)

        # Measure the effective inverse temperature
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

        # Compute the average (absolute) gradient to monitor convergence
        avg_grad = (grbm._linear.grad.abs().mean() + grbm._quadratic.grad.abs().mean())/2

        print(
            f"Iteration: {iteration}, Average |gradient|: {avg_grad.item():.2f}, Effective inverse temperature: {measured_beta:.4f}"
        )
