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


def run(use_qpu: bool, num_reads: int, batch_size: int, n_iterations: int, fully_visible: bool):
    """Run an example of fitting a graph-restricted Boltzmann machine to synthetic data generated
    uniformly randomly.

    Args:
        use_qpu (bool): Flag indicating whether to sample with a QPU or a classical sampler.
        num_reads (int): Sample size of spins sampled from the model.
        batch_size (int): Batch size of data.
        n_iterations (int): Number of training iterations.
        fully_visible (bool): Flag indicating whether the model should be fully visible.
    """
    if use_qpu:
        sampler = DWaveSampler(solver="Advantage2_system1.3")
        zephyr_grid_size = sampler.properties['topology']['shape'][0]
        G = sampler.to_networkx_graph()
        sample_kwargs = dict(
            num_reads=num_reads,
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
            num_reads=num_reads,
            beta_range=[1, 1],
            proposal_acceptance_criteria="Gibbs",
            randomize_order=True,
        )
        zephyr_grid_size = 6
        G = zephyr_graph(zephyr_grid_size)
        h_range = j_range = None
        # In contrast to the prefactor for the QPU, the MCMC sampler can sample at a designated
        # inverse temperature or annealing parameter (one), so no scaling is required (one).
        prefactor = 1.0

    if fully_visible:
        hidden_nodes = None
        n_vis = G.number_of_nodes()
        kind = None
    else:
        # Use a four-colouring of the Zephyr graph to determine a set of conditionally-independent
        # nodes to define as hidden units.
        linear_to_zephyr = zephyr_coordinates(zephyr_grid_size).linear_to_zephyr
        qubit_colour = {g: zephyr_four_color(linear_to_zephyr(g)) for g in G}
        hidden_nodes = [q for q, c in qubit_colour.items() if c == 0]
        n_hid = len(hidden_nodes)
        n_vis = G.number_of_nodes() - n_hid
        kind = "sampling"

    # Generate fake data to fit the Boltzmann machine to
    # Make sure ``x`` is of type float
    X = 1 - 2.0 * torch.randint(0, 2, (n_iterations, batch_size, n_vis))

    # Instantiate the model
    grbm = GRBM(G.nodes, G.edges, hidden_nodes)

    # Instantiate the optimizer
    opt_grbm = SGD(grbm.parameters(), 0.1)

    # Example of one iteration in a training loop
    # Generate a sample set from the model
    for iteration, x in enumerate(X):
        # Sample from the model
        s = grbm.sample(sampler, prefactor=prefactor, linear_range=h_range,
                        quadratic_range=j_range, sample_params=sample_kwargs)

        # Measure the effective inverse temperature
        measured_beta = grbm.estimate_beta(s)

        # Reset the gradients of the model weights
        opt_grbm.zero_grad()

        # Compute a quasi-objective---this quasi-objective yields the same gradient as the negative
        # log likelihood of the model
        quasi = grbm.quasi_objective(
            x, s, kind=kind, sampler=sampler, sample_kwargs=sample_kwargs,
            prefactor=prefactor, linear_range=h_range, quadratic_range=j_range
        )

        # Backpropgate gradients
        quasi.backward()

        # Update model weights with a step of stochastic gradient descent
        opt_grbm.step()

        # Compute the average (absolute) gradient to monitor convergence
        avg_grad = (grbm._linear.grad.abs().mean() + grbm._quadratic.grad.abs().mean())/2

        print(
            f"Iteration: {iteration}, Average |gradient|: {avg_grad.item():.2f}, Effective inverse temperature: {measured_beta:.4f}"
        )


if __name__ == "__main__":
    # Run example of fitting a fully-visible Boltzmann machine with a classical sampler
    run(use_qpu=False, num_reads=100, batch_size=100, n_iterations=3, fully_visible=True)
