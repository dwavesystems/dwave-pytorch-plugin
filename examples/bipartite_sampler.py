# Copyright 2026 D-Wave
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

from torch.optim import SGD
from dwave.plugins.torch.models.boltzmann_machine import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.samplers.bipartite_sampler import BipartiteGibbsSampler

def run():
    """Run an example of fitting a graph-restricted Boltzmann machine with a BipartiteGibbsSampler
    to synthetic data generated uniformly at random.
    """
    # RBM
    n_visible, n_hidden = 50, 20
    visible_nodes = [f"v{i}" for i in range(n_visible)]
    hidden_nodes  = [f"h{j}" for j in range(n_hidden)]
    nodes = visible_nodes + hidden_nodes
    edges = [[v, h] for v in visible_nodes for h in hidden_nodes]
    grbm = GRBM(nodes, edges, hidden_nodes=hidden_nodes)

    num_chains = batch_size = 100
    sampler = BipartiteGibbsSampler(grbm, num_chains=num_chains, schedule=[1.0], seed=123)


    n_iterations = 3
    n_visible = len(visible_nodes)

    X = 1 - 2.0 * torch.randint(0, 2, (n_iterations, batch_size, n_visible))

    optimizer = SGD(grbm.parameters(), lr=0.1)

    for iteration, x in enumerate(X):    
        # Positive phase: clamp visible, sample hidden
        obs = sampler.sample(grbm.pad_visible(x))

        # Negative phase: sample from the model
        s_model = sampler.sample()

        optimizer.zero_grad()

        # Compute a quasi-objective function whose gradient is that of the negative log likelihood
        loss = grbm.quasi_objective(obs, s_model)

        loss.backward()
        optimizer.step()

        avg_grad = (
            grbm.linear.grad.abs().mean()
            + grbm.quadratic.grad.abs().mean()
        ) / 2

        print(f"Iteration {iteration:3d} | "f"Average |gradient|: {avg_grad.item():.2f}")
    print("\nTraining finished.")

if __name__ == "__main__":
    torch.manual_seed(123)  
    run()