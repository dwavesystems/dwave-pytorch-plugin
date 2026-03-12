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

torch.manual_seed(123)  

def run():
    # RBM
    n_visible, n_hidden = 50, 20
    visible_nodes = [f"v{i}" for i in range(n_visible)]
    hidden_nodes  = [f"h{j}" for j in range(n_hidden)]
    nodes = visible_nodes + hidden_nodes
    edges = [[v, h] for v in visible_nodes for h in hidden_nodes]
    grbm = GRBM(nodes, edges, hidden_nodes=hidden_nodes)

    num_chains = 100
    sampler = BipartiteGibbsSampler(grbm, num_chains=num_chains, schedule=[1.0], seed=123)


    n_iterations = 3
    batch_size = 100
    n_visible = len(visible_nodes)

    X = 1 - 2.0 * torch.randint(0, 2, (n_iterations, batch_size, n_visible))

    optimizer = SGD(grbm.parameters(), lr=0.1)

    for iteration, x in enumerate(X):    
        # Positive phase: clamp visible, sample hidden
        full_obs = torch.full((batch_size, grbm.n_nodes), float("nan"))
        full_obs[:, grbm.visible_idx] = x
        obs = sampler.sample(full_obs)
        
        # Negative phase: sample from the model
        s_model = sampler.sample()

        optimizer.zero_grad()

        # Compute a quasi-objective function
        loss = (
            grbm.sufficient_statistics(obs).mean(0, True)
            - grbm.sufficient_statistics(s_model).mean(0, True)
        ) @ grbm.theta

        loss.backward()
        optimizer.step()

        avg_grad = (
            grbm._linear.grad.abs().mean()
            + grbm._quadratic.grad.abs().mean()
        ) / 2

        print(f"Iteration {iteration:3d} | "f"Average |gradient|: {avg_grad.item():.2f}")
    print("\nTraining finished.")

if __name__ == "__main__":
    run()