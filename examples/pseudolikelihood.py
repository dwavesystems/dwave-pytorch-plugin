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
"""Maximum pseudolikelihood: fitting a fully visible Boltzmann machine without a sampler, and
estimating the inverse temperature at which spins were sampled.

The script plants a Boltzmann machine on the complete graph of 5 nodes, draws exact samples of
its distribution by enumerating the 32 configurations, and fits a second model to them with
``fit_pseudolikelihood``: no sampler is involved, the fit is a convex optimization. It then draws
samples of the planted model at inverse temperature 1.7 and recovers that temperature with
``fit_inverse_temperature``, with a standard error, as one estimates the effective inverse
temperature of a quantum annealer from its reads. Offline, a few seconds on a CPU::

    python examples/pseudolikelihood.py
"""
import itertools

import torch

from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine
from dwave.plugins.torch.pseudolikelihood import fit_inverse_temperature


def exact_samples(model, num_samples: int, beta: float = 1.0, generator=None) -> torch.Tensor:
    """Samples of the Boltzmann distribution of a small model at inverse temperature ``beta``,
    by enumerating its configurations."""
    states = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=model.n_nodes)))
    with torch.no_grad():
        probabilities = torch.softmax(-beta * model(states), 0)
    rows = torch.multinomial(probabilities, num_samples, replacement=True, generator=generator)
    return states[rows]


def run(num_samples: int = 50_000, seed: int = 0) -> dict:
    """Fit a planted model by maximum pseudolikelihood and estimate the temperature of its samples.

    Args:
        num_samples (int): Samples drawn for each of the two fits.
        seed (int): Seed of the planted biases and of the samples.

    Returns:
        dict: The diagnostics of the joint fit, the largest errors of its biases, and the
        temperature estimates.
    """
    generator = torch.Generator().manual_seed(seed)
    nodes = list(range(5))
    edges = [(i, j) for i in nodes for j in nodes if i < j]
    linear = torch.rand(len(nodes), generator=generator) - 0.5
    quadratic = 2 * torch.rand(len(edges), generator=generator) - 1
    truth = GraphRestrictedBoltzmannMachine(
        nodes, edges,
        linear={node: float(h) for node, h in zip(nodes, linear)},
        quadratic={edge: float(J) for edge, J in zip(edges, quadratic)},
    )

    # The joint fit: the biases of a fresh model are replaced by the estimate
    data = exact_samples(truth, num_samples, generator=generator)
    model = GraphRestrictedBoltzmannMachine(nodes, edges)
    fit = model.fit_pseudolikelihood(data)
    with torch.no_grad():
        error_linear = float((model.linear - truth.linear).abs().max())
        error_quadratic = float((model.quadratic - truth.quadratic).abs().max())
    print(f"Joint fit: {fit.nlpl:.4f} nats per configuration after {fit.n_iterations} L-BFGS "
          f"iterations, converged={fit.converged}")
    print(f"Largest error of a linear bias {error_linear:.4f}, of a quadratic bias "
          f"{error_quadratic:.4f}")

    # The temperature fit: the planted biases are the fixed Hamiltonian of samples drawn at
    # inverse temperature 1.7
    beta_true = 1.7
    cold = exact_samples(truth, num_samples, beta=beta_true, generator=generator)
    temperature = fit_inverse_temperature(truth, truth.linear, truth.quadratic, cold)
    print(f"Inverse temperature of samples drawn at {beta_true}: {float(temperature.beta):.4f} "
          f"± {float(temperature.beta_stderr):.4f}")
    # Against the fitted model, the training data are at unit inverse temperature, up to the
    # rounding of the estimate to the model's float32 parameters
    own = fit_inverse_temperature(model, model.linear, model.quadratic, data)
    print(f"Inverse temperature of the training data under the fitted model: "
          f"{float(own.beta):.6f}")
    return dict(fit=fit, error_linear=error_linear, error_quadratic=error_quadratic,
                temperature=temperature, own_temperature=own)


if __name__ == "__main__":
    run()
