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
"""Evaluating generative models of spins by interaction order, on samples of a Boltzmann machine.

The data are samples of a graph-restricted Boltzmann machine on a random 4-regular graph of 48
nodes with random biases (the *teacher*), drawn by a ``BlockSampler`` with one chain per row
swept long enough to mix, and cut into the three splits of an ``Evaluator``. Four candidates
are scored against the test split: the two moment-matched baselines of
``dwave.plugins.torch.models`` fit on the training split (independent spins at the data's means,
and the dichotomized Gaussian matched to its means and pairwise tables); a *student*, a
Boltzmann machine on the teacher's graph trained on the training split for a few steps of
persistent contrastive divergence; and a fresh, independent sample of the teacher itself, which
reproduces the data's law and therefore reads at the noise floor. The script prints the
captured fractions, calibrates the headline kernel on the training and validation splits,
prints the normalized headlines, evaluates the differentiable training loss once, and, with
``--outdir``, draws the three figures of ``docs/evaluation.md``: ``fourier_weights.png`` (every
order), ``fourier_weights_low.png`` (the first 12 orders) and ``kernel_spectra.png`` (which
orders each fixed kernel can see). Offline, a minute or two on a CPU::

    python examples/evaluation.py [--outdir docs/media]
"""
import argparse

import networkx as nx
import torch

from dwave.plugins.torch.evaluation import Evaluator
from dwave.plugins.torch.models import DichotomizedGaussian, GraphRestrictedBoltzmannMachine
from dwave.plugins.torch.models import IndependentBernoulli
from dwave.plugins.torch.samplers import BlockSampler

TRAIN_ROWS, VAL_ROWS, TEST_ROWS = 16384, 4096, 8192
BLOCK_SIZE = 512  # rows per block: 32, 8 and 16 blocks for the three splits
SWEEPS = 1000  # block-Gibbs sweeps of every chain from a random state: far past mixing here
STUDENT_STEPS = 25  # steps of persistent contrastive divergence: an under-trained candidate


def teacher_samples(teacher: GraphRestrictedBoltzmannMachine, rows: int, seed: int) -> torch.Tensor:
    """``rows`` independent samples of ``teacher``: one chain per row, swept ``SWEEPS`` times."""
    sampler = BlockSampler(teacher, num_chains=rows, schedule=[1.0] * SWEEPS, seed=seed)
    return sampler.sample().to(torch.float64)


def train_student(graph: nx.Graph, train: torch.Tensor) -> GraphRestrictedBoltzmannMachine:
    """A Boltzmann machine on ``graph`` after ``STUDENT_STEPS`` steps of persistent contrastive
    divergence on ``train`` (see ``examples/boltzmann_machine.py`` for the training loop)."""
    student = GraphRestrictedBoltzmannMachine(graph.nodes, graph.edges)
    sampler = BlockSampler(student, num_chains=512, seed=5)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.05)
    generator = torch.Generator().manual_seed(6)
    for _ in range(STUDENT_STEPS):
        batch = train[torch.randint(0, train.shape[0], (512,), generator=generator)].float()
        optimizer.zero_grad()
        student.quasi_objective(batch, sampler.sample()).backward()
        optimizer.step()
    return student


def run(outdir: str | None = None) -> None:
    """Score the candidates against the teacher's samples; draw the figures.

    Args:
        outdir (str, optional): Directory for the figures, or ``None`` to print only.
    """
    graph = nx.random_regular_graph(4, 48, seed=0)
    teacher = GraphRestrictedBoltzmannMachine(graph.nodes, graph.edges)
    with torch.no_grad():  # random biases: strong enough for structure at every order
        teacher.linear.normal_(0.0, 0.3)
        teacher.quadratic.normal_(0.0, 1.0)
    dim = teacher.n_nodes
    train, val, test, fresh = (
        teacher_samples(teacher, rows, seed)
        for rows, seed in ((TRAIN_ROWS, 1), (VAL_ROWS, 2), (TEST_ROWS, 3), (TEST_ROWS, 4))
    )
    evaluator = Evaluator(train, val, test, block_size=BLOCK_SIZE)
    print(f"teacher: a Boltzmann machine on a random 4-regular graph, dim={dim}; splits of "
          f"{TRAIN_ROWS}, {VAL_ROWS} and {TEST_ROWS} rows in blocks of {BLOCK_SIZE}")

    candidates = {}
    for name, model in (("independent Bernoulli", IndependentBernoulli(dim)),
                        ("dichotomized Gaussian", DichotomizedGaussian(dim))):
        model.fit(train)
        candidates[name] = model.sample(TEST_ROWS).to(torch.float64)
    student = train_student(graph, train)
    candidates[f"student ({STUDENT_STEPS} steps)"] = teacher_samples(student, TEST_ROWS, 7)
    candidates["the teacher"] = fresh

    # Evaluation: per-profile scores read against the data's own distance from uniform
    results = {name: evaluator.score(sample, "test") for name, sample in candidates.items()}
    data, data_se = evaluator.uniform_score("test")
    print("\ncaptured fraction of the data's order-k Fourier weight, 1 - scores[k] / data[k]")
    print(f"{'':24s} " + " ".join(f"k={k:<6d}" for k in (1, 2, 3, 4, 6)) + " gaussian")
    for name, (scores, se) in results.items():
        captured = [1 - float(scores[k] / data[k]) for k in (1, 2, 3, 4, 6)]
        print(f"{name:24s} " + " ".join(f"{c:7.3f} " for c in captured)
              + f" {1 - float(scores[0] / data[0]):7.3f}")
    snr = data[1:] / data_se[1:]
    print(f"the data's own Fourier weight clears 3 standard errors at {int((snr >= 3).sum())} "
          f"of {dim} orders")

    # Calibration: the headline kernel follows the orders a test-sized sample can measure
    calibration = evaluator.calibrate()
    print(f"\ncalibrated headline: {int(calibration.resolved.sum())} of {dim} orders measurable, "
          "weights on orders 1-4: "
          + ", ".join(f"{w:.3f}" for w in calibration.alpha[:4].tolist()))
    baseline, _ = evaluator.uniform_score("test")
    for name, sample in candidates.items():
        scores, se = evaluator.score(sample, "test")
        print(f"{name:24s} headline {scores[0].item():.3e} +- {se[0].item():.1e}   "
              f"normalized {scores[0].item() / baseline[0].item():.3f}")
    evaluator.alpha = None  # the figures below keep the Gaussian headline

    # Training: a differentiable per-profile loss of a soft batch against a training batch
    batch = torch.tanh(2 * candidates["dichotomized Gaussian"][:512] + 0.1).requires_grad_()
    loss = evaluator.compute_loss(batch, block_size=128)
    loss[0].backward()
    print(f"\ncompute_loss on a 512-row soft batch (block_size=128): gaussian {loss[0].item():.3e}, "
          f"order 2 {loss[2].item():.3e}; mean |gradient| {batch.grad.abs().mean().item():.2e}")

    if outdir is not None:
        import pathlib

        from dwave.plugins.torch.evaluation.visualize import plot_fourier_weights, plot_kernel_spectra

        outdir = pathlib.Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        title = "Fourier weight by interaction order: samples of a Boltzmann machine on 48 nodes"
        figures = {
            "fourier_weights.png": plot_fourier_weights(evaluator, results, "test", title=title),
            "fourier_weights_low.png": plot_fourier_weights(
                evaluator, results, "test", title=title + ", k <= 12", max_order=12
            ),
            "kernel_spectra.png": plot_kernel_spectra(dims=(48, 4800), max_order=32),
        }
        for name, figure in figures.items():
            figure.savefig(outdir / name, dpi=150)
            print(f"wrote {outdir / name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outdir", default=None,
                        help="directory for the figures of docs/evaluation.md (default: none)")
    args = parser.parse_args()
    torch.manual_seed(0)
    run(args.outdir)
