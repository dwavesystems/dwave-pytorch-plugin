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
"""Moment-matched baselines against a synthetic sample, scored by interaction order.

The script draws a 12-spin dichotomized Gaussian with zero means (the family of
``DichotomizedGaussian`` itself), fits every baseline of ``BASELINES`` on one sample, builds the
*spin-space match* (the sample's spin mean and covariance fed to the family's sampler as if they
were the latent parameters, a natural but wrong moment match), scores everything with an
``Evaluator`` against an independent sample, and prints the captured fraction of the data's
distance from uniform at the Gaussian kernel and at interaction orders 2 and 4. With
``--outdir`` it also draws the two figures of ``docs/baselines.md``: ``link_function.png`` (the
family's link function and what each match realizes) and ``captured_by_order.png`` (the captured
fraction by interaction order). Offline, under a minute on a CPU::

    python examples/baselines.py [--outdir docs/media]
"""
import argparse
import math

import torch

from dwave.plugins.torch.evaluation import Evaluator
from dwave.plugins.torch.models import BASELINES


def sign_of_gaussian(mean: torch.Tensor, corr: torch.Tensor, n: int) -> torch.Tensor:
    """``n`` rows of ``sign(mean + L eps)``, ``L L^T = corr``: the family's definition, written out.

    The synthetic truth is drawn this way from known latent parameters, and the spin-space foil
    by feeding it the spin mean and covariance instead.
    """
    chol = torch.linalg.cholesky(corr.double())
    latent = torch.randn(n, mean.shape[0], dtype=torch.float64) @ chol.mT + mean.double()
    return torch.where(latent >= 0, 1.0, -1.0).to(torch.float64)


def run(outdir: str | None = None) -> dict:
    """Fit and score the baselines on a synthetic sample; draw the figures when ``outdir`` is given.

    Args:
        outdir (str, optional): Directory for the two figures, or ``None`` to print the table only.

    Returns:
        dict: Per model, the ``scores``/``se`` of ``Evaluator.score`` and the data's own
        ``uniform``/``uniform_se`` reference.
    """
    dim, rows, block_size = 12, 32768, 2048
    loadings = torch.randn(dim, 3, dtype=torch.float64) / math.sqrt(3)
    covariance = 0.6 * loadings @ loadings.mT + 0.4 * torch.eye(dim, dtype=torch.float64)
    scale = covariance.diagonal().sqrt()
    latent_corr = covariance / scale[:, None] / scale[None, :]
    zero = torch.zeros(dim, dtype=torch.float64)
    x_fit, x_val, x_ref = (sign_of_gaussian(zero, latent_corr, rows) for _ in range(3))
    evaluator = Evaluator(x_fit, x_val, x_ref, block_size=block_size)

    draws = {}
    for name, make in BASELINES.items():
        model = make(dim)
        model.fit(x_fit)
        draws[name] = model.sample(rows).to(torch.float64)
    # The foil: the spin mean and covariance passed off as latent parameters (rho = r)
    draws["spin-space"] = sign_of_gaussian(x_fit.mean(0), torch.cov(x_fit.mT), rows)

    data, data_se = evaluator.uniform_score("test")
    print(f"synthetic dichotomized Gaussian, dim={dim}, rows={rows} (zero means): captured")
    print("fraction of the data's distance from uniform at gaussian / order 2 / order 4")
    print("(1 = all of the data's structure, 0 = none)")
    results = {}
    for name, draw in draws.items():
        scores, se = evaluator.score(draw, "test")
        results[name] = {"scores": scores, "se": se, "uniform": data, "uniform_se": data_se}
        captured = [1 - float(scores[k] / data[k]) for k in (0, 2, 4)]
        print(f"  {name:12s} {captured[0]:6.3f} / {captured[1]:6.3f} / {captured[2]:6.3f}")
    if outdir is not None:
        render_figures(results, outdir)
    return results


def render_figures(results: dict, outdir: str) -> None:
    """Draw the two figures of ``docs/baselines.md``: the link function and the captured fractions.

    Args:
        results (dict): Per model, the ``scores``/``se`` and the data's ``uniform``/``uniform_se``
            reference, as :func:`run` assembles them; the keys ``"dichotomized"`` and
            ``"spin-space"`` are drawn.
        outdir (str): Directory to write ``link_function.png`` and ``captured_by_order.png`` into.
    """
    import os

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    ink, muted, grid = "#222222", "#666666", "#DDDDDD"
    orange, violet, blue = "#E69F00", "#4a3aa7", "#0072B2"

    def style(ax) -> None:
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(True, color=grid, linewidth=0.6)
        ax.set_axisbelow(True)

    # Figure 1: the link function and the two matches
    rho = torch.linspace(-1, 1, 401)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    style(axes[0])
    axes[0].plot(rho, rho, color=muted, linestyle="--", linewidth=1.2, label="identity")
    axes[0].plot(rho, 2 / math.pi * torch.asin(rho), color=blue, linewidth=2,
                 label="spin correlation r = (2/π) arcsin ρ")
    axes[0].set_xlabel("latent correlation ρ", color=ink)
    axes[0].set_ylabel("spin correlation r", color=ink)
    axes[0].legend(frameon=False, fontsize=9)
    style(axes[1])
    r = torch.linspace(-1, 1, 401)
    axes[1].plot(r, r, color=violet, linewidth=2, label="tetrachoric match: ρ = sin(πr/2)")
    axes[1].plot(r, 2 / math.pi * torch.asin(r), color=orange, linewidth=2, linestyle="-.",
                 label="spin-space match: ρ = r")
    axes[1].set_xlabel("target spin correlation r", color=ink)
    axes[1].set_ylabel("realized spin correlation", color=ink)
    axes[1].legend(frameon=False, fontsize=9)
    fig.suptitle("The dichotomized-Gaussian link and what each moment match realizes", color=ink)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "link_function.png"), dpi=140)
    plt.close(fig)

    # Figure 2: captured fraction by order on the synthetic sample
    ref = results["dichotomized"]  # every result shares the data reference
    orders = [k for k in range(1, 9) if float(ref["uniform"][k]) >= 3 * float(ref["uniform_se"][k])]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    style(ax)
    series = (("spin-space", orange, "D", "-."), ("dichotomized", violet, "^", "-"))
    labels = {
        "spin-space": "spin-space match: sign(mean + L ε) with the spin mean and covariance, ρ = r",
        "dichotomized": "tetrachoric match: DichotomizedGaussian.fit",
    }
    for name, color, marker, linestyle in series:
        res = results[name]
        captured = [1 - float(res["scores"][k] / res["uniform"][k]) for k in orders]
        err = [2 * abs(float(res["se"][k] / res["uniform"][k])) for k in orders]
        ax.errorbar(orders, captured, yerr=err, color=color, marker=marker, linestyle=linestyle,
                    linewidth=2, markersize=6, capsize=3, label=labels[name])
    even = [k for k in orders if k % 2 == 0]
    ax.plot(even, [1 - (1 - (2 / math.pi) ** (k / 2)) ** 2 for k in even], color=muted,
            marker="x", linestyle=":", linewidth=1.2,
            label="heuristic 1 − (1 − (2/π)^(k/2))² for the spin-space match")
    ax.set_xticks(orders)
    ax.axhline(1.0, color=grid, linewidth=0.8)
    ax.set_xlabel("interaction order k", color=ink)
    ax.set_ylabel("captured fraction of the data's order-k weight", color=ink)
    ax.set_ylim(-0.1, 1.15)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.set_title("Synthetic dichotomized Gaussian, 12 spins, zero means: the resolved orders",
                 color=ink, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "captured_by_order.png"), dpi=140)
    plt.close(fig)
    print(f"wrote {outdir}/link_function.png and {outdir}/captured_by_order.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--outdir", default=None,
                        help="directory for the two figures of docs/baselines.md (default: none)")
    args = parser.parse_args()
    torch.manual_seed(0)
    run(args.outdir)
