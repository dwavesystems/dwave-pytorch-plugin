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
"""Matplotlib figures of :class:`~dwave.plugins.torch.evaluation.Evaluator` scores and kernel
diagnostics.

- :func:`plot_fourier_weights` draws the Fourier weight against the interaction order ``k`` for
  one or several models scored by :meth:`~dwave.plugins.torch.evaluation.Evaluator.score`
  against one split: each model's misfit (its ``scores[k]``, the mean squared order-``k``
  Walsh-coefficient error) with +-2-standard-error bars, and the split's own weight against the
  analytic uniform distribution (:meth:`~dwave.plugins.torch.evaluation.Evaluator.uniform_score`)
  as the reference it is read against, on a symmetric-log axis whose linear band around zero is
  the floor of the view: a negative unbiased estimate is drawn where it falls inside the band
  and clamped onto the floor below it.
- :func:`plot_kernel_spectra` draws each fixed kernel's loss geometry: its Walsh eigenvalue per
  interaction order, normalized over orders ``>= 1``, i.e. the weight one unit of order-``j``
  mismatch energy receives (not the order's share of spectral mass), from the exact closed-form
  spectra at any dimension, one panel per dimension, on a symmetric-log axis whose linear band
  around zero clamps numerically-zero weights.

Design notes: identity is carried by axis labels and text. Series wear fixed Okabe-Ito colors
doubled by line style: the data-vs-uniform reference is green, the models take orange, blue,
vermillion, pink in listing order (at most four per figure); on the spectra plot the Gaussian
kernel wears a fixed pink dash-dot line and the order ladder a copper dotted line on the unit
level. Both y axes are symmetric-log (``symlog``): logarithmic above a linear threshold, linear
inside the shaded band around zero, which is the floor of the view, so numerically-zero and
negative values are clamped there instead of falling off a log axis or stretching it. Tensors
(any device) and NumPy arrays are accepted alike. Figures are built on
:class:`matplotlib.figure.Figure` directly (no pyplot import, no global state, headless-safe):
write with ``fig.savefig(path, dpi=150)``, or ``IPython.display.display(fig)`` in a notebook.
``examples/evaluation.py`` renders the figures of ``docs/evaluation.md``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

__all__ = ["plot_fourier_weights", "plot_kernel_spectra"]

_INK = "#1a1a1a"  # text: primary ink (text never wears a series color)
_MUTED = "#6a6a6a"  # text/axes: secondary ink
_GRID = "#dcdcdc"
# Fixed series anchors: Okabe-Ito, colorblind-safe
_REF = "#009E73"  # the data against the analytic uniform distribution (green reference)
# Model series in listing order: (color, linestyle), identity doubled by line style.
_MODEL_STYLES = (
    ("#E69F00", "-"),  # orange
    ("#0072B2", "--"),  # blue
    ("#D55E00", "-."),  # vermillion
    ("#CC79A7", ":"),  # pink
)
# Kernel identity on the spectra plot: the Gaussian kernel a fixed pink dash-dot line, the order
# ladder a copper dotted line on the unit level.
_GAUSSIAN_STYLE = ("#CC79A7", "-.")
_PROJ_STYLE = ("#A0522D", (0, (2, 1)))
# Linear threshold of the spectra plot's symlog axis: below the relative precision of a
# double, a normalized eigenvalue is numerically zero, and the linear band clamps it there.
_SPECTRA_LINTHRESH = 1e-16


def _np(values: Any) -> np.ndarray:
    """A NumPy array of a tensor (copied to the host, its dtype kept) or array-like, for matplotlib.

    The one device transfer in the package: matplotlib draws from host memory, so a CUDA result
    is copied over here rather than by every caller; nothing else is cast or moved.
    """
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _despine(ax: Any) -> None:
    """Recessive axes: drop top/right spines, mute the rest, small ink tick labels."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=_MUTED, labelcolor=_INK, labelsize=8, width=0.8)


def _symlog(ax: Any, linthresh: float) -> None:
    """Symmetric-log y axis: log beyond +-``linthresh``, linear and shaded inside, zero line."""
    # linscale=0.5: the whole linear band [-linthresh, linthresh] is as tall as one decade
    ax.set_yscale("symlog", linthresh=linthresh, linscale=0.5)
    ax.axhspan(-linthresh, linthresh, color=_GRID, alpha=0.4, zorder=0)
    ax.axhline(0.0, color=_MUTED, lw=0.7, ls=":", zorder=1)


def _draw_series(
    ax: Any,
    orders: np.ndarray,
    estimate: np.ndarray,
    se: np.ndarray,
    color: str,
    linestyle: Any,
    label: str,
    floor: float,
) -> Any:
    """Draw one Fourier-weight series as a line with +-2 se bars; return its legend handle.

    Every estimate is drawn where it falls, negative ones included (the symmetric-log axis of
    :func:`plot_fourier_weights` places them inside its linear band), except below ``floor``
    (the band's lower edge, the bottom of the view), where it is clamped onto the floor and
    its lower bar cut there; the upper bar keeps its true end. The line is unbroken.
    """
    drawn = np.maximum(estimate, floor)
    lower = drawn - np.maximum(estimate - 2.0 * se, floor)
    upper = np.maximum(estimate + 2.0 * se, floor) - drawn
    (handle,) = ax.plot(orders, drawn, color=color, ls=linestyle, lw=1.6, label=label, zorder=3)
    ax.errorbar(
        orders,
        drawn,
        yerr=[lower, upper],
        fmt="none",
        ecolor=color,
        elinewidth=1.0,
        capsize=2.5,
        alpha=0.7,
        zorder=2,
    )
    return handle


def plot_fourier_weights(
    evaluator: Any,
    results: Mapping[str, tuple[Any, Any]],
    split: str = "test",
    title: str | None = None,
    max_order: int | None = None,
    linthresh: float | None = None,
) -> Figure:
    """Fourier weight by interaction order for models scored against one split of an evaluator.

    One axis: orders ``k = 1..dim`` on the x axis (``max_order`` truncates; profile 0, the
    headline kernel, is not an order and is not drawn) and the order-``k`` Fourier weight on a
    symmetric-log y axis, the mean squared order-``k`` Walsh coefficient, entry ``[k]`` of the
    result tensors (the total weight of O'Donnell's Definition 1.19 is this times
    ``C(dim, k)``):

    - **data vs uniform** (green, solid): ``evaluator.uniform_score(split)``, the split's own
      Fourier weight at each order, the reference. It is what a model that captured *nothing*
      at order ``k`` would score, and it sets the scale a misfit there is read against.
    - one series per model (orange solid, blue dashed, vermillion dash-dot, pink dotted, in
      listing order; at most four): ``scores[k]``, the model's misfit at each order; at the
      reference it has captured none of that order's structure, at noise level all of it that
      this sample size can resolve. An empty ``results`` draws the reference alone.

    Bars are +-2 standard errors over blocks. The y axis is symmetric-log (``symlog``):
    logarithmic above ``linthresh`` and linear inside the shaded band
    ``[-linthresh, linthresh]``, which is also the bottom of the view. An unbiased estimate
    that came out negative is drawn where it falls inside the band; one below the band is
    clamped onto its floor, its lower bar cut there and its upper bar kept (the footnote counts
    them), so noise excursions at poorly resolved orders never stretch the axis, the same
    clamp the spectra plot applies to numerically-zero weights.

    Args:
        evaluator (Evaluator): The :class:`~dwave.plugins.torch.evaluation.Evaluator` whose
            ``score(xquery, split)`` produced ``results``; the figure reads its ``dim``,
            ``uniform_score(split)``, ``split(split)``, and ``num_blocks(split)``. One evaluator
            and one split is what makes the series comparable.
        results (Mapping[str, tuple]): A mapping from a model label to that model's
            ``(scores, se)`` pair from ``Evaluator.score``, each a tensor of ``dim + 1`` entries
            in profile order.
        split (str): The split the results were scored against, whose reference the figure
            draws. Defaults to ``"test"``.
        title (str, optional): Replaces the default headline. Defaults to ``None``.
        max_order (int, optional): The last order drawn. Defaults to ``None``, every order up to
            ``dim``.
        linthresh (float, optional): The linear threshold of the symmetric-log axis. Defaults to
            ``None``: the smallest +-2-se half-width on display, the sharpest noise floor in the
            figure, so the log scale covers everything the sample resolves and the band holds
            what it cannot.

    Raises:
        ValueError: If ``results`` holds more than four models or a pair whose tensors do not
            have ``dim + 1`` entries each, no order lies within ``max_order``, or ``linthresh``
            is not positive.

    Returns:
        matplotlib.figure.Figure: A figure with one axis.
    """
    dim = int(evaluator.dim)
    reference, reference_se = (_np(values) for values in evaluator.uniform_score(split))
    if len(results) > len(_MODEL_STYLES):
        raise ValueError(f"at most {len(_MODEL_STYLES)} models per figure, got {len(results)}")
    models: list[tuple[str, np.ndarray, np.ndarray]] = []
    for label, pair in results.items():
        scores, se = (_np(values) for values in pair)
        if scores.shape != (dim + 1,) or se.shape != (dim + 1,):
            raise ValueError(
                f"result {label!r} must be the (scores, se) pair of Evaluator.score: two tensors "
                f"of dim + 1 = {dim + 1} entries, got shapes {scores.shape} and {se.shape}"
            )
        models.append((label, scores, se))
    max_shown = dim if max_order is None else min(dim, int(max_order))
    if max_shown < 1:
        raise ValueError("max_order keeps no order: it must be >= 1")
    orders = np.arange(1, max_shown + 1, dtype=np.float64)
    order_slice = slice(1, max_shown + 1)
    series = [("data vs uniform", reference[order_slice], reference_se[order_slice], _REF, "-")]
    for (label, scores, se), (color, linestyle) in zip(models, _MODEL_STYLES, strict=False):
        series.append((f"{label} vs data", scores[order_slice], se[order_slice], color, linestyle))
    estimates = np.concatenate([estimate for _, estimate, _, _, _ in series])
    half_widths = 2.0 * np.concatenate([se for _, _, se, _, _ in series])
    if linthresh is None:
        # the sharpest noise floor on display; a degenerate figure with no spread falls back
        # to its smallest nonzero magnitude, then to 1
        resolved = half_widths[half_widths > 0]
        candidates = resolved if resolved.size else np.abs(estimates[estimates != 0])
        linthresh = float(candidates.min()) if candidates.size else 1.0
        threshold_note = "the smallest ±2-se half-width shown"
    elif not linthresh > 0:  # also rejects NaN
        raise ValueError(f"linthresh must be > 0, got {linthresh!r}")
    else:
        threshold_note = "set by linthresh"
    top = float((estimates + half_widths).max())
    clamped = int(np.count_nonzero(estimates < -linthresh))
    rows, blocks = int(evaluator.split(split).shape[0]), int(evaluator.num_blocks(split))

    fig = Figure(figsize=(7.6, 4.6), constrained_layout=True)
    ax = fig.subplots()
    handles = [
        _draw_series(ax, orders, estimate, se, color, linestyle, label, -linthresh)
        for label, estimate, se, color, linestyle in series
    ]
    _symlog(ax, linthresh)
    # the linear band is the floor of the view; room above the highest bar
    ax.set_ylim(-linthresh, 3.0 * max(top, linthresh))
    ax.set_xlim(0.5, max_shown + 0.5)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=min(12, max_shown)))
    ax.set_xlabel("interaction order $k$", fontsize=9, color=_INK)
    ax.set_ylabel(
        "Fourier weight at order $k$\n"
        "(mean squared Walsh coefficient; MMD$^2$ of the order-$k$ kernel)",
        fontsize=9,
        color=_INK,
    )
    ax.grid(axis="y", color=_GRID, lw=0.5)
    ax.set_axisbelow(True)
    _despine(ax)
    ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle(
        "Fourier weight by interaction order" if title is None else title,
        fontsize=11.5,
        color=_INK,
    )
    fig.supxlabel(
        f"dim={dim}; the {split} split, {rows} rows per sample, B={blocks} blocks.  "
        "Bars: ±2 standard errors over blocks.\n"
        f"Symmetric-log y axis: linear within ±{linthresh:.1e} ({threshold_note}; shaded band), "
        "logarithmic above.\n"
        f"{clamped} estimate(s) below the band are clamped onto its floor, with the lower bar "
        "cut there.\n"
        "Green: the data against the analytic uniform distribution, the weight a model that "
        "captures nothing would score.\nOther series: a model against the data; at noise level "
        "it has captured all the order's structure this sample size resolves.",
        fontsize=7.5,
        color=_MUTED,
    )
    return fig


def plot_kernel_spectra(
    dims: tuple[int, ...] = (48,),
    max_order: int = 32,
    title: str | None = None,
    linthresh: float = _SPECTRA_LINTHRESH,
) -> Figure:
    """Each fixed kernel's loss geometry: its normalized Walsh eigenvalue per interaction order.

    One panel per dimension in ``dims``: ``lambda_j / sum_{i >= 1} lambda_i`` on a symmetric-log
    scale, where ``lambda_j`` is the kernel's Walsh eigenvalue on order-``j`` interactions
    (order 0 is a constant offset that cancels in ``MMD^2`` and is dropped). Since
    ``MMD^2 = sum_j lambda_j E_j``, ``lambda_j`` is the exchange rate the score applies to one
    unit of order-``j`` mismatch energy ``E_j``, and the normalization puts every kernel on one
    scale. It is NOT the order's share of the kernel's spectral mass (its trace), which weighs
    each eigenvalue by the ``C(dim, j)`` characters of its band: for the Gaussian kernel at
    ``dim = 48`` the per-unit-energy weight puts 93% at order 1, while the spectral-mass share
    puts 13% there and peaks at order 3. Both fixed kinds have exact closed-form spectra (the
    order-``k`` entry a single band at order ``k``, normalized eigenvalue exactly 1 at its own
    order, so the ladder is the unit line; the Gaussian kernel exactly geometric with ratio
    ``tanh(1 / (2 sqrt(dim)))``), so no table is built and any dimension is plottable. The y
    axis is symmetric-log (``symlog``): logarithmic above ``linthresh`` and linear inside the
    shaded band around zero, so the Gaussian kernel's geometric decay is drawn at every order
    and clamps visibly onto zero once it drops below the threshold, the orders it is blind to in
    practice, instead of running off the bottom of a log axis. An entry with no point at order
    ``j`` puts zero weight there: an order-``k`` kernel literally cannot see a mismatch off its
    band. (An evaluator's alpha row, while it holds kernel weights, has eigenvalue
    ``alpha_k / C(dim, k)`` at order ``k``; it is not drawn here.)

    Args:
        dims (tuple[int, ...]): The dimensions, one panel each, integers ``>= 1``. Defaults to
            ``(48,)``.
        max_order (int): The last order drawn in every panel. Defaults to 32.
        title (str, optional): Replaces the default headline. Defaults to ``None``.
        linthresh (float): The linear threshold of the symmetric-log axis. Defaults to
            ``1e-16``, below the relative precision of a double, where a normalized eigenvalue
            is numerically zero.

    Raises:
        ValueError: If a dimension or ``max_order`` is not an integer ``>= 1``, or ``linthresh``
            is not positive.

    Returns:
        matplotlib.figure.Figure: A figure with one panel per dimension, sharing the y axis.
    """
    if not all(isinstance(dim, int | np.integer) and dim >= 1 for dim in dims):
        raise ValueError(f"dims must be integers >= 1, got {dims!r}")
    if not isinstance(max_order, int | np.integer) or max_order < 1:
        raise ValueError(f"max_order must be an integer >= 1, got {max_order!r}")
    if not linthresh > 0:  # also rejects NaN
        raise ValueError(f"linthresh must be > 0, got {linthresh!r}")
    fig = Figure(figsize=(5.2 * len(dims) + 0.4, 4.4), constrained_layout=True)
    axes = np.atleast_1d(fig.subplots(1, len(dims), sharey=True))
    for ax, dim in zip(axes, dims, strict=True):
        handles = []
        label_at: list[tuple[str, float, float]] = []
        # the Gaussian kernel: geometric, normalized: rho^j / sum_{i=1..dim} rho^i, closed form
        # (Kondor & Lafferty, 2002; pinned against direct enumeration in the unit tests), drawn
        # at every order; far out it underflows to exactly 0, which the symlog axis places
        rho = math.tanh(1.0 / (2.0 * math.sqrt(dim)))
        orders = np.arange(1, min(dim, max_order) + 1)
        total = rho * (1.0 - rho**dim) / (1.0 - rho)  # sum of rho^i, i = 1..dim
        weight = rho ** orders.astype(np.float64) / total
        color, linestyle = _GAUSSIAN_STYLE
        (line,) = ax.plot(
            orders, weight, color=color, ls=linestyle, lw=1.6, label="gaussian", zorder=2
        )
        handles.append(line)
        # label the Gaussian kernel at its knee: the last order still above the linear threshold
        knee = max(int(np.count_nonzero(weight > linthresh)) - 1, 0)
        label_at.append(("gaussian", float(orders[knee]), float(weight[knee])))
        # every order-k entry is a single Walsh band at its order: exactly 1 after
        # normalization, so the ladder is the unit line
        ladder_color, ladder_linestyle = _PROJ_STYLE
        (line,) = ax.plot(
            orders,
            np.ones(len(orders)),
            color=ladder_color,
            ls=ladder_linestyle,
            lw=1.6,
            label=f"order k, k = 1..{dim}",
            zorder=3,
        )
        handles.append(line)
        label_at.append(("k = 1", 1.0, 1.0))
        if len(orders) > 1:
            label_at.append((f"k = {orders[-1]}", float(orders[-1]), 1.0))
        # direct labels, stacked when several series end at (numerically) the same point
        occupied: dict[tuple[float, int], int] = {}
        for text, label_x, label_y in label_at:
            # quarter-decade bins; a clamped label sits in the linear band's slot
            slot = (label_x, round(math.log10(max(label_y, linthresh)) * 4))
            row = occupied.get(slot, 0)
            occupied[slot] = row + 1
            ax.annotate(
                text,
                (label_x, label_y),
                xytext=(5, 6 - 10 * row),
                textcoords="offset points",
                fontsize=7.5,
                color=_INK,
                va="center",
            )
        _symlog(ax, linthresh)
        ax.set_ylim(-linthresh, 3.0)  # the whole linear band below, room for labels above 1
        ax.set_xlabel("interaction order j", fontsize=9)
        ax.set_title(f"dim = {dim}  (gaussian sigma = {dim**0.5:.1f})", fontsize=10, color=_INK)
        ax.margins(x=0.14)
        ax.grid(axis="y", color=_GRID, lw=0.5)
        ax.set_axisbelow(True)
        _despine(ax)
        # framed: the legend sits on the linear band, over the clamped Gaussian line
        ax.legend(handles=handles, fontsize=7, loc="lower right", framealpha=0.9, edgecolor="none")
    # the normalized eigenvalue, the weight per unit of order-j mismatch energy, never "share":
    # an order's share of spectral mass carries the band size C(dim, j) as well
    axes[0].set_ylabel(
        "normalized eigenvalue  $\\lambda_j \\,/\\, \\sum_{i \\geq 1} \\lambda_i$\n"
        "(weight per unit of order-$j$ mismatch energy)",
        fontsize=9,
        color=_INK,
    )
    fig.suptitle(
        "kernel Walsh spectra: which interaction orders each score can see"
        if title is None
        else title,
        fontsize=11.5,
        color=_INK,
    )
    fig.supxlabel(
        "MMD = maximum mean discrepancy; MMD^2 = sum_j lambda_j E_j with E_j the order-j "
        "mismatch energy, so lambda_j (exact closed forms) is the weight one unit of\n"
        "order-j energy receives, normalized over orders >= 1 (order 0 cancels in MMD^2), "
        "not the order's share of spectral mass, which also carries the band size C(dim, j).\n"
        "The order ladder is the unit line: each order-k kernel puts unit weight at its own "
        "order and zero (blind) everywhere else; the Gaussian kernel's geometric decay covers "
        "every order.\n"
        f"Symmetric-log y axis: linear within ±{linthresh:.0e} (shaded band), where a normalized "
        "eigenvalue is numerically zero; the Gaussian kernel clamps onto zero past its knee.",
        fontsize=7.5,
        color=_MUTED,
    )
    return fig
