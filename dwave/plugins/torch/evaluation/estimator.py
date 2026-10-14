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
"""Blocked, exactly unbiased kernel-score estimation on hard spins from distance histograms.

Every score here is a squared maximum mean discrepancy (MMD), ``MMD^2_kappa(P, Q)``, for a
Hamming-isotropic kernel ``kappa(x, y) = f(d_H(x, y))`` given by its profile row ``f``, one
value per Hamming distance ``0..dim`` (see :mod:`dwave.plugins.torch.kernels`). The population
value is zero when ``Q`` matches ``P`` on every interaction order the kernel sees; smaller is
better. The reference sample is cut once into ``B`` blocks of ``r`` rows and every query model is
scored against the *same* blocks, so per-block differences between models cancel the reference
noise (the common-random-numbers device). Each block yields the U-statistic estimate
``within_x + within_y - 2 cross``: the within-sample means run over the ``r (r - 1)`` ordered
off-diagonal pairs and the cross mean over all ``r^2`` pairs, which is exactly unbiased for
independent and identically distributed (IID) rows and independent samples, so the ``B`` values
carry a standard error.

Each block pair is reduced to its Hamming-distance histogram from one Gram of ``+-1`` spins, and
every kernel's estimate is a dot product of its profile row with those counts: the whole battery
costs one matrix-vector product per block pair. Nothing is cast or moved: the computation runs
in the samples' floating dtype on their device, exactly while ``dim`` and ``r^2`` fit its
mantissa, and a mismatched tensor fails inside torch.

Public units: :func:`validate_spin_matrix`, :data:`UNIFORM` (the :class:`Uniform` marker for the
analytic uniform distribution, scored without a sample), :class:`BlockedScores`, and
:func:`blocked_scores`. The :class:`~dwave.plugins.torch.evaluation.Evaluator` wraps these with
cached reference histograms and three held-out splits; ``docs/evaluation.md`` carries
definitions, preconditions, and references.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from dwave.plugins.torch.kernels import uniform_kernel_mean

__all__ = ["UNIFORM", "BlockedScores", "Uniform", "blocked_scores", "validate_spin_matrix"]


def validate_spin_matrix(sample: torch.Tensor, name: str) -> None:
    """Check that ``sample`` is a non-empty 2-D matrix of floating spins in ``{-1, +1}``.

    The floating dtype is required rather than coerced: the Gram and the histogram run in the
    sample's own dtype, and an integer Gram wraps silently (an int8 Gram overflows at
    ``dim > 127``). The magnitude test rejects NaN, zeros, and soft values alike.

    Args:
        sample (torch.Tensor): The tensor to check.
        name (str): How to name it in the error message.

    Raises:
        ValueError: If ``sample`` is not 2-D, is empty, is not of a floating dtype, or holds an
            entry other than ``-1`` or ``+1``.
    """
    if sample.ndim != 2 or sample.numel() == 0:
        raise ValueError(
            f"{name} must be a non-empty 2-D (rows, dim) matrix, got shape {tuple(sample.shape)}"
        )
    if not sample.is_floating_point():
        raise ValueError(
            f"{name} must be spins in {{-1,+1}} of a floating dtype (the Gram and the histogram "
            f"run in the sample's dtype, and an integer Gram wraps silently), got {sample.dtype}"
        )
    if not bool((sample.abs() == 1).all()):
        raise ValueError(f"{name} must be spins in {{-1,+1}}")


def _validate_models(
    xqueries: Mapping[str, torch.Tensor | Uniform], reference: torch.Tensor, reference_name: str
) -> dict[str, torch.Tensor | None]:
    """Validate the query models once; ``None`` marks the analytic uniform model.

    Every sampled query must pass :func:`validate_spin_matrix` and have exactly the reference's
    shape, so block ``b`` of every sample holds the same row range; the :class:`Uniform` marker
    needs no sample. Raises :class:`ValueError` on an empty mapping or a violation.
    """
    if not xqueries:
        raise ValueError("xqueries must name at least one model")
    samples: dict[str, torch.Tensor | None] = {}
    for name, y in xqueries.items():
        if isinstance(y, Uniform):
            samples[name] = None
            continue
        validate_spin_matrix(y, name)  # validate once, not per block
        if y.shape != reference.shape:
            raise ValueError(
                f"model {name!r} must have the same shape as {reference_name}: got "
                f"{tuple(y.shape)} vs {tuple(reference.shape)}"
            )
        samples[name] = y
    return samples


def _check_exact_counting(dtype: torch.dtype, dim: int, block_size: int) -> None:
    """Refuse a ``(dtype, dim, block_size)`` at which the Grams or the histograms would round.

    A Gram entry is a sum of ``dim`` terms ``+-1`` and a histogram bin a count of up to
    ``block_size^2`` pairs; both are exact only while they stay within the largest integer the
    dtype represents contiguously, ``2 / eps`` (``2^24`` in float32, ``2^53`` in float64; ``eps``
    is a power of two, so the quotient is exact). Past it ``torch.histc`` rounds its increments
    away silently (a 4097-row float32 block already miscounts by one), so the default
    ``block_size = 4096`` is the float32 ceiling, and float64 never binds in practice.

    Raises:
        ValueError: If ``dim`` or ``block_size^2`` exceeds the limit.
    """
    limit = int(2.0 / torch.finfo(dtype).eps)  # eps is a power of two, so this is exact
    if max(dim, block_size * block_size) > limit:
        raise ValueError(
            f"{dtype} counts exactly only up to {limit}, but dim={dim} and block_size^2="
            f"{block_size * block_size} must both stay within it: lower block_size (<= "
            f"{int(limit**0.5)} here) or use float64 spins"
        )


@torch.no_grad()
def _distance_counts(left: torch.Tensor, right: torch.Tensor, dim: int) -> torch.Tensor:
    """Histogram over ``d = 0..dim`` of the Hamming distances between the rows of two blocks.

    One Gram ``left @ right.T`` gives every pairwise agreement count exactly (sums of ``+-1`` in
    the blocks' dtype), so ``d = (dim - agreement) / 2`` is an exact integer, formed in place;
    ``torch.histc`` then counts the distances into ``dim + 1`` unit bins centered on the
    integers. Returns the counts as a ``(dim + 1,)`` tensor in the blocks' dtype on their
    device. ``left @ left.T`` counts every row against itself at distance 0; the caller drops
    that diagonal.
    """
    agreement = left @ right.T
    distances = agreement.sub_(dim).mul_(-0.5)  # (dim - agreement) / 2
    return torch.histc(distances.reshape(-1), bins=dim + 1, min=-0.5, max=dim + 0.5)


# --------------------------------------------------------------------------------------
# Blocked paired scores
# --------------------------------------------------------------------------------------


class Uniform:
    """Marker for the analytic uniform distribution on ``{-1, +1}^dim``; see :data:`UNIFORM`.

    Pass :data:`UNIFORM` as a query value (``{"uniform": UNIFORM}``) to score the uniform
    distribution without sampling it: the kernel's mean against any point is the closed-form
    :func:`~dwave.plugins.torch.kernels.uniform_kernel_mean` of the profile, so the per-block
    score is the within-block reference mean minus that constant, with no cross term and no
    cross-term noise. At the order-``k`` row this is the reference's Fourier weight at order
    ``k``, the most a model can miss there.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNIFORM"


#: The analytic uniform distribution as a query model (:class:`Uniform`'s one instance).
UNIFORM = Uniform()


@dataclass(frozen=True)
class _Histograms:
    """Per-block Hamming-distance histograms: the sufficient statistics of every kernel score.

    ``xx[b, d]`` counts the ordered off-diagonal row pairs of reference block ``b`` at distance
    ``d``; for a sampled query, ``yy[name][b, d]`` counts the same within its own block ``b`` and
    ``xy[name][b, d]`` the pairs across the two blocks. The analytic uniform model has no
    histograms, so ``model_names`` (every query, in order) records which names are which. Each
    table is ``(B, dim + 1)`` in the samples' dtype on their device; ``block_size`` is ``r``.
    """

    model_names: tuple[str, ...]
    xx: torch.Tensor
    yy: dict[str, torch.Tensor]
    xy: dict[str, torch.Tensor]
    block_size: int


def _reference_histograms(x: torch.Tensor, num_blocks: int) -> torch.Tensor:
    """Within-block distance histograms of the reference, with the diagonal removed.

    Returns a ``(num_blocks, dim + 1)`` table whose row ``b`` counts the ``r (r - 1)`` ordered
    off-diagonal pairs of block ``b``, the pairs of the unbiased within-sample kernel mean.
    """
    sample_size, dim = x.shape
    block_size = sample_size // num_blocks
    xx = torch.stack([_distance_counts(block, block, dim) for block in x.split(block_size)])
    xx[:, 0] -= block_size  # drop the diagonal (distance 0): unbiased within-sample means
    return xx


def _block_histograms(
    x: torch.Tensor,
    samples: Mapping[str, torch.Tensor | None],
    num_blocks: int,
    xx: torch.Tensor | None = None,
) -> _Histograms:
    """Reduce every block pair to its Hamming-distance histogram, one exact Gram each.

    Block ``b`` holds rows ``[b r, (b + 1) r)`` of the reference ``x`` and of every sampled query
    (``None`` marks the analytic uniform model, which has no histograms). ``xx``, the
    reference's own histograms, is computed unless a cached table is passed; within-sample
    counts come back with the diagonal removed.
    """
    sample_size, dim = x.shape
    block_size = sample_size // num_blocks
    if xx is None:
        xx = _reference_histograms(x, num_blocks)
    x_blocks = x.split(block_size)
    yy: dict[str, torch.Tensor] = {}
    xy: dict[str, torch.Tensor] = {}
    for name, sample in samples.items():
        if sample is None:  # the analytic uniform model has no histograms
            continue
        y_blocks = sample.split(block_size)
        yy[name] = torch.stack([_distance_counts(y, y, dim) for y in y_blocks])
        yy[name][:, 0] -= block_size  # drop the diagonal, as for the reference
        xy[name] = torch.stack(
            [_distance_counts(xb, y, dim) for xb, y in zip(x_blocks, y_blocks, strict=True)]
        )
    return _Histograms(tuple(samples), xx, yy, xy, block_size)


def _scores_from_histograms(profile_table: torch.Tensor, histograms: _Histograms) -> torch.Tensor:
    """Per-block unbiased ``MMD^2`` estimates, shape ``(K, M, B)``, of every profile row.

    For a row ``f`` the within-sample term is ``f @ counts / (r (r - 1))`` over a block's ordered
    off-diagonal pairs and the cross term ``f @ counts / r^2`` over the ``r^2`` pairs of two
    blocks; the analytic uniform model has no within term of its own and takes the closed-form
    :func:`~dwave.plugins.torch.kernels.uniform_kernel_mean` of ``f`` as its cross term. One
    ``(K, dim + 1) x (dim + 1, B)`` product per histogram, whatever the kernel count.
    """
    n_within = float(histograms.block_size * (histograms.block_size - 1))
    n_between = float(histograms.block_size * histograms.block_size)
    within_x = profile_table @ histograms.xx.T / n_within  # (K, B)
    uniform_mean = uniform_kernel_mean(profile_table)  # (K,): the analytic arm's cross term
    per_model = []
    for name in histograms.model_names:
        if name not in histograms.yy:  # the analytic uniform model
            per_model.append(within_x - uniform_mean[:, None])
            continue
        per_model.append(
            within_x
            + profile_table @ histograms.yy[name].T / n_within
            - 2.0 * (profile_table @ histograms.xy[name].T) / n_between
        )
    return torch.stack(per_model, dim=1)


@dataclass(frozen=True)
class BlockedScores:
    """Per-block unbiased score estimates of ``M`` query models against shared reference blocks.

    Attributes:
        model_names (tuple[str, ...]): The query names, in the insertion order of the mapping
            they came from.
        scores (torch.Tensor): A ``(K, M, B)`` tensor in the samples' dtype: entry ``[j, i, b]``
            is the unbiased ``MMD^2`` estimate of model ``model_names[i]`` against reference
            block ``b`` under kernel row ``j`` of the profile table scored. Smaller is better;
            the mean over ``b`` is the estimate and the standard deviation over ``b`` divided by
            ``sqrt(B)`` its standard error.
        num_blocks (int): ``B``, the number of blocks.
        block_size (int): ``r``, the rows per block.
    """

    model_names: tuple[str, ...]
    scores: torch.Tensor
    num_blocks: int
    block_size: int


def blocked_scores(
    x: torch.Tensor,
    xqueries: Mapping[str, torch.Tensor | Uniform],
    num_blocks: int,
    profiles: torch.Tensor,
) -> BlockedScores:
    """Score every query model against the same ``num_blocks`` blocks of the reference ``x``.

    Rows are consumed in storage order: block ``b`` holds rows ``[b r, (b + 1) r)`` of every
    sample. A sample whose storage order is structured (sorted by label, grouped by writer, the
    reads of one hardware submission) must be permuted once by the caller, and that order kept:
    consumed as stored, its blocks are different generators and every score inflates.
    Unbiasedness and valid standard errors need IID rows within each sample and independence
    between samples (see the module docstring). Nothing is cast: ``profiles`` must share the
    samples' dtype and device, or the matmul fails inside torch.

    Args:
        x (torch.Tensor): The reference sample, a ``(rows, dim)`` floating spin matrix in
            ``{-1, +1}``.
        xqueries (Mapping[str, torch.Tensor | Uniform]): Query models by name: a spin matrix of
            exactly ``x``'s shape, or :data:`UNIFORM` for the analytic uniform distribution.
        num_blocks (int): ``B``, which must divide ``rows`` and leave at least two rows per
            block.
        profiles (torch.Tensor): A ``(K, dim + 1)`` table, one kernel profile per row:
            :func:`~dwave.plugins.torch.kernels.kernel_profiles` of ``dim`` or any custom stack.

    Raises:
        ValueError: If a sample fails :func:`validate_spin_matrix`, a query's shape differs from
            ``x``'s, ``xqueries`` is empty, ``profiles`` is not a non-empty ``(K, dim + 1)``
            table, ``num_blocks`` is below 1, does not divide the rows, or leaves fewer than two
            rows per block, or the dtype cannot count ``dim`` or ``block_size^2`` exactly.

    Returns:
        BlockedScores: A record whose ``scores`` has shape ``(K, M, B)``, in the row order of
        ``profiles`` and the insertion order of ``xqueries``.
    """
    validate_spin_matrix(x, "x")
    sample_size, dim = x.shape
    if profiles.ndim != 2 or profiles.shape[0] == 0 or profiles.shape[1] != dim + 1:
        raise ValueError(
            f"profiles must be a (K >= 1, dim + 1 = {dim + 1}) table, one kernel per row, got "
            f"shape {tuple(profiles.shape)}"
        )
    samples = _validate_models(xqueries, x, "x")
    if num_blocks < 1:
        raise ValueError(f"num_blocks must be >= 1, got {num_blocks!r}")
    if sample_size % num_blocks:
        raise ValueError(f"sample_size={sample_size} not divisible by num_blocks={num_blocks}")
    num_blocks = int(num_blocks)
    block_size = sample_size // num_blocks
    if block_size < 2:
        raise ValueError("need at least 2 rows per block for the U-statistic")
    _check_exact_counting(x.dtype, dim, block_size)
    histograms = _block_histograms(x, samples, num_blocks)
    scores = _scores_from_histograms(profiles, histograms)
    return BlockedScores(histograms.model_names, scores, num_blocks, block_size)
