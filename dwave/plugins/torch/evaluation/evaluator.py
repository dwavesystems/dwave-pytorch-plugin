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
"""The kernel-score battery as an evaluation score and as a training loss.

One :class:`Evaluator` holds three data splits of floating spins in ``{-1, +1}``, ``train``,
``val``, and ``test``, blocked once at a common block size, and serves both uses of the battery
of :mod:`dwave.plugins.torch.kernels`:

- **evaluation**: ``score(xquery, split)`` and ``blocked(xqueries, split)`` are the exactly
  unbiased blocked estimates of :mod:`dwave.plugins.torch.evaluation.estimator` for a full-size
  sample against a split, with standard errors over blocks, read against
  ``uniform_score(split)``, the split's own distance from the analytic uniform distribution;
- **training**: ``compute_loss(xquery, block_size)`` is the differentiable per-profile squared
  maximum mean discrepancy (MMD) of
  :func:`~dwave.plugins.torch.nn.functional.maximum_mean_discrepancy_loss` for a batch of
  generated spins (hard, or soft in ``[-1, 1]``) against a batch of a split.

Every result is indexed by *profile*: ``[0]`` the headline kernel (the Gaussian kernel, or the
alpha row, a nonnegative combination of the order kernels with weights ``alpha``, while kernel
weights are held) and ``[k]`` the order-``k`` kernel for ``k = 1..dim``. ``train`` and ``val``
are the calibration splits, the only data ``calibrate`` reads to fit ``alpha`` by the two-level
rule: equal weights on the orders a sample of the test split's size can measure (resolved on the
calibration data, and at least ``ratio`` times the per-order floor of real data scored against
real data), a small floor on every other order, returned as a :class:`Calibration` record; the
test split enters only through its block count, never through its rows. Nothing is cast or
moved: the computation runs in the splits' dtype on their device, and a mismatched tensor fails
inside torch. ``docs/evaluation.md`` carries definitions, preconditions, figures, and
references.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from dwave.plugins.torch.evaluation.estimator import (
    BlockedScores,
    Uniform,
    _block_histograms,
    _check_exact_counting,
    _Histograms,
    _reference_histograms,
    _scores_from_histograms,
    _validate_models,
    validate_spin_matrix,
)
from dwave.plugins.torch.kernels import _validate_alpha, kernel_profiles
from dwave.plugins.torch.nn.functional import maximum_mean_discrepancy_loss

__all__ = ["Calibration", "Evaluator"]


def _mean_se(per_block: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and standard error of the mean over axis 1 of ``(K, B)`` per-block estimates."""
    return per_block.mean(dim=1), per_block.std(dim=1) / math.sqrt(per_block.shape[1])


@dataclass(frozen=True)
class Calibration:
    """What :meth:`Evaluator.calibrate` fitted: the two-level kernel weights and what set them.

    Every tensor is in the splits' dtype on their device; ``alpha``, ``fourier_weight``, ``se``,
    ``resolved``, ``noise`` and ``shares`` hold one entry per order ``1..dim``.

    Attributes:
        alpha (torch.Tensor): The kernel weights the evaluator now holds; they sum to 1, so the
            calibrated alpha row has the unit diagonal of every fixed kernel. Two levels:
            ``(1 - floor) / m`` on each of the ``m`` measurable orders and ``floor / (dim - m)``
            on each other order; flat when no order, or every order, is measurable.
        fourier_weight (torch.Tensor): The calibration data's per-coefficient Fourier weight
            per order, its ``uniform_score`` pooled over the ``train`` and ``val`` blocks.
        se (torch.Tensor): The standard error of ``fourier_weight`` over the pooled blocks.
        resolved (torch.Tensor): The *measurable* orders, the gate that sets the weights:
            ``fourier_weight - 3 se > 0`` and ``fourier_weight >= ratio * noise`` with
            ``noise > 0``.
        noise (torch.Tensor): The floor per order: the standard error, over ``test_blocks``
            blocks, of real data scored against real data (the first ``val``-sized rows of
            ``train`` against the ``val`` blocks), what a sample that reproduces the law scores
            at that order.
        ratio (float): How many times the floor the data's weight had to reach, as passed.
        test_blocks (int): The block count the floor was scaled to.
        floor (float): ``eps``, the mass the unmeasurable orders share, as passed.
        effective_floor (float): The flat mass in force: ``eps`` when some but not all orders
            are measurable, 1 when none is (flat weights by default), 0 when all are.
        shares (torch.Tensor): Each order's share of the alpha row's baseline on the calibration
            data, ``alpha_k f_k / sum_j alpha_j f_j``, where the headline's baseline lives; NaN
            when that baseline is not positive.
        num_blocks (int): The number of pooled calibration blocks.
    """

    alpha: torch.Tensor
    fourier_weight: torch.Tensor
    se: torch.Tensor
    resolved: torch.Tensor
    noise: torch.Tensor
    ratio: float
    test_blocks: int
    floor: float
    effective_floor: float
    shares: torch.Tensor
    num_blocks: int


class Evaluator:
    """Three blocked data splits, the kernel battery in force, and both uses of it.

    Construction validates the splits (floating spins in ``{-1, +1}``, one common ``dim``), cuts
    each into blocks of ``block_size`` rows (every split must hold a whole number of at least
    eight blocks, so that each block standard deviation rests on at least seven degrees of
    freedom) and allocates the fixed profile table in the training split's dtype on its device.
    A split's reference half (its blocks' own histograms and its distance from the uniform
    distribution) is computed on first use and cached.

    Args:
        train (torch.Tensor): The training split, a ``(rows, dim)`` floating spin matrix.
            Models are fit on it, and it is the default reference of :meth:`compute_loss`; a
            calibration split.
        val (torch.Tensor): The validation split, a spin matrix with ``dim`` columns; a
            calibration split.
        test (torch.Tensor): The test split, a spin matrix with ``dim`` columns; the default
            reference of :meth:`blocked` and :meth:`uniform_score`.
        block_size (int): Rows per block, ``r``. Defaults to 4096, which is sized for large
            samples (a 60,000-row split trimmed to 57,344 rows gives 14 blocks); smaller splits
            want a smaller block. Work is ``O(rows * r)`` per model, and the smallest resolvable
            misfit scales like ``1 / sqrt(rows * r)``.
        alpha (torch.Tensor, optional): Kernel weights over orders ``1..dim`` to hold from the
            start; see the :attr:`alpha` property. Defaults to ``None``, the Gaussian headline.

    Attributes:
        splits (dict[str, torch.Tensor]): The three splits by name: ``"train"``, ``"val"``,
            ``"test"``.
        train (torch.Tensor): The training split, as passed.
        val (torch.Tensor): The validation split, as passed.
        test (torch.Tensor): The test split, as passed.
        dim (int): The number of spins.
        block_size (int): Rows per block.

    Raises:
        ValueError: If a split fails :func:`~dwave.plugins.torch.evaluation.validate_spin_matrix`,
            the column counts differ, ``block_size`` is below 2, a split's rows are not a whole
            number of at least eight blocks, the dtype cannot count ``dim`` or ``block_size^2``
            exactly, or ``alpha`` is invalid.
    """

    def __init__(
        self,
        train: torch.Tensor,
        val: torch.Tensor,
        test: torch.Tensor,
        block_size: int = 4096,
        alpha: torch.Tensor | None = None,
    ) -> None:
        self.splits = {"train": train, "val": val, "test": test}
        for name, split in self.splits.items():
            validate_spin_matrix(split, name)
        self.train, self.val, self.test = train, val, test
        self.dim = int(train.shape[1])
        for name, split in self.splits.items():
            if split.shape[1] != self.dim:
                raise ValueError(
                    f"{name} must have dim={self.dim} columns like train, got {split.shape[1]}"
                )
        if block_size < 2:
            raise ValueError(f"block_size must be >= 2, got {block_size!r}")
        self.block_size = int(block_size)
        for name, split in self.splits.items():
            rows = split.shape[0]
            if rows % self.block_size:
                raise ValueError(
                    f"{name} has {rows} rows, not a whole number of blocks of "
                    f"block_size={self.block_size}: trim it to {rows - rows % self.block_size} "
                    f"rows, or choose a block size dividing every split"
                )
            min_blocks = 8
            if rows // self.block_size < min_blocks:
                raise ValueError(
                    f"too few blocks: {name} has {rows} rows at block_size={self.block_size}, "
                    f"B={rows // self.block_size} < {min_blocks}; lower block_size, or supply "
                    f">= {min_blocks * self.block_size} rows"
                )
            _check_exact_counting(split.dtype, self.dim, self.block_size)
        # Row 0 of the fixed rows is the Gaussian kernel, never the alpha row; row k the order-k
        # profile. Allocated in the training split's dtype on its device.
        self._fixed_rows = kernel_profiles(self.dim, train.dtype, train.device)
        self._xx: dict[str, torch.Tensor] = {}  # per split: the blocks' own histograms
        self._uniform_fixed: dict[str, torch.Tensor] = {}  # per split: (dim + 1, B) uniform arm
        self._alpha: torch.Tensor | None = None
        self.alpha = alpha

    # -- the battery in force -----------------------------------------------------------

    @property
    def alpha(self) -> torch.Tensor | None:
        """The kernel weights over orders ``1..dim`` in force, as assigned, or ``None``.

        While weights are held, profile 0 is the alpha row ``sum_k alpha_k K_k / C(dim, k)``,
        whose Walsh eigenvalue at order ``k`` is ``alpha_k / C(dim, k)``: proper for every
        admissible ``alpha``, strictly proper exactly when every weight is positive, and a zero
        weight switches its order off. Nothing is normalized: the row's scale is the weights'
        own (its value at distance 0 is ``sum_k alpha_k``), and only the weights
        :meth:`calibrate` fits sum to 1. By linearity every alpha-row per-block score, and hence
        every loss, is the same combination of the order rows'; the alpha row's standard error is
        the spread of those combined block scores, not a combination of the order rows' standard
        errors. Assigning ``None`` returns profile 0 to the Gaussian kernel. The tensor is held
        as a copy, values as given, in its own dtype and device, which must match the splits' or
        the alpha row fails inside torch.

        Raises:
            ValueError: On assignment, if the value is not a 1-D tensor of ``dim`` finite,
                nonnegative entries with at least one positive.
        """
        return self._alpha

    @alpha.setter
    def alpha(self, weights: torch.Tensor | None) -> None:
        if weights is None:
            self._alpha = None
            return
        _validate_alpha(weights, self.dim)
        self._alpha = weights.clone()

    @property
    def profiles(self) -> torch.Tensor:
        """The ``(dim + 1, dim + 1)`` profile table in force, row ``j`` the profile of entry ``j``.

        Row 0 is the alpha row while weights are held, else the Gaussian kernel; rows
        ``1..dim`` are the order kernels of :func:`~dwave.plugins.torch.kernels.kernel_profiles`.
        """
        if self._alpha is None:
            return self._fixed_rows
        head = self._alpha @ self._fixed_rows[1:]
        return torch.cat([head[None], self._fixed_rows[1:]], dim=0)

    @property
    def profile_names(self) -> tuple[str, ...]:
        """The row names of :attr:`profiles`: the headline, then ``"order k"`` for ``k = 1..dim``.

        The headline is ``"alpha"`` while weights are held, else ``"gaussian"``.
        """
        head = "gaussian" if self._alpha is None else "alpha"
        return (head, *(f"order {k}" for k in range(1, self.dim + 1)))

    def _with_headline(self, scores: torch.Tensor) -> torch.Tensor:
        """Swap the alpha combination of the order entries into entry 0 while weights are held.

        ``scores`` is profile-indexed on axis 0, ``(dim + 1, ...)``, and comes back unchanged
        when no weights are held.
        """
        if self._alpha is None:
            return scores
        head = torch.einsum("k,k...->...", self._alpha, scores[1:])
        return torch.cat([head[None], scores[1:]], dim=0)

    def calibrate(
        self, ratio: float = 10.0, eps: float = 1e-3, test_blocks: int | None = None
    ) -> Calibration:
        """Fit the two-level kernel weights on the calibration splits and hold them.

        Order by order, the rule asks whether a sample of the test split's size that *misses*
        the order's structure can be told from one that reproduces it, and weights the orders
        where it can equally. Three steps:

        1. ``f_k`` and ``se_k``, the per-order scores of ``train`` and ``val`` against the
           analytic uniform distribution, pooled block by block (both splits share the block
           size), with their mean and standard error over the pooled blocks, and the resolution
           gate ``f_k - 3 se_k > 0``: an order whose estimate is within three standard errors of
           zero carries no demonstrable structure;
        2. the floor ``noise_k``, the per-order standard error, over ``test_blocks`` blocks, of
           real data scored against real data: the first ``val``-sized rows of ``train`` are
           scored against the ``val`` blocks (:meth:`blocked`), each order's block standard
           deviation is taken, and it is scaled by ``1 / sqrt(test_blocks)``. A sample that
           reproduces the law scores at about this level at order ``k`` (a degenerate
           U-statistic); one that misses the order's structure scores about ``f_k``;
        3. the gate: order ``k`` is *measurable* when it is resolved and ``f_k >= ratio *
           noise_k`` with ``noise_k > 0``. Both conditions are needed: an unresolved ``f_k`` is
           estimation noise, yet at the middle orders it still exceeds the degenerate floor by
           chance. The measurable orders share ``1 - eps`` equally and every other order shares
           ``eps``, so the row stays characteristic; with no measurable order the weights are
           flat (a :class:`RuntimeWarning` says so), and with every order measurable they are
           flat.

        The weights depend on the calibration splits alone (the test split enters only through
        its block count), so every model then meets one kernel on the same blocks.

        Args:
            ratio (float): How many times the floor the data's weight must reach; finite and
                positive. Defaults to 10.
            eps (float): The floor's total mass, spread evenly over the unmeasurable orders; in
                ``(0, 1)``. Defaults to ``1e-3``.
            test_blocks (int, optional): Blocks of the split the weights will score against;
                at least 1. Defaults to ``None``, the test split's count.

        Raises:
            ValueError: If ``ratio`` is not a finite positive number, ``eps`` is not in
                ``(0, 1)``, ``test_blocks`` is below 1, or ``train`` holds fewer rows than
                ``val`` (no real sample of ``val``'s size to score against it).

        Warns:
            RuntimeWarning: If no order is measurable (the calibration splits carry no resolved
                structure a test-sized sample can see above its own floor), so the weights held
                are flat by default, not by choice.

        Returns:
            Calibration: ``alpha`` (the weights now held; they sum to 1), ``fourier_weight`` and
            ``se``, ``resolved`` (the measurable orders), ``noise`` (the floor per order),
            ``ratio`` and ``test_blocks`` as used, ``floor`` (``eps``), ``effective_floor``
            (``eps``; 1 with no measurable order, 0 with every order measurable), ``shares``,
            and ``num_blocks``.
        """
        if not (math.isfinite(ratio) and ratio > 0.0):
            raise ValueError(f"ratio must be a finite number > 0, got {ratio!r}")
        if not (0.0 < eps < 1.0):
            raise ValueError(f"eps must lie in (0, 1), got {eps!r}")
        blocks_to_score = self.num_blocks("test") if test_blocks is None else int(test_blocks)
        if blocks_to_score < 1:
            raise ValueError(f"test_blocks must be >= 1, got {test_blocks!r}")
        n_val = self.val.shape[0]
        if self.train.shape[0] < n_val:
            raise ValueError(
                f"train has {self.train.shape[0]} rows but val has {n_val}: the floor scores the"
                " first val-sized rows of train against val, so train must hold at least that"
            )
        n = self.dim

        blocks = torch.cat([self._reference("train")[1][1:], self._reference("val")[1][1:]], dim=1)
        fourier_weight, se = _mean_se(blocks)
        resolved = fourier_weight - 3.0 * se > 0
        # the floor: real data against real data, per order, as a test-sized split would see it
        per_block = self.blocked({"real": self.train[:n_val]}, "val").scores[1:, 0, :]
        noise = per_block.std(dim=1) / math.sqrt(blocks_to_score)
        measurable = resolved & (noise > 0) & (fourier_weight >= ratio * noise)
        m = int(measurable.sum())
        if m == 0:
            warnings.warn(
                "Evaluator.calibrate: no order is measurable (the calibration splits carry no"
                f" resolved structure at {ratio:g} times the floor a test split of"
                f" {blocks_to_score} blocks sees), so the weights held are flat by default, not"
                " by choice",
                RuntimeWarning,
                stacklevel=2,
            )
        if m == 0 or m == n:
            alpha = torch.full_like(fourier_weight, 1.0 / n)
            effective_floor = 1.0 if m == 0 else 0.0
        else:
            high = torch.tensor(
                (1.0 - eps) / m, dtype=fourier_weight.dtype, device=fourier_weight.device
            )
            low = torch.tensor(
                eps / (n - m), dtype=fourier_weight.dtype, device=fourier_weight.device
            )
            alpha = torch.where(measurable, high, low)
            effective_floor = eps
        self.alpha = alpha  # held as a copy; the record below carries the same values
        baseline = alpha @ fourier_weight
        shares = (
            alpha * fourier_weight / baseline
            if baseline > 0
            else torch.full_like(fourier_weight, math.nan)
        )
        return Calibration(
            alpha=alpha,
            fourier_weight=fourier_weight,
            se=se,
            resolved=measurable,
            noise=noise,
            ratio=float(ratio),
            test_blocks=blocks_to_score,
            floor=eps,
            effective_floor=effective_floor,
            shares=shares,
            num_blocks=int(blocks.shape[1]),
        )

    # -- splits and their cached reference half ----------------------------------------

    def split(self, name: str) -> torch.Tensor:
        """The split named ``name``.

        Args:
            name (str): ``"train"``, ``"val"``, or ``"test"``.

        Raises:
            ValueError: If ``name`` is not one of the three.

        Returns:
            torch.Tensor: The split's spin matrix, as passed to the constructor.
        """
        try:
            return self.splits[name]
        except KeyError:
            raise ValueError(f"split must be one of {tuple(self.splits)}, got {name!r}") from None

    def num_blocks(self, split: str = "test") -> int:
        """``B``, the number of blocks of ``split``.

        Args:
            split (str): The split. Defaults to ``"test"``.

        Returns:
            int: The number of blocks.
        """
        return self.split(split).shape[0] // self.block_size

    def _reference(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """The split's cached reference half: its within-block histograms and its uniform arm.

        Returns the ``(B, dim + 1)`` histograms and the ``(dim + 1, B)`` per-block fixed-battery
        scores of the split against the analytic uniform distribution, computed on first use.
        """
        if split not in self._xx:
            rows = self.split(split)
            xx = _reference_histograms(rows, self.num_blocks(split))
            reference_only = _Histograms(("uniform",), xx, {}, {}, self.block_size)
            self._xx[split] = xx
            self._uniform_fixed[split] = _scores_from_histograms(self._fixed_rows, reference_only)[
                :, 0, :
            ]
        return self._xx[split], self._uniform_fixed[split]

    # -- evaluation --------------------------------------------------------------------

    def blocked(
        self, xqueries: Mapping[str, torch.Tensor | Uniform], split: str = "test"
    ) -> BlockedScores:
        """Per-block scores of several query models against one split, under the battery in force.

        :func:`~dwave.plugins.torch.evaluation.blocked_scores` with the split as the reference,
        its cached histograms, and :attr:`profiles` as the table: every query is scored against
        the same blocks, which pairs every comparison between them.

        Args:
            xqueries (Mapping[str, torch.Tensor | Uniform]): Query models by name: a spin matrix
                of exactly the split's shape, or :data:`~dwave.plugins.torch.evaluation.UNIFORM`
                for the analytic uniform distribution.
            split (str): The reference split. Defaults to ``"test"``.

        Raises:
            ValueError: If ``split`` is unknown, ``xqueries`` is empty, or a query is not a spin
                matrix of the split's shape.

        Returns:
            BlockedScores: A record whose ``scores`` has shape ``(dim + 1, M, B)``,
            profile-indexed.
        """
        reference = self.split(split)
        samples = _validate_models(xqueries, reference, split)
        xx, _ = self._reference(split)
        histograms = _block_histograms(reference, samples, self.num_blocks(split), xx=xx)
        fixed = _scores_from_histograms(self._fixed_rows, histograms)  # (dim + 1, M, B)
        return BlockedScores(
            histograms.model_names,
            self._with_headline(fixed),
            self.num_blocks(split),
            self.block_size,
        )

    def score(self, xquery: torch.Tensor, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Blocked ``MMD^2`` estimate of one sample against a split, with its standard error.

        The evaluation use of the battery. Smaller is better; the population value is zero
        exactly when the sample's law matches the split's on every order the kernel sees, and at
        entry ``k`` the unit is the mean squared error per order-``k`` parity coefficient
        (bounded by 4). Read it against :meth:`uniform_score`: a model whose score at order
        ``k`` equals the data's weight there has captured nothing at that order, and one at
        noise level has captured all of it this sample size resolves. Near an exact match the
        per-block estimates are degenerate U-statistics, so ``scores / se`` is a descriptive
        signal-to-noise ratio, not an exact test statistic. Estimates are unbiased, hence
        sometimes negative; nothing is clipped.

        Args:
            xquery (torch.Tensor): The generated sample: a floating spin matrix of exactly the
                split's shape, independent of the split, with IID rows in an unstructured
                storage order.
            split (str): The reference split, named explicitly.

        Raises:
            ValueError: If ``split`` is unknown or ``xquery`` is not a spin matrix of its shape.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(scores, se)``, two ``(dim + 1,)`` tensors in
            the split's dtype on its device, the per-profile estimates and their standard errors
            over the split's blocks.
        """
        blocked = self.blocked({"xquery": xquery}, split)
        return _mean_se(blocked.scores[:, 0, :])

    def uniform_score(self, split: str = "test") -> tuple[torch.Tensor, torch.Tensor]:
        """The split's own distance from the analytic uniform distribution, with standard errors.

        At entry ``k`` this is the data's *Fourier weight* at order ``k``, the mean squared
        order-``k`` Walsh coefficient: the structure the data carry there and the most a model
        can miss, estimated without a sample of the uniform distribution and without cross-term
        noise (the :data:`~dwave.plugins.torch.evaluation.UNIFORM` arm of
        :func:`~dwave.plugins.torch.evaluation.blocked_scores`), then cached. An order where this
        weight sits within its standard error of zero carries no measurable structure at this
        sample size.

        Args:
            split (str): The split. Defaults to ``"test"``.

        Raises:
            ValueError: If ``split`` is unknown.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(scores, se)`` as in :meth:`score`.
        """
        _, uniform_fixed = self._reference(split)
        return _mean_se(self._with_headline(uniform_fixed))

    # -- training ----------------------------------------------------------------------

    def compute_loss(
        self,
        xquery: torch.Tensor,
        block_size: int | None = None,
        x: torch.Tensor | None = None,
        split: str = "train",
    ) -> torch.Tensor:
        """Differentiable per-profile ``MMD^2`` of a generated batch against a batch of a split.

        :func:`~dwave.plugins.torch.nn.functional.maximum_mean_discrepancy_loss` with the
        evaluator's kernel weights and a reference batch of the split: with ``block_size=None``
        the whole batch is one block, the plain U-statistic over all ``m (m - 1)`` within-sample
        pairs and ``m^2`` cross pairs. A divisor ``block_size`` of ``m`` cuts both batches into
        ``m / block_size`` blocks scored block against block and averaged (the paired
        construction of :meth:`score`), with Gram work and memory falling from ``O(m^2)`` to
        ``O(m block_size)``. At hard spins each entry equals the histogram estimator of
        :func:`~dwave.plugins.torch.evaluation.blocked_scores` to rounding. At soft spins the
        kernels are evaluated at the real distance ``d = (dim - <x, y>) / 2``, a smooth function
        of the batch through its inner products alone, so the loss is a smooth surrogate whose
        minimizers at the corners of the cube are the hard-spin ones. The gradient flows through
        the within-batch and cross terms into ``xquery`` (``loss[profile].backward()``).

        Args:
            xquery (torch.Tensor): The generated batch, ``(m >= 2, dim)`` of a floating dtype:
                hard spins, or soft spins in ``[-1, 1]`` (a ``tanh`` output), with or without
                autograd history.
            block_size (int, optional): Rows per block; ``>= 2`` and a divisor of ``m``.
                Defaults to ``None``, one block.
            x (torch.Tensor, optional): The reference batch, a spin matrix of exactly
                ``xquery``'s shape and dtype. Defaults to ``None``, a fresh, unseeded draw of
                ``m`` distinct rows of ``split``.
            split (str): The split the default reference is drawn from. Defaults to
                ``"train"``.

        Raises:
            ValueError: If ``xquery`` is not an ``(m >= 2, dim)`` floating tensor of finite
                values in ``[-1, 1]``; ``block_size`` is below 2 or does not divide ``m``; ``x``
                fails :func:`~dwave.plugins.torch.evaluation.validate_spin_matrix` or differs in
                shape; the batch has more rows than ``split`` when ``x`` is omitted; or
                ``split`` is unknown.

        Returns:
            torch.Tensor: A ``(dim + 1,)`` tensor in the batch's dtype: per profile, an unbiased
            estimate of the ``MMD^2`` between the batch and the reference batch, differentiable
            in ``xquery``.
        """
        reference = self.split(split)
        y = xquery
        if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] != self.dim:
            raise ValueError(
                f"xquery must be a (m >= 2, dim={self.dim}) batch, got shape {tuple(y.shape)}"
            )
        m = y.shape[0]
        if x is None:
            if m > reference.shape[0]:
                raise ValueError(
                    f"the batch has {m} rows but {split} holds {reference.shape[0]}: pass x, or "
                    "a smaller batch"
                )
            x_ref = reference[torch.randperm(reference.shape[0], device=reference.device)[:m]]
        else:
            validate_spin_matrix(x, "x")
            if x.shape != y.shape:
                raise ValueError(
                    f"x must have the generated batch's shape {tuple(y.shape)}, got "
                    f"{tuple(x.shape)}"
                )
            x_ref = x
        return maximum_mean_discrepancy_loss(y, x_ref, alpha=self._alpha, block_size=block_size)
