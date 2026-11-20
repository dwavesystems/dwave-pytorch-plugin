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
r"""Maximum pseudolikelihood estimation of Ising models on a graph.

The pseudolikelihood of a spin configuration under a model is the product, over the nodes, of
the conditional probability of each spin given all the others (J. Besag, *Statistical Analysis
of Non-Lattice Data*, The Statistician 24(3), 1975). Under the Ising model on a graph with
biases :math:`(h, J)`, :math:`P(s) \propto \exp\{-E(s)\}` with
:math:`E(s) = \sum_i h_i s_i + \sum_{(i, j)} J_{ij} s_i s_j`, the spin of node :math:`i` is
logistic in its effective field :math:`\theta_i(s) = h_i + \sum_j J_{ij} s_j`
(:meth:`~dwave.plugins.torch.graph.GraphIndex.effective_field`),
:math:`P(s_i \mid s_{-i}) = \sigma(-2 s_i \theta_i(s))`, so the negative log-pseudolikelihood
(NLPL) of a configuration is :math:`\sum_i \mathrm{softplus}(2 s_i \theta_i(s))`. Averaged over
a sample, the NLPL is convex in the biases and involves neither the partition function nor
samples of the model, unlike the log likelihood, whose gradient needs equilibrium samples. Its
minimizer is a consistent estimator of the biases of a fully visible model, at some loss of
statistical efficiency relative to maximum likelihood.

Three functions share the objective:

* :func:`negative_log_pseudolikelihood` evaluates it under given biases, for a single model or
  a batch of models and differentiably in the biases, as a training loss or a held-out score.
* :func:`fit_pseudolikelihood` minimizes it over all the biases of one model with L-BFGS in
  float64 and returns the estimate with diagnostics.
* :func:`fit_inverse_temperature` holds the biases fixed as a reference Hamiltonian
  :math:`E^*` and minimizes it over the single scale :math:`\beta` of the family
  :math:`P_\beta(s) \propto \exp\{-\beta E^*(s)\}`, the inverse temperature at which the spins
  were sampled, by Newton's method, for a batch of models at once and with a standard error.

The two fits compose: on data drawn at inverse temperature :math:`\beta` from :math:`E^*`, the
joint fit recovers :math:`\beta E^*` while the one-parameter fit against :math:`E^*` recovers
:math:`\beta`, and against the joint fit's own estimate it returns exactly 1, the joint optimum
being stationary along its own ray.

The functions take complete spin configurations, one column per node of the graph in the order
of its nodes, like every function of the package. The pseudolikelihood of partially observed
configurations, which would marginalize the unobserved spins, is not implemented, so a
:class:`~dwave.plugins.torch.models.GraphRestrictedBoltzmannMachine` with hidden units cannot be
fitted. Degenerate data, a node whose spin is constant across the sample or an edge whose
endpoint product is, send the joint estimate to infinity along that coordinate, as logistic
regression diverges on separable data; the pure fit refuses such data, and the penalties ``l1``
and ``l2`` of :func:`fit_pseudolikelihood` bound the estimate for any data. The effective fields
are evaluated by a dense product with the coupling matrix of the graph, as elsewhere in the
package, or, on large sparse graphs, by two scatters over the edges, whichever is cheaper.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from dwave.plugins.torch.graph import GraphIndex

__all__ = [
    "InverseTemperatureFit",
    "PseudolikelihoodFit",
    "fit_inverse_temperature",
    "fit_pseudolikelihood",
    "negative_log_pseudolikelihood",
]


# ---------------------------------------------------------------------- validation ----------------


def _validate_spins(spins: torch.Tensor, n_nodes: int, batch_shape: torch.Size = torch.Size()) -> None:
    """Raise unless ``spins`` holds ``±1`` configurations of shape ``(*batch_shape, M, n_nodes)``."""
    if spins.ndim < 2 or spins.shape[-1] != n_nodes:
        raise ValueError(f"spins must have shape (..., M, {n_nodes}), got {tuple(spins.shape)}.")
    if batch_shape and spins.shape[:-2] != batch_shape:
        raise ValueError(
            f"spins must have shape {(*batch_shape, 'M', n_nodes)}, batched like the biases, got "
            f"{tuple(spins.shape)}."
        )
    if spins.shape[-2] == 0:
        raise ValueError("At least one sample per model is required, got none.")
    if not bool((spins.abs() == 1).all()):
        raise ValueError("spins must hold ±1 values only.")


def _validate(
    graph: GraphIndex, linear: torch.Tensor, quadratic: torch.Tensor, spins: torch.Tensor
) -> torch.Size:
    """Raise unless the biases define a batch of models on ``graph`` and ``spins`` are ``±1``
    configurations of the batch; return the batch shape."""
    n_nodes, n_edges = graph.n_nodes, graph.n_edges
    if linear.ndim < 1 or linear.shape[-1] != n_nodes:
        raise ValueError(
            f"linear must have shape (*batch, {n_nodes}), one bias per node, got "
            f"{tuple(linear.shape)}."
        )
    if quadratic.ndim < 1 or quadratic.shape[-1] != n_edges:
        raise ValueError(
            f"quadratic must have shape (*batch, {n_edges}), one bias per edge, got "
            f"{tuple(quadratic.shape)}."
        )
    batch_shape = linear.shape[:-1]
    if quadratic.shape[:-1] != batch_shape:
        raise ValueError(
            "linear and quadratic must define the same batch of models, got batch shapes "
            f"{tuple(batch_shape)} and {tuple(quadratic.shape[:-1])}."
        )
    _validate_spins(spins, n_nodes, batch_shape)
    return batch_shape


def _validate_chunk(sample_chunk: int | None) -> None:
    if sample_chunk is not None and sample_chunk < 1:
        raise ValueError(f"sample_chunk must be a positive int or None, got {sample_chunk}.")


# ---------------------------------------------------------------------- effective fields ----------


def _use_dense_fields(n_nodes: int, n_edges: int) -> bool:
    """Whether the effective fields go through the dense coupling matrix rather than two scatters
    over the edges.

    The scatters cost one pass over every (sample, edge) pair, the dense product ``n_nodes**2``
    multiply-adds per sample at a far higher throughput, so the product wins once ``n_nodes**2``
    is within a few hundred times the number of edges, and loses on sparse graphs with thousands
    of nodes and a few edges per node, such as the working graph of a QPU. Small graphs always
    take the dense path.
    """
    return n_nodes <= 256 or n_nodes * n_nodes <= 400 * n_edges


def _effective_fields(
    graph: GraphIndex, spins: torch.Tensor, linear: torch.Tensor, quadratic: torch.Tensor
) -> torch.Tensor:
    """Effective fields of spins of shape ``(..., M, n_nodes)`` under biases batched like them
    or unbatched, all floating tensors of one dtype and device; differentiable in the biases.

    The dense path is :meth:`~dwave.plugins.torch.graph.GraphIndex.effective_field`; the sparse
    path adds the bias of every edge, times the spin at one endpoint, to the field at the other.
    """
    if _use_dense_fields(graph.n_nodes, graph.n_edges):
        return graph.effective_field(spins, linear=linear, quadratic=quadratic)
    fields = torch.zeros_like(spins)
    if graph.n_edges:
        edge_biases = quadratic if quadratic.ndim == 1 else quadratic.unsqueeze(-2)
        fields = fields.index_add(-1, graph.edge_idx_i, edge_biases * spins[..., graph.edge_idx_j])
        fields = fields.index_add(-1, graph.edge_idx_j, edge_biases * spins[..., graph.edge_idx_i])
    return fields + (linear if linear.ndim == 1 else linear.unsqueeze(-2))


# ---------------------------------------------------------------------- the objective -------------


def negative_log_pseudolikelihood(
    graph: GraphIndex,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    spins: torch.Tensor,
    reduction: Literal["mean", "sum", "none"] = "mean",
) -> torch.Tensor:
    r"""Negative log-pseudolikelihood of spins under the Ising models of given biases.

    The negative log-pseudolikelihood of a configuration is the sum over the nodes of the
    negative log probability of each spin given all the others,
    :math:`\sum_i \mathrm{softplus}(2 s_i \theta_i(s))` with :math:`\theta` the effective
    fields (see the module docstring). It is differentiable in the biases, so it serves as a
    training loss, and, in nats per configuration, as a held-out score (lower is better). The
    biases define one model or a batch of them, as in
    :meth:`~dwave.plugins.torch.graph.GraphIndex.energy`; the ``(M, n_edges)``-sized
    intermediates are materialized in one pass, so chunk a large sample externally.

    Args:
        graph (GraphIndex): The graph of the models.
        linear (torch.Tensor): Linear biases of shape ``(*batch, n_nodes)``, or ``(n_nodes,)``
            for a single model.
        quadratic (torch.Tensor): Quadratic biases of the edges of shape ``(*batch, n_edges)``,
            or ``(n_edges,)``, in the order of the graph's edges.
        spins (torch.Tensor): ``±1`` configurations of shape ``(*batch, M, n_nodes)``, ``M`` per
            model, with one column per node in the order of the graph's nodes; or of any shape
            ``(..., M, n_nodes)`` for a single model.
        reduction (str): ``"mean"`` (default) or ``"sum"`` over the ``M`` configurations of
            each model, or ``"none"`` for the value of every configuration.

    Raises:
        ValueError: If the biases do not hold one bias per node and per edge or are not batched
            alike, ``spins`` are not ``±1`` values of the shape above, or ``reduction`` is not
            recognized.

    Returns:
        torch.Tensor: The values, of shape ``spins.shape[:-2]`` or, for ``"none"``,
        ``spins.shape[:-1]``, in the dtype of ``linear``.
    """
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f'reduction must be "mean", "sum" or "none", got {reduction!r}.')
    _validate(graph, linear, quadratic, spins)
    spins = spins.to(device=linear.device, dtype=linear.dtype)
    fields = _effective_fields(graph, spins, linear, quadratic)
    values = F.softplus(2.0 * spins * fields).sum(-1)
    if reduction == "mean":
        return values.mean(-1)
    if reduction == "sum":
        return values.sum(-1)
    return values


# ---------------------------------------------------------------------- the joint fit -------------


@dataclass(frozen=True)
class PseudolikelihoodFit:
    """The estimate of :func:`fit_pseudolikelihood` and its diagnostics.

    Attributes:
        linear (torch.Tensor): The fitted linear biases, shape ``(n_nodes,)``, in
            ``torch.float64`` on the graph's device.
        quadratic (torch.Tensor): The fitted quadratic biases of the edges, shape ``(n_edges,)``.
        nlpl (float): The mean negative log-pseudolikelihood per configuration at the estimate,
            in nats, without the penalties.
        grad_inf_norm (float): Infinity norm of the gradient of the minimized objective at the
            estimate, the first-order optimality measure.
        n_iterations (int): L-BFGS iterations performed; 0 if the origin already satisfied
            ``tolerance_grad``.
        n_function_evals (int): Evaluations of the objective and its gradient inside the
            optimizer; the line search evaluates more than once per iteration.
        converged (bool): Whether ``grad_inf_norm <= tolerance_grad``. ``False`` means that the
            iteration cap was reached, or that the line search could make no further progress
            at the precision of the objective, which is where tolerances below about 1e-9 land.
    """

    linear: torch.Tensor
    quadratic: torch.Tensor
    nlpl: float
    grad_inf_norm: float
    n_iterations: int
    n_function_evals: int
    converged: bool


def _raise_if_separable(graph: GraphIndex, spins: torch.Tensor, chunk: int) -> None:
    """Raise when the maximum pseudolikelihood estimate provably diverges along some bias.

    The spin of a node that is constant across the sample, or the product of the endpoint spins
    of an edge that is, sends the corresponding bias to infinity, as logistic regression diverges
    on linearly separable data. The check is exact: ``±1`` values are summed in float64, where
    the sums are integers, so a column is constant iff its sum is ``±num_samples``.
    """
    num_samples = spins.shape[0]
    device = graph.edge_idx_i.device
    node_sums = torch.zeros(graph.n_nodes, dtype=torch.float64, device=device)
    edge_sums = torch.zeros(graph.n_edges, dtype=torch.float64, device=device)
    for start in range(0, num_samples, chunk):
        s = spins[start:start + chunk].to(device=device, dtype=torch.float64)
        node_sums += s.sum(0)
        if graph.n_edges:
            edge_sums += (s[:, graph.edge_idx_i] * s[:, graph.edge_idx_j]).sum(0)
    constant_nodes = node_sums.abs() == num_samples
    constant_edges = edge_sums.abs() == num_samples
    if constant_nodes.any() or constant_edges.any():
        nodes = [graph.nodes[k] for k in constant_nodes.nonzero().flatten().tolist()]
        edges = [graph.edges[k] for k in constant_edges.nonzero().flatten().tolist()]
        raise ValueError(
            f"The data is degenerate: the spins of nodes {nodes} and the products of the "
            f"endpoint spins of edges {edges} are constant across all {num_samples} samples, so "
            "the maximum pseudolikelihood estimate diverges along those biases, as logistic "
            "regression does on separable data. Remove the constant statistics from the data "
            "and the graph (prune constant nodes, merge perfectly correlated pairs), or set l1 "
            "or l2 > 0 to bound the estimate."
        )


def fit_pseudolikelihood(
    graph: GraphIndex,
    spins: torch.Tensor,
    *,
    max_iter: int = 500,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 0.0,
    sample_chunk: int | None = 2048,
    l1: float = 0.0,
    l2: float = 0.0,
) -> PseudolikelihoodFit:
    r"""Fit the biases of an Ising model on a graph to spin data by maximum pseudolikelihood.

    Minimizes the mean negative log-pseudolikelihood of the sample (see the module docstring)
    over the linear and quadratic biases. The objective is convex, so the optimizer starts from
    the origin, where the estimate is independent of any initialization, and runs a full-batch
    L-BFGS with a strong-Wolfe line search on float64 copies of the biases; backends without
    float64, such as Apple's MPS, are not supported. The sample is processed in chunks of
    ``sample_chunk`` rows, each chunk backpropagated before the next, so the peak intermediate
    is ``sample_chunk`` times the number of edges (or, on the dense path, the number of nodes,
    plus the coupling matrix) rather than the size of the whole sample; the chunking is exact.
    The fit is deterministic.

    With ``l1`` or ``l2`` positive, the objective is the penalized
    :math:`\mathrm{NLPL} + l_1 (\|h\|_1 + \|J\|_1) + l_2 (\|h\|^2 + \|J\|^2)`, the maximum a
    posteriori estimate under a Laplace and a Gaussian prior, which is bounded for any data
    and, with ``l2 > 0``, strictly convex, so the estimate is unique. The penalties ride on the
    mean per-configuration NLPL, so ``l2 ~ 1 / num_samples`` is a fixed unit prior that washes
    out as the sample grows. The ``l1`` term is non-smooth and L-BFGS treats it by subgradient,
    a practical shrinkage rather than a proximal method: coefficients shrink but are not driven
    exactly to zero, and the first-order ``converged`` reading is approximate near kinks.

    Args:
        graph (GraphIndex): The graph of the model.
        spins (torch.Tensor): ``±1`` configurations of shape ``(num_samples, n_nodes)``, one
            column per node in the order of the graph's nodes.
        max_iter (int): Maximum L-BFGS iterations. Defaults to 500.
        tolerance_grad (float): Termination threshold on the infinity norm of the gradient.
            Defaults to 1e-7.
        tolerance_change (float): L-BFGS stagnation threshold on the change of the biases and
            of the objective. The default 0 disables stagnation stops, so termination is by
            ``tolerance_grad`` or ``max_iter`` and ``converged`` is a genuine first-order
            statement; a positive value, torch's own default being 1e-9, permits earlier stops
            that typically leave the gradient above ``tolerance_grad``.
        sample_chunk (int, optional): Rows per chunk. Defaults to 2048; ``None`` processes the
            whole sample at once.
        l1 (float): Strength of the Laplace (lasso) penalty. Defaults to 0.
        l2 (float): Strength of the Gaussian (ridge) penalty. Defaults to 0; the pure maximum
            pseudolikelihood estimate, which refuses degenerate data.

    Raises:
        ValueError: If ``spins`` are not ``±1`` values of the shape above, an option is invalid,
            or, with ``l1 == l2 == 0`` only, the data is degenerate: the spin of some node, or
            the product of the endpoint spins of some edge, is constant across the sample, where
            the estimate diverges.

    Returns:
        PseudolikelihoodFit: The fitted biases, in ``torch.float64`` on the graph's device,
        with the NLPL, gradient norm, iteration and evaluation counts, and a convergence flag.
    """
    _validate_spins(spins, graph.n_nodes)
    if spins.ndim != 2:
        raise ValueError(
            "fit_pseudolikelihood fits a single model: spins must have shape "
            f"(num_samples, {graph.n_nodes}), got {tuple(spins.shape)}."
        )
    _validate_chunk(sample_chunk)
    for name, value in (("l1", l1), ("l2", l2)):
        if not (0.0 <= value < math.inf):
            raise ValueError(f"{name} must be a finite float >= 0, got {value}.")

    device, dtype = graph.edge_idx_i.device, torch.float64
    num_samples, n_edges = spins.shape[0], graph.n_edges
    chunk = num_samples if sample_chunk is None else sample_chunk
    if l1 == 0.0 and l2 == 0.0:
        # Only the pure estimate can diverge; a penalty bounds it for any data
        _raise_if_separable(graph, spins, chunk)

    linear = torch.zeros(graph.n_nodes, dtype=dtype, device=device, requires_grad=True)
    quadratic = torch.zeros(n_edges, dtype=dtype, device=device, requires_grad=n_edges > 0)
    biases = [linear, quadratic] if n_edges else [linear]
    optimizer = torch.optim.LBFGS(
        biases,
        max_iter=max_iter,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        line_search_fn="strong_wolfe",
    )
    last_nlpl = 0.0

    def closure() -> torch.Tensor:
        # The objective is a mean over the sample, so the backward passes of the chunks
        # accumulate its exact gradient while each chunk's intermediates are freed before the next
        nonlocal last_nlpl
        optimizer.zero_grad()
        nlpl = torch.zeros((), dtype=dtype, device=device)
        for start in range(0, num_samples, chunk):
            s = spins[start:start + chunk].to(device=device, dtype=dtype)
            fields = _effective_fields(graph, s, linear, quadratic)
            chunk_nlpl = F.softplus(2.0 * s * fields).sum() / num_samples
            chunk_nlpl.backward()
            nlpl += chunk_nlpl.detach()
        last_nlpl = float(nlpl)
        objective = nlpl
        if l1 or l2:
            penalty = torch.zeros((), dtype=dtype, device=device)
            for bias in biases:
                penalty = penalty + l1 * bias.abs().sum() + l2 * bias.square().sum()
            penalty.backward()
            objective = objective + penalty.detach()
        return objective

    optimizer.step(closure)
    # Evaluated once more at the estimate, outside the step so that the optimizer's counters are
    # not inflated: the gradient of the objective and the NLPL without the penalties
    closure()
    grad_inf_norm = max(float(bias.grad.abs().max()) for bias in biases)
    state = optimizer.state_dict()["state"].get(0, {})
    return PseudolikelihoodFit(
        linear=linear.detach(),
        quadratic=quadratic.detach(),
        nlpl=last_nlpl,
        grad_inf_norm=grad_inf_norm,
        n_iterations=int(state.get("n_iter", 0)),
        n_function_evals=int(state.get("func_evals", 0)),
        converged=grad_inf_norm <= tolerance_grad,
    )


# ---------------------------------------------------------------------- the temperature fit -------


@dataclass(frozen=True)
class InverseTemperatureFit:
    """The estimate of :func:`fit_inverse_temperature`, its standard error and a flag, one entry
    per model.

    Attributes:
        beta (torch.Tensor): The maximum pseudolikelihood estimate of the inverse temperature,
            shape ``(*batch)`` in ``torch.float64``: the scale for which
            :math:`P(s) \\propto \\exp\\{-\\beta E^*(s)\\}` best explains the spins, :math:`E^*`
            being the fixed Hamiltonian of the given biases. ``inf`` when no spin of the sample
            is excited against its effective field (every spin in its lower-energy state, the
            zero-temperature limit) and ``-inf`` when every spin is.
        beta_stderr (torch.Tensor): Sandwich (Godambe) standard error of ``beta``, treating the
            configurations as independent draws; ``nan`` for an infinite estimate or a single
            configuration.
        converged (torch.Tensor): Whether the estimate met ``tolerance`` (an exact infinite
            limit counts as converged), as a boolean tensor; ``False`` means that ``max_iter``
            ran out.
    """

    beta: torch.Tensor
    beta_stderr: torch.Tensor
    converged: torch.Tensor


def _excitation_energies(
    graph: GraphIndex,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    spins: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    r"""Single-spin excitation energies :math:`\epsilon_i(s) = E(s) - E(s^{(i)}) = 2 s_i
    \theta_i(s)`, :math:`s^{(i)}` being :math:`s` with spin :math:`i` flipped: positive where the
    flip would lower the energy (the spin is excited against its field), negative where the spin
    is relaxed. Evaluated without autograd in float64, in ``chunk``-row chunks; the result has
    the shape of ``spins``, on the device of the biases.
    """
    device, dtype = linear.device, torch.float64
    linear, quadratic = linear.detach().to(dtype), quadratic.detach().to(dtype)
    energies = torch.empty(spins.shape, dtype=dtype, device=device)
    with torch.no_grad():
        for start in range(0, spins.shape[-2], chunk):
            s = spins[..., start:start + chunk, :].to(device=device, dtype=dtype)
            energies[..., start:start + chunk, :] = 2.0 * s * _effective_fields(
                graph, s, linear, quadratic
            )
    return energies


def fit_inverse_temperature(
    graph: GraphIndex,
    linear: torch.Tensor,
    quadratic: torch.Tensor,
    spins: torch.Tensor,
    *,
    max_iter: int = 100,
    tolerance: float = 1e-10,
    sample_chunk: int | None = 2048,
) -> InverseTemperatureFit:
    r"""Estimate the inverse temperature at which spins were sampled from the models of given
    biases, by maximum pseudolikelihood.

    The biases are held fixed as the reference Hamiltonian :math:`E^*` of each model and the
    single scale :math:`\beta` of the family :math:`P_\beta(s) \propto \exp\{-\beta E^*(s)\}`,
    the inverse temperature, is fitted by maximum pseudolikelihood: the mean negative
    log-pseudolikelihood of the sample restricted to that family,

    .. math::

        \mathrm{NLPL}(\beta) = \frac{1}{M} \sum_{s} \sum_i
        \mathrm{softplus}(\beta\, \epsilon_i(s)), \qquad
        \epsilon_i(s) = E^*(s) - E^*(s^{(i)}) = 2 s_i \theta_i(s),

    is minimized over :math:`\beta`. The :math:`\epsilon_i(s)` are the single-spin excitation
    energies under :math:`E^*`, positive where flipping spin :math:`i` would lower the energy
    (the spin is excited against its effective field :math:`\theta_i`) and negative where the
    spin is relaxed. The NLPL is a sum of softplus terms of a linear function of :math:`\beta`,
    hence convex, strictly so once any :math:`\epsilon_i(s)` is nonzero, so the estimate is
    unique and needs no initialization. This is the temperature estimator of Raymond, Yarkoni
    and Andriyash, `Global Warming: Temperature Estimation in Annealers
    <https://doi.org/10.3389/fict.2016.00023>`_, a logistic regression of every spin on its own
    effective field with one shared slope, and the estimate of
    :func:`dwave.system.temperatures.maximum_pseudolikelihood_temperature`, computed in PyTorch
    on the device of the biases, for a batch of models at once and with a standard error.

    If the spins are Boltzmann samples of :math:`E^*` at some inverse temperature, the estimate
    is consistent for it. If they are not, such as the reads of a quantum annealer or a Markov
    chain that has not equilibrated, :math:`\beta` measures the rate of single-spin excitations
    relative to :math:`E^*`, an *effective* inverse temperature, which is the quantity that
    decides how a model trained on such samples is to be rescaled before it is programmed: its
    reciprocal is the ``prefactor`` of a :class:`~dwave.plugins.torch.samplers.DimodSampler`. A
    negative estimate says that the spins are anti-aligned with their fields, usually a sign
    convention error upstream.

    The excitation energies do not depend on :math:`\beta`, so one pass over the sample builds
    them, one float64 tensor of the shape of ``spins`` plus ``sample_chunk``-bounded
    intermediates, and every later step is an elementwise pass over that tensor. The derivative
    :math:`\mathrm{NLPL}'(\beta) = \frac{1}{M} \sum \epsilon\, \sigma(\beta \epsilon)` is
    increasing in :math:`\beta`, and Newton's method with the analytic curvature finds its zero
    from :math:`\beta = 0`, where the first step is the moment estimate
    :math:`-2 \sum \epsilon / \sum \epsilon^2`. Each term :math:`\epsilon\, \sigma(\beta
    \epsilon)` has a second derivative in :math:`\beta` of the sign of :math:`-\beta`, so the
    derivative is convex on :math:`\beta < 0` and concave on :math:`\beta > 0`, and Newton
    started at that inflection point approaches the zero monotonically from one side, never
    overshooting, with quadratic convergence near it; no bracket is needed. The iteration of a
    model stops once its Newton step is at most ``tolerance * (1 + |beta|)``.

    The derivative crosses zero only if some :math:`\epsilon_i(s)` is positive and some
    negative. A sample without any excitation, every spin in its lower-energy state given its
    neighbours, as a repeated ground state, puts the maximizer at :math:`\beta = +\infty` (zero
    temperature), and a sample without any relaxed spin at :math:`-\infty`. These limits are
    returned as the exact estimate, converged and with a ``nan`` standard error: they are the
    complete, physically meaningful answer, and calibration loops must handle them, because
    excitations are rare in low-temperature annealer reads. A sample on which :math:`E^*` has
    no effective field anywhere (:math:`\epsilon \equiv 0`, as under an all-zero Hamiltonian)
    leaves :math:`\beta` unidentifiable and raises :class:`ValueError`.

    The pseudolikelihood is a composite likelihood, so the sampling variance of the estimate
    takes the sandwich form :math:`\mathrm{Var}(u) / (M\, \mathrm{NLPL}''(\beta)^2)` with
    :math:`u_s = \sum_i \epsilon_i(s)\, \sigma(\beta \epsilon_i(s))` the score of each
    configuration, of mean zero at the estimate, and :math:`\mathrm{NLPL}''` the mean curvature.
    It treats the configurations as independent draws; for an autocorrelated Markov chain, thin
    the chain or scale by its effective sample size.

    Args:
        graph (GraphIndex): The graph of the models.
        linear (torch.Tensor): Linear biases of shape ``(*batch, n_nodes)``, or ``(n_nodes,)``
            for a single model.
        quadratic (torch.Tensor): Quadratic biases of the edges of shape ``(*batch, n_edges)``,
            or ``(n_edges,)``, in the order of the graph's edges.
        spins (torch.Tensor): ``±1`` configurations of shape ``(*batch, M, n_nodes)``, ``M`` per
            model, with one column per node in the order of the graph's nodes; or of any shape
            ``(..., M, n_nodes)`` for a single model, one estimate per leading index.
        max_iter (int): Maximum Newton updates. Defaults to 100.
        tolerance (float): Relative stopping tolerance on ``beta``. Defaults to 1e-10.
        sample_chunk (int, optional): Rows per chunk while building the excitation energies and
            evaluating the derivatives. Defaults to 2048; ``None`` processes the whole sample at
            once.

    Raises:
        ValueError: If the biases do not hold one bias per node and per edge or are not batched
            alike, ``spins`` are not ``±1`` values of the shape above, an option is invalid, or
            the effective fields of some model vanish on every configuration of its sample, so
            that its inverse temperature is unidentifiable.

    Returns:
        InverseTemperatureFit: The estimates, their standard errors and convergence flags, of
        shape ``spins.shape[:-2]``, in ``torch.float64`` on the device of the biases.
    """
    _validate(graph, linear, quadratic, spins)
    _validate_chunk(sample_chunk)
    if max_iter < 1:
        raise ValueError(f"max_iter must be a positive int, got {max_iter}.")
    if not (0.0 < tolerance < math.inf):
        raise ValueError(f"tolerance must be a finite float > 0, got {tolerance}.")

    num_samples = spins.shape[-2]
    chunk = num_samples if sample_chunk is None else sample_chunk
    energies = _excitation_energies(graph, linear, quadratic, spins, chunk)
    batch_shape, device = energies.shape[:-2], energies.device

    num_excited = (energies > 0).sum((-2, -1))
    num_relaxed = (energies < 0).sum((-2, -1))
    unidentifiable = (num_excited == 0) & (num_relaxed == 0)
    if unidentifiable.any():
        raise ValueError(
            "The effective fields vanish on every configuration of the sample of model(s) "
            f"{unidentifiable.nonzero().tolist()} (every single-spin excitation energy is zero, "
            "as under an all-zero Hamiltonian), so the pseudolikelihood does not depend on beta "
            "and the inverse temperature is unidentifiable."
        )
    finite = (num_excited > 0) & (num_relaxed > 0)

    def derivatives(beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``NLPL'(beta)`` and ``NLPL''(beta)`` of every model, and the scores of the
        configurations, whose means are the first derivatives."""
        gradient = torch.zeros(batch_shape, dtype=torch.float64, device=device)
        curvature = torch.zeros(batch_shape, dtype=torch.float64, device=device)
        scores = torch.empty((*batch_shape, num_samples), dtype=torch.float64, device=device)
        scale = beta.unsqueeze(-1).unsqueeze(-1)
        for start in range(0, num_samples, chunk):
            eps = energies[..., start:start + chunk, :]
            z = scale * eps
            sigmoid = torch.sigmoid(z)
            chunk_scores = (eps * sigmoid).sum(-1)
            scores[..., start:start + chunk] = chunk_scores
            gradient += chunk_scores.sum(-1)
            # sigmoid(z) * sigmoid(-z) is the logistic variance, accurate where 1 - sigmoid(z)
            # would cancel
            curvature += (eps.square() * sigmoid * torch.sigmoid(-z)).sum((-2, -1))
        return gradient / num_samples, curvature / num_samples, scores

    # Newton's method from zero on the models with a finite estimate; the others are held at zero
    # while the iteration runs and receive their limits afterwards
    beta = torch.zeros(batch_shape, dtype=torch.float64, device=device)
    converged = ~finite
    active = finite.clone()
    for _ in range(max_iter):
        gradient, curvature, _ = derivatives(beta)
        # A vanishing curvature means saturated sigmoids; such a model cannot be advanced
        stuck = active & (curvature <= 0)
        active &= ~stuck
        step = torch.where(active, gradient / curvature.clamp_min(torch.finfo(torch.float64).tiny), 0.0)
        beta = beta - step
        done = active & (step.abs() <= tolerance * (1 + beta.abs()))
        converged |= done
        active &= ~done
        if not active.any():
            break

    _, curvature, scores = derivatives(beta)
    if num_samples < 2:
        beta_stderr = torch.full(batch_shape, math.nan, dtype=torch.float64, device=device)
    else:
        beta_stderr = (scores.var(-1, correction=1) / num_samples).sqrt() / curvature
    limits = torch.where(num_excited == 0, math.inf, -math.inf)
    return InverseTemperatureFit(
        beta=torch.where(finite, beta, limits),
        beta_stderr=torch.where(finite, beta_stderr, math.nan),
        converged=converged,
    )
