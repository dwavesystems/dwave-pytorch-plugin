# Mathematical Correctness Audit

**Repository:** [dwavesystems/dwave-pytorch-plugin](https://github.com/dwavesystems/dwave-pytorch-plugin)  
**Revision:** the working tree of branch `refactor` after the incorporation of the evaluation, kernel and baseline modules, on top of [`a5e7980a55c87e1ab3bd3741dd34d7ed02784189`](https://github.com/dwavesystems/dwave-pytorch-plugin/tree/a5e7980a55c87e1ab3bd3741dd34d7ed02784189) (uncommitted at the time of writing)  
**Date:** October 13, 2026 (a re-evaluation of the audit of September 28, 2026)  
**Scope:** Current mathematical implementation, tests, examples, README, API docstrings, and the new modules `dwave.plugins.torch.kernels`, `dwave.plugins.torch.evaluation` and `dwave.plugins.torch.models.baselines`.  
**What changed since the first audit:** The maximum-mean-discrepancy (MMD) implementation was replaced. The Euclidean `GaussianKernel` with a data-dependent bandwidth, `maximum_mean_discrepancy_loss(x, y, kernel)`, `MaximumMeanDiscrepancyLoss(kernel)` and `SquaredMMD(kernel)` were removed in favour of a fixed battery of Hamming-isotropic kernels (the Gaussian kernel at bandwidth $\sqrt{dim}$ and the order-$k$ Kravchuk kernels), a blocked unbiased estimator with standard errors, an `Evaluator`, and three moment-matched baselines. No recommendation of an open finding was applied, and no commits were created.

## Summary

Two of the eight findings of the first audit, the biased adaptive-bandwidth gradient (F2) and the `NaN` kernels of collapsed samples (F3), are **resolved** by the replacement: the battery's bandwidth is set by the dimension, so the leave-one-out gradient of `expectation` is exactly unbiased for every kernel of the battery (relative error $6 \times 10^{-16}$ on the first audit's exact counterexample), and identical samples give a finite, zero loss. The other six findings concern the samplers, the Ising layer's gradient contract, the Gumbel discretisation and the hidden-unit and dimod interfaces, none of which changed; **all six still reproduce**, with the same numbers as before.

The ported modules were audited by independent derivation, exact rational and enumeration checks, oracles (Owen's $T$ function, Higham's published example), finite differences and exact enumeration of sample sets, and by a second, independent auditor ([Appendix B](#appendix-b-independent-audit)), who found no mathematical defect in float64. Two new findings came out of this: F9 (low) records an intentional approximation that the documentation did not state, that at soft spins the order-kernel entries of the training loss are smooth extensions of the hard-spin estimates but not squared MMDs of any kernel, and the docstrings and the derivation now say so; F10 (medium, float32 only), found by the independent auditor, is a genuine numerical defect of the hand-written backward pass of `KravchukSums` at large dimension, **fixed** during this re-evaluation by reducing the reflection's angle to the nearest integer, with a regression test. Three documentation mismatches the auditor found were corrected.

| ID | Severity | Finding | Classification | Status |
|---|---|---|---|---|
| F1 | **High** | Metropolis has closed, non-equilibrium cycles at finite temperature | Incorrect sampling | Open, reproduces |
| F2 | ~~High~~ | Adaptive-bandwidth MMD gradients are biased | Incorrect statistical guarantee | **Resolved** by the fixed battery |
| F3 | ~~Medium~~ | Identical samples produce a zero bandwidth and NaN MMD | Numerical failure | **Resolved** by the fixed battery |
| F4 | Medium | Ising's general and composed-gradient claims exceed its finite-sample guarantees | Mathematical contract mismatch | Open, reproduces |
| F5 | Medium | Straight-through Gumbel spins do not yield exact encoder KL gradients | Mathematical contract mismatch | Open, reproduces |
| F6 | Medium | Valid conditional expectations are rejected when some hidden units are observed | Incorrect domain restriction | Open, reproduces |
| F7 | Medium | Edgeless models fail in explicit-bias sampling and layer temperature estimation | Tensor boundary failure | Open, reproduces |
| F8 | Medium | Dimod samples cannot be used directly with a float64 model | Numerical API incompatibility | Open, reproduces |
| F9 | Low | The soft-spin surrogate of the order kernels is not a positive-definite kernel | Intentional approximation, now documented | New |
| F10 | ~~Medium~~ | float32 gradients of the reflected orders $k > dim/2$ were dominated by rounding at large $dim$ | Numerical failure (float32 only) | New, **fixed** |

All findings have concrete counterexamples or reproductions, re-run on the current tree ([Appendix A](#appendix-a-reproductions)). F4 and F5 distinguish documented/intentional approximations from claims that those approximations are exact.

## Findings

### F1. Metropolis is non-ergodic even at finite, nonzero temperature (open)

**Sources:** [Metropolis acceptance rule][metropolis-update], [fixed block order][block-sweep], and [ergodicity claim][sampler-guarantees]. The sampler module is unchanged since the first audit.

The sampler always proposes a flip for every spin, and always visits blocks in the same order. Zero-energy moves are therefore accepted with probability one. This permits closed cycles even when all graph couplings are nonzero and the inverse temperature is one.

Consider a ferromagnetic triangle:

$$
E(s)=-s_0s_1-s_0s_2-s_1s_2,\qquad \beta=1.
$$

With update order $0,1,2$, the states `(-1, +1, -1)` and `(+1, -1, +1)` alternate forever. Every individual update on this trajectory has zero effective field and zero energy change. Both endpoint states have energy `+1`; neither ground state, with energy `-3`, is ever reached.

```python
import torch
from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.samplers import BlockSampler

edges = [(0, 1), (0, 2), (1, 2)]
model = GRBM(range(3), edges, quadratic={edge: -1.0 for edge in edges})
sampler = BlockSampler(
    model,
    colouring=lambda node: node,
    schedule=(1.0,),
    proposal_acceptance_criteria="Metropolis",
    initial_states=torch.tensor([[-1.0, 1.0, -1.0]]),
)
print([sampler.sample().tolist()[0] for _ in range(4)])
# [[1, -1, 1], [-1, 1, -1], [1, -1, 1], [-1, 1, -1]]
```

This also affects the default uniform initialization, not just explicitly supplied states. Its probability mass in this closed two-state class is `2/8 = 0.25`, whereas the class's true Boltzmann mass is

$$
\frac{2e^{-1}}{2e^3+6e^{-1}}
=\frac{1}{e^4+3}
=0.017361668724.
$$

The class mass cannot change under successive sweeps, so the distribution remains at least **0.232638331276** away from equilibrium in total variation, regardless of burn-in. Re-run today, 20,000 seeded chains after 30 sweeps put a mass of `0.2463` in the closed class.

**Impact:** Negative-phase statistics, conditional samples, and Ising-layer gradients can remain wrong indefinitely when using Metropolis. This is not merely slow mixing.

**Recommendation:** Use Gibbs until the Metropolis transition mechanism is made ergodic. Independent, per-spin lazy proposals with a positive probability of not proposing a flip are one remedy. Merely adding a chance of skipping an entire sweep does not connect the closed classes. Also qualify the detailed-balance wording: individual block kernels can be reversible while their fixed-order composition is only stationary, not reversible.

### F2. Adaptive-bandwidth MMD gradients are biased (resolved)

**Sources at the first audit:** [SquaredMMD's U-statistic claim][mmd-contract], [its leave-one-out implementation][mmd-loo], and [bandwidth estimation][kernel-bandwidth], all at revision `a5e7980`.

The first audit showed that `GaussianKernel(..., bandwidth=None)` estimated its bandwidth from the pooled samples, so the leave-one-out values of `SquaredMMD` held a bandwidth that still depended on the omitted sample, and the score-function gradient of `expectation` was biased by **6.5979 %** in relative vector error on an exactly enumerated two-spin example, in expectation over the sample distribution.

**Resolution.** The kernels are now the fixed Hamming-isotropic battery of [`dwave.plugins.torch.kernels`][kernels]: the Gaussian kernel $e^{-d_H / \sqrt{dim}}$ and the order-$k$ kernels $K_k(d_H) / \binom{dim}{k}$, with no data-dependent parameter. [`SquaredMMD`][squared-mmd] evaluates every profile at the real distance $(dim - \langle s, z \rangle) / 2$, so each entry is a degree-two U-statistic with a fixed kernel and the leave-one-out identity holds exactly. On the first audit's fixture (two spins, one edge, `h = [0.4, -0.2]`, `J = [0.3]`, three samples, reference `[[-1, -1], [1, -1]]`, every one of the $4^3 = 64$ ordered sample sets weighted by its Boltzmann probability, and an arbitrary combination `[1, 0.5, -0.25]` of the three profiles):

| Derivative | Exact derivative of the expected statistic | Expected implemented gradient |
|---|---:|---:|
| `h[0]` | 0.549709038184408 | 0.549709038184408 |
| `h[1]` | -1.083014101951357 | -1.083014101951357 |
| `J[0]` | 0.305471617665713 | 0.305471617665714 |

The relative vector error is **$5.9 \times 10^{-16}$** ([Appendix A](#appendix-a-reproductions)). On a three-node model with all three edges, random biases, $M = 3$ samples and random profile weights, the same enumeration gives $1.6 \times 10^{-15}$ (`tests/test_ising.py`, `TestExpectation.test_unbiased_for_means_and_u_statistics`, now runs the check for every profile). The frozen-bandwidth input-gradient caveat of the first audit no longer applies: there is no bandwidth to freeze, and the gradient with respect to a soft reference sample is the exact derivative of the surrogate, which `torch.autograd.gradcheck` confirms (`tests/test_functional.py`, `tests/test_evaluation.py`).

### F3. Collapsed samples produce NaN kernels, losses, and prior gradients (resolved)

**Source at the first audit:** [Bandwidth computation and division][kernel-bandwidth] at revision `a5e7980`.

All pairwise distances of identical items are zero, the adaptive bandwidth was zero, and the kernel computed `exp(-0 / 0)`.

**Resolution.** With the fixed battery, identical samples give the constant Gram $f(0) = 1$ for every profile and the estimate is exactly zero:

```python
import torch
from dwave.plugins.torch.nn.functional import maximum_mean_discrepancy_loss

x = torch.ones(3, 2)
print(maximum_mean_discrepancy_loss(x, x))  # tensor([0., 0., 0.])
```

The same configuration passed through `expectation(..., SquaredMMD(), ...)` gives a finite (zero) prior-bias gradient. There is no explicit bandwidth to validate.

### F4. Ising's unbiased-gradient contract is too broad (open)

**Sources:** [Public Ising description][ising-contract], [nonlinear-loss claim][ising-loss-claim], and [the more precise functional contract][expectation-contract]. The layer and the estimator are unchanged since the first audit.

There are two distinct finite-sample limitations.

**Arbitrary symmetric statistics target one sample fewer.** The functional documentation correctly states that, for a general symmetric statistic, its expected gradient is

$$
\mathbb E[\widehat g_M]
=\nabla_\theta\mathbb E[F_{M-1}],
$$

even though the forward value is $F_M$. These targets agree for sample means and fixed-kernel U-statistics, not for every symmetric statistic accepted by `Ising`.

For a single spin with `h=0.4`, `M=3`, and `F(spins)=exp(mean(spins))`, exact enumeration gives (re-verified today):

| Quantity | Derivative with respect to `h` |
|---|---:|
| True expected three-sample statistic | -0.749070350395 |
| Expected implemented gradient | -0.828992714049 |
| True expected two-sample statistic | -0.828992714049 |

This behavior is deliberately tested in the repository (`tests/test_ising.py`, `test_symmetric_statistics_target_one_sample_fewer`). The defect is the broader public claim that the layer differentiates the expectation of its returned statistic for an arbitrary symmetric `F`, not the implemented $M-1$ identity itself.

**An unbiased local mean gradient does not make a nonlinear downstream loss unbiased.** With one spin, write $\mu=-\tanh h$, use the default `Mean`, and apply $L(y)=y^2/2$ outside the layer. The true target derivative is

$$
\frac{d}{dh}\frac{\mu^2}{2}=-\mu(1-\mu^2).
$$

Exact enumeration of the implemented forward/backward pair gives

$$
\mathbb E[\widehat{\nabla_h L}]
=\frac{M-2}{M}\left[-\mu(1-\mu^2)\right].
$$

For `h=0.4` (re-verified today):

| Samples per model | True derivative of the loss of the mean | Expected implemented gradient |
|---:|---:|---:|
| 2 | 0.325099068837 | **0** |
| 3 | 0.325099068837 | 0.108366356279 |
| 5 | 0.325099068837 | 0.195059441302 |

For two samples the gradient vanishes for every possible sample pair: equal spins have zero sample variance, while unequal spins have zero sample mean. The upstream derivative and the covariance estimator are correlated.

The same distinction applies to the new `SquaredMMD`: the layer's gradient is unbiased for the expectation of each profile's estimate, which is what training on `loss[..., 0]` uses; a nonlinear function of the profiles (a ratio to the data's weight, say) placed outside the layer is a plug-in and inherits the bias above.

**Recommendation:** Align the README/layer contract with the narrower functional contract and distinguish local unbiasedness from end-to-end nonlinear-loss bias. If the desired objective is a squared population mean, an appropriate pairwise U-statistic inside `expectation`, with enough samples for leave-one-out evaluation, can target it without this plug-in bias. A general nonlinear objective requires its own estimator analysis.

### F5. The pseudo-KL gradient guarantee does not hold with straight-through Gumbel spins (open)

**Sources:** [Pseudo-KL guarantee][pseudo-kl-contract], [Gumbel discretization][gumbel-code], and [the exact-KL test][kl-test]. Unchanged since the first audit apart from their position in the file.

The entropy calculation is correct. The problem is claiming the same gradient as the true KL when the energy term is differentiated through hard straight-through Gumbel-softmax at finite relaxation temperature.

Consider one encoder logit $\ell=0$ and an Ising prior $E(s)=s$. The true encoder KL derivative is `0.5`: the entropy derivative vanishes, and

$$
\frac{d}{d\ell}\mathbb E_q[s]
=\frac{d}{d\ell}\tanh(\ell/2)
=\frac12
\quad\text{at }\ell=0.
$$

At `tau=1`, the relaxed positive-spin probability at this logit is uniform on `(0, 1)`, so the expected straight-through derivative is

$$
2\mathbb E[U(1-U)]=\frac13,
$$

not `1/2`. At the default `tau=1/7`, numerical integration of the exact logistic-noise distribution gives **0.491970726293** (re-verified today on a grid of two million points), still not `0.5`. Hard forward spins have the correct Bernoulli marginal; it is the backward estimator that is biased.

The exact-KL test substitutes `tanh(logits / 2)` for sampled spins, which verifies a different, exact path and does not establish the guarantee for the example's straight-through path.

**Recommendation:** Describe Gumbel-based encoder gradients as straight-through approximations. For a fully factorized encoder and pairwise Ising prior, evaluating the KL energy at `tanh(logits / 2)` gives the exact expected energy and encoder KL gradient; sampled spins can still be used for reconstruction. Exact prior gradients additionally require the correct negative-phase distribution.

### F6. Partially observed hidden units are incorrectly rejected (open)

**Source:** [`_unknown_are_adjacent`][unknown-mask]. The model module is unchanged since the first audit.

When no visible entry is unknown, the helper returns the global `connected_hidden` flag. That is valid for the all-hidden-unknown pattern produced by `pad_visible`, but not for the method's advertised arbitrary observation masks.

```python
import torch
from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM

model = GRBM(
    [0, 1, 2], [(1, 2)], hidden_nodes=[1, 2],
    linear={1: 0.3}, quadratic={(1, 2): 0.7},
)
model.conditional_expectation(torch.tensor([[1.0, float("nan"), 1.0]]))
# Raises ValueError, although only one spin is unknown.
```

The correct missing-spin expectation is `-tanh(0.3 + 0.7) = -0.761594155956`. A fully observed input to the same model is also incorrectly rejected (re-verified today).

**Recommendation:** Test adjacency of the actual per-row unknown mask. A topology-based shortcut can safely establish that conflicts are impossible, but the presence of hidden-hidden edges does not prove that both endpoints are currently unknown.

### F7. Edgeless models fail in two otherwise valid APIs (open)

**Sources:** [`DimodSampler.sample_biases`][dimod-batch] and [`Ising.estimate_betas`][ising-betas]. Both unchanged since the first audit.

Both flatten quadratic biases with `reshape(-1, n_edges)`. When `n_edges == 0`, PyTorch cannot infer the unspecified dimension of `reshape(-1, 0)`. Independent-spin models are valid Ising models and are explicitly supported elsewhere in the graph/model core.

```python
import torch
from dimod import ExactSolver
from dwave.plugins.torch.graph import GraphIndex
from dwave.plugins.torch.nn import Ising
from dwave.plugins.torch.samplers import DimodSampler

h, J = torch.tensor([0.4]), torch.empty(0)
DimodSampler(GraphIndex([0], []), ExactSolver()).sample_biases(h, J)
# RuntimeError: cannot reshape tensor of 0 elements into shape [-1, 0] ...

# Independently, this valid call also fails:
Ising([0], []).estimate_betas(h, J, torch.tensor([[-1.], [-1.], [-1.], [1.]]))
# ValueError claiming inconsistent shapes; the underlying failure is the same reshape.
```

Both failures reproduce today.

**Recommendation:** Calculate the number of models from the batch shape and use an explicit `(number_of_models, n_edges)` reshape. Cover both unbatched and batched zero-edge models.

### F8. Dimod output dtype breaks float64 inference and training (open)

**Sources:** [Sample-set conversion][sampleset-conversion], [unconditional sampling][dimod-unconditional], and [explicit-bias sampling][dimod-batch]. Unchanged since the first audit.

The conversion utility intentionally creates `float32` samples. The unconditional and explicit-bias sampler paths subsequently copy only the device, not the model/bias dtype. Consequently, a valid double-precision model cannot evaluate the energies of its own returned samples.

```python
import torch
from dimod import ExactSolver
from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM
from dwave.plugins.torch.samplers import DimodSampler

model = GRBM([0, 1], [(0, 1)], quadratic={(0, 1): 0.4}).double()
samples = DimodSampler(model, ExactSolver()).sample()
print(model.linear.dtype, samples.dtype)  # torch.float64 torch.float32
model(samples)
# RuntimeError: expected m1 and m2 to have the same dtype ...
```

The same mismatch prevents `quasi_objective` training and affects float64 `sample_biases` inputs. The conditional path does not have the same problem because it assembles its result in the model dtype. Reproduces today. Note that the new `Evaluator` casts nothing by design, so a float32 dimod sample must be cast by the caller before it is scored against float64 splits; that is documented, not a defect.

**Recommendation:** Preserve the utility's documented return type if desired, but cast sampler outputs to both the device and dtype of the owning model or supplied linear biases.

### F9. The soft-spin surrogate of the order kernels is not a positive-definite kernel (new, low)

**Sources:** [the real-distance extension in `maximum_mean_discrepancy_loss`][mmd-functional] and [`KravchukSums`][kravchuk-sums].

Every profile of the battery is evaluated at the real distance $d = (dim - \langle x, y \rangle) / 2$, which is the Hamming distance at hard spins and a smooth function of soft spins in $[-1, 1]$ (a `tanh` output, or straight-through Gumbel spins). The docstrings describe the result at soft spins as "a smooth surrogate whose minimizers at the corners of the cube are the hard-spin ones", which is correct, but the first version of the port did not say what kind of object each entry is between the corners.

The Gaussian row's extension is $e^{-dim / (2\sqrt{dim})}\, e^{\langle x, y \rangle / (2 \sqrt{dim})}$, the exponential of an inner product, hence a positive-definite kernel on soft points: entry 0 of a soft-spin loss is a squared MMD of the soft points. The order-$k$ rows' extensions are polynomials in the inner product and are **not** positive definite off the corners: on 60 random `tanh` points in 8 dimensions the Gram of the order-2 extension has minimum eigenvalue **-7.15**, the order-3 extension **-1.90** and the order-4 extension **-0.95**, while every hard-spin Gram of orders 1 to 8 is positive semidefinite to $10^{-14}$ ([Appendix A](#appendix-a-reproductions)). (The positive-definite extension of the order-$k$ kernel is the elementary symmetric polynomial $e_k(x_1 y_1, \dots, x_{dim} y_{dim})$, which depends on the coordinate-wise products and not on the inner product alone; the code's choice keeps the memory-light recurrence.) At soft spins the order entries are therefore smooth extensions of the hard-spin estimates, not squared MMDs of the soft points, and their value can be negative for reasons other than estimation noise.

**Impact:** None at hard spins, where every entry is exact. For training on soft spins the headline is a proper MMD; an order entry used alone as a loss is a surrogate without the MMD's non-negativity. The docstring of `maximum_mean_discrepancy_loss` and Step 1 of the derivation in `docs/evaluation.md` now state this.

**Recommendation (applied):** documentation only.

### F10. float32 gradients of the reflected orders were dominated by rounding at large dimension (new, fixed)

**Source:** [`KravchukSums.backward`][kravchuk-sums]; found by the independent auditor ([Appendix B](#appendix-b-independent-audit)).

The rows above $dim / 2$ are the reflections $\cos(\pi d)\, R_k(d)$ of the swept rows, whose derivative is $\cos(\pi d)\, R_k'(d) - \pi \sin(\pi d)\, R_k(d)$. At an integer distance the second term is exactly zero, but the port computed `torch.sin(math.pi * d)` directly, and in float32 the product $\pi d$ carries an angle error of about $8.7 \times 10^{-8}\, d$, so the sine was not zero and a spurious term of size about $\pi d \cdot 8.7 \times 10^{-8} \lvert R_k \rvert$, growing linearly in $d$, entered the gradient of every reflected row at hard spins. At $dim = 4800$ and $d = 4704$ the derivative of row $dim - 1$ (exactly $(-1)^d (-2 / dim) = -4.17 \times 10^{-4}$) came out as $+2.27 \times 10^{-3}$: the wrong sign and five times the magnitude. Per row, the maximum absolute error of the float32 gradient against float64 was $2.7 \times 10^{-3}$ for rows $4797$ to $4799$, against true magnitudes of $4 \times 10^{-4}$ to $1.3 \times 10^{-3}$. The swept rows were unaffected (relative errors $10^{-7}$ to $10^{-6}$), as was the Gaussian headline (relative error $2 \times 10^{-6}$), so a float32 loss on the Gaussian headline or on a calibrated alpha row (which puts $10^{-3}$ of its mass on the unmeasurable high orders) was contaminated by about one percent; a float32 loss with user-supplied weights on the top orders was wrong. Float64 was exact throughout.

**Fix (applied):** `KravchukSums` now reduces the angle to the nearest integer, $\cos(\pi d) = (-1)^m \cos(\pi (d - m))$ and $\sin(\pi d) = (-1)^m \sin(\pi (d - m))$ with $m = \mathrm{round}(d)$, in both passes. The difference $d - m$ is exact in floating point, so at an integer distance the sine is exactly zero and the cosine exactly $\pm 1$ in every dtype, and at a real distance the values are unchanged up to rounding. After the fix the float32 gradients of rows $4797$ to $4800$ at integer distances are within $1.5 \times 10^{-10}$ of float64 (the parity row's derivative is exactly $0$), `gradcheck` at real distances still passes, and the backward still equals autograd through the unrolled recurrence to $10^{-13}$ in float64 (`tests/test_kernels.py`, `test_float32_gradient_of_reflected_rows_at_integer_distances` and its neighbours).

## Lower-severity documentation corrections

| Topic | Source | Correction | Status |
|---|---|---|---|
| Negative-log-likelihood sign | [`sufficient_statistics` docstring][statistics-sign] | It says model statistics minus data statistics. For the implemented $p(s)\propto e^{-E(s)}$, the NLL gradient is **data minus model**. The implementation and `quasi_objective` documentation use the correct sign. | Open |
| Initialization versus critical temperature | [Initialization interpretation][initialization-contract] | The degree-normalized Gaussian scale is implemented as written, but an SK-inspired normalization is not a universal calibration of an arbitrary graph's actual critical temperature. In particular, a nearest-neighbor one-dimensional Ising family has no finite-temperature transition. Describe `init_temperature` as a mean-field/reference normalization rather than guaranteeing that `T0=1` is critical for every supported graph. | Open |
| float32 accuracy of the profile table | [`kernels.py`][kernels] module and `kernel_profiles` docstrings, `docs/evaluation.md` | The claim "float32 to about 5e-7" holds only up to $dim \approx 200$: the measured maximum error of the float32 table against the exact rationals is $2.3 \times 10^{-7}$ at $dim = 100$, $4.8 \times 10^{-7}$ at 200, $1.6 \times 10^{-6}$ at 1000 and $1.6 \times 10^{-5}$ at 4800 (the float64 claim of $2 \times 10^{-14}$ holds at every dimension, $1.2 \times 10^{-14}$ at 4800). | Corrected: the docstrings and the document now give both figures |
| A comment in `KravchukSums.forward` | [`kernels.py`][kravchuk-sums] | `if dim - 1 > half` was annotated "false at dim = 1, 2, 3 only"; at $dim = 3$ it is true (row 2 is written as the reflection of row 1), and the condition is reached and false at $dim = 2$ only. The behaviour was correct. | Corrected |
| Precision of the tetrachoric solve | `docs/baselines.md`, *The solve* and *Limitations* | The document claimed the error allowed in $\rho$ is below $10^{-8}$ for any table whose smallest smoothed cell is at least a few times $10^{-7}$. Measured on 20,000 random round trips, the maximum error is $8.2 \times 10^{-7}$ for cells in $[10^{-7}, 10^{-6})$, $9.1 \times 10^{-8}$ in $[10^{-6}, 10^{-5})$, $5.7 \times 10^{-9}$ in $[10^{-5}, 10^{-4})$ and at most $2.2 \times 10^{-10}$ for cells above $10^{-3}$; the conclusion that this is negligible against sampling error stands. | Corrected: the document now states the measured scale |
| `expectation` with `SquaredMMD` needs three samples | [`expectation`][expectation-contract] | `expectation` accepts two samples per model, but `SquaredMMD.leave_one_out` needs three (two remaining samples for its within-sample term), so the combination raises from the statistic; this was documented in `SquaredMMD` only. | Corrected: `expectation`'s docstring now says so |

## Important mathematical assumptions and intentional approximations

**Clipping and conditioning do not commute.** The documented Dimod implementation conditions the original model and then clips the reduced model; it does not condition the clipped joint model. This is intentional and tested, but it prevents interpreting all outputs as conditionals of one common clipped distribution. For example, let `h=[0, -2]`, `J01=3`, and clip quadratic biases to `[-1, 1]`, with no linear clipping. Conditioning the submitted joint on `s0=+1` gives a free-spin field of `-1`; the conditional submission instead has field `+1`. The corresponding conditional means are `+0.761594` and `-0.761594`. Exact learning-gradient claims need the approximation to be inactive or explicitly accounted for.

**Sampling assumptions matter independently of the formulas.** The score/covariance arguments assume independent samples from the specified Boltzmann distribution at unit inverse temperature. A finite number of Gibbs sweeps from a fresh state is not automatically an equilibrium draw; persistent contrastive divergence is an approximation; and arbitrary dimod optimizers or QPU reads are not automatically Boltzmann samples. The audit did not establish mixing-time bounds or validate a physical QPU's temperature calibration.

**Negative scalar values are not automatically errors.** The quasi-objective is not a numerical NLL, and an unbiased finite-sample estimate of squared MMD can be negative. Neither property is itself a defect. The evaluator clips nothing, by design.

**The evaluator's guarantees are conditional on independent rows.** Unbiasedness of `blocked_scores` and the validity of its standard errors need IID rows within each sample and independence between samples; rows are consumed in storage order, so a structured order (sorted by label, the reads of one hardware submission, consecutive states of one chain) inflates every score unless permuted once. Shuffling removes that blocking bias, not the dependence of correlated rows. The example `examples/evaluation.py` draws one chain per row.

**The battery's exactness is bounded by the dtype.** The histogram estimator counts Hamming distances exactly only while $dim$ and $block\_size^2$ stay below $2^{24}$ in float32 ($2^{53}$ in float64); `_check_exact_counting` refuses larger blocks. The float64 profile tables are within $2 \times 10^{-14}$ of the exact rationals, which makes the order rows single Walsh bands up to that leakage: at central orders past $dim \approx 100$ a reading is exact in absolute terms but no longer attributable to its order alone (pinned in `tests/test_kernels.py`, `TestExactTables`).

**The tetrachoric fit is exact for the smoothed tables, within its cap.** `DichotomizedGaussian.fit` smooths every $2\times2$ table with a pseudo-count of $1/4$ per cell (a bias of order $1/N$), caps latent correlations at $|\rho| \le 0.9999$, and projects an infeasible pairwise matrix onto the nearest correlation matrix with an eigenvalue floor of $10^{-4}$, after which the moved pairs are no longer reproduced; `fit_info` reports all three. The family has no free parameters above order two.

**The soft-spin loss is a surrogate (F9).** At hard spins every entry is exact; between the corners the headline is a squared MMD and the order entries are smooth extensions only.

## Re-evaluation of the ported modules

The new modules were checked by independent derivation and computation, beyond the unit tests that ship with them (`tests/test_kernels.py`, `tests/test_evaluation.py`, `tests/test_baselines.py`, and the rewritten `tests/test_functional.py`, `tests/test_loss.py` and `tests/test_ising.py`).

| Area | Check | Result |
|---|---|---|
| Kravchuk recurrence and reflection | Every row of `kernel_profiles(n)` against exact big-integer arithmetic for $n \in \{7, 8, 50, 100, 200\}$; the reflection $R_{n-k}(d) = (-1)^d R_k(d)$ against exact rows at $n = 4800$; dyadic rows stored exactly | within $2 \times 10^{-13}$; reflections exact; dyadic rows bitwise exact |
| Spectra | Walsh eigenvalues by enumeration of $\{-1, +1\}^8$: every order row a single band $1 / \binom{8}{k}$, the Gaussian row geometric with ratio $\tanh(1/(2\sqrt 8))$, the alpha row $\alpha_k / \binom{8}{k}$ with no order-0 component | within $10^{-10}$ |
| Uniform kernel mean | against the order-0 eigenvalue by enumeration and the closed form $((1 + e^{-1/\sqrt n}) / 2)^n$ | within $10^{-14}$ |
| `KravchukSums` | equals the table at integer distances; `gradcheck` at random real distances for $n \in \{8, 9\}$; backward equals autograd through the unrolled recurrence for $n \in \{1, 2, 8, 9, 24, 25\}$; the $(\dots, E)$ form equals the 1-D form row by row with its gradient | $10^{-14}$; passes; $10^{-13}$; $10^{-13}$ |
| `maximum_mean_discrepancy_loss` | against a naive table lookup at Hamming distances on hard spins (unequal sample sizes, $dim = 7$); symmetry in its arguments; blocks equal the mean of per-block estimates; `gradcheck` on soft inputs with and without `alpha`; float32 against float64 on a 256-row batch at $dim = 48$ | $10^{-12}$; exact; exact; passes; $4.6 \times 10^{-9}$ |
| `SquaredMMD` | equals the functional profile by profile, for batches of sets and per-set references; `leave_one_out` equals the statistic on every explicit subset of all samples but one; gradient to a soft reference equals the functional's; exact-enumeration unbiasedness of the score-function gradient on a two-node and a three-node model | $10^{-12}$; $10^{-12}$; $10^{-12}$; $5.9 \times 10^{-16}$ and $1.6 \times 10^{-15}$ |
| `blocked_scores` | against per-entry table lookups for sampled and analytic-uniform arms; per-block variances under uniform data against the closed forms $4/(r(r-1)) + 4/r^2$ and $2/(r(r-1))$ per $1/\binom{n}{k}$; the uniform arm against the exact $\sum_{S \ne \emptyset} \lambda_{|S|} W_P[S]^2$ of a random law on $n = 8$ by Monte Carlo; a point mass and $x$ against $-x$ exactly | $10^{-12}$; within 25 %; within 5 standard errors; exact (1 and 4) |
| `Evaluator` | bitwise equal to `blocked_scores` on the same rows; the calibration rule recomputed by hand from the returned record; `compute_loss` against `blocked_scores` on hard spins; end-to-end `gradcheck` on a soft batch; the alpha row by linearity | exact; exact; $10^{-8}$; passes; $10^{-12}$ |
| Bivariate normal | `_BivariateNormal.cdf` against Owen's $T$ (SciPy) on 20,000 random points with $\lvert\rho\rvert \le 0.9999$ and on the boundary layer $a \approx \pm b$ at the cap; the closed forms at zero thresholds and zero correlation; the complement identity; monotonicity in $\rho$; Plackett's identity by central difference | $3 \times 10^{-16}$ (bound $10^{-13}$); $9 \times 10^{-15}$ (bound $10^{-12}$); $10^{-14}$; $10^{-15}$; exact; $10^{-8}$ |
| Tetrachoric inversion | round trips on 5,000 random tables (residual, and $\rho$ where the density resolves it); the sine closed form at zero thresholds; clamping at the cap | $\le 10^{-12}$ and $10^{-8}$; $10^{-12}$; as documented |
| Nearest correlation | Higham's 2002 example $[[1,1,0],[1,1,1],[0,1,1]]$ (off-diagonals 0.7607 and 0.1573); a random indefinite matrix projected to a positive-definite unit-diagonal matrix nearer than the identity | within $10^{-3}$ (the eigenvalue floor); passes |
| `DichotomizedGaussian` | recovery of a known latent $(\mu, R)$ from 65,536 rows; samples reproduce the fitted means and correlations; the four-spin two-plus-states law forces the projection to the symmetric closed form $-(1 - 10^{-4}) / 3$; a degenerate column warns and decouples | within 0.06 / 0.08 (eight standard errors); within 0.05 / 0.06; within $10^{-3}$; as documented |
| Baselines | `IndependentBernoulli.fit` sets the exact frequencies; `UniformBernoulli` draws fair coins; constant columns stay constant | exact; within 7 standard errors; exact |
| Edge cases | $dim \in \{1, 2, 3\}$; $M = 2$ samples and $R = 2$ reference items; batch dimensions; float32; soft inputs exactly at $\pm 1$; the empty edge set of `GraphIndex([0], [])` in the enumeration of F4 | pass |
| float32 gradients at hard spins (F10) | `KravchukSums` gradients of rows $4797$ to $4800$ at integer distances, $dim = 4800$, float32 against float64, before and after the fix | before: errors up to $2.7 \times 10^{-3}$ against values of $4 \times 10^{-4}$ to $1.3 \times 10^{-3}$; after: within $1.5 \times 10^{-10}$ |
| Cost | `KravchukSums` at $dim = 5000$, $10^4$ distances, CPU with 4 threads | 0.08 s forward, 0.29 s forward and backward |

**Independent auditor.** A second auditor, given the code and the checks above as a brief but not their results, audited the same modules independently; its report is summarised in [Appendix B](#appendix-b-independent-audit).

## What was checked successfully

| Area | Result under the stated assumptions |
|---|---|
| Graph energy and parameter layout | $E=h^\mathsf{T}s+\sum_{(i,j)}J_{ij}s_i s_j$, with each edge counted once; consistent with dimod and sufficient statistics. |
| Effective fields and local update signs | $h_i^{\mathrm{eff}}=h_i+\sum_jJ_{ij}s_j$, conditional mean $-\tanh h_i^{\mathrm{eff}}$, Gibbs probability $\sigma(-2\beta h_i^{\mathrm{eff}})$, and flip energy $-2s_i h_i^{\mathrm{eff}}$ are consistent. |
| Hidden-unit likelihood gradients | Exact enumeration on a three-spin fixture, with an exact 48-row negative-phase distribution, matched marginalized NLL parameter gradients within `3.4e-16`; observed-input gradients also matched. |
| Fixed-kernel MMD | Off-diagonal normalization, cross-term factor of two, leave-one-out formula, and exact expected parameter gradients agree, now for every kernel of the battery (F2). |
| Factorized-encoder entropy and expected-energy KL | Summing entropy over spins and averaging over data is correct. Using mean spins and an exact negative phase matched enumerated KL gradients within `3.4e-16`. |
| Dimod conversions | Variable order, spin vartype, scale-before-clip behavior, and expansion of `num_occurrences` are consistent in the covered cases. |
| Temperature units | For one spin with `h=0.5` and three negative versus one positive observations, estimated inverse temperature was `log(3) = 1.0986122886681098`, matching the analytic pseudolikelihood optimum. |
| Kernel battery, estimator, evaluator and baselines | See the table of the previous section. |

## Method and limitations

The audit combined source inspection, independent derivations, exact state/sample-set enumeration, exact rational arithmetic on the shipped tables, external oracles (SciPy's Owen's $T$ function, Higham's published example), finite-difference and unrolled-autograd checks of the hand-written backward pass, Monte Carlo checks of unbiasedness and of closed-form variances, deterministic edge-case reproductions, and the repository's test suite. Main numerical counterexamples were evaluated in CPU float64. The Metropolis failure was established by a closed-class argument as well as actual sampler execution.

The test suite completed successfully:

```text
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MPLBACKEND=Agg \
  python -B -m unittest discover -s tests -q

Ran 249 tests
OK
```

The tested environment was Python `3.14.4`, PyTorch `2.11.0+cu130`, NumPy `2.4.4`, SciPy `1.17.1`, matplotlib `3.10.8`, dimod `0.12.22`, dwave-system `1.34.0`, and NetworkX `3.6.1`. CUDA was available, so the suite's CUDA-gated tests ran. **This was not the pinned PyTorch `2.9.1` environment in `requirements.txt`**; no dependencies were changed for the audit. Exact algebraic counterexamples do not depend on claiming validation of that pinned environment.

Imports were verified to resolve to this checkout, not the generated `build` copy. Generated packaging/build artifacts were not treated as separate implementations. No QPU was contacted. `ExactSolver` was used for enumeration or adapter reproduction, not misinterpreted as an unweighted finite-temperature Boltzmann sampler. This is an evidence-based audit, not a formal proof covering every graph, dtype, hardware backend, or custom statistic. The validation of the battery's Gaussian bandwidth on correlated data at large dimension rests on experiments outside this repository, as `docs/evaluation.md` states; this audit verified the battery's exactness and unbiasedness, not that choice.

## Appendix A: reproductions

Run from the repository root with bytecode writing disabled (`PYTHONPATH=. python -B script.py`). The first script re-runs every finding's reproduction; its output on the current tree is quoted inline above. The second is the exact check behind the resolution of F2: it enumerates every ordered set of three samples, weights each by its exact Boltzmann probability, and compares the true gradient of an arbitrary combination of the three profiles with the expectation of the gradient returned by the plugin.

```python
import itertools
import torch

from dwave.plugins.torch.graph import GraphIndex
from dwave.plugins.torch.nn import SquaredMMD
from dwave.plugins.torch.nn.functional import expectation

torch.set_default_dtype(torch.float64)
graph = GraphIndex([0, 1], [(0, 1)])
states = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=2)))
indices = torch.tensor(list(itertools.product(range(4), repeat=3)))
sample_sets = states[indices]
reference = torch.tensor([[-1.0, -1.0], [1.0, -1.0]])
weights = torch.tensor([1.0, 0.5, -0.25])  # the headline and the two order kernels

h = torch.tensor([0.4, -0.2], requires_grad=True)
J = torch.tensor([0.3], requires_grad=True)
statistic = SquaredMMD()
probabilities = torch.softmax(-graph.energy(states, h, J), 0)[indices].prod(-1)

values = (statistic(sample_sets, reference) @ weights).detach()
exact = torch.cat(torch.autograd.grad((probabilities * values).sum(), (h, J)))

values_with_gradient = expectation(graph, h, J, sample_sets, statistic, reference) @ weights
estimated = torch.cat(torch.autograd.grad(
    (probabilities.detach() * values_with_gradient).sum(), (h, J)
))

print("exact:", exact.tolist())
print("expected implemented gradient:", estimated.tolist())
print("relative error:", ((estimated - exact).norm() / exact.norm()).item())
# relative error: 5.9e-16
```

The definiteness check of F9:

```python
import math, torch
from dwave.plugins.torch.kernels import KravchukSums

torch.manual_seed(0)
n = 8
z = torch.tanh(torch.randn(60, n, dtype=torch.float64))
d = (n - z @ z.mT) / 2
rows = KravchukSums.apply(d.reshape(-1, 1), n).reshape(n + 1, 60, 60)  # per-pair values
print(torch.linalg.eigvalsh(torch.exp(-d / math.sqrt(n))).min())  # +1.1e-3: positive definite
print([float(torch.linalg.eigvalsh(rows[k]).min()) for k in (1, 2, 3, 4)])
# [-0.0, -7.15, -1.90, -0.95]: the order surrogates are indefinite on soft points
```

## Appendix B: independent audit

A second auditor, running as an independent agent with its own scripts (kept outside the repository), was given the five modules and a list of properties to derive and test, and not the results of the checks above. Its conclusion, quoted in substance: no mathematical defect in float64; every derivation it made independently (the Kravchuk recurrence and reflection, the single-band Walsh spectra, the geometric Gaussian spectrum, U-statistic unbiasedness, the leave-one-out closed forms, the score-function gradient identity, the Drezner-Wesolowsky integral, Plackett's identity, Sheppard's theorem, the Jeffreys smoothing, and the Dykstra-corrected Higham projection) is reproduced by the code to rounding. It reported one genuine float32 defect (F10 above, since fixed) and three documentation mismatches (the float32 table accuracy, a comment, and the precision claim of the tetrachoric solve, all corrected above), plus three observations that are not defects: `expectation` with `SquaredMMD` needs three samples (now documented); the relative error of `_BivariateNormal.cdf` reaches $1.3 \times 10^{-7}$ where the probability itself is $2.6 \times 10^{-10}$ (cancellation in $\Phi(a)\Phi(b)$ plus the correction), irrelevant at the absolute $10^{-12}$ tolerance of the solve; and the straight-through spins of `gumbel_spins` were exactly $\pm 1$ in 409,600 float32 and 16,384 float64 draws, so the comment allowing a rounding error away from $\{0, 1\}$ is over-cautious.

Its verified numbers, in addition to those of the table above:

- Profile table against exact rationals for $n \le 100$: float64 $5.6 \times 10^{-16}$, float32 $2.3 \times 10^{-7}$; against the explicit Kravchuk formula at $n = 200, 1000, 4800$: $4.2 \times 10^{-16}$, $1.5 \times 10^{-15}$, $1.2 \times 10^{-14}$; the orthogonality $\sum_d \binom{n}{d} 2^{-n} R_j R_k = \delta_{jk} / \binom{n}{k}$ at $n = 4800$ to $1.6 \times 10^{-12}$ relative on the diagonal and $6.6 \times 10^{-16}$ off it; `_log_binomials` to $5 \times 10^{-13}$ relative at $n = 4800$ and $n = 20000$.
- Walsh spectra by enumeration for $n = 1..9$: single bands to $1.3 \times 10^{-15}$; the Gaussian row's eigenvalues $((1 + e^{-1/\sqrt n}) / 2)^n \tanh(1 / (2 \sqrt n))^{|S|}$ to $1.5 \times 10^{-15}$.
- `KravchukSums`: `gradcheck` at real and integer distances for $n \in \{1..13, 16, 17, 31, 32, 64, 101\}$ and shapes `(6,)` and `(2, 3, 4)`; the hand-written vector-Jacobian product against autograd through an independent unrolled implementation to $1.9 \times 10^{-15}$ relative; CUDA against CPU to $1.4 \times 10^{-14}$.
- `maximum_mean_discrepancy_loss` against a naive table U-statistic over dimensions 1 to 30 and six sample-size pairs: $4.4 \times 10^{-16}$ (float64) and $2.1 \times 10^{-7}$ (float32); against an independent plain-autograd implementation at soft spins: $4.4 \times 10^{-16}$; gradients at inputs exactly $\pm 1$ finite and equal to a one-sided inward finite difference.
- `SquaredMMD`: permutation symmetry $2.8 \times 10^{-16}$; `leave_one_out` against explicit drop-one evaluation $5.6 \times 10^{-16}$ (float64) and $8.9 \times 10^{-8}$ (float32) over dimensions 1 to 11, $M = 3..9$, $R = 2..6$, hard and soft references, with and without weights.
- `expectation` unbiasedness by exact enumeration: a triangle with $M = 3$ (512 sets) $5.8 \times 10^{-16}$ with a hard reference and $2.0 \times 10^{-15}$ with a soft one; a path with $M = 4$ and weights (4096 sets) $3.4 \times 10^{-15}$; two nodes with $M = 5$: $1.9 \times 10^{-14}$; an empty edge set $1.0 \times 10^{-15}$; `Mean` equals minus the unbiased sample covariance to $2.8 \times 10^{-16}$; a non-U-statistic is unbiased for $\nabla \mathbb E[F_{M-1}]$ ($2.4 \times 10^{-15}$) and not for $\nabla \mathbb E[F_M]$ ($0.124$), as F4 states.
- `blocked_scores` against per-entry lookups: $6.1 \times 10^{-16}$ (float64), $1.5 \times 10^{-7}$ (float32); histograms exact at $dim = 4800$ in float32 and $5000$ in float64; the documented miscount of a 4097-row float32 block reproduced exactly (16,785,408 against 16,785,409) and refused by `_check_exact_counting`; the uniform arm's population value equals $\sum_{S \ne \emptyset} \lambda_{|S|} W_P[S]^2$ to $1.4 \times 10^{-17}$, its $r = 2$ block estimator matches exact enumeration to $7.6 \times 10^{-17}$, and ten Monte Carlo runs of 8,192 blocks of 256 rows at $n = 6$ give standardized errors of mean $0.30 \pm 0.32$ and root mean square $1.06$.
- `Evaluator`: `calibrate` recomputed independently (Fourier weights, standard errors, floor and gate identical; the weights sum to 1 to $10^{-12}$; the flat fallback warns).
- `_BivariateNormal.cdf` against Owen's $T$ on 2,160 points with $\lvert a \rvert, \lvert b \rvert \le 4$ and $\lvert \rho \rvert \le 0.9999$: $4.4 \times 10^{-16}$; against adaptive quadrature at thresholds $\pm 5$ and $\pm 6$: $4.4 \times 10^{-16}$; beyond the cap at $\rho = 0.99999$: $6.3 \times 10^{-10}$, as documented. `pdf` against SciPy to $5 \times 10^{-10}$ relative. `invert`: residual at most $10^{-12}$ and converged on all 24,000 round trips, $\lvert \hat\rho - \rho \rvert \le 3.6 \times 10^{-11}$ where the smallest cell is at least $10^{-2}$, $9.3 \times 10^{-11}$ against a bracketing root finder on Owen's $T$; 100,000 inversions in 1.9 s.
- The Jeffreys smoothing on a constructed ten-row table reproduces $\mu = \Phi^{-1}(7.5 / 11), \Phi^{-1}(4.5 / 11)$ exactly and the smoothed joint cell to $7.2 \times 10^{-16}$. `_nearest_correlation` on Higham's example: $0.760690$ and $0.157298$ with a zero floor, identical to a 30-restart brute-force minimization of the Frobenius distance ($0.527790$); a random indefinite $6 \times 6$ matrix matches brute force to $1.1 \times 10^{-9}$. A fitted `DichotomizedGaussian` (6 spins, 400,000 fit rows, 2,000,000 samples) reproduces the data's means and pair products within $1.1 \times 10^{-3}$ (at most 1.6 standard errors), and its exact fitted probabilities sit within $9.6 \times 10^{-7}$ of the raw frequencies (the smoothing shift); on non-Gaussian Ising-chain data the pair probabilities are matched to $3.4 \times 10^{-7}$.

[metropolis-update]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/samplers/block_spin_sampler.py#L280-L304
[block-sweep]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/samplers/block_spin_sampler.py#L306-L342
[sampler-guarantees]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/samplers/block_spin_sampler.py#L47-L51
[mmd-contract]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/nn/modules/statistics.py#L75-L100
[mmd-loo]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/nn/modules/statistics.py#L164-L175
[kernel-bandwidth]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/nn/modules/kernels.py#L113-L144
[kernels]: dwave/plugins/torch/kernels.py
[kravchuk-sums]: dwave/plugins/torch/kernels.py#L222-L330
[mmd-functional]: dwave/plugins/torch/nn/functional.py#L42-L151
[squared-mmd]: dwave/plugins/torch/nn/modules/statistics.py#L80-L209
[ising-contract]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/nn/modules/ising.py#L31-L49
[ising-loss-claim]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/nn/modules/ising.py#L65-L72
[expectation-contract]: dwave/plugins/torch/nn/functional.py#L255-L311
[pseudo-kl-contract]: dwave/plugins/torch/nn/functional.py#L159-L183
[gumbel-code]: dwave/plugins/torch/nn/functional.py#L194-L220
[kl-test]: tests/test_pseudo_kl_divergence.py#L120-L152
[unknown-mask]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/models/boltzmann_machine.py#L440-L455
[dimod-batch]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/samplers/dimod_sampler.py#L225-L235
[ising-betas]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/nn/modules/ising.py#L159-L171
[sampleset-conversion]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/utils.py#L29-L52
[dimod-unconditional]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/samplers/dimod_sampler.py#L158-L162
[statistics-sign]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/models/boltzmann_machine.py#L302-L308
[initialization-contract]: https://github.com/dwavesystems/dwave-pytorch-plugin/blob/a5e7980a55c87e1ab3bd3741dd34d7ed02784189/dwave/plugins/torch/models/boltzmann_machine.py#L59-L80
