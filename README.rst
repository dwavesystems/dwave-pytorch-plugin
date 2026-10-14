.. image:: https://img.shields.io/pypi/v/dwave-pytorch-plugin.svg
    :target: https://pypi.org/project/dwave-pytorch-plugin

.. image:: https://img.shields.io/pypi/pyversions/dwave-pytorch-plugin.svg
    :target: https://pypi.org/project/dwave-pytorch-plugin

.. image:: https://circleci.com/gh/dwavesystems/dwave-pytorch-plugin.svg?style=svg
    :target: https://circleci.com/gh/dwavesystems/dwave-pytorch-plugin

.. image:: https://codecov.io/gh/dwavesystems/dwave-pytorch-plugin/branch/main/graph/badge.svg
    :target: https://codecov.io/gh/dwavesystems/dwave-pytorch-plugin

====================
dwave-pytorch-plugin
====================

.. start_pytorch_plugin_about

``dwave-pytorch-plugin`` provides `PyTorch <https://pytorch.org>`_ modules and samplers for
training Boltzmann machines with D-Wave quantum computers.

Boltzmann machines are energy-based models whose training requires samples from the model
distribution, which are costly to obtain classically. This plugin lets a D-Wave quantum processing
unit (QPU), or any classical `dimod <https://github.com/dwavesystems/dimod>`_ sampler, provide
those samples inside a standard PyTorch training loop, so that Boltzmann machines can be trained
on their own or as priors and layers of larger neural networks.

The package provides the following components.

* **Models** (``dwave.plugins.torch.models``). ``GraphRestrictedBoltzmannMachine`` is an Ising
  model on the nodes and edges of a graph, such as the working graph of a QPU. Its parameters
  are one linear bias per node and one quadratic bias per edge; the coupling matrix is built on
  the fly, so energies and effective fields are dense matrix products that run efficiently on
  GPUs while parameters, gradients and optimizer state stay the size of the graph. The model
  supports hidden units,
  exact conditional expectations of hidden units that are not adjacent to each other, a
  quasi-objective whose gradient is the gradient of the negative log likelihood, and estimation of
  the effective inverse temperature of a set of samples. Through ``gumbel_spins`` and
  ``pseudo_kl_divergence_loss`` of ``dwave.plugins.torch.nn.functional``, a Boltzmann machine
  serves as the prior of a discrete variational autoencoder, whose encoder-spins-decoder wiring
  is a few lines of PyTorch shown in the examples. The same package provides moment-matched
  **baselines** of spins with one ``fit`` / ``sample`` interface, against which a candidate
  model's evaluation score is read: ``UniformBernoulli`` (fair coins), ``IndependentBernoulli``
  (the data's per-spin means) and ``DichotomizedGaussian`` (the signs of a latent Gaussian
  matched exactly to the data's means and pairwise correlations); see ``docs/baselines.md``.

* **Samplers** (``dwave.plugins.torch.samplers``). Samplers are ``torch.nn.Module`` objects that
  hold the model they sample from, so moving or saving a sampler moves or saves the model and the
  sampler's own state together. ``BlockSampler`` performs block-Gibbs or block-Metropolis
  sampling with persistent Markov chains on any colouring of the graph, such as the two layers
  of a restricted Boltzmann machine, and ``DimodSampler`` wraps any dimod sampler, including the
  ``DWaveSampler`` of `dwave-system <https://github.com/dwavesystems/dwave-system>`_, scaling and
  clipping the Hamiltonian before it is submitted. All samplers support conditional sampling of
  partially observed spins and sampling from batches of Ising models given by their biases, which
  is how the inputs of the ``Ising`` layer are sampled.

* **Neural network modules** (``dwave.plugins.torch.nn``). An ``Ising`` layer takes the biases
  of a batch of Ising models and spins sampled from them by any of the samplers, and returns a
  statistic of the samples together with an unbiased estimate of the gradient of the statistic's
  expectation with respect to the biases. The statistic is any module of the set of samples:
  ``Mean`` (the default, the mean spins or the mean of any per-sample transform, whose parameters
  are trained through the layer) and ``SquaredMMD`` (the squared maximum mean discrepancy to a
  reference sample, under every kernel of the battery below) are provided. The estimator itself,
  ``dwave.plugins.torch.nn.functional.expectation``, takes any graph module, which is how a
  Boltzmann machine prior is trained by the maximum mean discrepancy. Also included is a
  ``MaximumMeanDiscrepancyLoss`` for matching encoder samples to prior samples, and, in
  ``dwave.plugins.torch.nn.functional``, the functional losses and the Gumbel-softmax sampling
  of spins from encoder logits.

* **Evaluation** (``dwave.plugins.torch.evaluation`` and ``dwave.plugins.torch.kernels``). A
  generative model of spins is judged by its samples alone with kernel scores: exactly unbiased
  estimates of the squared maximum mean discrepancy, with standard errors, under a battery of
  Hamming-isotropic kernels whose entries decompose the mismatch by **interaction order** (the
  per-spin means, the pairwise correlations, the ``k``-way parities), read against the data's own
  distance from the uniform distribution. ``Evaluator`` holds three data splits and serves the
  evaluation score, a headline kernel calibrated to the orders the data demonstrably carry, and
  the differentiable training loss; ``blocked_scores`` scores many models against the same
  blocks; ``evaluation.visualize`` draws the results. See ``docs/evaluation.md``.

* **Graph core** (``dwave.plugins.torch.graph``). ``GraphIndex``, the graph module that models,
  layers and samplers are built on, and random spin generation. It imports nothing but PyTorch.

* **dimod utilities** (``dwave.plugins.torch.utils``). Conversions between tensors and dimod's
  binary quadratic models and sample sets, and temperature estimation with dwave-system.

.. end_pytorch_plugin_about

Installation
============

``dwave-pytorch-plugin`` requires Python 3.10 or later and is tested on Linux, macOS, and Windows.

Install from PyPI:

.. code-block:: bash

    pip install dwave-pytorch-plugin

Or install from source:

.. code-block:: bash

    pip install -r requirements.txt
    pip install .

PyTorch is installed as a dependency. To use a GPU, install a build of PyTorch for your hardware
first by following the `PyTorch installation instructions <https://pytorch.org/get-started/locally/>`_.

Sampling with a QPU requires a `Leap <https://cloud.dwavesys.com/leap/>`_ account. Run
``dwave config create`` to store your API token, or see the
`D-Wave documentation <https://docs.dwavequantum.com/>`_ for other ways to configure access to
Leap's solvers.

Examples
========

Complete scripts are in the repository's
`examples directory <https://github.com/dwavesystems/dwave-pytorch-plugin/tree/main/examples>`_;
the Boltzmann machine example additionally requires the ``dwave-graphs`` package. The discrete
variational autoencoder example trains an autoencoder with a Boltzmann machine prior sampled by
a ``BlockSampler``. The evaluation and baselines examples score Boltzmann machines and the
baselines against samples of a known model by interaction order, and draw the figures of
``docs/evaluation.md`` and ``docs/baselines.md`` (which requires ``matplotlib``).

Training a Boltzmann machine
----------------------------

The following example fits a fully visible Boltzmann machine on a 4-by-4 grid to synthetic data
using a classical block-Gibbs sampler. The negative phase of each step is a sweep of the sampler's
persistent Markov chains (persistent contrastive divergence).

.. code-block:: python

    import networkx as nx
    import torch

    from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine
    from dwave.plugins.torch.samplers import BlockSampler

    # A Boltzmann machine on a 4x4 grid: one spin per node, one coupling per edge
    graph = nx.grid_2d_graph(4, 4)
    grbm = GraphRestrictedBoltzmannMachine(graph.nodes, graph.edges)

    # Persistent block-Gibbs chains provide samples from the model (the negative phase)
    sampler = BlockSampler(grbm, num_chains=500, seed=0)

    # Training data: spins in {-1, +1} with one column per node. Here, all spins of a
    # configuration agree, so the model should learn ferromagnetic couplings.
    data = torch.sign(torch.randn(500, 1)).expand(-1, grbm.n_nodes)

    optimizer = torch.optim.SGD(grbm.parameters(), lr=0.05)
    for step in range(200):
        s_model = sampler.sample()  # advance every chain by one sweep
        optimizer.zero_grad()
        # The gradient of the quasi-objective is the gradient of the negative log likelihood
        grbm.quasi_objective(data, s_model).backward()
        optimizer.step()

    # Samples from the trained model reproduce the data: the spins (almost) always agree
    print(sampler.sample().mean(-1).abs().mean())  # close to 1

Sampling with a quantum computer
--------------------------------

To sample with a QPU, define the model on the QPU's working graph and wrap the ``DWaveSampler``
in a ``DimodSampler``. A QPU samples at an effective inverse temperature that differs from one,
so the Hamiltonian is scaled by a ``prefactor``, the reciprocal of an estimate of that inverse
temperature, and clipped to the programmable ranges before it is submitted. The training loop
is otherwise unchanged.

.. code-block:: python

    from dwave.system import DWaveSampler

    from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine
    from dwave.plugins.torch.samplers import DimodSampler

    qpu = DWaveSampler()
    graph = qpu.to_networkx_graph()
    grbm = GraphRestrictedBoltzmannMachine(graph.nodes, graph.edges)

    sampler = DimodSampler(
        grbm,
        qpu,
        prefactor=1 / 6.35,  # a ballpark estimate; refine it with grbm.estimate_beta
        linear_range=qpu.properties["h_range"],
        quadratic_range=qpu.properties["j_range"],
        # Return every read as its own sample and submit the Hamiltonian as is
        sample_kwargs=dict(num_reads=100, answer_mode="raw", auto_scale=False),
    )

    s_model = sampler.sample()  # a (100, n_nodes) tensor of spins
    # The inverse temperature at which the model was effectively sampled; ideally close to 1
    beta = grbm.estimate_beta(s_model)

Any dimod sampler can be wrapped in the same way; for example, a classical Markov chain Monte
Carlo sampler operating at unit inverse temperature needs no prefactor.

Hidden units
------------

Nodes passed as ``hidden_nodes`` are not observed in the data, and observations have one column
per visible node. The learning objective takes complete spin configurations, so the hidden units
of the data are filled in first: exactly, with their conditional expectations, when no two hidden
units are adjacent, or with conditional samples drawn by any sampler otherwise.

.. code-block:: python

    from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine
    from dwave.plugins.torch.samplers import BlockSampler

    # A restricted Boltzmann machine: 8 visible units fully connected to 4 hidden units
    visible = [f"v{i}" for i in range(8)]
    hidden = [f"h{j}" for j in range(4)]
    edges = [(v, h) for v in visible for h in hidden]
    grbm = GraphRestrictedBoltzmannMachine(visible + hidden, edges, hidden_nodes=hidden)
    # The two layers are the two blocks of a block-Gibbs sampler: each sweep samples the visible
    # units given the hidden units and then the hidden units given the visible units
    sampler = BlockSampler(grbm, colouring=lambda node: node in hidden, num_chains=100)

    x = ...  # observations of shape (batch_size, 8), one column per visible unit

    s_data = grbm.conditional_expectation(grbm.pad_visible(x))  # exact: hidden units are not adjacent
    s_data = sampler.complete(x)  # alternative: conditional samples, valid for any model
    grbm.quasi_objective(s_data, sampler.sample()).backward()

Training a prior by the maximum mean discrepancy
------------------------------------------------

Any statistic of samples of a Boltzmann machine that is symmetric in the samples can be
differentiated with respect to the machine's biases by
``dwave.plugins.torch.nn.functional.expectation``, which attaches an unbiased score-function
estimate of the gradient of the statistic's expectation to the statistic's value. With the squared
maximum mean discrepancy as the statistic, a Boltzmann machine prior is pulled towards a reference
sample ``z``, for example the spins an encoder assigns to a batch of data, while ``z`` receives its
own gradient through the kernel. The statistic returns one estimate per kernel of the battery:
entry 0 the headline kernel (the Gaussian kernel at the bandwidth set by the dimension, or a
calibrated combination of the orders) and entry ``k`` the order-``k`` kernel, which measures the
mismatch of the ``k``-spin parities alone.

.. code-block:: python

    from dwave.plugins.torch.nn import SquaredMMD
    from dwave.plugins.torch.nn.functional import expectation

    mmd = SquaredMMD()
    samples = sampler.sample()  # (M, n_nodes) samples of the prior, M >= 3
    loss = expectation(grbm, grbm.linear, grbm.quadratic, samples, mmd, z)  # z: (R, n_nodes)
    loss[0].backward()  # the headline: gradients for the prior and for whatever produced z

The ``Ising`` layer is the same estimator for biases produced by another network:
``Ising(nodes, edges, statistic)(linear, quadratic, spins, *inputs)``.

Evaluating a model by interaction order
---------------------------------------

Samples of a model are scored against held-out data with ``Evaluator``: exactly unbiased
estimates of the squared maximum mean discrepancy with standard errors over blocks, one entry
per kernel of the battery, read against the data's own distance from the uniform distribution.
The baselines say how much of the data's structure is cheap: whatever the dichotomized Gaussian,
matched to the means and pairwise correlations, already captures, a candidate cannot claim as
learned.

.. code-block:: python

    from dwave.plugins.torch.evaluation import Evaluator
    from dwave.plugins.torch.models import DichotomizedGaussian

    # Floating spin matrices in {-1, +1} of shape (rows, n_nodes), cut into blocks of 512 rows
    evaluator = Evaluator(x_train, x_val, x_test, block_size=512)
    data, _ = evaluator.uniform_score("test")  # the data's Fourier weight per interaction order

    baseline = DichotomizedGaussian(grbm.n_nodes)
    baseline.fit(x_train)
    for name, samples in {"baseline": baseline.sample(x_test.shape[0]), "model": s_model}.items():
        scores, se = evaluator.score(samples.to(x_test.dtype), "test")  # (n_nodes + 1,) each
        print(name, "captured at order 2:", 1 - scores[2] / data[2])  # 1 = all, 0 = nothing

    evaluator.calibrate()  # the headline now weighs the orders the test split can measure
    print(evaluator.score(s_model, "test")[0][0] / evaluator.uniform_score("test")[0][0])

``docs/evaluation.md`` and ``docs/baselines.md`` describe the kernels, the estimator, the
calibration and the baselines, with derivations and references.

License
=======

Released under the Apache License 2.0. See the
`LICENSE file <https://github.com/dwavesystems/dwave-pytorch-plugin/blob/main/LICENSE>`_.

The use of the Boltzmann machine and discrete autoencoder implementations in this package with a
quantum computing system is protected by the intellectual property rights of D-Wave Quantum Inc.
and its affiliates. Their use with D-Wave's quantum computing systems requires access to D-Wave's
Leap quantum cloud service and is governed by the
`Leap Cloud Subscription Agreement <https://cloud.dwavesys.com/leap/legal/cloud_subscription_agreement/>`_.

Contributing
============

Ocean's `contributing guide <https://docs.ocean.dwavesys.com/en/stable/contributing.html>`_ has
guidelines for contributing to Ocean packages. See
`CONTRIBUTING.rst <https://github.com/dwavesystems/dwave-pytorch-plugin/blob/main/CONTRIBUTING.rst>`_
for how this package manages its release notes.

Testing
-------

Install the test requirements and run the unit tests from the root of the repository:

.. code-block:: bash

    pip install -r requirements.txt -r tests/requirements.txt
    python -m unittest
