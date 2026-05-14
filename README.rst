D-Wave PyTorch Plugin
=====================

This plugin provides an interface between D-Wave's quantum computers and
the PyTorch framework, including neural network modules for building
and training Boltzmann Machines along with various sampler utility functions.

Example
-------
Boltzmann Machines are probabilistic generative models for high-dimensional binary data.
The following example walks through a typical workflow for fitting Boltzmann Machines via maximum likelihood.

Define a Graph-Restricted Boltzmann Machine with a square graph

.. code-block:: python

    import torch
    from torch.optim import SGD
    
    from dwave.plugins.torch.models import GraphRestrictedBoltzmannMachine as GRBM
    from dwave.plugins.torch.samplers import BlockSampler

    grbm = GRBM(nodes=["a", "b", "c", "d"], edges=[("a", "b"), ("b", "c"), ("c", "d"), ("d", "a")])
    print("Linear weights:", grbm.linear)
    print("Quadratic weights:", grbm.quadratic)


Instantiate a `block-Gibbs sampler <https://en.wikipedia.org/wiki/Gibbs_sampling#Blocked_Gibbs_sampler>`_.
Variables "a" and "c" are in block 0; variables "b" and "d" are in block 1.
The sampler consists of three parallel Markov chains of length ten each.
Each Markov chain samples at a constant unit inverse temperature.

.. code-block:: python

    sampler = BlockSampler(grbm=grbm, colouring=lambda v: v in {"b", "d"}, num_chains=3, schedule=[1]*10)


Create a batch of data and perform one likelihood-optimization step

.. code-block:: python

    x_data = torch.tensor([[1, -1, 1, -1], [-1, 1, 1, 1]], dtype=torch.float32)
    optimizer = SGD(grbm.parameters(), lr=1)
    x_model = sampler.sample()
    grbm.quasi_objective(x_data, x_model).backward()
    optimizer.step()
    print("Updated quadratic weights:", grbm.quadratic)

To use a `dimod <https://github.com/dwavesystems/dimod/>`_ sampler, replace the :code:`sampler = BlockSampler(...)` line with

.. code-block:: python

    from dwave.plugins.torch.samplers import DimodSampler
    from dwave.samplers import RandomSampler
    sampler = DimodSampler(grbm=grbm, sampler=RandomSampler(),
                           prefactor=1, sample_kwargs=dict(num_reads=5))


License
-------

Released under the Apache License 2.0. See LICENSE file.
