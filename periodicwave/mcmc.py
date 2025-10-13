# Copyright 2020 DeepMind Technologies Limited.
# Modifications Copyright (c) 2025 Max Geier Massachusetts Institute of Technology, MA, USA
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTICE: This file has been modified from the original DeepMind version.
# Changes:
# - Simplified MCMC move width update routine (called directly in train.py)

"""Metropolis-Hastings Monte Carlo.

NOTE: these functions operate on batches of MCMC configurations and should not
be vmapped.
"""

import chex
from periodicwave import constants
from periodicwave import networks
import jax
from jax import lax
from jax import numpy as jnp

def mh_accept(x1, x2, spins1, spins2, lp_1, lp_2, ratio, key, num_accepts):
  """Given state, proposal, and probabilities, execute MH accept/reject step."""
  key, subkey = jax.random.split(key)
  rnd = jnp.log(jax.random.uniform(subkey, shape=ratio.shape))
  cond = ratio > rnd
  x_new = jnp.where(cond[..., None], x2, x1)
  spins_new = jnp.where(cond[..., None], spins2, spins1)
  lp_new = jnp.where(cond, lp_2, lp_1)
  num_accepts += jnp.sum(cond)
  return x_new, spins_new, key, lp_new, num_accepts

def mh_update(
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    ndim=3,
    blocks=1,
    i=0
    
):
  """Performs one Metropolis-Hastings step using an all-electron move.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with signature f(params, x) which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configurations (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    ndim: dimensionality of system.
    blocks: Ignored.
    i: Ignored.
    allow_spin_flips: whether to allow spin flips during training

  Returns:
    (x, key, lp, num_accepts), where:
      x: Updated MCMC configurations.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.
  """
  del i, blocks, ndim  # electron index ignored for all-electron moves
  key, subkey = jax.random.split(key)
  x1 = data.positions
  spins1 = data.spins  

  x2 = x1 + stddev * jax.random.normal(subkey, shape=x1.shape)  # proposal
  spins2 = spins1                                  
  lp_2 = 2.0 * f(
      params, x2, spins2, data.atoms, data.charges
  )  # log prob of proposal
  ratio = lp_2 - lp_1

  x_new, spins_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, spins1, spins2, lp_1, lp_2, ratio, key, num_accepts)
  
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new, 'spins': spins_new}))
  return new_data, key, lp_new, num_accepts

def mh_block_update(                                   # SPIN UPDATES ARE NOT IMPLEMENTED
    params: networks.ParamTree,
    f: networks.LogFermiNetLike,
    data: networks.FermiNetData,
    key: chex.PRNGKey,
    lp_1,
    num_accepts,
    stddev=0.02,
    ndim=3,
    blocks=1,
    i=0,
):
  """Performs one Metropolis-Hastings step for a block of electrons.

  Args:
    params: Wavefuncttion parameters.
    f: Callable with LogFermiNetLike signature which returns the log of the
      wavefunction (i.e. the sqaure root of the log probability of x).
    data: Initial MCMC configuration (batched).
    key: RNG state.
    lp_1: log probability of f evaluated at x1 given parameters params.
    num_accepts: Number of MH move proposals accepted.
    stddev: width of Gaussian move proposal.
    ndim: dimensionality of system.
    blocks: number of blocks to split electron updates into.
    i: index of block of electrons to move.

  Returns:
    (x, key, lp, num_accepts), where:
      x: MCMC configurations with updated positions.
      key: RNG state.
      lp: log probability of f evaluated at x.
      num_accepts: update running total of number of accepted MH moves.

  Raises:
    NotImplementedError: if atoms is supplied.
  """
  key, subkey = jax.random.split(key)
  batch_size = data.positions.shape[0]
  nelec = data.positions.shape[1] // ndim
  pad = (blocks - nelec % blocks) % blocks
  x1 = jnp.reshape(
      jnp.pad(data.positions, ((0, 0), (0, pad * ndim))),
      [batch_size, blocks, -1, ndim],
  )
  ii = i % blocks
  x2 = x1.at[:, ii].add(
      stddev * jax.random.normal(subkey, shape=x1[:, ii].shape))
  x2 = jnp.reshape(x2, [batch_size, -1])
  if pad > 0:
    x2 = x2[..., :-pad*ndim]
  # log prob of proposal
  lp_2 = 2.0 * f(params, x2, data.spins, data.atoms, data.charges)
  ratio = lp_2 - lp_1

  x1 = jnp.reshape(x1, [batch_size, -1])
  if pad > 0:
    x1 = x1[..., :-pad*ndim]
  x_new, key, lp_new, num_accepts = mh_accept(
      x1, x2, lp_1, lp_2, ratio, key, num_accepts)
  new_data = networks.FermiNetData(**(dict(data) | {'positions': x_new}))
  return new_data, key, lp_new, num_accepts

def make_mcmc_step(batch_network,
                   batch_per_device,
                   steps=10,
                   ndim=3,
                   blocks=1
                   ):
  """Creates the MCMC step function.

  Args:
    batch_network: function, signature (params, x), which evaluates the log of
      the wavefunction (square root of the log probability distribution) at x
      given params. Inputs and outputs are batched.
    batch_per_device: Batch size per device.
    steps: Number of MCMC moves to attempt in a single call to the MCMC step
      function.
    atoms: atom positions. If given, an asymmetric move proposal is used based
      on the harmonic mean of electron-atom distances for each electron.
      Otherwise the (conventional) normal distribution is used.
    ndim: Dimensionality of the system (usually 3). Required only for block updates.
    blocks: Number of blocks to split the updates into. If 1, use all-electron
      moves.

  Returns:
    Callable which performs the set of MCMC steps.
  """
  inner_fun = mh_block_update if blocks > 1 else mh_update

  def mcmc_step(params, data, key, width):
    """Performs a set of MCMC steps.

    Args:
      params: parameters to pass to the network.
      data: (batched) MCMC configurations to pass to the network.
      key: RNG state.
      width: standard deviation to use in the move proposal.

    Returns:
      (data, pmove), where data is the updated MCMC configurations, key the
      updated RNG state and pmove the average probability a move was accepted.
    """
    pos = data.positions

    def step_fn(i, x):
      return inner_fun(
          params,
          batch_network,
          *x,
          stddev=width,
          ndim=ndim,
          blocks=blocks,
          i=i
          )

    nsteps = steps * blocks
    logprob = 2.0 * batch_network(
        params, pos, data.spins, data.atoms, data.charges
    )
    new_data, key, _, num_accepts = lax.fori_loop(
        0, nsteps, step_fn, (data, key, logprob, 0.0)
    )
    pmove = jnp.sum(num_accepts) / (nsteps * batch_per_device)
    pmove = constants.pmean(pmove)
    return new_data, pmove

  return mcmc_step


