# Copyright 2020 DeepMind Technologies Limited.
# Modifications Copyright (c) 2025 Max Geier, Khachatur Nazaryan, Massachusetts Institute of Technology, MA, USA
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
# - Streamlined for materials simulations
# - Included CustomPsiformer and SlaterNet architectures
# - Simplified batch initialization of electron positions

"""Core training loop for neural QMC in JAX."""

import functools
import importlib
import os
import time
from typing import Optional, Mapping, Sequence, Tuple, Union

from absl import logging
import chex
from periodicwave import checkpoint
from periodicwave import constants
from periodicwave import curvature_tags_and_blocks
from periodicwave import envelopes
from periodicwave import hamiltonian
from periodicwave import loss as qmc_loss_functions
from periodicwave import mcmc
from periodicwave import networks
from periodicwave import CustomPsiformer
from periodicwave import SlaterNet
from periodicwave.utils import statistics
from periodicwave.utils import utils
from periodicwave.utils import writers
from periodicwave.utils import jax_utils
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import kfac_jax
import ml_collections
import numpy as np
import optax
from typing_extensions import Protocol

def _assign_spin_configuration(
    nalpha: int, nbeta: int, batch_size: int = 1
) -> jnp.ndarray:
  """Returns the spin configuration for a fixed spin polarisation."""
  spins = jnp.concatenate((jnp.ones(nalpha), -jnp.ones(nbeta)))
  return jnp.tile(spins[None], reps=(batch_size, 1))

def init_electrons_gaussian(
    key,
    nspins: Sequence[int],
    ndim: int,
    batch_size: int,
    init_width: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
  """Initializes electron positions around each atom.

  Args:
    key: JAX RNG state.
    nspins: tuple of number of alpha and beta electrons.
    batch_size: total number of MCMC configurations to generate across all
      devices.
    init_width: width of (atom-centred) Gaussian used to generate initial
      electron configurations.

  Returns:
    array of (batch_size, (nalpha+nbeta)*ndim) of initial (random) electron
    positions in the initial MCMC configurations and ndim is the dimensionality
    of the space (i.e. typically 3), and array of (batch_size, (nalpha+nbeta))
    of spin configurations, where 1 and -1 indicate alpha and beta electrons
    respectively.
  """

  key, subkey = jax.random.split(key)
  electron_positions = jax.random.normal(subkey, shape=(batch_size, ndim * sum(nspins))) * init_width
  electron_spins = _assign_spin_configuration(nspins[0], nspins[1], batch_size)

  return electron_positions, electron_spins

# All optimizer states (KFAC and optax-based).
OptimizerState = Union[optax.OptState, kfac_jax.Optimizer.State]
OptUpdateResults = Tuple[networks.ParamTree, Optional[OptimizerState],
                         jnp.ndarray,
                         Optional[qmc_loss_functions.AuxiliaryLossData]]

class OptUpdate(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: optax.OptState,
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters accordingly.

    Args:
      params: network parameters.
      data: electron positions, spins and atomic positions.
      opt_state: optimizer internal state.
      key: RNG state.

    Returns:
      Tuple of (params, opt_state, loss, aux_data), where params and opt_state
      are the updated parameters and optimizer state, loss is the evaluated loss
      and aux_data auxiliary data (see AuxiliaryLossData docstring).
    """


StepResults = Tuple[
    networks.FermiNetData,
    networks.ParamTree,
    Optional[optax.OptState],
    jnp.ndarray,
    qmc_loss_functions.AuxiliaryLossData,
    jnp.ndarray,
]


class Step(Protocol):

  def __call__(
      self,
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: OptimizerState,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """Performs one set of MCMC moves and an optimization step.

    Args:
      data: batch of MCMC configurations, spins and atomic positions.
      params: network parameters.
      state: optimizer internal state.
      key: JAX RNG state.
      mcmc_width: width of MCMC move proposal. See mcmc.make_mcmc_step.

    Returns:
      Tuple of (data, params, state, loss, aux_data, pmove).
        data: Updated MCMC configurations drawn from the network given the
          *input* network parameters.
        params: updated network parameters after the gradient update.
        state: updated optimization state.
        loss: energy of system based on input network parameters averaged over
          the entire set of MCMC configurations.
        aux_data: AuxiliaryLossData object also returned from evaluating the
          loss of the system.
        pmove: probability that a proposed MCMC move was accepted.
    """


def null_update(
    params: networks.ParamTree,
    data: networks.FermiNetData,
    opt_state: Optional[optax.OptState],
    key: chex.PRNGKey,
) -> OptUpdateResults:
  """Performs an identity operation with an OptUpdate interface."""
  del data, key
  return params, opt_state, jnp.zeros(1), None


def make_opt_update_step(evaluate_loss: qmc_loss_functions.LossFn,
                         optimizer: optax.GradientTransformation) -> OptUpdate:
  """Returns an OptUpdate function for performing a parameter update."""

  # Define function that differentiates wrt parameters (argument 0)
  loss_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)

  def opt_update(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates the loss and gradients and updates the parameters using optax."""
    (loss, aux_data), grad = loss_and_grad(params, key, data)
    grad = constants.pmean(grad)
    updates, opt_state = optimizer.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, aux_data

  return opt_update


def make_loss_step(evaluate_loss: qmc_loss_functions.LossFn) -> OptUpdate:
  """Returns an OptUpdate function for evaluating the loss."""

  def loss_eval(
      params: networks.ParamTree,
      data: networks.FermiNetData,
      opt_state: Optional[optax.OptState],
      key: chex.PRNGKey,
  ) -> OptUpdateResults:
    """Evaluates just the loss and gradients with an OptUpdate interface."""
    loss, aux_data = evaluate_loss(params, key, data)
    return params, opt_state, loss, aux_data

  return loss_eval


def make_training_step(
    mcmc_step,
    optimizer_step: OptUpdate,
    reset_if_nan: bool = False,
) -> Step:
  """Factory to create traning step for non-KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    optimizer_step: OptUpdate callable which evaluates the forward and backward
      passes and updates the parameters and optimizer state, as required.
    reset_if_nan: If true, reset the params and opt state to the state at the
      previous step when the loss is NaN

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  @functools.partial(constants.pmap, donate_argnums=(0, 1, 2)) # applies pmap to step with args 0,1,2 donated for memory efficiency
  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: Optional[optax.OptState],
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration (except for KFAC): MCMC steps + optimization."""
    # MCMC loop
    mcmc_key, loss_key = jax.random.split(key, num=2)
    data, pmove = mcmc_step(params, data, mcmc_key, mcmc_width)

    # Optimization step
    new_params, new_state, loss, aux_data = optimizer_step(params,
                                                           data,
                                                           state,
                                                           loss_key)
    if reset_if_nan:
      new_params = jax.lax.cond(jnp.isnan(loss),
                                lambda: params,
                                lambda: new_params)
      new_state = jax.lax.cond(jnp.isnan(loss),
                               lambda: state,
                               lambda: new_state)
    return data, new_params, new_state, loss, aux_data, pmove

  return step


def make_kfac_training_step(
    mcmc_step,
    damping: float,
    optimizer: kfac_jax.Optimizer,
    reset_if_nan: bool = False) -> Step:
  """Factory to create traning step for KFAC optimizers.

  Args:
    mcmc_step: Callable which performs the set of MCMC steps. See make_mcmc_step
      for creating the callable.
    damping: value of damping to use for each KFAC update step.
    optimizer: KFAC optimizer instance.
    reset_if_nan: If true, reset the params and opt state to the state at the
      previous step when the loss is NaN

  Returns:
    step, a callable which performs a set of MCMC steps and then an optimization
    update. See the Step protocol for details.
  """
  mcmc_step = constants.pmap(mcmc_step, donate_argnums=1)
  shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
  shared_damping = kfac_jax.utils.replicate_all_local_devices(
      jnp.asarray(damping))
  # Due to some KFAC cleverness related to donated buffers, need to do this
  # to make state resettable
  copy_tree = constants.pmap(
      functools.partial(jax.tree_util.tree_map,
                        lambda x: (1.0 * x).astype(x.dtype)))

  def step(
      data: networks.FermiNetData,
      params: networks.ParamTree,
      state: kfac_jax.Optimizer.State,
      key: chex.PRNGKey,
      mcmc_width: jnp.ndarray,
  ) -> StepResults:
    """A full update iteration for KFAC: MCMC steps + optimization."""
    # KFAC requires control of the loss and gradient eval, so everything called
    # here must be already pmapped.

    # MCMC loop
    mcmc_keys, loss_keys = kfac_jax.utils.p_split(key)
    data, pmove = mcmc_step(params, data, mcmc_keys, mcmc_width)

    if reset_if_nan:
      old_params = copy_tree(params)
      old_state = copy_tree(state)

    # Optimization step
    new_params, new_state, stats = optimizer.step(
        params=params,
        state=state,
        rng=loss_keys,
        batch=data,
        momentum=shared_mom,
        damping=shared_damping,
    )

    if reset_if_nan and jnp.any(jnp.isnan(stats['loss'])):
      new_params = old_params
      new_state = old_state
    return data, new_params, new_state, stats['loss'], stats['aux'], pmove

  return step

def train(cfg: ml_collections.ConfigDict, writer_manager=None):
  """Runs training loop for QMC.

  Args:
    cfg: ConfigDict containing the system and training parameters to run on. See
      base_config.default for more details.
    writer_manager: context manager with a write method for logging output. If
      None, a default writer (ferminet.utils.writers.Writer) is used.

  Raises:
    ValueError: if an illegal or unsupported value in cfg is detected.
  """
  # Device logging
  num_devices = jax.local_device_count()
  num_hosts = jax.device_count() // num_devices
  logging.info('Starting QMC with %i XLA devices per host '
               'across %i hosts.', num_devices, num_hosts)
  if cfg.batch_size % (num_devices * num_hosts) != 0:
    raise ValueError('Batch size must be divisible by number of devices, '
                     f'got batch size {cfg.batch_size} for '
                     f'{num_devices * num_hosts} devices.')
  host_batch_size = cfg.batch_size // num_hosts  # batch size per host
  device_batch_size = host_batch_size // num_devices  # batch size per device
  data_shape = (num_devices, device_batch_size)

  # For consistency with the original FermiNet interface, we carry default
  # atom and charge information, set to zero. 
  # This data is unused in pbc code without atoms.
  atoms = jnp.stack([jnp.array([0.0, 0.0])])
  charges = jnp.array([0.0])
  nspins = cfg.system.electrons

  # Generate atomic configurations for each walker
  batch_atoms = jnp.tile(atoms[None, ...], [device_batch_size, 1, 1])
  batch_atoms = kfac_jax.utils.replicate_all_local_devices(batch_atoms)
  batch_charges = jnp.tile(charges[None, ...], [device_batch_size, 1])
  batch_charges = kfac_jax.utils.replicate_all_local_devices(batch_charges)

  if cfg.debug.deterministic:
    seed = 42
  else:
    seed = jnp.asarray([1e6 * time.time()])
    seed = int(multihost_utils.broadcast_one_to_all(seed)[0])
  prng_key = jax.random.PRNGKey(seed)

  if cfg.network.make_feature_layer_fn:
    feature_layer_module, feature_layer_fn = (
        cfg.network.make_feature_layer_fn.rsplit('.', maxsplit=1))
    feature_layer_module = importlib.import_module(feature_layer_module)
    make_feature_layer: networks.MakeFeatureLayer = getattr(
        feature_layer_module, feature_layer_fn
    )
    feature_layer = make_feature_layer(
        natoms=charges.shape[0],
        nspins=cfg.system.electrons,
        ndim=cfg.system.ndim,
        **cfg.network.make_feature_layer_kwargs)
  else:
    feature_layer = networks.make_ferminet_features(
        natoms=charges.shape[0],
        nspins=cfg.system.electrons,
        ndim=cfg.system.ndim,
    )

  if cfg.network.make_envelope_fn:
    envelope_module, envelope_fn = (
        cfg.network.make_envelope_fn.rsplit('.', maxsplit=1))
    envelope_module = importlib.import_module(envelope_module)
    make_envelope = getattr(envelope_module, envelope_fn)
    envelope = make_envelope(**cfg.network.make_envelope_kwargs)  # type: envelopes.Envelope
  else:
    # default: use no envelope
    envelope = envelopes.make_null_envelope()

  use_complex = cfg.network.get('complex', False)
  if cfg.network.network_type == 'CustomPsiformer':
    network = CustomPsiformer.make_fermi_net(
        nspins,
        charges,
        ndim=cfg.system.ndim,
        determinants=cfg.network.determinants,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow=cfg.network.get('jastrow', 'default'), # default: No Jastrow
        jastrow_kwargs=cfg.network.jastrow_kwargs,
        bias_orbitals=cfg.network.bias_orbitals,
        complex_output=use_complex,
        pbc_lattice=cfg.system.pbc_lattice,
        **cfg.network.CustomPsiformer,
    )
  elif cfg.network.network_type == 'SlaterNet':
    network = SlaterNet.make_fermi_net(
        nspins,
        charges,
        ndim=cfg.system.ndim,
        determinants=cfg.network.determinants,
        envelope=envelope,
        feature_layer=feature_layer,
        jastrow=cfg.network.get('jastrow', 'default'), # default: No Jastrow
        jastrow_kwargs=cfg.network.jastrow_kwargs,
        bias_orbitals=cfg.network.bias_orbitals,
        complex_output=use_complex,
        pbc_lattice=cfg.system.pbc_lattice,
        **cfg.network.SlaterNet,
    )
  prng_key, subkey = jax.random.split(prng_key)
  params = network.init(subkey)

  params = kfac_jax.utils.replicate_all_local_devices(params)
  signed_network = network.apply
  # Often just need log|psi(x)|.
  logabs_network = lambda *args, **kwargs: signed_network(*args, **kwargs)[1]
  batch_network = jax.vmap(
      logabs_network, in_axes=(None, 0, 0, 0, 0), out_axes=0
  )  # batched network

  # Exclusively when computing the gradient wrt the energy for complex
  # wavefunctions, it is necessary to have log(psi) rather than log(|psi|).
  # This is unused if the wavefunction is real-valued.
  def log_network(*args, **kwargs):
    if not use_complex:
      raise ValueError('This function should never be used if the '
                        'wavefunction is real-valued.')
    phase, mag = signed_network(*args, **kwargs)
    return mag + 1.j * phase

  # Set up checkpointing and restore params/data if necessary
  # Mirror behaviour of checkpoints in TF FermiNet.
  # Checkpoints are saved to save_path.
  # When restoring, we first check for a checkpoint in save_path. If none are
  # found, then we check in restore_path.  This enables calculations to be
  # started from a previous calculation but then resume from their own
  # checkpoints in the event of pre-emption.
  ckpt_save_path = checkpoint.create_save_path(cfg.log.save_path)
  ckpt_restore_path = checkpoint.get_restore_path(cfg.log.restore_path)

  ckpt_restore_filename = (
      checkpoint.find_last_checkpoint(ckpt_save_path) or
      checkpoint.find_last_checkpoint(ckpt_restore_path))

  if ckpt_restore_filename:
    (t_init,
     data,
     params,
     opt_state_ckpt,
     mcmc_width_ckpt) = checkpoint.restore(
         ckpt_restore_filename, host_batch_size)
  else:
    logging.info('No checkpoint found. Training new model.')
    prng_key, subkey = jax.random.split(prng_key)
    # make sure data on each host is initialized differently
    subkey = jax.random.fold_in(subkey, jax.process_index())
    # create electron state (position and spin)
    pos, spins = init_electrons_gaussian(
      subkey,
      nspins = cfg.system.electrons,
      ndim = cfg.system.ndim,
      batch_size=host_batch_size,
      init_width=cfg.mcmc.init_width,)

    pos = jnp.reshape(pos, data_shape + (-1,))
    pos = kfac_jax.utils.broadcast_all_local_devices(pos)
    spins = jnp.reshape(spins, data_shape + (-1,))
    spins = kfac_jax.utils.broadcast_all_local_devices(spins)
    data = networks.FermiNetData(
        positions=pos, spins=spins, atoms=batch_atoms, charges=batch_charges
    )

    t_init = 0
    opt_state_ckpt = None
    mcmc_width_ckpt = None

  # Set up logging and observables
  train_schema = ['step', 'energy', 'ewmean', 'ewvar', 'pmove', 'locstd']

  # Initialisation done. We now want to have different PRNG streams on each
  # device. Shard the key over devices
  sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(prng_key)

  # Main training
  
  # Construct MCMC step
  mcmc_step = mcmc.make_mcmc_step(
      batch_network,
      device_batch_size,
      steps=cfg.mcmc.steps,
      ndim=cfg.system.ndim,
      blocks=cfg.mcmc.blocks,
  )

  # Construct loss and optimizer
  # Requires a local energy function to be specified.
  local_energy_module, local_energy_fn = (
      cfg.system.make_local_energy_fn.rsplit('.', maxsplit=1))
  local_energy_module = importlib.import_module(local_energy_module)
  make_local_energy = getattr(local_energy_module, local_energy_fn)  # type: hamiltonian.MakeLocalEnergy
  local_energy_fn = make_local_energy(
      f=signed_network,
      charges=charges,
      nspins=nspins,
      complex_output=use_complex,
      use_scan=False,
      **cfg.system.make_local_energy_kwargs)

  local_energy = local_energy_fn

  evaluate_loss = qmc_loss_functions.make_loss(
      log_network if use_complex else logabs_network,
      local_energy,
      clip_local_energy=cfg.optim.clip_local_energy,
      clip_from_median=cfg.optim.clip_median,
      center_at_clipped_energy=cfg.optim.center_at_clip,
      complex_output=use_complex,
  )

  # Compute the learning rate
  def learning_rate_schedule(t_: jnp.ndarray) -> jnp.ndarray:
    return cfg.optim.lr.rate * jnp.power(
        (1.0 / (1.0 + (t_/cfg.optim.lr.delay))), cfg.optim.lr.decay)

  # Construct and setup optimizer
  if cfg.optim.optimizer == 'none':
    optimizer = None
  elif cfg.optim.optimizer == 'adam':
    optimizer = optax.chain(
        optax.scale_by_adam(**cfg.optim.adam),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.))
  elif cfg.optim.optimizer == 'lamb':
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(eps=1e-7),
        optax.scale_by_trust_ratio(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1))
  elif cfg.optim.optimizer == 'kfac':
    # Differentiate wrt parameters (argument 0)
    val_and_grad = jax.value_and_grad(evaluate_loss, argnums=0, has_aux=True)
    optimizer = kfac_jax.Optimizer(
        val_and_grad,
        l2_reg=cfg.optim.kfac.l2_reg,
        norm_constraint=cfg.optim.kfac.norm_constraint,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=cfg.optim.kfac.cov_ema_decay,
        inverse_update_period=cfg.optim.kfac.invert_every,
        min_damping=cfg.optim.kfac.min_damping,
        num_burnin_steps=0,
        register_only_generic=cfg.optim.kfac.register_only_generic,
        estimation_mode='fisher_exact',
        multi_device=True,
        pmap_axis_name=constants.PMAP_AXIS_NAME,
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
        ),
        # debug=True
    )
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    opt_state = optimizer.init(params, subkeys, data)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
  else:
    raise ValueError(f'Not a recognized optimizer: {cfg.optim.optimizer}')

  if not optimizer:
    opt_state = None
    step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=make_loss_step(evaluate_loss))
  elif isinstance(optimizer, optax.GradientTransformation):
    # optax/optax-compatible optimizer (ADAM, LAMB, ...)
    opt_state = jax.pmap(optimizer.init)(params)
    opt_state = opt_state_ckpt or opt_state  # avoid overwriting ckpted state
    step = make_training_step(
        mcmc_step=mcmc_step,
        optimizer_step=make_opt_update_step(evaluate_loss, optimizer),
        reset_if_nan=cfg.optim.reset_if_nan)
  elif isinstance(optimizer, kfac_jax.Optimizer):
    step = make_kfac_training_step(
        mcmc_step=mcmc_step,
        damping=cfg.optim.kfac.damping,
        optimizer=optimizer,
        reset_if_nan=cfg.optim.reset_if_nan)
  else:
    raise ValueError(f'Unknown optimizer: {optimizer}')

  if mcmc_width_ckpt is not None:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(mcmc_width_ckpt[0])
  else:
    mcmc_width = kfac_jax.utils.replicate_all_local_devices(
        jnp.asarray(cfg.mcmc.move_width))

  if t_init == 0: # MCMC burn in
    logging.info('Burning in MCMC chain for %d steps', cfg.mcmc.burn_in)

    burn_in_step = make_training_step(
        mcmc_step=mcmc_step, optimizer_step=null_update)

    for t in range(cfg.mcmc.burn_in):
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      data, params, *_ = burn_in_step(
          data,
          params,
          state=None,
          key=subkeys,
          mcmc_width=mcmc_width)
    logging.info('Completed burn-in MCMC steps')
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    ptotal_energy = constants.pmap(evaluate_loss)
    initial_energy, _ = ptotal_energy(params, subkeys, data)
    logging.info('Initial energy: %03.4f E_h', initial_energy[0])

  time_of_last_ckpt = time.time()
  weighted_stats = None

  if cfg.optim.optimizer == 'none' and opt_state_ckpt is not None:
    # If opt_state_ckpt is None, then we're restarting from a previous inference
    # run (most likely due to preemption) and so should continue from the last
    # iteration in the checkpoint. Otherwise, starting an inference run from a
    # training run.
    logging.info('No optimizer provided. Assuming inference run.')
    logging.info('Setting initial iteration to 0.')
    t_init = 0

    if not ckpt_restore_filename:
      logging.info('Inference aborted because no checkpoint loaded.')
      raise Exception("Inference aborted because no checkpoint loaded.")

  if writer_manager is None:
    if cfg.optim.optimizer == 'none':
      stats_name = "inference_stats"
    else:
      stats_name = "train_stats"
    # if ${stats_name}.csv already exists from previous run, rename it to train_stats_N.cnv
    writers.rename_file(stats_name, cfg.log.save_path, file_extension="csv")

    writer_manager = writers.Writer(
        name=stats_name,
        schema=train_schema,
        directory=ckpt_save_path,
        iteration_key=None,
        log=False)
    
  with writer_manager as writer:
    # Main training loop
    num_resets = 0  # used if reset_if_nan is true
    for t in range(t_init, cfg.optim.iterations):
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      data, params, opt_state, loss, aux_data, pmove = step(
          data,
          params,
          opt_state,
          subkeys,
          mcmc_width)
      
      # due to pmean, loss, and pmove should be the same across devices.
      loss = loss[0]
      # per batch variance isn't informative. Use weighted mean and variance instead.
      weighted_stats = statistics.exponentialy_weighted_stats(
          alpha=0.1, observation=loss, previous_stats=weighted_stats)

      pmove = pmove[0]

      # variance = aux_data.variance[0] 
      local_std = jnp.sqrt(aux_data.variance[0])

      # Update MCMC move width
      if cfg.mcmc.move_width_updater == 'adaptive':
        def do_width_update(w):
            w = jnp.where(pmove > 0.55, w * 1.1, w)
            w = jnp.where(pmove < 0.50, w * 0.9, w)
            return w
        pred = (jnp.remainder(t, cfg.mcmc.adapt_frequency) == 0)  # scalar ()
        mcmc_width = jax.lax.cond(pred, do_width_update, lambda w: w, mcmc_width)

      if cfg.debug.check_nan:
        tree = {'params': params, 'loss': loss}
        if cfg.optim.optimizer != 'none':
          tree['optim'] = opt_state
        try:
          chex.assert_tree_all_finite(tree)
          num_resets = 0  # Reset counter if check passes
        except AssertionError as e:
          if cfg.optim.reset_if_nan:  # Allow a certain number of NaNs
            num_resets += 1
            if num_resets > 100:
              raise e
          else:
            raise e

      # Logging
      if t % cfg.log.stats_frequency == 0:
        # write to train_stats
        writer_kwargs = {
            'step': t,
            'energy': np.asarray(loss),
            'ewmean': np.asarray(weighted_stats.mean),
            'ewvar': np.asarray(weighted_stats.variance),
            'pmove': np.asarray(pmove),
            'locstd': np.asarray(local_std),
        }
        writer.write(t, **writer_kwargs)

        # log training data
        logging_str = ('Step %05d: '
                       '%03.4f E_h, exp. variance=%03.4f E_h^2, pmove=%0.2f')
        logging_args = t, loss, weighted_stats.variance, pmove
        logging.info(logging_str, *logging_args)
        
      # Checkpointing every cfg.log.save_frequency minutes and at final iteration
      if time.time() - time_of_last_ckpt > cfg.log.save_frequency * 60 or t == (cfg.optim.iterations - 1):
        checkpoint.save(ckpt_save_path, t, data, params, opt_state, mcmc_width)
        time_of_last_ckpt = time.time()


