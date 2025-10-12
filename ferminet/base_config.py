# Copyright 2022 DeepMind Technologies Limited.
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
# - Streamlined for materials calculations

""" Default base configuration for VMC calculations for periodic systems. """

import ml_collections
from ml_collections import config_dict

def default() -> ml_collections.ConfigDict:
  """Create set of default parameters for running qmc.py.

  Note: placeholder (cfg.system.electrons must be replaced with appropriate values.

  Returns:
    ml_collections.ConfigDict containing default settings.
  """
  # wavefunction output.
  cfg = ml_collections.ConfigDict({
      'batch_size': 1024,  # Default value that empirically works well for two-dimensional problems.
      'optim': {
          # Objective type. Only 'vmc' implemented.
          # 'vmc': minimise <H> by standard VMC energy minimization
          'objective': 'vmc',
          'iterations': 1000000,  # number of iterations
          'optimizer': 'kfac',  # one of adam, kfac, lamb, none
          'laplacian': 'default',  # one of default or folx (for forward lapl)
          'lr': {
              'rate':  0.05,  # learning rate
              'decay': 1.0,  # exponent of learning rate decay
              'delay': 10000.0,  # term that sets the scale of the rate decay
          },
          # If greater than zero, scale (at which to clip local energy) in units
          # of the mean deviation from the mean.
          'clip_local_energy': 5.0,
          # If true, center the clipping window around the median rather than
          # the mean. More "correct" for removing outliers, but also potentially
          # slow, especially with multihost training.
          'clip_median': False,
          # If true, center the local energy differences in the gradient at the
          # average clipped energy rather than average energy, guaranteeing that
          # the average energy difference will be zero in each batch.
          'center_at_clip': True,
          # If true, keep the parameters and optimizer state from the previous
          # step and revert them if they become NaN after an update. 
          'reset_if_nan': False,
          # KFAC hyperparameters. See KFAC documentation for details.
          'kfac': {
              'invert_every': 1,
              'cov_update_every': 1,
              'damping': 0.001,
              'cov_ema_decay': 0.95,
              'momentum': 0.0,
              'momentum_type': 'regular',
              # Warning: adaptive damping is not currently available.
              'min_damping': 1.0e-6,
              'norm_constraint': 0.001,
              'mean_center': True,
              'l2_reg': 0.0,
              'register_only_generic': False,
          },
          # ADAM hyperparameters. See optax documentation for details.
          'adam': {
              'b1': 0.9,
              'b2': 0.999,
              'eps': 1.0e-8,
              'eps_root': 0.0,
          },
      },
      'log': {
          'stats_frequency': 1,  # iterations between logging of stats
          'save_frequency': 10.0,  # minutes between saving network params
          # Path to save/restore network to/from. If falsy,
          # creates a timestamped directory in the working directory.
          'save_path': '',
          # Path containing checkpoint to restore network from.
          # Ignored if falsy or save_path contains a checkpoint.
          'restore_path': '',
      },
      'system': {
          # Specify the system by setting variables below.
          # number of spin up, spin-down electrons
          'electrons': tuple(),
          # Dimensionality. 
          'ndim': 2,
          # String set to module.make_local_energy, where make_local_energy is a
          # callable (type: MakeLocalEnergy) which creates a function which
          # evaluates the local energy and module is the absolute module
          # containing make_local_energy.
          # If not set, hamiltonian.local_energy is used (only kinetic energy).
          'make_local_energy_fn': '',
          # Additional kwargs to pass into make_local_energy_fn.
          'make_local_energy_kwargs': {},
          # If periodic boundary conditions are used, store lattice:
          'pbc_lattice': None,
      },
      'mcmc': {
          # Note: HMC options are not currently used.
          # Number of burn in steps after pretraining.  If zero do not burn in
          # or reinitialize walkers.
          'burn_in': 300,
          'steps': 30,  # Number of MCMC steps to make between network updates.
          # Width of (atom-centred) Gaussian used to generate initial electron
          # configurations.
          'init_width': 1.0,
          # Width of Gaussian used for random moves for RMW or step size for HMC.
          'move_width': 0.1,
          # How to update move_width during runtime.
          # const: move_width remains constant
          # adaptive: increases or reduces move_width depending on pmove (default)
          'move_width_updater': 'adaptive',
          # Number of steps after which to update the adaptive MCMC step size
          'adapt_frequency': 100,
          'blocks': 1,  # Number of blocks to split the MCMC sampling into
      },
      'network': {
          'network_type': 'CustomPsiformer',  # One of 'SlaterNet' or 'CustomPsiformer'.
          # If true, the network outputs complex numbers rather than real.
          'complex': False,
          # Only used if network_type is 'CustomPsiformer'.
          'CustomPsiformer': {
              # Customized Psiformer architecture: Geier et al,. arXiv:2502.05383
              'num_layers': 4, # number of self-attention layers
              'num_heads': 4,  # number of heads per layer
              'attn_dim': 32,  # dimension of the scalar product within each attention head
              'value_dim': 32, # dimension of the value vector multiplied with the softmax result
              'mlp_dim': 128,  # MLP dimension equals internal token dimension
              'num_perceptrons_per_layer': 2, # number of subsequenct perceptrons after each attention layer
              'use_layer_norm': True, # whether to apply layer norm after each MLP and attention operation
              'mlp_activation_fct': "GELU", # MLP non-linearity
          },
          # Only used if network_type is 'SlaterNet': Geier et al,. arXiv:2502.05383
          # NOTE: SlaterNet is equivalent to Hartree-Fock if cfg.network.determinants = 1.
          'SlaterNet': {
              'num_layers': 4, # number layers over which a residual connection is applied
              'mlp_dim': 128,  # MLP dimension equals internal token dimension
              'num_perceptrons_per_layer': 2, # number of perceptrons between residual connections
              'use_layer_norm': True, # whether to apply layer norm in each layer
              'mlp_activation_fct': "GELU", # MLP non-linearity
          },
          # Config common to all architectures.
          'determinants': 4,  # Number of determinants.
          'bias_orbitals': False,  # include bias in last layer to orbitals
          # If specified, include a pre-determinant Jastrow factor.
          # One of 'default' (use network_type default), 'none', or 'simple_ee'.
          # NOTE: simple_ee Jastrow is specific to Coulomb interaction. 
          # It requires to specify the spatial dimensional as well as interaction_strength 
          # of Coulomb such that cusp conditions are correctly enforced.
          # These parameters are passed in jastrow_kwargs
          'jastrow': 'none',
          # Additional kwargs for custom jastrow
          'jastrow_kwargs': {'ndim': 2, "interaction_strength": 1.0}, 
          # String set to module.make_feature_layer, where make_feature_layer is
          # callable (type: MakeFeatureLayer) which creates an object with
          # member functions init() and apply() that initialize parameters
          # for custom input features and modify raw input features,
          # respectively. Module is the absolute module containing
          # make_feature_layer.
          # If not set, networks.make_ferminet_features is used.
          'make_feature_layer_fn': '',
          # Additional kwargs to pass into make_local_energy_fn.
          'make_feature_layer_kwargs': {},
          # Same structure as make_feature_layer
          'make_envelope_fn': '',
          'make_envelope_kwargs': {},
      },
      'debug': {
          # Check optimizer state, parameters and loss and raise an exception if
          # NaN is found.
          'check_nan': False,
          'deterministic': False,  # Use a deterministic seed.
      },
  })

  return cfg

