# Copyright (c) 2025 Max Geier, Khachatur Nazaryan
# Massachusetts Institute of Technology, MA, USA
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
#
# For interface consistency, the structure of this file is based on DeepMind's psiformer.py.

""" 
Feed-forward neural network to represent the best single Slater determinant wavefunction. 
This approach is equivalent to unrestricted Hartree-Fock when generating a single determinant 
from the network.
"""

from typing import Mapping, Optional, Sequence, Tuple, Union

import attr
import chex
from ferminet import envelopes
from ferminet import jastrows
from ferminet import network_blocks
from ferminet import networks
import jax
import jax.numpy as jnp


@attr.s(auto_attribs=True, kw_only=True)
class SlaterNetOptions(networks.BaseNetworkOptions):
  """Options controlling the Hartree-Fock part of the network architecture.

  Attributes:
    num_layers: Number of MLP layers representing the single orbitals.
    mlp_dim: Dimension of the perceptron layers and the internal token dimension
    num_perceptrons_per_layer: number of perceptrons per layer
    use_layer_norm: If true, include a layer norm after each num_perceptrons_per_layer of perceptrons.
  
  Notice: The total number of perceptrons is: num_layers * num_perceptrons_per_layer
      After each layer (containing num_perceptrons_per_layer subsequent perceptrons), 
      the residual output h^(l-1) from the previous layer is added to improve gradient flow.
      If use_layer_norm = True, after each layer a layer norm is applied, for numerical stability.
  """

  num_layers: int = 4
  mlp_dim: int = 128
  num_perceptrons_per_layer: int = 2 
  use_layer_norm: bool = False
  mlp_activation_fct: str = "TANH"

def make_layer_norm() ->...:
  """Implementation of LayerNorm."""

  def init(param_shape: int) -> Mapping[str, jnp.ndarray]:
    params = {}
    params['scale'] = jnp.ones(param_shape)
    params['offset'] = jnp.zeros(param_shape)
    return params

  def apply(params: networks.ParamTree,
            inputs: jnp.ndarray,
            axis: int = -1) -> jnp.ndarray:
    mean = jnp.mean(inputs, axis=axis, keepdims=True)
    variance = jnp.var(inputs, axis=axis, keepdims=True)
    eps = 1e-5
    inv = params['scale'] * jax.lax.rsqrt(variance + eps)
    return inv * (inputs - mean) + params['offset']

  return init, apply

def make_mlp(activation_fct_name: str = "TANH") ->...:
  """Construct MLP, with final linear projection to embedding size."""
  
  # Select non-linear activation function
  if activation_fct_name.upper() == "TANH":
    activation_fct = jnp.tanh
  elif activation_fct_name.upper() == "ELU":
    activation_fct = jax.nn.elu
  elif activation_fct_name.upper() == "GELU":
    activation_fct = jax.nn.gelu
  
  # Initialize parameters
  def init(key: chex.PRNGKey, input_dim: int, perceptron_dims: Tuple[int, ...],
           ) -> Sequence[networks.Param]:
    params = []
    for i in range(len(perceptron_dims)):
      if i == 0:
        in_dim = input_dim
        perceptron_dim = perceptron_dims[0]
      else:
        in_dim = perceptron_dims[i - 1]
        perceptron_dim = perceptron_dims[i]

      key, subkey = jax.random.split(key)
      params.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=in_dim,
              out_dim=perceptron_dim,
              include_bias=True))
    return params

  # Apply multilayer perceptron
  def apply(params: Sequence[networks.Param],
            inputs: jnp.ndarray) -> jnp.ndarray:
    x = inputs
    for i in range(len(params)):
      x = activation_fct(network_blocks.linear_layer(x, **params[i]))
    return x

  return init, apply


def make_SlaterNet_block(num_layers: int,
                              mlp_dim: int,
                              num_perceptrons_per_layer: int,
                              use_layer_norm: bool = False,
                              mlp_activation_fct: str = "TANH") ->...:
  if use_layer_norm:
    layer_norm_init, layer_norm_apply = make_layer_norm()

  mlp_init, mlp_apply = make_mlp(mlp_activation_fct)

  def init(key: chex.PRNGKey) -> networks.ParamTree: # replaced qkv_dim -> mlp_dim because attention is skipped so only mlp is present
    params = {}
    ln_params = []
    mlp_params = []

    for _ in range(num_layers):
      key, mlp_key = jax.random.split(key, 2)
      if use_layer_norm:
        ln_params.append([layer_norm_init(mlp_dim)])
      mlp_params.append(mlp_init(mlp_key, input_dim = mlp_dim, perceptron_dims = (mlp_dim, ) * num_perceptrons_per_layer))

    params['ln'] = ln_params
    params['mlp'] = mlp_params

    return params

  def apply(params: networks.ParamTree, single_particle_stream: jnp.ndarray) -> jnp.ndarray:
    x = single_particle_stream
    for layer in range(num_layers):
      # MLP
      assert isinstance(params['mlp'][layer], (tuple, list))
      mlp_output = mlp_apply(params['mlp'][layer], x)

      # Residual + optional LayerNorm.
      x = x + mlp_output
      if use_layer_norm:
        x = layer_norm_apply(params['ln'][layer][0], x)

    return x

  return init, apply


def make_SlaterNet_layers(
    nspins: Tuple[int, ...],
    natoms: int,
    options: SlaterNetOptions,
) -> Tuple[networks.InitLayersFn, networks.ApplyLayersFn]:
  """Creates the permutation-equivariant layers for SlaterNet.

  Args:
    nspins: Tuple with number of spin up and spin down electrons.
    natoms: number of atoms.
    options: network options.

  Returns:
    Tuple of init, apply functions.
  """
  del nspins, natoms  # Unused.

  SlaterNet_init, SlaterNet_apply = make_SlaterNet_block(
      num_layers=options.num_layers,
      mlp_dim=options.mlp_dim,
      num_perceptrons_per_layer=options.num_perceptrons_per_layer,
      use_layer_norm=options.use_layer_norm,
      mlp_activation_fct=options.mlp_activation_fct
  )

  def init(key: chex.PRNGKey) -> Tuple[int, networks.ParamTree]:
    """Returns tuple of output dimension from the final layer and parameters."""
    params = {}
    key, subkey = jax.random.split(key)
    feature_dims, params['input'] = options.feature_layer.init()
    one_electron_feature_dim, _ = feature_dims
    # Concatenate spin of each electron with other one-electron features.
    feature_dim = one_electron_feature_dim + 1

    # Map to MLP dim.
    key, subkey = jax.random.split(key)
    params['embed'] = network_blocks.init_linear_layer(
        subkey, in_dim=feature_dim, out_dim=options.mlp_dim, include_bias=False
    )['w']

    # MLP block params.
    key, subkey = jax.random.split(key)
    params.update(SlaterNet_init(key))

    return options.mlp_dim, params

  def apply(
      params,
      *,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      spins: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Applies the SlaterNet interaction layers to a walker configuration.

    Args:
      params: parameters for the interaction and permuation-equivariant layers.
      ae: electron-nuclear vectors.
      r_ae: electron-nuclear distances.
      ee: electron-electron vectors.
      r_ee: electron-electron distances.
      spins: spin of each electron.
      charges: nuclear charges.

    Returns:
      Array of shape (nelectron, output_dim), where the output dimension,
      output_dim, is given by init, and is suitable for projection into orbital
      space.
    """
    del charges  # Unused.

    # Only one-electron features are used by SlaterNet.
    ae_features, _ = options.feature_layer.apply(
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **params['input']
    )

    # For Hartree-Fock, the spin feature is required for correct permutation equivariance, as in PsiFormer.
    ae_features = jnp.concatenate((ae_features, spins[..., None]), axis=-1)

    features = ae_features  # Just 1-electron stream for now.

    # Embed into attention dimension.
    x = jnp.dot(features, params['embed'])

    return SlaterNet_apply(params, x)

  return init, apply


def make_fermi_net(
    nspins: Tuple[int, ...],
    charges: jnp.ndarray,
    *,
    ndim: int = 3,
    determinants: int = 1,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[networks.FeatureLayer] = None,
    jastrow: Union[str, jastrows.JastrowType] = jastrows.JastrowType.SIMPLE_EE,
    jastrow_kwargs: dict = {},
    complex_output: bool = False,
    bias_orbitals: bool = False,
    # SlaterNet-specific kwargs below.
    num_layers: int,
    mlp_dim: int,
    num_perceptrons_per_layer: int,
    use_layer_norm: bool,
    pbc_lattice: jnp.ndarray,
    mlp_activation_fct: str,
) -> networks.Network:
  """SlaterNet implementation of Hartree Fock.

  Includes standard envelope and determinants.

  Args:
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    ndim: Dimension of the system. Change only with caution.
    determinants: Number of determinants.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    jastrow: Type of Jastrow factor if used, or 'none' if 'default'.
    complex_output: If true, the wavefunction output is complex-valued.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    mlp_dim: dimension of the perceptron layers
    num_perceptrons_per_layer: number of perceptrons between residual addition and layer norm
    use_layer_norm: If true, use layer_norm on MLP.
    mlp_activation_fct: Which non-linearity to use in the MLP, one of TANH, ELU, GELU

  Returns:
    Network object containing init, apply, orbitals, options, where init and
    apply are callables which initialise the network parameters and apply the
    network respectively, orbitals is a callable which applies the network up to
    the orbitals, and options specifies the settings used in the network.
  """

  if not envelope:
    envelope = envelopes.make_isotropic_envelope()

  if not feature_layer:
    natoms = charges.shape[0]
    feature_layer = networks.make_ferminet_features(
        natoms, nspins, ndim=ndim, )

  # By default, use no Jastrow factor
  if isinstance(jastrow, str):
    if jastrow.upper() == 'DEFAULT':
      jastrow = jastrows.JastrowType.NONE
    else:
      jastrow = jastrows.JastrowType[jastrow.upper()]

  options = SlaterNetOptions(
      ndim=ndim,
      determinants=determinants,
      envelope=envelope,
      feature_layer=feature_layer,
      jastrow=jastrow,
      jastrow_kwargs=jastrow_kwargs,
      complex_output=complex_output,
      bias_orbitals=bias_orbitals,
      num_layers=num_layers,
      mlp_dim=mlp_dim,
      num_perceptrons_per_layer=num_perceptrons_per_layer,
      use_layer_norm=use_layer_norm,
      mlp_activation_fct=mlp_activation_fct,
      pbc_lattice = pbc_lattice,
  )  # pytype: disable=wrong-keyword-args

  SlaterNet_layers = make_SlaterNet_layers(nspins, charges.shape[0], options)

  orbitals_init, orbitals_apply = networks.make_orbitals(
      nspins=nspins,
      charges=charges,
      options=options,
      equivariant_layers=SlaterNet_layers,
  )

  def network_init(key: chex.PRNGKey) -> networks.ParamTree:
    return orbitals_init(key)

  def network_apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward evaluation of SlaterNet.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with nuclear charges.

    Returns:
      Output of antisymmetric neural network in log space, i.e. a tuple of sign
      of and log absolute value of the network evaluated at x.
    """
    orbitals = orbitals_apply(params, pos, spins, atoms, charges)
    result = network_blocks.logdet_matmul(orbitals)
    return result

  return networks.Network(
      options=options,
      init=network_init,
      apply=network_apply,
      orbitals=orbitals_apply,
  )
