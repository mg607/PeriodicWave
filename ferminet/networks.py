# Copyright 2020 DeepMind Technologies Limited.
# Modifications Copyright (c) 2025 Max Geier, Massachusetts Institute of Technology, MA, USA
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
# - Streamlined to keep only functions relevant for attention architecture
# - included make_periodic_construct_input_features that correctly estimates distances on a periodic system

"""Implementation of Fermionic Neural Network in JAX."""
import enum
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import attr
import chex
from ferminet import envelopes
from ferminet import jastrows
from ferminet import network_blocks
import jax
import jax.numpy as jnp
from typing_extensions import Protocol

FermiLayers = Tuple[Tuple[int, int], ...]
# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet
ParamTree = Union[
    jnp.ndarray, Iterable['ParamTree'], MutableMapping[Any, 'ParamTree']
]
# pytype: enable=not-supported-yet
# Parameters for a single part of the network are just a dict.
Param = MutableMapping[str, jnp.ndarray]


@chex.dataclass
class FermiNetData:
  """Data passed to network.

  Shapes given for an unbatched element (i.e. a single MCMC configuration).

  NOTE:
    the networks are written in batchless form. Typically one then maps
    (pmap+vmap) over every attribute of FermiNetData (nb this is required if
    using KFAC, as it assumes the FIM is estimated over a batch of data), but
    this is not strictly required. If some attributes are not mapped, then JAX
    simply broadcasts them to the mapped dimensions (i.e. those attributes are
    treated as identical for every MCMC configuration.

  Attributes:
    positions: walker positions, shape (nelectrons*ndim).
    spins: spins of each walker, shape (nelectrons).
    atoms: atomic positions, shape (natoms*ndim).
    charges: atomic charges, shape (natoms).
  """

  # We need to be able to construct instances of this with leaf nodes as jax
  # arrays (for the actual data) and as integers (to use with in_axes for
  # jax.vmap etc). We can type the struct to be either all arrays or all ints
  # using Generic, it just slightly complicates the type annotations in a few
  # functions (i.e. requires FermiNetData[jnp.ndarray] annotation).
  positions: Any
  spins: Any
  atoms: Any
  charges: Any


## Interfaces (public) ##


class InitFermiNet(Protocol):

  def __call__(self, key: chex.PRNGKey) -> ParamTree:
    """Returns initialized parameters for the network.

    Args:
      key: RNG state
    """


class FermiNetLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      electrons: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the sign and log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
      spins: 1D array specifying the spin state of each electron.
      atoms: positions of nuclei, shape: (natoms, ndim).
      charges: nuclei charges, shape: (natoms).
    """


class LogFermiNetLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      electrons: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns the log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
      spins: 1D array specifying the spin state of each electron.
      atoms: positions of nuclei, shape: (natoms, ndim).
      charges: nuclear charges, shape: (natoms).
    """


class OrbitalFnLike(Protocol):

  def __call__(
      self,
      params: ParamTree,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward evaluation of the Fermionic Neural Network up to the orbitals.

    Args:
      params: network parameter tree.
      pos: The electron positions, a 3N dimensional vector.
      spins: The electron spins, an N dimensional vector.
      atoms: Array with positions of atoms.
      charges: Array with atomic charges.

    Returns:
      Sequence of orbitals.
    """


class InitLayersFn(Protocol):

  def __call__(self, key: chex.PRNGKey) -> Tuple[int, ParamTree]:
    """Returns output dim and initialized parameters for the interaction layers.

    Args:
      key: RNG state
    """


class ApplyLayersFn(Protocol):

  def __call__(
      self,
      params: ParamTree,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      spins: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> jnp.ndarray:
    """Forward evaluation of the equivariant interaction layers.

    Args:
      params: parameters for the interaction and permutation-equivariant layers.
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


## Interfaces (network components) ##


class FeatureInit(Protocol):

  def __call__(self) -> Tuple[Tuple[int, int], Param]:
    """Creates the learnable parameters for the feature input layer.

    Returns:
      Tuple of ((x, y), params), where x and y are the number of one-electron
      features per electron and number of two-electron features per pair of
      electrons respectively, and params is a (potentially empty) mapping of
      learnable parameters associated with the feature construction layer.
    """


class FeatureApply(Protocol):

  def __call__(
      self,
      ae: jnp.ndarray,
      r_ae: jnp.ndarray,
      ee: jnp.ndarray,
      r_ee: jnp.ndarray,
      **params: jnp.ndarray,
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Creates the features to pass into the network.

    Args:
      ae: electron-atom vectors. Shape: (nelectron, natom, 3).
      r_ae: electron-atom distances. Shape: (nelectron, natom, 1).
      ee: electron-electron vectors. Shape: (nelectron, nelectron, 3).
      r_ee: electron-electron distances. Shape: (nelectron, nelectron).
      **params: learnable parameters, as initialised in the corresponding
        FeatureInit function.
    """


@attr.s(auto_attribs=True)
class FeatureLayer:
  init: FeatureInit
  apply: FeatureApply


class FeatureLayerType(enum.Enum):
  STANDARD = enum.auto()


class MakeFeatureLayer(Protocol):

  def __call__(
      self,
      natoms: int,
      nspins: Sequence[int],
      ndim: int,
      **kwargs: Any,
  ) -> FeatureLayer:
    """Builds the FeatureLayer object.

    Args:
      natoms: number of atoms.
      nspins: tuple of the number of spin-up and spin-down electrons.
      ndim: dimension of the system.
      **kwargs: additional kwargs to use for creating the specific FeatureLayer.
    """


## Network settings ##


@attr.s(auto_attribs=True, kw_only=True)
class BaseNetworkOptions:
  """Options controlling the overall network architecture.

  Attributes:
    ndim: dimension of system. Change only with caution.
    determinants: Number of determinants to use.
    states: Number of outputs, one per excited (or ground) state. Ignored if 0.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    envelope: Envelope object to create and apply the multiplicative envelope.
    feature_layer: Feature object to create and apply the input features for the
      one- and two-electron layers.
    jastrow: Type of Jastrow factor if used, or 'none' if no Jastrow factor.
    complex_output: If true, the network outputs complex numbers.
  """

  ndim: int = 3
  determinants: int = 16
  states: int = 0
  bias_orbitals: bool = False
  envelope: envelopes.Envelope = attr.ib(
      default=attr.Factory(
          envelopes.make_isotropic_envelope,
          takes_self=False))
  feature_layer: FeatureLayer = None
  jastrow: jastrows.JastrowType = jastrows.JastrowType.NONE
  jastrow_kwargs: dict = {}
  complex_output: bool = False
  pbc_lattice: jnp.ndarray = jnp.array([])


# Network class.
@attr.s(auto_attribs=True)
class Network:
  options: BaseNetworkOptions
  init: InitFermiNet
  apply: FermiNetLike
  orbitals: OrbitalFnLike


## Network layers: features ##
def make_periodic_construct_input_features(pbc_supercell: jnp.ndarray) -> ...:
  # WARNING: Using periodic norm for construct_input_features distorts the distances r_ee, r_ae on
  # length scales of the unit cell boundary. 
  # It only agrees with the real-space norm on short length scales.
  print("making construct_input_features with periodic norm")
  rec = 2 * jnp.pi * jnp.linalg.inv(pbc_supercell)
  supercell_metric =  pbc_supercell.T @ pbc_supercell

  # periodic_norm copied from ferminet.pbc.feature_layer
  def periodic_norm(metric: jnp.ndarray, scaled_r: jnp.ndarray) -> jnp.ndarray:
    """Returns the periodic norm of a set of vectors.

    Args:
      metric: metric tensor in fractional coordinate system, A.T A, where A is the
        lattice vectors.
      scaled_r: vectors in fractional coordinates of the lattice cell, with
        trailing dimension ndim, to compute the periodic norm of.
    """
    chex.assert_rank(metric, expected_ranks=2)
    a = (1 - jnp.cos(2 * jnp.pi * scaled_r))
    b = jnp.sin(2 * jnp.pi * scaled_r)
    cos_term = jnp.einsum('...m,mn,...n->...', a, metric, a)
    sin_term = jnp.einsum('...m,mn,...n->...', b, metric, b)
    return (1 / (2 * jnp.pi)) * jnp.sqrt(cos_term + sin_term)

  def construct_input_features(
      pos: jnp.ndarray,
      atoms: jnp.ndarray,
      ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Constructs inputs to Fermi Net from raw electron and atomic positions.

    Args:
      pos: electron positions. Shape (nelectrons*ndim,).
      atoms: atom positions. Shape (natoms, ndim).
      ndim: dimension of system. Change only with caution.

    Returns:
      ae, ee, r_ae, r_ee tuple, where:
        ae: atom-electron vector. Shape (nelectron, natom, ndim).
        ee: atom-electron vector. Shape (nelectron, nelectron, ndim).
        r_ae: atom-electron distance. Shape (nelectron, natom, 1).
        r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
      The diagonal terms in r_ee are masked out such that the gradients of these
      terms are also zero.
    """
    # jax.debug.print("atoms, ndim = {}", [atoms, ndim])
    assert atoms.shape[1] == ndim
    ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
    ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
    # convert to position in fraction of unit cell lattice vectors for application of periodic norm
    s_ae = jnp.einsum('il,jkl->jki', rec / 2 / jnp.pi, ae)
    s_ee = jnp.einsum('il,jkl->jki', rec / 2 / jnp.pi, ee)
    
    r_ae = periodic_norm(supercell_metric, s_ae)
    # Avoid computing the norm of zero, as it has undefined grad
    n = ee.shape[0]
    r_ee = (
        periodic_norm(supercell_metric, s_ee + jnp.eye(n)[..., None]) * (1.0 - jnp.eye(n)))

    return ae, ee, r_ae, r_ee[..., None]
    
  return construct_input_features

# Default construct input features used with default Hamiltonian
def construct_input_features(
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    ndim: int = 3) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
  """Constructs inputs to Fermi Net from raw electron and atomic positions.

  Args:
    pos: electron positions. Shape (nelectrons*ndim,).
    atoms: atom positions. Shape (natoms, ndim).
    ndim: dimension of system. Change only with caution.

  Returns:
    ae, ee, r_ae, r_ee tuple, where:
      ae: atom-electron vector. Shape (nelectron, natom, ndim).
      ee: atom-electron vector. Shape (nelectron, nelectron, ndim).
      r_ae: atom-electron distance. Shape (nelectron, natom, 1).
      r_ee: electron-electron distance. Shape (nelectron, nelectron, 1).
    The diagonal terms in r_ee are masked out such that the gradients of these
    terms are also zero.
  """
  assert atoms.shape[1] == ndim
  ae = jnp.reshape(pos, [-1, 1, ndim]) - atoms[None, ...]
  ee = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])

  r_ae = jnp.linalg.norm(ae, axis=2, keepdims=True)
  # Avoid computing the norm of zero, as it has undefined grad
  n = ee.shape[0]
  r_ee = (
      jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n)))
  return ae, ee, r_ae, r_ee[..., None]


def make_ferminet_features(
    natoms: int,
    nspins: Optional[Tuple[int, int]] = None,
    ndim: int = 3,
) -> FeatureLayer:
  """Returns the init and apply functions for the standard features."""

  del nspins

  def init() -> Tuple[Tuple[int, int], Param]:
    return (natoms * (ndim + 1), ndim + 1), {}

  def apply(ae, r_ae, ee, r_ee) -> Tuple[jnp.ndarray, jnp.ndarray]:
    ae_features = jnp.concatenate((r_ae, ae), axis=2)
    ee_features = jnp.concatenate((r_ee, ee), axis=2)
    ae_features = jnp.reshape(ae_features, [jnp.shape(ae_features)[0], -1])
    return ae_features, ee_features

  return FeatureLayer(init=init, apply=apply)


## Network layers: permutation-equivariance ##
def make_orbitals(
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    options: BaseNetworkOptions,
    equivariant_layers: Tuple[InitLayersFn, ApplyLayersFn]
) -> ...:
  """Returns init, apply pair for orbitals with spin-channels merged.

  All electrons are treated as a single channel (n_total, 0), matching the
  behavior of the original code path when use_spin_channels=False.
  """
  equivariant_layers_init, equivariant_layers_apply = equivariant_layers

  # Optional Jastrow factor.
  jastrow_init, jastrow_apply = jastrows.get_jastrow(
      options.jastrow, options.jastrow_kwargs
  )

  if options.pbc_lattice is not None:
    # periodic features for correct PBC treatment (esp. Jastrow)
    construct_input_features_internal = make_periodic_construct_input_features(options.pbc_lattice)
  else:
    construct_input_features_internal = construct_input_features

  # Merge spin channels: treat all electrons in one channel
  n_total = int(sum(nspins))
  nspins_merged = (n_total, 0)  # keep tuple shape for downstream utilities

  def init(key: chex.PRNGKey) -> ParamTree:
    """Initialise parameters."""
    if n_total == 0:
      raise ValueError('No electrons present!')

    key, subkey = jax.random.split(key)
    params = {}
    dims_orbital_in, params['layers'] = equivariant_layers_init(subkey)

    # Number of spin-orbitals for the single active channel
    num_states = max(options.states, 1)
    norbitals = n_total * options.determinants * num_states
    if options.complex_output:
      norbitals *= 2  # pack real/imag into even/odd outputs

    # Envelope params
    natom = charges.shape[0]
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      output_dims = dims_orbital_in
    elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      output_dims = [norbitals // 2] if options.complex_output else [norbitals]
    else:
      raise ValueError('Unknown envelope type')
    params['envelope'] = options.envelope.init(
        natom=natom, output_dims=output_dims, ndim=options.ndim
    )

    # Jastrow params
    if jastrow_init is not None:
      params['jastrow'] = jastrow_init()

    # Orbital-shaping linear layer (single channel)
    key, subkey = jax.random.split(key)
    params['orbital'] = [
        network_blocks.init_linear_layer(
            subkey,
            in_dim=dims_orbital_in,
            out_dim=norbitals,
            include_bias=options.bias_orbitals,
        )
    ]

    return params

  def apply(
      params,
      pos: jnp.ndarray,
      spins: jnp.ndarray,
      atoms: jnp.ndarray,
      charges: jnp.ndarray,
  ) -> Sequence[jnp.ndarray]:
    """Forward pass up to (merged) orbitals.

    Returns:
      A single matrix list [orbital_mat] with shape (ndet*states, n_total, n_total).
    """
    # Inputs / equivariant trunk
    ae, ee, r_ae, r_ee = construct_input_features_internal(pos, atoms, ndim=options.ndim)
    h = equivariant_layers_apply(
        params['layers'],
        ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee,
        spins=spins, charges=charges,
    )

    # PRE_ORBITAL envelope
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_ORBITAL:
      envelope_factor = options.envelope.apply(
          ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope']
      )
      h = envelope_factor * h

    # Linear projection to orbitals (single channel)
    orbital = network_blocks.linear_layer(h, **params['orbital'][0])

    # Complex packing
    if options.complex_output:
      orbital = orbital[..., ::2] + 1.0j * orbital[..., 1::2]

    # PRE_DETERMINANT envelope
    if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
      # With one channel, just apply params['envelope'][0] to the whole set
      orbital = orbital * options.envelope.apply(
          ae=ae, r_ae=r_ae, r_ee=r_ee, **params['envelope'][0]
      )

    # Reshape to (ndet*states, n_total, n_total)
    orbital = jnp.reshape(orbital, (n_total, -1, n_total))  # (elec, blocks, elec)
    orbital = jnp.transpose(orbital, (1, 0, 2))             # (blocks, elec, elec)

    orbitals = [orbital]  # already "concatenated" (only one channel)

    # Jastrow factor (pre-determinant for compatibility with pretraining)
    if jastrow_apply is not None:
      jastrow = jnp.exp(jastrow_apply(r_ee, params['jastrow'], nspins_merged) / n_total)
      orbitals = [o * jastrow for o in orbitals]

    return orbitals

  return init, apply
