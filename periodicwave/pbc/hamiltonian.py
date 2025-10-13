# Copyright (c) 2025 Max Geier and Khachatur Nazaryan, 
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

"""
Contains: 
- Ewald summation of Coulomb Hamiltonian in periodic boundary conditions.
- Triangular potential
- Local energy evaluation

See M Geier, K Nazaryan, T Zaklama, and L Fu, Phys. Rev. B 112, 045119
"""

import itertools
from typing import Callable, Optional, Sequence, Tuple

import chex
from periodicwave import hamiltonian
from periodicwave import networks
import jax
import jax.numpy as jnp
import numpy as np

from periodicwave.utils import utils
import folx

def make_triangular_potential(
    potential_lattice: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    potential_strength: float,
    phi: float
) -> Callable[[jnp.ndarray, jnp.ndarray], float]: #callable:
    """Creates a function to evaluate a triangular periodic potential."""
    del atoms, charges # unused
    a_M = potential_lattice[0, 0]  # Extract fixed moire period
    g1 = (4 * jnp.pi / (jnp.sqrt(3) * a_M)) * jnp.array([0, 1])
    g2 = (4 * jnp.pi / (jnp.sqrt(3) * a_M)) * jnp.array([jnp.sin(2 * jnp.pi / 3), jnp.cos(2 * jnp.pi / 3)])
    g3 = - (g1 + g2)
    Gs = [g1, g2, g3] # all three recip. vectors are related by 120 degree rotation
    print("make_triangular_potential: using [U, phi/pi] = " + str([potential_strength, phi / jnp.pi]))

    def potential(ae: jnp.ndarray):
      ae = jnp.reshape(ae, [-1, 2])
      return -2 * potential_strength * jnp.sum(jnp.array([
          jnp.cos(jnp.dot(aei, Gs[0]) + phi)
        + jnp.cos(jnp.dot(aei, Gs[1]) + phi)
        + jnp.cos(jnp.dot(aei, Gs[2]) + phi) for aei in ae]))
    return potential

def make_2DCoulomb_potential(
    lattice: jnp.ndarray,
    atoms: jnp.ndarray,
    charges: jnp.ndarray,
    truncation_limit: int = 5,
    interaction_energy_scale: float = 1.0,
) -> Callable[[jnp.ndarray, jnp.ndarray], float]:
  """Creates a function to evaluate infinite Coulomb sum for periodic lattice.
  A homogeneous charge background is subtracted, which is necessary because otherwise energies are infinite.
  The evaluation also contains the Madelung constant contribution, 
  i.e. the energy of particles from interaction with their own images in neighboring supercell.

    Args:
        lattice: Shape (2, 2). Matrix whose columns are the primitive lattice vectors.
        atoms: Shape (natoms, 2). Positions of the atoms.
        charges: Shape (natoms). Nuclear charges of the atoms.
        nspins: Tuple of the number of spin-up and spin-down electrons.
        truncation_limit: Integer. Half side length of square of nearest neighbours
        to primitive cell which are summed over in evaluation of Ewald sum.
        interaction_energy_scale: energy scale that multiplies the Coulomb interaction energy

    Returns:
        Callable with signature f(ae, ee, spins), where (ae, ee) are atom-electron and
        electron-electron displacement vectors respectively, and spins are electron spins,
        which evaluates the Coulomb sum for the periodic lattice via the Ewald method.
  """
  print("making 2DCoulomb potential with energy scale: " + str(interaction_energy_scale))
  del atoms, charges # unused
  rec = 2 * jnp.pi * jnp.linalg.inv(lattice)
  area = jnp.abs(jnp.linalg.det(lattice)) #area for 2D
  # the factor gamma tunes the width of the summands in real / reciprocal space
  # and this value is chosen to optimize the convergence trade-off between the
  # two sums. See CASINO QMC manual.
  gamma_factor = 2.8
  gamma = (gamma_factor / area**(1 / 2))**2  # Adjusted for 2D systems

  ordinals = sorted(range(-truncation_limit, truncation_limit + 1), key=abs)
  ordinals = jnp.array(list(itertools.product(ordinals, repeat=2)))  # Adjusted for 2D
  lat_vectors = jnp.einsum('kj,ij->ik', lattice, ordinals)
  rec_vectors = jnp.einsum('jk,ij->ik', rec, ordinals[1:])
  rec_vec_square = jnp.einsum('ij,ij->i', rec_vectors, rec_vectors)
  rec_vec_magnitude = jnp.sqrt(rec_vec_square) # |rec_vectors|
  lat_vec_norm = jnp.linalg.norm(lat_vectors[1:], axis=-1)

  def real_space_ewald(separation: jnp.ndarray):
      """Real-space Ewald potential between charges in 2D.
      """
      displacements = jnp.linalg.norm(separation - lat_vectors, axis=-1)  # |r - R|

      return jnp.sum(
          jax.scipy.special.erfc(gamma**0.5 * displacements) / displacements)

  def recp_space_ewald(separation: jnp.ndarray):
      """Reciprocal-space Ewald potential between charges in 2D.
      """
      # phase = jnp.exp(1.0j * jnp.dot(rec_vectors, separation))
      phase = jnp.cos(jnp.dot(rec_vectors, separation))

      factor = jax.scipy.special.erfc(rec_vec_magnitude / (2 * gamma**0.5) )
      return (2 * jnp.pi / area) * jnp.sum( phase * factor / rec_vec_magnitude)

  def ewald_sum(separation: jnp.ndarray):
      """Combined real and reciprocal space Ewald potential in 2D.
      """
      return real_space_ewald(separation) + recp_space_ewald(separation)
      
  # Compute Madelung constant components
  # Real-space part
  madelung_real = jnp.sum(
      jax.scipy.special.erfc(gamma**0.5 * lat_vec_norm) / lat_vec_norm
  )

  # q = 0 contribution of the real-space part, see discussion around eq. (A8) in Geier PRB 112, 045119 (2025)
  phi_S_q0 = (2 * jnp.pi) / area / gamma**0.5 / jnp.pi**0.5

  # Reciprocal-space part
  xi_L_0 = 2 * gamma**0.5 / jnp.pi**0.5
  madelung_recip = - 0*(2 * jnp.pi / area) * (1 / (gamma**0.5 * jnp.pi**0.5)) + \
      (2 * jnp.pi / area) * jnp.sum(
          jax.scipy.special.erfc(rec_vec_magnitude / (2 * gamma**0.5)) / rec_vec_magnitude
      ) - xi_L_0

  # Total Madelung constant
  madelung_const = madelung_real + madelung_recip

  batch_ewald_sum = jax.vmap(ewald_sum, in_axes=(0,))

  def electron_electron_potential(ee: jnp.ndarray):
      """Evaluates periodic electron-electron potential with charges.

      We always include neutralizing background term for homogeneous electron gas.
      """
      nelec = ee.shape[0]
      ee = jnp.reshape(ee, [-1, 2])
      ewald = batch_ewald_sum(ee)
      ewald = jnp.reshape(ewald, [nelec, nelec])
      # Set diagonal elements to zero (self-interaction)
      ewald = ewald.at[jnp.diag_indices(nelec)].set(0.0)

      # Add Madelung constant term: (1/2) * N * q_i^2 * Madelung_const
      # Since q_i^2 = 1, this simplifies to (1/2) * N * Madelung_const
      potential = 0.5 * jnp.sum(ewald) + 0.5 * nelec * madelung_const - 0.5 * nelec**2 * phi_S_q0
      return potential

  def potential(ae: jnp.ndarray, ee: jnp.ndarray):
    """Accumulates atom-electron, atom-atom, and electron-electron potential."""
    # Reduce vectors into first unit cell
    del ae # for HEG calculations, there are no atoms
    phase_ee = jnp.einsum('il,jkl->jki', rec / (2 * jnp.pi), ee)
    phase_prim_ee = (phase_ee + 0.5)  % 1 - 0.5
    prim_ee = jnp.einsum('il,jkl->jki', lattice, phase_prim_ee)
    return interaction_energy_scale * jnp.real(
        electron_electron_potential(prim_ee)
    )

  return potential

def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    complex_output: bool = False,
    lattice: Optional[jnp.ndarray] = None,
    convergence_radius: int = 5,
    potential_type = 'Coulomb',
    potential_kwargs = {},
    kinetic_kwargs = {},
) -> hamiltonian.LocalEnergy:
  """Creates the local energy function in periodic boundary conditions.

  Args:
    f: Callable which returns the sign and log of the magnitude of the
      wavefunction given the network parameters and configurations data.
    charges: Shape (natoms). Nuclear charges of the atoms.
    nspins: Number of particles of each spin.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: Set to True if complex-valued wavefunctions are used.
    states: Number of excited states to compute. Not implemented, only present
      for consistency of calling convention.
    lattice: Shape (ndim, ndim). Matrix of lattice vectors. Default: identity
      matrix.
    convergence_radius: int. Radius of cluster summed over by Ewald sums.

  Returns:
    Callable with signature e_l(params, key, data) which evaluates the local
    energy of the wavefunction given the parameters params, RNG state key,
    and a single MCMC configuration in data.
  """
  print("Using customized local_energy from pbc.hamiltonian ")
  del nspins
  assert lattice is not None, "pbc.hamiltonian.local_energy requires lattice to be passed"

  if kinetic_kwargs['laplacian_method'] in {'default'}:
    print("local_energy, using laplacian_method: " + kinetic_kwargs['laplacian_method'])
    ke = hamiltonian.local_kinetic_energy(f, use_scan=use_scan,
                                          complex_output=complex_output,
                                          laplacian_method=kinetic_kwargs['laplacian_method'])
    
  elif kinetic_kwargs['laplacian_method'] in {'folx'}:
    print("local_energy, using folx with complex_output: " + str(complex_output))
    if complex_output:
      def _lapl_over_f(params, data):
        f_closure = lambda x: f(params, x, data.spins, data.atoms, data.charges)
        f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=0) # setting sparsity_threshold = 0, finite value cause crash
        output = f_wrapped(data.positions)
        result = - (output[1].laplacian +
                    jnp.sum(output[1].jacobian.dense_array ** 2)) / 2
        result -= 0.5j * output[0].laplacian
        result += 0.5 * jnp.sum(output[0].jacobian.dense_array ** 2)
        result -= 1.j * jnp.sum(output[0].jacobian.dense_array *
                                output[1].jacobian.dense_array)
        return jnp.real(result)
    else:
      def _lapl_over_f(params, data):
        f_closure = lambda x: f(params, x, data.spins, data.atoms, data.charges)
        f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=0) # setting sparsity_threshold = 0, finite value cause crash
        output = f_wrapped(data.positions)
        result = - (output[1].laplacian +
                    jnp.sum(output[1].jacobian.dense_array ** 2)) / 2
        return result
      
    ke = _lapl_over_f

  if potential_type in {'Coulomb','CoulombMoire'}:
    if 'interaction_energy_scale' in potential_kwargs:
      interaction_energy_scale = potential_kwargs['interaction_energy_scale']
    else:
      interaction_energy_scale = 1.0
      
    potential_energy = make_2DCoulomb_potential(
      lattice, jnp.array([0.0]), charges, convergence_radius, interaction_energy_scale
    )
  elif potential_type == 'NoneMoire':
    pass
  else:
     raise NotImplementedError("pbc.local_energy: potential_type not implemented.")

  if potential_type in {'Coulomb'}:
    def _e_l(
        params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
      """Returns the total energy.

      Args:
        params: network parameters.
        key: RNG state.
        data: MCMC configuration.
      """
      del key  # unused
      ae, ee, _, _ = networks.construct_input_features(
          data.positions, data.atoms, ndim=2)
      potential = potential_energy(ae, ee)
      kinetic = ke(params, data)
      return potential + kinetic, None


  if potential_type in {'CoulombMoire'}:
    moire_potential_energy = make_triangular_potential(potential_kwargs['moire_lattice_vectors'], 
                                                  jnp.array([0.0]), charges,
                                                    potential_kwargs['moire_potential_strength'],
                                                    potential_kwargs['moire_potential_phi'])
    def _e_l(
        params: networks.ParamTree, key: chex.PRNGKey, data: networks.FermiNetData
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
      """Returns the total energy.

      Args:
        params: network parameters.
        key: RNG state.
        data: MCMC configuration.
      """
      del key  # unused
      ae, ee, _, _ = networks.construct_input_features(
          data.positions, data.atoms, ndim=2)
      potential = potential_energy(ae, ee)
      kinetic = ke(params, data)
      moire_potential = moire_potential_energy(ae)
      return potential + kinetic + moire_potential, None

  return _e_l
