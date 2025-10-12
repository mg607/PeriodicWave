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

""" Defines supercell lattice vectors for two-dimensional periodic systems. """

import numpy as np
from typing import Tuple

def largest_two_divisors(n):
    if n == 1:
        return 1, 1 
    elif n <= 0:
        return None, None  # No valid divisors for numbers < 1
    for i in range(int(np.sqrt(n)), 0, -1):
        if n % i == 0:  # Check if i is a divisor
            return n // i, i
    return None, None  # This should never happen for n > 1

def send_positions_to_first_unit_cell(positions, lattice, rec):
    phase_ee = np.einsum('il,kl->ki', rec / (2 * np.pi), positions)
    phase_prim_ee = (phase_ee + 0.5)  % 1 - 0.5
    prim_ee = np.einsum('il,kl->ki', lattice, phase_prim_ee)
    return prim_ee

def _square_lattice_vecs(L: float) -> np.ndarray:
    """Returns square lattice vectors for 2D with Wigner-Seitz radius rs."""
    return L * np.eye(2)  # 2D identity matrix for square lattice

def _triangular_lattice_vecs_periodic_potential(a: float, num_sites: int, return_lattice_M = False) -> np.ndarray:
    """Returns triangular lattice vectors for 2D with Wigner-Seitz radius rs."""
    # Calculate the area of the unit cell per electron, the total area is n_elec * area_per_electron
    # area_per_electron = np.pi * (rs**2)  # 2D area calculation

    # For a triangular lattice, the area of the primitive cell is sqrt(3)/2 * (a^2)
    # where a is the lattice constant (side length of the primitive cell).
    # a = (2 * area_per_electron / (np.sqrt(3)))**0.5  # Lattice constant

    periodic_potential_lattice_vectors = np.array([
        [a, -a / 2],
        [0, a * np.sqrt(3) / 2]
    ]) 
    a1 = periodic_potential_lattice_vectors[:,0]
    a2 = periodic_potential_lattice_vectors[:,1]
    
    # collection of C6 symmetric supercells
    if num_sites == 3:
        M = np.array([[1, 1], [2, -1]])
    elif num_sites == 12:
        M = np.array([[2, 2], [4, -2]])
    elif num_sites == 21:
        M = np.array([[1, 4], [5, -1]])
    elif num_sites == 27:
        M = np.array([[3, 3], [6, -3]])
    else:
        multipliers_lattice_vectors = largest_two_divisors(num_sites)
        M = np.array([[multipliers_lattice_vectors[0], 0], [0, multipliers_lattice_vectors[1]]])

    L1 = M[0,0]*a1 + M[1,0]*a2
    L2 = M[0,1]*a1 + M[1,1]*a2
    supercell_lattice_vectors = np.zeros((2,2))
    supercell_lattice_vectors[:,0] = L1
    supercell_lattice_vectors[:,1] = L2
    
    if return_lattice_M:
        return supercell_lattice_vectors, periodic_potential_lattice_vectors, M
    else:
        return supercell_lattice_vectors, periodic_potential_lattice_vectors


