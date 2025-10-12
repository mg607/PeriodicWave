# Copyright (c) 2025 Max Geier, Khachatur Nazaryan, Massachusetts Institute of Technology, MA, USA
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

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

def convert_moire_scales(me_eff_rel = 0.35, eps_inverse = 0.2, moire_a = 8.031, moire_potential_strength = 15):
    """ 
    Converts moire system parameters from SI to natural units used in the code.

    Default values as used in M Geier, K Nazaryan, T Zaklama, and L Fu, Phys. Rev. B 112, 045119

    Arguments:
        me_eff_rel = 0.35             # effective mass in units of bare electron mass
        moire_a = 8.031               # moire lattice constant, nm
        eps_inverse = 0.1             # inverse relative diel. constant of the surrounding dielectric
        moire_potential_strength = 15 # energy scale V of the hexagonal moire potential, meV 

    Returns:
        energy_scale: Conversion factor between energies returned from the code and SI units (meV)
        V: moire potential strength in natural units
        U: interaction energy scale in natural units
    """
    # Natural constants
    a_0  = 5.29177210544e-2      # Bohr radius, nm
    hbar = 6.582119569509066e-1  # meV * ps
    me   = 5.685630060215049e-3  # electron rest mass, meV/[(nm/ps)^2]

    # effective mass in SI units (meV/[(nm/ps)^2])
    me_eff = me_eff_rel * me      # effective mass, meV/[(nm/ps)^2]

    # scales converting dimensionless units from the code to SI units
    # Here we present units where distances are measured in terms of the moire lattice constant moire_a
    # and energies are measured in \hbar^2 / moire_a^2 / me_eff,
    energy_scale = hbar**2 / moire_a**2 / me_eff # meV
    # In these units, the kinetic energy term is
    # - 0.5 \sum_j \nabla_j^2 where j runs over all electrons.
    # In these units, the dimensionless moire potential strength is V, determining
    # - 2 V \sum_i \sum_{n = 1}^{3} \cos ( g_n \cdot r_i + \phi )
    V = moire_potential_strength / energy_scale 
    # The dimensionless Coulomb interaction energy scale is U, determining
    # 0.5 U \sum_{i \neq j} 1/{|r_j - r_i|}
    U = (moire_a / a_0) * eps_inverse * (me_eff / me)

    return energy_scale, V, U

def load_csv_data(folder_path, file_name):
    # Construct the full path to the CSV file
    file_path = os.path.join(folder_path, file_name)
    
    # Load the CSV file, only reading the first 5 columns
    data = pd.read_csv(file_path, usecols=["step", "energy", "ewmean", "ewvar", "pmove"])
    
    return data

ndim = 2 # spatial dimension of the system
network_type = "CustomPsiformer"

# system parameters
potential_type = "CoulombMoire"
num_unit_cells = 9
nspins = (6, 0)
num_electrons = sum(nspins)
me_eff_rel = 0.35 # in units of bare electron mass
eps_inverse = 0.1 # inverse dielectric constant of surrounding dielectric
moire_lattice_constant_nm = 8.031 # in nm
moire_potential_strength_meV = 15 # in meV
moire_potential_phi = 45.0 # potential shape angle in degrees

# Convert SI units to natural units
energy_scale, moire_potential_strength, interaction_energy_scale = convert_moire_scales(me_eff_rel, eps_inverse, moire_lattice_constant_nm, moire_potential_strength_meV)

# generate folder name
folder_name = f"results/2deg-CoulombMoire/{network_type}/el{nspins[0]}_{nspins[1]}_N{num_unit_cells}_V{np.round(moire_potential_strength,8)}_{moire_potential_phi}_U{np.round(interaction_energy_scale,8)}"
train_data = load_csv_data(folder_name, "train_stats.csv")

fig, ax = plt.subplots(1,1, figsize = (7,5))
ax.plot(train_data['step'][10000:], train_data['energy'][10000:] * energy_scale / num_electrons, marker='o', linestyle='-', linewidth=0.4, markersize=1, alpha=0.4)
ax.set_xlabel("step")
ax.set_ylabel("energy (meV)")
