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

import numpy as np
from matplotlib import pyplot as plt
import os
from ferminet.pbc import lattices

def convert_moire_scales(me_eff_rel = 0.35, eps_inverse = 0.05, moire_a = 8.031, moire_potential_strength = 15):
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

def get_positions_from_latest_npz_files(folder_path, N):
    """
    Loads position data from N latest checkpoints of FermiNet ouput stored in
    a folder with path folder_path
    """
    # Find all files that match the "qmcjax_ckpt_XXXXXX.npz" pattern
    files = [
        f for f in os.listdir(folder_path) 
        if f.startswith("qmcjax_ckpt_") and f.endswith(".npz")
    ]
    
    # Extract the six-digit number from each filename and store it in a tuple (number, filename)
    file_numbers = []
    for file in files:
        try:
            number = int(file.split('_')[-1].split('.')[0])
            file_numbers.append((number, file))
        except ValueError:
            print(f"Skipping file {file}, could not extract valid six-digit number.")

    # Sort the files by the six-digit number in descending order
    sorted_files = sorted(file_numbers, key=lambda x: x[0], reverse=True)

    # Select the top N files with the largest numbers
    largest_files = sorted_files[:N]

    print("In folder " + folder_path + "/ loading checkpoints: ")
    print([el[0] for el in largest_files])

    # Load the contents of the selected files
    positions_ckpt = []
    spins_ckpt = []
    for number, filename in largest_files:
        file_path = os.path.join(folder_path, filename)
        try:
            ckpt_data = np.load(file_path, allow_pickle=True)
            print(list(ckpt_data.keys()))
            data = ckpt_data['data'].item()
            print(list(data.keys()))
            positions_ckpt.append(data['positions'])
            spins_ckpt.append(data['spins'])
            # loaded_data[filename] = data
        except Exception as e:
            print(f"Failed to load {filename}: {e}")

    filenames = [el[1] for el in largest_files]

    return positions_ckpt, spins_ckpt, filenames

# system parameters
potential_type = "CoulombMoire"
network_type = "CustomPsiformer"
ndim = 2
num_unit_cells = 9
nspins = (6, 0)
num_electrons = sum(nspins)
me_eff_rel = 0.35 # in units of bare electron mass
eps_inverse = 0.15 # inverse dielectric constant of surrounding dielectric
moire_lattice_constant_nm = 8.031 # in nm
moire_potential_strength_meV = 15 # in meV
moire_potential_phi = 45.0 # potential shape angle in degrees

# Convert SI units to natural units
energy_scale, moire_potential_strength, interaction_energy_scale = convert_moire_scales(me_eff_rel, eps_inverse, moire_lattice_constant_nm, moire_potential_strength_meV)

# generate folder name
folder_name = f"results/2deg-CoulombMoire/{network_type}/el{nspins[0]}_{nspins[1]}_N{num_unit_cells}_V{np.round(moire_potential_strength,8)}_{moire_potential_phi}_U{np.round(interaction_energy_scale,8)}"

moire_a = 1.0
lat_vec, moire_lat_vec, lattice_M = lattices._triangular_lattice_vecs_periodic_potential(a = moire_a, num_sites=num_unit_cells, return_lattice_M = True)
rec = 2 * np.pi * np.linalg.inv(lat_vec)
area = np.linalg.det(lat_vec)
r_s = np.sqrt(area / sum(nspins) / np.pi)

load_N_ckpts = 3 # number of latest checkpoints to load 

# load configurations from latest checkpoints
positions_ckpt, spins_ckpt, filenames = get_positions_from_latest_npz_files(folder_name, load_N_ckpts)
shape_pos = np.shape(positions_ckpt)
positions_batch = np.reshape(np.array(positions_ckpt),(shape_pos[0]*shape_pos[1]*shape_pos[2], shape_pos[3]//ndim, ndim))
positions_batch = np.array([lattices.send_positions_to_first_unit_cell(config, lat_vec, rec) for config in positions_batch])

# plot electron density
fig_n, ax_n = plt.subplots(1, 1, figsize = (7, 5))
positions_plot = np.reshape(np.array(positions_batch),(shape_pos[0]*shape_pos[1]*shape_pos[2]*shape_pos[3]//ndim, ndim))
ax_n.scatter(positions_plot[:,0], positions_plot[:,1], color="tab:blue", s=1, alpha=0.2)
ax_n.set_xlabel("x / a_M")
ax_n.set_ylabel("y / a_M")
ax_n.set_title("Electron density")

# compute density-density correlator \int dR <n(R + dr/2) n(R - dr/2)>
relative_positions = []
for cntc in range(len(positions_batch)):
    config_up = positions_batch[cntc]
    for cntup in range(len(config_up)):
        relative_positions += np.subtract(config_up[cntup+1:], config_up[cntup]).tolist()
relative_positions = lattices.send_positions_to_first_unit_cell(relative_positions, lat_vec, rec)

# plot density-density correlator
fig_nn, ax_nn = plt.subplots(1, 1, figsize = (7, 5))
ax_nn.scatter(relative_positions[:,0], relative_positions[:,1], color="tab:blue", s=1, alpha=0.03)
ax_nn.set_xlabel("x / a_M")
ax_nn.set_ylabel("y / a_M")
ax_nn.set_title("Density-density correlation")

