# Copyright (c) 2025 Max Geier, Massachusetts Institute of Technology, MA, USA
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
This code runs a calculation of a two-dimensional electron gas with triangular potential.

Reference: M Geier, K Nazaryan, T Zaklama, and L Fu, Phys. Rev. B 112, 045119
"""

import os
import sys
from sys import argv
import numpy as np

from absl import logging
from periodicwave import base_config
from periodicwave import train

from periodicwave.utils import custom_logging
from periodicwave.utils import writers
from periodicwave.pbc import lattices

from typing import Tuple

from time import time

import jax
print("Jax Devices:", jax.devices())

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

# Always use float32 precision
print("Setting jax_default_matmul_precision to float32")
jax.config.update("jax_default_matmul_precision", "float32")
print(f"Matmul precision set to: {jax.default_matmul_precision}")

# Get parameters defining the physical system
print(f"Calculating 2DEG with triangular potential with parameters: {argv}")

if len(argv) >= 2:
    program_name = argv[0]
    # number of spin spin up, down electrons
    nspins = [int(n) for n in argv[1].split('_')] 
    num_unit_cells = int(argv[2])
    # Ratio between interaction and kinetic energy scale
    me_eff_rel = float(argv[3]) # in units of bare electron mass
    eps_inverse = float(argv[4]) # inverse dielectric constant of surrounding dielectric
    moire_lattice_constant_nm = float(argv[5]) # in nm
    moire_potential_strength_meV = float(argv[6]) # in meV
    moire_potential_phi = float(argv[7]) # potential shape angle in degrees
    network_type = argv[8]
else:
    nspins = [6, 0]
    num_unit_cells = 9
    me_eff_rel = 0.35 # in units of bare electron mass
    eps_inverse = 0.2 # inverse dielectric constant of surrounding dielectric
    moire_lattice_constant_nm = 8.031 # in nm
    moire_potential_strength_meV = 15 # in meV
    moire_potential_phi = 45 # potential shape angle in degrees
    network_type = "CustomPsiformer" # "SlaterNet", "CustomPsiformer"

# Convert SI units to natural units
energy_scale, moire_potential_strength, interaction_energy_scale = convert_moire_scales(me_eff_rel, eps_inverse, moire_lattice_constant_nm, moire_potential_strength_meV)
print(f"Energy scale = {energy_scale}, \n moire_potential_strength = {moire_potential_strength}, \n interaction_energy_scale = {interaction_energy_scale}")

# --------------------------- Set up config file ---------------------------
cfg = base_config.default()

# Set number of electrons
cfg.system.electrons = (nspins[0], nspins[1])  # (spin up, spin down) electrons

# Create the 2D lattice 
cfg.system.ndim = 2
num_electrons = sum(nspins)
supercell_a = np.sqrt(2 * np.pi / np.sqrt(3))
supercell_a = 1.0
supercell_lattice, moire_lattice = lattices._triangular_lattice_vecs_periodic_potential(supercell_a, num_unit_cells)
area = np.linalg.det(supercell_lattice)

# Set up the Hamiltonian parameters
cfg.system.make_local_energy_fn = "ferminet.pbc.hamiltonian.local_energy"
cfg.system.make_local_energy_kwargs = {"lattice": supercell_lattice, 
    "potential_type": 'CoulombMoire',
    "potential_kwargs": {'moire_lattice_vectors': moire_lattice,
                            'moire_potential_strength': moire_potential_strength, 
                            'moire_potential_phi':      moire_potential_phi/180*np.pi,
                            'interaction_energy_scale': interaction_energy_scale},
    "kinetic_kwargs": {'laplacian_method': 'folx'}} # use forward laplacian}
cfg.system.pbc_lattice = supercell_lattice

# feature layer for pbc
cfg.network.make_feature_layer_fn = ("ferminet.pbc.feature_layer.make_pbc_feature_layer")
cfg.network.make_feature_layer_kwargs = { "lattice": supercell_lattice, "include_r_ae": False }

# envelopes
cfg.network.make_envelope_fn = ( "ferminet.envelopes.make_null_envelope" ) # use no envelope with pbc

# Set batch size
cfg.batch_size = 1024

# learning rate parameters
cfg.optim.optimizer  = 'kfac'
cfg.optim.objective  = 'vmc'
cfg.optim.iterations = 200000
cfg.optim.lr.rate    = 0.1
cfg.optim.lr.delay   = 20000
cfg.optim.lr.decay   = 1.0

# logging parameters
cfg.log.save_frequency = 30.0 # minutes

# mcmc parameters
cfg.mcmc.burn_in    = 300
cfg.mcmc.init_width = 1.0
cfg.mcmc.move_width = 0.2
cfg.mcmc.steps      = 30
cfg.mcmc.move_width_updater = 'adaptive'

# general network parameters
cfg.network.complex = True # complex wavefunction work better for pbc systems
cfg.network.determinants = 4
cfg.network.jastrow = "NONE"
cfg.network.jastrow_kwargs = {'ndim':cfg.system.ndim}

# network architecture parameters
# Code to run Psiformer
if network_type == 'CustomPsiformer':
    cfg.network.network_type = 'CustomPsiformer'
    cfg.network.CustomPsiformer.num_layers = 4
    cfg.network.CustomPsiformer.num_heads  = 4
    cfg.network.CustomPsiformer.attn_dim   = 16
    cfg.network.CustomPsiformer.value_dim  = 16
    cfg.network.CustomPsiformer.mlp_dim    = 64
    cfg.network.CustomPsiformer.num_perceptrons_per_layer = 2
    cfg.network.CustomPsiformer.use_layer_norm = True
    cfg.network.CustomPsiformer.mlp_activation_fct = "GELU"

# Code to run SlaterNet
elif network_type == 'SlaterNet':
    cfg.network.network_type = 'SlaterNet'
    cfg.network.SlaterNet.num_layers = 4
    cfg.network.SlaterNet.mlp_dim    = 64
    cfg.network.SlaterNet.num_perceptrons_per_layer = 2
    cfg.network.SlaterNet.use_layer_norm = True
    cfg.network.SlaterNet.mlp_activation_fct = "GELU"
    cfg.network.determinants = 1 # when using a single determinant, SlaterNet is equivalent to Hartree-Fock

# Get folder name to save the results
folder_name = f"results/2deg-CoulombMoire/{network_type}/el{nspins[0]}_{nspins[1]}_N{num_unit_cells}_V{np.round(moire_potential_strength,8)}_{moire_potential_phi}_U{np.round(interaction_energy_scale,8)}"

# save path
cfg.log.save_path = folder_name

# log device info. 
# If same folder contains previous runs, rename old log file to keep information.
# The renaming follows the same convention as renaming train_stats.csv
writers.rename_file("device_info", folder_name, file_extension="log")
custom_logging.log_device_info(folder_name + "/device_info.log")
writers.rename_file("config", folder_name, file_extension="json")
custom_logging.save_config_dict_as_json(cfg, cfg.log.save_path + '/config.json')

t_init = time()
# --------------------------- train ---------------------------
train.train(cfg)
logging.info("Training completed after t [s] = " + str(int(time() - t_init)))