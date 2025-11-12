# PeriodicWave: Neural Network Variational Monte Carlo for Solids

This repository contains a Monte-Carlo solver to train neural-network variational wavefunction to solve continuous-space Fermi systems [M Geier, K Nazaryan, T Zaklama, L Fu, Phys. Rev. B 112, 045119 (2025)]. We provide an optimized self-attention neural network wavefunction ansatz to capture electron correlations. Additionally, we introduce a new neural network architecture -- SlaterNet -- that implements an unrestricted Hartree-Fock solver. 

Our AI-based method successfully solves the problem of moire semiconductors in two-dimensional materials. It outperforms traditional exact-diagonalization methods based on a projection to a finite number of orbitals. Thus, our approach advances the state-of-the-art in simulating quantum many-body systems from first principles.

Part of this code is built upon Google DeepMind's FermiNet [2] and extended to tackle periodic solids. 

This version includes the following network architectures:
- CustomPsiformer
- SlaterNet

CustomPsiformer is a self-attention neural network wavefunction ansatz that universally approximates many-fermion wavefunctions, see Ref. [1] for a description. The architecture is adapted from the original ``psiformer'' implementation from [4]. 

SlaterNet is a feed-forward neural network that realizes a variational implementation of unrestricted Hartree-Fock, i.e. it identifies the best Slater determinant of single-particle orbitals that minimizes the energy. See Ref. [1] for a description.

The code contains functionality to solve two-dimensional periodic systems:
- Modification for the code to work in 2d with PBC and complex wavefunctions
- 2D Coulomb interaction
- Triangular potential
- Optional Jastrow factors adapated to enforce cusp condition with Coulomb interaction in two dimensions and periodic boundary conditions

The code-base has been streamlined for materials problems. 

Additionally, there are the following interface and data analysis scripts:
- Data analysis scripts that load and plot energy
- Data analysis scripts that load and plot density and density-density correlators

Authors: Max Geier, Khachatur Nazaryan (2025)
Massachusetts Institute of Technology

## Installation (Verified on 10/09/2025)

1. Install python 3.13 environment:
```
conda create --name py313 python=3.13 numpy scipy pandas matplotlib virtualenv
```

2. Activate py313 environment and create virtualenv from there:
```
conda activate py313
virtualenv ~/venv/periodicwave
```

3. Activate new virtual environment
```
source ~/venv/periodicwave/bin/activate
```

4. Install PeriodicWave with all dependencies
In the periodicwave folder that contains setup.py:
```
pip install -e .
```

5. Optionally: Install GPU support if a GPU is available, The following updates JAX and installs all cuda versions.
```
pip install -U "jax[cuda12]"
```

6. Verify installation:
The following command should run a calculation of a two-dimensional Coulomb gas for 2 spin-up electrons at r_s = 10 with CustomPsiformer:
```
python3 periodicwave/configs/run_2deg.py
```

## Running two-dimensional electron gas calculations

The file periodicwave/configs/run-2deg.py contains default parameters to run calculations in a homogeneous two-dimensional electron gas with Coulomb interactions. The parameters are used to construct a config file which is then passed to train.train to start optimizing the neural network wavefunction towards the ground state.

The code allows to directly change electron number, Hamiltonian parameter r_s contolling the ratio between interaction and kinetic energy, shape of the periodic supercell, and whether CustomPsiformer or SlaterNet is used. 

The results are stored in a folder in results/2deg-Coulomb/{network_type}/el{nspins[0]}_{nspins[1]}_rs{r_s}_{supercell_shape}. In this folder, the calculation results are stored in train_stats.py and checkpoints are stored in qmcjax_ckpt_######.npy. 

## Running two-dimensional electron gas with periodic potential 

Similarly, the file periodicwave/configs/run-2deg-triangular-potential.py contains default parameters to run calculations in a two-dimensional electron gas with Coulomb interactions and triangular potential. Default parameters are set to reproduce results from Fig. 5 of [1].

## Monitoring and inference with evaluate scripts

To monitor the calculation, evaluate-energies-2dheg.py loads the energy as a function of steps from train_stats of the specified folder and plots it. 

Furthermore, evaluate-SpinDensity-2dheg.py loads the last check point and plots density and density-density correlators evaluated from the electron position configurations that were sampled by MCMC at the checkpoint step. 

## CustomPsiformer
CustomPsiformer is a modified version of the psiformer.py from Deepmind. We modified the network construction so that the internal network dimensions match the description in [1]. 

Specifically, the network options are:
    num_layers: Number of self-attention layers.
    num_heads: Number of self-attention heads per attention layer.
    attn_dim: Embedding dimension for each self-attention head.
    value_dim: Dimension of the value vector of each attention head.
    mlp_dim: Dimension of the perceptron layers and the internal token dimension.
    num_perceptrons_per_layer: Number of perceptron layers after each self-attention layer.
    use_layer_norm: If true, include a layer norm after both attention and MLP.
    mlp_activation_function: Selection of "TANH", "GELU", "ELU" non-linearity.

Furthermore, we included improvements: 
    1. The self-attention operation now always uses Flash attention for GPU optimization without loss of accuracy.
    2. There is now a selection for the non-linear activation function for the MLP. Typically, GELU or ELU outperform TANH.

Residual connections are applied across each attention layer and each MLP layer consisting of num_perceptrons_per_layer subsequent perceptrons.


## SlaterNet
SlaterNet is a pure feed-forward multilayer-perceptron neural network, i.e. all self-attention layers are taken out, see [1] for details. 
This calculation is equivalent to Hartree-Fock when constructing only a single Slater determinant and no Jastrow factor.
The default Jastrow option defaults to "NONE", i.e. NO Jastrow factor.

SlaterNet network options:
    num_layers: Number of MLP layers representing the single orbitals.
    mlp_dim: Dimension of the perceptron layers.
    num_perceptrons_per_layer: number of perceptrons per layer.
    use_layer_norm: If true, include a layer norm after each num_perceptrons_per_layer of perceptrons.
    mlp_activation_function: Selection of "TANH", "GELU", "ELU" non-linearity.
  
Notice: The total number of perceptrons is: num_layers * num_perceptrons_per_layer.
After each layer (including multiple perceptrons), the residual output h^(l-1) from the previous layer is added.
If use_layer_norm = True, after each layer a layer norm is applied, for numerical stability.

## Jastrow

We adapted the optional Jastrow factor for Coulomb interaction in two dimensions. It now takes the dimensionality and interaction strength as parameters.

## Inference (from original FermiNet documentation)

After training, it is useful to run calculations of the energy and other
observables over many time steps with the parameters fixed to accumulate
low-variance estimates of physical quantities. To do this, set 
`cfg.optim.optimizer = 'none'`. Make sure that either the value of 
`cfg.log.save_path` is the same, or that the value of `cfg.log.restore_path` 
is set to the value of `cfg.log.save_path` from the original training run.

## Output (from original FermiNet documentation)

The results directory contains `train_stats.csv` which contains the local energy
and MCMC acceptance probability for each iteration, and the `checkpoints`
directory, which contains the checkpoints generated during training. 

## List of changes relative to FermiNet GitHub
(i)   Include SlaterNet.py: SlaterNet architecture
(ii)  Include CustomPsiformer.py: Custom psiformer architecture
(iii) Implemented functionality for two-dimensional materials problems: Periodic boundary conditions, Coulomb interaction, moire potential, Jastrow factors
(iv)  Streamlining for materials problems: Removed molecule-specific functionality

## Giving Credit

The VMC method, basic neural network architecture, and adaptations for two-
dimensional materials are described in:

[1] M Geier, K Nazaryan, T Zaklama, L Fu, "Self-attention neural network for solving correlated electron problems in solids" Phys. Rev. B 112, 045119 (2025)
```
@article{geier2025,
  title = {Self-attention neural network for solving correlated electron problems in solids},
  author = {Geier, Max and Nazaryan, Khachatur and Zaklama, Timothy and Fu, Liang},
  journal = {Phys. Rev. B},
  volume = {112},
  issue = {4},
  pages = {045119},
  numpages = {16},
  year = {2025},
  month = {Jul},
  publisher = {American Physical Society},
  doi = {10.1103/qxc3-bkc7},
  url = {https://link.aps.org/doi/10.1103/qxc3-bkc7}
}
```

Please also cite these related works. Deepmind's original work introducing the generalized determinant and basic functionality:

[2] D Pfau, J S Spencer, A G de G Matthews, and W M C Foulkes "Ab-Initio Solution of the Many-Electron Schr{\"o}dinger Equation with Deep Neural Networks" Phys. Rev. Research 2, 033429 (2020)
```
@article{pfau2020ferminet,
  title={Ab-Initio Solution of the Many-Electron Schr{\"o}dinger Equation with Deep Neural Networks},
  author={D. Pfau and J.S. Spencer and A.G. de G. Matthews and W.M.C. Foulkes},
  journal={Phys. Rev. Research},
  year={2020},
  volume={2},
  issue = {3},
  pages={033429},
  doi = {10.1103/PhysRevResearch.2.033429},
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.2.033429}
}
```

and a NeurIPS Workshop Machine Learning and Physics paper describes the JAX
implementation:

[3] J S Spencer, D Pfau, A. Botev, and W M C Foulkes, "Better, Faster Fermionic Neural Networks", arXiv:2011.07125 (2020)
```
@misc{spencer2020better,
  title={Better, Faster Fermionic Neural Networks},
  author={James S. Spencer and David Pfau and Aleksandar Botev and W. M.C. Foulkes},
  year={2020},
  eprint={2011.07125},
  archivePrefix={arXiv},
  primaryClass={physics.comp-ph},
  url={https://arxiv.org/abs/2011.07125}
}
```

The PsiFormer architecture is detailed in an ICLR 2023 paper:

[4] I von Glehn, J S Spencer and D Pfau, "A Self-Attention Ansatz for Ab-initio Quantum Chemistry", ICLR (2023)
```
@misc{vonglehn2023psiformer,
  title={A Self-Attention Ansatz for Ab-initio Quantum Chemistry},
  author={Ingrid von Glehn and James S Spencer and David Pfau},
  journal={ICLR},
  year={2023},
}
```

Periodic boundary conditions were originally introduced in a Physical Review
Letters article:

[5] G Cassella, H. Sutterud, S Azadi, N D Drummond, D Pfau, J S Spencer, and W M C Foulkes, "Discovering quantum phase transitions with fermionic neural networks" Phys. Rev. Lett. 130, 036401 (2023)
```
@article{cassella2023discovering,
  title={Discovering quantum phase transitions with fermionic neural networks},
  author={Cassella, Gino and Sutterud, Halvard and Azadi, Sam and Drummond, ND and Pfau, David and Spencer, James S and Foulkes, W Matthew C},
  journal={Physical review letters},
  volume={130},
  number={3},
  pages={036401},
  year={2023},
  publisher={APS}
}
```

The Forward Laplacian was introduced in the following Nature Machine Intelligence article:

[6] R Li, H Ye, D Jiang, X Wen, C Wang, Z Li, X Li, D He, J Chen, W Ren, and L Wang, "A computational framework for neural network-based variational Monte Carlo with Forward Laplacian", Nat Mach Intell 6, 209–219 (2024)
```
@article{li2024computational,
	author = {Li, Ruichen and Ye, Haotian and Jiang, Du and Wen, Xuelan and Wang, Chuwei and Li, Zhe and Li, Xiang and He, Di and Chen, Ji and Ren, Weiluo and Wang, Liwei},
	date = {2024/02/01},
	doi = {10.1038/s42256-024-00794-x},
	id = {Li2024},
	isbn = {2522-5839},
	journal = {Nature Machine Intelligence},
	number = {2},
	pages = {209--219},
	title = {A computational framework for neural network-based variational Monte Carlo with Forward Laplacian},
	url = {https://doi.org/10.1038/s42256-024-00794-x},
	volume = {6},
	year = {2024},
}
```

Please cite our repository as:

```
@software{periodicwave_github,
  author = {Max Geier, Khachatur Nazaryan},
  title = {{PeriodicWave}},
  url = {http://github.com/mg607/periodicwave},
  year = {2025},
}
```

Cite Google DeepMind's FermiNet original repository:

```
@software{ferminet_github,
  author = {James S. Spencer, David Pfau and FermiNet Contributors},
  title = {{FermiNet}},
  url = {http://github.com/deepmind/ferminet},
  year = {2020},
}
```

## Troubleshooting for package compatibility
The list of packages for which the code works on a laptop (Apple M3, no GPU support) is posted below (obtained from pip freeze). The kfac-jax version during installation is 0.0.7.

```
absl-py==2.3.1
attrs==25.4.0
chex==0.1.91
cloudpickle==3.1.1
decorator==5.2.1
distrax==0.1.7
dm-tree==0.1.9
folx @ git+https://github.com/microsoft/folx@d05c107028e3f88239ebf9e894d4a8c01abf90f6
gast==0.6.0
h5py==3.14.0
immutabledict==4.2.1
jax==0.7.2
jaxlib==0.7.2
jaxtyping==0.3.3
kfac-jax @ git+https://github.com/deepmind/kfac-jax@d9ecae99e588e4abbb0dd3d4e977d1266824e14c
ml_collections==1.1.0
ml_dtypes==0.5.3
numpy==2.3.3
opt_einsum==3.4.0
optax==0.2.6
pandas==2.3.3
pyblock==0.6
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.3
scipy==1.16.2
six==1.17.0
tfp-nightly==0.26.0.dev20251007
toolz==1.0.0
typing_extensions==4.15.0
tzdata==2025.2
wadler_lindig==0.1.7
wrapt==1.17.3
```

The list of packages for which the code works with GPU is (verified 10/17/2025):

```
absl-py==2.3.1
attrs==25.4.0
chex==0.1.91
cloudpickle==3.1.1
decorator==5.2.1
distrax==0.1.7
dm-tree==0.1.9
folx @ git+https://github.com/microsoft/folx@d05c107028e3f88239ebf9e894d4a8c01abf90f6
gast==0.6.0
h5py==3.15.1
immutabledict==4.2.2
jax==0.8.0
jax-cuda12-pjrt==0.8.0
jax-cuda12-plugin==0.8.0
jaxlib==0.8.0
jaxtyping==0.3.3
kfac-jax @ git+https://github.com/deepmind/kfac-jax@55111a15e51b38a4cefbfd16cdda52472cda2632
ml_collections==1.1.0
ml_dtypes==0.5.3
numpy==2.3.4
nvidia-cublas-cu12==12.9.1.4
nvidia-cuda-cupti-cu12==12.9.79
nvidia-cuda-nvcc-cu12==12.9.86
nvidia-cuda-nvrtc-cu12==12.9.86
nvidia-cuda-runtime-cu12==12.9.79
nvidia-cudnn-cu12==9.14.0.64
nvidia-cufft-cu12==11.4.1.4
nvidia-cusolver-cu12==11.7.5.82
nvidia-cusparse-cu12==12.5.10.65
nvidia-nccl-cu12==2.28.3
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvshmem-cu12==3.4.5
opt_einsum==3.4.0
optax==0.2.6
pandas==2.3.3
-e git+https://github.com/mg607/PeriodicWave.git@3c54ee38e16eac1feb7190695351ef6affb3d439#egg=periodicwave
pyblock==0.6
python-dateutil==2.9.0.post0
pytz==2025.2
PyYAML==6.0.3
scipy==1.16.2
six==1.17.0
tfp-nightly==0.26.0.dev20251017
toolz==1.1.0
typing_extensions==4.15.0
tzdata==2025.2
wadler_lindig==0.1.7
wrapt==1.17.3
```