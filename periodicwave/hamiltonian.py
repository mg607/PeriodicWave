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
# - Streamlined for materials calculations

""" Basic function for evaluating the Hamiltonian on a wavefunction. """

from typing import Any, Callable, Optional, Sequence, Tuple, Union

import chex
from periodicwave import networks
from periodicwave.utils import utils
import folx
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
from typing_extensions import Protocol

Array = Union[jnp.ndarray, np.ndarray]


class LocalEnergy(Protocol):

  def __call__(
      self,
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    """Returns the local energy of a Hamiltonian at a configuration.

    Args:
      params: network parameters.
      key: JAX PRNG state.
      data: MCMC configuration to evaluate.
    """


class MakeLocalEnergy(Protocol):

  def __call__(
      self,
      f: networks.FermiNetLike,
      charges: jnp.ndarray,
      nspins: Sequence[int],
      use_scan: bool = False,
      complex_output: bool = False,
      **kwargs: Any
  ) -> LocalEnergy:
    """Builds the LocalEnergy function.

    Args:
      f: Callable which evaluates the sign and log of the magnitude of the
        wavefunction.
      charges: nuclear charges.
      nspins: Number of particles of each spin.
      use_scan: Whether to use a `lax.scan` for computing the laplacian.
      complex_output: If true, the output of f is complex-valued.
      **kwargs: additional kwargs to use for creating the specific Hamiltonian.
    """


KineticEnergy = Callable[
    [networks.ParamTree, networks.FermiNetData], jnp.ndarray
]


def local_kinetic_energy(
    f: networks.FermiNetLike,
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = 'default',
) -> KineticEnergy:
  r"""Creates a function to for the local kinetic energy, -1/2 \nabla^2 ln|f|.

  Args:
    f: Callable which evaluates the wavefunction as a
      (sign or phase, log magnitude) tuple.
    use_scan: Whether to use a `lax.scan` for computing the laplacian.
    complex_output: If true, the output of f is complex-valued.
    laplacian_method: Laplacian calculation method. One of:
      'default': take jvp(grad), looping over inputs
      'folx': use Microsoft's implementation of forward laplacian

  Returns:
    Callable which evaluates the local kinetic energy,
    -1/2f \nabla^2 f = -1/2 (\nabla^2 log|f| + (\nabla log|f|)^2).
  """

  phase_f = utils.select_output(f, 0)
  logabs_f = utils.select_output(f, 1)

  if laplacian_method == 'default':
    def _lapl_over_f(params, data):
      n = data.positions.shape[0]
      eye = jnp.eye(n)
      grad_f = jax.grad(logabs_f, argnums=1)
      def grad_f_closure(x):
        return grad_f(params, x, data.spins, data.atoms, data.charges)

      primal, dgrad_f = jax.linearize(grad_f_closure, data.positions)

      if complex_output:
        grad_phase = jax.grad(phase_f, argnums=1)
        def grad_phase_closure(x):
          return grad_phase(params, x, data.spins, data.atoms, data.charges)
        phase_primal, dgrad_phase = jax.linearize(
            grad_phase_closure, data.positions)
        hessian_diagonal = (
            lambda i: dgrad_f(eye[i])[i] + 1.j * dgrad_phase(eye[i])[i]
        )
      else:
        hessian_diagonal = lambda i: dgrad_f(eye[i])[i]

      if use_scan:
        _, diagonal = lax.scan(
            lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=n)
        result = -0.5 * jnp.sum(diagonal)
      else:
        result = -0.5 * lax.fori_loop(
            0, n, lambda i, val: val + hessian_diagonal(i), 0.0)
      result -= 0.5 * jnp.sum(primal ** 2)
      if complex_output:
        result += 0.5 * jnp.sum(phase_primal ** 2)
        result -= 1.j * jnp.sum(primal * phase_primal)
      return result

  elif laplacian_method == 'folx':
    def _lapl_over_f(params, data):
      f_closure = lambda x: f(params, x, data.spins, data.atoms, data.charges)
      f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=0) # setting sparsity threshold to zero, finite values cause crash
      output = f_wrapped(data.positions)
      result = - (output[1].laplacian +
                  jnp.sum(output[1].jacobian.dense_array ** 2)) / 2
      if complex_output:
        result -= 0.5j * output[0].laplacian
        result += 0.5 * jnp.sum(output[0].jacobian.dense_array ** 2)
        result -= 1.j * jnp.sum(output[0].jacobian.dense_array *
                                output[1].jacobian.dense_array)
      return result

  else:
    raise NotImplementedError(f'Laplacian method {laplacian_method} '
                              'not implemented.')

  return _lapl_over_f


