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
# - Updated simple-ee Jastrow for Coulomb interaction in two dimensional materials

""" Multiplicative Jastrow factors. """

import enum
from typing import Any, Callable, Iterable, Mapping, Union
import jax.numpy as jnp

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]

class JastrowType(enum.Enum):
  """Available multiplicative Jastrow factors."""

  NONE = enum.auto()
  SIMPLE_EE = enum.auto()
  SIMPLE_EE_SHORTRANGE = enum.auto()


def _jastrow_ee(
    r_ee: jnp.ndarray,
    params: ParamTree,
    nspins: tuple[int, int],
    jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray],
    ndim: int = 3,
    interaction_strength: float = 1.0,
) -> jnp.ndarray:
  """Jastrow factor for electron-electron cusps."""
  if ndim == 3:
    cusp_parallel = interaction_strength/4
    cusp_anti     = interaction_strength/2
  elif ndim == 2:
    cusp_parallel = interaction_strength/3
    cusp_anti     = interaction_strength 
  else:
    raise NotImplementedError("jastrow_ee: Factors to satisfy Coulomb cusp conditions implemented only for ndim = 2 and 3.")

  r_ees = [
      jnp.split(r, nspins[0:1], axis=1)
      for r in jnp.split(r_ee, nspins[0:1], axis=0)
  ]
  r_ees_parallel = jnp.concatenate([
      r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],
      r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
  ])

  if r_ees_parallel.shape[0] > 0:
    jastrow_ee_par = jnp.sum(
        jastrow_fun(r_ees_parallel, cusp_parallel, params['ee_par']) 
    )
  else:
    jastrow_ee_par = jnp.asarray(0.0)

  if r_ees[0][1].shape[0] > 0:
    jastrow_ee_anti = jnp.sum(jastrow_fun(r_ees[0][1], cusp_anti, params['ee_anti'])) 
  else:
    jastrow_ee_anti = jnp.asarray(0.0)

  return jastrow_ee_anti + jastrow_ee_par


def make_simple_ee_jastrow(ndim: int = 3, interaction_strength: float = 1.0) -> ...:
  """Creates a Jastrow factor for electron-electron cusps."""

  def simple_ee_cusp_fun(
      r: jnp.ndarray, cusp: float, alpha: jnp.ndarray
  ) -> jnp.ndarray:
    """Jastrow function satisfying electron cusp condition."""
    return -(cusp * alpha**2) / (alpha + r)

  def init() -> Mapping[str, jnp.ndarray]:
    params = {}
    params['ee_par'] = jnp.ones(
        shape=1,
    )
    params['ee_anti'] = jnp.ones(
        shape=1,
    )
    return params

  def apply(
      r_ee: jnp.ndarray,
      params: ParamTree,
      nspins: tuple[int, int],
  ) -> jnp.ndarray:
    """Jastrow factor for electron-electron cusps."""
    return _jastrow_ee(r_ee, params, nspins, jastrow_fun=simple_ee_cusp_fun, 
                       ndim=ndim, interaction_strength=interaction_strength)

  return init, apply

def get_jastrow(jastrow: JastrowType, jastrow_kwargs: dict) -> ...:
  jastrow_init, jastrow_apply = None, None
  if jastrow == JastrowType.SIMPLE_EE:
    print("get_jastrow: Using SIMPLE_EE Jastrow with parameters:")
    print(jastrow_kwargs)
    jastrow_init, jastrow_apply = make_simple_ee_jastrow(jastrow_kwargs["ndim"], jastrow_kwargs["interaction_strength"])
  elif jastrow != JastrowType.NONE:
    raise ValueError(f'Unknown Jastrow Factor type: {jastrow}')
  else:
    print("get_jastrow: NOT using Jastrow")

  return jastrow_init, jastrow_apply
