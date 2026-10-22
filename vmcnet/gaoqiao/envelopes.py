# Copyright 2022 DeepMind Technologies Limited.
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

"""Multiplicative envelope functions."""

import enum
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import attr
import vmcnet.gaoqiao.network_blocks as network_blocks
#import sto
import jax
import jax.numpy as jnp
from typing_extensions import Protocol
#import ds
import itertools

_MAX_POLY_ORDER = 5  # highest polynomial used in envelopes


class EnvelopeType(enum.Enum):
  """The point at which the envelope is applied."""
  PRE_DETERMINANT = enum.auto()
  PRE_DETERMINANT_Z = enum.auto()

class EnvelopeInit(Protocol):

  def __call__(
      self, natom: int, output_dims: Union[int, Sequence[int]],
      ndim: int
  ) -> Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]:
    """Returns the envelope parameters.

    Envelopes applied separately to each spin channel must create a sequence of
    parameters, one for each spin channel. Other envelope types must create a
    single mapping.

    Args:
      natom: Number of atoms in the system.
      output_dims: The dimension of the layer to which the envelope is applied,
        per-spin channel for pre_determinant envelopes and a scalar otherwise.
      ndim: Dimension of system. Change with care.
    """


class EnvelopeApply(Protocol):

  def __call__(self, *, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
               **kwargs: jnp.ndarray) -> jnp.ndarray:
    """Returns a multiplicative envelope to ensure boundary conditions are met.

    If the envelope is applied before orbital shaping or after determinant
    evaluation, the envelope function is called once and N is the number of
    electrons. If the envelope is applied after orbital shaping and before
    determinant evaluation, the envelope function is called once per spin
    channel and N is the number of electrons in the spin channel.

    The envelope applied post-determinant evaluation is assumed to be in
    log-space.

    Args:
      ae: atom-electron vectors, shape (N, natom, ndim).
      r_ae: atom-electron distances, shape (N, natom, 1).
      r_ee: electron-electron distances, shape (N, nel, 1).
      **kwargs: learnable parameters of the envelope function.
    """


@attr.s(auto_attribs=True)
class Envelope:
  apply_type: EnvelopeType
  init: EnvelopeInit
  apply: EnvelopeApply

@attr.s(auto_attribs=True)
class GemiEnvelope:
  init: EnvelopeInit
  apply: EnvelopeApply



def make_isotropic_envelope() -> Envelope:
  """Creates an isotropic exponentially decaying multiplicative envelope."""

  def init(
      natom: int, output_dims: Sequence[int], ndim: int = 3
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del ndim  # unused
    params = []
    for output_dim in output_dims:
      params.append({
          'pi': jnp.ones(shape=(natom, output_dim)),
          'sigma': jnp.ones(shape=(natom, output_dim))
      })
    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Computes an isotropic exponentially-decaying multiplicative envelope."""
    del ae, r_ee  # unused
    return jnp.sum(jnp.exp(-r_ae * sigma) * pi, axis=1)

  return Envelope(EnvelopeType.PRE_DETERMINANT, init, apply)


def make_ds_hz_envelope_positive(
    hiddens: Tuple[int, ...] = (8, 8),
    activation_fn=jax.nn.silu,
    hidden_scale: float = 1.0,
    output_scale: float = 1.0e-2,
    sigma_min: float = 0.05,
    sigma_init: float = 1.0,
    pi_base: float = 1.0,
    pi_scale: float = 0.1,
) -> Envelope:
  """Isotropic exponentially decaying multiplicative envelope.

  Envelope:
    env_i_k = sum_A pi_Ak * exp(-sigma_Ak * |r_i - R_A|)

  Improvements over original:
    1. Nonzero hidden initialization.
    2. Positive sigma via softplus.
    3. No residual on final layer.
    4. Different spin channels get different keys.
  """

  def _inverse_softplus(x):
    x = jnp.asarray(x)
    return x + jnp.log(-jnp.expm1(-x))

  def init_one(
      key,
      hz_size,
      output_dim,
  ) -> Sequence[Mapping[str, jnp.ndarray]]:

    dims_in = [hz_size] + list(hiddens)
    dims_out = list(hiddens) + [output_dim]

    params = {}
    params["pi"] = []
    params["sigma"] = []

    for ii in range(len(dims_in)):
      is_last = ii == len(dims_in) - 1
      layer_scale = output_scale if is_last else hidden_scale

      key, subkey = jax.random.split(key)
      params["pi"].append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_in[ii],
              out_dim=dims_out[ii],
              include_bias=True,
              scale=layer_scale,
          )
      )

      key, subkey = jax.random.split(key)
      sigma_layer = network_blocks.init_linear_layer(
          subkey,
          in_dim=dims_in[ii],
          out_dim=dims_out[ii],
          include_bias=True,
          scale=layer_scale,
      )

      # Initialize final sigma bias so that:
      # sigma_min + softplus(raw_sigma) ~= sigma_init
      if is_last:
        raw_init = _inverse_softplus(sigma_init - sigma_min)
        if "b" in sigma_layer:
          sigma_layer["b"] = jnp.ones((output_dim,)) * raw_init

      params["sigma"].append(sigma_layer)

    return params

  def init(
      key,
      hz_size,
      output_dims,
  ):
    params = []
    for output_dim in output_dims:
      key, subkey = jax.random.split(key)
      params.append(
        init_one(subkey, hz_size, output_dim)
      )
    return params

  def residual(x, y):
    return (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y

  def apply_net(
      hz,
      params,
  ):
    x = hz

    for ii in range(len(params)):
      y = network_blocks.linear_layer(x, **params[ii])

      # Hidden layers.
      if ii != len(params) - 1:
        y = activation_fn(y)
        x = residual(x, y)

      # Final layer: raw output, no activation, no residual.
      else:
        x = y
    return x

  def apply(
      *,
      hz,
      ae,
      pi,
      sigma,
  ) -> jnp.ndarray:
    """Computes isotropic exponentially decaying envelope.

    Args:
      hz:
        Nuclear hidden states, shape [natoms, hz_size].
      ae:
        Electron-nucleus displacement vectors for one spin channel,
        shape [ne_spin, natoms, 3].
      pi:
        pi network params.
      sigma:
        sigma network params.

    Returns:
      Envelope values, shape [ne_spin, output_dim].
    """

    raw_pi = apply_net(hz, pi)
    raw_sigma = apply_net(hz, sigma)

    pi = pi_base + pi_scale * raw_pi

    sigma = sigma_min + jax.nn.softplus(raw_sigma)

    r_ae = jnp.linalg.norm(ae, axis=-1)

    env = jnp.sum(
        jnp.exp(-r_ae[:, :, None] * sigma[None, :, :]) * pi[None, :, :],
        axis=1,
    )

    return env

  return Envelope(EnvelopeType.PRE_DETERMINANT_Z, init, apply)

def make_ds_hz_envelope(
		hiddens: Tuple[int] = (8,8),  #[8]
    activation_fn=jax.nn.tanh,
) -> Envelope:
  """Creates an isotropic exponentially decaying multiplicative envelope."""

  def init_one(
      key,
      hz_size,  #16
      output_dim, #odim+diff or odim
  ) -> Sequence[Mapping[str,jnp.ndarray]]:
    dims_in=[hz_size]+hiddens  #[16,8]
    #print('heddens:',hiddens) #hiddens=8
    ## last dim: pi and sigma
    dims_out=hiddens+[output_dim] #[8,16]
    params={}
    params['pi']=[]
    params['sigma']=[]
    for ii in range(len(dims_in)):
      key,subkey=jax.random.split(key)
      params['pi'].append(
        network_blocks.init_linear_layer(
          subkey,
          in_dim=dims_in[ii],
          out_dim=dims_out[ii],
          include_bias=True,
          scale=0.0,
        ))
      key,subkey=jax.random.split(key)
      params['sigma'].append(
        network_blocks.init_linear_layer(
          subkey,
          in_dim=dims_in[ii],
          out_dim=dims_out[ii],
          include_bias=True,
          scale=0.0,
        ))
    return params

  def init(
      key,
      hz_size,  #(16)
      output_dims, #[16,16] or [odim+diff,odim]
  ):
    params = []
    for output_dim in output_dims:
      params.append(
        init_one(key,hz_size,output_dim)
      )
    return params

  residual=lambda x,y:(x+y)/jnp.sqrt(2.0) if x.shape==y.shape else y

  def apply_net(
      hz,params,
  ):
    hz_in=hz
    for ii in range(len(params)):
      hz_out=network_blocks.linear_layer(hz_in,**(params[ii]))
      if ii != len(params) - 1:
        hz_out = activation_fn(hz_out)
      hz_in=residual(hz_in,hz_out)
    return hz_in

  def apply(
      *,
      hz,  #hz:(14,16)质子单流
      ae,  #(7,14,3)输入的是某种自旋的7个电子和14个质子的位移差
      pi,
      sigma,
  ) -> jnp.ndarray:
    """Computes an isotropic exponentially-decaying multiplicative envelope."""
    # hz: natoms x nfz
    # natoms x dims_out
    pi=apply_net(hz,pi)   #得到的就是pi:(14,16)
    sigma=apply_net(hz,sigma)       #得到的是sigma:(14,16)
    pi=1.0+1.*pi
    sigma=0.5+1.*sigma
    #print('sigma.shape:',sigma.shape)
    # ne x na
    #prim_periodic_sea,_=ds.nu_distance(ae,org_lattice,rec_lattice)  #(7,14)
    r_ae=jnp.linalg.norm(ae,axis=-1)
    # ne x na x None multiply na x odim -> ne x na x odim
    # sum(axis=1) -> ne x odim
    aa=jnp.sum(jnp.exp(-r_ae[:,:,None]*sigma)*pi,axis=1)
    #print('aa.shape:',aa.shape)
    return aa
    #(7,14,1)*(14,16)->(7,14,16)-=>(7,16)

  return Envelope(EnvelopeType.PRE_DETERMINANT_Z,init,apply)


def make_ds_hz_degenerate_envelope(
    hiddens: Tuple[int, ...] = (8, 8),
    activation_fn=jax.nn.tanh,
) -> Envelope:
  """Creates the nucleus-shared envelope of the degenerate ds_hz MLP.

  For each spin channel and output index a, the envelope is
  (1 + pi[a]) * sum_I exp(-(0.5 + sigma[a]) * |r_i - R_I|).
  The trainable arrays represent the original MLP's final biases, shared
  across nuclei. They have no positivity constraints. This matches ds_hz
  when its other parameters remain zero and no residual preserves hz;
  it is not equivalent to every possible ds_hz hidden-width configuration.

  Factory, init and apply call signatures match make_ds_hz_envelope.
  The parameter tree is intentionally compact, not MLP-checkpoint compatible.
  """
  del hiddens, activation_fn

  def init(key, hz_size, output_dims):
    del key, hz_size
    return [
        {'pi': jnp.zeros((output_dim,)),
         'sigma': jnp.zeros((output_dim,))}
        for output_dim in output_dims
    ]

  def apply(*, hz, ae, pi, sigma) -> jnp.ndarray:
    del hz
    r_ae = jnp.linalg.norm(ae, axis=-1)
    return jnp.sum(
        jnp.exp(-r_ae[:, :, None] * (0.5 + sigma)) * (1.0 + pi),
        axis=1,
    )

  return Envelope(EnvelopeType.PRE_DETERMINANT_Z, init, apply)

