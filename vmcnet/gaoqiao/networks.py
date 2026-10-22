# Copyright 2020 DeepMind Technologies Limited.
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

"""Implementation of Fermionic Neural Network in JAX."""
import enum
import functools
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

import attr
import chex
import vmcnet.gaoqiao.envelopes as envelopes
import vmcnet.gaoqiao.network_blocks as network_blocks
import jax
import jax.numpy as jnp
from typing_extensions import Protocol
import vmcnet.gaoqiao.dp as dp
from vmcnet.gaoqiao.jastrows import JastrowModel


FermiLayers = Tuple[Tuple[int, int], ...]
# Recursive types are not yet supported in pytype - b/109648354.
# pytype: disable=not-supported-yet
ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]
# pytype: enable=not-supported-yet
# Parameters for a single part of the network are just a dict.
Param = Mapping[str, jnp.ndarray]

## Interfaces (public) ##


class InitFermiNet(Protocol):

  def __call__(self, key: chex.PRNGKey) -> ParamTree:
    """Returns initialized parameters for the network.

    Args:
      key: RNG state
    """


class FermiNetLike(Protocol):

  def __call__(self, params: ParamTree,
               electrons: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the sign and log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
    """


class LogFermiNetLike(Protocol):

  def __call__(self, params: ParamTree, electrons: jnp.ndarray) -> jnp.ndarray:
    """Returns the log magnitude of the wavefunction.

    Args:
      params: network parameters.
      electrons: electron positions, shape (nelectrons*ndim), where ndim is the
        dimensionality of the system.
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

  def __call__(self, ae: jnp.ndarray, r_ae: jnp.ndarray, ee: jnp.ndarray,
               r_ee: jnp.ndarray,
               **params: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
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


class ModelInit(Protocol):

  def __call__(self, key) -> Param:
    """Creates the learnable parameters for the model.

    Returns:
      key: the random key
    """


class ModelApply(Protocol):

  def __call__(self, 
               Params,
               ae: jnp.ndarray, 
               ee: jnp.ndarray) -> jnp.ndarray:
    """Creates the model to pass into the network.

    Args:
      params: learnable model parameters
      ae: electron-atom vectors. Shape: (nelectron, natom, x).
      ee: electron-electron vectors. Shape: (nelectron, nelectron, x).
    """


@attr.s(auto_attribs=True)
class FerminetModel:
  init: ModelInit
  apply: ModelApply


class FeatureLayerType(enum.Enum):
  STANDARD = enum.auto()


class MakeFeatureLayer(Protocol):

  def __call__(self,
               charges: jnp.ndarray,
               nspins: Sequence[int],
               ndim: int,
               **kwargs: Any) -> FeatureLayer:
    """Builds the FeatureLayer object.

    Args:
      charges: (natom) array of atom nuclear charges.
      nspins: tuple of the number of spin-up and spin-down electrons.
      ndim: dimension of the system.
      **kwargs: additional kwargs to use for creating the specific FeatureLayer.
    """


## Network settings ##


@attr.s(auto_attribs=True, kw_only=True)
class FermiNetOptions:
  """Options controlling the FermiNet architecture.

  Attributes:
    ndim: dimension of system. Change only with caution.
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    determinants: Number of determinants to use.
    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.
    envelope: Envelope object to create and apply the multiplicative envelope.
    feature_layer: Feature object to create and apply the input features for the
      one- and two-electron layers.
  """
  ndim: int = 3
  hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32), (256, 32))
  determinants: int = 16
  full_det: bool = True
  bias_orbitals: bool = False
  envelope: envelopes.Envelope = attr.ib(
      default=attr.Factory(
          envelopes.make_isotropic_envelope,
          takes_self=False))
  feature_layer: FeatureLayer = None
  # attr.ib(default=attr.Factory(lambda self: make_ferminet_features(ndim=self.ndim), takes_self=True))
  ferminet_model: FerminetModel = None 
  # attr.ib(default=attr.Factory(lambda self: make_fermi_net_model, takes_self=True))
#   lattice : jnp.array = None
  do_complex : bool = False
  mes: dp.ManyElectronSystem = None
  det_mode: str = 'det'
  jastrow: JastrowModel =None
  # 原子核电荷；SO(3) 当前专用电子-电子 Kato Jastrow 不直接使用它，但保存在 options 中便于后续扩展电子-核 cusp。
  charges: Any = None

## Network initialisation ##

def init_layers(
    key: chex.PRNGKey, dims_one_in: Sequence[int], dims_one_out: Sequence[int],
    dims_two_in: Sequence[int],
    dims_two_out: Sequence[int]) -> Tuple[Sequence[Param], Sequence[Param]]:
  """Initialises parameters for the FermiNet layers.

  The final two-electron layer is not strictly necessary (i.e.
  FermiNetOptions.use_last_layer is False), in which case the two-electron
  stream contains one fewer layers than the one-electron stream.

  Args:
    key: JAX RNG state.
    dims_one_in: dimension of inputs to each one-electron layer.
    dims_one_out: dimension of outputs (number of hidden units) in each
      one-electron layer.
    dims_two_in: dimension of inputs to each two-electron layer.
    dims_two_out: dimension of outputs (number of hidden units) in each
      two-electron layer.

  Returns:
    Pair of sequences (length: number of layers) of parameters for one- and
    two-electon streams.

  Raises:
    ValueError: if dims_one_in and dims_one_out are different lengths, or
    similarly for dims_two_in and dims_two_out, or if the number of one-electron
    layers is not equal to or one more than the number of two electron layers.
  """
  if len(dims_one_in) != len(dims_one_out):
    raise ValueError(
        'Length of one-electron stream inputs and outputs not identical.')
  if len(dims_two_in) != len(dims_two_out):
    raise ValueError(
        'Length of two-electron stream inputs and outputs not identical.')
  if len(dims_two_in) not in (len(dims_one_out), len(dims_one_out) - 1):
    raise ValueError('Number of layers in two electron stream must match or be '
                     'one fewer than the number of layers in the one-electron '
                     'stream')
  single = []
  double = []
  ndouble_layers = len(dims_two_in)
  for i in range(len(dims_one_in)):
    key, subkey = jax.random.split(key)
    single.append(
        network_blocks.init_linear_layer(
            subkey,
            in_dim=dims_one_in[i],
            out_dim=dims_one_out[i],
            include_bias=True))

    if i < ndouble_layers:
      key, subkey = jax.random.split(key)
      double.append(
          network_blocks.init_linear_layer(
              subkey,
              in_dim=dims_two_in[i],
              out_dim=dims_two_out[i],
              include_bias=True))

  return single, double


def init_orbital_shaping(
    key: chex.PRNGKey,
    input_dim: int,
    nspin_orbitals: Sequence[int],
    bias_orbitals: bool,
) -> Sequence[Param]:
  """Initialises orbital shaping layer.

  Args:
    key: JAX RNG state.
    input_dim: dimension of input activations to the orbital shaping layer.
    nspin_orbitals: total number of orbitals in each spin-channel.
    bias_orbitals: whether to include a bias in the layer.

  Returns:
    Parameters of length len(nspin_orbitals) for the orbital shaping for each
    spin channel.
  """
  orbitals = []
  for nspin_orbital in nspin_orbitals:
    key, subkey = jax.random.split(key)
    orbitals.append(
        network_blocks.init_linear_layer(
            subkey,
            in_dim=input_dim,
            out_dim=nspin_orbital,
            include_bias=bias_orbitals))
  return orbitals

def init_fermi_net_params(
    key: chex.PRNGKey,
    natom: int,
    ndim: int,
    nspins: Tuple[int, ...],
    options: FermiNetOptions,

) -> ParamTree:
  """Initializes parameters for the Fermionic Neural Network.

  Args:
    key: JAX RNG state.
    atoms: (natom, ndim) array of atom positions.
    nspins: A tuple with either the number of spin-up and spin-down electrons,
      or the total number of electrons. If the latter, the spins are instead
      given as an input to the network.
    options: network options.

  Returns:
    PyTree of network parameters. Spin-dependent parameters are only created for
    spin channels containing at least one particle.
  """
  active_spin_channels = [spin for spin in nspins if spin > 0]
  if len(active_spin_channels) == 0:
    raise ValueError('No electrons present!')

  params = {}
  key, subkey = jax.random.split(key, num=2)
  _params, dims_orbital_in = options.ferminet_model.init(subkey)
  for kk, vv in _params.items():
    params[kk] = vv

  # How many spin-orbitals do we need to create per spin channel?
  nspin_orbitals = []
  if options.det_mode == "det":
    for nspin in active_spin_channels:
      if options.full_det:
        # Dense determinant. Need N orbitals per electron per determinant.
        norbitals = sum(nspins) * options.determinants  #最终的nspin_orbitals=[(n_up+n_dn)*ndet,(n_up+n_dn)*ndet]
      else:
        # Spin-factored block-diagonal determinant. Need nspin orbitals per electron per determinant.
        norbitals = nspin * options.determinants      #最终的nspin_orbitals=[n_up*ndet,n_dn*ndet]
      nspin_orbitals.append(norbitals)
  else:
    raise RuntimeError(f"unknown det_mode {options.det_mode}")

  if options.do_complex:
    # twice the output channels if do_complex
    nspin_orbitals_alloc = [ii*2 for ii in nspin_orbitals]
  else:
    nspin_orbitals_alloc = nspin_orbitals

  output_dims = nspin_orbitals
  if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    params['envelope'] = options.envelope.init(
      natom=natom, output_dims=output_dims, ndim=ndim)
  elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT_Z:
    params['envelope'] = options.envelope.init(
      key, hz_size=options.hidden_dims[-1][0], output_dims=output_dims)
  else:
    raise ValueError('Unknown envelope type')

  key, subkey = jax.random.split(key, num=2)
  params['orbital'] = init_orbital_shaping(
      key=subkey,
      input_dim=dims_orbital_in,
      nspin_orbitals=nspin_orbitals_alloc,
      bias_orbitals=options.bias_orbitals)

  if options.jastrow.init is not None:
    key, subkey = jax.random.split(key, num=2)
    params['jastrow'] = options.jastrow.init(
      key=subkey,
      dims_orbital_in=dims_orbital_in,
      include_bias=False,
      )
  else:
    params['jastrow'] = None

  params['det'] = None
  return params


def construct_input_features_ef(
    pos: jnp.ndarray,
    ndim: int = 3,
) -> Tuple[jnp.ndarray, jnp.ndarray]:

  pp = jnp.reshape(pos, [1, -1, ndim]) - jnp.reshape(pos, [-1, 1, ndim])
  np = pp.shape[0]
  r_pp = (jnp.linalg.norm(pp + jnp.eye(np)[..., None], axis=-1) * (1.0 - jnp.eye(np)))

  return pp, r_pp[..., None]


def make_fermi_net_model_ef_shrd_sym_ablation(
    natom,
    ndim,
    nspins,
    feature_layer,
    hidden_dims,   #i.e. ((64,16),(64,16),(64,16),(64,16))
    mes = None,
    activation_fn=jax.nn.tanh,
    # extra parameters
):
  if mes is None :
    raise RuntimeError('make_fermi_net_model_ef only support equal-footing models')
  
  update_alpha, do_rdt, resd_dt_shift, resd_dt_scale = None, False, None, None


  def init(
      key,
  ):    
    # dim_1_append = mes.get_dim_one_hot()    #3
    # dim_2_append = 2*mes.get_dim_one_hot()   #6
    dim_1_append = mes.get_dim_part_features()
    dim_2_append = mes.get_dim_pair_features()

    # number of spin channel
    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)  #2
    # init params
    params = {}
    (num_one_features, num_two_features), params['input'] = (feature_layer.init())

    key, subkey = jax.random.split(key)
    params['one_reshp'] = network_blocks.init_linear_layer(subkey, num_two_features + dim_1_append, hidden_dims[0][0])

    key, subkey = jax.random.split(key)
    params['two_reshp'] = network_blocks.init_linear_layer(subkey, num_two_features + dim_2_append, hidden_dims[0][1])
    distinguish_ele = True
    nfeatures = lambda out1, out2: (nchannels + 1) * out1 + (nchannels) * out2
    dims_1_in = [nfeatures(hidden_dims[0][0], hidden_dims[0][1])]
    dims_1_in += [nfeatures(hdim[0], hdim[1]) for hdim in hidden_dims[:-1]]
    dims_2_in = [hidden_dims[0][1]] + [hdim[1] for hdim in hidden_dims[:-2]]

 
    # nfeatures = lambda out1, out2: (nchannels+2) * out1 + (nchannels+1) * out2  # 3*xxx + 3*xxx
    # dims_1_in = [nfeatures(num_one_features + dim_1_append, num_two_features)]    
    # dims_1_in += [nfeatures(hdim[0], hdim[1]) for hdim in hidden_dims[:-1]]  
    # dims_2_in = ([num_two_features + dim_2_append] + [hdim[1] for hdim in hidden_dims[:-2]])  # [10,16,16]
    dims_1_out = [hdim[0] for hdim in hidden_dims]  # [64,64,64,64]
    dims_2_out = [hdim[1] for hdim in hidden_dims[:-1]]  # [16,16,16]
    key, subkey = jax.random.split(key)
    params['one'], params['two'] = init_layers(
        key=subkey,
        dims_one_in=dims_1_in,  
        dims_one_out=dims_1_out,  
        dims_two_in=dims_2_in,  
        dims_two_out=dims_2_out)  
    dims_orbital_in = hidden_dims[-1][0]
    dim_proj_1_in = dims_1_out[:len(dims_2_out)]  #[64,64,64]
    dim_proj_1_out = dims_2_out  # [16,16,16]
    params['proj'] = []
    for ii in range(len(params['two'])):
      if dim_proj_1_in[ii] != dim_proj_1_out[ii]:
        key, subkey = jax.random.split(key)
        params['proj'].append(network_blocks.init_linear_layer( subkey, dim_proj_1_in[ii], dim_proj_1_out[ii], ))
        #[64,64,64] 
        # |  |  |
        # |  |  |
        # V  V  V 
        #[16,16,16]
      else:
        # do not project if the input and output dims are the same
        params['proj'].append(None)
    # dim_proj_0_in = num_one_features + dim_1_append  #7
    # dim_proj_0_out = num_two_features  #4
    dim_proj_0_in = hidden_dims[0][0]
    dim_proj_0_out = hidden_dims[0][1]
    if dim_proj_0_in != dim_proj_0_out:
      key, subkey = jax.random.split(key)
      params['proj_0'] = network_blocks.init_linear_layer(subkey, dim_proj_0_in, dim_proj_0_out, )
      #7-->4
    else:
      params['proj_0'] = None


    return params, dims_orbital_in

  def construct_symmetric_features_conv(
          h1 : jnp.ndarray,
          h2 : jnp.ndarray,
          proj: Optional[Mapping[str,jnp.ndarray]] = None,
  ) -> jnp.ndarray:
    """
    hi  : ne x nfi
    hij : ne x ne x nfij
    """
    if proj is not None:
      ph1 = network_blocks.linear_layer(h1, **proj)  #(np,nf_two)
    else:
      ph1 = h1
    h2xh1 = h2 * ph1[:,None,:]   # (ne,ne,nf_two)* (ne,1,nf_two)->(ne,ne,nf_two)  p_h1 *(pointwise product) h2

    spin_partitions = network_blocks.array_partitions(nspins)
    # [(ne) x nf2, (ne) x nf2]
    g2 = [jnp.mean(h, axis=0) for h in jnp.split(h2xh1, spin_partitions, axis=0) if h.size > 0]
    g2 = jnp.concatenate(g2, axis=1)  #(ne,2*nf2)

    # [1 x nf1, 1 x nf1,]
    g1 = [jnp.mean(h, axis=0, keepdims=1) for h in jnp.split(h1, spin_partitions, axis=0) if h.size > 0]
    g1 = jnp.concatenate(g1, axis=1)  #(1,2*nf1)
    g1 = jnp.tile(g1, [h1.shape[0], 1])  #(ne,2*nf1)

    return jnp.concatenate([h1,g1,g2], axis=1)

  def _hi_next(
      hi_in,
      params,
  ):
    return activation_fn(network_blocks.linear_layer(hi_in, **params))

  def _hij_next(
      hij_in,
      params,
  ):
    return activation_fn(network_blocks.vmap_linear_layer(hij_in,params['w'],params['b'],))

  residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y 

  def apply(
      params,
      e2_features,
  ):
    # c1 = mes.get_part_one_hot()  #(nele+nz,3)
    # c2 = mes.get_pair_one_hot()  #(nele+nz,nele+nz,6)
    nele = sum(nspins)
    c1 = mes.get_part_features()
    c2 = mes.get_pair_features()
    c1 = c1[:nele,...]
    c2 = c2[:nele,:nele,...]
    h2 = e2_features[:nele,:nele,...]  #(ne,ne,nf_two)
    h1_=e2_features[:nele,...] #(ne,np,nf_two)
    h1 = jnp.mean(h1_,axis=1)  #(ne,np,nf_two)->(ne,nf_two)

    for i in range(len(params['two'])):
      if i == 0:
        # h1 = jnp.concatenate([h1, c1], axis=-1)  # (ne+na,nf_two+3)
        # h2 = jnp.concatenate([h2, c2], axis=-1)  # (ne+na,ne+na,nf_two+6)
        # h1_in = construct_symmetric_features_conv(h1, h2, params['proj_0'])

        h1 = jnp.concatenate([h1, c1], axis=-1)
        h2 = jnp.concatenate([h2, c2], axis=-1)
        h1 = network_blocks.linear_layer(h1, **params['one_reshp'])
        h2 = network_blocks.linear_layer(h2, **params['two_reshp'])
        h1 = activation_fn(h1)
        h2 = activation_fn(h2)
        h1_in = construct_symmetric_features_conv(h1, h2, params['proj_0'])
      else:
        h1_in = construct_symmetric_features_conv(h1, h2, params['proj'][i-1])
      h1_next = _hi_next(h1_in, params['one'][i])
      h2_next = _hij_next(h2, params['two'][i])
      h1 = residual(h1, h1_next)
      h2 = residual(h2, h2_next)

    assert len(params['two']) != len(params['one'])
    h1_in = construct_symmetric_features_conv(h1, h2, params['proj'][-1])
    h1_next = _hi_next(h1_in, params['one'][-1])
    h1 = residual(h1, h1_next)
    h_to_orbitals = h1

    return h_to_orbitals, None

  return FerminetModel(init, apply)


def make_fermi_net_model_ef_shrd_sym(
    natom,
    ndim,
    nspins,
    feature_layer,
    hidden_dims,   #i.e. ((64,16),(64,16),(64,16),(64,16))
    mes = None,
    activation_fn=jax.nn.silu,
    # extra parameters
):
  if mes is None :
    raise RuntimeError('make_fermi_net_model_ef only support equal-footing models')
  
  update_alpha, do_rdt, resd_dt_shift, resd_dt_scale = None, False, None, None


  def init(
      key,
  ):    
    # dim_1_append = mes.get_dim_one_hot()    #3
    # dim_2_append = 2*mes.get_dim_one_hot()   #6
    dim_1_append = mes.get_dim_part_features()
    dim_2_append = mes.get_dim_pair_features()

    # number of spin channel
    active_spin_channels = [spin for spin in nspins if spin > 0]
    nchannels = len(active_spin_channels)  #2
    # init params
    params = {}
    (num_one_features, num_two_features), params['input'] = (feature_layer.init())

    key, subkey = jax.random.split(key)
    params['one_reshp'] = network_blocks.init_linear_layer(subkey, num_one_features + dim_1_append, hidden_dims[0][0])

    key, subkey = jax.random.split(key)
    params['two_reshp'] = network_blocks.init_linear_layer(subkey, num_two_features + dim_2_append, hidden_dims[0][1])
    distinguish_ele = True
    nfeatures = lambda out1, out2: (nchannels + 2) * out1 + (nchannels + 1) * out2
    dims_1_in = [nfeatures(hidden_dims[0][0], hidden_dims[0][1])]
    dims_1_in += [nfeatures(hdim[0], hdim[1]) for hdim in hidden_dims[:-1]]
    dims_2_in = [hidden_dims[0][1]] + [hdim[1] for hdim in hidden_dims[:-2]]

 
    # nfeatures = lambda out1, out2: (nchannels+2) * out1 + (nchannels+1) * out2  # 3*xxx + 3*xxx
    # dims_1_in = [nfeatures(num_one_features + dim_1_append, num_two_features)]    
    # dims_1_in += [nfeatures(hdim[0], hdim[1]) for hdim in hidden_dims[:-1]]  
    # dims_2_in = ([num_two_features + dim_2_append] + [hdim[1] for hdim in hidden_dims[:-2]])  # [10,16,16]
    dims_1_out = [hdim[0] for hdim in hidden_dims]  # [64,64,64,64]
    dims_2_out = [hdim[1] for hdim in hidden_dims[:-1]]  # [16,16,16]
    key, subkey = jax.random.split(key)
    params['one'], params['two'] = init_layers(
        key=subkey,
        dims_one_in=dims_1_in,  
        dims_one_out=dims_1_out,  
        dims_two_in=dims_2_in,  
        dims_two_out=dims_2_out)  
    dims_orbital_in = hidden_dims[-1][0]
    dim_proj_1_in = dims_1_out[:len(dims_2_out)]  #[64,64,64]
    dim_proj_1_out = dims_2_out  # [16,16,16]
    params['proj'] = []
    for ii in range(len(params['two'])):
      if dim_proj_1_in[ii] != dim_proj_1_out[ii]:
        key, subkey = jax.random.split(key)
        params['proj'].append(network_blocks.init_linear_layer( subkey, dim_proj_1_in[ii], dim_proj_1_out[ii], ))
        #[64,64,64] 
        # |  |  |
        # |  |  |
        # V  V  V 
        #[16,16,16]
      else:
        # do not project if the input and output dims are the same
        params['proj'].append(None)
    # dim_proj_0_in = num_one_features + dim_1_append  #7
    # dim_proj_0_out = num_two_features  #4
    dim_proj_0_in = hidden_dims[0][0]
    dim_proj_0_out = hidden_dims[0][1]
    if dim_proj_0_in != dim_proj_0_out:
      key, subkey = jax.random.split(key)
      params['proj_0'] = network_blocks.init_linear_layer(subkey, dim_proj_0_in, dim_proj_0_out, )
      #7-->4
    else:
      params['proj_0'] = None


    return params, dims_orbital_in

  def construct_symmetric_features_conv(
          h1 : jnp.ndarray,
          h2 : jnp.ndarray,
          proj: Optional[Mapping[str,jnp.ndarray]] = None,
  ) -> jnp.ndarray:
    """
    hi  : np x nfi
    hij : np x np x nfij
    """
    if proj is not None:
      ph1 = network_blocks.linear_layer(h1, **proj)  #(np,nf_two)
    else:
      ph1 = h1
    h2xh1 = h2 * ph1[:,None,:]   # (np,np,nf_two)* (np,1,nf_two)->(np,np,nf_two)  p_h1 *(pointwise product) h2
    h2_upper, h2_below = mes.split_ea(h2xh1,axis=0)
    hi, hz = mes.split_ea(h1)

    spin_partitions = network_blocks.array_partitions(nspins)
    # [(nele+na) x nf2, (nele+na) x nf2, (nele+na) x nf2]
    g2 = [jnp.mean(h2_below, axis=0)]+[jnp.mean(h, axis=0) for h in jnp.split(h2_upper, spin_partitions, axis=0) if h.size > 0]
    g2 = jnp.concatenate(g2, axis=1)

    # [1 x nf1, 1 x nf1, 1 x nf1]
    g1 = [jnp.mean(hz, axis=0, keepdims=1)]+[jnp.mean(h, axis=0, keepdims=1) for h in jnp.split(hi, spin_partitions, axis=0) if h.size > 0]
    g1 = jnp.concatenate(g1, axis=1)
    g1 = jnp.tile(g1, [h1.shape[0], 1])

    return jnp.concatenate([h1,g1,g2], axis=1)

  def _hi_next(
      hi_in,
      params,
  ):
    return activation_fn(network_blocks.linear_layer(hi_in, **params))

  def _hij_next(
      hij_in,
      params,
  ):
    return activation_fn(network_blocks.vmap_linear_layer(hij_in,params['w'],params['b'],))

  residual = lambda x, y: (x + y) / jnp.sqrt(2.0) if x.shape == y.shape else y 

  def apply(
      params,
      e2_features,
  ):
    # c1 = mes.get_part_one_hot()  #(nele+nz,3)
    # c2 = mes.get_pair_one_hot()  #(nele+nz,nele+nz,6)
    c1 = mes.get_part_features()
    c2 = mes.get_pair_features()
    h2 = e2_features
    hee, _, haa = mes.split_ee_ea_aa(h2)
    ha = jnp.mean(haa, axis=0)  #\Sigma_x hxy : (na,na,nf_two)->(na,nf_two)
    he = jnp.mean(hee, axis=0)  #\Sigma_i hij :(ne,ne,nf_two)->(ne,nf_two)
    h1 = jnp.concatenate([he, ha], axis=0)    #(ne+na,nf_two)
    for i in range(len(params['two'])):
      if i == 0:
        # h1 = jnp.concatenate([h1, c1], axis=-1)  # (ne+na,nf_two+3)
        # h2 = jnp.concatenate([h2, c2], axis=-1)  # (ne+na,ne+na,nf_two+6)
        # h1_in = construct_symmetric_features_conv(h1, h2, params['proj_0'])

        h1 = jnp.concatenate([h1, c1], axis=-1)
        h2 = jnp.concatenate([h2, c2], axis=-1)
        h1 = network_blocks.linear_layer(h1, **params['one_reshp'])
        h2 = network_blocks.linear_layer(h2, **params['two_reshp'])
        h1 = activation_fn(h1)
        h2 = activation_fn(h2)
        h1_in = construct_symmetric_features_conv(h1, h2, params['proj_0'])
      else:
        h1_in = construct_symmetric_features_conv(h1, h2, params['proj'][i-1])
      h1_next = _hi_next(h1_in, params['one'][i])
      h2_next = _hij_next(h2, params['two'][i])
      h1 = residual(h1, h1_next)
      h2 = residual(h2, h2_next)
    if len(params['two']) != len(params['one']):
      h1_in = construct_symmetric_features_conv(h1, h2, params['proj'][-1])
      h1_next = _hi_next(h1_in, params['one'][-1])
      h1 = residual(h1, h1_next)
      [h_to_orbitals, hz] = mes.split_ea(h1)
    else:
      _, h_to_orbitals = construct_symmetric_features_conv(h1, h2, params['proj'][-1])
    return h_to_orbitals, hz

  return FerminetModel(init, apply)



def fermi_net_orbitals(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: FermiNetOptions = FermiNetOptions(),
):
  """Forward evaluation of the Fermionic Neural Network up to the orbitals.

  Args:
    params: A dictionary of parameters, containing fields:
      `atoms`: atomic positions, used to construct input features.
      `single`: a list of dictionaries with params 'w' and 'b', weights for the
        one-electron stream of the network.
      `double`: a list of dictionaries with params 'w' and 'b', weights for the
        two-electron stream of the network.
      `orbital`: a list of two weight matrices, for spin up and spin down (no
        bias is necessary as it only adds a constant to each row, which does not
        change the determinant).
      `dets`: weight on the linear combination of determinants
      `envelope`: a dictionary with fields `sigma` and `pi`, weights for the
        multiplicative envelope.
    pos: The electron positions, a 3N dimensional vector.
    atoms: Array with positions of atoms.
    nspins: Tuple with number of spin up and spin down electrons.
    options: Network configuration.

  Returns:
    One matrix (two matrices if options.full_det is False) that exchange columns
    under the exchange of inputs of shape (ndet, nalpha+nbeta, nalpha+nbeta) (or
    (ndet, nalpha, nalpha) and (ndet, nbeta, nbeta)) and a tuple of (ae, r_ae,
    r_ee), the atom-electron vectors, distances and electron-electron distances.
  """
  
  h_to_orbitals, hz, pairs = fermi_net_orbitals_part1_ef(params, pos, atoms, nspins, options)
  orbitals = fermi_net_orbitals_part2(params, pos, h_to_orbitals, hz, pairs, nspins, options)

  return orbitals


def fermi_net_orbitals_part1_ef(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: FermiNetOptions = FermiNetOptions(),
):

  all_pos = jnp.concatenate([pos.reshape(-1), atoms.reshape(-1)])  #(ne+na,3)
  pp, r_pp = construct_input_features_ef(all_pos)  # (ne+na,ne+na,3),(ne+na,ne+na,1)
  pp_features = options.feature_layer.apply(pp, r_pp, **params['input'])  # (ne+na,ne+na,nf_two)
  h_to_orbitals, hz = options.ferminet_model.apply(params, pp_features)
  # [h_to_orbitals, _] = options.mes.split_ea(h_to_orbitals)  #(ne,64)
  [ee, ea, aa] = options.mes.split_ee_ea_aa(pp)
  [r_ee, r_ea, r_aa] = options.mes.split_ee_ea_aa(r_pp)
  # [hi, hz] = options.mes.split_ea(hp)  #(ne,64),(na,64)
  return h_to_orbitals, hz, (ee, ea, aa, r_ee, r_ea, r_aa)


def fermi_net_orbitals_part2(
    params,
    pos,
    h_to_orbitals,
    hz,
    pair_dist,
    nspins: Tuple[int, ...],
    options: FermiNetOptions = FermiNetOptions(),
):
  model_h_to_orbitals = h_to_orbitals
  (ee, ae, aa, r_ee, r_ae, r_aa) = pair_dist

  # Note split creates arrays of size 0 for spin channels without any electrons.
  h_to_orbitals = jnp.split(h_to_orbitals, network_blocks.array_partitions(nspins), axis=0) #(nele,64)->[(n_up,64),(n_dn,64)]
  # Drop unoccupied spin channels
  h_to_orbitals = [h for h, spin in zip(h_to_orbitals, nspins) if spin > 0]  # [(n_up,64),(n_dn,64)]
  active_spin_channels = [spin for spin in nspins if spin > 0]    #[n_up,n_dn]
  active_spin_partitions = network_blocks.array_partitions(active_spin_channels)  #[n_up]
  # Create orbitals.

  orbitals = [network_blocks.linear_layer(h, **p)for h, p in zip(h_to_orbitals, params['orbital'])]
  #i.e. fulldet_type_orbitals and not do_complex:  [(n_up,64),(n_dn,64)]->[(n_up,nele*ndet),(n_dn,nele*ndet)]
  #i.e. fulldet_type_orbitals and do_complex:  [(n_up,64),(n_dn,64)]->[(n_up,nele*ndet*2),(n_dn,nele*ndet*2)]
  #i.e. not fulldet and not do_complex:  [(n_up,64),(n_dn,64)]->[(n_up,n_up*ndet),(n_dn,n_dn*ndet)]

  # if do_complex, make the orbitals complex numbers
  if options.do_complex:
    for ii in range(len(active_spin_channels)):
      nsplit = orbitals[ii].shape[-1] // 2
      split_orb = jnp.split(orbitals[ii], [nsplit], axis=-1)
      orbitals[ii] = split_orb[0] + 1j * split_orb[1]
    #[(n_up,nele*ndet*2),(n_dn,nele*ndet*2)]->[(n_up,nele*ndet),(n_dn,nele*ndet)]

  # Apply envelopes if required.
  if options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT:
    ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
    r_ae_channels = jnp.split(r_ae, active_spin_partitions, axis=0)
    r_ee_channels = jnp.split(r_ee, active_spin_partitions, axis=0)
    for i in range(len(active_spin_channels)):
      orbitals[i] = orbitals[i] * options.envelope.apply(
          ae=ae_channels[i],
          r_ae=r_ae_channels[i],
          r_ee=r_ee_channels[i],
          **params['envelope'][i],
      )
  elif options.envelope.apply_type == envelopes.EnvelopeType.PRE_DETERMINANT_Z:
    ae_channels = jnp.split(ae, active_spin_partitions, axis=0)
    for i in range(len(active_spin_channels)):
        orbitals[i] = orbitals[i] * options.envelope.apply(
        hz=hz,
        ae=ae_channels[i],
        **params['envelope'][i],     )    

  # Reshape into matrices.
  if options.det_mode == "det":
    shapes = [(spin, -1, sum(nspins) if options.full_det else spin) for spin in active_spin_channels]
    #full_det: [(n_up,-1,nele),(n_dn,-1,nele)]  else: [(n_up,-1,n_up),(n_dn,-1,n_dn)]
  else:
    shapes = [(spin, options.determinants, -1) for spin in active_spin_channels] # [(n_up,ndet,-1),(n_dn,ndet,-1)]
  orbitals = [jnp.reshape(orbital, shape) for orbital, shape in zip(orbitals, shapes)]  
  #full_det: [(n_up,nele*ndet),(n_dn,nele*ndet)]->[(n_up,ndet,nele),(n_dn,ndet,nele)] 
  #not full_det: [(n_up,n_up*ndet),(n_dn,n_dn*ndet)]->[(n_up,ndet,n_up),(n_dn,ndet,n_dn)]
  orbitals = [jnp.transpose(orbital, (1, 0, 2)) for orbital in orbitals]
  #full_det: [(n_up,ndet,nele),(n_dn,ndet,nele)]->[(ndet,n_up,nele),(ndet,n_dn,nele)]
  #not full_det: [(n_up,ndet,n_up),(n_dn,ndet,n_dn)]->[(ndet,n_up,n_up),(ndet,n_dn,n_dn)]
  if options.full_det:
    orbitals = [jnp.concatenate(orbitals, axis=1)] #[(ndet,nele,nele)]
  #本质上，ndet=1时，对于输入的orbitals_in:(nele,64)->[(n_up,64),(n_dn,64)]->[(n_up,nele),(n_dn,nele)]->[(nele,nele)] 
  #等效于：第一维是nele个轨道，第二维是nele个电子填充。
  
  if params['jastrow'] is not None:
    jastrow = options.jastrow.apply(params['jastrow'], r_ee, model_h_to_orbitals)
    orbitals = [orbital * jastrow for orbital in orbitals ]

  return orbitals, (ae, r_ae, r_ee, model_h_to_orbitals, hz)

## FermiNet ##


def fermi_net(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    return_det_dist=False,
    options: FermiNetOptions = FermiNetOptions(),
):
  """Forward evaluation of the Fermionic Neural Network for a single datum.

  Args:
    params: A dictionary of parameters, containing fields:
      `atoms`: atomic positions, used to construct input features.
      `single`: a list of dictionaries with params 'w' and 'b', weights for the
        one-electron stream of the network.
      `double`: a list of dictionaries with params 'w' and 'b', weights for the
        two-electron stream of the network.
      `orbital`: a list of two weight matrices, for spin up and spin down (no
        bias is necessary as it only adds a constant to each row, which does not
        change the determinant).
      `dets`: weight on the linear combination of determinants
      `envelope`: a dictionary with fields `sigma` and `pi`, weights for the
        multiplicative envelope.
    pos: The electron positions, a 3N dimensional vector.
    atoms: Array with positions of atoms.
    nspins: Tuple with number of spin up and spin down electrons.
    options: network options.

  Returns:
    Output of antisymmetric neural network in log space, i.e. a tuple of sign of
    and log absolute of the network evaluated at x.
  """

  orbitals, (ae, r_ae, r_ee, he, hz) = fermi_net_orbitals(
      params,
      pos,
      atoms=atoms,
      nspins=nspins,
      options=options,
  )

  sign_out, log_out, logdet = network_blocks.logdet_matmul(orbitals, w=params['det'], do_complex=options.do_complex)

  if return_det_dist:
    return jax.nn.log_softmax(logdet)

  return sign_out, log_out


def orbitals_fn(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: FermiNetOptions = FermiNetOptions(),
):

  orbitals, _ = fermi_net_orbitals(
      params,
      pos,
      atoms=atoms,
      nspins=nspins,
      options=options,
  )
  return orbitals


def return_hz_fn(
    params,
    pos: jnp.ndarray,
    atoms: jnp.ndarray,
    nspins: Tuple[int, ...],
    options: FermiNetOptions = FermiNetOptions(),
):
  orbitals, (ae, r_ae, r_ee, he, hz) = fermi_net_orbitals(
      params,
      pos,
      atoms=atoms,
      nspins=nspins,
      options=options,
  )
  return hz

def make_fermi_net(
    nele: int, 
    ndim:int, 
    nspins: Tuple[int, int],
    charges: jnp.ndarray,
    *,
    envelope: Optional[envelopes.Envelope] = None,
    feature_layer: Optional[FeatureLayer] = None,
    ferminet_model: Optional[FerminetModel] = None,
    bias_orbitals: bool = False,
    full_det: bool = True,
    hidden_dims: FermiLayers = ((256, 32), (256, 32), (256, 32)),
    determinants: int = 16,
    do_complex: bool = False,
    mes: dp.ManyElectronSystem = None,
    det_mode: str = 'det',
    jastrow : Optional[JastrowModel] = None,
) -> Tuple[InitFermiNet, FermiNetLike, FermiNetOptions]:
  """Creates functions for initializing parameters and evaluating ferminet.

  Args:
    atoms: (natom, ndim) array of atom positions.
    nspins: Tuple of the number of spin-up and spin-down electrons.
    charges: (natom) array of atom nuclear charges.
    envelope: Envelope to use to impose orbitals go to zero at infinity.
    feature_layer: Input feature construction.
    bias_orbitals: If true, include a bias in the final linear layer to shape
      the outputs into orbitals.

    full_det: If true, evaluate determinants over all electrons. Otherwise,
      block-diagonalise determinants into spin channels.
    hidden_dims: Tuple of pairs, where each pair contains the number of hidden
      units in the one-electron and two-electron stream in the corresponding
      layer of the FermiNet. The number of layers is given by the length of the
      tuple.
    determinants: Number of determinants to use.

  Returns:
    init, network, options tuple, where init and network are callables which
    initialise the network parameters and apply the network respectively, and
    options specifies the settings used in the network.
  """
  natom=len(charges)
  
  assert (det_mode == "det"),"only det mode is supported"
  assert (envelope and feature_layer and ferminet_model),("no envelope/feature_layer/ferminet_model")

  options = FermiNetOptions(
      hidden_dims=hidden_dims,
      determinants=determinants,
      full_det=full_det,
      bias_orbitals=bias_orbitals,
      envelope=envelope,
      feature_layer=feature_layer,
      ferminet_model=ferminet_model,
      do_complex=do_complex,
      mes=mes,
      det_mode=det_mode,
      jastrow=jastrow,
      charges=jnp.asarray(charges),
  )

  init = functools.partial(
      init_fermi_net_params,
      natom = natom, 
      ndim = ndim, 
      nspins=nspins,
      options=options,
  )

  network = functools.partial(
      fermi_net,
      nspins=nspins,
      return_det_dist=False,
      options=options,
  )

  det_fn = functools.partial(
      fermi_net,
      nspins=nspins,
      return_det_dist=True,
      options=options,
  )

  orb_fn = functools.partial(
      orbitals_fn,
      nspins=nspins,
      options=options, 
  )
   
  hz_fn = functools.partial(
      return_hz_fn,
      nspins=nspins,
      options=options, 
  )

  return init, network, det_fn, options, orb_fn, hz_fn
