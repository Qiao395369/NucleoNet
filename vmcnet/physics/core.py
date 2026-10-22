"""Core local energy and gradient construction routines."""

from typing import Callable, Optional, Sequence, Tuple, cast
import logging
import chex
import jax
import jax.numpy as jnp
from vmcnet.utils.distribute import PMAP_AXIS_NAME

import vmcnet.utils as utils
from vmcnet.utils.typing import (
    Array,
    ArrayLike,
    P,
    ClippingFn,
    PRNGKey,
    LocalEnergyApply,
    ModelApply,
    Dict,
    Any,
)
EnergyAuxData = Dict[str, Any]
ValueGradEnergyFn = Callable[[P, Array, Array], Tuple[Array, EnergyAuxData, P]]


def initialize_molecular_pos(
    key: PRNGKey,
    nchains: int,
    ion_pos: Array,
    ion_charges: Array,
    nelec_total: int,
    single_spins:Array,
    init_width: float = 0.2,
    dtype=chex.Numeric,
) -> Tuple[PRNGKey, Array]:
    natoms=len(ion_charges)
    walker=ion_pos.shape[0]
    assert ion_pos.shape==(walker,natoms,3)
    assert nelec_total==jnp.sum(single_spins)
    # print(single_spins)  # single_spins=[[5,2],[2,5]]
    # Assign each electron to an atom initially.
    ppp=[]
    for k in range(walker):
        electron_positions = []
        for i in range(2):
            for j in range(natoms):
                position = jnp.asarray(ion_pos[k][j])
                electron_positions.append(jnp.repeat(position[None, :], single_spins[j][i], axis=0))
        # print(electron_positions)
        # print(electron_positions.shape)
        ppp.append(jnp.concatenate(electron_positions,axis=0))
    ppp=jnp.stack(ppp,axis=0)
    ppp=ppp[:,None,...]
    key, subkey = jax.random.split(key)
    ppp += jax.random.normal(subkey, shape=(walker,nchains,)+ppp.shape[-2:] , dtype=dtype) * init_width
    logging.info("init_xp: %s \n " \
    "         init_xe: %s", ion_pos.shape, ppp.shape)
    return key, ppp

def combine_local_energy_terms(
    local_energy_terms: Sequence[ModelApply[P]],
) -> LocalEnergyApply[P]:
    """Combine a sequence of local energy terms by adding them.

    Args:
        local_energy_terms (Sequence): sequence of local energy terms, each with the
            signature (params, x) -> array of terms of shape (x.shape[0],)

    Returns:
        Callable: local energy function which computes the sum of the local energy
        terms. Has the signature
        (params, x) -> local energy array of shape (x.shape[0],)
    """

    def local_energy_fn(params: P, atoms_position: Array, elec_position: Array) -> Array:
        local_energy_sum = local_energy_terms[0](params, atoms_position, elec_position)
        for term in local_energy_terms[1:]:
            local_energy_sum = cast(Array, local_energy_sum + term(params, atoms_position, elec_position))
        return local_energy_sum

    return local_energy_fn

def get_statistics_from_local_energy(
    local_energies: Array, nan_safe: bool ,
) -> Tuple[Array, Array, Array]:
    """Collectively reduce local energies to an average energy and variance.

    Args:
        local_energies (Array): local energies of shape (nchains,), possibly
            distributed across multiple devices via utils.distribute.pmap.
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        nan_safe (bool, optional): flag which controls if jnp.nanmean is used instead of
            jnp.mean. Can be set to False when debugging if trying to find the source of
            unexpected nans. Defaults to True.

    Returns:
        (chex.Numeric, chex.Numeric): local energy average, local energy (sample)
        variance
    """
    # TODO(Jeffmin) might be worth investigating the numerical stability of the XLA
    # compiled version of these two computations, since the quality of the gradients
    # is fairly crucial to the success of the algorithm
    assert len(local_energies.shape) == 2  # local_energies:(W,B)
    W, B = local_energies.shape
    if nan_safe:
        allreduce_mean = utils.distribute.nanmean_all_local_devices
        w_mean = jnp.nanmean
    else:
        allreduce_mean = utils.distribute.mean_all_local_devices
        w_mean = jnp.mean

    energy_per_w = w_mean(local_energies, axis=1, keepdims=True)  # (W,1)
    var_per_w = jnp.sum(jnp.square(local_energies - energy_per_w), axis=1) / jnp.maximum(B - 1, 1)  # (W,)
    variance = allreduce_mean(var_per_w, axis=0)  # ()
    
    return energy_per_w, var_per_w ,variance


def get_clipped_energies_and_stats(
    local_energies_noclip: Array,
    clipping_fn: Optional[ClippingFn],
    nan_safe: bool,
) -> Tuple[Array, Array, EnergyAuxData]:
    """Clip local energies if requested and return auxiliary data."""
    energy_per_w_noclip, var_per_w_noclip, variance_noclip = get_statistics_from_local_energy(local_energies_noclip, nan_safe=nan_safe)

    if clipping_fn is not None:
        local_energies = clipping_fn(local_energies_noclip, energy_per_w_noclip)  #local_energies_noclip:(W,B)， energy_noclip:(W,1)-->local_energies: (W,B)
        energy_per_w, var_per_w, variance = get_statistics_from_local_energy(local_energies, nan_safe=nan_safe)  #energy: (W,1)  variance:(1,)
    else:
        local_energies, energy_per_w, var_per_w, variance= local_energies_noclip, energy_per_w_noclip, var_per_w_noclip, variance_noclip
    
    energy_stats = dict(
        variance=variance,  #()
        variance_noclip=variance_noclip,  #()
        multi_variance = jnp.maximum(var_per_w_noclip, var_per_w),    #(w,)
        energy_noclip=utils.distribute.nanmean_all_local_devices(energy_per_w_noclip,axis=(0,1)),  #()
        multi_energy = jnp.squeeze(energy_per_w, axis=-1),
        multi_energy_noclip = jnp.squeeze(energy_per_w_noclip, axis=-1)
    )

    return energy_per_w, local_energies, energy_stats


def create_energy_and_statistics_fn(
    local_energy_fn,
    clipping_fn: Optional[ClippingFn] = None,
    nan_safe: bool = True,
) -> ValueGradEnergyFn[P]:
    """Create a function which computes energies and associated statistics.

    Args:
        log_psi_apply (Callable): computes log|psi(x)|, where the signature of this
            function is (params, x) -> log|psi(x)|
        local_energy_fn (Callable): computes local energies Hpsi / psi. Has signature
            (params, x) -> (Hpsi / psi)(x)
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        clipping_fn (Callable, optional): post-processing function on the local energy,
            e.g. a function which clips the values to be within some multiple of the
            total variation from the median. The post-processed values are used for
            the gradient calculation, if available. Defaults to None.
        nan_safe (bool, optional): flag which controls if jnp.nanmean and jnp.nansum are
            used instead of jnp.mean and jnp.sum for the terms in the gradient
            calculation. Can be set to False when debugging if trying to find the source
            of unexpected nans. Defaults to True.

    Returns:
        Callable: function which computes the clipped energy and associated statistics.
        Has the signature
            (params, positions)
            -> (expected_energy, auxiliary_energy_data)
        where auxiliary_energy_data is the tuple
        (expected_variance, local_energies, unclipped_energy, unclipped_variance, centered_local_energies)
    """

    def energy_and_statistics(params,atoms_positions, positions):
        '''
        atoms_positions:(W,natom,dim)
        positions:(W,B,nele,dim)
        '''
        local_energies_noclip = jax.vmap(jax.vmap(local_energy_fn,in_axes=(None,None,0)),in_axes=(None,0,0))(params,atoms_positions, positions)  #(W,B)

        energy_per_w, local_energies, energy_stats = get_clipped_energies_and_stats(
            local_energies_noclip, clipping_fn, nan_safe
        )

        return energy_per_w, local_energies, energy_stats

    return energy_and_statistics