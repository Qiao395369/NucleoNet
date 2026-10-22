"""Routines which handle model parameter updating."""

from typing import Callable, Dict, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import vmcnet.physics as physics
import vmcnet.utils as utils
from vmcnet.utils.typing import (
    Array,
    D,
    OptimizerState,
    P,
    PRNGKey,
    S,
)

UpdateParamFn = Callable[[P, D, S, PRNGKey], Tuple[P, D, S, Dict, PRNGKey]]

def construct_eval_update_param_fn(
    local_energy_fn,
    nan_safe: bool = False,
    apply_pmap: bool = True,
) -> UpdateParamFn[P, D, OptimizerState]:
    """No update/clipping/grad function which simply evaluates the local energies.

    Can be used to do simple unclipped MCMC with :func:`~vmcnet.train.vmc.vmc_loop`.

    Arguments:
        local_energy_fn (Callable): computes local energies Hpsi / psi. Has signature
            (params, x) -> (Hpsi / psi)(x)
        nchains (int): total number of chains across all devices, used to compute a
            sample variance estimate of the local energy
        get_position_fn (GetPositionFromData): gets the walker positions from the MCMC
            data.
        nan_safe (bool): whether or not to mask local energy nans in the evaluation
            process. This option should not be used under normal circumstances, as the
            energy estimates are of unclear validity if nans are masked. However,
            it can be used to get a coarse estimate of the energy of a wavefunction even
            if a few walkers are returning nans for their local energies.

    Returns:
        Callable: function which evaluates the local energies and averages them, without
        updating the parameters
    """

    def eval_get_grad_and_E(key, params, optimizer_state, data, grad_acc, metrics_acc):
        positions = data["walker_data"]["elec_position"]
        atoms_positions = data["atoms_position"]

        local_energies=jax.vmap(jax.vmap(local_energy_fn, in_axes=(None,None,0)),in_axes=(None,0,0))(params, atoms_positions, positions) #(W,B)
        # kinetic,ei_potential,ee_potential,ii_potential = physics.core.get_statistics_from_other_energy(kinetic,ei_potential,ee_potential,ii_potential, nan_safe=nan_safe) #()

        energy_per_w, var_per_w, variance = physics.core.get_statistics_from_local_energy(
            local_energies, nan_safe=nan_safe
        )

        metrics = {"energy": utils.distribute.nanmean_all_local_devices(energy_per_w,axis=(0,1)),  #()
                   "variance": variance,  #()
                   "multi_variance": var_per_w,  #(W,)
                   "multi_energy": local_energies,  #(W,B)
                   }
        
        return grad_acc, metrics, optimizer_state, data, key
    
    def eval_update_param_fn(key, params, optimizer_state, data, grad):
        return params, data, optimizer_state, key

    if not apply_pmap:
        return jax.jit(eval_get_grad_and_E), jax.jit(eval_update_param_fn)
    else:
        return utils.distribute.pmap(eval_get_grad_and_E), utils.distribute.pmap(eval_update_param_fn)
