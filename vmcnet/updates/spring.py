"""SPRING implementation, see https://doi.org/10.1016/j.jcp.2024.113351."""

from typing import Callable, Dict, Tuple, Any, TypeAlias, Callable, NamedTuple, Protocol, Optional
import jax
import jax.flatten_util
import jax.numpy as jnp
from ml_collections import ConfigDict
import chex
from functools import partial
import logging
from vmcnet.utils.typing import Array, D, ModelApply, P, S
from vmcnet.utils.pytree_helpers import (
    multiply_tree_by_scalar,
    tree_inner_product,
)
from vmcnet.utils.distribute import pmean_if_pmap
from vmcnet.utils.distribute import PMAP_AXIS_NAME


OptimizerState: TypeAlias = Dict
WavefunctionParams: TypeAlias = Dict
class OptInitFunction(Protocol):
    """Protocol for optimizer initialization functions."""

    def __call__(
        self,
        params: P,
    ) -> OptimizerState:
        ...


class OptStepFunction(Protocol):
    """Protocol for optimizer step functions."""

    def __call__(
        self,
        params: P,
        data: D,
        opt_state: OptimizerState,
    ) -> Tuple[P,D, OptimizerState, Dict[str, Any]]:
        ...

class Optimizer(NamedTuple):
    """Optimizer interface with init and step function."""

    init: OptInitFunction
    grad_and_E: OptInitFunction
    update: OptStepFunction


def spring_wrapper(
    spring_opt,
    log_psi_apply,
    update_data_fn,
    energy_and_statistics_fn,
) -> Optimizer:

    def init(
        params: P,
    ) -> OptimizerState:
        return spring_opt.init(params)

    @partial(jax.vmap, in_axes=(None, 0, 0))
    @partial(jax.vmap, in_axes=(None, None, 0))
    def raveled_log_psi_grad(params: P, ion_pos: Array, positions: Array) -> Array:
        log_grads = jax.grad(log_psi_apply)(params, ion_pos, positions)
        return jax.flatten_util.ravel_pytree(log_grads)[0]

    def grad_and_E(
        key,
        params: P,
        opt_state: OptimizerState,
        data: D,
        grad_acc,
        metrics_acc,
    ) -> tuple[P, D, OptimizerState, Dict]:
        position = data["walker_data"]["elec_position"]
        atoms_position = data["atoms_position"]
        
        energy_per_w, E_loc, stats = energy_and_statistics_fn(params, atoms_position, position)
        log_psi_grads = raveled_log_psi_grad(params, atoms_position, position)
        grad, _, E_mean = spring_opt.get_grad(log_psi_grads, E_loc, energy_per_w, opt_state)

        metrics = {
            "energy": E_mean,
            "variance": stats["variance"],
            "variance_noclip": stats["variance_noclip"],
            "multi_variance": stats["multi_variance"],
            "energy_noclip": stats["energy_noclip"],
            "multi_energy": stats["multi_energy"],
            "multi_energy_noclip": stats["multi_energy_noclip"],
        }
        grad_acc = jax.tree_util.tree_map(lambda acc, i: acc + i, grad_acc, grad)
        metrics_acc = jax.tree_util.tree_map(lambda acc, i: acc + i, metrics_acc, metrics)
        return grad_acc, metrics_acc, opt_state, data, key

    def update(
        key,
        params: P,
        opt_state: OptimizerState,
        data: D,
        grad,
    ) -> tuple[P, D, OptimizerState, Dict]:
        updates, opt_state = spring_opt.update(grad, opt_state)
        params = apply_updates(params, updates)
        data = update_data_fn(data, params)

        return params, data, opt_state, key

    return Optimizer(init=init, grad_and_E=grad_and_E, update=update)

class Spring:

    def __init__(
        self,
        mu: float,
        norm_constraint: float,
        learning_rate_schedule: Callable[[int], float],
        damping_schedule: Callable[[int], float],
        repeat_single_mol: bool = False,
    ):
        self.mu = mu
        self.norm_constraint = norm_constraint
        self.lr_schedule = learning_rate_schedule
        self.dp_schedule = damping_schedule
        self.repeat_single_mol = repeat_single_mol

    def init(self, params: P):
        opt_state = {
            "prev_grad": jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), params),
            "step": jnp.asarray(0, dtype=jnp.int32),
        }
        return opt_state

    def get_grad(
        self,
        log_psi_grads,
        E_loc: P,
        E_mean_per_mol,
        opt_state,
    ) -> Tuple[Array, P]:
        prev_grad, unravel_fn = jax.flatten_util.ravel_pytree(opt_state["prev_grad"])
        prev_grad_decayed = self.mu * prev_grad  #(nparams,)
        W, B, nparams = log_psi_grads.shape
        nchains = W * B
        log_psi_grads = log_psi_grads.reshape((W * B, nparams)) / jnp.sqrt(nchains)  #(W*B,nparams)
        Ohat = log_psi_grads - jnp.mean(log_psi_grads, axis=0, keepdims=True)  #(W*B,nparams)
        T = Ohat @ Ohat.T  #(W*B,W*B)
        ones = jnp.ones((nchains, 1)) #(W*B,1)
        T_reg = T + ones @ ones.T / nchains + self.dp_schedule(opt_state["step"]) * jnp.eye(nchains)  #(W*B,W*B)
        E_mean = jnp.mean(E_mean_per_mol, keepdims=True)  #(1,1)
        E_mean = pmean_if_pmap(E_mean)  #(1,1)
        if self.repeat_single_mol:
            E_mean_per_mol = E_mean
        centered_energies = E_loc - E_mean_per_mol
        epsilon_bar = centered_energies.reshape((-1,)) / jnp.sqrt(nchains) #(W*B,)
        epsion_tilde = epsilon_bar - Ohat @ prev_grad_decayed   #(W*B,)
        dtheta_residual = Ohat.T @ jax.scipy.linalg.solve(T_reg, epsion_tilde, assume_a="pos") #(nparams,)
        dtheta_residual = pmean_if_pmap(dtheta_residual)
        SR_G = dtheta_residual + prev_grad_decayed
        return unravel_fn(SR_G), None, jnp.squeeze(E_mean)

    def update(
        self, grad, opt_state: OptimizerState
    ) -> tuple[WavefunctionParams, OptimizerState]:

        update = jax.tree_util.tree_map(lambda x: -self.lr_schedule(opt_state["step"]) * x, grad)
        update = constrain_norm(update, self.norm_constraint)

        return update, {
            "prev_grad": grad,
            "step": opt_state["step"] + 1,
        }


def apply_updates(params: WavefunctionParams, updates: WavefunctionParams) -> WavefunctionParams:
    """Apply updates to wave function parameters."""

    return jax.tree_util.tree_map(
        lambda p, u: jnp.asarray(p + u).astype(jnp.asarray(p).dtype), params, updates
    )


def constrain_norm(
    grad: P,
    norm_constraint: chex.Numeric = 0.001,
) -> P:
    """Euclidean norm constraint."""
    sq_norm_scaled_grads = tree_inner_product(grad, grad)

    # Sync the norms here, see:
    # https://github.com/deepmind/deepmind-research/blob/30799687edb1abca4953aec507be87ebe63e432d/kfac_ferminet_alpha/optimizer.py#L585
    sq_norm_scaled_grads = pmean_if_pmap(sq_norm_scaled_grads)

    norm_scale_factor = jnp.sqrt(norm_constraint / sq_norm_scaled_grads)
    coefficient = jnp.minimum(norm_scale_factor, 1)
    constrained_grads = multiply_tree_by_scalar(grad, coefficient)

    return constrained_grads