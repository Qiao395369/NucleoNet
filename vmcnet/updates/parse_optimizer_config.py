"""Get update functions from ConfigDicts."""

from typing import Optional, Tuple
from ml_collections import ConfigDict

import vmcnet.physics as physics
from vmcnet.utils.typing import (
    ClippingFn,
    D,
    GetPositionFromData,
    LearningRateSchedule,
    LocalEnergyApply,
    ModelApply,
    OptimizerState,
    P,
    PRNGKey,
    UpdateDataFn,
)
import vmcnet.utils as utils
from .update_param_fns import UpdateParamFn
import logging
from .spring import spring_wrapper, Spring
import jax

def _get_learning_rate_schedule(
    optimizer_config: ConfigDict,
) -> LearningRateSchedule:
    if optimizer_config.schedule_type == "constant":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate

    elif optimizer_config.schedule_type == "inverse_time":

        def learning_rate_schedule(t):
            return optimizer_config.learning_rate / (
                1.0 + optimizer_config.learning_decay_rate * t
            )

    else:
        raise ValueError(
            "Learning rate schedule type not supported; {} was requested".format(
                optimizer_config.schedule_type
            )
        )

    return learning_rate_schedule

def _get_damping_rate_schedule(
    vmc_config: ConfigDict,
) -> LearningRateSchedule:
    optimizer_config = vmc_config.optimizer[vmc_config.optimizer_type]
    if optimizer_config.damp_schedule_type == "constant":

        return lambda t : optimizer_config.damping

    elif optimizer_config.damp_schedule_type == "inverse_time":

        return _get_InverseSchedule(
            optimizer_config.damping, max(vmc_config.nepochs // 100, 1), optimizer_config.damping / 1000
            )

    else:
        raise ValueError(
            "damping rate schedule type not supported; {} was requested".format(
                optimizer_config.damp_schedule_type
            )
        )


def _get_InverseSchedule(init_value, decay_rate, offset=0.0):
        return lambda n: (init_value - offset) / (1 + n / decay_rate) + offset

def initialize_optimizer(
    log_psi_apply_novmap: ModelApply[P],
    energy_and_statistics_fn,
    vmc_config: ConfigDict,
    params: P,
    update_data_fn: UpdateDataFn[D, P],
    key: PRNGKey,
    apply_pmap: bool = False,
) -> Tuple[UpdateParamFn[P, D, OptimizerState], OptimizerState, PRNGKey]:
    """Get an update function and initialize optimizer state from the vmc configuration."""
    learning_rate_schedule = _get_learning_rate_schedule(
        vmc_config.optimizer[vmc_config.optimizer_type]
    )
    optimizer_config=vmc_config.optimizer[vmc_config.optimizer_type]
    logging.info("Using optimizer type: {}".format(vmc_config.optimizer_type))

    if vmc_config.optimizer_type == "spring":
        damping_rate_schedule = _get_damping_rate_schedule(vmc_config)
        opt_kwargs = {}
        opt_kwargs["mu"] = optimizer_config.mu
        opt_kwargs["norm_constraint"] = optimizer_config.norm_constraint
        opt_kwargs["learning_rate_schedule"] = learning_rate_schedule
        opt_kwargs["damping_schedule"] = damping_rate_schedule
        opt_kwargs["repeat_single_mol"] = vmc_config.repeat_single_mol
        opt = spring_wrapper(Spring(**opt_kwargs), log_psi_apply_novmap, update_data_fn, energy_and_statistics_fn)
        # init_fn = opt.init
        # update_param_fn = opt.step
        if apply_pmap:
            get_grad_and_E = utils.distribute.pmap(opt.grad_and_E)
            update_param_fn = utils.distribute.pmap(opt.update)
            init_fn = utils.distribute.pmap(opt.init)
        else:
            get_grad_and_E = jax.jit(opt.grad_and_E)
            update_param_fn = jax.jit(opt.update)
            init_fn = jax.jit(opt.init)
        optimizer_state = init_fn(params)
        return get_grad_and_E, update_param_fn, optimizer_state, key
    
    else:
        raise ValueError(
            "Requested optimizer type not supported; {} was requested".format(
                vmc_config.optimizer_type
            )
        )
