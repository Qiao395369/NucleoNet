"""Create configuration of hyperparameters."""

import os
from typing import Dict

from ml_collections import ConfigDict, FieldReference

from vmcnet.utils.checkpoint import DEFAULT_CHECKPOINT_FILE_NAME

NO_NAME = "NONE"
NO_PATH = "NONE"
NO_RELOAD_LOG_DIR = "NONE"
DEFAULT_CONFIG_FILE_NAME = "config.json"
DEFAULT_PRESETS_DIR = "preset_configs"


def _copy_all_dicts(config: Dict) -> Dict:
    """Recursively copy the top-level Dict and all sub-Dicts.

    This ensures that command line flags used to override fields of the config will only
    apply to the intended flag, even if the code used to generate the default values is
    shared among several different configuration flags. For example, the following code
    could be problematic:

    subconfig = {"val": 2}
    config = {
        "sub1": subconfig,
        "sub2": subconfig
    }

    If this config is used directly, setting a command line flag like
    --config.sub1.val=3 will override both config.sub1.val and config.sub2.val, which is
    not the intended behavior. Calling config=copy_all_dicts(config) before turning this
    into a ConfigDict solves the problem by making separate copies of both subconfigs.

    Note: using copy.deepcopy is not a valid replacement for this method, as deepcopy
    will not generate multiple copies of the same object if encountered multiple times.
    In the example above, deepcopy will make a single copy subconfig_copy of subconfig,
    and use it for both sub1 and sub2 in the returned copy. This does not solve the
    problem!
    """
    result = {}
    for key in config:
        if isinstance(config[key], Dict):
            result[key] = _copy_all_dicts(config[key])
        else:
            result[key] = config[key]
    return result


def get_default_reload_config() -> ConfigDict:
    """Make a default reload configuration (no logdir but valid defaults otherwise)."""
    return ConfigDict(
        {
            "logdir": NO_RELOAD_LOG_DIR,
            "use_config_file": True,
            "config_relative_file_path": DEFAULT_CONFIG_FILE_NAME,
            "use_checkpoint_file": True,
            "checkpoint_relative_file_path": DEFAULT_CHECKPOINT_FILE_NAME,
            "new_optimizer_state": False,
            "reburn": False,
            "append": True,
            "same_logdir": False,
            "to_pmap": False,
            "new_data": False,
            "nchains": 0,
            "end_epochs": 150000,
            "nburn":0,
        }
    )


def get_default_config() -> ConfigDict:
    """Make a default NucleoNet configuration for equilibrium N2."""
    config = ConfigDict(
        _copy_all_dicts(
            {
                "notes": "default",
                "problem": get_default_molecular_config(),
                "pretrain": get_default_pretrain_config(),
                "vmc": get_default_vmc_config(),
                "eval": get_default_eval_config(),
                "wfn_type":"gaoqiao",
                "gq": get_default_gq_config(),
                "logdir": os.path.join(
                    os.curdir,  # this will be relative to the calling script
                    "logs",
                ),
                # if save_to_current_datetime_subfolder=True, will log into a subfolder
                # named according to the datetime at start
                "save_to_current_datetime_subfolder": True,
                "subfolder_name": NO_NAME,
                "logging_level": "INFO",
                "dtype": "float64",
                "distribute": False,
                "debug_nans": False,  # If true, OVERRIDES config.distribute to be False
                "initial_seed": 0,
            }
        )
    )
    config.base_logdir = config.logdir
    return config


def get_default_molecular_config() -> Dict:
    """Get a default molecular configuration (N2, coordinates in bohr)."""
    problem_config = {
        "ion_pos": ((0.0, 0.0, -1.034), (0.0, 0.0, 1.034)),
        "ion_charges": (7.0, 7.0),
        "atoms_symbol" : ("N", "N"),
        "nspins": (7, 7),
        "single_nspins":((5,2),(2,5)),
        "ei_softening": 0.0,
        "ee_softening": 0.0,
    }
    return problem_config


def get_default_vmc_config() -> Dict:
    """Get a default VMC training configuration."""
    vmc_config = {
        "nchains": 4096,
        "repeat_single_mol": False,
        "repeat_single_molecule_walker": 0,
        "acc_steps": 1,
        "nepochs": 150000,
        "nburn": 4000,
        "fwdlap_inner_size": 0,
        "nsteps_per_param_update": 40,
        "nmoves_per_width_update": 100,
        "std_move": 0.25,
        "init_width":0.5,
        "checkpoint_every": 10000,
        "best_checkpoint_every": 10000,
        "checkpoint_dir": "checkpoints",
        "checkpoint_variance_scale": 10,
        "check_for_nans": True,
        "nhistory_max": 200,
        "record_amplitudes": False,
        "clip_threshold": 5.0,
        "clip_center": "mean",  # mean or median
        "nan_safe": True,
        "optimizer_type": "spring",
        "optimizer": {
            "spring": {
                # Learning rate settings
                "schedule_type": "inverse_time",  # constant or inverse_time
                "learning_rate": 5e-3,
                "learning_decay_rate": 1e-4,
                # SPRING hyperparams
                "mu": 0.99,
                "damp_schedule_type": "inverse_time",  # constant or inverse_time
                "damping": 0.001,
                "norm_constraint": 0.001,
            },
        },
    }
    return vmc_config

def get_default_pretrain_config() -> Dict:
    """Get a default VMC training configuration."""
    pretrain_config = {
        "optim": "adam", #choices=["adam", "lamb"]
        "nburn": 0,
        "iterations": 4000,
        "nsteps_per_param_update": 20,
        "nmoves_per_width_update": 100,
        "loss_mode":'fulldet',  #choices=['fulldet', 'blockdiag', 'l1', 'l2]
        "eps":1e-4,
        "offblock_lambda":0.0,
        "hf_coeff":1,
    }
    return pretrain_config


def get_default_gq_config() -> Dict:
    """Get a default attention configuration."""
    gq_config = {
        "det_mode": "det", 
        "feat_do_act": False,
        "feat_act_func": 'tanh',
        "feat_numb_divid": 1,
        "scale": 1.0,
        "dp_type":"full",  #["original", "charge", "full","element","element1"]
        "type":"ef_shrd_sym",  #["ef","fermi","shrd","ef_shrd","ef_shrd_sym","ef_shrd_sym_old","ef_shrd_sym_ablation"...]
        "envelope_type": "ds_hz_deg", #["ds_hz","iso","ds_hz_pos","ds_hz_deg"]
        "ndet":16,
        "wfn_depth":4,
        "h1":64,
        "h2":16,
        "nh":16,
        "do_complex":False,
        "jastrow_type": "mlp_res", #choices=["mlp_res", "simple_ee","none"]
        "jastrow_activation_type": "tanh",
        "jastrow_mlp_nlayer": 4,
        "jastrow_mlp_ndim": 32,
        "rescale_input": "all", #choices=["all", "all_eps", "e-a", "None"]
        "activation_type": "silu", #choices=["tanh", "relu", "silu]
    }
    return gq_config

def get_default_eval_config() -> Dict:
    """Get a default evaluation configuration."""
    eval_config = {
        "nchains": 128,
        "nburn": 40000,
        "nepochs": 50000,
        "repeat_single_mol": False,
        "repeat_single_molecule_walker": 0,
        "nsteps_per_param_update": 40,
        "nmoves_per_width_update": 100,
        "record_amplitudes": False,
        "std_move": 0.25,
        "init_width":0.5,
        # if use_data_from_training=True, nchains, nmoves_per_width_update, and
        # std_move are completely ignored, and the data output from training is
        # used as the initial positions instead
        "use_data_from_training": False,
        "record_local_energies": True,  # save local energies and compute statistics
        "nan_safe": False,
        "ion_pos": ((0.0, 0.0, -1.034), (0.0, 0.0, 1.034)),
        "ion_charges":(7.,7.),
        "single_nspins":((5,2),(2,5)),
        "nspins":(7,7),
    }
    return eval_config
