"""Small fixtures and assertions shared only by the new suite."""

from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np


FEATURE_RESCALE = "all"


def assert_tree_close(actual, expected, rtol=1e-9, atol=1e-10):
    a, a_structure = jax.tree_util.tree_flatten(actual)
    b, b_structure = jax.tree_util.tree_flatten(expected)
    assert a_structure == b_structure
    for x, y in zip(a, b):
        assert np.shape(x) == np.shape(y)
        np.testing.assert_allclose(x, y, rtol=rtol, atol=atol)


def assert_tree_finite(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        assert np.isfinite(np.asarray(leaf)).all()


def log_gaussian(params, atoms, electrons):
    """log|psi| for a Gaussian centered at the first nucleus."""
    return -params["alpha"] * jnp.sum((electrons - atoms[0]) ** 2)


def batch_log_gaussian(params, atoms, electrons):
    return jax.vmap(jax.vmap(log_gaussian, in_axes=(None, None, 0)),
                    in_axes=(None, 0, 0))(params, atoms, electrons)


def gaussian_data(w=2, b=6, ne=2):
    from vmcnet.mcmc.dynamic_width_position_amplitude import (
        make_dynamic_width_position_amplitude_data,
    )
    atoms = jnp.arange(w * 3, dtype=jnp.float64).reshape(w, 1, 3) / 7
    offsets = jax.random.normal(jax.random.PRNGKey(17), (w, b, ne, 3)) * 0.5
    electrons = atoms[:, None, :1, :] + offsets
    params = {"alpha": jnp.asarray(0.4)}
    data = make_dynamic_width_position_amplitude_data(
        atoms, electrons, batch_log_gaussian(params, atoms, electrons),
        jnp.full((w,), 0.3), jnp.zeros(w), jnp.zeros(w, dtype=int),
    )
    return params, data, jax.random.PRNGKey(11)


@lru_cache(maxsize=None)
def small_model(gq_type="ef_shrd_sym", envelope_type=None, jastrow_type="mlp_res",
                activation_type="silu", dp_type="full", do_complex=False):
    from vmcnet.gaoqiao.build import build_network
    if envelope_type is None:
        envelope_type = "ds_hz_deg" if gq_type == "ef_shrd_sym_ablation" else "ds_hz"
    return build_network(
        n=4, charges=jnp.array([2., 2.]), nspins=(2, 2),
        key=jax.random.PRNGKey(7), ndet=2, depth=2, h1=8, h2=4, nh=4,
        do_complex=do_complex, gq_type=gq_type, envelope_type=envelope_type,
        feat_params={"scale": [], "rescale": FEATURE_RESCALE, "numb_divid": 1},
        jastrow_type=jastrow_type, jastrow_mlp_nlayer=2, jastrow_mlp_ndim=8,
        activation_type=activation_type, dp_type=dp_type,
    )


def molecular_coordinates():
    atoms = jnp.array([[-0.7, 0.1, -0.2], [0.8, -0.1, 0.3]])
    electrons = jnp.array([
        [-0.5, 0.4, 0.2], [0.6, -0.3, 0.8],
        [-0.9, -0.5, 0.4], [0.2, 0.7, -0.6],
    ])
    return atoms, electrons


def tiny_config(logdir):
    from vmcnet.train.default_config import get_default_config
    cfg = get_default_config()
    positions = (((0., 0., -0.7), (0., 0., 0.7)),
                 ((0., 0., -0.9), (0., 0., 0.9)))
    cfg.problem.ion_pos = positions
    cfg.problem.ion_charges = (1., 1.)
    cfg.problem.atoms_symbol = ("H", "H")
    cfg.problem.nspins = (1, 1)
    cfg.problem.single_nspins = ((1, 0), (0, 1))
    cfg.eval.ion_pos = positions
    cfg.eval.ion_charges = (1., 1.)
    cfg.eval.nspins = (1, 1)
    cfg.eval.single_nspins = ((1, 0), (0, 1))
    cfg.logdir = str(logdir)
    cfg.base_logdir = str(logdir)
    cfg.save_to_current_datetime_subfolder = False
    cfg.distribute = False
    cfg.pretrain.iterations = 0
    cfg.gq.ndet = 1
    cfg.gq.wfn_depth = 2
    cfg.gq.rescale_input = FEATURE_RESCALE
    cfg.gq.h1 = 4
    cfg.gq.h2 = 2
    cfg.gq.nh = 2
    cfg.gq.jastrow_mlp_nlayer = 1
    cfg.gq.jastrow_mlp_ndim = 4
    for run in (cfg.vmc, cfg.eval):
        run.nchains = 4
        run.nburn = 1
        run.nepochs = 3
        run.nsteps_per_param_update = 1
        run.nmoves_per_width_update = 2
    cfg.vmc.checkpoint_every = 1
    cfg.vmc.best_checkpoint_every = 1
    cfg.vmc.acc_steps = 1
    return cfg
