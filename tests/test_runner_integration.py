"""Small actual molecular runs; only config injection and output paths are controlled."""

import json
import logging
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vmcnet.train import runners
from vmcnet.train.default_config import get_default_reload_config
from vmcnet.utils import io
from .helpers import assert_tree_close, assert_tree_finite, tiny_config


def call_runner(monkeypatch, cfg, reload):
    # Config parsing has its own tests; all model/MCMC/energy/SPRING/I/O code is real.
    monkeypatch.setattr(runners.train.parse_config_flags, "parse_flags", lambda fv: (reload, cfg))
    previous_level = logging.getLogger().level
    try:
        runners.run_molecule()
    finally:
        logging.getLogger().setLevel(previous_level)


def check_run(logdir, epochs, walkers, nchains):
    assert (logdir/"config.json").exists()
    assert (logdir/"best_checkpoint.npz").exists()
    for name in ("energy", "variance", "accept_ratio"):
        values = np.loadtxt(logdir/f"{name}.txt", ndmin=2)
        assert len(values) == epochs
        assert np.isfinite(values).all()
    multi = np.loadtxt(logdir/"multi_energy.txt", ndmin=2)
    assert multi.shape == (epochs, walkers)
    local = np.loadtxt(logdir/"eval"/"multi_energy.txt", ndmin=2)
    assert local.shape == (3, walkers*nchains)
    assert np.isfinite(local).all()
    statistics = json.loads((logdir/"eval"/"statistics.json").read_text())
    assert len(statistics["average"]) == walkers
    assert np.isfinite(statistics["average"]).all()


@pytest.mark.slow
def test_real_h2_training_evaluation_and_output(monkeypatch, tmp_path):
    cfg = tiny_config(tmp_path/"molecule")
    call_runner(monkeypatch, cfg, get_default_reload_config())
    check_run(tmp_path/"molecule", 3, len(cfg.problem.ion_pos), cfg.eval.nchains)


@pytest.mark.slow
def test_real_runner_checkpoint_replay(monkeypatch, tmp_path):
    full = tiny_config(tmp_path/"full")
    call_runner(monkeypatch, full, get_default_reload_config())
    partial = tiny_config(tmp_path/"part")
    partial.vmc.nepochs = 2
    call_runner(monkeypatch, partial, get_default_reload_config())
    resumed = tiny_config(tmp_path/"resumed")
    reload = get_default_reload_config()
    reload.logdir = str(tmp_path/"part")
    reload.checkpoint_relative_file_path = "checkpoints/2.npz"
    reload.end_epochs = 3
    reload.append = True
    call_runner(monkeypatch, resumed, reload)
    for name in ("energy", "multi_energy", "variance"):
        np.testing.assert_allclose(np.loadtxt(tmp_path/"resumed"/f"{name}.txt"),
                                   np.loadtxt(tmp_path/"full"/f"{name}.txt"), rtol=1e-9, atol=1e-9)
    full_state = io.reload_vmc_state(str(tmp_path/"full"/"checkpoints"), "3.npz")
    resumed_state = io.reload_vmc_state(str(tmp_path/"resumed"/"checkpoints"), "3.npz")
    assert_tree_close(resumed_state, full_state, rtol=1e-9, atol=1e-9)


def test_runner_model_wrapper_matches_single_configuration():
    cfg = tiny_config("unused")
    result = runners._get_gaoqiao_model(cfg.gq, 2, jnp.ones(2), (1, 1),
                                        jax.random.PRNGKey(5), False)
    batch_fn, one_axis, single, det, batch_det, orbitals, params, key = result
    atoms = jnp.asarray(cfg.problem.ion_pos)
    electrons = jax.random.normal(jax.random.PRNGKey(6), (2, 3, 2, 3))
    actual = batch_fn(params, atoms, electrons)
    expected = jnp.stack([jnp.stack([single(params, atoms[w], e) for e in electrons[w]])
                          for w in range(2)])
    np.testing.assert_allclose(actual, expected, rtol=1e-11)
    assert orbitals(params, atoms, electrons).shape == (2, 3, 1, 2, 2)
    assert_tree_finite(actual)
