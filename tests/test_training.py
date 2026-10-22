"""Two-stage VMC scheduling, accumulation, logging, and exact checkpoint replay."""

import logging
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vmcnet.train.vmc import vmc_loop
from vmcnet.utils import io
from vmcnet.mcmc.position_amplitude_core import get_update_data_fn
from .helpers import assert_tree_close, gaussian_data, batch_log_gaussian


def run_toy(logdir, *, nepochs=4, acc_steps=2, initial=None, start_epoch=0):
    if initial is None:
        params, data, key = gaussian_data()
        state = {"prev_grad": {"alpha": jnp.asarray(0.)}, "step": jnp.asarray(0)}
    else:
        params, state, data, key = initial
    calls = {"walk": 0, "gradient": 0, "update": 0}
    refresh = get_update_data_fn(batch_log_gaussian)

    def walk(p, d, k):
        calls["walk"] += 1
        d = {**d, "walker_data": {**d["walker_data"],
            "elec_position": d["walker_data"]["elec_position"]+1}}
        d = refresh(d, p)
        return jnp.full((2,), 0.5), d, jax.random.split(k)[0]

    def gradients(k, p, s, d, grad_acc, metrics_acc):
        calls["gradient"] += 1
        value = jnp.mean(d["walker_data"]["elec_position"])
        grad_acc = {"alpha": grad_acc["alpha"]+value}
        metrics = {name: value for name in metrics_acc}
        metrics["multi_energy"] = jnp.array([value, value+10.])
        metrics["multi_energy_noclip"] = metrics["multi_energy"]
        metrics["multi_variance"] = jnp.array([1., 2.])
        metrics_acc = jax.tree_util.tree_map(lambda a, b: a+b, metrics_acc, metrics)
        return grad_acc, metrics_acc, s, d, k

    def update(k, p, s, d, grad):
        calls["update"] += 1
        p = {"alpha": p["alpha"]-0.01*grad["alpha"]}
        s = {"prev_grad": grad, "step": s["step"]+1}
        return p, refresh(d, p), s, k

    result = vmc_loop(params, state, data, 6, nepochs, walk, gradients, update, key,
        logdir=str(logdir), checkpoint_every=1, best_checkpoint_every=2,
        is_pmapped=False, start_epoch=start_epoch, acc_steps=acc_steps)
    return result, calls


@pytest.mark.parametrize("acc_steps", [1, 3])
def test_accumulation_updates_counts_average_and_logs(tmp_path, caplog, acc_steps):
    params, data, _ = gaussian_data()
    mean0 = float(jnp.mean(data["walker_data"]["elec_position"]))
    with caplog.at_level(logging.INFO):
        result, calls = run_toy(tmp_path, nepochs=4, acc_steps=acc_steps)
    final, state, new_data, _, nans = result
    assert calls == {"walk": 4*acc_steps, "gradient": 4*acc_steps, "update": 4}
    epoch_gradients = mean0 + np.arange(4)*acc_steps + (acc_steps+1)/2
    np.testing.assert_allclose(final["alpha"], params["alpha"]-0.01*epoch_gradients.sum())
    np.testing.assert_allclose(np.loadtxt(tmp_path/"energy.txt"), epoch_gradients)
    np.testing.assert_allclose(np.loadtxt(tmp_path/"multi_energy.txt"),
                               np.stack([epoch_gradients, epoch_gradients+10.], axis=1))
    np.testing.assert_allclose(new_data["walker_data"]["elec_position"],
                               data["walker_data"]["elec_position"]+4*acc_steps)
    assert int(state["step"]) == 4
    assert not nans
    assert any("E:" in record.getMessage() for record in caplog.records)
    assert (tmp_path/"best_checkpoint.npz").exists()


def test_checkpoint_replay_reproduces_continuous_training(tmp_path):
    full, _ = run_toy(tmp_path/"full")
    run_toy(tmp_path/"part", nepochs=2)
    epoch, data, params, state, key = io.reload_vmc_state(str(tmp_path/"part"/"checkpoints"), "2.npz")
    # Current checkpoints contain the PRE-update state at zero-based epoch.
    assert epoch == 1
    resumed, calls = run_toy(tmp_path/"resumed", start_epoch=epoch,
                              initial=(params, state, data, key))
    assert calls["update"] == 3
    assert_tree_close(resumed, full)
    np.testing.assert_allclose(np.loadtxt(tmp_path/"resumed"/"energy.txt"),
                               np.loadtxt(tmp_path/"full"/"energy.txt")[epoch:])
