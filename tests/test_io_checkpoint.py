"""Real asynchronous writes and full current-state serialization round trips."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vmcnet.utils import io, checkpoint, distribute
from vmcnet.updates.spring import Spring
from .helpers import assert_tree_close, gaussian_data


@pytest.mark.parametrize("pmapped", [False, True])
def test_checkpoint_roundtrip_including_spring_state_and_rng(tmp_path, pmapped):
    params, data, key = gaussian_data(w=2*jax.local_device_count())
    opt = Spring(0.9, 1., lambda step: 0.1, lambda step: 0.1)
    state = opt.init(params)
    state["step"] = jnp.asarray(3)
    state["prev_grad"] = {"alpha": jnp.asarray(0.17)}
    if pmapped:
        # The runner shards geometries; each saved leaf retains its device axis.
        data = distribute.default_distribute_data(data)
        params = distribute.replicate_all_local_devices(params)
        state = distribute.replicate_all_local_devices(state)
        key = distribute.make_different_rng_key_on_all_devices(key)
    original = (5, data, params, state, key)
    with checkpoint.CheckpointWriter(is_pmapped=pmapped) as writer:
        writer.save_data(str(tmp_path), "state.npz", original)
    epoch, restored_data, restored_params, restored_state, restored_key = io.reload_vmc_state(str(tmp_path), "state.npz")
    if pmapped:
        restored_data, restored_params, restored_state, restored_key = distribute.distribute_vmc_state_from_checkpoint(
            restored_data, restored_params, restored_state, restored_key)
    assert_tree_close((epoch, restored_data, restored_params, restored_state, restored_key), original)


def test_metrics_writer_flushes_vector_rows_in_order(tmp_path):
    with checkpoint.MetricsWriter() as writer:
        for i in range(3):
            writer.save_data(str(tmp_path), "", {"energy": -float(i), "multi_energy": jnp.array([i, i+10])})
    np.testing.assert_allclose(np.loadtxt(tmp_path/"energy.txt"), [0., -1., -2.])
    np.testing.assert_allclose(np.loadtxt(tmp_path/"multi_energy.txt"), [[0, 10], [1, 11], [2, 12]])


def test_best_checkpoint_uses_best_observation_in_window(tmp_path):
    params, data, key = gaussian_data()
    best, pending = jnp.inf, None
    with checkpoint.CheckpointWriter(False) as writer:
        for epoch, energy in enumerate([3., 1., 5.]):
            # A singleton running window makes the best observation explicit.
            history = checkpoint.RunningEnergyVariance(checkpoint.RunningMetric(2), checkpoint.RunningMetric(2))
            _, candidate, pending = checkpoint.track_and_save_best_checkpoint(
                epoch, {"alpha": jnp.asarray(epoch, dtype=float)}, {"step": jnp.asarray(epoch)},
                data, key, {"energy": energy, "variance": 0.}, 6, history, writer, best,
                str(tmp_path), 0., "", 3, pending)
            best = jnp.minimum(best, candidate)
    saved = io.reload_vmc_state(str(tmp_path), checkpoint.BEST_CHECKPOINT_FILE_NAME)
    assert saved[0] == 1
    assert float(saved[2]["alpha"]) == 1.


def test_config_json_roundtrip_preserves_nested_tuples_and_unique_names(tmp_path):
    from ml_collections import ConfigDict
    original = ConfigDict({"positions": ((0., 1., 2.), (3., 4., 5.)), "nested": {"nspins": (2, 1)}})
    io.save_config_dict_to_json(original, str(tmp_path), "config")
    io.save_config_dict_to_json(original, str(tmp_path), "config")
    assert (tmp_path/"config.json").exists()
    assert (tmp_path/"config_1.json").exists()
    assert io.load_config_dict(str(tmp_path), "config.json").to_dict() == original.to_dict()


def test_copy_metrics_truncates_without_changing_source(tmp_path):
    src, dst = tmp_path/"source", tmp_path/"target"
    src.mkdir()
    dst.mkdir()
    text = "1\n2\n3\n4\n"
    (src/"energy.txt").write_text(text)
    io.copy_txt_stats(str(src), str(dst), truncate=2)
    assert (dst/"energy.txt").read_text() == "1\n2\n"
    assert (src/"energy.txt").read_text() == text


def test_checkpoint_metric_penalizes_uncertainty():
    np.testing.assert_allclose(checkpoint.get_checkpoint_metric(-2., 4., 100, 25.), -1.)
    np.testing.assert_allclose(checkpoint.get_checkpoint_metric(-2., 4., 100, 0.), -2.)
