"""Current W-by-B Metropolis interfaces and position/amplitude cache invariants."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmcnet.mcmc import metropolis as metro
from vmcnet.mcmc import position_amplitude_core as pacore
from vmcnet.mcmc import dynamic_width_position_amplitude as dwpa
from vmcnet.utils import distribute
from .helpers import assert_tree_close, gaussian_data, batch_log_gaussian, log_gaussian


@pytest.mark.parametrize("logabs", [True, False])
def test_acceptance_is_squared_wavefunction_ratio(logabs):
    old = jnp.array([1., -jnp.pi, 0., 1e-5])
    new = jnp.array([0., 1., 3., 4.])
    a, b = (jnp.log(jnp.abs(old)), jnp.log(jnp.abs(new))) if logabs else (old, new)
    actual = metro.metropolis_symmetric_acceptance(a, b, logabs=logabs)
    np.testing.assert_allclose(actual, [0., 1/np.pi**2, 1., 1.], rtol=1e-12)


def deterministic_step():
    def propose(params, data, key):
        return data + jnp.array([1., 2., 3., 4.]), key
    def accept(params, data, proposed):
        return jnp.array([1., 0., 1., 0.])
    def update(data, proposed, mask):
        return jnp.where(mask, proposed, data)
    return metro.make_metropolis_step(propose, accept, update)


def test_metropolis_step_has_geometry_axis_and_independent_keys():
    data = jnp.zeros((2, 4))
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    probability, result, next_keys = deterministic_step()(None, data, keys)
    np.testing.assert_allclose(probability, [0.5, 0.5])
    np.testing.assert_allclose(result, [[1., 0., 3., 0.]]*2)
    assert not np.array_equal(keys, next_keys)
    assert not np.array_equal(next_keys[0], next_keys[1])


def test_scan_walk_equals_repeated_steps():
    step = deterministic_step()
    data = jnp.zeros((2, 4))
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    probability, actual, actual_keys = metro.walk_data(5, None, data, keys, step, jnp.zeros(2))
    for _ in range(5):
        _, data, keys = step(None, data, keys)
    assert_tree_close((actual, actual_keys), (data, keys))
    np.testing.assert_allclose(probability, [0.5, 0.5])


@pytest.mark.parametrize("with_metadata_update", [False, True])
def test_mask_updates_position_and_amplitude_together(with_metadata_update):
    atoms = jnp.array([[0., 0., 0.]])
    x = jnp.arange(12.).reshape(4, 1, 3)
    data = pacore.make_position_amplitude_data(atoms, x, jnp.arange(4.), jnp.asarray(2))
    proposed = pacore.make_position_amplitude_data(atoms, x+100., jnp.arange(4.)+10., jnp.asarray(3))
    mask = jnp.array([True, False, False, True])
    meta_fn = (lambda old, m: old+jnp.sum(m)) if with_metadata_update else None
    result = pacore.make_position_amplitude_update(meta_fn)(data, proposed, mask)
    np.testing.assert_allclose(result["walker_data"]["elec_position"],
                               np.where(np.asarray(mask)[:, None, None], x+100, x))
    np.testing.assert_allclose(result["walker_data"]["amplitude"], [10., 1., 2., 13.])
    np.testing.assert_array_equal(result["atoms_position"], atoms)
    assert int(result["move_metadata"]) == (4 if with_metadata_update else 3)


def test_gaussian_proposal_recomputes_amplitudes_and_keeps_nuclei_fixed():
    params, batch, key = gaussian_data()
    data = jax.tree_util.tree_map(lambda x: x[0], batch)
    model = jax.vmap(log_gaussian, in_axes=(None, None, 0))
    proposal = pacore.make_position_amplitude_gaussian_proposal(model, lambda d: 0.3)
    proposed, next_key = proposal(params, data, key)
    assert not np.array_equal(data["walker_data"]["elec_position"], proposed["walker_data"]["elec_position"])
    np.testing.assert_array_equal(proposed["atoms_position"], data["atoms_position"])
    np.testing.assert_allclose(proposed["walker_data"]["amplitude"],
                               model(params, data["atoms_position"], proposed["walker_data"]["elec_position"]))
    assert not np.array_equal(key, next_key)


def test_parameter_change_refreshes_amplitude_cache():
    params, data, _ = gaussian_data()
    params = {"alpha": params["alpha"]*2}
    result = pacore.get_update_data_fn(batch_log_gaussian)(data, params)
    np.testing.assert_allclose(result["walker_data"]["amplitude"], 2*data["walker_data"]["amplitude"])
    assert_tree_close(result["move_metadata"], data["move_metadata"])


@pytest.mark.parametrize("acceptance,multiplier", [(0.5, 1.), (0.45, 1.), (0.55, 1.),
                                                    (0.9, 1.1), (0.1, 0.9)])
def test_width_thresholds(acceptance, multiplier):
    actual = dwpa.make_threshold_adjust_std_move(0.5, 0.1, 0.1)(0.3, acceptance)
    np.testing.assert_allclose(actual, 0.3*multiplier)


def test_metadata_updates_only_after_requested_moves():
    update = dwpa.make_update_move_metadata_fn(3, lambda width, mean: width*mean)
    metadata = dwpa.MoveMetadata(std_move=0.9, move_acceptance_sum=0., moves_since_update=0)
    for i, mask in enumerate([jnp.array([1., 0.]), jnp.array([1., 1.])]):
        metadata = update(metadata, mask)
        assert int(metadata["moves_since_update"]) == i+1
        np.testing.assert_allclose(metadata["std_move"], 0.9)
    metadata = update(metadata, jnp.array([0., 0.]))
    np.testing.assert_allclose(metadata["std_move"], 0.45)
    assert int(metadata["moves_since_update"]) == 0
    assert float(metadata["move_acceptance_sum"]) == 0.


def test_explicit_float_metadata_accumulates_boolean_acceptance():
    metadata = {
        "std_move": jnp.asarray(0.3),
        "move_acceptance_sum": jnp.asarray(0., dtype=jnp.float64),
        "moves_since_update": jnp.asarray(0),
    }
    update = dwpa.make_update_move_metadata_fn(2, lambda width, mean: width)
    result = jax.jit(update)(metadata, jnp.array([True, False]))
    np.testing.assert_allclose(result["move_acceptance_sum"], 0.5)
    assert int(result["moves_since_update"]) == 1


def test_jitted_walker_preserves_geometry_and_refreshes_cached_amplitude():
    from .helpers import log_gaussian
    params, data, key = gaussian_data()
    model = jax.vmap(log_gaussian, in_axes=(None, None, 0))
    step = dwpa.make_dynamic_pos_amp_gaussian_step(model, 2)
    walker = metro.make_jitted_walker_fn(3, step, apply_pmap=False)
    acceptance, result, next_key = walker(params, data, key)
    assert acceptance.shape == (2,)
    assert np.all((np.asarray(acceptance) >= 0) & (np.asarray(acceptance) <= 1))
    np.testing.assert_array_equal(result["atoms_position"], data["atoms_position"])
    np.testing.assert_allclose(result["walker_data"]["amplitude"],
                               batch_log_gaussian(params, result["atoms_position"], result["walker_data"]["elec_position"]))
    assert not np.array_equal(key, next_key)


def test_current_runner_distributes_geometries_not_particle_axes():
    from vmcnet.train.runners import _make_initial_data
    from vmcnet.train.default_config import get_default_config
    n = jax.local_device_count()
    params, data, _ = gaussian_data(w=2*n)
    p = distribute.replicate_all_local_devices(params)
    result = _make_initial_data(batch_log_gaussian, get_default_config().vmc,
        data["atoms_position"], data["walker_data"]["elec_position"], p,
        dtype=jnp.float64, apply_pmap=True)
    assert result["atoms_position"].shape == (n, 2, 1, 3)
    assert result["walker_data"]["elec_position"].shape == (n, 2, 6, 2, 3)
    np.testing.assert_allclose(result["walker_data"]["amplitude"],
                               np.asarray(data["walker_data"]["amplitude"]).reshape(n, 2, 6))


@pytest.mark.slow
def test_metropolis_samples_known_gaussian_distribution():
    # log|psi|=-r²/4 gives |psi|²=N(0,1), rather than N(0,2).
    # This detects the factor-of-two error when porting old log-probability APIs.
    params, data, key = gaussian_data(w=2, b=512, ne=1)
    params = {"alpha": jnp.asarray(0.25)}
    data = pacore.get_update_data_fn(batch_log_gaussian)(data, params)
    model = jax.vmap(log_gaussian, in_axes=(None, None, 0))
    step = pacore.make_position_amplitude_gaussian_metropolis_step(model, lambda d: 0.8)
    walk = metro.make_jitted_walker_fn(5, step, apply_pmap=False)

    def transition(carry, _):
        d, k = carry
        _, d, k = walk(params, d, k)
        displacement = d["walker_data"]["elec_position"]-d["atoms_position"][:, None, :1]
        return (d, k), displacement

    (data, key), _ = jax.lax.scan(transition, (data, key), xs=None, length=50)
    _, samples = jax.lax.scan(transition, (data, key), xs=None, length=40)
    values = np.asarray(samples).reshape(40, 2, 512, 3)
    np.testing.assert_allclose(values.mean((0, 2)), 0., atol=0.06)
    np.testing.assert_allclose(values.var((0, 2)), 1., atol=0.09)
