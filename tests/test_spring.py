"""SPRING against independent dense references, including wrapper differentiation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmcnet.updates.spring import Spring, apply_updates, constrain_norm, spring_wrapper
from vmcnet.physics.core import create_energy_and_statistics_fn
from vmcnet.mcmc.position_amplitude_core import get_update_data_fn
from vmcnet.utils import distribute
from .helpers import assert_tree_close, gaussian_data, log_gaussian, batch_log_gaussian


def optimizer(mu=0., damping=0.15, repeat=False):
    return Spring(mu, 100., lambda step: 0.03, lambda step: damping,
                  repeat_single_mol=repeat)


def reference_direction(scores, energies, means, previous, mu, damping):
    # Solve in parameter space, independently of the production sample-space solve.
    s = np.asarray(scores).reshape(-1, scores.shape[-1])
    o = (s-s.mean(0)) / np.sqrt(len(s))
    e = (np.asarray(energies)-np.asarray(means)).reshape(-1) / np.sqrt(len(s))
    momentum = mu*np.asarray(previous)
    return np.linalg.solve(o.T@o+damping*np.eye(o.shape[1]),
                           o.T@(e-o@momentum)) + momentum


@pytest.mark.parametrize("mu", [0., 0.8])
@pytest.mark.parametrize("repeat", [False, True])
def test_spring_matches_parameter_space_solve(mu, repeat):
    scores = jnp.array([[[1., 2., 0.], [2., -1., 1.], [0., 1., -1.]],
                        [[4., 0., 1.], [1., 2., 3.], [-2., 1., 2.]]])
    energies = jnp.array([[1., 2., 6.], [10., 14., 12.]])
    means = jnp.mean(energies, axis=1, keepdims=True)
    previous = jnp.array([0.2, -0.4, 0.7])
    opt = optimizer(mu=mu, repeat=repeat)
    state = {"prev_grad": {"w": previous}, "step": jnp.asarray(2)}
    grad, _, mean = jax.jit(opt.get_grad)(scores, energies, means, state)
    reference_means = means.mean() if repeat else means
    expected = reference_direction(scores, energies, reference_means, previous, mu, 0.15)
    np.testing.assert_allclose(grad["w"], expected, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(mean, np.asarray(energies).mean())
    # A global constant in the score is unobservable after centering.
    shifted, _, _ = opt.get_grad(scores + jnp.array([5., -7., 2.]), energies, means, state)
    assert_tree_close(shifted, grad)


@pytest.mark.parametrize("scale", [0., 0.01, 10.])
def test_norm_constraint_and_zero_gradient(scale):
    grad = {"w": jnp.array([3., 4.])*scale}
    result = constrain_norm(grad, 0.04)
    expected_scale = min(1., 0.2 / (5*scale)) if scale else 1.
    np.testing.assert_allclose(result["w"], grad["w"]*expected_scale, atol=1e-12)
    assert np.linalg.norm(result["w"]) <= 0.2 + 1e-12


def test_update_uses_schedule_and_advances_state():
    opt = Spring(0.7, 100., lambda step: 0.1/(step+1), lambda step: 0.2)
    params = {"w": jnp.array([1., 2.], dtype=jnp.float32)}
    state = opt.init(params)
    grad = {"w": jnp.array([2., -1.])}
    for step in range(3):
        updates, state = opt.update(grad, state)
        np.testing.assert_allclose(updates["w"], -0.1/(step+1)*grad["w"])
        assert int(state["step"]) == step+1
        assert_tree_close(state["prev_grad"], grad)
    assert apply_updates(params, updates)["w"].dtype == jnp.float32


def test_wrapper_score_gradients_and_data_cache():
    params, data, key = gaussian_data()
    def energy(p, atoms, x):
        return jnp.sum((x-atoms[0])**2) + 2*atoms[0, 0]
    stats_fn = create_energy_and_statistics_fn(energy)
    opt = optimizer()
    wrapped = spring_wrapper(opt, log_gaussian, get_update_data_fn(batch_log_gaussian), stats_fn)
    state = wrapped.init(params)
    grad_acc = jax.tree_util.tree_map(jnp.zeros_like, params)
    metrics = {name: jnp.asarray(0.) for name in (
        "energy", "variance", "variance_noclip", "multi_variance", "energy_noclip",
        "multi_energy", "multi_energy_noclip")}
    actual, metrics, state, _, key = wrapped.grad_and_E(key, params, state, data, grad_acc, metrics)
    xp = data["atoms_position"]
    xe = data["walker_data"]["elec_position"]
    radii = np.sum((np.asarray(xe)-np.asarray(xp)[:, None, :1])**2, axis=(-1, -2))
    local = radii+2*np.asarray(xp)[:, 0, 0, None]
    expected = reference_direction(-radii[..., None], local, local.mean(1, keepdims=True),
                                   np.zeros(1), 0., 0.15)
    np.testing.assert_allclose(actual["alpha"], expected[0], rtol=1e-11)
    updated, new_data, new_state, _ = wrapped.update(key, params, state, data, actual)
    assert int(new_state["step"]) == 1
    np.testing.assert_allclose(new_data["walker_data"]["amplitude"],
                               batch_log_gaussian(updated, xp, xe))
    np.testing.assert_allclose(metrics["multi_energy"], local.mean(1))


def test_pmapped_spring_averages_local_residuals():
    n = jax.local_device_count()
    scores = jnp.arange(n*2*3*2, dtype=float).reshape(n, 2, 3, 2)/11
    scores = scores + jnp.sin(scores*3)
    energies = jnp.arange(n*6, dtype=float).reshape(n, 2, 3)**2/7
    means = jnp.mean(energies, axis=2, keepdims=True)
    opt = optimizer()
    state = opt.init({"w": jnp.zeros(2)})
    expected = np.mean([reference_direction(scores[d], energies[d], means[d], np.zeros(2), 0., 0.15)
                        for d in range(n)], axis=0)
    grads, _, mean = distribute.pmap(opt.get_grad)(
        scores, energies, means, distribute.replicate_all_local_devices(state))
    np.testing.assert_allclose(grads["w"], np.broadcast_to(expected, (n, 2)), rtol=1e-10, atol=1e-11)
    np.testing.assert_allclose(mean, np.asarray(energies).mean())
