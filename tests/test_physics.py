"""Analytic references for molecular Coulomb energies and kinetic operators."""

import functools
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmcnet.physics import core, kinetic, potential
from vmcnet.train.runners import total_variation_clipping_fn
from .helpers import log_gaussian


def coulomb_reference(charges, atoms, electrons):
    en = -sum(z / np.linalg.norm(e - a)
              for e in electrons for z, a in zip(charges, atoms))
    ee = sum(1 / np.linalg.norm(electrons[i] - electrons[j])
             for i in range(len(electrons)) for j in range(i))
    nn = sum(charges[i] * charges[j] / np.linalg.norm(atoms[i] - atoms[j])
             for i in range(len(atoms)) for j in range(i))
    return np.array([en, ee, nn])


@pytest.mark.parametrize("distance", [1.2, 2.3])
def test_coulomb_values_and_geometry_dependence(distance):
    charges = jnp.array([1., 2.])
    atoms = jnp.array([[0., 0., 0.], [distance, 0.2, -0.1]])
    electrons = jnp.array([[0.3, 0.4, 0.5], [-0.4, -0.2, 0.6]])
    terms = [potential.create_electron_ion_coulomb_potential(charges),
             potential.create_electron_electron_coulomb_potential(),
             potential.create_ion_ion_coulomb_potential(charges)]
    actual = jnp.array([term(None, atoms, electrons) for term in terms])
    expected = coulomb_reference(charges, np.asarray(atoms), np.asarray(electrons))
    np.testing.assert_allclose(actual, expected, rtol=1e-12)
    combined = core.combine_local_energy_terms(terms)
    np.testing.assert_allclose(combined(None, atoms, electrons), expected.sum())
    rotation = jnp.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    shift = jnp.array([2., -1., 3.])
    np.testing.assert_allclose(
        combined(None, atoms @ rotation + shift, electrons @ rotation + shift),
        expected.sum(), rtol=1e-12,
    )


def test_coulomb_batch_and_softening():
    atoms = jnp.array([[0., 0., 0.]])
    electrons = jnp.array([[[1., 0., 0.]], [[0., 2., 0.]]])
    fn = potential.create_electron_ion_coulomb_potential(jnp.array([2.]), softening_term=0.5)
    np.testing.assert_allclose(fn(None, atoms, electrons), -2 / np.sqrt([1.25, 4.25]))
    ee = potential.create_electron_electron_coulomb_potential()
    np.testing.assert_array_equal(ee(None, atoms, electrons), [0., 0.])


@pytest.mark.parametrize("method,inner_size", [("old", 0), ("db", 0)] +
                         [("new", n) for n in (0, 1, 2, 3, 6)])
def test_gaussian_kinetic_matches_analytic(method, inner_size):
    params = {"alpha": jnp.asarray(0.7)}
    atoms = jnp.array([[0.1, -0.2, 0.3]])
    electrons = jnp.array([[0.4, 0.5, -0.7], [-0.2, 0.8, 0.9]])
    factory = getattr(kinetic, f"create_laplacian_kinetic_energy_{method}")
    fn = factory(log_gaussian, inner_size) if method == "new" else factory(log_gaussian)
    expected = 0.7 * electrons.size - 2 * 0.7**2 * jnp.sum((electrons - atoms[0])**2)
    np.testing.assert_allclose(jax.jit(fn)(params, atoms, electrons), expected, atol=1e-11)


def test_laplacian_ratio_polynomial_reference():
    def log_f(params, atoms, x):
        return jnp.log(jnp.sum(x*x + 3*x))
    x = jnp.array([[1., 2.], [3., 4.], [5., 6.]])
    actual = kinetic.laplacian_psi_over_psi(jax.grad(log_f, argnums=2), None, None, x)
    np.testing.assert_allclose(actual, 2 * x.size / jnp.sum(x*x + 3*x), rtol=1e-12)


@pytest.mark.parametrize("method", ["old", "new", "db"])
def test_hydrogen_local_energy_is_minus_half(method):
    def log_hydrogen(params, atoms, x):
        return -jnp.linalg.norm(x[0] - atoms[0])
    atoms = jnp.zeros((1, 3))
    x = jnp.array([[[0.4, 0.3, 0.2]], [[1.2, -0.4, 0.6]], [[-0.3, 1.4, 0.7]]])
    kin = getattr(kinetic, f"create_laplacian_kinetic_energy_{method}")(log_hydrogen)
    en = potential.create_electron_ion_coulomb_potential(jnp.ones(1))
    energy = core.combine_local_energy_terms([kin, en])
    np.testing.assert_allclose(jax.vmap(energy, in_axes=(None, None, 0))(None, atoms, x),
                               -0.5, atol=1e-11)


def test_harmonic_oscillator_local_energy_is_constant():
    omega = 1.3
    p = {"alpha": jnp.asarray(omega/2)}
    atoms = jnp.zeros((1, 3))
    x = jax.random.normal(jax.random.PRNGKey(6), (5, 2, 3))
    kin = kinetic.create_laplacian_kinetic_energy_new(log_gaussian)
    values = jax.vmap(kin, in_axes=(None, None, 0))(p, atoms, x)
    values += 0.5 * omega**2 * jnp.sum(x*x, axis=(-1, -2))
    np.testing.assert_allclose(values, 3*omega, atol=1e-11)


@pytest.mark.parametrize("center", ["mean", "median"])
def test_multigeometry_energy_statistics_and_clipping(center):
    # Different rows have very different offsets: centering must stay per geometry.
    energies = jnp.array([[1., 2., 3., 40.], [101., 102., 103., 140.]])
    clipping = functools.partial(total_variation_clipping_fn, threshold=1., clip_center=center)
    means, clipped, stats = core.get_clipped_energies_and_stats(energies, clipping, False)
    x = np.asarray(energies)
    midpoint = x.mean(1, keepdims=True) if center == "mean" else np.median(x, axis=1, keepdims=True)
    radius = np.abs(x-midpoint).mean(1, keepdims=True)
    expected = np.clip(x, midpoint-radius, midpoint+radius)
    np.testing.assert_allclose(clipped, expected)
    np.testing.assert_allclose(means, expected.mean(1, keepdims=True))
    np.testing.assert_allclose(stats["multi_energy"], expected.mean(1))
    np.testing.assert_allclose(stats["variance"], expected.var(1, ddof=1).mean())
    np.testing.assert_allclose(stats["variance_noclip"], x.var(1, ddof=1).mean())
    np.testing.assert_allclose(stats["multi_variance"], np.maximum(x.var(1, ddof=1), expected.var(1, ddof=1)))


def test_energy_factory_keeps_both_batch_axes():
    def local_energy(p, atoms, x):
        return p * jnp.sum(x) + jnp.sum(atoms)
    atoms = jnp.arange(6.).reshape(2, 1, 3)
    x = jnp.arange(36.).reshape(2, 3, 2, 3)
    means, values, stats = core.create_energy_and_statistics_fn(local_energy)(2., atoms, x)
    expected = 2*np.asarray(x).sum((-1, -2)) + np.asarray(atoms).sum((-1, -2))[:, None]
    np.testing.assert_allclose(values, expected)
    np.testing.assert_allclose(means, expected.mean(1, keepdims=True))
    assert stats["multi_energy"].shape == (2,)


def test_molecular_initialization_respects_geometry_and_spin_assignment():
    atoms = jnp.array([[[0., 0., 0.], [1., 0., 0.]],
                       [[0., 1., 0.], [2., 1., 0.]]])
    key = jax.random.PRNGKey(8)
    _, positions = core.initialize_molecular_pos(
        key, 3, atoms, jnp.array([1., 1.]), 2, jnp.array([[1, 0], [0, 1]]),
        init_width=0., dtype=jnp.float64,
    )
    np.testing.assert_allclose(positions, np.broadcast_to(np.asarray(atoms)[:, None], (2, 3, 2, 3)))
