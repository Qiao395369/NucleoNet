"""Current network contracts, signed antisymmetry, and determinant arithmetic."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vmcnet.gaoqiao import network_blocks, networks, open_feature_layer, dp, jastrows
from .helpers import FEATURE_RESCALE, assert_tree_finite, small_model, molecular_coordinates


@pytest.mark.parametrize("gq_type", ["ef_shrd_sym", "ef_shrd_sym_ablation"])
def test_wavefunction_translation_permutations_and_vmap(gq_type):
    params, signed, det_fn, orbitals, hz = small_model(gq_type)
    atoms, electrons = molecular_coordinates()
    sign, logabs = signed(params, electrons, atoms)
    assert np.shape(sign) == np.shape(logabs) == ()
    assert np.isfinite(logabs)
    assert abs(sign) == pytest.approx(1.)
    shift = jnp.array([1.3, -0.7, 2.1])
    for moved_e, moved_a, parity in [
        (electrons+shift, atoms+shift, 1),
        (electrons, atoms[::-1], 1),  # Both nuclei have Z=2.
        (electrons[jnp.array([1, 0, 2, 3])], atoms, -1),
        (electrons[jnp.array([0, 1, 3, 2])], atoms, -1),
        (electrons[jnp.array([1, 0, 3, 2])], atoms, 1),
    ]:
        actual_sign, actual_log = signed(params, moved_e, moved_a)
        np.testing.assert_allclose(actual_log, logabs, atol=2e-9, rtol=1e-9)
        np.testing.assert_allclose(actual_sign, parity*sign, atol=1e-12)
    batches = jnp.stack([electrons, electrons+0.2])
    batch_sign, batch_log = jax.vmap(signed, in_axes=(None, 0, None))(params, batches, atoms)
    for i in range(2):
        expected = signed(params, batches[i], atoms)
        np.testing.assert_allclose(batch_sign[i], expected[0])
        np.testing.assert_allclose(batch_log[i], expected[1])
    matrices = orbitals(params, electrons, atoms)
    assert len(matrices) == 1
    assert matrices[0].shape == (2, 4, 4)
    # Direct determinants are safe for this deliberately small reference problem.
    reference = np.linalg.det(np.asarray(matrices[0])).sum()
    np.testing.assert_allclose(sign*jnp.exp(logabs), reference, rtol=1e-9, atol=1e-18)


@pytest.mark.parametrize("gq_type,envelope", [
    ("ef_shrd_sym", name) for name in ("ds_hz", "ds_hz_pos", "iso", "ds_hz_deg")
] + [("ef_shrd_sym_ablation", name) for name in ("iso", "ds_hz_deg")])
def test_supported_envelopes_build_and_evaluate(gq_type, envelope):
    params, signed, *_ = small_model(gq_type, envelope_type=envelope)
    atoms, electrons = molecular_coordinates()
    assert_tree_finite(params)
    sign, logabs = signed(params, electrons, atoms)
    assert abs(sign) == pytest.approx(1.)
    assert np.isfinite(logabs)


@pytest.mark.parametrize("argument,values", [
    ("jastrow_type", ["mlp_res", "simple_ee", "null"]),
    ("activation_type", ["tanh", "relu", "silu"]),
    ("dp_type", ["original", "charge", "full", "element", "element1"]),
])
def test_supported_model_options(argument, values):
    atoms, electrons = molecular_coordinates()
    for value in values:
        params, signed, *_ = small_model(**{argument: value})
        sign, logabs = signed(params, electrons, atoms)
        assert abs(sign) == pytest.approx(1.)
        assert np.isfinite(logabs), (argument, value)


def test_complex_network_phase_and_odd_permutation():
    params, signed, *_ = small_model(do_complex=True)
    atoms, electrons = molecular_coordinates()
    phase, logabs = signed(params, electrons, atoms)
    other_phase, other_log = signed(params, electrons[jnp.array([1, 0, 2, 3])], atoms)
    np.testing.assert_allclose(abs(phase), 1., atol=1e-12)
    np.testing.assert_allclose(other_phase, -phase, atol=1e-9)
    np.testing.assert_allclose(other_log, logabs, atol=1e-9)


@pytest.mark.parametrize("complex_output", [False, True])
def test_logdet_weighted_sum_matches_direct_determinants(complex_output):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(3, 3, 3))
    if complex_output:
        a = a+1j*rng.normal(size=a.shape)
    weights = np.exp(np.array([-0.4, 0.2, 0.6]))
    phase, logabs, _ = network_blocks.logdet_matmul([jnp.asarray(a)], jnp.asarray(weights),
                                                   do_complex=complex_output)
    reference = np.dot(weights, np.linalg.det(a))
    np.testing.assert_allclose(phase*jnp.exp(logabs), reference, rtol=1e-12, atol=1e-12)


def test_logdet_stays_finite_when_direct_determinants_overflow():
    a = jnp.stack([jnp.diag(jnp.array([1e150, 2e150, 3e150])),
                   jnp.diag(jnp.array([2e150, 3e150, 4e150]))])
    sign, logabs, _ = network_blocks.logdet_matmul([a])
    assert sign == 1
    np.testing.assert_allclose(logabs, 450*np.log(10)+np.log(30.), atol=1e-10)


def test_logdet_handles_spin_products_and_one_by_one_blocks():
    up = jnp.array([[[2.]], [[-3.]]])
    down = jnp.array([[[1., 0.], [0., 4.]], [[1., 2.], [0., 2.]]])
    sign, logabs, _ = network_blocks.logdet_matmul([up, down])
    np.testing.assert_allclose(sign*jnp.exp(logabs), 2., atol=1e-12)
    zero_sign, zero_log = network_blocks.slogdet(jnp.zeros((1, 1)))
    assert zero_sign == 0
    assert np.isneginf(zero_log)


def test_open_features_shape_permutation_and_translation():
    atoms, electrons = molecular_coordinates()
    coords = jnp.concatenate([electrons, atoms])
    feature = open_feature_layer.make_open_features_ef(4, scale=[0.5, 2.], rescale=FEATURE_RESCALE)
    dimensions, params = feature.init()
    def apply(x):
        pp, distances = networks.construct_input_features_ef(x)
        return feature.apply(pp, distances, **params)
    actual = apply(coords)
    assert actual.shape == (6, 6, dimensions[1])
    assert_tree_finite(actual)
    np.testing.assert_allclose(apply(coords+jnp.array([2., 3., -1.])), actual, atol=1e-12)
    permutation = jnp.array([1, 0, 2, 3, 5, 4])
    np.testing.assert_allclose(apply(coords[permutation]), actual[permutation][:, permutation], atol=1e-12)


@pytest.mark.parametrize("kind,part_dim,pair_dim", [
    ("original", 3, 6), ("charge", 4, 9), ("full", 8, 13),
    ("element", 4, 8), ("element1", 4, 8),
])
def test_particle_descriptors_have_physical_charges_and_spin(kind, part_dim, pair_dim):
    system = dp.ManyElectronSystem(jnp.array([6., 1., 1.]), (2, 1), kind)
    assert system.get_part_features().shape == (6, part_dim)
    assert system.get_pair_features().shape == (6, 6, pair_dim)
    np.testing.assert_array_equal(system.get_part_charges(), [-1., -1., -1., 6., 1., 1.])
    np.testing.assert_array_equal(system.get_spin_marker(), [1., 1., -1., 0., 0., 0.])
    np.testing.assert_allclose(system.get_part_charges_norm(), np.array([-1., -1., -1., 6., 1., 1.])/6)
    if kind in ("element", "element1"):
        expected = [6, 1] if kind == "element" else [1, 6]
        np.testing.assert_array_equal(system.element_charges, expected)
        np.testing.assert_array_equal(system.get_part_features()[4], system.get_part_features()[5])


@pytest.mark.parametrize("kind", ["simple_ee", "mlp_res"])
def test_jastrow_is_positive_permutation_invariant_and_differentiable(kind):
    _, x = molecular_coordinates()
    # Finite diagonal placeholder: diagonal entries are excluded from ee cusps.
    distances = jnp.linalg.norm(x[:, None]-x[None, :]+jnp.eye(4)[..., None], axis=-1)
    hidden = jnp.arange(32.).reshape(4, 8)/30
    model = (jastrows.make_simple_ee_jastrow((2, 2)) if kind == "simple_ee" else
             jastrows.make_mlp_jastrow((2, 2), 2, 8, jax.nn.tanh))
    p = model.init(jax.random.PRNGKey(3), 8)
    original = model.apply(p, distances, hidden)
    perm = jnp.array([1, 0, 3, 2])
    permuted = model.apply(p, distances[perm][:, perm], hidden[perm])
    assert original > 0
    np.testing.assert_allclose(permuted, original, atol=1e-12)
    assert_tree_finite(jax.grad(lambda params: jnp.log(model.apply(params, distances, hidden)))(p))


@pytest.mark.slow
@pytest.mark.parametrize("gq_type", ["ef_shrd_sym", "ef_shrd_sym_ablation"])
def test_real_network_kinetic_implementations_agree(gq_type, record_property):
    from vmcnet.physics import kinetic
    params, signed, *_ = small_model(gq_type)
    atoms, x = molecular_coordinates()
    def logpsi(p, a, e):
        return signed(p, e, a)[1]
    assert_tree_finite(jax.grad(logpsi)(params, atoms, x))
    results = {
        "hessian": kinetic.create_laplacian_kinetic_energy_db(logpsi)(params, atoms, x),
        "old": kinetic.create_laplacian_kinetic_energy_old(logpsi)(params, atoms, x),
    }
    for inner_size in (0, 3, 6):
        results[f"fwdlap_{inner_size}"] = kinetic.create_laplacian_kinetic_energy_new(
            logpsi, inner_size)(params, atoms, x)

    # User policy: ignore coordinate-derivative NaNs under "all". Do not treat
    # two NaNs as evidence of agreement; compare only the finite results.
    finite = {}
    ignored = []
    for name, value in results.items():
        assert np.shape(value) == (), name
        assert not np.isinf(value), name
        if np.isnan(value):
            ignored.append(name)
        else:
            finite[name] = float(value)
    if len(finite) >= 2:
        np.testing.assert_allclose(list(finite.values()), next(iter(finite.values())),
                                   rtol=1e-8, atol=1e-8)
    record_property("feature_rescale", FEATURE_RESCALE)
    record_property("ignored_coordinate_derivative_nan", ",".join(ignored))
    record_property("finite_kinetic_comparison_count", len(finite))
