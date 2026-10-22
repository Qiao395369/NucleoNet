"""HF pretraining matrix/loss references and one real optimizer update on a tiny HF ansatz."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from vmcnet.train import pretrain_hf as pretrain
from .helpers import assert_tree_close, gaussian_data


def test_full_slater_matrix_and_split_blocks():
    up = jnp.array([[[2., 1.], [0., 3.]]])
    down = jnp.array([[[4.]]])
    full = pretrain.create_full_det((up, down))
    np.testing.assert_array_equal(full, [[[2., 1., 0.], [0., 3., 0.], [0., 0., 4.]]])
    a, b, off_a, off_b = pretrain.split_full_det_blocks(full, (2, 1))
    assert_tree_close((a, b), (up, down))
    np.testing.assert_array_equal(off_a, np.zeros((1, 2, 1)))
    np.testing.assert_array_equal(off_b, np.zeros((1, 1, 2)))


def test_occupied_subspace_loss_is_basis_invariant_but_detects_other_subspace():
    target = jnp.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])
    basis_change = jnp.array([[2., 1.], [0., 3.]])
    np.testing.assert_allclose(pretrain.occupied_subspace_loss(target, target@basis_change), 0., atol=1e-12)
    other = jnp.array([[1., 0.], [0., 0.], [0., 1.], [0., 0.]])
    np.testing.assert_allclose(pretrain.occupied_subspace_loss(target, other), 1., atol=1e-12)


@pytest.mark.parametrize("mode", ["l1", "l2", "fulldet", "blockdiag"])
def test_pretraining_loss_modes_and_determinant_axis(mode):
    target = jnp.broadcast_to(jnp.eye(2), (2, 3, 2, 2))
    orbitals = jnp.stack([target, target], axis=2)
    value = pretrain.compute_pretrain_loss(target, orbitals, (1, 1), loss_mode=mode)
    np.testing.assert_allclose(value, 0., atol=1e-12)
    if mode in ("l1", "l2"):
        shifted = orbitals+0.2
        expected = 0.2 if mode == "l1" else 0.04
        np.testing.assert_allclose(pretrain.compute_pretrain_loss(target, shifted, (1, 1), loss_mode=mode), expected)


def test_blockdiag_loss_penalizes_off_spin_blocks():
    target = jnp.broadcast_to(jnp.eye(2), (2, 3, 2, 2))
    mixed = target.at[..., 0, 1].set(0.5)
    actual = pretrain.compute_pretrain_loss(target, mixed, (1, 1), loss_mode="blockdiag", offblock_lambda=2.)
    np.testing.assert_allclose(actual, 0.5, atol=1e-12)
    with pytest.raises(ValueError, match="Unknown loss_mode"):
        pretrain.compute_pretrain_loss(target, mixed, (1, 1), loss_mode="wrong")


class TinyHF:
    """Analytic Slater blocks; avoids replacing production pretraining code or running SCF."""
    def jax_scf_(self, flat_positions):
        x = flat_positions.reshape(2, 3)
        return jnp.exp(-jnp.sum(x[0]**2))[None, None], jnp.exp(-jnp.sum(x[1]**2))[None, None]


def test_pretrain_step_updates_parameters_and_reduces_loss():
    _, data, _ = gaussian_data(ne=2)
    params = {"scale": jnp.asarray(0.4)}
    def orbitals(p, atoms, electrons):
        diagonal = jnp.exp(-jnp.sum(electrons**2, axis=-1))*p["scale"]
        full = jax.vmap(jax.vmap(jnp.diag))(diagonal)
        return full[:, :, None, :, :]
    opt = optax.sgd(0.1)
    state = opt.init(params)
    step = pretrain.make_pretrain_step(orbitals, opt.update, [TinyHF(), TinyHF()],
                                       lambda d, p: d, (1, 1), loss_mode="l2")
    _, updated, state, loss, grad_norm, grad_max, param_max, update_max = step(data, params, state)
    _, _, _, next_loss, *_ = step(data, updated, state)
    assert float(loss) > 0
    assert float(next_loss) < float(loss)
    assert float(updated["scale"]) > float(params["scale"])
    assert np.isfinite([loss, grad_norm, grad_max, param_max, update_max]).all()
