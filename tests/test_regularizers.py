import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pytest

import jax_gs.regularizers as regularizers
from jax_gs.regularizers import (
    compute_tv_loss_targeted,
    create_invisible_mask,
    dilate_mask,
)


def test_targeted_tv_constant_step_mask_and_gradient():
    constant = jnp.ones((1, 2, 3, 4))
    assert jnp.allclose(compute_tv_loss_targeted(constant), 0.0)
    step = constant.at[:, :, 1:, :].set(3.0)
    expected = (2.0 * 2 * 4) / step.size
    assert jnp.allclose(compute_tv_loss_targeted(step), expected)
    mask = jnp.zeros((1, 1, 3, 4))
    assert jnp.allclose(compute_tv_loss_targeted(step, mask), 0.0)
    gradient = jax.grad(compute_tv_loss_targeted)(step)
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_targeted_tv_validation_and_optional_binary_contract(monkeypatch):
    with pytest.raises(ValueError, match="4D"):
        compute_tv_loss_targeted(jnp.ones((3, 4)))
    monkeypatch.setattr(regularizers, "ENFORCE_CONTRACTS", True)
    with pytest.raises(ValueError, match="binary"):
        compute_tv_loss_targeted(
            jnp.ones((1, 1, 2, 2)),
            jnp.full((1, 1, 2, 2), 0.5),
        )


@pytest.mark.parametrize("ndim", [2, 3, 4])
def test_dilate_mask_grows_one_pixel_and_preserves_rank(ndim):
    shape = (5, 5) if ndim == 2 else ((1,) * (ndim - 2) + (5, 5))
    mask = jnp.zeros(shape).at[(0,) * (ndim - 2) + (2, 2)].set(1.0)
    dilated = dilate_mask(mask, 3)
    assert dilated.ndim == ndim
    assert int(jnp.sum(dilated)) == 9
    assert jnp.array_equal(dilate_mask(mask, 1), mask)


@pytest.mark.parametrize("kernel", [0, 2, 1.5])
def test_dilate_mask_rejects_invalid_kernel(kernel):
    with pytest.raises(ValueError, match="positive odd"):
        dilate_mask(jnp.zeros((3, 3)), kernel)


def test_create_invisible_mask_unions_arrays_and_inverts_png(tmp_path):
    first = jnp.asarray([[1.0, 0.0], [0.0, 0.0]])
    second = jnp.asarray([[0.0, 0.0], [0.0, 1.0]])
    assert jnp.array_equal(
        create_invisible_mask([first, second]),
        jnp.asarray([[1.0, 0.0], [0.0, 1.0]]),
    )
    path = tmp_path / "mask.png"
    Image.fromarray(np.asarray([[255, 0], [255, 255]], dtype=np.uint8)).save(path)
    assert jnp.array_equal(
        create_invisible_mask([str(path)]),
        jnp.asarray([[0.0, 1.0], [0.0, 0.0]]),
    )


def test_create_invisible_mask_rejects_bad_inputs():
    with pytest.raises(ValueError, match="at least one"):
        create_invisible_mask([])
    with pytest.raises(ValueError, match="shape"):
        create_invisible_mask([jnp.zeros((2, 2)), jnp.zeros((3, 2))])
    with pytest.raises(TypeError, match="JAX Array"):
        create_invisible_mask([object()])
