import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from jax_gs.contrib import dynamic
from jax_gs.contrib.dynamic import DeformNetwork
from jax_gs.contrib.dynamic.deformation import DeformationTable


def _inputs(count: int, feature_dim: int):
    key = jax.random.key(42)
    keys = jax.random.split(key, 4)
    return (
        jax.random.normal(keys[0], (count, 3)),
        jax.random.normal(keys[1], (count, 4)),
        jax.random.normal(keys[2], (count, 1)),
        jnp.zeros((count, 1), dtype=jnp.float32),
        jax.random.normal(keys[3], (count, feature_dim)),
    )


def test_deform_network_rejects_invalid_construction():
    with pytest.raises(ValueError, match="num_layers"):
        DeformNetwork(feature_dim=8, num_layers=0)
    with pytest.raises(ValueError, match="feature_dim"):
        DeformNetwork(feature_dim=0)


def test_deform_network_zero_heads_make_initial_forward_identity():
    network = DeformNetwork(8, hidden_dim=16, rngs=nnx.Rngs(1))
    inputs = _inputs(5, 8)
    outputs = network(*inputs)
    for output, expected in zip(outputs, inputs[:3]):
        assert jnp.array_equal(output, expected)


def test_deform_network_forward_alias_and_nnx_jit_match():
    network = DeformNetwork(8, hidden_dim=4, num_layers=2, rngs=nnx.Rngs(2))
    inputs = _inputs(3, 8)
    eager = network.forward(*inputs)
    compiled = nnx.jit(lambda module, values: module(*values))(network, inputs)
    for eager_value, compiled_value in zip(eager, compiled):
        assert jnp.array_equal(eager_value, compiled_value)


def test_deform_network_gradients_reach_heads_and_perturbed_trunk():
    network = DeformNetwork(6, hidden_dim=5, num_layers=2, rngs=nnx.Rngs(3))
    for layer in network.trunk:
        layer.kernel[...] = jnp.abs(layer.kernel[...]) + 0.1
        layer.bias[...] = 0.1
    for head in (network.pos_head, network.quat_head, network.opacity_head):
        head.kernel[...] = 0.01

    inputs = _inputs(4, 6)

    def loss(module):
        return sum(jnp.sum(value) for value in module(*inputs))

    gradients = nnx.grad(loss)(network)
    assert jnp.any(gradients.trunk[0].kernel[...] != 0.0)
    assert jnp.any(gradients.trunk[1].kernel[...] != 0.0)
    assert jnp.any(gradients.pos_head.kernel[...] != 0.0)
    assert jnp.any(gradients.quat_head.kernel[...] != 0.0)
    assert jnp.any(gradients.opacity_head.kernel[...] != 0.0)


def test_deform_network_zero_heads_block_only_trunk_gradient_at_init():
    network = DeformNetwork(6, hidden_dim=5, num_layers=2, rngs=nnx.Rngs(4))
    inputs = _inputs(4, 6)

    def loss(module):
        return sum(jnp.sum(value) for value in module(*inputs))

    gradients = nnx.grad(loss)(network)
    for layer in gradients.trunk.values():
        assert jnp.array_equal(layer.kernel[...], jnp.zeros_like(layer.kernel[...]))
        assert jnp.array_equal(layer.bias[...], jnp.zeros_like(layer.bias[...]))
    assert jnp.any(gradients.pos_head.kernel[...] != 0.0)


def test_deform_network_validates_batch_feature_and_dtype_contracts():
    network = DeformNetwork(8)
    means, quats, opacities, time, features = _inputs(4, 8)
    with pytest.raises(ValueError, match="batch dim"):
        network(means, quats[:3], opacities, time, features)
    with pytest.raises(ValueError, match="last dim"):
        network(means, quats, opacities, time, features[:, :7])
    with pytest.raises(ValueError, match="dtype"):
        network(means, quats, opacities, time, features.astype(jnp.float16))


def test_deformation_table_starts_static_and_sets_indices():
    assert dynamic.DeformationTable is DeformationTable
    table = DeformationTable(5)
    assert len(table) == 5
    assert not bool(jnp.any(table.mask))
    table.set_indices(jnp.asarray([1, 3]))
    assert jnp.array_equal(
        table.mask,
        jnp.asarray([False, True, False, True, False]),
    )
    table.set_indices(jnp.asarray([3]), value=False)
    assert not bool(table.mask[3])


def test_deformation_table_rejects_negative_size():
    with pytest.raises(ValueError, match="num_gaussians"):
        DeformationTable(-1)


def test_deformation_table_prune_preserves_survivor_flags():
    table = DeformationTable(5)
    table.set_indices(jnp.asarray([0, 2, 4]))
    table.prune(jnp.asarray([True, False, True, False, True]))
    assert len(table) == 3
    assert bool(jnp.all(table.mask))


def test_deformation_table_prune_rejects_shape_mismatch():
    table = DeformationTable(5)
    with pytest.raises(ValueError, match="shape"):
        table.prune(jnp.asarray([True, False]))


def test_deformation_table_duplicate_appends_inheriting_children():
    table = DeformationTable(5)
    table.set_indices(jnp.asarray([0, 2]))
    table.duplicate(jnp.asarray([0, 1]))
    assert jnp.array_equal(
        table.mask,
        jnp.asarray([True, False, True, False, False, True, False]),
    )


def test_deformation_table_split_replaces_parents_with_children():
    table = DeformationTable(5)
    table.set_indices(jnp.asarray([0, 2]))
    table.split(jnp.asarray([0, 2]), factor=2)
    assert jnp.array_equal(
        table.mask,
        jnp.asarray([False, False, False, True, True, True, True]),
    )


def test_deformation_table_split_rejects_invalid_factor():
    with pytest.raises(ValueError, match="factor"):
        DeformationTable(2).split(jnp.asarray([0]), factor=0)
