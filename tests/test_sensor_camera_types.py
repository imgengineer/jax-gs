import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_gs.sensors.kernels.cameras import (
    BivariateWindshieldDistortion,
    FThetaProjection,
    NoExternalDistortion,
    OpenCVFisheyeProjection,
    OpenCVPinholeProjection,
    ReferencePolynomial,
    ShutterType,
    from_components,
    validate_camera_projection,
)


def test_current_main_enum_values_are_stable():
    assert int(ShutterType.ROLLING_TOP_TO_BOTTOM) == 1
    assert int(ShutterType.GLOBAL) == 5
    assert int(ReferencePolynomial.FORWARD) == 0
    assert int(ReferencePolynomial.BACKWARD) == 1


def test_pinhole_projection_validates_shapes_and_transforms_intrinsics():
    projection = OpenCVPinholeProjection(
        focal_length=jnp.array([100.0, 120.0]),
        principal_point=jnp.array([50.0, 40.0]),
        radial_coeffs=jnp.zeros(6),
        tangential_coeffs=jnp.zeros(2),
        thin_prism_coeffs=jnp.zeros(4),
        resolution=(100, 80),
    )
    transformed = projection.transform((2.0, 0.5), (4.0, 3.0), (196, 37))
    np.testing.assert_allclose(transformed.focal_length, [200.0, 60.0])
    np.testing.assert_allclose(transformed.principal_point, [96.0, 17.0])
    assert transformed.resolution == (196, 37)
    np.testing.assert_array_equal(projection.principal_point, [50.0, 40.0])

    with pytest.raises(ValueError, match="focal_length"):
        OpenCVPinholeProjection(
            focal_length=jnp.ones(1),
            principal_point=jnp.ones(2),
            radial_coeffs=jnp.zeros(6),
            tangential_coeffs=jnp.zeros(2),
            thin_prism_coeffs=jnp.zeros(4),
            resolution=(100, 80),
        )


def test_windshield_factory_infers_degrees_and_packs_current_main_layout():
    h = jnp.array([0.0, 1.0, 0.0])
    v = jnp.array([0.0, 0.0, 1.0])
    distortion = from_components(h, v, h, v, ReferencePolynomial.FORWARD)
    assert isinstance(distortion, BivariateWindshieldDistortion)
    assert distortion.distortion_coeffs.shape == (42,)
    assert distortion.h_poly_degree == 1
    assert distortion.v_poly_degree == 1
    np.testing.assert_array_equal(distortion.distortion_coeffs[:3], h)
    np.testing.assert_array_equal(distortion.distortion_coeffs[6:9], v)
    np.testing.assert_array_equal(distortion.distortion_coeffs[21:24], h)
    np.testing.assert_array_equal(distortion.distortion_coeffs[27:30], v)

    with pytest.raises(ValueError, match="triangular"):
        from_components(jnp.ones(2), v, jnp.ones(2), v, 0)
    with pytest.raises(ValueError, match="matching triangular degree"):
        from_components(h, v, jnp.zeros(6), v, 0)


def test_ftheta_ainv_and_transform_follow_upstream_image_convention():
    projection = FThetaProjection(
        principal_point=jnp.array([50.0, 40.0]),
        fw_poly=jnp.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]),
        bw_poly=jnp.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
        A=jnp.array([2.0, 0.0, 0.0, 4.0]),
        resolution=(100, 80),
        reference_polynomial=ReferencePolynomial.FORWARD,
        fw_poly_degree=1,
        bw_poly_degree=1,
        newton_iterations=10,
        max_angle=1.4,
        min_2d_norm=1.0e-6,
    )
    np.testing.assert_allclose(projection.Ainv, [0.5, 0.0, 0.0, 0.25])
    transformed = projection.transform((2.0, 0.5), (3.0, 4.0), (194, 36))
    np.testing.assert_allclose(transformed.principal_point, [97.5, 15.75])
    np.testing.assert_allclose(transformed.fw_poly[1], 50.0)
    np.testing.assert_allclose(transformed.bw_poly[1], 0.02)
    np.testing.assert_allclose(transformed.A, [8.0, 0.0, 0.0, 4.0])


def test_fisheye_transform_recomputes_solver_initial_factor():
    projection = OpenCVFisheyeProjection(
        principal_point=jnp.array([50.0, 40.0]),
        focal_length=jnp.array([100.0, 120.0]),
        forward_poly=jnp.zeros(4),
        approx_backward_factor=jnp.array([1.0]),
        resolution=(100, 80),
        newton_iterations=10,
        max_angle=1.8,
        min_2d_norm=1.0e-6,
    )
    transformed = projection.transform((0.5, 2.0), (2.0, 3.0), (50, 160))
    np.testing.assert_allclose(transformed.principal_point, [22.75, 77.5])
    np.testing.assert_allclose(transformed.focal_length, [50.0, 240.0])
    expected_factor = 1.8 / max(50.0 / (2.0 * 50.0), 160.0 / (2.0 * 240.0))
    np.testing.assert_allclose(transformed.approx_backward_factor, [expected_factor])


def test_projection_types_are_jax_pytrees_and_value_validation_is_opt_in():
    projection = FThetaProjection(
        principal_point=jnp.array([50.0, 40.0]),
        fw_poly=jnp.array([0.0, 100.0, 0.0, 0.0, 0.0, 0.0]),
        bw_poly=jnp.array([0.0, 0.01, 0.0, 0.0, 0.0, 0.0]),
        A=jnp.array([1.0, 0.0, 0.0, 1.0]),
        resolution=(100, 80),
        reference_polynomial=0,
        fw_poly_degree=1,
        bw_poly_degree=1,
        newton_iterations=10,
        max_angle=1.4,
        min_2d_norm=1.0e-6,
    )
    leaves, tree = jax.tree_util.tree_flatten(projection)
    assert len(leaves) == 4
    assert isinstance(jax.tree_util.tree_unflatten(tree, leaves), FThetaProjection)
    validate_camera_projection(projection)
    validate_camera_projection(
        OpenCVPinholeProjection(
            jnp.ones(2), jnp.zeros(2), jnp.zeros(6), jnp.zeros(2), jnp.zeros(4), (1, 1)
        )
    )
    assert jax.tree_util.tree_leaves(NoExternalDistortion()) == []

    singular = FThetaProjection(**{**projection.__dict__, "A": jnp.zeros(4)})
    with pytest.raises(ValueError, match="non-singular"):
        validate_camera_projection(singular)


def test_constructor_rejects_nonfinite_or_out_of_range_scalar_configuration():
    base = {
        "principal_point": jnp.zeros(2),
        "fw_poly": jnp.zeros(6),
        "bw_poly": jnp.zeros(6),
        "A": jnp.array([1.0, 0.0, 0.0, 1.0]),
        "resolution": (1, 1),
        "reference_polynomial": 0,
        "fw_poly_degree": 0,
        "bw_poly_degree": 0,
        "newton_iterations": 0,
        "max_angle": math.pi,
        "min_2d_norm": 1.0e-6,
    }
    with pytest.raises(ValueError, match="max_angle"):
        FThetaProjection(**{**base, "max_angle": math.inf})
    with pytest.raises(ValueError, match="strictly positive"):
        FThetaProjection(**{**base, "min_2d_norm": 0.0})
