"""CuTe Gaussian projection and its parameter gradients exposed through JAX."""

from collections.abc import Callable

import chex
import jax
import jax.numpy as jnp

from ..config import CapacityConfig
from ..render.types import ProjectedGaussians
from ..scene.camera import Camera
from ..scene.point import GaussianArrays
from ..scene.types import ParameterGradients, VisibleClusters


def _project_gaussian_arrays(
    xyz: chex.Array,
    log_scale: chex.Array,
    rotation: chex.Array,
    opacity: chex.Array,
    sh: chex.Array,
    alive: chex.Array,
    world_to_camera: chex.Array,
    intrinsics: chex.Array,
    camera_center: chex.Array,
    config: CapacityConfig,
    width: int,
    height: int,
    near: float,
    far: float,
    active_degree: int | None = None,
    compacted_clusters: VisibleClusters | None = None,
    clear_invisible: bool = True,
) -> tuple[chex.Array, ...]:
    """Launch projection with the pool layout and optional visible-cluster prefix.

    Without clear_invisible, slots outside the visible clusters define only
    visible (False); their other fields are left unwritten.
    """
    from cutlass.jax import cutlass_call

    from .projection import launch_projection

    if jax.default_backend() != "gpu":
        raise RuntimeError("CuTe projection requires a JAX CUDA device")
    capacity = config.max_gaussians
    output_shapes = (
        jax.ShapeDtypeStruct((capacity * 2,), jnp.float32),
        jax.ShapeDtypeStruct((capacity,), jnp.float32),
        jax.ShapeDtypeStruct((capacity * 4,), jnp.float32),
        jax.ShapeDtypeStruct((capacity,), jnp.float32),
        jax.ShapeDtypeStruct((capacity * 3,), jnp.float32),
        jax.ShapeDtypeStruct((capacity,), jnp.float32),
        jax.ShapeDtypeStruct((capacity,), jnp.int8),
    )
    project_kernel = cutlass_call(
        launch_projection,
        output_shape_dtype=output_shapes,
        use_static_tensors=True,
        capacity=capacity,
        sh_dim=config.sh_dim,
        degree=config.sh_degree if active_degree is None else active_degree,
        width=width,
        height=height,
        near=near,
        far=far,
        cluster_size=config.cluster_size,
        compacted=compacted_clusters is not None,
        clear_invisible=clear_invisible,
    )
    cluster_arrays = (
        (jnp.zeros(1, jnp.int32), jnp.zeros(1, jnp.int32))
        if compacted_clusters is None
        else compacted_clusters
    )
    mean, depth, conic, radius, color, alpha, visible = project_kernel(
        xyz.reshape(-1),
        log_scale.reshape(-1),
        rotation.reshape(-1),
        opacity.reshape(-1),
        sh.reshape(-1),
        alive.astype(jnp.int8),
        world_to_camera.reshape(-1),
        intrinsics,
        camera_center,
        *cluster_arrays,
    )
    return (
        mean.reshape(capacity, 2),
        depth,
        conic.reshape(capacity, 2, 2),
        radius,
        color.reshape(capacity, 3),
        alpha,
        visible.astype(jnp.bool_),
    )


def project_cute(
    pool: GaussianArrays, camera: Camera, config: CapacityConfig
) -> ProjectedGaussians:
    intrinsics = jnp.stack([camera.fx, camera.fy, camera.cx, camera.cy])
    return ProjectedGaussians(
        *_project_gaussian_arrays(
            pool.xyz,
            pool.log_scale,
            pool.rotation,
            pool.opacity,
            pool.sh,
            pool.alive,
            camera.world_to_camera,
            intrinsics,
            camera.center,
            config,
            camera.width,
            camera.height,
            camera.near,
            camera.far,
        )
    )


def project_cute_vjp(
    pool: GaussianArrays,
    camera: Camera,
    config: CapacityConfig,
    active_degree: int | None = None,
    compacted_clusters: VisibleClusters | None = None,
) -> ProjectedGaussians:
    """CuTe projection forward and analytic CuTe parameter pullback."""
    if active_degree is not None and not 0 <= active_degree <= config.sh_degree:
        raise ValueError("active_degree must fit the pool's SH coefficients")
    intrinsics = jnp.stack([camera.fx, camera.fy, camera.cx, camera.cy])

    @jax.custom_vjp
    def project_arrays(
        xyz, log_scale, rotation, opacity, sh, alive, world_to_camera, intrinsics, camera_center
    ):
        return _project_gaussian_arrays(
            xyz,
            log_scale,
            rotation,
            opacity,
            sh,
            alive,
            world_to_camera,
            intrinsics,
            camera_center,
            config,
            camera.width,
            camera.height,
            camera.near,
            camera.far,
            active_degree,
            compacted_clusters,
        )

    def project_forward(
        xyz, log_scale, rotation, opacity, sh, alive, world_to_camera, intrinsics, camera_center
    ):
        result = _project_gaussian_arrays(
            xyz,
            log_scale,
            rotation,
            opacity,
            sh,
            alive,
            world_to_camera,
            intrinsics,
            camera_center,
            config,
            camera.width,
            camera.height,
            camera.near,
            camera.far,
            active_degree,
            compacted_clusters,
        )
        return result, (
            xyz,
            log_scale,
            rotation,
            opacity,
            sh,
            world_to_camera,
            intrinsics,
            camera_center,
            result[4],
        )

    def project_backward(residual, cotangents):
        xyz, log_scale, rotation, opacity, sh, world_to_camera, intrinsics, camera_center, color = (
            residual
        )
        gradients = _compute_projection_gradients(
            (xyz, log_scale, rotation, opacity, sh),
            world_to_camera,
            intrinsics,
            camera_center,
            color,
            cotangents,
            camera,
            config,
            active_degree,
            compacted_clusters,
            False,
        )
        return (
            *(
                gradient.reshape(value.shape)
                for gradient, value in zip(
                    gradients, (xyz, log_scale, rotation, opacity, sh), strict=True
                )
            ),
            None,
            None,
            None,
            None,
        )

    project_arrays.defvjp(project_forward, project_backward)
    return ProjectedGaussians(
        *project_arrays(
            pool.xyz,
            pool.log_scale,
            pool.rotation,
            pool.opacity,
            pool.sh,
            pool.alive,
            camera.world_to_camera,
            intrinsics,
            camera.center,
        )
    )


def _compute_projection_gradients(
    parameters: tuple[chex.Array, ...],
    world_to_camera: chex.Array,
    intrinsics: chex.Array,
    camera_center: chex.Array,
    projected_color: chex.Array,
    cotangents: tuple[chex.Array, ...],
    camera: Camera,
    config: CapacityConfig,
    active_degree: int | None,
    visible_clusters: VisibleClusters | None,
    compact_gradients: bool,
    rgb_only: bool = False,
    active_sh_only: bool = False,
    sh_color_only: bool = False,
) -> ParameterGradients:
    """Compute parameter gradients from projected-field cotangents in CuTe."""
    from cutlass.jax import cutlass_call

    from .projection_backward import launch_projection_backward

    degree = config.sh_degree if active_degree is None else active_degree
    sh_gradient_dim = (
        1 if sh_color_only else ((degree + 1) ** 2 if active_sh_only else config.sh_dim)
    )
    output_shapes = tuple(
        jax.ShapeDtypeStruct(
            (config.max_gaussians * sh_gradient_dim * 3 if index == 4 else value.size,),
            jnp.float32,
        )
        for index, value in enumerate(parameters)
    )
    backward_kernel = cutlass_call(
        launch_projection_backward,
        output_shape_dtype=output_shapes,
        use_static_tensors=True,
        capacity=config.max_gaussians,
        sh_dim=config.sh_dim,
        sh_gradient_dim=sh_gradient_dim,
        degree=degree,
        near=camera.near,
        cluster_size=config.cluster_size,
        compacted=visible_clusters is not None,
        compact_gradients=compact_gradients,
        rgb_only=rgb_only,
        sh_color_only=sh_color_only,
    )
    cluster_arrays = (
        (jnp.zeros(1, jnp.int32), jnp.zeros(1, jnp.int32))
        if visible_clusters is None
        else visible_clusters
    )
    mean_grad, depth_grad, conic_grad, radius_grad, color_grad, alpha_grad = cotangents[:6]
    return backward_kernel(
        *(value.reshape(-1) for value in parameters),
        world_to_camera.reshape(-1),
        intrinsics,
        camera_center,
        projected_color.reshape(-1),
        *cluster_arrays,
        mean_grad.reshape(-1),
        None if rgb_only else depth_grad,
        conic_grad.reshape(-1),
        None if rgb_only else radius_grad,
        color_grad.reshape(-1),
        alpha_grad,
    )


def project_with_compact_pullback(
    pool: GaussianArrays,
    camera: Camera,
    config: CapacityConfig,
    active_degree: int,
    clusters: VisibleClusters,
    *,
    rgb_only: bool = False,
    active_sh_only: bool = False,
    sh_color_only: bool = False,
) -> tuple[ProjectedGaussians, Callable[[ProjectedGaussians], ParameterGradients]]:
    """CuTe projection and a pullback producing gradients in visible-cluster order.

    The pullback returns fixed-capacity buffers in visible-cluster order.
    Only clusters[0][:clusters[1][0]] contribute valid entries. Parameter
    gradients follow xyz, log_scale, rotation, opacity and SH order; the tail
    is undefined. Ordinary autodiff uses project_cute_vjp instead.
    rgb_only specializes the pullback for zero depth and radius cotangents.
    active_sh_only returns only (active_degree + 1)**2 SH gradient coefficients;
    parameter storage remains unchanged. Optax restores the zero gradient tail.
    sh_color_only instead returns [capacity, 1, 3] masked color cotangents for
    reconstruction inside Optax. The position gradient still includes the
    SH direction derivative.
    Outside the visible clusters, only projected.visible is defined (False):
    binning and rasterization read the other fields of visible Gaussians only.
    """
    if not 0 <= active_degree <= config.sh_degree:
        raise ValueError("active_degree must fit the pool's SH coefficients")
    intrinsics = jnp.stack([camera.fx, camera.fy, camera.cx, camera.cy])
    parameters = (pool.xyz, pool.log_scale, pool.rotation, pool.opacity, pool.sh)
    projected = ProjectedGaussians(
        *_project_gaussian_arrays(
            *parameters,
            pool.alive,
            camera.world_to_camera,
            intrinsics,
            camera.center,
            config,
            camera.width,
            camera.height,
            camera.near,
            camera.far,
            active_degree,
            clusters,
            clear_invisible=False,
        )
    )

    def pullback(cotangents):
        arrays = _compute_projection_gradients(
            parameters,
            camera.world_to_camera,
            intrinsics,
            camera.center,
            projected.color,
            (
                cotangents.mean,
                cotangents.depth,
                cotangents.conic,
                cotangents.radius,
                cotangents.color,
                cotangents.alpha,
            ),
            camera,
            config,
            active_degree,
            clusters,
            True,
            rgb_only,
            active_sh_only,
            sh_color_only,
        )
        return tuple(
            array.reshape(
                (config.max_gaussians, 1 if sh_color_only else (active_degree + 1) ** 2, 3)
                if (active_sh_only or sh_color_only) and index == 4
                else value.shape
            )
            for index, (array, value) in enumerate(zip(arrays, parameters, strict=True))
        )

    return projected, pullback
