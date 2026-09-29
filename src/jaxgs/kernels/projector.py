import jax
import jax.numpy as jnp

from ..config import CapacityConfig
from ..render.projection import ProjectedGaussians
from ..scene.camera import Camera
from ..scene.point import GaussianPool


def _project_arrays(
    xyz,
    log_scale,
    rotation,
    opacity,
    sh,
    alive,
    view,
    intrinsic,
    center,
    config: CapacityConfig,
    width: int,
    height: int,
    near: float,
    far: float,
    active_degree: int | None = None,
    compacted_clusters=None,
):
    from cutlass.jax import cutlass_call

    from .projection import launch_projection

    if jax.default_backend() != "gpu":
        raise RuntimeError("CuTe projection requires a JAX CUDA device")
    c = config.max_gaussians
    shapes = (
        jax.ShapeDtypeStruct((c * 2,), jnp.float32),
        jax.ShapeDtypeStruct((c,), jnp.float32),
        jax.ShapeDtypeStruct((c * 4,), jnp.float32),
        jax.ShapeDtypeStruct((c,), jnp.float32),
        jax.ShapeDtypeStruct((c * 3,), jnp.float32),
        jax.ShapeDtypeStruct((c,), jnp.float32),
        jax.ShapeDtypeStruct((c,), jnp.int8),
    )
    call = cutlass_call(
        launch_projection,
        output_shape_dtype=shapes,
        use_static_tensors=True,
        capacity=c,
        sh_dim=config.sh_dim,
        degree=config.sh_degree if active_degree is None else active_degree,
        width=width,
        height=height,
        near=near,
        far=far,
        cluster_size=config.cluster_size,
        compacted=compacted_clusters is not None,
    )
    clusters = (
        (jnp.zeros(1, jnp.int32), jnp.zeros(1, jnp.int32))
        if compacted_clusters is None
        else compacted_clusters
    )
    mean, depth, conic, radius, color, alpha, visible = call(
        xyz.reshape(-1),
        log_scale.reshape(-1),
        rotation.reshape(-1),
        opacity.reshape(-1),
        sh.reshape(-1),
        alive.astype(jnp.int8),
        view.reshape(-1),
        intrinsic,
        center,
        *clusters,
    )
    return (
        mean.reshape(c, 2),
        depth,
        conic.reshape(c, 2, 2),
        radius,
        color.reshape(c, 3),
        alpha,
        visible.astype(jnp.bool_),
    )


def project_cute(pool: GaussianPool, camera: Camera, config: CapacityConfig) -> ProjectedGaussians:
    intrinsic = jnp.stack([camera.fx, camera.fy, camera.cx, camera.cy])
    return ProjectedGaussians(
        *_project_arrays(
            pool.xyz,
            pool.log_scale,
            pool.rotation,
            pool.opacity,
            pool.sh,
            pool.alive,
            camera.world_to_camera,
            intrinsic,
            camera.center,
            config,
            camera.width,
            camera.height,
            camera.near,
            camera.far,
        )
    )


def project_cute_vjp(
    pool: GaussianPool,
    camera: Camera,
    config: CapacityConfig,
    active_degree: int | None = None,
    compacted_clusters=None,
) -> ProjectedGaussians:
    """CuTe projection forward and analytic CuTe parameter pullback."""
    if active_degree is not None and not 0 <= active_degree <= config.sh_degree:
        raise ValueError("active_degree must fit the pool's SH coefficients")
    intrinsic = jnp.stack([camera.fx, camera.fy, camera.cx, camera.cy])

    @jax.custom_vjp
    def projected(xyz, log_scale, rotation, opacity, sh, alive, view, focal, center):
        return _project_arrays(
            xyz,
            log_scale,
            rotation,
            opacity,
            sh,
            alive,
            view,
            focal,
            center,
            config,
            camera.width,
            camera.height,
            camera.near,
            camera.far,
            active_degree,
            compacted_clusters,
        )

    def projected_fwd(xyz, log_scale, rotation, opacity, sh, alive, view, focal, center):
        result = _project_arrays(
            xyz,
            log_scale,
            rotation,
            opacity,
            sh,
            alive,
            view,
            focal,
            center,
            config,
            camera.width,
            camera.height,
            camera.near,
            camera.far,
            active_degree,
            compacted_clusters,
        )
        return result, (xyz, log_scale, rotation, opacity, sh, view, focal, center, result[4])

    def projected_bwd(residual, cotangents):
        xyz, log_scale, rotation, opacity, sh, view, focal, center, color = residual
        gradients = _project_pullback_arrays(
            (xyz, log_scale, rotation, opacity, sh),
            view,
            focal,
            center,
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

    projected.defvjp(projected_fwd, projected_bwd)
    return ProjectedGaussians(
        *projected(
            pool.xyz,
            pool.log_scale,
            pool.rotation,
            pool.opacity,
            pool.sh,
            pool.alive,
            camera.world_to_camera,
            intrinsic,
            camera.center,
        )
    )


def _project_pullback_arrays(
    params,
    view,
    focal,
    center,
    color,
    cotangents,
    camera,
    config,
    active_degree,
    clusters,
    compact_gradients,
):
    from cutlass.jax import cutlass_call

    from .projection_backward import launch_projection_backward

    shapes = tuple(jax.ShapeDtypeStruct((value.size,), jnp.float32) for value in params)
    call = cutlass_call(
        launch_projection_backward,
        output_shape_dtype=shapes,
        use_static_tensors=True,
        capacity=config.max_gaussians,
        sh_dim=config.sh_dim,
        degree=config.sh_degree if active_degree is None else active_degree,
        near=camera.near,
        cluster_size=config.cluster_size,
        compacted=clusters is not None,
        compact_gradients=compact_gradients,
    )
    cluster_arrays = (
        (jnp.zeros(1, jnp.int32), jnp.zeros(1, jnp.int32)) if clusters is None else clusters
    )
    gmean, gdepth, gconic, gradius, gcolor, galpha = cotangents[:6]
    return call(
        *(value.reshape(-1) for value in params),
        view.reshape(-1),
        focal,
        center,
        color.reshape(-1),
        *cluster_arrays,
        gmean.reshape(-1),
        gdepth,
        gconic.reshape(-1),
        gradius,
        gcolor.reshape(-1),
        galpha,
    )


def project_cute_sparse(pool, camera, config, active_degree, clusters):
    """Training projection with LiteGS's compact-gradient pullback.

    The pullback returns fixed-capacity buffers in visible-cluster order.
    Only slots belonging to clusters[:count] are valid. Sparse Adam consumes
    this prefix directly; ordinary autodiff uses project_cute_vjp instead.
    """
    if not 0 <= active_degree <= config.sh_degree:
        raise ValueError("active_degree must fit the pool's SH coefficients")
    focal = jnp.stack([camera.fx, camera.fy, camera.cx, camera.cy])
    params = (pool.xyz, pool.log_scale, pool.rotation, pool.opacity, pool.sh)
    projected = ProjectedGaussians(
        *_project_arrays(
            *params,
            pool.alive,
            camera.world_to_camera,
            focal,
            camera.center,
            config,
            camera.width,
            camera.height,
            camera.near,
            camera.far,
            active_degree,
            clusters,
        )
    )

    def pullback(cotangents):
        arrays = _project_pullback_arrays(
            params,
            camera.world_to_camera,
            focal,
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
        )
        return tuple(
            array.reshape(value.shape) for array, value in zip(arrays, params, strict=True)
        )

    return projected, pullback
