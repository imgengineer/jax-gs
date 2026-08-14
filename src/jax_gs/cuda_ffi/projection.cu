#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <string>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr int kBlockSize = 256;

ffi::Error cuda_error(const char* operation, cudaError_t error) {
  return ffi::Error::Internal(std::string(operation) + ": " +
                              cudaGetErrorString(error));
}

__device__ __forceinline__ float jax_max(float left, float right) {
  if (isnan(left) || isnan(right)) return NAN;
  return left > right ? left : right;
}

__device__ __forceinline__ float jax_min(float left, float right) {
  if (isnan(left) || isnan(right)) return NAN;
  return left < right ? left : right;
}

__device__ __forceinline__ float safe_denominator(float value) {
  const float sign = value < 0.0f ? -1.0f : 1.0f;
  return fabsf(value) < 1.0e-8f ? sign * 1.0e-8f : value;
}

__device__ __forceinline__ float dot3(const float left[3],
                                      const float right[3]) {
  return left[0] * right[0] + left[1] * right[1] + left[2] * right[2];
}

__global__ void projection_forward_kernel(
    int camera_count, int gaussian_count, int image_width, int image_height,
    float eps2d, float near_plane, float far_plane, float radius_clip,
    float alpha_threshold, const float* means_camera, const float* means2d,
    const float* covariances_camera, const float* opacities,
    const bool* active_mask, const float* intrinsics, int32_t* radii,
    float* conics, bool* valid_out) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t count =
      static_cast<int64_t>(camera_count) * gaussian_count;
  if (index >= count) return;

  const int camera_id = static_cast<int>(index / gaussian_count);
  const int gaussian_id = static_cast<int>(index % gaussian_count);
  const float* mean = means_camera + index * 3;
  const float* covariance = covariances_camera + index * 9;
  const float* K = intrinsics + camera_id * 9;

  const float z_safe = safe_denominator(mean[2]);
  const float inverse_z = __fdividef(1.0f, z_safe);
  const float inverse_z_squared = inverse_z * inverse_z;
  const float fx = K[0];
  const float fy = K[4];
  const float cx = K[2];
  const float cy = K[5];
  const float fx_safe = safe_denominator(fx);
  const float fy_safe = safe_denominator(fy);
  const float tan_fov_x = __fdividef(0.5f * image_width, fx_safe);
  const float tan_fov_y = __fdividef(0.5f * image_height, fy_safe);
  const float limit_x_positive =
      __fdividef(image_width - cx, fx_safe) + 0.3f * tan_fov_x;
  const float limit_x_negative =
      __fdividef(cx, fx_safe) + 0.3f * tan_fov_x;
  const float limit_y_positive =
      __fdividef(image_height - cy, fy_safe) + 0.3f * tan_fov_y;
  const float limit_y_negative =
      __fdividef(cy, fy_safe) + 0.3f * tan_fov_y;
  const float tx = mean[2] *
                   jax_min(jax_max(mean[0] * inverse_z, -limit_x_negative),
                           limit_x_positive);
  const float ty = mean[2] *
                   jax_min(jax_max(mean[1] * inverse_z, -limit_y_negative),
                           limit_y_positive);
  float jacobian[2][3] = {
      {fx * inverse_z, 0.0f, -fx * tx * inverse_z_squared},
      {0.0f, fy * inverse_z, -fy * ty * inverse_z_squared},
  };

  float projected_once[2][3];
#pragma unroll
  for (int row = 0; row < 2; ++row) {
#pragma unroll
    for (int column = 0; column < 3; ++column) {
      const float covariance_column[3] = {
          covariance[column], covariance[3 + column],
          covariance[6 + column]};
      projected_once[row][column] =
          dot3(jacobian[row], covariance_column);
    }
  }
  float covariance2d[2][2];
#pragma unroll
  for (int row = 0; row < 2; ++row) {
#pragma unroll
    for (int column = 0; column < 2; ++column) {
      covariance2d[row][column] =
          dot3(projected_once[row], jacobian[column]);
    }
  }

  const float covariance_xx = covariance2d[0][0] + eps2d;
  const float covariance_xy = covariance2d[0][1];
  const float covariance_yx = covariance2d[1][0];
  const float covariance_yy = covariance2d[1][1] + eps2d;
  const float determinant =
      covariance_xx * covariance_yy - covariance_xy * covariance_yx;
  const float safe_determinant = jax_max(determinant, 1.0e-10f);
  const float conic_xx = __fdividef(covariance_yy, safe_determinant);
  const float conic_xy = __fdividef(
      -0.5f * (covariance_xy + covariance_yx), safe_determinant);
  const float conic_yy = __fdividef(covariance_xx, safe_determinant);
  conics[index * 3] = conic_xx;
  conics[index * 3 + 1] = conic_xy;
  conics[index * 3 + 2] = conic_yy;

  const float opacity = opacities[gaussian_id];
  const bool opacity_valid = opacity >= alpha_threshold;
  const float opacity_ratio =
      jax_max(__fdividef(opacity, alpha_threshold), 1.0f);
  const float opacity_extend =
      sqrtf(jax_max(2.0f * logf(opacity_ratio), 0.0f));
  const float extend = jax_min(3.33f, opacity_extend);
  const float radius_x =
      ceilf(extend * sqrtf(jax_max(covariance_xx, 0.0f)));
  const float radius_y =
      ceilf(extend * sqrtf(jax_max(covariance_yy, 0.0f)));

  const float mean2d_x = means2d[index * 2];
  const float mean2d_y = means2d[index * 2 + 1];
  bool valid = determinant > 0.0f && mean[2] > near_plane &&
               mean[2] < far_plane && opacity_valid;
  valid = valid && (radius_x > radius_clip || radius_y > radius_clip);
  valid = valid && mean2d_x + radius_x > 0.0f &&
          mean2d_x - radius_x < image_width &&
          mean2d_y + radius_y > 0.0f &&
          mean2d_y - radius_y < image_height;
  valid = valid && active_mask[gaussian_id];
  valid = valid && isfinite(mean2d_x) && isfinite(mean2d_y) &&
          isfinite(mean[2]) && isfinite(conic_xx) && isfinite(conic_xy) &&
          isfinite(conic_yy);
  valid_out[index] = valid;
  radii[index * 2] = valid ? static_cast<int32_t>(radius_x) : 0;
  radii[index * 2 + 1] = valid ? static_cast<int32_t>(radius_y) : 0;
}

ffi::Error projection_forward_host(
    cudaStream_t stream, ffi::BufferR3<ffi::F32> means_camera,
    ffi::BufferR3<ffi::F32> means2d,
    ffi::BufferR4<ffi::F32> covariances_camera,
    ffi::BufferR1<ffi::F32> opacities,
    ffi::BufferR1<ffi::PRED> active_mask,
    ffi::BufferR3<ffi::F32> intrinsics,
    ffi::ResultBufferR3<ffi::S32> radii,
    ffi::ResultBufferR3<ffi::F32> conics,
    ffi::ResultBufferR2<ffi::PRED> valid,
    int64_t image_width, int64_t image_height, float eps2d,
    float near_plane, float far_plane, float radius_clip,
    float alpha_threshold) {
  const int64_t camera_count = means_camera.dimensions()[0];
  const int64_t gaussian_count = means_camera.dimensions()[1];
  if (camera_count > std::numeric_limits<int>::max() ||
      gaussian_count > std::numeric_limits<int>::max()) {
    return ffi::Error::InvalidArgument("projection dimensions exceed int32");
  }
  const int64_t count = camera_count * gaussian_count;
  const int64_t blocks = (count + kBlockSize - 1) / kBlockSize;
  projection_forward_kernel<<<static_cast<unsigned int>(blocks), kBlockSize,
                              0, stream>>>(
      static_cast<int>(camera_count), static_cast<int>(gaussian_count),
      static_cast<int>(image_width), static_cast<int>(image_height), eps2d,
      near_plane, far_plane, radius_clip, alpha_threshold,
      means_camera.typed_data(), means2d.typed_data(),
      covariances_camera.typed_data(), opacities.typed_data(),
      active_mask.typed_data(), intrinsics.typed_data(), radii->typed_data(),
      conics->typed_data(), valid->typed_data());
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return cuda_error("CUDA projection launch failed", error);
  }
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxGsProjectionForward, projection_forward_host,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::BufferR3<ffi::F32>>()
        .Arg<ffi::BufferR3<ffi::F32>>()
        .Arg<ffi::BufferR4<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::PRED>>()
        .Arg<ffi::BufferR3<ffi::F32>>()
        .Ret<ffi::BufferR3<ffi::S32>>()
        .Ret<ffi::BufferR3<ffi::F32>>()
        .Ret<ffi::BufferR2<ffi::PRED>>()
        .Attr<int64_t>("image_width")
        .Attr<int64_t>("image_height")
        .Attr<float>("eps2d")
        .Attr<float>("near_plane")
        .Attr<float>("far_plane")
        .Attr<float>("radius_clip")
        .Attr<float>("alpha_threshold"));
