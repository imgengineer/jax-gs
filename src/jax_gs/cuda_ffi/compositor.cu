/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of
 * California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA/XLA FFI adaptation of the public gsplat 1.5.3 rasterization
 * architecture. This file contains no PyTorch/ATen dependencies.
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

namespace {

constexpr float kMaxAlpha = 0.999f;
constexpr float kMinOneMinusAlpha = 1.0e-3f;
constexpr int kTileSize = 16;
constexpr int kTilePixels = kTileSize * kTileSize;

template <typename T>
__device__ __forceinline__ T clamp_value(T value, T minimum, T maximum) {
  return min(max(value, minimum), maximum);
}

__device__ __forceinline__ int safe_gaussian_id(int id, int count) {
  return id < 0 ? -1 : min(id, count - 1);
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  return value;
}

__device__ __forceinline__ bool evaluate_weight(
    const float3 conic, const float2 mean, const float opacity,
    const float px, const float py, const float alpha_threshold,
    float* visibility, float* alpha) {
  const float dx = px - mean.x;
  const float dy = py - mean.y;
  const float sigma =
      0.5f * (conic.x * dx * dx + conic.z * dy * dy) +
      conic.y * dx * dy;
  if (!isfinite(sigma) || sigma < 0.0f) {
    return false;
  }
  const float vis = __expf(-sigma);
  const float raw_alpha = opacity * vis;
  // Match jnp.minimum(..., MAX_ALPHA) followed by nan_to_num with
  // nan/neginf -> 0 and posinf -> MAX_ALPHA. CUDA fminf alone maps NaN to
  // its numeric operand, which would otherwise turn NaN opacity opaque.
  const float candidate_alpha =
      (isnan(raw_alpha) || (isinf(raw_alpha) && raw_alpha < 0.0f))
          ? 0.0f
          : fminf(kMaxAlpha, raw_alpha);
  if (candidate_alpha < alpha_threshold) {
    return false;
  }
  *visibility = vis;
  *alpha = candidate_alpha;
  return true;
}

template <int Channels>
__global__ void compositor_forward_kernel(
    int gaussian_count, int input_capacity, int image_width, int image_height,
    int tile_width, int tile_count, int per_tile_bound,
    float alpha_threshold, float transmittance_threshold,
    const float* means, const float* conics, const float* colors,
    const float* opacities, const int32_t* offsets, const int32_t* ids,
    const int32_t* valid_count, float* foreground, float* alpha_out,
    float* accepted_final_transmittance, int32_t* last_ids,
    bool* tile_overflow) {
  const int tile_id = blockIdx.x;
  const int tid = threadIdx.x;
  if (tile_id >= tile_count) return;

  const int tile_x = tile_id % tile_width;
  const int tile_y = tile_id / tile_width;
  const int pixel_x = tile_x * kTileSize + (tid & (kTileSize - 1));
  const int pixel_y = tile_y * kTileSize + (tid >> 4);
  const bool pixel_valid = pixel_x < image_width && pixel_y < image_height;
  const int pixel_id = min(pixel_y * image_width + pixel_x,
                           image_width * image_height - 1);

  const int bounded_valid_count =
      clamp_value(*valid_count, 0, input_capacity);
  int start = offsets[tile_id];
  int end = tile_id + 1 < tile_count ? offsets[tile_id + 1]
                                     : bounded_valid_count;
  start = clamp_value(start, 0, bounded_valid_count);
  end = clamp_value(end, start, bounded_valid_count);
  const int candidate_count = end - start;
  const int rendered_count = min(candidate_count, per_tile_bound);
  if (tid == 0) tile_overflow[tile_id] = candidate_count > per_tile_bound;

  extern __shared__ unsigned char shared[];
  int32_t* batch_ids = reinterpret_cast<int32_t*>(shared);
  float2* batch_means = reinterpret_cast<float2*>(batch_ids + kTilePixels);
  float3* batch_conics = reinterpret_cast<float3*>(batch_means + kTilePixels);
  float* batch_opacities = reinterpret_cast<float*>(batch_conics + kTilePixels);
  float* batch_colors = batch_opacities + kTilePixels;

  float transmittance = 1.0f;
  float accepted_transmittance = 1.0f;
  int32_t last_id = -1;
  float rendered[Channels] = {};
  bool done = !pixel_valid;
  const float px = static_cast<float>(pixel_x) + 0.5f;
  const float py = static_cast<float>(pixel_y) + 0.5f;

  for (int batch_start = 0; batch_start < rendered_count;
       batch_start += kTilePixels) {
    const int local = batch_start + tid;
    if (local < rendered_count) {
      const int position = start + local;
      const int raw_id = ids[position];
      const int gaussian_id = safe_gaussian_id(raw_id, gaussian_count);
      batch_ids[tid] = gaussian_id;
      if (gaussian_id >= 0) {
        batch_means[tid] = reinterpret_cast<const float2*>(means)[gaussian_id];
        batch_conics[tid] = reinterpret_cast<const float3*>(conics)[gaussian_id];
        batch_opacities[tid] = opacities[gaussian_id];
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
          batch_colors[tid * Channels + channel] =
              colors[gaussian_id * Channels + channel];
        }
      }
    }
    __syncthreads();

    const int batch_size = min(kTilePixels, rendered_count - batch_start);
    for (int t = 0; t < batch_size && !done; ++t) {
      const int gaussian_id = batch_ids[t];
      float visibility;
      float alpha;
      if (gaussian_id < 0 ||
          !evaluate_weight(batch_conics[t], batch_means[t],
                           batch_opacities[t], px, py, alpha_threshold,
                           &visibility, &alpha)) {
        continue;
      }
      const float next_transmittance = transmittance * (1.0f - alpha);
      if (!(next_transmittance > transmittance_threshold)) {
        transmittance = next_transmittance;
        done = true;
        continue;
      }
      const float weight = alpha * transmittance;
#pragma unroll
      for (int channel = 0; channel < Channels; ++channel) {
        rendered[channel] += batch_colors[t * Channels + channel] * weight;
      }
      transmittance = next_transmittance;
      accepted_transmittance = next_transmittance;
      last_id = start + batch_start + t;
    }
    __syncthreads();
  }

  if (pixel_valid) {
#pragma unroll
    for (int channel = 0; channel < Channels; ++channel) {
      foreground[pixel_id * Channels + channel] = rendered[channel];
    }
    alpha_out[pixel_id] = 1.0f - accepted_transmittance;
    accepted_final_transmittance[pixel_id] = accepted_transmittance;
    last_ids[pixel_id] = last_id;
  }
}

template <int Channels>
__global__ void compositor_backward_kernel(
    int gaussian_count, int input_capacity, int image_width, int image_height,
    int tile_width, int tile_count, int per_tile_bound,
    float alpha_threshold, float transmittance_threshold,
    const float* means, const float* conics, const float* colors,
    const float* opacities, const int32_t* offsets, const int32_t* ids,
    const int32_t* valid_count, const float* accepted_final_transmittance,
    const int32_t* last_ids, const float* render_cotangent,
    const float* alpha_cotangent, float* means_gradient,
    float* conics_gradient, float* colors_gradient,
    float* opacities_gradient) {
  const int tile_id = blockIdx.x;
  const int tid = threadIdx.x;
  if (tile_id >= tile_count) return;

  const int tile_x = tile_id % tile_width;
  const int tile_y = tile_id / tile_width;
  const int pixel_x = tile_x * kTileSize + (tid & (kTileSize - 1));
  const int pixel_y = tile_y * kTileSize + (tid >> 4);
  const bool inside = pixel_x < image_width && pixel_y < image_height;
  const int pixel_id = min(pixel_y * image_width + pixel_x,
                           image_width * image_height - 1);
  const float px = static_cast<float>(pixel_x) + 0.5f;
  const float py = static_cast<float>(pixel_y) + 0.5f;

  const int bounded_valid_count =
      clamp_value(*valid_count, 0, input_capacity);
  int start = offsets[tile_id];
  int end = tile_id + 1 < tile_count ? offsets[tile_id + 1]
                                     : bounded_valid_count;
  start = clamp_value(start, 0, bounded_valid_count);
  end = clamp_value(end, start, bounded_valid_count);
  const int rendered_end = min(end, start + per_tile_bound);

  extern __shared__ unsigned char shared[];
  int32_t* batch_ids = reinterpret_cast<int32_t*>(shared);
  float2* batch_means = reinterpret_cast<float2*>(batch_ids + kTilePixels);
  float3* batch_conics = reinterpret_cast<float3*>(batch_means + kTilePixels);
  float* batch_opacities = reinterpret_cast<float*>(batch_conics + kTilePixels);
  float* batch_colors = batch_opacities + kTilePixels;

  float transmittance = inside ? accepted_final_transmittance[pixel_id] : 1.0f;
  const int32_t pixel_last_id = inside ? last_ids[pixel_id] : -1;
  int warp_last_id = pixel_last_id;
  warp_last_id = max(warp_last_id, __shfl_xor_sync(0xffffffffu, warp_last_id, 16));
  warp_last_id = max(warp_last_id, __shfl_xor_sync(0xffffffffu, warp_last_id, 8));
  warp_last_id = max(warp_last_id, __shfl_xor_sync(0xffffffffu, warp_last_id, 4));
  warp_last_id = max(warp_last_id, __shfl_xor_sync(0xffffffffu, warp_last_id, 2));
  warp_last_id = max(warp_last_id, __shfl_xor_sync(0xffffffffu, warp_last_id, 1));

  __shared__ int s_block_last_id;
  if (tid < 8) {
    reinterpret_cast<int*>(shared)[tid] = -1;
  }
  __syncthreads();
  if ((tid & 31) == 0) {
    reinterpret_cast<int*>(shared)[tid >> 5] = warp_last_id;
  }
  __syncthreads();
  if (tid == 0) {
    int max_id = -1;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      max_id = max(max_id, reinterpret_cast<int*>(shared)[w]);
    }
    s_block_last_id = max_id;
  }
  __syncthreads();

  float trailing_cotangent = 0.0f;
  float render_gradient[Channels] = {};
  float output_alpha_gradient = 0.0f;
  if (inside) {
#pragma unroll
    for (int channel = 0; channel < Channels; ++channel) {
      render_gradient[channel] = render_cotangent[pixel_id * Channels + channel];
    }
    output_alpha_gradient = alpha_cotangent[pixel_id];
  }

  const int batches = (rendered_end - start + kTilePixels - 1) / kTilePixels;
  for (int batch = 0; batch < batches; ++batch) {
    const int batch_end = rendered_end - 1 - batch * kTilePixels;
    const int batch_size = min(kTilePixels, batch_end + 1 - start);
    if (batch_end - (batch_size - 1) > s_block_last_id) {
      continue;
    }
    const int position = batch_end - tid;
    if (position >= start) {
      const int raw_id = ids[position];
      const int gaussian_id = safe_gaussian_id(raw_id, gaussian_count);
      batch_ids[tid] = gaussian_id;
      if (gaussian_id >= 0) {
        batch_means[tid] = reinterpret_cast<const float2*>(means)[gaussian_id];
        batch_conics[tid] = reinterpret_cast<const float3*>(conics)[gaussian_id];
        batch_opacities[tid] = opacities[gaussian_id];
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
          batch_colors[tid * Channels + channel] =
              colors[gaussian_id * Channels + channel];
        }
      }
    }
    __syncthreads();

    const int t_start = max(0, batch_end - warp_last_id);
    for (int t = t_start; t < batch_size; ++t) {
      const int current_position = batch_end - t;
      const int gaussian_id = batch_ids[t];
      bool valid =
          inside && gaussian_id >= 0 && current_position <= pixel_last_id;
      float visibility = 0.0f;
      float alpha = 0.0f;
      float2 delta = {0.0f, 0.0f};
      float3 selected_conic = {0.0f, 0.0f, 0.0f};
      if (valid) {
        selected_conic = batch_conics[t];
        delta = {px - batch_means[t].x, py - batch_means[t].y};
        valid = evaluate_weight(
            selected_conic, batch_means[t], batch_opacities[t], px, py,
            alpha_threshold, &visibility, &alpha);
      }
      const unsigned valid_mask = __ballot_sync(0xffffffffu, valid);
      if (valid_mask == 0) continue;

      float color_local[Channels] = {};
      float means_x_local = 0.0f;
      float means_y_local = 0.0f;
      float conic_x_local = 0.0f;
      float conic_xy_local = 0.0f;
      float conic_y_local = 0.0f;
      float opacity_local = 0.0f;
      if (valid) {
        const float one_minus_alpha = 1.0f - alpha;
        const float transmittance_before =
            transmittance / fmaxf(kMinOneMinusAlpha, one_minus_alpha);
        const float weight = alpha * transmittance_before;
        float weight_cotangent = output_alpha_gradient;
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
          weight_cotangent +=
              render_gradient[channel] * batch_colors[t * Channels + channel];
          color_local[channel] = render_gradient[channel] * weight;
        }
        const float alpha_chain_cotangent =
            weight_cotangent * transmittance_before -
            trailing_cotangent / one_minus_alpha;
        const float raw_alpha = batch_opacities[t] * visibility;
        float clamp_cotangent = 0.0f;
        if (raw_alpha < kMaxAlpha) clamp_cotangent = 1.0f;
        else if (raw_alpha == kMaxAlpha) clamp_cotangent = 0.5f;
        const float raw_alpha_cotangent =
            alpha_chain_cotangent * clamp_cotangent;
        if (clamp_cotangent != 0.0f) {
          opacity_local = raw_alpha_cotangent * visibility;
          const float sigma_cotangent = -raw_alpha_cotangent * raw_alpha;
          means_x_local = -sigma_cotangent *
              (selected_conic.x * delta.x + selected_conic.y * delta.y);
          means_y_local = -sigma_cotangent *
              (selected_conic.z * delta.y + selected_conic.y * delta.x);
          conic_x_local = sigma_cotangent * 0.5f * delta.x * delta.x;
          conic_xy_local = sigma_cotangent * delta.x * delta.y;
          conic_y_local = sigma_cotangent * 0.5f * delta.y * delta.y;
        }
        trailing_cotangent += weight_cotangent * weight;
        transmittance = transmittance_before;
      }

#pragma unroll
      for (int channel = 0; channel < Channels; ++channel) {
        color_local[channel] = warp_sum(color_local[channel]);
      }
      means_x_local = warp_sum(means_x_local);
      means_y_local = warp_sum(means_y_local);
      conic_x_local = warp_sum(conic_x_local);
      conic_xy_local = warp_sum(conic_xy_local);
      conic_y_local = warp_sum(conic_y_local);
      opacity_local = warp_sum(opacity_local);
      if ((tid & 31) == 0) {
#pragma unroll
        for (int channel = 0; channel < Channels; ++channel) {
          atomicAdd(colors_gradient + gaussian_id * Channels + channel,
                    color_local[channel]);
        }
        atomicAdd(means_gradient + gaussian_id * 2, means_x_local);
        atomicAdd(means_gradient + gaussian_id * 2 + 1, means_y_local);
        atomicAdd(conics_gradient + gaussian_id * 3, conic_x_local);
        atomicAdd(conics_gradient + gaussian_id * 3 + 1, conic_xy_local);
        atomicAdd(conics_gradient + gaussian_id * 3 + 2, conic_y_local);
        atomicAdd(opacities_gradient + gaussian_id, opacity_local);
      }
    }
    __syncthreads();
  }
}

template <int Channels>
ffi::Error launch_forward(
    cudaStream_t stream, int gaussian_count, int input_capacity,
    int image_width, int image_height, int tile_width, int tile_count,
    int per_tile_bound, float alpha_threshold,
    float transmittance_threshold, const float* means, const float* conics,
    const float* colors, const float* opacities, const int32_t* offsets,
    const int32_t* ids, const int32_t* valid_count, float* foreground,
    float* alpha, float* accepted_final_transmittance, int32_t* last_ids,
    bool* tile_overflow) {
  const size_t shared_bytes =
      kTilePixels * (sizeof(int32_t) + sizeof(float2) + sizeof(float3) +
                     sizeof(float) + sizeof(float) * Channels);
  compositor_forward_kernel<Channels><<<tile_count, kTilePixels, shared_bytes,
                                        stream>>>(
      gaussian_count, input_capacity, image_width, image_height, tile_width,
      tile_count, per_tile_bound, alpha_threshold, transmittance_threshold,
      means, conics, colors, opacities, offsets, ids, valid_count, foreground,
      alpha, accepted_final_transmittance, last_ids, tile_overflow);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return ffi::Error::Internal(std::string("CUDA forward launch failed: ") +
                                cudaGetErrorString(error));
  }
  return ffi::Error::Success();
}

template <int Channels>
ffi::Error launch_backward(
    cudaStream_t stream, int gaussian_count, int input_capacity,
    int image_width, int image_height, int tile_width, int tile_count,
    int per_tile_bound, float alpha_threshold,
    float transmittance_threshold, const float* means, const float* conics,
    const float* colors, const float* opacities, const int32_t* offsets,
    const int32_t* ids, const int32_t* valid_count,
    const float* accepted_final_transmittance, const int32_t* last_ids,
    const float* render_cotangent, const float* alpha_cotangent,
    float* means_gradient, float* conics_gradient, float* colors_gradient,
    float* opacities_gradient) {
  const size_t shared_bytes =
      kTilePixels * (sizeof(int32_t) + sizeof(float2) + sizeof(float3) +
                     sizeof(float) + sizeof(float) * Channels);
  compositor_backward_kernel<Channels><<<tile_count, kTilePixels, shared_bytes,
                                         stream>>>(
      gaussian_count, input_capacity, image_width, image_height, tile_width,
      tile_count, per_tile_bound, alpha_threshold, transmittance_threshold,
      means, conics, colors, opacities, offsets, ids, valid_count,
      accepted_final_transmittance, last_ids, render_cotangent,
      alpha_cotangent, means_gradient, conics_gradient, colors_gradient,
      opacities_gradient);
  const cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return ffi::Error::Internal(std::string("CUDA backward launch failed: ") +
                                cudaGetErrorString(error));
  }
  return ffi::Error::Success();
}

ffi::Error compositor_forward_host(
    cudaStream_t stream, ffi::BufferR2<ffi::F32> means,
    ffi::BufferR2<ffi::F32> conics, ffi::BufferR2<ffi::F32> colors,
    ffi::BufferR1<ffi::F32> opacities, ffi::BufferR1<ffi::S32> offsets,
    ffi::BufferR1<ffi::S32> ids, ffi::BufferR0<ffi::S32> valid_count,
    ffi::ResultBufferR3<ffi::F32> foreground,
    ffi::ResultBufferR2<ffi::F32> alpha,
    ffi::ResultBufferR2<ffi::F32> accepted_final_transmittance,
    ffi::ResultBufferR2<ffi::S32> last_ids,
    ffi::ResultBufferR1<ffi::PRED> tile_overflow,
    int64_t image_width, int64_t image_height, int64_t tile_width,
    int64_t per_tile_bound, float alpha_threshold,
    float transmittance_threshold) {
  const int gaussian_count = static_cast<int>(means.dimensions()[0]);
  const int channels = static_cast<int>(colors.dimensions()[1]);
  const int input_capacity = static_cast<int>(ids.dimensions()[0]);
  const int tile_count = static_cast<int>(offsets.dimensions()[0]);
#define JAX_GS_FORWARD_CASE(ChannelCount)                                      \
  case ChannelCount:                                                          \
    return launch_forward<ChannelCount>(                                      \
        stream, gaussian_count, input_capacity, image_width, image_height,     \
        tile_width, tile_count, per_tile_bound, alpha_threshold,              \
        transmittance_threshold, means.typed_data(), conics.typed_data(),      \
        colors.typed_data(), opacities.typed_data(), offsets.typed_data(),     \
        ids.typed_data(), valid_count.typed_data(),                            \
        foreground->typed_data(), alpha->typed_data(),                        \
        accepted_final_transmittance->typed_data(), last_ids->typed_data(),   \
        tile_overflow->typed_data())
  switch (channels) {
    JAX_GS_FORWARD_CASE(1);
    JAX_GS_FORWARD_CASE(2);
    JAX_GS_FORWARD_CASE(3);
    JAX_GS_FORWARD_CASE(4);
    JAX_GS_FORWARD_CASE(8);
    JAX_GS_FORWARD_CASE(16);
    JAX_GS_FORWARD_CASE(32);
    default:
      return ffi::Error::InvalidArgument("unsupported channel count");
  }
#undef JAX_GS_FORWARD_CASE
}

ffi::Error compositor_backward_host(
    cudaStream_t stream, ffi::BufferR2<ffi::F32> means,
    ffi::BufferR2<ffi::F32> conics, ffi::BufferR2<ffi::F32> colors,
    ffi::BufferR1<ffi::F32> opacities, ffi::BufferR1<ffi::S32> offsets,
    ffi::BufferR1<ffi::S32> ids, ffi::BufferR0<ffi::S32> valid_count,
    ffi::BufferR2<ffi::F32> accepted_final_transmittance,
    ffi::BufferR2<ffi::S32> last_ids,
    ffi::BufferR3<ffi::F32> render_cotangent,
    ffi::BufferR2<ffi::F32> alpha_cotangent,
    ffi::ResultBufferR2<ffi::F32> means_gradient,
    ffi::ResultBufferR2<ffi::F32> conics_gradient,
    ffi::ResultBufferR2<ffi::F32> colors_gradient,
    ffi::ResultBufferR1<ffi::F32> opacities_gradient,
    int64_t image_width, int64_t image_height, int64_t tile_width,
    int64_t per_tile_bound, float alpha_threshold,
    float transmittance_threshold) {
  const int gaussian_count = static_cast<int>(means.dimensions()[0]);
  const int channels = static_cast<int>(colors.dimensions()[1]);
  const int input_capacity = static_cast<int>(ids.dimensions()[0]);
  const int tile_count = static_cast<int>(offsets.dimensions()[0]);
  cudaMemsetAsync(means_gradient->typed_data(), 0,
                  means_gradient->size_bytes(), stream);
  cudaMemsetAsync(conics_gradient->typed_data(), 0,
                  conics_gradient->size_bytes(), stream);
  cudaMemsetAsync(colors_gradient->typed_data(), 0,
                  colors_gradient->size_bytes(), stream);
  cudaMemsetAsync(opacities_gradient->typed_data(), 0,
                  opacities_gradient->size_bytes(), stream);
#define JAX_GS_BACKWARD_CASE(ChannelCount)                                     \
  case ChannelCount:                                                          \
    return launch_backward<ChannelCount>(                                     \
        stream, gaussian_count, input_capacity, image_width, image_height,     \
        tile_width, tile_count, per_tile_bound, alpha_threshold,              \
        transmittance_threshold, means.typed_data(), conics.typed_data(),      \
        colors.typed_data(), opacities.typed_data(), offsets.typed_data(),     \
        ids.typed_data(), valid_count.typed_data(),                            \
        accepted_final_transmittance.typed_data(), last_ids.typed_data(),     \
        render_cotangent.typed_data(), alpha_cotangent.typed_data(),          \
        means_gradient->typed_data(), conics_gradient->typed_data(),          \
        colors_gradient->typed_data(), opacities_gradient->typed_data())
  switch (channels) {
    JAX_GS_BACKWARD_CASE(1);
    JAX_GS_BACKWARD_CASE(2);
    JAX_GS_BACKWARD_CASE(3);
    JAX_GS_BACKWARD_CASE(4);
    JAX_GS_BACKWARD_CASE(8);
    JAX_GS_BACKWARD_CASE(16);
    JAX_GS_BACKWARD_CASE(32);
    default:
      return ffi::Error::InvalidArgument("unsupported channel count");
  }
#undef JAX_GS_BACKWARD_CASE
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxGsCompositorForward, compositor_forward_host,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::BufferR2<ffi::F32>>()
        .Arg<ffi::BufferR2<ffi::F32>>()
        .Arg<ffi::BufferR2<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR3<ffi::F32>>()
        .Ret<ffi::BufferR2<ffi::F32>>()
        .Ret<ffi::BufferR2<ffi::F32>>()
        .Ret<ffi::BufferR2<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::PRED>>()
        .Attr<int64_t>("image_width")
        .Attr<int64_t>("image_height")
        .Attr<int64_t>("tile_width")
        .Attr<int64_t>("per_tile_bound")
        .Attr<float>("alpha_threshold")
        .Attr<float>("transmittance_threshold"),
    {xla::ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxGsCompositorBackward, compositor_backward_host,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::BufferR2<ffi::F32>>()
        .Arg<ffi::BufferR2<ffi::F32>>()
        .Arg<ffi::BufferR2<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR0<ffi::S32>>()
        .Arg<ffi::BufferR2<ffi::F32>>()
        .Arg<ffi::BufferR2<ffi::S32>>()
        .Arg<ffi::BufferR3<ffi::F32>>()
        .Arg<ffi::BufferR2<ffi::F32>>()
        .Ret<ffi::BufferR2<ffi::F32>>()
        .Ret<ffi::BufferR2<ffi::F32>>()
        .Ret<ffi::BufferR2<ffi::F32>>()
        .Ret<ffi::BufferR1<ffi::F32>>()
        .Attr<int64_t>("image_width")
        .Attr<int64_t>("image_height")
        .Attr<int64_t>("tile_width")
        .Attr<int64_t>("per_tile_bound")
        .Attr<float>("alpha_threshold")
        .Attr<float>("transmittance_threshold"),
    {xla::ffi::Traits::kCmdBufferCompatible});
