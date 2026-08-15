/*
 * SPDX-FileCopyrightText: Copyright 2025 the Regents of the University of
 * California, Nerfstudio Team and contributors. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CUDA/XLA FFI fixed-capacity intersection prefix and topology stages. The
 * ordering and offset construction follow the public gsplat intersection
 * architecture. This file contains no PyTorch/ATen dependencies.
 */

#include <cuda_runtime.h>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_scan.cuh>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

extern "C" cudaError_t JaxGsLaunchCompositorForward(
    cudaStream_t stream, int channels, int gaussian_count, int input_capacity,
    int image_width, int image_height, int tile_width, int tile_count,
    int per_tile_bound, float alpha_threshold,
    float transmittance_threshold, const float* means, const float* conics,
    const float* colors, const float* opacities, const int32_t* offsets,
    const int32_t* ids, const int32_t* valid_count, float* foreground,
    float* alpha, float* accepted_final_transmittance, int32_t* last_ids,
    bool* tile_overflow);

namespace {

constexpr int32_t kCountLimit = (1 << 30) - 1;
constexpr int kThreads = 256;
constexpr int kAccuTileThreads = 128;

ffi::Error cuda_error(const char* operation, cudaError_t error) {
  if (error == cudaSuccess) {
    return ffi::Error::Success();
  }
  return ffi::Error::Internal(std::string(operation) + ": " +
                              cudaGetErrorString(error));
}

struct SaturatingAdd {
  __host__ __device__ int32_t operator()(int32_t left, int32_t right) const {
    const int64_t sum = static_cast<int64_t>(left) + right;
    return sum >= kCountLimit ? kCountLimit : static_cast<int32_t>(sum);
  }
};

__global__ void clamp_counts_kernel(const int32_t* counts, int32_t* cumulative,
                                    int count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    cumulative[index] = max(0, min(counts[index], kCountLimit));
  }
}

__global__ void finalize_prefix_kernel(int32_t* cumulative, int count,
                                       int32_t capacity,
                                       int32_t* valid_count, bool* overflow,
                                       int32_t* required_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int32_t required = cumulative[count - 1];
  if (index < count) {
    cumulative[index] = min(cumulative[index], capacity + 1);
  }
  if (index == 0) {
    required_count[0] = required;
    valid_count[0] = min(required, capacity);
    overflow[0] = required > capacity;
  }
}

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
  // JAX orders -0.0 and +0.0 equally and resolves that tie with Gaussian ID.
  // Canonicalizing zero keeps the stable CUB sort equivalent to that contract.
  if (value == 0.0f) {
    value = 0.0f;
  }
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) != 0u ? ~bits : bits ^ 0x80000000u;
}

__global__ void prepare_sort_keys_kernel(
    const int32_t* gaussian_ids, const int32_t* tile_ids, const float* depths,
    const int32_t* valid_count, int capacity, int gaussian_count,
    int tile_count, uint64_t* keys) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= capacity) {
    return;
  }
  const int count = valid_count[0] < 0
                        ? 0
                        : (valid_count[0] > capacity ? capacity
                                                     : valid_count[0]);
  if (index >= count) {
    keys[index] = UINT64_MAX;
    return;
  }
  const int32_t gaussian_id = gaussian_ids[index];
  const int32_t tile_id = tile_ids[index];
  if (gaussian_id < 0 || gaussian_id >= gaussian_count || tile_id < 0 ||
      tile_id >= tile_count) {
    keys[index] = UINT64_MAX;
    return;
  }
  const float depth = depths[gaussian_id];
  if (!isfinite(depth)) {
    keys[index] = UINT64_MAX;
    return;
  }
  keys[index] = (static_cast<uint64_t>(static_cast<uint32_t>(tile_id)) << 32) |
                ordered_float_bits(depth);
}

__global__ void accutile_count_kernel(
    const bool* valid, const float* b, const float* disc, const float* t,
    const float* p_u, const float* p_v, const float* coefficient,
    const int32_t* outer_min, const int32_t* outer_max,
    const int32_t* cross_min, const int32_t* cross_max,
    const float* outer_bbox_min, const float* outer_bbox_max,
    const float* cross_bbox_min, const float* cross_bbox_max,
    const float* argmin_outer, const float* argmax_outer, int count,
    int tile_size, int32_t* counts) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  if (!valid[index]) {
    counts[index] = 0;
    return;
  }

  const float block = static_cast<float>(tile_size);
  const float b_value = b[index];
  const float disc_value = disc[index];
  const float t_value = t[index];
  const float p_u_value = p_u[index];
  const float p_v_value = p_v[index];
  const float coefficient_value = coefficient[index];
  const int32_t outer_min_value = outer_min[index];
  const int32_t outer_max_value = outer_max[index];
  const int32_t cross_min_value = cross_min[index];
  const int32_t cross_max_value = cross_max[index];
  const float outer_bbox_min_value = outer_bbox_min[index];
  const float outer_bbox_max_value = outer_bbox_max[index];
  const float cross_bbox_min_value = cross_bbox_min[index];
  const float cross_bbox_max_value = cross_bbox_max[index];
  const float argmin_outer_value = argmin_outer[index];
  const float argmax_outer_value = argmax_outer[index];

  float min_line = static_cast<float>(outer_min_value) * block;
  float h = min_line - p_u_value;
  float radicand =
      max(disc_value * h * h + t_value * coefficient_value, 0.0f);
  float root = radicand > 0.0f ? radicand * rsqrtf(radicand) : 0.0f;
  float line_min =
      __fdividef(-b_value * h - root, coefficient_value) + p_v_value;
  float line_max =
      __fdividef(-b_value * h + root, coefficient_value) + p_v_value;
  const bool intersects = outer_bbox_min_value <= min_line;
  float previous_min = intersects ? line_min : cross_bbox_max_value;
  float previous_max = intersects ? line_max : cross_bbox_min_value;
  int32_t total = 0;

  for (int32_t outer = outer_min_value; outer < outer_max_value; ++outer) {
    min_line = static_cast<float>(outer) * block;
    const float max_line = min_line + block;
    h = max_line - p_u_value;
    radicand = max(disc_value * h * h + t_value * coefficient_value, 0.0f);
    root = radicand > 0.0f ? radicand * rsqrtf(radicand) : 0.0f;
    line_min =
        __fdividef(-b_value * h - root, coefficient_value) + p_v_value;
    line_max =
        __fdividef(-b_value * h + root, coefficient_value) + p_v_value;
    const bool current_intersects = max_line <= outer_bbox_max_value;
    const float current_min = current_intersects ? line_min : previous_min;
    const float current_max = current_intersects ? line_max : previous_max;
    const float ellipse_min =
        min_line <= argmin_outer_value && argmin_outer_value < max_line
            ? cross_bbox_min_value
            : min(previous_min, current_min);
    const float ellipse_max =
        min_line <= argmax_outer_value && argmax_outer_value < max_line
            ? cross_bbox_max_value
            : max(previous_max, current_max);
    const int32_t min_v = max(
        cross_min_value,
        min(cross_max_value,
            static_cast<int32_t>(__fdividef(ellipse_min, block))));
    const int32_t max_v = min(
        cross_max_value,
        max(cross_min_value,
            static_cast<int32_t>(__fdividef(ellipse_max, block) + 1.0f)));
    total += max_v - min_v;
    previous_min = current_min;
    previous_max = current_max;
  }
  counts[index] = max(total, 0);
}

__global__ void accutile_emit_keys_kernel(
    const bool* valid, const bool* is_y, const float* b, const float* disc,
    const float* t, const float* p_u, const float* p_v,
    const float* coefficient, const int32_t* outer_min,
    const int32_t* outer_max, const int32_t* cross_min,
    const int32_t* cross_max, const float* outer_bbox_min,
    const float* outer_bbox_max, const float* cross_bbox_min,
    const float* cross_bbox_max, const float* argmin_outer,
    const float* argmax_outer, const int32_t* cumulative,
    const int32_t* valid_count, const float* depths, int gaussian_count,
    int capacity, int tile_size, int tile_width, uint64_t* keys,
    int32_t* gaussian_ids) {
  const int rank = blockIdx.x * blockDim.x + threadIdx.x;
  if (rank >= capacity) return;
  const int count = max(0, min(valid_count[0], capacity));
  if (rank >= count) {
    keys[rank] = UINT64_MAX;
    gaussian_ids[rank] = -1;
    return;
  }

  int low = 0;
  int high = gaussian_count;
  while (low < high) {
    const int middle = low + (high - low) / 2;
    if (cumulative[middle] <= rank) low = middle + 1;
    else high = middle;
  }
  const int owner = min(low, gaussian_count - 1);
  const int32_t start = owner > 0 ? cumulative[owner - 1] : 0;
  const int32_t local_rank = rank - start;
  if (!valid[owner]) {
    keys[rank] = UINT64_MAX;
    gaussian_ids[rank] = -1;
    return;
  }

  const float block = static_cast<float>(tile_size);
  const float b_value = b[owner];
  const float disc_value = disc[owner];
  const float t_value = t[owner];
  const float p_u_value = p_u[owner];
  const float p_v_value = p_v[owner];
  const float coefficient_value = coefficient[owner];
  const int32_t outer_min_value = outer_min[owner];
  const int32_t outer_max_value = outer_max[owner];
  const int32_t cross_min_value = cross_min[owner];
  const int32_t cross_max_value = cross_max[owner];
  const float outer_bbox_min_value = outer_bbox_min[owner];
  const float outer_bbox_max_value = outer_bbox_max[owner];
  const float cross_bbox_min_value = cross_bbox_min[owner];
  const float cross_bbox_max_value = cross_bbox_max[owner];
  const float argmin_outer_value = argmin_outer[owner];
  const float argmax_outer_value = argmax_outer[owner];

  float min_line = static_cast<float>(outer_min_value) * block;
  float h = min_line - p_u_value;
  float radicand =
      max(disc_value * h * h + t_value * coefficient_value, 0.0f);
  float root = radicand > 0.0f ? radicand * rsqrtf(radicand) : 0.0f;
  float line_min =
      __fdividef(-b_value * h - root, coefficient_value) + p_v_value;
  float line_max =
      __fdividef(-b_value * h + root, coefficient_value) + p_v_value;
  const bool intersects = outer_bbox_min_value <= min_line;
  float previous_min = intersects ? line_min : cross_bbox_max_value;
  float previous_max = intersects ? line_max : cross_bbox_min_value;
  int32_t emitted = 0;
  int32_t selected_cross = 0;
  int32_t selected_outer = outer_min_value;
  bool found = false;

  for (int32_t outer = outer_min_value; outer < outer_max_value; ++outer) {
    min_line = static_cast<float>(outer) * block;
    const float max_line = min_line + block;
    h = max_line - p_u_value;
    radicand = max(disc_value * h * h + t_value * coefficient_value, 0.0f);
    root = radicand > 0.0f ? radicand * rsqrtf(radicand) : 0.0f;
    line_min =
        __fdividef(-b_value * h - root, coefficient_value) + p_v_value;
    line_max =
        __fdividef(-b_value * h + root, coefficient_value) + p_v_value;
    const bool current_intersects = max_line <= outer_bbox_max_value;
    const float current_min = current_intersects ? line_min : previous_min;
    const float current_max = current_intersects ? line_max : previous_max;
    const float ellipse_min =
        min_line <= argmin_outer_value && argmin_outer_value < max_line
            ? cross_bbox_min_value
            : min(previous_min, current_min);
    const float ellipse_max =
        min_line <= argmax_outer_value && argmax_outer_value < max_line
            ? cross_bbox_max_value
            : max(previous_max, current_max);
    const int32_t min_v = max(
        cross_min_value,
        min(cross_max_value,
            static_cast<int32_t>(__fdividef(ellipse_min, block))));
    const int32_t max_v = min(
        cross_max_value,
        max(cross_min_value,
            static_cast<int32_t>(__fdividef(ellipse_max, block) + 1.0f)));
    const int32_t span = max_v - min_v;
    if (!found && local_rank >= emitted && local_rank < emitted + span) {
      selected_cross = min_v + local_rank - emitted;
      selected_outer = outer;
      found = true;
    }
    emitted += span;
    previous_min = current_min;
    previous_max = current_max;
  }

  if (!found) {
    keys[rank] = UINT64_MAX;
    gaussian_ids[rank] = -1;
    return;
  }
  const int32_t tile_id =
      is_y[owner] ? selected_outer * tile_width + selected_cross
                  : selected_cross * tile_width + selected_outer;
  const float depth = depths[owner];
  if (!isfinite(depth)) {
    keys[rank] = UINT64_MAX;
    gaussian_ids[rank] = -1;
    return;
  }
  keys[rank] =
      (static_cast<uint64_t>(static_cast<uint32_t>(tile_id)) << 32) |
      ordered_float_bits(depth);
  gaussian_ids[rank] = owner;
}

__global__ void find_effective_valid_count_kernel(
    const uint64_t* keys, const int32_t* declared_valid_count, int capacity,
    int32_t* effective_valid_count) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int low = 0;
  int high = declared_valid_count[0] < 0
                 ? 0
                 : (declared_valid_count[0] > capacity
                        ? capacity
                        : declared_valid_count[0]);
  while (low < high) {
    const int middle = low + (high - low) / 2;
    if (keys[middle] < UINT64_MAX) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  effective_valid_count[0] = low;
}

__global__ void finalize_sorted_ids_kernel(const uint64_t* keys,
                                           const int32_t* valid_count,
                                           int capacity, int32_t* gaussian_ids,
                                           int32_t* tile_ids) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= capacity) {
    return;
  }
  const int count = valid_count[0] < 0
                        ? 0
                        : (valid_count[0] > capacity ? capacity
                                                     : valid_count[0]);
  const uint64_t key = keys[index];
  if (index < count && key != UINT64_MAX) {
    tile_ids[index] = static_cast<int32_t>(key >> 32);
  } else {
    gaussian_ids[index] = -1;
    tile_ids[index] = -1;
  }
}

__global__ void find_tile_offsets_kernel(const uint64_t* keys,
                                         const int32_t* valid_count_ptr,
                                         int capacity, int tile_count,
                                         int32_t* offsets) {
  const int count = valid_count_ptr[0] < 0
                        ? 0
                        : (valid_count_ptr[0] > capacity ? capacity
                                                         : valid_count_ptr[0]);
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (count <= 0) {
    if (idx < tile_count) {
      offsets[idx] = 0;
    }
    return;
  }
  if (idx >= count) {
    return;
  }

  const int32_t tile_curr =
      min(static_cast<int32_t>(keys[idx] >> 32), tile_count);

  if (idx == 0) {
    for (int32_t i = 0; i <= tile_curr && i < tile_count; ++i) {
      offsets[i] = 0;
    }
  }
  if (idx == count - 1) {
    for (int32_t i = tile_curr + 1; i < tile_count; ++i) {
      offsets[i] = count;
    }
  }
  if (idx > 0) {
    const int32_t tile_prev =
        min(static_cast<int32_t>(keys[idx - 1] >> 32), tile_count);
    if (tile_prev != tile_curr) {
      for (int32_t i = tile_prev + 1; i <= tile_curr && i < tile_count; ++i) {
        offsets[i] = idx;
      }
    }
  }
}

int radix_end_bit(int tile_count) {
  int tile_bits = 0;
  uint32_t value = static_cast<uint32_t>(tile_count);
  do {
    ++tile_bits;
    value >>= 1;
  } while (value != 0u);
  return 32 + tile_bits;
}

uint64_t prefix_workspace_bytes(int count) {
  if (count <= 0) {
    return UINT64_MAX;
  }
  size_t temporary_bytes = 0;
  int32_t* pointer = nullptr;
  const cudaError_t error = cub::DeviceScan::InclusiveScan(
      nullptr, temporary_bytes, pointer, pointer, SaturatingAdd{}, count);
  return error == cudaSuccess ? static_cast<uint64_t>(temporary_bytes)
                              : UINT64_MAX;
}

uint64_t sort_workspace_bytes(int capacity, int tile_count) {
  if (capacity <= 0 || tile_count <= 0) {
    return UINT64_MAX;
  }
  size_t temporary_bytes = 0;
  uint64_t* keys = nullptr;
  int32_t* values = nullptr;
  const cudaError_t error = cub::DeviceRadixSort::SortPairs(
      nullptr, temporary_bytes, keys, keys, values, values, capacity, 0,
      radix_end_bit(tile_count));
  return error == cudaSuccess ? static_cast<uint64_t>(temporary_bytes)
                              : UINT64_MAX;
}

uint64_t pipeline_workspace_bytes(int count, int capacity, int tile_count) {
  const uint64_t prefix = prefix_workspace_bytes(count);
  const uint64_t sort = sort_workspace_bytes(capacity, tile_count);
  if (prefix == UINT64_MAX || sort == UINT64_MAX) return UINT64_MAX;
  return max(prefix, sort);
}

ffi::Error intersection_prefix_host(
    cudaStream_t stream, ffi::BufferR1<ffi::S32> counts,
    ffi::ResultBufferR1<ffi::S32> cumulative,
    ffi::ResultBufferR0<ffi::S32> valid_count,
    ffi::ResultBufferR0<ffi::PRED> overflow,
    ffi::ResultBufferR0<ffi::S32> required_count,
    ffi::ResultBufferR1<ffi::U8> workspace, int64_t capacity_attr) {
  const int64_t count64 = counts.dimensions()[0];
  if (count64 <= 0 || count64 > std::numeric_limits<int>::max()) {
    return ffi::Error::InvalidArgument(
        "intersection prefix requires a non-empty int32 count vector");
  }
  if (capacity_attr < 0 || capacity_attr > kCountLimit - 1) {
    return ffi::Error::InvalidArgument("invalid intersection capacity");
  }
  if (cumulative->dimensions()[0] != count64) {
    return ffi::Error::InvalidArgument(
        "intersection cumulative output must match the count vector");
  }
  const int count = static_cast<int>(count64);
  const int32_t capacity = static_cast<int32_t>(capacity_attr);
  const int blocks = (count + kThreads - 1) / kThreads;

  clamp_counts_kernel<<<blocks, kThreads, 0, stream>>>(
      counts.typed_data(), cumulative->typed_data(), count);
  if (cudaError_t error = cudaGetLastError(); error != cudaSuccess) {
    return cuda_error("CUDA intersection count clamp launch failed", error);
  }

  size_t temporary_bytes = 0;
  cudaError_t error = cub::DeviceScan::InclusiveScan(
      nullptr, temporary_bytes, cumulative->typed_data(),
      cumulative->typed_data(), SaturatingAdd{}, count, stream);
  if (error != cudaSuccess) {
    return cuda_error("CUB intersection prefix size query failed", error);
  }
  if (workspace->size_bytes() < temporary_bytes) {
    return ffi::Error::Internal(
        "CUDA intersection prefix workspace is too small");
  }
  error = cub::DeviceScan::InclusiveScan(
      workspace->typed_data(), temporary_bytes, cumulative->typed_data(),
      cumulative->typed_data(), SaturatingAdd{}, count, stream);
  if (error != cudaSuccess) {
    return cuda_error("CUB intersection prefix failed", error);
  }

  finalize_prefix_kernel<<<blocks, kThreads, 0, stream>>>(
      cumulative->typed_data(), count, capacity, valid_count->typed_data(),
      overflow->typed_data(), required_count->typed_data());
  if (error = cudaGetLastError(); error != cudaSuccess) {
    return cuda_error("CUDA intersection prefix finalization launch failed",
                      error);
  }
  return ffi::Error::Success();
}

ffi::Error intersection_sort_offsets_host(
    cudaStream_t stream,
    ffi::BufferR1<ffi::S32> gaussian_ids,
    ffi::BufferR1<ffi::S32> tile_ids, ffi::BufferR1<ffi::F32> depths,
    ffi::BufferR0<ffi::S32> valid_count,
    ffi::ResultBufferR1<ffi::S32> sorted_gaussian_ids,
    ffi::ResultBufferR1<ffi::S32> sorted_tile_ids,
    ffi::ResultBufferR1<ffi::S32> offsets,
    ffi::ResultBufferR0<ffi::S32> effective_valid_count,
    ffi::ResultBufferR1<ffi::U8> key_workspace,
    ffi::ResultBufferR1<ffi::U8> cub_workspace, int64_t tile_count_attr) {
  const int64_t capacity64 = gaussian_ids.dimensions()[0];
  const int64_t gaussian_count64 = depths.dimensions()[0];
  if (capacity64 <= 0 || capacity64 > kCountLimit - 1 ||
      tile_ids.dimensions()[0] != capacity64 ||
      sorted_gaussian_ids->dimensions()[0] != capacity64 ||
      sorted_tile_ids->dimensions()[0] != capacity64) {
    return ffi::Error::InvalidArgument(
        "intersection sort requires equal non-empty fixed-capacity ID vectors");
  }
  if (gaussian_count64 <= 0 ||
      gaussian_count64 > std::numeric_limits<int>::max()) {
    return ffi::Error::InvalidArgument(
        "intersection sort requires a non-empty depth vector");
  }
  if (tile_count_attr <= 0 || tile_count_attr > kCountLimit - 1 ||
      offsets->dimensions()[0] != tile_count_attr) {
    return ffi::Error::InvalidArgument("invalid intersection tile count");
  }
  const int capacity = static_cast<int>(capacity64);
  const int gaussian_count = static_cast<int>(gaussian_count64);
  const int tile_count = static_cast<int>(tile_count_attr);
  const int blocks = (capacity + kThreads - 1) / kThreads;

  const size_t one_key_buffer_bytes =
      static_cast<size_t>(capacity) * sizeof(uint64_t);
  if (key_workspace->size_bytes() < 2 * one_key_buffer_bytes) {
    return ffi::Error::Internal(
        "CUDA intersection key workspace is too small");
  }
  auto* keys_in =
      reinterpret_cast<uint64_t*>(key_workspace->typed_data());
  auto* keys_out = reinterpret_cast<uint64_t*>(
      key_workspace->typed_data() + one_key_buffer_bytes);
  cudaError_t error = cudaSuccess;

  prepare_sort_keys_kernel<<<blocks, kThreads, 0, stream>>>(
      gaussian_ids.typed_data(), tile_ids.typed_data(), depths.typed_data(),
      valid_count.typed_data(), capacity, gaussian_count, tile_count, keys_in);
  if (error = cudaGetLastError(); error != cudaSuccess) {
    return cuda_error("CUDA intersection sort-key launch failed", error);
  }

  size_t temporary_bytes = 0;
  const int end_bit = radix_end_bit(tile_count);
  error = cub::DeviceRadixSort::SortPairs(
      nullptr, temporary_bytes, keys_in, keys_out, gaussian_ids.typed_data(),
      sorted_gaussian_ids->typed_data(), capacity, 0, end_bit, stream);
  if (error != cudaSuccess) {
    return cuda_error("CUB intersection sort size query failed", error);
  }
  if (cub_workspace->size_bytes() < temporary_bytes) {
    return ffi::Error::Internal(
        "CUB intersection sort workspace is too small");
  }
  error = cub::DeviceRadixSort::SortPairs(
      cub_workspace->typed_data(), temporary_bytes, keys_in, keys_out,
      gaussian_ids.typed_data(), sorted_gaussian_ids->typed_data(), capacity,
      0, end_bit, stream);
  if (error != cudaSuccess) {
    return cuda_error("CUB intersection sort failed", error);
  }

  find_effective_valid_count_kernel<<<1, 1, 0, stream>>>(
      keys_out, valid_count.typed_data(), capacity,
      effective_valid_count->typed_data());
  if (error = cudaGetLastError(); error != cudaSuccess) {
    return cuda_error(
        "CUDA intersection effective-count launch failed", error);
  }

  finalize_sorted_ids_kernel<<<blocks, kThreads, 0, stream>>>(
      keys_out, effective_valid_count->typed_data(), capacity,
      sorted_gaussian_ids->typed_data(), sorted_tile_ids->typed_data());
  if (error = cudaGetLastError(); error != cudaSuccess) {
    return cuda_error("CUDA intersection ID finalization launch failed", error);
  }

  const int offset_blocks =
      (max(capacity, tile_count) + kThreads - 1) / kThreads;
  find_tile_offsets_kernel<<<offset_blocks, kThreads, 0, stream>>>(
      keys_out, effective_valid_count->typed_data(), capacity, tile_count,
      offsets->typed_data());
  if (error = cudaGetLastError(); error != cudaSuccess) {
    return cuda_error("CUDA intersection offset launch failed", error);
  }
  return ffi::Error::Success();
}

ffi::Error intersection_compositor_pipeline_host(
    cudaStream_t stream, ffi::BufferR1<ffi::PRED> valid,
    ffi::BufferR1<ffi::PRED> is_y, ffi::BufferR1<ffi::F32> b,
    ffi::BufferR1<ffi::F32> disc, ffi::BufferR1<ffi::F32> t,
    ffi::BufferR1<ffi::F32> p_u, ffi::BufferR1<ffi::F32> p_v,
    ffi::BufferR1<ffi::F32> coefficient,
    ffi::BufferR1<ffi::S32> outer_min,
    ffi::BufferR1<ffi::S32> outer_max,
    ffi::BufferR1<ffi::S32> cross_min,
    ffi::BufferR1<ffi::S32> cross_max,
    ffi::BufferR1<ffi::F32> outer_bbox_min,
    ffi::BufferR1<ffi::F32> outer_bbox_max,
    ffi::BufferR1<ffi::F32> cross_bbox_min,
    ffi::BufferR1<ffi::F32> cross_bbox_max,
    ffi::BufferR1<ffi::F32> argmin_outer,
    ffi::BufferR1<ffi::F32> argmax_outer,
    ffi::BufferR1<ffi::F32> depths, ffi::BufferR2<ffi::F32> means,
    ffi::BufferR2<ffi::F32> conics, ffi::BufferR2<ffi::F32> colors,
    ffi::BufferR1<ffi::F32> opacities,
    ffi::ResultBufferR1<ffi::S32> sorted_gaussian_ids,
    ffi::ResultBufferR1<ffi::S32> sorted_tile_ids,
    ffi::ResultBufferR1<ffi::S32> offsets,
    ffi::ResultBufferR0<ffi::S32> valid_count,
    ffi::ResultBufferR0<ffi::PRED> overflow,
    ffi::ResultBufferR0<ffi::S32> required_count,
    ffi::ResultBufferR3<ffi::F32> foreground,
    ffi::ResultBufferR2<ffi::F32> alpha,
    ffi::ResultBufferR2<ffi::F32> accepted_final_transmittance,
    ffi::ResultBufferR2<ffi::S32> last_ids,
    ffi::ResultBufferR1<ffi::PRED> tile_overflow,
    ffi::ResultBufferR1<ffi::S32> cumulative,
    ffi::ResultBufferR1<ffi::U8> key_workspace,
    ffi::ResultBufferR1<ffi::U8> cub_workspace, int64_t capacity_attr,
    int64_t tile_size_attr, int64_t tile_width_attr,
    int64_t tile_height_attr, int64_t image_width_attr,
    int64_t image_height_attr, int64_t per_tile_bound_attr,
    float alpha_threshold, float transmittance_threshold) {
  const int64_t gaussian_count64 = valid.dimensions()[0];
  const int64_t tile_count64 = tile_width_attr * tile_height_attr;
  if (gaussian_count64 <= 0 ||
      gaussian_count64 > std::numeric_limits<int>::max() ||
      capacity_attr <= 0 || capacity_attr > kCountLimit - 1 ||
      tile_size_attr != 16 || tile_width_attr <= 0 || tile_height_attr <= 0 ||
      tile_width_attr > std::numeric_limits<int>::max() ||
      tile_height_attr > std::numeric_limits<int>::max() ||
      tile_count64 <= 0 || tile_count64 > kCountLimit - 1 ||
      image_width_attr <= 0 || image_height_attr <= 0 ||
      image_width_attr > std::numeric_limits<int>::max() ||
      image_height_attr > std::numeric_limits<int>::max() ||
      per_tile_bound_attr <= 0 ||
      per_tile_bound_attr > std::numeric_limits<int>::max()) {
    return ffi::Error::InvalidArgument(
        "invalid fused intersection/compositor dimensions");
  }
  const auto matches_count = [gaussian_count64](const auto& buffer) {
    return buffer.dimensions()[0] == gaussian_count64;
  };
  if (!matches_count(is_y) || !matches_count(b) || !matches_count(disc) ||
      !matches_count(t) || !matches_count(p_u) || !matches_count(p_v) ||
      !matches_count(coefficient) || !matches_count(outer_min) ||
      !matches_count(outer_max) || !matches_count(cross_min) ||
      !matches_count(cross_max) || !matches_count(outer_bbox_min) ||
      !matches_count(outer_bbox_max) || !matches_count(cross_bbox_min) ||
      !matches_count(cross_bbox_max) || !matches_count(argmin_outer) ||
      !matches_count(argmax_outer) || !matches_count(depths) ||
      !matches_count(means) || !matches_count(conics) ||
      !matches_count(colors) || !matches_count(opacities) ||
      cumulative->dimensions()[0] != gaussian_count64) {
    return ffi::Error::InvalidArgument(
        "fused intersection/compositor inputs must share Gaussian count");
  }
  const int64_t channels64 = colors.dimensions()[1];
  if (means.dimensions()[1] != 2 || conics.dimensions()[1] != 3 ||
      channels64 <= 0 ||
      sorted_gaussian_ids->dimensions()[0] != capacity_attr ||
      sorted_tile_ids->dimensions()[0] != capacity_attr ||
      offsets->dimensions()[0] != tile_count64 ||
      foreground->dimensions()[0] != image_height_attr ||
      foreground->dimensions()[1] != image_width_attr ||
      foreground->dimensions()[2] != channels64 ||
      alpha->dimensions()[0] != image_height_attr ||
      alpha->dimensions()[1] != image_width_attr ||
      accepted_final_transmittance->dimensions()[0] != image_height_attr ||
      accepted_final_transmittance->dimensions()[1] != image_width_attr ||
      last_ids->dimensions()[0] != image_height_attr ||
      last_ids->dimensions()[1] != image_width_attr ||
      tile_overflow->dimensions()[0] != tile_count64) {
    return ffi::Error::InvalidArgument(
        "invalid fused intersection/compositor buffer shape");
  }

  const int gaussian_count = static_cast<int>(gaussian_count64);
  const int capacity = static_cast<int>(capacity_attr);
  const int tile_size = static_cast<int>(tile_size_attr);
  const int tile_width = static_cast<int>(tile_width_attr);
  const int tile_count = static_cast<int>(tile_count64);
  const int channels = static_cast<int>(channels64);
  const size_t one_key_buffer_bytes =
      static_cast<size_t>(capacity) * sizeof(uint64_t);
  if (key_workspace->size_bytes() < 2 * one_key_buffer_bytes) {
    return ffi::Error::Internal(
        "CUDA fused pipeline key workspace is too small");
  }
  auto* keys_in =
      reinterpret_cast<uint64_t*>(key_workspace->typed_data());
  auto* keys_out = reinterpret_cast<uint64_t*>(
      key_workspace->typed_data() + one_key_buffer_bytes);
  const int count_blocks =
      (gaussian_count + kAccuTileThreads - 1) / kAccuTileThreads;
  accutile_count_kernel<<<count_blocks, kAccuTileThreads, 0, stream>>>(
      valid.typed_data(), b.typed_data(), disc.typed_data(), t.typed_data(),
      p_u.typed_data(), p_v.typed_data(), coefficient.typed_data(),
      outer_min.typed_data(), outer_max.typed_data(), cross_min.typed_data(),
      cross_max.typed_data(), outer_bbox_min.typed_data(),
      outer_bbox_max.typed_data(), cross_bbox_min.typed_data(),
      cross_bbox_max.typed_data(), argmin_outer.typed_data(),
      argmax_outer.typed_data(), gaussian_count, tile_size,
      cumulative->typed_data());
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    return cuda_error("CUDA fused intersection count launch failed", error);
  }

  size_t prefix_bytes = cub_workspace->size_bytes();
  error = cub::DeviceScan::InclusiveScan(
      cub_workspace->typed_data(), prefix_bytes, cumulative->typed_data(),
      cumulative->typed_data(), SaturatingAdd{}, gaussian_count, stream);
  if (error != cudaSuccess) {
    return cuda_error("CUB fused intersection prefix failed", error);
  }
  const int finalize_blocks =
      (gaussian_count + kThreads - 1) / kThreads;
  finalize_prefix_kernel<<<finalize_blocks, kThreads, 0, stream>>>(
      cumulative->typed_data(), gaussian_count, capacity,
      valid_count->typed_data(), overflow->typed_data(),
      required_count->typed_data());
  if (error = cudaGetLastError(); error != cudaSuccess) {
    return cuda_error("CUDA fused prefix finalization failed", error);
  }

  const int emit_blocks =
      (capacity + kAccuTileThreads - 1) / kAccuTileThreads;
  accutile_emit_keys_kernel<<<emit_blocks, kAccuTileThreads, 0, stream>>>(
      valid.typed_data(), is_y.typed_data(), b.typed_data(),
      disc.typed_data(), t.typed_data(), p_u.typed_data(), p_v.typed_data(),
      coefficient.typed_data(), outer_min.typed_data(),
      outer_max.typed_data(), cross_min.typed_data(), cross_max.typed_data(),
      outer_bbox_min.typed_data(), outer_bbox_max.typed_data(),
      cross_bbox_min.typed_data(), cross_bbox_max.typed_data(),
      argmin_outer.typed_data(), argmax_outer.typed_data(),
      cumulative->typed_data(), valid_count->typed_data(), depths.typed_data(),
      gaussian_count, capacity, tile_size, tile_width, keys_in,
      sorted_tile_ids->typed_data());
  if (error = cudaGetLastError(); error != cudaSuccess) {
    return cuda_error("CUDA fused intersection emit failed", error);
  }

  size_t sort_bytes = cub_workspace->size_bytes();
  const int end_bit = radix_end_bit(tile_count);
  error = cub::DeviceRadixSort::SortPairs(
      cub_workspace->typed_data(), sort_bytes, keys_in, keys_out,
      sorted_tile_ids->typed_data(), sorted_gaussian_ids->typed_data(),
      capacity, 0, end_bit, stream);
  if (error != cudaSuccess) {
    return cuda_error("CUB fused intersection sort failed", error);
  }

  find_effective_valid_count_kernel<<<1, 1, 0, stream>>>(
      keys_out, valid_count->typed_data(), capacity,
      valid_count->typed_data());
  if (error = cudaGetLastError(); error != cudaSuccess) {
    return cuda_error("CUDA fused effective-count launch failed", error);
  }
  const int capacity_blocks = (capacity + kThreads - 1) / kThreads;
  finalize_sorted_ids_kernel<<<capacity_blocks, kThreads, 0, stream>>>(
      keys_out, valid_count->typed_data(), capacity,
      sorted_gaussian_ids->typed_data(), sorted_tile_ids->typed_data());
  if (error = cudaGetLastError(); error != cudaSuccess) {
    return cuda_error("CUDA fused ID finalization launch failed", error);
  }
  const int offset_blocks =
      (max(capacity, tile_count) + kThreads - 1) / kThreads;
  find_tile_offsets_kernel<<<offset_blocks, kThreads, 0, stream>>>(
      keys_out, valid_count->typed_data(), capacity, tile_count,
      offsets->typed_data());
  if (error = cudaGetLastError(); error != cudaSuccess) {
    return cuda_error("CUDA fused topology finalization failed", error);
  }

  error = JaxGsLaunchCompositorForward(
      stream, channels, gaussian_count, capacity,
      static_cast<int>(image_width_attr), static_cast<int>(image_height_attr),
      tile_width, tile_count, static_cast<int>(per_tile_bound_attr),
      alpha_threshold, transmittance_threshold, means.typed_data(),
      conics.typed_data(), colors.typed_data(), opacities.typed_data(),
      offsets->typed_data(), sorted_gaussian_ids->typed_data(),
      valid_count->typed_data(), foreground->typed_data(), alpha->typed_data(),
      accepted_final_transmittance->typed_data(), last_ids->typed_data(),
      tile_overflow->typed_data());
  return cuda_error("CUDA fused compositor forward launch failed", error);
}

}  // namespace

extern "C" uint64_t JaxGsIntersectionPrefixWorkspaceBytes(int64_t count) {
  if (count > std::numeric_limits<int>::max()) {
    return UINT64_MAX;
  }
  return prefix_workspace_bytes(static_cast<int>(count));
}

extern "C" uint64_t JaxGsIntersectionSortWorkspaceBytes(
    int64_t capacity, int64_t tile_count) {
  if (capacity > std::numeric_limits<int>::max() ||
      tile_count > std::numeric_limits<int>::max()) {
    return UINT64_MAX;
  }
  return sort_workspace_bytes(static_cast<int>(capacity),
                              static_cast<int>(tile_count));
}

extern "C" uint64_t JaxGsIntersectionPipelineWorkspaceBytes(
    int64_t count, int64_t capacity, int64_t tile_count) {
  if (count > std::numeric_limits<int>::max() ||
      capacity > std::numeric_limits<int>::max() ||
      tile_count > std::numeric_limits<int>::max()) {
    return UINT64_MAX;
  }
  return pipeline_workspace_bytes(static_cast<int>(count),
                                  static_cast<int>(capacity),
                                  static_cast<int>(tile_count));
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxGsIntersectionPrefix, intersection_prefix_host,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR0<ffi::PRED>>()
        .Ret<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::U8>>()
        .Attr<int64_t>("capacity"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxGsIntersectionSortOffsets, intersection_sort_offsets_host,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::U8>>()
        .Ret<ffi::BufferR1<ffi::U8>>()
        .Attr<int64_t>("tile_count"));

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    JaxGsIntersectionCompositorPipeline,
    intersection_compositor_pipeline_host,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::BufferR1<ffi::PRED>>()
        .Arg<ffi::BufferR1<ffi::PRED>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::S32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Arg<ffi::BufferR2<ffi::F32>>()
        .Arg<ffi::BufferR2<ffi::F32>>()
        .Arg<ffi::BufferR2<ffi::F32>>()
        .Arg<ffi::BufferR1<ffi::F32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR0<ffi::PRED>>()
        .Ret<ffi::BufferR0<ffi::S32>>()
        .Ret<ffi::BufferR3<ffi::F32>>()
        .Ret<ffi::BufferR2<ffi::F32>>()
        .Ret<ffi::BufferR2<ffi::F32>>()
        .Ret<ffi::BufferR2<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::PRED>>()
        .Ret<ffi::BufferR1<ffi::S32>>()
        .Ret<ffi::BufferR1<ffi::U8>>()
        .Ret<ffi::BufferR1<ffi::U8>>()
        .Attr<int64_t>("capacity")
        .Attr<int64_t>("tile_size")
        .Attr<int64_t>("tile_width")
        .Attr<int64_t>("tile_height")
        .Attr<int64_t>("image_width")
        .Attr<int64_t>("image_height")
        .Attr<int64_t>("per_tile_bound")
        .Attr<float>("alpha_threshold")
        .Attr<float>("transmittance_threshold"));
