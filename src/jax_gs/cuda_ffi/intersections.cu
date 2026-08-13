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

namespace {

constexpr int32_t kCountLimit = (1 << 30) - 1;
constexpr int kThreads = 256;

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
                                         const int32_t* valid_count,
                                         int capacity, int tile_count,
                                         int32_t* offsets) {
  const int tile_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (tile_id >= tile_count) {
    return;
  }
  int low = 0;
  int high = valid_count[0] < 0
                 ? 0
                 : (valid_count[0] > capacity ? capacity : valid_count[0]);
  const uint64_t target = static_cast<uint64_t>(tile_id) << 32;
  while (low < high) {
    const int middle = low + (high - low) / 2;
    if (keys[middle] < target) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  offsets[tile_id] = low;
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

  const int offset_blocks = (tile_count + kThreads - 1) / kThreads;
  find_tile_offsets_kernel<<<offset_blocks, kThreads, 0, stream>>>(
      keys_out, effective_valid_count->typed_data(), capacity, tile_count,
      offsets->typed_data());
  if (error = cudaGetLastError(); error != cudaSuccess) {
    return cuda_error("CUDA intersection offset launch failed", error);
  }
  return ffi::Error::Success();
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
