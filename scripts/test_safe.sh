#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-/tmp/jax-gs-pycache-${UID}-$$}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export XLA_FLAGS="--xla_gpu_force_compilation_parallelism=1"
export MAX_JOBS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1

gpu_preflight() {
  local compiler_processes d_state_threads compute_processes available_kib

  compiler_processes="$({
    ps -eo pid=,comm=,args= | awk '
      $2 ~ /^(nvcc|cicc|cudafe\+\+|ptxas|ninja|sd-rmrf)$/ { print }
    '
  } || true)"
  if [[ -n "${compiler_processes}" ]]; then
    echo "Refusing GPU tests: CUDA/build processes are still present:" >&2
    echo "${compiler_processes}" >&2
    return 1
  fi

  d_state_threads="$({
    ps -eLo pid=,tid=,stat=,comm=,wchan= | awk '$3 ~ /^D/ { print }'
  } || true)"
  if [[ -n "${d_state_threads}" ]]; then
    if [[ "${ALLOW_D_STATE_GPU_TESTS:-0}" != "1" ]]; then
      echo "Refusing GPU tests: uninterruptible D-state threads are present:" >&2
      echo "${d_state_threads}" >&2
      echo "Set ALLOW_D_STATE_GPU_TESTS=1 only after explicitly accepting this risk." >&2
      return 1
    fi
    echo "Warning: proceeding with explicitly allowed D-state threads:" >&2
    echo "${d_state_threads}" >&2
  fi

  if ! compute_processes="$(
    timeout --kill-after=2s 10s nvidia-smi \
      --query-compute-apps=pid,process_name,used_memory \
      --format=csv,noheader,nounits
  )"; then
    echo "Refusing GPU tests: nvidia-smi failed" >&2
    return 1
  fi
  if [[ -n "${compute_processes}" ]]; then
    echo "Refusing GPU tests: NVIDIA compute processes are already active:" >&2
    echo "${compute_processes}" >&2
    return 1
  fi

  available_kib="$(awk '/MemAvailable:/ { print $2 }' /proc/meminfo)"
  if (( available_kib < 16 * 1024 * 1024 )); then
    echo "Refusing GPU tests: less than 16 GiB host memory is available" >&2
    return 1
  fi
}

acquire_gpu_lock() {
  local lock_path="${XDG_RUNTIME_DIR:-/tmp}/jax-gs-gpu-tests-${UID}.lock"
  exec 9>"${lock_path}"
  if ! flock -n 9; then
    echo "Refusing GPU tests: another safe GPU test run holds ${lock_path}" >&2
    return 1
  fi
}

gpu_postflight_on_exit() {
  local status=$?
  local postflight_status=0
  trap - EXIT
  if ! gpu_preflight; then
    postflight_status=1
  fi
  if (( status == 0 && postflight_status != 0 )); then
    status=${postflight_status}
  fi
  exit "${status}"
}

probe_gpu_device() {
  if ! timeout --kill-after=10s 30s uv run python - <<'PY'
import jax

if not any(device.platform == "gpu" for device in jax.devices()):
    raise SystemExit("RUN_GPU_TESTS=1 requires a CUDA JAX device")
print(jax.devices())
PY
  then
    echo "Refusing GPU tests: CUDA JAX device probe failed or timed out" >&2
    return 1
  fi
  gpu_preflight
}

run_gpu_test() {
  local node_id="$1"
  local marker_expression="${2:-not resource_heavy}"
  local test_status=0
  local postflight_status=0
  gpu_preflight
  if timeout --kill-after=10s "${GPU_TEST_TIMEOUT_SECONDS:-180}" \
    uv run pytest -q -o addopts="-ra" \
      -m "${marker_expression}" "${node_id}"; then
    test_status=0
  else
    test_status=$?
  fi
  if ! gpu_preflight; then
    postflight_status=1
  fi
  if (( test_status != 0 )); then
    return "${test_status}"
  fi
  return "${postflight_status}"
}

run_gpu_test_file_by_case() {
  local test_file="$1"
  local collected
  local node_id
  if ! collected="$(
    set -o pipefail
    JAX_PLATFORMS=cpu uv run pytest --collect-only -q \
      -m "not resource_heavy" "${test_file}" \
      | awk '/^tests\/.*::/ { print }'
  )"; then
    echo "Failed to collect GPU test cases from ${test_file}" >&2
    return 1
  fi
  if [[ -z "${collected}" ]]; then
    echo "No GPU test cases collected from ${test_file}" >&2
    return 1
  fi
  while IFS= read -r node_id; do
    run_gpu_test "${node_id}"
  done <<< "${collected}"
}

if [[ "${RUN_GPU_TESTS:-0}" == "1" ]]; then
  # Refuse immediately rather than completing the CPU suite and discovering
  # that the host was already unsafe for CUDA work.
  acquire_gpu_lock
  gpu_preflight
  trap gpu_postflight_on_exit EXIT
  probe_gpu_device
fi

for test_file in tests/test_*.py; do
  case "${test_file}" in
    tests/test_two_dgs.py|tests/test_two_dgs_low_level.py|tests/test_three_dgut_eval.py)
      continue
      ;;
  esac
  # A fresh process per file prevents JAX executable/compiler state from
  # accumulating until the CUDA/CPU compiler becomes unstable.
  JAX_PLATFORMS=cpu uv run pytest -q "${test_file}"
done
JAX_PLATFORMS=cpu uv run pytest -q -o addopts="-ra" \
  -m resource_heavy tests/test_two_dgs.py
JAX_PLATFORMS=cpu uv run pytest -q -o addopts="-ra" \
  -m resource_heavy tests/test_two_dgs_low_level.py
JAX_PLATFORMS=cpu uv run pytest -q -o addopts="-ra" \
  -m resource_heavy tests/test_three_dgut_eval.py
JAX_PLATFORMS=cpu uv run pytest -q -o addopts="-ra" \
  -m resource_heavy tests/test_sparse_rasterization.py
JAX_PLATFORMS=cpu uv run pytest -q -o addopts="-ra" \
  -m resource_heavy tests/test_visibility.py

if [[ "${RUN_GPU_TESTS:-0}" == "1" ]]; then
  # Every GPU case gets a fresh process. This prevents executable/compiler
  # state from accumulating inside pytest, which has produced pxla/ptxas
  # crashes on the local CUDA 13.3 + Blackwell stack.
  run_gpu_test_file_by_case tests/test_accutile_intersections.py
  run_gpu_test_file_by_case tests/test_rasterization_jax.py
  run_gpu_test_file_by_case tests/test_training.py
  run_gpu_test \
    tests/test_three_dgut.py::test_active_mask_is_jittable_with_static_chunked_shapes
  run_gpu_test \
    tests/test_three_dgut_eval.py::test_eval3d_custom_rays_hit_distance_and_normals \
    resource_heavy
  run_gpu_test \
    tests/test_three_dgut_eval.py::test_eval3d_extra_returns_last_ids_sample_counts_and_normals \
    resource_heavy
  run_gpu_test \
    'tests/test_sparse_rasterization.py::test_sparse_pixels_match_dense_gather_and_jit[False]' \
    resource_heavy
  run_gpu_test \
    tests/test_sparse_rasterization.py::test_sparse_pixels_gradients_match_dense_gather \
    resource_heavy
  run_gpu_test \
    tests/test_visibility.py::test_sparse_visibility_exact_weights_top_selection_and_padding
  if [[ "${RUN_RESOURCE_HEAVY_GPU_TESTS:-0}" == "1" ]]; then
    run_gpu_test \
      'tests/test_two_dgs.py::test_rasterization_2dgs_render_modes_and_auxiliary_maps[RGB-3-None]' \
      resource_heavy
    run_gpu_test \
      'tests/test_two_dgs.py::test_2dgs_absgrad_probe_sums_before_symmetric_pixel_cancellation[intersections]' \
      resource_heavy
    run_gpu_test \
      tests/test_two_dgs_low_level.py::test_low_level_2dgs_densify_probe_matches_ray_transform_vjp \
      resource_heavy
  fi
fi
