#!/usr/bin/env python
# coding: utf-8

# This file is part of pyDelPhi.
# Copyright (C) 2025 The pyDelPhi Project and contributors.
#
# pyDelPhi is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyDelPhi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with pyDelPhi. If not, see <https://www.gnu.org/licenses/>.

"""
pydelphi.space.core.vdw.process_bgp

Pure (no `self`) orchestrator for iterative boundary-grid-point (BGP) processing used in
VdW→MS surface construction.

Design goals
------------
- Pure orchestration: caller passes everything read; orchestrator returns everything updated.
- Backend-agnostic: identical wavefront loop for serial CPU / parallel CPU / CUDA backends.
- CUDA-friendly: orchestrator copies to device ONCE, launches per iteration, pulls back ONCE.

Backend contract
----------------
Each backend module must export:

    _process_bgp_one_iteration(
        *,
        boundary_point_start_index: int,
        boundary_point_end_index: int,
        num_external_boundary_points: int,
        boundary_grid_indices_1d,
        solute_bgp_type_1d,
        index_discrete_epsilon_map_1d,
        **backend_kwargs
    ) -> (
        exec_status: int,
        cycle_flag: bool,
        added_boundary_points_increment: int,
        removed_boundary_points_increment: int,
        num_external_boundary_points_new: int,
        boundary_grid_indices_1d,
        solute_bgp_type_1d,
        index_discrete_epsilon_map_1d,
    )

Notes:
- CPU backends take host arrays.
- CUDA backend takes device arrays; this orchestrator handles device staging + pullback.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any
import time
import os
import numpy as np

from pydelphi.config.global_runtime import vprint
from pydelphi.config.logging_config import DEBUG, get_effective_verbosity

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

BGPOrchestratorReturn = Tuple[
    int,  # exec_status
    np.ndarray,  # boundary_grid_indices_1d (HOST on return)
    np.ndarray,  # solute_bgp_type_1d (HOST on return)
    np.ndarray,  # index_discrete_epsilon_map_1d (HOST on return)
    int,  # num_discovered_bndy_grid_points (final)
    int,  # num_external_boundary_points (final)
    int,  # num_iterations
]

__all__ = [
    "process_bgp_orchestrate",
    "BGPOrchestratorReturn",
    "prepare_bgp_cuda_backend",
]


def _print_bgp_progress(
    iteration: int,
    added: int,
    removed: int,
    old_end: int,
    new_end: int,
    external: int,
    elapsed_s: Optional[float],
) -> None:
    if elapsed_s is None:
        vprint(
            DEBUG,
            _VERBOSITY,
            f"[BGP] iter={iteration:4d}  "
            f"old_total={old_end:11d}  "
            f"added={added:9d}  removed={removed:9d}  "
            f"new_total={new_end:11d}  "
            f"external={external:11d}",
        )
    else:
        vprint(
            DEBUG,
            _VERBOSITY,
            f"[BGP] iter={iteration:4d}  "
            f"old_total={old_end:11d}  "
            f"added={added:9d}  removed={removed:9d}  "
            f"new_total={new_end:11d}  "
            f"external={external:11d}  "
            f"dt={elapsed_s:7.3f}s",
        )


def _print_bgp_final(
    status: int,
    iters: int,
    total: int,
    external: int,
    reason: str,
) -> None:
    tag = "converged" if status == 0 else "aborted"
    vprint(
        DEBUG,
        _VERBOSITY,
        f"[BGP] {tag}: iters={iters}, total={total}, "
        f"external={external}, reason={reason}",
    )


def _fingerprint_array_debug(arr: np.ndarray):
    """
    Small deterministic debug fingerprint:
      - shape
      - dtype
      - size
      - int64 sum
      - sampled xor

    This is print-only/debug-only. Do not use for correctness.
    """
    a = np.asarray(arr)
    if a.size == 0:
        return {
            "shape": tuple(a.shape),
            "dtype": str(a.dtype),
            "size": 0,
            "sum": 0,
            "xor": 0,
        }

    v = a.astype(np.int64, copy=False).ravel()
    s = int(v.sum(dtype=np.int64))

    stride = max(1, v.size // 262144)
    sample = v[::stride]
    x = int(np.bitwise_xor.reduce(sample, dtype=np.int64))

    return {
        "shape": tuple(a.shape),
        "dtype": str(a.dtype),
        "size": int(a.size),
        "sum": s,
        "xor": x,
    }


def _print_pre_bgp_full_debug(
    *,
    label: str,
    epsilon_dimension: int,
    num_boundary_grid_points: int,
    num_external_boundary_points: int,
    boundary_grid_indices: np.ndarray,
    solute_bgp_type_1d: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
) -> None:
    """
    Host-side PRE-BGP full debug summary.

    Intended for serial-vs-parallel comparison before BGP iteration 1.
    Safe for CPU paths. For CUDA, call only before staging or after copy-back.
    """
    eps = np.asarray(index_discrete_epsilon_map_1d)
    sbt = np.asarray(solute_bgp_type_1d)
    bgi = np.asarray(boundary_grid_indices)

    epsdim = int(epsilon_dimension)

    # eps encoding:
    #   media  = eps // epsilon_dimension
    #   entity = eps %  epsilon_dimension
    # use abs for media/entity to match boundary-status classification behavior
    eps_abs = np.abs(eps.astype(np.int64, copy=False))
    media = eps_abs // epsdim
    entity = eps_abs % epsdim

    eps_min = int(eps.min()) if eps.size else 0
    eps_max = int(eps.max()) if eps.size else 0

    eps_zero_count = int(np.count_nonzero(eps == 0))
    entity_zero_count = int(np.count_nonzero(entity == 0))
    media_zero_count = int(np.count_nonzero(media == 0))

    sbt_vals, sbt_counts = np.unique(sbt, return_counts=True)
    sbt_counts_dict = {
        int(v): int(c) for v, c in zip(sbt_vals.tolist(), sbt_counts.tolist())
    }

    # Fingerprint full allocated BGI and active prefix separately.
    active_n = int(num_boundary_grid_points)
    bgi_active = bgi[:active_n]

    fp_eps = _fingerprint_array_debug(eps)
    fp_sbt = _fingerprint_array_debug(sbt)
    fp_bgi_all = _fingerprint_array_debug(bgi)
    fp_bgi_active = _fingerprint_array_debug(bgi_active)

    print(
        f"[PRE-BGP full] label={label} "
        f"num_boundary_grid_points={int(num_boundary_grid_points)} "
        f"num_external_boundary_points={int(num_external_boundary_points)} "
        f"epsilon_dimension={epsdim}"
    )
    print(
        f"[PRE-BGP full] label={label} "
        f"eps_min={eps_min} eps_max={eps_max} "
        f"eps_zero_count={eps_zero_count} "
        f"entity_zero_count={entity_zero_count} "
        f"media_zero_count={media_zero_count}"
    )
    print(f"[PRE-BGP full] label={label} " f"solute_bgp_type_counts={sbt_counts_dict}")
    print(f"[PRE-BGP full] label={label} " f"fp_eps={fp_eps} " f"fp_sbt={fp_sbt}")
    print(f"[PRE-BGP full] label={label} " f"fp_boundary_grid_indices_all={fp_bgi_all}")
    print(
        f"[PRE-BGP full] label={label} "
        f"fp_boundary_grid_indices_active={fp_bgi_active}"
    )

    # Keep existing head-style debug too, but now tied to this full summary.
    head_n = min(12, eps.size)
    if head_n > 0:
        head = eps[:head_n]
        head_abs = np.abs(head.astype(np.int64, copy=False))
        print(f"[PRE-BGP full] label={label} eps_head[{head_n}]={head.tolist()}")
        print(
            f"[PRE-BGP full] label={label} "
            f"head_media={(head_abs // epsdim).tolist()}"
        )
        print(
            f"[PRE-BGP full] label={label} "
            f"head_entity={(head_abs % epsdim).tolist()}"
        )


def _select_backend_module(use_cuda: bool, num_threads: int):
    # if use_cuda:
    #     from . import cuda_iter_gpu as backend
    #
    #     return backend
    if num_threads and num_threads > 1:
        from . import hybrid_colored_iter_cpu as backend

        return backend
    # Diagnostic path:
    # Run serial logic, but process frontier in 27-color order.
    # Use with num_threads=1:
    #
    #   PYDELPHI_BGP_SERIAL_COLORED=1 python ...
    #
    if os.environ.get("PYDELPHI_BGP_SERIAL_COLORED", "0") == "1":
        from . import serial_colored_iter_cpu as backend

        return backend

    from . import serial_iter_cpu as backend

    return backend


# ---------------------------------------------------------------------
# CUDA staging helpers
# ---------------------------------------------------------------------


def _is_numpy_array(x: Any) -> bool:
    return isinstance(x, np.ndarray)


def _is_scalarish(x: Any) -> bool:
    return isinstance(x, (int, float, bool, str, type(None), np.generic))


def _cuda_imports():
    from numba import cuda
    from numba.cuda.cudadrv.devicearray import DeviceNDArray

    return cuda, DeviceNDArray


def _to_device_maybe(cuda, DeviceNDArray, x: Any, stream=None):
    if _is_scalarish(x):
        return x
    if isinstance(x, DeviceNDArray):
        return x
    if _is_numpy_array(x):
        return (
            cuda.to_device(x, stream=stream)
            if stream is not None
            else cuda.to_device(x)
        )
    if isinstance(x, (tuple, list)) and all(_is_scalarish(t) for t in x):
        return x
    return x


# Cache compiled tiny fill kernels (so we don't re-jit every call)
_CUDA_FILL = {
    "ready": False,
    "fill_i32": None,
    "fill_i64": None,
    "clear_bgi_2d": None,
}


def _ensure_fill_kernels(cuda):
    if _CUDA_FILL["ready"]:
        return

    @cuda.jit
    def _fill_i32(arr, val):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = val

    @cuda.jit
    def _fill_i64(arr, val):
        i = cuda.grid(1)
        if i < arr.size:
            arr[i] = val

    @cuda.jit
    def _clear_bgi_2d(bgi):
        i = cuda.grid(1)
        if i < bgi.shape[0]:
            bgi[i, 0] = 0
            bgi[i, 1] = 0
            bgi[i, 2] = 0

    _CUDA_FILL["fill_i32"] = _fill_i32
    _CUDA_FILL["fill_i64"] = _fill_i64
    _CUDA_FILL["clear_bgi_2d"] = _clear_bgi_2d
    _CUDA_FILL["ready"] = True


def _memset_1d_i32(cuda, arr, value: int, threads: int = 256):
    n = int(arr.size)
    if n <= 0:
        return
    _ensure_fill_kernels(cuda)
    blocks = (n + threads - 1) // threads
    _CUDA_FILL["fill_i32"][blocks, threads](arr, np.int32(value))


def _memset_1d_i64(cuda, arr, value: int, threads: int = 256):
    n = int(arr.size)
    if n <= 0:
        return
    _ensure_fill_kernels(cuda)
    blocks = (n + threads - 1) // threads
    _CUDA_FILL["fill_i64"][blocks, threads](arr, np.int64(value))


def _clear_bgi2(cuda, bgi, threads: int = 256):
    _ensure_fill_kernels(cuda)
    n = int(bgi.shape[0])
    if n <= 0:
        return
    blocks = (n + threads - 1) // threads
    _CUDA_FILL["clear_bgi_2d"][blocks, threads](bgi)


def prepare_bgp_cuda_backend(
    *,
    boundary_grid_indices_host: np.ndarray,  # (nbgp,3)
    solute_bgp_type_1d_host: np.ndarray,
    index_discrete_epsilon_map_1d_host: np.ndarray,
    max_boundary_grid_points: int,
    grid_npoints: int,
    grid_shape: tuple[int, int, int],
    threads_per_block: int,
):
    cuda, _DeviceNDArray = _cuda_imports()
    stream = cuda.stream()

    stamp = np.int32(1)

    nx, ny, nz = grid_shape
    npoints = int(nx) * int(ny) * int(nz)
    idx_dtype = np.int32 if npoints <= (2**31 - 1) else np.int64

    ny_nz = idx_dtype(ny) * idx_dtype(nz)
    nz_ = idx_dtype(nz)

    bgi = boundary_grid_indices_host  # (nbgp,3)
    bgi_idx1d_host = (
        bgi[:, 0].astype(idx_dtype, copy=False) * ny_nz
        + bgi[:, 1].astype(idx_dtype, copy=False) * nz_
        + bgi[:, 2].astype(idx_dtype, copy=False)
    )
    bgi_idx1d_host = np.ascontiguousarray(bgi_idx1d_host, dtype=idx_dtype)

    # Core state arrays (device copies)
    d_bgi = cuda.to_device(bgi_idx1d_host, stream=stream)  # <-- 1D now
    d_sbt = cuda.to_device(solute_bgp_type_1d_host, stream=stream)
    d_eps = cuda.to_device(index_discrete_epsilon_map_1d_host, stream=stream)

    # Workspace buffers (match idx dtype)
    d_new_bgps_idx1d = cuda.device_array(
        shape=(max_boundary_grid_points,), dtype=idx_dtype, stream=stream
    )
    d_new_count = cuda.device_array(shape=(1,), dtype=np.int32, stream=stream)
    d_out_count = cuda.device_array(shape=(1,), dtype=np.int32, stream=stream)

    d_compact_stamp_arr = cuda.device_array(
        shape=(grid_npoints,), dtype=np.int32, stream=stream
    )

    d_touched_idx1d = cuda.device_array(
        shape=(max_boundary_grid_points * 7,), dtype=idx_dtype, stream=stream
    )
    d_touched_counter = cuda.device_array(shape=(1,), dtype=np.int32, stream=stream)

    d_exec_status = cuda.device_array(shape=(1,), dtype=np.int32, stream=stream)
    d_added_counter = cuda.device_array(shape=(1,), dtype=np.int32, stream=stream)
    d_removed_counter = cuda.device_array(shape=(1,), dtype=np.int32, stream=stream)
    d_num_external_counter = cuda.device_array(
        shape=(1,), dtype=np.int32, stream=stream
    )

    d_visited_next = cuda.device_array(
        shape=(grid_npoints,), dtype=np.int32, stream=stream
    )
    d_processed_stamp = cuda.device_array(
        shape=(grid_npoints,), dtype=np.int32, stream=stream
    )

    d_bgi_next = cuda.device_array(
        shape=(bgi_idx1d_host.shape[0],), dtype=idx_dtype, stream=stream
    )

    # ---- Explicit init ----
    _memset_1d_i32(cuda, d_processed_stamp, 0)
    _memset_1d_i32(cuda, d_visited_next, 0)
    _memset_1d_i32(cuda, d_compact_stamp_arr, 0)

    _memset_1d_i32(cuda, d_exec_status, 0)
    _memset_1d_i32(cuda, d_added_counter, 0)
    _memset_1d_i32(cuda, d_removed_counter, 0)
    _memset_1d_i32(cuda, d_num_external_counter, 0)
    _memset_1d_i32(cuda, d_new_count, 0)
    _memset_1d_i32(cuda, d_out_count, 0)
    _memset_1d_i32(cuda, d_touched_counter, 0)

    if idx_dtype == np.int32:
        _memset_1d_i32(cuda, d_new_bgps_idx1d, -1)
        _memset_1d_i32(cuda, d_touched_idx1d, -1)
        _memset_1d_i32(cuda, d_bgi_next, -1)
    else:
        _memset_1d_i64(cuda, d_new_bgps_idx1d, -1)
        _memset_1d_i64(cuda, d_touched_idx1d, -1)
        _memset_1d_i64(cuda, d_bgi_next, -1)

    cuda_ws = {
        "d_touched_idx1d": d_touched_idx1d,
        "d_touched_counter": d_touched_counter,
        "d_exec_status": d_exec_status,
        "d_added_counter": d_added_counter,
        "d_removed_counter": d_removed_counter,
        "d_num_external_counter": d_num_external_counter,
        "d_visited_next": d_visited_next,
        "d_processed_stamp": d_processed_stamp,
        "d_boundary_grid_indices_next": d_bgi_next,
        "d_new_bgps_idx1d": d_new_bgps_idx1d,
        "d_new_count": d_new_count,
        "d_out_count": d_out_count,
        "d_compact_stamp_arr": d_compact_stamp_arr,
        "stamp": stamp,
        "threads_per_block": int(threads_per_block),
        "stream": stream,
        "idx_dtype": idx_dtype,
        "ny_nz": ny_nz,
        "nz": nz_,
    }

    stream.synchronize()
    return (d_bgi, d_sbt, d_eps, cuda_ws)


def process_bgp_orchestrate(
    use_cuda: bool,
    num_threads: int,
    profile_timings: bool,
    max_points_added_in_iteration: int,
    max_no_divergence_count: int,
    max_total_iterations: Optional[int],
    num_boundary_grid_points: int,
    num_external_boundary_points: int,
    boundary_grid_indices: np.ndarray,
    solute_bgp_type_1d: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
    backend_kwargs: Dict[str, object],
    exit_njit_flag: int,
) -> BGPOrchestratorReturn:
    backend = _select_backend_module(use_cuda=use_cuda, num_threads=num_threads)

    # CUDA: stage to device ONCE
    d_bgi = d_sbt = d_eps = None
    cuda_ws = None
    backend_kwargs_iter: Dict[str, object] = backend_kwargs

    if use_cuda:
        cuda, DeviceNDArray = _cuda_imports()

        tpb = int(backend_kwargs.get("threads_per_block", 128))
        maxB = int(backend_kwargs["max_boundary_grid_points"])
        grid_npoints = int(solute_bgp_type_1d.size)

        d_bgi, d_sbt, d_eps, cuda_ws = prepare_bgp_cuda_backend(
            boundary_grid_indices_host=boundary_grid_indices,
            solute_bgp_type_1d_host=solute_bgp_type_1d,
            index_discrete_epsilon_map_1d_host=index_discrete_epsilon_map_1d,
            max_boundary_grid_points=maxB,
            grid_npoints=grid_npoints,
            grid_shape=backend_kwargs["grid_shape"],
            threads_per_block=tpb,
        )

        SKIP_TO_DEVICE = {
            "dtype_int",
            "dtype_real",
            "dtype_bool",
            "num_threads",
            "x_stride",
            "y_stride",
            "z_stride",
            "boundary_grid_indices_1d",
            "solute_bgp_type_1d",
            "index_discrete_epsilon_map_1d",
        }
        stream = cuda_ws["stream"]
        backend_kwargs_iter = {}
        for k, v in backend_kwargs.items():
            if k in SKIP_TO_DEVICE:
                continue
            backend_kwargs_iter[k] = _to_device_maybe(
                cuda, DeviceNDArray, v, stream=stream
            )

        backend_kwargs_iter["cuda_ws"] = cuda_ws

        boundary_grid_indices_iter = d_bgi
        solute_bgp_type_1d_iter = d_sbt
        index_discrete_epsilon_map_1d_iter = d_eps
    else:
        boundary_grid_indices_iter = boundary_grid_indices
        solute_bgp_type_1d_iter = solute_bgp_type_1d
        index_discrete_epsilon_map_1d_iter = index_discrete_epsilon_map_1d

    boundary_point_start_index = 1
    boundary_point_end_index = int(num_boundary_grid_points)
    num_discovered_bndy_grid_points = int(num_boundary_grid_points)

    # -----------------------------------------------------------------
    # PRE-BGP full host-side debug summary.
    # For CPU serial/parallel this is the exact state entering BGP iter 1.
    # Keep print-only; do not assert dataset-specific values.
    # -----------------------------------------------------------------
    if _VERBOSITY <= DEBUG:
        _print_pre_bgp_full_debug(
            label=f"cpu_threads_{int(num_threads)}",
            epsilon_dimension=int(backend_kwargs["epsilon_dimension"]),
            num_boundary_grid_points=int(num_boundary_grid_points),
            num_external_boundary_points=int(num_external_boundary_points),
            boundary_grid_indices=boundary_grid_indices_iter,
            solute_bgp_type_1d=solute_bgp_type_1d_iter,
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d_iter,
        )

    no_divergence_count = 0
    num_iterations = 0

    while True:
        num_iterations += 1

        if max_total_iterations is not None and num_iterations > max_total_iterations:
            status = int(exit_njit_flag)
            _print_bgp_final(
                status,
                num_iterations,
                num_discovered_bndy_grid_points,
                num_external_boundary_points,
                "max_total_iterations_exceeded",
            )
            break

        tic = time.perf_counter() if profile_timings else 0.0
        old_end = int(boundary_point_end_index)

        (
            exec_status,
            _cycle_flag,
            added_inc,
            removed_inc,
            num_external_boundary_points,
            boundary_grid_indices_iter,
            solute_bgp_type_1d_iter,
            index_discrete_epsilon_map_1d_iter,
        ) = backend._process_bgp_one_iteration(
            boundary_point_start_index=boundary_point_start_index,
            boundary_point_end_index=boundary_point_end_index,
            num_external_boundary_points=num_external_boundary_points,
            boundary_grid_indices=boundary_grid_indices_iter,
            solute_bgp_type_1d=solute_bgp_type_1d_iter,
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d_iter,
            **backend_kwargs_iter,
        )

        elapsed = (time.perf_counter() - tic) if profile_timings else None

        added_inc = int(added_inc)
        removed_inc = int(removed_inc)
        new_end = old_end + added_inc

        if int(exec_status) == int(exit_njit_flag):
            _print_bgp_final(
                int(exec_status),
                num_iterations,
                num_discovered_bndy_grid_points,
                num_external_boundary_points,
                "backend_exit_flag",
            )
            break
        if _VERBOSITY <= DEBUG:
            _print_bgp_progress(
                iteration=num_iterations,
                added=added_inc,
                removed=removed_inc,
                old_end=old_end,
                new_end=new_end,
                external=int(num_external_boundary_points),
                elapsed_s=elapsed,
            )

        boundary_point_start_index = old_end + 1
        boundary_point_end_index = new_end
        num_discovered_bndy_grid_points = new_end

        if added_inc > max_points_added_in_iteration:
            no_divergence_count += 1
            if no_divergence_count > max_no_divergence_count:
                _print_bgp_final(
                    int(exit_njit_flag),
                    num_iterations,
                    num_discovered_bndy_grid_points,
                    num_external_boundary_points,
                    "did_not_converge",
                )
                exec_status = int(exit_njit_flag)
                break
        else:
            no_divergence_count = 0

        if added_inc <= 0:
            exec_status = 0
            _print_bgp_final(
                0,
                num_iterations,
                num_discovered_bndy_grid_points,
                num_external_boundary_points,
                "converged",
            )
            break

    # CUDA: pull back ONCE
    if use_cuda:
        cuda, _ = _cuda_imports()
        stream = cuda_ws["stream"]
        stream.synchronize()

        boundary_grid_indices_1d = boundary_grid_indices_iter.copy_to_host(
            stream=stream
        )
        solute_bgp_type_1d = solute_bgp_type_1d_iter.copy_to_host(stream=stream)
        index_discrete_epsilon_map_1d = index_discrete_epsilon_map_1d_iter.copy_to_host(
            stream=stream
        )
        stream.synchronize()

        # grid dims
        nx, ny, nz = backend_kwargs_iter["grid_shape"]
        ny_nz = ny * nz

        # ensure int64 math for safety during division/mod
        idx = boundary_grid_indices_1d.astype(np.int64, copy=False)

        ix = idx // ny_nz
        rem = idx - ix * ny_nz
        iy = rem // nz
        iz = rem - iy * nz

        # pack into (N,3) int32
        # boundary_grid_indices = np.empty((idx.shape[0], 3), dtype=np.int32)
        boundary_grid_indices[:new_end, 0] = ix[:new_end]
        boundary_grid_indices[:new_end, 1] = iy[:new_end]
        boundary_grid_indices[:new_end, 2] = iz[:new_end]
    else:
        boundary_grid_indices = boundary_grid_indices_iter
        solute_bgp_type_1d = solute_bgp_type_1d_iter
        index_discrete_epsilon_map_1d = index_discrete_epsilon_map_1d_iter

    return (
        int(exec_status),
        boundary_grid_indices,
        solute_bgp_type_1d,
        index_discrete_epsilon_map_1d,
        int(num_discovered_bndy_grid_points),
        int(num_external_boundary_points),
        int(num_iterations),
    )
