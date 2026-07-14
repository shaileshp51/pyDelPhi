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

import numpy as np
from numba import njit, prange, cuda, set_num_threads, int32, uint8

from pydelphi.constants import (
    NON_SOLUTE_BOUNDARY,
    SOLUTE_BOUNDARY_ANY,
    SOLUTE_BOUNDARY_EXTERNAL,
)
from pydelphi.space.core.vdw.internals import _calculate_strides

THREADS_PER_BLOCK = 128  # must match your launch block size
MAX_LOCAL = 16  # per-thread local buffer capacity
SHARED_CAPACITY = THREADS_PER_BLOCK * MAX_LOCAL  # compile-time product


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _mark_boundary_points_cpu(
    epsilon_dimension,
    grid_shape,
    min_solute_grid_index,
    max_solute_grid_index,
    index_discrete_epsilon_map_1d,
    mark_array,
    per_thread_counts,
    per_thread_external_counts,
    num_threads,
    dtype_int,
):
    min_i = min_solute_grid_index[0]
    min_j = min_solute_grid_index[1]
    min_k = min_solute_grid_index[2]

    nx_solute = max_solute_grid_index[0] - min_i + 1
    ny_solute = max_solute_grid_index[1] - min_j + 1
    nz_solute = max_solute_grid_index[2] - min_k + 1

    x_stride_solute = ny_solute * nz_solute
    y_stride_solute = nz_solute
    total_points_solute = nx_solute * ny_solute * nz_solute

    block_size = (total_points_solute + num_threads - 1) // num_threads

    x_stride, y_stride, z_stride, xs3, ys3, zs3 = _calculate_strides(
        grid_shape, dtype_int
    )

    for thread_id in prange(num_threads):
        start = thread_id * block_size
        end = min((thread_id + 1) * block_size, total_points_solute)

        local_count = 0
        local_external = 0
        grid_neighbors_media_ids = np.zeros(7, dtype=dtype_int)

        for tid in range(start, end):
            i_rel = tid // x_stride_solute
            tmp = i_rel * x_stride_solute
            j_rel = (tid - tmp) // y_stride_solute
            k_rel = tid - tmp - j_rel * y_stride_solute

            i = i_rel + min_i
            j = j_rel + min_j
            k = k_rel + min_k

            ijk1d = i * x_stride + j * y_stride + k
            ijk1d_x_3 = ijk1d * 3

            grid_neighbors_media_ids[1] = (
                abs(index_discrete_epsilon_map_1d[ijk1d_x_3]) // epsilon_dimension
            )
            grid_neighbors_media_ids[2] = (
                abs(index_discrete_epsilon_map_1d[ijk1d_x_3 + 1]) // epsilon_dimension
            )
            grid_neighbors_media_ids[3] = (
                abs(index_discrete_epsilon_map_1d[ijk1d_x_3 + 2]) // epsilon_dimension
            )
            grid_neighbors_media_ids[4] = (
                abs(index_discrete_epsilon_map_1d[ijk1d_x_3 - xs3]) // epsilon_dimension
            )
            grid_neighbors_media_ids[5] = (
                abs(index_discrete_epsilon_map_1d[ijk1d_x_3 - ys3 + 1])
                // epsilon_dimension
            )
            grid_neighbors_media_ids[6] = (
                abs(index_discrete_epsilon_map_1d[ijk1d_x_3 - zs3 + 2])
                // epsilon_dimension
            )

            ext = False
            bnd = False
            for midpoint_index in range(1, 7):
                neighbor_index = midpoint_index % 6 + 1
                if grid_neighbors_media_ids[midpoint_index] == 0:
                    ext = True
                if (
                    grid_neighbors_media_ids[midpoint_index]
                    != grid_neighbors_media_ids[neighbor_index]
                ):
                    bnd = True

            if bnd:
                if ext:
                    mark_array[tid] = SOLUTE_BOUNDARY_ANY | SOLUTE_BOUNDARY_EXTERNAL
                    local_external += 1
                else:
                    mark_array[tid] = SOLUTE_BOUNDARY_ANY
                local_count += 1
            else:
                mark_array[tid] = NON_SOLUTE_BOUNDARY

        per_thread_counts[thread_id] = local_count
        per_thread_external_counts[thread_id] = local_external


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _collect_boundary_points_cpu(
    grid_shape,
    min_solute_grid_index,
    max_solute_grid_index,
    solute_bgp_type_1d,
    mark_array,
    base_offsets,
    boundary_grid_indices,
    num_threads,
    dtype_int,
):
    min_i = min_solute_grid_index[0]
    min_j = min_solute_grid_index[1]
    min_k = min_solute_grid_index[2]

    nx_solute = max_solute_grid_index[0] - min_i + 1
    ny_solute = max_solute_grid_index[1] - min_j + 1
    nz_solute = max_solute_grid_index[2] - min_k + 1

    x_stride_solute = ny_solute * nz_solute
    y_stride_solute = nz_solute
    total_points_solute = nx_solute * ny_solute * nz_solute

    block_size = (total_points_solute + num_threads - 1) // num_threads

    x_stride, y_stride, z_stride, xs3, ys3, zs3 = _calculate_strides(
        grid_shape, dtype_int
    )

    for thread_id in prange(num_threads):
        start = thread_id * block_size
        end = min((thread_id + 1) * block_size, total_points_solute)

        thread_base = base_offsets[thread_id]
        local_write_index = 0

        for tid in range(start, end):
            m = mark_array[tid]
            if m != NON_SOLUTE_BOUNDARY:
                i_rel = tid // x_stride_solute
                tmp = i_rel * x_stride_solute
                j_rel = (tid - tmp) // y_stride_solute
                k_rel = tid - tmp - j_rel * y_stride_solute

                i = i_rel + min_i
                j = j_rel + min_j
                k = k_rel + min_k

                ijk1d = i * x_stride + j * y_stride + k
                ijk1d_x_3 = ijk1d * 3

                idx_out = thread_base + local_write_index
                boundary_grid_indices[idx_out, 0] = i
                boundary_grid_indices[idx_out, 1] = j
                boundary_grid_indices[idx_out, 2] = k

                solute_bgp_type_1d[ijk1d] = m

                local_write_index += 1


def _find_boundary_grid_points_cpu_orchestrator(
    epsilon_dimension,
    grid_shape,
    min_solute_grid_index,
    max_solute_grid_index,
    index_discrete_epsilon_map_1d,
    dtype_int,
    num_threads,
):
    min_i = min_solute_grid_index[0]
    min_j = min_solute_grid_index[1]
    min_k = min_solute_grid_index[2]

    nx_solute = max_solute_grid_index[0] - min_i + 1
    ny_solute = max_solute_grid_index[1] - min_j + 1
    nz_solute = max_solute_grid_index[2] - min_k + 1

    total_points_solute = nx_solute * ny_solute * nz_solute

    mark_array = np.zeros(total_points_solute, dtype=np.uint8)
    per_thread_counts = np.zeros(num_threads, dtype=dtype_int)
    per_thread_external_counts = np.zeros(num_threads, dtype=dtype_int)

    set_num_threads(num_threads)
    _mark_boundary_points_cpu(
        epsilon_dimension,
        grid_shape,
        min_solute_grid_index,
        max_solute_grid_index,
        index_discrete_epsilon_map_1d,
        mark_array,
        per_thread_counts,
        per_thread_external_counts,
        num_threads,
        dtype_int,
    )

    n_boundary = int(np.sum(per_thread_counts))
    n_external = int(np.sum(per_thread_external_counts))

    # exclusive scan for offsets
    base_offsets = np.zeros(num_threads, dtype=dtype_int)
    running = 0
    for t in range(num_threads):
        base_offsets[t] = running
        running += int(per_thread_counts[t])

    solute_bgp_type_1d = np.full(
        grid_shape[0] * grid_shape[1] * grid_shape[2],
        fill_value=NON_SOLUTE_BOUNDARY,
        dtype=np.uint8,
    )
    boundary_grid_indices = (
        np.zeros((n_boundary, 3), dtype=dtype_int)
        if n_boundary > 0
        else np.zeros((0, 3), dtype=dtype_int)
    )

    set_num_threads(num_threads)
    _collect_boundary_points_cpu(
        grid_shape,
        min_solute_grid_index,
        max_solute_grid_index,
        solute_bgp_type_1d,
        mark_array,
        base_offsets,
        boundary_grid_indices,
        num_threads,
        dtype_int,
    )

    return (
        n_boundary,
        n_external,
        solute_bgp_type_1d,
        boundary_grid_indices,
    )


# ---------------------------------------------------------------------
# CUDA Pass 1: mark boundaries and accumulate per-block counts in shared memory
# ---------------------------------------------------------------------


@cuda.jit(cache=True)
def _mark_boundary_points_cuda(
    epsilon_dimension,
    grid_shape,
    min_solute_grid_index,
    max_solute_grid_index,
    index_discrete_epsilon_map_1d,
    mark_array,
    block_counts,
    block_external_counts,
):
    thread_idx = cuda.threadIdx.x
    block_dim = cuda.blockDim.x
    block_id = cuda.blockIdx.x
    grid_dim = cuda.gridDim.x

    sm_counts = cuda.shared.array(THREADS_PER_BLOCK, dtype=int32)
    sm_external = cuda.shared.array(THREADS_PER_BLOCK, dtype=int32)

    sm_counts[thread_idx] = 0
    sm_external[thread_idx] = 0
    cuda.syncthreads()

    min_i = min_solute_grid_index[0]
    min_j = min_solute_grid_index[1]
    min_k = min_solute_grid_index[2]

    nx_solute = max_solute_grid_index[0] - min_i + 1
    ny_solute = max_solute_grid_index[1] - min_j + 1
    nz_solute = max_solute_grid_index[2] - min_k + 1
    total_points_solute = nx_solute * ny_solute * nz_solute

    x_stride_solute = ny_solute * nz_solute
    y_stride_solute = nz_solute

    x_stride = grid_shape[1] * grid_shape[2]
    y_stride = grid_shape[2]
    xs3 = x_stride * 3
    ys3 = y_stride * 3
    zs3 = 3

    # Each block owns one contiguous chunk.
    chunk_size = (total_points_solute + grid_dim - 1) // grid_dim
    start = block_id * chunk_size
    end = start + chunk_size
    if end > total_points_solute:
        end = total_points_solute

    grid_neighbors_media_ids = cuda.local.array(7, dtype=int32)

    for idx in range(start + thread_idx, end, block_dim):
        i_rel = idx // x_stride_solute
        tmp = i_rel * x_stride_solute
        j_rel = (idx - tmp) // y_stride_solute
        k_rel = idx - tmp - j_rel * y_stride_solute

        i = i_rel + min_i
        j = j_rel + min_j
        k = k_rel + min_k

        ijk1d = i * x_stride + j * y_stride + k
        ijk1d_x3 = ijk1d * 3

        grid_neighbors_media_ids[1] = (
            abs(index_discrete_epsilon_map_1d[ijk1d_x3]) // epsilon_dimension
        )
        grid_neighbors_media_ids[2] = (
            abs(index_discrete_epsilon_map_1d[ijk1d_x3 + 1]) // epsilon_dimension
        )
        grid_neighbors_media_ids[3] = (
            abs(index_discrete_epsilon_map_1d[ijk1d_x3 + 2]) // epsilon_dimension
        )
        grid_neighbors_media_ids[4] = (
            abs(index_discrete_epsilon_map_1d[ijk1d_x3 - xs3]) // epsilon_dimension
        )
        grid_neighbors_media_ids[5] = (
            abs(index_discrete_epsilon_map_1d[ijk1d_x3 - ys3 + 1]) // epsilon_dimension
        )
        grid_neighbors_media_ids[6] = (
            abs(index_discrete_epsilon_map_1d[ijk1d_x3 - zs3 + 2]) // epsilon_dimension
        )

        bnd = False
        ext = False

        for midpoint_index in range(1, 7):
            neighbor_index = midpoint_index % 6 + 1

            if grid_neighbors_media_ids[midpoint_index] == 0:
                ext = True

            if (
                grid_neighbors_media_ids[midpoint_index]
                != grid_neighbors_media_ids[neighbor_index]
            ):
                bnd = True

        if bnd:
            if ext:
                mark_array[idx] = SOLUTE_BOUNDARY_ANY | SOLUTE_BOUNDARY_EXTERNAL
                sm_external[thread_idx] += 1
            else:
                mark_array[idx] = SOLUTE_BOUNDARY_ANY

            sm_counts[thread_idx] += 1
        else:
            mark_array[idx] = NON_SOLUTE_BOUNDARY

    cuda.syncthreads()

    if thread_idx == 0:
        total = 0
        total_ext = 0

        for t in range(block_dim):
            total += sm_counts[t]
            total_ext += sm_external[t]

        block_counts[block_id] = total
        block_external_counts[block_id] = total_ext


@cuda.jit(cache=True)
def _collect_boundary_points_cuda_block_prefix(
    grid_shape,
    min_solute_grid_index,
    max_solute_grid_index,
    solute_bgp_type_1d,
    mark_array,
    block_offsets,
    block_counts,
    boundary_grid_indices,
    block_written,
    n_boundary,
    error_flag,
):
    block_id = cuda.blockIdx.x
    thread_idx = cuda.threadIdx.x
    block_dim = cuda.blockDim.x
    grid_dim = cuda.gridDim.x

    sm_flags = cuda.shared.array(THREADS_PER_BLOCK, dtype=int32)
    sm_scan = cuda.shared.array(THREADS_PER_BLOCK, dtype=int32)

    min_i = min_solute_grid_index[0]
    min_j = min_solute_grid_index[1]
    min_k = min_solute_grid_index[2]

    nx_solute = max_solute_grid_index[0] - min_i + 1
    ny_solute = max_solute_grid_index[1] - min_j + 1
    nz_solute = max_solute_grid_index[2] - min_k + 1
    total_points_solute = nx_solute * ny_solute * nz_solute

    x_stride_solute = ny_solute * nz_solute
    y_stride_solute = nz_solute

    x_stride = grid_shape[1] * grid_shape[2]
    y_stride = grid_shape[2]

    # Must match pass 1 exactly.
    chunk_size = (total_points_solute + grid_dim - 1) // grid_dim
    start = block_id * chunk_size
    end = start + chunk_size
    if end > total_points_solute:
        end = total_points_solute

    base = block_offsets[block_id]
    limit = base + block_counts[block_id]

    running = 0
    tile_start = start

    while tile_start < end:
        idx = tile_start + thread_idx

        m = uint8(0)
        flag = int32(0)

        if idx < end:
            m = mark_array[idx]
            if m != NON_SOLUTE_BOUNDARY:
                flag = int32(1)

        sm_flags[thread_idx] = flag
        sm_scan[thread_idx] = flag
        cuda.syncthreads()

        # Inclusive scan over flags.
        offset = 1
        while offset < block_dim:
            val = int32(0)

            if thread_idx >= offset:
                val = sm_scan[thread_idx - offset]

            cuda.syncthreads()
            sm_scan[thread_idx] += val
            cuda.syncthreads()

            offset *= 2

        local_exclusive = sm_scan[thread_idx] - sm_flags[thread_idx]
        tile_count = sm_scan[block_dim - 1]

        if flag != 0:
            i_rel = idx // x_stride_solute
            tmp = i_rel * x_stride_solute
            j_rel = (idx - tmp) // y_stride_solute
            k_rel = idx - tmp - j_rel * y_stride_solute

            i = i_rel + min_i
            j = j_rel + min_j
            k = k_rel + min_k

            ijk1d = i * x_stride + j * y_stride + k
            dest = base + running + local_exclusive

            if dest >= limit or dest >= n_boundary:
                cuda.atomic.max(error_flag, 0, 1)
            else:
                boundary_grid_indices[dest, 0] = i
                boundary_grid_indices[dest, 1] = j
                boundary_grid_indices[dest, 2] = k

                solute_bgp_type_1d[ijk1d] = m

        cuda.syncthreads()

        running += tile_count
        tile_start += block_dim

    if thread_idx == 0:
        block_written[block_id] = running

        if running != block_counts[block_id]:
            cuda.atomic.max(error_flag, 0, 2)


# ---------------------------------------------------------------------
# CUDA orchestrator
# ---------------------------------------------------------------------

def _find_boundary_grid_points_cuda_orchestrator(
    epsilon_dimension,
    grid_shape,
    min_solute_grid_index,
    max_solute_grid_index,
    index_discrete_epsilon_map_1d,
    dtype_int=np.int32,
    threads_per_block=THREADS_PER_BLOCK,
):
    d_grid_shape = None
    d_min_solute = None
    d_max_solute = None

    d_index_map = None
    d_mark_array = None
    d_block_counts = None
    d_block_external = None

    d_solute_bgp_type_1d = None
    d_boundary_grid_indices = None
    d_block_offsets = None
    d_block_counts_for_collect = None
    d_block_written = None
    d_error_flag = None

    try:
        nx_solute = int(max_solute_grid_index[0] - min_solute_grid_index[0] + 1)
        ny_solute = int(max_solute_grid_index[1] - min_solute_grid_index[1] + 1)
        nz_solute = int(max_solute_grid_index[2] - min_solute_grid_index[2] + 1)
        total_points_solute = nx_solute * ny_solute * nz_solute

        n_blocks = (total_points_solute + threads_per_block - 1) // threads_per_block

        # Explicit device copies for all array-valued kernel metadata.
        d_grid_shape = cuda.to_device(np.array(grid_shape, dtype=dtype_int))
        d_min_solute = cuda.to_device(np.array(min_solute_grid_index, dtype=dtype_int))
        d_max_solute = cuda.to_device(np.array(max_solute_grid_index, dtype=dtype_int))

        d_index_map = cuda.to_device(index_discrete_epsilon_map_1d)

        # Explicit zero initialization is retained by design.
        d_mark_array = cuda.to_device(np.zeros(total_points_solute, dtype=np.uint8))
        d_block_counts = cuda.to_device(np.zeros(n_blocks, dtype=dtype_int))
        d_block_external = cuda.to_device(np.zeros(n_blocks, dtype=dtype_int))

        _mark_boundary_points_cuda[
            int(n_blocks),
            int(threads_per_block),
        ](
            epsilon_dimension,
            d_grid_shape,
            d_min_solute,
            d_max_solute,
            d_index_map,
            d_mark_array,
            d_block_counts,
            d_block_external,
        )
        cuda.synchronize()

        block_counts = d_block_counts.copy_to_host()
        block_external_counts = d_block_external.copy_to_host()

        n_boundary = int(np.sum(block_counts))
        n_external = int(np.sum(block_external_counts))

        # Host exclusive prefix sum over blocks only.
        block_offsets = np.zeros(n_blocks, dtype=dtype_int)
        running = 0

        for b in range(n_blocks):
            block_offsets[b] = running
            running += int(block_counts[b])

        if running != n_boundary:
            raise RuntimeError(
                f"CUDA boundary prefix mismatch: running={running}, "
                f"n_boundary={n_boundary}"
            )

        solute_bgp_type_1d = np.full(
            int(grid_shape[0]) * int(grid_shape[1]) * int(grid_shape[2]),
            fill_value=NON_SOLUTE_BOUNDARY,
            dtype=np.uint8,
        )

        boundary_grid_indices = (
            np.zeros((n_boundary, 3), dtype=dtype_int)
            if n_boundary > 0
            else np.zeros((0, 3), dtype=dtype_int)
        )

        d_solute_bgp_type_1d = cuda.to_device(solute_bgp_type_1d)
        d_boundary_grid_indices = cuda.to_device(boundary_grid_indices)
        d_block_offsets = cuda.to_device(block_offsets)
        d_block_counts_for_collect = cuda.to_device(block_counts)

        # Explicit zero initialization is retained by design.
        d_block_written = cuda.to_device(np.zeros(n_blocks, dtype=dtype_int))
        d_error_flag = cuda.to_device(np.zeros(1, dtype=np.int32))

        if n_boundary > 0:
            _collect_boundary_points_cuda_block_prefix[
                int(n_blocks),
                int(threads_per_block),
            ](
                d_grid_shape,
                d_min_solute,
                d_max_solute,
                d_solute_bgp_type_1d,
                d_mark_array,
                d_block_offsets,
                d_block_counts_for_collect,
                d_boundary_grid_indices,
                d_block_written,
                int(n_boundary),
                d_error_flag,
            )
            cuda.synchronize()

        h_error_flag = d_error_flag.copy_to_host()

        if h_error_flag[0] != 0:
            h_block_written = d_block_written.copy_to_host()

            raise RuntimeError(
                "CUDA boundary collection failed: "
                f"error_flag={int(h_error_flag[0])}, "
                f"sum_written={int(np.sum(h_block_written))}, "
                f"sum_counted={int(np.sum(block_counts))}"
            )

        h_block_written = d_block_written.copy_to_host()

        if not np.array_equal(
            h_block_written.astype(block_counts.dtype),
            block_counts,
        ):
            raise RuntimeError(
                "CUDA boundary collection count mismatch: "
                f"sum_written={int(np.sum(h_block_written))}, "
                f"sum_counted={int(np.sum(block_counts))}"
            )

        solute_bgp_type_1d = d_solute_bgp_type_1d.copy_to_host()
        boundary_grid_indices = d_boundary_grid_indices.copy_to_host()

        return (
            n_boundary,
            n_external,
            solute_bgp_type_1d,
            boundary_grid_indices,
        )

    finally:
        # Ensure all scheduled work using these buffers has completed before
        # dropping the local Python references.
        try:
            cuda.synchronize()
        except Exception:
            # Preserve the original exception if CUDA execution already failed.
            pass

        d_grid_shape = None
        d_min_solute = None
        d_max_solute = None

        d_index_map = None
        d_mark_array = None
        d_block_counts = None
        d_block_external = None

        d_solute_bgp_type_1d = None
        d_boundary_grid_indices = None
        d_block_offsets = None
        d_block_counts_for_collect = None
        d_block_written = None
        d_error_flag = None


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def find_boundary_grid_points(
    epsilon_dimension,
    grid_shape,
    min_solute_grid_index,
    max_solute_grid_index,
    index_discrete_epsilon_map_1d,
    dtype_int,
    use_cuda,
    num_threads,
):
    if use_cuda:
        return _find_boundary_grid_points_cuda_orchestrator(
            epsilon_dimension,
            grid_shape,
            min_solute_grid_index,
            max_solute_grid_index,
            index_discrete_epsilon_map_1d,
            dtype_int=dtype_int,
            threads_per_block=THREADS_PER_BLOCK,
        )
    else:
        return _find_boundary_grid_points_cpu_orchestrator(
            epsilon_dimension,
            grid_shape,
            min_solute_grid_index,
            max_solute_grid_index,
            index_discrete_epsilon_map_1d,
            dtype_int=dtype_int,
            num_threads=num_threads,
        )
