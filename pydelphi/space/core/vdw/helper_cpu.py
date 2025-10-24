#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

#
# pyDelPhi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# pyDelPhi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#

#
# PyDelphi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# PyDelphi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#


"""
CPU-parallel, block-based algorithm for building a zeta surface map.

This module provides a highly optimized, multi-step parallel algorithm for
identifying and collecting zeta surface boundary points from a 3D grid. The
approach is designed for performance on multi-core CPUs using Numba's
parallel execution capabilities.

The core function, `build_zeta_surface_map_cpu_parallel_block`, employs a
"two-pass" or "map-reduce"-style strategy to avoid race conditions and the
need for dynamic array resizing within parallel loops, which are common
performance bottlenecks.

The algorithm proceeds in the following steps:
1.  **Map & Mark**: The grid is divided into chunks, and each CPU thread
    processes one chunk in parallel. It identifies the zeta surface points
    within its chunk and records them in a local binary map (1 for a point,
    0 otherwise).
2.  **Count**: Each thread counts the number of points it found in its local
    map.
3.  **Prefix Sum (Offset Calculation)**: The main thread computes a cumulative
    sum of the counts from all threads. This gives each thread a unique offset,
    determining where it will write its results in the final output arrays.
4.  **Write**: Each thread, using its calculated offset, writes the coordinates
    and indices of its identified points directly into the final,
    pre-allocated global arrays.

This method ensures that memory for the final results is allocated only once
and that threads write to distinct, non-overlapping sections of the output
arrays, eliminating write hazards without the need for locks.
"""

from numba import njit, prange
import numpy as np
import numba


@njit(nogil=True, boundscheck=False, cache=True)
def _build_block_local_binary_map_cpu(
    grid_shape,
    zeta_surface_map_1d,
    index_discrete_epsilon_map_1d,
    epsdim,
    binary_map_block,
    start_index,
    num_elements,
    x_stride,
    y_stride,
    z_stride,
    x_stride_x_3,
    y_stride_x_3,
    z_stride_x_3,
):
    """Step 1: Build block-local binary map for a chunk of the grid (CPU)."""
    for i in range(num_elements):
        linear_index = start_index + i
        lz = linear_index % grid_shape[2]
        ly = (linear_index // grid_shape[2]) % grid_shape[1]
        lx = linear_index // (grid_shape[2] * grid_shape[1])

        if (
            1 <= lx < grid_shape[0] - 1
            and 1 <= ly < grid_shape[1] - 1
            and 1 <= lz < grid_shape[2] - 1
        ):
            lxyz1d = lx * x_stride + ly * y_stride + lz
            lxyz1d_x_3 = lxyz1d * 3

            zeta_tmp = np.zeros(7, dtype=np.bool_)
            zeta_tmp[0] = zeta_surface_map_1d[lxyz1d]
            zeta_tmp[1] = zeta_surface_map_1d[lxyz1d + z_stride]
            zeta_tmp[2] = zeta_surface_map_1d[lxyz1d - z_stride]
            zeta_tmp[3] = zeta_surface_map_1d[lxyz1d + y_stride]
            zeta_tmp[4] = zeta_surface_map_1d[lxyz1d - y_stride]
            zeta_tmp[5] = zeta_surface_map_1d[lxyz1d + x_stride]
            zeta_tmp[6] = zeta_surface_map_1d[lxyz1d - x_stride]

            zright = abs(index_discrete_epsilon_map_1d[lxyz1d_x_3]) // epsdim
            zfront = abs(index_discrete_epsilon_map_1d[lxyz1d_x_3 + 1]) // epsdim
            ztop = abs(index_discrete_epsilon_map_1d[lxyz1d_x_3 + 2]) // epsdim
            zleft = (
                abs(index_discrete_epsilon_map_1d[lxyz1d_x_3 - x_stride_x_3]) // epsdim
            )
            zback = (
                abs(index_discrete_epsilon_map_1d[lxyz1d_x_3 - y_stride_x_3 + 1])
                // epsdim
            )
            zbottom = (
                abs(index_discrete_epsilon_map_1d[lxyz1d_x_3 - z_stride_x_3 + 2])
                // epsdim
            )

            zext = 0
            if (
                zeta_tmp[0]
                and zright == 0
                and zleft == 0
                and ztop == 0
                and zbottom == 0
                and zfront == 0
                and zback == 0
            ):
                zext = 1

            zbgp = 0
            for midpoint_index in range(1, 7):
                if zeta_tmp[midpoint_index] != zeta_tmp[midpoint_index - 1]:
                    zbgp = 1

            if zeta_tmp[6] != zeta_tmp[1]:
                zbgp = 1

            if zbgp > 0 and zext == 1:
                binary_map_block[i] = 1
            else:
                binary_map_block[i] = 0
        else:
            binary_map_block[i] = 0


@njit(nogil=True, boundscheck=False, cache=True)
def _count_zeta_points_per_block_cpu(binary_map_block):
    """Step 2: Count zeta points in a block (CPU)."""
    count = 0
    for val in binary_map_block:
        if val == 1:
            count += 1
    return count


@njit(nogil=True, boundscheck=False, cache=True)
def _write_zeta_points_with_offset_cpu(
    grid_shape,
    grid_spacing,
    gridbox_center,
    indices_mid_grid,
    binary_map_block,
    out_coords,
    out_indices,
    offset,
    start_index,
    num_elements,
    x_stride,
    y_stride,
    z_stride,
):
    """Step 4: Write to global output arrays with precomputed offsets (CPU)."""
    local_count = 0
    for i in range(num_elements):
        if binary_map_block[i] == 1:
            linear_index = start_index + i
            lz = linear_index % grid_shape[2]
            ly = (linear_index // grid_shape[2]) % grid_shape[1]
            lx = linear_index // (grid_shape[2] * grid_shape[1])

            lxyz = np.array([lx, ly, lz], dtype=np.int32)
            xyz = (
                gridbox_center
                + (lxyz.astype(np.float32) - indices_mid_grid) * grid_spacing
            )
            global_index = offset + local_count
            out_indices[global_index] = lxyz
            out_coords[global_index] = xyz
            local_count += 1


@njit
def build_zeta_surface_map_cpu_parallel_block(
    grid_spacing,
    grid_shape,
    gridbox_center,
    indices_mid_grid,
    zeta_surface_map_1d,
    index_discrete_epsilon_map_1d,
    epsdim,
    zeta_coords_capacity,
    zeta_indices_capacity,
):
    """Two-step CPU parallel implementation with block-level strategy."""
    num_threads = numba.config.NUMBA_NUM_THREADS
    total_elements = np.prod(grid_shape)
    elements_per_chunk = (total_elements + num_threads - 1) // num_threads

    binary_maps = [
        np.zeros(elements_per_chunk, dtype=np.uint8) for _ in range(num_threads)
    ]
    block_counts = np.zeros(num_threads, dtype=np.int64)

    x_stride = grid_shape[1] * grid_shape[2]
    y_stride = grid_shape[2]
    z_stride = 1
    x_stride_x_3 = x_stride * 3
    y_stride_x_3 = y_stride * 3
    z_stride_x_3 = 3

    # Step 1: Build block-local binary maps in parallel
    for thread_id in prange(num_threads):
        start_index = thread_id * elements_per_chunk
        num_elements = min(elements_per_chunk, total_elements - start_index)
        _build_block_local_binary_map_cpu(
            grid_shape,
            zeta_surface_map_1d,
            index_discrete_epsilon_map_1d,
            epsdim,
            binary_maps[thread_id],
            start_index,
            num_elements,
            x_stride,
            y_stride,
            z_stride,
            x_stride_x_3,
            y_stride_x_3,
            z_stride_x_3,
        )

    # Step 2: Count zeta points per block in parallel
    for thread_id in prange(num_threads):
        block_counts[thread_id] = _count_zeta_points_per_block_cpu(
            binary_maps[thread_id]
        )

    # Step 3: Calculate offsets
    offsets = np.zeros_like(block_counts)
    offsets[1:] = np.cumsum(block_counts[:-1])
    total_points = np.sum(block_counts)

    # Step 4: Write to global output arrays with offsets in parallel
    out_coords = np.empty((total_points, 3), dtype=np.float32)
    out_indices = np.empty((total_points, 3), dtype=np.int32)

    for thread_id in prange(num_threads):
        start_index = thread_id * elements_per_chunk
        num_elements = min(elements_per_chunk, total_elements - start_index)
        _write_zeta_points_with_offset_cpu(
            grid_shape,
            grid_spacing,
            gridbox_center,
            indices_mid_grid,
            binary_maps[thread_id],
            out_coords,
            out_indices,
            offsets[thread_id],
            start_index,
            num_elements,
            x_stride,
            y_stride,
            z_stride,
        )

    return (
        out_coords,
        out_indices,
        total_points,
        total_points,
        out_coords.nbytes,
        out_indices.nbytes,
    )
