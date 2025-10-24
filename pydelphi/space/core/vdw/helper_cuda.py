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
GPU-accelerated algorithm for building a zeta surface map using CUDA.

This module provides a high-performance, multi-step parallel algorithm for
identifying and collecting zeta surface boundary points from a 3D grid,
specifically designed for execution on NVIDIA GPUs using the CUDA framework
via Numba.

The approach mirrors the block-based strategy of the CPU version but is adapted
to the GPU architecture, leveraging CUDA kernels and shared memory for maximum
parallelism and efficiency. The core function, `build_zeta_surface_map_cuda_parallel_block`,
orchestrates the process from the host (CPU).

The algorithm follows these main steps:
1.  **Data Transfer**: Input grid data is copied from the host (CPU) to the
    device (GPU) global memory.
2.  **Kernel 1: Map & Mark**: A CUDA kernel is launched where each thread block
    processes a chunk of the grid. Zeta surface points are identified and
    marked in a `shared memory` array, which is a fast, on-chip cache local
    to each block.
3.  **Kernel 2: Count per Block**: A second kernel counts the number of points
    found by each block (stored in its shared memory) and writes this count to
    a global array.
4.  **Prefix Sum (on Host)**: The block counts are transferred back to the host,
    which calculates a cumulative sum to determine the correct write offset for
    each block. This avoids write conflicts in the final output array. The
    offsets are then copied back to the device.
5.  **Kernel 3: Write to Global Memory**: A final kernel is launched. It first
    re-populates the shared binary map (as shared memory is not persistent
    across kernel launches). Then, threads corresponding to marked points use
    the block's offset and atomic operations to write the final coordinates
    and indices to the pre-allocated global output arrays in a conflict-free
    manner.
6.  **Final Data Transfer**: The resulting arrays are copied from the device
    back to the host.

This strategy is optimized to minimize slow global memory access by using
fast shared memory for intermediate, block-local results and ensures that the
final write to global memory is fully parallel and free of race conditions.
"""
from Cython.Shadow import nogil, boundscheck
from numba import cuda, float32, int32, bool_, njit
import numpy as np
import numba

TPB = 32  # Threads per block - adjust based on your GPU


@cuda.jit(cache=True)
def _build_block_local_binary_map_cuda(
    grid_shape_d,
    zeta_surface_map_1d_d,
    index_discrete_epsilon_map_1d_d,
    epsdim,
    binary_map_shared,
    start_index,
    num_elements,
    x_stride,
    y_stride,
    z_stride,
    x_stride_x_3,
    y_stride_x_3,
    z_stride_x_3,
):
    """Step 1: Build block-local binary map in shared memory."""
    thread_id_x = cuda.threadIdx.x
    thread_id_y = cuda.threadIdx.y
    thread_id_z = cuda.threadIdx.z
    block_dim_x = cuda.blockDim.x
    block_dim_y = cuda.blockDim.y
    block_dim_z = cuda.blockDim.z

    linear_index = (
        start_index
        + cuda.blockIdx.x * cuda.blockDim.x
        + cuda.blockIdx.y * cuda.blockDim.y * block_dim_x
        + cuda.blockIdx.z * cuda.blockDim.z * block_dim_x * block_dim_y
        + thread_id_x
        + thread_id_y * block_dim_x
        + thread_id_z * block_dim_x * block_dim_y
    )

    if linear_index < start_index + num_elements:
        lz = linear_index % grid_shape_d[2]
        ly = (linear_index // grid_shape_d[2]) % grid_shape_d[1]
        lx = linear_index // (grid_shape_d[2] * grid_shape_d[1])

        if (
            1 <= lx < grid_shape_d[0] - 1
            and 1 <= ly < grid_shape_d[1] - 1
            and 1 <= lz < grid_shape_d[2] - 1
        ):
            lxyz1d = lx * x_stride + ly * y_stride + lz
            lxyz1d_x_3 = lxyz1d * 3

            zeta_tmp = cuda.local.array(7, dtype=bool_)
            zeta_tmp[0] = zeta_surface_map_1d_d[lxyz1d]
            zeta_tmp[1] = zeta_surface_map_1d_d[lxyz1d + z_stride]
            zeta_tmp[2] = zeta_surface_map_1d_d[lxyz1d - z_stride]
            zeta_tmp[3] = zeta_surface_map_1d_d[lxyz1d + y_stride]
            zeta_tmp[4] = zeta_surface_map_1d_d[lxyz1d - y_stride]
            zeta_tmp[5] = zeta_surface_map_1d_d[lxyz1d + x_stride]
            zeta_tmp[6] = zeta_surface_map_1d_d[lxyz1d - x_stride]

            zright = abs(index_discrete_epsilon_map_1d_d[lxyz1d_x_3]) // epsdim
            zfront = abs(index_discrete_epsilon_map_1d_d[lxyz1d_x_3 + 1]) // epsdim
            ztop = abs(index_discrete_epsilon_map_1d_d[lxyz1d_x_3 + 2]) // epsdim
            zleft = (
                abs(index_discrete_epsilon_map_1d_d[lxyz1d_x_3 - x_stride_x_3])
                // epsdim
            )
            zback = (
                abs(index_discrete_epsilon_map_1d_d[lxyz1d_x_3 - y_stride_x_3 + 1])
                // epsdim
            )
            zbottom = (
                abs(index_discrete_epsilon_map_1d_d[lxyz1d_x_3 - z_stride_x_3 + 2])
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
                local_index = (
                    thread_id_z * block_dim_y * block_dim_x
                    + thread_id_y * block_dim_x
                    + thread_id_x
                )
                binary_map_shared[local_index] = 1
            else:
                local_index = (
                    thread_id_z * block_dim_y * block_dim_x
                    + thread_id_y * block_dim_x
                    + thread_id_x
                )
                binary_map_shared[local_index] = 0
        else:
            local_index = (
                thread_id_z * block_dim_y * block_dim_x
                + thread_id_y * block_dim_x
                + thread_id_x
            )
            binary_map_shared[local_index] = 0
    else:
        local_index = (
            thread_id_z * block_dim_y * block_dim_x
            + thread_id_y * block_dim_x
            + thread_id_x
        )
        binary_map_shared[local_index] = 0
    cuda.syncthreads()


@cuda.jit(cache=True)
def _count_zeta_points_per_block_cuda(binary_map_shared, block_counts):
    """Step 2: Count zeta points per block."""
    count = 0
    for i in range(binary_map_shared.shape[0]):
        if binary_map_shared[i] == 1:
            count += 1
    block_counts[cuda.blockIdx.x] = count


@cuda.jit(cache=True)
def _write_zeta_points_with_offset_cuda(
    grid_shape_d,
    grid_spacing,
    gridbox_center_d,
    indices_mid_grid_d,
    binary_map_shared,
    out_coords_d,
    out_indices_d,
    block_offset,
    start_index,
    num_elements,
    x_stride,
    y_stride,
    z_stride,
):
    """Step 4: Write to global output arrays with precomputed offsets."""
    thread_id_x = cuda.threadIdx.x
    thread_id_y = cuda.threadIdx.y
    thread_id_z = cuda.threadIdx.z
    block_dim_x = cuda.blockDim.x
    block_dim_y = cuda.blockDim.y
    block_dim_z = cuda.blockDim.z

    linear_index = (
        start_index
        + cuda.blockIdx.x * cuda.blockDim.x
        + cuda.blockIdx.y * cuda.blockDim.y * block_dim_x
        + cuda.blockIdx.z * cuda.blockDim.z * block_dim_x * block_dim_y
        + thread_id_x
        + thread_id_y * block_dim_x
        + thread_id_z * block_dim_x * block_dim_y
    )

    local_index_1d = (
        thread_id_z * block_dim_y * block_dim_x
        + thread_id_y * block_dim_x
        + thread_id_x
    )

    if (
        linear_index < start_index + num_elements
        and binary_map_shared[local_index_1d] == 1
    ):
        lz = linear_index % grid_shape_d[2]
        ly = (linear_index // grid_shape_d[2]) % grid_shape_d[1]
        lx = linear_index // (grid_shape_d[2] * grid_shape_d[1])

        lxyz = cuda.local.array(3, dtype=int32)
        lxyz[0] = lx
        lxyz[1] = ly
        lxyz[2] = lz
        xyz = (
            gridbox_center_d
            + (lxyz.astype(float32) - indices_mid_grid_d) * grid_spacing
        )

        local_count = cuda.shared.array(1, dtype=int32)
        if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0 and cuda.threadIdx.z == 0:
            local_count[0] = 0
        cuda.syncthreads()
        index_in_block = cuda.atomic.add(local_count, 0, 1)

        global_index = block_offset + index_in_block
        out_indices_d[global_index, 0] = lx
        out_indices_d[global_index, 1] = ly
        out_indices_d[global_index, 2] = lz
        out_coords_d[global_index, 0] = xyz[0]
        out_coords_d[global_index, 1] = xyz[1]
        out_coords_d[global_index, 2] = xyz[2]


@njit(nogil=True, boundscheck=False, cache=True)
def build_zeta_surface_map_cuda_parallel_block(
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
    """Two-step CUDA parallel implementation with block-level strategy."""
    block_dim = (TPB, TPB, 1)
    grid_dim_x = (grid_shape[0] + block_dim[0] - 1) // block_dim[0]
    grid_dim_y = (grid_shape[1] + block_dim[1] - 1) // block_dim[1]
    grid_dim_z = (grid_shape[2] + block_dim[2] - 1) // block_dim[2]
    grid_dim = (grid_dim_x, grid_dim_y, grid_dim_z)

    num_blocks = grid_dim_x * grid_dim_y * grid_dim_z
    num_threads_per_block = block_dim[0] * block_dim[1] * block_dim[2]
    total_elements = np.prod(grid_shape)

    grid_shape_d = cuda.to_device(np.array(grid_shape, dtype=np.int32))
    gridbox_center_d = cuda.to_device(gridbox_center.astype(np.float32))
    indices_mid_grid_d = cuda.to_device(indices_mid_grid.astype(np.float32))
    zeta_surface_map_1d_d = cuda.to_device(zeta_surface_map_1d.astype(np.bool_))
    index_discrete_epsilon_map_1d_d = cuda.to_device(
        index_discrete_epsilon_map_1d.astype(np.int32)
    )

    block_counts = cuda.device_array(num_blocks, dtype=np.int32)
    binary_map_shared = cuda.shared.array(block_dim, dtype=np.uint8)

    x_stride = grid_shape[1] * grid_shape[2]
    y_stride = grid_shape[2]
    z_stride = 1
    x_stride_x_3 = x_stride * 3
    y_stride_x_3 = y_stride * 3
    z_stride_x_3 = 3

    # Step 1: Build block-local binary map
    elements_per_block = num_threads_per_block
    for block_index in range(num_blocks):
        start_index = block_index * elements_per_block
        num_elements = min(elements_per_block, total_elements - start_index)
        grid_x = block_index % grid_dim_x
        grid_y = (block_index // grid_dim_x) % grid_dim_y
        grid_z = block_index // (grid_dim_x * grid_dim_y)
        current_block = (grid_x, grid_y, grid_z)
        _build_block_local_binary_map_cuda[current_block, block_dim](
            grid_shape_d,
            zeta_surface_map_1d_d,
            index_discrete_epsilon_map_1d_d,
            epsdim,
            binary_map_shared,
            np.int32(start_index),
            np.int32(num_elements),
            np.int32(x_stride),
            np.int32(y_stride),
            np.int32(z_stride),
            np.int32(x_stride_x_3),
            np.int32(y_stride_x_3),
            np.int32(z_stride_x_3),
        )

    # Step 2: Count zeta points per block
    count_grid_dim = (num_blocks + TPB - 1) // TPB
    _count_zeta_points_per_block_cuda[count_grid_dim, TPB](
        binary_map_shared, block_counts
    )

    # Step 3: Calculate offsets (on CPU for simplicity in this example)
    block_counts_host = block_counts.copy_to_host()
    offsets = np.zeros_like(block_counts_host)
    offsets[1:] = np.cumsum(block_counts_host[:-1])
    offsets_device = cuda.to_device(offsets.astype(np.int32))
    total_points = np.sum(block_counts_host)

    # Step 4: Write to global output arrays with offsets
    out_coords_d = cuda.device_array((total_points, 3), dtype=np.float32)
    out_indices_d = cuda.device_array((total_points, 3), dtype=np.int32)

    for block_index in range(num_blocks):
        start_index = block_index * elements_per_block
        num_elements = min(elements_per_block, total_elements - start_index)
        grid_x = block_index % grid_dim_x
        grid_y = (block_index // grid_dim_x) % grid_dim_y
        grid_z = block_index // (grid_dim_x * grid_dim_y)
        current_block = (grid_x, grid_y, grid_z)
        _build_block_local_binary_map_cuda[
            current_block, block_dim
        ](  # Re-run to have binary map in shared memory
            grid_shape_d,
            zeta_surface_map_1d_d,
            index_discrete_epsilon_map_1d_d,
            epsdim,
            binary_map_shared,
            np.int32(start_index),
            np.int32(num_elements),
            np.int32(x_stride),
            np.int32(y_stride),
            np.int32(z_stride),
            np.int32(x_stride_x_3),
            np.int32(y_stride_x_3),
            np.int32(z_stride_x_3),
        )
        _write_zeta_points_with_offset_cuda[current_block, block_dim](
            grid_shape_d,
            grid_spacing,
            gridbox_center_d,
            indices_mid_grid_d,
            binary_map_shared,
            out_coords_d,
            out_indices_d,
            offsets_device[block_index],
            np.int32(start_index),
            np.int32(num_elements),
            np.int32(x_stride),
            np.int32(y_stride),
            np.int32(z_stride),
        )

    final_coords = out_coords_d.copy_to_host()
    final_indices = out_indices_d.copy_to_host()

    return (
        final_coords,
        final_indices,
        total_points,
        total_points,
        final_coords.nbytes,
        final_indices.nbytes,
    )
