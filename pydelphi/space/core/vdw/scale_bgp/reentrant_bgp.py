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

#
# PyDelphi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# PyDelphi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#

import time
import math
import numpy as np
from numba import njit, prange, set_num_threads
from numba import cuda, float64, int32


from pydelphi.foundation.enums import Precision

from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
    nprint_cpu_if_verbose as nprint_cpu,
    vprint,
)
from pydelphi.config.logging_config import (
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

from pydelphi.constants import (
    ConstDelPhiFloats,
    ConstDelPhiInts,
)

# Define a small constant used for comparisons with zero.
APPROX_ZERO = ConstDelPhiFloats.ApproxZero.value
"""
A small constant used for comparisons with zero. Values smaller than this can be treated as numeric zero.
"""

# Define the maximum size of the neighbor list.
SPACE_NBRA_MAX_SIZE = ConstDelPhiInts.SpaceNBRASize.value
# Define the return value indicating an error in njit functions.
EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value

SHARED_SIZE = 512  # max threads per block we'll support in shared reduction
THREADS_PER_BLOCK = 512


@njit(nogil=True, boundscheck=False, cache=True)
def _find_closest_exposed_point(
    num_exposed_grid_points: delphi_int,
    s123: np.ndarray,
    system_min_coords: np.ndarray,
    cube_side_indver_length: delphi_real,
    cube_shape_indver: np.ndarray,
    cube_atom_index_start: np.ndarray,
    cube_atom_index_end: np.ndarray,
    cumulative_atom_index: np.ndarray,
    exposed_grid_point_coords: np.ndarray,
    is_outside_cube: np.ndarray,
    delphi_real_type: type,
    delphi_int_type: type,
) -> tuple[int, int]:
    """Finds the index of the closest solvent-exposed grid point to a given boundary point.

    Args:
        num_exposed_grid_points (delphi_int): Total number of solvent-exposed grid points.
        s123 (np.ndarray): Coordinates of the boundary point.
        system_min_coords (np.ndarray): Minimum coordinates of the system.
        cube_side_indver_length (delphi_real): The cube side length for indexing.
        cube_shape_indver (np.ndarray): Dimensions of the spatial partitioning cube for vertex indexing.
        cube_atom_index_start (np.ndarray): Array storing the starting index of exposed points in each cube.
        cube_atom_index_end (np.ndarray): Array storing the ending index of atoms in each cube.
        cumulative_atom_index (np.ndarray): Array mapping indices within cubes to global exposed point indices.
        exposed_grid_point_coords (np.ndarray): Array containing coordinates of all solvent-exposed grid points.
        is_outside_cube (np.ndarray): Boolean array indicating if a sub-cube is outside the main cube.
        delphi_real_type (type): Data type for real numbers.
        delphi_int_type (type): Data type for integers.

    Returns:
        closest_exposed_point_index (int): The index of the closest solvent-exposed grid point.
        is_fallthrough_point (delphi_int): 1 if a fallthrough-bgp-point else 0.
    """
    cube_side_indver_length_inverse = 1.0 / cube_side_indver_length
    # Calculate the relative coordinates of the boundary point within the system.
    relative_coords = (s123 - system_min_coords) * cube_side_indver_length_inverse
    # Calculate the integer grid indices of the boundary point.
    grid_indices_s123 = relative_coords.astype(delphi_int_type)
    grid_index_x = grid_indices_s123[0]
    grid_index_y = grid_indices_s123[1]
    grid_index_z = grid_indices_s123[2]
    # Calculate the fractional part of the relative coordinates.
    delta_coords_s123 = relative_coords - grid_indices_s123
    # Find the minimum and maximum fractional components.
    min_delta = min(
        delta_coords_s123[0], min(delta_coords_s123[1], delta_coords_s123[2])
    )
    max_delta = max(
        delta_coords_s123[0], max(delta_coords_s123[1], delta_coords_s123[2])
    )
    # Calculate the complement of the maximum fractional component.
    min_delta_complement = 1.0 - max_delta
    # Calculate a corner distance ratio.
    corner_distance_ratio = min(min_delta, min_delta_complement)
    # Calculate a cutoff threshold based on the corner distance.
    cutoff_threshold = cube_side_indver_length * (1 + corner_distance_ratio)
    cutoff_threshold_squared = cutoff_threshold**2
    closest_exposed_point_index = 0
    min_distance_to_exposed_squared = delphi_real_type(100.0)

    for extension_index in range(2):
        if extension_index == 0:
            off_min, off_max = -1, 1  # immidiate cube-voxels neighbor to search
        else:
            off_min, off_max = -2, 2  # extended cube-voxels neighbor to search
        for neighbor_x_offset in range(off_min, off_max + 1):
            for neighbor_y_offset in range(off_min, off_max + 1):
                for neighbor_z_offset in range(off_min, off_max + 1):
                    neighbor_x_index = grid_index_x + neighbor_x_offset
                    neighbor_y_index = grid_index_y + neighbor_y_offset
                    neighbor_z_index = grid_index_z + neighbor_z_offset

                    # Check if the neighboring cube indices are within bounds.
                    if (
                        0 <= neighbor_x_index <= cube_shape_indver[0]
                        and 0 <= neighbor_y_index <= cube_shape_indver[1]
                        and 0 <= neighbor_z_index <= cube_shape_indver[2]
                    ):
                        # For the extended search, check if the sub-cube is outside the main cube.
                        if extension_index == 1:
                            sub_cube_offset_x = neighbor_x_offset + 2
                            sub_cube_offset_y = neighbor_y_offset + 2
                            sub_cube_offset_z = neighbor_z_offset + 2
                            if not is_outside_cube[sub_cube_offset_x][
                                sub_cube_offset_y
                            ][sub_cube_offset_z]:
                                continue

                        # Iterate through the exposed points within the current neighboring cube.
                        for ii in range(
                            cube_atom_index_start[neighbor_x_index][neighbor_y_index][
                                neighbor_z_index
                            ],
                            cube_atom_index_end[neighbor_x_index][neighbor_y_index][
                                neighbor_z_index
                            ]
                            + 1,
                        ):
                            # Get the global index of the exposed point.
                            exposed_index = cumulative_atom_index[ii]
                            # Calculate the vector from the boundary point to the exposed point.
                            delta_coords_exposed_to_check = (
                                s123 - exposed_grid_point_coords[exposed_index]
                            )
                            # Calculate the squared distance.
                            distance_squared_to_exposed = (
                                delta_coords_exposed_to_check[0]
                                * delta_coords_exposed_to_check[0]
                                + delta_coords_exposed_to_check[1]
                                * delta_coords_exposed_to_check[1]
                                + delta_coords_exposed_to_check[2]
                                * delta_coords_exposed_to_check[2]
                            )
                            # Update the closest exposed point if a closer one is found.
                            if (
                                distance_squared_to_exposed
                                < min_distance_to_exposed_squared
                            ):
                                min_distance_to_exposed_squared = (
                                    distance_squared_to_exposed
                                )
                                closest_exposed_point_index = exposed_index
        # Break after the first extension if a close point is found within the cutoff.
        if (
            closest_exposed_point_index > 0
            and min_distance_to_exposed_squared < cutoff_threshold_squared
        ):
            break

    # Final fallback search: iterate through all exposed points if no close point is found.
    is_fallthrough_point = 0
    if closest_exposed_point_index <= 0:
        is_fallthrough_point = 1
        for exposed_index in range(
            num_exposed_grid_points
        ):  # Corrected range to 0-based indexing
            delta_coords_exposed_to_check = (
                s123 - exposed_grid_point_coords[exposed_index]
            )
            distance_squared_to_exposed = (
                delta_coords_exposed_to_check[0] * delta_coords_exposed_to_check[0]
                + delta_coords_exposed_to_check[1] * delta_coords_exposed_to_check[1]
                + delta_coords_exposed_to_check[2] * delta_coords_exposed_to_check[2]
            )
            if distance_squared_to_exposed < min_distance_to_exposed_squared:
                min_distance_to_exposed_squared = distance_squared_to_exposed
                closest_exposed_point_index = exposed_index

    return closest_exposed_point_index, is_fallthrough_point


@cuda.jit(device=True, inline="always")
def _cuda_find_closest_exposed_point(
    num_exposed_grid_points: delphi_int,
    s123: np.ndarray,
    system_min_coords: np.ndarray,
    cube_side_indver_length: delphi_real,
    cube_shape_indver: np.ndarray,
    cube_atom_index_start: np.ndarray,
    cube_atom_index_end: np.ndarray,
    cumulative_atom_index: np.ndarray,
    exposed_grid_point_coords: np.ndarray,
    is_outside_cube: np.ndarray,
) -> tuple[int, int]:
    """Finds the index of the closest solvent-exposed grid point to a given boundary point.

    Args:
        num_exposed_grid_points (delphi_int): Total number of solvent-exposed grid points.
        s123 (np.ndarray): Coordinates of the boundary point.
        system_min_coords (np.ndarray): Minimum coordinates of the system.
        cube_side_indver_length (delphi_real): The cube side length for indexing.
        cube_shape_indver (np.ndarray): Dimensions of the spatial partitioning cube for vertex indexing.
        cube_atom_index_start (np.ndarray): Array storing the starting index of exposed points in each cube.
        cube_atom_index_end (np.ndarray): Array storing the ending index of atoms in each cube.
        cumulative_atom_index (np.ndarray): Array mapping indices within cubes to global exposed point indices.
        exposed_grid_point_coords (np.ndarray): Array containing coordinates of all solvent-exposed grid points.
        is_outside_cube (np.ndarray): Boolean array indicating if a sub-cube is outside the main cube.
        delphi_real_type (type): Data type for real numbers.
        delphi_int_type (type): Data type for integers.

    Returns:
        closest_exposed_point_index (int): The index of the closest solvent-exposed grid point.
        is_fallthrough_point (delphi_int): 1 if a fallthrough-bgp-point else 0.
    """
    cube_side_indver_length_inverse = 1.0 / cube_side_indver_length
    # Calculate the relative coordinates of the boundary point within the system.
    relative_coords_x = (
        s123[0] - system_min_coords[0]
    ) * cube_side_indver_length_inverse
    relative_coords_y = (
        s123[1] - system_min_coords[1]
    ) * cube_side_indver_length_inverse
    relative_coords_z = (
        s123[2] - system_min_coords[2]
    ) * cube_side_indver_length_inverse

    # Calculate the integer grid indices of the boundary point.
    grid_index_x = int(relative_coords_x)
    grid_index_y = int(relative_coords_y)
    grid_index_z = int(relative_coords_z)

    # Calculate the fractional part of the relative coordinates.
    delta_coords_s123_x = relative_coords_x - grid_index_x
    delta_coords_s123_y = relative_coords_y - grid_index_y
    delta_coords_s123_z = relative_coords_z - grid_index_z

    # Find the minimum and maximum fractional components.
    min_delta = min(delta_coords_s123_x, min(delta_coords_s123_y, delta_coords_s123_z))
    max_delta = max(delta_coords_s123_x, max(delta_coords_s123_y, delta_coords_s123_z))
    # Calculate the complement of the maximum fractional component.
    min_delta_complement = 1.0 - max_delta
    # Calculate a corner distance ratio.
    corner_distance_ratio = min(min_delta, min_delta_complement)
    # Calculate a cutoff threshold based on the corner distance.
    cutoff_threshold = cube_side_indver_length * (1 + corner_distance_ratio)
    cutoff_threshold_squared = cutoff_threshold**2
    closest_exposed_point_index = 0
    min_distance_to_exposed_squared = float64(100.0)

    for extension_index in range(2):
        if extension_index == 0:
            off_min, off_max = -1, 1  # immidiate cube-voxels neighbor to search
        else:
            off_min, off_max = -2, 2  # extended cube-voxels neighbor to search
        for neighbor_x_offset in range(off_min, off_max + 1):
            for neighbor_y_offset in range(off_min, off_max + 1):
                for neighbor_z_offset in range(off_min, off_max + 1):
                    neighbor_x_index = grid_index_x + neighbor_x_offset
                    neighbor_y_index = grid_index_y + neighbor_y_offset
                    neighbor_z_index = grid_index_z + neighbor_z_offset

                    # Check if the neighboring cube indices are within bounds.
                    if (
                        0 <= neighbor_x_index <= cube_shape_indver[0]
                        and 0 <= neighbor_y_index <= cube_shape_indver[1]
                        and 0 <= neighbor_z_index <= cube_shape_indver[2]
                    ):
                        # For the extended search, check if the sub-cube is outside the main cube.
                        if extension_index == 1:
                            sub_cube_offset_x = neighbor_x_offset + 2
                            sub_cube_offset_y = neighbor_y_offset + 2
                            sub_cube_offset_z = neighbor_z_offset + 2
                            if not is_outside_cube[sub_cube_offset_x][
                                sub_cube_offset_y
                            ][sub_cube_offset_z]:
                                continue

                        # Iterate through the exposed points within the current neighboring cube.
                        for ii in range(
                            cube_atom_index_start[neighbor_x_index][neighbor_y_index][
                                neighbor_z_index
                            ],
                            cube_atom_index_end[neighbor_x_index][neighbor_y_index][
                                neighbor_z_index
                            ]
                            + 1,
                        ):
                            # Get the global index of the exposed point.
                            exposed_index = cumulative_atom_index[ii]
                            # Calculate the vector from the boundary point to the exposed point.
                            delta_coords_exposed_to_check_x = (
                                s123[0] - exposed_grid_point_coords[exposed_index][0]
                            )
                            delta_coords_exposed_to_check_y = (
                                s123[1] - exposed_grid_point_coords[exposed_index][1]
                            )
                            delta_coords_exposed_to_check_z = (
                                s123[2] - exposed_grid_point_coords[exposed_index][2]
                            )
                            # Calculate the squared distance.
                            distance_squared_to_exposed = (
                                delta_coords_exposed_to_check_x
                                * delta_coords_exposed_to_check_x
                                + delta_coords_exposed_to_check_y
                                * delta_coords_exposed_to_check_y
                                + delta_coords_exposed_to_check_z
                                * delta_coords_exposed_to_check_z
                            )
                            # Update the closest exposed point if a closer one is found.
                            if (
                                distance_squared_to_exposed
                                < min_distance_to_exposed_squared
                            ):
                                min_distance_to_exposed_squared = (
                                    distance_squared_to_exposed
                                )
                                closest_exposed_point_index = exposed_index
        # Break after the first extension if a close point is found within the cutoff.
        if (
            closest_exposed_point_index > 0
            and min_distance_to_exposed_squared < cutoff_threshold_squared
        ):
            break

    # Final fallback search: iterate through all exposed points if no close point is found.
    is_fallthrough_point = 0
    if closest_exposed_point_index <= 0:
        is_fallthrough_point = 1
        for exposed_index in range(
            num_exposed_grid_points
        ):  # Corrected range to 0-based indexing
            delta_coords_exposed_to_check_x = (
                s123[0] - exposed_grid_point_coords[exposed_index][0]
            )
            delta_coords_exposed_to_check_y = (
                s123[1] - exposed_grid_point_coords[exposed_index][1]
            )
            delta_coords_exposed_to_check_z = (
                s123[2] - exposed_grid_point_coords[exposed_index][2]
            )
            # Calculate the squared distance.
            distance_squared_to_exposed = (
                delta_coords_exposed_to_check_x * delta_coords_exposed_to_check_x
                + delta_coords_exposed_to_check_y * delta_coords_exposed_to_check_y
                + delta_coords_exposed_to_check_z * delta_coords_exposed_to_check_z
            )
            if distance_squared_to_exposed < min_distance_to_exposed_squared:
                min_distance_to_exposed_squared = distance_squared_to_exposed
                closest_exposed_point_index = exposed_index

    return closest_exposed_point_index, is_fallthrough_point


@njit(cache=True, nogil=True, parallel=True)
def _cpu_scale_reentrant_boundary_points(
    num_threads,
    num_boundary_points,
    surface_charge_positions,
    scaled_surface_normal_vectors,
    atom_index_for_boundary,
    grid_spacing,
    current_grid_origin,
    num_exposed_grid_points,
    system_min_coords,
    cube_side_length_index_vertices,
    cube_shape_indver,
    cube_voxel_atom_index_start,
    cube_voxel_atom_index_end,
    cube_voxel_atom_index_cumulative,
    exposed_grid_point_coords,
    is_outside_cube,
    probe_radius,
    probe_radius_2,
    delphi_real,
    delphi_int,
):
    local_counts = np.zeros(num_threads, dtype=delphi_int)
    block_size = (num_boundary_points + num_threads - 1) // num_threads

    # Initialize thread-local arrays outside the inner loop to eliminate temporary allocations
    # This must be done inside the parallel block (prange loop) to ensure thread-safety

    for thread_id in prange(num_threads):
        start = thread_id * block_size
        end = min(start + block_size, num_boundary_points)

        # 💡 Optimization: Thread-local, reusable, pre-allocated arrays
        s123 = np.zeros(3, dtype=delphi_real)
        delta_coords = np.zeros(3, dtype=delphi_real)

        local_count = 0
        for bgp_index in range(start, end):
            atom_idx = atom_index_for_boundary[bgp_index]

            # Use combined checks for early exit
            if atom_idx != 0:
                continue

            # --- 1. Compute current boundary point coords ---
            # Use scalar operations (faster than vector assignment for small fixed-size arrays)
            for i in range(3):
                s123[i] = (
                    surface_charge_positions[bgp_index, i] * grid_spacing
                    + current_grid_origin[i]
                )

            # --- 2. Find the closest solvent-exposed point ---
            closest_exposed_point_index, is_fallthrough_point = (
                _find_closest_exposed_point(
                    num_exposed_grid_points,
                    s123,
                    system_min_coords,
                    cube_side_length_index_vertices,
                    cube_shape_indver,
                    cube_voxel_atom_index_start,
                    cube_voxel_atom_index_end,
                    cube_voxel_atom_index_cumulative,
                    exposed_grid_point_coords,
                    is_outside_cube,
                    delphi_real,
                    delphi_int,
                )
            )
            local_count += is_fallthrough_point

            # --- 3. Compute delta vector and distance squared ---
            dist2 = 0.0

            # 💡 Optimization: Calculate delta_coords and dist2 in a single loop
            closest_coords = exposed_grid_point_coords[closest_exposed_point_index]
            for i in range(3):
                # Calculate delta and store in pre-allocated array
                delta_coords[i] = s123[i] - closest_coords[i]
                dist2 += delta_coords[i] * delta_coords[i]

            # --- 4. Calculate scaling factor ---
            if dist2 < APPROX_ZERO:
                # No scaling needed, early exit for inv_dist and scaling_factor calculation
                inv_dist = 0.0
                scaling_factor = 0.0
            else:
                inv_dist = 1.0 / np.sqrt(dist2)
                # 💡 Optimization: Combine probe_radius_2 division with inv_dist multiplication
                scaling_factor = probe_radius_2 * inv_dist

            # --- 5. Move boundary point & Set scaled surface normal ---
            if inv_dist > 0.0:
                # 💡 Optimization: Combine both updates into one loop for better cache use
                for i in range(3):
                    # Move boundary point (Updates surface_charge_positions)
                    surface_charge_positions[bgp_index, i] = (
                        closest_coords[i] + delta_coords[i] * scaling_factor
                    )

                    # Set scaled surface normal (Updates scaled_surface_normal_vectors)
                    scaled_surface_normal_vectors[bgp_index, i] = (
                        -delta_coords[i] * inv_dist
                    )
            elif dist2 < APPROX_ZERO:
                # If dist2 is zero, the point moves to the exposed point,
                # but normal must be zeroed (it may contain garbage from prior steps)
                for i in range(3):
                    surface_charge_positions[bgp_index, i] = closest_coords[i]
                    scaled_surface_normal_vectors[bgp_index, i] = 0.0

        local_counts[thread_id] = local_count

    num_fallthrough_bgps = np.sum(local_counts)
    return num_fallthrough_bgps


@cuda.jit(device=True, inline="always")
def _cuda_find_closest_exposed_point(
    sx,
    sy,
    sz,  # scalar coordinates (floats)
    num_exposed_grid_points,  # int scalar
    system_min_coords,  # float64[3]
    cube_side_indver_length,  # float scalar
    cube_shape_indver,  # int32[3] shape (nx,ny,nz)
    cube_atom_index_start,  # int32[:,:,:] start indices (per-voxel)
    cube_atom_index_end,  # int32[:,:,:] inclusive end indices
    cumulative_atom_index,  # int32[:] mapping within cube lists -> global exposed index
    exposed_grid_point_coords,  # float64[:,3] global coords
    is_outside_cube,  # bool/int array shape >= [5,5,5] for offsets -2..2
):
    """
    Finds the index of the closest solvent-exposed grid point to a given boundary point.

    Args:
        sx (delphi_real): scalar x-coordinate.
        sy (delphi_real): scalar y-coordinate.
        sz (delphi_real): scalar z-coordinate.
        num_exposed_grid_points (delphi_int): Total number of solvent-exposed grid points.
        system_min_coords (np.ndarray): Minimum coordinates of the system.
        cube_side_indver_length (delphi_real): The cube side length for indexing.
        cube_shape_indver (np.ndarray): Dimensions of the spatial partitioning cube for vertex indexing.
        cube_atom_index_start (np.ndarray): Array storing the starting index of exposed points in each cube.
        cube_atom_index_end (np.ndarray): Array storing the ending index of atoms in each cube.
        cumulative_atom_index (np.ndarray): Array mapping indices within cubes to global exposed point indices.
        exposed_grid_point_coords (np.ndarray): Array containing coordinates of all solvent-exposed grid points.
        is_outside_cube (np.ndarray): Boolean array indicating if a sub-cube is outside the main cube.

    Returns:
        closest_exposed_point_index (int): The index of the closest solvent-exposed grid point.
        is_fallthrough_point (delphi_int): 1 if a fallthrough-bgp-point else 0.
    """
    inv_len = 1.0 / cube_side_indver_length

    # relative indices (scalar)
    rel_x = (sx - system_min_coords[0]) * inv_len
    rel_y = (sy - system_min_coords[1]) * inv_len
    rel_z = (sz - system_min_coords[2]) * inv_len

    gx = int(rel_x)
    gy = int(rel_y)
    gz = int(rel_z)

    fx = rel_x - gx
    fy = rel_y - gy
    fz = rel_z - gz

    min_delta = fx if fx < fy else fy
    min_delta = min_delta if min_delta < fz else fz
    max_delta = fx if fx > fy else fy
    max_delta = max_delta if max_delta > fz else fz

    corner_distance_ratio = min(min_delta, 1.0 - max_delta)
    cutoff_threshold = cube_side_indver_length * (1.0 + corner_distance_ratio)
    cutoff_threshold_squared = cutoff_threshold * cutoff_threshold

    closest_exposed_point_index = -1
    min_distance_to_exposed_squared = 1e300

    # two extension levels
    for extension_index in range(2):
        if extension_index == 0:
            off_min = -1
            off_max = 1
        else:
            off_min = -2
            off_max = 2

        # iterate neighborhood
        for off_x in range(off_min, off_max + 1):
            nx = gx + off_x
            if nx < 0 or nx >= cube_shape_indver[0]:
                continue
            for off_y in range(off_min, off_max + 1):
                ny = gy + off_y
                if ny < 0 or ny >= cube_shape_indver[1]:
                    continue
                for off_z in range(off_min, off_max + 1):
                    nz = gz + off_z
                    if nz < 0 or nz >= cube_shape_indver[2]:
                        continue

                    if extension_index == 1:
                        sub_x = off_x + 2
                        sub_y = off_y + 2
                        sub_z = off_z + 2
                        # bounds check for is_outside_cube must be ensured by caller
                        if not is_outside_cube[sub_x, sub_y, sub_z]:
                            continue

                    start_idx = cube_atom_index_start[nx, ny, nz]
                    end_idx = cube_atom_index_end[nx, ny, nz]

                    if start_idx > end_idx:
                        continue

                    ii = start_idx
                    while ii <= end_idx:
                        exposed_index = cumulative_atom_index[ii]
                        ex = exposed_grid_point_coords[exposed_index, 0]
                        ey = exposed_grid_point_coords[exposed_index, 1]
                        ez = exposed_grid_point_coords[exposed_index, 2]
                        dx = sx - ex
                        dy = sy - ey
                        dz = sz - ez
                        d2 = dx * dx + dy * dy + dz * dz

                        if d2 < min_distance_to_exposed_squared:
                            min_distance_to_exposed_squared = d2
                            closest_exposed_point_index = int(exposed_index)
                        ii += 1

        # early exit check
        if (
            closest_exposed_point_index >= 0
            and min_distance_to_exposed_squared < cutoff_threshold_squared
        ):
            return closest_exposed_point_index, 0

    # fallback full scan
    is_fallthrough_point = 0
    if closest_exposed_point_index < 0:
        is_fallthrough_point = 1
        for exposed_index in range(num_exposed_grid_points):
            ex = exposed_grid_point_coords[exposed_index, 0]
            ey = exposed_grid_point_coords[exposed_index, 1]
            ez = exposed_grid_point_coords[exposed_index, 2]
            dx = sx - ex
            dy = sy - ey
            dz = sz - ez
            d2 = dx * dx + dy * dy + dz * dz
            if d2 < min_distance_to_exposed_squared:
                min_distance_to_exposed_squared = d2
                closest_exposed_point_index = int(exposed_index)

    return closest_exposed_point_index, is_fallthrough_point


# -------------------------
# Kernel: grid-stride + per-block reduction
# -------------------------
@cuda.jit(cache=True)
def _cuda_scale_reentrant_boundary_points_gridstride(
    num_boundary_points,
    surface_charge_positions,  # float64[:,3]
    scaled_surface_normal_vectors,  # float64[:,3]
    atom_index_for_boundary,  # int32[:]
    grid_spacing,  # float
    current_grid_origin,  # float64[3]
    num_exposed_grid_points,
    system_min_coords,  # float64[3]
    cube_side_length_index_vertices,  # float
    cube_shape_indver,  # int32[3]
    cube_voxel_atom_index_start,  # int32[:,:,:]
    cube_voxel_atom_index_end,  # int32[:,:,:] inclusive
    cube_voxel_atom_index_cumulative,  # int32[:]
    exposed_grid_point_coords,  # float64[:,3]
    is_outside_cube,  # bool/int[5,5,5]
    probe_radius,
    probe_radius_2,
    num_fallthrough_bgps,  # int32[1] - single scalar array for atomic adds
):
    # thread / block indices
    tid = cuda.threadIdx.x
    bdim = cuda.blockDim.x
    bid = cuda.blockIdx.x
    gdim = cuda.gridDim.x

    gidx = tid + bid * bdim
    stride = bdim * gdim

    # each thread accumulates local_sum in register
    local_sum = 0

    idx = gidx
    while idx < num_boundary_points:
        atom_idx = atom_index_for_boundary[idx]
        if atom_idx == 0:
            # compute s123 scalars
            sx = (
                surface_charge_positions[idx, 0] * grid_spacing + current_grid_origin[0]
            )
            sy = (
                surface_charge_positions[idx, 1] * grid_spacing + current_grid_origin[1]
            )
            sz = (
                surface_charge_positions[idx, 2] * grid_spacing + current_grid_origin[2]
            )

            # call device find (scalar variant)
            closest_idx, is_fallthrough_bgp = _cuda_find_closest_exposed_point(
                sx,
                sy,
                sz,
                num_exposed_grid_points,
                system_min_coords,
                cube_side_length_index_vertices,
                cube_shape_indver,
                cube_voxel_atom_index_start,
                cube_voxel_atom_index_end,
                cube_voxel_atom_index_cumulative,
                exposed_grid_point_coords,
                is_outside_cube,
            )

            # if a closest found, apply scaling & write back
            if closest_idx >= 0:
                ex = exposed_grid_point_coords[closest_idx, 0]
                ey = exposed_grid_point_coords[closest_idx, 1]
                ez = exposed_grid_point_coords[closest_idx, 2]
                dx = sx - ex
                dy = sy - ey
                dz = sz - ez
                dist2 = dx * dx + dy * dy + dz * dz

                if dist2 < APPROX_ZERO:
                    distance = 0.0
                    scaling_factor = 0.0
                else:
                    distance = math.sqrt(dist2)
                    scaling_factor = probe_radius_2 / distance

                # write back (coalescing depends on idx layout)
                surface_charge_positions[idx, 0] = ex + dx * scaling_factor
                surface_charge_positions[idx, 1] = ey + dy * scaling_factor
                surface_charge_positions[idx, 2] = ez + dz * scaling_factor

                if distance > APPROX_ZERO:
                    invd = 1.0 / distance
                    scaled_surface_normal_vectors[idx, 0] = -dx * invd
                    scaled_surface_normal_vectors[idx, 1] = -dy * invd
                    scaled_surface_normal_vectors[idx, 2] = -dz * invd
                else:
                    scaled_surface_normal_vectors[idx, 0] = 0.0
                    scaled_surface_normal_vectors[idx, 1] = 0.0
                    scaled_surface_normal_vectors[idx, 2] = 0.0

            local_sum += is_fallthrough_bgp
        # next grid-stride work item
        idx += stride

    # -------------------------
    # block-level reduction of local_sum -> one partial sum per block
    # -------------------------
    # static-size shared array (must be compile-time const)
    sdata = cuda.shared.array(SHARED_SIZE, int32)

    # bounds: threads may be > SHARED_SIZE in pathological configs, but we choose config to avoid that
    if tid < SHARED_SIZE:
        sdata[tid] = local_sum
    else:
        # if tid >= SHARED_SIZE, clamp to 0 (this shouldn't happen if threads_per_block <= SHARED_SIZE)
        sdata[tid % SHARED_SIZE] = local_sum

    cuda.syncthreads()

    # tree-reduction in shared mem (power-of-two-friendly)
    stride_sh = bdim // 2
    while stride_sh > 0:
        if tid < stride_sh:
            sdata[tid] += sdata[tid + stride_sh]
        cuda.syncthreads()
        stride_sh //= 2

    # thread 0 performs a single atomic add for this block
    if tid == 0:
        cuda.atomic.add(num_fallthrough_bgps, 0, sdata[0])
