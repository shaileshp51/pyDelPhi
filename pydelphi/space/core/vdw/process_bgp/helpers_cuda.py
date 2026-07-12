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

# pydelphi/space/core/vdw/process_bgp/helpers_cuda.py

import math
from numba import cuda

from pydelphi.constants import (
    NON_SOLUTE_BOUNDARY,
    SOLUTE_BOUNDARY_ANY,
    SOLUTE_BOUNDARY_EXTERNAL,
    ConstDelPhiInts,
    ATOMFIELD_X,
    ATOMFIELD_CRD_END,  # kept for parity/import sanity
    ATOMFIELD_RADIUS,
    ATOMFIELD_MEDIA_ID,
    ConstDelPhiFloats as ConstDelPhi,
    NEIGHBOR_VOXEL_RELATIVE_COORDINATES as NEIGHBOR_VOXEL_REL_COORDS,
)

EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value
APPROX_ZERO = ConstDelPhi.ApproxZero.value
MAX_NEIGHBR_BOUNDARY_ARRAY_LEN = ConstDelPhiInts.SpaceNBRASize.value

# (Optional but recommended) move neighbor offsets into constant memory
# NEIGHBOR_VOXEL_REL_COORDS_C = cuda.const.array_like(NEIGHBOR_VOXEL_REL_COORDS)


# -----------------------------------------------------------------------------
# Basic device utilities
# -----------------------------------------------------------------------------


@cuda.jit(device=True, inline=True)
def bgp_color_id_dev(ix: int, iy: int, iz: int) -> int:
    return (ix % 3) + 3 * (iy % 3) + 9 * (iz % 3)


@cuda.jit(device=True, inline=True)
def idx1d_from_ijk_dev(ix: int, iy: int, iz: int, ny_nz: int, nz: int) -> int:
    return ix * ny_nz + iy * nz + iz


@cuda.jit(device=True, inline=True)
def ijk_from_idx1d_dev(idx1d: int, x_stride: int, y_stride: int):
    ix = idx1d // x_stride
    rem = idx1d - ix * x_stride
    iy = rem // y_stride
    iz = rem - iy * y_stride
    return ix, iy, iz


@cuda.jit(device=True, inline=True)
def dot3_dev(dx, dy, dz):
    return dx * dx + dy * dy + dz * dz


@cuda.jit(device=True, inline=True)
def abs_i32(x):
    return -x if x < 0 else x


@cuda.jit(device=True, inline=True)
def stamp_claim_dev(stamp_arr, idx, stamp):
    """
    Claim idx for this stamp using CAS.
    REQUIREMENTS:
      - stamp_arr dtype int32 (or int64) on device
      - stamp scalar same width/type
    """
    old = stamp_arr[idx]
    if old == stamp:
        return False
    prev = cuda.atomic.cas(stamp_arr, idx, old, stamp)
    return prev == old


# -----------------------------------------------------------------------------
# Classification: equivalent to CPU neighbor-media scan
# -----------------------------------------------------------------------------

#
# @cuda.jit(device=True, inline=True)
# def classify_point_from_epsmap_dev(
#     idx1d: int,
#     epsilon_dimension: int,
#     epsmap,
#     x_stride_x_3: int,
#     y_stride_x_3: int,
# ):
#     """
#     epsmap is int32-like.
#     Reads 6 midpoint entries around this grid point (as CPU does).
#     """
#     base = idx1d * 3
#     epsdim = epsilon_dimension
#
#     # offsets in the *midpoint* (x3) layout
#     o1 = 0
#     o2 = 1
#     o3 = 2
#     o4 = -x_stride_x_3
#     o5 = -y_stride_x_3 + 1
#     o6 = -1  # NOTE: assumes z_stride_x_3 == 3, i.e. contiguous in z
#
#     v1 = abs_i32(epsmap[base + o1]) // epsdim
#     v2 = abs_i32(epsmap[base + o2]) // epsdim
#     v3 = abs_i32(epsmap[base + o3]) // epsdim
#     v4 = abs_i32(epsmap[base + o4]) // epsdim
#     v5 = abs_i32(epsmap[base + o5]) // epsdim
#     v6 = abs_i32(epsmap[base + o6]) // epsdim
#
#     is_external = (
#         1
#         if ((v1 == 0) or (v2 == 0) or (v3 == 0) or (v4 == 0) or (v5 == 0) or (v6 == 0))
#         else 0
#     )
#
#     is_boundary = 0
#     if v1 != v6:
#         is_boundary = 1
#     if (v2 != v1) or (v3 != v2) or (v4 != v3) or (v5 != v4) or (v6 != v5):
#         is_boundary = 1
#
#     return is_boundary, is_external

# @cuda.jit(device=True, inline=True)
# def classify_point_from_epsmap_dev(
#     idx1d: int,
#     epsilon_dimension: int,
#     epsmap,
#     x_stride_x_3: int,
#     y_stride_x_3: int,
# ):
#     base = idx1d * 3
#     epsdim = epsilon_dimension
#
#     o1 = 0
#     o2 = 1
#     o3 = 2
#     o4 = -x_stride_x_3
#     o5 = -y_stride_x_3 + 1
#     o6 = -1  # explicit "previous z plane, flag=3" in x3 layout => -1
#
#     v1 = abs_i32(epsmap[base + o1]) // epsdim
#     v2 = abs_i32(epsmap[base + o2]) // epsdim
#     v3 = abs_i32(epsmap[base + o3]) // epsdim
#     v4 = abs_i32(epsmap[base + o4]) // epsdim
#     v5 = abs_i32(epsmap[base + o5]) // epsdim
#     v6 = abs_i32(epsmap[base + o6]) // epsdim
#
#     is_external = 1 if (v1 == 0 or v2 == 0 or v3 == 0 or v4 == 0 or v5 == 0 or v6 == 0) else 0
#
#     is_boundary = 0
#     if v1 != v6:
#         is_boundary = 1
#     if (v2 != v1) or (v3 != v2) or (v4 != v3) or (v5 != v4) or (v6 != v5):
#         is_boundary = 1
#
#     return is_boundary, is_external


@cuda.jit(device=True, inline=True)
def classify_point_from_epsmap_dev(
    idx1d,
    epsilon_dimension,
    epsmap,
    x_stride_x3,
    y_stride_x3,
):
    base = int(idx1d * 3)
    epsdim = int(epsilon_dimension)

    # Forward midpoints (always safe if idx1d is valid)
    v1 = int(abs_i32(epsmap[base + 0]) // epsdim)
    v2 = int(abs_i32(epsmap[base + 1]) // epsdim)
    v3 = int(abs_i32(epsmap[base + 2]) // epsdim)

    # Backward midpoints - GUARDED against negative indices for GPU parity
    # If index would be < 0, we fallback to v1/v2/v3 to ensure "no boundary"
    # detection at the absolute grid floor, matching safe CPU behavior.

    idx4 = base - x_stride_x3
    v4 = int(abs_i32(epsmap[idx4]) // epsdim) if idx4 >= 0 else v1

    idx5 = base - y_stride_x3 + 1
    v5 = int(abs_i32(epsmap[idx5]) // epsdim) if idx5 >= 0 else v2

    idx6 = base - 1
    v6 = int(abs_i32(epsmap[idx6]) // epsdim) if idx6 >= 0 else v3

    is_external = 0
    if (v1 == 0) or (v2 == 0) or (v3 == 0) or (v4 == 0) or (v5 == 0) or (v6 == 0):
        is_external = 1

    is_boundary = 0
    # Exact CPU Comparison Chain
    if v1 != v6:
        is_boundary = 1
    if (v2 != v1) or (v3 != v2) or (v4 != v3) or (v5 != v4) or (v6 != v5):
        is_boundary = 1

    return is_boundary, is_external


# -----------------------------------------------------------------------------
# VdW-to-MS exposed-grid scanning (GPU port)
# -----------------------------------------------------------------------------


@cuda.jit(device=True)
def vdw_to_ms_all_voxels_buffer_dev(
    cycle_flag: int,  # 0/1
    midpoint_index: int,
    grid_x_index: int,
    grid_y_index: int,
    grid_z_index: int,
    probe_radius_squared: float,
    grid_midpoint_coords,  # (3,)
    grid_neighs_entity_ids,  # (>=7,)
    grid_neighs_media_ids,  # (>=7,)
    closest_exposed_grid_index: int,
    minimum_distance: float,
    cube_voxel_start_indices,  # 3D
    cube_voxel_end_indices,  # 3D
    grid_point_indices_in_voxels,  # 1D
    exposed_grids_coords,  # (N,3)
):
    min_d2 = minimum_distance

    pvx = 0
    pvy = 0
    pvz = 0

    mx0 = grid_midpoint_coords[0]
    mx1 = grid_midpoint_coords[1]
    mx2 = grid_midpoint_coords[2]

    # If you constant-memory the table, swap to NEIGHBOR_VOXEL_REL_COORDS_C
    for t in range(NEIGHBOR_VOXEL_REL_COORDS.shape[0]):
        vx = int(NEIGHBOR_VOXEL_REL_COORDS[t, 0])
        vy = int(NEIGHBOR_VOXEL_REL_COORDS[t, 1])
        vz = int(NEIGHBOR_VOXEL_REL_COORDS[t, 2])

        grid_x_index = grid_x_index + (vx - pvx)
        grid_y_index = grid_y_index + (vy - pvy)
        grid_z_index = grid_z_index + (vz - pvz)

        lower = int(cube_voxel_start_indices[grid_x_index, grid_y_index, grid_z_index])
        upper = int(cube_voxel_end_indices[grid_x_index, grid_y_index, grid_z_index])

        cycle_flag = 0
        for k in range(lower, upper + 1):
            eg_idx = int(grid_point_indices_in_voxels[k])

            dx = mx0 - exposed_grids_coords[eg_idx, 0]
            dy = mx1 - exposed_grids_coords[eg_idx, 1]
            dz = mx2 - exposed_grids_coords[eg_idx, 2]
            d2 = dot3_dev(dx, dy, dz)

            if d2 < probe_radius_squared:
                grid_neighs_entity_ids[midpoint_index] = -1
                grid_neighs_media_ids[midpoint_index] = -1
                cycle_flag = 1
                break

            if d2 < min_d2:
                closest_exposed_grid_index = eg_idx
                min_d2 = d2

        pvx = vx
        pvy = vy
        pvz = vz

        if cycle_flag == 1:
            break

    return cycle_flag, closest_exposed_grid_index, min_d2


# -----------------------------------------------------------------------------
# Atom candidate scan
# -----------------------------------------------------------------------------


@cuda.jit(device=True, inline=True)
def scan_candidates_and_find_closest_neighbor_dev(
    lower_limit,
    upper_limit,
    voxel_atom_ids,  # 1D int
    atom_surface_flags,  # 1D int/bool (NOTE: CPU uses atom_surface_flags[ao] with ao 1-based)
    neighbor_entity_id,  # scalar int
    min_distance_metric,  # scalar float
    closest_atom_or_object_index,  # scalar int
    num_atoms,  # scalar int
    atoms_data,  # (num_atoms, fields)
    midpoint_coords,  # (3,)
):
    previous_atom_or_object_index = 0
    num_neighbors_found = 0
    atom_or_object_index_current = 0

    mx0 = midpoint_coords[0]
    mx1 = midpoint_coords[1]
    mx2 = midpoint_coords[2]

    for kk in range(lower_limit, upper_limit + 1):
        ao = int(voxel_atom_ids[kk])
        if ao != 0:
            atom_or_object_index_current = ao

        if 0 < ao <= num_atoms:
            # IMPORTANT: this indexing must match CPU exactly.
            # If atom_surface_flags is length num_atoms+1 with 1-based indexing, keep ao.
            if atom_surface_flags[ao] == 0:
                num_neighbors_found += 1

                ax0 = atoms_data[ao - 1, ATOMFIELD_X + 0]
                ax1 = atoms_data[ao - 1, ATOMFIELD_X + 1]
                ax2 = atoms_data[ao - 1, ATOMFIELD_X + 2]
                ar = atoms_data[ao - 1, ATOMFIELD_RADIUS]

                dx = mx0 - ax0
                dy = mx1 - ax1
                dz = mx2 - ax2

                dist2 = dot3_dev(dx, dy, dz)
                dist = math.sqrt(dist2)
                ds = dist - ar

                if ds < min_distance_metric:
                    min_distance_metric = ds
                    closest_atom_or_object_index = ao

        else:
            # object branch
            if (ao != previous_atom_or_object_index) and (neighbor_entity_id == 0):
                previous_atom_or_object_index = ao
                num_neighbors_found += 1
                dist2 = 0.0
                if 0.0 <= dist2 < min_distance_metric:
                    min_distance_metric = dist2
                    closest_atom_or_object_index = ao

        if num_neighbors_found >= MAX_NEIGHBR_BOUNDARY_ARRAY_LEN:
            break

    return (
        closest_atom_or_object_index,
        min_distance_metric,
        atom_or_object_index_current,
    )


# -----------------------------------------------------------------------------
# Contact check (keep ONE version)
# -----------------------------------------------------------------------------


@cuda.jit(device=True)
def check_contact_region_dev(
    midpoint_entity_id: int,
    midpoint_coords,  # (3,)
    closest_atom_or_object_index: int,
    num_atoms: int,
    atoms_data,
    atom_plus_probe_radii_1d,
    atom_plus_probe_radii_is_valid: int,
    atom_plus_probe_radii_shrink_1d,
    cube_side_length_inverse: float,
    cube_shape,  # (3,)  (see NOTE below about bounds)
    cube_vertex_lowest_xyz,  # (3,)
    voxel_atom_count,  # 3D
    voxel_atom_count_cumulative,  # 3D
    voxel_atom_ids,  # 1D
    atomfield_x: int,
    atomfield_radius: int,  # unused but kept for signature stability
):
    in_contact = 1
    neighbor_atom_or_object_index = 0

    mx0 = midpoint_coords[0]
    mx1 = midpoint_coords[1]
    mx2 = midpoint_coords[2]

    sp0 = mx0
    sp1 = mx1
    sp2 = mx2

    if 0 < closest_atom_or_object_index <= num_atoms:
        ax0 = atoms_data[closest_atom_or_object_index - 1, atomfield_x + 0]
        ax1 = atoms_data[closest_atom_or_object_index - 1, atomfield_x + 1]
        ax2 = atoms_data[closest_atom_or_object_index - 1, atomfield_x + 2]

        dx = mx0 - ax0
        dy = mx1 - ax1
        dz = mx2 - ax2
        dist2 = dot3_dev(dx, dy, dz)
        dist = math.sqrt(dist2) if dist2 > 0.0 else 1.0

        if atom_plus_probe_radii_is_valid == 1:
            r = atom_plus_probe_radii_1d[closest_atom_or_object_index - 1]
            invd = 1.0 / dist if dist > 0.0 else 0.0
            sp0 = ax0 + r * dx * invd
            sp1 = ax1 + r * dy * invd
            sp2 = ax2 + r * dz * invd
        else:
            sp0 = ax0
            sp1 = ax1
            sp2 = ax2

    ix = int((sp0 - cube_vertex_lowest_xyz[0]) * cube_side_length_inverse)
    iy = int((sp1 - cube_vertex_lowest_xyz[1]) * cube_side_length_inverse)
    iz = int((sp2 - cube_vertex_lowest_xyz[2]) * cube_side_length_inverse)

    # NOTE: This bound check MUST match how cube_shape is defined on host.
    if (ix < 0) or (iy < 0) or (iz < 0):
        return 1, 0
    if (
        (ix > int(cube_shape[0]))
        or (iy > int(cube_shape[1]))
        or (iz > int(cube_shape[2]))
    ):
        return 1, 0

    lower = int(voxel_atom_count[ix, iy, iz])
    upper = int(voxel_atom_count_cumulative[ix, iy, iz])

    for kk in range(lower, upper + 1):
        neighbor_atom_or_object_index = int(voxel_atom_ids[kk])

        if 0 < neighbor_atom_or_object_index <= num_atoms:
            bx0 = atoms_data[neighbor_atom_or_object_index - 1, atomfield_x + 0]
            bx1 = atoms_data[neighbor_atom_or_object_index - 1, atomfield_x + 1]
            bx2 = atoms_data[neighbor_atom_or_object_index - 1, atomfield_x + 2]

            ddx = sp0 - bx0
            ddy = sp1 - bx1
            ddz = sp2 - bx2
            d2 = dot3_dev(ddx, ddy, ddz)

            if d2 < atom_plus_probe_radii_shrink_1d[neighbor_atom_or_object_index - 1]:
                return 0, neighbor_atom_or_object_index

        else:
            if (neighbor_atom_or_object_index != closest_atom_or_object_index) and (
                midpoint_entity_id == 0
            ):
                # CPU has a weird object-distance branch; currently no-op
                pass

    return in_contact, neighbor_atom_or_object_index


# -----------------------------------------------------------------------------
# Media ID + epsmap remap
# -----------------------------------------------------------------------------


@cuda.jit(device=True, inline=True)
def get_media_id_dev(
    atom_or_object_index_closest: int,
    atom_or_object_index_current: int,
    num_atoms: int,
    atoms_data,
    atomfield_media_id: int,
):
    if atom_or_object_index_closest == 0:
        if 0 < atom_or_object_index_current <= num_atoms:
            return int(atoms_data[atom_or_object_index_current - 1, atomfield_media_id])
        return int(atoms_data[0, atomfield_media_id])

    if 0 < atom_or_object_index_closest <= num_atoms:
        return int(atoms_data[atom_or_object_index_closest - 1, atomfield_media_id])

    return 0


@cuda.jit(device=True, inline=True)
def remap_epsilon_map_inplace_dev(
    grid_index_x: int,
    grid_index_y: int,
    grid_index_z: int,
    stride_x_3: int,
    stride_y_3: int,
    index_map,
    neighbor_index: int,
    epsilon_dimension: int,
    contact_region_media_id: int,
    neighbor_entity_id: int,
    epsmap,
):
    idx1d_x3 = (
        (grid_index_x + int(index_map[1, neighbor_index])) * stride_x_3
        + (grid_index_y + int(index_map[2, neighbor_index])) * stride_y_3
        + (grid_index_z + int(index_map[3, neighbor_index])) * 3
    )
    flag = int(index_map[4, neighbor_index])
    pos = idx1d_x3 + (flag - 1)
    epsmap[pos] = neighbor_entity_id + contact_region_media_id * epsilon_dimension


@cuda.jit(device=True, inline=True)
def remap_possibly_sign_updated_epsilon_map_inplace_dev(
    grid_index_x: int,
    grid_index_y: int,
    grid_index_z: int,
    stride_x_3: int,
    stride_y_3: int,
    index_map,
    neighbor_index: int,
    epsilon_dimension: int,
    neighbor_entity_id: int,
    epsmap,
):
    """
    Integer-only, % free, float free.
    Only does something if neighbor_entity_id < 0.
    """
    if neighbor_entity_id >= 0:
        return

    idx1d_x3 = (
        (grid_index_x + int(index_map[1, neighbor_index])) * stride_x_3
        + (grid_index_y + int(index_map[2, neighbor_index])) * stride_y_3
        + (grid_index_z + int(index_map[3, neighbor_index])) * 3
    )
    flag = int(index_map[4, neighbor_index])
    pos = idx1d_x3 + (flag - 1)

    eps_idx = epsmap[pos]  # int32

    # already sign-updated?
    if eps_idx < 0:
        return

    epsdim = epsilon_dimension
    a = abs_i32(eps_idx)
    mid = a // epsdim
    eid = a - mid * epsdim  # avoids % semantics

    if eid == 0:
        eid = 1

    epsmap[pos] = -(eid + mid * epsdim)
