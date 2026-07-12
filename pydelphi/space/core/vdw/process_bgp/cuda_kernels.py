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

# pydelphi/space/core/vdw/process_bgp/cuda_kernels.py
#
# DROP-IN REPLACEMENT (only the kernels you pasted, fixed)
#
# Key fixes:
# 1) touch_append_kernel: bounds check on touched_idx1d append (prevents memory corruption)
# 2) commit_kernel: "external->internal" now clears external bit, NOT deleting boundary
# 3) finalize_frontier_kernel: uses correct strides incl z_stride_x3
# 4) compact_keep_existing_kernel: now filters out NON_SOLUTE_BOUNDARY (prevents list explosion)

from numba import cuda, int64, int32
import math
import numpy as np

from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_RADIUS,
    ATOMFIELD_MEDIA_ID,
    NON_SOLUTE_BOUNDARY,
    SOLUTE_BOUNDARY_ANY,
    SOLUTE_BOUNDARY_EXTERNAL,
    ConstDelPhiInts,
)

from pydelphi.space.core.vdw.process_bgp.helpers_cuda import (
    EXIT_NJIT_FLAG,
    bgp_color_id_dev,
    idx1d_from_ijk_dev,
    ijk_from_idx1d_dev,
    vdw_to_ms_all_voxels_buffer_dev,
    classify_point_from_epsmap_dev,
    stamp_claim_dev,
    scan_candidates_and_find_closest_neighbor_dev,
    check_contact_region_dev,
    get_media_id_dev,
    remap_epsilon_map_inplace_dev,
    remap_possibly_sign_updated_epsilon_map_inplace_dev,
)

EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value

# Action codes
LOG_ACT_REMOVE = np.int32(1)
LOG_ACT_ADD = np.int32(2)
LOG_ACT_EXTCLR = np.int32(3)
LOG_ACT_KEEP = np.int32(4)  # optional


@cuda.jit(device=True, inline=True)
def _abs_i32_dev(x):
    return -x if x < 0 else x


@cuda.jit(device=True, inline=True)
def _decode_eps_dev(eps_val, epsdim):
    v = int64(eps_val)
    if v < 0:
        v = -v
    d = int64(epsdim)
    mid = v // d
    eid = v % d
    return int32(eid), int32(mid)


@cuda.jit(device=True, inline=True)
def log_commit_event_dev(
    log_count,  # (1,) int32
    log_cap: int,  # scalar
    log_idx1d,  # (cap,) int32
    log_action,  # (cap,) int32
    log_color,  # (cap,) int32   (optional, pass -1 if unknown)
    log_old_type,  # (cap,) int32
    log_new_type,  # (cap,) int32
    log_is_boundary,  # (cap,) int32
    log_is_external,  # (cap,) int32
    log_eid6,  # (cap,6) int16 or int32
    log_mid6,  # (cap,6) int16 or int32
    idx1d: int,
    action: int,
    color_id: int,
    old_type: int,
    new_type: int,
    is_boundary: int,
    is_external: int,
    epsmap,
    epsdim: int,
    x_stride_x3: int,
    y_stride_x3: int,
):
    pos = cuda.atomic.add(log_count, 0, 1)
    if pos >= log_cap:
        return

    log_idx1d[pos] = idx1d
    log_action[pos] = action
    log_color[pos] = color_id
    log_old_type[pos] = old_type
    log_new_type[pos] = new_type
    log_is_boundary[pos] = is_boundary
    log_is_external[pos] = is_external

    # same 6 positions as classify (base = idx1d*3)
    base = idx1d * 3
    o1 = 0
    o2 = 1
    o3 = 2
    o4 = -x_stride_x3
    o5 = -y_stride_x3 + 1
    o6 = -1  # NOTE: if your classify is wrong here, this log will expose it.

    v1 = int(epsmap[base + o1])
    v2 = int(epsmap[base + o2])
    v3 = int(epsmap[base + o3])
    v4 = int(epsmap[base + o4])
    v5 = int(epsmap[base + o5])
    v6 = int(epsmap[base + o6])

    e, m = _decode_eps_dev(v1, epsdim)
    log_eid6[pos, 0] = e
    log_mid6[pos, 0] = m
    e, m = _decode_eps_dev(v2, epsdim)
    log_eid6[pos, 1] = e
    log_mid6[pos, 1] = m
    e, m = _decode_eps_dev(v3, epsdim)
    log_eid6[pos, 2] = e
    log_mid6[pos, 2] = m
    e, m = _decode_eps_dev(v4, epsdim)
    log_eid6[pos, 3] = e
    log_mid6[pos, 3] = m
    e, m = _decode_eps_dev(v5, epsdim)
    log_eid6[pos, 4] = e
    log_mid6[pos, 4] = m
    e, m = _decode_eps_dev(v6, epsdim)
    log_eid6[pos, 5] = e
    log_mid6[pos, 5] = m


@cuda.jit
def touch_append_kernel(
    color_id: int,
    boundary_point_start_index: int,
    boundary_point_end_index: int,
    boundary_grid_indices,
    grid_shape,
    solute_bgp_type_1d,
    touched_idx1d,
    touched_counter,
    exec_status,
):
    bi = cuda.grid(1)
    B = boundary_point_end_index - boundary_point_start_index + 1
    if bi >= B:
        return

    nx = int(grid_shape[0])
    ny = int(grid_shape[1])
    nz = int(grid_shape[2])
    x_stride = ny * nz
    y_stride = nz

    bgp_index = boundary_point_start_index + bi
    idx_center = int(boundary_grid_indices[bgp_index - 1])

    ix, iy, iz = ijk_from_idx1d_dev(idx_center, x_stride, y_stride)

    if bgp_color_id_dev(ix, iy, iz) != color_id:
        return

    if solute_bgp_type_1d[idx_center] == NON_SOLUTE_BOUNDARY:
        return

    cap = int(touched_idx1d.shape[0])

    # local helper for bounds-safe append
    def _emit(val):
        pos = cuda.atomic.add(touched_counter, 0, 1)
        if pos < cap:
            touched_idx1d[pos] = val
        else:
            exec_status[0] = EXIT_NJIT_FLAG

    # 6-neighbors + center (exactly your intended visitation)
    if ix + 1 < nx - 1:
        _emit(idx1d_from_ijk_dev(ix + 1, iy, iz, x_stride, nz))
    if iy + 1 < ny - 1:
        _emit(idx1d_from_ijk_dev(ix, iy + 1, iz, x_stride, nz))
    if iz + 1 < nz - 1:
        _emit(idx1d_from_ijk_dev(ix, iy, iz + 1, x_stride, nz))
    if ix - 1 > 0:
        _emit(idx1d_from_ijk_dev(ix - 1, iy, iz, x_stride, nz))
    if iy - 1 > 0:
        _emit(idx1d_from_ijk_dev(ix, iy - 1, iz, x_stride, nz))
    if iz - 1 > 0:
        _emit(idx1d_from_ijk_dev(ix, iy, iz - 1, x_stride, nz))

    # include center (only if interior)
    if (0 < ix < nx - 1) and (0 < iy < ny - 1) and (0 < iz < nz - 1):
        _emit(idx_center)


@cuda.jit(device=True, inline=True)
def eps_write_component_dev(epsmap, base_x3: int, flag: int, val: int):
    # flag in {1,2,3}
    epsmap[base_x3 + (flag - 1)] = val


@cuda.jit(device=True, inline=True)
def remap_eps_full_dev(
    gx: int,
    gy: int,
    gz: int,
    stride_x_3: int,
    stride_y_3: int,
    index_map,
    neighbor_index: int,
    epsilon_dimension: int,
    contact_region_media_id: int,
    neighbor_entity_id: int,
    epsmap,
):
    base_x3 = (
        (gx + index_map[1, neighbor_index]) * stride_x_3
        + (gy + index_map[2, neighbor_index]) * stride_y_3
        + (gz + index_map[3, neighbor_index]) * 3
    )
    flag = int(index_map[4, neighbor_index])
    val = int(neighbor_entity_id + contact_region_media_id * epsilon_dimension)
    eps_write_component_dev(epsmap, base_x3, flag, val)


@cuda.jit
def commit_kernel(
    touched_idx1d,
    num_events: int,
    grid_shape,
    boundary_grid_indices_1d,
    solute_bgp_type_1d,
    epsmap,
    epsilon_dimension: int,
    num_discovered_total: int,
    max_boundary_grid_points: int,
    num_external_counter,
    added_counter,
    removed_counter,
    visited_next,
    stamp: int,
    new_bgps_idx1d,
    new_count,
    processed_stamp,
    exec_status,
    # --- optional debug log (pass log_enabled=0 to disable) ---
    log_enabled: int,
    log_color_id: int,
    log_count,
    log_cap: int,
    log_idx1d,
    log_action,
    log_color,
    log_old_type,
    log_new_type,
    log_is_boundary,
    log_is_external,
    log_eid6,
    log_mid6,
):
    tid = cuda.grid(1)
    if tid >= num_events:
        return

    idx1d = int(touched_idx1d[tid])
    if idx1d < 0:
        return

    # IMPORTANT: stamp-based claim (works across colors without memsetting)
    if not stamp_claim_dev(processed_stamp, idx1d, stamp):
        return

    ny = int(grid_shape[1])
    nz = int(grid_shape[2])
    x_stride_x3 = ny * nz * 3
    y_stride_x3 = nz * 3

    old_type = int(solute_bgp_type_1d[idx1d])
    was_marked = 1 if old_type != NON_SOLUTE_BOUNDARY else 0
    old_ext = (
        1 if ((old_type & SOLUTE_BOUNDARY_EXTERNAL) == SOLUTE_BOUNDARY_EXTERNAL) else 0
    )

    is_boundary, is_external = classify_point_from_epsmap_dev(
        idx1d, int(epsilon_dimension), epsmap, x_stride_x3, y_stride_x3
    )

    # ---- Case A: remove ----
    if (is_boundary == 0) and (was_marked == 1):
        new_type = NON_SOLUTE_BOUNDARY
        solute_bgp_type_1d[idx1d] = new_type
        if old_ext == 1:
            cuda.atomic.add(num_external_counter, 0, -1)
        cuda.atomic.add(removed_counter, 0, 1)

        if log_enabled == 1:
            log_commit_event_dev(
                log_count,
                log_cap,
                log_idx1d,
                log_action,
                log_color,
                log_old_type,
                log_new_type,
                log_is_boundary,
                log_is_external,
                log_eid6,
                log_mid6,
                idx1d,
                LOG_ACT_REMOVE,
                log_color_id,
                old_type,
                new_type,
                is_boundary,
                is_external,
                epsmap,
                int(epsilon_dimension),
                x_stride_x3,
                y_stride_x3,
            )
        return

    # ---- Case B: clear external bit (still boundary) ----
    if (
        (is_boundary == 1)
        and (was_marked == 1)
        and (old_ext == 1)
        and (is_external == 0)
    ):
        new_type = SOLUTE_BOUNDARY_ANY
        solute_bgp_type_1d[idx1d] = new_type
        cuda.atomic.add(num_external_counter, 0, -1)

        if log_enabled == 1:
            log_commit_event_dev(
                log_count,
                log_cap,
                log_idx1d,
                log_action,
                log_color,
                log_old_type,
                log_new_type,
                log_is_boundary,
                log_is_external,
                log_eid6,
                log_mid6,
                idx1d,
                LOG_ACT_EXTCLR,
                log_color_id,
                old_type,
                new_type,
                is_boundary,
                is_external,
                epsmap,
                int(epsilon_dimension),
                x_stride_x3,
                y_stride_x3,
            )
        # still boundary => can be part of frontier if you want; CPU behavior decides
        # (we keep the frontier logic below)
        # fallthrough

    # ---- Case C: newly boundary -> add ----
    if (is_boundary == 1) and (was_marked == 0):
        new_type = SOLUTE_BOUNDARY_ANY
        if is_external == 1:
            new_type = new_type | SOLUTE_BOUNDARY_EXTERNAL

        solute_bgp_type_1d[idx1d] = new_type

        inc0 = cuda.atomic.add(added_counter, 0, 1)
        new_id = num_discovered_total + (inc0 + 1)
        if new_id > max_boundary_grid_points:
            exec_status[0] = EXIT_NJIT_FLAG
            return

        boundary_grid_indices_1d[new_id - 1] = idx1d
        if is_external == 1:
            cuda.atomic.add(num_external_counter, 0, 1)

        if log_enabled == 1:
            log_commit_event_dev(
                log_count,
                log_cap,
                log_idx1d,
                log_action,
                log_color,
                log_old_type,
                log_new_type,
                log_is_boundary,
                log_is_external,
                log_eid6,
                log_mid6,
                idx1d,
                LOG_ACT_ADD,
                log_color_id,
                old_type,
                new_type,
                is_boundary,
                is_external,
                epsmap,
                int(epsilon_dimension),
                x_stride_x3,
                y_stride_x3,
            )

    # ---- Frontier list (dedup using visited_next stamp) ----
    if solute_bgp_type_1d[idx1d] != NON_SOLUTE_BOUNDARY:
        if stamp_claim_dev(visited_next, idx1d, stamp):
            pos = cuda.atomic.add(new_count, 0, 1)
            if pos >= new_bgps_idx1d.shape[0]:
                exec_status[0] = EXIT_NJIT_FLAG
                return
            new_bgps_idx1d[pos] = idx1d


@cuda.jit
def finalize_frontier_kernel(
    boundary_point_start_index: int,
    boundary_point_end_index: int,
    boundary_grid_indices,
    grid_shape,
    epsilon_dimension: int,
    epsmap,
    solute_bgp_type_1d,
    num_external_counter,
    removed_counter,
):
    bi = cuda.grid(1)
    B = boundary_point_end_index - boundary_point_start_index + 1
    if bi >= B:
        return

    bgp = boundary_point_start_index + bi
    idx1d = int(boundary_grid_indices[bgp - 1])

    if solute_bgp_type_1d[idx1d] == NON_SOLUTE_BOUNDARY:
        return

    old_ext = (
        1
        if (solute_bgp_type_1d[idx1d] & SOLUTE_BOUNDARY_EXTERNAL)
        == SOLUTE_BOUNDARY_EXTERNAL
        else 0
    )

    # Correct mid-map strides (x3 layout)
    y_stride_x3 = int(grid_shape[2]) * 3
    x_stride_x3 = int(grid_shape[1]) * y_stride_x3

    is_boundary, is_external = classify_point_from_epsmap_dev(
        idx1d, epsilon_dimension, epsmap, x_stride_x3, y_stride_x3
    )

    if is_boundary == 0:
        if old_ext == 1:
            cuda.atomic.add(num_external_counter, 0, -1)
        solute_bgp_type_1d[idx1d] = NON_SOLUTE_BOUNDARY
        cuda.atomic.add(removed_counter, 0, 1)

    elif (is_external == 0) and (old_ext == 1):
        solute_bgp_type_1d[idx1d] = SOLUTE_BOUNDARY_ANY
        cuda.atomic.add(num_external_counter, 0, -1)


@cuda.jit
def compact_keep_existing_kernel(
    total_discovered_end_index: int,
    boundary_grid_indices_in,  # 1D idx1d
    grid_shape,  # kept for signature stability (unused)
    solute_bgp_type_1d,
    stamp_arr,
    stamp: int,
    boundary_grid_indices_out,
    out_count,
):
    i = cuda.grid(1)
    if i >= total_discovered_end_index:
        return

    idx1d = int(boundary_grid_indices_in[i])
    if idx1d == 0:
        return

    # CRITICAL: only keep if still marked as boundary
    if solute_bgp_type_1d[idx1d] == NON_SOLUTE_BOUNDARY:
        return

    if not stamp_claim_dev(stamp_arr, idx1d, stamp):
        return

    pos = cuda.atomic.add(out_count, 0, 1)
    boundary_grid_indices_out[pos] = idx1d


@cuda.jit
def compact_append_new_kernel(
    new_bgps_idx1d,
    new_count_val: int,
    grid_shape,
    stamp_arr,
    stamp: int,
    boundary_grid_indices_out,
    out_count,
):
    i = cuda.grid(1)
    if i >= new_count_val:
        return

    idx1d = int(new_bgps_idx1d[i])

    ny = int(grid_shape[1])
    nz = int(grid_shape[2])
    x_stride = ny * nz
    y_stride = nz

    if not stamp_claim_dev(stamp_arr, idx1d, stamp):
        return

    # ix, iy, iz = ijk_from_idx1d_dev(idx1d, x_stride, y_stride)

    pos = cuda.atomic.add(out_count, 0, 1)
    boundary_grid_indices_out[pos] = idx1d


@cuda.jit
def clear_tail_kernel(start: int, end: int, boundary_grid_indices_out):
    i = cuda.grid(1)
    j = start + i
    if j >= end:
        return
    boundary_grid_indices_out[j] = 0


@cuda.jit
def midpoint_update_kernel(
    color_id,
    boundary_point_start_index,
    boundary_point_end_index,
    boundary_grid_indices_1d,
    grid_spacing,
    grid_shape,
    grid_origin_current,
    probe_radius_squared_1,
    probe_radius_squared_2,
    index_map,
    grid_neighbor_coords_offsets,
    index_discrete_epsilon_map_1d,
    solute_bgp_type_1d,
    epsilon_dimension,
    rm_boundary_pt_condition,
    min_xyz,
    cube_side_indver_inverse,
    cube_shape_indver,
    cube_voxel_start_indices,
    cube_voxel_end_indices,
    voxel_grid_point_indices,
    exposed_grids_coords,
    cube_vertex_lowest_xyz,
    cube_side_length_inverse,
    cube_shape,
    voxel_atom_count,
    voxel_atom_count_cumulative,
    voxel_atom_ids,
    atom_surface_flags,
    num_atoms,
    atoms_data,
    atom_plus_probe_radii_1d,
    atom_plus_probe_radii_shrink_1d,
    exec_status,
):
    bi = cuda.grid(1)
    B = boundary_point_end_index - boundary_point_start_index + 1
    if bi >= B:
        return

    bgp_index = boundary_point_start_index + bi
    idx_center = int(boundary_grid_indices_1d[bgp_index - 1])
    if idx_center <= 0:
        return

    nx, ny, nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
    ny_nz = ny * nz

    ix = idx_center // ny_nz
    rem = idx_center - ix * ny_nz
    iy = rem // nz
    iz = rem - iy * nz

    if bgp_color_id_dev(ix, iy, iz) != int(color_id):
        return

    if solute_bgp_type_1d[idx_center] == NON_SOLUTE_BOUNDARY:
        return

    stride_y_3 = nz * 3
    stride_x_3 = ny_nz * 3
    base3 = int(idx_center * 3)
    eps_dim = int(epsilon_dimension)
    n_atoms = int(num_atoms)

    # --- 1. UNIFIED PRE-PASS (Frozen State) ---
    # We use a local array to store the decoded neighbor info to avoid redundant math
    neigh_ents = cuda.local.array(6, dtype=cuda.int32)
    neigh_meds = cuda.local.array(6, dtype=cuda.int32)

    # Offsets matching your classify/log logic
    offsets = cuda.local.array(6, dtype=cuda.int32)
    offsets[0], offsets[1], offsets[2] = 0, 1, 2
    offsets[3], offsets[4], offsets[5] = -stride_x_3, -stride_y_3 + 1, -1

    rm_triggered = False
    for i in range(6):
        target_idx = base3 + offsets[i]
        val = int(index_discrete_epsilon_map_1d[target_idx]) if target_idx >= 0 else 0
        e, m = _decode_eps_dev(val, eps_dim)
        neigh_ents[i] = e
        neigh_meds[i] = m
        if 1 < e <= (n_atoms + 1):
            rm_triggered = True

    rm_boundary_pt = rm_triggered and (int(rm_boundary_pt_condition) == 1)

    # --- 2. NEIGHBOR LOOP ---
    origin0, origin1, origin2 = (
        grid_origin_current[0],
        grid_origin_current[1],
        grid_origin_current[2],
    )

    for neighbor_index in range(1, 7):
        off = neighbor_index - 1

        # Grid movement
        gx = ix + int(grid_neighbor_coords_offsets[off, 0])
        gy = iy + int(grid_neighbor_coords_offsets[off, 1])
        gz = iz + int(grid_neighbor_coords_offsets[off, 2])

        if (
            gx <= 0
            or gy <= 0
            or gz <= 0
            or gx >= nx - 1
            or gy >= ny - 1
            or gz >= nz - 1
        ):
            continue

        mx0 = origin0 + gx * grid_spacing
        mx1 = origin1 + gy * grid_spacing
        mx2 = origin2 + gz * grid_spacing

        # Determine Voxel for Candidate Search
        vx = int((mx0 - min_xyz[0]) * cube_side_indver_inverse)
        vy = int((mx1 - min_xyz[1]) * cube_side_indver_inverse)
        vz = int((mx2 - min_xyz[2]) * cube_side_indver_inverse)

        if (
            vx < 0
            or vy < 0
            or vz < 0
            or vx >= int(cube_shape_indver[0])
            or vy >= int(cube_shape_indver[1])
            or vz >= int(cube_shape_indver[2])
        ):
            continue

        midpoint_coords = cuda.local.array(3, dtype=cuda.float64)
        midpoint_coords[0], midpoint_coords[1], midpoint_coords[2] = mx0, mx1, mx2

        # Gate Logic (Using pre-decoded neighbors)
        curr_ent = neigh_ents[off]
        curr_med = neigh_meds[off]

        gate_condition = (
            (curr_ent == 0)
            or (rm_boundary_pt and curr_ent > n_atoms + 1)
            or (curr_med == 0 and curr_ent > 0)
        )

        probe2 = probe_radius_squared_2
        if gate_condition:
            if (curr_ent == 0) or (curr_med == 0):
                probe2 = probe_radius_squared_1

        # --- 3. VDW-TO-MS BUFFER CHECK ---
        tmp_e = cuda.local.array(8, dtype=cuda.int32)
        tmp_m = cuda.local.array(8, dtype=cuda.int32)
        for t in range(8):
            tmp_e[t] = 0
            tmp_m[t] = 0

        closest_exposed_idx = 0
        min_d2 = 1.0e30

        cycle_flag, closest_exposed_idx, min_d2 = vdw_to_ms_all_voxels_buffer_dev(
            0,
            neighbor_index,
            vx,
            vy,
            vz,
            probe2,
            midpoint_coords,
            tmp_e,
            tmp_m,
            closest_exposed_idx,
            min_d2,
            cube_voxel_start_indices,
            cube_voxel_end_indices,
            voxel_grid_point_indices,
            exposed_grids_coords,
        )

        if cycle_flag == 1:
            if int(rm_boundary_pt_condition) == 1:
                remap_possibly_sign_updated_epsilon_map_inplace_dev(
                    ix,
                    iy,
                    iz,
                    stride_x_3,
                    stride_y_3,
                    index_map,
                    neighbor_index,
                    epsilon_dimension,
                    -1,
                    index_discrete_epsilon_map_1d,
                )
            continue

        # --- 4. ATOM SCAN ---
        ix2 = int((mx0 - cube_vertex_lowest_xyz[0]) * cube_side_length_inverse)
        iy2 = int((mx1 - cube_vertex_lowest_xyz[1]) * cube_side_length_inverse)
        iz2 = int((mx2 - cube_vertex_lowest_xyz[2]) * cube_side_length_inverse)

        if (
            ix2 < 0
            or iy2 < 0
            or iz2 < 0
            or ix2 >= int(cube_shape[0])
            or iy2 >= int(cube_shape[1])
            or iz2 >= int(cube_shape[2])
        ):
            continue

        lower = int(voxel_atom_count[ix2, iy2, iz2])
        upper = int(voxel_atom_count_cumulative[ix2, iy2, iz2])

        # Get the gate entity_id specifically for candidate filtering
        # (This remains dynamic as it is used to skip the 'current' atom if applicable)
        idx1d_x3 = (
            (ix + int(index_map[1, neighbor_index])) * stride_x_3
            + (iy + int(index_map[2, neighbor_index])) * stride_y_3
            + (iz + int(index_map[3, neighbor_index])) * 3
        )
        flag = int(index_map[4, neighbor_index])
        raw_eps = int(index_discrete_epsilon_map_1d[idx1d_x3 + (flag - 1)])
        neighbor_entity_gate, _ = _decode_eps_dev(raw_eps, eps_dim)

        closest_atom_or_obj, min_metric, last_seen = (
            scan_candidates_and_find_closest_neighbor_dev(
                lower,
                upper,
                voxel_atom_ids,
                atom_surface_flags,
                neighbor_entity_gate,
                1.0e30,
                0,
                n_atoms,
                atoms_data,
                midpoint_coords,
            )
        )

        # Cavity Logic
        if closest_atom_or_obj == 0:
            neigh_media_id = get_media_id_dev(
                0, last_seen, n_atoms, atoms_data, ATOMFIELD_MEDIA_ID
            )
            if closest_atom_or_obj == 0:
                neigh_media_id = get_media_id_dev(
                    0, last_seen, n_atoms, atoms_data, ATOMFIELD_MEDIA_ID
                )
                remap_epsilon_map_inplace_dev(
                    ix,
                    iy,
                    iz,
                    stride_x_3,
                    stride_y_3,
                    index_map,
                    neighbor_index,
                    epsilon_dimension,
                    neigh_media_id,
                    1,  # OK only if CPU uses 1 for cavity/solvent entity
                    index_discrete_epsilon_map_1d,
                )
            if int(rm_boundary_pt_condition) == 1:
                remap_possibly_sign_updated_epsilon_map_inplace_dev(
                    ix,
                    iy,
                    iz,
                    stride_x_3,
                    stride_y_3,
                    index_map,
                    neighbor_index,
                    epsilon_dimension,
                    1,
                    index_discrete_epsilon_map_1d,
                )
            continue

        # Contact Check
        in_contact, _ = check_contact_region_dev(
            0,
            midpoint_coords,
            closest_atom_or_obj,
            n_atoms,
            atoms_data,
            atom_plus_probe_radii_1d,
            1,
            atom_plus_probe_radii_shrink_1d,
            cube_side_length_inverse,
            cube_shape,
            cube_vertex_lowest_xyz,
            voxel_atom_count,
            voxel_atom_count_cumulative,
            voxel_atom_ids,
            ATOMFIELD_X,
            ATOMFIELD_RADIUS,
        )

        if in_contact == 0:
            if int(rm_boundary_pt_condition) == 1:
                remap_possibly_sign_updated_epsilon_map_inplace_dev(
                    ix,
                    iy,
                    iz,
                    stride_x_3,
                    stride_y_3,
                    index_map,
                    neighbor_index,
                    epsilon_dimension,
                    -closest_atom_or_obj,
                    index_discrete_epsilon_map_1d,
                )
            continue

        # Not in contact
        neigh_media_id = get_media_id_dev(
            closest_atom_or_obj, last_seen, n_atoms, atoms_data, ATOMFIELD_MEDIA_ID
        )
        remap_epsilon_map_inplace_dev(
            ix,
            iy,
            iz,
            stride_x_3,
            stride_y_3,
            index_map,
            neighbor_index,
            epsilon_dimension,
            closest_atom_or_obj,
            neigh_media_id,
            index_discrete_epsilon_map_1d,
        )
        if int(rm_boundary_pt_condition) == 1:
            remap_possibly_sign_updated_epsilon_map_inplace_dev(
                ix,
                iy,
                iz,
                stride_x_3,
                stride_y_3,
                index_map,
                neighbor_index,
                epsilon_dimension,
                closest_atom_or_obj,
                index_discrete_epsilon_map_1d,
            )
