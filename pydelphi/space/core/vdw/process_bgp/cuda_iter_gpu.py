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

# pydelphi/space/core/vdw/process_bgp/cuda_iter_gpu.py
from __future__ import annotations
import numpy as np
from numba import cuda

from pydelphi.constants import ConstDelPhiInts
from pydelphi.space.core.vdw.process_bgp.cuda_kernels import (
    midpoint_update_kernel,
    touch_append_kernel,
    commit_kernel,
    finalize_frontier_kernel,
    compact_keep_existing_kernel,
    compact_append_new_kernel,
    clear_tail_kernel,
)

EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value


@cuda.jit
def _set_i32_1(arr, value):
    if cuda.grid(1) == 0:
        arr[0] = value


def _launch_1d(n: int, threads_per_block: int):
    blocks = (n + threads_per_block - 1) // threads_per_block
    return blocks, threads_per_block


def _process_bgp_one_iteration(
    *,
    probe_radius_squared_1: float,
    probe_radius_squared_2: float,
    boundary_point_start_index: int,
    boundary_point_end_index: int,
    grid_origin_current,
    max_boundary_grid_points: int,
    grid_neighbor_coords_offsets,
    boundary_grid_indices,  # DEVICE
    grid_spacing: float,
    grid_shape,
    exposed_grids_coords,
    index_discrete_epsilon_map_1d,  # DEVICE
    index_map,
    solute_bgp_type_1d,  # DEVICE
    epsilon_dimension: int,
    rm_boundary_pt_condition: bool,
    min_xyz,
    cube_side_indver_inverse: float,
    cube_shape_indver,
    cube_voxel_start_indices,
    cube_voxel_end_indices,
    cube_vertex_lowest_xyz,
    cube_side_length_inverse: float,
    cube_shape,
    voxel_grid_point_indices,
    voxel_atom_count,
    voxel_atom_count_cumulative,
    voxel_atom_ids,
    atom_surface_flags,
    num_atoms: int,
    atoms_data,
    atom_plus_probe_radii_1d,
    atom_plus_probe_radii_shrink_1d,
    num_external_boundary_points: int,
    cuda_ws,
):
    """
    CUDA backend: NO per-iter host<->device movement of big arrays.
    Only scalar counters are pulled back per iteration.
    """
    ws = cuda_ws
    threads = int(ws["threads_per_block"])

    # Reset device scalars (NO host->device copies)
    _set_i32_1[1, 1](ws["d_exec_status"], 0)
    _set_i32_1[1, 1](ws["d_added_counter"], 0)
    _set_i32_1[1, 1](ws["d_removed_counter"], 0)
    _set_i32_1[1, 1](ws["d_new_count"], 0)
    _set_i32_1[1, 1](ws["d_touched_counter"], 0)
    _set_i32_1[1, 1](ws["d_num_external_counter"], int(num_external_boundary_points))

    num_frontier = int(boundary_point_end_index - boundary_point_start_index + 1)
    blocks_frontier, threads = _launch_1d(num_frontier, threads)

    stamp = np.int32(ws["stamp"])
    # ws["d_processed_stamp"].copy_to_device(np.zeros(ws["d_processed_stamp"].shape, dtype=np.int32))
    # ws["d_visited_next"].copy_to_device(np.zeros(ws["d_visited_next"].shape, dtype=np.int32))

    log_cap = 20000  # tune (5k-50k typical)
    ws["log_cap"] = log_cap
    ws["d_log_count"] = cuda.to_device(np.zeros(1, dtype=np.int32))
    ws["d_log_idx1d"] = cuda.device_array(log_cap, dtype=np.int32)
    ws["d_log_action"] = cuda.device_array(log_cap, dtype=np.int32)
    ws["d_log_color"] = cuda.device_array(log_cap, dtype=np.int32)
    ws["d_log_old_type"] = cuda.device_array(log_cap, dtype=np.int32)
    ws["d_log_new_type"] = cuda.device_array(log_cap, dtype=np.int32)
    ws["d_log_is_boundary"] = cuda.device_array(log_cap, dtype=np.int32)
    ws["d_log_is_external"] = cuda.device_array(log_cap, dtype=np.int32)
    ws["d_log_eid6"] = cuda.device_array((log_cap, 6), dtype=np.int16)
    ws["d_log_mid6"] = cuda.device_array((log_cap, 6), dtype=np.int16)

    _set_i32_1[1, 1](ws["d_log_count"], 0)

    for color_id in range(27):
        stamp = np.int32(stamp + 1)
        midpoint_update_kernel[blocks_frontier, threads](
            np.int32(color_id),
            np.int32(boundary_point_start_index),
            np.int32(boundary_point_end_index),
            boundary_grid_indices,
            float(grid_spacing),
            grid_shape,
            grid_origin_current,
            float(probe_radius_squared_1),
            float(probe_radius_squared_2),
            index_map,
            grid_neighbor_coords_offsets,
            index_discrete_epsilon_map_1d,
            solute_bgp_type_1d,
            np.int32(epsilon_dimension),
            np.int32(1 if rm_boundary_pt_condition else 0),
            min_xyz,
            float(cube_side_indver_inverse),
            cube_shape_indver,
            cube_voxel_start_indices,
            cube_voxel_end_indices,
            voxel_grid_point_indices,
            exposed_grids_coords,
            cube_vertex_lowest_xyz,
            float(cube_side_length_inverse),
            cube_shape,
            voxel_atom_count,
            voxel_atom_count_cumulative,
            voxel_atom_ids,
            atom_surface_flags,
            np.int32(num_atoms),
            atoms_data,
            atom_plus_probe_radii_1d,
            atom_plus_probe_radii_shrink_1d,
            ws["d_exec_status"],
        )
        cuda.synchronize()

        if int(ws["d_exec_status"].copy_to_host()[0]) == EXIT_NJIT_FLAG:
            ws["stamp"] = stamp
            return (
                EXIT_NJIT_FLAG,
                False,
                0,
                0,
                int(ws["d_num_external_counter"].copy_to_host()[0]),
                boundary_grid_indices,
                solute_bgp_type_1d,
                index_discrete_epsilon_map_1d,
            )

        # Touch
        _set_i32_1[1, 1](ws["d_touched_counter"], 0)
        touch_append_kernel[blocks_frontier, threads](
            np.int32(color_id),
            np.int32(boundary_point_start_index),
            np.int32(boundary_point_end_index),
            boundary_grid_indices,
            grid_shape,
            solute_bgp_type_1d,
            ws["d_touched_idx1d"],
            ws["d_touched_counter"],
            ws["d_exec_status"],
        )
        cuda.synchronize()

        total_touched = int(ws["d_touched_counter"].copy_to_host()[0])
        if total_touched > 0:
            blocks_events, _ = _launch_1d(total_touched, threads)

            # Commit
            commit_kernel[blocks_events, threads](
                ws["d_touched_idx1d"],
                np.int32(total_touched),
                grid_shape,
                boundary_grid_indices,
                solute_bgp_type_1d,
                index_discrete_epsilon_map_1d,
                np.int32(epsilon_dimension),
                np.int32(boundary_point_end_index),
                np.int32(max_boundary_grid_points),
                ws["d_num_external_counter"],
                ws["d_added_counter"],
                ws["d_removed_counter"],
                ws["d_visited_next"],
                np.int32(stamp),
                ws["d_new_bgps_idx1d"],
                ws["d_new_count"],
                ws["d_processed_stamp"],
                ws["d_exec_status"],
                # ---- debug log ----
                np.int32(1),  # log_enabled (set 0 to disable)
                np.int32(color_id),  # log_color_id
                ws["d_log_count"],
                np.int32(ws["log_cap"]),
                ws["d_log_idx1d"],
                ws["d_log_action"],
                ws["d_log_color"],
                ws["d_log_old_type"],
                ws["d_log_new_type"],
                ws["d_log_is_boundary"],
                ws["d_log_is_external"],
                ws["d_log_eid6"],
                ws["d_log_mid6"],
            )
            cuda.synchronize()

            if int(ws["d_exec_status"].copy_to_host()[0]) == EXIT_NJIT_FLAG:
                ws["stamp"] = stamp
                return (
                    EXIT_NJIT_FLAG,
                    False,
                    0,
                    0,
                    int(ws["d_num_external_counter"].copy_to_host()[0]),
                    boundary_grid_indices,
                    solute_bgp_type_1d,
                    index_discrete_epsilon_map_1d,
                )

    # Finalize frontier centers
    finalize_frontier_kernel[blocks_frontier, threads](
        np.int32(boundary_point_start_index),
        np.int32(boundary_point_end_index),
        boundary_grid_indices,
        grid_shape,
        np.int32(epsilon_dimension),
        index_discrete_epsilon_map_1d,
        solute_bgp_type_1d,
        ws["d_num_external_counter"],
        ws["d_removed_counter"],
    )

    added_inc = int(ws["d_added_counter"].copy_to_host()[0])
    removed_inc = int(ws["d_removed_counter"].copy_to_host()[0])
    num_external_new = int(ws["d_num_external_counter"].copy_to_host()[0])

    total_discovered_end_index = int(boundary_point_end_index + added_inc)

    # Compact (double-buffer)
    stamp = np.int32(stamp + 1)
    _set_i32_1[1, 1](ws["d_out_count"], 0)

    nlog = int(ws["d_log_count"].copy_to_host()[0])
    nlog = min(nlog, ws["log_cap"])
    if nlog > 0:
        h_idx = ws["d_log_idx1d"][:nlog].copy_to_host()
        h_act = ws["d_log_action"][:nlog].copy_to_host()
        h_col = ws["d_log_color"][:nlog].copy_to_host()
        h_old = ws["d_log_old_type"][:nlog].copy_to_host()
        h_new = ws["d_log_new_type"][:nlog].copy_to_host()
        h_bnd = ws["d_log_is_boundary"][:nlog].copy_to_host()
        h_ext = ws["d_log_is_external"][:nlog].copy_to_host()
        h_eid6 = ws["d_log_eid6"][:nlog].copy_to_host()
        h_mid6 = ws["d_log_mid6"][:nlog].copy_to_host()

        # Print first 50 only (avoid spam)
        m = min(nlog, 5000)
        for i in range(m):
            if h_act[i] != 4:
                print(
                    f"[CUDA-LOG] i={i:4d} col={int(h_col[i]):2d} act={int(h_act[i])} idx1d={int(h_idx[i])} "
                    f"old={int(h_old[i])} new={int(h_new[i])} bnd={int(h_bnd[i])} ext={int(h_ext[i])} "
                    f"eid6={h_eid6[i].tolist()} mid6={h_mid6[i].tolist()}"
                )

    blocks_keep, _ = _launch_1d(total_discovered_end_index, threads)
    compact_keep_existing_kernel[blocks_keep, threads](
        np.int32(total_discovered_end_index),
        boundary_grid_indices,
        grid_shape,
        solute_bgp_type_1d,
        ws["d_compact_stamp_arr"],
        np.int32(stamp),
        ws["d_boundary_grid_indices_next"],
        ws["d_out_count"],
    )

    new_count_val = int(ws["d_new_count"].copy_to_host()[0])
    if new_count_val > 0:
        blocks_new, _ = _launch_1d(new_count_val, threads)
        compact_append_new_kernel[blocks_new, threads](
            ws["d_new_bgps_idx1d"],
            np.int32(new_count_val),
            grid_shape,
            ws["d_compact_stamp_arr"],
            np.int32(stamp),
            ws["d_boundary_grid_indices_next"],
            ws["d_out_count"],
        )

    next_count = int(ws["d_out_count"].copy_to_host()[0])
    if next_count < total_discovered_end_index:
        tail = total_discovered_end_index - next_count
        blocks_tail, _ = _launch_1d(tail, threads)
        clear_tail_kernel[blocks_tail, threads](
            np.int32(next_count),
            np.int32(total_discovered_end_index),
            ws["d_boundary_grid_indices_next"],
        )

    # Swap buffers for next iteration
    boundary_grid_indices_out = ws["d_boundary_grid_indices_next"]
    ws["d_boundary_grid_indices_next"] = boundary_grid_indices

    ws["stamp"] = stamp

    return (
        0,
        False,
        int(added_inc),
        int(removed_inc),
        int(num_external_new),
        boundary_grid_indices_out,
        solute_bgp_type_1d,
        index_discrete_epsilon_map_1d,
    )
