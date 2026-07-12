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

import time
import numpy as np
from numba import njit, prange, set_num_threads
from math import sqrt

from pydelphi.foundation.enums import Precision
from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_int,
    delphi_real,
)
from pydelphi.constants import (
    NON_SOLUTE_BOUNDARY,
    SOLUTE_BOUNDARY_ANY,
    SOLUTE_BOUNDARY_EXTERNAL,
    ConstDelPhiInts,
)

# Dynamic precision handling
if PRECISION.int_value in {Precision.SINGLE.int_value}:
    import pydelphi.utils.prec.single as size_cpu
elif PRECISION.int_value == Precision.DOUBLE.int_value:
    import pydelphi.utils.prec.double as size_cpu
else:
    raise ValueError(f"Unsupported PRECISION: {PRECISION}")

EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value

# ---- CPU helpers ----
from pydelphi.space.core.vdw.process_bgp.helpers_cpu import (
    _vdw_to_ms_all_voxels_buffer,
    _scan_candidates_and_find_closest_neighbor,
    _check_contact_region,
    _get_media_id,
)

from pydelphi.space.core.vdw.process_bgp.helpers_parallel import (
    _color_count_events,
    _exclusive_prefix_sum_int32,
    _color_fill_events,
    _commit_events_and_build_next,
    _finalize_frontier_centers_gridpt_status,
    _compact_boundary_grid_indices_simple,
    _remap_epsilon_map_parallel_inplace,
    _remap_possibly_sign_updated_epsilon_map_parallel_inplace,
)

import numpy as np
from numba import get_num_threads


# --- add this small helper in cpu_parallel.py (Python side) ---
def make_cpu_commit_log_ws(log_cap_total: int = 20000):
    nthreads = int(get_num_threads())
    cap_per = max(1, int(log_cap_total) // max(1, nthreads))
    ws = {
        "nthreads": nthreads,
        "cap_per": int(cap_per),
        "log_cap_total": int(log_cap_total),
    }
    ws["log_count"] = np.zeros(nthreads, dtype=np.int32)
    ws["log_idx1d"] = np.empty((nthreads, cap_per), dtype=np.int32)
    ws["log_action"] = np.empty((nthreads, cap_per), dtype=np.int32)
    ws["log_color"] = np.empty((nthreads, cap_per), dtype=np.int32)
    ws["log_old_type"] = np.empty((nthreads, cap_per), dtype=np.int32)
    ws["log_new_type"] = np.empty((nthreads, cap_per), dtype=np.int32)
    ws["log_is_boundary"] = np.empty((nthreads, cap_per), dtype=np.int32)
    ws["log_is_external"] = np.empty((nthreads, cap_per), dtype=np.int32)
    ws["log_eid6"] = np.empty((nthreads, cap_per, 6), dtype=np.int16)
    ws["log_mid6"] = np.empty((nthreads, cap_per, 6), dtype=np.int16)
    return ws


def print_cpu_commit_log(ws, max_print: int = 5000, skip_action: int = 4):
    nthreads = ws["nthreads"]
    cap_per = ws["cap_per"]
    printed = 0
    for t in range(nthreads):
        nt = int(ws["log_count"][t])
        nt = min(nt, cap_per)
        for k in range(nt):
            act = int(ws["log_action"][t, k])
            if act == skip_action:
                continue
            # print(
            #     f"[CPU-LOG] tid={t:2d} k={k:4d} col={int(ws['log_color'][t,k]):2d} "
            #     f"act={act} idx1d={int(ws['log_idx1d'][t,k])} "
            #     f"old={int(ws['log_old_type'][t,k])} new={int(ws['log_new_type'][t,k])} "
            #     f"bnd={int(ws['log_is_boundary'][t,k])} ext={int(ws['log_is_external'][t,k])} "
            #     f"eid6={ws['log_eid6'][t,k,:].tolist()} mid6={ws['log_mid6'][t,k,:].tolist()}"
            # )
            printed += 1
            if printed >= max_print:
                return


@njit(cache=True, inline="always")
def _cpu_try_log_event(
    log_enabled: int,
    cap_per: int,
    log_count,
    log_idx1d,
    log_action,
    log_color,
    log_old_type,
    log_new_type,
    log_is_boundary,
    log_is_external,
    log_eid6,
    log_mid6,
    idx1d: int,
    action: int,
    color_id: int,
    old_type: int,
    new_type: int,
    is_boundary: int,
    is_external: int,
    eid6,  # (6,) int16
    mid6,  # (6,) int16
):
    if log_enabled == 0:
        return

    tid = get_thread_id()
    k = log_count[tid]
    if k >= cap_per:
        return

    log_idx1d[tid, k] = idx1d
    log_action[tid, k] = action
    log_color[tid, k] = color_id
    log_old_type[tid, k] = old_type
    log_new_type[tid, k] = new_type
    log_is_boundary[tid, k] = is_boundary
    log_is_external[tid, k] = is_external

    for j in range(6):
        log_eid6[tid, k, j] = eid6[j]
        log_mid6[tid, k, j] = mid6[j]

    log_count[tid] = k + 1


def print_cpu_commit_log(ws, max_print: int = 5000, skip_action: int = 4):
    """
    Same spirit as your CUDA print loop; prints at most max_print entries.
    """
    nthreads = ws["nthreads"]
    cap_per = ws["cap_per"]
    total = int(ws["log_count"].sum())
    if total == 0:
        return

    printed = 0
    for t in range(nthreads):
        nt = int(ws["log_count"][t])
        nt = min(nt, cap_per)
        for k in range(nt):
            act = int(ws["log_action"][t, k])
            if act == skip_action:
                continue
            # print(
            #     f"[CPU-LOG] tid={t:2d} k={k:4d} col={int(ws['log_color'][t,k]):2d} "
            #     f"act={act} idx1d={int(ws['log_idx1d'][t,k])} "
            #     f"old={int(ws['log_old_type'][t,k])} new={int(ws['log_new_type'][t,k])} "
            #     f"bnd={int(ws['log_is_boundary'][t,k])} ext={int(ws['log_is_external'][t,k])} "
            #     f"eid6={ws['log_eid6'][t,k,:].tolist()} mid6={ws['log_mid6'][t,k,:].tolist()}"
            # )
            printed += 1
            if printed >= max_print:
                return


# @njit(nogil=True, boundscheck=False, cache=True)
# def _bgp_color_id(ix: int, iy: int, iz: int, dtype_int=int) -> int:
#     return dtype_int((ix % 3) + 3 * (iy % 3) + 9 * (iz % 3))


@njit(nogil=True, boundscheck=False, cache=True, inline="always")
def _bgp_color_id(ix: int, iy: int, iz: int, dtype_int=int) -> int:
    return dtype_int((ix % 5) + 5 * (iy % 5) + 25 * (iz % 5))


@njit(nogil=True, boundscheck=False, cache=True)
def _process_boundary_point_midpoint_parallel_noalloc(
    boundary_point_index: int,
    grid_index_x: int,
    grid_index_y: int,
    grid_index_z: int,
    grid_spacing: float,
    grid_shape: np.ndarray,
    grid_origin: np.ndarray,
    probe_radius_squared_1: float,
    probe_radius_squared_2: float,
    index_map: np.ndarray,
    grid_neighbor_coords_offsets: np.ndarray,
    # scratch rows (preallocated per bi)
    grid_neighs_entity_ids_row: np.ndarray,  # (7,)
    grid_neighs_media_ids_row: np.ndarray,  # (7,)
    midpoint_coords_row: np.ndarray,  # (3,)
    rm_boundary_pt: bool,
    min_xyz: np.ndarray,
    cube_side_indver_inverse: float,
    cube_shape_indver: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,  # mutated in-place
    cycle_flag: bool,
    cube_voxel_start_indices: np.ndarray,
    cube_voxel_end_indices: np.ndarray,
    grid_point_indices_in_voxels: np.ndarray,
    exposed_grids_coords: np.ndarray,
    cube_vertex_lowest_xyz: np.ndarray,
    cube_side_length_inverse: float,
    cube_shape: np.ndarray,
    voxel_atom_count: np.ndarray,
    voxel_atom_count_cumulative: np.ndarray,
    voxel_atom_ids: np.ndarray,
    atom_surface_flags: np.ndarray,
    num_atoms: int,
    atoms_data: np.ndarray,
    atom_plus_probe_radii_1d: np.ndarray,
    atom_plus_probe_radii_shrink_1d: np.ndarray,
    epsilon_dimension: int,
    num_cavity_points: int,
    dtype_int=int,
    dtype_real=float,
    dtype_bool=bool,
):
    exec_status = dtype_int(0)
    probe_radius = sqrt(probe_radius_squared_1)

    z_stride = dtype_int(1)
    y_stride = dtype_int(grid_shape[2])
    x_stride = dtype_int(grid_shape[1]) * y_stride

    gx = dtype_real(grid_origin[0] + grid_spacing * dtype_real(grid_index_x))
    gy = dtype_real(grid_origin[1] + grid_spacing * dtype_real(grid_index_y))
    gz = dtype_real(grid_origin[2] + grid_spacing * dtype_real(grid_index_z))

    last_x = dtype_int(cube_shape_indver[0] - 1)
    last_y = dtype_int(cube_shape_indver[1] - 1)
    last_z = dtype_int(cube_shape_indver[2] - 1)

    # compute center 1d*3
    ny_nz = grid_shape[1] * grid_shape[2]
    nz = grid_shape[2]
    idx1d = grid_index_x * ny_nz + grid_index_y * nz + grid_index_z
    base3 = idx1d * 3

    # same offsets as serial
    x_stride_x_3 = (ny_nz) * 3
    y_stride_x_3 = (nz) * 3
    grid_neighs_1d_offsets = np.empty(7, dtype_int)
    grid_neighs_1d_offsets[1:4] = 0, 1, 2
    grid_neighs_1d_offsets[4:7] = -x_stride_x_3, -y_stride_x_3 + 1, -dtype_int(3) + 2

    for neigh_id in range(1, 7):
        eps_idx = index_discrete_epsilon_map_1d[
            base3 + grid_neighs_1d_offsets[neigh_id]
        ]
        grid_neighs_entity_ids_row[neigh_id] = eps_idx % epsilon_dimension
        grid_neighs_media_ids_row[neigh_id] = eps_idx // epsilon_dimension

    for neighbor_index in range(1, 7):

        if (
            (grid_neighs_entity_ids_row[neighbor_index] == 0)
            or (
                rm_boundary_pt
                and grid_neighs_entity_ids_row[neighbor_index] > num_atoms + 1
            )
            or (
                (grid_neighs_media_ids_row[neighbor_index] == 0)
                and (grid_neighs_entity_ids_row[neighbor_index] > 0)
            )
        ):
            probe_radius_squared = probe_radius_squared_2
            if (grid_neighs_entity_ids_row[neighbor_index] == 0) or (
                grid_neighs_media_ids_row[neighbor_index] == 0
            ):
                probe_radius_squared = probe_radius_squared_1

            mx = dtype_real(gx + grid_neighbor_coords_offsets[neighbor_index, 0])
            my = dtype_real(gy + grid_neighbor_coords_offsets[neighbor_index, 1])
            mz = dtype_real(gz + grid_neighbor_coords_offsets[neighbor_index, 2])

            midpoint_coords_row[0] = mx
            midpoint_coords_row[1] = my
            midpoint_coords_row[2] = mz

            mid_ix = dtype_int((mx - min_xyz[0]) * cube_side_indver_inverse)
            mid_iy = dtype_int((my - min_xyz[1]) * cube_side_indver_inverse)
            mid_iz = dtype_int((mz - min_xyz[2]) * cube_side_indver_inverse)

            # out-of-range just falls through
            # if (
            #     mid_ix <= 0
            #     or mid_iy <= 0
            #     or mid_iz <= 0
            #     or mid_ix >= last_x
            #     or mid_iy >= last_y
            #     or mid_iz >= last_z
            # ):
            #     continue

            min_d2 = dtype_real(1000.0)
            closest_vox = dtype_int(0)

            cycle_flag, closest_vox, min_d2 = _vdw_to_ms_all_voxels_buffer(
                cycle_flag,
                neighbor_index,
                mid_ix,
                mid_iy,
                mid_iz,
                probe_radius_squared,
                midpoint_coords_row,
                grid_neighs_entity_ids_row,
                grid_neighs_media_ids_row,
                closest_vox,
                min_d2,
                cube_voxel_start_indices,
                cube_voxel_end_indices,
                grid_point_indices_in_voxels,
                exposed_grids_coords,
            )

            if cycle_flag:
                continue

            vix = dtype_int((mx - cube_vertex_lowest_xyz[0]) * cube_side_length_inverse)
            viy = dtype_int((my - cube_vertex_lowest_xyz[1]) * cube_side_length_inverse)
            viz = dtype_int((mz - cube_vertex_lowest_xyz[2]) * cube_side_length_inverse)

            if (
                vix < 0
                or viy < 0
                or viz < 0
                or vix > dtype_int(cube_shape[0])
                or viy > dtype_int(cube_shape[1])
                or viz > dtype_int(cube_shape[2])
            ):
                lower = dtype_int(1)
                upper = dtype_int(0)
            else:
                lower = voxel_atom_count[vix, viy, viz]
                upper = voxel_atom_count_cumulative[vix, viy, viz]

            min_d2 = dtype_real(100.0)
            closest = dtype_int(0)

            closest, min_d2, atom_or_object_index = (
                _scan_candidates_and_find_closest_neighbor(
                    lower_limit=lower,
                    upper_limit=upper,
                    voxel_atom_ids=voxel_atom_ids,
                    atom_surface_flags=atom_surface_flags,
                    neighbor_entity_id=grid_neighs_entity_ids_row[neighbor_index],
                    neighbor_index=neighbor_index,
                    min_distance_squared=min_d2,
                    closest_atom_or_object_index=closest,
                    num_atoms=num_atoms,
                    atoms_data=atoms_data,
                    midpoint_coords=midpoint_coords_row,
                    dtype_int=dtype_int,
                    dtype_real=dtype_real,
                )
            )

            if closest == 0:
                num_cavity_points += 1
            else:
                in_contact, _ = _check_contact_region(
                    midpoint_entity_id=grid_neighs_entity_ids_row[neighbor_index],
                    midpoint_coords=midpoint_coords_row,
                    closest_atom_or_object_index=closest,
                    num_atoms=num_atoms,
                    atoms_data=atoms_data,
                    atom_plus_probe_radii_1d=atom_plus_probe_radii_1d,
                    atom_plus_probe_radii_shrink_1d=atom_plus_probe_radii_shrink_1d,
                    cube_side_length_inverse=cube_side_length_inverse,
                    cube_shape=cube_shape,
                    cube_vertex_lowest_xyz=cube_vertex_lowest_xyz,
                    voxel_atom_count=voxel_atom_count,
                    voxel_atom_count_cumulative=voxel_atom_count_cumulative,
                    voxel_atom_ids=voxel_atom_ids,
                    dtype_int=dtype_int,
                    dtype_real=dtype_real,
                    dtype_bool=dtype_bool,
                )

                if in_contact:
                    grid_neighs_entity_ids_row[neighbor_index] = -closest
                    grid_neighs_media_ids_row[neighbor_index] = -closest

                    continue

            grid_neighs_entity_ids_row[neighbor_index] = dtype_int(1)

            contact_region_media_id = _get_media_id(
                closest,
                atom_or_object_index,
                num_atoms,
                atoms_data,
                dtype_int=dtype_int,
            )

            _remap_epsilon_map_parallel_inplace(
                grid_index_x=grid_index_x,
                grid_index_y=grid_index_y,
                grid_index_z=grid_index_z,
                stride_x_3=x_stride * 3,
                stride_y_3=y_stride * 3,
                index_map=index_map,
                neighbor_index=neighbor_index,
                epsilon_dimension=epsilon_dimension,
                contact_region_media_id=contact_region_media_id,
                neighbor_entity_id=grid_neighs_entity_ids_row[neighbor_index],
                index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
                dtype_int=dtype_int,
                dtype_real=dtype_real,
            )

            grid_neighs_media_ids_row[neighbor_index] = dtype_int(
                contact_region_media_id
            )

    return exec_status, num_cavity_points


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _color_midpoint_phase_strict_noalloc(
    color_id: int,
    boundary_point_start_index: int,
    boundary_point_end_index: int,
    boundary_grid_indices: np.ndarray,
    grid_spacing: float,
    grid_shape: np.ndarray,
    grid_origin_current: np.ndarray,
    probe_radius_squared_1: float,
    probe_radius_squared_2: float,
    index_map: np.ndarray,
    grid_neighbor_coords_offsets: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,  # mutated in-place
    solute_bgp_type_1d: np.ndarray,
    epsilon_dimension: int,
    rm_boundary_pt_condition: bool,
    min_xyz: np.ndarray,
    cube_side_indver_inverse: float,
    cube_shape_indver: np.ndarray,
    cube_voxel_start_indices: np.ndarray,
    cube_voxel_end_indices: np.ndarray,
    voxel_grid_point_indices: np.ndarray,
    exposed_grids_coords: np.ndarray,
    cube_vertex_lowest_xyz: np.ndarray,
    cube_side_length_inverse: float,
    cube_shape: np.ndarray,
    voxel_atom_count: np.ndarray,
    voxel_atom_count_cumulative: np.ndarray,
    voxel_atom_ids: np.ndarray,
    atom_surface_flags: np.ndarray,
    num_atoms: int,
    atoms_data: np.ndarray,
    atom_plus_probe_radii_1d: np.ndarray,
    atom_plus_probe_radii_shrink_1d: np.ndarray,
    neigh_entity_ids_buf: np.ndarray,  # (num_frontier_bgps,7)
    neigh_media_ids_buf: np.ndarray,  # (num_frontier_bgps,7)
    midpoint_coords_buf: np.ndarray,  # (num_frontier_bgps,3)
    num_threads: int,  # kept for signature stability
    dtype_int=int,
    dtype_real=float,
    dtype_bool=bool,
):
    exec_status = dtype_int(0)

    num_frontier_bgps = boundary_point_end_index - boundary_point_start_index + 1
    x_stride = dtype_int(grid_shape[1] * grid_shape[2])
    y_stride = dtype_int(grid_shape[2])

    for bi in prange(num_frontier_bgps):
        bgp_index = boundary_point_start_index + bi
        ixyz = boundary_grid_indices[bgp_index - 1]
        ix = dtype_int(ixyz[0])
        iy = dtype_int(ixyz[1])
        iz = dtype_int(ixyz[2])

        if _bgp_color_id(ix, iy, iz, dtype_int=dtype_int) != color_id:
            continue

        idx1d = ix * x_stride + iy * y_stride + iz
        if solute_bgp_type_1d[idx1d] == NON_SOLUTE_BOUNDARY:
            continue

        ent_row = neigh_entity_ids_buf[bi]
        med_row = neigh_media_ids_buf[bi]
        xyz_row = midpoint_coords_buf[bi]

        # build rm_boundary_pt exactly like serial
        rm = False
        base = idx1d * 3

        x_stride_x_3 = x_stride * 3
        y_stride_x_3 = y_stride * 3

        # neighbor offsets (match your serial definition)
        o1 = 0
        o2 = 1
        o3 = 2
        o4 = -x_stride_x_3
        o5 = -y_stride_x_3 + 1
        o6 = -3 + 2  # == -1

        # unrolled is safest for parfors
        eps_idx = index_discrete_epsilon_map_1d[base + o1]
        ent = eps_idx % epsilon_dimension
        if 1 < ent <= num_atoms + 1:
            rm = True

        eps_idx = index_discrete_epsilon_map_1d[base + o2]
        ent = eps_idx % epsilon_dimension
        if 1 < ent <= num_atoms + 1:
            rm = True

        eps_idx = index_discrete_epsilon_map_1d[base + o3]
        ent = eps_idx % epsilon_dimension
        if 1 < ent <= num_atoms + 1:
            rm = True

        eps_idx = index_discrete_epsilon_map_1d[base + o4]
        ent = eps_idx % epsilon_dimension
        if 1 < ent <= num_atoms + 1:
            rm = True

        eps_idx = index_discrete_epsilon_map_1d[base + o5]
        ent = eps_idx % epsilon_dimension
        if 1 < ent <= num_atoms + 1:
            rm = True

        eps_idx = index_discrete_epsilon_map_1d[base + o6]
        ent = eps_idx % epsilon_dimension
        if 1 < ent <= num_atoms + 1:
            rm = True

        rm_boundary_pt = rm and rm_boundary_pt_condition

        local_status, _ncav = _process_boundary_point_midpoint_parallel_noalloc(
            boundary_point_index=bgp_index,
            grid_index_x=ix,
            grid_index_y=iy,
            grid_index_z=iz,
            grid_spacing=grid_spacing,
            grid_shape=grid_shape,
            grid_origin=grid_origin_current,
            probe_radius_squared_1=probe_radius_squared_1,
            probe_radius_squared_2=probe_radius_squared_2,
            index_map=index_map,
            grid_neighbor_coords_offsets=grid_neighbor_coords_offsets,
            grid_neighs_entity_ids_row=ent_row,
            grid_neighs_media_ids_row=med_row,
            midpoint_coords_row=xyz_row,
            rm_boundary_pt=dtype_bool(rm_boundary_pt),
            min_xyz=min_xyz,
            cube_side_indver_inverse=cube_side_indver_inverse,
            cube_shape_indver=cube_shape_indver,
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
            cycle_flag=False,
            cube_voxel_start_indices=cube_voxel_start_indices,
            cube_voxel_end_indices=cube_voxel_end_indices,
            grid_point_indices_in_voxels=voxel_grid_point_indices,
            exposed_grids_coords=exposed_grids_coords,
            cube_vertex_lowest_xyz=cube_vertex_lowest_xyz,
            cube_side_length_inverse=cube_side_length_inverse,
            cube_shape=cube_shape,
            voxel_atom_count=voxel_atom_count,
            voxel_atom_count_cumulative=voxel_atom_count_cumulative,
            voxel_atom_ids=voxel_atom_ids,
            atom_surface_flags=atom_surface_flags,
            num_atoms=num_atoms,
            atoms_data=atoms_data,
            atom_plus_probe_radii_1d=atom_plus_probe_radii_1d,
            atom_plus_probe_radii_shrink_1d=atom_plus_probe_radii_shrink_1d,
            epsilon_dimension=epsilon_dimension,
            num_cavity_points=dtype_int(0),
            dtype_int=dtype_int,
            dtype_real=dtype_real,
            dtype_bool=dtype_bool,
        )

        for neigh in range(1, 7):
            _remap_possibly_sign_updated_epsilon_map_parallel_inplace(
                grid_index_x=ix,
                grid_index_y=iy,
                grid_index_z=iz,
                stride_x_3=(grid_shape[1] * grid_shape[2])
                * 3,  # or x_stride*3 if consistent
                stride_y_3=(grid_shape[2]) * 3,  # or y_stride*3
                index_map=index_map,
                neighbor_index=neigh,
                epsilon_dimension=epsilon_dimension,
                neighbor_entity_id=ent_row[neigh],  # <-- updated ids (may be negative)
                index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
                dtype_int=dtype_int,
                dtype_real=dtype_real,
            )

        if local_status == EXIT_NJIT_FLAG:
            exec_status = EXIT_NJIT_FLAG

    return exec_status


#
# # -----------------------------------------------------------------------------
# # ONLY interface: _process_bgp_one_iteration (signature preserved)
# # -----------------------------------------------------------------------------
# def _process_bgp_one_iteration(
#     num_threads: int,
#     probe_radius_squared_1: float,
#     probe_radius_squared_2: float,
#     boundary_point_start_index: int,
#     boundary_point_end_index: int,
#     x_stride: int,
#     y_stride: int,
#     z_stride: int,
#     grid_origin_current: np.ndarray,
#     max_boundary_grid_points: int,
#     grid_neighbor_coords_offsets: np.ndarray,
#     boundary_grid_indices: np.ndarray,
#     grid_spacing: float,
#     grid_shape: np.ndarray,
#     exposed_grids_coords: np.ndarray,
#     index_discrete_epsilon_map_1d: np.ndarray,
#     index_map: np.ndarray,
#     solute_bgp_type_1d: np.ndarray,
#     epsilon_dimension: int,
#     rm_boundary_pt_condition: bool,
#     min_xyz: np.ndarray,
#     cube_side_indver_inverse: float,
#     cube_shape_indver: np.ndarray,
#     cube_voxel_start_indices: np.ndarray,
#     cube_voxel_end_indices: np.ndarray,
#     cube_vertex_lowest_xyz: np.ndarray,
#     cube_side_length_inverse: float,
#     cube_shape: np.ndarray,
#     voxel_grid_point_indices: np.ndarray,
#     voxel_atom_count: np.ndarray,
#     voxel_atom_count_cumulative: np.ndarray,
#     voxel_atom_ids: np.ndarray,
#     atom_surface_flags: np.ndarray,
#     num_atoms: int,
#     atoms_data: np.ndarray,
#     atom_plus_probe_radii_1d: np.ndarray,
#     atom_plus_probe_radii_shrink_1d: np.ndarray,
#     num_external_boundary_points: int,
#     dtype_int=delphi_int,
#     dtype_real=delphi_real,
#     dtype_bool=bool,
# ):
#     """
#     Parallel driver for one VdW->MS boundary iteration.
#     Interface preserved.
#
#     Algorithm:
#       - For each of 27 colors:
#           (A) midpoint update (strict no-alloc in njit/prange, safe under coloring)
#           (B) build touched set matching serial visitation for num_frontier_bgps (neighbor_index=1..7)
#           (C) commit-by-recompute (serial update rules), dedup touched points via processed_stamp
#     """
#     set_num_threads(num_threads)
#     cycle_flag = False
#     neigh_update_exec_status = 0
#
#     added_boundary_points_increment = 0
#     removed_boundary_points_increment = 0
#
#     grid_npoints = int(grid_shape[0] * grid_shape[1] * grid_shape[2])
#
#     visited_next = np.zeros(grid_npoints, dtype=np.int32)
#     processed_stamp = np.zeros(grid_npoints, dtype=np.int32)
#
#     stamp = np.int32(1)
#
#     new_bgps_idx1d = np.zeros(max_boundary_grid_points, dtype=np.int64)
#     new_count = 0
#
#     num_discovered_bndy_grid_points = boundary_point_end_index
#     num_frontier_bgps = boundary_point_end_index - boundary_point_start_index + 1
#
#     # Python-side scratch (allowed): no allocations inside prange
#     neigh_entity_ids_buf = np.zeros(
#         (num_frontier_bgps, 7), dtype=boundary_grid_indices.dtype
#     )
#     neigh_media_ids_buf = np.zeros(
#         (num_frontier_bgps, 7), dtype=boundary_grid_indices.dtype
#     )
#     midpoint_coords_buf = np.zeros(
#         (num_frontier_bgps, 3), dtype=grid_origin_current.dtype
#     )
#
#     for color_id in range(27):
#         # advance stamp per color (critical: both visited_next + processed_stamp use this)
#         stamp = np.int32(stamp + 1)
#
#         # (A) midpoint update
#         mid_status = _color_midpoint_phase_strict_noalloc(
#             color_id=color_id,
#             boundary_point_start_index=boundary_point_start_index,
#             boundary_point_end_index=boundary_point_end_index,
#             boundary_grid_indices=boundary_grid_indices,
#             grid_spacing=grid_spacing,
#             grid_shape=grid_shape,
#             grid_origin_current=grid_origin_current,
#             probe_radius_squared_1=probe_radius_squared_1,
#             probe_radius_squared_2=probe_radius_squared_2,
#             index_map=index_map,
#             grid_neighbor_coords_offsets=grid_neighbor_coords_offsets,
#             index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
#             solute_bgp_type_1d=solute_bgp_type_1d,
#             epsilon_dimension=epsilon_dimension,
#             rm_boundary_pt_condition=rm_boundary_pt_condition,
#             min_xyz=min_xyz,
#             cube_side_indver_inverse=cube_side_indver_inverse,
#             cube_shape_indver=cube_shape_indver,
#             cube_voxel_start_indices=cube_voxel_start_indices,
#             cube_voxel_end_indices=cube_voxel_end_indices,
#             voxel_grid_point_indices=voxel_grid_point_indices,
#             exposed_grids_coords=exposed_grids_coords,
#             cube_vertex_lowest_xyz=cube_vertex_lowest_xyz,
#             cube_side_length_inverse=cube_side_length_inverse,
#             cube_shape=cube_shape,
#             voxel_atom_count=voxel_atom_count,
#             voxel_atom_count_cumulative=voxel_atom_count_cumulative,
#             voxel_atom_ids=voxel_atom_ids,
#             atom_surface_flags=atom_surface_flags,
#             num_atoms=num_atoms,
#             atoms_data=atoms_data,
#             atom_plus_probe_radii_1d=atom_plus_probe_radii_1d,
#             atom_plus_probe_radii_shrink_1d=atom_plus_probe_radii_shrink_1d,
#             neigh_entity_ids_buf=neigh_entity_ids_buf,
#             neigh_media_ids_buf=neigh_media_ids_buf,
#             midpoint_coords_buf=midpoint_coords_buf,
#             num_threads=num_threads,
#             dtype_int=dtype_int,
#             dtype_real=dtype_real,
#             dtype_bool=dtype_bool,
#         )
#
#         if mid_status == EXIT_NJIT_FLAG:
#             neigh_update_exec_status = EXIT_NJIT_FLAG
#             break
#
#         # (B) touched-set count/fill for num_frontier_bgps
#         counts, total_touched = _color_count_events(
#             color_id=color_id,
#             boundary_point_start_index=boundary_point_start_index,
#             boundary_point_end_index=boundary_point_end_index,
#             boundary_grid_indices=boundary_grid_indices,
#             grid_shape=grid_shape,
#             solute_bgp_type_1d=solute_bgp_type_1d,
#             dtype_int=dtype_int,
#         )
#
#         if int(total_touched) == 0:
#             continue
#
#         offsets = _exclusive_prefix_sum_int32(counts)
#
#         touched_idx1d = np.empty(int(total_touched), dtype=np.int64)
#
#         _color_fill_events(
#             color_id=color_id,
#             boundary_point_start_index=boundary_point_start_index,
#             boundary_point_end_index=boundary_point_end_index,
#             boundary_grid_indices=boundary_grid_indices,
#             grid_shape=grid_shape,
#             solute_bgp_type_1d=solute_bgp_type_1d,
#             counts=counts,
#             offsets=offsets,
#             event_idx1d=touched_idx1d,
#             dtype_int=dtype_int,
#         )
#
#         # (C) commit-by-recompute (serial rules), dedup touched via processed_stamp
#         (
#             commit_status,
#             boundary_grid_indices,
#             solute_bgp_type_1d,
#             num_external_boundary_points,
#             added_boundary_points_increment,
#             removed_boundary_points_increment,
#             new_bgps_idx1d,
#             new_count,
#         ) = _commit_events_and_build_next(
#             event_idx1d=touched_idx1d,
#             num_events=int(total_touched),
#             grid_shape=grid_shape,
#             boundary_grid_indices=boundary_grid_indices,
#             solute_bgp_type_1d=solute_bgp_type_1d,
#             index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
#             epsilon_dimension=epsilon_dimension,
#             num_discovered_bndy_grid_points=num_discovered_bndy_grid_points,
#             max_boundary_grid_points=max_boundary_grid_points,
#             num_external_boundary_points=num_external_boundary_points,
#             added_boundary_points_increment=added_boundary_points_increment,
#             removed_boundary_points_increment=removed_boundary_points_increment,
#             visited_next=visited_next,
#             stamp=stamp,
#             new_bgps_idx1d=new_bgps_idx1d,
#             new_count=new_count,
#             processed_stamp=processed_stamp,
#             dtype_int=dtype_int,
#         )
#
#         if commit_status == EXIT_NJIT_FLAG:
#             neigh_update_exec_status = EXIT_NJIT_FLAG
#             break
#
#         num_discovered_bndy_grid_points = (
#             boundary_point_end_index + added_boundary_points_increment
#         )
#
#     # --- FINALIZE: apply serial gridpoint-center status update for the frontier ---
#     num_external_boundary_points, removed_boundary_points_increment = (
#         _finalize_frontier_centers_gridpt_status(
#             boundary_point_start_index=boundary_point_start_index,
#             boundary_point_end_index=boundary_point_end_index,
#             boundary_grid_indices=boundary_grid_indices,
#             grid_shape=grid_shape,
#             epsilon_dimension=epsilon_dimension,
#             index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
#             solute_bgp_type_1d=solute_bgp_type_1d,
#             num_external_boundary_points=num_external_boundary_points,
#             removed_boundary_points_increment=removed_boundary_points_increment,
#             dtype_int=dtype_int,
#         )
#     )
#
#     # after all 27 colors + finalize
#     total_discovered_end_index = (
#         boundary_point_end_index + added_boundary_points_increment
#     )
#
#     stamp = np.int32(stamp + 1)
#
#     next_count = _compact_boundary_grid_indices_simple(
#         new_bgps_idx1d=new_bgps_idx1d,
#         new_count=new_count,
#         total_discovered_end_index=total_discovered_end_index,
#         boundary_grid_indices=boundary_grid_indices,
#         solute_bgp_type_1d=solute_bgp_type_1d,
#         index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
#         epsilon_dimension=epsilon_dimension,
#         grid_shape=grid_shape,
#         stamp_arr=visited_next,
#         stamp=stamp,
#         dtype_int=dtype_int,
#     )
#
#     boundary_point_start_index = 1
#     boundary_point_end_index = int(next_count)
#
#     print("Finished parallel bgp iteration.")
#
#     return (
#         neigh_update_exec_status,
#         cycle_flag,
#         added_boundary_points_increment,
#         removed_boundary_points_increment,
#         num_external_boundary_points,
#         boundary_grid_indices,
#         solute_bgp_type_1d,
#         index_discrete_epsilon_map_1d,
#     )


def _fingerprint_i64(arr: np.ndarray):
    """
    Cheap, stable-ish fingerprint for debugging discrepancies.
    Avoids full hashing; uses int64 sum + xor-reduction on a sampled view.
    """
    if arr is None:
        return (0, 0, 0)
    a = np.asarray(arr)
    n = a.size
    if n == 0:
        return (0, 0, 0)

    # view as int64 for stable ops (works for int32/int64/bool; floats will reinterpret bits)
    v = a.view(np.int64) if a.dtype.itemsize == 8 else a.astype(np.int64, copy=False)

    s = int(v.sum(dtype=np.int64))
    # sample stride to keep it cheap on huge arrays
    stride = max(1, n // 262144)  # sample up to ~262k items
    sample = v.ravel()[::stride]
    x = int(np.bitwise_xor.reduce(sample, dtype=np.int64))
    return (n, s, x)


def _process_bgp_one_iteration(
    num_threads: int,
    probe_radius_squared_1: float,
    probe_radius_squared_2: float,
    boundary_point_start_index: int,
    boundary_point_end_index: int,
    x_stride: int,
    y_stride: int,
    z_stride: int,
    grid_origin_current: np.ndarray,
    max_boundary_grid_points: int,
    grid_neighbor_coords_offsets: np.ndarray,
    boundary_grid_indices: np.ndarray,
    grid_spacing: float,
    grid_shape: np.ndarray,
    exposed_grids_coords: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
    index_map: np.ndarray,
    solute_bgp_type_1d: np.ndarray,
    epsilon_dimension: int,
    rm_boundary_pt_condition: bool,
    min_xyz: np.ndarray,
    cube_side_indver_inverse: float,
    cube_shape_indver: np.ndarray,
    cube_voxel_start_indices: np.ndarray,
    cube_voxel_end_indices: np.ndarray,
    cube_vertex_lowest_xyz: np.ndarray,
    cube_side_length_inverse: float,
    cube_shape: np.ndarray,
    voxel_grid_point_indices: np.ndarray,
    voxel_atom_count: np.ndarray,
    voxel_atom_count_cumulative: np.ndarray,
    voxel_atom_ids: np.ndarray,
    atom_surface_flags: np.ndarray,
    num_atoms: int,
    atoms_data: np.ndarray,
    atom_plus_probe_radii_1d: np.ndarray,
    atom_plus_probe_radii_shrink_1d: np.ndarray,
    num_external_boundary_points: int,
    dtype_int=delphi_int,
    dtype_real=delphi_real,
    dtype_bool=bool,
    # --- logging knobs ---
    log: bool = False,
    log_per_color: bool = True,
    log_fingerprints: bool = False,
    log_first_k_touched: int = 0,  # set e.g. 10 to print first few touched idx1d for a color
):
    """
    Parallel driver for one VdW->MS boundary iteration with CPU logging.

    Prints per-color timing + counts so you can compare with CUDA logs.
    """
    set_num_threads(num_threads)

    t_iter0 = time.perf_counter()

    cycle_flag = False
    neigh_update_exec_status = 0

    added_boundary_points_increment = 0
    removed_boundary_points_increment = 0

    grid_npoints = int(grid_shape[0] * grid_shape[1] * grid_shape[2])

    visited_next = np.zeros(grid_npoints, dtype=np.int32)
    processed_stamp = np.zeros(grid_npoints, dtype=np.int32)
    stamp = np.int32(1)

    new_bgps_idx1d = np.zeros(max_boundary_grid_points, dtype=np.int64)
    new_count = 0

    num_discovered_bndy_grid_points = boundary_point_end_index
    num_frontier_bgps = boundary_point_end_index - boundary_point_start_index + 1

    # Python-side scratch (allowed): no allocations inside prange
    neigh_entity_ids_buf = np.zeros(
        (num_frontier_bgps, 7), dtype=boundary_grid_indices.dtype
    )
    neigh_media_ids_buf = np.zeros(
        (num_frontier_bgps, 7), dtype=boundary_grid_indices.dtype
    )
    midpoint_coords_buf = np.zeros(
        (num_frontier_bgps, 3), dtype=grid_origin_current.dtype
    )

    if log:
        print(
            f"[CPU-BGP] start: frontier={num_frontier_bgps} "
            f"start={boundary_point_start_index} end={boundary_point_end_index} "
            f"grid_shape={tuple(int(x) for x in grid_shape)} threads={num_threads} "
            f"rm_boundary_pt_condition={rm_boundary_pt_condition}"
        )
        if log_fingerprints:
            fp_eps0 = _fingerprint_i64(index_discrete_epsilon_map_1d)
            fp_bgp0 = _fingerprint_i64(boundary_grid_indices)
            fp_typ0 = _fingerprint_i64(solute_bgp_type_1d)
            print(
                f"[CPU-BGP] fp(before) eps(n,sum,xor)={fp_eps0} bgp={fp_bgp0} type={fp_typ0}"
            )

    # accumulators for iteration-level timing
    t_mid_total = 0.0
    t_cnt_total = 0.0
    t_ps_total = 0.0
    t_fill_total = 0.0
    t_commit_total = 0.0

    total_touched_all_colors = 0

    cpu_log_ws = make_cpu_commit_log_ws(20000)
    if log:
        print(
            f"[CPU-LOG] nthreads={cpu_log_ws['nthreads']} cap_per={cpu_log_ws['cap_per']}"
        )

    # --- [DROP-IN #2] DELETE these two lines from inside the color loop ---
    # REMOVE / DELETE (do not keep):
    #     cpu_log_ws = make_cpu_commit_log_ws(20000)
    #     cpu_log_ws["log_count"].fill(0)
    #
    # Replace with JUST this reset (keep inside loop, right before commit):
    cpu_log_ws["log_count"].fill(0)

    # --- [DROP-IN #3] ADD THIS immediately after _commit_events_and_build_next returns ---
    # Put this right after the commit call block (after dt_commit is computed is fine):
    if log:
        print_cpu_commit_log(cpu_log_ws, max_print=50000, skip_action=4)

    for color_id in range(125):
        stamp = np.int32(stamp + 1)

        t0 = time.perf_counter()
        mid_status = _color_midpoint_phase_strict_noalloc(
            color_id=color_id,
            boundary_point_start_index=boundary_point_start_index,
            boundary_point_end_index=boundary_point_end_index,
            boundary_grid_indices=boundary_grid_indices,
            grid_spacing=grid_spacing,
            grid_shape=grid_shape,
            grid_origin_current=grid_origin_current,
            probe_radius_squared_1=probe_radius_squared_1,
            probe_radius_squared_2=probe_radius_squared_2,
            index_map=index_map,
            grid_neighbor_coords_offsets=grid_neighbor_coords_offsets,
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
            solute_bgp_type_1d=solute_bgp_type_1d,
            epsilon_dimension=epsilon_dimension,
            rm_boundary_pt_condition=rm_boundary_pt_condition,
            min_xyz=min_xyz,
            cube_side_indver_inverse=cube_side_indver_inverse,
            cube_shape_indver=cube_shape_indver,
            cube_voxel_start_indices=cube_voxel_start_indices,
            cube_voxel_end_indices=cube_voxel_end_indices,
            voxel_grid_point_indices=voxel_grid_point_indices,
            exposed_grids_coords=exposed_grids_coords,
            cube_vertex_lowest_xyz=cube_vertex_lowest_xyz,
            cube_side_length_inverse=cube_side_length_inverse,
            cube_shape=cube_shape,
            voxel_atom_count=voxel_atom_count,
            voxel_atom_count_cumulative=voxel_atom_count_cumulative,
            voxel_atom_ids=voxel_atom_ids,
            atom_surface_flags=atom_surface_flags,
            num_atoms=num_atoms,
            atoms_data=atoms_data,
            atom_plus_probe_radii_1d=atom_plus_probe_radii_1d,
            atom_plus_probe_radii_shrink_1d=atom_plus_probe_radii_shrink_1d,
            neigh_entity_ids_buf=neigh_entity_ids_buf,
            neigh_media_ids_buf=neigh_media_ids_buf,
            midpoint_coords_buf=midpoint_coords_buf,
            num_threads=num_threads,
            dtype_int=dtype_int,
            dtype_real=dtype_real,
            dtype_bool=dtype_bool,
        )
        t1 = time.perf_counter()
        dt_mid = t1 - t0
        t_mid_total += dt_mid

        if mid_status == EXIT_NJIT_FLAG:
            neigh_update_exec_status = EXIT_NJIT_FLAG
            if log:
                print(f"[CPU-BGP] color={color_id:02d} midpoint: EXIT_NJIT_FLAG")
            break

        t0 = time.perf_counter()
        counts, total_touched = _color_count_events(
            color_id=color_id,
            boundary_point_start_index=boundary_point_start_index,
            boundary_point_end_index=boundary_point_end_index,
            boundary_grid_indices=boundary_grid_indices,
            grid_shape=grid_shape,
            solute_bgp_type_1d=solute_bgp_type_1d,
            dtype_int=dtype_int,
        )
        t1 = time.perf_counter()
        dt_cnt = t1 - t0
        t_cnt_total += dt_cnt

        total_touched_i = int(total_touched)
        total_touched_all_colors += total_touched_i

        if total_touched_i == 0:
            if log and log_per_color:
                print(
                    f"[CPU-BGP] color={color_id:02d} "
                    f"t_mid={dt_mid:.6f}s t_cnt={dt_cnt:.6f}s touched=0 "
                    f"added={added_boundary_points_increment} removed={removed_boundary_points_increment} "
                    f"external={num_external_boundary_points}"
                )
            continue

        t0 = time.perf_counter()
        offsets = _exclusive_prefix_sum_int32(counts)
        t1 = time.perf_counter()
        dt_ps = t1 - t0
        t_ps_total += dt_ps

        touched_idx1d = np.empty(total_touched_i, dtype=np.int64)

        t0 = time.perf_counter()
        _color_fill_events(
            color_id=color_id,
            boundary_point_start_index=boundary_point_start_index,
            boundary_point_end_index=boundary_point_end_index,
            boundary_grid_indices=boundary_grid_indices,
            grid_shape=grid_shape,
            solute_bgp_type_1d=solute_bgp_type_1d,
            counts=counts,
            offsets=offsets,
            event_idx1d=touched_idx1d,
            dtype_int=dtype_int,
        )
        t1 = time.perf_counter()
        dt_fill = t1 - t0
        t_fill_total += dt_fill

        cpu_log_ws = make_cpu_commit_log_ws(20000)

        # before each commit call (per color), reset log_count:
        cpu_log_ws["log_count"].fill(0)

        t0 = time.perf_counter()
        (
            commit_status,
            boundary_grid_indices,
            solute_bgp_type_1d,
            num_external_boundary_points,
            added_boundary_points_increment,
            removed_boundary_points_increment,
            new_bgps_idx1d,
            new_count,
        ) = _commit_events_and_build_next(
            event_idx1d=touched_idx1d,
            num_events=total_touched_i,
            grid_shape=grid_shape,
            boundary_grid_indices=boundary_grid_indices,
            solute_bgp_type_1d=solute_bgp_type_1d,
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
            epsilon_dimension=epsilon_dimension,
            num_discovered_bndy_grid_points=num_discovered_bndy_grid_points,
            max_boundary_grid_points=max_boundary_grid_points,
            num_external_boundary_points=num_external_boundary_points,
            added_boundary_points_increment=added_boundary_points_increment,
            removed_boundary_points_increment=removed_boundary_points_increment,
            visited_next=visited_next,
            stamp=stamp,
            new_bgps_idx1d=new_bgps_idx1d,
            new_count=new_count,
            processed_stamp=processed_stamp,
            dtype_int=dtype_int,
            # ---- debug log (CPU) ----
            log_enabled=1 if log else 0,
            log_color_id=int(color_id),
            log_cap_per=int(cpu_log_ws["cap_per"]),
            log_count=cpu_log_ws["log_count"],
            log_idx1d=cpu_log_ws["log_idx1d"],
            log_action=cpu_log_ws["log_action"],
            log_color=cpu_log_ws["log_color"],
            log_old_type=cpu_log_ws["log_old_type"],
            log_new_type=cpu_log_ws["log_new_type"],
            log_is_boundary=cpu_log_ws["log_is_boundary"],
            log_is_external=cpu_log_ws["log_is_external"],
            log_eid6=cpu_log_ws["log_eid6"],
            log_mid6=cpu_log_ws["log_mid6"],
        )
        t1 = time.perf_counter()
        dt_commit = t1 - t0
        t_commit_total += dt_commit

        if log:
            print_cpu_commit_log(cpu_log_ws, max_print=50000, skip_action=4)

        if log and log_per_color:
            msg = (
                f"[CPU-BGP] color={color_id:02d} "
                f"t_mid={dt_mid:.6f}s t_cnt={dt_cnt:.6f}s t_ps={dt_ps:.6f}s "
                f"t_fill={dt_fill:.6f}s t_commit={dt_commit:.6f}s "
                f"touched={total_touched_i} new_count={int(new_count)} "
                f"added={added_boundary_points_increment} removed={removed_boundary_points_increment} "
                f"external={int(num_external_boundary_points)}"
            )
            print(msg)

            if log_first_k_touched > 0:
                k = min(log_first_k_touched, touched_idx1d.size)
                print(f"           touched_idx1d[:{k}] = {touched_idx1d[:k].tolist()}")

            if log_fingerprints:
                fp_eps = _fingerprint_i64(index_discrete_epsilon_map_1d)
                fp_typ = _fingerprint_i64(solute_bgp_type_1d)
                print(f"           fp eps={fp_eps} type={fp_typ}")

        if commit_status == EXIT_NJIT_FLAG:
            neigh_update_exec_status = EXIT_NJIT_FLAG
            if log:
                print(f"[CPU-BGP] color={color_id:02d} commit: EXIT_NJIT_FLAG")
            break

        num_discovered_bndy_grid_points = (
            boundary_point_end_index + added_boundary_points_increment
        )

    # FINALIZE (timed)
    t0 = time.perf_counter()
    num_external_boundary_points, removed_boundary_points_increment = (
        _finalize_frontier_centers_gridpt_status(
            boundary_point_start_index=boundary_point_start_index,
            boundary_point_end_index=boundary_point_end_index,
            boundary_grid_indices=boundary_grid_indices,
            grid_shape=grid_shape,
            epsilon_dimension=epsilon_dimension,
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
            solute_bgp_type_1d=solute_bgp_type_1d,
            num_external_boundary_points=num_external_boundary_points,
            removed_boundary_points_increment=removed_boundary_points_increment,
            dtype_int=dtype_int,
        )
    )
    t1 = time.perf_counter()
    dt_finalize = t1 - t0

    total_discovered_end_index = (
        boundary_point_end_index + added_boundary_points_increment
    )

    stamp = np.int32(stamp + 1)

    t0 = time.perf_counter()
    next_count = _compact_boundary_grid_indices_simple(
        new_bgps_idx1d=new_bgps_idx1d,
        new_count=new_count,
        total_discovered_end_index=total_discovered_end_index,
        boundary_grid_indices=boundary_grid_indices,
        solute_bgp_type_1d=solute_bgp_type_1d,
        index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
        epsilon_dimension=epsilon_dimension,
        grid_shape=grid_shape,
        stamp_arr=visited_next,
        stamp=stamp,
        dtype_int=dtype_int,
    )
    t1 = time.perf_counter()
    dt_compact = t1 - t0

    boundary_point_start_index = 1
    boundary_point_end_index = int(next_count)

    t_iter1 = time.perf_counter()
    dt_iter = t_iter1 - t_iter0

    if log:
        print(
            f"[CPU-BGP] done: "
            f"added={added_boundary_points_increment} removed={removed_boundary_points_increment} "
            f"external={int(num_external_boundary_points)} next_count={int(next_count)} "
            f"touched_total={int(total_touched_all_colors)}"
        )
        print(
            f"[CPU-BGP] timing: iter={dt_iter:.6f}s "
            f"mid={t_mid_total:.6f}s count={t_cnt_total:.6f}s prefix={t_ps_total:.6f}s "
            f"fill={t_fill_total:.6f}s commit={t_commit_total:.6f}s "
            f"finalize={dt_finalize:.6f}s compact={dt_compact:.6f}s"
        )
        if log_fingerprints:
            fp_eps1 = _fingerprint_i64(index_discrete_epsilon_map_1d)
            fp_bgp1 = _fingerprint_i64(boundary_grid_indices)
            fp_typ1 = _fingerprint_i64(solute_bgp_type_1d)
            print(
                f"[CPU-BGP] fp(after)  eps(n,sum,xor)={fp_eps1} bgp={fp_bgp1} type={fp_typ1}"
            )

    return (
        neigh_update_exec_status,
        cycle_flag,
        added_boundary_points_increment,
        removed_boundary_points_increment,
        num_external_boundary_points,
        boundary_grid_indices,
        solute_bgp_type_1d,
        index_discrete_epsilon_map_1d,
    )
