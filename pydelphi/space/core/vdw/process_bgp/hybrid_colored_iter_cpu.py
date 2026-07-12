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
Hybrid colored BGP backend.

Base semantics: serial_colored_iter_cpu.py with 125 colors.

This backend keeps the serial-colored APPLY order:

    for color_id in range(125):
        for this_bgp_index in active_frontier order:
            if color(center) == color_id:
                apply midpoint epsilon remaps
                update neighbor BGP status / append new BGPs
                remap sign-updated midpoint epsilons
                update center BGP status

The hybrid part is a per-color parallel PRECOMPUTE phase.  For all active BGPs
of the current color, it computes the expensive midpoint/contact decisions into
per-BGP buffers.  The actual mutation of index_discrete_epsilon_map_1d,
solute_bgp_type_1d, counters, and boundary_grid_indices is then replayed
serially in the same order as serial_colored_iter_cpu.py.

This is intended as a correctness-first bridge between 125-color serial and a
fully parallel BGP backend.  It intentionally avoids parallel appends and
parallel external-counter mutation.
"""

import numpy as np
from math import sqrt
from numba import njit, prange, set_num_threads

from pydelphi.foundation.enums import Precision
from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_int,
    delphi_real,
)

if PRECISION.int_value in {Precision.SINGLE.int_value}:
    import pydelphi.utils.prec.single as size_cpu
elif PRECISION.int_value == Precision.DOUBLE.int_value:
    import pydelphi.utils.prec.double as size_cpu
else:
    raise ValueError(f"Unsupported PRECISION: {PRECISION}")

from pydelphi.constants import (
    ConstDelPhiInts,
    NON_SOLUTE_BOUNDARY,
    ConstDelPhiFloats as ConstDelPhi,
)

EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value
APPROX_ZERO = ConstDelPhi.ApproxZero.value

from pydelphi.space.core.vdw.process_bgp.helpers_cpu import (
    _vdw_to_ms_all_voxels,
    _scan_candidates_and_find_closest_neighbor,
    _check_contact_region,
    _get_media_id,
    _remap_epsilon_map,
    _remap_possibly_sign_updated_epsilon_map,
)

from pydelphi.space.core.vdw.process_bgp.serial_iter_cpu import (
    _update_neighbor_status,
    _update_gridpoint_boundary_status,
)


@njit(nogil=True, boundscheck=False, cache=True, inline="always")
def _bgp_color_id_serial(ix: int, iy: int, iz: int, dtype_int=int) -> int:
    return dtype_int((ix % 5) + 5 * (iy % 5) + 25 * (iz % 5))


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _precompute_color_midpoint_decisions(
    color_id: int,
    boundary_point_start_index: int,
    boundary_point_end_index: int,
    x_stride: int,
    y_stride: int,
    z_stride: int,
    grid_origin_current: np.ndarray,
    grid_neighbor_coords_offsets: np.ndarray,
    boundary_grid_indices: np.ndarray,
    grid_spacing: float,
    grid_shape: np.ndarray,
    exposed_grids_coords: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
    solute_bgp_type_1d: np.ndarray,
    index_map: np.ndarray,
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
    probe_radius_squared_1: float,
    probe_radius_squared_2: float,
    active_mask: np.ndarray,  # uint8[num_frontier]
    action_remap: np.ndarray,  # uint8[num_frontier, 7]; 1 means remap+neighbor-status apply
    out_entity_ids: np.ndarray,  # int[num_frontier, 7]
    out_media_ids: np.ndarray,  # int[num_frontier, 7]
    dtype_int=int,
    dtype_real=float,
    dtype_bool=bool,
):
    """
    Parallel read-mostly midpoint/contact decision phase for one color.

    It mirrors the expensive decision logic in serial_iter_cpu._process_boundary_point_midpoint,
    but does not mutate global epsilon maps, solute_bgp_type_1d, counters, or
    boundary_grid_indices.  Those mutations are replayed serially by the caller.
    """
    num_frontier = boundary_point_end_index - boundary_point_start_index + 1

    x_stride_x_3 = dtype_int(x_stride * 3)
    y_stride_x_3 = dtype_int(y_stride * 3)

    # Local scalar offsets matching serial_colored_iter_cpu.py.
    off1 = dtype_int(0)
    off2 = dtype_int(1)
    off3 = dtype_int(2)
    off4 = dtype_int(-x_stride_x_3)
    off5 = dtype_int(-y_stride_x_3 + 1)
    off6 = dtype_int(-z_stride * 3 + 2)

    last_cube_indver_x = dtype_int(cube_shape_indver[0] - 1)
    last_cube_indver_y = dtype_int(cube_shape_indver[1] - 1)
    last_cube_indver_z = dtype_int(cube_shape_indver[2] - 1)

    for bi in prange(num_frontier):
        active_mask[bi] = 0
        for n in range(7):
            action_remap[bi, n] = 0
            out_entity_ids[bi, n] = 0
            out_media_ids[bi, n] = 0

        this_bgp_index = boundary_point_start_index + bi
        ixyz = boundary_grid_indices[this_bgp_index - 1]

        ix = dtype_int(ixyz[0])
        iy = dtype_int(ixyz[1])
        iz = dtype_int(ixyz[2])

        if _bgp_color_id_serial(ix, iy, iz, dtype_int=dtype_int) != color_id:
            continue

        grid_index_1d = dtype_int(ix * x_stride + iy * y_stride + iz * z_stride)
        if solute_bgp_type_1d[grid_index_1d] == NON_SOLUTE_BOUNDARY:
            continue

        active_mask[bi] = 1
        base3 = dtype_int(grid_index_1d * 3)

        # Load the six midpoint epsilon/entity/media values exactly like serial.
        eps_val = index_discrete_epsilon_map_1d[base3 + off1]
        out_entity_ids[bi, 1] = dtype_int(eps_val % epsilon_dimension)
        out_media_ids[bi, 1] = dtype_int(eps_val // epsilon_dimension)

        eps_val = index_discrete_epsilon_map_1d[base3 + off2]
        out_entity_ids[bi, 2] = dtype_int(eps_val % epsilon_dimension)
        out_media_ids[bi, 2] = dtype_int(eps_val // epsilon_dimension)

        eps_val = index_discrete_epsilon_map_1d[base3 + off3]
        out_entity_ids[bi, 3] = dtype_int(eps_val % epsilon_dimension)
        out_media_ids[bi, 3] = dtype_int(eps_val // epsilon_dimension)

        eps_val = index_discrete_epsilon_map_1d[base3 + off4]
        out_entity_ids[bi, 4] = dtype_int(eps_val % epsilon_dimension)
        out_media_ids[bi, 4] = dtype_int(eps_val // epsilon_dimension)

        eps_val = index_discrete_epsilon_map_1d[base3 + off5]
        out_entity_ids[bi, 5] = dtype_int(eps_val % epsilon_dimension)
        out_media_ids[bi, 5] = dtype_int(eps_val // epsilon_dimension)

        eps_val = index_discrete_epsilon_map_1d[base3 + off6]
        out_entity_ids[bi, 6] = dtype_int(eps_val % epsilon_dimension)
        out_media_ids[bi, 6] = dtype_int(eps_val // epsilon_dimension)

        rm_boundary_pt = (
            (1 < out_entity_ids[bi, 1] <= num_atoms + 1)
            or (1 < out_entity_ids[bi, 2] <= num_atoms + 1)
            or (1 < out_entity_ids[bi, 3] <= num_atoms + 1)
            or (1 < out_entity_ids[bi, 4] <= num_atoms + 1)
            or (1 < out_entity_ids[bi, 5] <= num_atoms + 1)
            or (1 < out_entity_ids[bi, 6] <= num_atoms + 1)
        )
        rm_boundary_pt = rm_boundary_pt and rm_boundary_pt_condition

        gx = dtype_real(grid_origin_current[0] + grid_spacing * dtype_real(ix))
        gy = dtype_real(grid_origin_current[1] + grid_spacing * dtype_real(iy))
        gz = dtype_real(grid_origin_current[2] + grid_spacing * dtype_real(iz))

        midpoint_coords = np.empty(3, dtype=dtype_real)
        cycle_flag = False

        for neighbor_index in range(1, 7):
            ent0 = out_entity_ids[bi, neighbor_index]
            med0 = out_media_ids[bi, neighbor_index]

            if (
                (ent0 == 0)
                or (rm_boundary_pt and ent0 > num_atoms + 1)
                or ((med0 == 0) and (ent0 > 0))
            ):
                probe_radius_squared = probe_radius_squared_2
                if (ent0 == 0) or (med0 == 0):
                    probe_radius_squared = probe_radius_squared_1

                midpoint_coords[0] = dtype_real(
                    gx + grid_neighbor_coords_offsets[neighbor_index, 0]
                )
                midpoint_coords[1] = dtype_real(
                    gy + grid_neighbor_coords_offsets[neighbor_index, 1]
                )
                midpoint_coords[2] = dtype_real(
                    gz + grid_neighbor_coords_offsets[neighbor_index, 2]
                )

                midpoint_index_x = dtype_int(
                    (midpoint_coords[0] - min_xyz[0]) * cube_side_indver_inverse
                )
                midpoint_index_y = dtype_int(
                    (midpoint_coords[1] - min_xyz[1]) * cube_side_indver_inverse
                )
                midpoint_index_z = dtype_int(
                    (midpoint_coords[2] - min_xyz[2]) * cube_side_indver_inverse
                )

                # Preserve serial behavior: out-of-cube is diagnostic-only there;
                # the search call still follows.
                if (
                    midpoint_index_x <= 0
                    or midpoint_index_y <= 0
                    or midpoint_index_z <= 0
                    or midpoint_index_x >= last_cube_indver_x
                    or midpoint_index_y >= last_cube_indver_y
                    or midpoint_index_z >= last_cube_indver_z
                ):
                    pass

                min_distance_squared = dtype_real(1000.0)
                closest_atom_or_object_index_voxel = dtype_int(0)

                (
                    cycle_flag,
                    closest_atom_or_object_index_voxel,
                    min_distance_squared,
                ) = _vdw_to_ms_all_voxels(
                    cycle_flag,
                    neighbor_index,
                    midpoint_index_x,
                    midpoint_index_y,
                    midpoint_index_z,
                    probe_radius_squared,
                    midpoint_coords,
                    out_entity_ids[bi],
                    out_media_ids[bi],
                    closest_atom_or_object_index_voxel,
                    min_distance_squared,
                    cube_voxel_start_indices,
                    cube_voxel_end_indices,
                    voxel_grid_point_indices,
                    exposed_grids_coords,
                )
                if cycle_flag:
                    continue

                indexing_cube_indices = (
                    (midpoint_coords - cube_vertex_lowest_xyz)
                    * cube_side_length_inverse
                ).astype(dtype_int)

                if size_cpu.or_lt_scalar(
                    indexing_cube_indices, dtype_int(0)
                ) or size_cpu.or_gt_vector(
                    indexing_cube_indices, cube_shape.astype(dtype_int)
                ):
                    lower_limit = dtype_int(1)
                    upper_limit = dtype_int(0)
                else:
                    lower_limit = voxel_atom_count[
                        indexing_cube_indices[0],
                        indexing_cube_indices[1],
                        indexing_cube_indices[2],
                    ]
                    upper_limit = voxel_atom_count_cumulative[
                        indexing_cube_indices[0],
                        indexing_cube_indices[1],
                        indexing_cube_indices[2],
                    ]

                min_distance_squared = dtype_real(100.0)
                closest_atom_or_object_index = dtype_int(0)

                (
                    closest_atom_or_object_index,
                    min_distance_squared,
                    atom_or_object_index,
                ) = _scan_candidates_and_find_closest_neighbor(
                    lower_limit=lower_limit,
                    upper_limit=upper_limit,
                    voxel_atom_ids=voxel_atom_ids,
                    atom_surface_flags=atom_surface_flags,
                    neighbor_entity_id=out_entity_ids[bi, neighbor_index],
                    neighbor_index=neighbor_index,
                    min_distance_squared=min_distance_squared,
                    closest_atom_or_object_index=closest_atom_or_object_index,
                    num_atoms=num_atoms,
                    atoms_data=atoms_data,
                    midpoint_coords=midpoint_coords,
                    dtype_int=dtype_int,
                    dtype_real=dtype_real,
                )

                if closest_atom_or_object_index != 0:
                    in_contact, _neighbor_atom_or_object_index = _check_contact_region(
                        midpoint_entity_id=out_entity_ids[bi, neighbor_index],
                        midpoint_coords=midpoint_coords,
                        closest_atom_or_object_index=closest_atom_or_object_index,
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
                        out_entity_ids[bi, neighbor_index] = dtype_int(
                            -closest_atom_or_object_index
                        )
                        out_media_ids[bi, neighbor_index] = dtype_int(
                            -closest_atom_or_object_index
                        )
                        continue

                # Serial path reaches here for cavity/reentrant/contact-region remap.
                out_entity_ids[bi, neighbor_index] = dtype_int(1)
                contact_region_media_id = _get_media_id(
                    closest_atom_or_object_index,
                    atom_or_object_index,
                    num_atoms,
                    atoms_data,
                    dtype_int=dtype_int,
                )
                out_media_ids[bi, neighbor_index] = dtype_int(contact_region_media_id)
                action_remap[bi, neighbor_index] = 1


@njit(nogil=True, boundscheck=False, cache=True)
def _process_bgp_one_iteration_core(
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
    active_mask: np.ndarray,
    action_remap: np.ndarray,
    out_entity_ids: np.ndarray,
    out_media_ids: np.ndarray,
    dtype_int=int,
    dtype_real=float,
    dtype_bool=bool,
):
    added_boundary_points_increment = dtype_int(0)
    removed_boundary_points_increment = dtype_int(0)

    x_stride_x_3 = dtype_int(x_stride * 3)
    y_stride_x_3 = dtype_int(y_stride * 3)

    cycle_flag = False
    neigh_update_exec_status = dtype_int(0)

    active_start = dtype_int(boundary_point_start_index)
    active_end = dtype_int(boundary_point_end_index)

    for color_id in range(125):
        _precompute_color_midpoint_decisions(
            color_id=color_id,
            boundary_point_start_index=active_start,
            boundary_point_end_index=active_end,
            x_stride=x_stride,
            y_stride=y_stride,
            z_stride=z_stride,
            grid_origin_current=grid_origin_current,
            grid_neighbor_coords_offsets=grid_neighbor_coords_offsets,
            boundary_grid_indices=boundary_grid_indices,
            grid_spacing=grid_spacing,
            grid_shape=grid_shape,
            exposed_grids_coords=exposed_grids_coords,
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
            solute_bgp_type_1d=solute_bgp_type_1d,
            index_map=index_map,
            epsilon_dimension=epsilon_dimension,
            rm_boundary_pt_condition=rm_boundary_pt_condition,
            min_xyz=min_xyz,
            cube_side_indver_inverse=cube_side_indver_inverse,
            cube_shape_indver=cube_shape_indver,
            cube_voxel_start_indices=cube_voxel_start_indices,
            cube_voxel_end_indices=cube_voxel_end_indices,
            cube_vertex_lowest_xyz=cube_vertex_lowest_xyz,
            cube_side_length_inverse=cube_side_length_inverse,
            cube_shape=cube_shape,
            voxel_grid_point_indices=voxel_grid_point_indices,
            voxel_atom_count=voxel_atom_count,
            voxel_atom_count_cumulative=voxel_atom_count_cumulative,
            voxel_atom_ids=voxel_atom_ids,
            atom_surface_flags=atom_surface_flags,
            num_atoms=num_atoms,
            atoms_data=atoms_data,
            atom_plus_probe_radii_1d=atom_plus_probe_radii_1d,
            atom_plus_probe_radii_shrink_1d=atom_plus_probe_radii_shrink_1d,
            probe_radius_squared_1=probe_radius_squared_1,
            probe_radius_squared_2=probe_radius_squared_2,
            active_mask=active_mask,
            action_remap=action_remap,
            out_entity_ids=out_entity_ids,
            out_media_ids=out_media_ids,
            dtype_int=dtype_int,
            dtype_real=dtype_real,
            dtype_bool=dtype_bool,
        )

        # Serial replay in the exact same active-frontier order as 125-color serial.
        for this_bgp_index in range(active_start, active_end + 1):
            bi = this_bgp_index - active_start
            if active_mask[bi] == 0:
                continue

            ixyz = boundary_grid_indices[this_bgp_index - 1]
            grid_index_x = dtype_int(ixyz[0])
            grid_index_y = dtype_int(ixyz[1])
            grid_index_z = dtype_int(ixyz[2])

            grid_neighs_entity_ids = out_entity_ids[bi]

            # Apply the remap+neighbor-status actions that the serial midpoint
            # function would have performed inside its neighbor loop.
            for neighbor_index in range(1, 7):
                if action_remap[bi, neighbor_index] == 0:
                    continue

                index_discrete_epsilon_map_1d = _remap_epsilon_map(
                    grid_index_x=grid_index_x,
                    grid_index_y=grid_index_y,
                    grid_index_z=grid_index_z,
                    stride_x_3=x_stride_x_3,
                    stride_y_3=y_stride_x_3,
                    index_map=index_map,
                    boundary_point_index=this_bgp_index,
                    neighbor_index=neighbor_index,
                    epsilon_dimension=epsilon_dimension,
                    contact_region_media_id=out_media_ids[bi, neighbor_index],
                    grid_neighs_entity_ids=grid_neighs_entity_ids,
                    index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
                    dtype_int=dtype_int,
                    dtype_real=dtype_real,
                )

                (
                    neigh_update_exec_status,
                    solute_bgp_type_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                ) = _update_neighbor_status(
                    grid_index_x,
                    grid_index_y,
                    grid_index_z,
                    neighbor_index,
                    epsilon_dimension,
                    grid_shape,
                    index_discrete_epsilon_map_1d,
                    boundary_grid_indices,
                    solute_bgp_type_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                    boundary_point_end_index,
                    max_boundary_grid_points,
                    dtype_int=dtype_int,
                )

                if neigh_update_exec_status == EXIT_NJIT_FLAG:
                    break

            if neigh_update_exec_status == EXIT_NJIT_FLAG:
                break

            # Same as serial-colored: after midpoint neighbor loop, remap possible
            # sign updates for all six midpoint directions.
            for neighbor_index in range(1, 7):
                index_discrete_epsilon_map_1d = (
                    _remap_possibly_sign_updated_epsilon_map(
                        grid_index_x=grid_index_x,
                        grid_index_y=grid_index_y,
                        grid_index_z=grid_index_z,
                        stride_x_3=x_stride_x_3,
                        stride_y_3=y_stride_x_3,
                        index_map=index_map,
                        boundary_point_index=this_bgp_index,
                        neighbor_index=neighbor_index,
                        epsilon_dimension=epsilon_dimension,
                        grid_neighs_entity_ids=grid_neighs_entity_ids,
                        index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
                        dtype_int=dtype_int,
                        dtype_real=dtype_real,
                    )
                )

            (
                neigh_update_exec_status,
                solute_bgp_type_1d,
                num_external_boundary_points,
                removed_boundary_points_increment,
            ) = _update_gridpoint_boundary_status(
                grid_index_x,
                grid_index_y,
                grid_index_z,
                epsilon_dimension,
                grid_shape,
                index_discrete_epsilon_map_1d,
                solute_bgp_type_1d,
                num_external_boundary_points,
                removed_boundary_points_increment,
                dtype_int=dtype_int,
            )

            if neigh_update_exec_status == EXIT_NJIT_FLAG:
                break

        if neigh_update_exec_status == EXIT_NJIT_FLAG:
            break

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
):
    """Interface-compatible Python wrapper with per-iteration scratch buffers."""
    if num_threads is not None and int(num_threads) > 0:
        set_num_threads(int(num_threads))

    num_frontier = int(boundary_point_end_index - boundary_point_start_index + 1)
    if num_frontier < 1:
        return (
            0,
            False,
            0,
            0,
            num_external_boundary_points,
            boundary_grid_indices,
            solute_bgp_type_1d,
            index_discrete_epsilon_map_1d,
        )

    active_mask = np.zeros(num_frontier, dtype=np.uint8)
    action_remap = np.zeros((num_frontier, 7), dtype=np.uint8)
    out_entity_ids = np.zeros((num_frontier, 7), dtype=boundary_grid_indices.dtype)
    out_media_ids = np.zeros((num_frontier, 7), dtype=boundary_grid_indices.dtype)

    return _process_bgp_one_iteration_core(
        num_threads,
        probe_radius_squared_1,
        probe_radius_squared_2,
        boundary_point_start_index,
        boundary_point_end_index,
        x_stride,
        y_stride,
        z_stride,
        grid_origin_current,
        max_boundary_grid_points,
        grid_neighbor_coords_offsets,
        boundary_grid_indices,
        grid_spacing,
        grid_shape,
        exposed_grids_coords,
        index_discrete_epsilon_map_1d,
        index_map,
        solute_bgp_type_1d,
        epsilon_dimension,
        rm_boundary_pt_condition,
        min_xyz,
        cube_side_indver_inverse,
        cube_shape_indver,
        cube_voxel_start_indices,
        cube_voxel_end_indices,
        cube_vertex_lowest_xyz,
        cube_side_length_inverse,
        cube_shape,
        voxel_grid_point_indices,
        voxel_atom_count,
        voxel_atom_count_cumulative,
        voxel_atom_ids,
        atom_surface_flags,
        num_atoms,
        atoms_data,
        atom_plus_probe_radii_1d,
        atom_plus_probe_radii_shrink_1d,
        num_external_boundary_points,
        active_mask,
        action_remap,
        out_entity_ids,
        out_media_ids,
        dtype_int=dtype_int,
        dtype_real=dtype_real,
        dtype_bool=dtype_bool,
    )
