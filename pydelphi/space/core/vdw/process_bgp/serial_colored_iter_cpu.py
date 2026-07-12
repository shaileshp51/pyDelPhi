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
Diagnostic colored-serial BGP backend.

Purpose
-------
This backend keeps the existing serial BGP logic, but processes the current
frontier in 27-color order:

    for color_id in range(27):
        for this_bgp_index in active_frontier:
            if color(center) == color_id:
                run the normal serial per-BGP logic

It is meant to answer:

    Does focused BGP diverge merely because of color ordering,
    or because of the parallel touched/commit/compact implementation?

This is NOT intended as a production backend.
"""

import numpy as np
from math import sqrt
from numba import njit

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
    _remap_possibly_sign_updated_epsilon_map,
)

# Reuse the existing serial implementation pieces directly.
# This keeps the midpoint/update semantics identical to serial_iter_cpu.py.
from pydelphi.space.core.vdw.process_bgp.serial_iter_cpu import (
    _process_boundary_point_midpoint,
    _update_gridpoint_boundary_status,
)


# @njit(nogil=True, boundscheck=False, cache=True, inline="always")
# def _bgp_color_id_serial(ix: int, iy: int, iz: int, dtype_int=int) -> int:
#     return dtype_int((ix % 3) + 3 * (iy % 3) + 9 * (iz % 3))


@njit(nogil=True, boundscheck=False, cache=True, inline="always")
def _bgp_color_id_serial(ix: int, iy: int, iz: int, dtype_int=int) -> int:
    return dtype_int((ix % 5) + 5 * (iy % 5) + 25 * (iz % 5))


@njit(nogil=True, boundscheck=False, cache=True)
def _process_bgp_one_iteration(
    num_threads: int,  # unused; kept for backend interface compatibility
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
    dtype_int=int,
    dtype_real=float,
    dtype_bool=bool,
):
    """
    Colored-serial version of serial_iter_cpu._process_bgp_one_iteration.

    Differences from normal serial:
      - outer loop is color_id = 0..26
      - inner loop scans the original active frontier
      - only BGP centers matching color_id are processed

    Preserved from serial:
      - midpoint handling
      - immediate neighbor-status update inside _process_boundary_point_midpoint
      - sign remap after each BGP
      - center/gridpoint boundary-status update after each BGP
      - added points are appended but not processed until the next orchestrator iteration
    """
    added_boundary_points_increment = dtype_int(0)
    removed_boundary_points_increment = dtype_int(0)

    grid_neighs_entity_ids = np.zeros(7, dtype=dtype_int)
    grid_neighs_media_ids = np.zeros(7, dtype=dtype_int)

    x_stride_x_3 = dtype_int(x_stride * 3)
    y_stride_x_3 = dtype_int(y_stride * 3)

    grid_neighs_1d_offsets = np.zeros(7, dtype=dtype_int)
    grid_neighs_1d_offsets[1] = dtype_int(0)
    grid_neighs_1d_offsets[2] = dtype_int(1)
    grid_neighs_1d_offsets[3] = dtype_int(2)
    grid_neighs_1d_offsets[4] = dtype_int(-x_stride_x_3)
    grid_neighs_1d_offsets[5] = dtype_int(-y_stride_x_3 + 1)
    grid_neighs_1d_offsets[6] = dtype_int(-z_stride * 3 + 2)

    cycle_flag = False
    num_cavity_points = dtype_int(0)
    neigh_update_exec_status = dtype_int(0)

    # Important: keep the active frontier fixed for this iteration.
    # Newly appended BGPs should not be processed until the orchestrator's next iteration.
    active_start = dtype_int(boundary_point_start_index)
    active_end = dtype_int(boundary_point_end_index)

    for color_id in range(125):
        for this_bgp_index in range(active_start, active_end + 1):
            ixyz = boundary_grid_indices[this_bgp_index - 1]

            grid_index_x = dtype_int(ixyz[0])
            grid_index_y = dtype_int(ixyz[1])
            grid_index_z = dtype_int(ixyz[2])

            if (
                _bgp_color_id_serial(
                    grid_index_x,
                    grid_index_y,
                    grid_index_z,
                    dtype_int=dtype_int,
                )
                != color_id
            ):
                continue

            grid_index_1d = dtype_int(
                grid_index_x * x_stride
                + grid_index_y * y_stride
                + grid_index_z * z_stride
            )
            grid_index_1d_times_3 = dtype_int(grid_index_1d * 3)

            if solute_bgp_type_1d[grid_index_1d] == NON_SOLUTE_BOUNDARY:
                continue

            rm_boundary_pt = False

            for neigh_id in range(1, 7):
                eps_val = index_discrete_epsilon_map_1d[
                    grid_index_1d_times_3 + grid_neighs_1d_offsets[neigh_id]
                ]
                grid_neighs_entity_ids[neigh_id] = dtype_int(
                    eps_val % epsilon_dimension
                )
                grid_neighs_media_ids[neigh_id] = dtype_int(
                    eps_val // epsilon_dimension
                )

            rm_boundary_pt = (
                (1 < grid_neighs_entity_ids[1] <= num_atoms + 1)
                or (1 < grid_neighs_entity_ids[2] <= num_atoms + 1)
                or (1 < grid_neighs_entity_ids[3] <= num_atoms + 1)
                or (1 < grid_neighs_entity_ids[4] <= num_atoms + 1)
                or (1 < grid_neighs_entity_ids[5] <= num_atoms + 1)
                or (1 < grid_neighs_entity_ids[6] <= num_atoms + 1)
                or rm_boundary_pt
            )

            rm_boundary_pt = rm_boundary_pt and rm_boundary_pt_condition

            (
                neigh_update_exec_status,
                grid_neighs_entity_ids,
                grid_neighs_media_ids,
                num_cavity_points,
                solute_bgp_type_1d,
                num_external_boundary_points,
                boundary_grid_indices,
                added_boundary_points_increment,
                removed_boundary_points_increment,
            ) = _process_boundary_point_midpoint(
                boundary_point_index=this_bgp_index,
                grid_index_x=grid_index_x,
                grid_index_y=grid_index_y,
                grid_index_z=grid_index_z,
                grid_spacing=grid_spacing,
                grid_shape=grid_shape,
                grid_origin=grid_origin_current,
                probe_radius_squared_1=probe_radius_squared_1,
                probe_radius_squared_2=probe_radius_squared_2,
                index_map=index_map,
                grid_neighbor_coords_offsets=grid_neighbor_coords_offsets,
                grid_neighs_entity_ids=grid_neighs_entity_ids,
                grid_neighs_media_ids=grid_neighs_media_ids,
                rm_boundary_pt=rm_boundary_pt,
                min_xyz=min_xyz,
                cube_side_indver_inverse=cube_side_indver_inverse,
                cube_shape_indver=cube_shape_indver,
                index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
                cycle_flag=cycle_flag,
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
                boundary_point_end_index=boundary_point_end_index,
                max_boundary_grid_points=max_boundary_grid_points,
                num_cavity_points=num_cavity_points,
                solute_bgp_type_1d=solute_bgp_type_1d,
                num_external_boundary_points=num_external_boundary_points,
                boundary_grid_indices=boundary_grid_indices,
                added_boundary_points_increment=added_boundary_points_increment,
                removed_boundary_points_increment=removed_boundary_points_increment,
                dtype_int=dtype_int,
                dtype_real=dtype_real,
                dtype_bool=dtype_bool,
            )

            if neigh_update_exec_status == EXIT_NJIT_FLAG:
                break

            # Same as serial: remap discrete_epsilon_map_1d after midpoint updates.
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

            # Same as serial: process the BGP center itself after neighbor remaps.
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
