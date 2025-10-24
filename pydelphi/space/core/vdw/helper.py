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
Core routines for molecular surface generation and boundary point refinement.

This module provides a collection of high-performance, Numba-accelerated functions
essential for constructing and processing molecular surfaces on a 3D grid. These
operations are fundamental pre-processing steps for grid-based biophysical
calculations, such as solving the Poisson-Boltzmann equation for electrostatic
potentials.

The main functionalities include:
- Identifying initial boundary points on a "zeta-surface".
- Converting a Van der Waals (VDW) surface to a Molecular Surface (MS) by
  assessing solvent accessibility using a probe sphere.
- Elaborating and consolidating the final list of dielectric boundary grid
  points, preparing them for subsequent calculations.

The functions are heavily optimized for speed using Numba's JIT compilation and
are designed to work seamlessly within the PyDelphi ecosystem, relying on its
specific data structures and configurations.
"""

import numpy as np
from numba import njit

from pydelphi.foundation.enums import Precision

from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
    nprint_cpu_if_verbose as nprint_cpu,
)
from pydelphi.config.logging_config import (
    DEBUG,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

from pydelphi.constants import ConstDelPhiFloats as ConstDelPhi
from pydelphi.constants import ConstDelPhiInts as ConstDelPhiInts
from pydelphi.constants import (
    NEIGHBOR_VOXEL_RELATIVE_COORDINATES as NEIGHBOR_VOXEL_REL_COORDS,
)

if PRECISION.value == Precision.DOUBLE.value:
    from pydelphi.utils.prec.double import or_lt_vector, or_gt_vector
else:
    from pydelphi.utils.prec.single import or_lt_vector, or_gt_vector

RESIZE_FACTOR = ConstDelPhi.ZetaArrayResizeFactor.value
INITIAL_SIZE_PERCENT = ConstDelPhi.ZetaArrayInitialSizePercent.value
EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value


@njit(nogil=True, boundscheck=False, cache=True)
def build_zeta_surface_map(
    grid_spacing,
    grid_shape,
    gridbox_center,
    indices_mid_grid,
    zeta_surface_map_1d,
    index_discrete_epsilon_map_1d,
    epsdim,
    zeta_surf_grid_coords,
    zeta_surf_grid_indices,
    zeta_coords_capacity,
    zeta_indices_capacity,
    num_zeta_surf_grid_coords,
    num_zeta_surf_grid_indices,
):
    """
    Finding the boundary points on the zeta-surface with resizable output arrays.

    Args:
        grid_spacing (float): Grid spacing.
        grid_shape (tuple): Shape of the grid (lx, ly, lz).
        gridbox_center (numpy array): Center of the grid box.
        indices_mid_grid (numpy array): Mid-grid indices.
        zeta_surface_map_1d (numpy array): 1D zeta surface map.
        index_discrete_epsilon_map_1d (numpy array): 1D discrete epsilon map.
        epsdim (int): Epsilon dimension.
        zeta_surf_grid_coords (numpy array): Initially allocated array for zeta surface coordinates.
        zeta_surf_grid_indices (numpy array): Initially allocated array for zeta surface indices.
        zeta_coords_capacity (int): Current allocated capacity of coords array.
        zeta_indices_capacity (int): Current allocated capacity of indices array.
        num_zeta_surf_grid_coords (numba.int64): Current count of zeta surface coordinates written.
        num_zeta_surf_grid_indices (numba.int64): Current count of zeta surface indices written.

    Returns:
        tuple: (zeta_surf_grid_coords, zeta_surf_grid_indices, num_zeta_point_coords, num_zeta_point_indices,
                zeta_coords_capacity, zeta_indices_capacity)  return values
               Returns potentially resized arrays and updated counts and capacities.
    """
    zeta_tmp = np.zeros(7, dtype=delphi_bool)
    z_stride = 1
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride

    x_stride_x_3 = x_stride * 3
    y_stride_x_3 = y_stride * 3
    z_stride_x_3 = 3

    for lx in range(1, grid_shape[0] - 1):
        for ly in range(1, grid_shape[1] - 1):
            for lz in range(1, grid_shape[2] - 1):
                zext = 0
                zbgp = 0
                lxyz1d = lx * x_stride + ly * y_stride + lz
                lxyz1d_x_3 = lxyz1d * 3

                zeta_tmp[0] = zeta_surface_map_1d[lxyz1d]
                # six neighbor midpoints
                zeta_tmp[1] = zeta_surface_map_1d[lxyz1d + 1]
                zeta_tmp[2] = zeta_surface_map_1d[lxyz1d - 1]
                zeta_tmp[3] = zeta_surface_map_1d[lxyz1d + y_stride]
                zeta_tmp[4] = zeta_surface_map_1d[lxyz1d - y_stride]
                zeta_tmp[5] = zeta_surface_map_1d[lxyz1d + x_stride]
                zeta_tmp[6] = zeta_surface_map_1d[lxyz1d - x_stride]

                zright = abs(index_discrete_epsilon_map_1d[lxyz1d_x_3]) // epsdim
                zfront = abs(index_discrete_epsilon_map_1d[lxyz1d_x_3 + 1]) // epsdim
                ztop = abs(index_discrete_epsilon_map_1d[lxyz1d_x_3 + 2]) // epsdim
                zleft = (
                    abs(index_discrete_epsilon_map_1d[lxyz1d_x_3 - x_stride_x_3])
                    // epsdim
                )
                zback = (
                    abs(index_discrete_epsilon_map_1d[lxyz1d_x_3 - y_stride_x_3 + 1])
                    // epsdim
                )
                zbottom = (
                    abs(index_discrete_epsilon_map_1d[lxyz1d_x_3 - z_stride_x_3 + 2])
                    // epsdim
                )

                # external point
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

                for midpoint_index in range(1, 7):
                    if zeta_tmp[midpoint_index] != zeta_tmp[midpoint_index - 1]:
                        zbgp = 1  # bnd point

                if zeta_tmp[6] != zeta_tmp[1]:
                    zbgp = 1  # bnd point

                if zbgp > 0 and zext == 1:
                    lxyz = np.array([lx, ly, lz], dtype=delphi_int)
                    xyz = (
                        gridbox_center
                        + (lxyz.astype(delphi_real) - indices_mid_grid) * grid_spacing
                    )
                    num_zeta_surf_grid_coords_x_3 = num_zeta_surf_grid_coords * 3
                    # Check if resizing is needed for coordinates array
                    if (num_zeta_surf_grid_coords_x_3 + 3) > zeta_coords_capacity:
                        new_coords_size = int(zeta_coords_capacity * RESIZE_FACTOR)
                        temp_coords_array = np.zeros(new_coords_size, dtype=delphi_real)
                        temp_coords_array[:zeta_coords_capacity] = zeta_surf_grid_coords
                        zeta_surf_grid_coords = temp_coords_array
                        zeta_coords_capacity = new_coords_size

                    zeta_surf_grid_coords[num_zeta_surf_grid_coords_x_3 + 0] = xyz[0]
                    zeta_surf_grid_coords[num_zeta_surf_grid_coords_x_3 + 1] = xyz[1]
                    zeta_surf_grid_coords[num_zeta_surf_grid_coords_x_3 + 2] = xyz[2]

                    # Check if resizing is needed for indices array
                    if (num_zeta_surf_grid_coords_x_3 + 3) > zeta_indices_capacity:
                        new_indices_size = int(zeta_indices_capacity * RESIZE_FACTOR)
                        temp_indices_array = np.zeros(
                            new_indices_size, dtype=delphi_int
                        )
                        temp_indices_array[:zeta_indices_capacity] = (
                            zeta_surf_grid_indices
                        )
                        zeta_surf_grid_indices = temp_indices_array
                        zeta_indices_capacity = new_indices_size

                    zeta_surf_grid_indices[num_zeta_surf_grid_coords_x_3] = lx
                    zeta_surf_grid_indices[num_zeta_surf_grid_coords_x_3 + 1] = ly
                    zeta_surf_grid_indices[num_zeta_surf_grid_coords_x_3 + 2] = lz

                    num_zeta_surf_grid_coords += 1
                    num_zeta_surf_grid_indices += 1

    return (
        zeta_surf_grid_coords,
        zeta_surf_grid_indices,
        num_zeta_surf_grid_coords,
        num_zeta_surf_grid_indices,
        zeta_coords_capacity,
        zeta_indices_capacity,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def vdw_to_ms_all_voxels(
    cycle_flag,
    midpoint_index,
    grid_x_index,
    grid_y_index,
    grid_z_index,
    probe_radius_squared,
    grid_midpoint_coords,
    grid_neighs_entity_ids,
    grid_neighs_media_ids,
    closest_exposed_grid_index,
    minimum_distance,
    cube_voxel_start_indices,
    cube_voxel_end_indices,
    grid_point_indices_in_voxels,
    exposed_grid_coordinates,
):
    """
    Calculates the Van der Waals to Molecular Surface (VDW to MS) for all neighboring voxels.

    This function iterates through all neighboring voxels of a given grid point and determines
    whether the atom at the current grid point is within the probe radius of any other atom
    in the neighboring voxels. It also identifies the closest atom.

    Args:
        cycle_flag (bool): A flag indicating whether a cycle has been detected.
        midpoint_index (int): The index of the atom being processed.
        grid_x_index (int): The x-coordinate of the grid point.
        grid_y_index (int): The y-coordinate of the grid point.
        grid_z_index (int): The z-coordinate of the grid point.
        probe_radius_squared (float): The squared probe radius.
        grid_midpoint_coords (numpy.ndarray): The coordinates of the atom.
        grid_neighs_entity_ids (numpy.ndarray): An array to store epsilon values.
        grid_neighs_media_ids (numpy.ndarray): An array to store squared epsilon values.
        closest_exposed_grid_index (int): The index of the closest exposed grid point in exposed_grids array.
        minimum_distance (float): The minimum distance to the closest atom.
        cube_voxel_start_indices (numpy.ndarray): An array of lower bounds for atom indices.
        cube_voxel_end_indices (numpy.ndarray): An array of upper bounds for atom indices.
        grid_point_indices_in_voxels (numpy.ndarray): An array of cumulative atom indices.
        exposed_grid_coordinates (numpy.ndarray): An array of exposed grid coordinates.

    Returns:
        tuple: A tuple containing the updated cycle_flag, closest_exposed_grid_index, and minimum_distance.
    """
    previous_voxel = np.array((0, 0, 0), dtype=np.int32)
    # iterate over all the 3x3x3=27 neighboring cubes and calculate voxel coords for each atom
    for voxel_coordinate in NEIGHBOR_VOXEL_REL_COORDS:
        relative_voxel_change = voxel_coordinate - previous_voxel

        grid_x_index, grid_y_index, grid_z_index = (
            grid_x_index + relative_voxel_change[0],
            grid_y_index + relative_voxel_change[1],
            grid_z_index + relative_voxel_change[2],
        )
        lower_limit = cube_voxel_start_indices[grid_x_index][grid_y_index][grid_z_index]
        upper_limit = cube_voxel_end_indices[grid_x_index][grid_y_index][grid_z_index]

        cycle_flag = False
        for current_atom_index in range(lower_limit, upper_limit + 1):
            grid_index_in_cumulative = grid_point_indices_in_voxels[current_atom_index]
            grid_difference = (
                grid_midpoint_coords
                - exposed_grid_coordinates[grid_index_in_cumulative]
            )
            distance_squared = (
                grid_difference[0] * grid_difference[0]
                + grid_difference[1] * grid_difference[1]
                + grid_difference[2] * grid_difference[2]
            )

            if distance_squared < probe_radius_squared:
                grid_neighs_entity_ids[midpoint_index] = -1
                grid_neighs_media_ids[midpoint_index] = -1
                cycle_flag = True
            elif distance_squared < minimum_distance:
                closest_exposed_grid_index = grid_index_in_cumulative
                minimum_distance = distance_squared

        previous_voxel[:] = voxel_coordinate

        if cycle_flag:
            break

    return cycle_flag, closest_exposed_grid_index, minimum_distance


@njit(nogil=True, boundscheck=False, cache=True)
def surface_elaborate_boundary_gridpoints(
    num_boundary_grid_indices,
    epsilon_dimension,
    max_boundary_grid_points,
    grid_shape,
    boundary_grid_points,
    boundary_grid_indices,
    dielectric_boundary_midpoints_1d,
    index_discrete_epsilon_map_1d,
    index_map,
):
    """
    Consolidates the list of boundary grid points, removing dead points and adding new ones.

    This function iterates through the provided boundary grid indices, updates the dielectric
    boundary and discrete epsilon maps, and populates the boundary grid points array.

    Args:
        num_boundary_grid_indices (int): The number of boundary grid indices.
        epsilon_dimension (int): The dimension of the epsilon values.
        max_boundary_grid_points (int): The maximum number of boundary grid points allowed.
        grid_shape (tuple): The shape of the grid (x, y, z).
        boundary_grid_points (numpy.ndarray): Array to store boundary grid point coordinates.
        boundary_grid_indices (numpy.ndarray): Array of boundary grid indices.
        dielectric_boundary_midpoints_1d (numpy.ndarray): 1D array representing the dielectric boundary.
        index_discrete_epsilon_map_1d (numpy.ndarray): 1D array representing the discrete epsilon map.
        index_map (numpy.ndarray): Array of neighbor offsets used for grid navigation.

    Returns:
        tuple: A tuple containing the number of boundary grid points and the updated
               boundary_grid_points array. Returns EXIT_NJIT_RETURN_VALUE and the
               boundary_grid_points array if an error occurs.
    """
    # consolidate the list, removing dead boundary points, adding new ones..
    boundary_grid_point_count = 0

    # Array is resized keeping old values in the memory.
    current_boundary_grid_points_count = len(boundary_grid_points)
    if 0 < current_boundary_grid_points_count < max_boundary_grid_points:
        boundary_grid_points_new = np.zeros(
            (max_boundary_grid_points + 1, 3), dtype=delphi_int
        )
        boundary_grid_points_new[
            : boundary_grid_points.shape[0], : boundary_grid_points.shape[1]
        ] = boundary_grid_points
        boundary_grid_points = boundary_grid_points_new
    else:
        boundary_grid_points = np.zeros(
            (max_boundary_grid_points + 1, 3), dtype=delphi_int
        )

    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride

    # bgp: boundary_grid_point.
    # NOTE: boundary_grid_indices array stores values with starting index 0.
    smallest_non_box_boundary_index = np.ones(3, dtype=delphi_int)
    larget_non_box_boundary_index = (grid_shape - 2).astype(delphi_int)
    for this_bgp_index in range(1, num_boundary_grid_indices + 1):
        grid_index_xyz = boundary_grid_indices[this_bgp_index - 1]
        # Skip this dielectric boundary grid point (bgp) if it is a gridbox boundary point,
        # as bgp must have all 6 neighbors defined for induced surface charge calculation.
        if or_lt_vector(
            grid_index_xyz, smallest_non_box_boundary_index
        ) or or_gt_vector(grid_index_xyz, larget_non_box_boundary_index):
            continue
        grid_x_index = grid_index_xyz[0]
        grid_y_index = grid_index_xyz[1]
        grid_z_index = grid_index_xyz[2]
        grid_index_1d = grid_x_index * x_stride + grid_y_index * y_stride + grid_z_index
        grid_index_1d_x_3 = grid_index_1d * 3
        if dielectric_boundary_midpoints_1d[grid_index_1d_x_3 + 1] != 0:
            boundary_grid_point_count += 1
            dielectric_boundary_midpoints_1d[grid_index_1d_x_3 + 1] = (
                boundary_grid_point_count
            )

            # Precaution not to exceed array size
            if boundary_grid_point_count <= max_boundary_grid_points:
                boundary_grid_points[boundary_grid_point_count - 1] = grid_index_xyz
            else:
                nprint_cpu(
                    DEBUG,
                    _VERBOSITY,
                    " VdMS> j=",
                    boundary_grid_point_count,
                    " is larger than ibmx= ",
                    max_boundary_grid_points,
                    " thus stopped...",
                )
                return EXIT_NJIT_FLAG, boundary_grid_points

        for neighbor_index in range(1, 7):
            neighbor_grid_index_1d = (
                (grid_x_index + index_map[1][neighbor_index]) * x_stride
                + (grid_y_index + index_map[2][neighbor_index]) * y_stride
                + (grid_z_index + index_map[3][neighbor_index])
            )
            neighbor_grid_index_1d_x_3 = neighbor_grid_index_1d * 3
            epsilon_index_value = index_discrete_epsilon_map_1d[
                neighbor_grid_index_1d_x_3 + index_map[4][neighbor_index] - 1
            ]
            epsilon_value = epsilon_index_value % epsilon_dimension

            if epsilon_value < 0:
                epsilon_index_value = -epsilon_index_value
                if index_map[4][neighbor_index] == 1:
                    index_discrete_epsilon_map_1d[neighbor_grid_index_1d_x_3] = (
                        epsilon_index_value
                    )
                elif index_map[4][neighbor_index] == 2:
                    index_discrete_epsilon_map_1d[neighbor_grid_index_1d_x_3 + 1] = (
                        epsilon_index_value
                    )
                elif index_map[4][neighbor_index] == 3:
                    index_discrete_epsilon_map_1d[neighbor_grid_index_1d_x_3 + 2] = (
                        epsilon_index_value
                    )

                if epsilon_value == -1:
                    epsilon_index_value = epsilon_index_value - 1
                    if index_map[4][neighbor_index] == 1:
                        index_discrete_epsilon_map_1d[neighbor_grid_index_1d_x_3] = (
                            epsilon_index_value
                        )
                    elif index_map[4][neighbor_index] == 2:
                        index_discrete_epsilon_map_1d[
                            neighbor_grid_index_1d_x_3 + 1
                        ] = epsilon_index_value
                    elif index_map[4][neighbor_index] == 3:
                        index_discrete_epsilon_map_1d[
                            neighbor_grid_index_1d_x_3 + 2
                        ] = epsilon_index_value

    if boundary_grid_point_count > max_boundary_grid_points:
        nprint_cpu(DEBUG, _VERBOSITY, " WARNING !!! Number of  MS points exceeds ibmx")
        return EXIT_NJIT_FLAG, boundary_grid_points

    num_boundary_grid_points = boundary_grid_point_count

    nprint_cpu(
        DEBUG,
        _VERBOSITY,
        " VdMS> After surface elaboration iBoundNum= ",
        num_boundary_grid_points,
    )
    return num_boundary_grid_points, boundary_grid_points


def print_4d_array(data_name, data_array, array_shape):
    print(f"list of {len(data_array)} {data_name}: ", end="")
    print(
        "[",
    )
    ic = 0
    y_stride_x_3 = array_shape[2] * 3
    x_stride_x_3 = array_shape[1] * array_shape[2] * 3
    for ijk1d_x_3 in range(0, len(data_array), 3):
        iix = ijk1d_x_3 // x_stride_x_3
        iiy = (ijk1d_x_3 - iix * x_stride_x_3) // y_stride_x_3
        iiz = ((ijk1d_x_3 - iix * x_stride_x_3) - iiy * y_stride_x_3) // 3
        if data_array[ijk1d_x_3 + 1] != 0:
            print(
                f"({ijk1d_x_3}, {iix}, {iiy}, {iiz}, {data_array[ijk1d_x_3 + 1]}, {data_array[ijk1d_x_3 + 2]}), ",
                end="",
            )
            if ic % 4 == 0:
                print()
            ic += 1
    print("]\n\n")
