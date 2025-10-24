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
Indexing Grid Utilities for Exposed Grid Points in Delphi

This module provides Numba-accelerated utilities for constructing and indexing
a 3D spatial grid over exposed dielectric surface grid points. It is designed to
support efficient neighborhood queries during molecular electrostatics calculations,
particularly in Gaussian or surface-based solvers.

Key Functions:
--------------
- setup_index_vertices:
    Computes the bounding box, cell size, and grid shape for the indexing volume
    that will contain all relevant exposed grid points, expanded based on probe
    radius and maximum atom radius.

- index_vertices:
    Maps individual exposed grid points into discrete voxel cells of the index grid.
    This mapping allows fast spatial lookup of grid points for each voxel, enabling
    localized neighbor operations.

- print_3d_array:
    Utility function to print out 3D index arrays for debugging and inspection.

Dependencies:
-------------
- Numba: for JIT acceleration of computational functions.
- NumPy: for array operations.
- PyDelphi's internal type alias: `delphi_int` for consistent integer width across platforms.
- PyDelphi's verbosity system: `VerbosityLevel` for potential logging extensions.

Usage Notes:
------------
- These functions are performance-critical and expected to be used inside higher-level
  solvers that perform Gaussian density computations, dielectric boundary processing,
  or electrostatic focusing.
- Arrays passed to these functions must be correctly preallocated to avoid runtime issues.

Author: Panday, Shailesh Kumar
"""

import numpy as np
from numba import njit

from pydelphi.config.global_runtime import (
    delphi_int,
)

from pydelphi.config.logging_config import (
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)


@njit(nogil=True, boundscheck=False, cache=True)
def setup_index_vertices(
    grid_spacing,
    probe_radius,
    max_atom_radius,
    coords_by_axis_min,
    coords_by_axis_max,
    id_max=50,
):
    """
    Calculates the dimensions and starting coordinates for the indexing grid.

    This function determines the size and origin of a 3D grid used to index
    exposed grid points. It ensures the grid encompasses all relevant atoms
    and probe radii, and that the grid cell size is appropriate for indexing.

    Args:
        grid_spacing (float): Spacing between grid points.
        probe_radius (float): Radius of the probe.
        max_atom_radius (float): Maximum radius of any atom.
        coords_by_axis_min (np.ndarray): Minimum coordinates of atoms along each axis (x, y, z).
        coords_by_axis_max (np.ndarray): Maximum coordinates of atoms along each axis (x, y, z).
        id_max (int): Maximum allowed index value for grid cells, to keep vertex index grid box of
                    reasonable size considering the memory-footprint and neighbor search processing.

    Returns:
        tuple: A tuple containing:
            - cube_side_index_vertex (float): Side length of the indexing grid cells.
            - min_xyz (np.ndarray): Minimum coordinates of the indexing grid.
            - index_vertex_cube_shape (np.ndarray): Shape of the indexing grid (number of cells along each axis).
    """
    cube_side_index_vertex = probe_radius
    grid_spacing_x_2 = 2.0 * grid_spacing

    min_xyz = coords_by_axis_min - (max_atom_radius + probe_radius)
    max_xyz = coords_by_axis_max + (max_atom_radius + probe_radius)

    min_xyz -= grid_spacing_x_2
    max_xyz += grid_spacing_x_2

    do_iterate = True
    while do_iterate:
        max_xyz += cube_side_index_vertex - probe_radius
        min_xyz -= cube_side_index_vertex - probe_radius
        index_vertex_cube_shape = (
            ((max_xyz - min_xyz) / cube_side_index_vertex) + 1
        ).astype(delphi_int)

        if np.any(index_vertex_cube_shape > id_max):
            idtemp = np.max(index_vertex_cube_shape)
            cube_side_index_vertex = cube_side_index_vertex * (idtemp + 1) / id_max
            print(
                """initial cube size too small in assigning accessible points to a grid
                therefore rescaling..."""
            )
        else:
            do_iterate = False

    return cube_side_index_vertex, min_xyz, index_vertex_cube_shape


@njit(nogil=True, boundscheck=False, cache=True)
def index_vertices(
    num_exposed_grids,
    cube_side_index_vertex_inverse,
    index_vertex_cube_shape,
    min_xyz,
    exposed_grids_coords,
    grid_point_indices_in_voxel,
):
    """
    Indexes exposed grid points into a 3D grid for efficient nearest neighbor searches.

    This function assigns each exposed grid point to a specific cell within a 3D
    grid. It creates lookup tables to quickly find all grid points within a given
    cell, facilitating nearest neighbor calculations.

    Args:
        num_exposed_grids (int): Number of exposed grid points.
        cube_side_index_vertex_inverse (float): Inverse of the indexing grid cell side length.
        index_vertex_cube_shape (np.ndarray): Shape of the indexing grid (number of cells along each axis).
        min_xyz (np.ndarray): Minimum coordinates of the indexing grid.
        exposed_grids_coords (np.ndarray): Coordinates of the exposed grid points.
        grid_point_indices_in_voxel (np.ndarray): Array to store indices of grid points within each box.

    Returns:
        tuple: A tuple containing:
            - cube_voxel_start_indices (np.ndarray): Array indicating the starting index of grid points within each cube voxel.
            - cube_voxel_end_indices (np.ndarray): Array indicating the ending index of grid points within each cube voxel.
            - grid_point_indices_in_voxel (np.ndarray): 1D Array containing indices of grid points, organized by cube voxel.
    """
    cube_voxel_start_indices = np.ones(
        (
            index_vertex_cube_shape[0] + 1,
            index_vertex_cube_shape[1] + 1,
            index_vertex_cube_shape[2] + 1,
        ),
        dtype=delphi_int,
    )
    cube_voxel_end_indices = np.zeros(
        (
            index_vertex_cube_shape[0] + 1,
            index_vertex_cube_shape[1] + 1,
            index_vertex_cube_shape[2] + 1,
        ),
        dtype=delphi_int,
    )

    exposed_grid_points = np.zeros((num_exposed_grids + 1, 3), dtype=delphi_int)

    # Convert exposed grid coordinates to grid cell indices and count points per cell.
    for i in range(1, num_exposed_grids + 1):
        exposed_grid_points[i][:] = (
            (exposed_grids_coords[i] - min_xyz) * cube_side_index_vertex_inverse
        ).astype(delphi_int)
        cube_voxel_end_indices[
            exposed_grid_points[i][0],
            exposed_grid_points[i][1],
            exposed_grid_points[i][2],
        ] += 1

    # Calculate starting and ending indices for grid points within each cell.
    n = 0
    for i in range(index_vertex_cube_shape[0] + 1):
        for j in range(index_vertex_cube_shape[1] + 1):
            for k in range(index_vertex_cube_shape[2] + 1):
                if cube_voxel_end_indices[i][j][k] != 0:
                    cube_voxel_start_indices[i][j][k] = n + 1
                    n += cube_voxel_end_indices[i][j][k]
                    cube_voxel_end_indices[i][j][k] = n

    # Populate grid_point_indices_in_voxel with the actual grid point indices.
    for i in range(1, num_exposed_grids + 1):
        ix, iy, iz = (
            exposed_grid_points[i][0],
            exposed_grid_points[i][1],
            exposed_grid_points[i][2],
        )
        j = cube_voxel_start_indices[ix][iy][iz]
        grid_point_indices_in_voxel[j] = i
        cube_voxel_start_indices[ix][iy][iz] += 1

    # Reset the cube_voxel_start_indices for use in nested loops.
    for i in range(1, num_exposed_grids + 1):
        ix, iy, iz = (
            exposed_grid_points[i][0],
            exposed_grid_points[i][1],
            exposed_grid_points[i][2],
        )
        cube_voxel_start_indices[ix][iy][iz] -= 1

    # Clear temporary array.
    exposed_grid_points = None
    return cube_voxel_start_indices, cube_voxel_end_indices, grid_point_indices_in_voxel


def print_3d_array(data_name, data_array, array_shape, n_per_line=8):
    print(f"{data_name} a list of {len(data_array)}: ", end="")
    print(
        "[",
    )
    ic = 0
    for i3 in range(array_shape[2] + 1):
        for i2 in range(array_shape[1] + 1):
            for i1 in range(array_shape[0] + 1):
                print(
                    f"({i1}, {i2}, {i3}, {data_array[i1][i2][i3]}), ",
                    end="",
                )
                if (ic + 1) % n_per_line == 0:
                    print("")
                ic += 1
    print("]\n\n")
