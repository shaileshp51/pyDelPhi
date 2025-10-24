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
Module for creating and populating a 3D cubic grid from atomic data.

This module provides high-performance, Numba-accelerated functions for spatial
partitioning of atomic coordinates into a voxel-based grid. This is a common
pre-processing step in computational biophysics and chemistry for tasks such
as calculating electrostatic potentials, identifying neighboring atoms, or
performing other grid-based analyses.

Key functionalities include:
- `setup_cube`: Determines the optimal dimensions and origin of the cubic grid
  based on the spatial extent of the atomic coordinates.
- `cube`: Maps each atom to its corresponding voxel and the 26 neighboring
  voxels, creating an efficient data structure for spatial queries.

The functions are designed to be used within the PyDelphi framework and rely on
its specific data types and constants.
"""

import numpy as np
from numba import njit, prange

from pydelphi.config.global_runtime import (
    delphi_int,
    delphi_real,
)

from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_CRD_END,
    ATOMFIELD_RADIUS,
)
from pydelphi.constants import (
    NEIGHBOR_VOXEL_RELATIVE_COORDINATES as NEIGHBOR_VOXEL_REL_COORDS,
)


@njit(nogil=True, boundscheck=False, cache=True)
def setup_cube(
    cube_side_length,
    coords_by_axis_min,
    coords_by_axis_max,
    scaling_factor=1.0,
    cube_extrema_offset=0.1,
):
    """
    Determines origin and shape of a cubic grid based on given parameters.

    This function calculates the dimensions and origin of a 3D cubic grid (voxel grid)
    based on the provided minimum and maximum coordinates, scaling factor, and offset.
    The grid is used to represent spatial data, such as atomic coordinates.

    Args:
        cube_side_length (float): Length of a side of each voxel in the grid.
        coords_by_axis_min (array-like): Minimum coordinate values for each axis (x, y, z).
        coords_by_axis_max (array-like): Maximum coordinate values for each axis (x, y, z).
        scaling_factor (float, optional): Factor to scale the grid padding. Defaults to 1.0.
        cube_extrema_offset (float, optional): Offset to extend the grid boundaries. Defaults to 0.1.

    Returns:
        tuple: A tuple containing:
            - cube_vertex_lowest_xyz (np.ndarray): Minimum coordinate values of the grid.
            - cube_shape (np.ndarray): Shape of the grid (number of voxels along each axis).
    """
    cube_padding = delphi_real(scaling_factor * cube_side_length + cube_extrema_offset)

    cube_vertex_lowest_xyz = np.empty(len(coords_by_axis_min), dtype=delphi_real)
    cube_vertex_lowest_xyz[:] = coords_by_axis_min - cube_padding

    cube_vertex_highest_xyz = np.empty(len(coords_by_axis_max), dtype=delphi_real)
    cube_vertex_highest_xyz[:] = coords_by_axis_max + cube_padding

    cube_shape = np.empty(len(coords_by_axis_max), dtype=delphi_int)
    cube_shape[:] = np.floor(
        (cube_vertex_highest_xyz - cube_vertex_lowest_xyz) / cube_side_length
    ).astype(delphi_int)[:]

    return cube_vertex_lowest_xyz, cube_shape


@njit
def str_to_int(substr):
    """Manually convert a numeric string to an integer (JIT-safe)."""
    result = 0
    for char in substr:
        if 48 <= ord(char) <= 57:  # ASCII range for '0'-'9'
            result = result * 10 + (ord(char) - 48)
    return result


@njit(nogil=True, boundscheck=False, cache=True)
def cube(
    cube_side_length,
    num_atoms,
    num_objects,
    num_molecules,
    cube_vertex_lowest_xyz,
    cube_shape,
    atoms_data,
    voxel_atom_ids,
):
    """
    Constructs a voxel grid representation of atomic and molecular data.

    This function populates a 3D voxel grid with atom IDs based on their
    spatial coordinates. It efficiently maps atoms to their respective
    voxels and neighboring voxels, facilitating spatial queries.

    Args:
        cube_side_length (float): Side length of a voxel in the grid.
        num_atoms (int): Number of atoms in the system.
        num_objects (int): Number of objects in the system.
        num_molecules (int): Number of molecules in the system.
        cube_vertex_lowest_xyz (np.ndarray): Minimum coordinate values of the grid.
        cube_shape (np.ndarray): Shape of the grid (number of voxels along each axis).
        atoms_data (np.ndarray): Array of atomic data (coordinates and radii).
        voxel_atom_ids (np.ndarray): Output array for voxel atom mapping.

    Returns:
        tuple: A tuple containing:
            - voxel_atom_ids (np.ndarray): Populated array with atom IDs mapped to voxels.
            - voxel_atom_count (np.ndarray): Array indicating the starting index of atoms in each voxel.
            - voxel_atom_count_cumulative (np.ndarray): Array indicating the ending index of atoms in each voxel.
    """
    voxel_atom_count = np.ones(
        (cube_shape[0] + 1, cube_shape[1] + 1, cube_shape[2] + 1),
        dtype=delphi_int,
    )
    voxel_atom_count_cumulative = np.zeros(
        (cube_shape[0] + 1, cube_shape[1] + 1, cube_shape[2] + 1),
        dtype=delphi_int,
    )
    atom_voxel_coords = np.zeros(
        (
            num_atoms
            + (num_objects - num_molecules)
            * (cube_shape[0] + 1)
            * (cube_shape[1] + 1)
            * (cube_shape[2] + 1)
            + 1,
            3,
        ),
        dtype=delphi_int,
    )

    cube_side_length_inverse = 1.0 / cube_side_length

    # First pass: Calculate voxel coordinates and cumulative counts (parallelized)
    for atom_index in range(atoms_data.shape[0]):
        this_atom_data = atoms_data[atom_index]
        if this_atom_data[ATOMFIELD_RADIUS] > 0.0:
            atom_coords = this_atom_data[ATOMFIELD_X:ATOMFIELD_CRD_END]

            ix = delphi_int(
                (atom_coords[0] - cube_vertex_lowest_xyz[0]) * cube_side_length_inverse
            )
            iy = delphi_int(
                (atom_coords[1] - cube_vertex_lowest_xyz[1]) * cube_side_length_inverse
            )
            iz = delphi_int(
                (atom_coords[2] - cube_vertex_lowest_xyz[2]) * cube_side_length_inverse
            )

            # Initialize the voxels atom counts with cumulative atom occupancy traversing voxels in
            # z-,y-,x- (-ve to +ve) order (voxel coords are relative to central voxel)
            for jz in range(iz - 1, iz + 2):
                for jy in range(iy - 1, iy + 2):
                    for jx in range(ix - 1, ix + 2):
                        voxel_atom_count_cumulative[jx][jy][jz] += 1

            atom_voxel_coords[atom_index][:] = ix, iy, iz

    # Calculate starting indices for each voxel
    temp_vox_atmcnt_cumulat = 1
    for iz in range(cube_shape[2] + 1):
        for iy in range(cube_shape[1] + 1):
            for ix in range(cube_shape[0] + 1):
                if voxel_atom_count_cumulative[ix][iy][iz] > 0:
                    voxel_atom_count[ix][iy][iz] = temp_vox_atmcnt_cumulat
                    temp_vox_atmcnt_cumulat += voxel_atom_count_cumulative[ix][iy][iz]

    # Second pass: Populate voxel_atom_ids with atomic data
    previous_voxel = np.array((0, 0, 0), dtype=np.int32)
    for voxel_coord in NEIGHBOR_VOXEL_REL_COORDS:
        voxel_coord_relative = voxel_coord - previous_voxel

        for atom_index in prange(atoms_data.shape[0]):
            this_atom_data = atoms_data[atom_index]
            if this_atom_data[ATOMFIELD_RADIUS] > 0.0:
                ix, iy, iz = (
                    atom_voxel_coords[atom_index][0] + voxel_coord_relative[0],
                    atom_voxel_coords[atom_index][1] + voxel_coord_relative[1],
                    atom_voxel_coords[atom_index][2] + voxel_coord_relative[2],
                )
                voxel_atom_ids[voxel_atom_count[ix][iy][iz]] = atom_index + 1
                voxel_atom_count[ix][iy][iz] += 1
                atom_voxel_coords[atom_index][0] = ix
                atom_voxel_coords[atom_index][1] = iy
                atom_voxel_coords[atom_index][2] = iz
        previous_voxel[:] = voxel_coord

    # Reset voxel_atom_count and calculate cumulative counts
    temp_vox_atmcnt_cumulat = 1
    for iz in range(cube_shape[2] + 1):
        for iy in range(cube_shape[1] + 1):
            for ix in range(cube_shape[0] + 1):
                if voxel_atom_count_cumulative[ix][iy][iz] > 0:
                    voxel_atom_count[ix][iy][iz] = temp_vox_atmcnt_cumulat
                    temp_vox_atmcnt_cumulat += voxel_atom_count_cumulative[ix][iy][iz]
                    voxel_atom_count_cumulative[ix][iy][iz] = (
                        temp_vox_atmcnt_cumulat - 1
                    )

    return voxel_atom_ids, voxel_atom_count, voxel_atom_count_cumulative


def print_3d_array(data_name, data_array, array_shape):
    print(f"list of {len(data_array)} {data_name}: ", end="")
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
                if (ic + 1) % 6 == 0:
                    print("")
                ic += 1
    print("]\n\n")
