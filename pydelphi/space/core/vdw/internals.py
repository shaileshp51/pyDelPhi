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


import numpy as np
from numba import njit

from pydelphi.foundation.enums import (
    Precision,
)

from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
    vprint,
    nprint_cpu_if_verbose as nprint_cpu,
)

from pydelphi.config.logging_config import (
    CRITICAL,
    ERROR,
    INFO,
    DEBUG,
    TRACE,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

from pydelphi.constants import (
    ConstDelPhiInts,
    ATOMFIELD_X,
    ATOMFIELD_CRD_END,
    ATOMFIELD_RADIUS,
    ATOMFIELD_MEDIA_ID,
)
from pydelphi.constants import ConstDelPhiFloats as ConstDelPhi

# Initialize module level constants based on global constants.
APPROX_ZERO = ConstDelPhi.ApproxZero.value
RADII_SQUARED_SHRINK_FACTOR = ConstDelPhi.SASSquaredRadiiShrinkFactor.value
MAX_NEIGHBR_BOUNDARY_ARRAY_LEN = ConstDelPhiInts.SpaceNBRASize.value
RESIZE_FACTOR = ConstDelPhi.ZetaArrayResizeFactor.value
INITIAL_SIZE_PERCENT = ConstDelPhi.ZetaArrayInitialSizePercent.value

EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value


@njit(nogil=True, boundscheck=False, cache=True)
def _calculate_solute_grid_boundaries(
    max_atom_radius,
    grid_spacing,
    grid_shape,
    grid_origin,
    coords_by_axis_min,
    coords_by_axis_max,
    dtype_int,
):
    """
    Calculates the minimum and maximum grid indices that encompass the solute
    (the molecule or set of atoms of interest), considering the maximum atom radius.

    Args:
        max_atom_radius (float): The maximum radius of any atom in the system.
        grid_spacing (float): The spacing between grid points.
        grid_shape (np.ndarray): The dimensions of the grid (nx, ny, nz).
        grid_origin (np.ndarray): The coordinates of the grid origin [x, y, z].
        coords_by_axis_min (np.ndarray): Minimum coordinates of the system along each axis.
        coords_by_axis_max (np.ndarray): Maximum coordinates of the system along each axis.
        dtype_int (type): Integer data type. Defaults to np.int32.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the minimum and maximum
                                         solute grid indices as NumPy arrays [x, y, z].
    """
    min_solute_indices = np.array(
        [
            dtype_int((c - o - max_atom_radius) / grid_spacing) - 2
            for c, o in zip(coords_by_axis_min, grid_origin)
        ],
        dtype=dtype_int,
    )
    max_solute_indices = np.array(
        [
            dtype_int((c - o + max_atom_radius) / grid_spacing) + 2
            for c, o in zip(coords_by_axis_max, grid_origin)
        ],
        dtype=dtype_int,
    )
    for axis_index in range(3):
        if min_solute_indices[axis_index] < 0:
            min_solute_indices[axis_index] = 0
        if max_solute_indices[axis_index] >= grid_shape[axis_index]:
            max_solute_indices[axis_index] = grid_shape[axis_index] - 1
    return min_solute_indices, max_solute_indices


@njit(nogil=True, boundscheck=False, cache=True)
def _setup_grid_neighbor_coords_offsets(grid_spacing_half, dtype_real):
    """
    Sets up the coordinate offsets for the six neighboring midpoints of a grid point.
    These offsets are used to quickly access the coordinates of the neighbors.

    Args:
        grid_spacing_half (float): Half the spacing between grid points.
        dtype_real (type): Real data type. Defaults to np.float64.

    Returns:
        np.ndarray: A NumPy array of shape (7, 3) where each row represents the
                    coordinate offset for a neighbor (index 0 is unused).
    """
    neighbor_coords_offsets = np.zeros((7, 3), dtype=dtype_real)
    neighbor_coords_offsets[1][0] = grid_spacing_half
    neighbor_coords_offsets[2][1] = grid_spacing_half
    neighbor_coords_offsets[3][2] = grid_spacing_half
    neighbor_coords_offsets[4][0] = -grid_spacing_half
    neighbor_coords_offsets[5][1] = -grid_spacing_half
    neighbor_coords_offsets[6][2] = -grid_spacing_half
    return neighbor_coords_offsets


@njit(nogil=True, boundscheck=False, cache=True)
def _calculate_grid_properties(
    grid_spacing,
    grid_shape,
    grid_origin,
    dtype_int,
    dtype_real,
) -> tuple[float, np.ndarray, np.ndarray, int, int]:
    """
    Calculates various properties of the computational grid.

    Args:
        grid_spacing (float): The spacing between grid points.
        grid_shape (np.ndarray): The dimensions of the grid (nx, ny, nz).
        grid_origin (np.ndarray): The coordinates of the grid origin [x, y, z].
        dtype_int (type): Integer data type. Defaults to np.int32.
        dtype_real (type): Real data type. Defaults to np.float64.

    Returns:
        tuple[float, np.ndarray, np.ndarray, int, int]: A tuple containing:
            - grid_spacing_half (float): Half the grid spacing.
            - mid_grid_point_indices (np.ndarray): Indices of the middle grid point.
            - gridbox_center (np.ndarray): Coordinates of the center of the grid box.
            - n_grid_points (int): Total number of grid points.
            - n_grid_points_x_3 (int): Total number of grid points multiplied by 3.
    """
    grid_spacing_half = 0.5 * grid_spacing
    mid_grid_indices = np.array([nx // 2 for nx in grid_shape], dtype=dtype_int)
    gridbox_center = np.array(
        [o + i_mid * grid_spacing for o, i_mid in zip(grid_origin, mid_grid_indices)],
        dtype=dtype_real,
    )
    total_grid_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
    total_grid_points_x_3 = total_grid_points * 3
    return (
        grid_spacing_half,
        mid_grid_indices,
        gridbox_center,
        total_grid_points,
        total_grid_points_x_3,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def _set_constant_values(dtype_int, dtype_bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Sets up constant values for index mapping and neighbor existence.

    Args:
        dtype_int (type): Integer data type. Defaults to np.int32.
        dtype_bool (type): Boolean data type. Defaults to np.bool_.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - index_map (np.ndarray): A 5x7 array for mapping indices.
            - neighbor_exists_array (np.ndarray): A boolean array indicating neighbor existence.
    """
    index_map = np.zeros((5, 7), dtype=dtype_int)
    neighbor_exists = np.zeros(7, dtype=dtype_bool)

    index_map[1][4] = -1
    index_map[2][5] = -1
    index_map[3][6] = -1
    index_map[4][1:4] = 1, 2, 3
    index_map[4][4:7] = 1, 2, 3

    neighbor_exists[1:6] = True

    return index_map, neighbor_exists


@njit(nogil=True, boundscheck=False, cache=True)
def _calculate_strides(grid_shape, dtype_int):
    """
    Calculates the strides needed to convert 3D grid indices to a 1D index.

    Args:
        grid_shape (np.ndarray): The dimensions of the grid (nx, ny, nz).
        dtype_int (type): Integer data type. Defaults to np.int32.

    Returns:
        tuple[int, int, int, int, int, int]: A tuple containing the strides:
            - x_stride
            - y_stride
            - z_stride
            - x_stride_x_3 (x_stride * 3)
            - y_stride_x_3 (y_stride * 3)
            - z_stride_x_3 (z_stride * 3)
    """
    x_stride = dtype_int(grid_shape[1] * grid_shape[2])
    y_stride = dtype_int(grid_shape[2])
    z_stride = dtype_int(1)
    x_stride_x_3 = dtype_int(x_stride * 3)
    y_stride_x_3 = dtype_int(y_stride * 3)
    z_stride_x_3 = dtype_int(z_stride * 3)
    return (
        x_stride,
        y_stride,
        z_stride,
        x_stride_x_3,
        y_stride_x_3,
        z_stride_x_3,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def _handle_zero_probe_radius(
    max_probe_radius: float,
    num_boundary_grid_points: int,
    boundary_grid_indices: np.ndarray,
    dtype_int,
    dtype_real,
) -> np.ndarray:
    """
    Handles the special case where the probe radius is zero. In this case,
    the boundary points are directly used as the molecular surface points.

    Args:
        max_probe_radius (float): The maximum probe radius.
        num_boundary_grid_points (int): The number of boundary grid points.
        boundary_grid_indices (np.ndarray): The 3D indices of the boundary grid points.
        dtype_int (type): Integer data type. Defaults to np.int32.
        dtype_real (type): Real data type. Defaults to np.float64.

    Returns:
        np.ndarray: A NumPy array containing the coordinates of the boundary points.
                    Returns an empty array if the probe radius is not zero.
    """
    if max_probe_radius < dtype_real(APPROX_ZERO):
        boundary_grid_points = np.zeros(
            (num_boundary_grid_points + 1, 3), dtype=dtype_int
        )
        for i in range(num_boundary_grid_points):
            boundary_grid_points[i] = boundary_grid_indices[
                i
            ]  # Adjust index for 0-based array
        return boundary_grid_points
    return np.zeros((0, 3), dtype=dtype_int)


@njit(nogil=True, boundscheck=False, cache=True)
def _calculate_atom_probe_radii(
    probe_radius: float,
    shrink_factor: float,
    num_atoms: int,
    atoms_data: np.ndarray,
    dtype_real: type,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculates the sum of atom radius and probe radius, as well as their squares
    (including a shrunken version for contact detection).

    Args:
        probe_radius (float): The radius of the probe molecule.
        shrink_factor (float): Factor to shrink squared radii.
        num_atoms (int): The number of atoms in the system.
        atoms_data (np.ndarray): Array containing atomic data (including radii).
        dtype_real (type): Real data type. Defaults to np.float64.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - atom_plus_probe_radii_1d (np.ndarray): 1D array of (atom_radius + probe_radius).
            - atom_plus_probe_radii_square_1d (np.ndarray): 1D array of (atom_radius + probe_radius)^2.
            - atom_plus_probe_radii_square_shrunk_1d (np.ndarray): 1D array of shrunken squared radii.
    """
    atom_plus_probe_radii_1d = np.zeros(num_atoms, dtype=dtype_real)
    atom_plus_probe_radii_square_1d = np.zeros(num_atoms, dtype=dtype_real)
    atom_plus_probe_radii_shrink_1d = np.zeros(num_atoms, dtype=dtype_real)
    for i in range(num_atoms):
        atom_plus_probe_radius = dtype_real(
            atoms_data[i][ATOMFIELD_RADIUS] + probe_radius
        )
        atom_plus_probe_radii_1d[i] = atom_plus_probe_radius
        atom_plus_probe_radii_square_1d[i] = dtype_real(
            atom_plus_probe_radius * atom_plus_probe_radius
        )
        atom_plus_probe_radii_shrink_1d[i] = dtype_real(
            atom_plus_probe_radii_square_1d[i] * shrink_factor
        )
    return (
        atom_plus_probe_radii_1d,
        atom_plus_probe_radii_square_1d,
        atom_plus_probe_radii_shrink_1d,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def _calculate_rms(
    only_molecule: bool,
    num_atoms: int,
    atoms_data: np.ndarray,
    dtype_real: type,
):
    """
    Calculates the root mean square (RMS) of the atomic positions. This might be
    used for centering or other geometric calculations.

    Args:
        only_molecule (bool): Flag indicating if only the molecule is considered.
        num_atoms (int): The number of atoms in the system.
        atoms_data (np.ndarray): Array containing atomic data (including coordinates).
        dtype_real (type): Real data type. Defaults to np.float64.

    Returns:
        float: The root-mean-square of the atomic positions. Returns 0.0 if no atoms.
    """
    rms = dtype_real(0.0)
    if only_molecule:
        for i in range(num_atoms):
            atom_radius = atoms_data[i][ATOMFIELD_RADIUS]
            atom_coords = atoms_data[i][ATOMFIELD_X:ATOMFIELD_CRD_END]
            rms = rms + atom_radius * np.sum(np.abs(atom_coords))
    return rms


@njit(nogil=True, boundscheck=False, cache=True)
def _calculate_cube_voxels_per_entity(
    num_objects: int,
    num_molecules: float,
    cube_shape: np.ndarray,
    dtype_int,
) -> int:
    """
    Calculates the number of voxels per entity (object or molecule) in the indexing cube.

    Args:
        num_objects (int): The number of geometric objects.
        num_molecules (int): The number of molecules.
        cube_shape (np.ndarray): The dimensions of the indexing cube.
        dtype_int (type): Integer data type. Defaults to np.int32.

    Returns:
        int: The number of voxels per entity.
    """
    total_cube_vertices = dtype_int(
        (cube_shape[0] + 1) * (cube_shape[1] + 1) * (cube_shape[2] + 1)
    )
    voxels_per_entity = dtype_int(27)
    if (num_objects - num_molecules) > 0:
        voxels_per_entity = dtype_int(max(total_cube_vertices, voxels_per_entity))
    return voxels_per_entity
