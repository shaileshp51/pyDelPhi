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
This module provides utility functions and a class (`SurfaceMolecularVdW`)
for setting up and processing the computational grid used in DelPhi calculations,
specifically for generating the Van der Waals molecular surface.

The module includes functionalities for:

- Defining grid properties and boundaries.
- Identifying boundary grid points based on dielectric constant changes.
- Handling the molecular surface using probe radius.
- Setting up an indexing cube for efficient neighbor searching.
- Remapping the dielectric constant map based on molecular surface information.
- Updating the status of neighboring grid points.
- Handling the zeta surface for advanced calculations.

The `SurfaceMolecularVdW` class orchestrates these functionalities to create the
molecular surface representation.

These functions and the class methods are often decorated with `@njit`
for performance optimization using Numba. The module also includes
precision-dependent imports to handle single, double, and mixed-precision
calculations.
"""

import time
import numpy as np

from math import sqrt
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

# # --- Dynamic Precision Handling ---
if PRECISION.int_value in {Precision.SINGLE.int_value}:
    import pydelphi.utils.prec.single as size_cpu

    try:
        import pydelphi.utils.cuda.single as size_gpu
    except ImportError:
        size_gpu = None
elif PRECISION.int_value == Precision.DOUBLE.int_value:
    import pydelphi.utils.prec.double as size_cpu

    try:
        import pydelphi.utils.cuda.double as size_gpu
    except ImportError:
        size_gpu = None
else:
    raise ValueError(f"Unsupported PRECISION: {PRECISION}")

import pydelphi.utils.io.writers as wrt
import pydelphi.space.core.vdw.sas as sas
import pydelphi.space.core.vdw.helper as helpers
import pydelphi.space.core.vdw.scale_boundary as sclbp
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


import pydelphi.space.core.voxelizer as voxelizer


# Deferred imports for precision-sensitive components
def configure_precision_dependent_imports():
    # Global identifiers (variables, imported-module-aliases and data_types) declaration
    global PRECISION, delphi_bool, delphi_int, delphi_real
    global size_cpu, size_gpu, wrt
    global sas, helpers, sclbp

    from pydelphi.config.global_runtime import (
        PRECISION,
        delphi_bool,
        delphi_int,
        delphi_real,
    )

    if PRECISION.int_value == Precision.SINGLE.int_value:
        import pydelphi.utils.prec.single as size_cpu

        try:
            import pydelphi.utils.cuda.single as size_gpu
        except ImportError:
            pass
            # print("No Cuda")

    elif PRECISION.int_value == Precision.DOUBLE.int_value:
        import pydelphi.utils.prec.double as size_cpu

        try:
            import pydelphi.utils.cuda.double as size_gpu
        except ImportError:
            pass
            # print("No Cuda")
    import pydelphi.utils.io.writers as wrt
    import pydelphi.space.core.vdw.sas as sas
    import pydelphi.space.core.vdw.helper as helpers
    import pydelphi.space.core.vdw.scale_boundary as sclbp


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
def _find_boundary_grid_points(
    epsilon_dimension,
    max_boundary_grid_points,
    grid_shape,
    min_solute_grid_index,
    max_solute_grid_index,
    index_discrete_epsilon_map_1d,
    dielectric_boundary_midpoints_1d,
    boundary_grid_indices,
    dtype_int,
):
    """
    Identifies the grid points that lie on the boundary between regions of different
    dielectric constants.

    Args:
        epsilon_dimension (int): The number of discrete dielectric constants.
        max_boundary_grid_points (int): The maximum number of boundary grid points to store.
        grid_shape (np.ndarray): The dimensions of the grid (nx, ny, nz).
        min_solute_grid_index (np.ndarray): Minimum grid indices of the solute.
        max_solute_grid_index (np.ndarray): Maximum grid indices of the solute.
        index_discrete_epsilon_map_1d (np.ndarray): 1D array of discrete dielectric indices.
        dielectric_boundary_midpoints_1d (np.ndarray): 1D array to store boundary point information.
        boundary_grid_indices (np.ndarray): Array to store the 3D indices of boundary points.
        dtype_int (type): Integer data type. Defaults to np.int32.

    Returns:
        tuple[int, int, np.ndarray, np.ndarray]: A tuple containing:
            - n_boundary_grid_points (int): The total number of boundary grid points found.
            - n_external_boundary_point (int): The number of boundary points on the exterior.
            - dielectric_boundary_midpoints_1d (np.ndarray): Updated boundary information array.
            - boundary_grid_indices (np.ndarray): Updated array of boundary point indices.
    """
    n_boundary_grid_points = 0
    n_external_boundary_point = 0
    grid_neighbors_media_ids = np.zeros(7, dtype=dtype_int)

    (
        x_stride,
        y_stride,
        z_stride,
        x_stride_x_3,
        y_stride_x_3,
        z_stride_x_3,
    ) = _calculate_strides(grid_shape, dtype_int=dtype_int)

    # Iterate through the grid points within the solute's bounding box
    for k in range(min_solute_grid_index[2], max_solute_grid_index[2] + 1):
        for j in range(min_solute_grid_index[1], max_solute_grid_index[1] + 1):
            for i in range(min_solute_grid_index[0], max_solute_grid_index[0] + 1):
                is_external_grid_point = False
                is_boundary_grid_point = False
                ijk1d = i * x_stride + j * y_stride + k
                ijk1d_x_3 = ijk1d * 3

                # Get the media IDs of the six neighboring midpoints
                grid_neighbors_media_ids[1] = (
                    abs(index_discrete_epsilon_map_1d[ijk1d_x_3]) // epsilon_dimension
                )  # +h/2 in x
                grid_neighbors_media_ids[2] = (
                    abs(index_discrete_epsilon_map_1d[ijk1d_x_3 + 1])
                    // epsilon_dimension
                )  # +h/2 in y
                grid_neighbors_media_ids[3] = (
                    abs(index_discrete_epsilon_map_1d[ijk1d_x_3 + 2])
                    // epsilon_dimension
                )  # +h/2 in z
                grid_neighbors_media_ids[4] = (
                    abs(index_discrete_epsilon_map_1d[ijk1d_x_3 - x_stride_x_3])
                    // epsilon_dimension
                )  # -h/2 in x
                grid_neighbors_media_ids[5] = (
                    abs(index_discrete_epsilon_map_1d[ijk1d_x_3 - y_stride_x_3 + 1])
                    // epsilon_dimension
                )  # -h/2 in y
                grid_neighbors_media_ids[6] = (
                    abs(index_discrete_epsilon_map_1d[ijk1d_x_3 - z_stride_x_3 + 2])
                    // epsilon_dimension
                )  # -h/2 in z

                # Check neighbors to determine if the current point is on the boundary
                for midpoint_index in range(1, 7):
                    neighbor_index = midpoint_index % 6 + 1
                    if grid_neighbors_media_ids[midpoint_index] == 0:
                        is_external_grid_point = True  # external point
                    if (
                        grid_neighbors_media_ids[midpoint_index]
                        != grid_neighbors_media_ids[neighbor_index]
                    ):
                        is_boundary_grid_point = True

                # If it's a boundary point, store its information
                if is_boundary_grid_point:
                    n_boundary_grid_points += 1  # //boundary bnd point
                    dielectric_boundary_midpoints_1d[ijk1d_x_3 + 1] = (
                        n_boundary_grid_points
                    )
                    dielectric_boundary_midpoints_1d[ijk1d_x_3 + 2] = (
                        is_external_grid_point
                    )

                    if is_external_grid_point:
                        n_external_boundary_point += 1
                        # bnd point and external point: boundary + surface
                        if n_boundary_grid_points < max_boundary_grid_points:
                            boundary_grid_indices[n_boundary_grid_points - 1] = (
                                np.array([i, j, k], dtype=dtype_int)
                            )
                        else:
                            print(
                                "Warning: Exceeded max_boundary_grid_points during assignment."
                            )
                else:
                    dielectric_boundary_midpoints_1d[ijk1d_x_3 + 1] = 0
                    dielectric_boundary_midpoints_1d[ijk1d_x_3 + 2] = 0
    return (
        n_boundary_grid_points,
        n_external_boundary_point,
        dielectric_boundary_midpoints_1d,
        boundary_grid_indices,
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
    atom_plus_probe_radii_1d = np.empty(num_atoms, dtype=dtype_real)
    atom_plus_probe_radii_square_1d = np.empty(num_atoms, dtype=dtype_real)
    atom_plus_probe_radii_shrink_1d = np.empty(num_atoms, dtype=dtype_real)
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


@njit(nogil=True, boundscheck=False, cache=True)
def _perform_cube_calculation(
    num_atoms: int,
    num_objects: int,
    num_molecules: int,
    voxels_per_entity: int,
    cube_side_length: float,
    cube_origin: np.ndarray,
    cube_shape: np.ndarray,
    atoms_data: np.ndarray,
    dtype_int,
    dtype_real,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs the calculation to populate the indexing cube with atom and object IDs (standalone version).

    Args:
        num_atoms (int): The total number of atoms.
        num_objects (int): The total number of objects.
        num_molecules (int): The total number of molecules.
        voxels_per_entity (int): The number of voxels per entity.
        cube_side_length (float): The length of each side of the cube.
        cube_origin (np.ndarray): The coordinates of the lowest vertex of the cube.
        cube_shape (np.ndarray): The dimensions (shape) of the cube grid.
        atoms_data (np.ndarray): Array containing atom properties.
        dtype_int (type): Integer data type. Defaults to int.
        dtype_real (type): Real data type. Defaults to float.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
               - voxel_atom_ids (np.ndarray): 1D array mapping voxel indices to atom/object IDs.
               - voxel_atom_start_indices (np.ndarray): Array storing the count of atoms/objects in each voxel.
               - voxel_atom_end_indices (np.ndarray): Array storing the cumulative count.
    """
    voxel_atom_ids = np.zeros(
        voxels_per_entity * (num_atoms + num_objects - num_molecules) + 1,
        dtype=dtype_int,
    )
    voxel_atom_ids, voxel_atom_start_indices, voxel_atom_end_indices = (
        voxelizer.build_atom_voxel_map(
            dtype_real(cube_side_length),
            dtype_int(num_atoms),
            dtype_int(num_objects),
            dtype_int(num_molecules),
            cube_origin.astype(dtype_real),
            cube_shape.astype(dtype_int),
            atoms_data.astype(dtype_real),
            voxel_atom_ids,
        )
    )
    return voxel_atom_ids, voxel_atom_start_indices, voxel_atom_end_indices


@njit(nogil=True, boundscheck=False, cache=True)
def _find_closest_neighbor(
    num_neighbors_found: int,
    probe_radius: float,
    min_distance_squared: float,
    closest_atom_or_object_index: int,
    num_atoms: int,
    atoms_data: np.ndarray,
    midpoint_coords: np.ndarray,
    neighbor_bounday_array: np.ndarray,
    dxyz: np.ndarray,
    dtype_int,
    dtype_real,
):
    """
    Finds the index of the closest neighboring atom or object to a given midpoint.

    Args:
        num_neighbors_found (int): The number of neighboring atoms or objects found.
        probe_radius (float): The radius of the probe molecule.
        min_distance_squared (float): The current minimum squared distance to a neighbor.
        closest_atom_or_object_index (int): The index of the closest neighbor found so far.
        num_atoms (int): The total number of atoms in the system.
        atoms_data (np.ndarray): Array containing atomic data (including coordinates and radii).
        midpoint_coords (np.ndarray): The coordinates of the midpoint being checked.
        neighbor_bounday_array (np.ndarray): Array containing indices of neighboring atoms or objects.
        dxyz (np.ndarray): Displacement vector (not fully clear from context, possibly related to surface normal).
        dtype_int (type): Integer data type.
        dtype_real (type): Real data type.

    Returns:
        tuple[int, float]: A tuple containing:
            - closest_atom_or_object_index (int): The index of the closest atom or object.
            - min_distance_squared (float): The minimum distance squared to the closest neighbor.
    """
    # Iterate through the found neighboring atoms or objects.
    for ii in range(1, num_neighbors_found + 1):
        if ii >= MAX_NEIGHBR_BOUNDARY_ARRAY_LEN:
            nprint_cpu(
                ERROR,
                _VERBOSITY,
                "space_vwtoms>> index beyond size of nbra: ii= ",
                ii,  # Adjusted index for printing
            )
        atom_or_object_index = dtype_int(neighbor_bounday_array[ii])
        # Check if the neighbor is an object (index greater than the number of atoms).
        if atom_or_object_index > num_atoms:
            object_index = dtype_int(atom_or_object_index - num_atoms)
            object_coords = midpoint_coords
            # an object can compete with an atom for a midpoint only if this midpoint is out of the object itself
            # Assuming distance_squared is calculated elsewhere or is always 0.0 here based on previous logic
            distance_squared = dtype_real(0.0)
            if 0.0 <= distance_squared < min_distance_squared:
                min_distance_squared = distance_squared
                closest_atom_or_object_index = atom_or_object_index
                if dxyz is not None:
                    distance_vector = (-dxyz) * (probe_radius - distance_squared)
        # If the neighbor is an atom
        else:
            # Get atom coordinates: atom_data index starts from 0, while atom_on_object from 1.
            atom_coords = (
                atoms_data[atom_or_object_index - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
            ).astype(dtype_real)
            atom_radius = atoms_data[atom_or_object_index - 1][ATOMFIELD_RADIUS]
            # Calculate the vector from the atom to the midpoint.
            distance_vector = (midpoint_coords - atom_coords).astype(dtype_real)
            distance_squared = size_cpu.dot_product(distance_vector, distance_vector)
            # Calculate the distance from the midpoint to the surface of the atom.
            distance_to_surface = sqrt(distance_squared) - atom_radius

            # Check if the distance to the atom surface is less than the current minimum.
            if distance_to_surface < min_distance_squared:
                min_distance_squared = distance_to_surface
                closest_atom_or_object_index = atom_or_object_index
    return closest_atom_or_object_index, min_distance_squared


@njit(nogil=True, boundscheck=False, cache=True)
def _check_contact_region(
    midpoint_entity_id: int,
    midpoint_coords: np.ndarray,
    closest_atom_or_object_index: int,
    num_atoms: int,
    atoms_data: np.ndarray,
    atom_plus_probe_radii_1d: np.ndarray,
    atom_plus_probe_radii_shrink_1d: np.ndarray,
    cube_side_length_inverse: float,
    cube_shape: np.ndarray,
    cube_vertex_lowest_xyz: np.ndarray,
    voxel_atom_count: np.ndarray,
    voxel_atom_count_cumulative: np.ndarray,
    voxel_atom_ids: np.ndarray,
    dtype_int,
    dtype_real,
    dtype_bool,
):
    """
    Checks if a midpoint is in the contact region of an atom or object (standalone version).

    Args:
        midpoint_entity_id (int): Entity id of the midpoint (0 is solvent).
        midpoint_coords (np.ndarray): The coordinates of the midpoint.
        closest_atom_or_object_index (int): The index of the atom or object closest to midpoint.
        num_atoms (int): Total number of atoms in the run.
        atoms_data (np.ndarray): Array containing atom properties.
        atom_plus_probe_radii_1d (np.ndarray): Sum of atom and probe radii.
        atom_plus_probe_radii_shrink_1d (np.ndarray): Shrunken squared radii.
        cube_side_length_inverse (float): Inverse of the cube side length.
        cube_shape (np.ndarray): Dimensions of the cube grid.
        cube_vertex_lowest_xyz (np.ndarray): Coordinates of the lowest vertex of the cube.
        voxel_atom_count (np.ndarray): Count of atoms/objects in each voxel.
        voxel_atom_count_cumulative (np.ndarray): Cumulative count.
        voxel_atom_ids (np.ndarray): IDs of atoms/objects in each voxel.
        dtype_int (type): Integer data type.
        dtype_real (type): Real data type.
        dtype_bool (type): Boolean data type.

    Returns:
        bool: True if the midpoint is in the contact region, False otherwise.
        neighbor_atom_or_object_index (int): The index of the atom or object in contact
    """
    in_contact = dtype_bool(True)
    if closest_atom_or_object_index <= num_atoms:
        atom_coords = (
            atoms_data[closest_atom_or_object_index - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
        ).astype(dtype_real)
        distance_vector = (midpoint_coords - atom_coords).astype(dtype_real)
        distance_to_center_squared = size_cpu.dot_product(
            distance_vector, distance_vector
        )
        distance_to_center = sqrt(distance_to_center_squared)
        if atom_plus_probe_radii_1d is not None:
            surface_point_coords = atom_coords + (
                (
                    atom_plus_probe_radii_1d[closest_atom_or_object_index - 1]
                    * distance_vector
                )
                / distance_to_center
            )
        else:
            surface_point_coords = atom_coords  # Fallback if radii are not available
    else:
        # Assuming distance_vector is defined or calculated appropriately for objects
        surface_point_coords = (
            midpoint_coords  # - self.dr123  # self.dr123 is not defined in this scope
        )

    indexing_cube_indices = (
        (surface_point_coords - cube_vertex_lowest_xyz) * cube_side_length_inverse
    ).astype(dtype_int)
    indexing_cube_dimensions = cube_shape.astype(dtype_int)
    if size_cpu.or_lt_scalar(
        indexing_cube_indices, dtype_int(0)
    ) or size_cpu.or_gt_vector(indexing_cube_indices, indexing_cube_dimensions):
        in_contact = dtype_bool(True)
        return in_contact, 0  # Treat as not in contact if outside cube

    lower_limit = voxel_atom_count[indexing_cube_indices[0]][indexing_cube_indices[1]][
        indexing_cube_indices[2]
    ]
    upper_limit = voxel_atom_count_cumulative[indexing_cube_indices[0]][
        indexing_cube_indices[1]
    ][indexing_cube_indices[2]]

    for kk in range(lower_limit, upper_limit + 1):
        neighbor_atom_or_object_index = voxel_atom_ids[kk]

        if neighbor_atom_or_object_index <= num_atoms:
            neighbor_atom_coords = (
                atoms_data[neighbor_atom_or_object_index - 1][
                    ATOMFIELD_X:ATOMFIELD_CRD_END
                ]
            ).astype(dtype_real)
            distance_vector_neighbor = (
                surface_point_coords - neighbor_atom_coords
            ).astype(dtype_real)
            distance_squared_neighbor = size_cpu.dot_product(
                distance_vector_neighbor, distance_vector_neighbor
            )
            if (
                distance_squared_neighbor
                < atom_plus_probe_radii_shrink_1d[neighbor_atom_or_object_index - 1]
            ):
                in_contact = dtype_bool(False)
                return in_contact, neighbor_atom_or_object_index
        else:
            if (
                neighbor_atom_or_object_index != closest_atom_or_object_index
                and midpoint_entity_id == 0
            ):
                object_coords = surface_point_coords
                # Assuming distance is calculated appropriately for objects
                distance = dtype_real(0.0)
                if distance < -APPROX_ZERO:
                    in_contact = dtype_bool(False)
                    return in_contact, neighbor_atom_or_object_index
    return in_contact, neighbor_atom_or_object_index


@njit(nogil=True, boundscheck=False, cache=True)
def _get_media_id(
    atom_or_object_index_closest: int,
    atom_or_object_index_current: int,
    num_atoms: int,
    atoms_data: np.ndarray,
    dtype_int,
):
    """
    Gets the media ID of an atom or assigns a default for cavities (standalone version).

    Args:
        atom_or_object_index_closest (int): Index of the closest atom or object.
        atom_or_object_index_current (int): Index of the current atom or object being considered.
        num_atoms (int): Number of atoms in the run.
        atoms_data (np.ndarray): Array containing atom properties.
        dtype_int (type): Integer data type. Defaults to int.

    Returns:
        int: The media ID.
    """
    if atom_or_object_index_closest == 0:
        if 0 < atom_or_object_index_current <= num_atoms:
            media_id = dtype_int(
                atoms_data[atom_or_object_index_current - 1][ATOMFIELD_MEDIA_ID]
            )

        else:
            nprint_cpu(
                DEBUG,
                _VERBOSITY,
                " VdMS> assigning arbitrary epsilon in cavity, atom_or_object_index_current=",
                atom_or_object_index_current,
            )
            media_id = dtype_int(atoms_data[0][ATOMFIELD_MEDIA_ID])
    elif 0 < atom_or_object_index_closest <= num_atoms:
        media_id = dtype_int(
            atoms_data[atom_or_object_index_closest - 1][ATOMFIELD_MEDIA_ID]
        )
    else:
        media_id = 0

    return media_id


@njit(nogil=True, boundscheck=False, cache=True)
def _remap_epsilon_map(
    grid_index_x: int,
    grid_index_y: int,
    grid_index_z: int,
    stride_x_3: int,
    stride_y_3: int,
    index_map: np.ndarray,
    boundary_point_index: int,
    neighbor_index: int,
    epsilon_dimension: int,
    contact_region_media_id: float,
    grid_neighs_entity_ids: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
    dtype_int=int,
    dtype_real=float,
):
    """
    Remaps the discrete epsilon map based on the media ID (standalone version).

    Args:
        grid_index_x (int): The x-index of the grid point.
        grid_index_y (int): The y-index of the grid point.
        grid_index_z (int): The z-index of the grid point.
        stride_x_3 (int): Stride along the x-dimension multiplied by 3.
        stride_y_3 (int): Stride along the y-dimension multiplied by 3.
        index_map (np.ndarray): Mapping of neighbor indices.
        boundary_point_index (int): The index of the current boundary point.
        neighbor_index (int): The index of the neighbor being considered.
        epsilon_dimension (int): The dimensions of epsilons distinct entities.
        grid_neighs_entity_ids (np.ndarray): Array of entity ids of grids neighbor midpoints.
        index_discrete_epsilon_map_1d (np.ndarray): 1D array representing the discrete dielectric constant map (to be updated).
        dtype_int (type): Integer data type. Defaults to int.
        dtype_real (type): Real data type. Defaults to float.

    Returns:
        np.ndarray: The updated index_discrete_epsilon_map_1d.
    """
    index_1d_x_3 = (
        (grid_index_x + index_map[1][neighbor_index]) * stride_x_3
        + (grid_index_y + index_map[2][neighbor_index]) * stride_y_3
        + (grid_index_z + index_map[3][neighbor_index]) * 3
    )
    entity_id = grid_neighs_entity_ids[neighbor_index]
    new_epsilon_value = dtype_real(
        entity_id + contact_region_media_id * epsilon_dimension
    )

    if index_map[4][neighbor_index] == 1:
        index_discrete_epsilon_map_1d[index_1d_x_3] = new_epsilon_value
        nprint_cpu(
            TRACE,
            _VERBOSITY,
            "i,j,ix+imap[1][j],iy+imap[2][j],iz+imap[3][j],discrete_epsilon_map_1d: ",
            boundary_point_index,
            neighbor_index,
            grid_index_x,
            index_map[1][neighbor_index],
            grid_index_y,
            index_map[2][neighbor_index],
            grid_index_z,
            index_map[3][neighbor_index],
            index_discrete_epsilon_map_1d[index_1d_x_3 : index_1d_x_3 + 3],
        )
    elif index_map[4][neighbor_index] == 2:
        index_discrete_epsilon_map_1d[index_1d_x_3 + 1] = new_epsilon_value
    elif index_map[4][neighbor_index] == 3:
        index_discrete_epsilon_map_1d[index_1d_x_3 + 2] = new_epsilon_value
    else:
        pass
        nprint_cpu(
            DEBUG,
            _VERBOSITY,
            " VdMS> ????? flag1",
            index_map[4][neighbor_index],
            ", j = ",
            neighbor_index,
        )
    return index_discrete_epsilon_map_1d


@njit(nogil=True, boundscheck=False, cache=True)
def _remap_possibly_sign_updated_epsilon_map(
    grid_index_x: int,
    grid_index_y: int,
    grid_index_z: int,
    stride_x_3: int,
    stride_y_3: int,
    index_map: np.ndarray,
    boundary_point_index: int,
    neighbor_index: int,
    epsilon_dimension: int,
    grid_neighs_entity_ids: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
    dtype_int=int,
    dtype_real=float,
):
    """
    Remaps the discrete epsilon map based on the media ID (standalone version).

    Args:
        grid_index_x (int): The x-index of the grid point.
        grid_index_y (int): The y-index of the grid point.
        grid_index_z (int): The z-index of the grid point.
        stride_x_3 (int): Stride along the x-dimension multiplied by 3.
        stride_y_3 (int): Stride along the y-dimension multiplied by 3.
        index_map (np.ndarray): Mapping of neighbor indices.
        boundary_point_index (int): The index of the current boundary point.
        neighbor_index (int): The index of the neighbor being considered.
        epsilon_dimension (int): The dimensions of epsilons distinct entities.
        grid_neighs_entity_ids (np.ndarray): Array of entity ids of grids neighbor midpoints.
        index_discrete_epsilon_map_1d (np.ndarray): 1D array representing the discrete dielectric constant map (to be updated).
        dtype_int (type): Integer data type. Defaults to int.
        dtype_real (type): Real data type. Defaults to float.

    Returns:
        np.ndarray: The updated index_discrete_epsilon_map_1d.
    """
    sign = dtype_real(1.0)
    index_1d_x_3 = (
        (grid_index_x + index_map[1][neighbor_index]) * stride_x_3
        + (grid_index_y + index_map[2][neighbor_index]) * stride_y_3
        + (grid_index_z + index_map[3][neighbor_index]) * 3
    )
    midpoint_original_epsilon_index = index_discrete_epsilon_map_1d[
        (index_1d_x_3 + index_map[4][neighbor_index] - 1)
    ]
    midpoint_original_entity_id = midpoint_original_epsilon_index % epsilon_dimension
    if midpoint_original_entity_id < 0:
        return index_discrete_epsilon_map_1d

    if grid_neighs_entity_ids[neighbor_index] < 0:
        sign = dtype_real(-1.0)
        if midpoint_original_entity_id == 0:
            midpoint_original_entity_id = 1

    midpoint_media_id = abs(midpoint_original_epsilon_index) // epsilon_dimension

    new_epsilon_value = dtype_real(
        sign * (midpoint_original_entity_id + midpoint_media_id * epsilon_dimension)
    )

    if index_map[4][neighbor_index] == 1:
        index_discrete_epsilon_map_1d[index_1d_x_3] = new_epsilon_value
    elif index_map[4][neighbor_index] == 2:
        index_discrete_epsilon_map_1d[index_1d_x_3 + 1] = new_epsilon_value
    elif index_map[4][neighbor_index] == 3:
        index_discrete_epsilon_map_1d[index_1d_x_3 + 2] = new_epsilon_value

    return index_discrete_epsilon_map_1d


@njit(nogil=True, boundscheck=False, cache=True)
def _update_neighbor_status(
    grid_index_x: int,
    grid_index_y: int,
    grid_index_z: int,
    neighbor_index: int,
    epsilon_dimension: int,
    grid_shape: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
    boundary_grid_indices: np.ndarray,
    dielectric_boundary_midpoints_1d: np.ndarray,
    num_external_boundary_points: int,
    added_boundary_points_increment: int,
    removed_boundary_points_increment: int,
    num_discovered_bndy_grid_points: int,
    max_boundary_grid_points: int,
    dtype_int,
) -> tuple[int, np.ndarray, int, int, int]:
    """
    Checks and updates the status of neighboring grid points based on epsilon values.

    This function examines a specific neighbor of a given grid point and determines
    if it lies on the dielectric boundary. It updates the `dielectric_boundary_midpoints_1d`
    array to mark boundary points and keeps track of the number of external boundary
    points and the number of added or removed boundary points.

    Args:
        grid_index_x (int): The x-index of the central grid point.
        grid_index_y (int): The y-index of the central grid point.
        grid_index_z (int): The z-index of the central grid point.
        neighbor_index (int): The index of the neighbor being considered (1-6).
            Passing 7 skips neighbor marking.
            1: +x, 2: +y, 3: +z, 4: -x, 5: -y, 6: -z.
        epsilon_dimension (int): The number of different epsilon media.
        grid_shape (np.ndarray): A NumPy array representing the dimensions of the grid (nx, ny, nz).
        index_discrete_epsilon_map_1d (np.ndarray): A 1D array representing the discrete
            dielectric constant map.
        boundary_grid_indices (np.ndarray): A NumPy array to store the 3D indices of boundary
            grid points. This array is updated if a new boundary point is found.
        dielectric_boundary_midpoints_1d (np.ndarray): A 1D array used to mark grid points as being on
            the dielectric boundary. Stores additional information as well.
        num_external_boundary_points (int): The current number of boundary grid points that
            are on the exterior of the solute. This count is updated.
        added_boundary_points_increment (int): A counter for the number of boundary points
            added in the current iteration. This is updated.
        removed_boundary_points_increment (int): A counter for the number of boundary points
            removed in the current iteration. This is updated.
        num_discovered_bndy_grid_points (int): The total count of boundary grid points
            discovered so far.
        max_boundary_grid_points (int): The maximum allowed number of boundary grid points.
        dtype_int (type): The integer data type used in the calculations (e.g., np.int32).

    Returns:
        tuple[int, np.ndarray, int, int, int]:
            - exec_status (int): An execution status flag. Returns `EXIT_NJIT_FLAG` if an
              error occurs (e.g., exceeding the maximum number of boundary points), otherwise 0.
            - dielectric_boundary_midpoints_1d (np.ndarray): The updated 1D array marking dielectric
              boundary points.
            - num_external_boundary_points (int): The updated count of external boundary points.
            - added_boundary_points_increment (int): The updated count of added boundary points.
            - removed_boundary_points_increment (int): The updated count of removed boundary points.
    """
    exec_status = 0  # Assume function run is successful, on error return EXIT_NJIT_FLAG
    epsilon_dimension = dtype_int(epsilon_dimension)

    # Calculate strides to convert 3D indices to 1D index
    x_stride, y_stride, z_stride, x_stride_x_3, y_stride_x_3, z_stride_x_3 = (
        _calculate_strides(grid_shape, dtype_int=dtype_int)
    )
    neighbor_x, neighbor_y, neighbor_z = grid_index_x, grid_index_y, grid_index_z

    # Adjust neighbor indices based on the neighbor_index
    if 1 <= neighbor_index <= 6:
        if neighbor_index == 1:
            neighbor_x += 1
            if neighbor_x >= grid_shape[0] - 1:
                return (
                    exec_status,
                    dielectric_boundary_midpoints_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                )
        elif neighbor_index == 2:
            neighbor_y += 1
            if neighbor_y >= grid_shape[1] - 1:
                return (
                    exec_status,
                    dielectric_boundary_midpoints_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                )
        elif neighbor_index == 3:
            neighbor_z += 1
            if neighbor_z >= grid_shape[2] - 1:
                return (
                    exec_status,
                    dielectric_boundary_midpoints_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                )
        elif neighbor_index == 4:
            neighbor_x -= 1
            if neighbor_x <= 0:
                return (
                    exec_status,
                    dielectric_boundary_midpoints_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                )
        elif neighbor_index == 5:
            neighbor_y -= 1
            if neighbor_y <= 0:
                return (
                    exec_status,
                    dielectric_boundary_midpoints_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                )
        elif neighbor_index == 6:
            neighbor_z -= 1
            if neighbor_z <= 0:
                return (
                    exec_status,
                    dielectric_boundary_midpoints_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                )

    is_external = dtype_int(0)
    is_boundary = dtype_int(0)
    neighbor_index_1d = neighbor_x * x_stride + neighbor_y * y_stride + neighbor_z
    neighbor_index_1d_x_3 = neighbor_index_1d * 3
    neighbor_media_ids = np.zeros(7, dtype=dtype_int)

    grid_neighs_1d_offsets = np.zeros(7, dtype=dtype_int)
    # Offsets for the six neighboring midpoints in the 1D epsilon map array.
    # Leading by h/2 neighbors in x, y, z directions.
    grid_neighs_1d_offsets[1:4] = 0, 1, 2
    # Lagging by h/2 neighbors in x, y, z directions.
    grid_neighs_1d_offsets[4:7] = -x_stride_x_3, -y_stride_x_3 + 1, -z_stride_x_3 + 2

    # Get the media IDs of the six neighboring midpoints of the current neighbor.
    for neigh_index in range(1, 7):
        neighbor_media_ids[neigh_index] = (
            abs(
                index_discrete_epsilon_map_1d[
                    neighbor_index_1d_x_3 + grid_neighs_1d_offsets[neigh_index]
                ]
            )
            // epsilon_dimension
        )

    # Determine if the neighbor is external (solvent) based on the first midpoint.
    if neighbor_media_ids[1] == 0:
        is_external = 1
    # Check if there's a change in media ID between the first and last midpoints
    if neighbor_media_ids[1] != neighbor_media_ids[6]:
        is_boundary = 1
    # Check for changes in media ID across all adjacent midpoints
    for midpoint_index in range(2, 7):
        if neighbor_media_ids[midpoint_index] == 0:
            is_external = 1
        if neighbor_media_ids[midpoint_index] != neighbor_media_ids[midpoint_index - 1]:
            is_boundary = 1

    # If the neighbor is no longer a boundary point
    if (is_boundary == 0) and (
        dielectric_boundary_midpoints_1d[neighbor_index_1d_x_3 + 1] != 0
    ):
        # Decrement the external boundary point count if it was external
        num_external_boundary_points -= dielectric_boundary_midpoints_1d[
            neighbor_index_1d_x_3 + 2
        ]
        # Reset the boundary point information
        dielectric_boundary_midpoints_1d[neighbor_index_1d_x_3 + 1] = 0
        dielectric_boundary_midpoints_1d[neighbor_index_1d_x_3 + 2] = 0
        removed_boundary_points_increment += 1
        nprint_cpu(
            TRACE,
            _VERBOSITY,
            "removing boundary point: ",
            (neighbor_x, neighbor_y, neighbor_z),
        )
    else:
        # If it's a boundary and was marked as external, but shouldn't be anymore
        if (
            is_boundary == 1
            and is_external == 0
            and dielectric_boundary_midpoints_1d[neighbor_index_1d_x_3 + 2] == 1
        ):
            num_external_boundary_points -= 1
            dielectric_boundary_midpoints_1d[neighbor_index_1d_x_3 + 2] = 0
            nprint_cpu(
                TRACE,
                _VERBOSITY,
                "changing ext_bgp into int_bgp: ",
                (neighbor_x, neighbor_y, neighbor_z),
            )

    # If the neighbor is a boundary point and not already marked as one
    if (
        is_boundary == 1
        and dielectric_boundary_midpoints_1d[neighbor_index_1d_x_3 + 1] == 0
    ):
        added_boundary_points_increment += 1
        dielectric_boundary_midpoints_1d[neighbor_index_1d_x_3 + 1] = (
            num_discovered_bndy_grid_points + added_boundary_points_increment
        )
        if (
            num_discovered_bndy_grid_points + added_boundary_points_increment
            > max_boundary_grid_points
        ):
            nprint_cpu(
                CRITICAL,
                _VERBOSITY,
                " ERROR> This case is too big, ibmx need to be increased.",
            )
            exec_status = EXIT_NJIT_FLAG

        if exec_status != EXIT_NJIT_FLAG:
            # If the limit is not exceeded, store the boundary point's index
            boundary_grid_indices[
                num_discovered_bndy_grid_points + added_boundary_points_increment - 1
            ][:] = np.array([neighbor_x, neighbor_y, neighbor_z], dtype=dtype_int)
            dielectric_boundary_midpoints_1d[neighbor_index_1d_x_3 + 2] = is_external
            num_external_boundary_points += dielectric_boundary_midpoints_1d[
                neighbor_index_1d_x_3 + 2
            ]
            nprint_cpu(
                TRACE,
                _VERBOSITY,
                "creating boundary_point: ",
                (neighbor_x, neighbor_y, neighbor_z),
            )
    return (
        exec_status,
        dielectric_boundary_midpoints_1d,
        num_external_boundary_points,
        added_boundary_points_increment,
        removed_boundary_points_increment,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def _update_gridpoint_boundary_status(
    grid_index_x: int,
    grid_index_y: int,
    grid_index_z: int,
    epsilon_dimension: int,
    grid_shape: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
    dielectric_boundary_midpoints_1d: np.ndarray,
    num_external_boundary_points: int,
    removed_boundary_points_increment: int,
    dtype_int,
) -> tuple[int, np.ndarray, int, int]:
    """
    Checks and updates the status of neighboring grid points based on epsilon values.

    This function examines a specific neighbor of a given grid point and determines
    if it lies on the dielectric boundary. It updates the `dielectric_boundary_midpoints_1d`
    array to mark boundary points and keeps track of the number of external boundary
    points and the number of added or removed boundary points.

    Args:
        grid_index_x (int): The x-index of the central grid point.
        grid_index_y (int): The y-index of the central grid point.
        grid_index_z (int): The z-index of the central grid point.
        epsilon_dimension (int): The number of different epsilon media.
        grid_shape (np.ndarray): A NumPy array representing the dimensions of the grid (nx, ny, nz).
        index_discrete_epsilon_map_1d (np.ndarray): A 1D array representing the discrete
            dielectric constant map.
        dielectric_boundary_midpoints_1d (np.ndarray): A 1D array used to mark grid points as being on
            the dielectric boundary. Stores additional information as well.
        num_external_boundary_points (int): The current number of boundary grid points that
            are on the exterior of the solute. This count is updated.
        removed_boundary_points_increment (int): A counter for the number of boundary points
            removed in the current iteration. This is updated.

    Returns:
        tuple[int, np.ndarray, int, int, int]:
            - exec_status (int): An execution status flag. Returns `EXIT_NJIT_FLAG` if an
              error occurs (e.g., exceeding the maximum number of boundary points), otherwise 0.
            - dielectric_boundary_midpoints_1d (np.ndarray): The updated 1D array marking dielectric
              boundary points.
            - num_external_boundary_points (int): The updated count of external boundary points.
            - removed_boundary_points_increment (int): The updated count of removed boundary points.
    """
    exec_status = 0  # Assume function run is successful, on error return EXIT_NJIT_FLAG
    epsilon_dimension = dtype_int(epsilon_dimension)

    # Calculate strides to convert 3D indices to 1D index
    x_stride, y_stride, z_stride, x_stride_x_3, y_stride_x_3, z_stride_x_3 = (
        _calculate_strides(grid_shape, dtype_int=dtype_int)
    )
    neighbor_x, neighbor_y, neighbor_z = grid_index_x, grid_index_y, grid_index_z
    process_neighbor = True

    is_external = dtype_int(0)
    is_boundary = dtype_int(0)
    neighbor_index_1d = neighbor_x * x_stride + neighbor_y * y_stride + neighbor_z
    neighbor_index_1d_x_3 = neighbor_index_1d * 3
    neighbor_media_ids = np.zeros(7, dtype=dtype_int)

    grid_neighs_1d_offsets = np.zeros(7, dtype=dtype_int)
    # Offsets for the six neighboring midpoints in the 1D epsilon map array.
    # Leading by h/2 neighbors in x, y, z directions.
    grid_neighs_1d_offsets[1:4] = 0, 1, 2
    # Lagging by h/2 neighbors in x, y, z directions.
    grid_neighs_1d_offsets[4:7] = -x_stride_x_3, -y_stride_x_3 + 1, -z_stride_x_3 + 2

    # Get the media IDs of the six neighboring midpoints of the current neighbor.
    for neigh_index in range(1, 7):
        neighbor_media_ids[neigh_index] = (
            abs(
                index_discrete_epsilon_map_1d[
                    neighbor_index_1d_x_3 + grid_neighs_1d_offsets[neigh_index]
                ]
            )
            // epsilon_dimension
        )

    # Determine if the neighbor is external (solvent) based on the first midpoint.
    if neighbor_media_ids[1] == 0:
        is_external = 1
    # Check if there's a change in media ID between the first and last midpoints
    if neighbor_media_ids[1] != neighbor_media_ids[6]:
        is_boundary = 1
    # Check for changes in media ID across all adjacent midpoints
    for midpoint_index in range(2, 7):
        if neighbor_media_ids[midpoint_index] == 0:
            is_external = 1
        if neighbor_media_ids[midpoint_index] != neighbor_media_ids[midpoint_index - 1]:
            is_boundary = 1

    # If the neighbor is no longer a boundary point
    if (is_external == 0) or (is_boundary == 0):
        # Decrement the external boundary point count if it was external
        num_external_boundary_points -= dielectric_boundary_midpoints_1d[
            neighbor_index_1d_x_3 + 2
        ]
        # Update the boundary point information
        if is_boundary == 1:
            dielectric_boundary_midpoints_1d[neighbor_index_1d_x_3 + 2] = is_external
        # Reset the boundary point information
        if is_boundary == 0:
            dielectric_boundary_midpoints_1d[neighbor_index_1d_x_3 + 1] = 0
            dielectric_boundary_midpoints_1d[neighbor_index_1d_x_3 + 2] = 0
            removed_boundary_points_increment += 1
            if is_external == 1:
                nprint_cpu(
                    DEBUG,
                    _VERBOSITY,
                    "born a new external point on gridpoint",
                    (neighbor_x, neighbor_y, neighbor_z),
                )

    return (
        exec_status,
        dielectric_boundary_midpoints_1d,
        num_external_boundary_points,
        removed_boundary_points_increment,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def _process_boundary_point_midpoint(
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
    grid_neighs_entity_ids: np.ndarray,
    grid_neighs_media_ids: np.ndarray,
    rm_boundary_pt: bool,
    min_xyz: np.ndarray,
    cube_side_indver_inverse: float,
    cube_shape_indver: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
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
    neighbor_boundary_array: np.ndarray,
    num_atoms: int,
    atoms_data: np.ndarray,
    atom_plus_probe_radii_1d: np.ndarray,
    atom_plus_probe_radii_shrink_1d: np.ndarray,
    epsilon_dimension: int,
    boundary_point_end_index: int,
    max_boundary_grid_points: int,
    dxyz: np.ndarray,
    num_cavity_points: int,
    dielectric_boundary_midpoints_1d: np.ndarray,
    num_external_boundary_points: int,
    boundary_grid_indices: np.ndarray,
    added_boundary_points_increment: int,
    removed_boundary_points_increment: int,
    dtype_int: type,
    dtype_real: type,
    dtype_bool: type,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
    int,
    np.ndarray,
    int,
    int,
]:
    """
    Processes a boundary grid point by checking its neighboring midpoints and updating
    dielectric boundary conditions based on atom proximity.

    This function determines whether a given midpoint lies inside or outside the indexing
    cube, finds the closest atom or object, and updates dielectric boundary values accordingly.

    Args:
        boundary_point_index (int): Index of the boundary point being processed.
        grid_index_x (int): X-coordinate index in the grid.
        grid_index_y (int): Y-coordinate index in the grid.
        grid_index_z (int): Z-coordinate index in the grid.
        grid_spacing (float): Spacing between grid points in the computational domain.
        grid_shape (np.ndarray): Dimensions (shape) of the grid.
        grid_origin (np.ndarray): Origin coordinates of the grid.
        probe_radius_squared_1 (float): Squared probe radius for boundary calculations.
        probe_radius_squared_2 (float): Alternative squared probe radius for different cases.
        grid_neighbor_coords_offsets (np.ndarray): Coordinate offsets for neighboring midpoints of a grid points.
        index_map (np.ndarray): Mapping from grid to atom indices.
        grid_neighs_entity_ids (np.ndarray): Epsilon entiry ids assigned to midpoints points.
        grid_neighs_media_ids (np.ndarray): Epsilon media ids for neighboring midpoints points.
        rm_boundary_pt (bool): Flag to indicate whether the boundary point should be removed.
        num_atoms (int): Total number of atoms in the system.
        min_xyz (np.ndarray): Minimum coordinates for the computational domain.
        cube_side_indver_inverse (float): Inverse cube side length factor for indexing.
        cube_shape_indver (np.ndarray): Shape of the indexing cube.
        index_discrete_epsilon_map_1d (np.ndarray): 1D mapping of epsilon values.
        cycle_flag (bool): Flag for iteration control in boundary processing.
        lower_atom_index_bounds (np.ndarray): Lower index bounds for atoms in the grid.
        upper_atom_index_bounds (np.ndarray): Upper index bounds for atoms in the grid.
        cumulative_atom_indices (np.ndarray): Cumulative indices for atoms in voxel storage.
        exposed_grids_coords (np.ndarray): Coordinates of exposed grid points.
        cube_vertex_lowest_xyz (np.ndarray): Lowest vertex coordinate of the cube.
        cube_side_length_inverse (float): Inverse length of a cube side.
        cube_shape (np.ndarray): Shape of the indexing cube.
        voxel_atom_count (np.ndarray): Number of atoms per voxel.
        voxel_atom_count_cumulative (np.ndarray): Cumulative count of atoms in voxels.
        voxel_atom_ids (np.ndarray): IDs of atoms in each voxel.
        atom_surface_flags (np.ndarray): Flags indicating which atoms are part of a surface.
        neighbor_boundary_array (np.ndarray): Array storing neighboring atom indices.
        atoms_data (np.ndarray): Atomic data such as positions and radii.
        epsilon_dimension (int): Dimensionality of the epsilon array.
        dxyz (np.ndarray): Optional displacement vector for atom positions.
        num_cavity_points (int): Counter for cavity points detected. Defaults to 0.
        dielectric_boundary_midpoints_1d (np.ndarray): Dielectric boundary mapping. Defaults to None.
        num_external_boundary_points (int): Count of external boundary points. Defaults to 0.
        boundary_grid_indices (np.ndarray): Grid indices marking boundaries. Defaults to None.
        added_boundary_points_increment (int): Loop index tracking media updates.
        removed_boundary_points_increment (int): Loop index tracking neighbor updates.
        dtype_int (type): Integer type for Delphi calculations.
        dtype_real (type): Floating-point type for Delphi calculations.
        dtype_bool (type): Boolean type for Delphi calculations.

    Returns:
        tuple:
            - Run status: neigh_update_exec_status, 0 on success EXIT_NJIT_FLAG on error
            - Updated `grid_neighs_entity_ids` array.
            - Updated `grid_neighs_media_ids` array.
            - Updated `num_cavity_points` count.
            - Updated `dielectric_boundary_midpoints_1d` array.
            - Updated `num_external_boundary_points` count.
            - Updated `boundary_grid_indices` array.
            - Updated `added_boundary_points_increment` count.
            - Updated `removed_boundary_points_increment` count.
    """
    probe_radius = sqrt(probe_radius_squared_1)
    last_cube_indver_indices = (cube_shape_indver - 1).astype(dtype_int)
    neigh_update_exec_status = (
        0  # Set it to success, it will be set to EXIT_NJIT_FLAG on error
    )

    grid_point_coords = (
        (
            grid_spacing
            * np.array([grid_index_x, grid_index_y, grid_index_z], dtype=dtype_int)
        )
        + grid_origin
    ).astype(dtype_real)

    z_stride = 1
    y_stride = dtype_int(grid_shape[2])
    x_stride = dtype_int(grid_shape[1] * y_stride)
    grid_index_1d = dtype_int(
        grid_index_x * x_stride + grid_index_y * y_stride + grid_index_z
    )
    grid_index_1d_times_3 = dtype_int(grid_index_1d * 3)

    for neighbor_index in range(1, 7):
        if (
            grid_neighs_entity_ids[neighbor_index] == 0
            or (
                rm_boundary_pt
                and grid_neighs_entity_ids[neighbor_index] > num_atoms + 1
            )
            or (
                grid_neighs_media_ids[neighbor_index] == 0
                and grid_neighs_entity_ids[neighbor_index] > 0
            )
        ):
            probe_radius_squared = probe_radius_squared_2
            if (
                grid_neighs_entity_ids[neighbor_index] == 0
                or grid_neighs_media_ids[neighbor_index] == 0
            ):
                probe_radius_squared = probe_radius_squared_1

            # add midpoint offset to grid point..
            midpoint_coords = (
                grid_point_coords + grid_neighbor_coords_offsets[neighbor_index]
            )
            # determine if this virgin midpoint is in or out of the indexing cube
            relative_coords = (midpoint_coords - min_xyz) * cube_side_indver_inverse
            midpoint_indices = relative_coords.astype(dtype_int)
            midpoint_index_x = midpoint_indices[0]
            midpoint_index_y = midpoint_indices[1]
            midpoint_index_z = midpoint_indices[2]

            if size_cpu.or_le_scalar(
                midpoint_indices, dtype_int(0)
            ) or size_cpu.or_ge_vector(midpoint_indices, last_cube_indver_indices):
                nprint_cpu(TRACE, _VERBOSITY, " VdMS> midpoint out of cube")
                nprint_cpu(
                    TRACE,
                    _VERBOSITY,
                    index_discrete_epsilon_map_1d[grid_index_1d_times_3],
                )

            min_distance_squared = dtype_real(1000.0)
            closest_atom_or_object_index_voxel = dtype_int(0)

            (
                cycle_flag,
                closest_atom_or_object_index_voxel,
                min_distance_squared,
            ) = helpers.vdw_to_ms_all_voxels(
                cycle_flag,
                neighbor_index,
                midpoint_index_x,
                midpoint_index_y,
                midpoint_index_z,
                probe_radius_squared,
                midpoint_coords,
                grid_neighs_entity_ids,
                grid_neighs_media_ids,
                closest_atom_or_object_index_voxel,
                min_distance_squared,
                cube_voxel_start_indices,
                cube_voxel_end_indices,
                grid_point_indices_in_voxels,
                exposed_grids_coords,
            )
            if cycle_flag:
                continue

            # it might be in the contact region find the closest atom surface
            indexing_cube_indices = (
                (midpoint_coords - cube_vertex_lowest_xyz) * cube_side_length_inverse
            ).astype(dtype_int)
            min_distance_squared = dtype_real(100.0)
            closest_atom_or_object_index = dtype_int(0)
            num_neighbors_found = dtype_int(0)

            indexing_cube_dimensions = cube_shape.astype(dtype_int)

            if size_cpu.or_lt_scalar(
                indexing_cube_indices, dtype_int(0)
            ) or size_cpu.or_gt_vector(indexing_cube_indices, indexing_cube_dimensions):
                pass  # Logic for objects might go here
            else:
                lower_limit = voxel_atom_count[indexing_cube_indices[0]][
                    indexing_cube_indices[1]
                ][indexing_cube_indices[2]]
                upper_limit = voxel_atom_count_cumulative[indexing_cube_indices[0]][
                    indexing_cube_indices[1]
                ][indexing_cube_indices[2]]

            previous_atom_or_object_index = dtype_int(0)

            for kk in range(lower_limit, upper_limit + 1):
                atom_or_object_index = voxel_atom_ids[kk]

                if kk > 0 and atom_or_object_index == 0:
                    nprint_cpu(DEBUG, _VERBOSITY, " VdMS> problems with cube")

                if 0 < atom_or_object_index <= num_atoms:
                    if atom_surface_flags[atom_or_object_index] == 0:
                        num_neighbors_found += 1
                        neighbor_boundary_array[num_neighbors_found] = (
                            atom_or_object_index
                        )
                else:
                    if (
                        atom_or_object_index != previous_atom_or_object_index
                        and grid_neighs_entity_ids[neighbor_index] == 0
                    ):
                        previous_atom_or_object_index = atom_or_object_index
                        num_neighbors_found = num_neighbors_found + 1
                        neighbor_boundary_array[num_neighbors_found - 1] = (
                            atom_or_object_index  # Adjust index for 0-based array
                        )

            (
                closest_atom_or_object_index,
                min_distance_squared,
            ) = _find_closest_neighbor(
                num_neighbors_found=num_neighbors_found,
                probe_radius=probe_radius,
                min_distance_squared=min_distance_squared,
                closest_atom_or_object_index=closest_atom_or_object_index,
                num_atoms=num_atoms,
                atoms_data=atoms_data,
                midpoint_coords=midpoint_coords,
                dxyz=dxyz,
                neighbor_bounday_array=neighbor_boundary_array,
                dtype_int=dtype_int,
                dtype_real=dtype_real,
            )

            if closest_atom_or_object_index == 0:
                num_cavity_points += 1  # possibly a cavity point
            else:
                (in_contact, neighbor_atom_or_object_index) = _check_contact_region(
                    midpoint_entity_id=grid_neighs_entity_ids[neighbor_index],
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
                    grid_neighs_entity_ids[neighbor_index] = (
                        -closest_atom_or_object_index
                    )
                    grid_neighs_media_ids[neighbor_index] = (
                        -closest_atom_or_object_index
                    )
                    continue

            grid_neighs_entity_ids[neighbor_index] = (
                1  # //eps = 1 means cavity or reentrant;
            )

            contact_region_media_id = _get_media_id(
                closest_atom_or_object_index,
                atom_or_object_index,
                num_atoms,
                atoms_data,
                dtype_int=dtype_int,
            )

            index_discrete_epsilon_map_1d = _remap_epsilon_map(
                grid_index_x=grid_index_x,
                grid_index_y=grid_index_y,
                grid_index_z=grid_index_z,
                stride_x_3=x_stride * 3,
                stride_y_3=y_stride * 3,
                index_map=index_map,
                boundary_point_index=boundary_point_index,
                neighbor_index=neighbor_index,
                epsilon_dimension=epsilon_dimension,
                contact_region_media_id=contact_region_media_id,
                grid_neighs_entity_ids=grid_neighs_entity_ids,
                index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
                dtype_int=dtype_int,
                dtype_real=dtype_real,
            )
            grid_neighs_media_ids[neighbor_index] = contact_region_media_id

            (
                neigh_update_exec_status,
                dielectric_boundary_midpoints_1d,
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
                dielectric_boundary_midpoints_1d,
                num_external_boundary_points,
                added_boundary_points_increment,
                removed_boundary_points_increment,
                boundary_point_end_index,
                max_boundary_grid_points,
                dtype_int=dtype_int,
            )

    return (
        neigh_update_exec_status,
        grid_neighs_entity_ids,
        grid_neighs_media_ids,
        num_cavity_points,
        dielectric_boundary_midpoints_1d,
        num_external_boundary_points,
        boundary_grid_indices,
        added_boundary_points_increment,
        removed_boundary_points_increment,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def _process_boundary_grid_points_loop(
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
    dielectric_boundary_midpoints_1d: np.ndarray,
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
    neighbor_bounday_array: np.ndarray,
    num_atoms: int,
    atoms_data: np.ndarray,
    atom_plus_probe_radii_1d: np.ndarray,
    atom_plus_probe_radii_shrink_1d: np.ndarray,
    num_external_boundary_points: int,
    dxyz: np.ndarray,
    dtype_int=int,
    dtype_real=float,
    dtype_bool=bool,
):
    """
    Processes the boundary grid points to determine the molecular surface.

    This function iterates through a range of identified boundary grid points and
    refines their status based on their local environment, specifically checking
    for overlap with atoms using the probe radius. It can remove boundary points
    that are deemed to be inside atoms (based on the `rm_boundary_pt_condition`)
    and updates the `dielectric_boundary_midpoints_1d` array accordingly. It also manages
    counters for added and removed boundary points and the number of external
    boundary points.

    Args:
        probe_radius_squared_1 (float): The square of the probe radius.
        probe_radius_squared_2 (float): Another value related to the probe radius squared (purpose might vary).
        boundary_point_start_index (int): The starting index (1-based) in the `boundary_grid_indices` array
            to begin processing.
        boundary_point_end_index (int): The ending index (inclusive, 1-based) in the `boundary_grid_indices`
            array to stop processing.
        x_stride (int): The stride in the 1D grid array along the x-axis.
        y_stride (int): The stride in the 1D grid array along the y-axis.
        z_stride (int): The stride in the 1D grid array along the z-axis.
        grid_origin_current (np.ndarray): A NumPy array containing the coordinates of the origin of the current grid.
        max_boundary_grid_points (int): The maximum number of boundary grid points allowed.
        grid_neighbor_coords_offsets (np.ndarray): A NumPy array containing the coordinate offsets
            for the six neighboring midpoints of a grid point.
        boundary_grid_indices (np.ndarray): A NumPy array storing the 3D indices of the identified
            boundary grid points.
        grid_spacing (float): The spacing between grid points.
        grid_shape (np.ndarray): A NumPy array representing the dimensions of the grid (nx, ny, nz).
        exposed_grids_coords (np.ndarray): A NumPy array containing the coordinates of grid points
            that are considered 'exposed' (e.g., on the surface).
        index_discrete_epsilon_map_1d (np.ndarray): A 1D array representing the discrete
            dielectric constant map.
        index_map (np.ndarray): A 5x7 array used for mapping indices to neighbor offsets.
        dielectric_boundary_midpoints_1d (np.ndarray): A 1D array used to mark grid points as being on
            the dielectric boundary. Stores additional information as well.
        epsilon_dimension (int): The number of different epsilon media.
        rm_boundary_pt_condition (bool): A boolean flag indicating whether to apply the condition
            for removing boundary points (e.g., based on the number of molecules).
        min_xyz (np.ndarray): A NumPy array containing the minimum x, y, and z coordinates of the system.
        cube_side_indver_inverse (float): The inverse of the side length of the indexing cube used for vertices.
        cube_shape_indver (np.ndarray): A NumPy array representing the shape of the indexing cube used for vertices.
        cube_voxel_start_indices (np.ndarray): A NumPy array containing the start indices of grid points
            within each voxel of the vertex indexing cube.
        cube_voxel_end_indices (np.ndarray): A NumPy array containing the end indices of grid points
            within each voxel of the vertex indexing cube.
        cube_vertex_lowest_xyz (np.ndarray): A NumPy array containing the coordinates of the lowest vertex
            of the indexing cube used for vertices.
        cube_side_length_inverse (float): The inverse of the side length of the main indexing cube.
        cube_shape (np.ndarray): A NumPy array representing the shape of the main indexing cube.
        voxel_grid_point_indices (np.ndarray): A NumPy array containing the 1D indices of exposed grid points
            that fall within each voxel of the vertex indexing cube.
        voxel_atom_count (np.ndarray): A NumPy array storing the number of atoms in each voxel of the
            main indexing cube.
        voxel_atom_count_cumulative (np.ndarray): A NumPy array storing the cumulative count of atoms
            in the voxels of the main indexing cube.
        voxel_atom_ids (np.ndarray): A NumPy array storing the IDs (1-based) of the atoms present in each
            voxel of the main indexing cube.
        atom_surface_flags (np.ndarray): A NumPy array of boolean flags indicating if an atom is on the surface.
        neighbor_bounday_array (np.ndarray): An array used to store the indices of neighboring atoms or objects
            for a given boundary point.
        num_atoms (int): The total number of atoms in the system.
        atoms_data (np.ndarray): A NumPy array containing the data for each atom (including coordinates and radii).
        atom_plus_probe_radii_1d (np.ndarray): A 1D array storing the sum of the radius of each atom and the probe radius.
        atom_plus_probe_radii_shrink_1d (np.ndarray): A 1D array storing the squared sum of the radius of each atom
            and the probe radius, shrunk by a factor for contact detection.
        num_external_boundary_points (int): The current number of boundary grid points that are on the
            exterior of the solute. This count will be updated.
        dxyz (np.ndarray): A NumPy array representing a displacement vector (its exact purpose might depend on the context).
        dtype_int (type): The integer data type used in the calculations (e.g., `int` or `np.int32`).
        dtype_real (type): The floating-point data type used in the calculations (e.g., `float` or `np.float64`).
        dtype_bool (type): The boolean data type used in the calculations (e.g., `bool` or `np.bool_`).

    Returns:
        tuple[int, bool, int, int, int]:
            - neigh_update_exec_status (int): An execution status flag. Returns `EXIT_NJIT_FLAG` if an
              error occurs in the neighbor update process, otherwise 0.
            - cycle_flag (bool): A boolean flag indicating if a cycle was detected during the process (purpose might vary).
            - added_boundary_points_increment (int): The number of boundary points added during this loop iteration.
            - removed_boundary_points_increment (int): The number of boundary points removed during this loop iteration.
            - num_external_boundary_points (int): The updated number of external boundary points.
    """
    added_boundary_points_increment = 0
    removed_boundary_points_increment = 0
    grid_neighs_entity_ids = np.zeros(7, dtype=dtype_int)  # Renamed from eps
    grid_neighs_media_ids = np.zeros(7, dtype=dtype_int)  # Renamed from eps2

    x_stride_x_3 = x_stride * 3
    y_stride_x_3 = y_stride * 3

    grid_neighs_1d_offsets = np.zeros(7, dtype=dtype_int)
    # Leading by h/2 neighbors in x, y, z directions
    grid_neighs_1d_offsets[1:4] = 0, 1, 2
    # Lagging by h/2 neighbors in x, y, z directions
    grid_neighs_1d_offsets[4:7] = -x_stride_x_3, -y_stride_x_3 + 1, -z_stride * 3 + 2

    probe_radius = sqrt(probe_radius_squared_1)
    cycle_flag = False
    num_cavity_points = 0

    for this_bgp_index in range(
        boundary_point_start_index, boundary_point_end_index + 1
    ):
        ixyz = boundary_grid_indices[
            this_bgp_index - 1
        ]  # Adjust index for 0-based array
        grid_index_x = ixyz[0]
        grid_index_y = ixyz[1]
        grid_index_z = ixyz[2]
        grid_index_1d = (
            grid_index_x * x_stride + grid_index_y * y_stride + grid_index_z * z_stride
        )
        grid_index_1d_times_3 = grid_index_1d * 3
        if dielectric_boundary_midpoints_1d[grid_index_1d_times_3 + 1] != 0:
            rm_boundary_pt = False
            for neigh_id in range(1, 7):
                grid_neighs_entity_ids[neigh_id] = (
                    index_discrete_epsilon_map_1d[
                        grid_index_1d_times_3 + grid_neighs_1d_offsets[neigh_id]
                    ]
                    % epsilon_dimension
                )
                grid_neighs_media_ids[neigh_id] = (
                    index_discrete_epsilon_map_1d[
                        grid_index_1d_times_3 + grid_neighs_1d_offsets[neigh_id]
                    ]
                    // epsilon_dimension
                )

            rm_boundary_pt = (
                (1 < grid_neighs_entity_ids[1] <= num_atoms + 1)
                or (1 < grid_neighs_entity_ids[2] <= num_atoms + 1)
                or (
                    (1 < grid_neighs_entity_ids[3] <= num_atoms + 1)
                    or (1 < grid_neighs_entity_ids[4] <= num_atoms + 1)
                )
                or (
                    (1 < grid_neighs_entity_ids[5] <= num_atoms + 1)
                    or (1 < grid_neighs_entity_ids[6] <= num_atoms + 1)
                )
                or rm_boundary_pt
            )

            rm_boundary_pt = rm_boundary_pt and rm_boundary_pt_condition

            (
                neigh_update_exec_status,
                grid_neighs_entity_ids,
                grid_neighs_media_ids,
                num_cavity_points,
                dielectric_boundary_midpoints_1d,
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
                neighbor_boundary_array=neighbor_bounday_array,
                num_atoms=num_atoms,
                atoms_data=atoms_data,
                atom_plus_probe_radii_1d=atom_plus_probe_radii_1d,
                atom_plus_probe_radii_shrink_1d=atom_plus_probe_radii_shrink_1d,
                epsilon_dimension=epsilon_dimension,
                boundary_point_end_index=boundary_point_end_index,
                max_boundary_grid_points=max_boundary_grid_points,
                dxyz=dxyz,
                num_cavity_points=num_cavity_points,
                dielectric_boundary_midpoints_1d=dielectric_boundary_midpoints_1d,
                num_external_boundary_points=num_external_boundary_points,
                boundary_grid_indices=boundary_grid_indices,
                added_boundary_points_increment=added_boundary_points_increment,
                removed_boundary_points_increment=removed_boundary_points_increment,
                dtype_int=dtype_int,
                dtype_real=dtype_real,
                dtype_bool=dtype_bool,
            )

            if neigh_update_exec_status != EXIT_NJIT_FLAG:
                # remap discrete_epsilon_map_1d in case there have been changes..
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

                # NOTE: this is final neighbor update and boundary point marking, here
                # Neighbors are not tested rather the point (ix, iy, iz) itself is processed.
                (
                    neigh_update_exec_status,
                    dielectric_boundary_midpoints_1d,
                    num_external_boundary_points,
                    removed_boundary_points_increment,
                ) = _update_gridpoint_boundary_status(
                    grid_index_x,
                    grid_index_y,
                    grid_index_z,
                    epsilon_dimension,
                    grid_shape,
                    index_discrete_epsilon_map_1d,
                    dielectric_boundary_midpoints_1d,
                    num_external_boundary_points,
                    removed_boundary_points_increment,
                    dtype_int=dtype_int,
                )
    return (
        neigh_update_exec_status,
        cycle_flag,
        added_boundary_points_increment,
        removed_boundary_points_increment,
        num_external_boundary_points,
        boundary_grid_indices,
    )


def _setup_cube_for_indexing(
    grid_scale: float,
    max_probe_radius: float,
    max_atom_radius: float,
    min_coords_by_axis: np.ndarray,
    max_coords_by_axis: np.ndarray,
    dtype_real,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Sets up a cubic grid for efficient indexing of atoms and objects (standalone version).

    Args:
        grid_scale (float): Number of grids per angstrom.
        max_probe_radius (float): The maximum radius of the probe molecule.
        max_atom_radius (float): The maximum radius of any atom in the system.
        min_coords_by_axis (np.ndarray): Minimum coordinates of the system along each axis (x, y, z).
        max_coords_by_axis (np.ndarray): Maximum coordinates of the system along each axis (x, y, z).
        dtype_real (type): Real data type.

    Returns:
        tuple[np.ndarray, np.ndarray, float, float]: A tuple containing:
               - voxel_space_origin (np.ndarray): The coordinates of the lowest vertex of the cube.
               - voxel_space_shape (np.ndarray): The dimensions (shape) of the cube grid.
               - voxel_side_length (float): The length of each side of the cube.
               - inverse_cube_side_length (float): The inverse of the cube side length.
    """
    delta = dtype_real(1.0 / grid_scale)
    delta = max(delta, max_probe_radius)
    voxel_side_length = dtype_real(max_atom_radius + delta)
    inverse_cube_side_length = dtype_real(1.0 / voxel_side_length)
    min_coords_array = np.asarray(min_coords_by_axis, dtype=dtype_real)
    max_coords_array = np.asarray(max_coords_by_axis, dtype=dtype_real)

    voxel_space_origin, voxel_space_shape = voxelizer.calculate_voxel_space_parameters(
        dtype_real(voxel_side_length),
        min_coords_array,
        max_coords_array,
        scaling_factor=dtype_real(2.0),
        voxel_space_offset=dtype_real(0.1),
    )
    return (
        voxel_space_origin,
        voxel_space_shape,
        voxel_side_length,
        inverse_cube_side_length,
    )


def _handle_zeta_surface(
    use_zeta_surface: bool,
    grid_spacing: float,
    grid_shape: tuple,
    gridbox_center: np.ndarray,
    mid_grid_indices: np.ndarray,
    zeta_surface_map_1d: np.ndarray,
    epsilon_dimension: int,
    index_discrete_epsilon_map_1d: np.ndarray,
    dtype_int,
    dtype_real,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """
    Builds the zeta surface map if use_zeta_surface is True (standalone version).

    Args:
        use_zeta_surface (bool): Flag indicating whether to build the zeta surface map.
        grid_spacing (float): The spacing between grid points.
        grid_shape (tuple): The dimensions of the grid (nx, ny, nz).
        gridbox_center (np.ndarray): The coordinates of the center of the grid.
        mid_grid_indices (np.ndarray): The indices of the middle grid point in each dimension.
        zeta_surface_map_1d (np.ndarray): 1D array representing the zeta surface.
        epsilon_dimension (int): The number of discrete dielectric constants.
        epsilon_map_1d (np.ndarray): 1D array representing the discrete dielectric constant map.
        dtype_int (type): Integer data type. Defaults to int.
        dtype_real (type): Real data type. Defaults to float.

    Returns:
        tuple[np.ndarray, np.ndarray, int, int, int, int]: A tuple containing the updated zeta surface grid coordinates, indices,
               number of points, and capacities.
    """
    zeta_surface_grid_coords = np.array(3, dtype=dtype_real)
    zeta_surface_grid_indices = np.array(3, dtype=dtype_int)
    num_zeta_point_coords = dtype_int(0)
    num_zeta_point_indices = dtype_int(0)
    zeta_coords_capacity = dtype_int(0)
    zeta_indices_capacity = dtype_int(0)

    if use_zeta_surface:
        max_possible_points = dtype_int(grid_shape[0] * grid_shape[1] * grid_shape[2])
        initial_size_coords = dtype_int(max_possible_points * INITIAL_SIZE_PERCENT) * 3
        initial_size_indices = dtype_int(max_possible_points * INITIAL_SIZE_PERCENT) * 3

        current_zeta_surf_grid_coords = np.zeros(initial_size_coords, dtype=dtype_real)
        current_zeta_surf_grid_indices = np.zeros(initial_size_indices, dtype=dtype_int)
        current_num_zeta_point_coords = dtype_int(0)
        current_num_zeta_point_indices = dtype_int(0)
        current_zeta_coords_capacity = dtype_int(initial_size_coords)
        current_zeta_indices_capacity = dtype_int(initial_size_indices)

        (
            zeta_surface_grid_coords,
            zeta_surface_grid_indices,
            num_zeta_point_coords,
            num_zeta_point_indices,
            zeta_coords_capacity,
            zeta_indices_capacity,
        ) = helpers.build_zeta_surface_map(
            grid_spacing=grid_spacing,
            grid_shape=grid_shape,
            gridbox_center=gridbox_center,
            indices_mid_grid=mid_grid_indices,
            zeta_surface_map_1d=zeta_surface_map_1d,
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
            epsdim=epsilon_dimension,
            zeta_surf_grid_coords=current_zeta_surf_grid_coords,
            zeta_surf_grid_indices=current_zeta_surf_grid_indices,
            zeta_coords_capacity=current_zeta_coords_capacity,
            zeta_indices_capacity=current_zeta_indices_capacity,
            num_zeta_surf_grid_coords=current_num_zeta_point_coords,
            num_zeta_surf_grid_indices=current_num_zeta_point_indices,
        )
    return (
        zeta_surface_grid_coords,
        zeta_surface_grid_indices,
        num_zeta_point_coords,
        num_zeta_point_indices,
        zeta_coords_capacity,
        zeta_indices_capacity,
    )


def _check_boundary_point_limit(
    num_boundary_grid_points: int,
    max_boundary_grid_points: int,
):
    """
    Checks if the number of boundary grid points exceeds the allowed maximum (standalone version).

    Args:
        num_boundary_grid_points (int): The current number of boundary grid points.
        max_boundary_grid_points (int): The maximum allowed number of boundary grid points.
    """
    if num_boundary_grid_points > max_boundary_grid_points:
        vprint(
            CRITICAL,
            _VERBOSITY,
            " WARNING iBoundNum= ",
            num_boundary_grid_points,
            " is greater than ibmx = ",
            max_boundary_grid_points,
        )
        vprint(CRITICAL, _VERBOSITY, " CRITICAL> Increase `max_boundary_grid_points`")
        exit(0)


def _print_dielectric_boundary(
    dielectric_boundary_midpoints_1d: np.ndarray,
    grid_shape: np.ndarray,
):
    """
    Prints the dielectric boundary information if the verbosity level is high enough (standalone version).

    Args:
        dielectric_boundary_midpoints_1d (np.ndarray): 1D array containing information about the dielectric boundary.
        grid_shape (np.ndarray): The dimensions of the grid (nx, ny, nz).
    """
    if _VERBOSITY <= TRACE:
        helpers.print_4d_array(
            "self.dielectric_boundary_midpoints_1d bndeps: (ijk1d_x_3, iix, iiy, iiz, bgp_id, iext)",
            dielectric_boundary_midpoints_1d,
            list(grid_shape) + [3],
        )


class SurfaceMolecularVdW:
    """
    A class responsible for creating the Van der Waals molecular surface
    representation used in DelPhi calculations.

    This class manages the grid, atom, and object data, and orchestrates the
    process of identifying boundary points and determining the molecular surface
    based on probe radius and atom/object radii.
    """

    def __init__(
        self,
        platform,
        grid_spacing,
        probe_radius,
        probe_radius_second,
        debye_length,
        salt_radius,
        radius_offset,
        max_radius,
        max_atom_radius,
        surface_offset,
        grid_shape,
        grid_origin,
        grid_scale,
        min_coords_by_axis,
        max_coords_by_axis,
        grid_shape_parentrun,
        grid_origin_parentrun,
        atoms_data,
        atom_index_array,
        objects_data,
        n_objects,
        n_molecules,
        surface_charge_positions,
        is_focusing,
        use_zeta_surface_calculation,
        index_discrete_epsilon_map_1d,
        dielectric_boundary_map_1d,
        zeta_surface_map_1d,
        verbosity,
        approx_zero,
    ):
        """
        Initializes the SurfaceMolecularVdW object.

        Args:
            platform (str): The computational platform (e.g., 'cpu', 'cuda').
            grid_spacing (float): The spacing between grid points in Angstroms.
            probe_radius (float): The radius of the probe molecule in Angstroms.
            probe_radius_second (float): A second probe radius for specific cases.
            debye_length (float): The Debye length of the solution.
            salt_radius (float): The radius of the salt ions.
            radius_offset (float): An offset applied to atomic radii.
            max_radius (float): The maximum radius considered in the system.
            max_atom_radius (float): The maximum radius of any atom in the system.
            surface_offset (float): An offset for surface calculations.
            grid_shape (tuple): A tuple (nx, ny, nz) representing the dimensions of the grid.
            grid_origin (np.ndarray): A NumPy array [x, y, z] representing the origin of the grid.
            grid_scale (float): The number of grid points per Angstrom.
            min_coords_by_axis (np.ndarray): Minimum coordinates of the system along each axis.
            max_coords_by_axis (np.ndarray): Maximum coordinates of the system along each axis.
            grid_shape_parentrun (tuple): Grid shape of the parent run (for focusing).
            grid_origin_parentrun (np.ndarray): Grid origin of the parent run (for focusing).
            atoms_data (np.ndarray): A NumPy array containing atomic data (coordinates, radii, etc.).
            atom_index_array (np.ndarray): An array mapping atom indices.
            objects_data (np.ndarray): A NumPy array containing data for geometric objects.
            n_objects (int): The number of geometric objects.
            n_molecules (int): The number of molecules in the system.
            surface_charge_positions (np.ndarray): Positions of surface charges.
            is_focusing (bool): A flag indicating if this is a focusing run.
            use_zeta_surface_calculation (bool): A flag to use zeta surface calculation.
            index_discrete_epsilon_map_1d (np.ndarray): 1D array of the discrete epsilon map.
            dielectric_boundary_map_1d (np.ndarray): 1D array of the dielectric boundary map at grid-points.
            zeta_surface_map_1d (np.ndarray): 1D array of the zeta surface map.
            verbosity (int): The verbosity level for output.
            approx_zero (float): A small value considered as zero.
        """
        # Lazy global precision dependent module imports
        configure_precision_dependent_imports()
        # set members
        self.num_cuda_threads = 256
        self.surface_method = None
        self.calculation_platform = platform
        self.grid_spacing = grid_spacing
        self.probe_radius = probe_radius
        self.probe_radius_second = probe_radius_second
        self.debye_length = debye_length
        self.salt_radius = salt_radius
        self.radius_offset = radius_offset
        self.surface_offset = surface_offset
        self.approx_zero = approx_zero
        self.grid_shape = grid_shape
        self.grid_origin = grid_origin
        self.grid_scale = grid_scale
        self.min_coords_by_axis = min_coords_by_axis
        self.max_coords_by_axis = max_coords_by_axis
        self.grid_shape_parentrun = grid_shape_parentrun
        self.grid_origin_parentrun = grid_origin_parentrun
        self.atoms_data = atoms_data
        self.atom_index_array = atom_index_array
        self.objects_data = objects_data
        self.maximum_radius = max_radius
        self.max_atom_radius = max_atom_radius
        self.num_molecules = n_molecules
        # positions of induced surface charges
        self.surface_charge_positions = surface_charge_positions
        self.use_zeta_surface_calculation = use_zeta_surface_calculation
        self.zeta_surface_grid_coords = None
        self.zeta_surface_grid_indices = None
        self.discrete_epsilon_index_map_1d = index_discrete_epsilon_map_1d
        self.dielectric_boundary_map_1d = dielectric_boundary_map_1d
        self.zeta_surface_map_1d = zeta_surface_map_1d
        self.surface_map_midpoints_1d = None
        self.is_focusing = is_focusing
        self.verbosity = verbosity
        # local fields initialized here
        self.num_grid_points = 0
        self.num_exposed_grids = 0
        self.num_atoms = len(self.atoms_data) if self.atoms_data is not None else 0
        self.num_objects = n_objects
        self.cube_side_length_inverse = None
        self.voxel_space_origin = np.zeros(3, dtype=delphi_int)
        self.cube_vertex_highest_xyz = np.zeros(3, dtype=delphi_int)
        self.voxel_space_shape = np.zeros(3, dtype=delphi_int)
        self.num_boundary_grid_points = None
        self.num_external_boundary_points = None
        self.epsilon_dimension = self.num_atoms + self.num_objects + 2

        # Boundary points related info
        self.max_boundary_grid_points = None
        self.boundary_grid_indices = None
        self.boundary_grid_points = None
        self.dielectric_boundary_midpoints_1d = None
        self.point_indices_by_voxel = None

        self.atom_plus_probe_radii_1d = None
        self.atom_plus_probe_radii_square_1d = None
        self.atom_plus_probe_radii_square_shrunk_1d = None

        # Zeta surface fields
        self.zeta_surface_grid_coords = None
        self.zeta_surface_grid_indices = None
        self.num_zeta_surface_point_coords = 0
        self.num_zeta_surface_point_indices = 0
        self.zeta_surface_coords_capacity = 0
        self.zeta_surface_indices_capacity = 0

        self.sLimObject = None
        self.voxel_atom_count = np.zeros(1, dtype=delphi_int)
        self.voxel_atom_count_cumulative = np.zeros(1, dtype=delphi_int)
        self.atom_accessibility = None
        self.limeps_min = np.zeros(3, dtype=delphi_int)
        self.limeps_max = np.zeros(3, dtype=delphi_int)
        self.exposed_grids_coords = np.zeros((0, 3), dtype=delphi_real)
        self.zeta_surfmap = None
        self.boundary_grid_points = np.zeros((0, 3), dtype=delphi_int)
        self.dxyz = np.zeros((0, 3), dtype=delphi_real)
        # Atoms-neighbor lookup utility fields
        self.indexing_voxel_scale = None
        self.indexing_voxel_origin = None
        self.indexing_voxel_shape = None
        self.voxel_point_start_indices = np.zeros(1, dtype=delphi_int)
        self.voxel_point_end_indices = np.zeros(1, dtype=delphi_int)
        self.neighbor_boundary_array = None
        self.scaled_surface_normal_vectors = None

    def _initialize_arrays(self, num_atoms):
        """Initializes NumPy arrays used in the VdwToMs method."""
        neighbor_boundary_array = np.zeros(
            MAX_NEIGHBR_BOUNDARY_ARRAY_LEN, dtype=delphi_int
        )
        index_map = np.zeros((5, 7), dtype=delphi_int)  # Initialize index_map
        neighbor_exists_array = np.zeros(
            7, dtype=np.bool_
        )  # Initialize neighbor_exists_array
        neighbor_grids_offset = np.zeros(
            (7, 3), dtype=delphi_real
        )  # Initialize neighbor_grids_offset

        return (
            neighbor_boundary_array,
            index_map,
            neighbor_grids_offset,
            neighbor_exists_array,
        )

    def create_vdw_molecular_surfaces(
        self,
        use_zeta_surface=True,
        solve_pbe=True,
        read_rxn_from_frc=True,
        calc_solvation_energy=True,
        calc_nonlinear_energy=False,
        calc_surface_energy=False,
        calc_surface_charge=False,
        only_molecule=True,
        profile_timings=False,
    ):
        """
        Creates the Van der Waals molecular surface.

        This method orchestrates the sequence of steps required to generate the
        molecular surface, including finding boundary grid points, handling the
        zeta surface (if enabled), and elaborating the boundary grid points.

        Args:
            use_zeta_surface (bool): Whether to use zeta surface calculation. Defaults to True.
            solve_pbe (bool): Flag for solvation energy calculation. Defaults to True.
            read_rxn_from_frc (bool): Flag related to reaction field energy. Defaults to True.
            calc_solvation_energy (bool): Flag for generating log information. Defaults to True.
            calc_nonlinear_energy (bool): Flag for non-linear Poisson-Boltzmann log. Defaults to False.
            calc_surface_energy (bool): Flag related to entropy calculation. Defaults to False.
            calc_surface_charge (bool): Flag related to enthalpy calculation. Defaults to False.
            only_molecule (bool): Flag to consider only the molecule. Defaults to True.
            profile_timings (bool): Flag to enable time-profiling of key steps. Defaults to False.

        Returns:
            int: 0 on success, or an error code (EXIT_NJIT_FLAG) on failure.
        """

        if profile_timings:
            tic_vdw_setup = time.perf_counter()
        # Initialization
        (
            self.neighbor_boundary_array,
            index_map,
            grid_neighbor_coords_offsets,
            neighbor_exists_array,
        ) = self._initialize_arrays(self.num_atoms)

        # Calculate grid properties
        (
            grid_spacing_half,
            mid_grid_point_indices,
            gridbox_center,
            n_grid_points,
            n_grid_points_x_3,
        ) = _calculate_grid_properties(
            self.grid_spacing,
            self.grid_shape,
            self.grid_origin,
            dtype_int=delphi_int,
            dtype_real=delphi_real,
        )
        grid_spacing = self.grid_spacing
        self.num_grid_points = n_grid_points
        self.num_grid_points_x_3 = n_grid_points_x_3

        cycle_flag = False

        # Set constant values
        index_map, neighbor_exists_array = _set_constant_values(
            dtype_int=delphi_int, dtype_bool=delphi_bool
        )

        # Set neighbor offsets
        grid_neighbor_coords_offsets = _setup_grid_neighbor_coords_offsets(
            grid_spacing_half=grid_spacing_half, dtype_real=delphi_real
        )

        # Determine maximum probe radius
        max_probe_radius = max(self.probe_radius, self.probe_radius_second)

        # Determine grid origin, grid_origin already accounts for parent or focusing. NO update is needed.
        grid_origin_current = (
            self.grid_origin if self.is_focusing else self.grid_origin
        )  # - 0.5 * self.grid_spacing

        # Determine exclusion radius
        exclusion_radius = (
            max(self.max_atom_radius, self.salt_radius)
            if self.debye_length
            != delphi_real(ConstDelPhi.ZeroMolarSaltDebyeLength.value)
            else self.max_atom_radius
        )

        # Calculate grid boundaries
        (
            min_solute_grid_index,
            max_solute_grid_index,
        ) = _calculate_solute_grid_boundaries(
            max_atom_radius=self.max_atom_radius,
            grid_spacing=self.grid_spacing,
            grid_shape=self.grid_shape,
            grid_origin=self.grid_origin,
            coords_by_axis_min=self.min_coords_by_axis,
            coords_by_axis_max=self.max_coords_by_axis,
            dtype_int=delphi_int,
        )

        # Calculate strides
        (
            x_stride,
            y_stride,
            z_stride,
            x_stride_x_3,
            y_stride_x_3,
            z_stride_x_3,
        ) = _calculate_strides(self.grid_shape, dtype_int=delphi_int)

        # Initialize boundary grid points arrays if not already initialized
        if self.max_boundary_grid_points is None:
            self.max_boundary_grid_points = 1000000
            if size_cpu.or_gt_scalar(
                self.grid_shape.astype(delphi_int), delphi_int(300)
            ):
                self.max_boundary_grid_points = 50000000
        if self.boundary_grid_indices is None:
            self.boundary_grid_indices = np.zeros(
                (self.max_boundary_grid_points, 3), dtype=delphi_int
            )
        if self.dielectric_boundary_midpoints_1d is None:
            self.dielectric_boundary_midpoints_1d = np.zeros(
                self.num_grid_points_x_3, dtype=delphi_int
            )

        vprint(DEBUG, _VERBOSITY, " Info> Drawing MS from vdW surface")
        iarv = 0
        if profile_timings:
            toc_vdw_setup = time.perf_counter()
            print(f"vdw to ms setup: {(toc_vdw_setup - tic_vdw_setup):.3f}s")
        # Find initial set of boundary grid points based on dielectric constant map.
        (
            num_boundary_grid_points_found,
            num_external_boundary_points,
            self.dielectric_boundary_midpoints_1d,
            self.boundary_grid_indices,
        ) = _find_boundary_grid_points(
            epsilon_dimension=self.epsilon_dimension,
            max_boundary_grid_points=self.max_boundary_grid_points,
            grid_shape=self.grid_shape,
            min_solute_grid_index=min_solute_grid_index,
            max_solute_grid_index=max_solute_grid_index,
            index_discrete_epsilon_map_1d=self.discrete_epsilon_index_map_1d,
            dielectric_boundary_midpoints_1d=self.dielectric_boundary_midpoints_1d,
            boundary_grid_indices=self.boundary_grid_indices,
            dtype_int=delphi_int,
        )
        if profile_timings:
            toc_vdw_init_bgp = time.perf_counter()
            print(
                f"vdw to ms find initial boundary grid points: {(toc_vdw_init_bgp - toc_vdw_setup):.3f}s"
            )

        # Handle zeta surface calculation if enabled
        (
            self.zeta_surface_grid_coords,
            self.zeta_surface_grid_indices,
            self.num_zeta_surface_point_coords,
            self.num_zeta_surface_point_indices,
            self.zeta_surface_coords_capacity,
            self.zeta_surface_indices_capacity,
        ) = _handle_zeta_surface(
            use_zeta_surface=use_zeta_surface,
            grid_spacing=self.grid_spacing,
            grid_shape=self.grid_shape,
            gridbox_center=gridbox_center,
            mid_grid_indices=mid_grid_point_indices,
            zeta_surface_map_1d=self.zeta_surface_map_1d,
            epsilon_dimension=self.epsilon_dimension,
            index_discrete_epsilon_map_1d=self.discrete_epsilon_index_map_1d,
            dtype_int=delphi_int,
            dtype_real=delphi_real,
        )

        self.num_boundary_grid_points = num_boundary_grid_points_found
        self.num_external_boundary_points = num_external_boundary_points

        vprint(
            DEBUG,
            _VERBOSITY,
            " VdMS> Boundary points facing continuum solvent= ",
            self.num_external_boundary_points,
        )
        vprint(
            DEBUG,
            _VERBOSITY,
            " VdMS> Total number of boundary points before elab.= ",
            self.num_boundary_grid_points,
        )

        # Check if the number of boundary grid points exceeds the limit.
        _check_boundary_point_limit(
            self.num_boundary_grid_points, self.max_boundary_grid_points
        )

        # Print the dielectric boundary information if verbosity is high enough.
        _print_dielectric_boundary(
            self.dielectric_boundary_midpoints_1d, self.grid_shape
        )
        if profile_timings:
            toc_vdw_zeta = time.perf_counter()
            print(
                f"vdw to ms _handle_zeta_surface: {(toc_vdw_zeta - toc_vdw_init_bgp):.3f}s"
            )

        # Handle the case where the probe radius is zero.
        self.boundary_grid_points = _handle_zero_probe_radius(
            max_probe_radius,
            self.num_boundary_grid_points,
            self.boundary_grid_indices,
            dtype_int=delphi_int,
            dtype_real=delphi_real,
        )
        if max_probe_radius > APPROX_ZERO and self.boundary_grid_points.shape[0] == 0:
            # Proceed if probe radius is non-zero
            if profile_timings:
                toc_vdw_zero_probe = time.perf_counter()
                print(
                    f"vdw to ms _handle_zero_probe_radius: {(toc_vdw_zero_probe - toc_vdw_zeta):.3f}s"
                )
            if profile_timings:
                toc_vdw_zero_probe = time.perf_counter()
            # Calculate atom plus probe radii and their squares (for contact detection).
            (
                self.atom_plus_probe_radii_1d,
                self.atom_plus_probe_radii_square_1d,
                self.atom_plus_probe_radii_square_shrunk_1d,
            ) = _calculate_atom_probe_radii(
                self.probe_radius,
                RADII_SQUARED_SHRINK_FACTOR,
                self.num_atoms,
                self.atoms_data,
                dtype_real=delphi_real,
            )

            probe_radius_squared_1 = self.probe_radius * self.probe_radius
            probe_radius_squared_2 = self.probe_radius_second * self.probe_radius_second

            # Calculate root mean square of atomic positions (used in some surface calculations).
            rms = _calculate_rms(
                only_molecule, self.num_atoms, self.atoms_data, dtype_real=delphi_real
            )

            surface_generation_flag = True
            if profile_timings:
                toc_vdw_radii_acc = time.perf_counter()
                print(
                    f"vdw to ms _calculate_atom_probe_radii: {(toc_vdw_radii_acc - toc_vdw_zero_probe):.3f}s"
                )
            # Calculate solvent accessible surface area (SAS) related information.
            (
                self.num_exposed_grids,
                self.exposed_grids_coords,
                self.atom_accessibility,
                self.voxel_atom_count,
                self.voxel_atom_count_cumulative,
                self.voxel_space_origin,
                self.voxel_space_shape,
            ) = sas.solvent_accessible_surface(
                probe_radius=self.probe_radius,
                probe_radius2=self.probe_radius_second,
                max_atom_radius=self.max_atom_radius,
                min_coords_by_axis=self.min_coords_by_axis,
                max_coords_by_axis=self.max_coords_by_axis,
                num_atoms=self.num_atoms,
                num_objects=self.num_objects,
                num_molecules=self.num_molecules,
                atoms_data=self.atoms_data,
                atom_plus_probe_radii_1d=self.atom_plus_probe_radii_1d,
                atom_plus_probe_radii_shrink_1d=self.atom_plus_probe_radii_square_shrunk_1d,
                num_vertices=520,
                num_edges=1040,
            )
            if profile_timings:
                toc_vdw_sas = time.perf_counter()
                print(
                    f"vdw to ms solvent_accessible_surface: {(toc_vdw_sas - toc_vdw_radii_acc):.3f}s"
                )

            # Setup a cubic grid for efficient indexing of vertices.
            (
                cube_vertex_lowest_xyz,
                cube_shape,
                cube_side_length,
                self.cube_side_length_inverse,
            ) = _setup_cube_for_indexing(
                grid_scale=self.grid_scale,
                max_probe_radius=max_probe_radius,
                max_atom_radius=self.max_atom_radius,
                min_coords_by_axis=self.min_coords_by_axis,
                max_coords_by_axis=self.max_coords_by_axis,
                dtype_real=delphi_real,
            )
            self.voxel_space_origin = cube_vertex_lowest_xyz
            self.voxel_space_shape = cube_shape

            # Calculate the number of voxels per entity (atom, object, molecule).
            num_voxel_per_entity = _calculate_cube_voxels_per_entity(
                num_objects=self.num_objects,
                num_molecules=self.num_molecules,
                cube_shape=self.voxel_space_shape,
                dtype_int=delphi_int,
            )
            if profile_timings:
                toc_vdw_cube_setup = time.perf_counter()
                print(
                    f"vdw to ms _setup_cube_for_indexing: {(toc_vdw_cube_setup - toc_vdw_sas):.3f}s"
                )

            # Perform the cube calculation to assign atoms to voxels.
            (
                voxel_atom_ids,
                voxel_atom_count,
                voxel_atom_count_cumulative,
            ) = _perform_cube_calculation(
                self.num_atoms,
                self.num_objects,
                self.num_molecules,
                num_voxel_per_entity,
                cube_side_length,
                self.voxel_space_origin,
                self.voxel_space_shape,
                self.atoms_data,
                delphi_int,
                delphi_real,
            )
            self.voxel_atom_ids = voxel_atom_ids
            self.voxel_atom_count = voxel_atom_count
            self.voxel_atom_count_cumulative = voxel_atom_count_cumulative
            if profile_timings:
                toc_vdw_cube_calc = time.perf_counter()
                print(
                    f"vdw to ms _perform_cube_calculation: {(toc_vdw_cube_calc - toc_vdw_cube_setup):.3f}s"
                )
            voxel_space_boundary_extension = self.max_atom_radius + self.probe_radius
            # voxels_per_dim must be small fraction of grids, we use maximum of 50 and 20% of largest grid_shape
            # to balance the voxelation size and very large-voxel to avoid very large neighborhood search.
            max_voxels_per_dimension_value = int(
                max(50, int(max(self.grid_shape) * 0.25))
            )
            # Setup vertex indexing data structures.
            (
                indexing_voxel_side,
                self.indexing_voxel_origin,
                self.indexing_voxel_shape,
            ) = voxelizer.calculate_indexing_voxel_parameters(
                self.grid_spacing,
                self.probe_radius,
                voxel_space_boundary_extension,
                self.min_coords_by_axis,
                self.max_coords_by_axis,
                max_voxels_per_dimension=max_voxels_per_dimension_value,
            )

            self.indexing_voxel_scale = 1.0 / indexing_voxel_side
            self.point_indices_by_voxel = np.zeros(
                self.num_exposed_grids + 1, dtype=delphi_int
            )

            vprint(
                DEBUG,
                _VERBOSITY,
                " VdMS> grid for indexing accessible points =  ",
                indexing_voxel_side,
            )
            if profile_timings:
                toc_vdw_index_setup = time.perf_counter()
                print(
                    f"vdw to ms setup_index_vertices: {(toc_vdw_index_setup - toc_vdw_cube_calc):.3f}s"
                )

            # Perform vertex indexing to find which atoms are accessible to the solvent
            (
                self.voxel_point_start_indices,
                self.voxel_point_end_indices,
                self.point_indices_by_voxel,
            ) = voxelizer.build_point_voxel_index_map(
                self.num_exposed_grids,
                self.indexing_voxel_scale,
                self.indexing_voxel_shape,
                self.indexing_voxel_origin,
                self.exposed_grids_coords,
                self.point_indices_by_voxel,
            )
            if profile_timings:
                toc_vdw_index_calc = time.perf_counter()
                print(
                    f"vdw to ms index_vertices: {(toc_vdw_index_calc - toc_vdw_index_setup):.3f}s"
                )

            rm_boundary_pt_condition = self.num_molecules > 0
            num_cavity_midpoints = 0
            boundary_point_start_index = 1
            boundary_point_end_index = self.num_boundary_grid_points

            max_points_added_in_iteration = 100000
            no_divergence_count = 0

            # Iteratively process boundary grid points to refine the molecular surface
            while True:
                if profile_timings:
                    tic_vdw_proc_bgp = time.perf_counter()
                (
                    neigh_update_exec_status,
                    cycle_flag,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                    num_external_boundary_points,
                    self.boundary_grid_indices,
                ) = _process_boundary_grid_points_loop(
                    probe_radius_squared_1=probe_radius_squared_1,
                    probe_radius_squared_2=probe_radius_squared_2,
                    boundary_point_start_index=boundary_point_start_index,
                    boundary_point_end_index=boundary_point_end_index,
                    x_stride=x_stride,
                    y_stride=y_stride,
                    z_stride=z_stride,
                    grid_origin_current=grid_origin_current,
                    max_boundary_grid_points=self.max_boundary_grid_points,
                    grid_neighbor_coords_offsets=grid_neighbor_coords_offsets,
                    boundary_grid_indices=self.boundary_grid_indices,
                    dielectric_boundary_midpoints_1d=self.dielectric_boundary_midpoints_1d,
                    grid_spacing=grid_spacing,
                    grid_shape=self.grid_shape,
                    exposed_grids_coords=self.exposed_grids_coords,
                    epsilon_dimension=self.epsilon_dimension,
                    index_discrete_epsilon_map_1d=self.discrete_epsilon_index_map_1d,
                    index_map=index_map,
                    rm_boundary_pt_condition=rm_boundary_pt_condition,
                    min_xyz=self.indexing_voxel_origin,
                    cube_side_indver_inverse=self.indexing_voxel_scale,
                    cube_shape_indver=self.indexing_voxel_shape,
                    cube_voxel_start_indices=self.voxel_point_start_indices,
                    cube_voxel_end_indices=self.voxel_point_end_indices,
                    cube_vertex_lowest_xyz=self.voxel_space_origin,
                    cube_side_length_inverse=self.cube_side_length_inverse,
                    cube_shape=self.voxel_space_shape,
                    voxel_grid_point_indices=self.point_indices_by_voxel,
                    voxel_atom_count=self.voxel_atom_count,
                    voxel_atom_count_cumulative=self.voxel_atom_count_cumulative,
                    voxel_atom_ids=self.voxel_atom_ids,
                    atom_surface_flags=self.atom_accessibility,
                    neighbor_bounday_array=self.neighbor_boundary_array,
                    num_atoms=self.num_atoms,
                    atoms_data=self.atoms_data,
                    atom_plus_probe_radii_1d=self.atom_plus_probe_radii_1d,
                    atom_plus_probe_radii_shrink_1d=self.atom_plus_probe_radii_square_shrunk_1d,
                    num_external_boundary_points=num_external_boundary_points,
                    dxyz=self.dxyz,
                    dtype_int=delphi_int,
                    dtype_real=delphi_real,
                    dtype_bool=delphi_bool,
                )
                if neigh_update_exec_status == EXIT_NJIT_FLAG:
                    exit(0)

                boundary_point_start_index = boundary_point_end_index + 1
                boundary_point_end_index += added_boundary_points_increment

                vprint(
                    DEBUG,
                    _VERBOSITY,
                    " VdMS> bgp added m=",
                    added_boundary_points_increment,
                    " bgp removed mr =",
                    removed_boundary_points_increment,
                )

                # Check for convergence of the surface iteration.
                if added_boundary_points_increment > max_points_added_in_iteration:
                    no_divergence_count += 1
                    if no_divergence_count > 50:
                        vprint(
                            ERROR,
                            _VERBOSITY,
                            " ERROR> Surface iteration did not converge",
                        )
                        exit(0)
                else:
                    no_divergence_count = 0
                if added_boundary_points_increment <= 0:
                    break
                if profile_timings:
                    toc_vdw_proc_bgp = time.perf_counter()
                    print(
                        f"vdw to ms _process_boundary_grid_points_loop: {(toc_vdw_proc_bgp - tic_vdw_proc_bgp):.3f}s"
                    )

            self.num_external_boundary_points = num_external_boundary_points

            if boundary_point_end_index > self.max_boundary_grid_points:
                vprint(
                    ERROR,
                    _VERBOSITY,
                    " Error> ibnd upper bound ",
                    boundary_point_end_index,
                    " exceeds ibmx",
                )
                exit(0)

            vprint(
                DEBUG,
                _VERBOSITY,
                " VdMS> Number of cavity mid-points inaccessible to solvent = ",
                num_cavity_midpoints,
            )
            if profile_timings:
                tic_vdw_elab_bgp = time.perf_counter()
            # Elaborate the boundary grid points to finalize the surface.
            (
                self.num_boundary_grid_points,
                self.boundary_grid_points,
            ) = helpers.surface_elaborate_boundary_gridpoints(
                num_boundary_grid_indices=boundary_point_end_index,
                epsilon_dimension=self.epsilon_dimension,
                max_boundary_grid_points=self.max_boundary_grid_points,
                grid_shape=self.grid_shape,
                boundary_grid_points=self.boundary_grid_points,
                boundary_grid_indices=self.boundary_grid_indices,
                dielectric_boundary_midpoints_1d=self.dielectric_boundary_midpoints_1d,
                index_discrete_epsilon_map_1d=self.discrete_epsilon_index_map_1d,
                index_map=index_map,
            )
            if profile_timings:
                toc_vdw_elab_bgp = time.perf_counter()
                print(
                    f"vdw to ms surface_elaborate_boundary_gridpoints: {(toc_vdw_elab_bgp - tic_vdw_elab_bgp):.3f}s"
                )
            if self.num_boundary_grid_points == EXIT_NJIT_FLAG:
                exit(0)

        # Scale boundary grid point positions relative to accessible data.
        if solve_pbe and (
            read_rxn_from_frc
            or calc_solvation_energy
            or calc_nonlinear_energy
            or calc_surface_energy
            or calc_surface_charge
        ):
            vprint(DEBUG, _VERBOSITY, " VdMS> Scaling boundary grid points ...")

            self.surface_charge_positions = np.zeros(
                (self.num_boundary_grid_points, 3), dtype=delphi_real
            )

            for boundary_point_index in range(self.num_boundary_grid_points):
                self.surface_charge_positions[boundary_point_index] = (
                    self.boundary_grid_points[boundary_point_index]
                ).astype(delphi_real)

            self.scaled_surface_normal_vectors = np.zeros(
                (self.num_boundary_grid_points, 3), dtype=delphi_real
            )
            self.atom_surface_index = np.zeros(
                self.num_boundary_grid_points, dtype=delphi_int
            )
            self.atom_index_array = np.zeros(
                self.num_boundary_grid_points, dtype=delphi_int
            )
            if _VERBOSITY <= TRACE:
                print("before scaling scspos: [")
                for surface_charge_index, surface_charge_position in enumerate(
                    self.surface_charge_positions
                ):
                    print(
                        f"({surface_charge_position[0]:.1f}, {surface_charge_position[1]:.1f}, {surface_charge_position[2]:.1f}), ",
                        end="",
                    )
                    if (surface_charge_index + 1) % 4 == 0:
                        print()
                print("]")
            if profile_timings:
                tic_vdw_scale_bgp = time.perf_counter()
                print(
                    f"vdw to ms surface_elaborate_boundary_gridpoints: {(toc_vdw_elab_bgp - tic_vdw_elab_bgp):.3f}s"
                )
            # Call the function to scale the Van der Waals surface boundary points.
            return_status, _ = sclbp.scale_vdw_surface_boundary_points(
                num_atoms=self.num_atoms,
                num_molecules=self.num_molecules,
                num_objects=self.num_objects,
                max_atom_radius=delphi_real(self.max_atom_radius),
                probe_radius=delphi_real(self.probe_radius),
                probe_radius_2=delphi_real(self.probe_radius_second),
                is_focusing_run=self.is_focusing,
                grid_spacing=delphi_real(self.grid_spacing),
                grid_dimensions=self.grid_shape.astype(delphi_int),
                grid_origin=self.grid_origin.astype(delphi_real),
                grid_origin_parentrun=self.grid_origin_parentrun.astype(delphi_real),
                atom_data=self.atoms_data,
                min_coords_by_axis=self.min_coords_by_axis,
                max_coords_by_axis=self.max_coords_by_axis,
                num_exposed_grid_points=self.num_exposed_grids,
                num_boundary_points=self.num_boundary_grid_points,
                num_external_boundary_points=self.num_external_boundary_points,
                surface_charge_positions=self.surface_charge_positions,
                discrete_epsilon_index_map_1d=self.discrete_epsilon_index_map_1d,
                neighboring_atom_indices=self.neighbor_boundary_array,
                scaled_surface_normal_vectors=self.scaled_surface_normal_vectors,
                exposed_grid_point_coords=self.exposed_grids_coords,
                atom_accessibility=self.atom_accessibility,
                atom_surface_index=self.atom_surface_index,
                atom_index_for_boundary=self.atom_index_array,
                atom_plus_probe_radii=self.atom_plus_probe_radii_1d,
                atom_plus_probe_radii_squared=self.atom_plus_probe_radii_square_1d,
                atom_plus_probe_radii_squared_shrunk=self.atom_plus_probe_radii_square_shrunk_1d,
                system_min_coords=self.indexing_voxel_origin,
                cube_side_indver_inverse=self.indexing_voxel_scale,
                cube_shape_indver=self.indexing_voxel_shape,
                cube_voxel_atom_index_start=self.voxel_point_start_indices,
                cube_voxel_atom_index_end=self.voxel_point_end_indices,
                cube_voxel_atom_index_cumulative=self.point_indices_by_voxel,
            )
            if profile_timings:
                toc_vdw_scale_bgp = time.perf_counter()
                print(
                    f"vdw to ms scale_vdw_surface_boundary_points: {(toc_vdw_scale_bgp - tic_vdw_scale_bgp):.3f}s"
                )
            if return_status == EXIT_NJIT_FLAG:
                return return_status

        # Print surface charge positions if verbosity is high enough
        if _VERBOSITY <= TRACE:
            print("scspos: [")
            for surface_charge_index, surface_charge_position in enumerate(
                self.surface_charge_positions
            ):
                print(
                    f"({surface_charge_position[0]:.4f}, {surface_charge_position[1]:.4f}, {surface_charge_position[2]:.4f}), ",
                    end="",
                )
                if (surface_charge_index + 1) % 4 == 0:
                    print()
            print("]")

        vprint(DEBUG, _VERBOSITY, " VdMS> MS creation done")
        return 0
