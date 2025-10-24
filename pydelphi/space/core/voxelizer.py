#!/usr/bin/env python3
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
Voxelizer: Voxelization and Spatial Indexing Utilities.

This module provides functions for creating and managing 3D spatial partitions
(voxel spaces) for efficiently mapping entities (atoms) or points (vertices)
to specific regions in 3D space, primarily for accelerated neighbor lookups
and spatial queries.

Functionalities:
1.  **Primary Voxel Space Setup (`calculate_voxel_space_parameters`)**:
    Defines parameters (origin, shape) for a voxel space typically used for
    atom distribution based on their physical extent and influence.
2.  **Atom-to-Voxel Mapping (`build_atom_voxel_map`)**:
    Populates the atom ID storage and creates lookup tables (start/end indices)
    for efficient retrieval of atoms belonging to specific voxels, considering
    neighboring voxels.
3.  **Consolidated Atom Voxel Map Builder (`build_consolidated_atoms_space_voxel_map`)**:
    A high-level function that orchestrates the setup, counting, and mapping
    processes to build the complete primary atom-to-voxel map structures.
4.  **Neighbor Voxel Unique Atom Index Map (`build_neighbor_voxel_unique_atom_index_map`)**:
    Constructs a per-voxel index map of sorted, unique atom indices that reside in the
    27-neighborhood (including itself) of each voxel in a 3D grid. This is used
    to accelerate spatial queries for Gaussian and solute-surface-based map construction.
5.  **Generalized Indexing Voxel Space Setup (`calculate_indexing_voxel_parameters`)**:
    Defines parameters for a potentially coarser secondary voxel space,
    optimized for spatially indexing arbitrary points with configurable
    boundary extensions, often used for grid points.
6.  **Point-to-Indexing-Voxel Mapping (`build_point_voxel_index_map`)**:
    Assigns points to voxels in the indexing voxel space and creates lookup tables
    (start/end indices) for efficient retrieval of points within each voxel.

Uses Numba for JIT compilation to optimize performance.
"""
import time
import numpy as np
from numba import njit

# --- Configuration Imports ---
from pydelphi.config.global_runtime import (
    delphi_bool,
    delphi_int,  # Custom integer type (e.g., np.int32 or np.int64)
    delphi_real,  # Custom real type (e.g., np.float32 or np.float64)
    vprint,
)

# --- Constant Imports ---
from pydelphi.constants import (
    ATOMFIELD_X,  # Index for X coordinate in atom data array
    ATOMFIELD_CRD_END,  # Index marking end of coordinates in atom data
    ATOMFIELD_RADIUS,  # Index for radius in atom data array
    NEIGHBOR_VOXEL_RELATIVE_COORDINATES as NEIGHBOR_VOXEL_REL_COORDS,  # 3x3x3 relative offsets
)
from pydelphi.config.logging_config import (
    DEBUG,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)


# ==============================================================================
# Primary Voxel Space and Atom Mapping Functions
# ==============================================================================


@njit(nogil=True, boundscheck=False, cache=True)
def calculate_voxel_space_parameters(
    voxel_side_length,
    coords_by_axis_min,
    coords_by_axis_max,
    scaling_factor=1.0,
    voxel_space_offset=0.1,
):
    """
    Determines the origin and dimensions (shape) of the primary voxel space.

    Calculates spatial boundaries and voxel counts for a 3D space partition based
    on entity coordinates plus padding.

    Args:
        voxel_side_length (float): Length of one side of a single voxel.
        coords_by_axis_min (np.ndarray): Min coordinate values (x, y, z).
        coords_by_axis_max (np.ndarray): Max coordinate values (x, y, z).
        scaling_factor (float, optional): Factor for padding. Defaults to 1.0.
        voxel_space_offset (float, optional): Additional absolute padding offset.
                                             Defaults to 0.1 to avoid missing points
                                             very close to boundary due to numeric processing.

    Returns:
        tuple: (voxel_space_origin (np.ndarray), voxel_space_shape (np.ndarray))
               Origin (min corner) and shape (voxel counts nx, ny, nz).
    """
    voxel_padding = delphi_real(scaling_factor * voxel_side_length + voxel_space_offset)

    voxel_space_origin = np.empty(len(coords_by_axis_min), dtype=delphi_real)
    voxel_space_origin[:] = coords_by_axis_min - voxel_padding

    voxel_space_highest_xyz = np.empty(len(coords_by_axis_max), dtype=delphi_real)
    voxel_space_highest_xyz[:] = coords_by_axis_max + voxel_padding

    voxel_space_shape = np.empty(len(coords_by_axis_max), dtype=delphi_int)
    voxel_space_shape[:] = np.floor(
        (voxel_space_highest_xyz - voxel_space_origin) / voxel_side_length
    ).astype(delphi_int)[:]

    return voxel_space_origin, voxel_space_shape


@njit(nogil=True, boundscheck=False, cache=True)
def build_atom_voxel_map(
    voxel_side_length,
    num_atoms,
    num_objects,
    num_molecules,
    voxel_space_origin,
    voxel_space_shape,
    atoms_data,
    voxel_atom_ids,
):
    """
    Maps atoms to locations within the defined voxel space and generates lookup tables.

    Populates voxel lookup structures with atom indices, considering neighboring
    voxels for proximity searches. Calculates start/end index ranges in
    `voxel_atom_ids` for atoms associated with each voxel.

    Args:
        voxel_side_length (float): Side length of a voxel.
        num_atoms (int): Total number of atoms.
        num_objects (int): Total number of objects (e.g., molecules).
        num_molecules (int): Number of molecules.
        voxel_space_origin (np.ndarray): Origin (min corner) of the voxel space.
        voxel_space_shape (np.ndarray): Shape (nx, ny, nz) of the voxel space.
        atoms_data (np.ndarray): Atom data (coordinates, radius, etc.).
        voxel_atom_ids (np.ndarray): Pre-allocated 1D array to store atom indices
                                     mapped to voxels.

    Returns:
        tuple: (voxel_atom_ids (np.ndarray),
                voxel_atom_start_index (np.ndarray),
                voxel_atom_end_index (np.ndarray))
               Populated atom ID list, 3D array of start indices, 3D array of end indices.

    Notes:
        - The sizing logic for `atom_central_voxel_coords` involving
          `(num_objects - num_molecules)` might need review based on specific use cases.
        - Temporary arrays are used internally; cleanup relies on Numba/Python GC.
    """
    voxel_atom_start_index = np.ones(
        (voxel_space_shape[0] + 1, voxel_space_shape[1] + 1, voxel_space_shape[2] + 1),
        dtype=delphi_int,
    )
    voxel_atom_end_index = np.zeros(
        (voxel_space_shape[0] + 1, voxel_space_shape[1] + 1, voxel_space_shape[2] + 1),
        dtype=delphi_int,
    )
    atom_central_voxel_coords = np.zeros(
        (
            num_atoms
            + (num_objects - num_molecules)
            * (voxel_space_shape[0] + 1)
            * (voxel_space_shape[1] + 1)
            * (voxel_space_shape[2] + 1)
            + 1,
            3,
        ),
        dtype=delphi_int,
    )
    voxel_space_scale = 1.0 / voxel_side_length

    # --- Pass 1: Calculate central voxel coordinates and count contributions per voxel ---
    for atom_index in range(atoms_data.shape[0]):
        this_atom_data = atoms_data[atom_index]
        if this_atom_data[ATOMFIELD_RADIUS] > 0.0:
            atom_coords = this_atom_data[ATOMFIELD_X:ATOMFIELD_CRD_END]
            ix = delphi_int(
                (atom_coords[0] - voxel_space_origin[0]) * voxel_space_scale
            )
            iy = delphi_int(
                (atom_coords[1] - voxel_space_origin[1]) * voxel_space_scale
            )
            iz = delphi_int(
                (atom_coords[2] - voxel_space_origin[2]) * voxel_space_scale
            )
            for jz in range(iz - 1, iz + 2):
                for jy in range(iy - 1, iy + 2):
                    for jx in range(ix - 1, ix + 2):
                        if (
                            0 <= jx <= voxel_space_shape[0]
                            and 0 <= jy <= voxel_space_shape[1]
                            and 0 <= jz <= voxel_space_shape[2]
                        ):
                            voxel_atom_end_index[jx][jy][jz] += 1
            atom_central_voxel_coords[atom_index][:] = ix, iy, iz

    # Calculate start indices and preliminary end indices
    temp_total_atom_count = 1
    for iz in range(voxel_space_shape[2] + 1):
        for iy in range(voxel_space_shape[1] + 1):
            for ix in range(voxel_space_shape[0] + 1):
                count_in_this_voxel = voxel_atom_end_index[ix][iy][iz]
                if count_in_this_voxel > 0:
                    voxel_atom_start_index[ix][iy][iz] = temp_total_atom_count
                    temp_total_atom_count += count_in_this_voxel
                    voxel_atom_end_index[ix][iy][iz] = temp_total_atom_count
                else:
                    voxel_atom_end_index[ix][iy][iz] = 0

    # --- Pass 2: Populate the flat `voxel_atom_ids` array ---
    current_voxel_indices = voxel_atom_start_index.copy()
    for neighbor_offset in NEIGHBOR_VOXEL_REL_COORDS:
        for atom_index in range(atoms_data.shape[0]):
            this_atom_data = atoms_data[atom_index]
            if this_atom_data[ATOMFIELD_RADIUS] > 0.0:
                cx, cy, cz = atom_central_voxel_coords[atom_index]
                nx, ny, nz = (
                    cx + neighbor_offset[0],
                    cy + neighbor_offset[1],
                    cz + neighbor_offset[2],
                )
                if (
                    0 <= nx <= voxel_space_shape[0]
                    and 0 <= ny <= voxel_space_shape[1]
                    and 0 <= nz <= voxel_space_shape[2]
                ):
                    target_index = current_voxel_indices[nx][ny][nz]
                    voxel_atom_ids[target_index] = atom_index + 1
                    current_voxel_indices[nx][ny][nz] += 1

    # Final calculation of the *inclusive* end index
    for iz in range(voxel_space_shape[2] + 1):
        for iy in range(voxel_space_shape[1] + 1):
            for ix in range(voxel_space_shape[0] + 1):
                count_in_this_voxel = (
                    current_voxel_indices[ix][iy][iz]
                    - voxel_atom_start_index[ix][iy][iz]
                )
                if count_in_this_voxel > 0:
                    voxel_atom_end_index[ix][iy][iz] = (
                        current_voxel_indices[ix][iy][iz] - 1
                    )
                else:
                    voxel_atom_start_index[ix][iy][iz] = 1
                    voxel_atom_end_index[ix][iy][iz] = 0

    atom_central_voxel_coords = None
    current_voxel_indices = None

    return voxel_atom_ids, voxel_atom_start_index, voxel_atom_end_index


# ==============================================================================
# Indexing Voxel Space and Point Mapping Functions
# ==============================================================================
@njit(nogil=True, boundscheck=False, cache=True)
def calculate_indexing_voxel_parameters(
    underlying_spacing,
    probe_radius,
    boundary_extension,
    coords_by_axis_min,
    coords_by_axis_max,
    max_voxels_per_dimension=150,
):
    """
    Calculates parameters for a coarse voxel space used for indexing points.

    Determines dimensions, origin, and voxel size for a secondary voxel space,
    optimized for spatially indexing points while managing memory by limiting
    voxel counts per dimension. The initial boundary padding is now configurable.

    Args:
        underlying_spacing (float): Spacing of the points being indexed or the
                                    underlying grid (for secondary padding).
        probe_radius (float): Probe radius (influences initial voxel size).
        boundary_extension (float): The primary distance to extend the boundaries
                                    beyond the min/max coordinates. This replaces
                                    the previous hardcoded (max_atom_radius +
                                    probe_radius) logic, allowing for general use
                                    (e.g., zeta_distance + probe_radius).
        coords_by_axis_min (np.ndarray): Min coordinates (x, y, z) of points/entities.
        coords_by_axis_max (np.ndarray): Max coordinates (x, y, z) of points/entities.
        max_voxels_per_dimension (int): Max allowed voxels per dimension to
                                        control memory usage. Defaults to 50.

    Returns:
        tuple: (indexing_voxel_side (float),
                indexing_voxel_origin (np.ndarray),
                indexing_voxel_shape (np.ndarray))
               Side length, origin, and shape of the indexing voxel space.

    Notes:
        - The iterative adjustment of `indexing_voxel_side` prevents excessive memory
          usage if the initial size leads to too many voxels.
        - Print statements within the loop are commented out for Numba compatibility.
    """
    # Initial voxel side guess often related to probe size
    indexing_voxel_side = probe_radius
    # Secondary padding based on underlying discretization/spacing
    spacing_x_2 = 2.0 * underlying_spacing

    # Calculate initial padded bounds using the configurable boundary_extension
    indexing_voxel_origin = coords_by_axis_min - boundary_extension  # Use new arg
    max_xyz = coords_by_axis_max + boundary_extension  # Use new arg

    # Apply secondary padding
    indexing_voxel_origin -= spacing_x_2
    max_xyz += spacing_x_2

    # Iteratively adjust voxel size if dimensions exceed the limit
    do_iterate = True
    while do_iterate:
        # Calculate temporary bounds and shape for checking size.
        # The subtraction of probe_radius here relates to the original algorithm's
        temp_max_xyz = max_xyz + indexing_voxel_side - probe_radius
        temp_min_xyz = indexing_voxel_origin - (indexing_voxel_side - probe_radius)
        current_shape = (
            ((temp_max_xyz - temp_min_xyz) / indexing_voxel_side) + 1
        ).astype(delphi_int)

        if np.any(current_shape > max_voxels_per_dimension):
            # If voxel space is too large, increase the voxel side length proportionally
            max_dim = np.max(current_shape)
            indexing_voxel_side = (
                indexing_voxel_side * (max_dim + 1) / max_voxels_per_dimension
            )
        else:
            # Dimensions are acceptable
            do_iterate = False

    # Final calculation of shape using the determined side length and original padded bounds
    indexing_voxel_shape = (
        ((max_xyz - indexing_voxel_origin) / indexing_voxel_side) + 1
    ).astype(delphi_int)
    # Ensure calculated shape dimensions are not negative
    indexing_voxel_shape = np.maximum(indexing_voxel_shape, 0).astype(delphi_int)

    return indexing_voxel_side, indexing_voxel_origin, indexing_voxel_shape


@njit(nogil=True, boundscheck=False, cache=True)
def _calculate_voxel_map_counts(
    voxel_side_length: delphi_real,
    voxel_space_origin: np.ndarray[delphi_real],
    voxel_space_shape: np.ndarray[delphi_int],
    atoms_data: np.ndarray[delphi_real],
):
    """
    Performs the first pass over atoms to count atom entries per voxel.

    This function iterates through the atoms and, for each atom with a positive
    radius, it determines its central voxel and then increments the count for
    its central voxel and all 26 neighboring voxels (a 3x3x3 cube). This count
    represents the number of potential atom contributions to a point located
    within that voxel or its immediate neighbors. This information is used
    to determine the necessary size for the flattened atom ID array
    (`voxel_atom_ids`) and the structure of the start/end index arrays.

    Args:
        voxel_side_length (delphi_real): Side length of a voxel.
        voxel_space_origin (np.ndarray[delphi_real]): Origin (min corner) of the voxel space.
        voxel_space_shape (np.ndarray[delphi_int]): Shape (nx, ny, nz) of the voxel space
                                                   (number of voxels in each dimension).
        atoms_data (np.ndarray[delphi_real]): Atom data array (must contain ATOMFIELD_X,
                                              ATOMFIELD_CRD_END, and ATOMFIELD_RADIUS).

    Returns:
        tuple: (total_atom_entries (delphi_int),
                voxel_atom_counts (np.ndarray[delphi_int]))
               - total_atom_entries: The total number of times an atom's influence
                                     region intersects any voxel, which determines
                                     the minimum size needed for the `voxel_atom_ids`
                                     array (before adding any buffer).
               - voxel_atom_counts: A 3D array mirroring the voxel space shape (plus
                                    a buffer for boundary handling), where each element
                                    stores the count of atom influence regions that
                                    overlap with that specific voxel.
    """
    n_atoms_actual = atoms_data.shape[0]
    voxel_atom_counts = np.zeros(  # Counts per voxel
        (voxel_space_shape[0] + 1, voxel_space_shape[1] + 1, voxel_space_shape[2] + 1),
        dtype=delphi_int,
    )
    # No need to store central coords here, just count
    voxel_space_scale = 1.0 / voxel_side_length
    total_atom_entries_needed: delphi_int = 0

    for atom_index in range(n_atoms_actual):
        this_atom_data = atoms_data[atom_index]
        if this_atom_data[ATOMFIELD_RADIUS] > 0.0:
            atom_coords = this_atom_data[ATOMFIELD_X:ATOMFIELD_CRD_END]
            ix = delphi_int(
                (atom_coords[0] - voxel_space_origin[0]) * voxel_space_scale
            )
            iy = delphi_int(
                (atom_coords[1] - voxel_space_origin[1]) * voxel_space_scale
            )
            iz = delphi_int(
                (atom_coords[2] - voxel_space_origin[2]) * voxel_space_scale
            )

            # Clamp indices
            ix = max(0, min(ix, voxel_space_shape[0]))
            iy = max(0, min(iy, voxel_space_shape[1]))
            iz = max(0, min(iz, voxel_space_shape[2]))

            # Iterate over the (3x3x3) 27 neighbors
            for neighbor_offset in NEIGHBOR_VOXEL_REL_COORDS:
                jx = ix + neighbor_offset[0]
                jy = iy + neighbor_offset[1]
                jz = iz + neighbor_offset[2]
                # Check bounds for (jx, jy, jz)
                if (
                    0 <= jx <= voxel_space_shape[0]
                    and 0 <= jy <= voxel_space_shape[1]
                    and 0 <= jz <= voxel_space_shape[2]
                ):
                    # Process voxel (jx, jy, jz)
                    voxel_atom_counts[jx, jy, jz] += 1
                    total_atom_entries_needed += 1

    return total_atom_entries_needed, voxel_atom_counts


@njit(nogil=True, boundscheck=False, cache=True)
def build_point_voxel_index_map(
    num_points,
    indexing_voxel_scale,
    indexing_voxel_shape,
    indexing_voxel_origin,
    point_coordinates,
    point_indices_by_voxel,
):
    """
    Assigns points to voxels in the indexing voxel space and creates lookup tables.

    Maps points to their corresponding indexing voxels and computes lookup tables
    (start/end indices) for efficient retrieval of points within each voxel.

    Args:
        num_points (int): Total number of points to index.
        indexing_voxel_scale (float): Inverse side length (num_voxel/angstrom) of indexing voxels.
        indexing_voxel_shape (np.ndarray): Shape (nx, ny, nz) of the indexing space.
        indexing_voxel_origin (np.ndarray): Origin (min corner) of the indexing space.
        point_coordinates (np.ndarray): Coordinates (N+1 x 3) of points.
                                        **Assumes points are 1-indexed.**
        point_indices_by_voxel (np.ndarray): Pre-allocated 1D array to store point
                                             indices sorted by voxel.

    Returns:
        tuple: (voxel_point_start_indices (np.ndarray),
                voxel_point_end_indices (np.ndarray),
                point_indices_by_voxel (np.ndarray))
               3D start indices, 3D end indices, and the populated 1D point index list.
    """
    voxel_point_start_indices = np.ones(
        (
            indexing_voxel_shape[0] + 1,
            indexing_voxel_shape[1] + 1,
            indexing_voxel_shape[2] + 1,
        ),
        dtype=delphi_int,
    )
    voxel_point_end_indices = np.zeros(
        (
            indexing_voxel_shape[0] + 1,
            indexing_voxel_shape[1] + 1,
            indexing_voxel_shape[2] + 1,
        ),
        dtype=delphi_int,
    )
    point_voxel_indices = np.zeros((num_points + 1, 3), dtype=delphi_int)

    # --- Pass 1: Calculate voxel index for each point and count points per voxel ---
    for i in range(1, num_points + 1):
        voxel_idx = (
            (point_coordinates[i] - indexing_voxel_origin) * indexing_voxel_scale
        ).astype(delphi_int)
        ix = max(0, min(voxel_idx[0], indexing_voxel_shape[0]))
        iy = max(0, min(voxel_idx[1], indexing_voxel_shape[1]))
        iz = max(0, min(voxel_idx[2], indexing_voxel_shape[2]))
        point_voxel_indices[i, :] = ix, iy, iz
        voxel_point_end_indices[ix, iy, iz] += 1

    # --- Pass 2: Calculate start and end indices for the flat array ---
    current_index = 0
    for k in range(indexing_voxel_shape[2] + 1):
        for j in range(indexing_voxel_shape[1] + 1):
            for i in range(indexing_voxel_shape[0] + 1):
                count_in_voxel = voxel_point_end_indices[i, j, k]
                if count_in_voxel != 0:
                    voxel_point_start_indices[i, j, k] = current_index + 1
                    current_index += count_in_voxel
                    voxel_point_end_indices[i, j, k] = current_index
                else:
                    voxel_point_start_indices[i, j, k] = 1
                    voxel_point_end_indices[i, j, k] = 0

    # --- Pass 3: Populate the flat `point_indices_by_voxel` array ---
    next_slot_in_voxel = voxel_point_start_indices.copy()
    for i in range(1, num_points + 1):
        ix, iy, iz = point_voxel_indices[i]
        target_flat_index = next_slot_in_voxel[ix, iy, iz]
        point_indices_by_voxel[target_flat_index] = i
        next_slot_in_voxel[ix, iy, iz] += 1

    point_voxel_indices = None
    next_slot_in_voxel = None

    return voxel_point_start_indices, voxel_point_end_indices, point_indices_by_voxel


def build_consolidated_atoms_space_voxel_map(
    voxel_side_length,
    coords_by_axis_min,
    coords_by_axis_max,
    scaling_factor,
    voxel_space_offset,
    num_atoms,
    num_objects,
    num_molecules,
    atoms_data,
):
    """
    Builds the complete primary atom-to-voxel map including parameters and lookup tables.

    This function orchestrates the voxel space parameter calculation, the counting
    pass to determine array sizing, and the final mapping pass to populate the
    atom ID list and generate the start/end index lookup tables for the primary
    voxel space used for atoms. It also measures the time taken for these steps.

    Args:
        voxel_side_length (float): Desired side length of the primary voxel cells.
        coords_by_axis_min (np.ndarray): Minimum coordinates (x, y, z) defining the extent
                                         of the data (e.g., min atom coordinates).
        coords_by_axis_max (np.ndarray): Maximum coordinates (x, y, z) defining the extent
                                         of the data (e.g., max atom coordinates).
        scaling_factor (float): Scaling factor used in `calculate_voxel_space_parameters`
                                for padding based on `voxel_side_length`.
        voxel_space_offset (float): Additional absolute offset used in
                                    `calculate_voxel_space_parameters` for padding.
        num_atoms (int): Total number of atoms.
        num_objects (int): Total number of objects.
                           (Passed to build_atom_voxel_map - specific usage may vary).
        num_molecules (int): Total number of molecules.
                             (Passed to build_atom_voxel_map - specific usage may vary).
        atoms_data (np.ndarray): Array containing atom data, including coordinates
                                 and radius, with structure corresponding to
                                 `ATOMFIELD_*` constants.

    Returns:
        tuple: (voxel_params (tuple), voxel_map_data (tuple), time_elapsed (float))
               - voxel_params: A tuple containing (voxel_space_origin (np.ndarray),
                                 voxel_space_shape (np.ndarray), voxel_scale (float),
                                 voxel_side_length (float)).
               - voxel_map_data: A tuple containing (voxel_atom_ids (np.ndarray),
                                   voxel_atom_start_index (np.ndarray),
                                   voxel_atom_end_index (np.ndarray)).
               - time_elapsed: The time taken in seconds to build the voxel map.
    """
    tic = time.perf_counter()

    # 1. Calculate primary voxel space parameters
    voxel_origin, voxel_shape = calculate_voxel_space_parameters(
        voxel_side_length,
        coords_by_axis_min,
        coords_by_axis_max,
        scaling_factor,
        voxel_space_offset,
    )
    voxel_scale = 1.0 / voxel_side_length

    # 2. Perform the first pass to count atom entries per voxel
    total_entries, _ = _calculate_voxel_map_counts(
        voxel_side_length, voxel_origin, voxel_shape, atoms_data
    )

    # 3. Pre-allocate the flat array for atom IDs
    # Add a +1 buffer for 1-based indexing compatibility and safety
    voxel_atom_ids = np.zeros(total_entries + 1, dtype=delphi_int)

    # 4. Perform the second pass to build the atom voxel map structures
    built_voxel_atom_ids, start_calc, end_calc = build_atom_voxel_map(
        voxel_side_length,
        num_atoms,
        num_objects,
        num_molecules,
        voxel_origin,
        voxel_shape,
        atoms_data,
        voxel_atom_ids,
    )
    toc = time.perf_counter()
    time_elapsed = round((toc - tic), 3)

    vprint(DEBUG, _VERBOSITY, f"Specific voxel map built in {time_elapsed:.3f} s.")

    # Package the results
    voxel_params = (voxel_origin, voxel_shape, voxel_scale, voxel_side_length)
    voxel_map_data = (built_voxel_atom_ids, start_calc, end_calc)

    return (
        voxel_params,
        voxel_map_data,
        time_elapsed,
    )


@njit(nogil=True, cache=True)
def _helper_build_neighbor_voxel_atom_index_map(
    voxel_atom_ids: np.ndarray,
    voxel_atom_start_index: np.ndarray,
    voxel_atom_end_index: np.ndarray,
    voxel_shape: np.ndarray,  # (nx, ny, nz)
    neighbor_voxel_atom_ids_flat: np.ndarray,
    neighbor_voxel_start_index: np.ndarray,
    neighbor_voxel_end_index: np.ndarray,
    seen_atoms: np.ndarray,  # shape (num_atoms,), dtype=uint8
    neighbor_offsets: np.ndarray,  # shape (27, 3), NEIGHBOR_VOXEL_REL_COORDS
) -> int:
    """
    Populates neighbor_voxel_atom_ids_flat, neighbor_voxel_start_index, neighbor_voxel_end_index
    for each voxel in the 3D voxel grid, collecting sorted unique atoms from all 27 neighboring voxels.

    Parameters
    ----------
    voxel_atom_ids : 1D array of atom indices assigned to voxels.
    voxel_atom_start_index, voxel_atom_end_index : (nx+1, ny+1, nz+1) arrays marking start/end in voxel_atom_ids.
    voxel_shape : tuple of 3 ints, the voxel grid shape (nx, ny, nz).
    neighbor_voxel_atom_ids_flat : 1D preallocated array for flattened neighbor atom indices.
    neighbor_voxel_start_index, neighbor_voxel_end_index : (nx+1, ny+1, nz+1) arrays to mark start/end per voxel.
    seen_atoms : (num_atoms,) scratch array for duplicate tracking; will be zeroed after each voxel.
    neighbor_offsets : (27, 3) array of relative offsets (e.g., NEIGHBOR_VOXEL_REL_COORDS).
    """
    nx, ny, nz = voxel_shape
    num_atoms = seen_atoms.shape[0]
    output_index = 0

    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                start_pos = output_index

                for n in range(neighbor_offsets.shape[0]):
                    dx, dy, dz = neighbor_offsets[n]
                    vx = i + dx
                    vy = j + dy
                    vz = k + dz

                    if 0 <= vx <= nx and 0 <= vy <= ny and 0 <= vz <= nz:
                        start_idx = voxel_atom_start_index[vx, vy, vz]
                        end_idx = voxel_atom_end_index[vx, vy, vz]

                        if start_idx <= end_idx:
                            for atom_ptr in range(start_idx, end_idx + 1):
                                atom_index = voxel_atom_ids[atom_ptr]
                                if seen_atoms[atom_index] == 0:
                                    seen_atoms[atom_index] = 1
                                    neighbor_voxel_atom_ids_flat[output_index] = (
                                        atom_index
                                    )
                                    output_index += 1

                # Sort the collected atom indices for this voxel
                if output_index > start_pos:
                    neighbor_voxel_atom_ids_flat[start_pos:output_index] = np.sort(
                        neighbor_voxel_atom_ids_flat[start_pos:output_index]
                    )

                neighbor_voxel_start_index[i, j, k] = start_pos
                neighbor_voxel_end_index[i, j, k] = output_index - 1

                # Reset the seen_atoms scratch array
                for a in range(num_atoms):
                    seen_atoms[a] = 0

    return output_index


def build_neighbor_voxel_unique_atom_index_map(
    num_atoms: delphi_int,
    voxel_atom_ids: np.ndarray,
    voxel_atom_start_index: np.ndarray,
    voxel_atom_end_index: np.ndarray,
    voxel_map_shape: np.ndarray,  # (nx, ny, nz)
    print_result: delphi_bool = False,
):
    """
    Constructs a per-voxel index map of sorted, unique atom indices that reside in the 27-neighborhood
    (including itself) of each voxel in a 3D grid. This is used to accelerate spatial queries for
    Gaussian and solute-surface-based map construction.

    For each voxel, the function looks at its 26 neighbors plus itself, gathers all atoms assigned
    to those voxels, removes duplicates using a scratch array (`seen_atoms`), and writes the sorted
    list of unique atom indices into a flattened output array. Start and end indices for each voxel
    are stored to allow direct slicing.

    Parameters
    ----------
    num_atoms : int
        Total number of atoms across the system; used to size scratch buffers.
    voxel_atom_ids : 1D array of int32
        Flat list of atom indices assigned to voxels, produced by voxelization stage.
    voxel_atom_start_index : 3D array of int32, shape (nx+1, ny+1, nz+1)
        Start index into `voxel_atom_ids` per voxel.
    voxel_atom_end_index : 3D array of int32, shape (nx+1, ny+1, nz+1)
        End index into `voxel_atom_ids` per voxel.
    voxel_map_shape : 1D array of 3 ints
        The dimensions of the voxel grid (nx, ny, nz).
    print_result: Whether to print the build neighbor_voxel_unique_atoms arrays for debuggin (default False)

    Returns
    -------
    neighbor_voxel_atom_ids_flat : 1D array of int32
        Flattened array of unique atom indices collected per voxel from its neighbors.
    neighbor_voxel_start_index : 3D array of int32
        Start index into `neighbor_voxel_atom_ids_flat` per voxel.
    neighbor_voxel_end_index : 3D array of int32
        End index into `neighbor_voxel_atom_ids_flat` per voxel.
    actual_neighbor_ids_count : int
        Number of entries filled in `neighbor_voxel_atom_ids_flat`.

    Notes
    -----
    - Output arrays are sized conservatively using an upper bound: 125 * num_atoms.
    - Assumes all atoms fit within the voxel map and indexing ranges are valid.
    - The `seen_atoms` array is reused and zeroed after each voxel's processing.
    - Optionally prints debugging info if `print_result` is True.
    """
    voxel_counts = (
        (voxel_map_shape[0] + 1) * (voxel_map_shape[1] + 1) * (voxel_map_shape[2] + 1)
    )
    # Conservative allocation:
    # - 125 * num_atoms accounts for worst-case atom-to-voxel influence due to padding.
    # - max(voxel_counts, 1000) adds buffer for indexing and low-volume grid fallback.
    neighbor_voxels_size = max(voxel_counts, 1000) + 125 * num_atoms
    neighbor_voxel_index_shape = (
        voxel_map_shape[0] + 1,
        voxel_map_shape[1] + 1,
        voxel_map_shape[2] + 1,
    )

    seen_atoms = np.full(num_atoms + 1, fill_value=False, dtype=np.bool_)
    neighbor_voxel_atom_ids_flat = np.full(
        neighbor_voxels_size, fill_value=-1, dtype=np.int32
    )
    neighbor_voxel_start_index = np.full(
        neighbor_voxel_index_shape, fill_value=-1, dtype=np.int32
    )
    neighbor_voxel_end_index = np.full(
        neighbor_voxel_index_shape, fill_value=-1, dtype=np.int32
    )

    actual_neighbor_ids_count = _helper_build_neighbor_voxel_atom_index_map(
        voxel_atom_ids,
        voxel_atom_start_index,
        voxel_atom_end_index,
        voxel_map_shape,
        neighbor_voxel_atom_ids_flat,
        neighbor_voxel_start_index,
        neighbor_voxel_end_index,
        seen_atoms,
        NEIGHBOR_VOXEL_REL_COORDS,
    )

    if print_result:
        _pretty_print_neighbor_voxel_unique_atom_ids(
            actual_neighbor_ids_count,
            voxel_map_shape,
            neighbor_voxel_atom_ids_flat,
            neighbor_voxel_start_index,
            neighbor_voxel_end_index,
        )

    return (
        neighbor_voxel_atom_ids_flat,
        neighbor_voxel_start_index,
        neighbor_voxel_end_index,
        actual_neighbor_ids_count,
    )


def _pretty_print_neighbor_voxel_unique_atom_ids(
    output_index,
    voxel_map_shape,
    neighbor_voxel_atom_ids_flat,
    neighbor_voxel_start_index,
    neighbor_voxel_end_index,
):
    """
    Prints a formatted representation of the neighbor voxel unique atom IDs for debugging purposes.

    This function iterates through each voxel in the grid and displays the start and end
    indices for its unique neighbor atoms in the flattened array, along with the actual
    list of unique atom IDs. It also provides an overview of the total number of
    filled entries in the flattened array compared to its maximum allocated size.

    Parameters
    ----------
    output_index : int
        The actual number of entries filled in `neighbor_voxel_atom_ids_flat`.
    voxel_map_shape : tuple of 3 ints
        The dimensions of the voxel grid (nx, ny, nz).
    neighbor_voxel_atom_ids_flat : 1D array of int32
        Flattened array of unique atom indices collected per voxel from its neighbors.
    neighbor_voxel_start_index : 3D array of int32
        Start index into `neighbor_voxel_atom_ids_flat` per voxel.
    neighbor_voxel_end_index : 3D array of int32
        End index into `neighbor_voxel_atom_ids_flat` per voxel.
    """
    nx, ny, nz = voxel_map_shape
    print(
        f"Neighbor Voxel Atom IDs (Flat actual/max): {output_index}/{neighbor_voxel_atom_ids_flat.size}"
    )
    print(
        f"Neighbor Voxel Atom IDs (Flat): {neighbor_voxel_atom_ids_flat[:output_index]}"
    )  # Only print filled part
    print("-" * 50)

    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                start_idx = neighbor_voxel_start_index[i, j, k]
                end_idx = neighbor_voxel_end_index[i, j, k]

                print(f"Voxel ({i}, {j}, {k}):")
                if (
                    start_idx == -1
                ):  # Indicates no atoms were found or processed for this voxel
                    print("  No neighbor atoms processed.")
                elif start_idx <= end_idx:
                    unique_atoms = neighbor_voxel_atom_ids_flat[start_idx : end_idx + 1]
                    print(f"  Start Index: {start_idx}, End Index: {end_idx}")
                    print(f"  Unique Neighbor Atoms: {unique_atoms}")
                else:
                    # This case could happen if start_pos == output_index (no atoms added)
                    # and output_index - 1 makes end_idx < start_idx
                    print("  No neighbor atoms found for this voxel.")
                print("-" * 20)
