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
    ERROR,
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

from pydelphi.constants import (
    ConstDelPhiInts,
    ATOMFIELD_X,
    ATOMFIELD_CRD_END,
    ATOMFIELD_RADIUS,
    ATOMFIELD_MEDIA_ID,
    ConstDelPhiFloats as ConstDelPhi,
    NEIGHBOR_VOXEL_RELATIVE_COORDINATES as NEIGHBOR_VOXEL_REL_COORDS,
)

# Initialize module level constants based on global constants.
APPROX_ZERO = ConstDelPhi.ApproxZero.value
RADII_SQUARED_SHRINK_FACTOR = ConstDelPhi.SASSquaredRadiiShrinkFactor.value
MAX_NEIGHBR_BOUNDARY_ARRAY_LEN = ConstDelPhiInts.SpaceNBRASize.value
RESIZE_FACTOR = ConstDelPhi.ZetaArrayResizeFactor.value
INITIAL_SIZE_PERCENT = ConstDelPhi.ZetaArrayInitialSizePercent.value


@njit(nogil=True, boundscheck=False, cache=True)
def _vdw_to_ms_all_voxels(
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
def _vdw_to_ms_all_voxels_buffer(
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
    exposed_grids_coords,
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
        exposed_grids_coords (numpy.ndarray): An array of exposed grid coordinates.

    Returns:
        tuple: A tuple containing the updated cycle_flag, closest_exposed_grid_index, and minimum_distance.
    """
    # force scalar local; do NOT ever assign to `minimum_distance` again
    min_d2 = minimum_distance

    # track previous voxel coordinate as scalars (no np.array alloc)
    pvx = 0
    pvy = 0
    pvz = 0

    # cache midpoint coords as scalars (avoids any shape weirdness)
    mx0 = grid_midpoint_coords[0]
    mx1 = grid_midpoint_coords[1]
    mx2 = grid_midpoint_coords[2]

    # iterate over 27 neighbor voxels
    for t in range(NEIGHBOR_VOXEL_REL_COORDS.shape[0]):
        vx = NEIGHBOR_VOXEL_REL_COORDS[t, 0]
        vy = NEIGHBOR_VOXEL_REL_COORDS[t, 1]
        vz = NEIGHBOR_VOXEL_REL_COORDS[t, 2]

        # relative move from previous
        grid_x_index = grid_x_index + (vx - pvx)
        grid_y_index = grid_y_index + (vy - pvy)
        grid_z_index = grid_z_index + (vz - pvz)

        lower = cube_voxel_start_indices[grid_x_index, grid_y_index, grid_z_index]
        upper = cube_voxel_end_indices[grid_x_index, grid_y_index, grid_z_index]

        # scan points in this voxel
        cycle_flag = False
        for k in range(lower, upper + 1):
            eg_idx = grid_point_indices_in_voxels[k]

            dx = mx0 - exposed_grids_coords[eg_idx, 0]
            dy = mx1 - exposed_grids_coords[eg_idx, 1]
            dz = mx2 - exposed_grids_coords[eg_idx, 2]

            d2 = dx * dx + dy * dy + dz * dz

            if d2 < probe_radius_squared:
                grid_neighs_entity_ids[midpoint_index] = -1
                grid_neighs_media_ids[midpoint_index] = -1
                cycle_flag = True
                break

            if d2 < min_d2:
                closest_exposed_grid_index = eg_idx
                min_d2 = d2

        # update previous voxel
        pvx = vx
        pvy = vy
        pvz = vz

        if cycle_flag:
            break

    return cycle_flag, closest_exposed_grid_index, min_d2


@njit(nogil=True, boundscheck=False, cache=True)
def _scan_candidates_and_find_closest_neighbor(
    lower_limit: int,
    upper_limit: int,
    voxel_atom_ids: np.ndarray,
    atom_surface_flags: np.ndarray,
    neighbor_entity_id: np.ndarray,
    neighbor_index: int,
    min_distance_squared: float,
    closest_atom_or_object_index: int,
    num_atoms: int,
    atoms_data: np.ndarray,
    midpoint_coords: np.ndarray,
    dtype_int,
    dtype_real,
):
    """
    Scans candidate atoms and objects within a voxel range and determines the
    closest neighboring atom or object to a given midpoint.

    This function combines two previously separate steps into a single streaming
    operation:

      1. Candidate enumeration:
         Iterates over atom/object indices stored in the voxel range
         [lower_limit, upper_limit], applying the same eligibility rules as the
         original neighbor-building logic:
           - Atoms are considered only if their surface flag indicates eligibility.
           - Objects are considered only once per contiguous object id and only
             if the midpoint is still unassigned (entity id == 0).

      2. Distance reduction:
         For each eligible candidate, computes the distance from the midpoint
         and keeps track of the closest atom or object encountered.

    The scan is performed without allocating or storing an explicit neighbor list.
    This design is intentional and enables:
      - O(1) per-thread memory usage
      - Straight-line control flow suitable for CPU parallelization (prange)
      - Future CUDA device portability without per-thread local arrays

    The semantics and variable naming closely follow the original implementation
    to preserve traceability and numerical behavior.

    Args:
        lower_limit (int):
            Lower bound (inclusive) of the voxel_atom_ids range to scan.

        upper_limit (int):
            Upper bound (inclusive) of the voxel_atom_ids range to scan.
            If lower_limit > upper_limit, the scan is skipped.

        voxel_atom_ids (np.ndarray):
            Packed array containing atom and object indices assigned to the voxel.

        atom_surface_flags (np.ndarray):
            Array indicating whether an atom is eligible for surface consideration.
            A value of 0 means the atom is eligible.

        grid_neighs_entity_ids (np.ndarray):
            Entity ids of neighboring midpoints. Used to gate object candidates
            so that objects are only considered for virgin (unassigned) midpoints.

        neighbor_index (int):
            Index of the midpoint neighbor being processed (1–6).

        min_distance_squared (float):
            Current minimum distance metric. For atoms, this represents the
            distance to the atomic surface (not center).

        closest_atom_or_object_index (int):
            Index of the closest atom or object found so far. Updated during scan.

        num_atoms (int):
            Total number of atoms in the system. Indices greater than num_atoms
            are treated as objects.

        atoms_data (np.ndarray):
            Array containing atomic data, including coordinates, radius, and
            media id fields.

        midpoint_coords (np.ndarray):
            Coordinates of the midpoint being evaluated.

        dtype_int (type):
            Integer data type used for indices.

        dtype_real (type):
            Floating-point data type used for distance calculations.

    Returns:
        tuple:
            - closest_atom_or_object_index (int):
                Index of the closest atom or object identified during the scan.
                A value of 0 indicates no valid neighbor was found.

            - min_distance_squared (float):
                Minimum distance metric corresponding to the closest candidate.
                For atoms, this is the distance to the atomic surface.

            - atom_or_object_index_current (int):
                The most recently encountered non-zero atom or object index during
                the scan. This value preserves compatibility with downstream logic
                (e.g., media id assignment) that previously relied on the last
                scanned candidate.
    """
    previous_atom_or_object_index = dtype_int(0)
    num_neighbors_found = dtype_int(0)

    # This replaces the old 'atom_or_object_index' that existed after the kk-loop.
    # We keep the same name for traceability at the call site.
    atom_or_object_index_current = dtype_int(0)

    for kk in range(lower_limit, upper_limit + 1):
        atom_or_object_index = voxel_atom_ids[kk]

        if kk > 0 and atom_or_object_index == 0:
            nprint_cpu(DEBUG, _VERBOSITY, " VdMS> problems with cube")

        # Preserve "last seen" semantics (closest to old behavior)
        if atom_or_object_index != 0:
            atom_or_object_index_current = dtype_int(atom_or_object_index)

        if 0 < atom_or_object_index <= num_atoms:
            if atom_surface_flags[atom_or_object_index] == 0:
                num_neighbors_found += 1

                atom_coords = (
                    atoms_data[atom_or_object_index - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
                ).astype(dtype_real)
                atom_radius = atoms_data[atom_or_object_index - 1][ATOMFIELD_RADIUS]

                distance_vector = (midpoint_coords - atom_coords).astype(dtype_real)
                distance_squared = size_cpu.dot_product(
                    distance_vector, distance_vector
                )
                distance_to_surface = sqrt(distance_squared) - atom_radius

                if distance_to_surface < min_distance_squared:
                    min_distance_squared = distance_to_surface
                    closest_atom_or_object_index = atom_or_object_index
        else:
            if (
                atom_or_object_index != previous_atom_or_object_index
                and neighbor_entity_id == 0
            ):
                previous_atom_or_object_index = atom_or_object_index
                num_neighbors_found = num_neighbors_found + 1

                # object branch preserved
                distance_squared = dtype_real(0.0)
                if 0.0 <= distance_squared < min_distance_squared:
                    min_distance_squared = distance_squared
                    closest_atom_or_object_index = dtype_int(atom_or_object_index)

        if num_neighbors_found >= MAX_NEIGHBR_BOUNDARY_ARRAY_LEN:
            nprint_cpu(
                ERROR,
                _VERBOSITY,
                "space_vwtoms>> index beyond size of nbra: ii= ",
                num_neighbors_found,
            )

    return (
        closest_atom_or_object_index,
        min_distance_squared,
        atom_or_object_index_current,
    )


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
    contact_region_media_id: int,
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
