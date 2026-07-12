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

import gc
import math
import numpy as np
from numba import njit, prange, set_num_threads
from numba import cuda, int32, float32, float64

from pydelphi.config.global_runtime import (
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

from pydelphi.constants import (
    ConstDelPhiFloats,
    ConstDelPhiInts,
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_CRD_END,
    ATOMFIELD_RADIUS,
    ATOMFIELD_MEDIA_ID,
)

# Define a small constant used for comparisons with zero.
APPROX_ZERO = ConstDelPhiFloats.ApproxZero.value
"""
A small constant used for comparisons with zero. Values smaller than this can be treated as numeric zero.
"""

# Define the maximum size of the neighbor list.
SPACE_NBRA_MAX_SIZE = ConstDelPhiInts.SpaceNBRASize.value
# Define the return value indicating an error in njit functions.
EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value

SHARED_SIZE = 512  # max threads per block we'll support in shared reduction
THREADS_PER_BLOCK = 512


@njit(nogil=True, boundscheck=False, cache=True)
def _is_internal_boundary(
    discrete_epsilon_map_1d_index: np.ndarray,
    grid_index_1d: delphi_int,
    epsilon_dimension: delphi_int,
    num_atoms: delphi_int,
    x_stride_x_3: delphi_int,
    y_stride_x_3: delphi_int,
) -> delphi_bool:
    """Checks if a given grid point lies on an internal boundary between different dielectric regions.

    Refactored to use 0-based indexing (0 to 5) for the 6 neighboring midpoints.

    Args:
        discrete_epsilon_map_1d_index (np.ndarray): 1D array representing the discrete epsilon map.
        grid_index_1d (delphi_int): The 1D index of the grid point.
        epsilon_dimension (delphi_int): The total number of dielectric regions.
        num_atoms (delphi_int): The number of atoms in the system.
        x_stride_x_3 (delphi_int): Stride used to access neighbors in the x-direction.
        y_stride_x_3 (delphi_int): Stride used to access neighbors in the y-direction.

    Mutates:
        None: This function reads input arrays but does not modify any input arguments.

    Returns:
        delphi_bool: True if the grid point is on an internal boundary, False otherwise.
    """
    # Use size 6 for 0-based indexing of the 6 midpoints (0 to 5)
    grid_neighs_entity_ids = np.zeros(6, dtype=delphi_int)
    grid_neighs_media_ids = np.zeros(6, dtype=delphi_int)
    grid_index_1d_x_3 = grid_index_1d * 3
    z_stride = 1

    # Stride offsets for the 6 neighbors (corresponding to indices 0 to 5)
    # [0, 1, 2] -> Leading +h/2 in x, y, z
    # [3, 4, 5] -> Lagging -h/2 in x, y, z
    grid_neighs_1d_offsets = np.zeros(6, dtype=delphi_int)

    # Indices 0, 1, 2: Leading by h/2 neighbors in x, y, z directions
    grid_neighs_1d_offsets[0] = 0
    grid_neighs_1d_offsets[1] = 1
    grid_neighs_1d_offsets[2] = 2

    # Indices 3, 4, 5: Lagging by h/2 neighbors in x, y, z directions
    grid_neighs_1d_offsets[3] = -x_stride_x_3
    # Note: We used +1 for y-lagging and +2 for z-lagging
    grid_neighs_1d_offsets[4] = -y_stride_x_3 + 1
    grid_neighs_1d_offsets[5] = -z_stride * 3 + 2

    # Iterate through the 6 neighboring midpoint indices (0 to 5).
    for neigh_id in range(6):
        # Calculate the index in the 1D epsilon map.
        entity_index = grid_index_1d_x_3 + grid_neighs_1d_offsets[neigh_id]

        # Calculate the entity ID and media ID.
        value = discrete_epsilon_map_1d_index[entity_index]
        grid_neighs_entity_ids[neigh_id] = value % epsilon_dimension
        grid_neighs_media_ids[neigh_id] = value // epsilon_dimension

    is_internal_boundary_grid_point = False

    # Check dielectric discontinuities by iterating over adjacent midpoint pairs (0→1, 1→2, …, 5→0).
    for midpoint_index in range(6):
        # Calculate the index of the adjacent neighbor in the offset list (0, 1, 2, 3, 4, 5)
        # (midpoint_index + 1) % 6 ensures 5 -> 0 wrap-around
        neighbor_index = (midpoint_index + 1) % 6

        # Check if the media IDs of the two midpoints are different and neither is zero (solvent).
        midpoint_media = grid_neighs_media_ids[midpoint_index]
        neighbor_media = grid_neighs_media_ids[neighbor_index]

        if midpoint_media != neighbor_media and midpoint_media * neighbor_media != 0:
            # Check if both entities belong to atoms (or are very close to atoms).
            midpoint_entity = grid_neighs_entity_ids[midpoint_index]
            neighbor_entity = grid_neighs_entity_ids[neighbor_index]

            is_internal_boundary_grid_point |= (
                midpoint_entity <= num_atoms + 1 and neighbor_entity <= num_atoms + 1
            )

    return is_internal_boundary_grid_point


@njit(nogil=True, boundscheck=False, cache=True)
def _process_neighbors(
    gp_x: delphi_real,
    gp_y: delphi_real,
    gp_z: delphi_real,
    num_atoms: delphi_int,
    atom_data: np.ndarray,
    neighboring_atom_indices: np.ndarray,
    neighbor_count: delphi_int,
    previous_media_id: delphi_int,
    closest_atom_or_object_index: delphi_int,
    min_distance_squared: delphi_real,
    debug_level: delphi_int,  # Non-type constant argument
    verbosity_level: delphi_int,  # Non-type constant argument
    delphi_real_type: type,  # <-- MOVED TO LAST
    delphi_int_type: type,  # <-- MOVED TO LAST
) -> tuple[delphi_int, delphi_int, delphi_real]:
    """Finds the closest atom or object to a given grid point.

    Args:
        gp_x (delphi_real): Scalar x-coordinate of the grid point.
        gp_y (delphi_real): Scalar y-coordinate of the grid point.
        gp_z (delphi_real): Scalar z-coordinate of the grid point.
        num_atoms (delphi_int): The number of atoms in the system.
        atom_data (np.ndarray): Array containing atom properties.
        neighboring_atom_indices (np.ndarray): Array storing indices of neighboring atoms.
        neighbor_count (delphi_int): The number of neighboring atoms found so far.
        previous_media_id (delphi_int): The media ID of the previously considered closest entity.
        closest_atom_or_object_index (delphi_int): Index of the closest atom or object found so far.
        min_distance_squared (delphi_real): Minimum distance squared found so far.
        debug_level (delphi_int): Current global DEBUG level.
        verbosity_level (delphi_int): Current global _VERBOSITY level.
        delphi_real_type (type): Data type for real numbers.
        delphi_int_type (type): Data type for integers.

    Mutates:
        None: This function only updates local scalar variables and returns them; it does not modify any input arrays.

    Returns:
        delphi_int: Index of the closest atom or object.
        delphi_int: Media ID of the closest atom or object.
        delphi_real: Minimum distance squared to the closest atom or object.
    """
    distance = delphi_real_type(0.0)
    # Iterate through the neighboring grid points.
    for ii in range(1, neighbor_count + 1):
        if ii >= SPACE_NBRA_MAX_SIZE:
            nprint_cpu(
                debug_level,
                verbosity_level,
                "space_sclbp>> index beyond size of nbra: ii= ",
                ii,
            )

        atom_or_object_index = neighboring_atom_indices[ii]
        this_media_id = delphi_int_type(
            atom_data[atom_or_object_index - 1][ATOMFIELD_MEDIA_ID]
        )
        # Check if the media ID has changed.
        is_media_different = this_media_id != previous_media_id

        # Process if the current entity is an atom.
        if atom_or_object_index <= num_atoms:
            # Calculate the vector from the atom center to the grid point (scalar ops).
            atom_x = atom_data[atom_or_object_index - 1, ATOMFIELD_X]
            atom_y = atom_data[atom_or_object_index - 1, ATOMFIELD_Y]
            atom_z = atom_data[atom_or_object_index - 1, ATOMFIELD_Z]
            atom_radius = atom_data[atom_or_object_index - 1, ATOMFIELD_RADIUS]

            dx = gp_x - atom_x
            dy = gp_y - atom_y
            dz = gp_z - atom_z

            # Calculate the distance from the grid point to the atom surface.
            delta_norm_squared = dx * dx + dy * dy + dz * dz
            distance = np.sqrt(delta_norm_squared) - atom_radius

            # Define precedence and proximity conditions for updating the closest entity.
            precedence = (
                atom_or_object_index > closest_atom_or_object_index
                or closest_atom_or_object_index > num_atoms
            )
            proximity = abs(distance) < abs(min_distance_squared)
            condition = (precedence and (proximity or distance < 0.0)) or (
                proximity and min_distance_squared > 0.0
            )

            # Update the closest atom if the current atom is closer or satisfies the conditions.
            if (distance < min_distance_squared and not is_media_different) or (
                condition and is_media_different
            ):
                previous_media_id = this_media_id
                min_distance_squared = distance
                closest_atom_or_object_index = atom_or_object_index

    return closest_atom_or_object_index, previous_media_id, min_distance_squared


@njit(nogil=True, boundscheck=False, cache=True, fastmath=True)
def _is_solvent_exposed(
    u_x: delphi_real,
    u_y: delphi_real,
    u_z: delphi_real,
    num_atoms: delphi_int,
    atom_data: np.ndarray,
    voxel_atom_indices: np.ndarray,
    atoms_per_voxel_count: np.ndarray,
    cumulative_atoms_per_voxel: np.ndarray,
    cube_vertex_min_xyz: np.ndarray,
    inverse_cube_side_length: delphi_real,
    cube_nx: delphi_int,
    cube_ny: delphi_int,
    cube_nz: delphi_int,
    shrunk_atom_plus_probe_radii_squared: np.ndarray,
    delphi_real_type: type,
) -> delphi_bool:
    """
    Checks if a given point (u_x,u_y,u_z) is solvent-exposed by checking proximity
    to atoms within the voxel it falls into.
    Returns True (1) if exposed, False (0) if buried.
    """
    # --- 1. Compute voxel index (ci0,ci1,ci2) ---
    ci0 = int((u_x - cube_vertex_min_xyz[0]) * inverse_cube_side_length)
    ci1 = int((u_y - cube_vertex_min_xyz[1]) * inverse_cube_side_length)
    ci2 = int((u_z - cube_vertex_min_xyz[2]) * inverse_cube_side_length)

    # --- 2. Boundary rejection ---
    if (
        ci0 < 0
        or ci0 >= cube_nx
        or ci1 < 0
        or ci1 >= cube_ny
        or ci2 < 0
        or ci2 >= cube_nz
    ):
        return True  # outside cube = solvent exposed

    # --- 3. Get atom list for this voxel ---

    lower = atoms_per_voxel_count[ci0, ci1, ci2]
    upper = cumulative_atoms_per_voxel[ci0, ci1, ci2]

    # --- 4. Loop over atoms in voxel ---
    for kk in range(lower, upper + 1):  # note: inclusive upper in original
        atom_idx = voxel_atom_indices[kk]

        if atom_idx <= num_atoms:
            # --- fetch atom data ---
            ax = atom_data[atom_idx - 1, ATOMFIELD_X]
            ay = atom_data[atom_idx - 1, ATOMFIELD_Y]
            az = atom_data[atom_idx - 1, ATOMFIELD_Z]

            dx = u_x - ax
            dy = u_y - ay
            dz = u_z - az
            dist2 = delphi_real_type(dx * dx + dy * dy + dz * dz)

            # --- overlap check ---
            if dist2 < shrunk_atom_plus_probe_radii_squared[atom_idx - 1]:
                return False  # buried

    # --- 5. No overlaps found ---
    return True


@njit(nogil=True, boundscheck=False, cache=True)
def _scale_exposed_point(
    closest_atom_or_object_index: delphi_int,
    num_atoms: delphi_int,
    atom_data: np.ndarray,
    distance_to_surface: delphi_real,
    dr_x: delphi_real,
    dr_y: delphi_real,
    dr_z: delphi_real,
    atom_data_radius_index: delphi_int,
    delphi_real_type: type,
) -> (delphi_real, delphi_real, delphi_real, delphi_real, delphi_real, delphi_real):
    """Scales the position of a solvent-exposed boundary point to the van der Waals surface of the closest atom.

    Args:
        bgp_index (delphi_int): **Index of the boundary grid point** currently being processed.
        closest_atom_or_object_index (delphi_int): Index of the closest atom or object.
        num_atoms (delphi_int): The number of atoms in the system.
        atom_data (np.ndarray): Array containing atom properties.
        distance_to_surface (delphi_real): Distance from the atom center to the original boundary point.
        dr_x (delphi_real): Vector from the atom center to the original boundary point (x-component).
        dr_y (delphi_real): Vector from the atom center to the original boundary point (y-component).
        dr_z (delphi_real): Vector from the atom center to the original boundary point (z-component).

        atom_data_radius_index (delphi_int): Index of the atom radius in the atom_data array.
        delphi_real_type (type): Data type for real numbers.

    Mutates:
       None.

    Returns:
        surf_pos_x, surf_pos_y, surf_pos_z, surf_norm_x, surf_norm_y, surf_norm_z
    """
    surf_pos_x, surf_pos_y, surf_pos_z, surf_norm_x, surf_norm_y, surf_norm_z = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    if closest_atom_or_object_index <= num_atoms:
        atom_index = closest_atom_or_object_index - 1

        atom_x = delphi_real_type(atom_data[atom_index, ATOMFIELD_X])
        atom_y = delphi_real_type(atom_data[atom_index, ATOMFIELD_Y])
        atom_z = delphi_real_type(atom_data[atom_index, ATOMFIELD_Z])

        atom_radius = atom_data[atom_index, atom_data_radius_index]
        scale_factor = atom_radius / distance_to_surface

        # Mutates surface_charge_point_positions at bgp_index
        surf_pos_x = atom_x + dr_x * scale_factor
        surf_pos_y = atom_y + dr_y * scale_factor
        surf_pos_z = atom_z + dr_z * scale_factor

        # Mutates scaled_surface_normal_vectors at bgp_index
        surf_norm_x = dr_x / distance_to_surface
        surf_norm_y = dr_y / distance_to_surface
        surf_norm_z = dr_z / distance_to_surface

    return surf_pos_x, surf_pos_y, surf_pos_z, surf_norm_x, surf_norm_y, surf_norm_z


@njit(cache=True, nogil=True)
def _scale_initial_boundary_points(
    num_boundary_points,
    surface_charge_positions,
    atom_data,
    atom_plus_probe_radii,
    atom_plus_probe_radii_squared_shrunk,
    voxel_atom_indices,
    voxel_atom_count,
    voxel_atom_count_cumulative,
    voxel_space_origin,
    voxel_space_scale,
    voxel_space_shape,
    atom_accessibility,
    num_atoms,
    num_molecules,
    discrete_epsilon_index_map_1d,
    epsilon_dimension,
    x_stride,
    y_stride,
    x_stride_x_3,
    y_stride_x_3,
    current_grid_origin,
    grid_spacing,
    max_probe_radius,
    atom_surface_index,
    atom_index_for_boundary,
    scaled_surface_normal_vectors,
    delphi_real,
    delphi_int,
):
    """
    Scales the initial set of boundary points to align with the van der Waals surface of the closest atoms.

    This function iterates over all boundary points, determining whether each point lies on an
    internal or external boundary. For internal boundaries, it calculates a scaled position
    between neighboring atoms. For external boundaries, it identifies the closest atom or object,
    evaluates solvent exposure, and scales points onto the solvent-accessible surface if applicable.

    Args:
        num_boundary_points (delphi_int): Total number of boundary points.
        surface_charge_positions (np.ndarray): Array of boundary point coordinates.
        atom_data (np.ndarray): Array of atomic properties.
        atom_plus_probe_radii (np.ndarray): Array of atom radii plus solvent probe radius.
        atom_plus_probe_radii_squared_shrunk (np.ndarray): Array of shrunk atom radii squared.
        voxel_atom_indices (np.ndarray): Array mapping voxels to contained atom indices.
        voxel_atom_count (np.ndarray): Array of starting indices of atoms in each voxel.
        voxel_atom_count_cumulative (np.ndarray): Array of ending indices of atoms in each voxel.
        voxel_space_origin (np.ndarray): Origin coordinates of the voxelized space.
        voxel_space_scale (delphi_real): Inverse of voxel side length.
        voxel_space_shape (tuple): Shape of the voxelized space.
        atom_accessibility (np.ndarray): Accessibility flag for each atom.
        num_atoms (delphi_int): Total number of atoms.
        num_molecules (delphi_int): Total number of molecules.
        discrete_epsilon_index_map_1d (np.ndarray): Flattened dielectric map.
        epsilon_dimension (delphi_int): Number of discrete dielectric regions.
        x_stride (delphi_int): X-direction stride for 1D indexing of the grid.
        y_stride (delphi_int): Y-direction stride for 1D indexing of the grid.
        x_stride_x_3 (delphi_int): X-stride multiplied by 3 for midpoint indexing.
        y_stride_x_3 (delphi_int): Y-stride multiplied by 3 for midpoint indexing.
        current_grid_origin (np.ndarray): Coordinates of the current grid origin.
        grid_spacing (delphi_real): Grid spacing.
        max_probe_radius (delphi_real): Maximum probe radius used for solvent-accessible surface.
        atom_surface_index (np.ndarray): Array to store closest atom index per boundary point.
        atom_index_for_boundary (np.ndarray): Array indicating atom index responsible for each boundary point.
        scaled_surface_normal_vectors (np.ndarray): Array to store scaled surface normal vectors.
        delphi_real (type): Floating-point type used in calculations.
        delphi_int (type): Integer type used in calculations.

    Returns:
        int: Status flag (0 for success, EXIT_NJIT_FLAG for failure).
        int: Count of successfully scaled boundary points.
    """
    scaled_boundary_point_count = 0

    for bgp_index in range(num_boundary_points):
        # --- Internal boundary check ---
        is_internal_bgp = False
        if num_boundary_points != num_atoms and num_molecules > 1:
            grid_coords_i = int(surface_charge_positions[bgp_index, 0])
            grid_coords_j = int(surface_charge_positions[bgp_index, 1])
            grid_coords_k = int(surface_charge_positions[bgp_index, 2])

            grid_index_1d = (
                grid_coords_i * x_stride + grid_coords_j * y_stride + grid_coords_k
            )
            is_internal_bgp = _is_internal_boundary(
                discrete_epsilon_index_map_1d,
                grid_index_1d,
                epsilon_dimension,
                num_atoms,
                x_stride_x_3,
                y_stride_x_3,
            )

        # --- Cartesian coordinates ---
        grid_point_coords = np.zeros(3, dtype=delphi_real)
        for i in range(3):
            grid_point_coords[i] = (
                surface_charge_positions[bgp_index][i] * grid_spacing
                + current_grid_origin[i]
            )

        min_distance_squared_1 = 1000.0
        min_distance_squared_2 = 1000.0
        closest_atom_index_1 = 0
        closest_atom_index_2 = 0
        neighbor_count = 0
        closest_atom_or_object_index = 0
        previous_object_index = 0
        previous_media_id = 0
        neighboring_atom_indices = np.zeros(SPACE_NBRA_MAX_SIZE, dtype=delphi_int)

        # --- Cube indices ---
        cube_indices = np.zeros(3, dtype=delphi_int)
        for i in range(3):
            cube_indices[i] = int(
                (grid_point_coords[i] - voxel_space_origin[i]) * voxel_space_scale
            )

        lower_limit = 0
        upper_limit = -1

        out_of_bounds = False
        for i in range(3):
            if cube_indices[i] < 0 or cube_indices[i] >= voxel_space_shape[i]:
                out_of_bounds = True
                break

        if not out_of_bounds:
            lower_limit = voxel_atom_count[
                cube_indices[0], cube_indices[1], cube_indices[2]
            ]
            upper_limit = voxel_atom_count_cumulative[
                cube_indices[0], cube_indices[1], cube_indices[2]
            ]
        else:
            lower_limit = 0
            upper_limit = -1

        # --- Loop over atoms/objects ---
        for kk in range(lower_limit, upper_limit + 1):
            atom_or_object_index = voxel_atom_indices[kk]

            if atom_or_object_index <= num_atoms:
                if is_internal_bgp:
                    delta_coords = (
                        grid_point_coords
                        - atom_data[
                            atom_or_object_index - 1, ATOMFIELD_X:ATOMFIELD_CRD_END
                        ]
                    ).astype(np.float64)
                    this_atom_radius = atom_data[
                        atom_or_object_index - 1, ATOMFIELD_RADIUS
                    ]
                    delta_norm_squared = (
                        delta_coords[0] * delta_coords[0]
                        + delta_coords[1] * delta_coords[1]
                        + delta_coords[2] * delta_coords[2]
                    )
                    distance_squared = delta_norm_squared - this_atom_radius**2

                    if distance_squared < min_distance_squared_1:
                        closest_atom_index_2 = closest_atom_index_1
                        min_distance_squared_2 = min_distance_squared_1
                        closest_atom_index_1 = atom_or_object_index
                        min_distance_squared_1 = distance_squared
                    elif distance_squared <= min_distance_squared_2:
                        closest_atom_index_2 = atom_or_object_index
                        min_distance_squared_2 = distance_squared
                else:
                    if atom_accessibility[atom_or_object_index] == 0:
                        neighbor_count += 1
                        if neighbor_count < SPACE_NBRA_MAX_SIZE:
                            neighboring_atom_indices[neighbor_count] = (
                                atom_or_object_index
                            )
            else:
                if atom_or_object_index != previous_object_index:
                    previous_object_index = atom_or_object_index
                    neighbor_count += 1
                    if neighbor_count < SPACE_NBRA_MAX_SIZE:
                        neighboring_atom_indices[neighbor_count] = atom_or_object_index

        # --- Internal boundary handling ---
        if is_internal_bgp:
            atom_surface_index[bgp_index] = closest_atom_index_1
            if (
                closest_atom_index_1 * closest_atom_index_2 == 0
                or closest_atom_index_1 == closest_atom_index_2
            ):
                nprint_cpu(
                    DEBUG,
                    _VERBOSITY,
                    "Problems in Scaling multidielectric Boundary Grid Points",
                )
                return EXIT_NJIT_FLAG, 0
            atom_index_for_boundary[bgp_index] = -1

            delta_coords_12 = (
                atom_data[closest_atom_index_2 - 1, ATOMFIELD_X:ATOMFIELD_CRD_END]
                - atom_data[closest_atom_index_1 - 1, ATOMFIELD_X:ATOMFIELD_CRD_END]
            )
            temp = (
                delta_coords_12[0] * delta_coords_12[0]
                + delta_coords_12[1] * delta_coords_12[1]
                + delta_coords_12[2] * delta_coords_12[2]
            )
            temp = 0.5 * (min_distance_squared_2 - min_distance_squared_1) / temp
            surface_charge_positions[bgp_index] = (
                grid_point_coords + temp * delta_coords_12
            )
            scaled_surface_normal_vectors[bgp_index] = np.zeros_like(grid_point_coords)
            continue

        # --- External boundary ---
        (closest_atom_or_object_index, previous_media_id, _) = _process_neighbors(
            grid_point_coords[0],
            grid_point_coords[1],
            grid_point_coords[2],
            num_atoms,
            atom_data,
            neighboring_atom_indices,
            neighbor_count,
            previous_media_id,
            closest_atom_or_object_index,
            100.0,
            DEBUG,
            _VERBOSITY,
            delphi_real,
            delphi_int,
        )
        atom_surface_index[bgp_index] = closest_atom_or_object_index

        if closest_atom_or_object_index == 0 and closest_atom_index_1 == 0:
            nprint_cpu(
                DEBUG,
                _VERBOSITY,
                "no close atom or object for boundary pointer ",
                bgp_index,
            )
            return EXIT_NJIT_FLAG, 0

        # --- Distance vector ---
        if closest_atom_or_object_index <= num_atoms:
            delta_r = (
                grid_point_coords
                - atom_data[
                    closest_atom_or_object_index - 1, ATOMFIELD_X:ATOMFIELD_CRD_END
                ]
            ).astype(np.float64)
        else:
            delta_r = np.zeros(3, dtype=np.float64)

        distance_to_surface = np.sqrt(
            delta_r[0] * delta_r[0] + delta_r[1] * delta_r[1] + delta_r[2] * delta_r[2]
        )
        is_outside = True

        # --- Solvent exposure check ---
        if max_probe_radius > 0.0:
            if closest_atom_or_object_index <= num_atoms:
                u123 = atom_data[
                    closest_atom_or_object_index - 1, ATOMFIELD_X:ATOMFIELD_CRD_END
                ] + (
                    atom_plus_probe_radii[closest_atom_or_object_index - 1]
                    * delta_r
                    / distance_to_surface
                )
            else:
                u123 = grid_point_coords - delta_r

            is_outside = _is_solvent_exposed(
                u123[0],
                u123[1],
                u123[2],
                num_atoms,
                atom_data,
                voxel_atom_indices,
                voxel_atom_count,
                voxel_atom_count_cumulative,
                voxel_space_origin,
                voxel_space_scale,
                voxel_space_shape[0],
                voxel_space_shape[1],
                voxel_space_shape[2],
                atom_plus_probe_radii_squared_shrunk,
                delphi_real,
            )

        # --- Scale exposed boundary point ---
        if is_outside:
            scaled_boundary_point_count += 1
            surf_pos = np.zeros(3, dtype=delphi_real)
            surf_norm = np.zeros(3, dtype=delphi_real)

            (
                surf_pos_x,
                surf_pos_y,
                surf_pos_z,
                surf_norm_x,
                surf_norm_y,
                surf_norm_z,
            ) = _scale_exposed_point(
                closest_atom_or_object_index,
                num_atoms,
                atom_data,
                distance_to_surface,
                delta_r[0],
                delta_r[1],
                delta_r[2],
                ATOMFIELD_RADIUS,
                delphi_real,
            )

            surface_charge_positions[bgp_index, 0] = surf_pos_x
            surface_charge_positions[bgp_index, 1] = surf_pos_y
            surface_charge_positions[bgp_index, 2] = surf_pos_z

            scaled_surface_normal_vectors[bgp_index, 0] = surf_norm_x
            scaled_surface_normal_vectors[bgp_index, 1] = surf_norm_y
            scaled_surface_normal_vectors[bgp_index, 2] = surf_norm_z

            atom_index_for_boundary[bgp_index] = closest_atom_or_object_index
        else:
            atom_index_for_boundary[bgp_index] = 0

    return 0, scaled_boundary_point_count


@njit(nogil=True, boundscheck=False, inline="always")
def _scan_voxel_for_closest_atom(
    gp0,
    gp1,
    gp2,
    lower_limit,
    upper_limit,
    voxel_atom_indices,
    atom_data,
    atom_accessibility,
    num_atoms,
    # state / init
    prev_media_id_init,
    closest_init,
    best_dist_init,
    # types
    delphi_real,
    delphi_int,
):
    """
    List-free replacement for (_fill neighbor list + _process_neighbors).

    Matches _process_neighbors' atom-selection logic:
      - only considers atoms with atom_accessibility[idx] == 0
      - compares signed distance to surface: sqrt(r2) - radius
      - applies the same (precedence/proximity/media-change) update rules
    """
    previous_media_id = prev_media_id_init
    closest_atom_or_object_index = closest_init
    best_distance = best_dist_init  # NOTE: this is signed distance-to-surface

    for kk in range(lower_limit, upper_limit + 1):
        atom_or_object_index = voxel_atom_indices[kk]

        # Only atoms participate in the "closest" choice (same as your _process_neighbors)
        if atom_or_object_index <= num_atoms:
            if atom_accessibility[atom_or_object_index] != 0:
                continue

            # media id
            this_media_id = delphi_int(
                atom_data[atom_or_object_index - 1, ATOMFIELD_MEDIA_ID]
            )
            is_media_different = this_media_id != previous_media_id

            # signed distance to surface
            ax = atom_data[atom_or_object_index - 1, ATOMFIELD_X]
            ay = atom_data[atom_or_object_index - 1, ATOMFIELD_Y]
            az = atom_data[atom_or_object_index - 1, ATOMFIELD_Z]
            ar = atom_data[atom_or_object_index - 1, ATOMFIELD_RADIUS]

            dx = gp0 - ax
            dy = gp1 - ay
            dz = gp2 - az
            r2 = dx * dx + dy * dy + dz * dz
            dist = delphi_real(math.sqrt(r2) - ar)  # signed distance to atom surface

            # replicate your _process_neighbors selection logic
            precedence = (atom_or_object_index > closest_atom_or_object_index) or (
                closest_atom_or_object_index > num_atoms
            )
            proximity = abs(dist) < abs(best_distance)
            condition = (precedence and (proximity or dist < 0.0)) or (
                proximity and best_distance > 0.0
            )

            if (dist < best_distance and (not is_media_different)) or (
                condition and is_media_different
            ):
                previous_media_id = this_media_id
                best_distance = dist
                closest_atom_or_object_index = atom_or_object_index

    return closest_atom_or_object_index, previous_media_id, best_distance


@njit(cache=True, nogil=True, parallel=True)
def _scale_initial_boundary_points_parallel(
    num_threads,
    num_boundary_points,
    surface_charge_positions,
    atom_data,
    atom_plus_probe_radii,
    atom_plus_probe_radii_squared_shrunk,
    voxel_atom_indices,
    voxel_atom_count,
    voxel_atom_count_cumulative,
    voxel_space_origin,
    voxel_space_scale,
    voxel_space_shape: tuple,
    atom_accessibility,
    num_atoms,
    num_molecules,
    discrete_epsilon_index_map_1d,
    epsilon_dimension,
    x_stride,
    y_stride,
    x_stride_x_3,
    y_stride_x_3,
    current_grid_origin,
    grid_spacing,
    max_probe_radius,
    atom_surface_index,
    atom_index_for_boundary,
    scaled_surface_normal_vectors,
    debug_level,
    verbosity_level,
    exit_status_flag,
    delphi_real,
    delphi_int,
):
    nt = num_threads
    if nt <= 0:
        nt = 1

    chunk_size = (num_boundary_points + nt - 1) // nt

    thread_counts = np.zeros(nt, dtype=delphi_int)
    thread_error_flags = np.zeros(nt, dtype=delphi_int)

    for thread_id in prange(nt):
        start = thread_id * chunk_size
        end = start + chunk_size
        if end > num_boundary_points:
            end = num_boundary_points

        local_count = delphi_int(0)
        local_error = delphi_int(0)

        for bgp_index in range(start, end):
            # temps preserve current values unless successfully updated
            temp_pos_x = surface_charge_positions[bgp_index, 0]
            temp_pos_y = surface_charge_positions[bgp_index, 1]
            temp_pos_z = surface_charge_positions[bgp_index, 2]

            temp_norm_x = scaled_surface_normal_vectors[bgp_index, 0]
            temp_norm_y = scaled_surface_normal_vectors[bgp_index, 1]
            temp_norm_z = scaled_surface_normal_vectors[bgp_index, 2]

            temp_atom_index_for_boundary = atom_index_for_boundary[bgp_index]
            current_atom_surface_index = atom_surface_index[bgp_index]

            # --- Internal boundary check ---
            is_internal_bgp = False
            if (num_boundary_points != num_atoms) and (num_molecules > 1):
                gi = int(surface_charge_positions[bgp_index, 0])
                gj = int(surface_charge_positions[bgp_index, 1])
                gk = int(surface_charge_positions[bgp_index, 2])
                grid_index_1d = gi * x_stride + gj * y_stride + gk
                is_internal_bgp = _is_internal_boundary(
                    discrete_epsilon_index_map_1d,
                    grid_index_1d,
                    epsilon_dimension,
                    num_atoms,
                    x_stride_x_3,
                    y_stride_x_3,
                )

            # --- Cartesian coordinates (elementwise) ---
            gp0 = temp_pos_x * grid_spacing + current_grid_origin[0]
            gp1 = temp_pos_y * grid_spacing + current_grid_origin[1]
            gp2 = temp_pos_z * grid_spacing + current_grid_origin[2]

            # --- voxel index (elementwise) ---
            ci0 = int((gp0 - voxel_space_origin[0]) * voxel_space_scale)
            ci1 = int((gp1 - voxel_space_origin[1]) * voxel_space_scale)
            ci2 = int((gp2 - voxel_space_origin[2]) * voxel_space_scale)

            if (
                (ci0 < 0)
                or (ci0 >= voxel_space_shape[0])
                or (ci1 < 0)
                or (ci1 >= voxel_space_shape[1])
                or (ci2 < 0)
                or (ci2 >= voxel_space_shape[2])
            ):
                lower_limit = 0
                upper_limit = -1
            else:
                lower_limit = voxel_atom_count[ci0, ci1, ci2]
                upper_limit = voxel_atom_count_cumulative[ci0, ci1, ci2]

            # --- Internal boundary handling (unchanged logic, list-free already) ---
            if is_internal_bgp:
                min_distance_squared_1 = 1000.0
                min_distance_squared_2 = 1000.0
                closest_atom_index_1 = 0
                closest_atom_index_2 = 0

                for kk in range(lower_limit, upper_limit + 1):
                    atom_or_object_index = voxel_atom_indices[kk]
                    if atom_or_object_index <= num_atoms:
                        ax = atom_data[atom_or_object_index - 1, ATOMFIELD_X]
                        ay = atom_data[atom_or_object_index - 1, ATOMFIELD_Y]
                        az = atom_data[atom_or_object_index - 1, ATOMFIELD_Z]
                        dx = gp0 - ax
                        dy = gp1 - ay
                        dz = gp2 - az
                        ar = atom_data[atom_or_object_index - 1, ATOMFIELD_RADIUS]
                        dist2 = dx * dx + dy * dy + dz * dz - ar * ar

                        if dist2 < min_distance_squared_1:
                            closest_atom_index_2 = closest_atom_index_1
                            min_distance_squared_2 = min_distance_squared_1
                            closest_atom_index_1 = atom_or_object_index
                            min_distance_squared_1 = dist2
                        else:
                            if dist2 <= min_distance_squared_2:
                                closest_atom_index_2 = atom_or_object_index
                                min_distance_squared_2 = dist2

                current_atom_surface_index = closest_atom_index_1

                if (
                    (closest_atom_index_1 == 0)
                    or (closest_atom_index_2 == 0)
                    or (closest_atom_index_1 == closest_atom_index_2)
                ):
                    local_error = exit_status_flag
                    temp_atom_index_for_boundary = 0
                    temp_norm_x = 0.0
                    temp_norm_y = 0.0
                    temp_norm_z = 0.0
                else:
                    temp_atom_index_for_boundary = -1

                    a2x = atom_data[closest_atom_index_2 - 1, ATOMFIELD_X]
                    a2y = atom_data[closest_atom_index_2 - 1, ATOMFIELD_Y]
                    a2z = atom_data[closest_atom_index_2 - 1, ATOMFIELD_Z]
                    a1x = atom_data[closest_atom_index_1 - 1, ATOMFIELD_X]
                    a1y = atom_data[closest_atom_index_1 - 1, ATOMFIELD_Y]
                    a1z = atom_data[closest_atom_index_1 - 1, ATOMFIELD_Z]

                    dcx = a2x - a1x
                    dcy = a2y - a1y
                    dcz = a2z - a1z
                    denom = dcx * dcx + dcy * dcy + dcz * dcz

                    if denom == 0.0:
                        local_error = exit_status_flag
                        temp_atom_index_for_boundary = 0
                    else:
                        temp_factor = (
                            0.5
                            * (min_distance_squared_2 - min_distance_squared_1)
                            / denom
                        )
                        temp_pos_x = gp0 + temp_factor * dcx
                        temp_pos_y = gp1 + temp_factor * dcy
                        temp_pos_z = gp2 + temp_factor * dcz
                        temp_norm_x = 0.0
                        temp_norm_y = 0.0
                        temp_norm_z = 0.0
                        # local_count = local_count + 1

            # --- External boundary handling (list-free scan) ---
            if (not is_internal_bgp) and (local_error == 0):
                closest_atom_or_object_index = 0
                previous_media_id = 0

                closest_atom_or_object_index, previous_media_id, _best_dist = (
                    _scan_voxel_for_closest_atom(
                        gp0,
                        gp1,
                        gp2,
                        lower_limit,
                        upper_limit,
                        voxel_atom_indices,
                        atom_data,
                        atom_accessibility,
                        num_atoms,
                        previous_media_id,
                        closest_atom_or_object_index,
                        delphi_real(100.0),
                        delphi_real,
                        delphi_int,
                    )
                )

                current_atom_surface_index = closest_atom_or_object_index

                if closest_atom_or_object_index == 0:
                    local_error = exit_status_flag
                    temp_atom_index_for_boundary = 0
                else:
                    # distance vector to atom center
                    ax = atom_data[closest_atom_or_object_index - 1, ATOMFIELD_X]
                    ay = atom_data[closest_atom_or_object_index - 1, ATOMFIELD_Y]
                    az = atom_data[closest_atom_or_object_index - 1, ATOMFIELD_Z]
                    drx = gp0 - ax
                    dry = gp1 - ay
                    drz = gp2 - az
                    distance_to_surface = math.sqrt(drx * drx + dry * dry + drz * drz)

                    is_outside = True
                    if max_probe_radius > 0.0:
                        ar = atom_plus_probe_radii[closest_atom_or_object_index - 1]
                        ux = ax + (ar * drx / distance_to_surface)
                        uy = ay + (ar * dry / distance_to_surface)
                        uz = az + (ar * drz / distance_to_surface)

                        is_outside = _is_solvent_exposed(
                            ux,
                            uy,
                            uz,
                            num_atoms,
                            atom_data,
                            voxel_atom_indices,
                            voxel_atom_count,
                            voxel_atom_count_cumulative,
                            voxel_space_origin,
                            voxel_space_scale,
                            voxel_space_shape[0],
                            voxel_space_shape[1],
                            voxel_space_shape[2],
                            atom_plus_probe_radii_squared_shrunk,
                            delphi_real,
                        )

                    if is_outside:
                        atom_radius = atom_data[
                            closest_atom_or_object_index - 1, ATOMFIELD_RADIUS
                        ]
                        scale_factor = atom_radius / distance_to_surface

                        # scale point to vdW surface
                        temp_pos_x = ax + drx * scale_factor
                        temp_pos_y = ay + dry * scale_factor
                        temp_pos_z = az + drz * scale_factor

                        # normals
                        temp_norm_x = drx / distance_to_surface
                        temp_norm_y = dry / distance_to_surface
                        temp_norm_z = drz / distance_to_surface

                        temp_atom_index_for_boundary = closest_atom_or_object_index
                        local_count = local_count + 1
                    else:
                        temp_atom_index_for_boundary = 0

            # --- Final unconditional write-back ---
            atom_surface_index[bgp_index] = current_atom_surface_index
            atom_index_for_boundary[bgp_index] = temp_atom_index_for_boundary

            surface_charge_positions[bgp_index, 0] = temp_pos_x
            surface_charge_positions[bgp_index, 1] = temp_pos_y
            surface_charge_positions[bgp_index, 2] = temp_pos_z

            scaled_surface_normal_vectors[bgp_index, 0] = temp_norm_x
            scaled_surface_normal_vectors[bgp_index, 1] = temp_norm_y
            scaled_surface_normal_vectors[bgp_index, 2] = temp_norm_z

        thread_counts[thread_id] = local_count
        thread_error_flags[thread_id] = local_error

    return thread_counts, thread_error_flags


@cuda.jit(device=True, inline=True)
def _is_internal_boundary_dev(
    discrete_epsilon_map_1d_index,
    grid_index_1d,
    epsilon_dimension,
    num_atoms,
    x_stride_x_3,
    y_stride_x_3,
):
    # Local fixed arrays: small only
    neigh_entity = cuda.local.array(6, dtype=int32)
    neigh_media = cuda.local.array(6, dtype=int32)

    grid_index_1d_x_3 = grid_index_1d * 3
    z_stride_x_3 = 3  # since z_stride = 1 then *3

    # offsets for 6 midpoints
    # [0,1,2] = leading x,y,z
    # [3,4,5] = lagging x,y,z (with your specific encoding)
    offsets = cuda.local.array(6, dtype=int32)
    offsets[0] = 0
    offsets[1] = 1
    offsets[2] = 2
    offsets[3] = -x_stride_x_3
    offsets[4] = -y_stride_x_3 + 1
    offsets[5] = -z_stride_x_3 + 2

    for t in range(6):
        entity_index = grid_index_1d_x_3 + offsets[t]
        value = discrete_epsilon_map_1d_index[entity_index]
        neigh_entity[t] = value % epsilon_dimension
        neigh_media[t] = value // epsilon_dimension

    # check discontinuities in cyclic adjacent pairs
    is_internal = False
    for m in range(6):
        n = (m + 1) % 6
        mm = neigh_media[m]
        mn = neigh_media[n]
        if mm != mn and (mm * mn) != 0:
            em = neigh_entity[m]
            en = neigh_entity[n]
            if (em <= num_atoms + 1) and (en <= num_atoms + 1):
                is_internal = True
    return is_internal


@cuda.jit(device=True, inline=True)
def _is_solvent_exposed_dev(
    u_x,
    u_y,
    u_z,
    num_atoms,
    atom_data,
    voxel_atom_indices,
    atoms_per_voxel_count,
    cumulative_atoms_per_voxel,
    cube_vertex_min_xyz,
    inverse_cube_side_length,
    cube_nx,
    cube_ny,
    cube_nz,
    shrunk_atom_plus_probe_radii_squared,
):
    # voxel coords
    ci0 = int32((u_x - cube_vertex_min_xyz[0]) * inverse_cube_side_length)
    ci1 = int32((u_y - cube_vertex_min_xyz[1]) * inverse_cube_side_length)
    ci2 = int32((u_z - cube_vertex_min_xyz[2]) * inverse_cube_side_length)

    if (
        (ci0 < 0)
        or (ci0 >= cube_nx)
        or (ci1 < 0)
        or (ci1 >= cube_ny)
        or (ci2 < 0)
        or (ci2 >= cube_nz)
    ):
        return True  # outside => exposed

    lower = atoms_per_voxel_count[ci0, ci1, ci2]
    upper = cumulative_atoms_per_voxel[ci0, ci1, ci2]

    for kk in range(lower, upper + 1):  # inclusive upper (as in your CPU)
        atom_idx = voxel_atom_indices[kk]
        if atom_idx <= num_atoms:
            ax = atom_data[atom_idx - 1, ATOMFIELD_X]
            ay = atom_data[atom_idx - 1, ATOMFIELD_Y]
            az = atom_data[atom_idx - 1, ATOMFIELD_Z]
            dx = u_x - ax
            dy = u_y - ay
            dz = u_z - az
            dist2 = dx * dx + dy * dy + dz * dz
            if dist2 < shrunk_atom_plus_probe_radii_squared[atom_idx - 1]:
                return False
    return True


@cuda.jit(device=True, inline=True)
def _update_closest_external_candidate(
    gp0,
    gp1,
    gp2,
    num_atoms,
    atom_data,
    atom_or_object_index,
    prev_media_id,
    closest_idx,
    min_dist,
    # returns updated
):
    # This is the “inlined _process_neighbors” logic without a neighbor array.
    # It only updates when atom_or_object_index is an atom, same as your original.
    if atom_or_object_index <= num_atoms:
        # media id
        this_media = int32(atom_data[atom_or_object_index - 1, ATOMFIELD_MEDIA_ID])
        is_media_different = this_media != prev_media_id

        ax = atom_data[atom_or_object_index - 1, ATOMFIELD_X]
        ay = atom_data[atom_or_object_index - 1, ATOMFIELD_Y]
        az = atom_data[atom_or_object_index - 1, ATOMFIELD_Z]
        ar = atom_data[atom_or_object_index - 1, ATOMFIELD_RADIUS]

        dx = gp0 - ax
        dy = gp1 - ay
        dz = gp2 - az
        # distance-to-surface (signed)
        dist_center = math.sqrt(dx * dx + dy * dy + dz * dz)
        dist = dist_center - ar

        # precedence / proximity logic (ported)
        precedence = (atom_or_object_index > closest_idx) or (closest_idx > num_atoms)
        proximity = abs(dist) < abs(min_dist)
        cond = (precedence and (proximity or dist < 0.0)) or (
            proximity and (min_dist > 0.0)
        )

        if ((dist < min_dist) and (not is_media_different)) or (
            cond and is_media_different
        ):
            closest_idx = atom_or_object_index
            prev_media_id = this_media
            min_dist = dist

    return closest_idx, prev_media_id, min_dist


# ---------------------------------------
# Main CUDA kernel
# ---------------------------------------


@cuda.jit(cache=True)
def _cuda_scale_initial_boundary_points(
    num_boundary_points,
    surface_charge_positions,
    atom_data,
    atom_plus_probe_radii,
    atom_plus_probe_radii_squared_shrunk,
    voxel_atom_indices,
    voxel_atom_count,
    voxel_atom_count_cumulative,
    voxel_space_origin,
    voxel_space_scale,
    voxel_space_shape0,
    voxel_space_shape1,
    voxel_space_shape2,
    atom_accessibility,
    num_atoms,
    num_molecules,
    discrete_epsilon_index_map_1d,
    epsilon_dimension,
    x_stride,
    y_stride,
    x_stride_x_3,
    y_stride_x_3,
    current_grid_origin,
    grid_spacing,
    max_probe_radius,
    atom_surface_index,
    atom_index_for_boundary,
    scaled_surface_normal_vectors,
    # outputs for host reduction:
    scaled_flag,  # uint8 or int32, length num_boundary_points
    error_flag,  # int32, length num_boundary_points (0 ok / EXIT_NJIT_FLAG)
):
    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    for bgp_index in range(tid, num_boundary_points, stride):

        # defaults
        error_flag[bgp_index] = 0
        scaled_flag[bgp_index] = 0

        # Load current grid coords (these are the “unscaled” coords in your CPU code)
        gi = surface_charge_positions[bgp_index, 0]
        gj = surface_charge_positions[bgp_index, 1]
        gk = surface_charge_positions[bgp_index, 2]

        # internal boundary check
        is_internal = False
        if (num_boundary_points != num_atoms) and (num_molecules > 1):
            grid_index_1d = int32(gi) * x_stride + int32(gj) * y_stride + int32(gk)
            is_internal = _is_internal_boundary_dev(
                discrete_epsilon_index_map_1d,
                grid_index_1d,
                epsilon_dimension,
                num_atoms,
                x_stride_x_3,
                y_stride_x_3,
            )

        # Cartesian gp
        gp0 = gi * grid_spacing + current_grid_origin[0]
        gp1 = gj * grid_spacing + current_grid_origin[1]
        gp2 = gk * grid_spacing + current_grid_origin[2]

        # cube index
        ci0 = int32((gp0 - voxel_space_origin[0]) * voxel_space_scale)
        ci1 = int32((gp1 - voxel_space_origin[1]) * voxel_space_scale)
        ci2 = int32((gp2 - voxel_space_origin[2]) * voxel_space_scale)

        if (
            (ci0 < 0)
            or (ci0 >= voxel_space_shape0)
            or (ci1 < 0)
            or (ci1 >= voxel_space_shape1)
            or (ci2 < 0)
            or (ci2 >= voxel_space_shape2)
        ):
            lower = 0
            upper = -1
        else:
            lower = voxel_atom_count[ci0, ci1, ci2]
            upper = voxel_atom_count_cumulative[ci0, ci1, ci2]

        # Scan voxel list once.
        # For internal: track two closest atoms (by distance_squared to surface)
        min1 = 1.0e30
        min2 = 1.0e30
        a1 = int32(0)
        a2 = int32(0)

        # For external: track closest candidate using your precedence/media logic
        closest_idx = int32(0)
        prev_media_id = int32(0)
        min_dist = 1.0e30

        prev_object_index = int32(0)

        for kk in range(lower, upper + 1):
            idx = voxel_atom_indices[kk]

            if idx <= num_atoms:
                if is_internal:
                    ax = atom_data[idx - 1, ATOMFIELD_X]
                    ay = atom_data[idx - 1, ATOMFIELD_Y]
                    az = atom_data[idx - 1, ATOMFIELD_Z]
                    ar = atom_data[idx - 1, ATOMFIELD_RADIUS]
                    dx = gp0 - ax
                    dy = gp1 - ay
                    dz = gp2 - az
                    dsq = dx * dx + dy * dy + dz * dz - ar * ar

                    if dsq < min1:
                        a2 = a1
                        min2 = min1
                        a1 = idx
                        min1 = dsq
                    elif dsq <= min2:
                        a2 = idx
                        min2 = dsq
                else:
                    # external: only consider "accessible==0" atoms (same as your CPU path)
                    if atom_accessibility[idx] == 0:
                        closest_idx, prev_media_id, min_dist = (
                            _update_closest_external_candidate(
                                gp0,
                                gp1,
                                gp2,
                                num_atoms,
                                atom_data,
                                idx,
                                prev_media_id,
                                closest_idx,
                                min_dist,
                            )
                        )
            else:
                # objects: preserve uniqueness like CPU (if needed later)
                if idx != prev_object_index:
                    prev_object_index = idx
                    # If you later need object handling, extend here.

        # Write atom_surface_index always (mirror CPU)
        if is_internal:
            atom_surface_index[bgp_index] = a1
        else:
            atom_surface_index[bgp_index] = closest_idx

        # Internal boundary action
        if is_internal:
            if (a1 == 0) or (a2 == 0) or (a1 == a2):
                error_flag[bgp_index] = EXIT_NJIT_FLAG
                atom_index_for_boundary[bgp_index] = 0
                scaled_surface_normal_vectors[bgp_index, 0] = 0.0
                scaled_surface_normal_vectors[bgp_index, 1] = 0.0
                scaled_surface_normal_vectors[bgp_index, 2] = 0.0
                continue

            atom_index_for_boundary[bgp_index] = -1

            a2x = atom_data[a2 - 1, ATOMFIELD_X]
            a2y = atom_data[a2 - 1, ATOMFIELD_Y]
            a2z = atom_data[a2 - 1, ATOMFIELD_Z]
            a1x = atom_data[a1 - 1, ATOMFIELD_X]
            a1y = atom_data[a1 - 1, ATOMFIELD_Y]
            a1z = atom_data[a1 - 1, ATOMFIELD_Z]

            dcx = a2x - a1x
            dcy = a2y - a1y
            dcz = a2z - a1z
            denom = dcx * dcx + dcy * dcy + dcz * dcz
            if denom == 0.0:
                error_flag[bgp_index] = EXIT_NJIT_FLAG
                atom_index_for_boundary[bgp_index] = 0
                continue

            factor = 0.5 * (min2 - min1) / denom

            # scaled position stored into surface_charge_positions
            surface_charge_positions[bgp_index, 0] = gp0 + factor * dcx
            surface_charge_positions[bgp_index, 1] = gp1 + factor * dcy
            surface_charge_positions[bgp_index, 2] = gp2 + factor * dcz

            scaled_surface_normal_vectors[bgp_index, 0] = 0.0
            scaled_surface_normal_vectors[bgp_index, 1] = 0.0
            scaled_surface_normal_vectors[bgp_index, 2] = 0.0

            scaled_flag[bgp_index] = 1
            continue

        # External boundary action
        if (closest_idx == 0) and (a1 == 0):
            error_flag[bgp_index] = EXIT_NJIT_FLAG
            atom_index_for_boundary[bgp_index] = 0
            continue

        if closest_idx <= num_atoms:
            ax = atom_data[closest_idx - 1, ATOMFIELD_X]
            ay = atom_data[closest_idx - 1, ATOMFIELD_Y]
            az = atom_data[closest_idx - 1, ATOMFIELD_Z]
            drx = gp0 - ax
            dry = gp1 - ay
            drz = gp2 - az
        else:
            drx = 0.0
            dry = 0.0
            drz = 0.0

        dist = math.sqrt(drx * drx + dry * dry + drz * drz)
        if dist == 0.0:
            # avoid nan scaling
            error_flag[bgp_index] = EXIT_NJIT_FLAG
            atom_index_for_boundary[bgp_index] = 0
            continue

        is_outside = True
        if max_probe_radius > 0.0:
            if closest_idx <= num_atoms:
                ar = atom_plus_probe_radii[closest_idx - 1]
                ux = ax + (ar * drx / dist)
                uy = ay + (ar * dry / dist)
                uz = az + (ar * drz / dist)
            else:
                ux = gp0 - drx
                uy = gp1 - dry
                uz = gp2 - drz

            is_outside = _is_solvent_exposed_dev(
                ux,
                uy,
                uz,
                num_atoms,
                atom_data,
                voxel_atom_indices,
                voxel_atom_count,
                voxel_atom_count_cumulative,
                voxel_space_origin,
                voxel_space_scale,
                voxel_space_shape0,
                voxel_space_shape1,
                voxel_space_shape2,
                atom_plus_probe_radii_squared_shrunk,
            )

        if is_outside and (closest_idx <= num_atoms):
            atom_radius = atom_data[closest_idx - 1, ATOMFIELD_RADIUS]
            scale = atom_radius / dist

            surface_charge_positions[bgp_index, 0] = ax + drx * scale
            surface_charge_positions[bgp_index, 1] = ay + dry * scale
            surface_charge_positions[bgp_index, 2] = az + drz * scale

            scaled_surface_normal_vectors[bgp_index, 0] = drx / dist
            scaled_surface_normal_vectors[bgp_index, 1] = dry / dist
            scaled_surface_normal_vectors[bgp_index, 2] = drz / dist

            atom_index_for_boundary[bgp_index] = closest_idx
            scaled_flag[bgp_index] = 1
        else:
            atom_index_for_boundary[bgp_index] = 0
            # scaled_flag remains 0


def scale_initial_vdw_surface_boundary_points(
    use_cuda: bool,
    num_threads,  # CPU threads
    num_cuda_threads,  # CUDA threads-per-block (preferred)
    num_boundary_points,
    surface_charge_positions,
    atom_data,
    atom_plus_probe_radii,
    atom_plus_probe_radii_squared_shrunk,
    voxel_atom_indices,
    voxel_atom_count,
    voxel_atom_count_cumulative,
    voxel_space_origin,
    voxel_space_scale,
    voxel_space_shape,  # tuple/list len=3
    atom_accessibility,
    num_atoms,
    num_molecules,
    discrete_epsilon_index_map_1d,
    epsilon_dimension,
    x_stride,
    y_stride,
    x_stride_x_3,
    y_stride_x_3,
    current_grid_origin,
    grid_spacing,
    max_probe_radius,
    atom_surface_index,
    atom_index_for_boundary,
    scaled_surface_normal_vectors,
    debug_level,
    verbosity_level,
    exit_status_flag,
    delphi_real,
    delphi_int,
):
    n = int(num_boundary_points)
    if n <= 0:
        return 0, 0

    # CPU thread count
    nt = int(num_threads) if (num_threads is not None) else 1
    if nt <= 0:
        nt = 1

    # ============================================================
    # CUDA BRANCH (INLINE of your tested CUDA setup + launch)
    # ============================================================
    if use_cuda:
        # threads-per-block
        tpb = int(num_cuda_threads) if (num_cuda_threads is not None) else 256
        if tpb <= 0:
            tpb = 256
        # Optional safety clamp (Numba supports up to 1024 on most GPUs)
        if tpb < 32:
            tpb = 32
        if tpb > 1024:
            tpb = 1024

        blocks = (n + tpb - 1) // tpb

        # -----------------------------
        # Device transfers (DEFINE d_*)
        # -----------------------------
        d_surface_charge_positions = cuda.to_device(surface_charge_positions)
        d_atom_data = cuda.to_device(atom_data)
        d_atom_plus_probe_radii = cuda.to_device(atom_plus_probe_radii)
        d_atom_plus_probe_radii_squared_shrunk = cuda.to_device(
            atom_plus_probe_radii_squared_shrunk
        )

        d_voxel_atom_indices = cuda.to_device(voxel_atom_indices)
        d_voxel_atom_count = cuda.to_device(voxel_atom_count)
        d_voxel_atom_count_cumulative = cuda.to_device(voxel_atom_count_cumulative)

        d_voxel_space_origin = cuda.to_device(voxel_space_origin)
        d_atom_accessibility = cuda.to_device(atom_accessibility)

        d_discrete_epsilon_index_map_1d = cuda.to_device(discrete_epsilon_index_map_1d)
        d_current_grid_origin = cuda.to_device(current_grid_origin)

        # outputs that are mutated by kernel
        d_atom_surface_index = cuda.to_device(atom_surface_index)
        d_atom_index_for_boundary = cuda.to_device(atom_index_for_boundary)
        d_scaled_surface_normal_vectors = cuda.to_device(scaled_surface_normal_vectors)

        # per-bgp flags for host reduction
        scaled_flag = np.zeros(n, dtype=np.uint8)
        error_flag = np.zeros(n, dtype=np.int32)
        d_scaled_flag = cuda.to_device(scaled_flag)
        d_error_flag = cuda.to_device(error_flag)

        # -----------------------------
        # Kernel launch
        # -----------------------------
        _cuda_scale_initial_boundary_points[int(blocks), int(tpb)](
            n,
            d_surface_charge_positions,
            d_atom_data,
            d_atom_plus_probe_radii,
            d_atom_plus_probe_radii_squared_shrunk,
            d_voxel_atom_indices,
            d_voxel_atom_count,
            d_voxel_atom_count_cumulative,
            d_voxel_space_origin,
            voxel_space_scale,
            int(voxel_space_shape[0]),
            int(voxel_space_shape[1]),
            int(voxel_space_shape[2]),
            d_atom_accessibility,
            int(num_atoms),
            int(num_molecules),
            d_discrete_epsilon_index_map_1d,
            int(epsilon_dimension),
            int(x_stride),
            int(y_stride),
            int(x_stride_x_3),
            int(y_stride_x_3),
            d_current_grid_origin,
            delphi_real(grid_spacing),
            delphi_real(max_probe_radius),
            d_atom_surface_index,
            d_atom_index_for_boundary,
            d_scaled_surface_normal_vectors,
            d_scaled_flag,
            d_error_flag,
        )
        cuda.synchronize()

        # -----------------------------
        # Copy back
        # -----------------------------
        d_surface_charge_positions.copy_to_host(surface_charge_positions)
        d_atom_surface_index.copy_to_host(atom_surface_index)
        d_atom_index_for_boundary.copy_to_host(atom_index_for_boundary)
        d_scaled_surface_normal_vectors.copy_to_host(scaled_surface_normal_vectors)

        error_flag = d_error_flag.copy_to_host()
        if np.any(error_flag != 0):
            return exit_status_flag, 0

        scaled_flag = d_scaled_flag.copy_to_host()
        scaled_count = int(scaled_flag.sum())

        d_surface_charge_positions = None
        d_atom_data = None
        d_atom_plus_probe_radii = None
        d_atom_plus_probe_radii_squared_shrunk = None
        d_voxel_atom_indices = None
        d_voxel_atom_count = None
        d_voxel_atom_count_cumulative = None
        d_voxel_space_origin = None
        d_atom_accessibility = None
        d_discrete_epsilon_index_map_1d = None
        d_current_grid_origin = None

        d_atom_surface_index = None
        d_atom_index_for_boundary = None
        d_scaled_surface_normal_vectors = None
        d_scaled_flag = None
        d_error_flag = None

        gc.collect()
        return 0, scaled_count

    # ============================================================
    # CPU PARALLEL BRANCH
    # ============================================================
    elif nt > 1:
        set_num_threads(nt)
        thread_counts, thread_error_flags = _scale_initial_boundary_points_parallel(
            nt,
            n,
            surface_charge_positions,
            atom_data,
            atom_plus_probe_radii,
            atom_plus_probe_radii_squared_shrunk,
            voxel_atom_indices,
            voxel_atom_count,
            voxel_atom_count_cumulative,
            voxel_space_origin,
            voxel_space_scale,
            voxel_space_shape,
            atom_accessibility,
            num_atoms,
            num_molecules,
            discrete_epsilon_index_map_1d,
            epsilon_dimension,
            x_stride,
            y_stride,
            x_stride_x_3,
            y_stride_x_3,
            current_grid_origin,
            grid_spacing,
            max_probe_radius,
            atom_surface_index,
            atom_index_for_boundary,
            scaled_surface_normal_vectors,
            debug_level,
            verbosity_level,
            exit_status_flag,
            delphi_real,
            delphi_int,
        )

        # scalarize error + reduce here
        if np.any(thread_error_flags != 0):
            return exit_status_flag, 0

        scaled_count = int(np.sum(thread_counts))
        return 0, scaled_count

    # ============================================================
    # CPU SERIAL (REFERENCE) BRANCH
    # ============================================================
    else:
        return _scale_initial_boundary_points(
            n,
            surface_charge_positions,
            atom_data,
            atom_plus_probe_radii,
            atom_plus_probe_radii_squared_shrunk,
            voxel_atom_indices,
            voxel_atom_count,
            voxel_atom_count_cumulative,
            voxel_space_origin,
            voxel_space_scale,
            voxel_space_shape,
            atom_accessibility,
            num_atoms,
            num_molecules,
            discrete_epsilon_index_map_1d,
            epsilon_dimension,
            x_stride,
            y_stride,
            x_stride_x_3,
            y_stride_x_3,
            current_grid_origin,
            grid_spacing,
            max_probe_radius,
            atom_surface_index,
            atom_index_for_boundary,
            scaled_surface_normal_vectors,
            delphi_real,
            delphi_int,
        )
