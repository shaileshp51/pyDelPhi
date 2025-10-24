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

# Determine the floating-point precision to use based on the DelphiPrecision setting.
if PRECISION.value == Precision.SINGLE.value:
    from pydelphi.utils.prec.single import dot_product

elif PRECISION.value == Precision.DOUBLE.value:
    from pydelphi.utils.prec.double import dot_product

from pydelphi.constants import (
    ConstDelPhiFloats,
    ConstDelPhiInts,
    ATOMFIELD_X,
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

import pydelphi.space.core.voxelizer as voxelizer
import pydelphi.space.core.vdw.sas as sas


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

    Args:
        discrete_epsilon_map_1d_index (np.ndarray): 1D array representing the discrete epsilon map.
        grid_index_1d (delphi_int): The 1D index of the grid point.
        epsilon_dimension (delphi_int): The total number of dielectric regions.
        num_atoms (delphi_int): The number of atoms in the system.
        x_stride_x_3 (delphi_int): Stride used to access neighbors in the x-direction within the 1D epsilon map.
            Multiplied by 3 to account for the three +h/2 leading midpoints corresponding to each grid point.
        y_stride_x_3 (delphi_int): Stride used to access neighbors in the y-direction within the 1D epsilon map.
            Multiplied by 3 to account for the three +h/2 leading midpoints corresponding to each grid point.

    Returns:
        delphi_bool: True if the grid point is on an internal boundary, False otherwise.
    """
    grid_neighs_entity_ids = np.zeros(7, dtype=delphi_int)
    grid_neighs_media_ids = np.zeros(7, dtype=delphi_int)
    grid_index_1d_x_3 = grid_index_1d * 3
    z_stride = 1

    grid_neighs_1d_offsets = np.zeros(7, dtype=delphi_int)
    # Leading by h/2 neighbors in x, y, z directions
    grid_neighs_1d_offsets[1:4] = 0, 1, 2
    # Lagging by h/2 neighbors in x, y, z directions
    grid_neighs_1d_offsets[4:7] = -x_stride_x_3, -y_stride_x_3 + 1, -z_stride * 3 + 2

    # Iterate through the neighboring grid points.
    for neigh_id in range(1, 7):
        # Calculate the entity ID of the neighbor.
        grid_neighs_entity_ids[neigh_id] = (
            discrete_epsilon_map_1d_index[
                grid_index_1d_x_3 + grid_neighs_1d_offsets[neigh_id]
            ]
            % epsilon_dimension
        )
        # Calculate the media ID of the neighbor.
        grid_neighs_media_ids[neigh_id] = (
            discrete_epsilon_map_1d_index[
                grid_index_1d_x_3 + grid_neighs_1d_offsets[neigh_id]
            ]
            // epsilon_dimension
        )

    is_internal_boundary_grid_point = False
    # Iterate through pairs of neighboring midpoints to check for dielectric discontinuities.
    for midpoint_index in range(1, 7):
        neighbor_index = (
            midpoint_index % 6
        ) + 1  # Ensures (1,6), (2,3), (3,4), ..., (6,1)

        # Check if the media IDs of the two midpoints are different and neither is zero (solvent).
        if (
            grid_neighs_media_ids[midpoint_index]
            != grid_neighs_media_ids[neighbor_index]
            and grid_neighs_media_ids[midpoint_index]
            * grid_neighs_media_ids[neighbor_index]
            != 0
        ):
            # If the media IDs are different and non-zero, check if both entities belong to atoms (or are very close to atoms).
            is_internal_boundary_grid_point |= (
                grid_neighs_entity_ids[midpoint_index] <= num_atoms + 1
                and grid_neighs_entity_ids[neighbor_index] <= num_atoms + 1
            )
    return is_internal_boundary_grid_point


@njit(nogil=True, boundscheck=False, cache=True)
def _process_neighbors(
    grid_point_coords: np.ndarray,
    num_atoms: delphi_int,
    atom_data: np.ndarray,
    neighboring_atom_indices: np.ndarray,
    neighbor_count: delphi_int,
    previous_media_id: delphi_int,
    closest_atom_or_object_index: delphi_int,
    min_distance_squared: delphi_real,
    delphi_real_type: type,
    delphi_int_type: type,
) -> tuple[delphi_int, delphi_int, delphi_real]:
    """Finds the closest atom or object to a given grid point.

    Args:
        grid_point_coords (np.ndarray): Coordinates of the grid point.
        num_atoms (delphi_int): The number of atoms in the system.
        atom_data (np.ndarray): Array containing atom properties.
        neighboring_atom_indices (np.ndarray): Array to store indices of neighboring atoms.
        neighbor_count (delphi_int): The number of neighboring atoms found so far.
        previous_media_id (delphi_int): The media ID of the previously considered closest entity.
        closest_atom_or_object_index (delphi_int): Index of the closest atom or object found so far.
        min_distance_squared (delphi_real): Minimum distance squared found so far.
        delphi_real_type (type): Data type for real numbers.
        delphi_int_type (type): Data type for integers.

    Returns:
        delphi_int: Index of the closest atom or object.
        delphi_int: Media ID of the closest atom or object.
        delphi_real: Minimum distance squared to the closest atom or object.
    """
    distance = delphi_real_type(0.0)
    for ii in range(1, neighbor_count + 1):
        if ii >= SPACE_NBRA_MAX_SIZE:
            nprint_cpu(
                DEBUG,
                _VERBOSITY,
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
            # Calculate the vector from the atom center to the grid point.
            delta_coords = (
                grid_point_coords
                - atom_data[atom_or_object_index - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
            ).astype(delphi_real_type)
            # Calculate the distance from the grid point to the atom surface.
            distance = (
                np.sqrt(dot_product(delta_coords, delta_coords))
                - atom_data[atom_or_object_index - 1][ATOMFIELD_RADIUS]
            )
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


@njit(nogil=True, boundscheck=False, cache=True)
def _is_solvent_exposed(
    u123: np.ndarray,
    num_atoms: delphi_int,
    atom_data: np.ndarray,
    voxel_atom_indices: np.ndarray,
    atoms_per_voxel_count: np.ndarray,
    cumulative_atoms_per_voxel: np.ndarray,
    cube_vertex_min_xyz: np.ndarray,
    inverse_cube_side_length: delphi_real,
    cube_shape: tuple,
    shrunk_atom_plus_probe_radii_squared: np.ndarray,
    probe_radius: delphi_real,
    delphi_real_type: type,
    delphi_int_type: type,
) -> delphi_bool:
    """Checks if a given point is solvent-exposed.

    Args:
        u123 (np.ndarray): Coordinates of the point to check.
        num_atoms (delphi_int): The number of atoms in the system.
        atom_data (np.ndarray): Array containing atom properties.
        voxel_atom_indices (np.ndarray): Array mapping voxel indices to atom indices.
        atoms_per_voxel_count (np.ndarray): Array storing the starting index of atoms in each voxel.
        cumulative_atoms_per_voxel (np.ndarray): Array storing the ending index of atoms in each voxel.
        cube_vertex_min_xyz (np.ndarray): Coordinates of the minimum vertex of the spatial partitioning cube.
        inverse_cube_side_length (delphi_real): Inverse of the side length of the spatial partitioning cube.
        cube_shape (tuple): Dimensions of the spatial partitioning cube.
        shrunk_atom_plus_probe_radii_squared (np.ndarray): Squared radii of atoms shrunk by the probe radius.
        probe_radius (delphi_real): Radius of the solvent probe.
        delphi_real_type (type): Data type for real numbers.
        delphi_int_type (type): Data type for integers.

    Returns:
        delphi_bool: True if the point is solvent-exposed, False otherwise.
    """
    is_outside = True
    # Calculate the voxel indices for the given point.
    cube_indices_u = ((u123 - cube_vertex_min_xyz) * inverse_cube_side_length).astype(
        delphi_int_type
    )
    # Check if the voxel indices are within the bounds of the spatial partitioning cube.
    if np.all(cube_indices_u >= 0) and np.all(cube_indices_u <= cube_shape):
        # Get the range of atom indices for the current voxel.
        lower_limit_u = atoms_per_voxel_count[cube_indices_u[0]][cube_indices_u[1]][
            cube_indices_u[2]
        ]
        upper_limit_u = cumulative_atoms_per_voxel[cube_indices_u[0]][
            cube_indices_u[1]
        ][cube_indices_u[2]]
        # Iterate through the atoms in the current voxel.
        for kk in range(lower_limit_u, upper_limit_u + 1):
            atom_or_object_index_u = voxel_atom_indices[kk]
            if atom_or_object_index_u <= num_atoms:
                # Calculate the vector from the atom center to the point.
                delta_coords_u = (
                    u123
                    - atom_data[atom_or_object_index_u - 1][
                        ATOMFIELD_X:ATOMFIELD_CRD_END
                    ]
                ).astype(delphi_real_type)
                # Calculate the squared distance between the point and the atom center.
                distance_squared_u = delphi_real_type(
                    dot_product(delta_coords_u, delta_coords_u)
                )
                # If the squared distance is less than the squared shrunk radius, the point is inside the atom.
                if (
                    distance_squared_u
                    < shrunk_atom_plus_probe_radii_squared[atom_or_object_index_u - 1]
                ):
                    is_outside = False

    return is_outside


@njit(nogil=True, boundscheck=False, cache=True)
def _scale_exposed_point(
    i: delphi_int,
    closest_atom_or_object_index: delphi_int,
    num_atoms: delphi_int,
    grid_point_coords: np.ndarray,
    atom_data: np.ndarray,
    distance_to_surface: delphi_real,
    delta_r: np.ndarray,
    surface_charge_point_positions: np.ndarray,
    scaled_surface_normal_vectors: np.ndarray,
    atom_plus_probe_radii: np.ndarray,
    atom_data_radius_index: delphi_int,
    delphi_real_type: type,
) -> tuple[np.ndarray, np.ndarray]:
    """Scales the position of a solvent-exposed boundary point to the van der Waals surface of the closest atom.

    Args:
        i (delphi_int): Index of the boundary point.
        closest_atom_or_object_index (delphi_int): Index of the closest atom or object.
        num_atoms (delphi_int): The number of atoms in the system.
        grid_point_coords (np.ndarray): Original coordinates of the boundary point.
        atom_data (np.ndarray): Array containing atom properties.
        distance_to_surface (delphi_real): Distance from the atom center to the original boundary point.
        delta_r (np.ndarray): Vector from the atom center to the original boundary point.
        surface_charge_point_positions (np.ndarray): Array to store the scaled positions of boundary points.
        scaled_surface_normal_vectors (np.ndarray): Array to store the scaled surface normal vectors.
        atom_plus_probe_radii (np.ndarray): Array containing the sum of atom radii and probe radius.
        atom_data_radius_index (delphi_int): Index of the atom radius in the atom_data array.
        delphi_real_type (type): Data type for real numbers.

    Returns:
        np.ndarray: Updated surface charge point positions.
        np.ndarray: Scaled surface normal vectors.
    """
    # Process if the closest entity is an atom.
    if closest_atom_or_object_index <= num_atoms:
        # Calculate the scaled position on the van der Waals surface.
        surface_charge_point_positions[i] = (
            atom_data[closest_atom_or_object_index - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
            + delta_r
            * (
                atom_data[closest_atom_or_object_index - 1][atom_data_radius_index]
                / distance_to_surface
            )
        ).astype(delphi_real_type)
        # Calculate the scaled surface normal vector.
        scaled_surface_normal_vectors[i] = (delta_r / distance_to_surface).astype(
            delphi_real_type
        )

    return surface_charge_point_positions, scaled_surface_normal_vectors


@njit(nogil=True, boundscheck=False, cache=True)
def _find_closest_exposed_point(
    all_boundary_points_processed: delphi_int,
    num_exposed_grid_points: delphi_int,
    s123: np.ndarray,
    system_min_coords: np.ndarray,
    cube_side_indver_length: delphi_real,
    cube_shape_indver: np.ndarray,
    cube_atom_index_start: np.ndarray,
    cube_atom_index_end: np.ndarray,
    cumulative_atom_index: np.ndarray,
    exposed_grid_point_coords: np.ndarray,
    is_outside_cube: np.ndarray,
    delphi_real_type: type,
    delphi_int_type: type,
) -> tuple[int, int]:
    """Finds the index of the closest solvent-exposed grid point to a given boundary point.

    Args:
        all_boundary_points_processed (delphi_int): Counter for processed boundary points (likely unused).
        num_exposed_grid_points (delphi_int): Total number of solvent-exposed grid points.
        s123 (np.ndarray): Coordinates of the boundary point.
        system_min_coords (np.ndarray): Minimum coordinates of the system.
        cube_side_indver_length (delphi_real): The cube side length for indexing.
        cube_shape_indver (np.ndarray): Dimensions of the spatial partitioning cube for vertex indexing.
        cube_atom_index_start (np.ndarray): Array storing the starting index of exposed points in each cube.
        cube_atom_index_end (np.ndarray): Array storing the ending index of atoms in each cube.
        cumulative_atom_index (np.ndarray): Array mapping indices within cubes to global exposed point indices.
        exposed_grid_point_coords (np.ndarray): Array containing coordinates of all solvent-exposed grid points.
        is_outside_cube (np.ndarray): Boolean array indicating if a sub-cube is outside the main cube.
        delphi_real_type (type): Data type for real numbers.
        delphi_int_type (type): Data type for integers.

    Returns:
        closest_exposed_point_index (int): The index of the closest solvent-exposed grid point.
        all_boundary_points_processed (delphi_int): Counter for processed boundary points (likely unused).
    """
    cube_side_indver_length_inverse = 1.0 / cube_side_indver_length
    # Calculate the relative coordinates of the boundary point within the system.
    relative_coords = (s123 - system_min_coords) * cube_side_indver_length_inverse
    # Calculate the integer grid indices of the boundary point.
    grid_indices_s123 = relative_coords.astype(delphi_int_type)
    grid_index_x = grid_indices_s123[0]
    grid_index_y = grid_indices_s123[1]
    grid_index_z = grid_indices_s123[2]
    # Calculate the fractional part of the relative coordinates.
    delta_coords_s123 = relative_coords - grid_indices_s123
    # Find the minimum and maximum fractional components.
    min_delta = min(
        delta_coords_s123[0], min(delta_coords_s123[1], delta_coords_s123[2])
    )
    max_delta = max(
        delta_coords_s123[0], max(delta_coords_s123[1], delta_coords_s123[2])
    )
    # Calculate the complement of the maximum fractional component.
    min_delta_complement = 1.0 - max_delta
    # Calculate a corner distance ratio.
    corner_distance_ratio = min(min_delta, min_delta_complement)
    # Calculate a cutoff threshold based on the corner distance.
    cutoff_threshold = cube_side_indver_length * (1 + corner_distance_ratio)
    cutoff_threshold_squared = cutoff_threshold**2
    closest_exposed_point_index = 0
    min_distance_to_exposed_squared = delphi_real_type(100.0)

    cube_extensions = [
        np.arange(-1, 2, dtype=delphi_int_type),
        np.arange(-2, 3, dtype=delphi_int_type),
    ]

    for extension_index in range(len(cube_extensions)):
        cube_extension = cube_extensions[extension_index]
        for neighbor_x_offset in cube_extension:
            for neighbor_y_offset in cube_extension:
                for neighbor_z_offset in cube_extension:
                    neighbor_x_index = grid_index_x + neighbor_x_offset
                    neighbor_y_index = grid_index_y + neighbor_y_offset
                    neighbor_z_index = grid_index_z + neighbor_z_offset

                    # Check if the neighboring cube indices are within bounds.
                    if (
                        0 <= neighbor_x_index <= cube_shape_indver[0]
                        and 0 <= neighbor_y_index <= cube_shape_indver[1]
                        and 0 <= neighbor_z_index <= cube_shape_indver[2]
                    ):
                        # For the extended search, check if the sub-cube is outside the main cube.
                        if extension_index == 1:
                            sub_cube_offset_x = neighbor_x_offset + 2
                            sub_cube_offset_y = neighbor_y_offset + 2
                            sub_cube_offset_z = neighbor_z_offset + 2
                            if not is_outside_cube[sub_cube_offset_x][
                                sub_cube_offset_y
                            ][sub_cube_offset_z]:
                                continue

                        # Iterate through the exposed points within the current neighboring cube.
                        for ii in range(
                            cube_atom_index_start[neighbor_x_index][neighbor_y_index][
                                neighbor_z_index
                            ],
                            cube_atom_index_end[neighbor_x_index][neighbor_y_index][
                                neighbor_z_index
                            ]
                            + 1,
                        ):
                            # Get the global index of the exposed point.
                            exposed_index = cumulative_atom_index[ii]
                            # Calculate the vector from the boundary point to the exposed point.
                            delta_coords_exposed_to_check = (
                                s123 - exposed_grid_point_coords[exposed_index]
                            )
                            # Calculate the squared distance.
                            distance_squared_to_exposed = dot_product(
                                delta_coords_exposed_to_check,
                                delta_coords_exposed_to_check,
                            )
                            # Update the closest exposed point if a closer one is found.
                            if (
                                distance_squared_to_exposed
                                < min_distance_to_exposed_squared
                            ):
                                min_distance_to_exposed_squared = (
                                    distance_squared_to_exposed
                                )
                                closest_exposed_point_index = exposed_index
        # Break after the first extension if a close point is found within the cutoff.
        if (
            closest_exposed_point_index > 0
            and min_distance_to_exposed_squared < cutoff_threshold_squared
        ):
            break

    # Final fallback search: iterate through all exposed points if no close point is found.
    if closest_exposed_point_index <= 0:
        all_boundary_points_processed += 1
        for exposed_index in range(
            num_exposed_grid_points
        ):  # Corrected range to 0-based indexing
            delta_coords_exposed_to_check = (
                s123 - exposed_grid_point_coords[exposed_index]
            )
            distance_squared_to_exposed = dot_product(
                delta_coords_exposed_to_check,
                delta_coords_exposed_to_check,
            )
            if distance_squared_to_exposed < min_distance_to_exposed_squared:
                min_distance_to_exposed_squared = distance_squared_to_exposed
                closest_exposed_point_index = exposed_index

    return closest_exposed_point_index, all_boundary_points_processed


@njit(nogil=True, boundscheck=False, cache=True)
def scale_vdw_surface_boundary_points(
    num_atoms: delphi_int,
    num_molecules: delphi_int,
    num_objects: delphi_int,
    max_atom_radius: delphi_real,
    probe_radius: delphi_real,
    probe_radius_2: delphi_real,
    is_focusing_run: delphi_bool,
    grid_spacing: delphi_real,
    grid_origin: np.ndarray,
    grid_dimensions: tuple,
    grid_origin_parentrun: np.ndarray,
    atom_data: np.ndarray,
    min_coords_by_axis: np.ndarray,
    max_coords_by_axis: np.ndarray,
    num_exposed_grid_points: delphi_int,
    num_boundary_points: delphi_int,
    num_external_boundary_points: delphi_int,
    surface_charge_positions: np.ndarray,
    discrete_epsilon_index_map_1d: np.ndarray,
    neighboring_atom_indices: np.ndarray,
    scaled_surface_normal_vectors: np.ndarray,
    exposed_grid_point_coords: np.ndarray,
    atom_accessibility: np.ndarray,
    atom_surface_index: np.ndarray,
    atom_index_for_boundary: np.ndarray,
    atom_plus_probe_radii: np.ndarray,
    atom_plus_probe_radii_squared: np.ndarray,
    atom_plus_probe_radii_squared_shrunk: np.ndarray,
    system_min_coords: np.ndarray,
    cube_side_indver_inverse: delphi_real,
    cube_shape_indver: np.ndarray,
    cube_voxel_atom_index_start: np.ndarray,
    cube_voxel_atom_index_end: np.ndarray,
    cube_voxel_atom_index_cumulative: np.ndarray,
) -> tuple[int, float]:
    """Scales the initial set of boundary points to accurately represent the molecular van der Waals surface.

    This function iterates through the boundary points and adjusts their positions to lie on the solvent-accessible surface.
    It handles both external and internal boundaries, as well as solvent exposure.

    Args:
        num_atoms (delphi_int): The number of atoms in the system.
        num_molecules (delphi_int): The number of molecules in the system.
        num_objects (delphi_int): The number of other objects in the system.
        max_atom_radius (delphi_real): The maximum radius of any atom.
        probe_radius (delphi_real): The radius of the solvent probe.
        probe_radius_2 (delphi_real): The radius of the second solvent probe (used in some cases).
        is_focusing_run (delphi_bool): Flag indicating if this is a focusing run.
        grid_spacing (delphi_real): The spacing between grid points.
        grid_origin (np.ndarray): The coordinates of the origin of the grid.
        grid_dimensions (tuple): The dimensions of the grid (nx, ny, nz).
        grid_origin_parentrun (np.ndarray): The origin of the grid in the parent run (for focusing).
        atom_data (np.ndarray): Array containing atom properties.
        object_data (np.ndarray): Array containing object properties.
        min_coords_by_axis (np.ndarray): Minimum coordinates of the system along each axis.
        max_coords_by_axis (np.ndarray): Maximum coordinates of the system along each axis.
        voxel_atom_count (np.ndarray): Array storing the starting index of atoms in each voxel.
        voxel_atom_count_cumulative (np.ndarray): Array storing the ending index of atoms in each voxel.
        num_exposed_grid_points (delphi_int): The number of solvent-exposed grid points.
        num_boundary_points (delphi_int): The total number of boundary points.
        num_external_boundary_points (delphi_int): The number of external boundary points.
        surface_charge_positions (np.ndarray): Array storing the positions of surface charge points.
        discrete_epsilon_index_map_1d (np.ndarray): 1D array representing the discrete epsilon map.
        neighboring_atom_indices (np.ndarray): Array to store indices of neighboring atoms.
        scaled_surface_normal_vectors (np.ndarray): Array to store the scaled surface normal vectors.
        exposed_grid_point_coords (np.ndarray): Array containing coordinates of all solvent-exposed grid points.
        atom_accessibility (np.ndarray): Array indicating the accessibility of each atom.
        atom_surface_index (np.ndarray): Array to store the index of the closest atom to each boundary point.
        atom_index_for_boundary (np.ndarray): Array to store the index of the atom associated with each boundary point.
        atom_plus_probe_radii (np.ndarray): Array containing the sum of atom radii and probe radius.
        atom_plus_probe_radii_squared (np.ndarray): Array containing the squared sum of atom radii and probe radius.
        atom_plus_probe_radii_squared_shrunk (np.ndarray): Array containing the squared sum of shrunk atom radii and probe radius.
        system_min_coords (np.ndarray): Minimum coordinates of the system.
        cube_side_indver_inverse (delphi_real): Inverse of the cube side length used for indexing.
        cube_voxel_atom_index_start (np.ndarray): Array storing the starting index of atoms in each cube.
        cube_voxel_atom_index_end (np.ndarray): Array storing the ending index of atoms in each cube.
        cube_voxel_atom_index_cumulative (np.ndarray): Array storing the cumulative index of atoms in each cube.

    Returns:
        int: An exit flag (0 for success).
        float: The inverse of the cube side length.
    """
    temp_grid_coords = np.zeros(3, dtype=delphi_real)
    # Initialize a boolean array to track if sub-cubes are outside the main cube.
    is_outside_cube = np.ones((5, 5, 5), dtype=delphi_bool)
    # Mark the central 3x3x3 sub-cubes as inside.
    for bgp_index in range(3):
        for j in range(3):
            for k in range(3):
                is_outside_cube[bgp_index + 1][j + 1][k + 1] = False

    epsilon_dimension = num_atoms + num_objects + 2
    half_grid_spacing = grid_spacing / 2.0
    is_internal_bgp = False

    # Determine the current grid origin based on whether it's a focusing run.
    current_grid_origin = grid_origin if is_focusing_run else grid_origin
    max_probe_radius = max(probe_radius_2, probe_radius)

    # Calculate the solvent-accessible surface if no exposed grid points are found and probe radius is greater than zero.
    if (
        num_exposed_grid_points == 0
        and max_probe_radius > 0.0
        and (num_objects > 1 or num_atoms > 1)
    ):
        nprint_cpu(DEBUG, _VERBOSITY, "Scaling routine in action")
        (
            num_exposed_grid_points,
            exposed_grid_point_coords,
            atom_accessibility,
            voxel_atom_count,
            voxel_atom_count_cumulative,
            voxel_space_origin,
            voxel_space_shape,
        ) = sas.solvent_accessible_surface(
            probe_radius=probe_radius,
            probe_radius2=probe_radius_2,
            max_atom_radius=max_atom_radius,
            min_coords_by_axis=min_coords_by_axis,
            max_coords_by_axis=max_coords_by_axis,
            num_atoms=num_atoms,
            num_objects=num_objects,
            num_molecules=num_molecules,
            atoms_data=atom_data,
            atom_plus_probe_radii_1d=atom_plus_probe_radii,
            # atom_plus_probe_radii_squared_1d=atom_plus_probe_radii_squared,
            atom_plus_probe_radii_shrink_1d=atom_plus_probe_radii_squared_shrunk,
            num_vertices=delphi_int(520),
            num_edges=delphi_int(1040),
        )

    # Determine the side length of the spatial partitioning cube
    delta = max(max_probe_radius, half_grid_spacing)
    voxel_side_length = max_atom_radius + delta
    voxel_space_scale = 1.0 / voxel_side_length

    # Set up the spatial partitioning into voxels
    (voxel_space_origin, voxel_space_shape) = (
        voxelizer.calculate_voxel_space_parameters(
            voxel_side_length,
            min_coords_by_axis,
            max_coords_by_axis,
            scaling_factor=2.0,
        )
    )
    voxels_extended_flat_size = (
        (voxel_space_shape[0] + 1)
        * (voxel_space_shape[1] + 1)
        * (voxel_space_shape[2] + 1)
    )
    voxels_per_entity = 27
    # Adjust the number of voxels per entity if there are non-molecule objects.
    if (num_objects - num_molecules) > 0:
        voxels_per_entity = voxels_extended_flat_size
        if voxels_per_entity < 27:
            voxels_per_entity = delphi_int(27)
    # Initialize an array to store atom indices for each voxel.
    voxel_atom_indices = np.zeros(
        voxels_per_entity * (num_atoms + num_objects - num_molecules) + 1,
        dtype=delphi_int,
    )

    # Populate the spatial partitioning cube with atom indices.
    (
        voxel_atom_indices,
        voxel_atom_count,
        voxel_atom_count_cumulative,
    ) = voxelizer.build_atom_voxel_map(
        voxel_side_length,
        num_atoms,
        num_objects,
        num_molecules,
        voxel_space_origin,
        voxel_space_shape,
        atom_data,
        voxel_atom_indices,
    )

    scaled_boundary_point_count = 0
    y_stride = grid_dimensions[2]
    x_stride = grid_dimensions[1] * grid_dimensions[2]
    x_stride_x_3 = x_stride * 3
    y_stride_x_3 = y_stride * 3

    # Iterate through each boundary grid point.
    for bgp_index in range(num_boundary_points):
        # Handle molecules with different epsilon values at internal points.
        if num_boundary_points != num_external_boundary_points and num_molecules > 1:
            # Get the grid coordinates of the current boundary point.
            grid_coords_ijk = [
                delphi_int(v) for v in surface_charge_positions[bgp_index]
            ]
            grid_coord_x = grid_coords_ijk[0]
            grid_coord_y = grid_coords_ijk[1]
            grid_coord_z = grid_coords_ijk[2]
            grid_index_1d = (
                grid_coord_x * x_stride + grid_coord_y * y_stride + grid_coord_z
            )
            # Check if the current boundary grid point is on an internal boundary.
            is_internal_bgp = _is_internal_boundary(
                discrete_epsilon_index_map_1d,
                grid_index_1d,
                epsilon_dimension,
                num_atoms,
                x_stride_x_3,
                y_stride_x_3,
            )

        # Calculate the Cartesian coordinates of the current boundary grid point.
        grid_point_coords = (
            surface_charge_positions[bgp_index] * grid_spacing + current_grid_origin
        )

        # Initialize min_distance_squared to arbitrary large number than expected.
        min_distance_squared = delphi_real(100.0)
        min_distance_squared_1 = delphi_real(1000.0)
        min_distance_squared_2 = delphi_real(1000.0)

        previous_media_id = 0
        closest_atom_or_object_index = 0
        neighbor_count = 0

        closest_atom_index_1 = 0
        closest_atom_index_2 = 0

        # Calculate the indices of the spatial partitioning cube containing the current grid point.
        cube_indices = np.array(
            [
                (a - b) * voxel_space_scale
                for a, b in zip(
                    grid_point_coords,
                    voxel_space_origin,
                )
            ],
            dtype=delphi_int,
        )

        lower_limit = 0
        upper_limit = -1
        # Get the range of atom indices for the current cube if the indices are within bounds.
        if not (np.any(cube_indices < 0) or np.any(cube_indices > voxel_space_shape)):
            lower_limit = voxel_atom_count[cube_indices[0]][cube_indices[1]][
                cube_indices[2]
            ]
            upper_limit = voxel_atom_count_cumulative[cube_indices[0]][cube_indices[1]][
                cube_indices[2]
            ]
        previous_object_index = 0

        # Iterate through the atoms in the current spatial partitioning cube.
        for kk in range(lower_limit, upper_limit + 1):
            atom_or_object_index = voxel_atom_indices[kk]
            if atom_or_object_index <= num_atoms:
                if is_internal_bgp:
                    # Calculate the squared distance from the grid point to the atom surface.
                    delta_coords = (
                        grid_point_coords
                        - atom_data[atom_or_object_index - 1][
                            ATOMFIELD_X:ATOMFIELD_CRD_END
                        ]
                    ).astype(delphi_real)
                    this_atom_radius = atom_data[atom_or_object_index - 1][
                        ATOMFIELD_RADIUS
                    ]
                    distance_squared = (
                        dot_product(delta_coords, delta_coords)
                        - this_atom_radius * this_atom_radius
                    )

                    # Keep track of the two closest atoms
                    if distance_squared < min_distance_squared_1:
                        closest_atom_index_2 = closest_atom_index_1
                        min_distance_squared_2 = min_distance_squared_1
                        closest_atom_index_1 = atom_or_object_index
                        min_distance_squared_1 = distance_squared
                    elif distance_squared <= min_distance_squared_2:
                        closest_atom_index_2 = atom_or_object_index
                        min_distance_squared_2 = distance_squared
                else:
                    # If it's a regular boundary point and the atom is not a surface atom.
                    if atom_accessibility[atom_or_object_index] == 0:
                        neighbor_count += 1
                        if neighbor_count < SPACE_NBRA_MAX_SIZE:
                            neighboring_atom_indices[neighbor_count] = (
                                atom_or_object_index
                            )
                        else:
                            nprint_cpu(
                                DEBUG,
                                _VERBOSITY,
                                "space_sclbp>> index beyond size of nbra: nnbr= ",
                                neighbor_count,
                            )
            else:
                # Handle objects (molecules or other structures)
                if atom_or_object_index != previous_object_index:
                    previous_object_index = atom_or_object_index
                    neighbor_count += 1
                    if neighbor_count >= SPACE_NBRA_MAX_SIZE:
                        nprint_cpu(
                            DEBUG,
                            _VERBOSITY,
                            "space_sclbp>> index beyond size of nbra: nnbr= ",
                            neighbor_count,
                        )
                    neighboring_atom_indices[neighbor_count] = atom_or_object_index

        # Handle internal boundary points by adjusting their position between the two closest atoms.
        if is_internal_bgp:
            atom_surface_index[bgp_index] = closest_atom_index_1
            # Check for errors in identifying the two closest atoms
            if (
                closest_atom_index_1 * closest_atom_index_2 == 0
                or closest_atom_index_1 == closest_atom_index_2
            ):
                nprint_cpu(
                    DEBUG,
                    _VERBOSITY,
                    "Problems in Scaling multidielectric Boundary Grid Points",
                )
                return EXIT_NJIT_FLAG, 0.0
            atom_index_for_boundary[bgp_index] = -1

            # Calculate the displacement vector between the two closest atoms.
            delta_coords_12 = (
                atom_data[closest_atom_index_2 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
                - atom_data[closest_atom_index_1 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
            )
            temp = dot_product(delta_coords_12, delta_coords_12)
            # temp is a scaling factor used to position the internal boundary point.
            # It represents a fraction along the vector connecting the centers of the two closest atoms,
            # determined by the difference in the squared distances from the original grid point to their surfaces
            # and the squared distance between the atom centers.
            temp = 0.5 * (min_distance_squared_2 - min_distance_squared_1) / temp
            # Update the position of the internal boundary point.
            surface_charge_positions[bgp_index] = grid_point_coords + (
                temp * delta_coords_12
            )
            scaled_surface_normal_vectors[bgp_index] = temp_grid_coords[:]
            continue
        else:
            # For external boundary points, find the closest atom or object.
            (
                closest_atom_or_object_index,
                previous_media_id,
                min_distance_squared,
            ) = _process_neighbors(
                grid_point_coords,
                num_atoms,
                atom_data,
                neighboring_atom_indices,
                neighbor_count,
                previous_media_id,
                closest_atom_or_object_index,
                min_distance_squared,
                delphi_real,
                delphi_int,
            )

            atom_surface_index[bgp_index] = closest_atom_or_object_index

        # Check if a closest atom or object was found.
        if closest_atom_or_object_index == 0 and closest_atom_index_1 == 0:
            nprint_cpu(
                DEBUG,
                _VERBOSITY,
                "no close atom or object for boundary pointer ",
                bgp_index,
            )
            return EXIT_NJIT_FLAG, 0.0

        # Calculate the vector from the closest atom/object to the grid point.
        if closest_atom_or_object_index <= num_atoms:
            delta_r = (
                grid_point_coords
                - atom_data[closest_atom_or_object_index - 1][
                    ATOMFIELD_X:ATOMFIELD_CRD_END
                ]
            ).astype(delphi_real)
        else:
            delta_r = np.zeros(3, dtype=delphi_real)  # Placeholder

        distance_to_surface = np.sqrt(dot_product(delta_r, delta_r))
        is_outside = True

        # Check for solvent exposure if the probe radius is greater than zero.
        if max_probe_radius > 0.0:
            if closest_atom_or_object_index <= num_atoms:
                u123 = atom_data[closest_atom_or_object_index - 1][
                    ATOMFIELD_X:ATOMFIELD_CRD_END
                ] + (
                    (atom_plus_probe_radii[closest_atom_or_object_index - 1] * delta_r)
                    / distance_to_surface
                )
            else:
                u123 = (grid_point_coords - delta_r).astype(
                    delphi_real
                )  # Placeholder for zero radius

            # Check if the test point is inside any other atom (excluding the closest one).
            is_outside = _is_solvent_exposed(
                u123,
                num_atoms,
                atom_data,
                voxel_atom_indices,
                voxel_atom_count,
                voxel_atom_count_cumulative,
                voxel_space_origin,
                voxel_space_scale,
                voxel_space_shape,
                atom_plus_probe_radii_squared_shrunk,
                probe_radius,
                delphi_real,
                delphi_int,
            )

        # If the boundary point is solvent-exposed, scale its position to the van der Waals surface.
        if is_outside:
            scaled_boundary_point_count += 1
            (
                surface_charge_positions,
                scaled_surface_normal_vectors,
            ) = _scale_exposed_point(
                bgp_index,
                closest_atom_or_object_index,
                num_atoms,
                grid_point_coords,
                atom_data,
                distance_to_surface,
                delta_r,
                surface_charge_positions,
                scaled_surface_normal_vectors,
                atom_plus_probe_radii,
                ATOMFIELD_RADIUS,
                delphi_real,
            )
            atom_index_for_boundary[bgp_index] = closest_atom_or_object_index
        else:
            atom_index_for_boundary[bgp_index] = 0

    # Handle re-entrant surfaces by finding the closest exposed point for buried boundary points.
    if max_probe_radius > 0.0:
        all_boundary_points_processed = 0
        cube_side_length_index_vertices = 1.0 / cube_side_indver_inverse

        nprint_cpu(
            DEBUG,
            _VERBOSITY,
            "SCALE-BGP>> n_boundary_points:",
            num_boundary_points,
        )
        nprint_cpu(
            DEBUG,
            _VERBOSITY,
            "SCALE-BGP>> num_exposed_grid_points:",
            num_exposed_grid_points,
        )

        # Iterate through boundary points again to handle re-entrant cases
        for bgp_index in range(num_boundary_points):
            if atom_index_for_boundary[bgp_index] == -1:
                continue  # Skip internal boundary points

            # If the boundary point is not solvent-exposed (atom_index_for_boundary is 0).
            if atom_index_for_boundary[bgp_index] == 0:
                # Calculate coordinates of the current boundary point.
                s123 = (
                    surface_charge_positions[bgp_index] * grid_spacing
                    + current_grid_origin
                ).astype(delphi_real)

                # Find the closest solvent-exposed grid point.
                (
                    closest_exposed_point_index,
                    all_boundary_points_processed,
                ) = _find_closest_exposed_point(
                    all_boundary_points_processed,
                    num_exposed_grid_points,
                    s123,
                    system_min_coords,
                    cube_side_length_index_vertices,
                    cube_shape_indver,
                    cube_voxel_atom_index_start,
                    cube_voxel_atom_index_end,
                    cube_voxel_atom_index_cumulative,
                    exposed_grid_point_coords,
                    is_outside_cube,
                    delphi_real,
                    delphi_int,
                )

                # Calculate the vector from the closest exposed point to the current boundary point.
                delta_coords_exposed = (
                    s123 - exposed_grid_point_coords[closest_exposed_point_index]
                )
                distance_to_exposed = np.sqrt(
                    dot_product(delta_coords_exposed, delta_coords_exposed)
                )
                if distance_to_exposed < APPROX_ZERO:
                    scaling_factor = 0.0
                else:
                    is_buried_in_object = True
                    # Determine the scaling factor based on whether the point is buried.
                    if is_buried_in_object:
                        scaling_factor = probe_radius_2 / distance_to_exposed
                        is_buried_in_object = False
                    else:
                        scaling_factor = probe_radius / distance_to_exposed

                # Move the boundary point to the surface of the probe sphere around the closest exposed point.
                surface_charge_positions[bgp_index] = exposed_grid_point_coords[
                    closest_exposed_point_index
                ] + (delta_coords_exposed * scaling_factor)

                # Calculate the scaled surface normal vector.
                if distance_to_exposed > APPROX_ZERO:
                    scaled_surface_normal_vectors[bgp_index] = (
                        -delta_coords_exposed
                    ) / distance_to_exposed
                else:
                    nprint_cpu(
                        DEBUG,
                        _VERBOSITY,
                        "bdp close to arcp ",
                        bgp_index,
                        distance_to_exposed,
                    )
    nprint_cpu(
        DEBUG,
        _VERBOSITY,
        "SCALE-BGP fall-through points>> num_exposed_grid_points:",
        all_boundary_points_processed,
    )

    return 0, voxel_space_scale
