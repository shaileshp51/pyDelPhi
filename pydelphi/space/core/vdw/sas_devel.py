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

from numba import njit, prange
import numpy as np

# Constants used in the original function
APPROX_ZERO = 1e-6
ATOMFIELD_RADIUS = 3  # Placeholder index for radius field in atoms_data
ATOM_PAIRS_COUNT_MINIMAL = 100
ATOM_PAIRS_LINEAR_FACTOR = 4
ATOM_PAIRS_N_SQUARED_FACTOR = 0.01


# Dummy helpers for clarity (replace with actual implementation)
@njit
def get_atom_coords(atom_data):
    return atom_data[:3]  # assuming first 3 entries are x, y, z


@njit
def dot_product(vec1, vec2):
    return vec1[0] * vec2[0] + vec1[1] * vec2[1] + vec1[2] * vec2[2]


@njit(parallel=True, nogil=True)
def _count_atom_pairs(
    num_atoms,
    atoms_data,
    atom_plus_probe_radii,
    cube_vertex_lowest_xyz,
    cube_side_length_inverse,
    voxel_atom_counts,
    voxel_cumulative_atom_counts,
    voxel_atom_indices,
):
    pair_counts = np.zeros(num_atoms, dtype=np.int32)

    for atom_index1 in prange(num_atoms):
        radius1 = atom_plus_probe_radii[atom_index1]
        if atoms_data[atom_index1][ATOMFIELD_RADIUS] < APPROX_ZERO:
            continue

        atom1_coords = get_atom_coords(atoms_data[atom_index1])
        voxel_indices = (
            (atom1_coords - cube_vertex_lowest_xyz) * cube_side_length_inverse
        ).astype(np.int32)

        lower_limit = voxel_atom_counts[voxel_indices[0]][voxel_indices[1]][
            voxel_indices[2]
        ]
        upper_limit = voxel_cumulative_atom_counts[voxel_indices[0]][voxel_indices[1]][
            voxel_indices[2]
        ]

        count = 0
        for voxel_atom_index in range(lower_limit, upper_limit + 1):
            atom_index2 = voxel_atom_indices[voxel_atom_index]
            if atom_index2 <= num_atoms and atom_index2 > atom_index1 + 1:
                radius2 = atom_plus_probe_radii[atom_index2 - 1]
                if atoms_data[atom_index2 - 1][ATOMFIELD_RADIUS] > 0:
                    combined_radius = radius1 + radius2
                    combined_radius_squared = combined_radius * combined_radius
                    radius_diff = abs(radius1 - radius2)
                    radius_diff_squared = radius_diff * radius_diff

                    dist_vec = (
                        get_atom_coords(atoms_data[atom_index2 - 1]) - atom1_coords
                    )
                    dist_squared = dot_product(dist_vec, dist_vec)

                    if (
                        combined_radius_squared - dist_squared > 0.01
                        and dist_squared > radius_diff_squared
                    ):
                        count += 1

        pair_counts[atom_index1] = count

    return pair_counts


@njit(nogil=True)
def _prefix_sum(arr):
    out = np.empty_like(arr)
    total = 0
    for i in range(len(arr)):
        out[i] = total
        total += arr[i]
    return out, total


@njit(parallel=True, nogil=True)
def _populate_atom_pairs(
    num_atoms,
    atoms_data,
    atom_plus_probe_radii,
    atom_contact_state,
    cube_vertex_lowest_xyz,
    cube_side_length_inverse,
    voxel_atom_counts,
    voxel_cumulative_atom_counts,
    voxel_atom_indices,
    contacting_atom_pairs,
    offsets,
):
    for atom_index1 in prange(num_atoms):
        radius1 = atom_plus_probe_radii[atom_index1]
        if atoms_data[atom_index1][ATOMFIELD_RADIUS] < APPROX_ZERO:
            atom_contact_state[atom_index1 + 1] = 0
            continue

        atom1_coords = get_atom_coords(atoms_data[atom_index1])
        voxel_indices = (
            (atom1_coords - cube_vertex_lowest_xyz) * cube_side_length_inverse
        ).astype(np.int32)

        lower_limit = voxel_atom_counts[voxel_indices[0]][voxel_indices[1]][
            voxel_indices[2]
        ]
        upper_limit = voxel_cumulative_atom_counts[voxel_indices[0]][voxel_indices[1]][
            voxel_indices[2]
        ]

        idx = 0
        for voxel_atom_index in range(lower_limit, upper_limit + 1):
            atom_index2 = voxel_atom_indices[voxel_atom_index]
            if atom_index2 <= num_atoms and atom_index2 > atom_index1 + 1:
                radius2 = atom_plus_probe_radii[atom_index2 - 1]
                if atoms_data[atom_index2 - 1][ATOMFIELD_RADIUS] > 0:
                    combined_radius = radius1 + radius2
                    combined_radius_squared = combined_radius * combined_radius
                    radius_diff = abs(radius1 - radius2)
                    radius_diff_squared = radius_diff * radius_diff

                    dist_vec = (
                        get_atom_coords(atoms_data[atom_index2 - 1]) - atom1_coords
                    )
                    dist_squared = dot_product(dist_vec, dist_vec)

                    if (
                        combined_radius_squared - dist_squared > 0.01
                        and dist_squared > radius_diff_squared
                    ):
                        offset = offsets[atom_index1] + idx
                        contacting_atom_pairs[offset, 0] = atom_index1 + 1
                        contacting_atom_pairs[offset, 1] = atom_index2
                        contacting_atom_pairs[offset, 2] = 0
                        idx += 1

        if idx == 0:
            atom_contact_state[atom_index1 + 1] = 0


@njit(nogil=True)
def find_atom_pairs_parallel(
    num_atoms,
    atoms_data,
    atom_plus_probe_radii,
    atom_contact_state,
    cube_vertex_lowest_xyz,
    cube_side_length_inverse,
    voxel_atom_counts,
    voxel_cumulative_atom_counts,
    voxel_atom_indices,
):
    pair_counts = _count_atom_pairs(
        num_atoms,
        atoms_data,
        atom_plus_probe_radii,
        cube_vertex_lowest_xyz,
        cube_side_length_inverse,
        voxel_atom_counts,
        voxel_cumulative_atom_counts,
        voxel_atom_indices,
    )

    offsets, total_pairs = _prefix_sum(pair_counts)
    contacting_atom_pairs = np.zeros((total_pairs, 3), dtype=np.int32)

    _populate_atom_pairs(
        num_atoms,
        atoms_data,
        atom_plus_probe_radii,
        atom_contact_state,
        cube_vertex_lowest_xyz,
        cube_side_length_inverse,
        voxel_atom_counts,
        voxel_cumulative_atom_counts,
        voxel_atom_indices,
        contacting_atom_pairs,
        offsets,
    )

    return atom_contact_state, contacting_atom_pairs


from numba import njit, prange
import numpy as np


@njit(parallel=True, nogil=True, boundscheck=False, cache=True)
def _calculate_exposed_grids(
    num_atoms,
    atoms_data,
    atom_plus_probe_radii_1d,
    atom_plus_probe_radii_shrink_1d,
    atom_contact_state,
    atom_pairs,
    vertex_array,
    edge_array,
    edges,
    voxel_space_origin,
    voxel_space_scale,
    voxel_atom_start_indices,
    voxel_atom_end_indices,
    voxel_atom_indices,
    num_vertices,
    num_edges,
    refinement_level,
    initial_vertex_count,
    num_extended_cube_grid_points,
    real_type,
    int_type,
):
    n_pairs = atom_pairs.shape[0]
    estimated_grid_counts = _estimate_grid_counts(
        n_pairs,
        atom_pairs,
        atoms_data,
        atom_plus_probe_radii_1d,
        atom_plus_probe_radii_shrink_1d,
        vertex_array,
        voxel_space_origin,
        voxel_space_scale,
        voxel_atom_start_indices,
        voxel_atom_end_indices,
        voxel_atom_indices,
        num_atoms,
        initial_vertex_count,
        real_type,
        int_type,
    )

    total = 0
    offsets = np.empty_like(estimated_grid_counts)
    for i in range(n_pairs):
        offsets[i] = total
        total += estimated_grid_counts[i]

    exposed_grid_coordinates = np.zeros((total + 1, 3), dtype=real_type)
    vertex_occupation = np.zeros(num_vertices + 1, dtype=int_type)
    edge_state = np.zeros(num_edges + 1, dtype=int_type)

    per_pair_results = _populate_exposed_grids(
        n_pairs,
        atom_pairs,
        atoms_data,
        atom_plus_probe_radii_1d,
        atom_plus_probe_radii_shrink_1d,
        vertex_array,
        voxel_space_origin,
        voxel_space_scale,
        voxel_atom_start_indices,
        voxel_atom_end_indices,
        voxel_atom_indices,
        num_atoms,
        initial_vertex_count,
        offsets,
        exposed_grid_coordinates,
        real_type,
        int_type,
    )

    num_exposed_atom_pairs = 0
    for i in range(n_pairs):
        if per_pair_results[i, 0] == 1:
            num_exposed_atom_pairs += 1
            atom1 = atom_pairs[i, 0]
            atom2 = atom_pairs[i, 1]
            atom_contact_state[atom1] = 1
            atom_contact_state[atom2] = 1

    num_accessible_atoms = 0
    for atom_index in range(1, num_atoms + 1):
        if atom_contact_state[atom_index] == 0:
            num_accessible_atoms += 1

    return (
        offsets[n_pairs - 1] + estimated_grid_counts[n_pairs - 1],
        exposed_grid_coordinates,
        num_accessible_atoms,
        atom_contact_state,
        num_exposed_atom_pairs,
    )


@njit(parallel=True, nogil=True, boundscheck=False, cache=True)
def _estimate_grid_counts(
    n_pairs,
    atom_pairs,
    atoms_data,
    atom_plus_probe_radii_1d,
    atom_plus_probe_radii_shrink_1d,
    vertex_array,
    voxel_space_origin,
    voxel_space_scale,
    voxel_atom_start_indices,
    voxel_atom_end_indices,
    voxel_atom_indices,
    num_atoms,
    initial_vertex_count,
    real_type,
    int_type,
):
    counts = np.zeros(n_pairs, dtype=int_type)
    for pair_idx in prange(n_pairs):
        atom1 = atom_pairs[pair_idx, 0] - 1
        atom2 = atom_pairs[pair_idx, 1] - 1

        coord1 = atoms_data[atom1, 0:3]
        coord2 = atoms_data[atom2, 0:3]

        center = 0.5 * (coord1 + coord2)
        radius = atom_plus_probe_radii_1d[atom1] + atom_plus_probe_radii_1d[atom2]

        min_corner = center - radius
        max_corner = center + radius

        grid_min = np.floor(
            (min_corner - voxel_space_origin) * voxel_space_scale
        ).astype(int_type)
        grid_max = np.floor(
            (max_corner - voxel_space_origin) * voxel_space_scale
        ).astype(int_type)

        dx = grid_max[0] - grid_min[0] + 1
        dy = grid_max[1] - grid_min[1] + 1
        dz = grid_max[2] - grid_min[2] + 1

        estimate = dx * dy * dz * refinement_level
        counts[pair_idx] = estimate

    return counts


@njit(parallel=True, nogil=True, boundscheck=False, cache=True)
def _populate_exposed_grids(
    n_pairs,
    atom_pairs,
    atoms_data,
    atom_plus_probe_radii_1d,
    atom_plus_probe_radii_shrink_1d,
    vertex_array,
    voxel_space_origin,
    voxel_space_scale,
    voxel_atom_start_indices,
    voxel_atom_end_indices,
    voxel_atom_indices,
    num_atoms,
    initial_vertex_count,
    offsets,
    exposed_grid_coordinates,
    real_type,
    int_type,
):
    pair_results = np.zeros((n_pairs, 1), dtype=int_type)
    for pair_idx in prange(n_pairs):
        atom1 = atom_pairs[pair_idx, 0] - 1
        atom2 = atom_pairs[pair_idx, 1] - 1

        coord1 = atoms_data[atom1, 0:3]
        coord2 = atoms_data[atom2, 0:3]
        center = 0.5 * (coord1 + coord2)

        radius1 = atom_plus_probe_radii_1d[atom1]
        radius2 = atom_plus_probe_radii_1d[atom2]

        combined_radius = radius1 + radius2
        combined_radius_squared = combined_radius * combined_radius

        distance_vector = coord1 - coord2
        distance_squared = np.dot(distance_vector, distance_vector)

        if distance_squared > 0 and distance_squared < combined_radius_squared:
            base_index = offsets[pair_idx]
            exposed_grid_coordinates[base_index][0] = center[0]
            exposed_grid_coordinates[base_index][1] = center[1]
            exposed_grid_coordinates[base_index][2] = center[2]

            pair_results[pair_idx, 0] = 1

    return pair_results
