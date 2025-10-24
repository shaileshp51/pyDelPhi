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
This module calculates the Solvent Accessible Surface (SAS) of a molecular system.

It uses a sphere-based approach to determine the exposed surface area of atoms
and objects, considering a probe radius. The process involves:
1. Initializing and refining a set of vertices and edges on a sphere.
2. Identifying pairs of atoms that are in contact.
3. Calculating the exposed grid points on the surface, considering inter-atomic
   and atom-object occlusions.

The module relies on Numba for performance optimization of numerical operations.
"""

import numpy as np
from numba import njit

from pydelphi.foundation.enums import Precision

from pydelphi.config.global_runtime import (
    PRECISION,
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

if PRECISION.value == Precision.SINGLE.value:
    from pydelphi.utils.prec.single import dot_product, get_atom_coords

elif PRECISION.value == Precision.DOUBLE.value:
    from pydelphi.utils.prec.double import dot_product, get_atom_coords

import pydelphi.space.core.voxelizer as voxelizer

from pydelphi.constants import (
    ConstPhysical,
    ConstDelPhiFloats,
    ATOMFIELD_RADIUS,
)

APPROX_ZERO = ConstDelPhiFloats.ApproxZero.value
CONST_PI = ConstPhysical.Pi.value
RESIZE_FACTOR = ConstDelPhiFloats.ZetaArrayResizeFactor.value

ATOM_PAIRS_COUNT_MINIMAL = ConstDelPhiFloats.SASAtomPairsMinimalCount.value
ATOM_PAIRS_LINEAR_FACTOR = ConstDelPhiFloats.SASLinearPairsFactorPerAtom.value
ATOM_PAIRS_N_SQUARED_FACTOR = ConstDelPhiFloats.SASQuadraticPairsFactorOfNSquared.value

EXPOSED_GRIDS_COUNT_MINIMAL = ConstDelPhiFloats.SASExposedGridsMinimalCount.value
EXPOSED_GRIDS_SURFACE_FACTOR = ConstDelPhiFloats.SASExposedGridsSurfaceAreaFactor.value


@njit(nogil=True, boundscheck=False, cache=True)
def _initialize_vertices_and_edges(
    num_vertices, num_edges, initial_vertex_count, real_type, int_type
):
    vertex_array = np.zeros((num_vertices + 1, 3), dtype=real_type)
    edge_array = np.zeros((num_edges + 1, 3), dtype=int_type)
    edges = np.zeros(num_edges + 1, dtype=int_type)

    angle_increment = 2.0 * CONST_PI / initial_vertex_count

    for vertex_index in range(1, initial_vertex_count + 1):
        rotation_angle = (vertex_index - 1) * angle_increment
        vertex_array[vertex_index][0] = real_type(np.cos(rotation_angle))
        vertex_array[vertex_index][1] = real_type(np.sin(rotation_angle))
        vertex_array[vertex_index][2] = real_type(0)

        next_vertex_index = (
            vertex_index + 1 if vertex_index != initial_vertex_count else 1
        )
        edge_array[vertex_index][1] = vertex_index
        edge_array[vertex_index][2] = next_vertex_index

    return vertex_array, edge_array, edges


@njit(nogil=True, boundscheck=False, cache=True)
def _refine_vertices_and_edges(
    vertex_array,
    edge_array,
    edges,
    refinement_level,
    initial_vertex_count,
    real_type,
    int_type,
):
    current_vertex_count = initial_vertex_count
    current_edge_count = initial_vertex_count
    start_edge_index = 1
    end_edge_index = 0

    for level in range(1, refinement_level + 1):
        start_edge_index = end_edge_index + 1
        end_edge_index = current_edge_count
        for edge_index in range(start_edge_index, end_edge_index + 1):
            vertex_index1 = edge_array[edge_index][1]
            vertex_index2 = edge_array[edge_index][2]

            midpoint_vector = vertex_array[vertex_index1] + vertex_array[vertex_index2]
            midpoint_magnitude = np.sqrt(dot_product(midpoint_vector, midpoint_vector))

            current_vertex_count += 1
            vertex_array[current_vertex_count] = (
                midpoint_vector / midpoint_magnitude
            ).astype(real_type)
            current_edge_count += 1
            edges[edge_index] = current_edge_count
            edge_array[current_edge_count][1] = vertex_index1
            edge_array[current_edge_count][2] = current_vertex_count
            current_edge_count += 1
            edge_array[current_edge_count][1] = current_vertex_count
            edge_array[current_edge_count][2] = vertex_index2

    current_edge_count = end_edge_index

    for edge_index in range(start_edge_index, current_edge_count + 1):
        edges[edge_index] = -1

    return (
        vertex_array,
        edge_array,
        edges,
        current_vertex_count,
        current_edge_count,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def _find_atom_pairs(
    num_atoms,
    atoms_data,
    atom_plus_probe_radii,
    atom_contact_state,
    cube_vertex_lowest_xyz,
    cube_side_length_inverse,
    voxel_atom_counts,
    voxel_cumulative_atom_counts,
    voxel_atom_indices,
    real_type,
    int_type,
):
    num_contacting_atom_pairs = 0
    num_contacting_atom_pairs_current = 0
    initial_pair_count = int(
        max(
            ATOM_PAIRS_COUNT_MINIMAL,
            min(
                ATOM_PAIRS_LINEAR_FACTOR * num_atoms,
                ATOM_PAIRS_N_SQUARED_FACTOR * num_atoms**2,
            ),
        )
    )
    contacting_atom_pairs = np.zeros((initial_pair_count, 3), dtype=int_type)

    for atom_index1 in range(num_atoms):
        radius1 = atom_plus_probe_radii[atom_index1]
        if atoms_data[atom_index1][ATOMFIELD_RADIUS] < APPROX_ZERO:
            continue
        atom1_coords = get_atom_coords(atoms_data[atom_index1])
        voxel_indices = (
            (atom1_coords - cube_vertex_lowest_xyz) * cube_side_length_inverse
        ).astype(int_type)

        lower_limit = voxel_atom_counts[voxel_indices[0]][voxel_indices[1]][
            voxel_indices[2]
        ]
        upper_limit = voxel_cumulative_atom_counts[voxel_indices[0]][voxel_indices[1]][
            voxel_indices[2]
        ]
        previous_atom_index = 0
        for voxel_atom_index in range(lower_limit, upper_limit + 1):
            atom_index2 = voxel_atom_indices[voxel_atom_index]
            if atom_index2 <= num_atoms:
                radius2 = atom_plus_probe_radii[atom_index2 - 1]

                if (
                    atoms_data[atom_index2 - 1][ATOMFIELD_RADIUS] > 0
                    and atom_index2 > atom_index1 + 1
                ):
                    combined_radius = radius1 + radius2
                    combined_radius_squared = combined_radius * combined_radius
                    radius_difference = abs(radius1 - radius2)
                    radius_difference_squared = radius_difference * radius_difference
                    distance_vector = (
                        get_atom_coords(atoms_data[atom_index2 - 1]) - atom1_coords
                    )
                    distance_squared = dot_product(distance_vector, distance_vector)
                    delta = combined_radius_squared - distance_squared

                    if delta > 0.01 and distance_squared > radius_difference_squared:
                        if (
                            num_contacting_atom_pairs_current
                            >= contacting_atom_pairs.shape[0]
                        ):
                            new_pair_count = int(
                                contacting_atom_pairs.shape[0] * RESIZE_FACTOR
                            )
                            contacting_atom_pairs = np.resize(
                                contacting_atom_pairs, (new_pair_count, 3)
                            )
                        contacting_atom_pairs[num_contacting_atom_pairs_current][0] = (
                            atom_index1 + 1
                        )
                        contacting_atom_pairs[num_contacting_atom_pairs_current][
                            1
                        ] = atom_index2
                        contacting_atom_pairs[num_contacting_atom_pairs_current][2] = 0
                        num_contacting_atom_pairs_current += 1
            previous_atom_index = atom_index2

        # For this atom1 no new contact pairs found
        if num_contacting_atom_pairs_current == num_contacting_atom_pairs:
            atom_contact_state[atom_index1 + 1] = (
                0  # Atom1 not in contact to any other atoms
            )

        num_contacting_atom_pairs = num_contacting_atom_pairs_current

    return atom_contact_state, contacting_atom_pairs[:num_contacting_atom_pairs]


@njit(nogil=True, boundscheck=False, cache=True)
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
    num_exposed_grids = 0
    initial_exposed_grid_count = int(
        min(
            num_extended_cube_grid_points,
            max(
                EXPOSED_GRIDS_COUNT_MINIMAL,
                EXPOSED_GRIDS_SURFACE_FACTOR
                * (num_extended_cube_grid_points) ** (2 / 3),
            ),
        )
    )
    exposed_grid_coordinates = np.zeros(
        (initial_exposed_grid_count + 1, 3), dtype=real_type
    )
    vertex_occupation = np.zeros(num_vertices + 1, dtype=int_type)
    edge_state = np.zeros(num_edges + 1, dtype=int_type)
    rotation_matrix = np.zeros((4, 3), dtype=real_type)
    contact_point = np.zeros(3, dtype=real_type)

    num_exposed_atom_pairs = 0
    for pair_index in range(atom_pairs.shape[0]):
        atom_index1 = atom_pairs[pair_index][0]
        atom_index2 = atom_pairs[pair_index][1]
        atom_coords1 = get_atom_coords(atoms_data[atom_index1 - 1])
        atom_coords2 = get_atom_coords(atoms_data[atom_index2 - 1])
        r1 = atom_plus_probe_radii_1d[atom_index1 - 1]
        r2 = atom_plus_probe_radii_1d[atom_index2 - 1]

        distance_vector = atom_coords2 - atom_coords1
        distance_squared = dot_product(distance_vector, distance_vector)
        distance_magnitude = real_type(np.sqrt(distance_squared))
        prefactor = 1.0 + (r1**2 - r2**2) / distance_squared
        midpoint_coords = atom_coords1 + ((0.5 * prefactor) * distance_vector).astype(
            real_type
        )

        sum_radii = r1 + r2
        diff_radii = r1 - r2

        term1_sq = sum_radii * sum_radii - distance_squared
        term2_sq = distance_squared - diff_radii * diff_radii

        interaction_radius = real_type(
            0.5 * np.sqrt(term1_sq) * np.sqrt(term2_sq) / distance_magnitude
        )

        dx1 = distance_vector[0]
        dx2 = distance_vector[1]
        dx3 = distance_vector[2]
        xy_projection_magnitude = real_type(np.sqrt(dx1 * dx1 + dx2 * dx2))

        if xy_projection_magnitude > 1.0e-8:
            tangential_vector_x = -dx2 / xy_projection_magnitude
            tangential_vector_y = dx1 / xy_projection_magnitude
            cosine_theta = dx3 / distance_magnitude

            sine_theta = real_type(np.sqrt(1.0 - cosine_theta * cosine_theta))

            one_minus_cosine_theta = real_type(1.0 - cosine_theta)
            temp_multiplier = real_type(one_minus_cosine_theta * tangential_vector_x)
            sine_tangential_x = real_type(sine_theta * tangential_vector_x)
            sine_tangential_y = real_type(sine_theta * tangential_vector_y)

            rotation_matrix[1][0] = real_type(
                temp_multiplier * tangential_vector_x + cosine_theta
            )
            rotation_matrix[1][1] = real_type(temp_multiplier * tangential_vector_y)
            rotation_matrix[1][2] = real_type(sine_tangential_y)

            rotation_matrix[2][0] = real_type(temp_multiplier * tangential_vector_y)
            rotation_matrix[2][1] = real_type(
                one_minus_cosine_theta * tangential_vector_y * tangential_vector_y
                + cosine_theta
            )
            rotation_matrix[2][2] = real_type(-sine_tangential_x)

            rotation_matrix[3][0] = real_type(-sine_tangential_y)
            rotation_matrix[3][1] = real_type(sine_tangential_x)
            rotation_matrix[3][2] = real_type(cosine_theta)
        else:
            rotation_matrix[1][0:3] = real_type(1.0), real_type(0.0), real_type(0.0)
            rotation_matrix[2][0:3] = real_type(0.0), real_type(1.0), real_type(0.0)
            rotation_matrix[3][0:3] = real_type(0.0), real_type(0.0), real_type(1.0)

        num_visible_vertices = 0

        vertex_index = 1
        while vertex_index <= initial_vertex_count:
            contact_point[0] = real_type(
                rotation_matrix[1][0] * vertex_array[vertex_index][0]
                + rotation_matrix[1][1] * vertex_array[vertex_index][1]
            )
            contact_point[1] = real_type(
                rotation_matrix[2][0] * vertex_array[vertex_index][0]
                + rotation_matrix[2][1] * vertex_array[vertex_index][1]
            )
            contact_point[2] = real_type(
                rotation_matrix[3][0] * vertex_array[vertex_index][0]
                + rotation_matrix[3][1] * vertex_array[vertex_index][1]
            )
            contact_point = (
                midpoint_coords + (contact_point * interaction_radius)
            ).astype(real_type)
            grid_indices = (
                (contact_point - voxel_space_origin) * voxel_space_scale
            ).astype(int_type)
            lower_limit = voxel_atom_start_indices[grid_indices[0]][grid_indices[1]][
                grid_indices[2]
            ]
            upper_limit = voxel_atom_end_indices[grid_indices[0]][grid_indices[1]][
                grid_indices[2]
            ]

            voxel_atom_index = lower_limit
            no_jump_to_outer = True
            while no_jump_to_outer and voxel_atom_index <= upper_limit:
                atom_index = voxel_atom_indices[voxel_atom_index]
                if atom_index > num_atoms:
                    vertex_occupation[vertex_index] = atom_index
                    voxel_atom_index += 1
                    continue

                distance_vector = (
                    get_atom_coords(atoms_data[atom_index - 1]) - contact_point
                )
                distance_squared = dot_product(distance_vector, distance_vector)

                if distance_squared < atom_plus_probe_radii_shrink_1d[atom_index - 1]:
                    vertex_occupation[vertex_index] = atom_index
                    vertex_index += 1
                    no_jump_to_outer = False
                    break
                voxel_atom_index += 1
            if no_jump_to_outer:
                num_visible_vertices += 1
                num_exposed_grids += 1
                if num_exposed_grids >= exposed_grid_coordinates.shape[0]:
                    new_exposed_grid_count = int(
                        exposed_grid_coordinates.shape[0] * RESIZE_FACTOR
                    )
                    exposed_grid_coordinates = np.resize(
                        exposed_grid_coordinates, (new_exposed_grid_count + 1, 3)
                    )
                exposed_grid_coordinates[num_exposed_grids] = contact_point
                vertex_occupation[vertex_index] = 0
                vertex_index += 1

        edge_state_count = 0
        if refinement_level > 0:
            for edge_index in range(initial_vertex_count, 0, -1):
                vertex_index1 = vertex_occupation[edge_array[edge_index][1]]
                vertex_index2 = vertex_occupation[edge_array[edge_index][2]]

                if vertex_index1 > 0 and vertex_index1 == vertex_index2:
                    continue
                edge_state_count += 1
                edge_state[edge_state_count] = edge_index

        if edge_state_count > 0:
            loop_30_break = True
            while loop_30_break:
                loop_30_cont = True
                edge_index = edge_state[edge_state_count]
                edge_state_count -= 1
                vertex_index1 = vertex_occupation[edge_array[edge_index][1]]
                vertex_index2 = vertex_occupation[edge_array[edge_index][2]]

                if vertex_index1 > num_atoms or vertex_index2 > num_atoms:
                    if edge_state_count > 0:
                        loop_30_cont = False
                        continue
                    loop_30_break = False
                    break

                refined_vertex_index = edge_index + initial_vertex_count
                contact_point[0] = real_type(
                    dot_product(rotation_matrix[1], vertex_array[refined_vertex_index])
                )
                contact_point[1] = real_type(
                    dot_product(rotation_matrix[2], vertex_array[refined_vertex_index])
                )
                contact_point[2] = real_type(
                    dot_product(rotation_matrix[3], vertex_array[refined_vertex_index])
                )
                contact_point = midpoint_coords + (
                    contact_point
                    * real_type(
                        interaction_radius
                    )  # Ensure interaction_radius is real_type
                )

                if vertex_index1 != 0:
                    distance_vector = (
                        get_atom_coords(atoms_data[vertex_index1 - 1]) - contact_point
                    )
                    distance_squared = dot_product(distance_vector, distance_vector)

                    if (
                        distance_squared
                        < atom_plus_probe_radii_shrink_1d[vertex_index1 - 1]
                    ):
                        vertex_occupation[refined_vertex_index] = vertex_index1

                        if edges[edge_index] > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index] + 1

                        if edge_state_count > 0:
                            loop_30_cont = False
                            continue
                        loop_30_break = False
                        break

                if vertex_index2 != 0:
                    distance_vector = (
                        get_atom_coords(atoms_data[vertex_index2 - 1]) - contact_point
                    )
                    distance_squared = dot_product(distance_vector, distance_vector)

                    if (
                        distance_squared
                        < atom_plus_probe_radii_shrink_1d[vertex_index2 - 1]
                    ):
                        vertex_occupation[refined_vertex_index] = vertex_index2

                        if edges[edge_index] > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index]

                        if edge_state_count > 0:
                            loop_30_cont = False
                            continue
                        loop_30_break = False
                        break

                grid_indices = (
                    (contact_point - voxel_space_origin) * voxel_space_scale
                ).astype(int_type)
                lower_limit = voxel_atom_start_indices[grid_indices[0]][
                    grid_indices[1]
                ][grid_indices[2]]
                upper_limit = voxel_atom_end_indices[grid_indices[0]][grid_indices[1]][
                    grid_indices[2]
                ]

                voxel_atom_index = lower_limit
                loop_50_break = True
                while (
                    (loop_30_break and loop_30_cont)
                    and loop_50_break
                    and voxel_atom_index <= upper_limit
                ):
                    loop_50_cont = True
                    atom_index = voxel_atom_indices[voxel_atom_index]

                    if atom_index > num_atoms:
                        vertex_occupation[refined_vertex_index] = atom_index
                        voxel_atom_index += 1
                        loop_50_cont = False
                        continue

                    distance_vector = (
                        get_atom_coords(atoms_data[atom_index - 1]) - contact_point
                    )
                    distance_squared = dot_product(distance_vector, distance_vector)

                    if (
                        distance_squared
                        < atom_plus_probe_radii_shrink_1d[atom_index - 1]
                    ):
                        vertex_occupation[refined_vertex_index] = atom_index

                        if edges[edge_index] > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index] + 1
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index]

                        if edge_state_count > 0:
                            loop_30_cont = False
                            break
                        loop_30_break = False
                        break
                    if loop_30_break:
                        voxel_atom_index += 1
                if loop_30_break and loop_30_cont:
                    num_visible_vertices += 1
                    num_exposed_grids += 1
                    if num_exposed_grids >= exposed_grid_coordinates.shape[0]:
                        new_exposed_grid_count = int(
                            exposed_grid_coordinates.shape[0] * RESIZE_FACTOR
                        )
                        exposed_grid_coordinates = np.resize(
                            exposed_grid_coordinates, (new_exposed_grid_count + 1, 3)
                        )
                    exposed_grid_coordinates[num_exposed_grids] = contact_point

                    vertex_occupation[refined_vertex_index] = 0
                    if edges[edge_index] > 0:
                        if edges[edges[edge_index] + 1] > 0 or vertex_index2 > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index] + 1

                        if edges[edges[edge_index]] > 0 or vertex_index1 > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index]

                    if edge_state_count <= 0:
                        loop_30_break = False
                        break

        if num_visible_vertices > 0:
            num_exposed_atom_pairs += 1
            atom_contact_state[atom_index1] = 0
            if atom_index2 <= num_atoms:
                atom_contact_state[atom_index2] = 0

    num_accessible_atoms = 0
    for atom_index in range(1, num_atoms + 1):
        if atom_contact_state[atom_index] == 0:
            num_accessible_atoms += 1

    return (
        num_exposed_grids,
        exposed_grid_coordinates,
        num_accessible_atoms,
        atom_contact_state,
        num_exposed_atom_pairs,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def solvent_accessible_surface(
    probe_radius,
    probe_radius2,
    max_atom_radius,
    min_coords_by_axis,
    max_coords_by_axis,
    num_atoms,
    num_objects,
    num_molecules,
    atoms_data,
    atom_plus_probe_radii_1d,
    atom_plus_probe_radii_shrink_1d,
    num_vertices=520,
    num_edges=1040,
):
    initial_vertex_count = 12
    refinement_level = 5

    max_probe_radius = np.maximum(probe_radius, probe_radius2)
    voxel_side_length = 2.0 * (max_atom_radius + max_probe_radius)
    voxel_space_scale = 1.0 / voxel_side_length

    voxel_space_origin, voxel_space_shape = voxelizer.calculate_voxel_space_parameters(
        voxel_side_length,
        min_coords_by_axis,
        max_coords_by_axis,
        scaling_factor=1.0,
    )

    num_extended_voxel_space_grids = (
        (voxel_space_shape[0] + 1)
        * (voxel_space_shape[1] + 1)
        * (voxel_space_shape[2] + 1)
    )
    max_voxel_atoms = delphi_int(27)
    if (num_objects - num_molecules) > 0:
        max_voxel_atoms = delphi_int(
            max(num_extended_voxel_space_grids, max_voxel_atoms)
        )

    voxel_atom_ids = np.zeros(
        max_voxel_atoms * (num_atoms + num_objects - num_molecules) + 1,
        dtype=delphi_int,
    )

    (voxel_atom_ids, voxel_atom_start_indices, voxel_atom_end_indices) = (
        voxelizer.build_atom_voxel_map(
            voxel_side_length,
            num_atoms,
            num_objects,
            num_molecules,
            voxel_space_origin,
            voxel_space_shape,
            atoms_data,
            voxel_atom_ids,
        )
    )

    vertex_array, edge_array, edges = _initialize_vertices_and_edges(
        num_vertices, num_edges, initial_vertex_count, delphi_real, delphi_int
    )
    (
        vertex_array,
        edge_array,
        edges,
        current_vertex_count,
        current_edge_count,
    ) = _refine_vertices_and_edges(
        vertex_array,
        edge_array,
        edges,
        refinement_level,
        initial_vertex_count,
        delphi_real,
        delphi_int,
    )

    nprint_cpu(
        DEBUG,
        _VERBOSITY,
        " # of vertex_array             :",
        current_vertex_count,
    )
    nprint_cpu(DEBUG, _VERBOSITY, " # of edges                :", current_edge_count)

    # For each atom whether the atom is in contact to any other atom.
    atom_contact_state = np.ones(
        num_atoms + 1, dtype=np.bool_
    )  # first atom index is 1.
    atom_contact_state, atom_pairs = _find_atom_pairs(
        num_atoms,
        atoms_data,
        atom_plus_probe_radii_1d,
        atom_contact_state,
        voxel_space_origin,
        voxel_space_scale,
        voxel_atom_start_indices,
        voxel_atom_end_indices,
        voxel_atom_ids,
        delphi_real,
        delphi_int,
    )

    nprint_cpu(DEBUG, _VERBOSITY, " # of pairs                :", atom_pairs.shape[0])

    voxel_side_length = max_atom_radius + max_probe_radius
    voxel_space_scale = 1.0 / voxel_side_length

    (voxel_space_origin, voxel_space_shape) = (
        voxelizer.calculate_voxel_space_parameters(
            voxel_side_length,
            min_coords_by_axis,
            max_coords_by_axis,
            scaling_factor=2.0,
        )
    )
    # Needed to recalculate as te scaling_factor has changed to 2.0
    num_extended_voxel_space_grids = (
        (voxel_space_shape[0] + 1)
        * (voxel_space_shape[1] + 1)
        * (voxel_space_shape[2] + 1)
    )

    max_voxel_atoms = delphi_int(27)
    if (num_objects - num_molecules) > 0:
        max_voxel_atoms = delphi_int(
            max(max_voxel_atoms, num_extended_voxel_space_grids)
        )

    voxel_atom_ids = np.zeros(
        max_voxel_atoms * (num_atoms + num_objects - num_molecules) + 1,
        dtype=delphi_int,
    )

    (voxel_atom_ids, voxel_atom_start_indices, voxel_atom_end_indices) = (
        voxelizer.build_atom_voxel_map(
            voxel_side_length,
            num_atoms,
            num_objects,
            num_molecules,
            voxel_space_origin,
            voxel_space_shape,
            atoms_data,
            voxel_atom_ids,
        )
    )

    (
        num_exposed_grids,
        exposed_grids_coords,
        num_accessible_atoms,
        atom_contact_state,
        num_exposed_atom_pairs,
    ) = _calculate_exposed_grids(
        num_atoms=num_atoms,
        atoms_data=atoms_data,
        atom_plus_probe_radii_1d=atom_plus_probe_radii_1d,
        atom_plus_probe_radii_shrink_1d=atom_plus_probe_radii_shrink_1d,
        atom_contact_state=atom_contact_state,
        atom_pairs=atom_pairs,
        vertex_array=vertex_array,
        edge_array=edge_array,
        edges=edges,
        voxel_space_origin=voxel_space_origin,
        voxel_space_scale=voxel_space_scale,
        voxel_atom_start_indices=voxel_atom_start_indices,
        voxel_atom_end_indices=voxel_atom_end_indices,
        voxel_atom_indices=voxel_atom_ids,
        num_vertices=current_vertex_count,
        num_edges=num_edges,
        refinement_level=refinement_level,
        initial_vertex_count=initial_vertex_count,
        num_extended_cube_grid_points=num_extended_voxel_space_grids,
        real_type=delphi_real,
        int_type=delphi_int,
    )
    nprint_cpu(
        DEBUG,
        _VERBOSITY,
        "# exposed pairs (atom-atom and atom-object)= ",
        num_exposed_atom_pairs,
    )
    nprint_cpu(DEBUG, _VERBOSITY, "no. of exposed/arc points = ", num_exposed_grids)
    nprint_cpu(
        DEBUG,
        _VERBOSITY,
        "no. surface atoms = ",
        num_accessible_atoms,
        " no. burried atoms = ",
        num_atoms - num_accessible_atoms,
    )

    return (
        num_exposed_grids,
        exposed_grids_coords,
        atom_contact_state,
        voxel_atom_start_indices,
        voxel_atom_end_indices,
        voxel_space_origin,
        voxel_space_shape,
    )
