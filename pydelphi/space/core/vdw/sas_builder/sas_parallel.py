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

import gc
import math
import numpy as np
from numba import njit, prange, cuda, int64, float64, set_num_threads

from pydelphi.foundation.enums import Precision

from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_int,
    delphi_real,
    nprint_cpu_if_verbose as nprint_cpu,
    vprint,
)
from pydelphi.config.logging_config import (
    TRACE,
    VERBOSE,
    DEBUG,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

if PRECISION.value == Precision.SINGLE.value:
    from pydelphi.utils.prec.single import dot_product

elif PRECISION.value == Precision.DOUBLE.value:
    from pydelphi.utils.prec.double import dot_product

import pydelphi.space.core.voxelizer as voxelizer

from pydelphi.constants import (
    ConstPhysical,
    ConstDelPhiFloats,
    ConstDelPhiInts,
    ATOMFIELD_RADIUS,
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_CRD_END,
)

APPROX_ZERO = ConstDelPhiFloats.ApproxZero.value
CONST_PI = ConstPhysical.Pi.value
RESIZE_FACTOR = ConstDelPhiFloats.ZetaArrayResizeFactor.value

ATOM_PAIRS_COUNT_MINIMAL = ConstDelPhiFloats.SASAtomPairsMinimalCount.value
ATOM_PAIRS_LINEAR_FACTOR = ConstDelPhiFloats.SASLinearPairsFactorPerAtom.value
ATOM_PAIRS_N_SQUARED_FACTOR = ConstDelPhiFloats.SASQuadraticPairsFactorOfNSquared.value

EXPOSED_GRIDS_COUNT_MINIMAL = ConstDelPhiFloats.SASExposedGridsMinimalCount.value
EXPOSED_GRIDS_SURFACE_FACTOR = ConstDelPhiFloats.SASExposedGridsSurfaceAreaFactor.value

SERIAL_SURF_THREADS_THRESHOLD = ConstDelPhiInts.SerialVDWSurfaceThreadsThreshold.value

SAS_MAX_VERTICES = ConstDelPhiInts.SASMaxVertices.value
SAS_MAX_EDGES = ConstDelPhiInts.SASMaxEdges.value
SAS_THREADS_PER_BLOCK = ConstDelPhiInts.SASThreadsPerBlock.value
SAS_BLOCKS_PER_GRID = ConstDelPhiInts.SASBlocksPerGrid.value


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


@njit(inline="always", nogil=True, boundscheck=False, cache=True)
def _cpu_clamped_voxel_indices(
    atom_coords, cube_vertex_lowest_xyz, cube_side_length_inverse, dims, int_type
):
    """Return clamped voxel indices (x,y,z) as integers for CPU code."""
    voxel_indices = (
        (atom_coords - cube_vertex_lowest_xyz) * cube_side_length_inverse
    ).astype(int_type)
    x = voxel_indices[0]
    if x < 0:
        x = 0
    elif x > dims[0] - 1:
        x = dims[0] - 1
    y = voxel_indices[1]
    if y < 0:
        y = 0
    elif y > dims[1] - 1:
        y = dims[1] - 1
    z = voxel_indices[2]
    if z < 0:
        z = 0
    elif z > dims[2] - 1:
        z = dims[2] - 1
    return x, y, z


@njit(inline="always", nogil=True, boundscheck=False, cache=True)
def _cpu_is_atom_valid_for_tests(atoms_data, atom_index, approx_zero_threshold):
    """Check atom exists and has positive radius. `atom_index` is 1-based in some places; caller should pass correct index."""
    # atom_index here assumed to be 0-based for direct indexing; if callers pass 1-based adjust accordingly
    return atoms_data[atom_index][ATOMFIELD_RADIUS] > approx_zero_threshold


@njit(inline="always", nogil=True, boundscheck=False, cache=True)
def _cpu_pair_contact_test(atom_coords1, atom_coords2, radius1, radius2):
    """Return True if the pair (atom1, atom2) is considered contacting by your geometric test."""
    distance_vector = atom_coords2 - atom_coords1
    distance_squared = dot_product(distance_vector, distance_vector)

    combined_radius = radius1 + radius2
    combined_radius_squared = combined_radius * combined_radius

    radius_difference = abs(radius1 - radius2)
    radius_difference_squared = radius_difference * radius_difference

    delta = combined_radius_squared - distance_squared

    # keep same thresholds as original
    return (delta > 0.01) and (distance_squared > radius_difference_squared)


@njit(inline="always", nogil=True, boundscheck=False, cache=True)
def _cpu_write_contacting_pair(array, idx, atom1_1based, atom2_1based):
    array[idx, 0] = atom1_1based
    array[idx, 1] = atom2_1based
    array[idx, 2] = 0


@njit(nogil=True, boundscheck=False, cache=True)
def _cpu_find_atom_pairs_serial(
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

    dims = voxel_atom_counts.shape  # (nx, ny, nz)

    for atom_index1 in range(num_atoms):
        # skip zero-radius atoms
        if atoms_data[atom_index1][ATOMFIELD_RADIUS] < APPROX_ZERO:
            continue

        radius1 = atom_plus_probe_radii[atom_index1]
        atom1_coords = atoms_data[atom_index1][ATOMFIELD_X:ATOMFIELD_CRD_END]

        x, y, z = _cpu_clamped_voxel_indices(
            atom1_coords,
            cube_vertex_lowest_xyz,
            cube_side_length_inverse,
            dims,
            int_type,
        )

        lower_limit = voxel_atom_counts[x][y][z]
        upper_limit = voxel_cumulative_atom_counts[x][y][z]

        previous_atom_index = 0
        for voxel_atom_index in range(lower_limit, upper_limit + 1):
            atom_index2 = voxel_atom_indices[voxel_atom_index]
            # keep previous semantics: code treated atom_index2 as 1-based sometimes
            if atom_index2 <= 0 or atom_index2 > num_atoms:
                continue

            if atom_index2 <= atom_index1 + 1:
                continue

            a2_zero = atom_index2 - 1  # zero based index of atom: a2

            if atoms_data[a2_zero][ATOMFIELD_RADIUS] < APPROX_ZERO:
                continue

            radius2 = atom_plus_probe_radii[a2_zero]
            coords2 = atoms_data[a2_zero][ATOMFIELD_X:ATOMFIELD_CRD_END]

            if _cpu_pair_contact_test(atom1_coords, coords2, radius1, radius2):
                if num_contacting_atom_pairs_current >= contacting_atom_pairs.shape[0]:
                    new_pair_count = int(contacting_atom_pairs.shape[0] * RESIZE_FACTOR)
                    contacting_atom_pairs = np.resize(
                        contacting_atom_pairs, (new_pair_count, 3)
                    )

                _cpu_write_contacting_pair(
                    contacting_atom_pairs,
                    num_contacting_atom_pairs_current,
                    atom_index1 + 1,
                    atom_index2,
                )
                num_contacting_atom_pairs_current += 1

            previous_atom_index = atom_index2

        if num_contacting_atom_pairs_current == num_contacting_atom_pairs:
            atom_contact_state[atom_index1 + 1] = 0

        num_contacting_atom_pairs = num_contacting_atom_pairs_current

    return atom_contact_state, contacting_atom_pairs[:num_contacting_atom_pairs]


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _cpu_find_atom_pairs_parallel(
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
    dims = voxel_atom_counts.shape  # (nx, ny, nz)

    # --- PASS 1: count pairs per atom ---
    counts = np.zeros(num_atoms, dtype=int_type)
    for atom_index1 in prange(num_atoms):
        if atoms_data[atom_index1][ATOMFIELD_RADIUS] < APPROX_ZERO:
            counts[atom_index1] = 0
            continue
        radius1 = atom_plus_probe_radii[atom_index1]
        atom1_coords = atoms_data[atom_index1][ATOMFIELD_X:ATOMFIELD_CRD_END]

        x, y, z = _cpu_clamped_voxel_indices(
            atom1_coords,
            cube_vertex_lowest_xyz,
            cube_side_length_inverse,
            dims,
            int_type,
        )

        lower = voxel_atom_counts[x, y, z]
        upper = voxel_cumulative_atom_counts[x, y, z]

        count = 0
        for voxel_atom_index in range(lower, upper + 1):
            atom_index2 = voxel_atom_indices[voxel_atom_index]

            # check valid atom index range (keeps original semantics)
            if atom_index2 <= 0 or atom_index2 > num_atoms:
                continue

            if atom_index2 <= atom_index1 + 1:
                continue

            a2_zero = atom_index2 - 1
            if atoms_data[a2_zero][ATOMFIELD_RADIUS] < APPROX_ZERO:
                continue

            radius2 = atom_plus_probe_radii[a2_zero]
            coords2 = atoms_data[a2_zero][ATOMFIELD_X:ATOMFIELD_CRD_END]

            if _cpu_pair_contact_test(atom1_coords, coords2, radius1, radius2):
                count += 1

        counts[atom_index1] = count

    # --- compute offsets ---
    offsets = np.zeros(num_atoms + 1, dtype=int_type)
    total = 0
    for i in range(num_atoms):
        offsets[i] = total
        total += counts[i]
    offsets[num_atoms] = total

    # --- PASS 2: fill array ---
    contacting_atom_pairs = np.zeros((total, 3), dtype=int_type)
    for atom_index1 in prange(num_atoms):
        start = offsets[atom_index1]
        idx = start

        if atoms_data[atom_index1][ATOMFIELD_RADIUS] < APPROX_ZERO:
            # mark no contacts consistent with original
            continue
        radius1 = atom_plus_probe_radii[atom_index1]
        atom1_coords = atoms_data[atom_index1][ATOMFIELD_X:ATOMFIELD_CRD_END]

        x, y, z = _cpu_clamped_voxel_indices(
            atom1_coords,
            cube_vertex_lowest_xyz,
            cube_side_length_inverse,
            dims,
            int_type,
        )

        lower = voxel_atom_counts[x, y, z]
        upper = voxel_cumulative_atom_counts[x, y, z]

        for voxel_atom_index in range(lower, upper + 1):
            atom_index2 = voxel_atom_indices[voxel_atom_index]

            if atom_index2 <= 0 or atom_index2 > num_atoms:
                continue
            if atom_index2 <= atom_index1 + 1:
                continue
            a2_zero = atom_index2 - 1
            if atoms_data[a2_zero][ATOMFIELD_RADIUS] < APPROX_ZERO:
                continue

            radius2 = atom_plus_probe_radii[a2_zero]
            coords2 = atoms_data[a2_zero][ATOMFIELD_X:ATOMFIELD_CRD_END]

            if _cpu_pair_contact_test(atom1_coords, coords2, radius1, radius2):
                _cpu_write_contacting_pair(
                    contacting_atom_pairs, idx, atom_index1 + 1, atom_index2
                )
                idx += 1

        if idx == start:  # no contacts
            atom_contact_state[atom_index1 + 1] = 0

    return atom_contact_state, contacting_atom_pairs


@cuda.jit(device=True)
def _cuda_clamped_voxel_indices(
    atom_coords,
    cube_vertex_lowest_xyz,
    cube_side_length_inverse,
    dims_x,
    dims_y,
    dims_z,
):
    """Return clamped voxel indices (x,y,z) as ints for CUDA device kernels."""
    # Numba CUDA doesn't support vectorized .astype; do scalar ops
    vx = int((atom_coords[0] - cube_vertex_lowest_xyz[0]) * cube_side_length_inverse)
    if vx < 0:
        vx = 0
    elif vx > dims_x - 1:
        vx = dims_x - 1

    vy = int((atom_coords[1] - cube_vertex_lowest_xyz[1]) * cube_side_length_inverse)
    if vy < 0:
        vy = 0
    elif vy > dims_y - 1:
        vy = dims_y - 1

    vz = int((atom_coords[2] - cube_vertex_lowest_xyz[2]) * cube_side_length_inverse)
    if vz < 0:
        vz = 0
    elif vz > dims_z - 1:
        vz = dims_z - 1

    return vx, vy, vz


@cuda.jit(device=True)
def _cuda_pair_contact_test(atom_coords1, atom_coords2, radius1, radius2):
    """CUDA device-side geometric test. Returns 1 if contacting, else 0."""
    dx = atom_coords2[0] - atom_coords1[0]
    dy = atom_coords2[1] - atom_coords1[1]
    dz = atom_coords2[2] - atom_coords1[2]
    distance_squared = dx * dx + dy * dy + dz * dz

    combined_radius = radius1 + radius2
    combined_radius_squared = combined_radius * combined_radius

    radius_diff = radius1 - radius2
    if radius_diff < 0:
        radius_diff = -radius_diff
    radius_diff_squared = radius_diff * radius_diff

    delta = combined_radius_squared - distance_squared

    if delta > 0.01 and distance_squared > radius_diff_squared:
        return 1
    else:
        return 0


@cuda.jit(device=True)
def _cuda_write_contacting_pair(contacting_array, idx, atom1_1based, atom2_1based):
    contacting_array[idx, 0] = atom1_1based
    contacting_array[idx, 1] = atom2_1based
    contacting_array[idx, 2] = 0


@cuda.jit(cache=True)
def _cuda_find_atom_pairs_count_contacts_pass1(
    num_atoms,
    atoms_data,
    atom_plus_probe_radii,
    voxel_atom_counts,
    voxel_cumulative_atom_counts,
    voxel_atom_indices,
    cube_vertex_lowest_xyz,
    cube_side_length_inverse,
    counts,
):
    atom_index1 = cuda.grid(1)
    if atom_index1 >= num_atoms:
        return

    if atoms_data[atom_index1][ATOMFIELD_RADIUS] < APPROX_ZERO:
        counts[atom_index1] = 0
        return

    radius1 = atom_plus_probe_radii[atom_index1]
    atom1_coords = atoms_data[atom_index1][ATOMFIELD_X:ATOMFIELD_CRD_END]

    dims_x, dims_y, dims_z = voxel_atom_counts.shape

    x, y, z = _cuda_clamped_voxel_indices(
        atom1_coords,
        cube_vertex_lowest_xyz,
        cube_side_length_inverse,
        dims_x,
        dims_y,
        dims_z,
    )

    lower = voxel_atom_counts[x, y, z]
    upper = voxel_cumulative_atom_counts[x, y, z]

    count = 0
    for voxel_atom_index in range(lower, upper + 1):
        atom_index2 = voxel_atom_indices[voxel_atom_index]
        if atom_index2 <= 0 or atom_index2 > num_atoms:
            continue
        if atom_index2 <= atom_index1 + 1:
            continue
        # convert to zero-based
        a2_zero = atom_index2 - 1
        if atoms_data[a2_zero][ATOMFIELD_RADIUS] < APPROX_ZERO:
            continue
        radius2 = atom_plus_probe_radii[a2_zero]
        coords2 = atoms_data[a2_zero][ATOMFIELD_X:ATOMFIELD_CRD_END]

        if _cuda_pair_contact_test(atom1_coords, coords2, radius1, radius2):
            count += 1

    counts[atom_index1] = count


@cuda.jit(cache=True)
def _cuda_find_atom_pairs_fill_contacts_pass2(
    num_atoms,
    atoms_data,
    atom_plus_probe_radii,
    voxel_atom_counts,
    voxel_cumulative_atom_counts,
    voxel_atom_indices,
    cube_vertex_lowest_xyz,
    cube_side_length_inverse,
    offsets,
    contacting_atom_pairs,
    atom_contact_state,
):
    atom_index1 = cuda.grid(1)
    if atom_index1 >= num_atoms:
        return

    start = offsets[atom_index1]
    idx = start

    if atoms_data[atom_index1][ATOMFIELD_RADIUS] < APPROX_ZERO:
        return

    radius1 = atom_plus_probe_radii[atom_index1]
    atom1_coords = atoms_data[atom_index1][ATOMFIELD_X:ATOMFIELD_CRD_END]

    dims_x, dims_y, dims_z = voxel_atom_counts.shape
    x, y, z = _cuda_clamped_voxel_indices(
        atom1_coords,
        cube_vertex_lowest_xyz,
        cube_side_length_inverse,
        dims_x,
        dims_y,
        dims_z,
    )

    lower = voxel_atom_counts[x, y, z]
    upper = voxel_cumulative_atom_counts[x, y, z]

    for voxel_atom_index in range(lower, upper + 1):
        atom_index2 = voxel_atom_indices[voxel_atom_index]
        if atom_index2 <= 0 or atom_index2 > num_atoms:
            continue
        if atom_index2 <= atom_index1 + 1:
            continue
        a2_zero = atom_index2 - 1
        if atoms_data[a2_zero][ATOMFIELD_RADIUS] < APPROX_ZERO:
            continue

        radius2 = atom_plus_probe_radii[a2_zero]
        coords2 = atoms_data[a2_zero][ATOMFIELD_X:ATOMFIELD_CRD_END]

        if _cuda_pair_contact_test(atom1_coords, coords2, radius1, radius2):
            _cuda_write_contacting_pair(
                contacting_atom_pairs, idx, atom_index1 + 1, atom_index2
            )
            idx += 1

    if idx == start:
        atom_contact_state[atom_index1 + 1] = 0


def _cuda_find_atom_pairs(
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
    # --- Pass 1: count contacts ---
    counts = np.zeros(num_atoms, dtype=np.int32)
    d_counts = cuda.to_device(counts)
    d_atoms_data = cuda.to_device(atoms_data)
    d_atom_plus_probe_radii = cuda.to_device(atom_plus_probe_radii)
    d_voxel_atom_counts = cuda.to_device(voxel_atom_counts)
    d_voxel_cumulative_atom_counts = cuda.to_device(voxel_cumulative_atom_counts)
    d_voxel_atom_indices = cuda.to_device(voxel_atom_indices)
    d_cube_vertex_lowest_xyz = cuda.to_device(cube_vertex_lowest_xyz)

    threads_per_block = 128
    blocks = (num_atoms + threads_per_block - 1) // threads_per_block
    _cuda_find_atom_pairs_count_contacts_pass1[int(blocks), int(threads_per_block)](
        num_atoms,
        d_atoms_data,
        d_atom_plus_probe_radii,
        d_voxel_atom_counts,
        d_voxel_cumulative_atom_counts,
        d_voxel_atom_indices,
        d_cube_vertex_lowest_xyz,
        cube_side_length_inverse,
        d_counts,
    )
    cuda.synchronize()
    counts = d_counts.copy_to_host()

    # --- CPU prefix sum / offsets ---
    offsets = np.zeros(num_atoms + 1, dtype=np.int32)
    total = 0
    for i in range(num_atoms):
        offsets[i] = total
        total += counts[i]
    offsets[num_atoms] = total

    # --- Allocate output ---
    contacting_atom_pairs = np.zeros((total, 3), dtype=np.int32)
    d_offsets = cuda.to_device(offsets)
    d_contacting_atom_pairs = cuda.to_device(contacting_atom_pairs)
    d_atom_contact_state = cuda.to_device(atom_contact_state)

    # --- Pass 2: fill contacts ---
    _cuda_find_atom_pairs_fill_contacts_pass2[int(blocks), int(threads_per_block)](
        num_atoms,
        d_atoms_data,
        d_atom_plus_probe_radii,
        d_voxel_atom_counts,
        d_voxel_cumulative_atom_counts,
        d_voxel_atom_indices,
        d_cube_vertex_lowest_xyz,
        cube_side_length_inverse,
        d_offsets,
        d_contacting_atom_pairs,
        d_atom_contact_state,
    )
    cuda.synchronize()
    contacting_atom_pairs = d_contacting_atom_pairs.copy_to_host()
    atom_contact_state = d_atom_contact_state.copy_to_host()

    d_counts = None
    d_atoms_data = None
    d_atom_plus_probe_radii = None
    d_voxel_atom_counts = None
    d_voxel_cumulative_atom_counts = None
    d_voxel_atom_indices = None
    d_cube_vertex_lowest_xyz = None
    d_offsets = None
    d_contacting_atom_pairs = None
    d_atom_contact_state = None

    return atom_contact_state, contacting_atom_pairs


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
    use_cuda,
    num_threads,
    real_type,
    int_type,
):
    if use_cuda:
        # two-pass CUDA version
        return _cuda_find_atom_pairs(
            num_atoms,
            atoms_data,
            atom_plus_probe_radii,
            atom_contact_state,
            cube_vertex_lowest_xyz,
            cube_side_length_inverse,
            voxel_atom_counts,
            voxel_cumulative_atom_counts,
            voxel_atom_indices,
        )
    else:
        # serial CPU implementation
        if num_threads < SERIAL_SURF_THREADS_THRESHOLD:
            return _cpu_find_atom_pairs_serial(
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
            )
        else:
            set_num_threads(num_threads)
            return _cpu_find_atom_pairs_parallel(
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
            )


def _debug_print_contact_pairs(atom_pairs, atoms_data, atom_plus_probe_radii_1d, label):
    """
    Debug-only contact-pair dump.

    Prints one stable line per discovered contacting pair so serial and parallel
    logs can be diffed with grep/sort/comm.
    """
    if _VERBOSITY > DEBUG:
        return

    print(f"[SAS-CONTACT-PAIRS] label={label} count={atom_pairs.shape[0]}")

    for pair_index in range(atom_pairs.shape[0]):
        atom1 = int(atom_pairs[pair_index, 0])
        atom2 = int(atom_pairs[pair_index, 1])

        a1z = atom1 - 1
        a2z = atom2 - 1

        x1 = float(atoms_data[a1z, ATOMFIELD_X])
        y1 = float(atoms_data[a1z, ATOMFIELD_Y])
        z1 = float(atoms_data[a1z, ATOMFIELD_Z])

        x2 = float(atoms_data[a2z, ATOMFIELD_X])
        y2 = float(atoms_data[a2z, ATOMFIELD_Y])
        z2 = float(atoms_data[a2z, ATOMFIELD_Z])

        r1 = float(atom_plus_probe_radii_1d[a1z])
        r2 = float(atom_plus_probe_radii_1d[a2z])

        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        d2 = dx * dx + dy * dy + dz * dz

        combined = r1 + r2
        delta = combined * combined - d2

        print(
            "[SAS-CONTACT-PAIR] "
            f"label={label} "
            f"pair_index={pair_index} "
            f"atom1={atom1} atom2={atom2} "
            f"a1_xyz=({x1:.6f},{y1:.6f},{z1:.6f}) "
            f"a2_xyz=({x2:.6f},{y2:.6f},{z2:.6f}) "
            f"r1p={r1:.6f} r2p={r2:.6f} "
            f"d2={d2:.12e} delta={delta:.12e}"
        )


# --- Reusable Inline Helpers For _calculate_exposed_grids_* functions ---


@njit(inline="always", nogil=True)
def dot_product(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@njit(inline="always", nogil=True)
def _compute_midpoint_and_radius(
    atom_coords1, atom_coords2, r1, r2, out_midpoint, out_dv
):
    # Compute distance vector element-wise
    for i in range(3):
        out_dv[i] = atom_coords2[i] - atom_coords1[i]

    # Compute squared distance
    d2 = out_dv[0] * out_dv[0] + out_dv[1] * out_dv[1] + out_dv[2] * out_dv[2]

    if d2 <= APPROX_ZERO * APPROX_ZERO:
        # Midpoint for coincident atoms
        for i in range(3):
            out_midpoint[i] = 0.5 * (atom_coords1[i] + atom_coords2[i])
        interaction_radius = 0.0
        distance = 0.0
        return interaction_radius, distance

    d = math.sqrt(d2)
    prefactor = 1.0 + (r1 * r1 - r2 * r2) / d2

    # Midpoint
    for i in range(3):
        out_midpoint[i] = atom_coords1[i] + 0.5 * prefactor * out_dv[i]

    sum_r = r1 + r2
    diff_r = r1 - r2
    term1_sq = sum_r * sum_r - d2
    term2_sq = d2 - diff_r * diff_r

    if term1_sq <= 0.0 or term2_sq <= 0.0:
        interaction_radius = 0.0
    else:
        interaction_radius = 0.5 * math.sqrt(term1_sq) * math.sqrt(term2_sq) / d

    return interaction_radius, d


@njit(inline="always", nogil=True)
def _compute_rotation_matrix(
    distance_vector, distance_magnitude, xy_proj, rotation_matrix
):
    dx1, dx2, dx3 = distance_vector

    if xy_proj > 1.0e-8:
        tx = -dx2 / xy_proj
        ty = dx1 / xy_proj
        cos_t = dx3 / distance_magnitude

        # Clamp for safety
        if cos_t > 1.0:
            cos_t = 1.0
        elif cos_t < -1.0:
            cos_t = -1.0

        sin_t = math.sqrt(1.0 - cos_t * cos_t)
        one_minus_cos = 1.0 - cos_t
        temp_mul = one_minus_cos * tx
        sin_t_tx = sin_t * tx
        sin_t_ty = sin_t * ty

        rotation_matrix[0][0] = temp_mul * tx + cos_t
        rotation_matrix[0][1] = temp_mul * ty
        rotation_matrix[0][2] = sin_t_ty

        rotation_matrix[1][0] = temp_mul * ty
        rotation_matrix[1][1] = one_minus_cos * ty * ty + cos_t
        rotation_matrix[1][2] = -sin_t_tx

        rotation_matrix[2][0] = -sin_t_ty
        rotation_matrix[2][1] = sin_t_tx
        rotation_matrix[2][2] = cos_t
    else:
        rotation_matrix[0][0], rotation_matrix[0][1], rotation_matrix[0][2] = (
            1.0,
            0.0,
            0.0,
        )
        rotation_matrix[1][0], rotation_matrix[1][1], rotation_matrix[1][2] = (
            0.0,
            1.0,
            0.0,
        )
        rotation_matrix[2][0], rotation_matrix[2][1], rotation_matrix[2][2] = (
            0.0,
            0.0,
            1.0,
        )

    return rotation_matrix


@njit(inline="always", nogil=True, boundscheck=False)
def _calculate_contact_point_coords(
    midpoint_coords,
    interaction_radius,
    rotation_matrix,
    vertex_coords,
    contact_point_out,
):
    """
    Calculates the 3D world coordinate of the potential contact point based on a
    rotated sphere vertex. Writes the result to contact_point_out (in-place).
    """
    # 1. Rotate vertex_coords by rotation_matrix
    cp0 = (
        rotation_matrix[0, 0] * vertex_coords[0]
        + rotation_matrix[0, 1] * vertex_coords[1]
        + rotation_matrix[0, 2] * vertex_coords[2]
    )
    cp1 = (
        rotation_matrix[1, 0] * vertex_coords[0]
        + rotation_matrix[1, 1] * vertex_coords[1]
        + rotation_matrix[1, 2] * vertex_coords[2]
    )
    cp2 = (
        rotation_matrix[2, 0] * vertex_coords[0]
        + rotation_matrix[2, 1] * vertex_coords[1]
        + rotation_matrix[2, 2] * vertex_coords[2]
    )

    # 2. Scale by interaction radius and translate by midpoint
    contact_point_out[0] = midpoint_coords[0] + cp0 * interaction_radius
    contact_point_out[1] = midpoint_coords[1] + cp1 * interaction_radius
    contact_point_out[2] = midpoint_coords[2] + cp2 * interaction_radius


@njit(inline="always", nogil=True, boundscheck=False)
def _get_voxel_limits(
    contact_point,
    voxel_space_origin,
    voxel_space_scale,
    voxel_atom_start_indices,
    voxel_atom_end_indices,
):
    """
    Calculates voxel grid indices for the contact point and returns
    (lower_limit, upper_limit) for the voxel atom list.

    Args:
        contact_point (array-like[3]): The (x, y, z) contact point.
        voxel_space_origin (array-like[3]): The voxel space origin (x0, y0, z0).
        voxel_space_scale (float): Scale factor for voxel space.
        voxel_atom_start_indices (3D array): Start indices.
        voxel_atom_end_indices (3D array): End indices.
        int_type (callable): Integer type caster (e.g., np.int32).

    Returns:
        (int, int): lower_limit, upper_limit
    """

    grid_x = int((contact_point[0] - voxel_space_origin[0]) * voxel_space_scale)
    grid_y = int((contact_point[1] - voxel_space_origin[1]) * voxel_space_scale)
    grid_z = int((contact_point[2] - voxel_space_origin[2]) * voxel_space_scale)

    nx = voxel_atom_start_indices.shape[0]
    ny = voxel_atom_start_indices.shape[1]
    nz = voxel_atom_start_indices.shape[2]

    if grid_x < 0:
        grid_x = 0
    elif grid_x >= nx:
        grid_x = nx - 1

    if grid_y < 0:
        grid_y = 0
    elif grid_y >= ny:
        grid_y = ny - 1

    if grid_z < 0:
        grid_z = 0
    elif grid_z >= nz:
        grid_z = nz - 1

    lower_limit = voxel_atom_start_indices[grid_x][grid_y][grid_z]
    upper_limit = voxel_atom_end_indices[grid_x][grid_y][grid_z]

    return lower_limit, upper_limit


@njit(inline="always", nogil=True, boundscheck=False)
def _check_occupancy_by_voxel(
    contact_point,
    lower_limit,
    upper_limit,
    voxel_atom_indices,
    num_atoms,
    atoms_data,
    atom_plus_probe_radii_shrink_1d,
    vertex_occupation_out,
    index_to_update,
):
    """
    Iterates through the local voxel atom list to check if contact_point is covered.

    Args:
        ...
        vertex_occupation_out (array): Array where the found occupying atom index (or marker)
                                       is written at index_to_update.
    Returns:
        is_occupied (bool): True if an atom was found, False otherwise.
    """
    voxel_atom_index = lower_limit
    is_occupied = False

    while voxel_atom_index <= upper_limit:
        atom_index = voxel_atom_indices[voxel_atom_index]

        if atom_index <= 0:
            voxel_atom_index += 1
            continue

        if atom_index > num_atoms:
            # Marker for non-solute atom / boundary condition
            vertex_occupation_out[index_to_update] = atom_index
            voxel_atom_index += 1
            continue

        dx = atoms_data[atom_index - 1][ATOMFIELD_X] - contact_point[0]
        dy = atoms_data[atom_index - 1][ATOMFIELD_Y] - contact_point[1]
        dz = atoms_data[atom_index - 1][ATOMFIELD_Z] - contact_point[2]

        distance_squared = dx * dx + dy * dy + dz * dz

        if distance_squared < atom_plus_probe_radii_shrink_1d[atom_index - 1]:
            # Occupied by atom_index
            vertex_occupation_out[index_to_update] = atom_index
            is_occupied = True
            break

        voxel_atom_index += 1

    if not is_occupied:
        # If no atom was found, occupation remains 0 (exposed)
        vertex_occupation_out[index_to_update] = 0

    return is_occupied


@njit(nogil=True, boundscheck=False, cache=True)
def _calculate_exposed_grids_serial(
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
    num_extended_solute_grid_points,
    real_type,
    int_type,
):
    num_exposed_grids = 0
    initial_exposed_grid_count = int(
        min(
            num_extended_solute_grid_points,
            max(
                EXPOSED_GRIDS_COUNT_MINIMAL,
                EXPOSED_GRIDS_SURFACE_FACTOR
                * (num_extended_solute_grid_points) ** (2 / 3),
            ),
        )
    )
    # The array must be resized inside the function for Numba to handle the array reassignment
    exposed_grid_coordinates = np.zeros(
        (initial_exposed_grid_count + 1, 3), dtype=real_type
    )
    vertex_occupation = np.zeros(num_vertices + 1, dtype=int_type)
    edge_state = np.zeros(num_edges + 1, dtype=int_type)
    rotation_matrix = np.zeros((3, 3), dtype=real_type)
    contact_point = np.zeros(3, dtype=real_type)

    midpoint_coords = np.zeros(3, dtype=real_type)
    distance_vector = np.zeros(3, dtype=real_type)

    num_exposed_atom_pairs = 0
    for pair_index in range(atom_pairs.shape[0]):
        vertex_occupation[:] = 0
        edge_state[:] = 0
        atom_index1 = atom_pairs[pair_index][0]
        atom_index2 = atom_pairs[pair_index][1]
        nprint_cpu(
            TRACE,
            _VERBOSITY,
            f"[PAIR {pair_index}] atoms=({atom_index1}, {atom_index2})",
        )

        atom_coords1 = atoms_data[atom_index1 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
        atom_coords2 = atoms_data[atom_index2 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
        r1 = atom_plus_probe_radii_1d[atom_index1 - 1]
        r2 = atom_plus_probe_radii_1d[atom_index2 - 1]

        # Use helper to compute geometry
        interaction_radius, distance_magnitude = _compute_midpoint_and_radius(
            atom_coords1, atom_coords2, r1, r2, midpoint_coords, distance_vector
        )

        dx1 = distance_vector[0]
        dx2 = distance_vector[1]
        xy_projection_magnitude = real_type(np.sqrt(dx1 * dx1 + dx2 * dx2))

        # Use helper to compute rotation matrix
        _ = _compute_rotation_matrix(
            distance_vector,
            distance_magnitude,
            xy_projection_magnitude,
            rotation_matrix,
        )

        num_visible_vertices = 0
        vertex_index = 1
        while vertex_index <= initial_vertex_count:
            vertex_coords = vertex_array[vertex_index]

            # Use helper 1: Calculate contact point
            _calculate_contact_point_coords(
                midpoint_coords,
                interaction_radius,
                rotation_matrix,
                vertex_coords,
                contact_point,
            )

            # Use helper 2: Get voxel limits
            lower_limit, upper_limit = _get_voxel_limits(
                contact_point,
                voxel_space_origin,
                voxel_space_scale,
                voxel_atom_start_indices,
                voxel_atom_end_indices,
            )

            # Use helper 3: Check occupancy
            is_occupied = _check_occupancy_by_voxel(
                contact_point,
                lower_limit,
                upper_limit,
                voxel_atom_indices,
                num_atoms,
                atoms_data,
                atom_plus_probe_radii_shrink_1d,
                vertex_occupation,
                vertex_index,
            )

            if not is_occupied:
                num_visible_vertices += 1
                num_exposed_grids += 1

                # Manual resize logic for serial function
                if num_exposed_grids >= exposed_grid_coordinates.shape[0]:
                    new_exposed_grid_count = int(
                        exposed_grid_coordinates.shape[0] * RESIZE_FACTOR
                    )
                    exposed_grid_coordinates = np.resize(
                        exposed_grid_coordinates, (new_exposed_grid_count + 1, 3)
                    )

                exposed_grid_coordinates[num_exposed_grids, :] = contact_point

            vertex_index += 1

        nprint_cpu(
            TRACE,
            _VERBOSITY,
            f"[PAIR {pair_index}] visible_vertices={num_visible_vertices}, "
            f"initial_exposed={num_exposed_grids}",
        )

        # --- Refinement Loop Setup ---
        edge_state_count = 0
        if refinement_level > 0:
            for edge_index in range(initial_vertex_count, 0, -1):
                vertex_index1 = vertex_occupation[edge_array[edge_index][1]]
                vertex_index2 = vertex_occupation[edge_array[edge_index][2]]

                if vertex_index1 > 0 and vertex_index1 == vertex_index2:
                    continue
                edge_state_count += 1
                edge_state[edge_state_count] = edge_index

        # --- Refinement Loop Execution (Refactored) ---
        # Loop continues until the stack is empty (edge_state_count <= 0)
        if edge_state_count > 0:
            refinement_level_curr = 1
            while edge_state_count > 0:
                nprint_cpu(
                    TRACE,
                    _VERBOSITY,
                    f"[PAIR {pair_index}] refinement_iter={refinement_level_curr}, "
                    f"edge_state_count={edge_state_count}, "
                    f"exposed={num_exposed_grids}",
                )

                edge_index = edge_state[edge_state_count]
                edge_state_count -= 1
                vertex_index1 = vertex_occupation[edge_array[edge_index][1]]
                vertex_index2 = vertex_occupation[edge_array[edge_index][2]]

                # Case 1: External/Boundary Check
                if vertex_index1 > num_atoms or vertex_index2 > num_atoms:
                    continue

                refined_vertex_index = edge_index + initial_vertex_count
                vertex_coords = vertex_array[refined_vertex_index]

                # Use helper 1: Calculate contact point
                _calculate_contact_point_coords(
                    midpoint_coords,
                    interaction_radius,
                    rotation_matrix,
                    vertex_coords,
                    contact_point,
                )

                # Case 2: Check occupancy by V1 (pre-check)
                found_v1_occupancy = False
                if vertex_index1 != 0 and vertex_index1 <= num_atoms:
                    distance_vector = (
                        atoms_data[vertex_index1 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
                        - contact_point
                    )
                    distance_squared = dot_product(distance_vector, distance_vector)
                    if (
                        distance_squared
                        < atom_plus_probe_radii_shrink_1d[vertex_index1 - 1]
                    ):
                        vertex_occupation[refined_vertex_index] = vertex_index1
                        found_v1_occupancy = True
                        if edges[edge_index] > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index] + 1
                        continue

                # Case 3: Check occupancy by V2 (pre-check)
                found_v2_occupancy = False
                if (
                    not found_v1_occupancy
                    and vertex_index2 != 0
                    and vertex_index2 <= num_atoms
                ):
                    distance_vector = (
                        atoms_data[vertex_index2 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
                        - contact_point
                    )
                    distance_squared = dot_product(distance_vector, distance_vector)
                    if (
                        distance_squared
                        < atom_plus_probe_radii_shrink_1d[vertex_index2 - 1]
                    ):
                        vertex_occupation[refined_vertex_index] = vertex_index2
                        found_v2_occupancy = True
                        if edges[edge_index] > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index]
                        continue

                # Case 4: Check general occupation via voxel grid
                if not found_v1_occupancy and not found_v2_occupancy:

                    # Use helper 2: Get voxel limits
                    lower_limit, upper_limit = _get_voxel_limits(
                        contact_point,
                        voxel_space_origin,
                        voxel_space_scale,
                        voxel_atom_start_indices,
                        voxel_atom_end_indices,
                    )

                    # Use helper 3: Check occupancy
                    is_occupied = _check_occupancy_by_voxel(
                        contact_point,
                        lower_limit,
                        upper_limit,
                        voxel_atom_indices,
                        num_atoms,
                        atoms_data,
                        atom_plus_probe_radii_shrink_1d,
                        vertex_occupation,
                        refined_vertex_index,
                    )

                    if is_occupied:
                        if edges[edge_index] > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index] + 1
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index]
                        continue

                # Case 5: Exposed
                num_visible_vertices += 1
                num_exposed_grids += 1

                # Manual resize logic for serial function
                if num_exposed_grids >= exposed_grid_coordinates.shape[0]:
                    new_exposed_grid_count = int(
                        exposed_grid_coordinates.shape[0] * RESIZE_FACTOR
                    )
                    exposed_grid_coordinates = np.resize(
                        exposed_grid_coordinates, (new_exposed_grid_count + 1, 3)
                    )

                exposed_grid_coordinates[num_exposed_grids, :] = contact_point
                vertex_occupation[refined_vertex_index] = 0

                if edges[edge_index] > 0:
                    # Logic for adding new edges to state array for further refinement
                    if edges[edges[edge_index] + 1] > 0 or vertex_index2 > 0:
                        edge_state_count += 1
                        edge_state[edge_state_count] = edges[edge_index] + 1

                    if edges[edges[edge_index]] > 0 or vertex_index1 > 0:
                        edge_state_count += 1
                        edge_state[edge_state_count] = edges[edge_index]
            refinement_level_curr += 1

            # --- Pair Finalization ---
        if num_visible_vertices > 0:
            num_exposed_atom_pairs += 1
            atom_contact_state[atom_index1] = 0
            if atom_index2 <= num_atoms:
                atom_contact_state[atom_index2] = 0
        nprint_cpu(
            TRACE,
            _VERBOSITY,
            f"[PAIR {pair_index}] DONE: total_exposed={num_exposed_grids}, "
            f"final_visible={num_visible_vertices}",
        )

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


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _pass1_calculate_counts(
    num_atoms,
    atoms_data,
    atom_plus_probe_radii_1d,
    atom_plus_probe_radii_shrink_1d,
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
    num_threads,
    real_type,
    int_type,
):
    """
    Pass 1: Tracks count of exposed grids per thread AND exposed pairs per thread.
    Uses new inline helpers for clean, parallel-friendly loops.
    """
    num_pairs = atom_pairs.shape[0]
    pairs_per_thread = (num_pairs + num_threads - 1) // num_threads

    per_thread_pair_count = np.zeros(num_threads, dtype=int_type)
    per_thread_coords_count = np.zeros(num_threads, dtype=int_type)

    for tid in prange(num_threads):
        p_start = tid * pairs_per_thread
        p_end = min(p_start + pairs_per_thread, num_pairs)

        local_coords_count = 0
        local_pair_count = 0

        # thread-local scratch arrays
        vertex_occupation = np.zeros(num_vertices + 1, dtype=int_type)
        edge_state = np.zeros(num_edges + 1, dtype=int_type)
        rotation_matrix = np.zeros((3, 3), dtype=real_type)
        contact_point = np.zeros(3, dtype=real_type)

        midpoint_coords = np.zeros(3, dtype=real_type)
        distance_vector = np.zeros(3, dtype=real_type)

        for pair_index in range(p_start, p_end):
            for i in range(num_vertices + 1):
                vertex_occupation[i] = 0

            for i in range(num_edges + 1):
                edge_state[i] = 0

            # --- START: Serial Logic for a Single Pair ---
            atom_index1 = atom_pairs[pair_index][0]
            atom_index2 = atom_pairs[pair_index][1]
            atom_coords1 = atoms_data[atom_index1 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
            atom_coords2 = atoms_data[atom_index2 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
            r1 = atom_plus_probe_radii_1d[atom_index1 - 1]
            r2 = atom_plus_probe_radii_1d[atom_index2 - 1]

            interaction_radius, distance_magnitude = _compute_midpoint_and_radius(
                atom_coords1,
                atom_coords2,
                r1,
                r2,
                midpoint_coords,
                distance_vector,
            )

            dx1 = distance_vector[0]
            dx2 = distance_vector[1]
            xy_proj = real_type(math.sqrt(dx1 * dx1 + dx2 * dx2))

            # Use helper to compute rotation matrix
            _ = _compute_rotation_matrix(
                distance_vector,
                distance_magnitude,
                xy_proj,
                rotation_matrix,
            )

            num_visible_vertices = 0

            # A. Initial Vertex Pass (Counting Grids)
            vertex_index = 1
            while vertex_index <= initial_vertex_count:
                vertex_coords = vertex_array[vertex_index]

                # Use helper 1: Calculate contact point
                _calculate_contact_point_coords(
                    midpoint_coords,
                    interaction_radius,
                    rotation_matrix,
                    vertex_coords,
                    contact_point,
                )

                # Use helper 2: Get voxel limits
                lower_limit, upper_limit = _get_voxel_limits(
                    contact_point,
                    voxel_space_origin,
                    voxel_space_scale,
                    voxel_atom_start_indices,
                    voxel_atom_end_indices,
                )

                # Use helper 3: Check occupancy
                is_occupied = _check_occupancy_by_voxel(
                    contact_point,
                    lower_limit,
                    upper_limit,
                    voxel_atom_indices,
                    num_atoms,
                    atoms_data,
                    atom_plus_probe_radii_shrink_1d,
                    vertex_occupation,
                    vertex_index,
                )

                if not is_occupied:
                    num_visible_vertices += 1
                    local_coords_count += 1  # COUNT INCREMENT

                vertex_index += 1

            # B. Edge State Setup
            edge_state_count = 0
            if refinement_level > 0:
                vertex_occupation[0] = 0  # Ensure 0 is not treated as an atom index

                for edge_index in range(initial_vertex_count, 0, -1):
                    v1 = vertex_occupation[edge_array[edge_index][1]]
                    v2 = vertex_occupation[edge_array[edge_index][2]]

                    if v1 > 0 and v1 == v2:
                        continue
                    edge_state_count += 1
                    edge_state[edge_state_count] = edge_index

            # C. Refinement Loop (Counting Grids - Refactored)
            if edge_state_count > 0:
                while edge_state_count > 0:  # Use stack size for loop control
                    edge_index = edge_state[edge_state_count]
                    edge_state_count -= 1
                    vertex_index1 = vertex_occupation[edge_array[edge_index][1]]
                    vertex_index2 = vertex_occupation[edge_array[edge_index][2]]

                    # Case 1: External/Boundary Check
                    if vertex_index1 > num_atoms or vertex_index2 > num_atoms:
                        continue  # Go to the next edge on the stack

                    refined_vertex_index = edge_index + initial_vertex_count
                    vertex_coords = vertex_array[refined_vertex_index]

                    # Use helper 1: Calculate contact point
                    _calculate_contact_point_coords(
                        midpoint_coords,
                        interaction_radius,
                        rotation_matrix,
                        vertex_coords,
                        contact_point,
                    )

                    # Case 2: Check occupancy by V1 (pre-check, only needed for state update)
                    found_v1_occupancy = False
                    if vertex_index1 != 0 and vertex_index1 <= num_atoms:
                        distance_vector = (
                            atoms_data[vertex_index1 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
                            - contact_point
                        )
                        distance_squared = dot_product(distance_vector, distance_vector)
                        if (
                            distance_squared
                            < atom_plus_probe_radii_shrink_1d[vertex_index1 - 1]
                        ):
                            vertex_occupation[refined_vertex_index] = vertex_index1
                            found_v1_occupancy = True
                            if edges[edge_index] > 0:
                                edge_state_count += 1
                                edge_state[edge_state_count] = edges[edge_index] + 1
                            continue  # Go to the next edge on the stack

                    # Case 3: Check occupancy by V2 (pre-check, only needed for state update)
                    found_v2_occupancy = False
                    if (
                        not found_v1_occupancy
                        and vertex_index2 != 0
                        and vertex_index2 <= num_atoms
                    ):
                        distance_vector = (
                            atoms_data[vertex_index2 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
                            - contact_point
                        )
                        distance_squared = dot_product(distance_vector, distance_vector)
                        if (
                            distance_squared
                            < atom_plus_probe_radii_shrink_1d[vertex_index2 - 1]
                        ):
                            vertex_occupation[refined_vertex_index] = vertex_index2
                            found_v2_occupancy = True
                            if edges[edge_index] > 0:
                                edge_state_count += 1
                                edge_state[edge_state_count] = edges[edge_index]
                            continue  # Go to the next edge on the stack

                    # Case 4: Check general occupation via voxel grid
                    if not found_v1_occupancy and not found_v2_occupancy:

                        # Use helper 2: Get voxel limits
                        lower_limit, upper_limit = _get_voxel_limits(
                            contact_point,
                            voxel_space_origin,
                            voxel_space_scale,
                            voxel_atom_start_indices,
                            voxel_atom_end_indices,
                        )

                        # Use helper 3: Check occupancy
                        is_occupied = _check_occupancy_by_voxel(
                            contact_point,
                            lower_limit,
                            upper_limit,
                            voxel_atom_indices,
                            num_atoms,
                            atoms_data,
                            atom_plus_probe_radii_shrink_1d,
                            vertex_occupation,
                            refined_vertex_index,
                        )

                        if is_occupied:
                            if edges[edge_index] > 0:
                                edge_state_count += 1
                                edge_state[edge_state_count] = edges[edge_index] + 1
                                edge_state_count += 1
                                edge_state[edge_state_count] = edges[edge_index]
                            continue  # Go to the next edge on the stack

                    # Case 5: Exposed
                    num_visible_vertices += 1
                    local_coords_count += 1  # COUNT INCREMENT

                    vertex_occupation[refined_vertex_index] = 0

                    if edges[edge_index] > 0:
                        if edges[edges[edge_index] + 1] > 0 or vertex_index2 > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index] + 1

                        if edges[edges[edge_index]] > 0 or vertex_index1 > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index]

            # Loop naturally ends when edge_state_count <= 0

            # D. Pair Finalization
            if num_visible_vertices > 0:
                local_pair_count += 1

        # --- END: Serial Logic for a Single Pair ---

        per_thread_coords_count[tid] = local_coords_count
        per_thread_pair_count[tid] = local_pair_count

    return per_thread_coords_count, per_thread_pair_count


# ====================================================================
# --- PASS 2: Coordinate Generation and Indexed Global Write ---
# ====================================================================
@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _pass2_generate_coords(
    num_atoms,
    atoms_data,
    atom_plus_probe_radii_1d,
    atom_plus_probe_radii_shrink_1d,
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
    global_coords_offsets,
    global_pair_indices,  # Pre-allocated array to store exposed pair indices
    global_pair_offsets,
    exposed_grid_coordinates,  # Global output buffer (N x 3) - 0-based
    num_threads,
    real_type,
    int_type,
):
    """
    Pass 2: Executes exposed grid logic and writes coordinates directly to the global array,
    leveraging helper functions for geometry and occupancy checks.
    """
    num_pairs = atom_pairs.shape[0]
    pairs_per_thread = (num_pairs + num_threads - 1) // num_threads

    for tid in prange(num_threads):
        p_start = tid * pairs_per_thread
        p_end = min(p_start + pairs_per_thread, num_pairs)

        # Thread-local scratchpads
        global_coords_write_offset = global_coords_offsets[tid]
        global_pair_write_offset = global_pair_offsets[tid]

        local_coords_write_idx = 0
        local_pair_write_idx = 0

        # Note: arrays are 1-based indexed in logic
        vertex_occupation = np.zeros(num_vertices + 1, dtype=int_type)
        edge_state = np.zeros(num_edges + 1, dtype=int_type)
        rotation_matrix = np.zeros((3, 3), dtype=real_type)
        contact_point = np.zeros(3, dtype=real_type)

        midpoint_coords = np.zeros(3, dtype=real_type)
        distance_vector = np.zeros(3, dtype=real_type)

        for pair_index in range(p_start, p_end):
            for i in range(num_vertices + 1):
                vertex_occupation[i] = 0

            for i in range(num_edges + 1):
                edge_state[i] = 0

            # --- GEOMETRY SETUP (REFACTORED) ---
            atom_index1 = atom_pairs[pair_index][0]
            atom_index2 = atom_pairs[pair_index][1]
            atom_coords1 = atoms_data[atom_index1 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
            atom_coords2 = atoms_data[atom_index2 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
            r1 = atom_plus_probe_radii_1d[atom_index1 - 1]
            r2 = atom_plus_probe_radii_1d[atom_index2 - 1]

            interaction_radius, distance_magnitude = _compute_midpoint_and_radius(
                atom_coords1,
                atom_coords2,
                r1,
                r2,
                midpoint_coords,
                distance_vector,
            )

            dx1 = distance_vector[0]
            dx2 = distance_vector[1]
            xy_proj = real_type(math.sqrt(dx1 * dx1 + dx2 * dx2))

            # Use helper to compute rotation matrix (updates rotation_matrix in-place)
            _ = _compute_rotation_matrix(
                distance_vector,
                distance_magnitude,
                xy_proj,
                rotation_matrix,
            )
            # --- END GEOMETRY SETUP ---

            num_visible_vertices = 0

            # A. Initial Vertex Pass (REFACTORED)
            vertex_index = 1
            while vertex_index <= initial_vertex_count:
                vertex_coords = vertex_array[vertex_index]

                # Use helper 1: Calculate contact point
                _calculate_contact_point_coords(
                    midpoint_coords,
                    interaction_radius,
                    rotation_matrix,
                    vertex_coords,
                    contact_point,
                )

                # Use helper 2: Get voxel limits
                lower_limit, upper_limit = _get_voxel_limits(
                    contact_point,
                    voxel_space_origin,
                    voxel_space_scale,
                    voxel_atom_start_indices,
                    voxel_atom_end_indices,
                )

                # Use helper 3: Check occupancy (updates vertex_occupation[vertex_index])
                is_occupied = _check_occupancy_by_voxel(
                    contact_point,
                    lower_limit,
                    upper_limit,
                    voxel_atom_indices,
                    num_atoms,
                    atoms_data,
                    atom_plus_probe_radii_shrink_1d,
                    vertex_occupation,
                    vertex_index,
                )

                if not is_occupied:
                    num_visible_vertices += 1
                    # WRITE TO GLOBAL BUFFER (elementwise)
                    write_idx = global_coords_write_offset + local_coords_write_idx
                    exposed_grid_coordinates[write_idx, 0] = contact_point[0]
                    exposed_grid_coordinates[write_idx, 1] = contact_point[1]
                    exposed_grid_coordinates[write_idx, 2] = contact_point[2]
                    local_coords_write_idx += 1

                vertex_index += 1

            # B. Edge State Setup (Identical to Pass 1)
            edge_state_count = 0
            if refinement_level > 0:
                vertex_occupation[0] = 0

                for edge_index in range(initial_vertex_count, 0, -1):
                    v1 = vertex_occupation[edge_array[edge_index][1]]
                    v2 = vertex_occupation[edge_array[edge_index][2]]

                    if v1 > 0 and v1 == v2:
                        continue
                    edge_state_count += 1
                    edge_state[edge_state_count] = edge_index

            # C. Refinement Loop (REFACTORED)
            if edge_state_count > 0:
                while edge_state_count > 0:
                    edge_index = edge_state[edge_state_count]
                    edge_state_count -= 1
                    vertex_index1 = vertex_occupation[edge_array[edge_index][1]]
                    vertex_index2 = vertex_occupation[edge_array[edge_index][2]]
                    refined_vertex_index = edge_index + initial_vertex_count

                    # Case 1: External/Boundary Check
                    if vertex_index1 > num_atoms or vertex_index2 > num_atoms:
                        continue

                        # Use helper 1: Calculate contact point
                    vertex_coords = vertex_array[refined_vertex_index]
                    _calculate_contact_point_coords(
                        midpoint_coords,
                        interaction_radius,
                        rotation_matrix,
                        vertex_coords,
                        contact_point,
                    )

                    is_occupied = False

                    # Case 2: Check occupancy by V1 (pre-check)
                    if vertex_index1 != 0 and vertex_index1 <= num_atoms:
                        distance_vector_check = (
                            atoms_data[vertex_index1 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
                            - contact_point
                        )
                        distance_squared = dot_product(
                            distance_vector_check, distance_vector_check
                        )
                        if (
                            distance_squared
                            < atom_plus_probe_radii_shrink_1d[vertex_index1 - 1]
                        ):
                            vertex_occupation[refined_vertex_index] = vertex_index1
                            is_occupied = True
                            if edges[edge_index] > 0:
                                edge_state_count += 1
                                edge_state[edge_state_count] = edges[edge_index] + 1
                            continue  # Occupied, move to next edge

                    # Case 3: Check occupancy by V2 (pre-check)
                    if (
                        not is_occupied
                        and vertex_index2 != 0
                        and vertex_index2 <= num_atoms
                    ):
                        distance_vector_check = (
                            atoms_data[vertex_index2 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
                            - contact_point
                        )
                        distance_squared = dot_product(
                            distance_vector_check, distance_vector_check
                        )
                        if (
                            distance_squared
                            < atom_plus_probe_radii_shrink_1d[vertex_index2 - 1]
                        ):
                            vertex_occupation[refined_vertex_index] = vertex_index2
                            is_occupied = True
                            if edges[edge_index] > 0:
                                edge_state_count += 1
                                edge_state[edge_state_count] = edges[edge_index]
                            continue  # Occupied, move to next edge

                    # Case 4: Check general occupation via voxel grid (REFACTORED)
                    if not is_occupied:

                        # Use helper 2: Get voxel limits
                        lower_limit, upper_limit = _get_voxel_limits(
                            contact_point,
                            voxel_space_origin,
                            voxel_space_scale,
                            voxel_atom_start_indices,
                            voxel_atom_end_indices,
                        )

                        # Use helper 3: Check occupancy (updates vertex_occupation[refined_vertex_index])
                        is_occupied = _check_occupancy_by_voxel(
                            contact_point,
                            lower_limit,
                            upper_limit,
                            voxel_atom_indices,
                            num_atoms,
                            atoms_data,
                            atom_plus_probe_radii_shrink_1d,
                            vertex_occupation,
                            refined_vertex_index,
                        )

                        if is_occupied:
                            if edges[edge_index] > 0:
                                # Both new edges require further checking
                                edge_state_count += 1
                                edge_state[edge_state_count] = edges[edge_index] + 1
                                edge_state_count += 1
                                edge_state[edge_state_count] = edges[edge_index]
                            continue  # Occupied, move to next edge

                    # Case 5: Exposed (not is_occupied)
                    if not is_occupied:
                        num_visible_vertices += 1

                        # WRITE TO GLOBAL BUFFER (elementwise)
                        write_idx = global_coords_write_offset + local_coords_write_idx
                        exposed_grid_coordinates[write_idx, 0] = contact_point[0]
                        exposed_grid_coordinates[write_idx, 1] = contact_point[1]
                        exposed_grid_coordinates[write_idx, 2] = contact_point[2]
                        local_coords_write_idx += 1

                        # Note: vertex_occupation[refined_vertex_index] is already 0 from helper

                        if edges[edge_index] > 0:
                            # Logic for adding new edges to state array for further refinement
                            if edges[edges[edge_index] + 1] > 0 or vertex_index2 > 0:
                                edge_state_count += 1
                                edge_state[edge_state_count] = edges[edge_index] + 1

                            if edges[edges[edge_index]] > 0 or vertex_index1 > 0:
                                edge_state_count += 1
                                edge_state[edge_state_count] = edges[edge_index]

            # Final state update for the pair
            if num_visible_vertices > 0:
                # WRITE EXPOSED PAIR INDEX (elementwise)
                global_pair_indices[global_pair_write_offset + local_pair_write_idx] = (
                    pair_index
                )
                local_pair_write_idx += 1

    # in-place writes; no return required


# ====================================================================
# --- Two Pass Orchestrator (Signature Match) ---
# ====================================================================
@njit(nogil=True, boundscheck=False, cache=True)
def _calculate_exposed_grids_two_pass(
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
    num_extended_cube_grid_points,  # Unused in two-pass but kept for signature match
    use_cuda,  # Unused but kept for signature match
    num_threads,
    real_type,
    int_type,
):
    """
    Two-Pass Parallel Orchestrator:
    1. Count required buffers.
    2. Allocate exact buffers.
    3. Fill buffers and collect exposed pair indices (for sequential state update).
    4. Perform sequential state update and return.
    """
    # 1. PASS 1: Calculate Counts
    per_thread_coords_count, per_thread_pair_count = _pass1_calculate_counts(
        num_atoms,
        atoms_data,
        atom_plus_probe_radii_1d,
        atom_plus_probe_radii_shrink_1d,
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
        num_threads,
        real_type,
        int_type,
    )

    # 2. Buffer Allocation and Offset Calculation (Sequential)
    total_exposed_grids = np.sum(per_thread_coords_count)
    num_exposed_atom_pairs = np.sum(per_thread_pair_count)

    # Global Buffer Allocation (1-based indexing for coordinates)
    if total_exposed_grids == 0:
        exposed_grid_coordinates = np.zeros((1, 3), dtype=real_type)
    else:
        exposed_grid_coordinates = np.zeros(
            (total_exposed_grids + 1, 3), dtype=real_type
        )

    # Exposed Pair Index Buffer (Used for final atomic state update)
    # The pair indices are 0-based in atom_pairs, but we track the index itself
    global_pair_indices = np.zeros(num_exposed_atom_pairs, dtype=int_type)

    # Calculate global start offsets for each thread's write:
    # Coords: Start at 1 (due to 1-based indexing), Pair Indices: Start at 0
    global_coords_offsets = np.zeros(num_threads + 1, dtype=int_type)
    global_pair_offsets = np.zeros(num_threads + 1, dtype=int_type)

    current_coords_offset = 1  # Coords start at index 1
    current_pair_offset = 0  # Pairs start at index 0

    for tid in range(num_threads):
        global_coords_offsets[tid] = current_coords_offset
        global_pair_offsets[tid] = current_pair_offset

        current_coords_offset += per_thread_coords_count[tid]
        current_pair_offset += per_thread_pair_count[tid]

    # Sanity check: Should match total counts
    # assert current_coords_offset == total_exposed_grids + 1
    # assert current_pair_offset == num_exposed_atom_pairs

    # 3. PASS 2: Generate Coordinates and Collect Pair Indices
    _pass2_generate_coords(
        num_atoms,
        atoms_data,
        atom_plus_probe_radii_1d,
        atom_plus_probe_radii_shrink_1d,
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
        global_coords_offsets,
        global_pair_indices,
        global_pair_offsets,
        exposed_grid_coordinates,
        num_threads,
        real_type,
        int_type,
    )

    # 4. Sequential Aggregation of Atom Contact State (Identical to Single-Pass Final Step)
    atom_contact_state_out = np.copy(atom_contact_state)

    # Iterate over the collected unique exposed pair indices
    for i in range(num_exposed_atom_pairs):
        pair_index = global_pair_indices[i]
        atom_index1 = atom_pairs[pair_index][0]
        atom_index2 = atom_pairs[pair_index][1]

        # The *fact* that it was exposed is sufficient to mark state 0
        atom_contact_state_out[atom_index1] = 0
        if atom_index2 <= num_atoms:
            atom_contact_state_out[atom_index2] = 0

    num_accessible_atoms = np.sum(atom_contact_state_out[1 : num_atoms + 1] == 0)

    # Return matches the single-pass signature
    return (
        total_exposed_grids,
        exposed_grid_coordinates,
        num_accessible_atoms,
        atom_contact_state_out,
        num_exposed_atom_pairs,
    )


# ====================================================================
# --- CUDA PASS 1 KERNEL: Calculate Counts ---
# ====================================================================
@cuda.jit(cache=True)
def _cuda_pass1_calculate_counts_kernel(
    num_atoms,
    atoms_data,
    atom_plus_probe_radii_1d,
    atom_plus_probe_radii_shrink_1d,
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
    num_threads,
    d_per_thread_coords_count,  # Output buffer
    d_per_thread_pair_count,  # Output buffer
):
    """Pass 1: Counts exposed grids and exposed pairs per thread block."""
    tid = cuda.grid(1)  # Thread index (0 to num_threads - 1)

    num_pairs = atom_pairs.shape[0]
    pairs_per_thread = (num_pairs + num_threads - 1) // num_threads

    p_start = tid * pairs_per_thread
    p_end = min(p_start + pairs_per_thread, num_pairs)

    if p_start >= p_end:
        d_per_thread_coords_count[tid] = 0
        d_per_thread_pair_count[tid] = 0
        return

    local_coords_count = 0
    local_pair_count = 0

    # Thread-local scratch arrays (automatically allocated in local/private memory)
    # SAS_MAX_VERTICES + 1: 1041, literal int will be used allow cuda compilation
    # SAS_MAX_EDGES + 1: 2081, literal int will be used allow cuda compilation
    vertex_occupation = cuda.local.array(1041, int64)
    edge_state = cuda.local.array(2081, int64)
    rotation_matrix = cuda.local.array((3, 3), float64)
    contact_point = cuda.local.array(3, float64)

    midpoint_coords = cuda.local.array(3, dtype=float64)
    distance_vector = cuda.local.array(3, dtype=float64)

    for pair_index in range(p_start, p_end):
        # Reset scratchpads (initialization)
        for i in range(num_vertices + 1):
            vertex_occupation[i] = 0
        for i in range(num_edges + 1):
            edge_state[i] = 0

        # --- GEOMETRY SETUP ---
        atom_index1 = atom_pairs[pair_index, 0]
        atom_index2 = atom_pairs[pair_index, 1]
        atom_coords1 = atoms_data[atom_index1 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
        atom_coords2 = atoms_data[atom_index2 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
        r1 = atom_plus_probe_radii_1d[atom_index1 - 1]
        r2 = atom_plus_probe_radii_1d[atom_index2 - 1]

        interaction_radius, distance_magnitude = _compute_midpoint_and_radius(
            atom_coords1,
            atom_coords2,
            r1,
            r2,
            midpoint_coords,
            distance_vector,
        )

        dx1 = distance_vector[0]
        dx2 = distance_vector[1]
        xy_proj = float64(math.sqrt(dx1 * dx1 + dx2 * dx2))

        _ = _compute_rotation_matrix(
            distance_vector,
            distance_magnitude,
            xy_proj,
            rotation_matrix,
        )
        # --- END GEOMETRY SETUP ---

        num_visible_vertices = 0

        # A. Initial Vertex Pass
        vertex_index = 1
        while vertex_index <= initial_vertex_count:
            vertex_coords = vertex_array[vertex_index]

            _calculate_contact_point_coords(
                midpoint_coords,
                interaction_radius,
                rotation_matrix,
                vertex_coords,
                contact_point,
            )

            lower_limit, upper_limit = _get_voxel_limits(
                contact_point,
                voxel_space_origin,
                voxel_space_scale,
                voxel_atom_start_indices,
                voxel_atom_end_indices,
            )

            is_occupied = _check_occupancy_by_voxel(
                contact_point,
                lower_limit,
                upper_limit,
                voxel_atom_indices,
                num_atoms,
                atoms_data,
                atom_plus_probe_radii_shrink_1d,
                vertex_occupation,
                vertex_index,
            )

            if not is_occupied:
                num_visible_vertices += 1
                local_coords_count += 1

            vertex_index += 1

        # B. Edge State Setup
        edge_state_count = 0
        if refinement_level > 0:
            vertex_occupation[0] = 0

            for edge_index in range(initial_vertex_count, 0, -1):
                v1 = vertex_occupation[edge_array[edge_index, 1]]
                v2 = vertex_occupation[edge_array[edge_index, 2]]

                if v1 > 0 and v1 == v2:
                    continue
                edge_state_count += 1
                edge_state[edge_state_count] = edge_index

        # C. Refinement Loop
        if edge_state_count > 0:
            while edge_state_count > 0:
                edge_index = edge_state[edge_state_count]
                edge_state_count -= 1
                vertex_index1 = vertex_occupation[edge_array[edge_index, 1]]
                vertex_index2 = vertex_occupation[edge_array[edge_index, 2]]
                refined_vertex_index = edge_index + initial_vertex_count

                if vertex_index1 > num_atoms or vertex_index2 > num_atoms:
                    continue

                vertex_coords = vertex_array[refined_vertex_index]
                _calculate_contact_point_coords(
                    midpoint_coords,
                    interaction_radius,
                    rotation_matrix,
                    vertex_coords,
                    contact_point,
                )

                is_occupied = False

                # Check V1
                if vertex_index1 != 0 and vertex_index1 <= num_atoms:
                    dx1 = atoms_data[vertex_index1 - 1][ATOMFIELD_X] - contact_point[0]
                    dy1 = atoms_data[vertex_index1 - 1][ATOMFIELD_Y] - contact_point[1]
                    dz1 = atoms_data[vertex_index1 - 1][ATOMFIELD_Z] - contact_point[2]

                    distance_squared = dx1 * dx1 + dy1 * dy1 + dz1 * dz1
                    if (
                        distance_squared
                        < atom_plus_probe_radii_shrink_1d[vertex_index1 - 1]
                    ):
                        vertex_occupation[refined_vertex_index] = vertex_index1
                        is_occupied = True
                        if edges[edge_index] > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index] + 1
                        continue

                # Check V2
                if (
                    not is_occupied
                    and vertex_index2 != 0
                    and vertex_index2 <= num_atoms
                ):
                    dx2 = atoms_data[vertex_index2 - 1][ATOMFIELD_X] - contact_point[0]
                    dy2 = atoms_data[vertex_index2 - 1][ATOMFIELD_Y] - contact_point[1]
                    dz2 = atoms_data[vertex_index2 - 1][ATOMFIELD_Z] - contact_point[2]

                    distance_squared = dx2 * dx2 + dy2 * dy2 + dz2 * dz2

                    if (
                        distance_squared
                        < atom_plus_probe_radii_shrink_1d[vertex_index2 - 1]
                    ):
                        vertex_occupation[refined_vertex_index] = vertex_index2
                        is_occupied = True
                        if edges[edge_index] > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index]
                        continue

                # Check Voxel Grid
                if not is_occupied:
                    lower_limit, upper_limit = _get_voxel_limits(
                        contact_point,
                        voxel_space_origin,
                        voxel_space_scale,
                        voxel_atom_start_indices,
                        voxel_atom_end_indices,
                    )
                    is_occupied = _check_occupancy_by_voxel(
                        contact_point,
                        lower_limit,
                        upper_limit,
                        voxel_atom_indices,
                        num_atoms,
                        atoms_data,
                        atom_plus_probe_radii_shrink_1d,
                        vertex_occupation,
                        refined_vertex_index,
                    )

                    if is_occupied:
                        if edges[edge_index] > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index] + 1
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index]
                        continue

                # Exposed
                if not is_occupied:
                    num_visible_vertices += 1
                    local_coords_count += 1

                    # vertex_occupation is already 0 from helper

                    if edges[edge_index] > 0:
                        if edges[edges[edge_index] + 1] > 0 or vertex_index2 > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index] + 1

                        if edges[edges[edge_index]] > 0 or vertex_index1 > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index]

        # Final state update for the pair
        if num_visible_vertices > 0:
            local_pair_count += 1

    # Write results to the global thread output arrays
    d_per_thread_coords_count[tid] = local_coords_count
    d_per_thread_pair_count[tid] = local_pair_count


# ====================================================================
# --- CUDA PASS 2 KERNEL: Coordinate Generation and Indexed Global Write ---
# ====================================================================
@cuda.jit(cache=True)
def _cuda_pass2_generate_coords_kernel(
    num_atoms,
    atoms_data,
    atom_plus_probe_radii_1d,
    atom_plus_probe_radii_shrink_1d,
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
    num_threads,
    d_global_coords_offsets,
    d_global_pair_indices,  # Output
    d_global_pair_offsets,
    d_exposed_grid_coordinates,  # Output
):
    """Pass 2: Executes exposed grid logic and writes coordinates directly to global device memory."""
    tid = cuda.grid(1)

    num_pairs = atom_pairs.shape[0]
    pairs_per_thread = (num_pairs + num_threads - 1) // num_threads

    p_start = tid * pairs_per_thread
    p_end = min(p_start + pairs_per_thread, num_pairs)

    if p_start >= p_end:
        return

    # Thread-local offsets (read from device global memory)
    global_coords_write_offset = d_global_coords_offsets[tid]
    global_pair_write_offset = d_global_pair_offsets[tid]

    local_coords_write_idx = 0
    local_pair_write_idx = 0

    # Thread-local scratch arrays
    # SAS_MAX_VERTICES + 1: 1041, literal int will be used to allow cuda compilation
    # SAS_MAX_EDGES + 1: 2081, literal int will be used to allow cuda compilation
    vertex_occupation = cuda.local.array(1041, int64)
    edge_state = cuda.local.array(2081, int64)
    rotation_matrix = cuda.local.array((3, 3), float64)
    contact_point = cuda.local.array(3, float64)

    midpoint_coords = cuda.local.array(3, float64)
    distance_vector = cuda.local.array(3, float64)

    for pair_index in range(p_start, p_end):
        # Reset scratchpads (initialization)
        for i in range(num_vertices + 1):
            vertex_occupation[i] = 0
        for i in range(num_edges + 1):
            edge_state[i] = 0

        # --- GEOMETRY SETUP (Identical to Pass 1) ---
        atom_index1 = atom_pairs[pair_index, 0]
        atom_index2 = atom_pairs[pair_index, 1]
        atom_coords1 = atoms_data[atom_index1 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
        atom_coords2 = atoms_data[atom_index2 - 1][ATOMFIELD_X:ATOMFIELD_CRD_END]
        r1 = atom_plus_probe_radii_1d[atom_index1 - 1]
        r2 = atom_plus_probe_radii_1d[atom_index2 - 1]

        interaction_radius, distance_magnitude = _compute_midpoint_and_radius(
            atom_coords1,
            atom_coords2,
            r1,
            r2,
            midpoint_coords,
            distance_vector,
        )

        dx1 = distance_vector[0]
        dx2 = distance_vector[1]
        xy_proj = float64(math.sqrt(dx1 * dx1 + dx2 * dx2))

        _ = _compute_rotation_matrix(
            distance_vector,
            distance_magnitude,
            xy_proj,
            rotation_matrix,
        )
        # --- END GEOMETRY SETUP ---

        num_visible_vertices = 0

        # A. Initial Vertex Pass (Coordinate WRITE)
        vertex_index = 1
        while vertex_index <= initial_vertex_count:
            vertex_coords = vertex_array[vertex_index]

            _calculate_contact_point_coords(
                midpoint_coords,
                interaction_radius,
                rotation_matrix,
                vertex_coords,
                contact_point,
            )

            lower_limit, upper_limit = _get_voxel_limits(
                contact_point,
                voxel_space_origin,
                voxel_space_scale,
                voxel_atom_start_indices,
                voxel_atom_end_indices,
            )

            is_occupied = _check_occupancy_by_voxel(
                contact_point,
                lower_limit,
                upper_limit,
                voxel_atom_indices,
                num_atoms,
                atoms_data,
                atom_plus_probe_radii_shrink_1d,
                vertex_occupation,
                vertex_index,
            )

            if not is_occupied:
                num_visible_vertices += 1
                # WRITE TO GLOBAL DEVICE BUFFER
                write_idx = global_coords_write_offset + local_coords_write_idx
                if write_idx >= d_global_coords_offsets[tid + 1]:
                    return

                d_exposed_grid_coordinates[write_idx, 0] = contact_point[0]
                d_exposed_grid_coordinates[write_idx, 1] = contact_point[1]
                d_exposed_grid_coordinates[write_idx, 2] = contact_point[2]
                local_coords_write_idx += 1

            vertex_index += 1

        # B. Edge State Setup
        edge_state_count = 0
        if refinement_level > 0:
            vertex_occupation[0] = 0

            for edge_index in range(initial_vertex_count, 0, -1):
                v1 = vertex_occupation[edge_array[edge_index, 1]]
                v2 = vertex_occupation[edge_array[edge_index, 2]]

                if v1 > 0 and v1 == v2:
                    continue
                edge_state_count += 1
                edge_state[edge_state_count] = edge_index

        # C. Refinement Loop (Coordinate WRITE)
        if edge_state_count > 0:
            while edge_state_count > 0:
                edge_index = edge_state[edge_state_count]
                edge_state_count -= 1
                vertex_index1 = vertex_occupation[edge_array[edge_index, 1]]
                vertex_index2 = vertex_occupation[edge_array[edge_index, 2]]
                refined_vertex_index = edge_index + initial_vertex_count

                if vertex_index1 > num_atoms or vertex_index2 > num_atoms:
                    continue

                vertex_coords = vertex_array[refined_vertex_index]
                _calculate_contact_point_coords(
                    midpoint_coords,
                    interaction_radius,
                    rotation_matrix,
                    vertex_coords,
                    contact_point,
                )

                is_occupied = False

                # Check V1
                if vertex_index1 != 0 and vertex_index1 <= num_atoms:
                    dx1 = atoms_data[vertex_index1 - 1][ATOMFIELD_X] - contact_point[0]
                    dy1 = atoms_data[vertex_index1 - 1][ATOMFIELD_Y] - contact_point[1]
                    dz1 = atoms_data[vertex_index1 - 1][ATOMFIELD_Z] - contact_point[2]

                    distance_squared = dx1 * dx1 + dy1 * dy1 + dz1 * dz1
                    if (
                        distance_squared
                        < atom_plus_probe_radii_shrink_1d[vertex_index1 - 1]
                    ):
                        vertex_occupation[refined_vertex_index] = vertex_index1
                        is_occupied = True
                        if edges[edge_index] > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index] + 1
                        continue

                # Check V2
                if (
                    not is_occupied
                    and vertex_index2 != 0
                    and vertex_index2 <= num_atoms
                ):
                    dx2 = atoms_data[vertex_index2 - 1][ATOMFIELD_X] - contact_point[0]
                    dy2 = atoms_data[vertex_index2 - 1][ATOMFIELD_Y] - contact_point[1]
                    dz2 = atoms_data[vertex_index2 - 1][ATOMFIELD_Z] - contact_point[2]

                    distance_squared = dx2 * dx2 + dy2 * dy2 + dz2 * dz2
                    if (
                        distance_squared
                        < atom_plus_probe_radii_shrink_1d[vertex_index2 - 1]
                    ):
                        vertex_occupation[refined_vertex_index] = vertex_index2
                        is_occupied = True
                        if edges[edge_index] > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index]
                        continue

                # Check Voxel Grid
                if not is_occupied:
                    lower_limit, upper_limit = _get_voxel_limits(
                        contact_point,
                        voxel_space_origin,
                        voxel_space_scale,
                        voxel_atom_start_indices,
                        voxel_atom_end_indices,
                    )
                    is_occupied = _check_occupancy_by_voxel(
                        contact_point,
                        lower_limit,
                        upper_limit,
                        voxel_atom_indices,
                        num_atoms,
                        atoms_data,
                        atom_plus_probe_radii_shrink_1d,
                        vertex_occupation,
                        refined_vertex_index,
                    )

                    if is_occupied:
                        if edges[edge_index] > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index] + 1
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index]
                        continue

                # Exposed
                if not is_occupied:
                    num_visible_vertices += 1

                    # WRITE TO GLOBAL DEVICE BUFFER
                    write_idx = global_coords_write_offset + local_coords_write_idx
                    if write_idx >= d_global_coords_offsets[tid + 1]:
                        return

                    d_exposed_grid_coordinates[write_idx, 0] = contact_point[0]
                    d_exposed_grid_coordinates[write_idx, 1] = contact_point[1]
                    d_exposed_grid_coordinates[write_idx, 2] = contact_point[2]
                    local_coords_write_idx += 1

                    if edges[edge_index] > 0:
                        if edges[edges[edge_index] + 1] > 0 or vertex_index2 > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index] + 1

                        if edges[edges[edge_index]] > 0 or vertex_index1 > 0:
                            edge_state_count += 1
                            edge_state[edge_state_count] = edges[edge_index]

        # Final state update for the pair
        if num_visible_vertices > 0:
            # WRITE EXPOSED PAIR INDEX
            d_global_pair_indices[global_pair_write_offset + local_pair_write_idx] = (
                pair_index
            )
            local_pair_write_idx += 1


# ====================================================================
# --- CUDA Orchestrator (Host Function) ---
# ====================================================================
def _calculate_exposed_grids_two_pass_cuda(
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
    int_type,
    real_type,
):
    """
    CUDA Two-Pass Parallel Orchestrator:
    1. Transfer inputs to Device.
    2. Launch CUDA Pass 1 (Count).
    3. Transfer counts back.
    4. Calculate offsets (Host).
    5. Transfer offsets back to Device.
    6. Launch CUDA Pass 2 (Write Coords & Pair Indices).
    7. Transfer results back (Host).
    8. Sequential state update (Host).
    """
    # ----------------------------
    # --- Safety Checks Before Launch
    # ----------------------------
    if num_vertices > SAS_MAX_VERTICES:
        raise RuntimeError(
            f"ERROR: num_vertices={num_vertices} exceeds SAS_MAX_VERTICES_SMALL={SAS_MAX_VERTICES}. "
            f"Aborting to prevent kernel overflow."
        )

    if num_edges > SAS_MAX_EDGES:
        raise RuntimeError(
            f"ERROR: num_edges={num_edges} exceeds SAS_MAX_EDGES_SMALL={SAS_MAX_EDGES}. "
            f"Aborting to prevent kernel overflow."
        )

    # ----------------------------
    # --- Launch Config (CUDA)
    # ----------------------------
    threads_per_block = SAS_THREADS_PER_BLOCK
    blocks_per_grid = SAS_BLOCKS_PER_GRID
    total_threads = SAS_THREADS_PER_BLOCK * SAS_BLOCKS_PER_GRID

    # ----------------------------
    # --- 1. Transfer Inputs to Device (D prefix)
    # ----------------------------
    d_atoms_data = cuda.to_device(atoms_data)
    d_atom_plus_probe_radii_1d = cuda.to_device(atom_plus_probe_radii_1d)
    d_atom_plus_probe_radii_shrink_1d = cuda.to_device(atom_plus_probe_radii_shrink_1d)
    d_atom_pairs = cuda.to_device(atom_pairs)
    d_vertex_array = cuda.to_device(vertex_array)
    d_edge_array = cuda.to_device(edge_array)
    d_edges = cuda.to_device(edges)
    d_voxel_space_origin = cuda.to_device(voxel_space_origin)
    d_voxel_atom_start_indices = cuda.to_device(voxel_atom_start_indices)
    d_voxel_atom_end_indices = cuda.to_device(voxel_atom_end_indices)
    d_voxel_atom_indices = cuda.to_device(voxel_atom_indices)

    # Initialize device output buffers for Pass 1 (one count per thread)
    d_per_thread_coords_count = cuda.to_device(np.zeros(total_threads, dtype=int_type))
    d_per_thread_pair_count = cuda.to_device(np.zeros(total_threads, dtype=int_type))

    # ----------------------------
    # --- 2. Launch CUDA Pass 1
    # ----------------------------
    _cuda_pass1_calculate_counts_kernel[int(blocks_per_grid), int(threads_per_block)](
        num_atoms,
        d_atoms_data,
        d_atom_plus_probe_radii_1d,
        d_atom_plus_probe_radii_shrink_1d,
        d_atom_pairs,
        d_vertex_array,
        d_edge_array,
        d_edges,
        d_voxel_space_origin,
        voxel_space_scale,
        d_voxel_atom_start_indices,
        d_voxel_atom_end_indices,
        d_voxel_atom_indices,
        num_vertices,
        num_edges,
        refinement_level,
        initial_vertex_count,
        total_threads,
        d_per_thread_coords_count,
        d_per_thread_pair_count,
    )

    # ----------------------------
    # --- 3. Transfer Counts Back (Host H prefix)
    # ----------------------------
    h_per_thread_coords_count = d_per_thread_coords_count.copy_to_host()
    h_per_thread_pair_count = d_per_thread_pair_count.copy_to_host()

    # ----------------------------
    # --- 4. Buffer Allocation and Offset Calculation (Sequential on Host)
    # ----------------------------
    total_exposed_grids = np.sum(h_per_thread_coords_count)
    num_exposed_atom_pairs = np.sum(h_per_thread_pair_count)

    # Global Buffer Allocation (1-based indexing for coordinates)
    h_exposed_grid_coordinates = np.zeros((total_exposed_grids + 1, 3), dtype=real_type)
    h_global_pair_indices = np.zeros(num_exposed_atom_pairs, dtype=int_type)

    # Calculate global start offsets for each thread's write:
    h_global_coords_offsets = np.zeros(total_threads + 1, dtype=int_type)
    h_global_pair_offsets = np.zeros(total_threads + 1, dtype=int_type)

    current_coords_offset = 1  # Coords start at index 1
    current_pair_offset = 0  # Pairs start at index 0

    for tid in range(total_threads):
        h_global_coords_offsets[tid] = current_coords_offset
        h_global_pair_offsets[tid] = current_pair_offset

        current_coords_offset += h_per_thread_coords_count[tid]
        current_pair_offset += h_per_thread_pair_count[tid]

    h_global_coords_offsets[total_threads] = current_coords_offset
    h_global_pair_offsets[total_threads] = current_pair_offset

    # ----------------------------
    # --- 5. Transfer Offsets and Outputs to Device
    # ----------------------------
    d_global_coords_offsets = cuda.to_device(h_global_coords_offsets)
    d_global_pair_offsets = cuda.to_device(h_global_pair_offsets)
    d_exposed_grid_coordinates = cuda.to_device(h_exposed_grid_coordinates)
    d_global_pair_indices = cuda.to_device(h_global_pair_indices)

    # ----------------------------
    # --- 6. Launch CUDA Pass 2
    # ----------------------------
    _cuda_pass2_generate_coords_kernel[
        int(SAS_BLOCKS_PER_GRID), int(SAS_THREADS_PER_BLOCK)
    ](
        num_atoms,
        d_atoms_data,
        d_atom_plus_probe_radii_1d,
        d_atom_plus_probe_radii_shrink_1d,
        d_atom_pairs,
        d_vertex_array,
        d_edge_array,
        d_edges,
        d_voxel_space_origin,
        voxel_space_scale,
        d_voxel_atom_start_indices,
        d_voxel_atom_end_indices,
        d_voxel_atom_indices,
        num_vertices,
        num_edges,
        refinement_level,
        initial_vertex_count,
        total_threads,
        d_global_coords_offsets,
        d_global_pair_indices,
        d_global_pair_offsets,
        d_exposed_grid_coordinates,
    )
    cuda.synchronize()

    # ----------------------------
    # --- 7. Transfer Results Back (Host)
    # ----------------------------
    exposed_grid_coordinates = d_exposed_grid_coordinates.copy_to_host()
    global_pair_indices = d_global_pair_indices.copy_to_host()

    # ----------------------------
    # --- 8. Sequential Aggregation of Atom Contact State (Host)
    # ----------------------------
    atom_contact_state_out = np.copy(atom_contact_state)

    for i in range(num_exposed_atom_pairs):
        pair_index = global_pair_indices[i]
        atom_index1 = atom_pairs[pair_index, 0]
        atom_index2 = atom_pairs[pair_index, 1]

        atom_contact_state_out[atom_index1] = 0
        if atom_index2 <= num_atoms:
            atom_contact_state_out[atom_index2] = 0

    num_accessible_atoms = np.sum(atom_contact_state_out[1 : num_atoms + 1] == 0)

    # Drop references to large device arrays
    d_atoms_data = None
    d_atom_plus_probe_radii_1d = None
    d_atom_plus_probe_radii_shrink_1d = None
    d_atom_pairs = None
    d_vertex_array = None
    d_edge_array = None
    d_edges = None
    d_voxel_space_origin = None
    d_voxel_atom_start_indices = None
    d_voxel_atom_end_indices = None
    d_voxel_atom_indices = None

    d_per_thread_coords_count = None
    d_per_thread_pair_count = None
    d_global_coords_offsets = None
    d_global_pair_offsets = None
    d_exposed_grid_coordinates = None
    d_global_pair_indices = None

    # Encourage Python GC
    gc.collect()

    # Return matches the CPU signature
    return (
        total_exposed_grids,
        exposed_grid_coordinates,
        num_accessible_atoms,
        atom_contact_state_out,
        num_exposed_atom_pairs,
    )


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
    num_extended_solute_grid_points,  # Unused in two-pass but kept for signature match
    use_cuda,  # Unused but kept for signature match
    num_threads,
    real_type,
    int_type,
):
    if use_cuda:
        return _calculate_exposed_grids_two_pass_cuda(
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
            int_type,
            real_type,
        )
    else:
        if num_threads < SERIAL_SURF_THREADS_THRESHOLD:
            return _calculate_exposed_grids_serial(
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
                num_extended_solute_grid_points,
                real_type,
                int_type,
            )
        else:
            return _calculate_exposed_grids_two_pass(
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
                num_extended_solute_grid_points,  # Unused in two-pass but kept for signature match
                use_cuda,  # Unused but kept for signature match
                num_threads,
                real_type,
                int_type,
            )


def _debug_print_accessible_atoms(atom_contact_state, label):
    accessible = np.nonzero(atom_contact_state[1:] == 0)[0] + 1
    print(f"[SAS-ACCESSIBLE-ATOMS] label={label} count={accessible.shape[0]}")
    for atom_id in accessible:
        print(f"[SAS-ACCESSIBLE-ATOM] label={label} atom={int(atom_id)}")


def solvent_accessible_surface(
    grid_spacing,
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
    use_cuda,
    num_threads,
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

    solute_space_range = (
        max_coords_by_axis - min_coords_by_axis
    ) + 2 * max_probe_radius
    solute_space_grids = np.ceil(solute_space_range / grid_spacing).astype(delphi_int)
    num_extended_solute_grid_points = delphi_int(np.prod(solute_space_grids))

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
        use_cuda,
        num_threads,
        delphi_real,
        delphi_int,
    )

    if _VERBOSITY <= DEBUG:
        vprint(
            DEBUG,
            _VERBOSITY,
            "initial atom_contact_state=[",
            ", ".join([str(a) for a in atom_contact_state]),
            "]",
        )
        backend_label = (
            "cuda"
            if use_cuda
            else (
                "cpu-parallel"
                if num_threads >= SERIAL_SURF_THREADS_THRESHOLD
                else "cpu-serial"
            )
        )
        _debug_print_contact_pairs(
            atom_pairs=atom_pairs,
            atoms_data=atoms_data,
            atom_plus_probe_radii_1d=atom_plus_probe_radii_1d,
            label=backend_label,
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
        num_extended_solute_grid_points=num_extended_solute_grid_points,
        use_cuda=use_cuda,
        num_threads=num_threads,
        real_type=delphi_real,
        int_type=delphi_int,
    )

    backend_label = (
        "cuda"
        if use_cuda
        else (
            "cpu-parallel"
            if num_threads >= SERIAL_SURF_THREADS_THRESHOLD
            else "cpu-serial"
        )
    )

    if _VERBOSITY <= DEBUG:
        _debug_print_accessible_atoms(atom_contact_state, backend_label)

    nprint_cpu(
        DEBUG,
        _VERBOSITY,
        "# exposed pairs (atom-atom and atom-object)= ",
        num_exposed_atom_pairs,
        "num_exposed_grids:",
        num_exposed_grids,
        "num_accessible_atoms:",
        num_accessible_atoms,
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
