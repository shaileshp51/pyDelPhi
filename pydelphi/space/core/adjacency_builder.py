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

import math
import numpy as np
from numba import set_num_threads, njit, prange, cuda

from pydelphi.foundation.platforms import Platform
from pydelphi.foundation.enums import Precision

from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_RADIUS,
    ATOMFIELD_ATOMIC_NUMBER,
    ConstDelPhiFloats as ConstDelPhi,
)

# --- PRIVATE CONSTANT: EXPANDED NEIGHBOR VOXEL RELATIVE COORDINATES FOR 5x5x5 SEARCH ---
# This array pre-calculates all (dx, dy, dz) offsets needed to check a 5x5x5 cube
# of voxels around a central voxel.
#
# Purpose:
# When determining atom overlaps, we use a coarse voxel map to quickly identify
# potential neighbors. An atom's influence (its Van der Waals radius + probe radius)
# can extend beyond its immediate 3x3x3 neighboring voxels if the atom's center
# is close to a voxel boundary and its radius is larger than the voxel size.
#
# Sufficiency of 5x5x5:
# A 5x5x5 search cube (ranging from -2 to +2 in each dimension for dx, dy, dz)
# ensures that all potentially overlapping atoms are found. It covers a region
# large enough to capture all overlaps while being more efficient than a full
# N*N all-pairs check. This size is robust for typical voxel sizes where voxel
# size is often set relative to atom radii.
#
# Initialization and Robustness:
# This array is explicitly sized to (125, 3) (5*5*5 = 125 coordinates) using `np.empty`
# and filled directly. This makes its dimensions transparent and consistent with its name.
# It is populated once when the module loads and then explicitly set to read-only
# (`flags.writeable = False`) to prevent any accidental modification during runtime.
# This ensures its immutability as a constant lookup table for both CPU and CUDA kernels.

_NEIGHBOR_VOXEL_REL_COORDS_5X5X5 = np.empty((125, 3), dtype=np.int32)
_idx = 0
for dz in range(-2, 3):
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            _NEIGHBOR_VOXEL_REL_COORDS_5X5X5[_idx, 0] = dx
            _NEIGHBOR_VOXEL_REL_COORDS_5X5X5[_idx, 1] = dy
            _NEIGHBOR_VOXEL_REL_COORDS_5X5X5[_idx, 2] = dz
            _idx += 1
_NEIGHBOR_VOXEL_REL_COORDS_5X5X5.flags.writeable = False  # Make the array read-only

APPROX_ZERO = ConstDelPhi.ApproxZero.value

from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
    vprint,
)
from pydelphi.config.logging_config import (
    DEBUG,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

# --- Dynamic Precision Handling ---
if PRECISION.int_value in {
    Precision.SINGLE.int_value,
}:
    try:
        import pydelphi.utils.cuda.single as size_gpu
    except ImportError:
        size_gpu = None
elif PRECISION.int_value == Precision.DOUBLE.int_value:
    try:
        import pydelphi.utils.cuda.double as size_gpu
    except ImportError:
        size_gpu = None
else:
    raise ValueError(f"Unsupported PRECISION: {PRECISION}")


# --- Numba CPU Kernel ---
@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_populate_overlap_matrices_atom_centric(
    atoms_data: np.ndarray[delphi_real],
    pairwise_overlap_matrix: np.ndarray[delphi_bool],
    voxel_atom_ids: np.ndarray[delphi_int],
    voxel_atom_start_index: np.ndarray[delphi_int],
    voxel_atom_end_index: np.ndarray[delphi_int],
    voxel_map_origin: np.ndarray[delphi_real],
    voxel_map_shape: np.ndarray[delphi_int],
    voxel_map_scale: delphi_real,
    neighbor_voxel_rel_coords: np.ndarray[delphi_int],  # Explicitly passed
) -> None:
    """
    (PRIVATE) Populates pairwise_overlap_matrix using an atom-centric approach on CPU.
    Utilizes pre-computed relative voxel coordinates for neighbor search.
    """
    num_atoms = atoms_data.shape[0]

    v_origin = voxel_map_origin
    v_shape = voxel_map_shape
    v_scale = voxel_map_scale

    for atom_idx1 in prange(num_atoms):
        atom1_data = atoms_data[atom_idx1]
        atom1_x = atom1_data[ATOMFIELD_X]
        atom1_y = atom1_data[ATOMFIELD_Y]
        atom1_z = atom1_data[ATOMFIELD_Z]
        atom1_radius = atom1_data[ATOMFIELD_RADIUS]
        atom1_is_hydrogen = int(atom1_data[ATOMFIELD_ATOMIC_NUMBER]) == 1

        central_vx = max(
            0, min(delphi_int((atom1_x - v_origin[0]) * v_scale), v_shape[0] - 1)
        )
        central_vy = max(
            0, min(delphi_int((atom1_y - v_origin[1]) * v_scale), v_shape[1] - 1)
        )
        central_vz = max(
            0, min(delphi_int((atom1_z - v_origin[2]) * v_scale), v_shape[2] - 1)
        )

        # Iterate through the pre-computed relative voxel coordinates
        for i in range(neighbor_voxel_rel_coords.shape[0]):
            dx = neighbor_voxel_rel_coords[i, 0]
            dy = neighbor_voxel_rel_coords[i, 1]
            dz = neighbor_voxel_rel_coords[i, 2]

            vx = central_vx + dx
            vy = central_vy + dy
            vz = central_vz + dz

            if 0 <= vx < v_shape[0] and 0 <= vy < v_shape[1] and 0 <= vz < v_shape[2]:
                start_idx = voxel_atom_start_index[vx, vy, vz]
                end_idx = voxel_atom_end_index[vx, vy, vz]

                if start_idx <= end_idx:
                    for atom_list_pos in range(start_idx, end_idx + 1):
                        atom_id2 = voxel_atom_ids[atom_list_pos]
                        atom_idx2 = atom_id2 - 1

                        if atom_id2 == 0 or atom_idx1 == atom_idx2:
                            continue

                        if atom_idx1 >= atom_idx2:
                            continue

                        atom2_data = atoms_data[atom_idx2]
                        atom2_x = atom2_data[ATOMFIELD_X]
                        atom2_y = atom2_data[ATOMFIELD_Y]
                        atom2_z = atom2_data[ATOMFIELD_Z]
                        atom2_radius = atom2_data[ATOMFIELD_RADIUS]
                        atom2_is_hydrogen = (
                            int(atom2_data[ATOMFIELD_ATOMIC_NUMBER]) == 1
                        )

                        delta_x = atom1_x - atom2_x
                        delta_y = atom1_y - atom2_y
                        delta_z = atom1_z - atom2_z
                        dist_sq = (
                            delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
                        )

                        sum_radii = atom1_radius + atom2_radius
                        overlap_dist_sq_threshold = sum_radii * sum_radii

                        none_is_hydrogen = not (atom1_is_hydrogen or atom2_is_hydrogen)

                        if dist_sq < overlap_dist_sq_threshold and none_is_hydrogen:
                            pairwise_overlap_matrix[atom_idx1, atom_idx2] = True
                            pairwise_overlap_matrix[atom_idx2, atom_idx1] = True


# --- Numba CUDA Kernel ---
@cuda.jit(cache=True)
def _cuda_populate_overlap_matrices_atom_centric(
    d_atoms_data,
    d_pairwise_overlap_matrix,
    d_voxel_atom_ids,
    d_voxel_atom_start_index,
    d_voxel_atom_end_index,
    d_voxel_map_origin,
    d_voxel_map_shape,
    d_voxel_map_scale,
    d_neighbor_voxel_rel_coords,  # Explicitly passed
):
    """
    (PRIVATE) Populates pairwise_overlap_matrix using an atom-centric approach on CUDA.
    Utilizes pre-computed relative voxel coordinates for neighbor search.
    """
    atom_idx1 = cuda.grid(1)

    num_atoms = d_atoms_data.shape[0]
    if atom_idx1 >= num_atoms:
        return

    v_origin = d_voxel_map_origin
    v_shape = d_voxel_map_shape
    v_scale = d_voxel_map_scale

    atom1_data = d_atoms_data[atom_idx1]
    atom1_x = atom1_data[ATOMFIELD_X]
    atom1_y = atom1_data[ATOMFIELD_Y]
    atom1_z = atom1_data[ATOMFIELD_Z]
    atom1_radius = atom1_data[ATOMFIELD_RADIUS]
    atom1_is_hydrogen = int(atom1_data[ATOMFIELD_ATOMIC_NUMBER]) == 1

    central_vx = max(
        0, min(delphi_int((atom1_x - v_origin[0]) * v_scale), v_shape[0] - 1)
    )
    central_vy = max(
        0, min(delphi_int((atom1_y - v_origin[1]) * v_scale), v_shape[1] - 1)
    )
    central_vz = max(
        0, min(delphi_int((atom1_z - v_origin[2]) * v_scale), v_shape[2] - 1)
    )

    # Iterate through the pre-computed relative voxel coordinates
    for i in range(d_neighbor_voxel_rel_coords.shape[0]):
        dx = d_neighbor_voxel_rel_coords[i, 0]
        dy = d_neighbor_voxel_rel_coords[i, 1]
        dz = d_neighbor_voxel_rel_coords[i, 2]

        vx = central_vx + dx
        vy = central_vy + dy
        vz = central_vz + dz

        if 0 <= vx < v_shape[0] and 0 <= vy < v_shape[1] and 0 <= vz < v_shape[2]:
            start_idx = d_voxel_atom_start_index[vx, vy, vz]
            end_idx = d_voxel_atom_end_index[vx, vy, vz]

            if start_idx <= end_idx:
                for atom_list_pos in range(start_idx, end_idx + 1):
                    atom_id2 = d_voxel_atom_ids[atom_list_pos]
                    atom_idx2 = atom_id2 - 1

                    if atom_id2 == 0 or atom_idx1 == atom_idx2:
                        continue

                    if atom_idx1 >= atom_idx2:
                        continue

                    atom2_data = d_atoms_data[atom_idx2]
                    atom2_x = atom2_data[ATOMFIELD_X]
                    atom2_y = atom2_data[ATOMFIELD_Y]
                    atom2_z = atom2_data[ATOMFIELD_Z]
                    atom2_radius = atom2_data[ATOMFIELD_RADIUS]
                    atom2_is_hydrogen = int(atom2_data[ATOMFIELD_ATOMIC_NUMBER]) == 1

                    delta_x = atom1_x - atom2_x
                    delta_y = atom1_y - atom2_y
                    delta_z = atom1_z - atom2_z
                    dist_sq = delta_x * delta_x + delta_y * delta_y + delta_z * delta_z

                    sum_radii = atom1_radius + atom2_radius
                    overlap_dist_sq_threshold = sum_radii * sum_radii

                    none_is_hydrogen = not (atom1_is_hydrogen or atom2_is_hydrogen)

                    if dist_sq < overlap_dist_sq_threshold and none_is_hydrogen:
                        d_pairwise_overlap_matrix[atom_idx1, atom_idx2] = True
                        d_pairwise_overlap_matrix[atom_idx2, atom_idx1] = True


# --- CPU Helper Kernels (remain unchanged) ---
@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_count_overlaps_per_atom(
    pairwise_overlap_matrix: np.ndarray[delphi_bool],
    overlap_counts_per_atom: np.ndarray[delphi_int],
) -> None:
    """
    (PRIVATE) Counts the number of overlapping atoms for each atom.
    """
    num_atoms = pairwise_overlap_matrix.shape[0]
    for ix in prange(num_atoms):
        current_atom_overlap_count = 0
        for iy in range(num_atoms):
            if ix == iy:
                continue
            if pairwise_overlap_matrix[ix, iy]:
                current_atom_overlap_count += 1
        overlap_counts_per_atom[ix] = current_atom_overlap_count


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_build_adjacency_map(
    pairwise_overlap_matrix: np.ndarray[delphi_bool],
    adjacency_map: np.ndarray[delphi_int],
    sentinel_value: delphi_int,
) -> None:
    """
    (PRIVATE) Populates a pre-allocated 2D NumPy array with the adjacency information.
    """
    num_atoms = pairwise_overlap_matrix.shape[0]
    for ix in prange(num_atoms):
        current_overlap_idx = 0
        for iy in range(num_atoms):
            if ix == iy:
                continue
            if pairwise_overlap_matrix[ix, iy]:
                adjacency_map[ix, current_overlap_idx] = iy
                current_overlap_idx += 1


# --- Public Orchestrator Function ---
def calculate_atom_overlap_adjacency(
    platform: Platform,
    atoms_data: np.ndarray[delphi_real],
    voxel_atom_ids: np.ndarray[delphi_int],
    voxel_atom_start_index: np.ndarray[delphi_int],
    voxel_atom_end_index: np.ndarray[delphi_int],
    voxel_map_origin: np.ndarray[delphi_real],
    voxel_map_shape: np.ndarray[delphi_int],
    voxel_map_scale: delphi_real,
    sentinel_value: delphi_int = -1,
) -> np.ndarray[delphi_int]:
    """
    Calculates the Van der Waals overlap adjacency for atoms and returns it as a
    fixed-size 2D NumPy array. This function orchestrates the entire process,
    dispatching to CPU or CUDA kernels based on the 'platform' argument.

    All intermediate calculations are encapsulated within this module.

    Args:
        platform (str): The platform instance storing configured platform information.
        atoms_data (np.ndarray): 2D array where each row represents an atom (position, radius, etc.).
        voxel_atom_ids (np.ndarray): Pre-built 1D array of 1-based atom IDs in their voxels.
        voxel_atom_start_index (np.ndarray): Pre-built 3D array of start indices for atoms in voxels.
        voxel_atom_end_index (np.ndarray): Pre-built 3D array of end indices for atoms in voxels.
        voxel_map_origin (np.ndarray): Physical coordinates of the origin of the pre-built voxel map.
        voxel_map_shape (np.ndarray): Dimensions (vx, vy, vz) of the pre-built voxel map.
        voxel_map_scale (delphi_real): Scale factor (1.0 / voxel_size) of the pre-built voxel map.
        sentinel_value (delphi_int): The value to use for empty slots in the adjacency map (-1 by default).


    Returns:
        np.ndarray[delphi_int]: A 2D NumPy array of shape (num_atoms, max_overlaps_per_atom).
                                Each row `i` contains the 0-based indices of atoms
                                overlapping with atom `i`, followed by the sentinel_value.
    """
    num_atoms = atoms_data.shape[0]

    # pairwise_overlap_matrix is a temporary, internal structure used to facilitate
    # determining the size of the adjacency map and populating it. It is not returned,
    # ensuring efficient memory utilization and a clean public API.
    pairwise_overlap_matrix = np.zeros((num_atoms, num_atoms), dtype=delphi_bool)

    if num_atoms == 0:
        return np.full((0, 0), sentinel_value, dtype=delphi_int)

    if platform.active == "cuda":
        if not cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot run on GPU.")

        vprint(f"Running atom overlap calculation on CUDA...", min_verbosity=DEBUG)

        # Transfer data to GPU
        d_atoms_data = cuda.to_device(atoms_data)
        d_pairwise_overlap_matrix = cuda.to_device(pairwise_overlap_matrix)
        d_voxel_atom_ids = cuda.to_device(voxel_atom_ids)
        d_voxel_atom_start_index = cuda.to_device(voxel_atom_start_index)
        d_voxel_atom_end_index = cuda.to_device(voxel_atom_end_index)
        d_voxel_map_origin = cuda.to_device(voxel_map_origin)
        d_voxel_map_shape = cuda.to_device(voxel_map_shape)
        d_neighbor_voxel_rel_coords = cuda.to_device(
            _NEIGHBOR_VOXEL_REL_COORDS_5X5X5
        )  # Transfer this to device

        # Configure CUDA launch
        threads_per_block = 256
        blocks_per_grid = (num_atoms + (threads_per_block - 1)) // threads_per_block

        _cuda_populate_overlap_matrices_atom_centric[
            blocks_per_grid, threads_per_block
        ](
            d_atoms_data,
            d_pairwise_overlap_matrix,
            d_voxel_atom_ids,
            d_voxel_atom_start_index,
            d_voxel_atom_end_index,
            d_voxel_map_origin,
            d_voxel_map_shape,
            voxel_map_scale,
            d_neighbor_voxel_rel_coords,  # Pass device array to kernel
        )
        cuda.synchronize()

        # Copy updated pairwise_overlap_matrix back to CPU for subsequent steps
        pairwise_overlap_matrix = d_pairwise_overlap_matrix.copy_to_host()

    elif platform.active == "cpu":
        vprint(DEBUG, _VERBOSITY, f"Running atom overlap calculation on CPU...")
        _cpu_populate_overlap_matrices_atom_centric(
            atoms_data,
            pairwise_overlap_matrix,
            voxel_atom_ids,
            voxel_atom_start_index,
            voxel_atom_end_index,
            voxel_map_origin,
            voxel_map_shape,
            voxel_map_scale,
            _NEIGHBOR_VOXEL_REL_COORDS_5X5X5,  # Pass CPU array to kernel
        )
    else:
        raise ValueError(f"Unsupported platform: {platform}. Choose 'cpu' or 'cuda'.")

    # These steps remain on CPU, operating on the pairwise_overlap_matrix
    overlap_counts_per_atom = np.zeros(num_atoms, dtype=delphi_int)
    _cpu_count_overlaps_per_atom(pairwise_overlap_matrix, overlap_counts_per_atom)

    max_overlaps_per_atom = 0
    if num_atoms > 0:
        max_overlaps_per_atom = np.max(overlap_counts_per_atom)

    adjacency_map = np.full(
        (num_atoms, max_overlaps_per_atom), sentinel_value, dtype=delphi_int
    )

    _cpu_build_adjacency_map(pairwise_overlap_matrix, adjacency_map, sentinel_value)

    _pretty_print_adjacency_map(
        adjacency_map, sentinel_value=-1, max_rows_to_print=num_atoms
    )

    return adjacency_map


# --- Private Debugging Utility ---
def _pretty_print_adjacency_map(
    adjacency_map: np.ndarray[delphi_int],
    sentinel_value: delphi_int = -1,
    max_rows_to_print: int = 20,
    atom_id_offset: int = 0,
) -> None:
    """
    (PRIVATE) Prints the 2D adjacency map in a human-readable format.
    """
    num_atoms, _ = adjacency_map.shape
    print(f"\n--- Adjacency Map ({num_atoms} atoms) ---")

    rows_to_show = num_atoms
    if max_rows_to_print > 0:
        rows_to_show = min(num_atoms, max_rows_to_print)

    for i in range(rows_to_show):
        neighbors = []
        for j in range(adjacency_map.shape[1]):
            neighbor_id = adjacency_map[i, j]
            if neighbor_id != sentinel_value:
                neighbors.append(str(neighbor_id + atom_id_offset))
            else:
                break

        if neighbors:
            print(f"Atom {i + atom_id_offset}: [{', '.join(neighbors)}]")
        else:
            print(f"Atom {i + atom_id_offset}: [No overlaps]")

    if num_atoms > rows_to_show:
        print(f"... (Truncated, showing first {rows_to_show} atoms) ...")

    print("--- End Adjacency Map ---")
