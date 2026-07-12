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
# This array is explicitly sized to (125, 3) (5*5*5 = 125 coordinates) using `np.zeros`
# and filled directly. This makes its dimensions transparent and consistent with its name.
# It is populated once when the module loads and then explicitly set to read-only
# (`flags.writeable = False`) to prevent any accidental modification during runtime.
# This ensures its immutability as a constant lookup table for both CPU and CUDA kernels.

_NEIGHBOR_VOXEL_REL_COORDS_5X5X5 = np.zeros((125, 3), dtype=np.int32)
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
    VERBOSE,
)

from pydelphi.foundation.models import AtomAdjacencyCSR

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)


# --------------------------------------------------------------------------
# CPU kernels: COUNT and FILL for symmetric CSR
# --------------------------------------------------------------------------


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _cpu_count_overlaps_atom_centric(
    atoms_data: np.ndarray,
    probe_radius: delphi_real,
    r_offset: delphi_real,
    overlap_counts_per_atom: np.ndarray,  # shape (num_atoms,)
    voxel_atom_ids: np.ndarray,
    voxel_atom_start_index: np.ndarray,
    voxel_atom_end_index: np.ndarray,
    voxel_map_origin: np.ndarray,
    voxel_map_shape: np.ndarray,
    voxel_map_scale: delphi_real,
    neighbor_voxel_rel_coords: np.ndarray,  # (125, 3)
) -> None:
    """
    CPU pass 1: compute outgoing degree for each atom.
    Each thread only writes overlap_counts_per_atom[atom_idx1], so prange is safe.
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

        count = 0

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
                        if atom_id2 == 0:
                            continue

                        atom_idx2 = atom_id2 - 1
                        if atom_idx1 == atom_idx2:
                            continue  # only skip self – no i>j/i<j logic

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
                        sum_radii_extended = sum_radii + 2 * (probe_radius + r_offset)
                        overlap_dist_sq_threshold = (
                            sum_radii_extended * sum_radii_extended
                        )

                        none_is_hydrogen = not (atom1_is_hydrogen or atom2_is_hydrogen)

                        if dist_sq < overlap_dist_sq_threshold and none_is_hydrogen:
                            count += 1

        overlap_counts_per_atom[atom_idx1] = count


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _cpu_fill_csr_atom_centric(
    atoms_data: np.ndarray,
    probe_radius: delphi_real,
    r_offset: delphi_real,
    csr_row_ptr: np.ndarray,  # length N+1
    csr_col_idx: np.ndarray,  # length sum(degree)
    voxel_atom_ids: np.ndarray,
    voxel_atom_start_index: np.ndarray,
    voxel_atom_end_index: np.ndarray,
    voxel_map_origin: np.ndarray,
    voxel_map_shape: np.ndarray,
    voxel_map_scale: delphi_real,
    neighbor_voxel_rel_coords: np.ndarray,
) -> None:
    """
    CPU pass 2: fill CSR adjacency, row by row.
    Each atom only writes into its own row, so prange is safe and no atomics needed.
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

        row_start = csr_row_ptr[atom_idx1]
        pos = 0  # offset within this row

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
                        if atom_id2 == 0:
                            continue

                        atom_idx2 = atom_id2 - 1
                        if atom_idx1 == atom_idx2:
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
                        sum_radii_extended = sum_radii + 2 * (probe_radius + r_offset)
                        overlap_dist_sq_threshold = (
                            sum_radii_extended * sum_radii_extended
                        )

                        none_is_hydrogen = not (atom1_is_hydrogen or atom2_is_hydrogen)

                        if dist_sq < overlap_dist_sq_threshold and none_is_hydrogen:
                            csr_col_idx[row_start + pos] = atom_idx2
                            pos += 1


# --------------------------------------------------------------------------
# CUDA kernels: COUNT and FILL for symmetric CSR
# --------------------------------------------------------------------------


@cuda.jit(cache=True)
def _cuda_count_overlaps_atom_centric(
    d_atoms_data,
    probe_radius,
    r_offset,
    d_overlap_counts_per_atom,  # int32, length N
    d_voxel_atom_ids,
    d_voxel_atom_start_index,
    d_voxel_atom_end_index,
    d_voxel_map_origin,
    d_voxel_map_shape,
    d_voxel_map_scale,
    d_neighbor_voxel_rel_coords,
):
    """
    CUDA pass 1: per-atom outgoing degree; each thread owns one atom, no atomics.
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

    count = 0

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
                    if atom_id2 == 0:
                        continue

                    atom_idx2 = atom_id2 - 1
                    if atom_idx1 == atom_idx2:
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
                    sum_radii_extended = sum_radii + 2 * (probe_radius + r_offset)
                    overlap_dist_sq_threshold = sum_radii_extended * sum_radii_extended

                    none_is_hydrogen = not (atom1_is_hydrogen or atom2_is_hydrogen)

                    if dist_sq < overlap_dist_sq_threshold and none_is_hydrogen:
                        count += 1

    d_overlap_counts_per_atom[atom_idx1] = count


@cuda.jit(cache=True)
def _cuda_fill_csr_atom_centric(
    d_atoms_data,
    probe_radius,
    r_offset,
    d_csr_row_ptr,
    d_csr_col_idx,
    d_voxel_atom_ids,
    d_voxel_atom_start_index,
    d_voxel_atom_end_index,
    d_voxel_map_origin,
    d_voxel_map_shape,
    d_voxel_map_scale,
    d_neighbor_voxel_rel_coords,
):
    """
    CUDA pass 2: fill CSR; each thread writes only its row, so no atomics needed.
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

    row_start = d_csr_row_ptr[atom_idx1]
    pos = 0

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
                    if atom_id2 == 0:
                        continue

                    atom_idx2 = atom_id2 - 1
                    if atom_idx1 == atom_idx2:
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
                    sum_radii_extended = sum_radii + 2 * (probe_radius + r_offset)
                    overlap_dist_sq_threshold = sum_radii_extended * sum_radii_extended

                    none_is_hydrogen = not (atom1_is_hydrogen or atom2_is_hydrogen)

                    if dist_sq < overlap_dist_sq_threshold and none_is_hydrogen:
                        d_csr_col_idx[row_start + pos] = atom_idx2
                        pos += 1


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _cpu_sort_csr_rows(
    csr_row_ptr: np.ndarray,
    csr_col_idx: np.ndarray,
) -> None:
    """
    Sort column indices within each CSR row in-place, in parallel over rows.
    """
    n_rows = csr_row_ptr.shape[0] - 1

    for i in prange(n_rows):
        start = csr_row_ptr[i]
        end = csr_row_ptr[i + 1]
        if end > start:
            csr_col_idx[start:end] = np.sort(csr_col_idx[start:end])


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _cpu_count_unique_per_row(
    csr_row_ptr: np.ndarray,
    csr_col_idx: np.ndarray,
    row_counts: np.ndarray,  # preallocated, length = n_rows
) -> None:
    """
    Count number of unique entries in each sorted CSR row.
    Assumes rows are already sorted.
    """
    n_rows = csr_row_ptr.shape[0] - 1

    for i in prange(n_rows):
        start = csr_row_ptr[i]
        end = csr_row_ptr[i + 1]

        if end <= start:
            row_counts[i] = 0
        else:
            cnt = 1
            last = csr_col_idx[start]
            for k in range(start + 1, end):
                v = csr_col_idx[k]
                if v != last:
                    cnt += 1
                    last = v
            row_counts[i] = cnt


@njit(nogil=True, boundscheck=False, cache=True)
def _cpu_prefix_sum_row_counts(
    row_counts: np.ndarray,
    new_row_ptr: np.ndarray,  # length n_rows + 1
) -> None:
    """
    Sequential prefix sum over row_counts to build new_row_ptr.
    """
    n_rows = row_counts.shape[0]
    new_row_ptr[0] = 0
    total = 0
    for i in range(n_rows):
        total += row_counts[i]
        new_row_ptr[i + 1] = total


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _cpu_fill_dedup_csr_rows(
    csr_row_ptr: np.ndarray,
    csr_col_idx: np.ndarray,
    new_row_ptr: np.ndarray,
    new_col_idx: np.ndarray,
) -> None:
    """
    Fill new_col_idx with unique entries from each sorted CSR row.
    Assumes new_row_ptr is already computed via prefix sum of unique counts.
    """
    n_rows = new_row_ptr.shape[0] - 1

    for i in prange(n_rows):
        start = csr_row_ptr[i]
        end = csr_row_ptr[i + 1]
        write = new_row_ptr[i]

        if end <= start:
            continue

        last = csr_col_idx[start]
        new_col_idx[write] = last
        write += 1

        for k in range(start + 1, end):
            v = csr_col_idx[k]
            if v != last:
                new_col_idx[write] = v
                write += 1
                last = v


@njit(nogil=True, boundscheck=False, cache=True)
def _cpu_dedup_csr_rows(
    csr_row_ptr: np.ndarray,
    csr_col_idx: np.ndarray,
):
    """
    Top-level dedup wrapper: sort rows first (outside), then call this to:
      - compute unique counts per row (parallel),
      - prefix-sum them to new_row_ptr (sequential),
      - fill new_col_idx with uniques (parallel).

    Returns:
        new_row_ptr, new_col_idx
    """
    n_rows = csr_row_ptr.shape[0] - 1

    # A) count uniques per row (parallel)
    row_counts = np.empty(n_rows, dtype=csr_row_ptr.dtype)
    _cpu_count_unique_per_row(csr_row_ptr, csr_col_idx, row_counts)

    # B) prefix sum (sequential but cheap)
    new_row_ptr = np.empty(n_rows + 1, dtype=csr_row_ptr.dtype)
    _cpu_prefix_sum_row_counts(row_counts, new_row_ptr)

    total = new_row_ptr[-1]
    new_col_idx = np.empty(total, dtype=csr_col_idx.dtype)

    # C) fill uniques per row (parallel)
    _cpu_fill_dedup_csr_rows(csr_row_ptr, csr_col_idx, new_row_ptr, new_col_idx)

    return new_row_ptr, new_col_idx


# --------------------------------------------------------------------------
# PUBLIC API: calculate_atom_overlap_adjacency -> symmetric CSR
# --------------------------------------------------------------------------


def build_atom_overlap_adjacency_csr(
    platform: Platform,
    atoms_data: np.ndarray[delphi_real],
    probe_radius: delphi_real,
    r_offset: delphi_real,
    voxel_atom_ids: np.ndarray[delphi_int],
    voxel_atom_start_index: np.ndarray[delphi_int],
    voxel_atom_end_index: np.ndarray[delphi_int],
    voxel_map_origin: np.ndarray[delphi_real],
    voxel_map_shape: np.ndarray[delphi_int],
    voxel_map_scale: delphi_real,
):
    """
    Calculates Van der Waals overlap adjacency for atoms and returns a
    **symmetric CSR adjacency** instead of a dense or fixed-width adjacency map.

    Returns:
        csr_row_ptr : np.ndarray[delphi_int], shape (N+1,)
        csr_col_idx : np.ndarray[delphi_int], shape (2E,)
        overlap_counts_per_atom : np.ndarray[delphi_int], shape (N,)

    where:
        - N is number of atoms
        - E is number of undirected overlapping pairs
        - neighbors of atom i are csr_col_idx[csr_row_ptr[i] : csr_row_ptr[i+1]]
    """
    num_atoms = atoms_data.shape[0]

    if num_atoms == 0:
        empty = np.zeros(0, dtype=delphi_int)
        return empty, empty, empty

    # degrees (overlaps per atom); CPU uses delphi_int, CUDA uses int32 internally
    if platform.active == "cuda":
        # int32 for cuda.atomic.add
        overlap_counts_host = np.zeros(num_atoms, dtype=np.int32)
    else:
        overlap_counts_host = np.zeros(num_atoms, dtype=delphi_int)

    if platform.active == "cuda":
        if not cuda.is_available():
            raise RuntimeError("CUDA is not available. Cannot run on GPU.")

        vprint(
            DEBUG,
            _VERBOSITY,
            "Running atom overlap calculation on CUDA (symmetric CSR)...",
        )

        d_atoms_data = cuda.to_device(atoms_data)
        d_overlap_counts = cuda.to_device(overlap_counts_host)
        d_voxel_atom_ids = cuda.to_device(voxel_atom_ids)
        d_voxel_atom_start_index = cuda.to_device(voxel_atom_start_index)
        d_voxel_atom_end_index = cuda.to_device(voxel_atom_end_index)
        d_voxel_map_origin = cuda.to_device(voxel_map_origin)
        d_voxel_map_shape = cuda.to_device(voxel_map_shape)
        d_neighbor_voxel_rel_coords = cuda.to_device(_NEIGHBOR_VOXEL_REL_COORDS_5X5X5)

        threads_per_block = 256
        blocks_per_grid = (num_atoms + (threads_per_block - 1)) // threads_per_block

        _cuda_count_overlaps_atom_centric[int(blocks_per_grid), int(threads_per_block)](
            d_atoms_data,
            probe_radius,
            r_offset,
            d_overlap_counts,
            d_voxel_atom_ids,
            d_voxel_atom_start_index,
            d_voxel_atom_end_index,
            d_voxel_map_origin,
            d_voxel_map_shape,
            voxel_map_scale,
            d_neighbor_voxel_rel_coords,
        )
        cuda.synchronize()

        overlap_counts_host = d_overlap_counts.copy_to_host()

    elif platform.active == "cpu":
        vprint(
            DEBUG,
            _VERBOSITY,
            "Running atom overlap calculation on CPU (symmetric CSR)...",
        )
        _cpu_count_overlaps_atom_centric(
            atoms_data,
            probe_radius,
            r_offset,
            overlap_counts_host,
            voxel_atom_ids,
            voxel_atom_start_index,
            voxel_atom_end_index,
            voxel_map_origin,
            voxel_map_shape,
            voxel_map_scale,
            _NEIGHBOR_VOXEL_REL_COORDS_5X5X5,
        )
    else:
        raise ValueError(f"Unsupported platform: {platform}. Choose 'cpu' or 'cuda'.")

    # Cast CUDA counts to delphi_int if needed
    if overlap_counts_host.dtype != delphi_int:
        overlap_counts_per_atom = overlap_counts_host.astype(delphi_int)
    else:
        overlap_counts_per_atom = overlap_counts_host

    # Build CSR row_ptr
    csr_row_ptr = np.empty(num_atoms + 1, dtype=delphi_int)
    csr_row_ptr[0] = 0
    for i in range(num_atoms):
        csr_row_ptr[i + 1] = csr_row_ptr[i] + overlap_counts_per_atom[i]

    total_neighbors = int(csr_row_ptr[-1])
    csr_col_idx = np.empty(total_neighbors, dtype=delphi_int)

    # Fill CSR
    if platform.active == "cuda":
        d_atoms_data = cuda.to_device(atoms_data)
        d_csr_row_ptr = cuda.to_device(csr_row_ptr)
        d_csr_col_idx = cuda.to_device(csr_col_idx)
        d_current_pos = cuda.to_device(np.zeros(num_atoms, dtype=np.int32))
        d_voxel_atom_ids = cuda.to_device(voxel_atom_ids)
        d_voxel_atom_start_index = cuda.to_device(voxel_atom_start_index)
        d_voxel_atom_end_index = cuda.to_device(voxel_atom_end_index)
        d_voxel_map_origin = cuda.to_device(voxel_map_origin)
        d_voxel_map_shape = cuda.to_device(voxel_map_shape)
        d_neighbor_voxel_rel_coords = cuda.to_device(_NEIGHBOR_VOXEL_REL_COORDS_5X5X5)

        threads_per_block = 256
        blocks_per_grid = (num_atoms + (threads_per_block - 1)) // threads_per_block

        _cuda_fill_csr_atom_centric[int(blocks_per_grid), int(threads_per_block)](
            d_atoms_data,
            probe_radius,
            r_offset,
            d_csr_row_ptr,
            d_csr_col_idx,
            d_voxel_atom_ids,
            d_voxel_atom_start_index,
            d_voxel_atom_end_index,
            d_voxel_map_origin,
            d_voxel_map_shape,
            voxel_map_scale,
            d_neighbor_voxel_rel_coords,
        )
        cuda.synchronize()

        csr_col_idx = d_csr_col_idx.copy_to_host()
    else:
        _cpu_fill_csr_atom_centric(
            atoms_data,
            probe_radius,
            r_offset,
            csr_row_ptr,
            csr_col_idx,
            voxel_atom_ids,
            voxel_atom_start_index,
            voxel_atom_end_index,
            voxel_map_origin,
            voxel_map_shape,
            voxel_map_scale,
            _NEIGHBOR_VOXEL_REL_COORDS_5X5X5,
        )

    # 1) sort neighbors (enables dedup + binary search)
    _cpu_sort_csr_rows(csr_row_ptr, csr_col_idx)

    # 2) deduplicate neighbors (parallel counting + filling)
    csr_row_ptr, csr_col_idx = _cpu_dedup_csr_rows(csr_row_ptr, csr_col_idx)

    # 3) degrees from final non-redundant CSR
    overlap_counts_per_atom = csr_row_ptr[1:] - csr_row_ptr[:-1]

    if VERBOSE >= _VERBOSITY:
        _pretty_print_csr_adjacency(
            csr_row_ptr, csr_col_idx, max_rows_to_print=50, atom_id_offset=0
        )

    return AtomAdjacencyCSR(csr_row_ptr, csr_col_idx, overlap_counts_per_atom)


def _pretty_print_csr_adjacency(
    csr_row_ptr: np.ndarray,
    csr_col_idx: np.ndarray,
    max_rows_to_print: int = 20,
    atom_id_offset: int = 0,
) -> None:
    """
    Pretty-print a CSR adjacency structure for debugging/testing.

    Args:
        csr_row_ptr (np.ndarray): CSR row pointer array, shape (N+1,)
        csr_col_idx (np.ndarray): CSR column indices array, shape (nnz,)
        max_rows_to_print (int): Maximum number of atom rows to print (0 = unlimited)
        atom_id_offset (int): Value to add to printed atom IDs (e.g., +1 for 1-based)
    """
    num_atoms = csr_row_ptr.shape[0] - 1
    nnz = csr_col_idx.shape[0]

    print(f"\n--- CSR Adjacency ({num_atoms} atoms, {nnz} total neighbors) ---")

    rows_to_show = (
        num_atoms if max_rows_to_print <= 0 else min(num_atoms, max_rows_to_print)
    )

    for i in range(rows_to_show):
        start = csr_row_ptr[i]
        end = csr_row_ptr[i + 1]

        if start == end:
            print(f"Atom {i + atom_id_offset}: [No overlaps]")
            continue

        # neighbors sorted + deduped already
        neighbors = csr_col_idx[start:end]
        neighbors_str = ", ".join(str(int(n) + atom_id_offset) for n in neighbors)

        print(f"Atom {i + atom_id_offset}: [{neighbors_str}]")

    if num_atoms > rows_to_show:
        print(f"... (Truncated, showing first {rows_to_show} atoms) ...")

    print("--- End CSR Adjacency ---")
