#!/usr/bin/env python
# coding: utf-8
#
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

import numpy as np
from numba import njit, prange, float64, int32, int64, float32

from pydelphi.constants import (
    ATOMFIELD_ATOMIC_NUMBER,
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_RADIUS,
    ATOMFIELD_LJ_GAMMA,
    LEN_ATOMFIELDS,
)

from pydelphi.constants import ConstPhysical

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

PI = ConstPhysical.Pi.value
FOUR_PI = 4.0 * PI
FOUR_THIRD_PI = FOUR_PI / 3.0

K_CONSTANT = 2.227
SIGMOID_G = 5.0
K_OVER_PI_POW_3_2 = (K_CONSTANT / PI) ** 1.5

MAX_ORDER = 10
MIN_VOLUME = 1.0e-3

KT_TO_KCAL = 0.6162  # R * T at 298K / 4.184 J/cal (approx)

ATOMPROP_X = 0  # coordinate X
ATOMPROP_Y = 1  # coordinate Y
ATOMPROP_Z = 2  # coordinate Z
ATOMPROP_Z_NUM = 3  # atomic number Z
ATOMPROP_GAMMA = 4  # atomic gamma (for SA energy)
ATOMPROP_R_VDW = 5  # raw VDW radius
ATOMPROP_R_EFF = 6  # effective radius (Rvdw + roffset)
ATOMPROP_ALPHA = 7  # alpha_i for Gaussian overlap
ATOMPROP_P = 8  # P_i parameter (Gaussian prefactor)
NUM_ATOMPROPS = 9


# --- Index Constants for Homogeneous Array (NUM_FIELDS_OVERLAP = 19) ---
# OVR_* = OverlapRecord field indices for N-body atomic overlap expansion

OVR_ORDER_N = 0  # Overlap order N
OVR_VOLUME_VN = 1  # Volume contribution V_N
OVR_AREA_DERIV_AN = 2  # Surface-area derivative A_N
OVR_ALPHA_SUM = 3  # Σ alpha_i
OVR_PRODUCT_P = 4  # Π p_i
OVR_LAMBDA_SUM = 5  # Σ lambda_ij

OVR_CENTER_COORDS = 6  # rX, rY, rZ (flat: indices 6–8)

OVR_ATOM_IDS = 9  # Atom indices i1..i10 (padding −1)

NUM_FIELDS_OVERLAP = 19  # Total length of one OverlapRecord

from pydelphi.config.logging_config import (
    CRITICAL,
    ERROR,
    VERBOSE,
    DEBUG,
    TRACE,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)


def _debug_print_overlap_summary(
    total_volume,
    total_SA,
    cavity_volume,
    overlap_records,
    max_print_records=50,
):
    """
    Internal debug helper for inspecting Gaussian-overlap geometry and
    overlap records in a structured, C++-like format.

    Not part of the public API; do not call from outside this module.
    """
    num_records = overlap_records.shape[0]

    # ---- Geometry summary ----
    print("\n" + "=" * 70)
    print("                 GAUSSIAN OVERLAP GEOMETRY SUMMARY                 ")
    print("=" * 70)
    print(f"{'Total Molecular Volume (Gaussian):':<45} {total_volume:10.4f} Ang^3")
    print(
        f"{'Total Molecular Volume corrected for SEV:':<45} "
        f"{cavity_volume:10.4f} Ang^3"
    )
    print(f"{'Total Molecular Surface Area:':<45} {total_SA:10.4f} Ang^2")
    print("-" * 70)
    print(
        f"[gaussian_overlap] overlap records: {num_records} total; "
        f"showing up to {min(num_records, max_print_records)}"
    )
    print("=" * 70)

    # ---- Overlap record table (rich format) ----
    n_show = min(num_records, max_print_records)

    header = (
        "| {:<5} | {:<5} | {:<10} | {:<10} | {:<10} | {:<10} | " "{:<20} | {:<30}"
    ).format(
        "Idx",
        "Order",
        "Volume",
        "Rad.Deriv.",
        "Alpha",
        "P_Factor",
        "R_Center (x, y, z)",
        "Atoms (Indices)",
    )

    print("\nInternal Overlap Records:")
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    for i in range(n_show):
        rec = overlap_records[i]

        order = int(rec[OVR_ORDER_N])
        volume = rec[OVR_VOLUME_VN]
        rad_deriv = rec[OVR_AREA_DERIV_AN]
        alpha = rec[OVR_ALPHA_SUM]
        P_factor = rec[OVR_PRODUCT_P]

        r_center = rec[OVR_CENTER_COORDS : OVR_CENTER_COORDS + 3]

        # Only first `order` atom indices are meaningful.
        atom_indices = rec[OVR_ATOM_IDS : OVR_ATOM_IDS + order].astype(np.int32)
        atoms_str = ", ".join(map(str, atom_indices))

        print(
            "| {:5d} | {:5d} | {:10.4f} | {:10.4f} | {:10.4f} | {:10.4f} | "
            "({:6.2f}, {:6.2f}, {:6.2f}) | {:<30}".format(
                i,
                order,
                volume,
                rad_deriv,
                alpha,
                P_factor,
                r_center[0],
                r_center[1],
                r_center[2],
                atoms_str,
            )
        )

    print("-" * len(header))
    if num_records > max_print_records:
        print(f"... and {num_records - max_print_records} more records.")


@njit(nogil=True, boundscheck=False, cache=True)
def _compute_lambda_matrix(atoms_props, adjacency_map, sentinel_adj):
    """
    Computes the Lambda matrix.
    atom_props_list: [x, y, z, Radius, Alpha, P, Rvdw, gamma]
    """
    N = atoms_props.shape[0]
    lambda_matrix = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        if i != sentinel_adj and int(atoms_props[i, ATOMPROP_Z_NUM]) != 1:
            for j in adjacency_map[i]:
                if j != sentinel_adj and int(atoms_props[j, ATOMPROP_Z_NUM]) != 1:
                    dx = atoms_props[i, ATOMPROP_X] - atoms_props[j, ATOMPROP_X]
                    dy = atoms_props[i, ATOMPROP_Y] - atoms_props[j, ATOMPROP_Y]
                    dz = atoms_props[i, ATOMPROP_Z] - atoms_props[j, ATOMPROP_Z]

                    dist2 = dx * dx + dy * dy + dz * dz
                    alpha_i = atoms_props[i, ATOMPROP_ALPHA]
                    alpha_j = atoms_props[j, ATOMPROP_ALPHA]

                    if dist2 > 1.0e-6:
                        Lambda_term = alpha_i * alpha_j * dist2
                        lambda_matrix[i, j] = Lambda_term
                        lambda_matrix[j, i] = Lambda_term

    return lambda_matrix


@njit(nogil=True, boundscheck=False, cache=True)
def _compute_lambda_csr_upper(
    atoms_props,
    csr_row_ptr,
    csr_col_idx,
):
    """
    Build a sparse, *upper-triangular* CSR-like representation of Lambda,
    using an existing atom adjacency CSR as input.

    Args:
        atoms_props : 2D array [num_atoms, ATOMPROP_*]
        csr_row_ptr : 1D int64 [num_atoms+1]
        csr_col_idx : 1D int32 [nnz]  (neighbors, symmetric adjacency)

    Returns:
        row_start      : int64 [num_atoms+1]
        neighbors_flat : int32 [M]    (j indices, j > i)
        lambda_flat    : float64 [M]  (Lambda(i, j) values)

    Note:
        Because csr_col_idx rows are already sorted and we iterate
        in that order, while only filtering (j > i, non-H, dist2 > eps),
        the resulting neighbors_flat is already sorted within each row.
        No extra sort step is required.
    """
    num_atoms = atoms_props.shape[0]

    # ---------- 1st pass: count entries per row (only j > i) ----------
    row_start = np.empty(num_atoms + 1, dtype=np.int64)
    row_start[0] = 0

    for i in range(num_atoms):
        count = 0

        # skip hydrogens for i
        if int(atoms_props[i, ATOMPROP_Z_NUM]) != 1:
            xi = atoms_props[i, ATOMPROP_X]
            yi = atoms_props[i, ATOMPROP_Y]
            zi = atoms_props[i, ATOMPROP_Z]
            alpha_i = atoms_props[i, ATOMPROP_ALPHA]

            start = csr_row_ptr[i]
            end = csr_row_ptr[i + 1]

            for k in range(start, end):
                j = csr_col_idx[k]

                # upper triangle only: keep only (i, j) with j > i
                if j <= i:
                    continue

                # skip hydrogens for j
                if int(atoms_props[j, ATOMPROP_Z_NUM]) == 1:
                    continue

                dx = xi - atoms_props[j, ATOMPROP_X]
                dy = yi - atoms_props[j, ATOMPROP_Y]
                dz = zi - atoms_props[j, ATOMPROP_Z]
                dist2 = dx * dx + dy * dy + dz * dz

                if dist2 > 1.0e-6:
                    count += 1

        row_start[i + 1] = row_start[i] + count

    M = row_start[num_atoms]

    # ---------- allocate flat arrays ----------
    neighbors_flat = np.empty(M, dtype=np.int32)
    lambda_flat = np.empty(M, dtype=np.float64)

    # ---------- 2nd pass: fill arrays ----------
    for i in range(num_atoms):
        start = row_start[i]
        idx = start

        if int(atoms_props[i, ATOMPROP_Z_NUM]) != 1:
            xi = atoms_props[i, ATOMPROP_X]
            yi = atoms_props[i, ATOMPROP_Y]
            zi = atoms_props[i, ATOMPROP_Z]
            alpha_i = atoms_props[i, ATOMPROP_ALPHA]

            adj_start = csr_row_ptr[i]
            adj_end = csr_row_ptr[i + 1]

            for k in range(adj_start, adj_end):
                j = csr_col_idx[k]

                if j <= i:
                    continue

                if int(atoms_props[j, ATOMPROP_Z_NUM]) == 1:
                    continue

                dx = xi - atoms_props[j, ATOMPROP_X]
                dy = yi - atoms_props[j, ATOMPROP_Y]
                dz = zi - atoms_props[j, ATOMPROP_Z]
                dist2 = dx * dx + dy * dy + dz * dz

                if dist2 > 1.0e-6:
                    alpha_j = atoms_props[j, ATOMPROP_ALPHA]
                    Lambda_term = alpha_i * alpha_j * dist2

                    neighbors_flat[idx] = j
                    lambda_flat[idx] = Lambda_term
                    idx += 1

        # if no valid neighbors: row_start[i] == row_start[i+1] and idx == start

    # rows of neighbors_flat are already sorted by j because csr_col_idx rows
    # are sorted and we only filter (no reordering), so no extra sort here.

    return row_start, neighbors_flat, lambda_flat


@njit(nogil=True, boundscheck=False, cache=True)
def _lambda_lookup_symmetric(i, j, row_start, neighbors_flat, lambda_flat):
    """
    Symmetric Λ(i, j) lookup from upper-triangular CSR.

    We store only entries with i < j.
    For i > j, we flip to (j, i) and search that row.
    """

    if i == j:
        # diagonal is zero in your current formulation
        return 0.0

    if i > j:
        # use canonical order (imin, imax)
        tmp = i
        i = j
        j = tmp

    # binary search j in row i
    start = row_start[i]
    end = row_start[i + 1] - 1  # inclusive

    lo = start
    hi = end
    while lo <= hi:
        mid = (lo + hi) // 2
        jm = neighbors_flat[mid]

        if jm == j:
            return lambda_flat[mid]
        elif jm < j:
            lo = mid + 1
        else:
            hi = mid - 1

    # if (i, j) wasn't in adjacency, Λ(i, j) = 0
    return 0.0


@njit(nogil=True, boundscheck=False, cache=True)
def _calculate_intermediate_props_kernel(
    parent_r,
    child_r,
    parent_alpha,
    child_alpha,
    parent_sum_lambda,
    lambda_parent_child,
    parent_order,
    new_P_factor,
):
    """
    Calculates essential properties for the next iteration.
    Returns: [Order, Alpha, P_factor, SumLambda, rX, rY, rZ] (7 floats)
    """
    alpha_new = parent_alpha + child_alpha
    r_new = (parent_r * parent_alpha + child_r * child_alpha) / alpha_new
    sum_lambda_new = parent_sum_lambda + lambda_parent_child
    new_order = parent_order + 1

    result = np.zeros(7, dtype=np.float64)
    result[0] = float(new_order)
    result[1] = alpha_new
    result[2] = new_P_factor  # Corrected P_new (product of all Ps)
    result[3] = sum_lambda_new
    result[4:7] = r_new

    return result


@njit(nogil=True, boundscheck=False, cache=True)
def _calculate_full_overlap_kernel(intermediate_props_array):
    """
    Calculates the full Volume using intermediate properties from Pass 1.
    Input: [Order, Alpha, P_factor (P_N), SumLambda, rX, rY, rZ] (7 floats)
    Returns: [Volume V_N, Alpha_N, P_N, SumLambda_N, rX, rY, rZ] (7 floats)
    """
    order = intermediate_props_array[0]
    alpha_new = intermediate_props_array[1]
    P_new = intermediate_props_array[2]
    sum_lambda_new = intermediate_props_array[3]
    r_new = intermediate_props_array[4:7]

    # Volume calculation
    volume_factor = np.exp(-sum_lambda_new / alpha_new) * ((PI / alpha_new) ** 1.5)
    volume = P_new * volume_factor

    # Return structure for the main properties calculated in Pass 2
    result = np.zeros(7, dtype=np.float64)
    result[0] = volume
    result[1] = alpha_new
    result[2] = P_new
    result[3] = sum_lambda_new
    result[4:7] = r_new

    return result


@njit(nogil=True, boundscheck=False, cache=True)
def _calculate_radii_derivative_term(volume_new, alpha_new, radius_eff_i, r_i, r_new):
    """
    Compute the radii-derivative contribution a_i for atom i
    in an N-body Gaussian overlap.

    Parameters
    ----------
    volume_new : float
        Overlap volume V_N of the N-body cluster.
    alpha_new : float
        Combined alpha of the N-body Gaussian cluster.
    radius_eff_i : float
        Effective radius of atom i (Rvdw + radius_offset).
    r_i : array-like (3,)
        Position of atom i.
    r_new : array-like (3,)
        Centroid of the N-body overlap.

    Returns
    -------
    float
        a_i, the radii-derivative term for atom i.
    """
    if volume_new <= MIN_VOLUME:
        return 0.0

    # squared distance between atom i and the cluster centroid
    dx = r_i[0] - r_new[0]
    dy = r_i[1] - r_new[1]
    dz = r_i[2] - r_new[2]
    r_iter_sq = dx * dx + dy * dy + dz * dz

    # helper term from analytic Gaussian derivative
    term = (1.5 / alpha_new) + r_iter_sq

    # compute a_i
    a_i = 2.0 * K_CONSTANT * volume_new * term / (radius_eff_i**3)
    return a_i


@njit(nogil=True, boundscheck=False, cache=True)
def _find_common_neighbors(
    parent_indices,  # 1D int32
    last_atom_idx,  # int32
    csr_row_ptr,  # 1D int64, shape (N+1)
    csr_col_idx,  # 1D int32, shape (nnz)
):
    """
    Finds atoms 'c' that are neighbors of ALL atoms in parent_indices
    AND c > last_atom_idx, using CSR adjacency.

    Returns:
        1D int32 array of common neighbor indices (sorted ascending).
    """
    num_parents = parent_indices.shape[0]

    # neighbors of the first parent (we'll scan these as candidates)
    i0 = parent_indices[0]
    start0 = csr_row_ptr[i0]
    end0 = csr_row_ptr[i0 + 1]
    deg0 = end0 - start0

    # temp buffer: upper bound on matches = deg0
    tmp = np.empty(deg0, dtype=np.int32)
    match_count = 0

    for k in range(start0, end0):
        c = csr_col_idx[k]

        # enforce c > last_atom_idx
        if c <= last_atom_idx:
            continue

        is_common = True

        # check membership of c in neighbors of all other parents
        for p in range(1, num_parents):
            ip = parent_indices[p]
            s = csr_row_ptr[ip]
            e = csr_row_ptr[ip + 1]

            # binary search for c in row ip (rows are sorted)
            lo = s
            hi = e
            found = False
            while lo < hi:
                mid = (lo + hi) // 2
                v = csr_col_idx[mid]
                if v == c:
                    found = True
                    break
                elif v < c:
                    lo = mid + 1
                else:
                    hi = mid

            if not found:
                is_common = False
                break

        if is_common:
            tmp[match_count] = c
            match_count += 1

    # pack exact-size result
    result = np.empty(match_count, dtype=np.int32)
    for t in range(match_count):
        result[t] = tmp[t]

    return result


@njit(parallel=True, nogil=True, boundscheck=False, cache=True)
def _fill_atom_props_kernel(atom_data, atom_props, radius_offset, P_const, K_constant):
    """
    Populate atom_props from atom_data and runtime parameters.

    atom_data:  (num_atoms, LEN_ATOMFIELDS)
    atom_props:(num_atoms, N_ATOMPROPS) preallocated
               [x, y, z, radius_eff, alpha, P, radius_vdw, gamma, Z_num]

    radius_offset: scalar
        radius_eff = radius_vdw + radius_offset

    P_const: scalar
        FOUR_THIRD_PI * K_OVER_PI_POW_3_2

    K_constant: scalar
        K in alpha = K / radius_eff^2
    """
    num_atoms = atom_data.shape[0]

    for i in prange(num_atoms):
        x = atom_data[i, ATOMFIELD_X]
        y = atom_data[i, ATOMFIELD_Y]
        z = atom_data[i, ATOMFIELD_Z]

        radius_vdw = atom_data[i, ATOMFIELD_RADIUS]
        gamma = atom_data[i, ATOMFIELD_LJ_GAMMA]
        z_num = atom_data[i, ATOMFIELD_ATOMIC_NUMBER]

        radius_eff = radius_vdw + radius_offset
        if radius_eff <= 0.0:
            radius_eff = 1e-6  # safety guard

        alpha = K_constant / (radius_eff * radius_eff)

        atom_props[i, ATOMPROP_X] = x
        atom_props[i, ATOMPROP_Y] = y
        atom_props[i, ATOMPROP_Z] = z
        atom_props[i, ATOMPROP_Z_NUM] = z_num
        atom_props[i, ATOMPROP_GAMMA] = gamma
        atom_props[i, ATOMPROP_R_VDW] = radius_vdw
        atom_props[i, ATOMPROP_R_EFF] = radius_eff
        atom_props[i, ATOMPROP_ALPHA] = alpha
        atom_props[i, ATOMPROP_P] = P_const


@njit(nogil=True, boundscheck=False, cache=True)
def _sum_lambda_over_parents(
    parent_indices, child_idx, row_start, neighbors_flat, lambda_flat
):
    total = 0.0
    for p in parent_indices:
        total += _lambda_lookup_symmetric(
            p, child_idx, row_start, neighbors_flat, lambda_flat
        )
    return total


@njit(nogil=True, boundscheck=False, cache=True)
def _exclusive_scan(counts):
    n = counts.shape[0]
    offsets = np.empty(n + 1, dtype=np.int64)
    offsets[0] = 0
    for i in range(n):
        offsets[i + 1] = offsets[i] + counts[i]
    return offsets


@njit(parallel=True, nogil=True, boundscheck=False, cache=True)
def _fill_order1_tuples(
    atom_props,
    offsets,  # len num_atoms+1
    overlap_atoms,  # (N1, MAX_ORDER) int32
    intermediate_props,
):  # (N1, 7) float64
    num_atoms = atom_props.shape[0]

    for i in prange(num_atoms):
        out_idx = offsets[i]
        if out_idx == offsets[i + 1]:
            continue  # no tuple for this atom

        atomic_number = int(atom_props[i, ATOMPROP_Z_NUM])
        if atomic_number == 1:
            continue

        alpha = atom_props[i, ATOMPROP_ALPHA]
        p = atom_props[i, ATOMPROP_P]

        volume = p * ((PI / alpha) ** 1.5)
        if volume <= MIN_VOLUME:
            continue

        # one tuple at out_idx
        rec = intermediate_props[out_idx]
        rec[0] = 1.0  # order
        rec[1] = alpha
        rec[2] = p
        rec[3] = 0.0  # sum_lambda
        rec[4] = atom_props[i, ATOMPROP_X]
        rec[5] = atom_props[i, ATOMPROP_Y]
        rec[6] = atom_props[i, ATOMPROP_Z]

        for k in range(MAX_ORDER):
            overlap_atoms[out_idx, k] = -1
        overlap_atoms[out_idx, 0] = i


@njit(parallel=True, nogil=True, boundscheck=False, cache=True)
def _count_order1_tuples(atom_props):
    num_atoms = atom_props.shape[0]
    counts = np.zeros(num_atoms, dtype=np.int32)

    for i in prange(num_atoms):
        atomic_number = int(atom_props[i, ATOMPROP_Z_NUM])
        if atomic_number != 1:
            alpha_i = atom_props[i, ATOMPROP_ALPHA]
            p_i = atom_props[i, ATOMPROP_P]
            volume = p_i * ((PI / alpha_i) ** 1.5)
            if volume > MIN_VOLUME:
                counts[i] = 1

    return counts


def _build_order1_tuples(atom_props):
    counts = _count_order1_tuples(atom_props)
    offsets = _exclusive_scan(counts)
    N1 = offsets[-1]

    overlap_atoms_1 = np.full((N1, MAX_ORDER), -1, dtype=np.int32)
    intermediate_1 = np.zeros((N1, 7), dtype=np.float64)

    _fill_order1_tuples(
        atom_props,
        offsets,
        overlap_atoms_1,
        intermediate_1,
    )
    return overlap_atoms_1, intermediate_1


@njit(parallel=True, nogil=True, boundscheck=False, cache=True)
def _fill_children_for_parents(
    parent_atoms,
    parent_props,
    atom_props,
    adj_list,
    neighbor_counts,
    row_start,
    neighbors_flat,
    lambda_flat,
    sentinel_adj,
    offsets,  # len Nk+1
    child_atoms,  # (Nk1, MAX_ORDER) int32
    child_props,
):  # (Nk1, 7) float64
    Nk = parent_atoms.shape[0]

    for idx in prange(Nk):
        parent_order = int(parent_props[idx, 0])
        if parent_order >= MAX_ORDER:
            continue

        parent_alpha = parent_props[idx, 1]
        parent_P = parent_props[idx, 2]
        parent_sum_lambda = parent_props[idx, 3]
        parent_r_x = parent_props[idx, 4]
        parent_r_y = parent_props[idx, 5]
        parent_r_z = parent_props[idx, 6]

        parent_indices = np.empty(parent_order, dtype=np.int32)
        for s in range(parent_order):
            parent_indices[s] = parent_atoms[idx, s]
        last_atom_idx = parent_indices[parent_order - 1]

        out_pos = offsets[idx]

        if parent_order == 1:
            i0 = parent_indices[0]
            n0 = neighbor_counts[i0]
            for pos in range(n0):
                child_atom_idx = adj_list[i0, pos]
                if child_atom_idx == sentinel_adj:
                    break
                if child_atom_idx <= last_atom_idx:
                    continue
                if int(atom_props[child_atom_idx, ATOMPROP_Z_NUM]) == 1:
                    continue

                child_r_x = atom_props[child_atom_idx, ATOMPROP_X]
                child_r_y = atom_props[child_atom_idx, ATOMPROP_Y]
                child_r_z = atom_props[child_atom_idx, ATOMPROP_Z]
                child_alpha = atom_props[child_atom_idx, ATOMPROP_ALPHA]
                child_P = atom_props[child_atom_idx, ATOMPROP_P]

                lambda_pc = _sum_lambda_over_parents(
                    parent_indices,
                    child_atom_idx,
                    row_start,
                    neighbors_flat,
                    lambda_flat,
                )
                new_P = parent_P * child_P

                parent_r = np.array([parent_r_x, parent_r_y, parent_r_z])
                child_r = np.array([child_r_x, child_r_y, child_r_z])

                new_rec = _calculate_intermediate_props_kernel(
                    parent_r,
                    child_r,
                    parent_alpha,
                    child_alpha,
                    parent_sum_lambda,
                    lambda_pc,
                    parent_order,
                    new_P,
                )

                alpha_new = new_rec[1]
                sum_lambda_new = new_rec[3]
                volume_factor = np.exp(-sum_lambda_new / alpha_new) * (
                    (PI / alpha_new) ** 1.5
                )
                new_volume = new_P * volume_factor
                if new_volume <= MIN_VOLUME:
                    continue

                # write tuple
                for t in range(7):
                    child_props[out_pos, t] = new_rec[t]

                for s in range(MAX_ORDER):
                    child_atoms[out_pos, s] = -1
                child_atoms[out_pos, 0] = i0
                child_atoms[out_pos, 1] = child_atom_idx

                out_pos += 1

        else:
            common_neighbors = _find_common_neighbors(
                parent_indices, last_atom_idx, adj_list, neighbor_counts
            )
            n_common = common_neighbors.size
            for ci in range(n_common):
                child_atom_idx = common_neighbors[ci]
                if int(atom_props[child_atom_idx, ATOMPROP_Z_NUM]) == 1:
                    continue

                child_r_x = atom_props[child_atom_idx, ATOMPROP_X]
                child_r_y = atom_props[child_atom_idx, ATOMPROP_Y]
                child_r_z = atom_props[child_atom_idx, ATOMPROP_Z]
                child_alpha = atom_props[child_atom_idx, ATOMPROP_ALPHA]
                child_P = atom_props[child_atom_idx, ATOMPROP_P]

                lambda_pc = _sum_lambda_over_parents(
                    parent_indices,
                    child_atom_idx,
                    row_start,
                    neighbors_flat,
                    lambda_flat,
                )
                new_P = parent_P * child_P

                parent_r = np.array([parent_r_x, parent_r_y, parent_r_z])
                child_r = np.array([child_r_x, child_r_y, child_r_z])

                new_rec = _calculate_intermediate_props_kernel(
                    parent_r,
                    child_r,
                    parent_alpha,
                    child_alpha,
                    parent_sum_lambda,
                    lambda_pc,
                    parent_order,
                    new_P,
                )

                alpha_new = new_rec[1]
                sum_lambda_new = new_rec[3]
                volume_factor = np.exp(-sum_lambda_new / alpha_new) * (
                    (PI / alpha_new) ** 1.5
                )
                new_volume = new_P * volume_factor
                if new_volume <= MIN_VOLUME:
                    continue

                new_order = parent_order + 1
                if new_order > MAX_ORDER:
                    continue

                for t in range(7):
                    child_props[out_pos, t] = new_rec[t]

                for s in range(MAX_ORDER):
                    child_atoms[out_pos, s] = -1
                for s in range(parent_order):
                    child_atoms[out_pos, s] = parent_indices[s]
                child_atoms[out_pos, new_order - 1] = child_atom_idx

                out_pos += 1


@njit(inline="always", nogil=True)
def _try_add_child(
    parent_indices,
    parent_order,
    parent_alpha,
    parent_P,
    parent_sum_lambda,
    parent_r_x,
    parent_r_y,
    parent_r_z,
    child_atom_idx,
    atom_props,
    row_start,
    neighbors_flat,
    lambda_flat,
    child_atoms,
    child_props,
    out_pos,
):
    """
    If child_atom_idx is a valid child for the given parent cluster, compute its
    intermediate props, apply volume filter, write into child_atoms/child_props
    at out_pos, and return 1.

    Otherwise, return 0 and DO NOT modify outputs.
    """

    # skip hydrogens
    if int(atom_props[child_atom_idx, ATOMPROP_Z_NUM]) == 1:
        return 0

    child_r_x = atom_props[child_atom_idx, ATOMPROP_X]
    child_r_y = atom_props[child_atom_idx, ATOMPROP_Y]
    child_r_z = atom_props[child_atom_idx, ATOMPROP_Z]
    child_alpha = atom_props[child_atom_idx, ATOMPROP_ALPHA]
    child_P = atom_props[child_atom_idx, ATOMPROP_P]

    lambda_pc = _sum_lambda_over_parents(
        parent_indices,
        child_atom_idx,
        row_start,
        neighbors_flat,
        lambda_flat,
    )
    new_P = parent_P * child_P

    parent_r = np.array([parent_r_x, parent_r_y, parent_r_z])
    child_r = np.array([child_r_x, child_r_y, child_r_z])

    new_rec = _calculate_intermediate_props_kernel(
        parent_r,
        child_r,
        parent_alpha,
        child_alpha,
        parent_sum_lambda,
        lambda_pc,
        parent_order,
        new_P,
    )

    alpha_new = new_rec[1]
    sum_lambda_new = new_rec[3]
    volume_factor = np.exp(-sum_lambda_new / alpha_new) * ((PI / alpha_new) ** 1.5)
    new_volume = new_P * volume_factor

    if new_volume <= MIN_VOLUME:
        return 0

    new_order = parent_order + 1
    if new_order > MAX_ORDER:
        return 0

    # write child_props tuple (7 floats)
    for t in range(7):
        child_props[out_pos, t] = new_rec[t]

    # reset and fill child_atoms row
    for s in range(MAX_ORDER):
        child_atoms[out_pos, s] = -1

    if parent_order == 1:
        # parents: [i0], children: [i0, child_atom_idx]
        child_atoms[out_pos, 0] = parent_indices[0]
        child_atoms[out_pos, 1] = child_atom_idx
    else:
        # multi-parent: copy existing parent indices, append child
        for s in range(parent_order):
            child_atoms[out_pos, s] = parent_indices[s]
        child_atoms[out_pos, new_order - 1] = child_atom_idx

    return 1


@njit(parallel=True, nogil=True, boundscheck=False, cache=True)
def _fill_children_for_parents_csr(
    parent_atoms,
    parent_props,
    atom_props,
    csr_row_ptr,
    csr_col_idx,
    row_start,
    neighbors_flat,
    lambda_flat,
    offsets,  # len Nk+1
    child_atoms,  # (Nk1, MAX_ORDER) int32
    child_props,  # (Nk1, 7) float64
):
    Nk = parent_atoms.shape[0]

    for idx in prange(Nk):
        parent_order = int(parent_props[idx, 0])
        if parent_order >= MAX_ORDER:
            continue

        parent_alpha = parent_props[idx, 1]
        parent_P = parent_props[idx, 2]
        parent_sum_lambda = parent_props[idx, 3]
        parent_r_x = parent_props[idx, 4]
        parent_r_y = parent_props[idx, 5]
        parent_r_z = parent_props[idx, 6]

        # gather parent_indices
        parent_indices = np.empty(parent_order, dtype=np.int32)
        for s in range(parent_order):
            parent_indices[s] = parent_atoms[idx, s]
        last_atom_idx = parent_indices[parent_order - 1]

        out_pos = offsets[idx]

        if parent_order == 1:
            # single-parent case: use CSR neighbors of i0
            i0 = parent_indices[0]
            row_s = csr_row_ptr[i0]
            row_e = csr_row_ptr[i0 + 1]

            for pos in range(row_s, row_e):
                child_atom_idx = csr_col_idx[pos]

                # enforce ordering: only children with index > last_atom_idx
                if child_atom_idx <= last_atom_idx:
                    continue

                added = _try_add_child(
                    parent_indices,
                    parent_order,
                    parent_alpha,
                    parent_P,
                    parent_sum_lambda,
                    parent_r_x,
                    parent_r_y,
                    parent_r_z,
                    child_atom_idx,
                    atom_props,
                    row_start,
                    neighbors_flat,
                    lambda_flat,
                    child_atoms,
                    child_props,
                    out_pos,
                )
                if added == 1:
                    out_pos += 1

        else:
            # multi-parent case: intersect neighbors via CSR
            common_neighbors = _find_common_neighbors(
                parent_indices,
                last_atom_idx,
                csr_row_ptr,
                csr_col_idx,
            )
            n_common = common_neighbors.size

            for ci in range(n_common):
                child_atom_idx = common_neighbors[ci]

                added = _try_add_child(
                    parent_indices,
                    parent_order,
                    parent_alpha,
                    parent_P,
                    parent_sum_lambda,
                    parent_r_x,
                    parent_r_y,
                    parent_r_z,
                    child_atom_idx,
                    atom_props,
                    row_start,
                    neighbors_flat,
                    lambda_flat,
                    child_atoms,
                    child_props,
                    out_pos,
                )
                if added == 1:
                    out_pos += 1
        # offsets[idx+1] bounds how far out_pos can go; counts were precomputed


@njit(inline="always", nogil=True)
def _child_contribution(
    parent_indices,
    child_atom_idx,
    atom_props,
    parent_alpha,
    parent_P,
    parent_sum_lambda,
    parent_order,
    parent_r_x,
    parent_r_y,
    parent_r_z,
    row_start,
    neighbors_flat,
    lambda_flat,
):
    """
    Return 1 if child_atom_idx is a valid child for the given parent cluster,
    otherwise 0.

    This encapsulates the common logic used in both the parent_order == 1 and
    parent_order > 1 branches.
    """

    # skip hydrogens
    if int(atom_props[child_atom_idx, ATOMPROP_Z_NUM]) == 1:
        return 0

    child_r_x = atom_props[child_atom_idx, ATOMPROP_X]
    child_r_y = atom_props[child_atom_idx, ATOMPROP_Y]
    child_r_z = atom_props[child_atom_idx, ATOMPROP_Z]
    child_alpha = atom_props[child_atom_idx, ATOMPROP_ALPHA]
    child_P = atom_props[child_atom_idx, ATOMPROP_P]

    lambda_pc = _sum_lambda_over_parents(
        parent_indices,
        child_atom_idx,
        row_start,
        neighbors_flat,
        lambda_flat,
    )
    new_P = parent_P * child_P

    parent_r = np.array([parent_r_x, parent_r_y, parent_r_z])
    child_r = np.array([child_r_x, child_r_y, child_r_z])

    new_rec = _calculate_intermediate_props_kernel(
        parent_r,
        child_r,
        parent_alpha,
        child_alpha,
        parent_sum_lambda,
        lambda_pc,
        parent_order,
        new_P,
    )

    alpha_new = new_rec[1]
    sum_lambda_new = new_rec[3]
    volume_factor = np.exp(-sum_lambda_new / alpha_new) * ((PI / alpha_new) ** 1.5)
    new_volume = new_P * volume_factor

    if new_volume <= MIN_VOLUME:
        return 0

    return 1


@njit(parallel=True, nogil=True, boundscheck=False, cache=True)
def _count_children_for_parents(
    parent_atoms,
    parent_props,
    atom_props,
    csr_row_ptr,
    csr_col_idx,
    row_start,
    neighbors_flat,
    lambda_flat,
):
    num_parent_atoms = parent_atoms.shape[0]
    counts = np.zeros(num_parent_atoms, dtype=int64)

    for idx in prange(num_parent_atoms):
        parent_order = int(parent_props[idx, 0])
        if parent_order >= MAX_ORDER:
            continue

        parent_alpha = parent_props[idx, 1]
        parent_P = parent_props[idx, 2]
        parent_sum_lambda = parent_props[idx, 3]
        parent_r_x = parent_props[idx, 4]
        parent_r_y = parent_props[idx, 5]
        parent_r_z = parent_props[idx, 6]

        # gather parent_indices
        parent_indices = np.empty(parent_order, dtype=np.int32)
        for s in range(parent_order):
            parent_indices[s] = parent_atoms[idx, s]
        last_atom_idx = parent_indices[parent_order - 1]

        c_children = 0

        if parent_order == 1:
            # single-parent case: scan neighbors of i0
            i0 = parent_indices[0]
            row_s = csr_row_ptr[i0]
            row_e = csr_row_ptr[i0 + 1]

            for pos in range(row_s, row_e):
                child_atom_idx = csr_col_idx[pos]

                # enforce ordering: only children with index > last_atom_idx
                if child_atom_idx <= last_atom_idx:
                    continue

                c_children += _child_contribution(
                    parent_indices,
                    child_atom_idx,
                    atom_props,
                    parent_alpha,
                    parent_P,
                    parent_sum_lambda,
                    parent_order,
                    parent_r_x,
                    parent_r_y,
                    parent_r_z,
                    row_start,
                    neighbors_flat,
                    lambda_flat,
                )

        else:
            # multi-parent case: intersect neighbors via CSR
            common_neighbors = _find_common_neighbors(
                parent_indices,
                last_atom_idx,
                csr_row_ptr,
                csr_col_idx,
            )
            n_common = common_neighbors.size

            for ci in range(n_common):
                child_atom_idx = common_neighbors[ci]

                c_children += _child_contribution(
                    parent_indices,
                    child_atom_idx,
                    atom_props,
                    parent_alpha,
                    parent_P,
                    parent_sum_lambda,
                    parent_order,
                    parent_r_x,
                    parent_r_y,
                    parent_r_z,
                    row_start,
                    neighbors_flat,
                    lambda_flat,
                )

        counts[idx] = c_children

    return counts


def _expand_to_next_order(
    parent_atoms,
    parent_props,
    atom_props,
    atom_adjacency_crs,
    neighbor_counts,
    row_start,
    neighbors_flat,
    lambda_flat,
    sentinel_adj,
):
    counts = _count_children_for_parents(
        parent_atoms,
        parent_props,
        atom_props,
        atom_adjacency_crs.row_ptr,
        atom_adjacency_crs.col_idx,
        row_start,
        neighbors_flat,
        lambda_flat,
    )
    offsets = _exclusive_scan(counts)
    Nk1 = offsets[-1]
    if Nk1 == 0:
        return (
            np.zeros((0, MAX_ORDER), dtype=np.int32),
            np.zeros((0, 7), dtype=np.float64),
        )

    child_atoms = np.full((Nk1, MAX_ORDER), -1, dtype=np.int32)
    child_props = np.zeros((Nk1, 7), dtype=np.float64)

    _fill_children_for_parents_csr(
        parent_atoms,
        parent_props,
        atom_props,
        atom_adjacency_crs.row_ptr,
        atom_adjacency_crs.col_idx,
        row_start,
        neighbors_flat,
        lambda_flat,
        offsets,
        child_atoms,
        child_props,
    )
    return child_atoms, child_props


def _run_pass1_all_orders(
    atom_props,
    atom_adjacency_csr,
    row_start,
    neighbors_flat,
    lambda_flat,
    sentinel_adj,
):
    # Note: always use adj_list for neighbor_count, as csr is only upper-trianlgular
    # neighbor_counts_csr = build_neighbor_counts_from_csr(row_start)
    # neighbor_counts = _build_neighbor_counts_from_adj(atom_adjacency_csr, sentinel_adj)
    neighbor_counts = atom_adjacency_csr.degree

    # print("neighbor_counts_csr:", neighbor_counts_csr, "neighbor_counts:", neighbor_counts)

    # order 1
    atoms_k, props_k = _build_order1_tuples(atom_props)

    # collect all tuples across orders
    all_atoms = [atoms_k]
    all_props = [props_k]

    # expand order by order
    for order in range(1, MAX_ORDER):
        if atoms_k.shape[0] == 0:
            break

        atoms_k1, props_k1 = _expand_to_next_order(
            atoms_k,
            props_k,
            atom_props,
            atom_adjacency_csr,
            neighbor_counts,
            row_start,
            neighbors_flat,
            lambda_flat,
            sentinel_adj,
        )

        if atoms_k1.shape[0] == 0:
            break

        all_atoms.append(atoms_k1)
        all_props.append(props_k1)

        atoms_k = atoms_k1
        props_k = props_k1

    # concatenate all orders into final arrays
    final_overlap_atoms = (
        np.concatenate(all_atoms, axis=0).astype(np.float64)
        if all_atoms
        else np.zeros((0, MAX_ORDER), dtype=np.float64)
    )
    final_intermediate_props = (
        np.concatenate(all_props, axis=0)
        if all_props
        else np.zeros((0, 7), dtype=np.float64)
    )
    return final_overlap_atoms, final_intermediate_props


@njit(nogil=True, boundscheck=False, cache=True)
def _build_neighbor_counts_from_adj(adj_list, sentinel_adj):
    """
    Count valid neighbors per atom from adj_list, stopping at sentinel.

    This MUST match the semantics of the original Python code:
    for i in range(adj_list.shape[0]):
        count = 0
        for j in adj_list[i]:
            if j != sentinel: count += 1
            else: break
    """
    num_atoms, max_neighbors = adj_list.shape
    counts = np.zeros(num_atoms, dtype=np.int32)
    for i in prange(num_atoms):
        c = 0
        for k in range(max_neighbors):
            j = adj_list[i, k]
            if j == sentinel_adj:
                break
            c += 1
        counts[i] = c
    return counts


@njit(nogil=True, boundscheck=False, cache=True)
def _pass2_compute_overlaps_and_derivatives(
    final_intermediate_props,
    final_overlap_atoms,
    atom_props,
):
    """
    Pass 2:
    - For each overlap tuple (intermediate_rec, atoms_arr),
      compute full overlap properties, contribution to total volume,
      and per-atom radii derivatives A_i^(k).
    - Fill final_overlap_records (homogeneous array).
    - Accumulate radii_derivative_per_atom[atom_idx, order].

    Returns:
        total_V,
        final_overlap_records,
        radii_derivative_per_atom
    """

    num_ovr_tuples = final_intermediate_props.shape[0]
    num_atoms = atom_props.shape[0]

    # Allocate outputs
    final_overlap_records = np.full(
        (num_ovr_tuples, NUM_FIELDS_OVERLAP), -1.0, dtype=np.float64
    )
    radii_derivative_per_atom = np.zeros((num_atoms, MAX_ORDER + 1), dtype=np.float64)

    total_V = 0.0

    for t in range(num_ovr_tuples):
        intermediate_rec = final_intermediate_props[t]
        atoms_arr = final_overlap_atoms[t]

        order = int(intermediate_rec[0])
        if order <= 0:
            continue

        # Full overlap: [V_N, alpha_N, P_N, sum_lambda_N, r_Nx, r_Ny, r_Nz, ...]
        overlap_props = _calculate_full_overlap_kernel(intermediate_rec)

        volume = overlap_props[0]
        alpha = overlap_props[1]
        P_N = overlap_props[2]
        sum_lambda = overlap_props[3]
        r_new_x = overlap_props[4]
        r_new_y = overlap_props[5]
        r_new_z = overlap_props[6]

        # Inclusion–exclusion contribution to total volume
        # (-1)^(order+1) * V_N
        if order % 2 == 1:
            # order odd: factor = +1 for 1, -1 for 2, ...
            factor = 1.0
        else:
            factor = -1.0
        total_V += factor * volume

        # Per-atom radii derivatives (A_i^(order))
        new_radii_deriv_total = 0.0

        # Extract integer indices for the atoms in this tuple
        for k in range(order):
            atom_idx = int(atoms_arr[k])
            if atom_idx < 0:
                break

            r_i_x = atom_props[atom_idx, ATOMPROP_X]
            r_i_y = atom_props[atom_idx, ATOMPROP_Y]
            r_i_z = atom_props[atom_idx, ATOMPROP_Z]
            radius_eff_i = atom_props[atom_idx, ATOMPROP_R_EFF]

            # Rebuild small vectors (Numba-friendly)
            r_i = np.array((r_i_x, r_i_y, r_i_z), dtype=np.float64)
            r_new = np.array((r_new_x, r_new_y, r_new_z), dtype=np.float64)

            A_iter = _calculate_radii_derivative_term(
                volume, alpha, radius_eff_i, r_i, r_new
            )
            radii_derivative_per_atom[atom_idx, order] += A_iter
            new_radii_deriv_total += A_iter

        # Store this tuple's record into final_overlap_records[t, :]
        rec = final_overlap_records[t]

        rec[OVR_ORDER_N] = float(order)
        rec[OVR_VOLUME_VN] = volume
        rec[OVR_AREA_DERIV_AN] = new_radii_deriv_total
        rec[OVR_ALPHA_SUM] = alpha
        rec[OVR_PRODUCT_P] = P_N
        rec[OVR_LAMBDA_SUM] = sum_lambda

        rec[OVR_CENTER_COORDS + 0] = r_new_x
        rec[OVR_CENTER_COORDS + 1] = r_new_y
        rec[OVR_CENTER_COORDS + 2] = r_new_z

        # Copy atom indices (as floats, to match your schema)
        # Initialize with -1
        for k in range(MAX_ORDER):
            rec[OVR_ATOM_IDS + k] = -1.0

        for k in range(order):
            rec[OVR_ATOM_IDS + k] = atoms_arr[k]

    return total_V, final_overlap_records, radii_derivative_per_atom


@njit(nogil=True, boundscheck=False, cache=True)
def _pass3_surface_and_energy(
    atom_props,
    radii_derivative_per_atom,
    temperature,
    probe_radius,
    sigmoid_g,
):
    """
    Final reduction:
      - per-atom surface areas (inclusion–exclusion)
      - SA filter
      - SEV volume correction
      - cavity SA and volume energies

    Returns:
        total_SA,
        total_corrV,
        energy_cavity_SA,
        energy_cavity_volume,
        V_SEV_corrected
    """

    num_atoms = atom_props.shape[0]

    # Precompute sign factors for SA:
    # sign_factors[k] = (-1)^(k+1)
    sign_factors = np.zeros(MAX_ORDER + 1, dtype=np.float64)
    sign_factors[0] = 0.0  # unused
    for k in range(1, MAX_ORDER + 1):
        if k % 2 == 1:
            sign_factors[k] = 1.0
        else:
            sign_factors[k] = -1.0

    total_SA = 0.0
    total_corrV = 0.0
    energy_cavity_SA = 0.0

    temperature_factor = temperature / 298.0

    for i in range(num_atoms):
        atomic_number = int(atom_props[i, ATOMPROP_Z_NUM])
        if atomic_number == 1:  # skip hydrogens
            continue

        # SA_i = sum_{k=1..MAX_ORDER} sign_factors[k] * A_i^(k)
        atomic_SA = 0.0
        for k in range(1, MAX_ORDER + 1):
            derv_radius_ik = radii_derivative_per_atom[i, k]
            if derv_radius_ik != 0.0:
                atomic_SA += sign_factors[k] * derv_radius_ik

        radius_i_vdw = atom_props[i, ATOMPROP_R_VDW]
        radius_i_eff = atom_props[i, ATOMPROP_R_EFF]
        gamma_i = atom_props[i, ATOMPROP_GAMMA]

        # SA Filtering
        rel_surf_area_i = 0.5 * (
            1.0 - np.cos(2.0 * np.arctan2(probe_radius, probe_radius + radius_i_eff))
        )
        # sigmoid argument: -atomic_SA + atomic_SA * rel_surf_area_i
        # = atomic_SA * (rel_surf_area_i - 1)
        arg = -atomic_SA + atomic_SA * rel_surf_area_i
        sa_multiplier = 1.0 / (1.0 + np.exp(sigmoid_g * arg))

        atomic_SA_filtered = atomic_SA * sa_multiplier
        total_SA += atomic_SA_filtered

        # Cavity SA energy
        energy_cavity_SA += atomic_SA_filtered * gamma_i

        # SEV volume correction term
        # corrV_term = A_filtered * R / 3 * (1 - (Rvdw/R)^3)
        if radius_i_eff != 0.0:
            ratio = radius_i_vdw / radius_i_eff
            corrV_term = (
                atomic_SA_filtered * radius_i_eff / 3.0 * (1.0 - ratio * ratio * ratio)
            )
            total_corrV += corrV_term

    energy_cavity_SA *= temperature_factor

    return total_SA, total_corrV, energy_cavity_SA


@njit(nogil=True, boundscheck=False, cache=True)
def _pass3_surface_and_volume(
    atom_props,
    radii_derivative_per_atom,
    probe_radius,
    sigmoid_g,
):
    """
    Final reduction (pure geometry):

      - per-atom filtered surface areas via inclusion–exclusion
      - SEV volume correction term

    Returns
    -------
    total_SA : float
        Total filtered surface area.
    total_corrV : float
        SEV correction term to be subtracted from total_V.
    per_atom_SA : (num_atoms,) float64
        Filtered per-atom surface areas.
    """
    num_atoms = atom_props.shape[0]

    # Precompute sign factors: sign_factors[k] = (-1)^(k+1)
    sign_factors = np.zeros(MAX_ORDER + 1, dtype=np.float64)
    sign_factors[0] = 0.0  # unused
    for k in range(1, MAX_ORDER + 1):
        sign_factors[k] = 1.0 if (k % 2 == 1) else -1.0

    total_SA = 0.0
    total_corrV = 0.0
    per_atom_SA = np.zeros(num_atoms, dtype=np.float64)

    for i in range(num_atoms):
        atomic_number = int(atom_props[i, ATOMPROP_Z_NUM])
        if atomic_number == 1:  # skip hydrogens
            continue

        # SA_i = sum_{k=1..MAX_ORDER} (-1)^(k+1) * A_i^(k)
        atomic_SA = 0.0
        for k in range(1, MAX_ORDER + 1):
            derv_radius_ik = radii_derivative_per_atom[i, k]
            if derv_radius_ik != 0.0:
                atomic_SA += sign_factors[k] * derv_radius_ik

        radius_i_vdw = atom_props[i, ATOMPROP_R_VDW]
        radius_i_eff = atom_props[i, ATOMPROP_R_EFF]

        # Geometric SA filtering
        rel_surf_area_i = 0.5 * (
            1.0
            - np.cos(
                2.0
                * np.arctan2(
                    probe_radius,
                    probe_radius + radius_i_eff,
                )
            )
        )
        arg = -atomic_SA + atomic_SA * rel_surf_area_i
        sa_multiplier = 1.0 / (1.0 + np.exp(sigmoid_g * arg))

        atomic_SA_filtered = atomic_SA * sa_multiplier
        per_atom_SA[i] = atomic_SA_filtered
        total_SA += atomic_SA_filtered

        # SEV volume correction: A_filtered * R_eff / 3 * (1 - (Rvdw/R_eff)^3)
        if radius_i_eff != 0.0:
            ratio = radius_i_vdw / radius_i_eff
            corrV_term = (
                atomic_SA_filtered * radius_i_eff / 3.0 * (1.0 - ratio * ratio * ratio)
            )
            total_corrV += corrV_term

    return total_SA, total_corrV, per_atom_SA


def compute_gaussian_overlap_surface_volume(
    atom_data,
    atom_adjacency_csr,
    probe_radius=1.4,
    radius_offset=0.0,
    sigmoid_g=5.0,
    sentinel_adj=-1,
):
    """
    Gaussian-overlap geometry: diffuse solute representation.

    Computes:
        total_volume  : raw SEV volume from Gaussian overlap expansion
        cavity_volume : SEV-corrected cavity volume (diffuse cavity)
        total_SA      : filtered surface area from smoothed indicator gradient
        per_atom_SA   : filtered per-atom surface areas

    Parameters
    ----------
    atom_data : ndarray (N, M)
        Per-atom geometry and radius information.
        Must contain at least:
            • van der Waals radius or Gaussian radius parameters
            • atomic positions (x, y, z)
        The exact column layout follows pyDelPhi's internal ATOMFIELD schema.

    atom_adjacency_csr : tuple or CSR-like structure
        Compressed-sparse-row adjacency describing all atom–atom overlaps.
        Consisting `(neighbors_flat, row_start, neighbor_counts)`.
        Only atoms listed as neighbors contribute to Gaussian overlap terms.

    probe_radius : float, optional (default: 1.4)
        Probe radius used to extend the effective Gaussian radius
        (analogous to SES/SAS probe inflation).
        Applied uniformly to all atoms.

    radius_offset : float, optional (default: 0.0)
        Additional user-specified offset added to each atom's base radius
        before Gaussian expansion. Useful for tuning effective solute size.

    sigmoid_g : float, optional (default: 5.0)
        Sharpness parameter controlling the smooth transition in
        the filtered surface-area indicator.
        Larger values → sharper boundary (approaches hard-sphere limit).

    sentinel_adj : int, optional (default: -1)
        Sentinel value marking the end of neighbor lists in the flattened
        adjacency array. Must match the internal adjacency builder.

    Notes
    -----
    • This routine performs geometry-only calculations.
    • No hard-sphere or "homogeneous solute" assumptions are made.
    • Nonpolar energetics (γ, γᵢ, p, temperature scaling) are handled separately.
    """
    num_atoms = atom_data.shape[0]

    # --- Data preparation ---
    atom_props = np.zeros((num_atoms, NUM_ATOMPROPS), dtype=np.float64)
    P_const = FOUR_THIRD_PI * K_OVER_PI_POW_3_2
    K_constant = K_CONSTANT

    _fill_atom_props_kernel(atom_data, atom_props, radius_offset, P_const, K_constant)

    row_start, neighbors_flat, lambda_flat = _compute_lambda_csr_upper(
        atom_props,
        atom_adjacency_csr.row_ptr,
        atom_adjacency_csr.col_idx,
    )

    # ----------------------------------------------------------------------
    # PASS 1: counting and tuple collection
    # ----------------------------------------------------------------------
    final_overlap_atoms, final_intermediate_props = _run_pass1_all_orders(
        atom_props,
        atom_adjacency_csr,
        row_start,
        neighbors_flat,
        lambda_flat,
        sentinel_adj,
    )
    num_tuples = final_overlap_atoms.shape[0]

    if VERBOSE > _VERBOSITY:
        print(
            f"--- Pass 1 Complete: Found {num_tuples} overlap tuples "
            f"(orders 1 to {MAX_ORDER}) ---"
        )

    # ----------------------------------------------------------------------
    # PASS 2: overlap integrals and derivatives
    # ----------------------------------------------------------------------
    total_volume, final_overlap_records, radii_derivative_per_atom = (
        _pass2_compute_overlaps_and_derivatives(
            final_intermediate_props,
            final_overlap_atoms,
            atom_props,
        )
    )

    # ----------------------------------------------------------------------
    # PASS 3: SA + SEV correction (geometry only)
    # ----------------------------------------------------------------------
    total_SA, total_corrV, per_atom_SA = _pass3_surface_and_volume(
        atom_props,
        radii_derivative_per_atom,
        probe_radius,
        sigmoid_g,
    )

    cavity_volume = total_volume - total_corrV

    # --- internal debug only, no overlap_records in return ---
    if VERBOSE >= _VERBOSITY:
        _debug_print_overlap_summary(
            total_volume=total_volume,
            total_SA=total_SA,
            cavity_volume=cavity_volume,
            overlap_records=final_overlap_records,
        )

    # Public, stable, geometry-only API:
    return total_volume, cavity_volume, total_SA, per_atom_SA
