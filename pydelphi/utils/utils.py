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
Utility functions for the pydelphi package, including numerical operations,
array manipulations, and Numba-optimized functions for performance.

Key functions:
    crd3d_grid_distance_factor: Calculates the distance factor for charge distribution.
    crd3d_grid_neighbors: Determines the 8 neighboring grid points and their distance factors.
    sort_2d_by_single_column: Sorts a 2D NumPy array based on a single column.
    is_contained_in_sorted: Checks if a key exists in a sorted array column using binary search.
    accumulate_sorted_by_key: Accumulates values in a pre-sorted array based on a key.
    generate_atomic_charge_grid_contributions: Generates charge contributions from atoms to grid points.
    format_and_count_grid_charges: Formats unique grid point contributions and counts charged points.
"""

import numpy as np

# Ensure Numba and necessary submodules are imported correctly
_numba_usable = False
try:
    from numba import jit, njit, prange, types, float64, int64

    _numba_usable = True
except ImportError:
    print("Warning: Numba package not found. Parallel execution will be disabled.")
    _numba_usable = False

    def _dummy_decorator(func):
        # This decorator does nothing, just returns the function itself.
        return func

    # njit becomes a lambda that returns the dummy decorator
    njit = lambda *args, **kwargs: _dummy_decorator
    # prange falls back to standard range
    prange = range
    NumbaPerformanceWarning = None

from pydelphi.config.global_runtime import nprint_cpu_if_verbose as nprint_cpu
from pydelphi.config.logging_config import (
    DEBUG,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

from pydelphi.constants import ConstDelPhiFloats
from pydelphi.constants import (
    ATOMFIELD_GRID_X,
    ATOMFIELD_GRID_END,
    ATOMFIELD_CHARGE,
)

from pydelphi.utils.prec.double import and_gt_scalar, and_lt_scalar

# GPCHRG: Constants for grid point charge fields
from pydelphi.constants import (
    GPCHRGFIELD_INDX_1D,  # Index of the 1D grid point index
    GPCHRGFIELD_CHARGE,  # Index of the charge at the grid point
    GPCHRGFIELD_INDX_X,  # Index of the x-coordinate of the grid point
    GPCHRGFIELD_INDX_Y,  # Index of the y-coordinate of the grid point
    GPCHRGFIELD_INDX_Z,  # Index of the z-coordinate of the grid point
    GRID_NEIGHBOR_OFFSETS,  # Constant for neighbor offsets
    NUM_DIMENSIONS,  # Constant for the number of dimensions
)

APPROX_ZERO = ConstDelPhiFloats.ApproxZero.value


@jit(
    [
        "float64(float64[:],float64[:],float64[:])",
        "float64(float64[:],float64[:],int64[:])",
        "float64(float64[:],int64[:],float64[:])",
        "float64(int64[:],float64[:],float64[:])",
        "float64(float64[:],int64[:],int64[:])",
        "float64(int64[:],int64[:],float64[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_grid_distance_factor(crd3d_grid_coords, neighbor_grid_index, neighbor_offset):
    """
    Calculates the distance factor for charge distribution to neighboring grid points.

    Args:
        crd3d_grid_coords (np.ndarray): The 3D coordinates of a point within the grid.
        neighbor_grid_index (np.ndarray): The integer 3D index of a neighboring grid point.
        neighbor_offset (np.ndarray): The offset (0 or 1) from the floor of the coordinates to the neighbor.

    Returns:
        float: The distance factor.
    """
    d_diff = 1.0
    for i in range(NUM_DIMENSIONS):
        d_diff *= (
            neighbor_grid_index[i]
            - neighbor_offset[i]
            - crd3d_grid_coords[i]
            + 1
            - neighbor_offset[i]
        )
    d_diff = np.abs(d_diff)
    return d_diff


@jit(
    [
        "float64[:,:](float64[:])",
        "float64[:,:](int64[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_grid_neighbors(crd3d_grid_coords):
    """
    Determines the 8 neighboring grid points and their distance factors for a given coordinate.

    Args:
        crd3d_grid_coords (np.ndarray): The 3D coordinates (float or int) of a point within the grid.

    Returns:
        np.ndarray: A 2D array of shape (8, 4), where each row represents a neighbor
                    and contains [ix, iy, iz, distance_factor].
    """
    floor_coords = np.floor(crd3d_grid_coords).astype(np.int64)
    points = np.zeros((8, 4), dtype=np.float64)

    for i in GRID_NEIGHBOR_OFFSETS:
        for j in GRID_NEIGHBOR_OFFSETS:
            for k in GRID_NEIGHBOR_OFFSETS:
                p_index = int(i * 4 + j * 2 + k)
                points[p_index][0] = float(floor_coords[0] + i)
                points[p_index][1] = float(floor_coords[1] + j)
                points[p_index][2] = float(floor_coords[2] + k)
                points[p_index][3] = crd3d_grid_distance_factor(
                    crd3d_grid_coords,
                    points[p_index][0:3].astype(np.int64),
                    np.array([i, j, k], dtype=np.float64),
                )

    return points


@njit(nogil=True, boundscheck=False, cache=True)
def sort_2d_by_single_column(data_array, column_index):
    """Sorts a 2D NumPy array based on a single column."""
    if data_array.shape[0] == 0:
        return data_array
    idx = np.argsort(data_array[:, column_index], kind="quicksort")
    return data_array[idx]


@njit(nogil=True, boundscheck=False, cache=True)
def is_contained_in_sorted(sorted_array, key, column_index=0):
    """
    Checks if a key exists in a sorted array column using binary search.

    Args:
        sorted_array (np.ndarray): The sorted NumPy array to search within.
        key (float or int): The value to search for.
        column_index (int, optional): The index of the column to search in. Defaults to 0.

    Returns:
        tuple: A tuple containing:
            - bool: True if the key is found, False otherwise.
            - int: The index of the key if found, -1 otherwise.
    """
    if sorted_array.size == 0:  # Handle empty array case
        return False, -1

    if sorted_array.ndim == 2:
        if column_index >= sorted_array.shape[1]:
            # Avoid index error if column_index is out of bounds
            return False, -1
        sorted_data = sorted_array[:, column_index]
    elif sorted_array.ndim == 1:
        sorted_data = sorted_array
    else:  # Handle unexpected dimensions
        # Or raise an error
        return False, -1

    # The searchsorted finds insertion point; check if element at insertion point matches
    left = np.searchsorted(sorted_data, key, side="left")
    # Must check bounds *before* accessing index `left`
    if left < sorted_data.shape[0] and sorted_data[left] == key:
        return True, left
    return False, -1


@njit(nogil=True, boundscheck=False, cache=True)
def accumulate_sorted_by_key(
    sorted_data: np.ndarray,  # Input array, MUST be sorted by key_col_idx
    key_col_idx: int,  # Index of the key column (e.g., index_1d)
    value_col_idx: int,  # Index of the value column to sum (e.g., charge)
    carry_cols_indices: tuple,  # Tuple of indices for other columns to carry along
    invalid_key_marker: float = -1.0,  # Key value indicating rows to ignore
):
    """
    Accumulates values in a pre-sorted array based on a key column,
    carrying along specified columns from the first instance of each key.
    Rows with the invalid_key_marker are skipped.

    Args:
        sorted_data (np.ndarray): The input NumPy array, which must be sorted by the column specified in `key_col_idx`.
        key_col_idx (int): The index of the column to use as the key for accumulation.
        value_col_idx (int): The index of the column containing the values to be summed.
        carry_cols_indices (tuple): A tuple of integer indices representing columns whose values from the
                                    first occurrence of each key should be carried over to the output.
        invalid_key_marker (float, optional): A value in the key column that indicates rows to be ignored. Defaults to -1.0.

    Returns:
        np.ndarray: A 2D NumPy array where each row represents a unique key,
                    containing the key, the sum of the values, and the carried-over values.
                    The columns are in the order: [key, sum(values), carried_value_1, carried_value_2, ...].
    """
    num_rows = sorted_data.shape[0]
    num_carry_cols = len(carry_cols_indices)
    output_num_cols = 1 + 1 + num_carry_cols  # key + summed_value + carried_cols
    # Here dtype for float is kept to float64 to avoid mapping large (> 10**6) index1d being incorrect due to precision limits.
    if num_rows == 0:
        return np.empty((0, output_num_cols), dtype=np.float64)

    output_array = np.empty((num_rows, output_num_cols), dtype=np.float64)
    output_count = 0
    i = 0

    # Find the first valid row
    while i < num_rows and sorted_data[i, key_col_idx] == invalid_key_marker:
        i += 1

    if i >= num_rows:
        return np.empty((0, output_num_cols), dtype=np.float64)

    # Initialize with the first valid row
    current_key = sorted_data[i, key_col_idx]
    current_value_sum = sorted_data[i, value_col_idx]
    carried_values = np.empty(num_carry_cols, dtype=np.float64)
    for k, carry_idx in enumerate(carry_cols_indices):
        carried_values[k] = sorted_data[i, carry_idx]

    i += 1
    while i < num_rows:
        row_key = sorted_data[i, key_col_idx]

        if row_key == invalid_key_marker:
            i += 1
            continue

        if row_key == current_key:
            current_value_sum += sorted_data[i, value_col_idx]
        else:
            # Finalize previous group
            output_array[output_count, 0] = current_key
            output_array[output_count, 1] = current_value_sum
            for k, val in enumerate(carried_values):
                output_array[output_count, 2 + k] = val
            output_count += 1
            # Start new group
            current_key = row_key
            current_value_sum = sorted_data[i, value_col_idx]
            for k, carry_idx in enumerate(carry_cols_indices):
                carried_values[k] = sorted_data[i, carry_idx]  # Corrected: use i, not 0

        i += 1

    # Process the very last valid group
    output_array[output_count, 0] = current_key
    output_array[output_count, 1] = current_value_sum
    for k, val in enumerate(carried_values):
        output_array[output_count, 2 + k] = val
    output_count += 1

    return output_array[:output_count]


@njit(nogil=True, boundscheck=False, cache=True, parallel=_numba_usable)
def generate_atomic_charge_grid_contributions(
    is_focusing: np.bool_,
    atoms_data: np.ndarray,
    grid_shape: np.ndarray,
    contributions_array: np.ndarray,  # Output array, size (num_atoms * max_neighbors, 5)
    max_neighbors_per_atom: int,
):
    """
    Generates (index_1d, charge_contribution, ix, iy, iz) for all atoms on their surrounding grid points.

    Args:
        is_focusing: If this a focusing run then True otherwise False
        atoms_data (np.ndarray): Array containing atom information.
        grid_shape (np.ndarray): Array representing the shape of the grid (nx, ny, nz).
        contributions_array (np.ndarray): Pre-allocated output array of shape (num_atoms * max_neighbors, 5).
        max_neighbors_per_atom (int): Maximum number of neighboring grid points to consider per atom.
    """
    num_atoms = atoms_data.shape[0]
    nx, ny, nz = grid_shape

    y_stride = nz
    x_stride = ny * y_stride

    num_charge_inside_gridbox = 0
    for i_atom in prange(num_atoms):
        this_atom = atoms_data[i_atom]
        atom_charge = this_atom[ATOMFIELD_CHARGE]
        atom_grid_coords = this_atom[ATOMFIELD_GRID_X:ATOMFIELD_GRID_END].astype(
            np.float64
        )  # Get atoms grid coords (float)

        # Check if the charge is contained completely inside the focusing gridbox (focusing run only)
        is_inside_focusing_gridbox = and_gt_scalar(
            atom_grid_coords, 0.0
        ) and and_lt_scalar(atom_grid_coords, max(grid_shape) - 1)
        if not is_focusing or is_inside_focusing_gridbox:
            num_charge_inside_gridbox += 1

            # Get neighbors and weights using the provided helper
            neighbor_iterable = crd3d_grid_neighbors(atom_grid_coords)

            for neighbor_index, p in enumerate(neighbor_iterable):
                if neighbor_index >= max_neighbors_per_atom:
                    break  # Safety break

                ix, iy, iz = np.int64(p[0]), np.int64(p[1]), np.int64(p[2])
                weight = np.float64(p[3])
                target_idx = i_atom * max_neighbors_per_atom + neighbor_index

                # Boundary check
                if (0 <= ix < nx) and (0 <= iy < ny) and (0 <= iz < nz):
                    charge_contribution = atom_charge * weight
                    index_1d = ix * x_stride + iy * y_stride + iz
                    contributions_array[target_idx, GPCHRGFIELD_INDX_1D] = index_1d
                    contributions_array[target_idx, GPCHRGFIELD_CHARGE] = (
                        charge_contribution
                    )
                    contributions_array[target_idx, GPCHRGFIELD_INDX_X] = float(ix)
                    contributions_array[target_idx, GPCHRGFIELD_INDX_Y] = float(iy)
                    contributions_array[target_idx, GPCHRGFIELD_INDX_Z] = float(iz)
                else:
                    # Mark as invalid
                    contributions_array[target_idx, GPCHRGFIELD_INDX_1D] = -1.0
                    contributions_array[target_idx, GPCHRGFIELD_CHARGE] = 0.0
                    contributions_array[target_idx, GPCHRGFIELD_INDX_X] = -1.0
                    contributions_array[target_idx, GPCHRGFIELD_INDX_Y] = -1.0
                    contributions_array[target_idx, GPCHRGFIELD_INDX_Z] = -1.0
    nprint_cpu(
        DEBUG,
        _VERBOSITY,
        " Total #charged-atoms contained in grid: ",
        num_charge_inside_gridbox,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def format_and_count_grid_charges(unique_contributions, index_sorted_charged_grids):
    """
    Formats unique grid point contributions and counts the number of charged points.

    Args:
        unique_contributions (np.ndarray): A 2D array where each row represents a unique
            grid point with accumulated charge and coordinates, sorted by index_1d.
            Expected columns: [index_1d, total_charge, ix, iy, iz].
        index_sorted_charged_grids (np.ndarray): A pre-allocated 2D array to store the
            formatted charged grid points. Rows correspond to charged grid points,
            and columns are [index_1d, charge, ix, iy, iz].

    Returns:
        tuple: A tuple containing:
            - int: The total number of charged grid points (where absolute charge > APPROX_ZERO).
            - int: The number of positively charged grid points.
            - int: The number of negatively charged grid points.
    """
    num_unique_points = unique_contributions.shape[0]
    num_charged_grids = 0
    num_pve_charged_grids = 0
    num_nve_charged_grids = 0

    if num_unique_points > 0:
        for i in range(num_unique_points):
            total_charge = unique_contributions[i, GPCHRGFIELD_CHARGE]
            if abs(total_charge) > APPROX_ZERO:
                index_1d = unique_contributions[i, GPCHRGFIELD_INDX_1D]
                ix = unique_contributions[i, GPCHRGFIELD_INDX_X]
                iy = unique_contributions[i, GPCHRGFIELD_INDX_Y]
                iz = unique_contributions[i, GPCHRGFIELD_INDX_Z]

                index_sorted_charged_grids[num_charged_grids, GPCHRGFIELD_INDX_1D] = (
                    index_1d
                )
                index_sorted_charged_grids[num_charged_grids, GPCHRGFIELD_CHARGE] = (
                    total_charge
                )
                index_sorted_charged_grids[num_charged_grids, GPCHRGFIELD_INDX_X] = ix
                index_sorted_charged_grids[num_charged_grids, GPCHRGFIELD_INDX_Y] = iy
                index_sorted_charged_grids[num_charged_grids, GPCHRGFIELD_INDX_Z] = iz

                if total_charge > 0:
                    num_pve_charged_grids += 1
                elif total_charge < 0:
                    num_nve_charged_grids += 1

                num_charged_grids += 1

    return num_charged_grids, num_pve_charged_grids, num_nve_charged_grids


@njit
def safe_str_to_int(substr):
    """
    Converts a numeric string representation to an integer (JIT-safe).

    Args:
        substr (str): String containing numeric digits.

    Returns:
        int: Integer value, or 0 if conversion fails or string is empty.
    """
    result = 0
    for char in substr:
        if 48 <= ord(char) <= 57:  # ASCII '0'-'9'
            result = result * 10 + (ord(char) - 48)
    return result


def seconds_to_hms(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"


def print_3d_array_data(data_name, data_array, array_shape, n_per_line=8):
    """
    Prints the contents of a 3D NumPy array in a formatted manner.

    Iterates through elements up to `array_shape` bounds and prints indices/values.

    Args:
        data_name (str): Descriptive name for the data being printed.
        data_array (np.ndarray): 3D NumPy array to print.
        array_shape (tuple/list): Shape (nx, ny, nz) for iteration bounds.
                                  Assumes array dims are at least shape + 1.
        n_per_line (int): Max entries per line. Defaults to 8.
    """
    total_elements = (array_shape[0] + 1) * (array_shape[1] + 1) * (array_shape[2] + 1)
    print(f"{data_name} (total {total_elements} entries): [")
    ic = 0
    for i3 in range(array_shape[2] + 1):
        for i2 in range(array_shape[1] + 1):
            for i1 in range(array_shape[0] + 1):
                if (
                    i1 < data_array.shape[0]
                    and i2 < data_array.shape[1]
                    and i3 < data_array.shape[2]
                ):
                    print(f"({i1},{i2},{i3},{data_array[i1, i2, i3]}), ", end="")
                else:  # OOB = Out Of Bounds
                    print(f"({i1},{i2},{i3}, OOB), ", end="")
                if (ic + 1) % n_per_line == 0:
                    print("")
                ic += 1
    if ic % n_per_line != 0:
        print("")
    print("]\n\n")
