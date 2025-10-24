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
This module provides functionality for distributing atomic charges onto a 3D grid
and aggregating them at individual grid points. It's a crucial component in
preparing charge source terms for electrostatic potential calculations, such as
those performed by Poisson-Boltzmann solvers.

The module implements a deterministic approach to assign atomic charges to
neighboring grid points, ensuring that the total charge is conserved and
properly distributed based on a predefined charge spreading method (e.g.,
Gaussian-like distributions). The core workflow involves:
1. Generating contributions: Calculating how much charge each atom contributes
   to its surrounding grid points.
2. Filtering valid contributions: Removing any invalid or out-of-bounds contributions.
3. Sorting contributions: Ordering the contributions by their 1D grid index
   to facilitate efficient aggregation.
4. Accumulating charges: Summing up all contributions for each unique grid point
   to obtain the net charge at that point.
5. Formatting and counting: Organizing the final charged grid points into a
   standardized format and counting the number of positive, negative, and total
   charged grid points.

This process is designed to be efficient for both CPU and potential GPU
implementations, leveraging optimized NumPy operations and potentially Numba
for further acceleration in other parts of the larger codebase.
"""

import numpy as np

from pydelphi.constants import (
    # GPCHRG: Constants for grid point charge fields
    GPCHRGFIELD_INDX_1D,  # Index of the 1D grid point index
    GPCHRGFIELD_CHARGE,  # Index of the charge at the grid point
    GPCHRGFIELD_INDX_X,  # Index of the x-coordinate of the grid point
    GPCHRGFIELD_INDX_Y,  # Index of the y-coordinate of the grid point
    GPCHRGFIELD_INDX_Z,  # Index of the z-coordinate of the grid point
    ConstDelPhiFloats as ConstDelPhi,
)

APPROX_ZERO = ConstDelPhi.ApproxZero.value
GAUSSIAN_INFLUENCE_RADIUS_FACTOR = ConstDelPhi.GaussianInfluenceRadiusFactor.value

from pydelphi.config.global_runtime import (
    vprint,
    nprint_cpu_if_verbose as nprint_cpu,
)
from pydelphi.config.logging_config import (
    DEBUG,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

from pydelphi.utils.utils import (
    generate_atomic_charge_grid_contributions,
    accumulate_sorted_by_key,
    format_and_count_grid_charges,
)


def set_grid_charges_sorted_by_index1d(
    is_focusing: np.bool_,
    atoms_data: np.ndarray,
    grid_shape: np.ndarray,  # e.g. np.array([nx, ny, nz])
    max_neighbors_per_atom: int = 8,
):
    """
    Distributes atomic charges to a grid and aggregates charges at each grid point.

    This method takes atomic data and a grid shape as input, calculates the
    contribution of each atom to its neighboring grid points, and then
    aggregates these contributions to determine the total charge at each unique
    grid point. The resulting charged grid points are sorted by their 1D index.

    Args:
        is_focusing: If this a focusing run then True otherwise False.
        atoms_data (np.ndarray): A 2D array containing information about each atom.
            The exact structure depends on the context (e.g., coordinates, charge).
        grid_shape (np.ndarray): A 1D array (or list) of length 3 representing the
            number of grid points along the x, y, and z axes (nx, ny, nz).
        max_neighbors_per_atom (int, optional): The maximum number of neighboring
            grid points an atom can contribute to. Defaults to 8.

    Returns:
        tuple: A tuple containing:
            - int: The total number of grid points with a non-zero charge.
            - int: The number of grid points with a positive charge.
            - int: The number of grid points with a negative charge.
            - np.ndarray: A 2D array where each row represents a unique grid
              point with a non-zero accumulated charge, sorted by its 1D index.
              Columns are [index_1d, total_charge, ix, iy, iz].
    """
    num_atoms = atoms_data.shape[0]

    nprint_cpu(
        DEBUG,
        _VERBOSITY,
        "INFO: Starting charge distribution (Sort-Reduce Deterministic)...",
    )
    # --- Allocation ---
    total_slots = num_atoms * max_neighbors_per_atom
    fill_value = np.array([-1.0, 0.0, -1.0, -1.0, -1.0], dtype=np.float64)
    contributions_array = np.empty((total_slots, 5), dtype=np.float64)
    contributions_array[:] = fill_value

    # --- Step 1: Generate Contributions ---
    generate_atomic_charge_grid_contributions(
        is_focusing,
        atoms_data,
        grid_shape,
        contributions_array,
        max_neighbors_per_atom,
    )

    # --- Step 2: Filter ---
    valid_contributions_mask = contributions_array[:, GPCHRGFIELD_INDX_1D] != -1.0
    valid_contributions = contributions_array[valid_contributions_mask]
    actual_num_contributions = valid_contributions.shape[0]

    # --- Step 3: Sort ---
    if actual_num_contributions > 0:
        sort_indices = np.argsort(
            valid_contributions[:, GPCHRGFIELD_INDX_1D], kind="quicksort"
        )
        sorted_contributions = valid_contributions[sort_indices]
    else:
        sorted_contributions = np.zeros((0, 5), dtype=np.float64)

    # --- Step 4: Accumulate ---
    if actual_num_contributions > 0:
        unique_contributions = accumulate_sorted_by_key(
            sorted_data=sorted_contributions,
            key_col_idx=GPCHRGFIELD_INDX_1D,
            value_col_idx=GPCHRGFIELD_CHARGE,
            carry_cols_indices=(
                GPCHRGFIELD_INDX_X,
                GPCHRGFIELD_INDX_Y,
                GPCHRGFIELD_INDX_Z,
            ),
            invalid_key_marker=-1.0,
        )
    else:
        unique_contributions = np.zeros((0, 5), dtype=np.float64)

    # --- Step 5: Final Formatting and Counting ---
    num_unique_points = unique_contributions.shape[0]
    index_sorted_charged_grids = np.empty((num_unique_points, 5), dtype=np.float64)
    index_sorted_charged_grids[:] = fill_value  # Initialize with default value

    (num_charged_grids, num_pve_charged_grids, num_nve_charged_grids) = (
        format_and_count_grid_charges(unique_contributions, index_sorted_charged_grids)
    )

    uniq_index_sorted_aggr_charge_grids = index_sorted_charged_grids[:num_charged_grids]
    vprint(
        DEBUG,
        _VERBOSITY,
        "uniq_index_sorted_aggr_charge_grids:",
        uniq_index_sorted_aggr_charge_grids,
    )
    return (
        num_charged_grids,
        num_pve_charged_grids,
        num_nve_charged_grids,
        uniq_index_sorted_aggr_charge_grids,
    )
