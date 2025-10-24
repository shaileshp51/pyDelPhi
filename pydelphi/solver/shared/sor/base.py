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


"""
This module implements iterative relaxation methods for solving Poisson-like equations
on a 3D grid, optimized for performance using Numba and CUDA.

It supports single, double, and mixed precision based on configuration,
and provides both CPU and CUDA implementations for core iteration routines.
"""

import numpy as np

from numba import njit, prange, cuda
from math import sin, sqrt

from pydelphi.foundation.enums import Precision
from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
)

from pydelphi.constants import (
    BOX_BOUNDARY,
    BOX_INTERIOR,
    BOX_HOMO_EPSILON,
    XYZ_COMPONENTS,
    HALF_GRID_OFFSET_LAGGING,
    HALF_GRID_OFFSET_LEADING,
    ConstPhysical as Constants,
)

# Import precision-specific utils
precision = PRECISION.int_value
if precision in (Precision.SINGLE.int_value,):
    from pydelphi.utils.prec.single import *

    try:
        from pydelphi.utils.cuda.single import *  # Optional CUDA utils for single precision
    except ImportError:
        pass

elif precision == Precision.DOUBLE.int_value:
    from pydelphi.utils.prec.double import *

    try:
        from pydelphi.utils.cuda.double import *  # Optional CUDA utils for double precision
    except ImportError:
        pass


# --- Core Iteration Functions ---


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_iterate_relaxation_factor(
    iteration_block_size: delphi_int,
    grid_shape: np.ndarray[delphi_int],
    phi_map_odds_1d: np.ndarray[delphi_real],
    phi_map_even_1d: np.ndarray[delphi_real],
    epsilon_map_midpoints_1d: np.ndarray[delphi_real],
    epsilon_sum_neighbors_1d: np.ndarray[delphi_real],
    is_boundary_gridpoint_1d: np.ndarray[delphi_bool],
) -> None:
    """
    Performs relaxation iterations to update the potential (phi) map using
    the relaxation factor method. This CPU version iterates over blocks of
    iterations and uses an odd-even update scheme for grid points.

    Algorithm:
    The Relaxation Factor method is implemented using an even-odd iteration scheme
    to improve convergence. In each iteration block, we perform two sub-iterations:
    one for even grid points and one for odd grid points. Potential (phi) values
    at each grid point are updated based on the weighted average of their neighbors'
    potential values and the dielectric constant at midpoints. Boundary grid points
    are skipped during the update process. After each even-odd iteration cycle, the
    phi maps for even and odd points are conceptually swapped to prepare for the next cycle.

    Args:
        iteration_block_size (delphi_int): Number of iterations to perform in this block.
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid (nz, nx, ny).
        phi_map_odds_1d (np.ndarray[delphi_real]): 1D array for phi values at odd grid points.
        phi_map_even_1d (np.ndarray[delphi_real]): 1D array for phi values at even grid points.
        epsilon_map_midpoints_1d (np.ndarray[delphi_real]): 1D array of epsilon values at midpoints.
        epsilon_sum_neighbors_1d (np.ndarray[delphi_real]): 1D array of sum of epsilons for neighbors.
        is_boundary_gridpoint_1d (np.ndarray[delphi_bool]): 1D boolean array indicating boundary points.
    """
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride

    x_stride_x_3 = x_stride * XYZ_COMPONENTS
    y_stride_x_3_minus_1 = y_stride * XYZ_COMPONENTS - 1

    y_stride_half = y_stride // 2
    x_stride_half = x_stride // 2

    num_grid_points = grid_shape[0] * x_stride
    num_grid_points_half = (num_grid_points + 1) // 2

    for iteration_number in range(1, iteration_block_size + 1):
        for update_type in (0, 1):  # 0 for even, 1 for odd
            phi_map_neighbors = phi_map_odds_1d if update_type == 0 else phi_map_even_1d
            phi_map_to_update = phi_map_even_1d if update_type == 0 else phi_map_odds_1d

            if update_type == 0:
                z_lag = HALF_GRID_OFFSET_LAGGING
                z_lead = HALF_GRID_OFFSET_LEADING
                y_lag = y_stride_half + 1
                y_lead = y_stride_half
                x_lag = x_stride_half + 1
                x_lead = x_stride_half
            else:
                z_lag = HALF_GRID_OFFSET_LEADING
                z_lead = HALF_GRID_OFFSET_LAGGING
                y_lag = y_stride_half
                y_lead = y_stride_half + 1
                x_lag = x_stride_half
                x_lead = x_stride_half + 1

            for ijk1d_half in prange(num_grid_points_half):
                ijk1d = update_type + ijk1d_half * 2
                ijk1d_x_3 = ijk1d * XYZ_COMPONENTS

                if ijk1d >= num_grid_points:
                    continue

                if is_boundary_gridpoint_1d[ijk1d] != BOX_BOUNDARY:
                    epsilon_sum_local = epsilon_sum_neighbors_1d[ijk1d]
                    eps_k_minus_half = epsilon_map_midpoints_1d[
                        ijk1d_x_3 - HALF_GRID_OFFSET_LAGGING
                    ]
                    eps_k_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 2]
                    eps_j_minus_half = epsilon_map_midpoints_1d[
                        ijk1d_x_3 - y_stride_x_3_minus_1
                    ]
                    eps_j_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 1]
                    eps_i_minus_half = epsilon_map_midpoints_1d[
                        ijk1d_x_3 - x_stride_x_3
                    ]
                    eps_i_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3]

                    phi_k_minus_1 = phi_map_neighbors[ijk1d_half - z_lag]
                    phi_k_plus_1 = phi_map_neighbors[ijk1d_half + z_lead]
                    phi_j_minus_1 = phi_map_neighbors[ijk1d_half - y_lag]
                    phi_j_plus_1 = phi_map_neighbors[ijk1d_half + y_lead]
                    phi_i_minus_1 = phi_map_neighbors[ijk1d_half - x_lag]
                    phi_i_plus_1 = phi_map_neighbors[ijk1d_half + x_lead]

                    numerator = (
                        phi_k_minus_1 * eps_k_minus_half
                        + phi_k_plus_1 * eps_k_plus_half
                        + phi_j_minus_1 * eps_j_minus_half
                        + phi_j_plus_1 * eps_j_plus_half
                        + phi_i_minus_1 * eps_i_minus_half
                        + phi_i_plus_1 * eps_i_plus_half
                    )
                    phi_map_to_update[ijk1d_half] = numerator / epsilon_sum_local


@cuda.jit(cache=True)
def _cuda_iterate_relaxation_factor(
    even_odd: delphi_int,
    grid_shape: np.ndarray[delphi_int],
    phi_map_current_half_1d: np.ndarray[delphi_real],
    phi_map_next_half_1d: np.ndarray[delphi_real],
    epsilon_map_midpoints_1d: np.ndarray[delphi_real],
    epsilon_sum_neighbors_1d: np.ndarray[delphi_real],
    is_boundary_gridpoint_1d: np.ndarray[delphi_bool],
) -> None:
    """
    CUDA kernel for relaxation iteration. Performs the same potential update
    as `_iterate_relaxation_factor` but on the GPU.

    Args:
        even_odd (delphi_int): Flag to distinguish even (0) and odd (1) iterations.
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid (nz, nx, ny).
        phi_map_current_half_1d (np.ndarray[delphi_real]): Current phi values (odd or even half).
        phi_map_next_half_1d (np.ndarray[delphi_real]): Next phi values (even or odd half).
        epsilon_map_midpoints_1d (np.ndarray[delphi_real]): Epsilon values at midpoints.
        epsilon_sum_neighbors_1d (np.ndarray[delphi_real]): Sum of epsilons for neighbors.
        is_boundary_gridpoint_1d (np.ndarray[delphi_bool]): Boundary grid point flags.
    """
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    num_grid_points = grid_shape[0] * x_stride

    y_stride_x_3 = y_stride * XYZ_COMPONENTS
    x_stride_x_3 = x_stride * XYZ_COMPONENTS
    y_stride_x_3_minus_1 = y_stride_x_3 - 1

    y_stride_half = y_stride // 2
    x_stride_half = x_stride // 2

    if even_odd.item() == 0:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LEADING
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LAGGING
        y_stride_half_leading_offset = y_stride_half
        y_stride_half_lagging_offset = y_stride_half + 1
        x_stride_half_leading_offset = x_stride_half
        x_stride_half_lagging_offset = x_stride_half + 1
    elif even_odd.item() == 1:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LAGGING  # Changed from 1 to HALF_GRID_OFFSET_LAGGING for clarity
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LEADING  # Changed from 0 to HALF_GRID_OFFSET_LEADING for clarity
        y_stride_half_leading_offset = y_stride_half + 1
        y_stride_half_lagging_offset = y_stride_half
        x_stride_half_leading_offset = x_stride_half + 1
        x_stride_half_lagging_offset = x_stride_half

    ijk1d_half = cuda.grid(1)  # Get 1D grid index for CUDA thread
    ijk1d = 2 * ijk1d_half + even_odd.item()  # Convert half index to full index
    ijk1d_x_3 = ijk1d * XYZ_COMPONENTS

    if ijk1d < num_grid_points:  # Boundary check
        if is_boundary_gridpoint_1d[ijk1d] != BOX_BOUNDARY:
            epsilon_sum_local = epsilon_sum_neighbors_1d[
                ijk1d
            ]  # Sum of epsilons for neighbors

            # Retrieve epsilon values for each neighbor midpoint
            eps_k_minus_half = epsilon_map_midpoints_1d[
                ijk1d_x_3 - HALF_GRID_OFFSET_LAGGING
            ]
            eps_k_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 2]
            eps_j_minus_half = epsilon_map_midpoints_1d[
                ijk1d_x_3 - y_stride_x_3_minus_1
            ]
            eps_j_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 1]
            eps_i_minus_half = epsilon_map_midpoints_1d[ijk1d_x_3 - x_stride_x_3]
            eps_i_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3]

            # Apply relaxation only to non-boundary points
            # Retrieve phi values from neighbor grid points based on current_half phi map
            phi_k_minus_1, phi_k_plus_1 = (
                phi_map_current_half_1d[ijk1d_half - z_stride_half_lagging_offset],
                phi_map_current_half_1d[ijk1d_half + z_stride_half_leading_offset],
            )
            phi_j_minus_1, phi_j_plus_1 = (
                phi_map_current_half_1d[ijk1d_half - y_stride_half_lagging_offset],
                phi_map_current_half_1d[ijk1d_half + y_stride_half_leading_offset],
            )
            phi_i_minus_1, phi_i_plus_1 = (
                phi_map_current_half_1d[ijk1d_half - x_stride_half_lagging_offset],
                phi_map_current_half_1d[ijk1d_half + x_stride_half_leading_offset],
            )

            # Calculate numerator for potential update
            numerator = (
                phi_k_minus_1 * eps_k_minus_half
                + phi_k_plus_1 * eps_k_plus_half
                + phi_j_minus_1 * eps_j_minus_half
                + phi_j_plus_1 * eps_j_plus_half
                + phi_i_minus_1 * eps_i_minus_half
                + phi_i_plus_1 * eps_i_plus_half
            )

            phi_map_next_half_1d[ijk1d_half] = (
                numerator / epsilon_sum_local
            )  # Update phi value


@cuda.jit(cache=True)
def _cuda_iterate_SOR(
    even_odd: delphi_int,
    omega: delphi_real,
    approx_zero: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    phi_map_current_half_1d: np.ndarray[delphi_real],
    phi_map_next_half_1d: np.ndarray[delphi_real],
    epsilon_map_midpoints_1d: np.ndarray[delphi_real],
    epsilon_sum_neighbors_plus_salt_screening_1d: np.ndarray[delphi_real],
    is_boundary_gridpoint_1d: np.ndarray[delphi_bool],
    charge_map_1d: np.ndarray[delphi_real],
) -> None:
    """
    CUDA kernel for Successive Over-Relaxation (SOR) iteration.

    Args:
        even_odd (delphi_int): Flag for even/odd iterations.
        omega (delphi_real): Over-relaxation factor.
        approx_zero (delphi_real): Threshold to consider charge density as zero.
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid.
        phi_map_current_half_1d (np.ndarray[delphi_real]): Current phi values (odd or even half).
        phi_map_next_half_1d (np.ndarray[delphi_real]): Next phi values (even or odd half).
        epsilon_map_midpoints_1d (np.ndarray[delphi_real]): Epsilon values at midpoints.
        epsilon_sum_neighbors_plus_salt_screening_1d (np.ndarray[delphi_real]): Sum of epsilons for neighbors.
        is_boundary_gridpoint_1d (np.ndarray[delphi_bool]): Boundary grid point flags.
        charge_map_1d (np.ndarray[delphi_real]): Charge density map.
    """
    nx, ny, nz = grid_shape
    y_stride = nz
    x_stride = ny * y_stride

    y_stride_x_3 = y_stride * XYZ_COMPONENTS
    x_stride_x_3 = x_stride * XYZ_COMPONENTS
    y_stride_x_3_minus_1 = y_stride_x_3 - 1

    y_stride_half = y_stride // 2
    x_stride_half = x_stride // 2

    # Offsets relative to current grid to access potentials (phi) at neighboring grids
    if even_odd.item() == 0:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LEADING
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LAGGING
        y_stride_half_leading_offset = y_stride_half
        y_stride_half_lagging_offset = y_stride_half + 1
        x_stride_half_leading_offset = x_stride_half
        x_stride_half_lagging_offset = x_stride_half + 1
    elif even_odd.item() == 1:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LAGGING
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LEADING
        y_stride_half_leading_offset = y_stride_half + 1
        y_stride_half_lagging_offset = y_stride_half
        x_stride_half_leading_offset = x_stride_half + 1
        x_stride_half_lagging_offset = x_stride_half

    num_grid_points = nx * x_stride

    one_minus_omega = 1.0 - omega.item()  # Pre-calculate (1 - omega)

    ijk1d_half = cuda.grid(1)  # Get CUDA grid index
    ijk1d = 2 * ijk1d_half + even_odd.item()  # Convert half index to full index
    ijk1d_x_3 = ijk1d * XYZ_COMPONENTS

    if ijk1d < num_grid_points:  # Grid array bounds check
        # Apply SOR only to non-boundary points
        if is_boundary_gridpoint_1d[ijk1d] != BOX_BOUNDARY:
            epsilon_sum_local = epsilon_sum_neighbors_plus_salt_screening_1d[
                ijk1d
            ]  # Sum of neighbor epsilons

            # Retrieve phi values from neighbor grid points from current_half phi map
            phi_k_minus_1, phi_k_plus_1 = (
                phi_map_current_half_1d[ijk1d_half - z_stride_half_lagging_offset],
                phi_map_current_half_1d[ijk1d_half + z_stride_half_leading_offset],
            )
            phi_j_minus_1, phi_j_plus_1 = (
                phi_map_current_half_1d[ijk1d_half - y_stride_half_lagging_offset],
                phi_map_current_half_1d[ijk1d_half + y_stride_half_leading_offset],
            )
            phi_i_minus_1, phi_i_plus_1 = (
                phi_map_current_half_1d[ijk1d_half - x_stride_half_lagging_offset],
                phi_map_current_half_1d[ijk1d_half + x_stride_half_leading_offset],
            )

            numerator = 0.0  # ensure defined regardless of branching
            # Calculate numerator for SOR update
            if is_boundary_gridpoint_1d[ijk1d] == BOX_HOMO_EPSILON:
                eps = epsilon_map_midpoints_1d[ijk1d_x_3]
                phi_sum = (
                    phi_k_minus_1
                    + phi_k_plus_1
                    + phi_j_minus_1
                    + phi_j_plus_1
                    + phi_i_minus_1
                    + phi_i_plus_1
                )
                numerator = eps * phi_sum
            else:
                # Retrieve epsilon values for neighbor midpoints
                eps_k_minus_half = epsilon_map_midpoints_1d[
                    ijk1d_x_3 - HALF_GRID_OFFSET_LAGGING
                ]
                eps_k_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 2]
                eps_j_minus_half = epsilon_map_midpoints_1d[
                    ijk1d_x_3 - y_stride_x_3_minus_1
                ]
                eps_j_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 1]
                eps_i_minus_half = epsilon_map_midpoints_1d[ijk1d_x_3 - x_stride_x_3]
                eps_i_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3]

                numerator = (
                    phi_k_minus_1 * eps_k_minus_half
                    + phi_k_plus_1 * eps_k_plus_half
                    + phi_j_minus_1 * eps_j_minus_half
                    + phi_j_plus_1 * eps_j_plus_half
                    + phi_i_minus_1 * eps_i_minus_half
                    + phi_i_plus_1 * eps_i_plus_half
                )

            denominator = epsilon_sum_local
            charge_density = charge_map_1d[ijk1d]  # Local charge density

            # Apply SOR update, considering charge density
            if (
                abs(charge_density) > approx_zero.item()
            ):  # Check if charge density is significant
                phi_map_next_half_1d[ijk1d_half] = (
                    omega.item() * phi_map_next_half_1d[ijk1d_half]
                    + one_minus_omega * (numerator + charge_density) / denominator
                )
            else:  # If charge density is negligible
                phi_map_next_half_1d[ijk1d_half] = omega.item() * phi_map_next_half_1d[
                    ijk1d_half
                ] + one_minus_omega * (numerator / denominator)


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_iterate_SOR(
    even_odd: delphi_int,
    omega: delphi_real,
    approx_zero: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    phi_map_current_half_1d: np.ndarray[delphi_real],
    phi_map_next_half_1d: np.ndarray[delphi_real],
    epsilon_map_midpoints_1d: np.ndarray[delphi_real],
    epsilon_sum_neighbors_plus_salt_screening_1d: np.ndarray[delphi_real],
    is_boundary_gridpoint_1d: np.ndarray[delphi_bool],
    charge_map_1d: np.ndarray[delphi_real],
) -> None:
    """
    CPU function for Successive Over-Relaxation (SOR) iteration, similar to CUDA version.

    Args:
        even_odd (delphi_int): Flag for even/odd iterations.
        omega (delphi_real): Over-relaxation factor.
        approx_zero (delphi_real): Threshold for negligible charge density.
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid.
        phi_map_current_half_1d (np.ndarray[delphi_real]): Current phi values (odd or even half).
        phi_map_next_half_1d (np.ndarray[delphi_real]): Next phi values (even or odd half).
        epsilon_map_midpoints_1d (np.ndarray[delphi_real]): Epsilon values at midpoints.
        epsilon_sum_neighbors_plus_salt_screening_1d (np.ndarray[delphi_real]): Sum of epsilons for neighbors.
        is_boundary_gridpoint_1d (np.ndarray[delphi_bool]): Boundary grid point flags.
        charge_map_1d (np.ndarray[delphi_real]): Charge density map.
    """
    nx, ny, nz = grid_shape
    y_stride = nz
    x_stride = ny * y_stride

    y_stride_x_3 = y_stride * XYZ_COMPONENTS
    x_stride_x_3 = x_stride * XYZ_COMPONENTS
    y_stride_x_3_minus_1 = y_stride_x_3 - 1

    y_stride_half = y_stride // 2
    x_stride_half = x_stride // 2

    if even_odd == 0:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LEADING
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LAGGING
        y_stride_half_leading_offset = y_stride_half
        y_stride_half_lagging_offset = y_stride_half + 1
        x_stride_half_leading_offset = x_stride_half
        x_stride_half_lagging_offset = x_stride_half + 1
    elif even_odd == 1:
        z_stride_half_leading_offset = HALF_GRID_OFFSET_LAGGING  # Changed from 1 to HALF_GRID_OFFSET_LAGGING for clarity
        z_stride_half_lagging_offset = HALF_GRID_OFFSET_LEADING  # Changed from 0 to HALF_GRID_OFFSET_LEADING for clarity
        y_stride_half_leading_offset = y_stride_half + 1
        y_stride_half_lagging_offset = y_stride_half
        x_stride_half_leading_offset = x_stride_half + 1
        x_stride_half_lagging_offset = x_stride_half

    num_grid_points = nx * x_stride
    num_grid_points_half = (num_grid_points + 1) // 2

    one_minus_omega = 1 - omega  # Pre-calculate (1 - omega)

    for ijk1d_half in prange(
        num_grid_points_half
    ):  # Parallel loop over half grid points
        ijk1d = 2 * ijk1d_half + even_odd  # Convert half index to full index
        if ijk1d < num_grid_points:  # Boundary check
            ijk1d_x_3 = ijk1d * XYZ_COMPONENTS

            # Apply SOR only to non-boundary points
            if is_boundary_gridpoint_1d[ijk1d] != BOX_BOUNDARY:
                epsilon_sum_local = epsilon_sum_neighbors_plus_salt_screening_1d[
                    ijk1d
                ]  # Sum of neighbor epsilons

                # Retrieve phi values from neighbor grid points from current_half phi map
                phi_k_minus_1, phi_k_plus_1 = (
                    phi_map_current_half_1d[ijk1d_half - z_stride_half_lagging_offset],
                    phi_map_current_half_1d[ijk1d_half + z_stride_half_leading_offset],
                )
                phi_j_minus_1, phi_j_plus_1 = (
                    phi_map_current_half_1d[ijk1d_half - y_stride_half_lagging_offset],
                    phi_map_current_half_1d[ijk1d_half + y_stride_half_leading_offset],
                )
                phi_i_minus_1, phi_i_plus_1 = (
                    phi_map_current_half_1d[ijk1d_half - x_stride_half_lagging_offset],
                    phi_map_current_half_1d[ijk1d_half + x_stride_half_leading_offset],
                )

                numerator = 0.0  # ensure defined regardless of branching
                if is_boundary_gridpoint_1d[ijk1d] == BOX_HOMO_EPSILON:
                    eps = epsilon_map_midpoints_1d[ijk1d_x_3]
                    phi_sum = (
                        phi_k_minus_1
                        + phi_k_plus_1
                        + phi_j_minus_1
                        + phi_j_plus_1
                        + phi_i_minus_1
                        + phi_i_plus_1
                    )
                    numerator = eps * phi_sum
                else:
                    # Retrieve epsilon values for neighbor midpoints
                    eps_k_minus_half = epsilon_map_midpoints_1d[
                        ijk1d_x_3 - HALF_GRID_OFFSET_LAGGING
                    ]
                    eps_k_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 2]
                    eps_j_minus_half = epsilon_map_midpoints_1d[
                        ijk1d_x_3 - y_stride_x_3_minus_1
                    ]
                    eps_j_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3 + 1]
                    eps_i_minus_half = epsilon_map_midpoints_1d[
                        ijk1d_x_3 - x_stride_x_3
                    ]
                    eps_i_plus_half = epsilon_map_midpoints_1d[ijk1d_x_3]

                    # Calculate numerator and denominator for SOR update
                    numerator = (
                        phi_k_minus_1 * eps_k_minus_half
                        + phi_k_plus_1 * eps_k_plus_half
                        + phi_j_minus_1 * eps_j_minus_half
                        + phi_j_plus_1 * eps_j_plus_half
                        + phi_i_minus_1 * eps_i_minus_half
                        + phi_i_plus_1 * eps_i_plus_half
                    )

                denominator = epsilon_sum_local
                charge_density = charge_map_1d[ijk1d]  # Local charge density

                # Apply SOR update, considering charge density if it is significant
                if abs(charge_density) > approx_zero:
                    updated_phi = omega * phi_map_next_half_1d[ijk1d_half] + (
                        one_minus_omega * (numerator + charge_density) / denominator
                    )
                else:  # If charge density is negligible
                    updated_phi = omega * phi_map_next_half_1d[ijk1d_half] + (
                        one_minus_omega * numerator / denominator
                    )
                phi_map_next_half_1d[ijk1d_half] = updated_phi  # Update phi value


def _cpu_iterate_block_SOR(
    iterations: delphi_int,
    iteration_block_size: delphi_int,
    omega: delphi_real,
    approx_zero: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    phi_map_odds_1d: np.ndarray[delphi_real],
    phi_map_even_1d: np.ndarray[delphi_real],
    epsilon_map_midpoints_1d: np.ndarray[delphi_real],
    epsilon_sum_neighbors_plus_salt_screening_1d: np.ndarray[delphi_real],
    is_boundary_gridpoint_1d: np.ndarray[delphi_bool],
    charge_map_1d: np.ndarray[delphi_real],
) -> None:
    """
    CPU function to perform SOR iteration for a block of iterations,
    alternating between even and odd grid point updates.

    Args:
        iterations (delphi_int): Starting iteration number.
        iteration_block_size (delphi_int): Number of iterations in the block.
        omega (delphi_real): Over-relaxation factor.
        approx_zero (delphi_real): Threshold for negligible charge density.
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid.
        phi_map_odds_1d (np.ndarray[delphi_real]): Phi values for odd grid points.
        phi_map_even_1d (np.ndarray[delphi_real]): Phi values for even grid points.
        epsilon_map_midpoints_1d (np.ndarray[delphi_real]): Epsilon values at midpoints.
        epsilon_sum_neighbors_plus_salt_screening_1d (np.ndarray[delphi_real]): Sum of epsilons for neighbors.
        is_boundary_gridpoint_1d (np.ndarray[delphi_bool]): Boundary grid point flags.
        charge_map_1d (np.ndarray[delphi_real]): Charge density map.
    """
    for iteration_number in range(
        iterations + 1, iterations + iteration_block_size + 1
    ):  # Iterate for block size
        for even_odd_flag in [0, 1]:  # Iterate over even and odd points
            if even_odd_flag == 0:
                _cpu_iterate_SOR(
                    even_odd_flag,
                    omega,
                    approx_zero,
                    grid_shape,
                    phi_map_odds_1d,
                    phi_map_even_1d,
                    epsilon_map_midpoints_1d,
                    epsilon_sum_neighbors_plus_salt_screening_1d,
                    is_boundary_gridpoint_1d,
                    charge_map_1d,
                )
            elif even_odd_flag == 1:
                _cpu_iterate_SOR(
                    even_odd_flag,
                    omega,
                    approx_zero,
                    grid_shape,
                    phi_map_even_1d,
                    phi_map_odds_1d,
                    epsilon_map_midpoints_1d,
                    epsilon_sum_neighbors_plus_salt_screening_1d,
                    is_boundary_gridpoint_1d,
                    charge_map_1d,
                )


# --- Phi Map Initialization Functions ---
@cuda.jit(cache=True)
def _cuda_init_relaxfactor_phimap(
    grid_shape: np.ndarray[delphi_int],
    sin_values_x: np.ndarray[delphi_real],
    sin_values_y: np.ndarray[delphi_real],
    sin_values_z: np.ndarray[delphi_real],
    phi_map_current_1d: np.ndarray[delphi_real],
) -> None:
    """
    CUDA kernel to initialize the potential (phi) map using sine function products.

    Args:
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid (nx, ny, nx).
        sin_values_x (np.ndarray[delphi_real]): Sine values along x-axis.
        sin_values_y (np.ndarray[delphi_real]): Sine values along y-axis.
        sin_values_z (np.ndarray[delphi_real]): Sine values along z-axis.
        phi_map_current_1d (np.ndarray[delphi_real]): 1D array for the phi map to be initialized.
    """
    nx, ny, nz = grid_shape
    y_stride = nz
    x_stride = ny * y_stride
    num_grid_points = nx * x_stride

    ijk1d = cuda.grid(1)  # Get CUDA grid index
    if ijk1d < num_grid_points:  # Boundary check
        i = ijk1d // x_stride  # Calculate x-index
        j = (ijk1d - i * x_stride) // y_stride  # Calculate y-index
        k = ijk1d - i * x_stride - j * y_stride  # Calculate z-index

        # Initialize phi map using product of sine values along each axis
        phi_map_current_1d[ijk1d] = sin_values_x[i] * sin_values_y[j] * sin_values_z[k]


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_init_relaxfactor_phimap(
    grid_shape: np.ndarray[delphi_int],
    sin_values_x: np.ndarray[delphi_real],
    sin_values_y: np.ndarray[delphi_real],
    sin_values_z: np.ndarray[delphi_real],
    phi_map_current_1d: np.ndarray[delphi_real],
) -> None:
    """
    CPU function to initialize the potential (phi) map using sine function products.

    Args:
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid (nx, ny, nx).
        sin_values_x (np.ndarray[delphi_real]): Sine values along x-axis.
        sin_values_y (np.ndarray[delphi_real]): Sine values along y-axis.
        sin_values_z (np.ndarray[delphi_real]): Sine values along z-axis.
        phi_map_current_1d (np.ndarray[delphi_real]): 1D array for the phi map to be initialized.
    """
    nx, ny, nz = grid_shape
    y_stride = nz
    x_stride = ny * y_stride
    num_grid_points = nx * x_stride

    for ijk1d in prange(num_grid_points):  # Parallel loop over grid points
        i = ijk1d // x_stride  # Calculate x-index
        j = (ijk1d - i * x_stride) // y_stride  # Calculate y-index
        k = ijk1d - i * x_stride - j * y_stride  # Calculate z-index

        # Initialize phi map using product of sine values along each axis
        phi_map_current_1d[ijk1d] = sin_values_x[i] * sin_values_y[j] * sin_values_z[k]


@njit(nogil=True, boundscheck=False, cache=True)
def _prepare_to_init_relaxfactor_phimap(
    grid_shape: np.ndarray[delphi_int],
    periodic_boundary_xyz: np.ndarray[delphi_bool],
    sin_values_x: np.ndarray[delphi_real],
    sin_values_y: np.ndarray[delphi_real],
    sin_values_z: np.ndarray[delphi_real],
) -> None:
    """
    Prepares sine value arrays for initializing the potential map,
    handling periodic boundary conditions if specified.

    Args:
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid (nx, ny, nz).
        periodic_boundary_xyz (np.ndarray[delphi_bool]): Boolean array for periodic boundaries in x, y, z.
        sin_values_x (np.ndarray[delphi_real]): Array to store sine values for x-axis.
        sin_values_y (np.ndarray[delphi_real]): Array to store sine values for y-axis.
        sin_values_z (np.ndarray[delphi_real]): Array to store sine values for z-axis.
    """
    nx, ny, nz = grid_shape
    pi_value = delphi_real(Constants.Pi.value)

    sqrt_of_2 = sqrt(2.0)
    sqrt_of_last_grid_x = sqrt(nx - 1.0)
    sqrt_of_last_grid_y = sqrt(ny - 1.0)
    sqrt_of_last_grid_z = sqrt(nz - 1.0)

    # Handle periodic boundary conditions by filling sine value arrays with reciprocal of sqrt(grid_size)
    if periodic_boundary_xyz[0]:
        reciprocal_x = 1.0 / sqrt(nx)
        sin_values_x.fill(reciprocal_x)  # Fill x-sine values for periodic boundary
    else:
        # Calculate sine values along x-axis
        for ix in range(nx):
            temp_angle = pi_value * ix / (nx - 1)  # Angle for sine function
            sin_values_x[ix] = (
                sqrt_of_2 * sin(temp_angle) / sqrt_of_last_grid_x
            )  # Sine value calculation

    if periodic_boundary_xyz[1]:
        reciprocal_y = 1.0 / sqrt(ny)
        sin_values_y.fill(reciprocal_y)  # Fill y-sine values for periodic boundary
    else:
        # Calculate sine values along y-axis
        for iy in range(ny):
            temp_angle = pi_value * iy / (ny - 1)  # Angle for sine function
            sin_values_y[iy] = (
                sqrt_of_2 * sin(temp_angle) / sqrt_of_last_grid_y
            )  # Sine value calculation

    if periodic_boundary_xyz[2]:
        reciprocal_z = 1.0 / sqrt(nz)
        sin_values_z.fill(reciprocal_z)  # Fill z-sine values for periodic boundary
    else:
        # Calculate sine values along z-axis
        for iz in range(nz):
            temp_angle = pi_value * iz / (nz - 1)  # Angle for sine function
            sin_values_z[iz] = (
                sqrt_of_2 * sin(temp_angle) / sqrt_of_last_grid_z
            )  # Sine value calculation
