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

from pydelphi.foundation.enums import Precision
from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
)

from pydelphi.constants import (
    XYZ_COMPONENTS,
    HALF_GRID_OFFSET_LAGGING,
    HALF_GRID_OFFSET_LEADING,
    BOX_BOUNDARY,
    BOX_HOMO_EPSILON,
)

# Import precision-specific utils
precision = PRECISION.value
if precision in (Precision.SINGLE.value,):
    from pydelphi.utils.prec.single import *

    try:
        from pydelphi.utils.cuda.single import *  # Optional CUDA utils for single precision
    except ImportError:
        pass

elif precision == Precision.DOUBLE.value:
    from pydelphi.utils.prec.double import *

    try:
        from pydelphi.utils.cuda.double import *  # Optional CUDA utils for double precision
    except ImportError:
        pass


# --- Core Iteration Functions ---


@cuda.jit(cache=True)
def _cuda_iterate_nwt(
    vacuum: delphi_bool,
    even_odd: delphi_int,
    non_zero_salt: delphi_bool,
    approx_zero: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    phi_map_current_half_1d: np.ndarray[delphi_real],
    phi_map_next_half_1d: np.ndarray[delphi_real],
    salt_ions_solvation_penalty_map_1d: np.ndarray[delphi_real],
    epsilon_map_midpoints_1d: np.ndarray[delphi_real],
    epsilon_sum_neighbors_plus_salt_screening_1d: np.ndarray[delphi_real],
    is_boundary_gridpoint_1d: np.ndarray[delphi_bool],
    charge_map_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_real],
) -> None:
    """
    CUDA kernel for Newton-like non-linear iteration.

    Args:
        even_odd (delphi_int): Flag for even/odd iterations.
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
    num_grid_points_half = (num_grid_points + 1) // 2

    ijk1d_half = cuda.grid(1)  # Get CUDA grid index
    ijk1d = 2 * ijk1d_half + even_odd.item()  # Convert half index to full index
    ijk1d_x_3 = ijk1d * XYZ_COMPONENTS

    if ijk1d < num_grid_points:  # Grid array bounds check
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
            # Calculate numerator for NWT update
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

                # Calculate $\sigma_{p=1}^{6}{phi_ijk_neighbor_p*eps_ijk_midpoint_p}$ term of numerator
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

            # Non-linear screening due to salt
            if (not vacuum) and non_zero_salt and (not ion_exclusion_map_1d[ijk1d]):
                phimap_ijk1d = phi_map_next_half_1d[ijk1d_half]
                exp_phi = math.exp(phimap_ijk1d)  # exp(x)
                exp_phi_inv = 1.0 / exp_phi  # exp(-x)
                sinh_phi = 0.5 * (exp_phi - exp_phi_inv)
                cosh_phi = 0.5 * (exp_phi + exp_phi_inv)
                # cosh_phi = math.cosh(phimap_ijk1d)
                # sinh_phi = math.sinh(phimap_ijk1d)

                salt_factor_numerator = salt_ions_solvation_penalty_map_1d[ijk1d] * (
                    phimap_ijk1d * cosh_phi - sinh_phi
                )
                salt_factor_denominator = (
                    salt_ions_solvation_penalty_map_1d[ijk1d] * cosh_phi
                )

                numerator += salt_factor_numerator
                denominator += salt_factor_denominator

            # Apply update, considering charge density
            if (
                abs(charge_density) > approx_zero.item()
            ):  # Check if charge density is significant
                phi_map_next_half_1d[ijk1d_half] = (
                    numerator + charge_density
                ) / denominator
            else:  # If charge density is negligible
                phi_map_next_half_1d[ijk1d_half] = numerator / denominator


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_iterate_nwt(
    vacuum: delphi_bool,
    even_odd: delphi_int,
    non_zero_salt: delphi_bool,
    approx_zero: delphi_real,
    epkt: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    phi_map_current_half_1d: np.ndarray[delphi_real],
    phi_map_next_half_1d: np.ndarray[delphi_real],
    salt_ions_solvation_penalty_map_1d: np.ndarray[delphi_real],
    epsilon_map_midpoints_1d: np.ndarray[delphi_real],
    epsilon_sum_neighbors_sum_only_1d: np.ndarray[delphi_real],
    is_boundary_gridpoint_1d: np.ndarray[delphi_bool],
    charge_map_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_real],
) -> None:
    """
    CPU function for Successive Over-Relaxation (SOR) iteration, similar to CUDA version.

    Args:
        vacuum (delphi_bool): Whether is run corresponds to vacuum phase.
        even_odd (delphi_int): Flag for even/odd iterations.
        non_zero_salt (delphi_bool): Whether salt-concentration is non-zero.
        approx_zero (delphi_real): Threshold for negligible charge density.
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid.
        phi_map_current_half_1d (np.ndarray[delphi_real]): Current phi values (odd or even half).
        phi_map_next_half_1d (np.ndarray[delphi_real]): Next phi values (even or odd half).
        epsilon_map_midpoints_1d (np.ndarray[delphi_real]): Epsilon values at midpoints.
        epsilon_sum_neighbors_sum_only_1d (np.ndarray[delphi_real]): Sum of epsilons for neighbors.
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

    # Parallel loop over half grid points
    for ijk1d_half in prange(num_grid_points_half):
        ijk1d = 2 * ijk1d_half + even_odd  # Convert half index to full index
        if ijk1d < num_grid_points:  # Boundary check
            ijk1d_x_3 = ijk1d * XYZ_COMPONENTS

            if is_boundary_gridpoint_1d[ijk1d] != BOX_BOUNDARY:
                epsilon_sum_local = epsilon_sum_neighbors_sum_only_1d[
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
                # Calculate numerator for NWT update
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

                    # Calculate $\sigma_{p=1}^{6}{phi_ijk_neighbor_p*eps_ijk_midpoint_p}$ term of numerator
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

                if (not vacuum) and non_zero_salt and (not ion_exclusion_map_1d[ijk1d]):
                    phimap_ijk1d = phi_map_next_half_1d[ijk1d_half]
                    exp_phi = math.exp(phimap_ijk1d)  # exp(x)
                    exp_phi_inv = 1.0 / exp_phi  # exp(-x)
                    sinh_phi = 0.5 * (exp_phi - exp_phi_inv)
                    cosh_phi = 0.5 * (exp_phi + exp_phi_inv)
                    # cosh_phi = math.cosh(phimap_ijk1d )
                    # sinh_phi = math.sinh(phimap_ijk1d )

                    salt_factor_numerator = salt_ions_solvation_penalty_map_1d[
                        ijk1d
                    ] * (phimap_ijk1d * cosh_phi - sinh_phi)
                    salt_factor_denominator = (
                        salt_ions_solvation_penalty_map_1d[ijk1d] * cosh_phi
                    )
                    # [20, 20, 20] 0-based
                    # if ijk1d == 20 * (
                    #     grid_shape[1] * grid_shape[2] + grid_shape[2] + 1
                    # ):
                    #     print(
                    #         "epssum:",
                    #         epsilon_sum_local,
                    #         "numer:",
                    #         numerator,
                    #         "debfct:",
                    #         kappa_x_grid_spacing_wholesquare,
                    #         "salt_num:",
                    #         salt_factor_numerator,
                    #         "salt_denum:",
                    #         salt_factor_denominator,
                    #
                    #     )

                    numerator += salt_factor_numerator
                    denominator += salt_factor_denominator

                # Apply NWT update, considering charge density if it is significant
                if abs(charge_density) > approx_zero:
                    updated_phi = (numerator + charge_density) / denominator
                else:  # If charge density is negligible
                    updated_phi = numerator / denominator

                # if ijk1d == 20 * (grid_shape[1] * grid_shape[2] + grid_shape[2] + 1):
                #     print(
                #         " phi_old:",
                #         phi_map_next_half_1d[ijk1d_half],
                #         " phi_new:",
                #         updated_phi,
                #     )

                phi_map_next_half_1d[ijk1d_half] = updated_phi  # Update phi value


def _cpu_iterate_block_nwt(
    vacuum: delphi_bool,
    iterations: delphi_int,
    iteration_block_size: delphi_int,
    non_zero_salt: delphi_bool,
    approx_zero: delphi_real,
    epkt: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    phi_map_odds_1d: np.ndarray[delphi_real],
    phi_map_even_1d: np.ndarray[delphi_real],
    salt_ions_solvation_penalty_map_1d: np.ndarray[delphi_real],
    epsilon_map_midpoints_1d: np.ndarray[delphi_real],
    epsilon_sum_neighbors_sum_only_1d: np.ndarray[delphi_real],
    is_boundary_gridpoint_1d: np.ndarray[delphi_bool],
    charge_map_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_real],
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
        epsilon_sum_neighbors_sum_only_1d (np.ndarray[delphi_real]): Sum of epsilons for neighbors.
        is_boundary_gridpoint_1d (np.ndarray[delphi_bool]): Boundary grid point flags.
        charge_map_1d (np.ndarray[delphi_real]): Charge density map.
    """
    for iteration_number in range(
        iterations + 1, iterations + iteration_block_size + 1
    ):  # Iterate for block size
        for even_odd_flag in [0, 1]:  # Iterate over even and odd points
            if even_odd_flag == 0:
                _cpu_iterate_nwt(
                    vacuum,
                    even_odd_flag,
                    non_zero_salt,
                    approx_zero,
                    epkt,
                    grid_shape,
                    phi_map_odds_1d,
                    phi_map_even_1d,
                    salt_ions_solvation_penalty_map_1d,
                    epsilon_map_midpoints_1d,
                    epsilon_sum_neighbors_sum_only_1d,
                    is_boundary_gridpoint_1d,
                    charge_map_1d,
                    ion_exclusion_map_1d,
                )
            elif even_odd_flag == 1:
                _cpu_iterate_nwt(
                    vacuum,
                    even_odd_flag,
                    non_zero_salt,
                    approx_zero,
                    epkt,
                    grid_shape,
                    phi_map_even_1d,
                    phi_map_odds_1d,
                    salt_ions_solvation_penalty_map_1d,
                    epsilon_map_midpoints_1d,
                    epsilon_sum_neighbors_sum_only_1d,
                    is_boundary_gridpoint_1d,
                    charge_map_1d,
                    ion_exclusion_map_1d,
                )
