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
PyDelphi Grid Interpolation Utilities (njit compatible).

This module provides functions for performing interpolation on 3D grids.
Functions are designed to be compatible with Numba's @njit decorator.
Error handling uses an explicit integer return status:
    - 0: Indicates successful execution.
    - EXIT_NJIT_FLAG: Indicates an error occurred. An error message is printed
      using nprint_cpu with PRINT_MANDATORY_VALUE. Callers are responsible
      for checking the return status and potentially aborting execution.

Functions:
    interpolate: Performs trilinear interpolation on a 3D grid.
    tricubic_interpolation: Performs 3D tricubic interpolation on a 3D grid.
    bool_interpolation: Performs trilinear interpolation on a 3D boolean grid.
"""

import math
import numpy as np
from numba import njit

from pydelphi.config.global_runtime import (
    delphi_int,
    delphi_real,
    nprint_cpu_if_verbose as nprint_cpu,
)
from pydelphi.config.logging_config import (
    ERROR,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

# All the intermediate real values are float64, so single precision utility functions are not needed.
from pydelphi.utils.prec.double import (
    or_lt_scalar,
    or_lt_vector,
    or_ge_vector,
    or_gt_vector,
)

from pydelphi.constants import ConstDelPhiInts

EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value


@njit(nogil=True, boundscheck=False, cache=True, fastmath=True)
def cubic_interpolation(values_4pt: np.ndarray, frac_x_local: float) -> float:
    """
    Performs 1D cubic interpolation.

    This is a helper function used by bicubic_interpolation and tricubic_interpolation to perform
    cubic interpolation along a single dimension.

    Args:
        values_4pt (np.ndarray): A 1D numpy array of 4 values representing the grid values around the point of interpolation.
                        Specifically, values_4pt = [value_{i-1}, value_{i}, value_{i+1}, value_{i+2}] where i is the floor index (float64).
        frac_x_local (float): The fractional part of the coordinate (between 0 and 1) along the interpolation dimension,
                           relative to the second point in values_4pt (value_i) (float64).

    Returns:
        float: The cubic interpolated value (float64).
    """
    p = values_4pt
    x = frac_x_local

    # Ensure intermediate calculations are float64
    interpolated_value = np.float64(
        x * x * (p[3] - 2.0 * p[2] + p[1]) / 2.0
        + x * (4.0 * p[2] - 3.0 * p[1] - p[3]) / 2.0
        + p[1]
    )
    return interpolated_value


@njit(nogil=True, boundscheck=False, cache=True, fastmath=True)
def bicubic_interpolation(
    values_4x4: np.ndarray, frac_x_local: float, frac_y_local: float
) -> float:
    """
    Performs 2D bicubic interpolation.

    This function interpolates a value at a point with fractional coordinates (frac_x_local, frac_y_local)
    within a local 2D grid represented by values_4x4.
    It uses cubic interpolation twice: first along one dimension (Y), and then along the other (X).

    Args:
        values_4x4 (np.ndarray): A 4x4 numpy array representing a local 4x4 grid of values needed for interpolation.
                                 values_4x4[i, j] represents the grid value at offset (i-1, j-1)
                                 from the floor index in each dimension (X and Y respectively) (float64).
        frac_x_local (float): The fractional part of the x-coordinate (between 0 and 1) within the cell (float64).
        frac_y_local (float): The fractional part of the y-coordinate (between 0 and 1) within the cell (float64).

    Returns:
        float: The bicubic interpolated value (float64).
    """
    p = values_4x4
    x = frac_x_local
    y = frac_y_local

    # Perform cubic interpolation along Y, results in float64
    interpolated_value_along_y_and_x_offset = np.zeros(4, dtype=np.float64)
    for i in range(4):
        interpolated_value_along_y_and_x_offset[i] = cubic_interpolation(p[i], y)

    # Perform cubic interpolation along X using the Y-interpolated values, result is float64
    interpolated_value = np.float64(
        cubic_interpolation(interpolated_value_along_y_and_x_offset, x)
    )

    return interpolated_value


@njit(nogil=True, boundscheck=False, cache=True, fastmath=True)
def interpolate(
    grid_shape: np.ndarray, grid_data_map: np.ndarray, grid_coords: np.ndarray
) -> tuple[delphi_int, delphi_real]:
    """
    Performs trilinear interpolation on a 3D grid at a given fractional grid coordinate.

    Args:
        grid_shape (np.ndarray): 1D array (nx, ny, nz) of grid shape (dtype=delphi_int).
        grid_data_map (np.ndarray): The 3D numpy array of shape (nx, ny, nz) containing the grid data, in C-order.
        grid_coords (np.ndarray): 1D array (fx, fy, fz) of fractional grid coordinates (0-based).

    Returns:
        Returns:
        tuple[delphi_int, delphi_real]: (status, interpolated_value). Status is 0 on success,
                                        EXIT_NJIT_FLAG on error.
    """
    nx = grid_shape[0]
    ny = grid_shape[1]
    nz = grid_shape[2]

    # Use imported helper functions for boundary checks
    # Grid extent for trilinear interpolation is [0, N-1] inclusive.
    last_grid_indices_float = np.array(
        [float(nx - 1), float(ny - 1), float(nz - 1)], dtype=np.float64
    )

    if or_lt_scalar(grid_coords, 0.0) or or_gt_vector(
        grid_coords, last_grid_indices_float
    ):
        nprint_cpu(
            ERROR,
            _VERBOSITY,
            "ERROR: Point out of trilinear interpolation box: VALUES: (",
            round(grid_coords[0], 4),
            ", ",
            round(grid_coords[1], 4),
            ", ",
            round(grid_coords[2], 4),
            ") GRID SHAPE: (",
            nx,
            ", ",
            ny,
            ", ",
            nz,
            ")",
        )
        # Return error flag and a default value cast to delphi_real
        return delphi_int(EXIT_NJIT_FLAG), delphi_real(0.0)

    grid_floor_index = np.array(
        [
            int(math.floor(grid_coords[0])),
            int(math.floor(grid_coords[1])),
            int(math.floor(grid_coords[2])),
        ],
        dtype=np.int32,
    )

    # Determine ceiling indices, clamped to max index (N-1)
    grid_ceiling_index = np.array(
        [
            min(grid_floor_index[0] + 1, nx - 1),
            min(grid_floor_index[1] + 1, ny - 1),
            min(grid_floor_index[2] + 1, nz - 1),
        ],
        dtype=np.int32,
    )

    fractional_part = grid_coords - grid_floor_index.astype(
        np.float64
    )  # Ensure float subtraction
    if np.allclose(fractional_part, 0.0, atol=1.0e-6):
        # Point is exactly on a grid point, return value cast to delphi_real
        return delphi_int(0), delphi_real(
            grid_data_map[grid_floor_index[0], grid_floor_index[1], grid_floor_index[2]]
        )

    # Weights are float64
    weights = grid_coords - grid_floor_index.astype(np.float64)

    # Get corner values, ensure they are treated as float for interpolation
    v000 = float(
        grid_data_map[grid_floor_index[0], grid_floor_index[1], grid_floor_index[2]]
    )
    v100 = float(
        grid_data_map[grid_ceiling_index[0], grid_floor_index[1], grid_floor_index[2]]
    )
    v010 = float(
        grid_data_map[grid_floor_index[0], grid_ceiling_index[1], grid_floor_index[2]]
    )
    v001 = float(
        grid_data_map[grid_floor_index[0], grid_floor_index[1], grid_ceiling_index[2]]
    )
    v110 = float(
        grid_data_map[grid_ceiling_index[0], grid_ceiling_index[1], grid_floor_index[2]]
    )
    v101 = float(
        grid_data_map[grid_ceiling_index[0], grid_floor_index[1], grid_ceiling_index[2]]
    )
    v011 = float(
        grid_data_map[grid_floor_index[0], grid_ceiling_index[1], grid_ceiling_index[2]]
    )
    v111 = float(
        grid_data_map[
            grid_ceiling_index[0], grid_ceiling_index[1], grid_ceiling_index[2]
        ]
    )

    # Interpolation calculations use float64
    v_ix_iy0_iz0 = v000 * (1.0 - weights[0]) + v100 * weights[0]
    v_ix_iy1_iz0 = v010 * (1.0 - weights[0]) + v110 * weights[0]
    v_ix_iy0_iz1 = v001 * (1.0 - weights[0]) + v101 * weights[0]
    v_ix_iy1_iz1 = v011 * (1.0 - weights[0]) + v111 * weights[0]

    v_ix_iy_iz0 = v_ix_iy0_iz0 * (1.0 - weights[1]) + v_ix_iy1_iz0 * weights[1]
    v_ix_iy_iz1 = v_ix_iy0_iz1 * (1.0 - weights[1]) + v_ix_iy1_iz1 * weights[1]

    interpolated_value = v_ix_iy_iz0 * (1.0 - weights[2]) + v_ix_iy_iz1 * weights[2]

    # Return success and interpolated value cast to delphi_real
    return delphi_int(0), delphi_real(interpolated_value)


@njit(nogil=True, boundscheck=False, cache=True, fastmath=True)
def tricubic_interpolation(
    grid_shape: np.ndarray, grid_data_map: np.ndarray, point_frac: np.ndarray
) -> tuple[delphi_int, delphi_real]:
    """
    Performs 3D tricubic interpolation on a 3D grid.

    Args:
        grid_shape (np.ndarray): 1D array (nx, ny, nz) of grid shape (dtype=delphi_int).
        grid_data_map (np.ndarray): 3D grid data (C-order [X][Y][Z]).
        point_frac (np.ndarray): 1D array (fx, fy, fz) of fractional grid coordinates (0-based).

    Returns:
        tuple[delphi_int, delphi_real]: (status, interpolated_value). Status is 0 on success,
                                        EXIT_NJIT_FLAG on error.
    """
    nx = grid_shape[0]
    ny = grid_shape[1]
    nz = grid_shape[2]

    # Valid domain for 0-based fractional point for tricubic interpolation is [1.0, N-2.0).
    min_valid_point_frac = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    max_valid_point_frac = np.array(
        [float(nx - 2.0), float(ny - 2.0), float(nz - 2.0)], dtype=np.float64
    )

    # Use imported helper functions for boundary checks
    if or_lt_vector(point_frac, min_valid_point_frac) or or_ge_vector(
        point_frac, max_valid_point_frac
    ):
        nprint_cpu(
            ERROR,
            _VERBOSITY,
            "ERROR: Point out of tricubic interpolation box: VALUES: (",
            round(point_frac[0], 4),
            ", ",
            round(point_frac[1], 4),
            ", ",
            round(point_frac[2], 4),
            ") GRID SHAPE: (",
            nx,
            ", ",
            ny,
            ", ",
            nz,
            ")",
        )
        # Return error flag and a default value cast to delphi_real
        return delphi_int(EXIT_NJIT_FLAG), delphi_real(0.0)

    floor_index = np.array(
        [
            int(math.floor(point_frac[0])),
            int(math.floor(point_frac[1])),
            int(math.floor(point_frac[2])),
        ],
        dtype=np.int32,
    )

    # Check if point falls exactly on a grid point
    if np.allclose(point_frac, floor_index.astype(np.float64), atol=1.0e-6):
        # Point is exactly on a grid point, return value cast to delphi_real
        return delphi_int(0), delphi_real(
            grid_data_map[floor_index[0], floor_index[1], floor_index[2]]
        )

    # Weights are float64
    weights_local = point_frac - floor_index.astype(np.float64)

    # Extract 4x4x4 subgrid, values should be float64 for intermediate calculations
    data_neighborhood_4x4x4 = np.zeros((4, 4, 4), dtype=np.float64)

    for local_x in range(4):
        for local_y in range(4):
            for local_z in range(4):
                global_x = floor_index[0] + local_x - 1
                global_y = floor_index[1] + local_y - 1
                global_z = floor_index[2] + local_z - 1

                # Check grid boundaries (defensive)
                if (
                    0 <= global_x < grid_shape[0]
                    and 0 <= global_y < grid_shape[1]
                    and 0 <= global_z < grid_shape[2]
                ):
                    data_neighborhood_4x4x4[local_x, local_y, local_z] = float(
                        grid_data_map[global_x, global_y, global_z]
                    )
                else:
                    data_neighborhood_4x4x4[local_x, local_y, local_z] = (
                        0.0  # Handle out of bounds
                    )

    # Perform bicubic interpolation along YZ for each X slice, results are float64
    bicubic_interpolants = np.zeros(4, dtype=np.float64)
    for local_x in range(4):
        bicubic_interpolants[local_x] = bicubic_interpolation(
            data_neighborhood_4x4x4[local_x], weights_local[1], weights_local[2]
        )

    # Perform cubic interpolation along X using bicubic results, result is float64
    interpolated_value = cubic_interpolation(bicubic_interpolants, weights_local[0])

    # Return success and interpolated value cast to delphi_real
    return delphi_int(0), delphi_real(interpolated_value)


@njit(nogil=True, boundscheck=False, cache=True, fastmath=True)
def bool_interpolation(
    grid_shape: np.ndarray, grid_data_map: np.ndarray, point_frac: np.ndarray
) -> tuple[delphi_int, delphi_real]:
    """
    Performs trilinear interpolation on a 3D boolean grid.

    This function is similar to `interpolation` but is optimized for boolean grids.
    The grid values are treated as boolean (True/False or 1/0), and trilinear interpolation is performed,
    resulting in a float value between 0.0 and 1.0, representing the weighted boolean value at the given point.

    Args:
        grid_shape (np.ndarray): 1D array (nx, ny, nz) of grid shape (dtype=delphi_int).
        grid_data_map (np.ndarray): 3D boolean grid data (C-order [X][Y][Z]).
        point_frac (np.ndarray): 1D array (fx, fy, fz) of fractional grid coordinates (0-based).

    Returns:
        tuple[delphi_int, delphi_real]: (status, interpolated_value). Status is 0 on success,
                                        EXIT_NJIT_FLAG on error. Value is between 0.0 and 1.0.
    """
    nx = grid_shape[0]
    ny = grid_shape[1]
    nz = grid_shape[2]
    max_grid_index = np.array(grid_shape, dtype=np.float64) - 1.0

    # Use imported helper functions for boundary checks
    # Grid extent for boolean interpolation is [0, N-1] inclusive.
    if or_lt_scalar(point_frac, 0.0) or or_gt_vector(point_frac, max_grid_index):
        nprint_cpu(
            ERROR,
            _VERBOSITY,
            "ERROR: Point out of boolean interpolation box: VALUES: (",
            round(point_frac[0], 4),
            ", ",
            round(point_frac[1], 4),
            ", ",
            round(point_frac[2], 4),
            ") GRID SHAPE: (",
            nx,
            ", ",
            ny,
            ", ",
            nz,
            ")",
        )
        # Return error flag and a default value cast to delphi_real
        return delphi_int(EXIT_NJIT_FLAG), delphi_real(0.0)

    floor_index = np.array(
        [
            int(math.floor(point_frac[0])),
            int(math.floor(point_frac[1])),
            int(math.floor(point_frac[2])),
        ],
        dtype=np.int32,
    )  # 0-based

    # Check if point falls exactly on a grid point
    if np.allclose(point_frac, floor_index.astype(np.float64), atol=1.0e-6):
        # Point is exactly on a grid point, return value cast to delphi_real
        return delphi_int(0), delphi_real(
            float(grid_data_map[floor_index[0], floor_index[1], floor_index[2]])
        )

    # Determine ceiling indices, clamped to max index (N-1)
    ceiling_index = floor_index + 1
    ceiling_index[0] = min(ceiling_index[0], int(max_grid_index[0]))
    ceiling_index[1] = min(ceiling_index[1], int(max_grid_index[1]))
    ceiling_index[2] = min(ceiling_index[2], int(max_grid_index[2]))

    # Weights are float64
    weights_local = point_frac - floor_index.astype(np.float64)

    # Get corner values, explicitly convert 0/non-zero to 0.0/1.0 float64
    val_000_int = grid_data_map[floor_index[0], floor_index[1], floor_index[2]]
    v000 = 0.0 if val_000_int == 0 else 1.0

    val_100_int = grid_data_map[ceiling_index[0], floor_index[1], floor_index[2]]
    v100 = 0.0 if val_100_int == 0 else 1.0

    val_010_int = grid_data_map[floor_index[0], ceiling_index[1], floor_index[2]]
    v010 = 0.0 if val_010_int == 0 else 1.0

    val_001_int = grid_data_map[floor_index[0], floor_index[1], ceiling_index[2]]
    v001 = 0.0 if val_001_int == 0 else 1.0

    val_110_int = grid_data_map[ceiling_index[0], ceiling_index[1], floor_index[2]]
    v110 = 0.0 if val_110_int == 0 else 1.0

    val_101_int = grid_data_map[ceiling_index[0], floor_index[1], ceiling_index[2]]
    v101 = 0.0 if val_101_int == 0 else 1.0

    val_011_int = grid_data_map[floor_index[0], ceiling_index[1], ceiling_index[2]]
    v011 = 0.0 if val_011_int == 0 else 1.0

    val_111_int = grid_data_map[ceiling_index[0], ceiling_index[1], ceiling_index[2]]
    v111 = 0.0 if val_111_int == 0 else 1.0

    # Calculate the 8 coefficients using float64
    a8 = v000
    a5 = v100 - v000
    a6 = v010 - v000
    a7 = v001 - v000
    a2 = v110 - v100 - a6
    a3 = v101 - v100 - a7
    a4 = v011 - v010 - a7
    a1 = v111 - v110 - a3 - a4 + a7

    # Perform trilinear interpolation using float64
    interpolated_value = (
        a1 * weights_local[0] * weights_local[1] * weights_local[2]
        + a2 * weights_local[0] * weights_local[1]
        + a3 * weights_local[0] * weights_local[2]
        + a4 * weights_local[1] * weights_local[2]
        + a5 * weights_local[0]
        + a6 * weights_local[1]
        + a7 * weights_local[2]
        + a8
    )

    # Return success and interpolated value cast to delphi_real
    return delphi_int(0), delphi_real(interpolated_value)
