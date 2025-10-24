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
This module provides functions for calculating the spatial dielectric map and its gradient
on a 3D grid, supporting both CPU (Numba) and GPU (Numba CUDA) acceleration.

It focuses on:
- Calculating the gradient of the spatial epsilon map based on the surface map,
  internal epsilon map gradient, and grid point epsilon map.
- The module supports both single and double precision floating-point arithmetic,
  determined by the `PRECISION` global runtime configuration.
"""

import numpy as np

from numba import njit, prange
from numba import cuda

from pydelphi.foundation.enums import (
    Precision,
)
from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_int,
    delphi_real,
)

if PRECISION.value == Precision.SINGLE.value:
    from pydelphi.utils.prec.single import *

    try:
        from pydelphi.utils.cuda.single import *
    except ImportError:
        pass
        # print("No Cuda")

elif PRECISION.value == Precision.DOUBLE.value:
    from pydelphi.utils.prec.double import *

    try:
        from pydelphi.utils.cuda.double import *
    except ImportError:
        pass
        # print("No Cuda")


@cuda.jit(cache=True)
def _cuda_helper_calc_spatial_epsilon_map(
    epsout: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    surface_map_1d: np.ndarray[delphi_real],
    epsmap_gridpoints_1d: np.ndarray[delphi_real],
    grad_surface_map_1d: np.ndarray[delphi_real],
    grad_epsin_map_1d: np.ndarray[delphi_real],
    grad_epsmap_1d: np.ndarray[delphi_real],
):
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride

    last_grid_id_x = delphi_int(grid_shape[0] - 1)
    last_grid_id_y = delphi_int(grid_shape[1] - 1)
    last_grid_id_z = delphi_int(grid_shape[2] - 1)

    ijk1d = cuda.grid(1)
    if ijk1d < n_grid_points:
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3 = 3 * ijk1d
        if not (
            i == 0
            or i == last_grid_id_x
            or j == 0
            or j == last_grid_id_y
            or k == 0
            or k == last_grid_id_z
        ):
            grad_epsmap_1d[ijk1d_x_3] = (
                grad_epsin_map_1d[ijk1d_x_3] * surface_map_1d[ijk1d]
            ) + (epsmap_gridpoints_1d[ijk1d] - epsout) * grad_surface_map_1d[ijk1d_x_3]
            grad_epsmap_1d[ijk1d_x_3 + 1] = (
                grad_epsin_map_1d[ijk1d_x_3 + 1] * surface_map_1d[ijk1d]
            ) + (epsmap_gridpoints_1d[ijk1d] - epsout) * grad_surface_map_1d[
                ijk1d_x_3 + 1
            ]
            grad_epsmap_1d[ijk1d_x_3 + 2] = (
                grad_epsin_map_1d[ijk1d_x_3 + 2] * surface_map_1d[ijk1d]
            ) + (epsmap_gridpoints_1d[ijk1d] - epsout) * grad_surface_map_1d[
                ijk1d_x_3 + 2
            ]
        elif ijk1d_x_3 + 2 < 3 * n_grid_points:
            grad_epsmap_1d[ijk1d_x_3] = 0.0
            grad_epsmap_1d[ijk1d_x_3 + 1] = 0.0
            grad_epsmap_1d[ijk1d_x_3 + 2] = 0.0


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_helper_calc_spatial_epsilon_map(
    epsout: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    surface_map_1d: np.ndarray[delphi_real],
    epsmap_gridpoints_1d: np.ndarray[delphi_real],
    grad_surface_map_1d: np.ndarray[delphi_real],
    grad_epsin_map_1d: np.ndarray[delphi_real],
    grad_epsmap_1d: np.ndarray[delphi_real],
):
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride

    last_grid_id_x = delphi_int(grid_shape[0] - 1)
    last_grid_id_y = delphi_int(grid_shape[1] - 1)
    last_grid_id_z = delphi_int(grid_shape[2] - 1)

    for ijk1d in prange(n_grid_points):
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3 = 3 * ijk1d
        if not (
            i == 0
            or i == last_grid_id_x
            or j == 0
            or j == last_grid_id_y
            or k == 0
            or k == last_grid_id_z
        ):
            grad_epsmap_1d[ijk1d_x_3] = (
                grad_epsin_map_1d[ijk1d_x_3] * surface_map_1d[ijk1d]
            ) + (epsmap_gridpoints_1d[ijk1d] - epsout) * grad_surface_map_1d[ijk1d_x_3]
            grad_epsmap_1d[ijk1d_x_3 + 1] = (
                grad_epsin_map_1d[ijk1d_x_3 + 1] * surface_map_1d[ijk1d]
            ) + (epsmap_gridpoints_1d[ijk1d] - epsout) * grad_surface_map_1d[
                ijk1d_x_3 + 1
            ]
            grad_epsmap_1d[ijk1d_x_3 + 2] = (
                grad_epsin_map_1d[ijk1d_x_3 + 2] * surface_map_1d[ijk1d]
            ) + (epsmap_gridpoints_1d[ijk1d] - epsout) * grad_surface_map_1d[
                ijk1d_x_3 + 2
            ]
        elif ijk1d_x_3 + 2 < 3 * n_grid_points:
            grad_epsmap_1d[ijk1d_x_3] = 0.0
            grad_epsmap_1d[ijk1d_x_3 + 1] = 0.0
            grad_epsmap_1d[ijk1d_x_3 + 2] = 0.0
