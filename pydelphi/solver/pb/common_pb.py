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
This module provides functions for setting up boundary conditions and preparing
grid-related data for electrostatic potential calculations, supporting both
CPU (Numba) and GPU (Numba CUDA) acceleration.

It includes:
- `_set_gridpoint_charges`: Sets charges to specific grid points, handling
  multiple charges at the same point by summing them.
- `_cpu_setup_coulombic_boundary_condition`: CPU implementation for applying
  Coulombic boundary conditions to the potential map.
- `_cuda_setup_coulombic_boundary_condition`: CUDA implementation for applying
  Coulombic boundary conditions.
- `_cpu_setup_dipolar_boundary_condition`: CPU implementation for applying
  Dipolar boundary conditions based on charge centroids.
- `_cuda_setup_dipolar_boundary_condition`: CUDA implementation for applying
  Dipolar boundary conditions.
- `_cpu_prepare_charge_neigh_eps_sum_to_iterate`: CPU function to prepare
  charge and neighboring epsilon sums for iterative solvers, including
  Debye-Huckel screening.
- `_cuda_prepare_charge_neigh_eps_sum_to_iterate`: CUDA implementation for
  preparing charge and neighboring epsilon sums.
- `_cpu_setup_focusing_boundary_condition`: CPU implementation for focusing
  the potential map from a parent run onto a finer grid.

The module supports both single and double precision floating-point arithmetic,
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
    delphi_bool,
    delphi_int,
    delphi_real,
    vprint,
)

from pydelphi.config.logging_config import INFO, DEBUG, get_effective_verbosity

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

# GPCHRG: Constants for grid point charge fields
from pydelphi.constants import (
    # Index of the 1D grid point index
    GPCHRGFIELD_CHARGE,  # Index of the charge at the grid point
    GPCHRGFIELD_INDX_X,  # Index of the x-coordinate of the grid point
    GPCHRGFIELD_INDX_Y,  # Index of the y-coordinate of the grid point
    GPCHRGFIELD_INDX_Z,  # Index of the z-coordinate of the grid point
    BOX_BOUNDARY,  # The grid-point is on the gridbox boundary
    BOX_INTERIOR,  # The grid-point is box interior point
    BOX_HOMO_EPSILON,  # The grid-point's all neighbor midpoints dielectric are equal.
)

from pydelphi.constants import ConstDelPhiFloats as ConstDelPhi
from pydelphi.utils.interpolation import interpolate

APPROX_ZERO = ConstDelPhi.ApproxZero.value

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


@njit(nogil=True, boundscheck=False, cache=True)
def _set_gridpoint_charges(
    grid_shape, charged_gridpoints_1d, grid_charge_map_1d
) -> None:
    """Sets charges to specified grid points, handling multiple charges at the same point using atomic addition.

    Args:
        grid_shape (tuple): Shape of the grid (nx, ny, nz).
        charged_gridpoints_1d (np.ndarray): Array of grid points with charges [index_1d, total_charge, ix, iy, iz].
        grid_charge_map_1d (np.ndarray): 1D array to store grid point charges.
    """
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    num_grid_points = grid_shape[0] * x_stride

    # Iterate using prange for parallelization
    for idx in range(charged_gridpoints_1d.shape[0]):
        grid_ijkq = charged_gridpoints_1d[idx]
        i = int(grid_ijkq[GPCHRGFIELD_INDX_X])
        j = int(grid_ijkq[GPCHRGFIELD_INDX_Y])
        k = int(grid_ijkq[GPCHRGFIELD_INDX_Z])
        this_gridpoint_charge = delphi_real(grid_ijkq[GPCHRGFIELD_CHARGE])
        ijk1d = int(i * x_stride + j * y_stride + k)
        if 0 <= ijk1d < num_grid_points:
            grid_charge_map_1d[ijk1d] = this_gridpoint_charge


def _cpu_setup_focusing_boundary_condition(
    scale_parentrun,
    scale,
    grid_center_parentrun,
    grid_center,
    grid_shape_parentrun,
    grid_shape,
    approx_zero,
    phimap_parentrun,
    phimap_1d,
):
    if abs(scale_parentrun - scale) < approx_zero:  # comparing float to fZero
        vprint(INFO, _VERBOSITY, "scales are the same.")
        vprint(INFO, _VERBOSITY, "therefore assuming this to be a continuence")
    else:
        vprint(DEBUG, _VERBOSITY, "\n focussing potential map:")
        vprint(DEBUG, _VERBOSITY, f"original scale (grids/A)      : {scale_parentrun}")
        vprint(
            DEBUG,
            _VERBOSITY,
            f"object centre at (A) : {grid_center_parentrun[0]} {grid_center_parentrun[1]} {grid_center_parentrun[2]}",
        )  # Using array indexing

        all_focus_bgp_within_parent = True

        grid_spacing = delphi_real(1.0 / scale)
        mid_grid_index = grid_shape // 2  # Integer division for 0-based indexing
        mid_grid_index_parentrun = grid_shape_parentrun // 2

        y_stride = grid_shape[2]
        x_stride = grid_shape[1] * y_stride

        grid_real_index = np.zeros(3, dtype=delphi_real)

        last_x_p = grid_shape_parentrun[0] - 1
        last_y_p = grid_shape_parentrun[1] - 1
        last_z_p = grid_shape_parentrun[2] - 1

        # handles grid_shape=1 case, using max to avoid range(0, 1, 0)
        for iz in range(0, grid_shape[2], max(1, grid_shape[2] - 1)):
            for iy in range(0, grid_shape[1], max(1, grid_shape[1] - 1)):
                for ix in range(0, grid_shape[0], max(1, grid_shape[0] - 1)):
                    grid_real_index[0:3] = (
                        delphi_real(ix),
                        delphi_real(iy),
                        delphi_real(iz),
                    )
                    grid_coords = (
                        grid_center + (grid_real_index - mid_grid_index) * grid_spacing
                    )
                    grid_index_parentrun = (
                        mid_grid_index_parentrun
                        + (grid_coords - grid_center_parentrun) * scale_parentrun
                    )

                    if not (
                        0 <= grid_index_parentrun[0] < last_x_p
                        and 0 <= grid_index_parentrun[1] < last_y_p
                        and 0 <= grid_index_parentrun[2] < last_z_p
                    ):
                        all_focus_bgp_within_parent = False

        if not all_focus_bgp_within_parent:
            raise ValueError(
                f"Focusing outside parent boundary: {scale_parentrun}, {grid_center_parentrun}, {scale}, {grid_center}"
            )

        vprint(DEBUG, _VERBOSITY, "pulling boundary values out of old potential map...")

        for iz in range(grid_shape[2]):
            for iy in range(grid_shape[1]):
                for ix in range(0, grid_shape[0], max(1, grid_shape[0] - 1)):
                    ijk_1d = ix * x_stride + iy * y_stride + iz
                    grid_real_index[0:3] = (
                        delphi_real(ix),
                        delphi_real(iy),
                        delphi_real(iz),
                    )
                    grid_coords = (
                        grid_center + (grid_real_index - mid_grid_index) * grid_spacing
                    )
                    grid_index_parentrun = (
                        grid_coords - grid_center_parentrun
                    ) * scale_parentrun + mid_grid_index_parentrun  # NumPy array operations
                    interpolation_status, phimap_1d[ijk_1d] = interpolate(
                        grid_shape_parentrun, phimap_parentrun, grid_index_parentrun
                    )  # Note index swap ix, iy, iz

        for iz in range(grid_shape[2]):
            for iy in range(0, grid_shape[1], max(1, grid_shape[1] - 1)):
                for ix in range(grid_shape[0]):
                    ijk_1d = ix * x_stride + iy * y_stride + iz
                    grid_real_index[0:3] = (
                        delphi_real(ix),
                        delphi_real(iy),
                        delphi_real(iz),
                    )
                    grid_coords = (
                        grid_center + (grid_real_index - mid_grid_index) * grid_spacing
                    )
                    grid_index_parentrun = (
                        grid_coords - grid_center_parentrun
                    ) * scale_parentrun + mid_grid_index_parentrun

                    interpolation_status, phimap_1d[ijk_1d] = interpolate(
                        grid_shape_parentrun, phimap_parentrun, grid_index_parentrun
                    )

        for iz in range(0, grid_shape[2], max(1, grid_shape[2] - 1)):
            for iy in range(grid_shape[1]):
                for ix in range(grid_shape[0]):
                    ijk_1d = ix * x_stride + iy * y_stride + iz
                    grid_real_index[0:3] = (
                        delphi_real(ix),
                        delphi_real(iy),
                        delphi_real(iz),
                    )
                    grid_coords = (
                        grid_center + (grid_real_index - mid_grid_index) * grid_spacing
                    )
                    grid_index_parentrun = (
                        grid_coords - grid_center_parentrun
                    ) * scale_parentrun + mid_grid_index_parentrun

                    interpolation_status, phimap_1d[ijk_1d] = interpolate(
                        grid_shape_parentrun, phimap_parentrun, grid_index_parentrun
                    )


@njit(nogil=True, boundscheck=False, parallel=True, fastmath=True, cache=True)
def _cpu_setup_coulombic_boundary_condition(
    vacuum: delphi_bool,
    grid_spacing: delphi_real,
    exdi_scaled: delphi_real,
    indi_scaled: delphi_real,
    debye_length: delphi_real,
    non_zero_salt: delphi_bool,
    grid_shape: np.ndarray[delphi_int],
    atoms_data: np.ndarray[delphi_real],
    phimap_1d: np.ndarray[delphi_real],
) -> None:
    """Sets up Coulombic boundary condition on CPU.

    Calculates and sets the electrostatic potential (phimap_1d) at the boundary grid points
    assuming a Coulombic potential from all atoms.

    Args:
        vacuum (delphi_bool): Flag indicating vacuum conditions.
        grid_spacing (delphi_real): Grid spacing.
        exdi_scaled (delphi_real): Scaled exterior dielectric constant.
        indi_scaled (delphi_real): Scaled interior dielectric constant.
        debye_length (delphi_real): Debye length.
        grid_shape (np.ndarray[delphi_int]): Shape of the grid (nx, ny, nz).
        atoms_data (np.ndarray[delphi_real]): Array of atom data.
        phimap_1d (np.ndarray[delphi_real]): 1D array to store the potential map.
    """
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride

    last_grid_id_x = delphi_int(grid_shape[0] - 1)
    last_grid_id_y = delphi_int(grid_shape[1] - 1)
    last_grid_id_z = delphi_int(grid_shape[2] - 1)

    debye_length_inverse = 1.0 / debye_length
    epsilon_temp = exdi_scaled
    this_grid_phi = 0.0

    if vacuum:
        epsilon_temp = indi_scaled

    for ijk1d in prange(n_grid_points):
        ix = ijk1d // x_stride
        jy = (ijk1d - ix * x_stride) // y_stride
        kz = ijk1d - ix * x_stride - jy * y_stride
        # these are non-bounday points just ignore these
        if (
            (0 < ix < last_grid_id_x)
            and (0 < jy < last_grid_id_y)
            and (0 < kz < last_grid_id_z)
        ):
            phimap_1d[ijk1d] = 0.0  # Set potential to 0 for internal points
        else:
            # Initialize potential to 0 for boundary points
            this_grid_phi = 0.0
            for this_atom in atoms_data:
                atom_charge = get_atom_charge(this_atom)
                # Consider only atoms with significant charge
                if abs(atom_charge) > delphi_real(ConstDelPhi.ApproxZero.value):
                    grid_dx = ix - get_atom_grid_x(this_atom)
                    grid_dy = jy - get_atom_grid_y(this_atom)
                    grid_dz = kz - get_atom_grid_z(this_atom)
                    grid_dist_square = (
                        grid_dx * grid_dx + grid_dy * grid_dy + grid_dz * grid_dz
                    )
                    fdist = delphi_real(math.sqrt(grid_dist_square) * grid_spacing)
                    tmpval = atom_charge
                    if (not vacuum) and non_zero_salt:
                        tmpval *= math.exp(
                            -fdist * debye_length_inverse
                        )  # Apply Debye-Huckel screening if not vacuum
                    tmpval /= fdist * epsilon_temp
                    this_grid_phi += tmpval
            phimap_1d[ijk1d] = this_grid_phi  # Set boundary potential


@cuda.jit(cache=True,fastmath=True)
def _cuda_setup_coulombic_boundary_condition(
    vacuum: delphi_bool,
    grid_spacing: delphi_real,
    exdi_scaled: delphi_real,
    indi_scaled: delphi_real,
    debye_length: delphi_real,
    non_zero_salt: delphi_bool,
    grid_shape: np.ndarray[delphi_int],
    atoms_data: np.ndarray[delphi_real],
    phimap_1d: np.ndarray[delphi_real],
) -> None:
    """Sets up Coulombic boundary condition on CUDA.

    Calculates and sets the electrostatic potential (phimap_1d) at the boundary grid points
    assuming a Coulombic potential from all atoms, using CUDA for parallel execution.

    Args:
        vacuum (delphi_bool): Flag indicating vacuum conditions.
        grid_spacing (delphi_real): Grid spacing.
        exdi_scaled (delphi_real): Scaled exterior dielectric constant.
        indi_scaled (delphi_real): Scaled interior dielectric constant.
        debye_length (delphi_real): Debye length.
        grid_shape (np.ndarray[delphi_int]): Shape of the grid (nx, ny, nz).
        atoms_data (np.ndarray[delphi_real]): Array of atom data.
        phimap_1d (np.ndarray[delphi_real]): 1D array to store the potential map.
    """
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride

    last_grid_id_x = delphi_int(grid_shape[0] - 1)
    last_grid_id_y = delphi_int(grid_shape[1] - 1)
    last_grid_id_z = delphi_int(grid_shape[2] - 1)

    debye_length_inverse = 1.0 / debye_length
    epsilon_temp = exdi_scaled

    if vacuum:
        epsilon_temp = indi_scaled

    ijk1d = cuda.grid(1)  # Get the global thread ID
    if ijk1d < n_grid_points:
        ix = ijk1d // x_stride
        jy = (ijk1d - ix * x_stride) // y_stride
        kz = ijk1d - ix * x_stride - jy * y_stride
        # these are non-bounday points
        if (
            (0 < ix < last_grid_id_x)
            and (0 < jy < last_grid_id_y)
            and (0 < kz < last_grid_id_z)
        ):
            phimap_1d[ijk1d] = 0.0  # Set potential to 0 for internal points
        else:
            this_grid_phi_total = 0.0  # Initialize potential to 0 for boundary points
            for this_atom in atoms_data:
                atom_charge = cu_get_atom_charge(this_atom)
                # Consider only atoms with significant charge
                if abs(atom_charge) > delphi_real(ConstDelPhi.ApproxZero.value):
                    grid_dx = ix - cu_get_atom_grid_x(this_atom)
                    grid_dy = jy - cu_get_atom_grid_y(this_atom)
                    grid_dz = kz - cu_get_atom_grid_z(this_atom)
                    grid_dist_square = (
                        grid_dx * grid_dx + grid_dy * grid_dy + grid_dz * grid_dz
                    )
                    fdist = delphi_real(math.sqrt(grid_dist_square) * grid_spacing)
                    tmpval = atom_charge
                    if (not vacuum) and non_zero_salt:
                        tmpval *= math.exp(
                            -fdist * debye_length_inverse
                        )  # Apply Debye-Huckel screening if not vacuum
                    tmpval = tmpval / (fdist * epsilon_temp)
                    this_grid_phi_total += tmpval
            phimap_1d[
                ijk1d
            ] += this_grid_phi_total  # Accumulate potential from each atom


@njit(nogil=True, boundscheck=False, parallel=True, fastmath=True, cache=True)
def _cpu_setup_dipolar_boundary_condition(
    vacuum: delphi_bool,
    has_pve_charges: delphi_bool,
    has_nve_charges: delphi_bool,
    grid_spacing: delphi_real,
    exdi_scaled: delphi_real,
    indi_scaled: delphi_real,
    debye_length: delphi_real,
    non_zero_salt: delphi_bool,
    total_pve_charge: delphi_real,
    total_nve_charge: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_centroid_pve_charge: np.ndarray[delphi_real],
    grid_centroid_nve_charge: np.ndarray[delphi_real],
    phimap_1d: np.ndarray[delphi_real],
) -> None:
    """Sets up Dipolar boundary condition on CPU.

    Calculates and sets the electrostatic potential (phimap_1d) at the boundary grid points
    assuming a dipolar potential based on total positive and negative charges and their centroids.

    Args:
        vacuum (delphi_bool): Flag indicating vacuum conditions.
        grid_spacing (delphi_real): Grid spacing.
        exdi_scaled (delphi_real): Scaled exterior dielectric constant.
        indi_scaled (delphi_real): Scaled interior dielectric constant.
        debye_length (delphi_real): Debye length.
        total_pve_charge (delphi_real): Total positive mobile ion charge.
        total_nve_charge (delphi_real): Total negative mobile ion charge.
        grid_shape (np.ndarray[delphi_int]): Shape of the grid (nx, ny, nz).
        grid_centroid_pve_charge (np.ndarray[delphi_real]): Centroid of positive charges.
        grid_centroid_nve_charge (np.ndarray[delphi_real]): Centroid of negative charges.
        phimap_1d (np.ndarray[delphi_real]): 1D array to store the potential map.
    """
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride

    last_grid_id_x = delphi_int(grid_shape[0] - 1)
    last_grid_id_y = delphi_int(grid_shape[1] - 1)
    last_grid_id_z = delphi_int(grid_shape[2] - 1)

    debye_length_inverse = 1.0 / debye_length
    epsilon_temp = exdi_scaled
    if vacuum:
        epsilon_temp = indi_scaled

    for ijk1d in prange(n_grid_points):
        ix = ijk1d // x_stride
        jy = (ijk1d - ix * x_stride) // y_stride
        kz = ijk1d - ix * x_stride - jy * y_stride
        # These are non-boundary points just initialize them to 0.0.
        if (
            (0 < ix < last_grid_id_x)
            and (0 < jy < last_grid_id_y)
            and (0 < kz < last_grid_id_z)
        ):
            phimap_1d[ijk1d] = 0.0  # Set potential to 0 for internal points
        else:
            temp_phi_this_grid = 0.0  # Initialize potential to 0 for boundary points
            if has_pve_charges:
                grid_pve_dx = ix - grid_centroid_pve_charge[0]
                grid_pve_dy = jy - grid_centroid_pve_charge[1]
                grid_pve_dz = kz - grid_centroid_pve_charge[2]
                grid_pve_dist_square = (
                    grid_pve_dx * grid_pve_dx
                    + grid_pve_dy * grid_pve_dy
                    + grid_pve_dz * grid_pve_dz
                )
                dist_to_pve = delphi_real(
                    math.sqrt(grid_pve_dist_square) * grid_spacing
                )
                temp_phi_pve = total_pve_charge / (dist_to_pve * epsilon_temp)
                if (not vacuum) and non_zero_salt:
                    temp_phi_pve *= math.exp(
                        -dist_to_pve * debye_length_inverse
                    )  # Apply Debye-Huckel screening if not vacuum
                # Accumulate potential from positive charges
                temp_phi_this_grid += temp_phi_pve
            if has_nve_charges:
                grid_nve_dx = ix - grid_centroid_nve_charge[0]
                grid_nve_dy = jy - grid_centroid_nve_charge[1]
                grid_nve_dz = kz - grid_centroid_nve_charge[2]
                grid_nve_dist_square = (
                    grid_nve_dx * grid_nve_dx
                    + grid_nve_dy * grid_nve_dy
                    + grid_nve_dz * grid_nve_dz
                )
                dist_to_nve = delphi_real(
                    math.sqrt(grid_nve_dist_square) * grid_spacing
                )
                temp_phi_nve = total_nve_charge / (dist_to_nve * epsilon_temp)
                if (not vacuum) and non_zero_salt:
                    temp_phi_nve *= math.exp(
                        -dist_to_nve * debye_length_inverse
                    )  # Apply Debye-Huckel screening if not vacuum
                # Accumulate potential from negative charges
                temp_phi_this_grid += temp_phi_nve
            phimap_1d[ijk1d] = temp_phi_this_grid


@cuda.jit(cache=True,fastmath=True)
def _cuda_setup_dipolar_boundary_condition(
    vacuum: delphi_bool,
    has_pve_charges: delphi_bool,
    has_nve_charges: delphi_bool,
    grid_spacing: delphi_real,
    exdi_scaled: delphi_real,
    indi_scaled: delphi_real,
    debye_length: delphi_real,
    non_zero_salt: delphi_bool,
    total_pve_charge: delphi_real,
    total_nve_charge: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_centroid_pve_charge: np.ndarray[delphi_real],
    grid_centroid_nve_charge: np.ndarray[delphi_real],
    phimap_1d: np.ndarray[delphi_real],
) -> None:
    """Sets up Dipolar boundary condition on CUDA.

    Calculates and sets the electrostatic potential (phimap_1d) at the boundary grid points
    assuming a dipolar potential based on total positive and negative charges and their centroids, using CUDA.

    Args:
        vacuum (delphi_bool): Flag indicating vacuum conditions.
        grid_spacing (delphi_real): Grid spacing.
        exdi_scaled (delphi_real): Scaled exterior dielectric constant.
        indi_scaled (delphi_real): Scaled interior dielectric constant.
        debye_length (delphi_real): Debye length.
        total_pve_charge (delphi_real): Total positive mobile ion charge.
        total_nve_charge (delphi_real): Total negative mobile ion charge.
        grid_shape (np.ndarray[delphi_int]): Shape of the grid (nx, ny, nz).
        phimap_1d (np.ndarray[delphi_real]): 1D array to store the potential map.
    """
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride

    last_grid_id_x = delphi_int(grid_shape[0] - 1)
    last_grid_id_y = delphi_int(grid_shape[1] - 1)
    last_grid_id_z = delphi_int(grid_shape[2] - 1)

    debye_length_inverse = 1.0 / debye_length
    epsilon_temp = exdi_scaled
    if vacuum:
        epsilon_temp = indi_scaled

    ijk1d = cuda.grid(1)  # Get the global thread ID
    if ijk1d < n_grid_points:
        ix = ijk1d // x_stride
        jy = (ijk1d - ix * x_stride) // y_stride
        kz = ijk1d - ix * x_stride - jy * y_stride
        # These are non-boundary points just ignore
        if (
            (0 < ix < last_grid_id_x)
            and (0 < jy < last_grid_id_y)
            and (0 < kz < last_grid_id_z)
        ):
            phimap_1d[ijk1d] = 0.0  # Set potential to 0 for internal points
        else:
            temp_phi_this_grid = 0.0  # Initialize potential to 0 for boundary points
            if has_pve_charges:
                grid_pve_dx = ix - grid_centroid_pve_charge[0]
                grid_pve_dy = jy - grid_centroid_pve_charge[1]
                grid_pve_dz = kz - grid_centroid_pve_charge[2]
                grid_pve_dist_square = (
                    grid_pve_dx * grid_pve_dx
                    + grid_pve_dy * grid_pve_dy
                    + grid_pve_dz * grid_pve_dz
                )
                dist_to_pve = delphi_real(
                    math.sqrt(grid_pve_dist_square) * grid_spacing
                )
                temp_phi_pve = total_pve_charge / (dist_to_pve * epsilon_temp)
                if not vacuum:
                    temp_phi_pve *= math.exp(
                        -dist_to_pve * debye_length_inverse
                    )  # Apply Debye-Huckel screening if not vacuum
                # Accumulate potential from positive charges
                temp_phi_this_grid += temp_phi_pve
            if has_nve_charges:
                grid_nve_dx = ix - grid_centroid_nve_charge[0]
                grid_nve_dy = jy - grid_centroid_nve_charge[1]
                grid_nve_dz = kz - grid_centroid_nve_charge[2]
                grid_nve_dist_square = (
                    grid_nve_dx * grid_nve_dx
                    + grid_nve_dy * grid_nve_dy
                    + grid_nve_dz * grid_nve_dz
                )
                dist_to_nve = delphi_real(
                    math.sqrt(grid_nve_dist_square) * grid_spacing
                )
                temp_phi_nve = total_nve_charge / (dist_to_nve * epsilon_temp)
                if (not vacuum) and non_zero_salt:
                    temp_phi_nve *= math.exp(
                        -dist_to_nve * debye_length_inverse
                    )  # Apply Debye-Huckel screening if not vacuum
                # Accumulate potential from negative charges
                temp_phi_this_grid += temp_phi_nve
            phimap_1d[ijk1d] = temp_phi_this_grid


@cuda.jit(cache=True)
def _cuda_prepare_charge_neigh_eps_sum_to_iterate(
    vacuum: delphi_bool,
    exdi: delphi_real,
    grid_spacing: delphi_real,
    four_pi: delphi_real,
    epkt: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    epsmap_midpoints_1d: np.ndarray[delphi_real],
    charge_map_1d: np.ndarray[delphi_real],
    eps_midpoint_neighs_sum_salt_screening_1d: np.ndarray[delphi_real],
    boundary_gridpoints_1d: np.ndarray[delphi_bool],
):
    """Prepares data for iteration on CUDA.

    This CUDA kernel calculates parameters needed for the iterative solver, such as
    identifying boundary grid points, calculating the sum of dielectric constants around each grid point,
    and scaling the charge map.

    Args:
        vacuum (delphi_bool): Flag indicating vacuum conditions.
        exdi (delphi_real): Exterior dielectric constant.
        grid_spacing (delphi_real): Grid spacing.
        four_pi (delphi_real): Constant 4*pi.
        epkt (delphi_real): Constant to convert to kT/e.
        grid_shape (np.ndarray[delphi_int]): Shape of the grid.
        epsmap_midpoints_1d (np.ndarray[delphi_real]): 1D midpoint dielectric map.
        charge_map_1d (np.ndarray[delphi_real]): 1D charge map.
        eps_midpoint_neighs_sum_salt_screening_1d (np.ndarray[delphi_real]): 1D array to store sum of neighbor epsilons.
        boundary_gridpoints_1d (np.ndarray[delphi_bool]): 1D array to mark boundary points.
    """
    four_pi_epkt_grid_spacing = four_pi * epkt / grid_spacing
    epsout = 1.0 if vacuum else exdi

    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride
    y_stride_x_3 = 3 * y_stride
    x_stride_x_3 = 3 * x_stride

    last_grid_id_x = delphi_int(grid_shape[0] - 1)
    last_grid_id_y = delphi_int(grid_shape[1] - 1)
    last_grid_id_z = delphi_int(grid_shape[2] - 1)

    ijk1d = cuda.grid(1)  # Get the global thread ID
    if ijk1d < n_grid_points:
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3 = 3 * ijk1d
        if (
            (0 < i < last_grid_id_x)
            and (0 < j < last_grid_id_y)
            and (0 < k < last_grid_id_z)
        ):
            # Internal point, not a gridbox boundary
            eps_k_minus_half = epsmap_midpoints_1d[ijk1d_x_3 - 1]  # i,j,k-h/2
            eps_k_plus_half = epsmap_midpoints_1d[ijk1d_x_3 + 2]  # i,j,k+h/2
            eps_j_minus_half = epsmap_midpoints_1d[
                ijk1d_x_3 - y_stride_x_3 + 1
            ]  # i,j-h/2,k
            eps_j_plus_half = epsmap_midpoints_1d[ijk1d_x_3 + 1]  # i,j+h/2,k
            eps_i_minus_half = epsmap_midpoints_1d[
                ijk1d_x_3 - x_stride_x_3
            ]  # i-h/2,j,k
            eps_i_plus_half = epsmap_midpoints_1d[ijk1d_x_3]  # i+h/2,j,k

            eps_midpoint_neighs_sum_salt_screening_1d[ijk1d] = (
                eps_k_minus_half
                + eps_k_plus_half
                + eps_j_minus_half
                + eps_j_plus_half
                + eps_i_minus_half
                + eps_i_plus_half
            )

            # Check for homogeneous dielectric: all 6 epsilon midpoints are the same
            if (
                eps_k_minus_half == eps_k_plus_half
                and eps_k_minus_half == eps_j_minus_half
                and eps_k_minus_half == eps_j_plus_half
                and eps_k_minus_half == eps_i_minus_half
                and eps_k_minus_half == eps_i_plus_half
            ):
                boundary_gridpoints_1d[ijk1d] = BOX_HOMO_EPSILON
            else:
                boundary_gridpoints_1d[ijk1d] = BOX_INTERIOR
        else:
            boundary_gridpoints_1d[ijk1d] = BOX_BOUNDARY  # Mark as boundary point.
            eps_midpoint_neighs_sum_salt_screening_1d[ijk1d] = (
                6 * epsout
            )  # For boundary, use exterior dielectric * 6

    # Note: charge scaling is needed for both vacuum & water phase
    if ijk1d < n_grid_points:
        charge_map_1d[ijk1d] *= four_pi_epkt_grid_spacing  # Scale the charge map


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_prepare_charge_neigh_eps_sum_to_iterate(
    vacuum: delphi_bool,
    exdi: delphi_real,
    grid_spacing: delphi_real,
    four_pi: delphi_real,
    epkt: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    epsmap_midpoints_1d: np.ndarray[delphi_real],
    charge_map_1d: np.ndarray[delphi_real],
    eps_midpoint_neighs_sum_plus_salt_screening_1d: np.ndarray[delphi_real],
    boundary_gridpoints_1d: np.ndarray[delphi_bool],
):
    """Prepares dielectric and charge parameters for the iterative Poisson-Boltzmann solver.

    This function initializes key numerical terms for iterative updates in the solver, including:
    - Identifying boundary grid points.
    - Computing the sum of dielectric constants from neighboring grid midpoints.
    - Scaling the charge distribution for numerical stability.
    - Adjusting dielectric terms when a Debye screening length is used (non-vacuum cases).

    The function runs efficiently on the CPU using Numba's parallel execution.

    Args:
        vacuum (delphi_bool): Flag indicating if the system is in vacuum (True) or solvent (False).
        exdi (delphi_real): Exterior dielectric constant (used for boundary conditions).
        grid_spacing (delphi_real): Grid spacing (h) in physical units.
        four_pi (delphi_real): The numerical constant `4π`.
        epkt (delphi_real): The `e/kT` constant, used for electrostatic scaling.
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D computational grid (nx, ny, nz).
        epsmap_midpoints_1d (np.ndarray[delphi_real]): 1D array storing dielectric values at midpoints.
        charge_map_1d (np.ndarray[delphi_real]): 1D array storing charge distributions.
        eps_midpoint_neighs_sum_plus_salt_screening_1d (np.ndarray[delphi_real]): Output array storing sum of neighbor dielectric constants and screening due to salt.
        boundary_gridpoints_1d (np.ndarray[delphi_bool]): Output array marking grid points that lie on computational gridbox boundaries.

    Notes:
        - The function computes `eps_nd_midpoint_neighs_sum_1d` as the sum of dielectric constants around each grid point.
        - Boundary points are identified based on grid location and are treated separately.
        - If `vacuum == False` and a Debye length is provided, an additional screening term is added.
        - Charge values in `charge_map_1d` are scaled for use in the iterative solver.

    Parallel Execution:
        - The function uses `prange` to parallelize iteration over grid points.
        - No explicit locking or atomic operations are needed since updates to different indices are independent.

    """
    # Precompute common numerical constants for efficiency
    four_pi_epkt_grid_spacing = four_pi * epkt / grid_spacing  # Charge scaling factor
    epsout = 1.0 if vacuum else exdi  # Dielectric constant in vacuum vs. solvent

    # Compute strides for accessing the 1D representation of the 3D grid
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride
    y_stride_x_3 = 3 * y_stride
    x_stride_x_3 = 3 * x_stride

    # Precompute `y_stride_x_3 - 1` once outside the loop for performance
    # Note: ijk1d_x_3 - y_stride_x_3 + 1 is same as ijk1d_x_3 - y_stride_x_3_minus_1
    y_stride_x_3_minus_1 = y_stride_x_3 - 1

    # Compute last valid indices for boundary checks (avoids redundant calculations inside loop)
    last_grid_x = delphi_int(grid_shape[0] - 1)
    last_grid_y = delphi_int(grid_shape[1] - 1)
    last_grid_z = delphi_int(grid_shape[2] - 1)

    # Step 1: Identify boundary points and compute neighbor dielectric sums
    for ijk1d in prange(n_grid_points):
        # Convert 1D index back to 3D indices
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3 = 3 * ijk1d  # Used for accessing midpoint dielectric values

        # Identify boundary points based on their position in the grid
        if (0 < i < last_grid_x) and (0 < j < last_grid_y) and (0 < k < last_grid_z):
            # Internal point, not a gridbox boundary
            eps_k_minus_half = epsmap_midpoints_1d[ijk1d_x_3 - 1]  # i,j,k-h/2
            eps_k_plus_half = epsmap_midpoints_1d[ijk1d_x_3 + 2]  # i,j,k+h/2
            eps_j_minus_half = epsmap_midpoints_1d[
                ijk1d_x_3 - y_stride_x_3_minus_1
            ]  # i,j-h/2,k
            eps_j_plus_half = epsmap_midpoints_1d[ijk1d_x_3 + 1]  # i,j+h/2,k
            eps_i_minus_half = epsmap_midpoints_1d[
                ijk1d_x_3 - x_stride_x_3
            ]  # i-h/2,j,k
            eps_i_plus_half = epsmap_midpoints_1d[ijk1d_x_3]  # i+h/2,j,k

            eps_midpoint_neighs_sum_plus_salt_screening_1d[ijk1d] = (
                eps_k_minus_half
                + eps_k_plus_half
                + eps_j_minus_half
                + eps_j_plus_half
                + eps_i_minus_half
                + eps_i_plus_half
            )

            # Check for homogeneous dielectric: all 6 epsilon midpoints are the same
            if (
                eps_k_minus_half == eps_k_plus_half
                and eps_k_minus_half == eps_j_minus_half
                and eps_k_minus_half == eps_j_plus_half
                and eps_k_minus_half == eps_i_minus_half
                and eps_k_minus_half == eps_i_plus_half
            ):
                boundary_gridpoints_1d[ijk1d] = BOX_HOMO_EPSILON
            else:
                boundary_gridpoints_1d[ijk1d] = BOX_INTERIOR
        else:
            boundary_gridpoints_1d[ijk1d] = BOX_BOUNDARY
            eps_midpoint_neighs_sum_plus_salt_screening_1d[ijk1d] = (
                6 * epsout
            )  # Boundary points use exterior dielectric constant

    # Step 3: Scale charge distribution for term: \frac{4\times\pi\times{q}\times{epkt}}{h} (in kT/e units)
    for ijk1d in prange(n_grid_points):
        charge_map_1d[
            ijk1d
        ] *= four_pi_epkt_grid_spacing  # Apply precomputed charge scaling factor


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_salt_ions_solvation_penalty(
    vacuum: delphi_bool,
    non_zero_salt: delphi_bool,
    is_gaussian_diel_model: delphi_bool,
    exdi: delphi_real,
    ion_radius: delphi_real,
    ions_valance: delphi_real,
    debye_length: delphi_real,
    epkt: delphi_real,
    grid_spacing: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    epsilon_map_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_real],
    salt_ions_solvation_penalty_map_1d: np.ndarray[delphi_real],
) -> None:
    """Calculates solvation energy penalty factor.

    This function computes a penalty factor based on solvation energy, used in some contexts
    to adjust for ion solvation effects.  This is currently not used in the main PBE solver.

    Args:
        epsout (delphi_real): Exterior dielectric constant.
        ion_radius (delphi_real): Radius of the ion.
        ions_valance (delphi_real): Valance of the ion.
        srfcut (delphi_real): Surface dielectric cutoff value.
        epkt (delphi_real): Constant to convert to kT/e.
        epsilon_map_1d (np.ndarray[delphi_real]): 1D dielectric map.

    Returns:
        np.ndarray[delphi_real]: 1D array of Boltzmann factor energy penalty.
    """
    grid_spacing_square = grid_spacing**2
    kappa_square = exdi / debye_length**2  # Related to ionic screening

    kappa_x_grid_spacing_wholesquare = (
        kappa_square * grid_spacing_square
    )  # Screening term
    epsout = (
        delphi_real(1.0) if vacuum else delphi_real(exdi)
    )  # Dielectric constant in vacuum vs. solvent
    n_grid_points = grid_shape[0] * grid_shape[1] * grid_shape[2]

    # Debye screening factor if ion is present (non-vacuum)
    if (not vacuum) and non_zero_salt:
        # print("kappa_x_grid_spacing_wholesquare:", kappa_x_grid_spacing_wholesquare)
        if is_gaussian_diel_model:
            penalty_factor = epkt * ((ions_valance**2) * 1 / (2.0 * ion_radius))
            inverse_epsout = 1.0 / epsout
            for ijk1d in prange(n_grid_points):
                # if not ion_exclusion_map_1d[ijk1d]:
                # Note: ion_exclusion_map_1d is True/False for solvent, solute regions. It can be
                # real number between (0.0, 1.0) for diffused interface models like (Gaussian/GCS).
                energy_factor = 1.0 / epsilon_map_1d[ijk1d] - inverse_epsout
                boltz_factor_energy_penalty = math.exp(-penalty_factor * energy_factor)
                if boltz_factor_energy_penalty < APPROX_ZERO:
                    boltz_factor_energy_penalty = 0.0
                screening_factor = (
                    kappa_x_grid_spacing_wholesquare * boltz_factor_energy_penalty
                )

                salt_ions_solvation_penalty_map_1d[ijk1d] = screening_factor
        else:
            for ijk1d in prange(n_grid_points):
                if not ion_exclusion_map_1d[ijk1d]:
                    # Note: ion_exclusion_map_1d is True/False for solvent, solute regions. It can be
                    # real number between (0.0, 1.0) for diffused interface models like (Gaussian/GCS).
                    salt_ions_solvation_penalty_map_1d[ijk1d] = (
                        kappa_x_grid_spacing_wholesquare
                        * (1 - ion_exclusion_map_1d[ijk1d])
                    )

    return salt_ions_solvation_penalty_map_1d
