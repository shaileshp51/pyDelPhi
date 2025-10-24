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
This module provides functions for calculating electrostatic potentials and related properties
on a 3D grid, supporting both CPU (Numba) and GPU (Numba CUDA) acceleration.

It includes:
- Calculation of Coulombic potential maps from atom charges.
- Calculation of the gradient of Coulombic potential maps.
- Calculation of the dot product between the gradient of the epsilon map and the gradient of the Coulomb map.
- Calculation of the gradient of the internal epsilon map (for Gaussian dielectric models).
- Setup of Coulombic and Dipolar boundary conditions for the potential map.
- Preparation of charge and neighboring epsilon sums for iterative solvers.

The module supports both single and double precision floating-point arithmetic,
determined by the `PRECISION` global runtime configuration.
"""

import numpy as np

from numba import njit, prange
from numba import cuda

from pydelphi.config.logging_config import DEBUG
from pydelphi.foundation.enums import (
    Precision,
    IonExclusionRegion,
)
from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
    nprint_cpu,
)

from pydelphi.constants import (
    ConstDelPhiFloats as ConstDelPhi,
    BOX_BOUNDARY,
    BOX_INTERIOR,
    BOX_HOMO_EPSILON,
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

APPROX_ZERO = ConstDelPhi.ApproxZero.value
STERN_LAYER = IonExclusionRegion.STERNLAYER.int_value
GAUSSIAN_LAYER = IonExclusionRegion.GAUSSIANLAYER.int_value
SOLUTE_SURFACE = IonExclusionRegion.SOLUTESURFACE.int_value


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_calc_coulomb_map(
    grid_spacing: delphi_real,
    indi_scaled: delphi_real,
    approx_zero: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    coulomb_map_1d: np.ndarray[delphi_real],
) -> None:
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride

    for ijk1d in prange(n_grid_points):
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride
        grid_pos_x = i * grid_spacing + grid_origin[0]
        grid_pos_y = j * grid_spacing + grid_origin[1]
        grid_pos_z = k * grid_spacing + grid_origin[2]
        this_grid_coulomb: delphi_real = 0.0
        for this_atom in atoms_data:
            atom_crd_x = get_atom_x(this_atom)
            atom_crd_y = get_atom_y(this_atom)
            atom_crd_z = get_atom_z(this_atom)
            atom_charge = get_atom_charge(this_atom)

            if atom_charge != 0.0:
                delta_rx = grid_pos_x - atom_crd_x
                delta_ry = grid_pos_y - atom_crd_y
                delta_rz = grid_pos_z - atom_crd_z
                fg_dist = math.sqrt(
                    delta_rx * delta_rx + delta_ry * delta_ry + delta_rz * delta_rz
                )
                if fg_dist > approx_zero:
                    potential = atom_charge / (fg_dist * indi_scaled)
                    this_grid_coulomb = delphi_real(this_grid_coulomb + potential)
        coulomb_map_1d[ijk1d] = this_grid_coulomb


@cuda.jit(cache=True)
def _cuda_calc_coulomb_map(
    grid_spacing: delphi_real,
    indi_scaled: delphi_real,
    approx_zero: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    coulomb_map_1d: np.ndarray[delphi_real],
) -> None:
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    num_grid_points = grid_shape[0] * x_stride
    ijk1d = cuda.grid(1)

    if ijk1d < num_grid_points:
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride
        grid_pos_x = i * grid_spacing + grid_origin[0]
        grid_pos_y = j * grid_spacing + grid_origin[1]
        grid_pos_z = k * grid_spacing + grid_origin[2]
        this_grid_coulomb: delphi_real = 0.0
        for this_atom in atoms_data:
            atom_crd_x = cu_get_atom_x(this_atom)
            atom_crd_y = cu_get_atom_y(this_atom)
            atom_crd_z = cu_get_atom_z(this_atom)
            atom_charge = cu_get_atom_charge(this_atom)

            if atom_charge != 0.0:
                delta_rx = grid_pos_x - atom_crd_x
                delta_ry = grid_pos_y - atom_crd_y
                delta_rz = grid_pos_z - atom_crd_z
                gridpoint_dist = math.sqrt(delta_rx**2 + delta_ry**2 + delta_rz**2)
                if gridpoint_dist > approx_zero:
                    potential = atom_charge / (gridpoint_dist * indi_scaled)
                    this_grid_coulomb = delphi_real(this_grid_coulomb + potential)
        coulomb_map_1d[ijk1d] = this_grid_coulomb


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_calc_grad_coulomb_map(
    grid_spacing: delphi_real,
    indi_scaled: delphi_real,
    approx_zero: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    grad_coulomb_map_1d: np.ndarray[delphi_real],
) -> None:
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride

    for ijk1d in prange(n_grid_points):
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3 = 3 * ijk1d

        grid_pos_x = i * grid_spacing + grid_origin[0]
        grid_pos_y = j * grid_spacing + grid_origin[1]
        grid_pos_z = k * grid_spacing + grid_origin[2]

        this_grid_coulomb_dx: delphi_real = 0.0
        this_grid_coulomb_dy: delphi_real = 0.0
        this_grid_coulomb_dz: delphi_real = 0.0
        for this_atom in atoms_data:
            atom_crd_x = get_atom_x(this_atom)
            atom_crd_y = get_atom_y(this_atom)
            atom_crd_z = get_atom_z(this_atom)
            atom_charge = get_atom_charge(this_atom)

            delta_rx = grid_pos_x - atom_crd_x
            delta_ry = grid_pos_y - atom_crd_y
            delta_rz = grid_pos_z - atom_crd_z

            if atom_charge != 0.0:
                fg_dist = math.sqrt(delta_rx**2 + delta_ry**2 + delta_rz**2)
                if fg_dist > approx_zero:
                    atom_charge_over_denom = atom_charge / (
                        fg_dist * fg_dist * fg_dist * indi_scaled
                    )
                    # del(coulomb_map)/del(x)
                    this_grid_coulomb_dx = delphi_real(
                        this_grid_coulomb_dx - delta_rx * atom_charge_over_denom
                    )
                    # del(coulomb_map)/del(y)
                    this_grid_coulomb_dy = delphi_real(
                        this_grid_coulomb_dy - delta_ry * atom_charge_over_denom
                    )
                    # del(coulomb_map)/del(z)
                    this_grid_coulomb_dz = delphi_real(
                        this_grid_coulomb_dz - delta_rz * atom_charge_over_denom
                    )
        grad_coulomb_map_1d[ijk1d_x_3] = this_grid_coulomb_dx
        grad_coulomb_map_1d[ijk1d_x_3 + 1] = this_grid_coulomb_dy
        grad_coulomb_map_1d[ijk1d_x_3 + 2] = this_grid_coulomb_dz


@cuda.jit(cache=True)
def _cuda_calc_grad_coulomb_map(
    grid_spacing: delphi_real,
    indi_scaled: delphi_real,
    approx_zero: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    grad_coulomb_map_1d: np.ndarray[delphi_real],
) -> None:
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    num_grid_points = grid_shape[0] * x_stride
    ijk1d = cuda.grid(1)

    if ijk1d < num_grid_points:
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3 = 3 * ijk1d

        grid_pos_x = i * grid_spacing + grid_origin[0]
        grid_pos_y = j * grid_spacing + grid_origin[1]
        grid_pos_z = k * grid_spacing + grid_origin[2]

        this_grid_coulomb_dx: delphi_real = 0.0
        this_grid_coulomb_dy: delphi_real = 0.0
        this_grid_coulomb_dz: delphi_real = 0.0
        for this_atom in atoms_data:
            atom_crd_x = cu_get_atom_x(this_atom)
            atom_crd_y = cu_get_atom_y(this_atom)
            atom_crd_z = cu_get_atom_z(this_atom)
            atom_charge = cu_get_atom_charge(this_atom)

            delta_rx = grid_pos_x - atom_crd_x
            delta_ry = grid_pos_y - atom_crd_y
            delta_rz = grid_pos_z - atom_crd_z

            if atom_charge != 0.0:
                gridpoint_dist = math.sqrt(
                    delta_rx * delta_rx + delta_ry * delta_ry + delta_rz * delta_rz
                )
                if gridpoint_dist > approx_zero:
                    atom_charge_over_denom = atom_charge / (
                        (gridpoint_dist**3) * indi_scaled
                    )
                    # del(coulomb_map)/del(x)
                    this_grid_coulomb_dx = delphi_real(
                        this_grid_coulomb_dx - delta_rx * atom_charge_over_denom
                    )
                    # del(coulomb_map)/del(y)
                    this_grid_coulomb_dy = delphi_real(
                        this_grid_coulomb_dy - delta_ry * atom_charge_over_denom
                    )
                    # del(coulomb_map)/del(z)
                    this_grid_coulomb_dz = delphi_real(
                        this_grid_coulomb_dz - delta_rz * atom_charge_over_denom
                    )
        grad_coulomb_map_1d[ijk1d_x_3] = this_grid_coulomb_dx
        grad_coulomb_map_1d[ijk1d_x_3 + 1] = this_grid_coulomb_dy
        grad_coulomb_map_1d[ijk1d_x_3 + 2] = this_grid_coulomb_dz


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_grad_epsilon_dot_coulomb_map(
    grad_epsmap_1d: np.ndarray[delphi_real],
    grad_coulomb_map_1d: np.ndarray[delphi_real],
    eps_dot_coul_map_1d: np.ndarray[delphi_real],
) -> None:
    for ijk1d in prange(eps_dot_coul_map_1d.size):
        ijk1d_x_3 = 3 * ijk1d
        eps_dot_coul_map_1d[ijk1d] = (
            grad_epsmap_1d[ijk1d_x_3] * grad_coulomb_map_1d[ijk1d_x_3]
            + grad_epsmap_1d[ijk1d_x_3 + 1] * grad_coulomb_map_1d[ijk1d_x_3 + 1]
            + grad_epsmap_1d[ijk1d_x_3 + 2] * grad_coulomb_map_1d[ijk1d_x_3 + 2]
        )


@cuda.jit(cache=True)
def _cuda_grad_epsilon_dot_coulomb_map(
    grad_epsmap_1d: np.ndarray[delphi_real],
    grad_coulomb_map_1d: np.ndarray[delphi_real],
    eps_dot_coul_map_1d: np.ndarray[delphi_real],
) -> None:
    ijk1d = cuda.grid(1)
    if ijk1d < eps_dot_coul_map_1d.size:
        ijk1d_x_3 = 3 * ijk1d
        eps_dot_coul_map_1d[ijk1d] = (
            grad_epsmap_1d[ijk1d_x_3] * grad_coulomb_map_1d[ijk1d_x_3]
            + grad_epsmap_1d[ijk1d_x_3 + 1] * grad_coulomb_map_1d[ijk1d_x_3 + 1]
            + grad_epsmap_1d[ijk1d_x_3 + 2] * grad_coulomb_map_1d[ijk1d_x_3 + 2]
        )


@cuda.jit(cache=True)
def _cuda_calc_grad_epsilon_in_map(
    gaussian_exponent: delphi_int,
    grid_spacing: delphi_real,
    diff_gap_indi: delphi_real,
    approx_zero: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    density_gridpoint_map_1d: np.ndarray[delphi_real],
    grad_epsin_map_1d: np.ndarray[delphi_real],
):
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride
    ijk1d = cuda.grid(1)
    if ijk1d < n_grid_points:
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3 = 3 * ijk1d
        grid_pos_x = i * grid_spacing + grid_origin[0]
        grid_pos_y = j * grid_spacing + grid_origin[1]
        grid_pos_z = k * grid_spacing + grid_origin[2]

        this_epsin_map_dx = 0.0
        this_epsin_map_dy = 0.0
        this_epsin_map_dz = 0.0
        gaussian_exponent_minus_1 = gaussian_exponent - 1
        gaussian_exponent_x_2 = gaussian_exponent * 2
        total_density = density_gridpoint_map_1d[ijk1d]
        for this_atom in atoms_data:
            atom_crd_x = cu_get_atom_x(this_atom)
            atom_crd_y = cu_get_atom_y(this_atom)
            atom_crd_z = cu_get_atom_z(this_atom)
            atom_radius = cu_get_atom_radius(this_atom)
            atom_sigma = cu_get_atom_gaussiansigma(this_atom)
            atom_sigma_x_atom_radius_square = (
                atom_sigma * atom_sigma * atom_radius * atom_radius
            )
            delta_rx = grid_pos_x - atom_crd_x
            delta_ry = grid_pos_y - atom_crd_y
            delta_rz = grid_pos_z - atom_crd_z
            dist_square = (
                delta_rx * delta_rx + delta_ry * delta_ry + delta_rz * delta_rz
            )
            if dist_square < approx_zero:
                continue
            dist_factor = dist_square**gaussian_exponent_minus_1
            atom_factor = gaussian_exponent_x_2 / (
                atom_sigma_x_atom_radius_square**gaussian_exponent
            )
            density = math.exp(
                -((dist_square / atom_sigma_x_atom_radius_square) ** gaussian_exponent)
            )

            if 1 - density > approx_zero:
                density_factor = density * (1 - total_density) / (1 - density)
                all_but_crd_factor = (
                    diff_gap_indi * atom_factor * density_factor * dist_factor
                )
                this_epsin_map_dx = this_epsin_map_dx + delta_rx * all_but_crd_factor
                this_epsin_map_dy = this_epsin_map_dy + delta_ry * all_but_crd_factor
                this_epsin_map_dz = this_epsin_map_dz + delta_rz * all_but_crd_factor
        grad_epsin_map_1d[ijk1d_x_3] = delphi_real(this_epsin_map_dx)
        grad_epsin_map_1d[ijk1d_x_3 + 1] = delphi_real(this_epsin_map_dy)
        grad_epsin_map_1d[ijk1d_x_3 + 2] = delphi_real(this_epsin_map_dz)


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_calc_grad_epsilon_in_map(
    gaussian_exponent: delphi_int,
    grid_spacing: delphi_real,
    diff_gap_indi: delphi_real,
    approx_zero: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    density_gridpoint_map_1d: np.ndarray[delphi_real],
    grad_epsin_map_1d: np.ndarray[delphi_real],
):
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride

    for ijk1d in prange(n_grid_points):
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3 = 3 * ijk1d
        grid_pos_x = i * grid_spacing + grid_origin[0]
        grid_pos_y = j * grid_spacing + grid_origin[1]
        grid_pos_z = k * grid_spacing + grid_origin[2]

        this_epsin_map_dx = 0.0
        this_epsin_map_dy = 0.0
        this_epsin_map_dz = 0.0

        total_density = density_gridpoint_map_1d[ijk1d]
        for this_atom in atoms_data:
            atom_crd_x = get_atom_x(this_atom)
            atom_crd_y = get_atom_y(this_atom)
            atom_crd_z = get_atom_z(this_atom)
            atom_radius = get_atom_radius(this_atom)
            atom_sigma = get_atom_gaussiansigma(this_atom)
            delta_rx = grid_pos_x - atom_crd_x
            delta_ry = grid_pos_y - atom_crd_y
            delta_rz = grid_pos_z - atom_crd_z
            dist_square = delta_rx**2 + delta_ry**2 + delta_rz**2
            if dist_square < approx_zero:
                continue
            dist_factor = dist_square ** (gaussian_exponent - 1)
            atom_factor = (2 * gaussian_exponent) / (
                (atom_sigma * atom_radius) ** (2 * gaussian_exponent)
            )
            density = math.exp(
                -(
                    (dist_square / ((atom_sigma * atom_radius) ** 2))
                    ** gaussian_exponent
                )
            )

            if 1 - density > approx_zero:
                density_factor = density * (1 - total_density) / (1 - density)
                all_but_crd_factor = (
                    diff_gap_indi * atom_factor * density_factor * dist_factor
                )
                # if i==29 and j==23 and k==22:
                #     print("atom_factor * density_factor * dist_factor * srf_func_dr * (1/total_density**2)=", atom_factor , density_factor , dist_factor, srf_func_dr , (1/total_density**2), srf_square, delta_r)
                #     print(surfacemap_dz[i][j][k], srf_factor * srf_square * delta_r[delphi_int(2)])
                #     print("total_density=", total_density, "density=", density, srf_factor, srf_square, delta_r)
                #     print()
                this_epsin_map_dx = this_epsin_map_dx + delta_rx * all_but_crd_factor
                this_epsin_map_dy = this_epsin_map_dy + delta_ry * all_but_crd_factor
                this_epsin_map_dz = this_epsin_map_dz + delta_rz * all_but_crd_factor
        grad_epsin_map_1d[ijk1d_x_3] = delphi_real(this_epsin_map_dx)
        grad_epsin_map_1d[ijk1d_x_3 + 1] = delphi_real(this_epsin_map_dy)
        grad_epsin_map_1d[ijk1d_x_3 + 2] = delphi_real(this_epsin_map_dz)


@njit(nogil=True, boundscheck=False, parallel=True, fastmath=True, cache=True)
def _cpu_setup_coulombic_boundary_condition(
    vacuum: delphi_bool,
    grid_spacing: delphi_real,
    exdi_scaled: delphi_real,
    indi_scaled: delphi_real,
    debye_length: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    atoms_data: np.ndarray[delphi_real],
    coulomb_map_1d: np.ndarray[delphi_real],
    phimap_1d: np.ndarray[delphi_real],
) -> None:
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
        # these are non-boundary points
        if not (
            0 < ix < last_grid_id_x
            and 0 < jy < last_grid_id_y
            and 0 < kz < last_grid_id_z
        ):
            this_grid_phimap = 0.0
            for this_atom in atoms_data:
                atom_charge = get_atom_charge(this_atom)
                if abs(atom_charge) > delphi_real(ConstDelPhi.ApproxZero.value):
                    grid_dx = ix - get_atom_grid_x(this_atom)
                    grid_dy = jy - get_atom_grid_y(this_atom)
                    grid_dz = kz - get_atom_grid_z(this_atom)
                    grid_dist_square = (
                        grid_dx * grid_dx + grid_dy * grid_dy + grid_dz * grid_dz
                    )
                    gridpoint_dist = delphi_real(
                        math.sqrt(grid_dist_square) * grid_spacing
                    )
                    tmpval = atom_charge
                    if not vacuum:
                        tmpval *= math.exp(-gridpoint_dist * debye_length_inverse)
                    tmpval = tmpval / (gridpoint_dist * epsilon_temp)
                    this_grid_phimap += tmpval
            phimap_1d[ijk1d] = this_grid_phimap - coulomb_map_1d[ijk1d]
        else:
            phimap_1d[ijk1d] = 0.0


@cuda.jit(cache=True,fastmath=True)
def _cuda_setup_coulombic_boundary_condition(
    vacuum: delphi_bool,
    grid_spacing: delphi_real,
    exdi_scaled: delphi_real,
    indi_scaled: delphi_real,
    debye_length: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    atoms_data: np.ndarray[delphi_real],
    coulomb_map_1d: np.ndarray[delphi_real],
    phimap_1d: np.ndarray[delphi_real],
) -> None:
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

    ijk1d = cuda.grid(1)
    if ijk1d < n_grid_points:
        ix = ijk1d // x_stride
        jy = (ijk1d - ix * x_stride) // y_stride
        kz = ijk1d - ix * x_stride - jy * y_stride
        # these are non-boundary points
        if not (
            0 < ix < last_grid_id_x
            and 0 < jy < last_grid_id_y
            and 0 < kz < last_grid_id_z
        ):
            this_grid_phimap = 0.0
            for this_atom in atoms_data:
                atom_charge = cu_get_atom_charge(this_atom)
                if abs(atom_charge) > delphi_real(ConstDelPhi.ApproxZero.value):
                    grid_dx = ix - cu_get_atom_grid_x(this_atom)
                    grid_dy = jy - cu_get_atom_grid_y(this_atom)
                    grid_dz = kz - cu_get_atom_grid_z(this_atom)
                    grid_dist_square = (
                        grid_dx * grid_dx + grid_dy * grid_dy + grid_dz * grid_dz
                    )
                    gridpoint_dist = delphi_real(
                        math.sqrt(grid_dist_square) * grid_spacing
                    )
                    tmpval = atom_charge
                    if not vacuum:
                        tmpval *= math.exp(-gridpoint_dist * debye_length_inverse)
                    tmpval = tmpval / (gridpoint_dist * epsilon_temp)
                    this_grid_phimap += tmpval
            phimap_1d[ijk1d] = this_grid_phimap - coulomb_map_1d[ijk1d]
        else:
            phimap_1d[ijk1d] = 0.0


@njit(nogil=True, boundscheck=False, parallel=True, fastmath=True, cache=True)
def _cpu_setup_dipolar_boundary_condition(
    vacuum: delphi_bool,
    grid_spacing: delphi_real,
    exdi_scaled: delphi_real,
    indi_scaled: delphi_real,
    debye_length: delphi_real,
    total_pve_charge: delphi_real,
    total_nve_charge: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    centroid_pve_charge: np.ndarray[delphi_real],
    centroid_nve_charge: np.ndarray[delphi_real],
    coulomb_map_1d: np.ndarray[delphi_real],
    phimap_1d: np.ndarray[delphi_real],
) -> None:
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

    has_pve_charges, has_nve_charges = False, False
    grid_centroid_pve_charge = None
    grid_centroid_nve_charge = None

    if not centroid_pve_charge is None:
        has_pve_charges = True
    if not centroid_nve_charge is None:
        has_nve_charges = True
    if has_pve_charges:
        grid_centroid_pve_charge = to_grid_coords(
            centroid_pve_charge, grid_origin, grid_spacing
        )
    if has_nve_charges:
        grid_centroid_nve_charge = to_grid_coords(
            centroid_nve_charge, grid_origin, grid_spacing
        )
    for ijk1d in prange(n_grid_points):
        ix = ijk1d // x_stride
        jy = (ijk1d - ix * x_stride) // y_stride
        kz = ijk1d - ix * x_stride - jy * y_stride
        # these are non-boundary points
        if not (
            0 < ix < last_grid_id_x
            and 0 < jy < last_grid_id_y
            and 0 < kz < last_grid_id_z
        ):
            phimap_1d[ijk1d] = 0.0
            if has_pve_charges:
                grid_pve_dx = ix - grid_centroid_pve_charge[0]
                grid_pve_dy = jy - grid_centroid_pve_charge[1]
                grid_pve_dz = kz - grid_centroid_pve_charge[2]
                grid_pve_dist_square = (
                    grid_pve_dx * grid_pve_dx
                    + grid_pve_dy * grid_pve_dy
                    + grid_pve_dz * grid_pve_dz
                )
                fdist_pve = delphi_real(math.sqrt(grid_pve_dist_square) * grid_spacing)
                fphi_pve = total_pve_charge / (fdist_pve * epsilon_temp)
                if not vacuum:
                    fphi_pve *= math.exp(-fdist_pve * debye_length_inverse)
                phimap_1d[ijk1d] += fphi_pve
            if has_nve_charges:
                grid_nve_dx = ix - grid_centroid_nve_charge[0]
                grid_nve_dy = jy - grid_centroid_nve_charge[1]
                grid_nve_dz = kz - grid_centroid_nve_charge[2]
                grid_nve_dist_square = (
                    grid_nve_dx * grid_nve_dx
                    + grid_nve_dy * grid_nve_dy
                    + grid_nve_dz * grid_nve_dz
                )
                fdist_nve = delphi_real(math.sqrt(grid_nve_dist_square) * grid_spacing)
                # fdist_nve = distance(grid_3d_indices, grid_centroid_nve_charge)
                fphi_nve = total_nve_charge / (fdist_nve * epsilon_temp)
                if not vacuum:
                    fphi_nve *= math.exp(-fdist_nve * debye_length_inverse)
                phimap_1d[ijk1d] += fphi_nve
            phimap_1d[ijk1d] -= coulomb_map_1d[ijk1d]
        else:
            phimap_1d[ijk1d] = 0.0


@njit(nogil=True, boundscheck=False, parallel=True, fastmath=True, cache=True)
def _cuda_setup_dipolar_boundary_condition(
    vacuum: delphi_bool,
    grid_spacing: delphi_real,
    exdi_scaled: delphi_real,
    indi_scaled: delphi_real,
    debye_length: delphi_real,
    total_pve_charge: delphi_real,
    total_nve_charge: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    centroid_pve_charge: np.ndarray[delphi_real],
    centroid_nve_charge: np.ndarray[delphi_real],
    coulomb_map_1d: np.ndarray[delphi_real],
    phimap_1d: np.ndarray[delphi_real],
) -> None:
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

    has_pve_charges, has_nve_charges = False, False
    grid_centroid_pve_charge = None
    grid_centroid_nve_charge = None

    if not centroid_pve_charge is None:
        has_pve_charges = True
    if not centroid_nve_charge is None:
        has_nve_charges = True
    if has_pve_charges:
        grid_centroid_pve_charge = np.zeros(3, dtype=delphi_real)
        cu_to_grid_coords(
            centroid_pve_charge, grid_centroid_pve_charge, grid_origin, grid_spacing
        )
    if has_nve_charges:
        grid_centroid_nve_charge = np.zeros(3, dtype=delphi_real)
        cu_to_grid_coords(
            centroid_nve_charge, grid_centroid_nve_charge, grid_origin, grid_spacing
        )

    ijk1d = cuda.grid(1)
    if ijk1d < n_grid_points:
        ix = ijk1d // x_stride
        jy = (ijk1d - ix * x_stride) // y_stride
        kz = ijk1d - ix * x_stride - jy * y_stride
        # these are non-boundary points
        if not (
            0 < ix < last_grid_id_x
            and 0 < jy < last_grid_id_y
            and 0 < kz < last_grid_id_z
        ):
            phimap_1d[ijk1d] = 0.0
            if has_pve_charges:
                grid_pve_dx = ix - grid_centroid_pve_charge[0]
                grid_pve_dy = jy - grid_centroid_pve_charge[1]
                grid_pve_dz = kz - grid_centroid_pve_charge[2]
                grid_pve_dist_square = (
                    grid_pve_dx * grid_pve_dx
                    + grid_pve_dy * grid_pve_dy
                    + grid_pve_dz * grid_pve_dz
                )
                dist_pve = delphi_real(math.sqrt(grid_pve_dist_square) * grid_spacing)
                phi_pve = total_pve_charge / (dist_pve * epsilon_temp)
                if not vacuum:
                    phi_pve *= math.exp(-dist_pve * debye_length_inverse)
                phimap_1d[ijk1d] += phi_pve
            if has_nve_charges:
                grid_nve_dx = ix - grid_centroid_nve_charge[0]
                grid_nve_dy = jy - grid_centroid_nve_charge[1]
                grid_nve_dz = kz - grid_centroid_nve_charge[2]
                grid_nve_dist_square = (
                    grid_nve_dx * grid_nve_dx
                    + grid_nve_dy * grid_nve_dy
                    + grid_nve_dz * grid_nve_dz
                )
                dist_nve = delphi_real(math.sqrt(grid_nve_dist_square) * grid_spacing)
                phi_nve = total_nve_charge / (dist_nve * epsilon_temp)
                if not vacuum:
                    phi_nve *= math.exp(-dist_nve * debye_length_inverse)
                phimap_1d[ijk1d] += phi_nve
            phimap_1d[ijk1d] -= coulomb_map_1d[ijk1d]
        else:
            phimap_1d[ijk1d] = 0.0


@cuda.jit(cache=True)
def _cuda_prepare_charge_neigh_eps_sum_to_iterate(
    vacuum: delphi_bool,
    exdi: delphi_real,
    grid_spacing: delphi_real,
    debye_length: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    solute_surface_map_1d: np.ndarray[delphi_real],
    epsmap_midpoints_1d: np.ndarray[delphi_real],
    coulomb_map_1d: np.ndarray[delphi_real],
    charge_map_1d: np.ndarray[delphi_real],
    eps_nd_midpoint_neighs_sum_1d: np.ndarray[delphi_real],
    boundary_gridpoints_1d: np.ndarray[delphi_bool],
):
    grid_spacing_square = grid_spacing**2
    kappa_square = exdi / debye_length**2
    kappa_x_grid_spacing_wholesquare = kappa_square * grid_spacing_square
    epsout = 1 if vacuum != 0 else exdi
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride
    y_stride_x_3 = 3 * y_stride
    x_stride_x_3 = 3 * x_stride

    last_grid_id_x = delphi_int(grid_shape[0] - 1)
    last_grid_id_y = delphi_int(grid_shape[1] - 1)
    last_grid_id_z = delphi_int(grid_shape[2] - 1)

    ijk1d = cuda.grid(1)
    if ijk1d < n_grid_points:
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3 = 3 * ijk1d
        if (
            i == 0
            or j == 0
            or k == 0
            or i == last_grid_id_x
            or j == last_grid_id_y
            or k == last_grid_id_z
        ):
            boundary_gridpoints_1d[ijk1d] = BOX_BOUNDARY
            eps_nd_midpoint_neighs_sum_1d[ijk1d] = 6 * epsout
        else:
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

            eps_nd_midpoint_neighs_sum_1d[ijk1d] = (
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

    if vacuum == 0 and debye_length != delphi_real(
        ConstDelPhi.ZeroMolarSaltDebyeLength.value
    ):
        if ijk1d < n_grid_points:
            # add the: h^2.\kappa^2 term to the \sigma_{j=1}^{6}{eps_j} in the denomrator
            # i.e. epsmap_node_and_midpoints[ijk_1d][7] of iteration formula.
            eps_nd_midpoint_neighs_sum_1d[ijk1d] += kappa_x_grid_spacing_wholesquare * (
                1 - solute_surface_map_1d[ijk1d]
            )
            # Update the charge-source with the green function for water case with
            # in regions accessible to salt at non-boundary grid points .
            if boundary_gridpoints_1d[ijk1d] != BOX_BOUNDARY:
                charge_map_1d[ijk1d] -= (
                    (1 - solute_surface_map_1d[ijk1d])
                    * kappa_square
                    * coulomb_map_1d[ijk1d]
                )
    # Note: charge scaling is needed for both vacuum & water phase
    if ijk1d < n_grid_points:
        charge_map_1d[ijk1d] *= grid_spacing_square


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_prepare_charge_neigh_eps_sum_to_iterate(
    vacuum: delphi_bool,
    non_zero_salt: delphi_bool,
    ion_exclusion_method_int_value: delphi_int,
    exdi: delphi_real,
    ion_radius: delphi_real,
    ions_valance: delphi_real,
    grid_spacing: delphi_real,
    debye_length: delphi_real,
    epkt: delphi_real,
    grid_shape: np.ndarray[delphi_real],
    solute_surface_map_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_real],
    epsilon_map_1d: np.ndarray[delphi_real],
    epsmap_midpoints_1d: np.ndarray[delphi_real],
    coulomb_map_1d: np.ndarray[delphi_real],
    charge_map_1d: np.ndarray[delphi_real],
    eps_midpoint_neighs_sum_plus_salt_screening_1d: np.ndarray[delphi_real],
    boundary_gridpoints_1d: np.ndarray[delphi_bool],
):
    grid_spacing_square = grid_spacing**2
    kappa_square = exdi / debye_length**2
    kappa_x_grid_spacing_wholesquare = kappa_square * grid_spacing_square
    epsout = 1 if vacuum != 0 else exdi
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride
    y_stride_x_3 = 3 * y_stride
    x_stride_x_3 = 3 * x_stride

    last_grid_id_x = delphi_int(grid_shape[0] - 1)
    last_grid_id_y = delphi_int(grid_shape[1] - 1)
    last_grid_id_z = delphi_int(grid_shape[2] - 1)

    for ijk1d in prange(n_grid_points):
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3 = 3 * ijk1d
        if (
            i == 0
            or j == 0
            or k == 0
            or i == last_grid_id_x
            or j == last_grid_id_y
            or k == last_grid_id_z
        ):
            boundary_gridpoints_1d[ijk1d] = BOX_BOUNDARY
            eps_midpoint_neighs_sum_plus_salt_screening_1d[ijk1d] = 6 * epsout
        else:
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

    if (not vacuum) and non_zero_salt:
        if ion_exclusion_method_int_value == SOLUTE_SURFACE:
            for ijk1d in prange(n_grid_points):
                # add the: h^2.\kappa^2 term to the \sigma_{j=1}^{6}{eps_j} in the denomrator
                # i.e. epsmap_node_and_midpoints[ijk_1d][7] of iteration formula.
                eps_midpoint_neighs_sum_plus_salt_screening_1d[
                    ijk1d
                ] += kappa_x_grid_spacing_wholesquare * (
                    1 - solute_surface_map_1d[ijk1d]
                )
                # Update the charge-source with the green function for water case with
                # in regions accessible to salt at non-boundary grid points.
                if boundary_gridpoints_1d[ijk1d] != BOX_BOUNDARY:
                    charge_map_1d[ijk1d] -= (
                        (1 - solute_surface_map_1d[ijk1d])
                        * kappa_square
                        * coulomb_map_1d[ijk1d]
                    )
        elif ion_exclusion_method_int_value == GAUSSIAN_LAYER:
            penalty_factor = epkt * ((ions_valance**2) * 1 / (2.0 * ion_radius))
            inverse_epsout = 1.0 / epsout
            for ijk1d in prange(n_grid_points):
                energy_factor = 1.0 / epsilon_map_1d[ijk1d] - inverse_epsout
                boltz_factor_energy_penalty = math.exp(-penalty_factor * energy_factor)
                if boltz_factor_energy_penalty < APPROX_ZERO:
                    boltz_factor_energy_penalty = 0.0
                screening_factor = (
                    kappa_x_grid_spacing_wholesquare * boltz_factor_energy_penalty
                )
                eps_midpoint_neighs_sum_plus_salt_screening_1d[
                    ijk1d
                ] += screening_factor
                if boundary_gridpoints_1d[ijk1d] != BOX_BOUNDARY:
                    charge_map_1d[ijk1d] -= (
                        (1 - solute_surface_map_1d[ijk1d])
                        * kappa_square
                        * coulomb_map_1d[ijk1d]
                    )
        elif ion_exclusion_method_int_value == STERN_LAYER:
            for ijk1d in prange(n_grid_points):
                eps_midpoint_neighs_sum_plus_salt_screening_1d[
                    ijk1d
                ] += kappa_x_grid_spacing_wholesquare * (
                    1 - ion_exclusion_map_1d[ijk1d]
                )
                if boundary_gridpoints_1d[ijk1d] != BOX_BOUNDARY:
                    charge_map_1d[ijk1d] -= (
                        (1 - solute_surface_map_1d[ijk1d])
                        * kappa_square
                        * coulomb_map_1d[ijk1d]
                    )

    # Note: charge scaling is needed for both vacuum & water phase
    for ijk1d in prange(n_grid_points):
        charge_map_1d[ijk1d] *= grid_spacing_square


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
        # nprint_cpu(DEBUG,_VERBOSITY, "kappa_x_grid_spacing_wholesquare:", kappa_x_grid_spacing_wholesquare)
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
