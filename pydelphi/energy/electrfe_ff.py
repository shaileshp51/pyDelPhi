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

def phase_surf_integral_energy(
    phimap_phase_w, phimap_phase_v, eps_boxsurf_phase2, grid_scale, grid_shape
):

    x_face_integral = 0.0
    y_face_integral = 0.0
    z_face_integral = 0.0

    grid_spacing = 1.0 / grid_scale
    inv_grid_spacing_x_2 = 1.0 / (grid_spacing * 2.0)

    for j in range(1, grid_shape[1] - 1):
        temp_sum = 0.0
        for k in range(1, grid_shape[2] - 1):
            sum_w1 = phimap_phase_w[0, j, k] + phimap_phase_w[1, j, k]
            diff_v1 = phimap_phase_v[0, j, k] - phimap_phase_v[1, j, k]

            sum_w2 = (
                phimap_phase_w[grid_shape[0] - 1, j, k]
                + phimap_phase_w[grid_shape[0] - 2, j, k]
            )
            diff_v2 = (
                phimap_phase_v[grid_shape[0] - 1, j, k]
                - phimap_phase_v[grid_shape[0] - 2, j, k]
            )

            term1 = sum_w1 * diff_v1 * inv_grid_spacing_x_2
            term2 = sum_w2 * diff_v2 * inv_grid_spacing_x_2

            temp_sum += term1
            temp_sum += term2

        x_face_integral += temp_sum

    for i in range(1, grid_shape[0] - 1):
        temp_sum = 0.0
        for k in range(1, grid_shape[2] - 1):
            sum_w1 = phimap_phase_w[i, 0, k] + phimap_phase_w[i, 1, k]
            diff_v1 = phimap_phase_v[i, 0, k] - phimap_phase_v[i, 1, k]

            sum_w2 = (
                phimap_phase_w[i, grid_shape[1] - 1, k]
                + phimap_phase_w[i, grid_shape[1] - 2, k]
            )
            diff_v2 = (
                phimap_phase_v[i, grid_shape[1] - 1, k]
                - phimap_phase_v[i, grid_shape[1] - 2, k]
            )

            term1 = sum_w1 * diff_v1 * inv_grid_spacing_x_2
            term2 = sum_w2 * diff_v2 * inv_grid_spacing_x_2

            temp_sum += term1
            temp_sum += term2

        y_face_integral += temp_sum

    for i in range(1, grid_shape[0] - 1):
        temp_sum = 0.0
        for j in range(1, grid_shape[1] - 1):
            sum_w1 = phimap_phase_w[i, j, 0] + phimap_phase_w[i, j, 1]
            diff_v1 = phimap_phase_v[i, j, 0] - phimap_phase_v[i, j, 1]

            sum_w2 = (
                phimap_phase_w[i, j, grid_shape[2] - 1]
                + phimap_phase_w[i, j, grid_shape[2] - 2]
            )
            diff_v2 = (
                phimap_phase_v[i, j, grid_shape[2] - 1]
                - phimap_phase_v[i, j, grid_shape[2] - 2]
            )

            term1 = sum_w1 * diff_v1 * inv_grid_spacing_x_2
            term2 = sum_w2 * diff_v2 * inv_grid_spacing_x_2

            temp_sum += term1
            temp_sum += term2

        z_face_integral += temp_sum

    total_integral = grid_spacing * grid_spacing(
        x_face_integral + y_face_integral + z_face_integral
    )

    return total_integral * eps_boxsurf_phase2


def integral_grad_u_grad_v(
    phimap_phase_w,
    phimap_phase_v,
    epsilon_v,
    eps_map,
    ion_exclusion_map,
    grid_scale,
    grid_shape,
):
    pass
