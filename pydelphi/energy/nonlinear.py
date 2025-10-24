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
This module calculates the non-linear Poisson-Boltzmann energy terms,
specifically the $\\rho \\cdot \\phi$ term (solvation energy) and the
osmotic pressure term, for a given electrostatic potential grid.

It implements the calculation based on a polynomial approximation of the
charge density ($\\rho$) and a related function for osmotic pressure,
iterating through grid cells and accounting for ion exclusion regions
and boundary conditions.
"""

import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads, cuda, float64


@njit(nogil=True, boundscheck=False, cache=True)
def energy_nonlinear_serial(
    grid_shape: np.ndarray,  # shape = (3,), dtype=int
    scale: float,
    grid_origin: np.ndarray,  # precomputed origin = (-offset / scale) + box_center
    ion_exclusion_map_1d: np.ndarray,  # shape = (nx, ny, nz), dtype=bool
    phimap: np.ndarray,  # shape = (nx, ny, nz), dtype=float
    taylor_coeffs: np.ndarray,  # shape = (5,), highest order first
    # ions_energy_enabled: bool,
    # grid_output_count: list,  # length-1 list for reference-like output
) -> float:
    nx, ny, nz = grid_shape
    y_stride = nz
    x_stride = ny * nz
    grid_spacing = 1.0 / scale
    cell_volume = grid_spacing**3

    # Precompute derivatives of the polynomial terms
    d_coeffs = -taylor_coeffs / np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    energy_rho_times_phi_over_2 = 0.0
    energy_osmotic = 0.0

    # Loop through each cell in the 3D grid
    for i in range(nx):
        cut_edge_x = 0.5 if i == 0 or i == nx - 1 else 1.0

        for j in range(ny):
            cut_edge_y = cut_edge_x * (0.5 if j == 0 or j == ny - 1 else 1.0)

            for k in range(nz):
                cut_edge = cut_edge_y * (0.5 if k == 0 or k == nz - 1 else 1.0)

                # Calculate the 1D index for the current (k, j, z) cell
                ijk1d = i * x_stride + j * y_stride + k

                # Check if the current cell is part of the ion exclusion map
                if ion_exclusion_map_1d[ijk1d]:
                    phi = phimap[i, j, k]

                    # Solvation charge density (ρ = P(phi))
                    # Using Horner's method: P(phi) = c4*phi^4 + c3*phi^3 + c2*phi^2 + c1*phi + c0
                    # For taylor_coeffs = [c4, c3, c2, c1, c0] (highest order first):
                    charge_density_horner = taylor_coeffs[4]  # c4
                    charge_density_horner = (
                        charge_density_horner * phi + taylor_coeffs[3]
                    )  # c4*phi + c3
                    charge_density_horner = (
                        charge_density_horner * phi + taylor_coeffs[2]
                    )  # ... + c2
                    charge_density_horner = (
                        charge_density_horner * phi + taylor_coeffs[1]
                    )  # ... + c1
                    charge_density_horner = (
                        charge_density_horner * phi + taylor_coeffs[0]
                    )  # ... + c0. This is P(phi).

                    # Calculate 'ionic_charge_in_solvent'
                    ionic_charge_in_solvent = (
                        cut_edge * charge_density_horner * phi * cell_volume
                    )

                    # Osmotic density (Q(phi))
                    # Using Horner's method: Q(phi) = d4*phi^4 + d3*phi^3 + d2*phi^2 + d1*phi + d0
                    # For d_coeffs = [d4, d3, d2, d1, d0] (highest order first):
                    osmotic_density_horner = d_coeffs[4]  # d4
                    osmotic_density_horner = (
                        osmotic_density_horner * phi + d_coeffs[3]
                    )  # d4*phi + d3
                    osmotic_density_horner = (
                        osmotic_density_horner * phi + d_coeffs[2]
                    )  # ... + d2
                    osmotic_density_horner = (
                        osmotic_density_horner * phi + d_coeffs[1]
                    )  # ... + d1
                    osmotic_density_horner = (
                        osmotic_density_horner * phi + d_coeffs[0]
                    )  # ... + d0. This is Q(phi).

                    # Accumulate osmotic energy (fEnergy_Osmetic in C++)
                    energy_osmotic += cut_edge * osmotic_density_horner * phi * phi

                    # Populate Numba-friendly output arrays
                    # Ensure grid_origin elements are handled as integers for array indexing if needed.
                    # k, j, z are already integers from range().
                    # output_grids[current_output_idx, 0] = k + int(grid_origin[0]) + 1
                    # output_grids[current_output_idx, 1] = j + int(grid_origin[1]) + 1
                    # output_grids[current_output_idx, 2] = z + int(grid_origin[2]) + 1
                    # output_values[current_output_idx] = charge
                    # current_output_idx += 1

                    # Accumulate rho_times_phi_over_2 energy (fEnergy_Solvation in C++)
                    energy_rho_times_phi_over_2 -= ionic_charge_in_solvent * phi

    # Set the actual count of output elements processed
    # grid_output_count[0] = current_output_idx

    # Apply the final scaling factors
    energy_unit_conversion_factor = 0.0006023
    energy_rho_times_phi_over_2 *= 0.5 * energy_unit_conversion_factor
    energy_osmotic *= -energy_unit_conversion_factor * cell_volume

    # grid_output_count[0] = len(output)

    return energy_rho_times_phi_over_2, energy_osmotic


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def energy_nonlinear(
    grid_shape,
    scale,
    grid_origin: np.ndarray,  # precomputed origin = (-offset / scale) + box_center
    ion_exclusion_map_1d,
    phimap,
    taylor_coeffs,
    nthreads,
):
    nx, ny, nz = grid_shape
    total_size = nx * ny * nz
    grid_spacing = 1.0 / scale
    cell_volume = grid_spacing**3

    d_coeffs = -taylor_coeffs / np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    # per-thread accumulators
    thread_rho_phi = np.zeros(nthreads, dtype=np.float64)
    thread_osmotic = np.zeros(nthreads, dtype=np.float64)

    # partition work by thread index
    for tid in prange(nthreads):
        start = (tid * total_size) // nthreads
        end = ((tid + 1) * total_size) // nthreads

        rho_phi_local = 0.0
        osmotic_local = 0.0

        for idx in range(start, end):
            i = idx // (ny * nz)
            j = (idx % (ny * nz)) // nz
            k = idx % nz

            cut_edge_x = 0.5 if i == 0 or i == nx - 1 else 1.0
            cut_edge_y = cut_edge_x * (0.5 if j == 0 or j == ny - 1 else 1.0)
            cut_edge = cut_edge_y * (0.5 if k == 0 or k == nz - 1 else 1.0)

            if ion_exclusion_map_1d[idx]:
                phi = phimap[i, j, k]

                # Horner eval for charge density
                charge_density = (
                    (
                        (taylor_coeffs[4] * phi + taylor_coeffs[3]) * phi
                        + taylor_coeffs[2]
                    )
                    * phi
                    + taylor_coeffs[1]
                ) * phi + taylor_coeffs[0]

                ionic_charge = cut_edge * charge_density * phi * cell_volume

                osmotic_density = (
                    ((d_coeffs[4] * phi + d_coeffs[3]) * phi + d_coeffs[2]) * phi
                    + d_coeffs[1]
                ) * phi + d_coeffs[0]

                osmotic_local += cut_edge * osmotic_density * phi * phi
                rho_phi_local += -ionic_charge * phi

        thread_rho_phi[tid] = rho_phi_local
        thread_osmotic[tid] = osmotic_local

    # reduction
    energy_rho_times_phi_over_2 = thread_rho_phi.sum()
    energy_osmotic = thread_osmotic.sum()

    # scaling
    energy_unit_conversion_factor = 0.0006023
    energy_rho_times_phi_over_2 *= 0.5 * energy_unit_conversion_factor
    energy_osmotic *= -energy_unit_conversion_factor * cell_volume

    return energy_rho_times_phi_over_2, energy_osmotic
