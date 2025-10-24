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
This module provides functions for calculating reaction field energy, which
arises from the interaction between fixed charges (atoms) and induced charges
on a dielectric boundary. This is a crucial component in continuum electrostatics
calculations (e.g., Poisson-Boltzmann equation) for determining solvation energy.

The module handles:
1. Identifying charged grid points that lie on the dielectric boundary.
2. Calculating the induced surface charges at these boundary points based
   on the local electrostatic potential gradient.
3. Computing the reaction field energy by summing the interactions between
   the fixed atomic charges and the induced surface charges.

It leverages Numba for performance optimization, with parallelization where
applicable, and includes fallbacks if Numba is not available.

Key functions:
- `_calc_charged_boundary_gridpoints`: Filters charged grid points to find
  those on the dielectric boundary and computes an adjustment factor.
- `_calculate_induced_surface_charges`: Computes the actual induced charge
  values on the dielectric boundary based on potentials.
- `_calculate_rf_energy_outer_surf` / `_calculate_rf_energy_outer_atom`:
  Core kernels for summing reaction field energy, optimized based on
  the number of atoms vs. surface charges.
- `_calc_induced_charge_rf_energy_helper`: A helper to dispatch to the
  appropriate reaction field energy kernel and apply final scaling.
- `calc_induced_charge_rf_energy`: The main high-level function for
  calculating the reaction field energy and the induced charges.
- `calc_reactionfield_energy`: A general function to calculate reaction
  field energy from pre-existing charged grid points and a potential map.
"""

import math
import time
import numpy as np

from pydelphi.config.global_runtime import (
    nprint_cpu_if_verbose as nprint_cpu,
    vprint,
)

from pydelphi.config.logging_config import DEBUG, get_effective_verbosity, INFO

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

# Ensure Numba and necessary submodules are imported correctly
_numba_usable = False
try:
    from numba import jit, njit, prange, types, float64, int64, set_num_threads, cuda

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

from pydelphi.foundation.platforms import Platform
from pydelphi.constants import ConstPhysical, ConstDelPhiFloats
from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_CHARGE,
)

# GPCHRG: Constants for grid point charge fields
from pydelphi.constants import (
    GPCHRGFIELD_INDX_1D,  # Index of the 1D grid point index
    GPCHRGFIELD_CHARGE,  # Index of the charge at the grid point
    GPCHRGFIELD_INDX_X,  # Index of the x-coordinate of the grid point
    GPCHRGFIELD_INDX_Y,  # Index of the y-coordinate of the grid point
    GPCHRGFIELD_INDX_Z,  # Index of the z-coordinate of the grid point
)

APPROX_ZERO = ConstDelPhiFloats.ApproxZero.value
FOUR_PI = ConstPhysical.FourPi.value
ONE_SIXTH = ConstPhysical.Sixth.value
RXN_ENERGY_FACTOR = ConstPhysical.ReactionFieldEnergyFactor.value

from pydelphi.utils.utils import (
    is_contained_in_sorted,
)


@njit(nogil=True, boundscheck=False, cache=True)
def _calculate_induced_surface_charges(
    charged_bgp_info_accum,  # Input: array [index_1d, adjustment_factor]
    diel_boundary_gridpoints,
    phimap_in_solvent,
    grid_shape,
    scale,
    epkt,
):
    """
    Calculates the induced surface charges at the dielectric boundary.

    Args:
        charged_bgp_info_accum (np.ndarray): A 2D array of shape (K, 2) containing [index_1d, adjustment_factor]
                                             for charged grid points on the boundary.
        diel_boundary_gridpoints (np.ndarray): A 2D array of shape (M, 3) containing [ix, iy, iz] of the dielectric boundary grid points.
        phimap_in_solvent (np.ndarray): A 3D array representing the electrostatic potential map in the solvent.
        grid_shape (np.ndarray): Array representing the shape of the grid (nx, ny, nz).
        scale (float): The grid scale factor.
        epkt (float): The thermal energy scale factor.

    Returns:
        np.ndarray: A 1D array of shape (M * 4) containing [ix, iy, iz, charge] for each boundary grid point.
    """
    num_boundary_gridpoints = diel_boundary_gridpoints.shape[0]
    # Output needs space for ix, iy, iz, charge
    induced_surf_charges_flat = np.zeros(num_boundary_gridpoints * 4, dtype=np.float64)
    energy_factor = RXN_ENERGY_FACTOR / (
        2.0 * scale * epkt
    )  # Corrected constant? Check definition
    phi_neighbors = np.zeros(7, dtype=np.float64)

    # --- Pre-sort the charged_bgp_info_accum by index_1d (col 0) ---
    charged_bgp_info_sorted = charged_bgp_info_accum

    nx, ny, nz = grid_shape.astype(int64)
    x_stride, y_stride = np.int64(ny * nz), np.int64(nz)

    missing_neighbors_count = 0  # Local counter for sequential loop

    # This loop is sequential
    for i in range(num_boundary_gridpoints):
        i_start = i * 4
        # Ensure indices from boundary points are integers
        ix = int(diel_boundary_gridpoints[i, 0])
        iy = int(diel_boundary_gridpoints[i, 1])
        iz = int(diel_boundary_gridpoints[i, 2])

        # Basic check if boundary point itself is valid
        if not (0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz):
            # Handle invalid boundary point? Skip or set charge to 0?
            induced_surf_charges_flat[i_start + 3] = 0.0
            continue  # Skip this boundary point

        index_1d = ix * x_stride + iy * y_stride + iz

        induced_surf_charges_flat[i_start + 0] = float(ix)
        induced_surf_charges_flat[i_start + 1] = float(iy)
        induced_surf_charges_flat[i_start + 2] = float(iz)

        phi_neighbors[:] = 0.0  # Reset for current iteration
        num_valid_neighbors = 0
        phi_neigh_sum = 0.0

        # Check neighbors and sum potentials
        # X neighbors
        if ix > 0:
            phi_neighbors[0] = phimap_in_solvent[ix - 1, iy, iz]
            num_valid_neighbors += 1
        if ix < nx - 1:
            phi_neighbors[1] = phimap_in_solvent[ix + 1, iy, iz]
            num_valid_neighbors += 1
        # Y neighbors
        if iy > 0:
            phi_neighbors[2] = phimap_in_solvent[ix, iy - 1, iz]
            num_valid_neighbors += 1
        if iy < ny - 1:
            phi_neighbors[3] = phimap_in_solvent[ix, iy + 1, iz]
            num_valid_neighbors += 1
        # Z neighbors
        if iz > 0:
            phi_neighbors[4] = phimap_in_solvent[ix, iy, iz - 1]
            num_valid_neighbors += 1
        if iz < nz - 1:
            phi_neighbors[5] = phimap_in_solvent[ix, iy, iz + 1]
            num_valid_neighbors += 1

        phi_neigh_sum = np.sum(phi_neighbors[0:6])  # Sum only the 6 neighbor slots

        # Current grid point potential
        phi_center = phimap_in_solvent[ix, iy, iz]

        if num_valid_neighbors > 0:
            # Normalize by actual number of valid neighbors found
            phi_effective = phi_center - (phi_neigh_sum / num_valid_neighbors)
            if num_valid_neighbors < 6:
                missing_neighbors_count += (
                    1  # Count points processed with missing neighbors
                )
        else:  # Should not happen for interior points, maybe edge cases?
            phi_effective = phi_center

        # If the boundary gridpoint is also a charged point, adjust potential
        # Use the pre-sorted accumulated charged boundary point info
        is_charged_bgp, index_charged_bgp = is_contained_in_sorted(
            charged_bgp_info_sorted,
            index_1d,
            column_index=0,  # Search by index_1d
        )
        if is_charged_bgp:
            # Get the pre-calculated charge factor (col 1 in accumulated array)
            charge_adjustment_factor = charged_bgp_info_sorted[index_charged_bgp, 1]
            phi_effective -= charge_adjustment_factor  # Apply adjustment

        induced_surf_charges_flat[i_start + 3] = phi_effective * energy_factor

    if missing_neighbors_count > 0:
        print(
            f"WARNING: {missing_neighbors_count} boundary points had < 6 valid neighbors during induced charge calculation."
        )

    return induced_surf_charges_flat


@njit(nogil=True, boundscheck=False, cache=True, parallel=_numba_usable)
def _calculate_rf_energy_outer_surf(
    induced_surf_charges_flat,  # Shape [N*4]
    induced_surf_charge_positions,  # Shape [N, 3]
    atoms_data,
):
    """Calculates the reaction field energy contribution from induced surface charges to atoms."""
    num_induced_surf_charges = induced_surf_charge_positions.shape[0]
    num_atoms = atoms_data.shape[0]

    if num_induced_surf_charges == 0 or num_atoms == 0:
        return 0.0

    partial_energy_contribs = np.zeros(num_induced_surf_charges, dtype=np.float64)

    for surf_charge_index in prange(num_induced_surf_charges):
        # Correctly get position and charge value
        isurf_x = induced_surf_charge_positions[surf_charge_index, 0]
        isurf_y = induced_surf_charge_positions[surf_charge_index, 1]
        isurf_z = induced_surf_charge_positions[surf_charge_index, 2]
        # Charge value is at index 3 of the corresponding 4-element block
        this_surf_charge_value = induced_surf_charges_flat[surf_charge_index * 4 + 3]

        this_potential_sum = 0.0
        for atom_index in range(num_atoms):
            atom_x = atoms_data[atom_index, ATOMFIELD_X]
            atom_y = atoms_data[atom_index, ATOMFIELD_Y]
            atom_z = atoms_data[atom_index, ATOMFIELD_Z]
            atom_charge = atoms_data[atom_index, ATOMFIELD_CHARGE]

            dx = isurf_x - atom_x
            dy = isurf_y - atom_y
            dz = isurf_z - atom_z
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq > APPROX_ZERO:
                dist = math.sqrt(dist_sq)
                this_potential_sum += atom_charge / dist

        partial_energy_contribs[surf_charge_index] = (
            this_potential_sum * this_surf_charge_value
        )

    energy_temp = np.sum(partial_energy_contribs)
    return energy_temp


@njit(nogil=True, boundscheck=False, cache=True, parallel=_numba_usable)
def _calculate_rf_energy_outer_atom(
    induced_surf_charges_flat,  # Shape [N*4]
    induced_surf_charge_positions,  # Shape [N, 3]
    atoms_data,
):
    """Calculates the reaction field energy contribution to atoms from induced surface charges."""
    num_induced_surf_charges = induced_surf_charge_positions.shape[0]
    num_atoms = atoms_data.shape[0]

    if num_induced_surf_charges == 0 or num_atoms == 0:
        return 0.0

    partial_energy_contribs = np.zeros(num_atoms, dtype=np.float64)

    for atom_index in prange(num_atoms):
        atom_x = atoms_data[atom_index, ATOMFIELD_X]
        atom_y = atoms_data[atom_index, ATOMFIELD_Y]
        atom_z = atoms_data[atom_index, ATOMFIELD_Z]
        atom_charge = atoms_data[atom_index, ATOMFIELD_CHARGE]

        potential_at_atom_j = 0.0
        for surf_charge_index in range(num_induced_surf_charges):
            isurf_x = induced_surf_charge_positions[surf_charge_index, 0]
            isurf_y = induced_surf_charge_positions[surf_charge_index, 1]
            isurf_z = induced_surf_charge_positions[surf_charge_index, 2]
            this_surf_charge_value = induced_surf_charges_flat[
                surf_charge_index * 4 + 3
            ]

            dx = atom_x - isurf_x
            dy = atom_y - isurf_y
            dz = atom_z - isurf_z
            dist_sq = dx * dx + dy * dy + dz * dz

            if dist_sq > APPROX_ZERO:
                dist = math.sqrt(dist_sq)
                potential_at_atom_j += this_surf_charge_value / dist

        partial_energy_contribs[atom_index] = atom_charge * potential_at_atom_j

    energy_temp = np.sum(partial_energy_contribs)
    return energy_temp


def _calc_induced_charge_rf_energy_helper(
    induced_surf_charges_flat,  # Shape [N*4]
    induced_surf_charge_positions,  # Shape [N, 3]
    atoms_data,
    epkt,
):
    """Helper function to calculate the reaction field energy."""
    num_induced_surf_charges = induced_surf_charge_positions.shape[0]
    num_atoms = atoms_data.shape[0]

    if num_induced_surf_charges == 0 or num_atoms == 0:
        return 0.0

    energy_temp = 0.0
    if num_induced_surf_charges >= num_atoms:
        energy_temp = _calculate_rf_energy_outer_surf(
            induced_surf_charges_flat, induced_surf_charge_positions, atoms_data
        )
    else:
        energy_temp = _calculate_rf_energy_outer_atom(
            induced_surf_charges_flat, induced_surf_charge_positions, atoms_data
        )

    energy_solvation = energy_temp * epkt * 0.5
    return energy_solvation


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _calculate_rf_energy_unified(
    induced_surf_charges_flat,
    induced_surf_charge_positions,
    atoms_data,
    epkt,
    nthreads,
    thread_local_sums,
):
    """
    Calculates the reaction field energy by manually splitting the work
    and accumulating results in a thread-local array, with correct remainder handling.
    """
    num_induced_surf_charges = induced_surf_charge_positions.shape[0]
    num_atoms = atoms_data.shape[0]

    if num_induced_surf_charges == 0 or num_atoms == 0:
        return 0.0

    # Use a parallel loop over the threads themselves
    for thread_id in prange(nthreads):
        # Determine the work range for this thread
        if num_induced_surf_charges >= num_atoms:
            total_work = num_induced_surf_charges
        else:
            total_work = num_atoms

        # Correctly handle the work distribution
        chunk_size = (total_work + nthreads - 1) // nthreads
        start_index = thread_id * chunk_size
        end_index = start_index + chunk_size

        # The last thread handles the remaining iterations to ensure all work is covered.
        if thread_id == nthreads - 1:
            end_index = total_work

        # Process the assigned work
        partial_sum = 0.0
        if num_induced_surf_charges >= num_atoms:
            for i in range(start_index, end_index):
                isurf_x = induced_surf_charge_positions[i, 0]
                isurf_y = induced_surf_charge_positions[i, 1]
                isurf_z = induced_surf_charge_positions[i, 2]
                this_surf_charge_value = induced_surf_charges_flat[i * 4 + 3]

                this_potential_sum = 0.0
                for atom_index in range(num_atoms):
                    atom_x = atoms_data[atom_index, ATOMFIELD_X]
                    atom_y = atoms_data[atom_index, ATOMFIELD_Y]
                    atom_z = atoms_data[atom_index, ATOMFIELD_Z]
                    atom_charge = atoms_data[atom_index, ATOMFIELD_CHARGE]

                    dx = isurf_x - atom_x
                    dy = isurf_y - atom_y
                    dz = isurf_z - atom_z
                    dist_sq = dx * dx + dy * dy + dz * dz

                    if dist_sq > APPROX_ZERO:
                        dist = math.sqrt(dist_sq)
                        this_potential_sum += atom_charge / dist

                partial_sum += this_potential_sum * this_surf_charge_value
        else:
            for i in range(start_index, end_index):
                atom_x = atoms_data[i, ATOMFIELD_X]
                atom_y = atoms_data[i, ATOMFIELD_Y]
                atom_z = atoms_data[i, ATOMFIELD_Z]
                atom_charge = atoms_data[i, ATOMFIELD_CHARGE]

                potential_at_atom_j = 0.0
                for surf_charge_index in range(num_induced_surf_charges):
                    isurf_x = induced_surf_charge_positions[surf_charge_index, 0]
                    isurf_y = induced_surf_charge_positions[surf_charge_index, 1]
                    isurf_z = induced_surf_charge_positions[surf_charge_index, 2]
                    this_surf_charge_value = induced_surf_charges_flat[
                        surf_charge_index * 4 + 3
                    ]

                    dx = atom_x - isurf_x
                    dy = atom_y - isurf_y
                    dz = atom_z - isurf_z
                    dist_sq = dx * dx + dy * dy + dz * dz

                    if dist_sq > APPROX_ZERO:
                        dist = math.sqrt(dist_sq)
                        potential_at_atom_j += this_surf_charge_value / dist

                partial_sum += atom_charge * potential_at_atom_j

        thread_local_sums[thread_id] = partial_sum

    return np.sum(thread_local_sums) * epkt * 0.5


@njit(nogil=True, boundscheck=False, cache=True, parallel=_numba_usable)
def _calc_charged_boundary_gridpoints(
    unique_charged_gridpoints: np.ndarray,  # Input: array [index_1d, total_charge, ix, iy, iz]
    boundary_indices_3d: np.ndarray,  # Input: boundary points [M, 3]
    dielectric_boundary_map_1d: np.ndarray,  # Input: 1D map for boundary check
    grid_shape: np.ndarray,  # Grid dimensions
    scale: float,  # Grid scale
    indi_scaled: float,  # Internal dielectric related
    debye_factor: float,  # Debye factor
):
    """
    Identifies unique charged grid points that lie on the dielectric boundary and calculates an adjustment factor.

    Args:
        unique_charged_gridpoints (np.ndarray): A 2D array of shape (K, 5) containing
                                                [index_1d, total_charge, ix, iy, iz]
                                                of unique charged grid points, sorted by index_1d.
        boundary_indices_3d (np.ndarray): A 2D array of shape (M, 3) containing [ix, iy, iz]
                                          of the dielectric boundary grid points.
        dielectric_boundary_map_1d (np.ndarray): A 1D array representing the dielectric boundary map.
        grid_shape (np.ndarray): Array representing the shape of the grid (nx, ny, nz).
        scale (float): The grid scale factor.
        indi_scaled (float): A scaling factor related to the internal dielectric constant.
        debye_factor (float): The Debye screening factor.

    Returns:
        tuple: A tuple containing:
            - int: The number of unique charged boundary grid points.
            - np.ndarray: A 2D array of shape (L, 2) containing [index_1d, adjustment_factor]
                          for charged grid points on the boundary.
    """
    nx, ny, nz = grid_shape  # Assume grid_shape is array/tuple
    num_boundary = boundary_indices_3d.shape[0]
    map_size = dielectric_boundary_map_1d.shape[0]
    num_unique_charged = unique_charged_gridpoints.shape[0]

    x_stride, y_stride = np.int64(ny * nz), np.int64(nz)

    # Check for empty inputs early
    if num_unique_charged == 0:
        return 0, np.empty((0, 2), dtype=np.float64)  # Return format [idx1d, factor]

    # --- Step 4: Build sorted boundary 1D index array ---
    if num_boundary == 0:
        sorted_boundary_point_indices_1d = np.empty(0, dtype=np.int64)
    else:
        boundary_point_indices_1d = np.empty(num_boundary, dtype=np.int64)
        # This loop can be parallel
        for i in prange(num_boundary):
            # Ensure indices are integer
            bx = int(boundary_indices_3d[i, 0])
            by = int(boundary_indices_3d[i, 1])
            bz = int(boundary_indices_3d[i, 2])
            if 0 <= bx < nx and 0 <= by < ny and 0 <= bz < nz:
                index_1d = bx * x_stride + by * y_stride + bz
                boundary_point_indices_1d[i] = index_1d
            else:
                boundary_point_indices_1d[i] = -1  # Invalid marker

        # Sort and filter invalid boundary points
        valid_boundary_mask = boundary_point_indices_1d != -1
        valid_boundary_indices = boundary_point_indices_1d[valid_boundary_mask]
        sorted_boundary_point_indices_1d = np.sort(valid_boundary_indices)

    # --- Step 5: Final pass to calculate adjustment factor ---
    charged_bgp_info_final = np.empty((num_unique_charged, 2), dtype=np.float64)
    final_count = 0  # Count points that are actually on the boundary

    for i in range(num_unique_charged):
        index_1d = int(unique_charged_gridpoints[i, GPCHRGFIELD_INDX_1D])
        accumulated_charge = unique_charged_gridpoints[
            i, GPCHRGFIELD_CHARGE
        ]  # total_charge

        # Assume adjustment factor is 0 unless it's a valid boundary point
        adjustment_factor = 0.0

        # Check if this unique charged point is on the dielectric boundary
        is_on_boundary, _ = is_contained_in_sorted(
            sorted_boundary_point_indices_1d, index_1d
        )
        adjusted_charge_factor = accumulated_charge

        is_on_boundary, _ = is_contained_in_sorted(
            sorted_boundary_point_indices_1d, index_1d
        )

        if is_on_boundary and 0 <= index_1d < map_size:
            in_solvent = 1.0 if dielectric_boundary_map_1d[index_1d] else 0.0
            denominator = 6.0 * indi_scaled + debye_factor * in_solvent
            if abs(denominator) > APPROX_ZERO:
                adjustment_factor = FOUR_PI * scale / denominator
                adjusted_charge_factor = accumulated_charge * adjustment_factor
            else:
                adjusted_charge_factor = 0.0
        else:
            adjusted_charge_factor = 0.0

        charged_bgp_info_final[final_count, 0] = float(index_1d)
        charged_bgp_info_final[final_count, 1] = adjusted_charge_factor
        final_count += 1

    return final_count, charged_bgp_info_final[:final_count]

@cuda.jit(cache=True)
def _rf_energy_kernel_outer_surf(induced_surf_charges_flat, induced_surf_charge_positions,
                                 atoms_data, outer_start, n_active, out_nthreads):
    """
    CUDA kernel where outer loop is over surface charges.
    Each thread computes one surface charge contribution summing over all atoms.
    """
    tid = cuda.grid(1)
    if tid >= n_active:
        return

    i = outer_start + tid
    n_surf = induced_surf_charge_positions.shape[0]
    n_atoms = atoms_data.shape[0]
    if i >= n_surf:
        return

    surf_x = induced_surf_charge_positions[i, 0]
    surf_y = induced_surf_charge_positions[i, 1]
    surf_z = induced_surf_charge_positions[i, 2]
    surf_q = induced_surf_charges_flat[i*4 + 3]

    acc = 0.0
    for j in range(n_atoms):
        atom_x = atoms_data[j, ATOMFIELD_X]
        atom_y = atoms_data[j, ATOMFIELD_Y]
        atom_z = atoms_data[j, ATOMFIELD_Z]
        atom_q = atoms_data[j, ATOMFIELD_CHARGE]

        dx = surf_x - atom_x
        dy = surf_y - atom_y
        dz = surf_z - atom_z
        dist_sq = dx*dx + dy*dy + dz*dz

        if dist_sq > APPROX_ZERO:
            acc += atom_q / math.sqrt(dist_sq)

    out_nthreads[tid] = acc * surf_q


@cuda.jit(cache=True)
def _rf_energy_kernel_outer_atom(induced_surf_charges_flat, induced_surf_charge_positions,
                                atoms_data, outer_start, n_active, out_nthreads):
    """
    CUDA kernel where outer loop is over atoms.
    Each thread computes one atom contribution summing over all surface charges.
    """
    tid = cuda.grid(1)
    if tid >= n_active:
        return

    j = outer_start + tid
    n_surf = induced_surf_charge_positions.shape[0]
    n_atoms = atoms_data.shape[0]
    if j >= n_atoms:
        return

    atom_x = atoms_data[j, ATOMFIELD_X]
    atom_y = atoms_data[j, ATOMFIELD_Y]
    atom_z = atoms_data[j, ATOMFIELD_Z]
    atom_q = atoms_data[j, ATOMFIELD_CHARGE]

    acc = 0.0
    for i in range(n_surf):
        surf_x = induced_surf_charge_positions[i, 0]
        surf_y = induced_surf_charge_positions[i, 1]
        surf_z = induced_surf_charge_positions[i, 2]
        surf_q = induced_surf_charges_flat[i*4 + 3]

        dx = atom_x - surf_x
        dy = atom_y - surf_y
        dz = atom_z - surf_z
        dist_sq = dx*dx + dy*dy + dz*dz

        if dist_sq > APPROX_ZERO:
            acc += surf_q / math.sqrt(dist_sq)

    out_nthreads[tid] = acc * atom_q


def _rf_energy_gpu(induced_surf_charges_flat, induced_surf_charge_positions, atoms_data, epkt):
    """
    Computes the reaction field energy on GPU using CUDA.
    Dynamically chooses outer dimension based on sizes and chunks the computation
    to avoid GPU memory blowup for huge systems.
    """
    n_surf = induced_surf_charge_positions.shape[0]
    n_atoms = atoms_data.shape[0]

    if n_surf == 0 or n_atoms == 0:
        return 0.0

    # Choose kernel based on smaller dimension as inner loop
    if n_surf >= n_atoms:
        kernel = _rf_energy_kernel_outer_surf
        outer_size = n_surf
    else:
        kernel = _rf_energy_kernel_outer_atom
        outer_size = n_atoms

    threads_per_block = 256
    max_blocks = 1024
    total_energy = 0.0

    # Chunking loop to limit device memory usage
    for outer_start in range(0, outer_size, threads_per_block * max_blocks):
        n_active = min(threads_per_block * max_blocks, outer_size - outer_start)
        blocks = (n_active + threads_per_block - 1) // threads_per_block

        out_nthreads_device = cuda.device_array(shape=(threads_per_block*blocks,), dtype=np.float64)
        kernel[blocks, threads_per_block](induced_surf_charges_flat,
                                          induced_surf_charge_positions,
                                          atoms_data,
                                          outer_start, n_active,
                                          out_nthreads_device)
        cuda.synchronize()
        partial_host = out_nthreads_device.copy_to_host()
        total_energy += np.sum(partial_host[:n_active])

    return total_energy * epkt * 0.5

def calc_induced_charge_rf_energy(
    platform: Platform,
    unique_charged_gridpoints: np.ndarray,  # Input: array [index_1d, total_charge, ix, iy, iz]
    dielectric_boundary_grids: np.ndarray,  # Input: boundary points [M, 3]
    dielectric_boundary_map_1d: np.ndarray,  # Input: 1D map
    atoms_data: np.ndarray,
    induced_surf_charge_positions: np.ndarray,  # Input: boundary point positions [M, 3] ? Should match diel_boundary_grids
    phimap_in_solvent: np.ndarray,
    grid_shape: np.ndarray,
    scale: float,
    indi_scaled: float,
    debye_factor: float,
    epkt: float,
    dump_arrays: bool = False,
):
    """
    Calculates the reaction field energy due to induced charges on the dielectric boundary.

    Args:

        unique_charged_gridpoints (np.ndarray): A 2D array of shape (K, 5) containing
                                                [index_1d, total_charge, ix, iy, iz]
                                                of unique charged grid points, sorted by index_1d
                                                (output of set_grid_charges_sorted_by_index1d).
        dielectric_boundary_grids (np.ndarray): A 2D array of shape (M, 3) containing [ix, iy, iz]
                                                 of the dielectric boundary grid points.
        dielectric_boundary_map_1d (np.ndarray): A 1D array representing the dielectric boundary map.
        atoms_data (np.ndarray): Array containing atom information.
        induced_surf_charge_positions (np.ndarray): A 2D array of shape (M, 3) containing the [ix, iy, iz]
                                                    coordinates of the induced surface charges
                                                    (should match dielectric_boundary_grids).
        phimap_in_solvent (np.ndarray): A 3D array representing the electrostatic potential map in the solvent.
        grid_shape (np.ndarray): Array representing the shape of the grid (nx, ny, nz).
        scale (float): The grid scale factor.
        indi_scaled (float): A scaling factor related to the internal dielectric constant.
        debye_factor (float): The Debye screening factor.
        epkt (float): The thermal energy scale factor.
        dump_arrays (bool): Dump unique_charged_gridpoints, dielectric_boundary_grids, induced_surf_charge_positions arrays.

    Returns:
        tuple: A tuple containing:
            - float: The solvation energy due to the reaction field.
            - np.ndarray: A 1D array of shape (M * 4) containing [ix, iy, iz, charge] for each boundary grid point.
    """
    if dump_arrays:
        from pprint import PrettyPrinter
        import sys

        np.set_printoptions(
            threshold=sys.maxsize,  # Print all elements
            edgeitems=sys.maxsize,  # Print all items at the beginning/end
            linewidth=200,  # Large linewidth to avoid wrapping for long arrays
            formatter={
                "float_kind": lambda x: "%.6f" % x
            },  # Optional: format floats consistently
        )
        fout = open("induced-surf-rf-cubic.txt", "w")
        pp = PrettyPrinter(
            indent=4, width=10**6, depth=None, compact=False, stream=fout
        )
    tic_crg_gp = time.perf_counter()
    num_unique_charged, charged_bgp_info = _calc_charged_boundary_gridpoints(
        unique_charged_gridpoints,
        dielectric_boundary_grids,
        dielectric_boundary_map_1d,
        grid_shape,
        scale,
        indi_scaled,
        debye_factor,
    )
    toc_crg_gp = time.perf_counter()
    vprint(
        INFO,
        _VERBOSITY,
        f"INFO: Time calculating charged grid points: {toc_crg_gp-tic_crg_gp:0.3f}",
    )
    nprint_cpu(
        DEBUG,
        _VERBOSITY,
        "INFO: Found ",
        num_unique_charged,
        " unique charged boundary gridpoints.",
    )
    if dump_arrays:
        pp.pprint(f"unique_charged_gridpoints(len={num_unique_charged})")
        pp.pprint((num_unique_charged, unique_charged_gridpoints[:num_unique_charged]))

    nprint_cpu(DEBUG, _VERBOSITY, "INFO: Calculating induced surface charges...")
    tic_srf_crg = time.perf_counter()
    induced_surf_charges_flat = _calculate_induced_surface_charges(
        charged_bgp_info,  # Pass the [index_1d, factor] array
        dielectric_boundary_grids,  # Pass the original boundary points [M, 3]
        phimap_in_solvent,
        grid_shape,
        scale,
        epkt,
    )
    toc_srf_crg = time.perf_counter()
    vprint(
        INFO,
        _VERBOSITY,
        f"INFO: Time calculating induced surf charges: {toc_srf_crg - tic_srf_crg:0.3f}",
    )
    nprint_cpu(DEBUG, _VERBOSITY, "INFO: Calculating reaction field energy...")

    if dump_arrays:
        pp.pprint("dielectric_boundary_grids")
        pp.pprint(dielectric_boundary_grids)

        pp.pprint("induced_surf_charges_flat")
        pp.pprint(induced_surf_charges_flat)

    tic_rf_erg = time.perf_counter()
    thread_local_sums = np.zeros(platform.names["cpu"]["num_threads"], dtype=np.float64)
    if platform.active == "cpu":
        set_num_threads(platform.names["cpu"]["num_threads"])
        energy_solvation = _calculate_rf_energy_unified(
            induced_surf_charges_flat,
            induced_surf_charge_positions,  # Ensure this matches dielectric_boundary_grids if used as positions
            atoms_data,
            epkt,
            platform.names["cpu"]["num_threads"],
            thread_local_sums,
        )
    else:
        energy_solvation = _rf_energy_gpu(induced_surf_charges_flat, induced_surf_charge_positions, atoms_data, epkt)
    toc_rf_erg = time.perf_counter()

    vprint(
        INFO,
        _VERBOSITY,
        f"INFO: Time calculating induced surf rf energy: {toc_rf_erg - tic_rf_erg:0.3f}",
    )
    nprint_cpu(
        DEBUG, _VERBOSITY, "INFO: Calculated Solvation Energy: {energy_solvation}"
    )
    if dump_arrays:
        pp.pprint(f"induced_surf_charge_positions:")
        pp.pprint(induced_surf_charge_positions)

    # Return energy and the flat induced charges [ix, iy, iz, charge_val]
    return energy_solvation, induced_surf_charges_flat


@njit(nogil=True, boundscheck=False, cache=True)
def calc_reactionfield_energy(uniq_index_sorted_aggr_charge_grids, phimap_in_media):
    """
    Calculates the reaction field energy within a medium given the unique, index-sorted,
        aggregated charged grid points and the potential map.

    Args:
        uniq_index_sorted_aggr_charge_grids (np.ndarray): A 2D array where each row represents a charged grid point
                                                        [index_1d, acc_charge, ix, iy, iz], sorted by index_1d and
                                                        with aggregated charges for unique grid points.
        phimap_in_media (np.ndarray): A 3D array representing the electrostatic potential map in the medium.

    Returns:
        float: The total reaction field energy.
    """
    total_energy_phase_sum = 0.0
    for i in range(uniq_index_sorted_aggr_charge_grids.shape[0]):
        qi = uniq_index_sorted_aggr_charge_grids[i]
        # Extract ix, iy, iz, and charge using constants
        ix = int(qi[GPCHRGFIELD_INDX_X])
        iy = int(qi[GPCHRGFIELD_INDX_Y])
        iz = int(qi[GPCHRGFIELD_INDX_Z])
        charge = qi[GPCHRGFIELD_CHARGE]

        # Add bounds check for safety unless absolutely sure indices are valid
        if (
            0 <= ix < phimap_in_media.shape[0]
            and 0 <= iy < phimap_in_media.shape[1]
            and 0 <= iz < phimap_in_media.shape[2]
        ):
            phi_xyz = phimap_in_media[ix, iy, iz]  # Use tuple indexing
            total_energy_phase_sum += charge * phi_xyz
    total_energy_phase_sum *= 0.5
    return total_energy_phase_sum
