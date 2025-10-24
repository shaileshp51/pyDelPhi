#!/usr/bin/env python3
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
Module: delphi_routines

This module contains Numba-optimized functions for calculating various
spatial maps related to dielectric properties and Gaussian surface definitions
within the pydelhi software for solving PB equations. It provides both CPU and CUDA
implementations for performance-critical computations, leveraging Numba's `@njit`
and `@cuda.jit` decorators for just-in-time compilation and GPU acceleration, respectively.

Key functionalities include:
- Calculation of Gaussian epsilon maps based on atomic densities.
- Determination of spatial epsilon maps at grid points and midpoints,
  incorporating surface and density-based cutoffs.
- Identification of dielectric boundary regions.
- Computation of Gaussian-like surface maps.
- Analytical calculation of the gradient of the Gaussian-like surface map,
  employing a neighbor voxel mapping strategy for efficient atom lookup.

The module is designed to be highly performant, utilizing parallel processing
(`prange` for CPU, `cuda.grid` for GPU) and minimizing memory transfers
where possible, particularly in the CUDA implementations.
"""

import math
import numpy as np
from numba import set_num_threads, njit, prange, cuda

from pydelphi.foundation.platforms import Platform
from pydelphi.foundation.enums import Precision

from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_RADIUS,
    ATOMFIELD_GAUSS_SIGMA,
    ConstDelPhiFloats as ConstDelPhi,
)

APPROX_ZERO = ConstDelPhi.ApproxZero.value
GAUSSIAN_INFLUENCE_RADIUS_FACTOR = ConstDelPhi.GaussianInfluenceRadiusFactor.value

from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
    vprint,
)
from pydelphi.config.logging_config import (
    DEBUG,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

from pydelphi.space.core.voxelizer import build_neighbor_voxel_unique_atom_index_map

# --- Dynamic Precision Handling ---
if PRECISION.int_value in {
    Precision.SINGLE.int_value,
}:

    try:
        import pydelphi.utils.cuda.single as size_gpu
    except ImportError:
        size_gpu = None
elif PRECISION.int_value == Precision.DOUBLE.int_value:

    try:
        import pydelphi.utils.cuda.double as size_gpu
    except ImportError:
        size_gpu = None
else:
    raise ValueError(f"Unsupported PRECISION: {PRECISION}")


@njit(nogil=True, boundscheck=False, cache=True)
def calc_atom_gaussian_influence_radius(
    probe_radius: delphi_real,
    salt_radius: delphi_real,
    offset: delphi_real,
    max_atom_radius: delphi_real,
    atoms_data: np.ndarray[delphi_real],
    gaussian_decay_factor: float = GAUSSIAN_INFLUENCE_RADIUS_FACTOR,
    dtype_real: type = float,
) -> delphi_real:
    """
    Estimates the maximum Gaussian influence radius among all atoms.

    This version considers the maximum of the Gaussian decay influence (controlled by
    gaussian_decay_factor) and the physical extent with the exclusion radius.

    Args:
        probe_radius (delphi_real): Radius of the probe molecule.
        salt_radius (delphi_real): Radius of the salt ions.
        offset (delphi_real): An offset value.
        max_atom_radius (delphi_real): The maximum radius of any atom in the system.
        atoms_data (np.ndarray[delphi_real]): A 2D array containing atom information.
            It is expected to have columns corresponding to ATOMFIELD_RADIUS and
            ATOMFIELD_GAUSS_SIGMA.
        gaussian_decay_factor (float, optional): Factor controlling the extent of the
            Gaussian influence. Higher values lead to a larger radius and smaller
            contribution at the boundary. Defaults to the value defined in
            ConstDelPhi.GaussianInfluenceRadiusFactor.

    Returns:
        delphi_real: The estimated maximum Gaussian influence radius.
    """
    exclusion_radius = max(probe_radius, salt_radius) + offset
    max_influence_radius: dtype_real = (
        max_atom_radius + exclusion_radius
    )  # Initialize here

    for this_atom in atoms_data:
        atom_radius = this_atom[ATOMFIELD_RADIUS]
        atom_sigma = this_atom[ATOMFIELD_GAUSS_SIGMA]

        # Calculate the Gaussian decay influence radius.
        gaussian_influence = gaussian_decay_factor * atom_radius * atom_sigma

        # Calculate the physical extent with the exclusion radius.
        physical_extent = atom_radius + exclusion_radius

        # The influence radius is the maximum of these two.
        this_atom_influence_radius = max(gaussian_influence, physical_extent)

        max_influence_radius = max(max_influence_radius, this_atom_influence_radius)

    # Return the maximum of the initial estimate and the calculated max.
    return dtype_real(max(max_atom_radius, max_influence_radius))


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_calc_gaussian_density_map(
    generate_ion_exclusion_map: delphi_bool,
    scale: delphi_real,
    gaussian_exponent: delphi_int,
    gaussian_influence_radius_factor: delphi_real,
    surface_offset: delphi_real,
    atom_influence_radius: delphi_real,
    salt_radius: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    gauss_density_map_1d: np.ndarray[delphi_real],
    gauss_density_map_midpoints_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_bool],
    neighbor_voxel_atom_ids_flat: np.ndarray[delphi_int],
    neighbor_voxel_atom_start_index: np.ndarray[delphi_int],
    neighbor_voxel_atom_end_index: np.ndarray[delphi_int],
    voxel_map_origin: np.ndarray[delphi_real],
    voxel_map_shape: np.ndarray[delphi_int],
    voxel_map_scale: delphi_real,
) -> None:
    """
    MODIFIED: Uses voxel map for inner atom loop via NEIGHBOR_VOXEL_REL_COORDS.
    Calculates Gaussian density map on CPU.
    Warning: Parallel execution might cause race conditions on map updates.
    """
    grid_spacing = 1.0 / scale
    grid_spacing_half = 0.5 * grid_spacing
    atom_influence_radius_square = atom_influence_radius * atom_influence_radius
    y_stride: delphi_int = grid_shape[2]
    x_stride: delphi_int = grid_shape[1] * y_stride
    num_grid_points: delphi_int = grid_shape[0] * x_stride

    v_origin = voxel_map_origin
    v_shape = voxel_map_shape
    v_scale = voxel_map_scale

    for ijk1d in prange(num_grid_points):  # Parallel loop over grid points
        i: delphi_int = ijk1d // x_stride
        j: delphi_int = (ijk1d - i * x_stride) // y_stride
        k: delphi_int = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3: delphi_int = 3 * ijk1d

        grid_pos_x = i * grid_spacing + grid_origin[0]
        grid_pos_y = j * grid_spacing + grid_origin[1]
        grid_pos_z = k * grid_spacing + grid_origin[2]

        valid_half_h_x = i != (grid_shape[0] - 1)
        valid_half_h_y = j != (grid_shape[1] - 1)
        valid_half_h_z = k != (grid_shape[2] - 1)

        total_density: delphi_real = 0.0
        total_density_half_hx: delphi_real = 0.0
        total_density_half_hy: delphi_real = 0.0
        total_density_half_hz: delphi_real = 0.0
        ion_excluded_flag = False  # Local flag

        central_vx = max(
            0, min(delphi_int((grid_pos_x - v_origin[0]) * v_scale), v_shape[0])
        )
        central_vy = max(
            0, min(delphi_int((grid_pos_y - v_origin[1]) * v_scale), v_shape[1])
        )
        central_vz = max(
            0, min(delphi_int((grid_pos_z - v_origin[2]) * v_scale), v_shape[2])
        )

        vx = central_vx
        vy = central_vy
        vz = central_vz

        # --- Optimized Inner Loop Start ---
        if 0 <= vx <= v_shape[0] and 0 <= vy <= v_shape[1] and 0 <= vz <= v_shape[2]:
            start = neighbor_voxel_atom_start_index[vx, vy, vz]
            end = neighbor_voxel_atom_end_index[vx, vy, vz]
            if start <= end:
                for atom_list_idx in range(start, end + 1):
                    atom_idx = neighbor_voxel_atom_ids_flat[atom_list_idx] - 1
                    if atom_idx == -1:
                        continue

                    # --- Start of original inner loop logic ---
                    this_atom = atoms_data[atom_idx]
                    atom_crd_x = this_atom[ATOMFIELD_X]
                    atom_crd_y = this_atom[ATOMFIELD_Y]
                    atom_crd_z = this_atom[ATOMFIELD_Z]
                    atom_sigma = this_atom[ATOMFIELD_GAUSS_SIGMA]
                    atom_radius = this_atom[ATOMFIELD_RADIUS]
                    atom_rad_with_offset = atom_radius + surface_offset
                    if atom_sigma > APPROX_ZERO and atom_rad_with_offset > APPROX_ZERO:
                        mult_inv = 1.0 / (atom_sigma**2 * atom_rad_with_offset**2)
                    else:
                        mult_inv = 1.0e2  # Very large inverse
                    r_dx = grid_pos_x - atom_crd_x
                    r_dy = grid_pos_y - atom_crd_y
                    r_dz = grid_pos_z - atom_crd_z
                    dist_square = r_dx * r_dx + r_dy * r_dy + r_dz * r_dz

                    if generate_ion_exclusion_map and not ion_excluded_flag:
                        ion_exclusion_limit = atom_radius + salt_radius
                        ion_exclusion_limit_square = ion_exclusion_limit**2
                        if dist_square < ion_exclusion_limit_square:
                            ion_excluded_flag = True

                    if dist_square <= atom_influence_radius_square and mult_inv > 0.0:
                        density = math.exp(
                            -((dist_square * mult_inv) ** gaussian_exponent)
                        )
                        total_density = 1.0 - (1.0 - total_density) * (1.0 - density)
                        if valid_half_h_x:
                            r_half_h_dx = r_dx + grid_spacing_half
                            dist_sq_hx = r_half_h_dx**2 + r_dy**2 + r_dz**2
                            if dist_sq_hx <= atom_influence_radius_square:
                                density_hx = math.exp(
                                    -((dist_sq_hx * mult_inv) ** gaussian_exponent)
                                )
                                total_density_half_hx = 1.0 - (
                                    1.0 - total_density_half_hx
                                ) * (1.0 - density_hx)
                        if valid_half_h_y:
                            r_half_h_dy = r_dy + grid_spacing_half
                            dist_sq_hy = r_dx**2 + r_half_h_dy**2 + r_dz**2
                            if dist_sq_hy <= atom_influence_radius_square:
                                density_hy = math.exp(
                                    -((dist_sq_hy * mult_inv) ** gaussian_exponent)
                                )
                                total_density_half_hy = 1.0 - (
                                    1.0 - total_density_half_hy
                                ) * (1.0 - density_hy)
                        if valid_half_h_z:
                            r_half_h_dz = r_dz + grid_spacing_half
                            dist_sq_hz = r_dx**2 + r_dy**2 + r_half_h_dz**2
                            if dist_sq_hz <= atom_influence_radius_square:
                                density_hz = math.exp(
                                    -((dist_sq_hz * mult_inv) ** gaussian_exponent)
                                )
                                total_density_half_hz = 1.0 - (
                                    1.0 - total_density_half_hz
                                ) * (1.0 - density_hz)
                    # --- End of original inner loop logic ---
        # --- Optimized Inner Loop End ---

        # Assign final densities for this grid point ijk1d
        gauss_density_map_1d[ijk1d] = delphi_real(total_density)
        gauss_density_map_midpoints_1d[ijk1d_x_3] = delphi_real(total_density_half_hx)
        gauss_density_map_midpoints_1d[ijk1d_x_3 + 1] = delphi_real(
            total_density_half_hy
        )
        gauss_density_map_midpoints_1d[ijk1d_x_3 + 2] = delphi_real(
            total_density_half_hz
        )
        # Update ion exclusion map
        if generate_ion_exclusion_map and ion_excluded_flag:
            if not ion_exclusion_map_1d[ijk1d]:
                ion_exclusion_map_1d[ijk1d] = ion_excluded_flag


@cuda.jit(cache=True)
def _cuda_calc_gaussian_density_map(
    generate_ion_exclusion_map: delphi_bool,
    scale: delphi_real,
    gaussian_exponent,
    gaussian_stddev_covered: delphi_real,
    surface_offset: delphi_real,
    atom_influence_radius: delphi_real,
    salt_radius: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    gauss_density_map_1d: np.ndarray[delphi_real],
    gauss_density_map_midpoints_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_bool],
    neighbor_voxel_atom_ids_flat: np.ndarray[delphi_int],
    neighbor_voxel_start_index: np.ndarray[delphi_int],
    neighbor_voxel_end_index: np.ndarray[delphi_int],
    voxel_map_origin: np.ndarray[delphi_real],
    voxel_map_shape: np.ndarray[delphi_int],
    voxel_map_scale: delphi_real,
) -> None:
    grid_spacing = 1.0 / scale
    grid_spacing_half = 0.5 * grid_spacing
    atom_influence_radius_square = atom_influence_radius * atom_influence_radius
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    ijk1d = cuda.grid(1)
    num_grid_points = grid_shape[0] * grid_shape[1] * grid_shape[2]

    v_origin = voxel_map_origin
    v_shape = voxel_map_shape
    v_scale = voxel_map_scale

    if ijk1d < num_grid_points:
        i: delphi_int = ijk1d // x_stride
        j: delphi_int = (ijk1d - i * x_stride) // y_stride
        k: delphi_int = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3: delphi_int = 3 * ijk1d

        grid_pos_x = i * grid_spacing + grid_origin[0]
        grid_pos_y = j * grid_spacing + grid_origin[1]
        grid_pos_z = k * grid_spacing + grid_origin[2]

        valid_half_h_x = i != (grid_shape[0] - 1)
        valid_half_h_y = j != (grid_shape[1] - 1)
        valid_half_h_z = k != (grid_shape[2] - 1)

        total_density: delphi_real = 0.0
        total_density_half_hx: delphi_real = 0.0
        total_density_half_hy: delphi_real = 0.0
        total_density_half_hz: delphi_real = 0.0
        ion_excluded_flag = False

        central_vx = max(
            0, min(delphi_int((grid_pos_x - v_origin[0]) * v_scale), v_shape[0])
        )
        central_vy = max(
            0, min(delphi_int((grid_pos_y - v_origin[1]) * v_scale), v_shape[1])
        )
        central_vz = max(
            0, min(delphi_int((grid_pos_z - v_origin[2]) * v_scale), v_shape[2])
        )

        vx = central_vx
        vy = central_vy
        vz = central_vz

        if 0 <= vx <= v_shape[0] and 0 <= vy <= v_shape[1] and 0 <= vz <= v_shape[2]:
            start = neighbor_voxel_start_index[vx, vy, vz]
            end = neighbor_voxel_end_index[vx, vy, vz]
            if start <= end:
                for atom_list_idx in range(start, end + 1):
                    atom_idx = neighbor_voxel_atom_ids_flat[atom_list_idx] - 1
                    if atom_idx == -1:
                        continue

                    this_atom = atoms_data[atom_idx]
                    atom_crd_x = this_atom[ATOMFIELD_X]
                    atom_crd_y = this_atom[ATOMFIELD_Y]
                    atom_crd_z = this_atom[ATOMFIELD_Z]
                    atom_sigma = this_atom[ATOMFIELD_GAUSS_SIGMA]
                    atom_radius = this_atom[ATOMFIELD_RADIUS]

                    atom_rad_with_offset = atom_radius + surface_offset
                    if atom_sigma > APPROX_ZERO and atom_rad_with_offset > APPROX_ZERO:
                        mult_inv = 1.0 / (atom_sigma**2 * atom_rad_with_offset**2)
                    else:
                        mult_inv = 0.0
                    r_dx = grid_pos_x - atom_crd_x
                    r_dy = grid_pos_y - atom_crd_y
                    r_dz = grid_pos_z - atom_crd_z
                    dist_square = r_dx * r_dx + r_dy * r_dy + r_dz * r_dz
                    if generate_ion_exclusion_map and not ion_excluded_flag:
                        ion_exclusion_limit = (
                            atom_sigma * atom_radius * gaussian_stddev_covered
                        ) + salt_radius
                        ion_exclusion_limit_square = ion_exclusion_limit**2
                        if dist_square < ion_exclusion_limit_square:
                            ion_excluded_flag = True
                    if dist_square <= atom_influence_radius_square and mult_inv > 0.0:
                        g_arg = (dist_square * mult_inv) ** gaussian_exponent
                        density = math.exp(-g_arg)
                        total_density = 1.0 - (1.0 - total_density) * (1.0 - density)
                        if valid_half_h_x:
                            r_half_h_dx = r_dx + grid_spacing_half
                            dist_sq_hx = r_half_h_dx**2 + r_dy**2 + r_dz**2
                            if dist_sq_hx <= atom_influence_radius_square:
                                g_arg_hx = (dist_sq_hx * mult_inv) ** gaussian_exponent
                                density_hx = math.exp(-g_arg_hx)
                                total_density_half_hx = 1.0 - (
                                    1.0 - total_density_half_hx
                                ) * (1.0 - density_hx)
                        if valid_half_h_y:
                            r_half_h_dy = r_dy + grid_spacing_half
                            dist_sq_hy = r_dx**2 + r_half_h_dy**2 + r_dz**2
                            if dist_sq_hy <= atom_influence_radius_square:
                                g_arg_hy = (dist_sq_hy * mult_inv) ** gaussian_exponent
                                density_hy = math.exp(-g_arg_hy)
                                total_density_half_hy = 1.0 - (
                                    1.0 - total_density_half_hy
                                ) * (1.0 - density_hy)
                        if valid_half_h_z:
                            r_half_h_dz = r_dz + grid_spacing_half
                            dist_sq_hz = r_dx**2 + r_dy**2 + r_half_h_dz**2
                            if dist_sq_hz <= atom_influence_radius_square:
                                g_arg_hz = (dist_sq_hz * mult_inv) ** gaussian_exponent
                                density_hz = math.exp(-g_arg_hz)
                                total_density_half_hz = 1.0 - (
                                    1.0 - total_density_half_hz
                                ) * (1.0 - density_hz)
                # --- End of original inner loop logic ---

        # Assign final densities for this grid point ijk1d
        gauss_density_map_1d[ijk1d] = delphi_real(total_density)
        gauss_density_map_midpoints_1d[ijk1d_x_3] = delphi_real(total_density_half_hx)
        gauss_density_map_midpoints_1d[ijk1d_x_3 + 1] = delphi_real(
            total_density_half_hy
        )
        gauss_density_map_midpoints_1d[ijk1d_x_3 + 2] = delphi_real(
            total_density_half_hz
        )
        if generate_ion_exclusion_map and ion_excluded_flag:
            if not ion_exclusion_map_1d[ijk1d]:
                ion_exclusion_map_1d[ijk1d] = True


def calc_gaussian_density_map(
    platform,
    num_cuda_threads,
    generate_ion_exclusion_map,
    scale,
    gaussian_exponent,
    gaussian_influence_radius_factor,
    surface_offset,
    atom_influence_radius,
    salt_radius,
    grid_shape,
    grid_origin,
    atoms_data,
    gauss_density_map_1d,
    gauss_density_map_midpoints_1d,
    ion_exclusion_map_1d,
    voxel_atom_ids,
    voxel_atom_start_index,
    voxel_atom_end_index,
    voxel_map_origin,
    voxel_map_shape,
    voxel_map_scale,
):
    num_atoms = atoms_data.shape[0]

    # Step 1: build neighbor voxel map
    (
        neighbor_voxel_atom_ids_flat,
        neighbor_voxel_start_index,
        neighbor_voxel_end_index,
        actual_neighbor_ids_count,
    ) = build_neighbor_voxel_unique_atom_index_map(
        num_atoms=num_atoms,
        voxel_atom_ids=voxel_atom_ids,
        voxel_atom_start_index=voxel_atom_start_index,
        voxel_atom_end_index=voxel_atom_end_index,
        voxel_map_shape=voxel_map_shape,
    )

    # Step 2: calculate Gaussian density
    if platform.active == "cpu":
        vprint(DEBUG, _VERBOSITY, "Calling _cpu_calc_gaussian_density_map")
        _cpu_calc_gaussian_density_map(
            generate_ion_exclusion_map=generate_ion_exclusion_map,
            scale=scale,
            gaussian_exponent=gaussian_exponent,
            gaussian_influence_radius_factor=gaussian_influence_radius_factor,
            surface_offset=surface_offset,
            atom_influence_radius=atom_influence_radius,
            salt_radius=salt_radius,
            grid_shape=grid_shape,
            grid_origin=grid_origin,
            atoms_data=atoms_data,
            gauss_density_map_1d=gauss_density_map_1d,
            gauss_density_map_midpoints_1d=gauss_density_map_midpoints_1d,
            ion_exclusion_map_1d=ion_exclusion_map_1d,
            neighbor_voxel_atom_ids_flat=neighbor_voxel_atom_ids_flat,
            neighbor_voxel_atom_start_index=neighbor_voxel_start_index,
            neighbor_voxel_atom_end_index=neighbor_voxel_end_index,
            voxel_map_origin=voxel_map_origin,
            voxel_map_shape=voxel_map_shape,
            voxel_map_scale=voxel_map_scale,
        )
    elif platform.active == "cuda" and hasattr(cuda, "jit"):
        vprint(DEBUG, _VERBOSITY, "Calling _cuda_calc_gaussian_density_map")
        # Transfer data to GPU
        grid_shape_dev = cuda.to_device(grid_shape.astype(delphi_int))
        grid_origin_dev = cuda.to_device(grid_origin.astype(delphi_real))
        atoms_data_dev = cuda.to_device(atoms_data.astype(delphi_real))
        gauss_density_map_1d_dev = cuda.to_device(gauss_density_map_1d)
        gauss_density_map_midpoints_1d_dev = cuda.to_device(
            gauss_density_map_midpoints_1d
        )
        ion_exclusion_map_1d_dev = cuda.to_device(ion_exclusion_map_1d)
        neighbor_voxel_ids_dev = cuda.to_device(
            neighbor_voxel_atom_ids_flat.astype(delphi_int)
        )
        neighbor_voxel_start_dev = cuda.to_device(
            neighbor_voxel_start_index.astype(delphi_int)
        )
        neighbor_voxel_end_dev = cuda.to_device(
            neighbor_voxel_end_index.astype(delphi_int)
        )
        voxel_origin_dev = cuda.to_device(voxel_map_origin.astype(delphi_real))
        voxel_shape_dev = cuda.to_device(voxel_map_shape.astype(delphi_int))

        # Configure kernel launch
        num_grid_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
        threads_per_block = num_cuda_threads
        blocks_per_grid = (
            num_grid_points + (threads_per_block - 1)
        ) // threads_per_block

        # Launch CUDA kernel
        _cuda_calc_gaussian_density_map[blocks_per_grid, threads_per_block](
            delphi_bool(generate_ion_exclusion_map),
            delphi_real(scale),
            delphi_real(gaussian_exponent),
            delphi_real(gaussian_influence_radius_factor),
            delphi_real(surface_offset),
            delphi_real(atom_influence_radius),
            delphi_real(salt_radius),
            grid_shape_dev,
            grid_origin_dev,
            atoms_data_dev,
            gauss_density_map_1d_dev,
            gauss_density_map_midpoints_1d_dev,
            ion_exclusion_map_1d_dev,
            neighbor_voxel_ids_dev,
            neighbor_voxel_start_dev,
            neighbor_voxel_end_dev,
            voxel_origin_dev,
            voxel_shape_dev,
            delphi_real(voxel_map_scale),
        )
        cuda.synchronize()

        # Copy results back to host
        gauss_density_map_1d_dev.copy_to_host(gauss_density_map_1d)
        gauss_density_map_midpoints_1d_dev.copy_to_host(gauss_density_map_midpoints_1d)
        ion_exclusion_map_1d_dev.copy_to_host(ion_exclusion_map_1d)

    else:
        raise RuntimeError(f"Unsupported platform: {platform.active}")


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _calc_gaussian_epsilon_map(
    gapdi: delphi_real,
    indi: delphi_real,
    gaussian_density_1d: np.ndarray[delphi_real],
    gauss_epsmap_1d: np.ndarray[delphi_real],
):
    """
    Calculates the Gaussian epsilon map based on Gaussian density.

    The dielectric constant at each grid point is determined by a linear
    interpolation between the interior dielectric constant (indi) and
    the difference between the gap dielectric constant (gapdi) and indi,
    scaled by the inverse of the Gaussian density.

    Args:
        gapdi (delphi_real): Dielectric constant in the gap region.
        indi (delphi_real): Dielectric constant in the interior (solute) region.
        gaussian_density_1d (np.ndarray[delphi_real]): 1D array of Gaussian densities
                                                       at each grid point.
        gauss_epsmap_1d (np.ndarray[delphi_real]): 1D array to store the calculated
                                                   Gaussian epsilon map at grid/mid-points,
                                                   This array is modified in-place.
    """
    gap_indi = gapdi - indi
    for ijk1d in prange(gauss_epsmap_1d.shape[0]):
        gauss_epsmap_1d[ijk1d] = indi + gap_indi * (1.0 - gaussian_density_1d[ijk1d])


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def calc_spatial_epsilon_map_midpoints(
    num_cpu_threads: delphi_int,
    is_surf_midpoints: delphi_bool,
    vacuum: delphi_bool,
    exdi: delphi_real,
    gapdi: delphi_real,
    indi: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    gauss_density_map_1d: np.ndarray[delphi_real],
    solute_surface_map_1d: np.ndarray[delphi_real],
    gauss_density_map_midpoints_1d: np.ndarray[delphi_real],
    surface_map_midpoints_1d: np.ndarray[delphi_real],
):
    """
    Calculates the spatial epsilon map, including values at midpoints, based on
    Gaussian and surface maps.

    This function determines the dielectric constant at each grid point and
    at the midpoints between grid points. It distinguishes between cases
    where surface midpoints are used or not.

    Args:
        num_cpu_threads (delphi_int): Number of CPU threads to use (currently unused in numba parallel prange).
        is_surf_midpoints (delphi_bool): If True, surface map midpoints are used for epsilon calculation.
        vacuum (delphi_bool): If True, the exterior dielectric constant is set to 1.0 (vacuum).
        exdi (delphi_real): Exterior dielectric constant.
        gapdi (delphi_real): Dielectric constant in the gap region.
        indi (delphi_real): Interior dielectric constant.
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid (nx, ny, nz).
        gauss_density_map_1d (np.ndarray[delphi_real]): 1D array of Gaussian densities at grid points.
        solute_surface_map_1d (np.ndarray[delphi_real]): 1D array of solute surface values at grid points.
        gauss_density_map_midpoints_1d (np.ndarray[delphi_real]): 1D array of Gaussian densities
                                                                  at midpoints.
        surface_map_midpoints_1d (np.ndarray[delphi_real]): 1D array of surface values at midpoints.

    Returns:
        tuple: A tuple containing:
            - epsilon_gauss_1d (np.ndarray[delphi_real]): Gaussian epsilon map at grid points.
            - epsilon_r_1d (np.ndarray[delphi_real]): Final spatial epsilon map at grid points.
            - epsilon_r_midpoints_1d (np.ndarray[delphi_real]): Final spatial epsilon map at midpoints.
    """
    num_grid_points = gauss_density_map_1d.shape[0]
    num_mid_points = num_grid_points * 3
    epsout = delphi_real(1.0) if vacuum else exdi

    epsilon_gauss_1d = np.zeros(num_grid_points, dtype=delphi_real)
    epsilon_r_1d = np.zeros(num_grid_points, dtype=delphi_real)

    epsilon_r_midpoints_1d = np.empty(num_mid_points, dtype=delphi_real)
    epsilon_r_midpoints_1d.fill(epsout)

    y_stride: delphi_int = grid_shape[2]
    x_stride: delphi_int = grid_shape[1] * y_stride

    if not is_surf_midpoints:
        _calc_gaussian_epsilon_map(gapdi, indi, gauss_density_map_1d, epsilon_gauss_1d)
        for ijk1d in prange(num_grid_points):
            epsilon_r_1d[ijk1d] = (
                epsilon_gauss_1d[ijk1d] * solute_surface_map_1d[ijk1d]
                + (1.0 - solute_surface_map_1d[ijk1d]) * epsout
            )
        for ijk1d in prange(num_grid_points):
            i = ijk1d // x_stride
            j = (ijk1d - i * x_stride) // y_stride
            k = ijk1d - i * x_stride - j * y_stride
            ijk1d_x_3 = ijk1d * 3
            if not (
                i == 0
                or i == grid_shape[0] - 1
                or j == 0
                or j == grid_shape[1] - 1
                or k == 0
                or k == grid_shape[2] - 1
            ):
                # Check indices exist before accessing neighbors
                idx_xp = ijk1d + x_stride
                idx_yp = ijk1d + y_stride
                idx_zp = ijk1d + 1
                if idx_xp < num_grid_points:
                    epsilon_r_midpoints_1d[ijk1d_x_3] = (
                        epsilon_r_1d[ijk1d] + epsilon_r_1d[idx_xp]
                    ) / 2.0
                if idx_yp < num_grid_points:
                    epsilon_r_midpoints_1d[ijk1d_x_3 + 1] = (
                        epsilon_r_1d[ijk1d] + epsilon_r_1d[idx_yp]
                    ) / 2.0
                if idx_zp < num_grid_points:
                    epsilon_r_midpoints_1d[ijk1d_x_3 + 2] = (
                        epsilon_r_1d[ijk1d] + epsilon_r_1d[idx_zp]
                    ) / 2.0

    elif is_surf_midpoints:
        if surface_map_midpoints_1d is not None:
            epsilon_gauss_midpoints_1d = np.zeros(
                gauss_density_map_midpoints_1d.size, dtype=delphi_real
            )
            _calc_gaussian_epsilon_map(
                gapdi, indi, gauss_density_map_1d, epsilon_gauss_1d
            )
            _calc_gaussian_epsilon_map(
                gapdi, indi, gauss_density_map_midpoints_1d, epsilon_gauss_midpoints_1d
            )
            for ijk1d in prange(num_grid_points):
                epsilon_r_1d[ijk1d] = (
                    epsilon_gauss_1d[ijk1d] * solute_surface_map_1d[ijk1d]
                    + (1.0 - solute_surface_map_1d[ijk1d]) * epsout
                )
            for ijkm1d in prange(num_mid_points):
                epsilon_r_midpoints_1d[ijkm1d] = (
                    epsilon_gauss_midpoints_1d[ijkm1d]
                    * surface_map_midpoints_1d[ijkm1d]
                    + (1.0 - surface_map_midpoints_1d[ijkm1d]) * epsout
                )
    return epsilon_gauss_1d, epsilon_r_1d, epsilon_r_midpoints_1d


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def calc_gaussian_cutoff_spatial_epsilon_map_midpoints(
    num_cpu_threads: delphi_int,
    vaccum: delphi_bool,
    exdi: delphi_real,
    gapdi: delphi_real,
    indi: delphi_real,
    density_cutoff: delphi_real,
    epsilon_cutoff: delphi_real,
    solute_density_threshold: delphi_real,
    eps_maxmin_ratio_threshold: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    gauss_density_map_1d: np.ndarray[delphi_real],
    gauss_density_map_midpoints_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_bool],
):
    """
    Calculates the spatial epsilon map with Gaussian and cutoff-based dielectric assignment
    for grid points and midpoints.

    This function applies density and epsilon cutoffs to define dielectric regions,
    and includes a smoothing pass to ensure continuity of the dielectric map.
    It also updates the ion exclusion map based on solute density.

    Args:
        num_cpu_threads (delphi_int): Number of CPU threads to use (currently unused).
        vaccum (delphi_bool): If True, the exterior dielectric constant is set to 1.0 (vacuum).
        exdi (delphi_real): Exterior dielectric constant.
        gapdi (delphi_real): Dielectric constant in the gap region.
        indi (delphi_real): Interior dielectric constant.
        density_cutoff (delphi_real): Density threshold below which the dielectric constant
                                      defaults to the exterior value.
        epsilon_cutoff (delphi_real): Epsilon threshold above which the dielectric constant
                                      defaults to the exterior value (if vacuum is True and
                                      density_cutoff is not applied).
        solute_density_threshold (delphi_real): Density threshold to identify solute region
                                                for ion exclusion and smoothing.
        eps_maxmin_ratio_threshold (delphi_real): Ratio threshold for epsilon values in a neighborhood
                                                  to trigger smoothing.
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid (nx, ny, nz).
        gauss_density_map_1d (np.ndarray[delphi_real]): 1D array of Gaussian densities at grid points.
        gauss_density_map_midpoints_1d (np.ndarray[delphi_real]): 1D array of Gaussian densities
                                                                  at midpoints.
        ion_exclusion_map_1d (np.ndarray[delphi_bool]): 1D boolean array indicating ion exclusion regions.
                                                        This array is modified in-place.

    Returns:
        tuple: A tuple containing:
            - epsilon_r_1d (np.ndarray[delphi_real]): Final spatial epsilon map at grid points.
            - epsilon_r_midpoints_1d (np.ndarray[delphi_real]): Final spatial epsilon map at midpoints.
    """
    num_grid_points = gauss_density_map_1d.shape[0]
    num_mid_points = gauss_density_map_midpoints_1d.shape[0]
    epsout = delphi_real(1.0) if vaccum else exdi

    spatial_outeps_limit = np.empty(2, dtype=delphi_real)
    spatial_outeps_limit[0] = exdi
    spatial_outeps_limit[1] = gapdi

    epsilon_r_1d = np.empty(num_grid_points, dtype=delphi_real)
    epsilon_r_1d.fill(epsout)
    # Calculate density dependent epsilon on gridpoints.
    # ATTENTION: here we use exdi in contrast to gapdi used with gaussian surface to mimic original gaussiancutoff.
    _calc_gaussian_epsilon_map(exdi, indi, gauss_density_map_1d, epsilon_r_1d)

    epsilon_r_midpoints_1d = np.empty(num_mid_points, dtype=delphi_real)
    epsilon_r_midpoints_1d.fill(epsout)

    z_stride: delphi_int = 1
    y_stride: delphi_int = grid_shape[2]
    x_stride: delphi_int = y_stride * grid_shape[1]

    last_grid_x = grid_shape[0] - 1
    last_grid_y = grid_shape[1] - 1
    last_grid_z = grid_shape[2] - 1

    # Loop 1: Assign midpoint epsilon values
    for ijk1d in prange(num_grid_points):
        ijk1d_x_3__x = ijk1d * 3
        ijk1d_x_3__y = ijk1d_x_3__x + 1
        ijk1d_x_3__z = ijk1d_x_3__x + 2
        regional_epsilon_limit = spatial_outeps_limit[
            delphi_int(ion_exclusion_map_1d[ijk1d])
        ]
        if ijk1d_x_3__z < num_mid_points:
            for idx in (ijk1d_x_3__x, ijk1d_x_3__y, ijk1d_x_3__z):
                rho = gauss_density_map_midpoints_1d[idx]
                if rho < solute_density_threshold:
                    epsilon_r_midpoints_1d[idx] = epsout
                else:
                    epsilon_r_midpoints_1d[idx] = (
                        rho * indi + (1.0 - rho) * regional_epsilon_limit
                    )

    # Loop 2: Apply density filter
    if vaccum:
        if density_cutoff > 0.0:
            for ijk1d_x_3 in prange(num_mid_points):
                if gauss_density_map_midpoints_1d[ijk1d_x_3] < density_cutoff:
                    epsilon_r_midpoints_1d[ijk1d_x_3] = epsout
        elif epsilon_cutoff > indi:
            for ijk1d_x_3 in prange(num_mid_points):
                if epsilon_r_midpoints_1d[ijk1d_x_3] > epsilon_cutoff:
                    epsilon_r_midpoints_1d[ijk1d_x_3] = epsout

    # Precompute offsets outside the loop
    minus_x_offset = -x_stride * 3
    minus_y_offset = -y_stride * 3 + 1
    minus_z_offset = -z_stride * 3 + 2
    # Loop 3: Smoothing pass (split into even/odd sets to avoid race conditions)
    num_grid_points_one_kind = num_grid_points // 2 + 1
    for index_kind in (0, 1):  # 0 = even, 1 = odd
        for ijk1d_strided in prange(num_grid_points_one_kind):
            ijk1d = ijk1d_strided * 2 + index_kind
            if ijk1d < num_grid_points:
                i = ijk1d // x_stride
                j = (ijk1d - i * x_stride) // y_stride
                k = ijk1d - i * x_stride - j * y_stride

                if 0 < i < last_grid_x and 0 < j < last_grid_y and 0 < k < last_grid_z:
                    ijk1d_x_3 = ijk1d * 3
                    minus_x_idx = ijk1d_x_3 + minus_x_offset
                    minus_y_idx = ijk1d_x_3 + minus_y_offset
                    minus_z_idx = ijk1d_x_3 + minus_z_offset

                    if (
                        gauss_density_map_midpoints_1d[ijk1d_x_3]
                        > solute_density_threshold
                        or gauss_density_map_midpoints_1d[ijk1d_x_3 + 1]
                        > solute_density_threshold
                        or gauss_density_map_midpoints_1d[ijk1d_x_3 + 2]
                        > solute_density_threshold
                        or gauss_density_map_midpoints_1d[minus_x_idx]
                        > solute_density_threshold
                        or gauss_density_map_midpoints_1d[minus_y_idx]
                        > solute_density_threshold
                        or gauss_density_map_midpoints_1d[minus_z_idx]
                        > solute_density_threshold
                    ):
                        if not ion_exclusion_map_1d[ijk1d]:
                            ion_exclusion_map_1d[ijk1d] = True

                        neigh_eps = np.empty(6, dtype=delphi_real)
                        neigh_eps[0] = epsilon_r_midpoints_1d[ijk1d_x_3]
                        neigh_eps[1] = epsilon_r_midpoints_1d[ijk1d_x_3 + 1]
                        neigh_eps[2] = epsilon_r_midpoints_1d[ijk1d_x_3 + 2]
                        neigh_eps[3] = epsilon_r_midpoints_1d[minus_x_idx]
                        neigh_eps[4] = epsilon_r_midpoints_1d[minus_y_idx]
                        neigh_eps[5] = epsilon_r_midpoints_1d[minus_z_idx]

                        epsmax_neigh = np.max(neigh_eps)
                        epsmin_neigh = np.min(neigh_eps)

                        if (
                            epsmin_neigh > 1e-6
                            and epsmax_neigh / epsmin_neigh < eps_maxmin_ratio_threshold
                        ):
                            epsmean_neigh = np.sum(neigh_eps) / 6.0
                            epsilon_r_midpoints_1d[ijk1d_x_3] = epsmean_neigh
                            epsilon_r_midpoints_1d[ijk1d_x_3 + 1] = epsmean_neigh
                            epsilon_r_midpoints_1d[ijk1d_x_3 + 2] = epsmean_neigh
                            epsilon_r_midpoints_1d[minus_x_idx] = epsmean_neigh
                            epsilon_r_midpoints_1d[minus_y_idx] = epsmean_neigh
                            epsilon_r_midpoints_1d[minus_z_idx] = epsmean_neigh
                            epsilon_r_1d[ijk1d] = epsmean_neigh

    return epsilon_r_1d, epsilon_r_midpoints_1d


@njit(nogil=True, cache=True)
def _calc_gaussian_dielectric_boundary_map(
    gaussian_density_midpoints_1d,
    dielectric_boundary_map_1d,
    grid_shape,
    density_threshold=0.02,
):
    """
    Calculates the Gaussian dielectric boundary map.

    This function identifies grid points that are part of the solute region
    (i.e., within the dielectric boundary) based on a density threshold
    applied to the Gaussian density at midpoints. Points above the threshold
    are marked as `False` (solute region), meaning they are not part of the
    dielectric boundary.

    Args:
        gaussian_density_midpoints_1d (np.ndarray[delphi_real]): 1D array of Gaussian densities
                                                                  at grid midpoints.
        dielectric_boundary_map_1d (np.ndarray[delphi_bool]): 1D boolean array to store the
                                                               dielectric boundary information.
                                                               This array is modified in-place.
                                                               Initially, all points are likely True (boundary/solvent).
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid (nx, ny, nz).
        density_threshold (delphi_real, optional): The density threshold to determine if a point
                                                   is within the solute region. Defaults to 0.02.
    """
    nx, ny, nz = grid_shape[0], grid_shape[1], grid_shape[2]
    y_stride: int = nz
    x_stride: int = ny * y_stride
    num_grid_points = nx * ny * nz
    num_mid_points = gaussian_density_midpoints_1d.size
    last_grid_indices = grid_shape - 1  # Calculate last indices once

    # Calculate offsets outside the loop
    offset_plus_x = 0
    offset_plus_y = 1
    offset_plus_z = 2
    offset_minus_x = -x_stride * 3
    offset_minus_y = -y_stride * 3 + 1
    offset_minus_z = -1 * 3 + 2  # or -1

    for ijk1d in prange(num_grid_points):
        i = ijk1d // x_stride
        remainder = ijk1d % x_stride
        j = remainder // y_stride
        k = remainder % y_stride

        # Process only inner grid points using pre-calculated last indices
        if (
            1 <= i < last_grid_indices[0]
            and 1 <= j < last_grid_indices[1]
            and 1 <= k < last_grid_indices[2]
        ):
            ijk1d_x3 = ijk1d * 3
            idx_plus_x = ijk1d_x3 + offset_plus_x
            idx_plus_y = ijk1d_x3 + offset_plus_y
            idx_plus_z = ijk1d_x3 + offset_plus_z
            idx_minus_x = ijk1d_x3 + offset_minus_x
            idx_minus_y = ijk1d_x3 + offset_minus_y
            idx_minus_z = ijk1d_x3 + offset_minus_z

            if (
                0 <= idx_plus_x < num_mid_points
                and 0 <= idx_plus_y < num_mid_points
                and 0 <= idx_plus_z < num_mid_points
                and 0 <= idx_minus_x < num_mid_points
                and 0 <= idx_minus_y < num_mid_points
                and 0 <= idx_minus_z < num_mid_points
            ):
                if (
                    gaussian_density_midpoints_1d[idx_plus_x] > density_threshold
                    or gaussian_density_midpoints_1d[idx_plus_y] > density_threshold
                    or gaussian_density_midpoints_1d[idx_plus_z] > density_threshold
                    or gaussian_density_midpoints_1d[idx_minus_x] > density_threshold
                    or gaussian_density_midpoints_1d[idx_minus_y] > density_threshold
                    or gaussian_density_midpoints_1d[idx_minus_z] > density_threshold
                ):
                    dielectric_boundary_map_1d[ijk1d] = False  # Solute region


@njit(nogil=True, boundscheck=False, parallel=True)
def _cpu_calc_gaussian_like_surface(
    srfexp: delphi_real,
    approx_zero: delphi_real,
    gaussian_density_1d: np.ndarray[delphi_real],
    surface_map_1d: np.ndarray[delphi_real],
):
    """
    Calculates a Gaussian-like surface map on the CPU.

    This function computes a surface value for each grid point based on its
    Gaussian density and a surface exponent. The result is stored in
    `surface_map_1d`.

    Args:
        srfexp (delphi_real): Surface exponent, controlling the steepness of the surface.
        approx_zero (delphi_real): A small value used for floating-point comparisons to avoid division by zero.
        gaussian_density_1d (np.ndarray[delphi_real]): 1D array of Gaussian densities at grid points.
        surface_map_1d (np.ndarray[delphi_real]): 1D array to store the calculated surface map.
                                                  This array is modified in-place.
    """
    for ijk1d in prange(surface_map_1d.size):
        if gaussian_density_1d[ijk1d] > approx_zero:
            surface_map_1d[ijk1d] = 1.0 / (
                1.0 + (1.0 / gaussian_density_1d[ijk1d] - 1.0) ** srfexp
            )
        else:
            surface_map_1d[ijk1d] = 0.0


@cuda.jit(cache=True)
def _cuda_calc_gaussian_like_surface(
    srfexp: delphi_real,
    approx_zero: delphi_real,
    gaussian_density_1d: np.ndarray[delphi_real],
    surface_map_1d: np.ndarray[delphi_real],
):
    """
    Calculates a Gaussian-like surface map on the GPU using CUDA.

    This is the CUDA kernel version of `_cpu_calc_gaussian_like_surface`,
    performing the same calculation but optimized for GPU execution.

    Args:
        srfexp (delphi_real): Surface exponent, controlling the steepness of the surface.
        approx_zero (delphi_real): A small value used for floating-point comparisons to avoid division by zero.
        gaussian_density_1d (np.ndarray[delphi_real]): 1D array of Gaussian densities at grid points.
        surface_map_1d (np.ndarray[delphi_real]): 1D array to store the calculated surface map.
                                                  This array is modified in-place.
    Dependencies:
    - `numba.cuda.jit` for CUDA kernel compilation.
    - `numba.cuda.grid` for thread indexing.
    - `numpy` for array handling.
    - `delphi_real` (type alias).
    """
    ijk1d = cuda.grid(1)
    if ijk1d < surface_map_1d.size:
        if gaussian_density_1d[ijk1d] > approx_zero:
            surface_map_1d[ijk1d] = 1.0 / (
                1.0 + (1.0 / gaussian_density_1d[ijk1d] - 1.0) ** srfexp
            )
        else:
            surface_map_1d[ijk1d] = 0.0


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_calc_grad_surface_map_analytical(
    gaussian_exponent: delphi_int,
    grid_spacing: delphi_real,
    surface_density_exponent: delphi_real,
    approx_zero: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    density_map_gridpoints_1d: np.ndarray[delphi_real],
    surface_map_1d: np.ndarray[delphi_real],
    grad_surface_map_1d: np.ndarray[delphi_real],
    # --- Added Voxel Map Parameters ---
    neighbor_voxel_atom_ids_flat: np.ndarray[delphi_int],
    neighbor_voxel_atom_start_index: np.ndarray[delphi_int],
    neighbor_voxel_atom_end_index: np.ndarray[delphi_int],
    voxel_map_origin: np.ndarray[delphi_real],
    voxel_map_shape: np.ndarray[delphi_int],
    voxel_map_scale: delphi_real,
    # ----------------------------------
) -> None:
    """
    Calculates the analytical gradient of the Gaussian-like surface map on the CPU.

    This function computes the gradient of the surface map at each grid point
    by analytically differentiating the Gaussian density function. It uses a
    neighbor voxel map to efficiently identify relevant atoms for each grid point.

    Args:
        gaussian_exponent (delphi_int): The exponent used in the Gaussian function.
        grid_spacing (delphi_real): The spacing between grid points.
        surface_density_exponent (delphi_real): The exponent used in the surface function.
        approx_zero (delphi_real): A small value used for floating-point comparisons.
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid (nx, ny, nz).
        grid_origin (np.ndarray[delphi_real]): Origin (x, y, z) of the grid.
        atoms_data (np.ndarray[delphi_real]): 2D array of atom data (coordinates, radius, sigma, etc.).
        density_map_gridpoints_1d (np.ndarray[delphi_real]): 1D array of Gaussian densities at grid points.
        surface_map_1d (np.ndarray[delphi_real]): 1D array of calculated surface values at grid points.
        grad_surface_map_1d (np.ndarray[delphi_real]): 1D array to store the x, y, z components
                                                       of the gradient at each grid point.
                                                       This array is modified in-place.
        neighbor_voxel_atom_ids_flat (np.ndarray[delphi_int]): Flattened array of atom IDs per neighbor voxel.
        neighbor_voxel_atom_start_index (np.ndarray[delphi_int]): Start indices for atoms in
                                                                  `neighbor_voxel_atom_ids_flat` for each voxel.
        neighbor_voxel_atom_end_index (np.ndarray[delphi_int]): End indices for atoms in
                                                                `neighbor_voxel_atom_ids_flat` for each voxel.
        voxel_map_origin (np.ndarray[delphi_real]): Origin of the voxel map.
        voxel_map_shape (np.ndarray[delphi_int]): Shape of the voxel map.
        voxel_map_scale (delphi_real): Scale of the voxel map.
    Dependencies:
    - `numba.njit` for JIT compilation.
    - `numba.prange` for parallel loop execution.
    - `numpy` for array handling.
    - `math` for `exp` function.
    - `delphi_real`, `delphi_int` (type aliases).
    - `ATOMFIELD_X`, `ATOMFIELD_Y`, `ATOMFIELD_Z`, `ATOMFIELD_RADIUS`, `ATOMFIELD_GAUSS_SIGMA` (constants).
    """
    gaussian_exponent_minus_1 = gaussian_exponent - 1
    gaussian_exponent_x_2 = gaussian_exponent * 2
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    num_grid_points = grid_shape[0] * x_stride

    num_atoms = atoms_data.shape[0]

    v_origin = voxel_map_origin
    v_shape = voxel_map_shape
    v_scale = voxel_map_scale

    for ijk1d in prange(num_grid_points):  # Parallel loop over fine grid points
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = (ijk1d - i * x_stride) - j * y_stride
        ijk1d_x_3 = 3 * ijk1d

        # real-space grid position
        grid_pos_x = i * grid_spacing + grid_origin[0]
        grid_pos_y = j * grid_spacing + grid_origin[1]
        grid_pos_z = k * grid_spacing + grid_origin[2]

        total_density = density_map_gridpoints_1d[ijk1d]

        srf_square: delphi_real = 0.0
        srf_func_dr: delphi_real = 0.0

        if total_density > approx_zero:
            # Calculate srf_square and srf_func_dr once per grid point if density is valid
            srf_val = surface_map_1d[ijk1d]
            if srf_val > approx_zero:  # Check srf_val before squaring
                srf_square = srf_val**2
            # Avoid issues if (1/total_density - 1) is negative or zero for fractional powers
            term = (1.0 / total_density) - 1.0
            if abs(term) > approx_zero or surface_density_exponent == delphi_int(
                surface_density_exponent
            ):  # Check for integer exponent
                srf_func_dr = surface_density_exponent * (
                    term ** (surface_density_exponent - 1.0)
                )

        (
            grad_surface_map_1d[ijk1d_x_3],
            grad_surface_map_1d[ijk1d_x_3 + 1],
            grad_surface_map_1d[ijk1d_x_3 + 2],
        ) = (0.0, 0.0, 0.0)

        # --- Map fine grid point to central coarse voxel ---
        central_vx = max(
            0, min(delphi_int((grid_pos_x - v_origin[0]) * v_scale), v_shape[0])
        )
        central_vy = max(
            0, min(delphi_int((grid_pos_y - v_origin[1]) * v_scale), v_shape[1])
        )
        central_vz = max(
            0, min(delphi_int((grid_pos_z - v_origin[2]) * v_scale), v_shape[2])
        )

        if (
            0 <= central_vx <= v_shape[0]
            and 0 <= central_vy <= v_shape[1]
            and 0 <= central_vz <= v_shape[2]
        ):
            start = neighbor_voxel_atom_start_index[central_vx, central_vy, central_vz]
            end = neighbor_voxel_atom_end_index[central_vx, central_vy, central_vz]
            if start <= end:
                for atom_list_idx in range(start, end + 1):
                    # Atom index is 1-based in ids array, to map to 0-based index -1 is needed.
                    atom_id_raw = neighbor_voxel_atom_ids_flat[atom_list_idx]
                    if atom_id_raw == 0:
                        continue  # Sentinel: skip

                    atom_idx = atom_id_raw - 1  # Now safe to subtract
                    # Ensure atom_idx is valid before accessing atoms_data
                    if 0 <= atom_idx < num_atoms:
                        this_atom = atoms_data[atom_idx]

                        # --- Original analytical gradient calculation logic for this atom ---
                        atom_crd_x = this_atom[ATOMFIELD_X]
                        atom_crd_y = this_atom[ATOMFIELD_Y]
                        atom_crd_z = this_atom[ATOMFIELD_Z]
                        atom_radius = this_atom[ATOMFIELD_RADIUS]
                        atom_sigma = this_atom[ATOMFIELD_GAUSS_SIGMA]

                        # Check if atom has valid sigma and radius
                        if atom_sigma > approx_zero and atom_radius > approx_zero:
                            delta_rx = grid_pos_x - atom_crd_x
                            delta_ry = grid_pos_y - atom_crd_y
                            delta_rz = grid_pos_z - atom_crd_z
                            atom_sigma_x_radius = atom_sigma * atom_radius
                            atom_sigma_x_radius_square = atom_sigma_x_radius**2
                            dist_square = delta_rx**2 + delta_ry**2 + delta_rz**2

                            if (
                                dist_square > approx_zero
                                and atom_sigma_x_radius_square > approx_zero
                            ):  # Avoid division by zero and point at atom center
                                dist_sq_norm = dist_square / atom_sigma_x_radius_square
                                g_arg = dist_sq_norm**gaussian_exponent

                                # Check if g_arg is too large (prevents overflow in exp)
                                if (
                                    g_arg < 700.0
                                ):  # A common practical limit for exp(x) to avoid overflow for float64
                                    density = math.exp(-g_arg)

                                    # Ensure neither density nor (1.0 - density) is too close to zero for division
                                    if (
                                        abs(1.0 - density) > approx_zero
                                        and density > approx_zero
                                    ):
                                        # Calculate components of the gradient contribution
                                        density_factor_numerator = density * (
                                            1.0 - total_density
                                        )
                                        density_factor_denominator = 1.0 - density

                                        # Check if density_factor_denominator is not zero
                                        if (
                                            abs(density_factor_denominator)
                                            > approx_zero
                                        ):
                                            density_factor = (
                                                density_factor_numerator
                                                / density_factor_denominator
                                            )

                                            dist_factor = (
                                                dist_square**gaussian_exponent_minus_1
                                                if gaussian_exponent_minus_1 >= 0
                                                else 1.0
                                                / (
                                                    dist_square
                                                    ** abs(gaussian_exponent_minus_1)
                                                )
                                            )  # Handle negative exponents if needed, though exp is usually > 0
                                            atom_factor = gaussian_exponent_x_2 / (
                                                atom_sigma_x_radius_square
                                                ** gaussian_exponent
                                            )

                                            srf_factor_base = (
                                                atom_factor
                                                * density_factor
                                                * dist_factor
                                            )

                                            srf_factor: delphi_real = 0.0
                                            if (
                                                total_density > approx_zero
                                                and srf_square > approx_zero
                                                and abs(srf_func_dr) > approx_zero
                                            ):
                                                srf_factor = (
                                                    srf_factor_base
                                                    * srf_func_dr
                                                    * (1.0 / total_density**2)
                                                    * srf_square
                                                )  # Combine factors

                                            # Accumulate gradient contribution from this atom
                                            # Note the negative sign in the original formula
                                            grad_surface_map_1d[ijk1d_x_3] -= (
                                                srf_factor * delta_rx
                                            )
                                            grad_surface_map_1d[ijk1d_x_3 + 1] -= (
                                                srf_factor * delta_ry
                                            )
                                            grad_surface_map_1d[ijk1d_x_3 + 2] -= (
                                                srf_factor * delta_rz
                                            )
                            # --- End of original analytical gradient calculation logic for this atom ---
        # --- End of neighbor voxel loop ---
    # --- End of fine grid point loop ---


@cuda.jit(cache=True)
def _cuda_calc_grad_surface_map_analytical(
    gaussian_exponent: delphi_int,
    grid_spacing: delphi_real,
    srfdensityexp: delphi_real,
    approx_zero: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    densitymap_gridpoints_1d: np.ndarray[delphi_real],
    surface_map_1d: np.ndarray[delphi_real],
    grad_surface_map_1d: np.ndarray[delphi_real],
    # --- Added Voxel Map Parameters ---
    neighbor_voxel_atom_ids_flat: np.ndarray[delphi_int],
    neighbor_voxel_atom_start_index: np.ndarray[delphi_int],
    neighbor_voxel_atom_end_index: np.ndarray[delphi_int],
    voxel_map_origin: np.ndarray[delphi_real],
    voxel_map_shape: np.ndarray[delphi_int],
    voxel_map_scale: delphi_real,
    # ----------------------------------
) -> None:
    """
    Calculates the analytical gradient of the Gaussian-like surface map on the GPU using CUDA.

    This is the CUDA kernel version of `_cpu_calc_grad_surface_map_analytical`,
    performing the same analytical differentiation of the Gaussian density function
    but optimized for GPU execution. It utilizes a neighbor voxel map for efficient
    atom lookup.

    Args:
        gaussian_exponent (delphi_int): The exponent used in the Gaussian function.
        grid_spacing (delphi_real): The spacing between grid points.
        srfdensityexp (delphi_real): The exponent used in the surface function.
        approx_zero (delphi_real): A small value for floating-point comparisons.
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid (nx, ny, nz).
        grid_origin (np.ndarray[delphi_real]): Origin (x, y, z) of the grid.
        atoms_data (np.ndarray[delphi_real]): 2D array of atom data (coordinates, radius, sigma, etc.).
        densitymap_gridpoints_1d (np.ndarray[delphi_real]): 1D array of Gaussian densities at grid points.
        surface_map_1d (np.ndarray[delphi_real]): 1D array of calculated surface values at grid points.
        grad_surface_map_1d (np.ndarray[delphi_real]): 1D array to store the x, y, z components
                                                       of the gradient at each grid point.
                                                       This array is modified in-place.
        neighbor_voxel_atom_ids_flat (np.ndarray[delphi_int]): Flattened array of atom IDs per neighbor voxel.
        neighbor_voxel_atom_start_index (np.ndarray[delphi_int]): Start indices for atoms in
                                                                  `neighbor_voxel_atom_ids_flat` for each voxel.
        neighbor_voxel_atom_end_index (np.ndarray[delphi_int]): End indices for atoms in
                                                                `neighbor_voxel_atom_ids_flat` for each voxel.
        voxel_map_origin (np.ndarray[delphi_real]): Origin of the voxel map.
        voxel_map_shape (np.ndarray[delphi_int]): Shape of the voxel map.
        voxel_map_scale (delphi_real): Scale of the voxel map.
    Dependencies:
    - `numba.cuda.jit` for CUDA kernel compilation.
    - `numba.cuda.grid` for thread indexing.
    - `numba.cuda.libdevice.fabs`, `numba.cuda.libdevice.powf`, `numba.cuda.libdevice.expf` for CUDA math functions.
    - `numpy` for array handling.
    - `delphi_real`, `delphi_int` (type aliases).
    - `ATOMFIELD_X`, `ATOMFIELD_Y`, `ATOMFIELD_Z`, `ATOMFIELD_RADIUS`, `ATOMFIELD_GAUSS_SIGMA` (constants).
    """
    gaussian_exponent_minus_1 = gaussian_exponent - 1
    gaussian_exponent_x_2 = gaussian_exponent * 2
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride

    num_atoms = atoms_data.shape[0]
    num_grid_points = (
        grid_shape[0] * grid_shape[1] * grid_shape[2]
    )  # Correct total grid points calculation
    ijk1d = cuda.grid(1)

    if ijk1d < num_grid_points:
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride  # Correct k index calculation
        ijk1d_x_3 = 3 * ijk1d

        # real-space grid position
        grid_pos_x = i * grid_spacing + grid_origin[0]
        grid_pos_y = j * grid_spacing + grid_origin[1]
        grid_pos_z = k * grid_spacing + grid_origin[2]

        total_density = densitymap_gridpoints_1d[ijk1d]

        srf_square: delphi_real = 0.0
        srf_func_dr: delphi_real = 0.0

        if total_density > approx_zero:
            # Calculate srf_square and srf_func_dr once per grid point if density is valid
            srf_val = surface_map_1d[ijk1d]
            if srf_val > approx_zero:  # Check srf_val before squaring
                srf_square = srf_val**2
            # Avoid issues if (1/total_density - 1) is negative or zero for fractional powers
            term = (1.0 / total_density) - 1.0
            if cuda.libdevice.fabs(term) > approx_zero or srfdensityexp == delphi_int(
                srfdensityexp
            ):  # Check for integer exponent
                srf_func_dr = srfdensityexp * (
                    cuda.libdevice.powf(term, srfdensityexp - 1.0)
                    if isinstance(term, float) or term > 0
                    else cuda.libdevice.pow(term, srfdensityexp - 1.0)
                )

        # Initialize gradient components for this grid point
        grad_x: delphi_real = 0.0
        grad_y: delphi_real = 0.0
        grad_z: delphi_real = 0.0

        v_origin = voxel_map_origin  # Use device pointer directly
        v_shape = voxel_map_shape  # Use device pointer directly
        v_scale = voxel_map_scale  # Use device value directly

        # --- Map fine grid point to central coarse voxel ---
        central_vx = max(
            0, min(delphi_int((grid_pos_x - v_origin[0]) * v_scale), v_shape[0])
        )
        central_vy = max(
            0, min(delphi_int((grid_pos_y - v_origin[1]) * v_scale), v_shape[1])
        )
        central_vz = max(
            0, min(delphi_int((grid_pos_z - v_origin[2]) * v_scale), v_shape[2])
        )

        if (
            0 <= central_vx <= v_shape[0]
            and 0 <= central_vy <= v_shape[1]
            and 0 <= central_vz <= v_shape[2]
        ):
            start = neighbor_voxel_atom_start_index[central_vx, central_vy, central_vz]
            end = neighbor_voxel_atom_end_index[central_vx, central_vy, central_vz]
            if start <= end:
                for atom_list_idx in range(start, end + 1):
                    # Atom index is 1-based in ids array, to map to 0-based index -1 is needed.
                    atom_idx = neighbor_voxel_atom_ids_flat[atom_list_idx] - 1
                    if atom_idx == -1:
                        continue

                    # Ensure atom_idx is valid before accessing atoms_data
                    if 0 <= atom_idx < num_atoms:
                        this_atom = atoms_data[atom_idx]

                        # --- Original analytical gradient calculation logic for this atom ---
                        atom_crd_x = this_atom[ATOMFIELD_X]
                        atom_crd_y = this_atom[ATOMFIELD_Y]
                        atom_crd_z = this_atom[ATOMFIELD_Z]
                        atom_radius = this_atom[ATOMFIELD_RADIUS]
                        atom_sigma = this_atom[ATOMFIELD_GAUSS_SIGMA]

                        # Check if atom has valid sigma and radius
                        if atom_sigma > approx_zero and atom_radius > approx_zero:
                            delta_rx = grid_pos_x - atom_crd_x
                            delta_ry = grid_pos_y - atom_crd_y
                            delta_rz = grid_pos_z - atom_crd_z
                            atom_sigma_x_radius = atom_sigma * atom_radius
                            atom_sigma_x_radius_sq = atom_sigma_x_radius**2
                            dist_square = delta_rx**2 + delta_ry**2 + delta_rz**2

                            if (
                                dist_square > approx_zero
                                and atom_sigma_x_radius_sq > approx_zero
                            ):  # Avoid division by zero and point at atom center
                                dist_sq_norm = dist_square / atom_sigma_x_radius_sq
                                # Use CUDA math functions
                                g_arg = cuda.libdevice.powf(
                                    dist_sq_norm, gaussian_exponent
                                )

                                # Check if g_arg is too large (prevents overflow in exp)
                                if (
                                    g_arg < 700.0
                                ):  # A common practical limit for exp(x) to avoid overflow for float64
                                    density = cuda.libdevice.expf(-g_arg)

                                    # Ensure neither (1.0 - density) nor density is too close to zero for division
                                    if (
                                        cuda.libdevice.fabs(1.0 - density) > approx_zero
                                        and density > approx_zero
                                    ):
                                        # Calculate components of the gradient contribution
                                        density_factor_numerator = density * (
                                            1.0 - total_density
                                        )
                                        density_factor_denominator = 1.0 - density

                                        # Check if density_factor_denominator is not zero
                                        if (
                                            cuda.libdevice.fabs(
                                                density_factor_denominator
                                            )
                                            > approx_zero
                                        ):
                                            density_factor = (
                                                density_factor_numerator
                                                / density_factor_denominator
                                            )

                                            dist_factor = (
                                                cuda.libdevice.powf(
                                                    dist_square,
                                                    gaussian_exponent_minus_1,
                                                )
                                                if gaussian_exponent_minus_1 >= 0
                                                else 1.0
                                                / (
                                                    cuda.libdevice.powf(
                                                        dist_square,
                                                        cuda.libdevice.fabs(
                                                            gaussian_exponent_minus_1
                                                        ),
                                                    )
                                                )
                                            )
                                            atom_factor = gaussian_exponent_x_2 / (
                                                cuda.libdevice.powf(
                                                    atom_sigma_x_radius_sq,
                                                    gaussian_exponent,
                                                )
                                            )

                                            srf_factor_base = (
                                                atom_factor
                                                * density_factor
                                                * dist_factor
                                            )

                                            srf_factor: delphi_real = 0.0
                                            if (
                                                total_density > approx_zero
                                                and srf_square > approx_zero
                                                and cuda.libdevice.fabs(srf_func_dr)
                                                > approx_zero
                                            ):
                                                srf_factor = (
                                                    srf_factor_base
                                                    * srf_func_dr
                                                    * (
                                                        1.0
                                                        / cuda.libdevice.powf(
                                                            total_density, 2.0
                                                        )
                                                    )
                                                    * srf_square
                                                )  # Combine factors

                                            # Accumulate gradient contribution from this atom
                                            # Note the negative sign in the original formula
                                            grad_x -= srf_factor * delta_rx
                                            grad_y -= srf_factor * delta_ry
                                            grad_z -= srf_factor * delta_rz
                        # --- End of original analytical gradient calculation logic for this atom ---
        # --- End of neighbor voxel loop ---

        # Assign accumulated gradient components to the output array
        grad_surface_map_1d[ijk1d_x_3] = grad_x
        grad_surface_map_1d[ijk1d_x_3 + 1] = grad_y
        grad_surface_map_1d[ijk1d_x_3 + 2] = grad_z
    # --- End of fine grid point loop ---


def calc_grad_surface_map_analytical(
    platform: Platform,
    num_cuda_threads: delphi_int,
    gaussian_exponent: delphi_int,
    grid_spacing: delphi_real,
    srfdensityexp: delphi_real,
    approx_zero: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    densitymap_gridpoints_1d: np.ndarray[delphi_real],
    surface_map_1d: np.ndarray[delphi_real],
    grad_surface_map_1d: np.ndarray[delphi_real],
    # --- Pass Voxel Map Parameters ---
    voxel_atom_ids: np.ndarray[delphi_int],
    voxel_atom_start_index: np.ndarray[delphi_int],
    voxel_atom_end_index: np.ndarray[delphi_int],
    voxel_map_origin: np.ndarray[delphi_real],
    voxel_map_shape: np.ndarray[delphi_int],
    voxel_map_scale: delphi_real,
    # -------------------------------
) -> None:
    """
    Manages the calculation of the analytical gradient of the Gaussian-like surface map,
    dispatching to either CPU or CUDA implementations based on the `platform` setting.

    This function sets up the necessary data structures for the neighbor voxel map
    which helps to efficiently identify relevant atoms for each grid point during
    gradient calculation. It then calls the appropriate CPU or CUDA kernel.

    Args:
        platform (Platform): An object defining the active platform (CPU or CUDA)
                             and its configuration.
        num_cuda_threads (delphi_int): Number of CUDA threads per block if running on GPU.
        gaussian_exponent (delphi_int): The exponent used in the Gaussian function.
        grid_spacing (delphi_real): The spacing between grid points.
        srfdensityexp (delphi_real): The exponent used in the surface function.
        approx_zero (delphi_real): A small value for floating-point comparisons.
        grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid (nx, ny, nz).
        grid_origin (np.ndarray[delphi_real]): Origin (x, y, z) of the grid.
        atoms_data (np.ndarray[delphi_real]): 2D array of atom data (coordinates, radius, sigma, etc.).
        densitymap_gridpoints_1d (np.ndarray[delphi_real]): 1D array of Gaussian densities at grid points.
        surface_map_1d (np.ndarray[delphi_real]): 1D array of calculated surface values at grid points.
        grad_surface_map_1d (np.ndarray[delphi_real]): 1D array to store the x, y, z components
                                                       of the gradient at each grid point.
                                                       This array is modified in-place.
        voxel_atom_ids (np.ndarray[delphi_int]): Original flattened array of atom IDs within voxels.
        voxel_atom_start_index (np.ndarray[delphi_int]): Original start indices for atoms in `voxel_atom_ids`.
        voxel_atom_end_index (np.ndarray[delphi_int]): Original end indices for atoms in `voxel_atom_ids`.
        voxel_map_origin (np.ndarray[delphi_real]): Origin of the voxel map.
        voxel_map_shape (np.ndarray[delphi_int]): Shape of the voxel map.
        voxel_map_scale (delphi_real): Scale of the voxel map.
    Dependencies:
    - `numpy` for array handling.
    - `Platform` (custom class for platform management).
    - `set_num_threads` (utility function).
    - `_cpu_calc_grad_surface_map_analytical` (CPU kernel).
    - `_cuda_calc_grad_surface_map_analytical` (CUDA kernel).
    - `numba.cuda` for CUDA device operations (to_device, device_array_like, copy_to_host).
    - `_build_neighbor_voxel_atom_index_map` (utility function to build voxel map).
    - `pretty_print_neighbor_voxel_unique_atom_ids` (utility for debugging).
    - `NEIGHBOR_VOXEL_REL_COORDS` (constant).
    - `delphi_real`, `delphi_int` (type aliases).
    """
    num_atoms = atoms_data.shape[0]

    # Step 1: build neighbor voxel map
    (
        neighbor_voxel_atom_ids_flat,
        neighbor_voxel_start_index,
        neighbor_voxel_end_index,
        actual_neighbor_ids_count,
    ) = build_neighbor_voxel_unique_atom_index_map(
        num_atoms,
        voxel_atom_ids,
        voxel_atom_start_index,
        voxel_atom_end_index,
        voxel_map_shape,
    )

    if platform.active == "cpu":
        set_num_threads(platform.names["cpu"]["num_threads"])
        _cpu_calc_grad_surface_map_analytical(
            gaussian_exponent,
            grid_spacing,
            srfdensityexp,
            approx_zero,
            grid_shape,
            grid_origin,
            atoms_data,
            densitymap_gridpoints_1d,
            surface_map_1d,
            grad_surface_map_1d,
            # --- Pass Voxel Map Parameters ---
            neighbor_voxel_atom_ids_flat,
            neighbor_voxel_start_index,
            neighbor_voxel_end_index,
            voxel_map_origin,
            voxel_map_shape,
            voxel_map_scale,
            # -------------------------------
        )
    elif platform.active == "cuda":
        num_grid_points = np.prod(grid_shape)
        num_blocks = (num_grid_points + num_cuda_threads - 1) // num_cuda_threads
        # Transfer necessary data to device
        grid_shape_device = cuda.to_device(grid_shape)
        grid_origin_device = cuda.to_device(grid_origin)
        atoms_data_device = cuda.to_device(atoms_data)
        densitymap_gridpoints_1d_device = cuda.to_device(densitymap_gridpoints_1d)
        surface_map_1d_device = cuda.to_device(surface_map_1d)
        grad_surface_map_1d_device = cuda.device_array_like(grad_surface_map_1d)

        # Transfer Voxel Map Parameters to device
        neighbor_voxel_atom_ids_flat_device = cuda.to_device(
            neighbor_voxel_atom_ids_flat
        )
        neighbor_voxel_start_index_device = cuda.to_device(neighbor_voxel_start_index)
        neighbor_voxel_end_index_device = cuda.to_device(neighbor_voxel_end_index)
        voxel_map_origin_device = cuda.to_device(voxel_map_origin)
        voxel_map_shape_device = cuda.to_device(voxel_map_shape)

        _cuda_calc_grad_surface_map_analytical[num_blocks, num_cuda_threads](
            gaussian_exponent,
            delphi_real(grid_spacing),  # Pass scalar directly
            srfdensityexp,  # Pass scalar directly
            delphi_real(approx_zero),  # Pass scalar directly
            grid_shape_device,
            grid_origin_device,
            atoms_data_device,
            densitymap_gridpoints_1d_device,
            surface_map_1d_device,
            grad_surface_map_1d_device,
            # --- Pass Voxel Map Parameters (device pointers) ---
            neighbor_voxel_atom_ids_flat_device,
            neighbor_voxel_start_index_device,
            neighbor_voxel_end_index_device,
            voxel_map_origin_device,
            voxel_map_shape_device,
            delphi_real(voxel_map_scale),  # Pass scalar directly
            # ---------------------------------------------------
        )
        grad_surface_map_1d_device.copy_to_host(grad_surface_map_1d)

        # Explicitly free device memory (optional, but good practice)
        del grid_shape_device
        del grid_origin_device
        del atoms_data_device
        del densitymap_gridpoints_1d_device
        del surface_map_1d_device
        del grad_surface_map_1d_device
        del neighbor_voxel_atom_ids_flat_device
        del neighbor_voxel_start_index_device
        del neighbor_voxel_end_index_device
        del voxel_map_origin_device
        del voxel_map_shape_device
        # voxel_map_scale was scalar, no device memory to free


def calc_gaussian_like_surface(
    platform,
    num_cuda_threads,
    surf_den_exp_scaled,
    approx_zero,
    gauss_density_solvent_1d,
    surface_map_1d,
):
    """
    Calculates the Gaussian-like surface map, dispatching to either CPU or CUDA
    implementations based on the `platform` setting.

    Args:
        platform (Platform): An object defining the active platform (CPU or CUDA)
                             and its configuration.
        num_cuda_threads (delphi_int): Number of CUDA threads per block if running on GPU.
        surf_den_exp_scaled (delphi_real): Surface exponent, controlling the steepness of the surface.
        approx_zero (delphi_real): A small value for floating-point comparisons.
        gauss_density_solvent_1d (np.ndarray[delphi_real]): 1D array of Gaussian densities for solvent.
        surface_map_1d (np.ndarray[delphi_real]): 1D array to store the calculated surface map.
                                                  This array is modified in-place.
    Dependencies:
    - `numpy` for array handling.
    - `Platform` (custom class for platform management).
    - `set_num_threads` (utility function).
    - `_cpu_calc_gaussian_like_surface` (CPU kernel).
    - `_cuda_calc_gaussian_like_surface` (CUDA kernel).
    - `numba.cuda` for CUDA device operations (to_device, device_array_like, copy_to_host).
    - `delphi_real` (type alias).
    - `APPROX_ZERO` (constant, though `approx_zero` is passed as an argument).
    """
    if platform.active == "cpu":
        set_num_threads(platform.names["cpu"]["num_threads"])
        _cpu_calc_gaussian_like_surface(
            surf_den_exp_scaled,
            approx_zero,
            gauss_density_solvent_1d,
            surface_map_1d,
        )
    elif platform.active == "cuda":
        num_blocks = (
            gauss_density_solvent_1d.size + num_cuda_threads - 1
        ) // num_cuda_threads
        gauss_density_solvent_1d_device = cuda.to_device(gauss_density_solvent_1d)
        surface_map_1d_device = cuda.device_array_like(surface_map_1d)
        _cuda_calc_gaussian_like_surface[num_blocks, num_cuda_threads](
            surf_den_exp_scaled,
            delphi_real(APPROX_ZERO),  # Ensure approx_zero is correct type
            gauss_density_solvent_1d_device,
            surface_map_1d_device,
        )
        surface_map_1d_device.copy_to_host(surface_map_1d)
        gauss_density_solvent_1d_device = None
        surface_map_1d_device = None
