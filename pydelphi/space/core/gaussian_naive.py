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
    NEIGHBOR_VOXEL_RELATIVE_COORDINATES as NEIGHBOR_VOXEL_REL_COORDS,
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


@njit(nogil=True, cache=True)
def _build_neighbor_voxel_atom_index_map(
    voxel_atom_ids: np.ndarray,
    voxel_atom_start_index: np.ndarray,
    voxel_atom_end_index: np.ndarray,
    voxel_shape: np.ndarray,  # (nx, ny, nz)
    neighbor_voxel_atom_ids_flat: np.ndarray,
    neighbor_voxel_start_index: np.ndarray,
    neighbor_voxel_end_index: np.ndarray,
    seen_atoms: np.ndarray,  # shape (num_atoms,), dtype=uint8
    neighbor_offsets: np.ndarray,  # shape (27, 3), NEIGHBOR_VOXEL_REL_COORDS
) -> int:
    """
    Populates neighbor_voxel_atom_ids_flat, neighbor_voxel_start_index, neighbor_voxel_end_index
    for each voxel in the 3D voxel grid, collecting unique atoms from all 27 neighboring voxels.

    Parameters
    ----------
    voxel_atom_ids : 1D array of atom indices assigned to voxels.
    voxel_atom_start_index, voxel_atom_end_index : (nx, ny, nz) arrays marking start/end in voxel_atom_ids.
    voxel_shape : tuple of 3 ints, the voxel grid shape (nx, ny, nz).
    neighbor_voxel_atom_ids_flat : 1D preallocated array for flattened neighbor atom indices.
    neighbor_voxel_start_index, neighbor_voxel_end_index : (nx, ny, nz) arrays to mark start/end per voxel.
    seen_atoms : (num_atoms,) scratch array for duplicate tracking; will be zeroed after each voxel.
    neighbor_offsets : (27, 3) array of relative offsets (e.g., NEIGHBOR_VOXEL_REL_COORDS).
    """
    nx, ny, nz = voxel_shape
    num_atoms = seen_atoms.shape[0]
    output_index = 0

    for i in range(nx + 1):
        for j in range(ny + 1):
            for k in range(nz + 1):
                start_pos = output_index

                for n in range(neighbor_offsets.shape[0]):
                    dx, dy, dz = neighbor_offsets[n]
                    vx = i + dx
                    vy = j + dy
                    vz = k + dz

                    if 0 <= vx <= nx and 0 <= vy <= ny and 0 <= vz <= nz:
                        start_idx = voxel_atom_start_index[vx, vy, vz]
                        end_idx = voxel_atom_end_index[vx, vy, vz]

                        if start_idx <= end_idx:
                            for atom_ptr in range(start_idx, end_idx + 1):
                                atom_index = voxel_atom_ids[atom_ptr]
                                if seen_atoms[atom_index] == 0:
                                    seen_atoms[atom_index] = 1
                                    neighbor_voxel_atom_ids_flat[output_index] = (
                                        atom_index
                                    )
                                    output_index += 1

                neighbor_voxel_start_index[i, j, k] = start_pos
                neighbor_voxel_end_index[i, j, k] = output_index - 1

                # Fully reset seen_atoms for next voxel
                for a in range(num_atoms):
                    seen_atoms[a] = 0
    return output_index


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_calc_gaussian_density_map(
    generate_ion_exclusion_map: delphi_bool,
    scale: delphi_real,
    gaussian_exponent: delphi_int,
    surface_offset: delphi_real,
    atom_influence_radius: delphi_real,
    salt_radius: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    gauss_density_map_1d: np.ndarray[delphi_real],
    gauss_density_map_midpoints_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_bool],
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

    num_atoms = atoms_data.shape[0]

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

        # --- Optimized Inner Loop Start ---
        for atom_idx in range(num_atoms):
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
                mult_inv = 0.0  # Very large inverse
                continue
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
                density = math.exp(-((dist_square * mult_inv) ** gaussian_exponent))
                total_density = 1.0 - (1.0 - total_density) * (1.0 - density)
                if valid_half_h_x:
                    r_half_h_dx = r_dx + grid_spacing_half
                    dist_sq_hx = r_half_h_dx**2 + r_dy**2 + r_dz**2
                    if dist_sq_hx <= atom_influence_radius_square:
                        density_hx = math.exp(
                            -((dist_sq_hx * mult_inv) ** gaussian_exponent)
                        )
                        total_density_half_hx = 1.0 - (1.0 - total_density_half_hx) * (
                            1.0 - density_hx
                        )
                if valid_half_h_y:
                    r_half_h_dy = r_dy + grid_spacing_half
                    dist_sq_hy = r_dx**2 + r_half_h_dy**2 + r_dz**2
                    if dist_sq_hy <= atom_influence_radius_square:
                        density_hy = math.exp(
                            -((dist_sq_hy * mult_inv) ** gaussian_exponent)
                        )
                        total_density_half_hy = 1.0 - (1.0 - total_density_half_hy) * (
                            1.0 - density_hy
                        )
                if valid_half_h_z:
                    r_half_h_dz = r_dz + grid_spacing_half
                    dist_sq_hz = r_dx**2 + r_dy**2 + r_half_h_dz**2
                    if dist_sq_hz <= atom_influence_radius_square:
                        density_hz = math.exp(
                            -((dist_sq_hz * mult_inv) ** gaussian_exponent)
                        )
                        total_density_half_hz = 1.0 - (1.0 - total_density_half_hz) * (
                            1.0 - density_hz
                        )
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
    surface_offset: delphi_real,
    atom_influence_radius: delphi_real,
    salt_radius: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    gauss_density_map_1d: np.ndarray[delphi_real],
    gauss_density_map_midpoints_1d: np.ndarray[delphi_real],
    ion_exclusion_map_1d: np.ndarray[delphi_bool],
) -> None:
    grid_spacing = 1.0 / scale
    grid_spacing_half = 0.5 * grid_spacing
    atom_influence_radius_square = atom_influence_radius * atom_influence_radius
    num_atoms = atoms_data.shape[0]
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    ijk1d = cuda.grid(1)
    num_grid_points = grid_shape[0] * grid_shape[1] * grid_shape[2]

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

        for atom_idx in range(num_atoms):
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
                ion_exclusion_limit = atom_radius + salt_radius
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
                        total_density_half_hx = 1.0 - (1.0 - total_density_half_hx) * (
                            1.0 - density_hx
                        )
                if valid_half_h_y:
                    r_half_h_dy = r_dy + grid_spacing_half
                    dist_sq_hy = r_dx**2 + r_half_h_dy**2 + r_dz**2
                    if dist_sq_hy <= atom_influence_radius_square:
                        g_arg_hy = (dist_sq_hy * mult_inv) ** gaussian_exponent
                        density_hy = math.exp(-g_arg_hy)
                        total_density_half_hy = 1.0 - (1.0 - total_density_half_hy) * (
                            1.0 - density_hy
                        )
                if valid_half_h_z:
                    r_half_h_dz = r_dz + grid_spacing_half
                    dist_sq_hz = r_dx**2 + r_dy**2 + r_half_h_dz**2
                    if dist_sq_hz <= atom_influence_radius_square:
                        g_arg_hz = (dist_sq_hz * mult_inv) ** gaussian_exponent
                        density_hz = math.exp(-g_arg_hz)
                        total_density_half_hz = 1.0 - (1.0 - total_density_half_hz) * (
                            1.0 - density_hz
                        )
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
                ion_exclusion_map_1d[ijk1d] = ion_excluded_flag


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
    voxel_counts = (
        (voxel_map_shape[0] + 1) * (voxel_map_shape[1] + 1) * (voxel_map_shape[2] + 1)
    )
    # Note: Assumed any atom can be counted at max 27 times, when it influences all the neighboring voxel regions.
    neighbor_voxels_size = voxel_counts + 125 * num_atoms
    neighbor_voxel_index_shape = (
        voxel_map_shape[0] + 1,
        voxel_map_shape[1] + 1,
        voxel_map_shape[2] + 1,
    )

    seen_atoms = np.full(num_atoms + 1, fill_value=False, dtype=delphi_bool)
    neighbor_voxel_atom_ids_flat = np.full(
        neighbor_voxels_size, fill_value=-1, dtype=delphi_int
    )
    neighbor_voxel_start_index = np.full(
        neighbor_voxel_index_shape, fill_value=-1, dtype=delphi_int
    )
    neighbor_voxel_end_index = np.full(
        neighbor_voxel_index_shape, fill_value=-1, dtype=delphi_int
    )

    # Step 1: build neighbor voxel map
    output_index = _build_neighbor_voxel_atom_index_map(
        voxel_atom_ids,
        voxel_atom_start_index,
        voxel_atom_end_index,
        voxel_map_shape,
        neighbor_voxel_atom_ids_flat,
        neighbor_voxel_start_index,
        neighbor_voxel_end_index,
        seen_atoms,
        NEIGHBOR_VOXEL_REL_COORDS,
    )

    # Step 2: calculate Gaussian density
    if platform.active == "cpu":
        vprint(DEBUG, _VERBOSITY, "Calling _cpu_calc_gaussian_density_map")
        _cpu_calc_gaussian_density_map(
            generate_ion_exclusion_map=generate_ion_exclusion_map,
            scale=scale,
            gaussian_exponent=gaussian_exponent,
            surface_offset=surface_offset,
            atom_influence_radius=atom_influence_radius,
            salt_radius=salt_radius,
            grid_shape=grid_shape,
            grid_origin=grid_origin,
            atoms_data=atoms_data,
            gauss_density_map_1d=gauss_density_map_1d,
            gauss_density_map_midpoints_1d=gauss_density_map_midpoints_1d,
            ion_exclusion_map_1d=ion_exclusion_map_1d,
        )
    elif platform.active == "cuda" and hasattr(cuda, "jit"):
        vprint(DEBUG, _VERBOSITY, "Calling _cuda_calc_gaussian_density_map")
        # Transfer data to GPU
        grid_shape_dev = cuda.to_device(grid_shape.astype(delphi_int))
        grid_origin_dev = cuda.to_device(grid_origin.astype(delphi_real))
        atoms_data_dev = cuda.to_device(atoms_data.astype(delphi_real))
        gauss_density_map_1d_dev = cuda.to_device(
            gauss_density_map_1d.astype(delphi_real)
        )
        gauss_density_map_midpoints_1d_dev = cuda.to_device(
            gauss_density_map_midpoints_1d.astype(delphi_real)
        )
        ion_exclusion_map_1d_dev = cuda.to_device(
            ion_exclusion_map_1d.astype(delphi_bool)
        )

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
            delphi_real(surface_offset),
            delphi_real(atom_influence_radius),
            delphi_real(salt_radius),
            grid_shape_dev,
            grid_origin_dev,
            atoms_data_dev,
            gauss_density_map_1d_dev,
            gauss_density_map_midpoints_1d_dev,
            ion_exclusion_map_1d_dev,
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
    gap_indi = gapdi - indi
    for ijk1d in prange(gauss_epsmap_1d.shape[0]):
        gauss_epsmap_1d[ijk1d] = indi + gap_indi * (1.0 - gaussian_density_1d[ijk1d])


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def calc_spatial_epsilon_map_midpoints(
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
    num_grid_points = gauss_density_map_1d.shape[0]
    num_mid_points = gauss_density_map_midpoints_1d.shape[0]
    epsout = delphi_real(1.0) if vaccum else exdi

    spatial_outeps_limit = np.empty(2, dtype=delphi_real)
    spatial_outeps_limit[0] = exdi
    spatial_outeps_limit[1] = gapdi

    epsilon_r_1d = np.full(num_grid_points, fill_value=epsout, dtype=delphi_real)
    epsilon_r_midpoints_1d = np.empty(num_mid_points, dtype=delphi_real)
    epsilon_r_midpoints_1d.fill(epsout)

    z_stride: delphi_int = 1
    y_stride: delphi_int = grid_shape[2]
    x_stride: delphi_int = y_stride * grid_shape[1]

    last_grid_x = grid_shape[0] - 1
    last_grid_y = grid_shape[1] - 1
    last_grid_z = grid_shape[2] - 1

    # Loop 1: Assign gridpoint epsilon values
    for ijk1d in prange(num_grid_points):
        rho = gauss_density_map_1d[ijk1d]
        regional_epsilon_limit = spatial_outeps_limit[
            delphi_int(ion_exclusion_map_1d[ijk1d])
        ]
        if rho < solute_density_threshold:
            epsilon_r_1d[ijk1d] = epsout
        else:
            epsilon_r_1d[ijk1d] = rho * indi + (1.0 - rho) * regional_epsilon_limit

    # Loop 2: Assign midpoint epsilon values
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

    # Loop 3: Apply density filter
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
) -> None:
    gaussian_exponent_minus_1 = gaussian_exponent - 1
    gaussian_exponent_x_2 = gaussian_exponent * 2
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    num_grid_points = grid_shape[0] * x_stride

    num_atoms = atoms_data.shape[0]

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

        for atom_idx in range(num_atoms):
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
                        if abs(1.0 - density) > approx_zero and density > approx_zero:
                            # Calculate components of the gradient contribution
                            density_factor_numerator = density * (1.0 - total_density)
                            density_factor_denominator = 1.0 - density

                            # Check if density_factor_denominator is not zero
                            if abs(density_factor_denominator) > approx_zero:
                                density_factor = (
                                    density_factor_numerator
                                    / density_factor_denominator
                                )

                                dist_factor = (
                                    dist_square**gaussian_exponent_minus_1
                                    if gaussian_exponent_minus_1 >= 0
                                    else 1.0
                                    / (dist_square ** abs(gaussian_exponent_minus_1))
                                )  # Handle negative exponents if needed, though exp is usually > 0
                                atom_factor = gaussian_exponent_x_2 / (
                                    atom_sigma_x_radius_square**gaussian_exponent
                                )

                                srf_factor_base = (
                                    atom_factor * density_factor * dist_factor
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
                                grad_surface_map_1d[ijk1d_x_3] -= srf_factor * delta_rx
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
) -> None:
    gaussian_exponent_minus_1 = gaussian_exponent - 1
    gaussian_exponent_x_2 = gaussian_exponent * 2
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
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

        for atom_idx in range(atoms_data.shape[0]):
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
                    dist_square > approx_zero and atom_sigma_x_radius_sq > approx_zero
                ):  # Avoid division by zero and point at atom center
                    dist_sq_norm = dist_square / atom_sigma_x_radius_sq
                    # Use CUDA math functions
                    g_arg = cuda.libdevice.powf(dist_sq_norm, gaussian_exponent)

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
                            density_factor_numerator = density * (1.0 - total_density)
                            density_factor_denominator = 1.0 - density

                            # Check if density_factor_denominator is not zero
                            if (
                                cuda.libdevice.fabs(density_factor_denominator)
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
                                    atom_factor * density_factor * dist_factor
                                )

                                srf_factor: delphi_real = 0.0
                                if (
                                    total_density > approx_zero
                                    and srf_square > approx_zero
                                    and cuda.libdevice.fabs(srf_func_dr) > approx_zero
                                ):
                                    srf_factor = (
                                        srf_factor_base
                                        * srf_func_dr
                                        * (
                                            1.0
                                            / cuda.libdevice.powf(total_density, 2.0)
                                        )
                                        * srf_square
                                    )  # Combine factors

                                # Accumulate gradient contribution from this atom
                                # Note the negative sign in the original formula
                                grad_x -= srf_factor * delta_rx
                                grad_y -= srf_factor * delta_ry
                                grad_z -= srf_factor * delta_rz
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
        )
        grad_surface_map_1d_device.copy_to_host(grad_surface_map_1d)

        # Explicitly free device memory (optional, but good practice)
        del grid_shape_device
        del grid_origin_device
        del atoms_data_device
        del densitymap_gridpoints_1d_device
        del surface_map_1d_device
        del grad_surface_map_1d_device


def calc_gaussian_like_surface(
    platform,
    num_cuda_threads,
    surf_den_exp_scaled,
    approx_zero,
    gauss_density_solvent_1d,
    surface_map_1d,
):
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
