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

from math import floor, sqrt, sin, log, exp
from numba import njit, prange, cuda
import numpy as np

from pydelphi.foundation.enums import Precision

from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_RADIUS,
    ATOMFIELD_GAUSS_SIGMA,
    ConstDelPhiFloats as ConstDelPhi,
    ConstPhysical,
)

APPROX_ZERO = ConstDelPhi.ApproxZero.value
GAUSSIAN_INFLUENCE_RADIUS_FACTOR = ConstDelPhi.GaussianInfluenceRadiusFactor.value
TWO_PI = 2.0 * ConstPhysical.Pi.value

from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
)

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

# --- IMPORT HELPER FUNCTIONS FOR VOXEL MAPPING ---
from pydelphi.space.core.voxelizer import (
    build_neighbor_voxel_unique_atom_index_map,
)


@njit(nogil=True, boundscheck=False, parallel=True)
def _cpu_mark_solute_and_heaviside_solute_surface_maps(
    grid_spacing: delphi_real,
    probe_radius: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    solute_inside_map_1d: np.ndarray[delphi_bool],
    solute_outside_map_1d: np.ndarray[delphi_bool],
    surface_heaviside_map_1d: np.ndarray[delphi_real],
    neighbor_voxel_atom_ids_flat: np.ndarray[delphi_int],
    neighbor_voxel_atom_start_index: np.ndarray[delphi_int],
    neighbor_voxel_atom_end_index: np.ndarray[delphi_int],
    voxel_map_origin: np.ndarray[delphi_real],
    voxel_map_shape: np.ndarray[delphi_int],
    voxel_map_scale: delphi_real,
) -> None:
    """
    CPU kernel to classify grid points with respect to solute and solvent-accessible surface.

    This function computes three maps in a single pass over the grid:
      1. solute_inside_map_1d     — True if the grid point lies strictly inside the solute volume.
      2. solute_outside_map_1d    — True if the grid point lies outside the solvent-accessible boundary.
      3. surface_heaviside_map_1d — Heaviside (0/1) step map for the solvent-accessible solute surface.

    Neighbor atoms are queried efficiently using a precomputed voxel-based spatial map.

    Notes:
        - This function is intended for CPU execution. A matching `_cuda_` version may be defined for GPU platforms.
        - Uses parallel execution across the grid using Numba `@njit(parallel=True)`.

    Parameters:
        grid_spacing: Uniform spacing between grid points (in angstrom).
        probe_radius: Probe radius used to define the solvent-accessible surface.
        grid_shape: Shape of the 3D grid (x, y, z).
        grid_origin: Real-space origin of the grid.
        atoms_data: Atom data array with expected layout (X, Y, Z, Radius, GaussSigma, ...).
        solute_inside_map_1d: Output boolean array, marks grid points inside solute.
        solute_outside_map_1d: Output boolean array, marks grid points outside extended surface.
        surface_heaviside_map_1d: Output float array, 1.0 where inside extended surface, 0.0 elsewhere.
        neighbor_voxel_atom_ids_flat: Flattened array of neighbor atom indices for all voxels.
        neighbor_voxel_atom_start_index: Starting index in flat list per voxel.
        neighbor_voxel_atom_end_index: Ending index in flat list per voxel.
        voxel_map_origin: Real-space origin of the voxel map.
        voxel_map_shape: Shape of the voxel grid (x, y, z).
        voxel_map_scale: Scale factor from real-space to voxel index space.
    """
    x_dim, y_dim, z_dim = grid_shape
    y_stride = z_dim
    x_stride = y_dim * z_dim

    v_origin = voxel_map_origin
    v_shape = voxel_map_shape
    v_scale = voxel_map_scale

    solute_outside_map_1d.fill(True)
    solute_inside_map_1d.fill(False)
    surface_heaviside_map_1d.fill(0.0)

    gcs_probe = (
        2.0 * probe_radius
    )  # Gaussian Convolution Surface traces enveloping solvent accessible surface

    for ix in prange(x_dim):
        for iy in range(y_dim):
            for iz in range(z_dim):
                flat_index = ix * x_stride + iy * y_stride + iz

                grid_x = grid_origin[0] + ix * grid_spacing
                grid_y = grid_origin[1] + iy * grid_spacing
                grid_z = grid_origin[2] + iz * grid_spacing

                is_inside_gcs_extended_region = (
                    False  # GCS: note surface traced by farther side of probe.
                )
                is_inside_hsf_extended_region = (
                    False  # Heavy side function path traced by center of probe.
                )
                is_strictly_inside_solute = False

                central_vx = max(
                    0, min(delphi_int((grid_x - v_origin[0]) * v_scale), v_shape[0])
                )
                central_vy = max(
                    0, min(delphi_int((grid_y - v_origin[1]) * v_scale), v_shape[1])
                )
                central_vz = max(
                    0, min(delphi_int((grid_z - v_origin[2]) * v_scale), v_shape[2])
                )

                if (
                    0 <= central_vx <= v_shape[0]
                    and 0 <= central_vy <= v_shape[1]
                    and 0 <= central_vz <= v_shape[2]
                ):
                    start_atom_list_idx = neighbor_voxel_atom_start_index[
                        central_vx, central_vy, central_vz
                    ]
                    end_atom_list_idx = neighbor_voxel_atom_end_index[
                        central_vx, central_vy, central_vz
                    ]

                    if start_atom_list_idx <= end_atom_list_idx:
                        for atom_list_ptr in range(
                            start_atom_list_idx, end_atom_list_idx + 1
                        ):
                            atom_idx = neighbor_voxel_atom_ids_flat[atom_list_ptr] - 1
                            if atom_idx < 0:
                                continue

                            atom = atoms_data[atom_idx]
                            atom_x = atom[ATOMFIELD_X]
                            atom_y = atom[ATOMFIELD_Y]
                            atom_z = atom[ATOMFIELD_Z]
                            atom_radius = atom[ATOMFIELD_RADIUS]
                            atom_sigma = atom[ATOMFIELD_GAUSS_SIGMA]

                            dx = grid_x - atom_x
                            dy = grid_y - atom_y
                            dz = grid_z - atom_z
                            dist_sq = dx * dx + dy * dy + dz * dz

                            radius_core = atom_radius * atom_sigma
                            radius_hsf = radius_core + probe_radius
                            radius_gcs = radius_core + gcs_probe

                            radius_gcs_sq = radius_gcs**2
                            radius_hsf_sq = radius_hsf**2
                            radius_core_sq = radius_core**2

                            if dist_sq <= radius_gcs_sq:
                                is_inside_gcs_extended_region = True

                            if dist_sq <= radius_hsf_sq:
                                is_inside_hsf_extended_region = True

                            if dist_sq <= radius_core_sq:
                                is_strictly_inside_solute = True

                            if (
                                is_inside_gcs_extended_region
                                and is_strictly_inside_solute
                            ):
                                break  # Ensures both inside/outside-most are set, safe to exit now

                solute_inside_map_1d[flat_index] = is_strictly_inside_solute
                solute_outside_map_1d[flat_index] = not is_inside_gcs_extended_region
                surface_heaviside_map_1d[flat_index] = (
                    1.0 if is_inside_hsf_extended_region else 0.0
                )


@cuda.jit
def _cuda_mark_solute_and_heaviside_solute_surface_maps(
    grid_spacing: delphi_real,
    probe_radius: delphi_real,
    grid_shape: np.ndarray,  # (x_dim, y_dim, z_dim)
    grid_origin: np.ndarray,  # (3,)
    atoms_data: np.ndarray,  # shape: (n_atoms, fields)
    solute_inside_map_1d: np.ndarray,  # output
    solute_outside_map_1d: np.ndarray,  # output
    surface_heaviside_map_1d: np.ndarray,  # output
    neighbor_voxel_atom_ids_flat: np.ndarray,
    neighbor_voxel_atom_start_index: np.ndarray,
    neighbor_voxel_atom_end_index: np.ndarray,
    voxel_map_origin: np.ndarray,
    voxel_map_shape: np.ndarray,
    voxel_map_scale: delphi_real,
):
    tid = cuda.grid(1)

    x_dim = grid_shape[0]
    y_dim = grid_shape[1]
    z_dim = grid_shape[2]
    num_grid_pts = x_dim * y_dim * z_dim
    if tid >= num_grid_pts:
        return

    # Compute 3D grid index from flat tid
    x_stride = y_dim * z_dim
    y_stride = z_dim
    ix = tid // x_stride
    iy = (tid % x_stride) // y_stride
    iz = tid % z_dim

    grid_x = grid_origin[0] + ix * grid_spacing
    grid_y = grid_origin[1] + iy * grid_spacing
    grid_z = grid_origin[2] + iz * grid_spacing

    is_inside_gcs_extended_region = False
    is_inside_hsf_extended_region = False
    is_strictly_inside_solute = False

    v_origin_x = voxel_map_origin[0]
    v_origin_y = voxel_map_origin[1]
    v_origin_z = voxel_map_origin[2]

    v_shape_x = voxel_map_shape[0]
    v_shape_y = voxel_map_shape[1]
    v_shape_z = voxel_map_shape[2]

    central_vx = min(max(0, int((grid_x - v_origin_x) * voxel_map_scale)), v_shape_x)
    central_vy = min(max(0, int((grid_y - v_origin_y) * voxel_map_scale)), v_shape_y)
    central_vz = min(max(0, int((grid_z - v_origin_z) * voxel_map_scale)), v_shape_z)

    if (
        0 <= central_vx <= v_shape_x
        and 0 <= central_vy <= v_shape_y
        and 0 <= central_vz <= v_shape_z
    ):
        start_idx = neighbor_voxel_atom_start_index[central_vx, central_vy, central_vz]
        end_idx = neighbor_voxel_atom_end_index[central_vx, central_vy, central_vz]

        if start_idx <= end_idx:
            for atom_ptr in range(start_idx, end_idx + 1):
                atom_idx = neighbor_voxel_atom_ids_flat[atom_ptr] - 1
                if atom_idx < 0:
                    continue

                atom = atoms_data[atom_idx]
                atom_x = atom[ATOMFIELD_X]
                atom_y = atom[ATOMFIELD_Y]
                atom_z = atom[ATOMFIELD_Z]
                atom_radius = atom[ATOMFIELD_RADIUS]
                atom_sigma = atom[ATOMFIELD_GAUSS_SIGMA]

                dx = grid_x - atom_x
                dy = grid_y - atom_y
                dz = grid_z - atom_z
                dist_sq = dx * dx + dy * dy + dz * dz

                gcs_probe = 2.0 * probe_radius

                r_core = atom_radius * atom_sigma
                r_hsf = r_core + probe_radius
                r_gcs = r_core + gcs_probe

                if dist_sq <= r_core * r_core:
                    is_strictly_inside_solute = True
                if dist_sq <= r_hsf * r_hsf:
                    is_inside_hsf_extended_region = True
                if dist_sq <= r_gcs * r_gcs:
                    is_inside_gcs_extended_region = True

                if is_inside_gcs_extended_region and is_strictly_inside_solute:
                    break

    solute_inside_map_1d[tid] = is_strictly_inside_solute
    solute_outside_map_1d[tid] = not is_inside_gcs_extended_region
    surface_heaviside_map_1d[tid] = 1.0 if is_inside_hsf_extended_region else 0.0


@njit(nogil=True, boundscheck=False)
def _perform_fft(
    complex_data_array: np.ndarray[delphi_real],
    num_complex_points: delphi_int,
    fft_sign: delphi_int,
) -> None:
    """
    Performs a Fast Fourier Transform (FFT) on a 1D real array representing complex numbers
    using the Cooley-Tukey algorithm. This function modifies the input `complex_data_array` in-place.

    The `complex_data_array` is expected to be a 1D array of 2 * num_complex_points,
    representing num_complex_points complex numbers, where complex_data_array[2*k] is the real part
    and complex_data_array[2*k+1] is the imaginary part for the k-th complex number.

    Parameters
    ----------
    complex_data_array : np.ndarray[delphi_real]
        The 1D array containing interleaved real and imaginary parts of complex numbers.
        Modified in-place.
    num_complex_points : delphi_int
        The number of complex data points (complex_data_array length is 2 * num_complex_points).
    fft_sign : delphi_int
        The sign of the exponent in the FFT, typically 1 for forward FFT
        (e.g., exp(-i*theta)) and -1 for inverse FFT (e.g., exp(i*theta)).
    """
    total_data_length = 2 * num_complex_points
    reverse_index = 0

    # Bit-reversal permutation: Rearrange data array elements for in-place FFT
    # This step ensures that when we combine sub-FFTs, the data is in the correct order.
    for current_index in range(0, total_data_length, 2):
        if reverse_index > current_index:
            # Swap real parts
            complex_data_array[reverse_index], complex_data_array[current_index] = (
                complex_data_array[current_index],
                complex_data_array[reverse_index],
            )
            # Swap imaginary parts
            (
                complex_data_array[reverse_index + 1],
                complex_data_array[current_index + 1],
            ) = (
                complex_data_array[current_index + 1],
                complex_data_array[reverse_index + 1],
            )

        power_of_two_half = total_data_length // 2  # Represents 'm' in some texts
        while (power_of_two_half >= 1) and (reverse_index >= power_of_two_half):
            reverse_index -= power_of_two_half
            power_of_two_half //= 2
        reverse_index += power_of_two_half

    # Danielson-Lanczos section: Iterative butterfly computations
    # This loop proceeds through stages of the FFT, combining smaller FFTs into larger ones.
    # 'current_block_size' doubles in each iteration, representing the size of the FFT blocks.
    current_block_size = 2  # Initial block size (2 points, i.e., 1 complex pair)
    while total_data_length > current_block_size:
        # 'step_between_blocks' is the stride to get to the next block in the current stage
        step_between_blocks = 2 * current_block_size
        # 'angle_increment' is the fundamental angle for the twiddle factors (W_N^k)
        angle_increment = TWO_PI / (fft_sign * current_block_size)

        # Pre-calculate sine and cosine components for the angle increment
        # These are used to update the complex rotation factor (twiddle factor).
        cos_angle_increment = (
            -2.0 * sin(0.5 * angle_increment) ** 2
        )  # cos(x) - 1 for half angle
        sin_angle_increment = sin(angle_increment)

        # 'twiddle_real' and 'twiddle_imag' are the real and imaginary parts of the
        # complex rotation factor (W_N^k). They are updated multiplicatively.
        twiddle_real = 1.0
        twiddle_imag = 0.0

        # Loop through each sub-block within the current stage
        for block_offset in range(0, current_block_size, 2):
            # Loop through the data, applying butterflies for the current twiddle factor
            for data_start_index in range(
                block_offset, total_data_length, step_between_blocks
            ):
                # 'conjugate_pair_index' is the index of the second element in the butterfly pair
                conjugate_pair_index = data_start_index + current_block_size

                # Calculate the product of the twiddle factor and the second complex number
                # temp_real = (twiddle_real * data_real_part) - (twiddle_imag * data_imag_part)
                # temp_imag = (twiddle_real * data_imag_part) + (twiddle_imag * data_real_part)
                temp_real_product = (
                    twiddle_real * complex_data_array[conjugate_pair_index]
                    - twiddle_imag * complex_data_array[conjugate_pair_index + 1]
                )
                temp_imag_product = (
                    twiddle_real * complex_data_array[conjugate_pair_index + 1]
                    + twiddle_imag * complex_data_array[conjugate_pair_index]
                )

                # Perform the butterfly operation
                # A' = A + (W*B)
                # B' = A - (W*B)
                complex_data_array[conjugate_pair_index] = (
                    complex_data_array[data_start_index] - temp_real_product
                )
                complex_data_array[conjugate_pair_index + 1] = (
                    complex_data_array[data_start_index + 1] - temp_imag_product
                )
                complex_data_array[data_start_index] = (
                    complex_data_array[data_start_index] + temp_real_product
                )
                complex_data_array[data_start_index + 1] = (
                    complex_data_array[data_start_index + 1] + temp_imag_product
                )

            # Update the twiddle factor for the next sub-block using a rotation formula
            # w_{k+1} = w_k * w_inc
            # Here, an optimized form (from numerical recipes) is used to avoid repeated sin/cos calls
            # wr_new = wr * wpr - wi * wpi + wr
            # wi_new = wi * wpr + wr_old * wpi + wi
            old_twiddle_real = twiddle_real
            twiddle_real = (
                twiddle_real * cos_angle_increment
                - twiddle_imag * sin_angle_increment
                + twiddle_real
            )
            twiddle_imag = (
                twiddle_imag * cos_angle_increment
                + old_twiddle_real * sin_angle_increment
                + twiddle_imag
            )

        current_block_size = (
            step_between_blocks  # Move to the next stage (double the block size)
        )


@njit(nogil=True, boundscheck=False)
def _cpu_calculate_gaussian_kernel(
    probe_radius: delphi_real,
    grid_spacing: delphi_real,
    gaussian_sigma: delphi_real,
    gcs_tolerance: delphi_real,
) -> np.ndarray:
    """
    Computes a 1D normalized Gaussian kernel using multiplicative inverses
    for efficiency and clarity.

    Parameters:
        probe_radius: Truncation radius for the kernel.
        grid_spacing: Spacing between grid points.
        gaussian_sigma: Standard deviation of the Gaussian.
        gcs_tolerance: Truncation tolerance threshold.

    Returns:
        A 1D numpy array of normalized Gaussian weights.
    """
    inv_sigma = 1.0 / gaussian_sigma
    inv_sigma_squared = inv_sigma * inv_sigma
    coeff = sqrt(TWO_PI)
    inv_norm_factor = 1.0 / (coeff * gaussian_sigma)

    num_points = floor(probe_radius / grid_spacing) * 2 + 1

    # Precompute width = sqrt(-2 * ln(tol * σ * sqrt(2π))) * σ
    logarg = gcs_tolerance * gaussian_sigma * coeff
    log_term = log(logarg)
    width = sqrt(-2.0 * log_term) * gaussian_sigma
    inv_num_minus1 = 1.0 / (num_points - 1)
    step = 2.0 * width * inv_num_minus1

    gaussian_function = np.empty(num_points, dtype=delphi_real)
    kernel_sum = 0.0

    for i in range(num_points):
        coord = -width + i * step
        exponent = -(coord * coord) * 0.5 * inv_sigma_squared
        value = exp(exponent) * inv_norm_factor
        gaussian_function[i] = value
        kernel_sum += value

    inv_sum = 1.0 / kernel_sum
    for i in range(num_points):
        gaussian_function[i] *= inv_sum

    return gaussian_function


@njit(nogil=True, boundscheck=False)
def _get_padded_dimension_size(
    original_size: delphi_int, padding: delphi_int
) -> delphi_int:
    # No changes
    padded_size = original_size + padding
    for i in range(1, 11):
        if padded_size < 2**i:
            return 2**i
    return padded_size


@njit(nogil=True, boundscheck=False, parallel=True)
def _convolve_axis(
    surface_map_1d: np.ndarray[delphi_real],
    grid_shape: np.ndarray[delphi_int],
    kernel_function: np.ndarray[delphi_real],
    axis: int,
    padded_dimension_size: delphi_int,
    points_in_kernel: delphi_int,
) -> None:
    # No changes
    x_stride = grid_shape[1] * grid_shape[2]
    y_stride = grid_shape[2]

    fft_kernel = np.zeros(2 * padded_dimension_size, dtype=delphi_real)
    for i in range(points_in_kernel):
        fft_kernel[2 * i] = kernel_function[i]

    _perform_fft(fft_kernel, padded_dimension_size, 1)

    iter_dim1_idx = (axis + 1) % 3
    iter_dim2_idx = (axis + 2) % 3

    iter_dim1_size = grid_shape[iter_dim1_idx]
    iter_dim2_size = grid_shape[iter_dim2_idx]

    for combined_idx in prange(iter_dim1_size * iter_dim2_size):
        dim1_val = combined_idx // iter_dim2_size
        dim2_val = combined_idx % iter_dim2_size

        data_line = np.zeros(2 * padded_dimension_size, dtype=delphi_real)
        convolution_line = np.zeros(2 * padded_dimension_size, dtype=delphi_real)

        for i in range(grid_shape[axis]):
            if axis == 0:
                flat_index = i * x_stride + dim1_val * y_stride + dim2_val
            elif axis == 1:
                flat_index = dim1_val * x_stride + i * y_stride + dim2_val
            else:
                flat_index = dim1_val * x_stride + dim2_val * y_stride + i
            data_line[2 * i] = surface_map_1d[flat_index]

        _perform_fft(data_line, padded_dimension_size, 1)

        for i in range(padded_dimension_size):
            convolution_line[2 * i] = (
                data_line[2 * i] * fft_kernel[2 * i]
                - data_line[2 * i + 1] * fft_kernel[2 * i + 1]
            )
            convolution_line[2 * i + 1] = (
                data_line[2 * i] * fft_kernel[2 * i + 1]
                + data_line[2 * i + 1] * fft_kernel[2 * i]
            )

        _perform_fft(convolution_line, padded_dimension_size, -1)

        offset = (points_in_kernel - 1) // 2
        for i in range(grid_shape[axis]):
            if axis == 0:
                flat_index = i * x_stride + dim1_val * y_stride + dim2_val
            elif axis == 1:
                flat_index = dim1_val * x_stride + i * y_stride + dim2_val
            else:
                flat_index = dim1_val * x_stride + dim2_val * y_stride + i
            surface_map_1d[flat_index] = convolution_line[2 * (i + offset)]


def generate_gaussian_convolution_surface(
    grid_spacing: delphi_real,
    probe_radius: delphi_real,
    gaussian_sigma: delphi_real,
    gcs_tolerance: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    grid_origin: np.ndarray[delphi_real],
    atoms_data: np.ndarray[delphi_real],
    solute_inside_map_1d: np.ndarray[delphi_bool],
    solute_outside_map_1d: np.ndarray[delphi_bool],
    surface_map_1d: np.ndarray[delphi_real],
    # --- Parameters for Voxel Mapping ---
    voxel_atom_ids: np.ndarray[delphi_int],  # Output from atom-to-voxel assignment
    voxel_atom_start_index: np.ndarray[
        delphi_int
    ],  # Output from atom-to-voxel assignment
    voxel_atom_end_index: np.ndarray[
        delphi_int
    ],  # Output from atom-to-voxel assignment
    voxel_map_origin: np.ndarray[delphi_real],  # Origin of the voxel grid
    voxel_map_shape: np.ndarray[delphi_int],  # Shape of the voxel grid
    voxel_map_scale: delphi_real,  # Scale (1/voxel_spacing) of the voxel grid
) -> None:
    """
    Generates a Gaussian Convolution Surface (GCS) map for a given solute
    within a grid, leveraging a pre-built voxel map for efficient atom lookup.
    """
    num_grid_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
    num_atoms = atoms_data.shape[0]

    # --- Step 0: Pre-compute neighbor voxel atom map ---
    (
        neighbor_voxel_atom_ids_flat,
        neighbor_voxel_start_index,
        neighbor_voxel_end_index,
        actual_neighbor_ids_count,
    ) = build_neighbor_voxel_unique_atom_index_map(
        num_atoms,
        voxel_atom_ids,  # This comes from the atom-to-voxel assignment in a higher level
        voxel_atom_start_index,  # This comes from the atom-to-voxel assignment
        voxel_atom_end_index,  # This comes from the atom-to-voxel assignment
        voxel_map_shape,
    )
    debug_module = False
    # Step 1: Mark grid points which are inside and outside the solute regions and initial Heaviside solute surface map.
    _cpu_mark_solute_and_heaviside_solute_surface_maps(
        grid_spacing,
        probe_radius,
        grid_shape,
        grid_origin,
        atoms_data,
        solute_inside_map_1d,
        solute_outside_map_1d,
        surface_map_1d,
        # Pass the pre-computed voxel neighbor map info
        neighbor_voxel_atom_ids_flat,
        neighbor_voxel_start_index,
        neighbor_voxel_end_index,
        voxel_map_origin,
        voxel_map_shape,
        voxel_map_scale,
    )
    if debug_module:
        np.save("new_solute_inside_map_1d.npy", solute_inside_map_1d)
        np.save("new_solute_outside_map_1d.npy", solute_outside_map_1d)
        np.save("new_heavyside_surface_map_1d.npy", surface_map_1d)

    # Step 2: Calculate the 1D Gaussian kernel for convolution.
    gaussian_kernel = _cpu_calculate_gaussian_kernel(
        probe_radius, grid_spacing, gaussian_sigma, gcs_tolerance
    )
    num_kernel_points = gaussian_kernel.shape[0]
    if debug_module:
        print("gaussian_kernel=", gaussian_kernel)

    # Step 3: Determine padded dimensions for efficient FFT.
    padding = num_kernel_points
    padded_x_size = _get_padded_dimension_size(grid_shape[0], padding)
    padded_y_size = _get_padded_dimension_size(grid_shape[1], padding)
    padded_z_size = _get_padded_dimension_size(grid_shape[2], padding)

    # Step 4: Perform convolution along each axis using FFT.
    _convolve_axis(
        surface_map_1d,
        grid_shape,
        gaussian_kernel,
        axis=0,
        padded_dimension_size=padded_x_size,
        points_in_kernel=num_kernel_points,
    )
    if debug_module:
        np.save("new_convolve_x_surface_map_1d.npy", surface_map_1d)

    _convolve_axis(
        surface_map_1d,
        grid_shape,
        gaussian_kernel,
        axis=1,
        padded_dimension_size=padded_y_size,
        points_in_kernel=num_kernel_points,
    )
    if debug_module:
        np.save("new_convolve_y_surface_map_1d.npy", surface_map_1d)

    _convolve_axis(
        surface_map_1d,
        grid_shape,
        gaussian_kernel,
        axis=2,
        padded_dimension_size=padded_z_size,
        points_in_kernel=num_kernel_points,
    )
    if debug_module:
        np.save("new_convolve_z_surface_map_1d.npy", surface_map_1d)

    # Step 5: Normalize the surface map and apply boundary conditions.
    surface_max_value = surface_map_1d.max()
    if surface_max_value > APPROX_ZERO:  # Use APPROX_ZERO for float comparisons
        for ijk1d in prange(num_grid_points):
            surface_map_1d[ijk1d] /= surface_max_value

            if solute_outside_map_1d[ijk1d]:
                surface_map_1d[ijk1d] = 0.0
            elif solute_inside_map_1d[ijk1d]:
                surface_map_1d[ijk1d] = 1.0
    if debug_module:
        np.save("new_normalized_surface_map_1d.npy", surface_map_1d)
