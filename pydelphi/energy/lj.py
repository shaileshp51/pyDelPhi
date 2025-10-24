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
This module provides functions for calculating the Lennard-Jones energy between atoms,
supporting both CPU (Numba-parallelized) and CUDA (GPU) backends.

It defines:
- `_geometric_mean`: Helper function for geometric mean.
- `_arithmetic_mean`: Helper function for arithmetic mean.
- `_cpu_lj_energy_kernel`: A Numba-jitted function for calculating
  pairwise Lennard-Jones energy on the CPU.
- `_cuda_compute_pair_index`: A Numba CUDA device function to map a flat index
  to a unique pair of atom indices (copied for self-containment).
- `_cuda_lj_energy_kernel`: A Numba CUDA kernel for GPU-accelerated
  pairwise Lennard-Jones energy calculation with shared memory reduction.
- `_cuda_calc_lj_energy`: A host function to manage CUDA operations for LJ,
  including data transfer, kernel launch configuration, and result retrieval.
- `calc_lj_energy`: The public interface to calculate the total scaled
  Lennard-Jones energy, abstracting away the backend selection and providing
  error handling and final scaling based on temperature.
"""

import math
import numpy as np
from numba import njit, prange, get_num_threads, set_num_threads, cuda, float64


from pydelphi.config.global_runtime import vprint

from pydelphi.config.logging_config import DEBUG, get_effective_verbosity

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

from pydelphi.foundation.platforms import Platform
from pydelphi.config.global_runtime import delphi_real
from pydelphi.constants import ConstDelPhiFloats

from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_LJ_SIGMA,
    ATOMFIELD_LJ_EPSILON,
    LJCUTOFF_MIN,
    LJCUTOFF_MAX,
    MAX_KERNEL_SHARED_MEM_THREADS,
)

APPROX_ZERO = ConstDelPhiFloats.ApproxZero.value


# --- Helper functions for LJ energy calculation (Numba-compatible) ---
@njit(nogil=True, boundscheck=False, cache=True, inline=True)
def _arithmetic_mean(a: delphi_real, b: delphi_real) -> delphi_real:
    """Calculates the arithmetic mean of two numbers."""
    return (a + b) / 2.0


@njit(nogil=True, boundscheck=False, cache=True, inline=True)
def _geometric_mean(a: delphi_real, b: delphi_real) -> delphi_real:
    """
    Calculates the geometric mean of two numbers, handling non-positive inputs
    to prevent math domain errors for sqrt. Epsilon values are typically positive.
    """
    # Clamp to zero if negative to avoid sqrt(negative)
    a = max(0.0, a)
    b = max(0.0, b)
    return math.sqrt(a * b)


# --- CUDA Device Helper (reused from original template for self-containment) ---
@cuda.jit(device=True, inline=True)
def _cuda_compute_pair_index(flat_idx: int, n_atoms: int) -> tuple[int, int]:
    """
    Maps a flat pair index (for unique pairs i < j) to 2D indices (i, j).
    This is a standard formula for enumerating unique pairs.
    """
    # Using float constants for intermediate calculation robustness to avoid integer overflow issues
    # and maintain precision before final integer cast.
    term_under_sqrt = -8.0 * flat_idx + 4.0 * n_atoms * (n_atoms - 1.0) - 7.0
    term_under_sqrt = max(0.0, term_under_sqrt)  # Ensure non-negative for sqrt
    i_float = n_atoms - 2.0 - math.floor(math.sqrt(term_under_sqrt) / 2.0 - 0.5)
    i = int(i_float)

    # Using integer arithmetic where possible for j calculation for precision
    j = int(
        flat_idx
        + i
        + 1
        - (n_atoms * (n_atoms - 1)) // 2
        + ((n_atoms - i) * (n_atoms - i - 1)) // 2
    )
    # Basic bounds clamp as safety (should ideally not be needed if formula is correct)
    i = max(0, min(i, n_atoms - 1))
    j = max(0, min(j, n_atoms - 1))
    return i, j


# --- Lennard-Jones CPU Kernel ---
@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _cpu_lj_energy_kernel(atoms_data: np.ndarray) -> np.float64:
    """
    Computes the raw Lennard-Jones energy sum on the CPU using Numba parallel loops.

    Calculates pairwise interactions using the Lennard-Jones potential for all unique
    pairs (j > i) and sums them. This function performs the core summation
    without applying temperature scaling factors.

    Args:
        atoms_data: NumPy array containing atom data [N, fields], dtype should
                    match `delphi_real`. Expected fields are defined by
                    ATOMFIELD_X, Y, Z, LJ_SIGMA, LJ_EPSILON.

    Returns:
        The raw total Lennard-Jones energy sum as a np.float64 value, suitable
        for high-precision accumulation.
    """
    num_atoms = atoms_data.shape[0]
    if num_atoms < 2:
        return 0.0

    # Use float64 for accumulator array for better precision during summation
    energy_contrib = np.zeros(num_atoms, dtype=np.float64)

    # Parallel loop over atoms i
    for i in prange(num_atoms):
        # Cache atom i data
        xi = atoms_data[i, ATOMFIELD_X]
        yi = atoms_data[i, ATOMFIELD_Y]
        zi = atoms_data[i, ATOMFIELD_Z]
        i_sigma = atoms_data[i, ATOMFIELD_LJ_SIGMA]
        i_eps = atoms_data[i, ATOMFIELD_LJ_EPSILON]

        local_sum = 0.0  # Accumulate per outer loop iteration (float64)
        # Inner loop calculates interactions for pairs (i, j) where j > i
        for j in range(i + 1, num_atoms):
            # Cache atom j data
            xj = atoms_data[j, ATOMFIELD_X]
            yj = atoms_data[j, ATOMFIELD_Y]
            zj = atoms_data[j, ATOMFIELD_Z]
            j_sigma = atoms_data[j, ATOMFIELD_LJ_SIGMA]
            j_eps = atoms_data[j, ATOMFIELD_LJ_EPSILON]

            # Calculate components of distance vector
            dx = xi - xj
            dy = yi - yj
            dz = zi - zj

            # Filter pairs outside the LJ cutoff range based on absolute component distances
            # This replicates the C++ `optABS` check used for initial filtering.
            # It's an unusual filter, but directly translated from the C++ logic provided.
            if (
                abs(dx) > LJCUTOFF_MAX
                or abs(dx) < LJCUTOFF_MIN
                or abs(dy) > LJCUTOFF_MAX
                or abs(dy) < LJCUTOFF_MIN
                or abs(dz) > LJCUTOFF_MAX
                or abs(dz) < LJCUTOFF_MIN
            ):
                continue

            dist2 = dx * dx + dy * dy + dz * dz

            # Ensure distance is not zero (or too close) before division
            if dist2 <= APPROX_ZERO:
                continue

            # Calculate 6th and 12th powers of distance
            dist6 = dist2 * dist2 * dist2
            dist12 = dist6 * dist6

            # Calculate geometric mean of epsilon values for cross-interaction strength.
            # Handles non-positive inputs to prevent math domain errors for sqrt.
            ij_eps = math.sqrt(max(0.0, i_eps) * max(0.0, j_eps))

            # Calculate arithmetic mean of sigma values for cross-interaction distance.
            ij_sigma = (i_sigma + j_sigma) * 0.5

            # Attractive term (6th power)
            f_att = math.pow(ij_sigma, 6) / dist6

            # Repulsive term (12th power)
            f_rep = math.pow(ij_sigma, 12) / dist12

            # Difference of the terms multiplied by 4*eps_ij
            f_diff = 4.0 * ij_eps * (f_rep - f_att)

            # Add the pair's contribution to the total energy
            local_sum += f_diff

        energy_contrib[i] = local_sum

    # Final reduction on the host CPU
    total_raw_energy = np.sum(energy_contrib)

    return total_raw_energy


# --- Lennard-Jones CUDA Kernel ---
@cuda.jit(cache=True, fastmath=False)  # fastmath=False recommended for scientific accuracy
def _cuda_lj_energy_kernel(atoms_data, energy_block):
    """
    CUDA kernel: Computes raw Lennard-Jones energy sum using shared memory reduction.

    Shared memory size is fixed at compile time via MAX_KERNEL_SHARED_MEM_THREADS.
    Reduction logic uses cuda.blockDim.x for runtime thread count. Skips pairs
    with distance squared < APPROX_ZERO.

    Args:
        atoms_data: Device array [N, fields], dtype=`delphi_real`.
        energy_block: Device output array [num_blocks], dtype=`np.float64`.
    """
    n_atoms = atoms_data.shape[0]
    total_pairs = (n_atoms * (n_atoms - 1)) // 2

    # Allocate shared memory based on the compile-time maximum constant
    shared_energy = cuda.shared.array(
        shape=MAX_KERNEL_SHARED_MEM_THREADS, dtype=float64
    )

    tid = cuda.threadIdx.x
    block_dim = cuda.blockDim.x  # Actual number of threads launched in this block
    grid_dim = cuda.gridDim.x
    global_idx_start = cuda.blockIdx.x * block_dim + tid
    grid_stride = grid_dim * block_dim

    # Handle edge case of no pairs
    if total_pairs == 0:
        if tid == 0 and cuda.blockIdx.x == 0:  # Only one thread needs to zero output
            if cuda.blockIdx.x < energy_block.shape[0]:
                energy_block[cuda.blockIdx.x] = 0.0
        return

    local_sum = 0.0  # Use float64 for accumulation

    # Grid-stride loop over unique pairs
    for flat_idx in range(global_idx_start, total_pairs, grid_stride):
        i, j = _cuda_compute_pair_index(flat_idx, n_atoms)  # Call device function

        # Safety check for indices (should ideally not fail if formula is correct)
        if i >= n_atoms or j >= n_atoms or i >= j:
            continue

        # Fetch atom data
        xi = atoms_data[i, ATOMFIELD_X]
        yi = atoms_data[i, ATOMFIELD_Y]
        zi = atoms_data[i, ATOMFIELD_Z]
        i_sigma = atoms_data[i, ATOMFIELD_LJ_SIGMA]
        i_eps = atoms_data[i, ATOMFIELD_LJ_EPSILON]

        xj = atoms_data[j, ATOMFIELD_X]
        yj = atoms_data[j, ATOMFIELD_Y]
        zj = atoms_data[j, ATOMFIELD_Z]
        j_sigma = atoms_data[j, ATOMFIELD_LJ_SIGMA]
        j_eps = atoms_data[j, ATOMFIELD_LJ_EPSILON]

        # Calculate components of distance vector
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj

        # Filter pairs outside the LJ cutoff range based on absolute component distances
        # Using math.fabs for device-side absolute value
        if (
            math.fabs(dx) > LJCUTOFF_MAX
            or math.fabs(dx) < LJCUTOFF_MIN
            or math.fabs(dy) > LJCUTOFF_MAX
            or math.fabs(dy) < LJCUTOFF_MIN
            or math.fabs(dz) > LJCUTOFF_MAX
            or math.fabs(dz) < LJCUTOFF_MIN
        ):
            continue

        dist2 = dx * dx + dy * dy + dz * dz

        # Avoid division by zero for coincident or very close atoms
        if dist2 <= APPROX_ZERO:
            continue

        dist6 = dist2 * dist2 * dist2
        dist12 = dist6 * dist6

        # Combination sigma/epsilon (using device-friendly math functions)
        ij_eps = math.sqrt(i_eps * j_eps)  # Geometric mean for epsilon
        ij_sigma = (i_sigma + j_sigma) / 2.0  # Arithmetic mean for sigma

        # Attractive term (6th power)
        f_att = math.pow(ij_sigma, 6.0) / dist6

        # Repulsive term (12th power)
        f_rep = math.pow(ij_sigma, 12.0) / dist12

        # Difference of the terms multiplied by 4*eps_ij
        f_diff = 4.0 * ij_eps * (f_rep - f_att)

        local_sum += f_diff

    # Store local sum in shared memory (check bounds against allocation size)
    if tid < MAX_KERNEL_SHARED_MEM_THREADS:
        shared_energy[tid] = local_sum
    # Sync before reduction
    cuda.syncthreads()

    # In-block reduction using shared memory (uses runtime block_dim)
    stride = block_dim // 2
    while stride > 0:
        # Check thread index against current stride before accessing shared memory
        if tid < stride:
            # Ensure read index is also within allocated bounds
            read_idx = tid + stride
            if read_idx < MAX_KERNEL_SHARED_MEM_THREADS:
                shared_energy[tid] += shared_energy[read_idx]

        cuda.syncthreads()  # Sync after each reduction level
        stride //= 2

    # First thread writes the block's total sum to the global output array
    if tid == 0:
        if cuda.blockIdx.x < energy_block.shape[0]:
            energy_block[cuda.blockIdx.x] = shared_energy[0]


# --- CUDA Host Function for LJ ---
def _cuda_calc_lj_energy(platform: Platform, atoms_data: np.ndarray) -> np.float64:
    """
    Host function: Orchestrates raw Lennard-Jones energy calculation on GPU via CUDA.

    Uses properties from the activated Platform object (e.g., max threads per block)
    to determine an adaptive grid configuration. Allocates memory, launches kernel,
    retrieves block sums, and performs final host summation.

    Args:
        platform: Activated pydelphi.platforms.Platform object for CUDA.
        atoms_data: NumPy array [N, fields], dtype=`delphi_real`.

    Returns:
        The raw total Lennard-Jones energy sum computed on the GPU, as np.float64.
    """
    n_atoms = atoms_data.shape[0]
    if n_atoms < 2:
        return 0.0
    total_pairs = (n_atoms * (n_atoms - 1)) // 2
    if total_pairs == 0:
        return 0.0

    # --- Get Device Properties from Platform Object ---
    active_props = platform.properties()
    if not active_props or platform.active != "cuda":
        raise RuntimeError(
            "CUDA calculation called, but Platform object is not activated for CUDA or has no properties."
        )

    device_max_threads_per_block = active_props.get("MAX_THREADS_PER_BLOCK", None)
    selected_device_id = platform.names["cuda"]["selected_id"]  # Get selected ID

    if device_max_threads_per_block is None:
        vprint(
            DEBUG,
            _VERBOSITY,
            f"Warning: MAX_THREADS_PER_BLOCK not found for device {selected_device_id}. Using default: {MAX_KERNEL_SHARED_MEM_THREADS}",
        )
        device_max_threads_per_block = MAX_KERNEL_SHARED_MEM_THREADS  # Fallback

    # --- Adaptive Grid Configuration ---
    initial_threads_per_block = min(
        256, device_max_threads_per_block
    )  # Start guess capped by device max
    min_threads_per_block = 32
    threads_per_block = max(min_threads_per_block, initial_threads_per_block)

    if threads_per_block <= 0:
        threads_per_block = min_threads_per_block  # Safety check
    blocks_per_grid = (total_pairs + threads_per_block - 1) // threads_per_block

    # Adaptation loop
    while (
        blocks_per_grid <= threads_per_block
        and threads_per_block > min_threads_per_block
    ):
        threads_per_block_prev = threads_per_block
        threads_per_block = max(min_threads_per_block, threads_per_block // 2)
        if threads_per_block == threads_per_block_prev:
            break  # Avoid infinite loop if it can't reduce further
        blocks_per_grid = (total_pairs + threads_per_block - 1) // threads_per_block

    if blocks_per_grid == 0 and total_pairs > 0:
        blocks_per_grid = 1
    threads_per_block = min(
        threads_per_block, device_max_threads_per_block
    )  # Final cap

    vprint(
        DEBUG,
        _VERBOSITY,
        f"LJ CUDA Grid Config: blocks={blocks_per_grid}, threads={threads_per_block}",
    )

    # --- Prepare Data & Launch ---
    if threads_per_block <= 0:
        raise ValueError("Invalid threads_per_block calculated for LJ CUDA kernel.")
    try:
        # Ensure context is set to the selected device
        cuda.select_device(selected_device_id)
        atoms_data_device = cuda.to_device(atoms_data)
        energy_block_host_zeros = np.zeros(blocks_per_grid, dtype=np.float64)
        energy_block_device = cuda.to_device(energy_block_host_zeros)
    except Exception as e:
        print(
            f"Error preparing CUDA memory/context on device {selected_device_id} for LJ: {e}"
        )
        raise  # Re-raise critical error

    # Launch kernel
    _cuda_lj_energy_kernel[blocks_per_grid, threads_per_block](
        atoms_data_device, energy_block_device
    )
    cuda.synchronize()  # Wait for completion

    # Copy results back and sum on host
    energy_block_host = energy_block_device.copy_to_host()
    total_raw_energy = np.sum(energy_block_host)

    return total_raw_energy


# === Public Interface ===
def calc_lj_energy(
    platform: Platform, atoms_data: np.ndarray, temperature: float
) -> np.float64:
    """
    Calculates the total scaled Lennard-Jones energy for a set of atoms.

    Uses the provided activated `platform` object to select the backend (CPU or CUDA)
    and its specific configuration. Computes the raw pairwise energy sum and
    applies the final temperature scaling factor (fTemper / 298.0).

    Args:
        platform: Activated `pydelphi.platforms.Platform` object.
        atoms_data: NumPy array [N, fields] with atom coordinates, LJ sigma, and LJ epsilon.
                    Will be cast to `delphi_real` if necessary.
                    Expected fields: ATOMFIELD_X, Y, Z, LJ_SIGMA, LJ_EPSILON.
        temperature: The current temperature in Kelvin (e.g., 298.0 K).

    Returns:
        The final, scaled total Lennard-Jones energy as a `np.float64` value.
        Returns 0.0 if fewer than two atoms are provided.

    Raises:
        RuntimeError: If the platform is not activated, or if CUDA is selected
                      but unavailable or fails during execution.
        ImportError: If the selected backend requires Numba and it's not installed.
    """
    # Ensure input data has the correct precision type for compute kernels
    if atoms_data.dtype != delphi_real:
        atoms_data_cast = atoms_data.astype(delphi_real)
    else:
        atoms_data_cast = atoms_data

    n_atoms = atoms_data_cast.shape[0]
    if n_atoms < 2:
        return np.float64(0.0)

    raw_energy_sum = np.float64(0.0)  # Initialize as float64

    # --- Select Backend ---
    if platform.active == "cuda":
        if not platform.names.get("cuda", {}).get("available"):
            raise RuntimeError(
                "CUDA platform selected, but no usable CUDA devices were detected by Platform object."
            )
        try:
            raw_energy_sum = _cuda_calc_lj_energy(platform, atoms_data_cast)
        except Exception as e:
            print(f"Error during CUDA Lennard-Jones energy calculation: {e}")
            # Optional: Add traceback print for debugging
            # import traceback; traceback.print_exc()
            raise e  # Re-raise the exception to signal failure

    elif platform.active == "cpu":
        try:
            vprint(
                DEBUG,
                _VERBOSITY,
                f"Number of CPU threads used for LJ: {get_num_threads()}",
            )
            num_cpu_threads = platform.names["cpu"]["num_threads"]
            set_num_threads(num_cpu_threads)
            raw_energy_sum = _cpu_lj_energy_kernel(atoms_data_cast)
        except (
            ImportError
        ) as e:  # Catch error if dummy njit was used or Numba isn't installed
            raise ImportError(
                f"CPU LJ calculation requires Numba, but it's not installed or dummy was used: {e}"
            )
        except Exception as e:
            print(f"Error during CPU Lennard-Jones energy calculation: {e}")
            raise e

    else:
        raise RuntimeError(
            f"Unknown active platform in Platform object for LJ calculation: '{platform.active}'"
        )

    # --- Apply final temperature scaling ---
    # Epsilon values are typically in kT at a reference temperature (e.g., 298K).
    scaled_energy = raw_energy_sum * (np.float64(temperature) / 298.0)

    return np.float64(scaled_energy)  # Ensure final return is float64
