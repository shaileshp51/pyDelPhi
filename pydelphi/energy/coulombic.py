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
from numba import njit, prange, get_num_threads, set_num_threads, cuda, float64

from pydelphi.config.logging_config import DEBUG
from pydelphi.foundation.platforms import Platform
from pydelphi.config.global_runtime import delphi_real, vprint
from pydelphi.constants import ConstDelPhiFloats
from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_CHARGE,
    LEN_ATOMFIELDS,
)

APPROX_ZERO = ConstDelPhiFloats.ApproxZero.value

# Sliced array indices for the compact arrays (SoA uses separate arrays, but keep these for compatibility)
SLICEDFIELD_X = 0
SLICEDFIELD_Y = 1
SLICEDFIELD_Z = 2
SLICEDFIELD_CHARGE = 3

LEN_XYZCHARGE = 4

# CUDA configuration constants (global)
MAX_BLOCKS_DIM = 256  # used only when computing tile size
THREADS_PER_BLOCK = 256  # 1D block size; tune (128/256/512) depending on GPU
# Note: choose THREADS_PER_BLOCK such that THREADS_PER_BLOCK <= device max threads per block


# --- CPU kernel (unchanged) ---
@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _cpu_coulombic_energy_kernel(atoms_data: np.ndarray) -> np.float64:
    num_atoms = atoms_data.shape[0]
    if num_atoms < 2:
        return 0.0

    energy_contrib = np.zeros(num_atoms, dtype=np.float64)

    for i in prange(num_atoms):
        xi, yi, zi, qi = (
            atoms_data[i, ATOMFIELD_X],
            atoms_data[i, ATOMFIELD_Y],
            atoms_data[i, ATOMFIELD_Z],
            atoms_data[i, ATOMFIELD_CHARGE],
        )

        local_sum = 0.0
        for j in range(i + 1, num_atoms):
            xj, yj, zj, qj = (
                atoms_data[j, ATOMFIELD_X],
                atoms_data[j, ATOMFIELD_Y],
                atoms_data[j, ATOMFIELD_Z],
                atoms_data[j, ATOMFIELD_CHARGE],
            )

            dx, dy, dz = xi - xj, yi - yj, zi - zj
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq > APPROX_ZERO:
                dist = math.sqrt(dist_sq)
                local_sum += (qi * qj) / dist

        energy_contrib[i] = local_sum

    return np.sum(energy_contrib)


# === CUDA kernel (SoA, no shared memory) ===
@cuda.jit(cache=True, fastmath=False)
def _cuda_coulombic_energy_kernel_soa(x_arr, y_arr, z_arr, q_arr, energy_out, n_atoms):
    """
    Per-thread computes contributions for several i indices (grid-stride) with j loop.
    No shared memory used. Arrays are SoA: x_arr, y_arr, z_arr, q_arr (device arrays).
    energy_out is a device array where each thread writes its partial sum (one element per thread).
    """
    tid = cuda.grid(1)  # global 1D thread id
    stride = cuda.gridsize(1)  # total threads in grid

    # Each thread will compute partial sums for i = tid, tid+stride, ...
    # energy_out should have at least as many elements as number of launched threads
    # We'll write partial sums to energy_out[tid]
    # Ensure we don't access out-of-bounds on write if tid >= energy_out.size

    # Precompute n_atoms as local variable
    n = n_atoms

    # Safety: if tid >= energy_out.size we still can compute nothing
    if tid >= energy_out.shape[0]:
        return

    total_local = 0.0

    # Outer loop: iterate over i atoms assigned to this thread
    i = tid
    while i < n:
        xi = x_arr[i]
        yi = y_arr[i]
        zi = z_arr[i]
        qi = q_arr[i]

        # Inner loop: j > i (to avoid double counting)
        # We iterate j from i+1 to n-1
        # Note: the memory access pattern on j is fully sequential; every thread repeats it.
        # This is bandwidth heavy; using SoA helps coalescing per read.
        for j in range(i + 1, n):
            dx = xi - x_arr[j]
            dy = yi - y_arr[j]
            dz = zi - z_arr[j]
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq > APPROX_ZERO:
                total_local += (qi * q_arr[j]) / math.sqrt(dist_sq)

        i += stride

    # Write partial sum to device output (one slot per thread index)
    energy_out[tid] = total_local


# --- Tile-size chooser (keeps same semantics) ---
@njit(nogil=True, boundscheck=False, cache=True)
def _choose_tile_size(
    n_atoms,
    threads_per_block=THREADS_PER_BLOCK,
    blocks_per_grid=MAX_BLOCKS_DIM * MAX_BLOCKS_DIM,
    safety_factor=1.5,
    max_tile_size=32768,
):
    """
    Compute a suitable tile size for GPU execution (kept for compatibility).
    When using SoA kernel without per-tile transfers, tile_size can be set large;
    this function preserves previous heuristics.
    """
    total_threads = threads_per_block * blocks_per_grid
    base_tile = math.ceil(n_atoms / total_threads) if total_threads > 0 else n_atoms
    tile_size = int(base_tile * safety_factor)

    tile_size = max(threads_per_block, tile_size)  # at least one block worth
    tile_size = min(tile_size, max_tile_size)

    return tile_size


def _cuda_calc_coulombic_energy(
    platform: Platform, atoms_data_slice_xyzq: np.ndarray
) -> np.float64:
    """
    Orchestrate CUDA execution with SoA layout and a single persistent device copy.

    atoms_data_slice_xyzq is expected to be shape (n_atoms, 4) but we convert to SoA.
    """
    n_atoms = atoms_data_slice_xyzq.shape[0]
    if n_atoms < 2:
        return 0.0

    if platform.active != "cuda":
        raise RuntimeError("CUDA platform not active.")

    # Convert sliced AoS (n,4) into SoA flattened arrays for better coalescing
    x_host = atoms_data_slice_xyzq[:, SLICEDFIELD_X].astype(delphi_real, copy=True)
    y_host = atoms_data_slice_xyzq[:, SLICEDFIELD_Y].astype(delphi_real, copy=True)
    z_host = atoms_data_slice_xyzq[:, SLICEDFIELD_Z].astype(delphi_real, copy=True)
    q_host = atoms_data_slice_xyzq[:, SLICEDFIELD_CHARGE].astype(delphi_real, copy=True)

    # Transfer arrays once to device (keep resident)
    x_dev = cuda.to_device(x_host)
    y_dev = cuda.to_device(y_host)
    z_dev = cuda.to_device(z_host)
    q_dev = cuda.to_device(q_host)

    # Decide grid/block sizing
    threads_per_block = THREADS_PER_BLOCK
    # make grid large enough to cover CPU-wide concurrency; we want many threads to saturate GPU
    # Choose number of blocks such that total threads >= min(n_atoms, some multiple)
    # A reasonable starting grid:
    blocks = max(1, (n_atoms + threads_per_block - 1) // threads_per_block)
    # cap number of blocks to avoid creating enormous grids (tune as needed)
    blocks = min(blocks, 65536)  # conservative max blocks
    grid_size = blocks * threads_per_block
    total_threads = blocks * threads_per_block

    # Allocate device array for partial energy per thread (size = total_threads)
    energy_block_device = cuda.device_array(shape=(total_threads,), dtype=np.float64)

    # Launch kernel: each thread computes contributions for some i's in grid-stride
    _cuda_coulombic_energy_kernel_soa[blocks, threads_per_block](
        x_dev, y_dev, z_dev, q_dev, energy_block_device, np.int32(n_atoms)
    )

    # Copy back partial sums and reduce on host
    energy_host = energy_block_device.copy_to_host()
    total_raw_energy = float(np.sum(energy_host))

    return total_raw_energy


@njit(nogil=True, boundscheck=False, cache=True, parallel=True)
def _slice_atoms_xyzq_for_cuda(n_atoms, atoms_data, atoms_data_slice_xyzq, dtype_real):
    """
    Extract x,y,z,q fields into a compact (n,4) array for initial conversion to SoA.
    This is still AoS on host but small and fast; then _cuda_calc_coulombic_energy converts to SoA.
    """
    for i in prange(n_atoms):
        atoms_data_slice_xyzq[i, SLICEDFIELD_X] = dtype_real(atoms_data[i, ATOMFIELD_X])
        atoms_data_slice_xyzq[i, SLICEDFIELD_Y] = dtype_real(atoms_data[i, ATOMFIELD_Y])
        atoms_data_slice_xyzq[i, SLICEDFIELD_Z] = dtype_real(atoms_data[i, ATOMFIELD_Z])
        atoms_data_slice_xyzq[i, SLICEDFIELD_CHARGE] = dtype_real(
            atoms_data[i, ATOMFIELD_CHARGE]
        )


# === Public API ===
def calc_coulombic_energy(
    platform: Platform, atoms_data: np.ndarray, indi: float, epkt: float
) -> np.float64:
    """
    Compute Coulombic energy using either CPU or CUDA backend.
    For CUDA backend we convert the input to a compact (n,4) slice and then to SoA arrays.
    """
    if atoms_data.dtype != delphi_real:
        atoms_data_cast = atoms_data.astype(delphi_real)
    else:
        atoms_data_cast = atoms_data

    n_atoms = atoms_data_cast.shape[0]
    if n_atoms < 2:
        return np.float64(0.0)

    raw_energy_sum = np.float64(0.0)

    if platform.active == "cuda":
        if not platform.names.get("cuda", {}).get("available"):
            raise RuntimeError("CUDA selected but no devices available.")

        # Prepare compact (n,4) array on host (fast)
        atoms_data_slice_xyzq = np.empty((n_atoms, LEN_XYZCHARGE), dtype=delphi_real)
        _slice_atoms_xyzq_for_cuda(
            n_atoms, atoms_data_cast, atoms_data_slice_xyzq, dtype_real=delphi_real
        )

        # Run the SoA CUDA kernel (which itself converts to SoA device arrays)
        raw_energy_sum = _cuda_calc_coulombic_energy(platform, atoms_data_slice_xyzq)

    elif platform.active == "cpu":
        vprint(DEBUG, 0, f"Number of threads used: {get_num_threads()}")
        set_num_threads(platform.names["cpu"]["num_threads"])
        raw_energy_sum = _cpu_coulombic_energy_kernel(atoms_data_cast)

    else:
        raise RuntimeError(f"Unknown platform: {platform.active}")

    if abs(float(indi)) < APPROX_ZERO:
        raise ValueError(f"Dielectric constant 'indi' ({indi}) is too small.")

    scaled_energy = raw_energy_sum * (1.0 / np.float64(indi)) * np.float64(epkt)
    return np.float64(scaled_energy)
