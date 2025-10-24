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
This module calculates the non-linear Coulombic energy by summing the product
of atomic charges and their interpolated electrostatic potentials from a grid.
It supports both CPU (Numba-optimized) and CUDA (GPU-accelerated) computation.

The core idea is to obtain the electrostatic potential at each atom's position
by trilinear interpolation from a pre-computed 3D potential grid (phi_grid),
and then sum (charge * potential) for all atoms.

Functions:
- `_cpu_interp_trilinear`: Performs trilinear interpolation on the CPU.
- `_cpu_calc_energy_clbnonl`: Calculates non-linear Coulombic energy on the CPU.
- `_cuda_interp_trilinear`: Performs trilinear interpolation on the GPU (device function).
- `_cuda_calc_energy_clbnonl_kernel`: CUDA kernel for parallel non-linear Coulombic energy calculation.
- `_cuda_calc_energy_clbnonl`: Host function to orchestrate CUDA-based calculation.
- `calc_energy_clbnonl`: Public interface to dispatch to the appropriate backend.
"""

import numpy as np
from numba import njit, cuda
from typing import Tuple
from pydelphi.foundation.platforms import Platform
from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_CHARGE,
)


@njit(nogil=True, boundscheck=False, cache=True)
def _cpu_interp_trilinear(phi_grid, pos, grid_origin, grid_spacing, grid_shape):
    """
    Performs trilinear interpolation of the electrostatic potential at a given
    position on the CPU.

    Parameters
    ----------
    phi_grid : np.ndarray
        A 3D NumPy array representing the electrostatic potential grid.
    pos : np.ndarray
        The (x, y, z) coordinates (in Angstroms) at which to interpolate the potential.
    grid_origin : np.ndarray
        The (xmin, ymin, zmin) coordinates of the grid's origin.
    grid_spacing : float
        The spacing between grid points (Angstroms/grid unit).
    grid_shape : np.ndarray
        The shape of the `phi_grid` array (nx, ny, nz).

    Returns
    -------
    float
        The interpolated electrostatic potential at the given position.
        Returns 0.0 if the position is outside the grid boundaries.
    """
    x, y, z = pos
    xmin, ymin, zmin = grid_origin
    nx, ny, nz = grid_shape

    fx = (x - xmin) / grid_spacing
    fy = (y - ymin) / grid_spacing
    fz = (z - zmin) / grid_spacing

    i = int(fx)
    j = int(fy)
    k = int(fz)

    if i < 0 or i >= nx - 1 or j < 0 or j >= ny - 1 or k < 0 or k >= nz - 1:
        return 0.0

    dx = fx - i
    dy = fy - j
    dz = fz - k

    c000 = phi_grid[k, j, i]
    c001 = phi_grid[k, j, i + 1]
    c010 = phi_grid[k, j + 1, i]
    c011 = phi_grid[k, j + 1, i + 1]
    c100 = phi_grid[k + 1, j, i]
    c101 = phi_grid[k + 1, j, i + 1]
    c110 = phi_grid[k + 1, j + 1, i]
    c111 = phi_grid[k + 1, j + 1, i + 1]

    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy

    return c0 * (1 - dz) + c1 * dz


@njit(nogil=True, boundscheck=False, cache=True)
def _cpu_calc_energy_clbnonl(
    atoms_data, phi_grid, grid_origin, grid_spacing, grid_shape
):
    """
    Calculates the non-linear Coulombic energy on the CPU.

    This is done by summing (charge * interpolated_potential) for each atom.

    Parameters
    ----------
    atoms_data : np.ndarray
        A 2D NumPy array of atom data. Expected columns for atom `i` are:
        `atoms_data[i, ATOMFIELD_X]`, `atoms_data[i, ATOMFIELD_Y]`,
        `atoms_data[i, ATOMFIELD_Z]` for coordinates, and
        `atoms_data[i, ATOMFIELD_CHARGE]` for charge.
    phi_grid : np.ndarray
        A 3D NumPy array representing the electrostatic potential grid.
    grid_origin : np.ndarray
        The (xmin, ymin, zmin) coordinates of the grid's origin.
    grid_spacing : float
        The spacing between grid points (Angstroms/grid unit).
    grid_shape : np.ndarray
        The shape of the `phi_grid` array (nx, ny, nz).

    Returns
    -------
    float
        The total non-linear Coulombic energy.
    """
    energy = 0.0
    for i in range(atoms_data.shape[0]):
        x = atoms_data[i, ATOMFIELD_X]
        y = atoms_data[i, ATOMFIELD_Y]
        z = atoms_data[i, ATOMFIELD_Z]
        q = atoms_data[i, ATOMFIELD_CHARGE]

        phi_i = _cpu_interp_trilinear(
            phi_grid, (x, y, z), grid_origin, grid_spacing, grid_shape
        )
        energy += q * phi_i
    return energy


# === CUDA VERSION ===
@cuda.jit(cache=True, device=True)
def _cuda_interp_trilinear(phi_grid, x, y, z, grid_origin, grid_spacing, grid_shape):
    """
    Performs trilinear interpolation of the electrostatic potential at a given
    position on a CUDA device.

    Parameters
    ----------
    phi_grid : np.ndarray
        A 3D NumPy array representing the electrostatic potential grid (on device).
    x, y, z : float
        The x, y, and z coordinates (in Angstroms) at which to interpolate the potential.
    grid_origin : np.ndarray
        The (xmin, ymin, zmin) coordinates of the grid's origin.
    grid_spacing : float
        The spacing between grid points (Angstroms/grid unit).
    grid_shape : np.ndarray
        The shape of the `phi_grid` array (nx, ny, nz).

    Returns
    -------
    float
        The interpolated electrostatic potential at the given position.
        Returns 0.0 if the position is outside the grid boundaries.
    """
    xmin, ymin, zmin = grid_origin
    nx, ny, nz = grid_shape

    fx = (x - xmin) / grid_spacing
    fy = (y - ymin) / grid_spacing
    fz = (z - zmin) / grid_spacing

    i = int(fx)
    j = int(fy)
    k = int(fz)

    if i < 0 or i >= nx - 1 or j < 0 or j >= ny - 1 or k < 0 or k >= nz - 1:
        return 0.0

    dx = fx - i
    dy = fy - j
    dz = fz - k

    c000 = phi_grid[k, j, i]
    c001 = phi_grid[k, j, i + 1]
    c010 = phi_grid[k, j + 1, i]
    c011 = phi_grid[k, j + 1, i + 1]
    c100 = phi_grid[k + 1, j, i]
    c101 = phi_grid[k + 1, j, i + 1]
    c110 = phi_grid[k + 1, j + 1, i]
    c111 = phi_grid[k + 1, j + 1, i + 1]

    c00 = c000 * (1 - dx) + c001 * dx
    c01 = c010 * (1 - dx) + c011 * dx
    c10 = c100 * (1 - dx) + c101 * dx
    c11 = c110 * (1 - dx) + c111 * dx

    c0 = c00 * (1 - dy) + c01 * dy
    c1 = c10 * (1 - dy) + c11 * dy

    return c0 * (1 - dz) + c1 * dz


@cuda.jit(cache=True)
def _cuda_calc_energy_clbnonl_kernel(
    atoms_data, phi_grid, grid_origin, grid_spacing, grid_shape, energy_out
):
    """
    CUDA kernel for calculating the non-linear Coulombic energy.

    Each thread processes one atom, interpolates its potential, and atomically
    adds its contribution to a shared global energy sum.

    Parameters
    ----------
    atoms_data : np.ndarray
        Device array of atom data.
    phi_grid : np.ndarray
        Device array of the 3D electrostatic potential grid.
    grid_origin : np.ndarray
        Device array containing the grid's origin coordinates.
    grid_spacing : float
        The spacing between grid points.
    grid_shape : Tuple[int, int, int]
        Device array containing the grid's shape (dimensions).
    energy_out : np.ndarray
        A single-element device array (e.g., `cuda.device_array(1)`)
        where the total energy will be atomically accumulated.
    """
    i = cuda.grid(1)
    if i >= atoms_data.shape[0]:
        return

    x = atoms_data[i, ATOMFIELD_X]
    y = atoms_data[i, ATOMFIELD_Y]
    z = atoms_data[i, ATOMFIELD_Z]
    q = atoms_data[i, ATOMFIELD_CHARGE]

    phi_i = _cuda_interp_trilinear(
        phi_grid, x, y, z, grid_origin, grid_spacing, grid_shape
    )
    cuda.atomic.add(energy_out, 0, q * phi_i)


def _cuda_calc_energy_clbnonl(
    atoms_data, phi_grid, grid_origin, grid_spacing, grid_shape
):
    """
    CUDA kernel for calculating the non-linear Coulombic energy.

    Each thread processes one atom, interpolates its potential, and atomically
    adds its contribution to a shared global energy sum.

    Parameters
    ----------
    atoms_data : np.ndarray
        Device array of atom data.
    phi_grid : np.ndarray
        Device array of the 3D electrostatic potential grid.
    grid_origin : np.ndarray
        Device array of 3-floats containing the grid's origin coordinates.
    grid_spacing : float
        The spacing between grid points.
    grid_shape : np.ndarray
        Device array of 3-ints containing the grid's shape (dimensions).
    energy_out : np.ndarray
        A single-element device array (e.g., `cuda.device_array(1)`)
        where the total energy will be atomically accumulated.
    """
    n_atoms = atoms_data.shape[0]
    threads_per_block = 128
    blocks_per_grid = (n_atoms + threads_per_block - 1) // threads_per_block

    d_atoms = cuda.to_device(atoms_data)
    d_phi = cuda.to_device(phi_grid)
    d_energy = cuda.device_array(1, dtype=atoms_data.dtype)
    d_energy[0] = 0.0

    grid_origin_tuple = tuple(grid_origin)
    grid_shape_tuple = tuple(grid_shape)

    _cuda_calc_energy_clbnonl_kernel[blocks_per_grid, threads_per_block](
        d_atoms, d_phi, grid_origin_tuple, grid_spacing, grid_shape_tuple, d_energy
    )

    return d_energy.copy_to_host()[0]


def calc_energy_clbnonl(
    platform: Platform,
    atoms_data: np.ndarray,
    phi_grid: np.ndarray,
    grid_origin: Tuple[float, float, float],
    grid_spacing: float,
    grid_shape: Tuple[int, int, int],
) -> float:
    """
    Calculates the non-linear Coulombic energy based on atomic charges and
    an electrostatic potential grid.

    This function acts as a dispatcher, choosing between CPU and CUDA backends
    based on the provided `platform` object.

    Parameters
    ----------
    platform : Platform
        An activated `Platform` object indicating the desired computation backend ('cpu' or 'cuda').
    atoms_data : np.ndarray
        A 2D NumPy array of atom data. It is expected to contain at least X, Y, Z coordinates
        and CHARGE, accessible via `ATOMFIELD_X`, `ATOMFIELD_Y`, `ATOMFIELD_Z`, `ATOMFIELD_CHARGE` constants.
    phi_grid : np.ndarray
        A 3D NumPy array representing the pre-computed electrostatic potential field.
        Its dimensions should match `grid_shape`.
    grid_origin : np.ndarray
        A tuple `(xmin, ymin, zmin)` specifying the physical coordinates of the
        origin (lowest corner) of the `phi_grid`.
    grid_spacing : float
        The uniform spacing (voxel size) between grid points in Angstroms.
    grid_shape : np.ndarray
        A tuple `(nx, ny, nz)` representing the dimensions (number of points)
        of the `phi_grid` along X, Y, and Z axes, respectively.

    Returns
    -------
    float
        The total non-linear Coulombic energy calculated from `sum(charge_i * phi_i)`.

    Raises
    ------
    ValueError
        If an unsupported platform is specified.
    """
    if platform.name == "cpu":
        return _cpu_calc_energy_clbnonl(
            atoms_data, phi_grid, grid_origin, grid_spacing, grid_shape
        )
    elif platform.name == "cuda":
        return _cuda_calc_energy_clbnonl(
            atoms_data, phi_grid, grid_origin, grid_spacing, grid_shape
        )
    else:
        raise ValueError(f"Unsupported platform: {platform.name}")
