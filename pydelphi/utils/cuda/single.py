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

"""
This module defines CUDA kernels for accessing and manipulating single-precision
atom data within the pydelphi framework. These kernels are designed for use with
Numba's CUDA target and provide efficient access to atom properties on the GPU.
"""

import numpy as np
from numba import cuda
from numba import float32

from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_GRID_X,
    ATOMFIELD_GRID_Y,
    ATOMFIELD_GRID_Z,
    ATOMFIELD_CHARGE,
    ATOMFIELD_RADIUS,
    ATOMFIELD_GAUSS_SIGMA,
    ATOMFIELD_RES_KEY,
    # ResKey mapping constants
    CHEM_KIND_BASE_OFFSET_VALUES,
    RES_KIND_BLOCK_SIZE,
    RES_KIND_UNKNOWN,
    RES_KIND_PROTEIN,
    RES_KIND_NUCLEIC,
    RES_KIND_LIPID,
    RES_KIND_CARBOHYDRATE,
)


@cuda.jit(["bool_(float32[:])"], device=True)
def cu_is_atom_res_protein(atom_data):
    """
    Check if an atom belongs to a protein residue by examining its residue key.

    Args:
        atom_data (np.ndarray): Atom record array with residue key at ATOMFIELD_RES_KEY.

    Returns:
        bool: True if the atom is part of a protein residue.
    """
    res_key = atom_data[ATOMFIELD_RES_KEY]
    return (
        CHEM_KIND_BASE_OFFSET_VALUES[RES_KIND_PROTEIN]
        <= res_key
        < CHEM_KIND_BASE_OFFSET_VALUES[RES_KIND_PROTEIN] + RES_KIND_BLOCK_SIZE
    )


@cuda.jit(["int32(float32)"], device=True)
def cu_get_residue_kind(res_key):
    """
    Determine the residue kind based on its numeric key.

    Checks the key against known kind ranges using if/elif statements.

    Args:
        res_key (float): Residue key.

    Returns:
        int: RES_KIND_PROTEIN, RES_KIND_NUCLEIC, etc., or 0 if unknown/UNK.
        Note: Returns 0 for keys that do not fall into any defined kind range,
        including the specific UNK key (99999.0) if it's outside these ranges.
    """
    # Unrolling the dictionary iteration into explicit checks is Numba-friendly
    # Accessing values from global constant dict KIND_BASE_OFFSET is fine
    if (
        CHEM_KIND_BASE_OFFSET_VALUES[RES_KIND_PROTEIN]
        <= res_key
        < CHEM_KIND_BASE_OFFSET_VALUES[RES_KIND_PROTEIN] + RES_KIND_BLOCK_SIZE
    ):
        return RES_KIND_PROTEIN
    elif (
        CHEM_KIND_BASE_OFFSET_VALUES[RES_KIND_NUCLEIC]
        <= res_key
        < CHEM_KIND_BASE_OFFSET_VALUES[RES_KIND_NUCLEIC] + RES_KIND_BLOCK_SIZE
    ):
        return RES_KIND_NUCLEIC
    elif (
        CHEM_KIND_BASE_OFFSET_VALUES[RES_KIND_LIPID]
        <= res_key
        < CHEM_KIND_BASE_OFFSET_VALUES[RES_KIND_LIPID] + RES_KIND_BLOCK_SIZE
    ):
        return RES_KIND_LIPID
    elif (
        CHEM_KIND_BASE_OFFSET_VALUES[RES_KIND_CARBOHYDRATE]
        <= res_key
        < CHEM_KIND_BASE_OFFSET_VALUES[RES_KIND_CARBOHYDRATE] + RES_KIND_BLOCK_SIZE
    ):
        return RES_KIND_CARBOHYDRATE

    # Handle the 'UNK' key or any key outside the defined ranges
    return RES_KIND_UNKNOWN


@cuda.jit(["float32(float32[:])", "float32(float64[:])"], device=True)
def cu_get_atom_x(atom_data):
    """Returns the x-coordinate of an atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atomic properties.

    Returns:
        float32: The x-coordinate of the atom.
    """
    return float32(atom_data[ATOMFIELD_X])


@cuda.jit(["float32(float32[:])", "float32(float64[:])"], device=True)
def cu_get_atom_y(atom_data):
    """Returns the y-coordinate of an atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atomic properties.

    Returns:
        float32: The y-coordinate of the atom.
    """
    return float32(atom_data[ATOMFIELD_Y])


@cuda.jit(["float32(float32[:])", "float32(float64[:])"], device=True)
def cu_get_atom_z(atom_data):
    """Returns the z-coordinate of an atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atomic properties.

    Returns:
        float32: The z-coordinate of the atom.
    """
    return float32(atom_data[ATOMFIELD_Z])


@cuda.jit(["float32(float32[:])", "float32(float64[:])"], device=True)
def cu_get_atom_grid_x(atom_data):
    """Returns the atom's x-coordinate in grid units.

    Args:
        atom_data (np.ndarray): A NumPy array containing atomic properties.

    Returns:
        float32: The x-coordinate in grid units.
    """
    return float32(atom_data[ATOMFIELD_GRID_X])


@cuda.jit(["float32(float32[:])", "float32(float64[:])"], device=True)
def cu_get_atom_grid_y(atom_data):
    """Returns the atom's y-coordinate in grid units.

    Args:
        atom_data (np.ndarray): A NumPy array containing atomic properties.

    Returns:
        float32: The y-coordinate in grid units.
    """
    return float32(atom_data[ATOMFIELD_GRID_Y])


@cuda.jit(["float32(float32[:])", "float32(float64[:])"], device=True)
def cu_get_atom_grid_z(atom_data):
    """Returns the atom's z-coordinate in grid units.

    Args:
        atom_data (np.ndarray): A NumPy array containing atomic properties.

    Returns:
        float32: The z-coordinate in grid units.
    """
    return float32(atom_data[ATOMFIELD_GRID_Z])


@cuda.jit(["float32(float32[:])", "float32(float64[:])"], device=True)
def cu_get_atom_charge(atom_data):
    """Returns the charge of the atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atomic properties.

    Returns:
        float32: The atomic charge.
    """
    return float32(atom_data[ATOMFIELD_CHARGE])


@cuda.jit(["float32(float32[:])", "float32(float64[:])"], device=True)
def cu_get_atom_radius(atom_data):
    """Returns the van der Waals radius of the atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atomic properties.

    Returns:
        float32: The atomic radius.
    """
    return float32(atom_data[ATOMFIELD_RADIUS])


@cuda.jit(["float32(float32[:])", "float32(float64[:])"], device=True)
def cu_get_atom_gaussiansigma(atom_data):
    """Returns the Gaussian sigma value of the atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atomic properties.

    Returns:
        float32: The Gaussian sigma value.
    """
    return float32(atom_data[ATOMFIELD_GAUSS_SIGMA])


@cuda.jit(
    [
        "float32[:](float32[:], float32[:], float32[:], float32)",
        "float64[:](float64[:], float64[:], float64[:], float64)",
    ],
    device=True,
)
def cu_to_grid_coords(crd3d_nparray, crd3d_grid_crd3d, grid_origin, h):
    """Computes the grid coordinates from real-space coordinates.

    Args:
        crd3d_nparray (np.ndarray): 1D NumPy array (size 3) containing real-space coordinates.
        crd3d_grid_crd3d (np.ndarray): 1D NumPy array (size 3) to store computed grid coordinates.
        grid_origin (np.ndarray): 1D NumPy array (size 3) representing the origin of the grid.
        h (float): The spacing between grid points in Angstroms.

    Returns:
        np.ndarray: The updated `crd3d_grid_crd3d` array with computed grid coordinates.
    """
    crd3d_grid_crd3d[0] = (crd3d_nparray[ATOMFIELD_X] - grid_origin[ATOMFIELD_X]) / h
    crd3d_grid_crd3d[1] = (crd3d_nparray[ATOMFIELD_Y] - grid_origin[ATOMFIELD_Y]) / h
    crd3d_grid_crd3d[2] = (crd3d_nparray[ATOMFIELD_Z] - grid_origin[ATOMFIELD_Z]) / h
    return crd3d_grid_crd3d
