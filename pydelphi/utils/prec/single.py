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

# !/usr/bin/env python
# coding: utf-8
"""
This module provides utility functions for 3D coordinate operations and atom data manipulation.

It includes functions for:
    - Basic 3D vector operations (distance, vector addition, subtraction, etc.)
    - Atom data handling (accessing and setting atom properties)
    - Coordinate transformations (real to grid coordinates)
    - Optimized mathematical operations using Numba for performance.

The module leverages NumPy for numerical operations and Numba for just-in-time compilation,
enhancing the speed of coordinate calculations and atom-related computations.
"""

import math
import numpy as np

from numba import jit

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
    ATOMFIELD_ATOMIC_NUMBER,
    ATOMFIELD_LJ_SIGMA,
    ATOMFIELD_LJ_EPSILON,
    ATOMFIELD_LJ_GAMMA,
    ATOMFIELD_MEDIA_ID,
    # ResKey mapping constants
    CHEM_KIND_BASE_OFFSET_VALUES,
    RES_KIND_BLOCK_SIZE,
    RES_KIND_UNKNOWN,
    RES_KIND_PROTEIN,
    RES_KIND_NUCLEIC,
    RES_KIND_LIPID,
    RES_KIND_CARBOHYDRATE,
)


@jit(["bool_(float32[:])"], nopython=True, cache=True)
def is_atom_res_protein(atom_data):
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


@jit(["int32(float32)"], nopython=True, cache=True)
def get_residue_kind(res_key):
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


@jit(
    [
        "float32(float32[:],float32[:])",
        "float32(float32[:],int32[:])",
        "float32(int32[:],float32[:])",
        "float32(int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def dist_square(a, b):
    """
    Calculate the square of the Euclidean distance between two 3D points.

    This function is optimized for performance using Numba's just-in-time compilation.

    Args:
        a (np.ndarray): A NumPy array of shape (3,) representing the coordinates of the first point.
        b (np.ndarray): A NumPy array of shape (3,) representing the coordinates of the second point.

    Returns:
        float32: The square of the distance between point 'a' and point 'b'.
    """
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2


@jit(
    [
        "float32(float32[:],float32[:])",
        "float32(float32[:],int32[:])",
        "float32(int32[:],float32[:])",
        "float32(int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def distance(a, b):
    """
    Calculate the Euclidean distance between two 3D points.

    This function is optimized for performance using Numba's just-in-time compilation.

    Args:
        a (np.ndarray): A NumPy array of shape (3,) representing the coordinates of the first point.
        b (np.ndarray): A NumPy array of shape (3,) representing the coordinates of the second point.

    Returns:
        float32: The distance between point 'a' and point 'b'.
    """
    d_sq = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2
    return math.sqrt(d_sq)


@jit(["float32(float32[:], int32)"], nopython=True, cache=True)
def get_atom_field(atom_data, field_id):
    """
    Retrieve a specific field value from atom data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.
        field_id (int): An integer representing the index or ID of the desired field
                         in the atom data array, as defined by constants like ATOMFIELD_X, etc.

    Returns:
        float32: The value of the specified atom field.
    """
    return atom_data[field_id]


@jit(["void(float32[:], float32, int32)"], nopython=True, cache=True)
def set_atom_field(atom_data, value, field_id):
    """
    Set a specific field value within the atom data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.
        value (float32): The value to set for the specified field.
        field_id (int): An integer representing the index or ID of the field to be set
                         in the atom data array, as defined by constants like ATOMFIELD_X, etc.
    """
    atom_data[field_id] = value


@jit(["float32(float32[:])"], nopython=True, cache=True)
def get_atom_x(atom_data):
    """
    Get the x-coordinate of an atom from its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The x-coordinate of the atom.
    """
    return atom_data[ATOMFIELD_X]


@jit(["float32(float32[:])"], nopython=True, cache=True)
def get_atom_y(atom_data):
    """
    Get the y-coordinate of an atom from its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The y-coordinate of the atom.
    """
    return atom_data[ATOMFIELD_Y]


@jit(["float32(float32[:])"], nopython=True, cache=True)
def get_atom_z(atom_data):
    """
    Get the z-coordinate of an atom from its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The z-coordinate of the atom.
    """
    return atom_data[ATOMFIELD_Z]


@jit(["float32(float32[:])"], nopython=True, cache=True)
def atom_media_id(atom_data):
    """
    Get the media-id of an atom from its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The media-id  of the atom.
    """
    return atom_data[ATOMFIELD_MEDIA_ID]


@jit(["void(float32[:], float32[:], float32)"], nopython=True, cache=True)
def set_atom_grid_coords(atom_data, grid_origin, h):
    """
    Calculate and set the grid coordinates for an atom based on its real-space coordinates, grid origin, and grid spacing.

    The grid coordinates are calculated as:
        grid_x = (atom_x - grid_origin_x) / h
        grid_y = (atom_y - grid_origin_y) / h
        grid_z = (atom_z - grid_origin_z) / h

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties, where grid coordinates will be stored.
        grid_origin (np.ndarray): A NumPy array of shape (3,) representing the real-space coordinates of the grid origin (0, 0, 0).
        h (float32): The grid spacing, i.e., the real-space distance between adjacent grid lines.
    """
    atom_data[ATOMFIELD_GRID_X] = (atom_data[ATOMFIELD_X] - grid_origin[0]) / h
    atom_data[ATOMFIELD_GRID_Y] = (atom_data[ATOMFIELD_Y] - grid_origin[1]) / h
    atom_data[ATOMFIELD_GRID_Z] = (atom_data[ATOMFIELD_Z] - grid_origin[2]) / h


@jit(["float32[:](float32[:], float32[:], float32)"], nopython=True, cache=True)
def to_grid_coords(crd3d_nparray, grid_origin, h):
    """
    Convert real-space coordinates to grid coordinates.

    Args:
        crd3d_nparray (np.ndarray): A NumPy array of shape (3,) representing real-space coordinates.
        grid_origin (np.ndarray): A NumPy array of shape (3,) representing the real-space coordinates of the grid origin.
        h (float32): The grid spacing.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the corresponding grid coordinates.
    """
    crd3d_grid_crd3d = np.array([0, 0, 0], dtype=np.float32)
    crd3d_grid_crd3d[0] = (crd3d_nparray[0] - grid_origin[0]) / h
    crd3d_grid_crd3d[1] = (crd3d_nparray[1] - grid_origin[1]) / h
    crd3d_grid_crd3d[2] = (crd3d_nparray[2] - grid_origin[2]) / h
    return crd3d_grid_crd3d


@jit(["void(float32[:], float32)"], nopython=True, cache=True)
def set_atom_charge(atom_data, charge):
    """
    Set the charge of an atom in its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.
        charge (float32): The charge value to set for the atom.
    """
    atom_data[ATOMFIELD_CHARGE] = charge


@jit(["void(float32[:], float32)"], nopython=True, cache=True)
def set_atom_radius(atom_data, radius):
    """
    Set the radius of an atom in its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.
        radius (float32): The radius value to set for the atom.
    """
    atom_data[ATOMFIELD_RADIUS] = radius


@jit(["void(float32[:], float32)"], nopython=True, cache=True)
def set_atom_gaussiansigma(atom_data, gaussiansigma):
    """
    Set the Gaussian sigma value of an atom in its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.
        gaussiansigma (float32): The Gaussian sigma value to set for the atom.
    """
    atom_data[ATOMFIELD_GAUSS_SIGMA] = gaussiansigma


@jit(["float32(float32[:])"], nopython=True, cache=True)
def get_atom_charge(atom_data):
    """
    Get the charge of an atom from its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The charge of the atom.
    """
    return atom_data[ATOMFIELD_CHARGE]


@jit(["float32(float32[:])"], nopython=True, cache=True)
def get_atom_radius(atom_data):
    """
    Get the radius of an atom from its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The radius of the atom.
    """
    return atom_data[ATOMFIELD_RADIUS]


@jit(["float32(float32[:])"], nopython=True, cache=True)
def get_atom_gaussiansigma(atom_data):
    """
    Get the Gaussian sigma value of an atom from its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The Gaussian sigma value of the atom.
    """
    return atom_data[ATOMFIELD_GAUSS_SIGMA]


@jit(["float32[:](float32[:])"], nopython=True, cache=True)
def get_atom_coords(atom_data):
    """
    Get the real-space coordinates of an atom from its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the real-space coordinates (x, y, z) of the atom.
    """
    return atom_data[0:3]


@jit(["float32[:](float32[:])"], nopython=True, cache=True)
def get_atom_grid_coords(atom_data):
    """
    Get the grid coordinates of an atom from its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the grid coordinates (grid_x, grid_y, grid_z) of the atom.
    """
    return atom_data[3:6]


@jit(["float32(float32[:])"], nopython=True, cache=True)
def get_atom_grid_x(atom_data):
    """
    Get the x-component of the grid coordinate of an atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The grid x-coordinate of the atom.
    """
    return atom_data[ATOMFIELD_GRID_X]


@jit(["float32(float32[:])"], nopython=True, cache=True)
def get_atom_grid_y(atom_data):
    """
    Get the y-component of the grid coordinate of an atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The grid y-coordinate of the atom.
    """
    return atom_data[ATOMFIELD_GRID_Y]


@jit(["float32(float32[:])"], nopython=True, cache=True)
def get_atom_grid_z(atom_data):
    """
    Get the z-component of the grid coordinate of an atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The grid z-coordinate of the atom.
    """
    return atom_data[ATOMFIELD_GRID_Z]


@jit(["float32(float32[:])"], nopython=True, cache=True)
def get_atom_reskey(atom_data):
    """
    Get the residue key of an atom from its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The residue key of the atom.
    """
    return atom_data[ATOMFIELD_RES_KEY]


@jit(["float32(float32[:])"], nopython=True, cache=True)
def get_atom_lj_sigma(atom_data):
    """
    Get the Lennard-Jones sigma parameter of an atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The Lennard-Jones sigma value of the atom.
    """
    return atom_data[ATOMFIELD_LJ_SIGMA]


@jit(["float32(float32[:])"], nopython=True, cache=True)
def get_atom_lj_epsilon(atom_data):
    """
    Get the Lennard-Jones epsilon parameter of an atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The Lennard-Jones epsilon value of the atom.
    """
    return atom_data[ATOMFIELD_LJ_EPSILON]


@jit(["float32(float32[:])"], nopython=True, cache=True)
def get_atom_vdw_gamma(atom_data):
    """
    Get the van der Waals gamma parameter of an atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        float32: The van der Waals gamma value of the atom.
    """
    return atom_data[ATOMFIELD_LJ_GAMMA]


@jit(["float32(float32[:], int32)"], nopython=True, cache=True)
def get_atom_field(atom_data, atomfield_id):
    """
    Retrieve a specific atom field value by its ID.  Redundant, consider removing in favor of the other get_atom_field.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.
        atomfield_id (int): The ID of the atom field to retrieve (e.g., ATOMFIELD_CHARGE).

    Returns:
        float32: The value of the specified atom field.
    """
    return atom_data[atomfield_id]


@jit(["void(float32[:], float32)"], nopython=True, cache=True)
def set_atom_reskey(atom_data, value):
    """
    Set the residue key for an atom in its data array.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.
        value (float32): The residue key value to set for the atom.
    """
    atom_data[ATOMFIELD_RES_KEY] = value


@jit(["void(float32[:], float32)"], nopython=True, cache=True)
def set_atom_lj_sigma(atom_data, value):
    """
    Set the Lennard-Jones sigma parameter for an atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.
        value (float32): The Lennard-Jones sigma value to set.
    """
    atom_data[ATOMFIELD_LJ_SIGMA] = value


@jit(["void(float32[:], float32)"], nopython=True, cache=True)
def set_atom_lj_epsilon(atom_data, value):
    """
    Set the Lennard-Jones epsilon parameter for an atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.
        value (float32): The Lennard-Jones epsilon value to set.
    """
    atom_data[ATOMFIELD_LJ_EPSILON] = value


@jit(["void(float32[:], float32)"], nopython=True, cache=True)
def set_atom_vdw_gamma(atom_data, value):
    """
    Set the van der Waals gamma parameter for an atom.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.
        value (float32): The van der Waals gamma value to set.
    """
    atom_data[ATOMFIELD_LJ_GAMMA] = value


@jit(["bool_(float32[:])"], nopython=True, cache=True)
def is_atom_hydrogen(atom_data):
    """
    Check if an atom is a hydrogen atom based on its atomic number.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        bool: True if the atom is hydrogen (atomic number is 1.0), False otherwise.
    """
    return atom_data[ATOMFIELD_ATOMIC_NUMBER] == 1.0


@jit(["bool_(float32[:])"], nopython=True, cache=True)
def is_atom_res_protein(atom_data):
    """
    Check if an atom belongs to a protein residue based on its residue key.

    Args:
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        bool: True if the atom's residue key is within the protein residue range (0.0 < res_key < 210.0), False otherwise.
    """
    return 0.0 < atom_data[ATOMFIELD_RES_KEY] < 210.0


@jit(["float32(float32[:])", "float32(int32[:])"], nopython=True, cache=True)
def crd3d_len(crd3d_nparr):
    """
    Calculate the length (magnitude) of a 3D vector.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.

    Returns:
        float32: The length of the 3D vector.
    """
    d_sum = 0.0
    for i in crd3d_nparr:
        d_sum += i * i
    return math.sqrt(d_sum)


@jit(
    [
        "float32[:](float32[:])",
        "int32[:](int32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_neg(crd3d_nparr):
    """
    Calculate the negative of a 3D vector.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the negative of the input vector.
    """
    result = np.copy(crd3d_nparr)
    result *= -1
    return result


@jit(
    [
        "float32[:](float32[:],float32)",
        "float32[:](float32[:],int32)",
        "float32[:](int32[:],float32)",
    ],
    nopython=True,
    cache=True,
)
def crd3d_scalar_add(crd3d_nparr, v):
    """
    Add a scalar value to each component of a 3D vector.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.
        v (float32 or int32): The scalar value to add.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the resulting vector after scalar addition.
    """
    result = np.copy(crd3d_nparr).astype(np.float32)
    result += v
    return result


@jit(
    [
        "int32[:](int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def crd3d_int_scalar_add(crd3d_nparr, v):
    """
    Add a scalar integer value to each component of a 3D integer vector.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D integer vector.
        v (int32): The scalar integer value to add.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the resulting integer vector after scalar addition.
    """
    result = np.copy(crd3d_nparr).astype(np.int32)
    result += v
    return result


@jit(
    [
        "float32[:](float32[:],float32[:])",
        "float32[:](float32[:],int32[:])",
        "float32[:](int32[:],float32[:])",
        "float32[:](int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_vector_add(crd3d_nparr, v):
    """
    Add two 3D vectors component-wise.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D vector.
        v (np.ndarray): A NumPy array of shape (3,) representing the second 3D vector.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the vector sum.
    """
    result = np.copy(crd3d_nparr).astype(np.float32)
    result[0] = crd3d_nparr[0] + v[0]
    result[1] = crd3d_nparr[1] + v[1]
    result[2] = crd3d_nparr[2] + v[2]
    return result


@jit(
    [
        "float32[:](float32[:],float32)",
        "float32[:](float32[:],int32)",
        "float32[:](int32[:],float32)",
        "float32[:](int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def crd3d_scalar_sub(crd3d_nparr, v):
    """
    Subtract a scalar value from each component of a 3D vector.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.
        v (float32 or int32): The scalar value to subtract.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the resulting vector after scalar subtraction.
    """
    result = np.copy(crd3d_nparr).astype(np.float32)
    result -= v
    return result


@jit(
    [
        "int32[:](int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def crd3d_int_scalar_sub(crd3d_nparr, v):
    """
    Subtract a scalar integer value from each component of a 3D integer vector.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D integer vector.
        v (int32): The scalar integer value to subtract.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the resulting integer vector after scalar subtraction.
    """
    result = np.copy(crd3d_nparr).astype(np.int32)
    result -= v
    return result


@jit(
    [
        "float32[:](float32[:],float32[:])",
        "float32[:](float32[:],int32[:])",
        "float32[:](int32[:],float32[:])",
        "float32[:](int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_vector_sub(crd3d_nparr, v):
    """
    Subtract the second 3D vector from the first 3D vector component-wise.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D vector.
        v (np.ndarray): A NumPy array of shape (3,) representing the second 3D vector to subtract.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the vector difference.
    """
    result = np.copy(crd3d_nparr).astype(np.float32)
    result[0] = crd3d_nparr[0] - v[0]
    result[1] = crd3d_nparr[1] - v[1]
    result[2] = crd3d_nparr[2] - v[2]
    return result


@jit(
    [
        "float32[:](float32[:],float32)",
        "float32[:](float32[:],int32)",
        "float32[:](int32[:],float32)",
        "float32[:](int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def crd3d_scalar_mult(crd3d_nparr, v):
    """
    Multiply each component of a 3D vector by a scalar value.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.
        v (float32 or int32): The scalar value to multiply by.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the resulting vector after scalar multiplication.
    """
    result = np.copy(crd3d_nparr).astype(np.float32)
    result *= v
    return result


@jit(
    [
        "float32[:](float32[:],float32[:])",
        "float32[:](float32[:],int32[:])",
        "float32[:](int32[:],float32[:])",
        "float32[:](int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_vector_mult(crd3d_nparr, v):
    """
    Multiply two 3D vectors component-wise.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D vector.
        v (np.ndarray): A NumPy array of shape (3,) representing the second 3D vector.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the component-wise product vector.
    """
    result = np.copy(crd3d_nparr).astype(np.float32)
    result[0] = crd3d_nparr[0] * v[0]
    result[1] = crd3d_nparr[1] * v[1]
    result[2] = crd3d_nparr[2] * v[2]
    return result


@jit(
    [
        "float32[:](float32[:],float32[:])",
        "float32[:](float32[:],int32[:])",
        "float32[:](int32[:],float32[:])",
        "float32[:](int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_cross(crd3d_nparr, v):
    """
    Calculate the cross product of two 3D vectors.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D vector.
        v (np.ndarray): A NumPy array of shape (3,) representing the second 3D vector.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the cross product vector.
    """
    result = np.copy(crd3d_nparr).astype(np.float32)
    result[0] = crd3d_nparr[1] * v[2] - crd3d_nparr[2] * v[1]
    result[1] = crd3d_nparr[2] * v[0] - crd3d_nparr[0] * v[2]
    result[2] = crd3d_nparr[0] * v[1] - crd3d_nparr[1] * v[0]
    return result


@jit(
    [
        "float32(float32[:])",
        "float32(int32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_sum(crd3d_nparr):
    """
    Calculate the sum of the components of a 3D vector.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.

    Returns:
        float32: The sum of the x, y, and z components of the vector.
    """
    result = np.zeros(1, dtype=np.float32)
    result[0] = crd3d_nparr[0] + crd3d_nparr[1] + crd3d_nparr[2]
    return result[0]


@jit(
    [
        "float32[:](float32[:],float32)",
        "float32[:](float32[:],int32)",
        "float32[:](int32[:],float32)",
        "float32[:](int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def crd3d_scalar_div(crd3d_nparr, v):
    """
    Divide each component of a 3D vector by a scalar value.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.
        v (float32 or int32): The scalar value to divide by.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the resulting vector after scalar division.
    """
    result = np.copy(crd3d_nparr).astype(np.float32)
    result /= v
    return result


@jit(
    [
        "float32[:](float32[:],float32[:])",
        "float32[:](float32[:],int32[:])",
        "float32[:](int32[:],float32[:])",
        "float32[:](int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_vector_div(crd3d_nparr, v):
    """
    Divide the first 3D vector by the second 3D vector component-wise.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D vector (numerator).
        v (np.ndarray): A NumPy array of shape (3,) representing the second 3D vector (denominator).

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the component-wise division result vector.
    """
    result = np.copy(crd3d_nparr).astype(np.float32)
    result[0] = crd3d_nparr[0] / v[0]
    result[1] = crd3d_nparr[1] / v[1]
    result[2] = crd3d_nparr[2] / v[2]
    return result


@jit(
    [
        "float32(int32[:],int32[:])",
        "float32(int32[:],float32[:])",
        "float32(float32[:],int32[:])",
        "float32(float32[:],float32[:])",
    ],
    nopython=True,
    cache=True,
)
def dot_product(crd3d_nparr, b):
    """
    Calculate the dot product of two 3D vectors. Optimized version for performance.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D vector.
        b (np.ndarray): A NumPy array of shape (3,) representing the second 3D vector.

    Returns:
        float32: The dot product of the two vectors.
    """
    d_sum = 0.0
    for i, j in zip(crd3d_nparr, b):
        d_sum += i * j
    return d_sum


@jit(
    [
        "boolean(float32[:],float32)",
        "boolean(float32[:],int32)",
        "boolean(int32[:],float32)",
        "boolean(int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def or_ge_scalar(crd3d_nparr, value):
    """
    Check if any component of a 3D vector is greater than or equal to a scalar value (optimized OR-Greater or Equal).

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.
        value (float32 or int32): The scalar value to compare against.

    Returns:
        bool: True if at least one component of the vector is greater than or equal to 'value', False otherwise.
    """
    result = False or crd3d_nparr[0] >= value
    result = result or crd3d_nparr[1] >= value
    result = result or crd3d_nparr[2] >= value
    return result


@jit(
    [
        "boolean(float32[:],float32[:])",
        "boolean(float32[:],int32[:])",
        "boolean(int32[:],float32[:])",
        "boolean(int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def or_ge_vector(crd3d_nparr, value):
    """
    Check if any component of a 3D vector is greater than or equal to the corresponding component of another 3D vector (optimized OR-Greater or Equal - Vector version).

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D vector.
        value (np.ndarray): A NumPy array of shape (3,) representing the second 3D vector to compare against.

    Returns:
        bool: True if for at least one component i, crd3d_nparr[i] is greater than or equal to crd3d_nparr2[i], False otherwise.
    """
    result = (
        crd3d_nparr[0] >= value[0]
        or crd3d_nparr[1] >= value[1]
        or crd3d_nparr[2] >= value[2]
    )
    return result


@jit(
    [
        "boolean(float32[:],float32)",
        "boolean(float32[:],int32)",
        "boolean(int32[:],float32)",
        "boolean(int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def or_le_scalar(crd3d_nparr, value):
    """
    Check if any component of a 3D vector is less than or equal to a scalar value (optimized OR-Less or Equal).

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.
        value (float32 or int32): The scalar value to compare against.

    Returns:
        bool: True if at least one component of the vector is less than or equal to 'value', False otherwise.
    """
    result = False or crd3d_nparr[0] <= value
    result = result or crd3d_nparr[1] <= value
    result = result or crd3d_nparr[2] <= value
    return result


@jit(
    [
        "boolean(float32[:],float32)",
        "boolean(float32[:],int32)",
        "boolean(int32[:],float32)",
        "boolean(int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def or_lt_scalar(crd3d_nparr, value):
    """
    Check if any component of a 3D vector is less than a scalar value (optimized OR-Less Than).

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.
        value (float32 or int32): The scalar value to compare against.

    Returns:
        bool: True if at least one component of the vector is less than 'value', False otherwise.
    """
    result = crd3d_nparr[0] < value
    result = result or crd3d_nparr[1] < value
    result = result or crd3d_nparr[2] < value
    return result


@jit(
    [
        "boolean(float32[:],float32[:])",
        "boolean(float32[:],int32[:])",
        "boolean(int32[:],float32[:])",
        "boolean(int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def or_lt_vector(crd3d_nparr, value):
    """
    Check if any component of a 3D vector is less than the corresponding component of another 3D vector (optimized OR-Less Than - Vector version).

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D vector.
        value (np.ndarray): A NumPy array of shape (3,) representing the second 3D vector to compare against.

    Returns:
        bool: True if for at least one component i, crd3d_nparr[i] is less than crd3d_nparr2[i], False otherwise.
    """
    result = False or crd3d_nparr[0] < value[0]
    result = result or crd3d_nparr[1] < value[1]
    result = result or crd3d_nparr[2] < value[2]
    return result


@jit(
    [
        "boolean(float32[:],float32)",
        "boolean(float32[:],int32)",
        "boolean(int32[:],float32)",
        "boolean(int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def and_lt_scalar(crd3d_nparr, value):
    """
    Check if all components of a 3D vector are less than a scalar value (optimized AND-Less Than).

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.
        value (float32 or int32): The scalar value to compare against.

    Returns:
        bool: True if all components of the vector are less than 'value', False otherwise.
    """
    result = True and crd3d_nparr[0] < value
    result = result and crd3d_nparr[1] < value
    result = result and crd3d_nparr[2] < value
    return result


@jit(
    [
        "boolean(float32[:],float32[:])",
        "boolean(float32[:],int32[:])",
        "boolean(int32[:],float32[:])",
        "boolean(int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def and_lt_vector(crd3d_nparr, value):
    """
    Check if all components of a 3D vector are less than the corresponding components of another 3D vector (optimized AND-Less Than - Vector version).

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D vector.
        value (np.ndarray): A NumPy array of shape (3,) representing the second 3D vector to compare against.

    Returns:
        bool: True if for all components i, crd3d_nparr[i] is less than crd3d_nparr2[i], False otherwise.
    """
    result = True and crd3d_nparr[0] < value[0]
    result = result and crd3d_nparr[1] < value[1]
    result = result and crd3d_nparr[2] < value[2]
    return result


@jit(
    [
        "boolean(float32[:],float32)",
        "boolean(float32[:],int32)",
        "boolean(int32[:],float32)",
        "boolean(int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def and_le_scalar(crd3d_nparr, value):
    """
    Check if all components of a 3D vector are less than or equal to a scalar value (optimized AND-Less or Equal).

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.
        value (float32 or int32): The scalar value to compare against.

    Returns:
        bool: True if all components of the vector are less than or equal to 'value', False otherwise.
    """
    result = True
    result = result and crd3d_nparr[0] <= value
    result = result and crd3d_nparr[1] <= value
    result = result and crd3d_nparr[2] <= value
    return result


@jit(
    [
        "boolean(float32[:],float32[:])",
        "boolean(float32[:],int32[:])",
        "boolean(int32[:],float32[:])",
        "boolean(int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def and_le_vector(crd3d_nparr, value):
    """
    Check if all components of a 3D vector are less than or equal to the corresponding components of another 3D vector (optimized AND-Less or Equal - Vector version).

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D vector.
        value (np.ndarray): A NumPy array of shape (3,) representing the second 3D vector to compare against.

    Returns:
        bool: True if for all components i, crd3d_nparr[i] is less than or equal to crd3d_nparr2[i], False otherwise.
    """
    result = True and crd3d_nparr[0] <= value[0]
    result = result and crd3d_nparr[1] <= value[1]
    result = result and crd3d_nparr[2] <= value[2]
    return result


@jit(
    [
        "boolean(float32[:],float32)",
        "boolean(float32[:],int32)",
        "boolean(int32[:],float32)",
        "boolean(int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def or_gt_scalar(crd3d_nparr, value):
    """
    Check if any component of a 3D vector is greater than a scalar value (optimized OR-Greater Than).

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.
        value (float32 or int32): The scalar value to compare against.

    Returns:
        bool: True if at least one component of the vector is greater than 'value', False otherwise.
    """
    result = False or crd3d_nparr[0] > value
    result = result or crd3d_nparr[1] > value
    result = result or crd3d_nparr[2] > value
    return result


@jit(
    [
        "boolean(float32[:],float32[:])",
        "boolean(float32[:],int32[:])",
        "boolean(int32[:],float32[:])",
        "boolean(int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def or_gt_vector(crd3d_nparr, crd3d_nparr2):
    """
    Check if any component of a 3D vector is greater than the corresponding component of another 3D vector (optimized OR-Greater Than - Vector version). Redundant, consider removing in favor of the other optimized_or_greater_than_vector.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D vector.
        crd3d_nparr2 (np.ndarray): A NumPy array of shape (3,) representing the second 3D vector to compare against.

    Returns:
        bool: True if for at least one component i, crd3d_nparr[i] is greater than crd3d_nparr2[i], False otherwise.
    """
    result = False
    result = result or crd3d_nparr[0] > crd3d_nparr2[0]
    result = result or crd3d_nparr[1] > crd3d_nparr2[1]
    result = result or crd3d_nparr[2] > crd3d_nparr2[2]
    return result


@jit(
    [
        "boolean(float32[:],float32[:])",
        "boolean(float32[:],int32[:])",
        "boolean(int32[:],float32[:])",
        "boolean(int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def or_gt_vector_ref(crd3d_nparr, crd3d_ref):
    """
    Check if any component of a 3D vector is greater than the corresponding component of another 3D vector (optimized OR-Greater Than - Vector version). Redundant function with the same name, consider removing one.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D vector.
        crd3d_ref (np.ndarray): A NumPy array of shape (3,) representing the second 3D vector to compare against.

    Returns:
        bool: True if for at least one component i, crd3d_nparr[i] is greater than crd3d_ref[i], False otherwise.
    """
    result = False
    result = result or crd3d_nparr[0] > crd3d_ref[0]
    result = result or crd3d_nparr[1] > crd3d_ref[1]
    result = result or crd3d_nparr[2] > crd3d_ref[2]
    return result


@jit(
    [
        "boolean(float32[:],float32)",
        "boolean(float32[:],int32)",
        "boolean(int32[:],float32)",
        "boolean(int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def and_gt_scalar(crd3d_nparr, value):
    """
    Check if all components of a 3D vector are greater than a scalar value (optimized AND-Greater Than).

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.
        value (float32 or int32): The scalar value to compare against.

    Returns:
        bool: True if all components of the vector are greater than 'value', False otherwise.
    """
    result = True
    result = result and crd3d_nparr[0] > value
    result = result and crd3d_nparr[1] > value
    result = result and crd3d_nparr[2] > value
    return result


@jit(
    [
        "boolean(float32[:],float32[:])",
        "boolean(float32[:],int32[:])",
        "boolean(int32[:],float32[:])",
        "boolean(int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def and_gt_vector(crd3d_nparr, crd3d_ref):
    """
    Check if all components of a 3D vector are greater than the corresponding components of another 3D vector (optimized AND-Greater Than - Vector version).

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.
        crd3d_ref (np.ndarray): A NumPy array of shape (3,) representing the second 3D vector to compare against.

    Returns:
        bool: True if for all components i, crd3d_nparr[i] is greater than crd3d_ref[i], False otherwise.
    """
    result = True
    result = result and crd3d_nparr[0] > crd3d_ref[0]
    result = result and crd3d_nparr[1] > crd3d_ref[1]
    result = result and crd3d_nparr[2] > crd3d_ref[2]
    return result


@jit(
    [
        "float32(float32[:])",
        "int32(int32[:])",
    ],
    nopython=True,
    cache=True,
)
def minimum_component(crd3d_nparr):
    """
    Find the minimum component value in a 3D vector.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.

    Returns:
        float32 or int32: The minimum value among the x, y, and z components.
    """
    return np.min(crd3d_nparr)


@jit(
    [
        "int32[:](int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def component_min_scalar(crd3d_nparr, value):
    """
    Component-wise minimum of a 3D integer vector and a scalar integer value.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D integer vector.
        value (int32): The scalar integer value to compare with each component.

    Returns:
        np.ndarray: A NumPy array of shape (3,) where each component is the minimum of the original component and 'value'.
    """
    result = np.copy(crd3d_nparr).astype(np.int32)
    result[0] = value if value < result[0] else result[0]
    result[1] = value if value < result[1] else result[1]
    result[2] = value if value < result[2] else result[2]
    return result


@jit(
    [
        "int32[:](int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def component_min_vector(crd3d_nparr, value):
    """
    Component-wise minimum of two 3D integer vectors.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D integer vector.
        value (np.ndarray): A NumPy array of shape (3,) representing the second 3D integer vector to compare against.

    Returns:
        np.ndarray: A NumPy array of shape (3,) where each component is the minimum of the corresponding components from 'crd3d_nparr' and 'value'.
    """
    result = np.copy(crd3d_nparr).astype(np.int32)
    result[0] = value[0] if value[0] < result[0] else result[0]
    result[1] = value[1] if value[1] < result[1] else result[1]
    result[2] = value[2] if value[2] < result[2] else result[2]
    return result


@jit(
    [
        "float32(float32[:])",
        "int32(int32[:])",
    ],
    nopython=True,
    cache=True,
)
def maximum_component(crd3d_nparr):
    """
    Find the maximum component value in a 3D vector.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.

    Returns:
        float32 or int32: The maximum value among the x, y, and z components.
    """
    return np.max(crd3d_nparr)


@jit(
    [
        "int32[:](int32[:],int32)",
    ],
    nopython=True,
    cache=True,
)
def component_max_scalar(crd3d_nparr, value):
    """
    Component-wise maximum of a 3D integer vector and a scalar integer value.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D integer vector.
        value (int32): The scalar integer value to compare with each component.

    Returns:
        np.ndarray: A NumPy array of shape (3,) where each component is the maximum of the original component and 'value'.
    """
    result = np.copy(crd3d_nparr).astype(np.int32)
    result[0] = value if value > result[0] else result[0]
    result[1] = value if value > result[1] else result[1]
    result[2] = value if value > result[2] else result[2]
    return result


@jit(
    [
        "int32[:](int32[:],int32[:])",
    ],
    nopython=True,
    cache=True,
)
def component_max_vector(crd3d_nparr, value):
    """
    Component-wise maximum of two 3D integer vectors.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing the first 3D integer vector.
        value (np.ndarray): A NumPy array of shape (3,) representing the second 3D integer vector to compare against.

    Returns:
        np.ndarray: A NumPy array of shape (3,) where each component is the maximum of the corresponding components from 'crd3d_nparr' and 'value'.
    """
    result = np.copy(crd3d_nparr).astype(np.int32)
    result[0] = value[0] if value[0] > result[0] else result[0]
    result[1] = value[1] if value[1] > result[1] else result[1]
    result[2] = value[2] if value[2] > result[2] else result[2]
    return result


@jit(
    [
        "float32[:](float32[:])",
        "float32[:](int32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_to_float(crd3d_nparr):
    """
    Convert a 3D vector array to float32 type.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the same vector, but with float32 data type.
    """
    c = crd3d_nparr.astype(np.float32)
    return c


@jit(
    [
        "int32[:](float32[:])",
        "int32[:](int32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_to_int(crd3d_nparr):
    """
    Convert a 3D vector array to int32 type.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.

    Returns:
        np.ndarray: A NumPy array of shape (3,) representing the same vector, but with int32 data type.
    """
    c = crd3d_nparr.astype(np.int32)
    return c


@jit(
    [
        "float32[:](float32[:])",
        "int32[:](int32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_closest(crd3d_nparr):
    """
    Round each component of a 3D vector to the nearest integer.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.

    Returns:
        np.ndarray: A NumPy array of shape (3,) where each component is rounded to the nearest integer.
    """
    res = np.copy(crd3d_nparr)
    for i in range(3):
        if crd3d_nparr[i] > 0.0:
            res[i] = int(crd3d_nparr[i] + 0.5)
        else:
            res[i] = int(crd3d_nparr[i] - 0.5)
    return res


@jit(
    [
        "float32[:](float32[:])",
        "int32[:](int32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_ceil(crd3d_nparr):
    """
    Calculate the ceiling of each component of a 3D vector.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.

    Returns:
        np.ndarray: A NumPy array of shape (3,) where each component is the ceiling of the original component.
    """
    res = np.copy(crd3d_nparr)
    for i in range(3):
        res[i] = math.ceil(crd3d_nparr[i])
    return res


@jit(
    [
        "float32[:](float32[:])",
        "int32[:](int32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_floor(crd3d_nparr):
    """
    Calculate the floor of each component of a 3D vector.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing a 3D vector.

    Returns:
        np.ndarray: A NumPy array of shape (3,) where each component is the floor of the original component.
    """
    res = np.copy(crd3d_nparr)
    for i in range(3):
        res[i] = math.floor(crd3d_nparr[i])
    return res


@jit(
    [
        "float32(float32[:],float32[:],float32[:])",
        "float32(float32[:],float32[:],int32[:])",
        "float32(float32[:],int32[:],float32[:])",
        "float32(int32[:],float32[:],float32[:])",
        "float32(float32[:],int32[:],int32[:])",
        "float32(int32[:],int32[:],float32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_grid_distance_factor(crd3d_grid_coords, neighbor_grid_index, neighbor_offset):
    """Calculates a trilinear interpolation distance factor.

    This function computes a distance factor used in trilinear interpolation for neighboring grid points.
    It's based on the difference between a given coordinate and the integer index of a neighboring grid point,
    taking into account the neighbor's offset within its cell.

    Args:
        crd3d_grid_coords (float32[:]): The 3D coordinates of a point within the grid.
        neighbor_grid_index (float32[:] or int32[:]): The 3D integer indices of a neighboring grid point.
        neighbor_offset (float32[:] or int32[:]): The 3D offset of the neighbor relative to the
            floor of its grid cell.  Represents one vertex of the unit cube.

    Returns:
        float32: The calculated distance factor used in trilinear interpolation.
    """
    d_diff = 1.0
    for i in range(3):
        d_diff *= (
            neighbor_grid_index[i]
            - neighbor_offset[i]
            - crd3d_grid_coords[i]
            + 1
            - neighbor_offset[i]
        )
    d_diff = np.abs(d_diff)
    return d_diff


@jit(
    [
        "float32[:,:](float32[:])",
        "float32[:,:](int32[:])",
    ],
    nopython=True,
    cache=True,
)
def crd3d_grid_neighbors(crd3d_grid_coords):
    """Generates neighbor grid points and their distance factors.

    This function identifies the 8 neighboring grid points for a given 3D coordinate and calculates
    their corresponding distance factors used in trilinear interpolation.

    Args:
        crd3d_grid_coords (float32[:] or int32[:]): The 3D coordinates of a point within the grid.

    Returns:
        float32[:,:]: A NumPy array of shape (8, 4). Each row represents a neighbor:
                      - Columns 0-2: Integer grid indices (x, y, z).
                      - Column 3: The distance factor for trilinear interpolation.
    """
    floor_coords = crd3d_to_int(
        crd3d_floor(crd3d_grid_coords)
    )  # Assuming crd3d_to_int and crd3d_floor are defined
    points = np.zeros((8, 4), dtype=np.float32)

    for i in range(2):
        for j in range(2):
            for k in range(2):
                p_index = int(i * 4 + j * 2 + k)
                points[p_index][0] = float(floor_coords[0] + i)
                points[p_index][1] = float(floor_coords[1] + j)
                points[p_index][2] = float(floor_coords[2] + k)
                points[p_index][3] = crd3d_grid_distance_factor(
                    crd3d_grid_coords,
                    points[p_index][0:3].astype(np.int32),
                    np.array([i, j, k], dtype=np.float32),
                )

    return points
