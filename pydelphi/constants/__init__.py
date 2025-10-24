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
This module defines various constants used in the pyDelphi program.

It re-exports enumerations for:
    - AtomFields:         Fields within the atom_data array.
    - ConstPhysical:      Physical and universal constants (double precision).
    - ConstDelPhiFloats:  Delphi numerical constants (double precision).
    - ConstDelPhiInts:    Delphi integer constants.
    - ConstChemElement:   Known chemical elements.

It also re-exports:
    - NEIGHBOR_VOXEL_RELATIVE_COORDINATES: Voxel coordinates for neighbor calculations.
    - ATOMFIELD_*: Constants for efficient atom field access.
    - LEN_ATOMFIELDS: Total number of atom fields.
    - MAX_KERNEL_SHARED_MEM_THREADS: Constant for CUDA kernel shared memory.
    - GRID_POINT_NEIGHBOR_COUNT, XYZ_COMPONENTS, HALF_GRID_OFFSET_LAGGING, HALF_GRID_OFFSET_LEADING: General grid constants.
    - GPCHRGFIELD_*: Constants for grid point charge fields.
    - GRID_NEIGHBOR_OFFSETS: Constant for neighbor offsets.
    - NUM_DIMENSIONS: Constant for number of dimensions.
    - CCOI_*: Constants for circle-of-intersection in SAS.
    - AtomicNumToElement: Dictionary mapping atomic number to element.

The following are placeholders for future "Part 2 logic" related to residues and polymers:
    - ResNameProtein:      Dictionary of standard protein residue names and keys (3-letter to 1-letter).
    - ResNameToResKey:     Dictionary mapping residue names to unique integer keys based on kind and variant.
    - ResKeyToResName:     Dictionary mapping unique integer keys back to residue names.
    - get_element_by_atomic_number: Function to retrieve element by atomic number.

The residue keying scheme follows a structured approach based on biomolecule kind,
base residue type, and variant status, enabling efficient lookup and classification
using integer arithmetic.
"""

# Re-export constants from physical.py
from .physical import ConstPhysical

# Re-export constants from application.py
from .application import (
    NEIGHBOR_VOXEL_RELATIVE_COORDINATES,
    AtomFields,
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_CRD_END,
    ATOMFIELD_GRID_X,
    ATOMFIELD_GRID_Y,
    ATOMFIELD_GRID_Z,
    ATOMFIELD_GRID_END,
    ATOMFIELD_CHARGE,
    ATOMFIELD_RADIUS,
    ATOMFIELD_GAUSS_SIGMA,
    ATOMFIELD_RES_KEY,
    ATOMFIELD_ATOMIC_NUMBER,
    ATOMFIELD_LJ_SIGMA,
    ATOMFIELD_LJ_EPSILON,
    ATOMFIELD_LJ_GAMMA,
    ATOMFIELD_MEDIA_ID,
    LEN_ATOMFIELDS,
    MAX_KERNEL_SHARED_MEM_THREADS,
    GRID_POINT_NEIGHBOR_COUNT,
    XYZ_COMPONENTS,
    HALF_GRID_OFFSET_LAGGING,
    HALF_GRID_OFFSET_LEADING,
    GPCHRGFIELD_INDX_1D,
    GPCHRGFIELD_CHARGE,
    GPCHRGFIELD_INDX_X,
    GPCHRGFIELD_INDX_Y,
    GPCHRGFIELD_INDX_Z,
    GRID_NEIGHBOR_OFFSETS,
    NUM_DIMENSIONS,
    LJCUTOFF_MIN,
    LJCUTOFF_MAX,
    CCOI_X,
    CCOI_Y,
    CCOI_Z,
    CCOI_CRD_END,
    CCOI_RADIANS,
    CCOI_COUNT,
    CCOI_FIELDS_COUNT,
    ConstDelPhiFloats,
    ConstDelPhiInts,
    BOX_BOUNDARY,
    BOX_INTERIOR,
    BOX_HOMO_EPSILON,
)

# Re-export constants from elements.py
from .elements import (
    ConstChemElement,
    AtomicNumToElement,
    get_element_by_symbol,
    get_element_by_atomic_number,
    get_element_symbol_by_atomic_number,
)


# Re-export constants from residues.py
from .residues import (
    CHEM_KIND_BASE_OFFSET_VALUES,
    RES_KIND_BLOCK_SIZE,
    RES_KIND_UNKNOWN,
    RES_KIND_PROTEIN,
    RES_KIND_NUCLEIC,
    RES_KIND_LIPID,
    RES_KIND_CARBOHYDRATE,
    ResNameProtein,
    ResNameToResKey,
    ResKeyToResName,
)
