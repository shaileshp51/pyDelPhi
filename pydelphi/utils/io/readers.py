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


"""
This module provides functions for reading various file formats commonly used
in molecular modeling and bioinformatics, particularly within the Delphi software
suite. It includes readers for:

- Gaussian cube files (.cube, .phi): Scalar data grids for visualizing properties
  like electrostatic potential. Supports both text and binary formats with
  endianness and marker size auto-detection for binary files.
- Protein Data Bank files (.pdb): Atomic coordinates and basic structural information.
- PQR files (.pqr): PDB files with added charge and radius information for each atom.
- Size configuration files (.siz): Custom atom size definitions.
- Charge configuration files (.crg): Custom atom charge definitions.
- Van der Waals parameter files (.vdw): (To be implemented)
- Sigma parameter files (.sigma): (To be implemented)
- Force center files (.frc): Atomic coordinate files used for force field calculations.

These functions are designed to parse specific file formats and return structured
data, often in the form of dictionaries or numpy arrays, suitable for further
processing within the pydelphi package or other scientific applications.
"""

import numpy as np

from pydelphi.constants import (
    ConstDelPhiInts,
)
from pydelphi.config.logging_config import (
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

RES_NUMBER_UNKNOWN = ConstDelPhiInts.ResidueNumberUnknown.value


from pydelphi.utils.io.format.assorted.custom_reader import (
    read_crg,
    read_siz,
    read_vdw,
    read_zphi,
    read_grid_charges,
    read_gaussian_sigma,
    read_induced_surface_charges,
    compare_zphi,
)

from pydelphi.utils.io.format.pdb_pqr import read_pdb, read_pqr
from pydelphi.utils.io.format.cube.cube_io import read_cube
from pydelphi.utils.io.format.assorted.frc_reader import (
    read_frc,
    calculate_center_of_frc_atoms,
)
