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
This module provides precision independent utility functions for atom data representation.
"""

import numpy as np

from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_CHARGE,
    ATOMFIELD_RADIUS,
    ATOMFIELD_GAUSS_SIGMA,
)


def atom_repr(atom_key, atom_data):
    """
    Generate a string representation of atom data for logging or printing.

    Args:
        atom_key (tuple): A tuple containing atom identification information (e.g., atom name, index).
        atom_data (np.ndarray): A NumPy array containing atom properties.

    Returns:
        str: A formatted string representing the atom's key and selected properties.
    """
    ostr = f"|{atom_key[0]:20s}; {atom_data[ATOMFIELD_X]:8.3f}{atom_data[ATOMFIELD_Y]:8.3f}{atom_data[ATOMFIELD_Z]:8.3f};"
    ostr += f" {atom_data[ATOMFIELD_CHARGE]:8.4f}; {atom_data[ATOMFIELD_RADIUS]:8.4f}; {atom_data[ATOMFIELD_GAUSS_SIGMA]:6.2f}|\n"
    return ostr


def crd3d_to_str(crd3d_nparr):
    """
    Convert a 3D coordinate NumPy array to a string representation.

    Args:
        crd3d_nparr (np.ndarray): A NumPy array of shape (3,) representing 3D coordinates.

    Returns:
        str: A string in the format "(x, y, z)" representing the coordinates.
    """
    return f"({crd3d_nparr[0]}, {crd3d_nparr[1]}, {crd3d_nparr[2]})"
