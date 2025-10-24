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
This module provides a Numba-optimized function to calculate the electrostatic
energy contribution from charged grid points within a given phase.

The primary function `calc_grid_energy` computes the sum of (charge * potential)
for each charged grid point, then applies a factor of 0.5. This is typically
used to calculate the self-energy of the grid charges in a particular medium
(e.g., vacuum or solvent).
"""

from numba import njit
import numpy as np

# GPCHRG: Constants for grid point charge fields
from pydelphi.constants import (
    # Index of the 1D grid point index
    GPCHRGFIELD_CHARGE,  # Index of the charge at the grid point
    GPCHRGFIELD_INDX_X,  # Index of the x-coordinate of the grid point
    GPCHRGFIELD_INDX_Y,  # Index of the y-coordinate of the grid point
    GPCHRGFIELD_INDX_Z,  # Index of the z-coordinate of the grid point
)


@njit(nogil=True, boundscheck=False, cache=True)
def calc_grid_energy(uniq_index_sorted_aggr_charge_grids, phimap_in_phase):
    """
    Calculates the energy of charged grid points within a phase given the charged grid points and the potential map.

    Args:
        uniq_index_sorted_aggr_charge_grids (np.ndarray): A 2D array where each row represents a charged grid
                                                            point [index_1d, acc_charge, ix, iy, iz].
        phimap_in_phase (np.ndarray): A 3D array representing the electrostatic potential map in the phase.

    Returns:
        float: The total energy of the charged grid points in the phase.
    """
    tmp_energy_phase_sum = 0.0
    phi_xyz = 0.0
    for i in range(len(uniq_index_sorted_aggr_charge_grids)):
        qi = uniq_index_sorted_aggr_charge_grids[i]
        # Extract ix, iy, iz, and charge using constants
        ix = int(qi[GPCHRGFIELD_INDX_X])
        iy = int(qi[GPCHRGFIELD_INDX_Y])
        iz = int(qi[GPCHRGFIELD_INDX_Z])
        charge = qi[GPCHRGFIELD_CHARGE]

        phi_xyz = phimap_in_phase[ix][iy][iz]
        tmp_energy_phase_sum += charge * phi_xyz
    tmp_energy_phase_sum *= 0.5  # 1/2 * qi*phi[i]
    return tmp_energy_phase_sum
