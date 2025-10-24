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
This module defines an enumeration `ConstPhysical` containing commonly used
physical and universal constants in Delphi. All constants are represented
with double-precision (float64) for accuracy in calculations.
"""

from enum import Enum


class ConstPhysical(Enum):
    """
    Physical and Universal Constants used in Delphi (double precision).

    This enumeration provides commonly used physical and universal constants
    with double-precision (float64) representation for Delphi calculations.

    Constants:
        Pi (float): Ratio of circle's circumference to its diameter.
        FourPi (float): 4 * Pi.
        Sixth (float): 1.0 / 6.0.
        AbsoluteZero (float): Absolute zero temperature in Kelvin.
        AtomicUnitCrg (float): Charge of a proton in Coulomb.
        BoltzmannConstant (float): Boltzmann constant (J/K).
        VacuumPermittivity (float): Vacuum permittivity (F/m).
        EPK (float): Energy conversion factor to kT/e units.
                     Specifically, e^2 / (4*pi*epsilon_0*k*1.0e-10).
        AvogadroNumber (float): Avogadro's constant (mol^-1).
        Calories2Joules (float): Conversion factor from calories to Joules.
        DebyConstant (float): Debye constant in Delphi units.
        BohrFactor (float): Bohr radius factor in angstrom.
    """

    Pi = 3.141592653589793
    FourPi = 1.256637061435917e1
    """4 times Pi."""
    Sixth = 1.666666666666667e-1
    """One-sixth (1/6)."""
    AbsoluteZero = -273.15
    AtomicUnitCrg = 1.602176487e-19
    BoltzmannConstant = 1.3806504e-23
    VacuumPermittivity = 8.8541878176e-12
    EPK = 167100.9162872952
    AvogadroNumber = 6.022140857e23
    Calories2Joules = 4.184
    DebyConstant = 0.01990076478
    BohrFactor = 0.5291772108
    ReactionFieldEnergyFactor = 0.9549296586
