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
This module defines an enumeration for chemical elements and provides utility functions
for working with atomic numbers.

It includes:
- `ConstChemElement`: An enumeration of known chemical elements, mapping their symbols
  to their atomic numbers (represented as float64 for compatibility with data structures).
- `AtomicNumToElement`: A reverse mapping (dictionary) for efficient lookup of
  `ConstChemElement` members by their atomic number.
- `get_element_by_atomic_number`: A function to retrieve a `ConstChemElement` member
  given an atomic number.
"""

from enum import Enum


class ConstChemElement(Enum):
    """
    Enumeration of known chemical elements.

    Source: IUPAC Periodic Table (May 4, 2022).
    URL: https://www.degruyter.com/document/doi/10.1515/pac-2019-0603/html

    Each element is represented by its atomic number (float64, although integers,
    using float allows storage in atom_data without type conversion).

    Members are element symbols mapped to their atomic numbers.
    """

    UNK = 0.0  # UNKNOWN (dummy atom/element, not in Periodic table)
    H = 1.0  # hydrogen
    He = 2.0  # helium
    Li = 3.0  # lithium
    Be = 4.0  # beryllium
    B = 5.0  # boron
    C = 6.0  # carbon
    N = 7.0  # nitrogen
    O = 8.0  # oxygen
    F = 9.0  # fluorine
    Ne = 10.0  # neon
    Na = 11.0  # sodium
    Mg = 12.0  # magnesium
    Al = 13.0  # aluminium
    Si = 14.0  # silicon
    P = 15.0  # phosphorus
    S = 16.0  # sulfur
    Cl = 17.0  # chlorine
    Ar = 18.0  # argon
    K = 19.0  # potassium
    Ca = 20.0  # calcium
    Sc = 21.0  # scandium
    Ti = 22.0  # titanium
    V = 23.0  # vanadium
    Cr = 24.0  # chromium
    Mn = 25.0  # manganese
    Fe = 26.0  # iron
    Co = 27.0  # cobalt
    Ni = 28.0  # nickel
    Cu = 29.0  # copper
    Zn = 30.0  # zinc
    Ga = 31.0  # gallium
    Ge = 32.0  # germanium
    As = 33.0  # arsenic
    Se = 34.0  # selenium
    Br = 35.0  # bromine
    Kr = 36.0  # krypton
    Rb = 37.0  # rubidium
    Sr = 38.0  # strontium
    Y = 39.0  # yttrium
    Zr = 40.0  # zirconium
    Nb = 41.0  # niobium
    Mo = 42.0  # molybdenum
    Tc = 43.0  # technetiuma
    Ru = 44.0  # ruthenium
    Rh = 45.0  # rhodium
    Pd = 46.0  # palladium
    Ag = 47.0  # silver
    Cd = 48.0  # cadmium
    In = 49.0  # indium
    Sn = 50.0  # tin
    Sb = 51.0  # antimony
    Te = 52.0  # tellurium
    I = 53.0  # iodine
    Xe = 54.0  # xenon
    Cs = 55.0  # caesium
    Ba = 56.0  # barium
    La = 57.0  # lanthanum
    Ce = 58.0  # cerium
    Pr = 59.0  # praseodymium
    Nd = 60.0  # neodymium
    Pm = 61.0  # promethiuma
    Sm = 62.0  # samarium
    Eu = 63.0  # europium
    Gd = 64.0  # gadolinium
    Tb = 65.0  # terbium
    Dy = 66.0  # dysprosium
    Ho = 67.0  # holmium
    Er = 68.0  # erbium
    Tm = 69.0  # thulium
    Yb = 70.0  # ytterbium
    Lu = 71.0  # lutetium
    Hf = 72.0  # hafnium
    Ta = 73.0  # tantalum
    W = 74.0  # tungsten
    Re = 75.0  # rhenium
    Os = 76.0  # osmium
    Ir = 77.0  # iridium
    Pt = 78.0  # platinum
    Au = 79.0  # gold
    Hg = 80.0  # mercury
    Tl = 81.0  # thallium
    Pb = 82.0  # lead
    Bi = 83.0  # bismutha
    Po = 84.0  # poloniuma
    At = 85.0  # astatinea
    Rn = 86.0  # radona
    Fr = 87.0  # franciuma
    Ra = 88.0  # radiuma
    Ac = 89.0  # actiniuma
    Th = 90.0  # thoriuma
    Pa = 91.0  # protactiniuma
    U = 92.0  # uraniuma
    Np = 93.0  # neptuniuma
    Pu = 94.0  # plutoniuma
    Am = 95.0  # americiuma
    Cm = 96.0  # curiuma
    Bk = 97.0  # berkeliuma
    Cf = 98.0  # californiuma
    Es = 99.0  # einsteiniuma
    Fm = 100.0  # fermiuma
    Md = 101.0  # mendeleviuma
    No = 102.0  # nobeliuma
    Lr = 103.0  # lawrenciuma
    Rf = 104.0  # rutherfordiuma
    Db = 105.0  # dubniuma
    Sg = 106.0  # seaborgiuma
    Bh = 107.0  # bohriuma
    Hs = 108.0  # hassiuma
    Mt = 109.0  # meitneriuma
    Ds = 110.0  # darmstadtiuma
    Rg = 111.0  # roentgeniuma
    Cn = 112.0  # coperniciuma
    Nh = 113.0  # nihoniuma
    Fl = 114.0  # fleroviuma
    Mc = 115.0  # moscoviuma
    Lv = 116.0  # livermoriuma
    Ts = 117.0  # tennessinea
    Og = 118.0  # oganessona

    @classmethod
    def has_member(cls, key):
        """Check if the enumeration has a member with the given key."""
        return key in cls.__members__


# Create a reverse mapping for atomic number to element for efficient lookup
AtomicNumToElement = {elm.value: elm for elm in ConstChemElement}


def get_element_by_symbol(symbol: str) -> ConstChemElement:
    """
    Get the ConstChemElement enum member from a chemical element symbol string.

    Args:
        symbol (str): The element symbol (e.g., 'C', 'Cl', 'Fe').

    Returns:
        ConstChemElement: The corresponding enum member (e.g., ConstChemElement.C).
    """
    symbol = symbol.capitalize()
    if ConstChemElement.has_member(symbol):
        return ConstChemElement[symbol]
    return ConstChemElement.UNK


def get_element_by_atomic_number(atomic_number: float) -> ConstChemElement:
    """
    Retrieves the chemical element Enum member for a given atomic number.

    This function uses the `AtomicNumToElement` dictionary for efficient lookup.

    Parameters
    ----------
    atomic_number : float
        The atomic number of the element as a float (matches the type in atom_data).
        This value is expected to be a float representation of an integer.

    Returns
    -------
    ConstChemElement
        The corresponding `ConstChemElement` Enum member.value.

    Raises
    ------
    KeyError
        If the provided atomic number, after rounding and conversion to float,
        does not correspond to a known element in `ConstChemElement`.
    """
    # Convert atomic_number to the nearest integer and then to float for dictionary lookup.
    # Rounding is used to handle potential floating-point inaccuracies in the input.
    int_atomic_number = int(round(atomic_number, 0))
    # The lookup dictionary keys are floats (e.g., 1.0, 6.0), so we cast the integer back to float.
    if float(int_atomic_number) in AtomicNumToElement:
        return AtomicNumToElement[float(int_atomic_number)].value
    else:
        return ConstChemElement.UNK.value


def get_element_symbol_by_atomic_number(atomic_number: float) -> ConstChemElement:
    """
    Retrieves the chemical element Enum member for a given atomic number.

    This function uses the `AtomicNumToElement` dictionary for efficient lookup.

    Parameters
    ----------
    atomic_number : float
        The atomic number of the element as a float (matches the type in atom_data).
        This value is expected to be a float representation of an integer.

    Returns
    -------
    ConstChemElement
        The corresponding `ConstChemElement` Enum member.name.

    Raises
    ------
    KeyError
        If the provided atomic number, after rounding and conversion to float,
        does not correspond to a known element in `ConstChemElement`.
    """
    # Convert atomic_number to the nearest integer and then to float for dictionary lookup.
    # Rounding is used to handle potential floating-point inaccuracies in the input.
    int_atomic_number = int(round(atomic_number, 0))
    # The lookup dictionary keys are floats (e.g., 1.0, 6.0), so we cast the integer back to float.
    if float(int_atomic_number) in AtomicNumToElement:
        return AtomicNumToElement[float(int_atomic_number)].name
    else:
        return ConstChemElement.UNK.name
