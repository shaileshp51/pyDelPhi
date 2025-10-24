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

import numpy as np

from pydelphi.config.global_runtime import (
    delphi_real,
    vprint,
)
from pydelphi.constants import (
    LEN_ATOMFIELDS,
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_GRID_X,
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
    ResNameToResKey,
    get_element_by_symbol,
    get_element_symbol_by_atomic_number,
    ConstChemElement,
    ConstDelPhiInts,
)
from pydelphi.config.logging_config import (
    ERROR,
    WARNING,
    TRACE,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

RES_NUMBER_UNKNOWN = ConstDelPhiInts.ResidueNumberUnknown


def _guess_element_from_atom_name(atom_name: str) -> str:
    """
    Guess the chemical element symbol from an atom name string, commonly found
    in PDB/PQR files.

    This function attempts to infer the correct chemical element symbol based on
    standard naming conventions and element symbol capitalization rules.

    Steps:
    1. Remove all digit characters from the input atom name.
       For example, 'H1' → 'H', 'HG2' → 'HG'.
    2. If the resulting string is empty, return 'UNK' (unknown element).
    3. If the string length is 1, return the uppercase version as the element symbol.
       Example: 'C' → 'C'.
    4. If the string length is 2 or more:
       - Check if the second character is lowercase.
         - If yes, try to match the first two characters as a known chemical element symbol
           (e.g., 'Ca' for calcium, 'He' for helium).
         - If the two-character match is successful (found in `ConstChemElement`), return it.
       - Otherwise, fallback to the first character uppercase only.
         This handles cases like 'CA' (C-alpha carbon, not calcium) which should be
         interpreted as 'C'.

    This logic respects chemical element symbol conventions where element symbols are
    either one uppercase letter (e.g., 'C', 'H') or one uppercase followed by one lowercase
    letter (e.g., 'Ca', 'Fe').

    Parameters
    ----------
    atom_name : str
        The atom name string from which to guess the element symbol.

    Returns
    -------
    str
        The guessed chemical element symbol, or 'UNK' if unknown.

    Examples
    --------
    >>> _guess_element_from_atom_name('H1')
    'H'
    >>> _guess_element_from_atom_name('Ca')
    'Ca'
    >>> _guess_element_from_atom_name('CA')
    'C'
    >>> _guess_element_from_atom_name('CH3')
    'C'
    >>> _guess_element_from_atom_name('')
    'UNK'
    """
    element_name = "UNK"
    # Remove digits by filtering characters
    name_no_digits = "".join(ch for ch in atom_name if not ch.isdigit()).strip()

    if not name_no_digits:
        element_name = "UNK"
    elif len(name_no_digits) == 1:
        # Length 1: simply uppercase
        element_name = name_no_digits.upper()
    else:
        # Length 2 or more:
        first_char = name_no_digits[0].upper()
        second_char = name_no_digits[1]
        if second_char.islower() and ConstChemElement.has_member(
            first_char + second_char
        ):
            element_name = first_char + second_char
        else:
            # Fallback to first char uppercase
            element_name = first_char

    vprint(
        TRACE, _VERBOSITY, f"TRACE>> atom_name='{atom_name}' → element='{element_name}'"
    )
    return element_name


def _get_element_symbol(line: str) -> str:
    """
    Extracts the element symbol from a PDB or PQR line.
    If element column (77–78) is present and non-empty, use it.
    Otherwise, infer from the atom name.

    Args:
        line (str): A line from a PDB or PQR file.

    Returns:
        str: The guessed or extracted element symbol, e.g., 'C', 'Fe', 'Cl'.
    """
    # Try to read element column (77–78)
    if len(line) >= 78:
        elem_field = line[76:78].strip()
        if elem_field:
            return elem_field.capitalize()

    # Fallback: guess from atom name field (columns 13–16, 0-based: 12–16)
    atom_name = line[12:16]
    return _guess_element_from_atom_name(atom_name)


def read_pdb(filename):
    """
    Reads a PDB (Protein Data Bank) file and extracts atomic data.

    Parses atomic records (ATOM or HETATM) from the PDB file and stores them
    in a dictionary. The dictionary keys are tuples containing atom and residue
    identification information, and values are numpy arrays representing
    various atomic properties.

    Args:
        filename (str): The path to the PDB file.

    Returns:
        tuple: A tuple containing:
            - atoms (dict): A dictionary of atomic data.
              Keys are tuples of the format:
              (atom number, atom ID, atom name, residue name, chain, residue number).
              Values are numpy arrays with the following structure:
                - [0]: x-coordinate (Å)
                - [1]: y-coordinate (Å)
                - [2]: z-coordinate (Å)
                - [3-5]: Grid coordinates (default 0.0)
                - [6]: Charge (default 0.0)
                - [7]: Radius (default 1.08 Å, hydrogen radius)
                - [8]: Gaussian sigma (default 1.0)
                - [9]: Residue key (integer, from `ResNameToResKey` mapping)
                - [10]: Element atomic number (integer, from `ConstChemElement` enum)
                - [11]: LJ-sigma (default 0.0)
                - [12]: LJ-epsilon (default 0.0)
                - [13]: van der Waals gamma (default 0.0)
                - [14]: Object media number (default 1.0)
            - objects (list): A list of strings, currently containing molecule
              object data like ["is a molecule  0", " "].
    """
    from pydelphi.config.global_runtime import delphi_real

    object_media_number = 1.0
    atoms = {}
    objects = ["is a molecule  0", " "]

    with open(filename) as fin:
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue

            record = ln[0:6].upper().strip()
            if record in ("ATOM", "HETATM"):
                atomnum = ln[6:11].strip()
                atomname = ln[12:16].strip()
                atom_elm = ln[12:14].strip()
                atomid = ln[11:26]
                resname = ln[17:20].strip()
                chain = ln[21:22].strip()
                resnum = ln[22:26].strip()

                # Convert residue number to integer or set as unknown
                resnum = int(resnum) if resnum else RES_NUMBER_UNKNOWN

                # Create the key for the atom
                atom_key = (record, atomnum, atomid, atomname, resname, chain, resnum)

                # Initialize atomic data array
                atom_data = np.zeros(LEN_ATOMFIELDS, dtype=delphi_real)
                atom_data[ATOMFIELD_X] = delphi_real(ln[30:38].strip())  # x-coordinate
                atom_data[ATOMFIELD_Y] = delphi_real(ln[38:46].strip())  # y-coordinate
                atom_data[ATOMFIELD_Z] = delphi_real(ln[46:54].strip())  # z-coordinate
                atom_data[ATOMFIELD_GRID_X:ATOMFIELD_GRID_END] = [
                    0.0,
                    0.0,
                    0.0,
                ]  # Grid coordinates (x, y, z)
                atom_data[ATOMFIELD_CHARGE] = delphi_real(0.0)  # Default charge
                atom_data[ATOMFIELD_RADIUS] = delphi_real(
                    1.08
                )  # Default radius (for H)
                atom_data[ATOMFIELD_GAUSS_SIGMA] = delphi_real(
                    1.0
                )  # Gaussian sigma (default)

                # Residue key lookup
                atom_data[ATOMFIELD_RES_KEY] = ResNameToResKey.get(
                    resname.upper(), ResNameToResKey["UNK"]
                )

                # Element atomic number
                element_symbol = _get_element_symbol(ln)
                element_enum = get_element_by_symbol(element_symbol)
                atom_data[ATOMFIELD_ATOMIC_NUMBER] = element_enum.value

                # Additional properties (default values)
                atom_data[ATOMFIELD_LJ_SIGMA] = 0.0  # LJ-sigma
                atom_data[ATOMFIELD_LJ_EPSILON] = 0.0  # LJ-epsilon
                atom_data[ATOMFIELD_LJ_GAMMA] = 0.0  # van der Waals gamma
                atom_data[ATOMFIELD_MEDIA_ID] = (
                    object_media_number  # Object media number
                )

                # Store the atom data in the dictionary
                atoms[atom_key] = atom_data

    return atoms, objects


def read_pqr(filename):
    """
    Reads a PQR (Protein Data Bank with Charges and Radii) file and extracts atomic data.

    Similar to read_pdb, but additionally parses charge and radius information
    directly from the PQR file format.

    Args:
        filename (str): The path to the PQR file.

    Returns:
        tuple: A tuple containing:
            - atoms (dict): A dictionary of atomic data.
              Keys are tuples of the format:
              (atom number, atom ID, atom name, residue name, chain, residue number).
              Values are numpy arrays with the following structure:
                - [0]: x-coordinate (Å)
                - [1]: y-coordinate (Å)
                - [2]: z-coordinate (Å)
                - [3-5]: Grid coordinates (default 0.0)
                - [6]: Charge (parsed from PQR file)
                - [7]: Radius (parsed from PQR file)
                - [8]: Gaussian sigma (default 1.0)
                - [9]: Residue key (integer, from `ResNameToResKey` mapping)
                - [10]: Element atomic number (integer, from `ConstChemElement` enum)
                - [11]: LJ-sigma (default 0.0)
                - [12]: LJ-epsilon (default 0.0)
                - [13]: van der Waals gamma (default 0.0)
                - [14]: Object media number (default 1.0)
            - objects (list): A list of strings, currently containing molecule
              object data like ["is a molecule  0", " "].
    """
    from pydelphi.config.global_runtime import delphi_real

    object_media_number = 1.0
    atoms = {}
    objects = ["is a molecule  0", " "]

    with open(filename) as fin:
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue

            record = ln[0:6].upper().strip()
            if record in ("ATOM", "HETATM"):
                atomnum = ln[6:11].strip()
                atomname = ln[12:16].strip()
                atomid = ln[11:26]
                resname = ln[17:20].strip()
                chain = ln[21:22].strip()
                resnum = ln[22:26].strip()

                resnum = int(resnum) if resnum else RES_NUMBER_UNKNOWN

                atom_key = (
                    record,
                    atomnum,
                    atomid,
                    atomname,
                    resname,
                    chain,
                    resnum,
                )  # record is first element of key

                atom_data = np.zeros(LEN_ATOMFIELDS, dtype=delphi_real)
                atom_data[ATOMFIELD_X] = delphi_real(ln[30:38].strip())
                atom_data[ATOMFIELD_Y] = delphi_real(ln[38:46].strip())
                atom_data[ATOMFIELD_Z] = delphi_real(ln[46:54].strip())
                atom_data[ATOMFIELD_GRID_X:ATOMFIELD_GRID_END] = [0.0, 0.0, 0.0]
                atom_data[ATOMFIELD_CHARGE] = delphi_real(ln[54:62].strip())
                atom_data[ATOMFIELD_RADIUS] = delphi_real(ln[62:70].strip())
                atom_data[ATOMFIELD_GAUSS_SIGMA] = delphi_real(1.0)
                atom_data[ATOMFIELD_RES_KEY] = ResNameToResKey.get(
                    resname.upper(), ResNameToResKey["UNK"]
                )
                element_symbol = _guess_element_from_atom_name(atomname)
                element_enum = get_element_by_symbol(element_symbol)
                atom_data[ATOMFIELD_ATOMIC_NUMBER] = element_enum.value
                atom_data[ATOMFIELD_LJ_SIGMA] = 0.0
                atom_data[ATOMFIELD_LJ_EPSILON] = 0.0
                atom_data[ATOMFIELD_LJ_GAMMA] = 0.0
                atom_data[ATOMFIELD_MEDIA_ID] = object_media_number
                atoms[atom_key] = atom_data
    # print(np.array(list(atoms.values()))[:,ATOMFIELD_ATOMIC_NUMBER])
    return atoms, objects


def write_pqr(filename, atoms, objects):
    """
    Writes atomic data back to a PQR file.

    Args:
        filename (str): The path to the output PQR file.
        atoms (dict): A dictionary of atomic data, as produced by read_pqr.
        objects (list): A list of strings, currently containing molecule
              object data like ["is a molecule  0", " "].
    """
    with open(filename, "w") as fout:
        # once = True
        for atom_key, atom_data in sorted(
            atoms.items(), key=lambda x: int(x[0][1])
        ):  # sorts by atomnum which is now the second element of the key.
            record, atomnum, atomid, atomname, resname, chain, resnum = atom_key
            x = atom_data[ATOMFIELD_X]
            y = atom_data[ATOMFIELD_Y]
            z = atom_data[ATOMFIELD_Z]
            charge = atom_data[ATOMFIELD_CHARGE]
            radius = atom_data[ATOMFIELD_RADIUS]
            atomic_number = atom_data[ATOMFIELD_ATOMIC_NUMBER]
            element = get_element_symbol_by_atomic_number(atomic_number)
            # if once:
            #     print(atomic_number, element)
            #     once = False
            line = (
                f"{record:<6}{atomnum:>5} {atomname:<4} {resname:>3} {chain:>1}{resnum:>4}    "  # 30
                f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{charge:>8.4f}{radius:>8.4f}"  # 70
                f"{'':7s}{element:<3s}\n"  # 80
            )
            fout.write(line)

        for obj in objects:
            fout.write(obj + "\n")
