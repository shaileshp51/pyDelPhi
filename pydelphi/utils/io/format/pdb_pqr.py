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

from os import path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Mapping, Tuple

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

from pydelphi.utils.io.atomkey_fields import (
    AK_RECORD,
    AK_ATOMNUM,
    AK_ATOMINDEX,
    AK_NAME,
    AK_RESNAME,
    AK_CHAIN,
    AK_RESNUM,
    AK_ATOMTYPE,
    AK_SEGID,
    AK_ATOMIC_NUMBER,
    AK_LEN_V1,
    AK_LEN_V2,
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


def get_atomic_number_from_atomname(atom_name):
    element_symbol = _guess_element_from_atom_name(atom_name)
    element_enum = get_element_by_symbol(element_symbol)
    return element_enum.value


def _read_reserved_resname(line: str) -> str:
    """
    Read a residue name from PDB/PQR fixed-width columns with pyDelPhi's
    reserved 4th-character convention.

    Base residue name uses columns 18-20, matching the legacy 3-character
    PDB/DelPhi convention. Column 21 is treated as an optional reserved
    residue-name extension only when it is nonspace. Chain remains column 22.

    This preserves legacy behavior for ordinary 3-character residue names while
    allowing meaningful 4-character names such as TIP3, TIP4, NADP, and NADH.
    """
    resname = line[17:20].strip()
    resname4 = line[20:21].strip()
    if resname4:
        resname = f"{resname}{resname4}"
    return resname


def _resname_to_reskey(resname: str):
    """Return residue key, falling back from 4-char names to legacy 3-char names."""
    name = str(resname or "").strip().upper()
    return ResNameToResKey.get(
        name,
        ResNameToResKey.get(name[:3], ResNameToResKey["UNK"]),
    )


def _split_resname_reserved4(resname: str) -> tuple[str, str]:
    """
    Split a residue name into legacy columns 18-20 plus reserved column 21.

    The 4th character is emitted only when present. Longer residue names are
    intentionally truncated to this 3+1 representation because the writer uses
    PDB/PQR fixed-width output.
    """
    name = str(resname or "").strip()
    if len(name) > 3:
        return name[:3], name[3:4]
    return name, " "


def _format_pdb_residue_fields(resname, chain, resnum) -> str:
    """
    Format residue name, optional reserved 4th character, chain, and residue
    number for PDB/PQR output.

    Layout after atom name:
        altLoc blank, resname columns 18-20, reserved char column 21,
        chain column 22, residue number columns 23-26.
    """
    res3, res4 = _split_resname_reserved4(resname)
    chain1 = str(chain or "").strip()[:1]
    return f" {res3:>3}{res4:1}{chain1:>1}{int(resnum):>4}"


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
              (atom number, atom ID, atom name, residue name, chain, residue number, atom_type).
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
        atomindex = 0
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue

            record = ln[0:6].upper().strip()
            if record in ("ATOM", "HETATM"):
                atomnum = ln[6:11].strip()
                atomname = ln[12:16].strip()
                # atomid = ln[11:26]
                resname = _read_reserved_resname(ln)
                chain = ln[21:22].strip()
                resnum = ln[22:26].strip()
                atomtype = ""
                segid = ln[72:76].strip()

                # Convert residue number to integer or set as unknown
                resnum = int(resnum) if resnum else RES_NUMBER_UNKNOWN

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
                atom_data[ATOMFIELD_RES_KEY] = _resname_to_reskey(resname)

                # Element atomic number
                element_symbol = _get_element_symbol(ln)
                element_enum = get_element_by_symbol(element_symbol)
                atomic_number = element_enum.value
                atom_data[ATOMFIELD_ATOMIC_NUMBER] = atomic_number

                # Additional properties (default values)
                atom_data[ATOMFIELD_LJ_SIGMA] = 0.0  # LJ-sigma
                atom_data[ATOMFIELD_LJ_EPSILON] = 0.0  # LJ-epsilon
                atom_data[ATOMFIELD_LJ_GAMMA] = 0.0  # van der Waals gamma
                atom_data[ATOMFIELD_MEDIA_ID] = (
                    object_media_number  # Object media number
                )

                # Create the key for the atom
                atom_key = (
                    record,
                    atomnum,
                    atomindex,
                    atomname,
                    resname,
                    chain,
                    resnum,
                    atomtype,
                    segid,
                    atomic_number,
                )
                # Store the atom data in the dictionary
                atoms[atom_key] = atom_data
                atomindex += 1

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
              (atom number, atom ID, atom name, residue name, chain, residue number, atom type).
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
        atomindex = 0
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue

            record = ln[0:6].upper().strip()
            if record in ("ATOM", "HETATM"):
                atomserial = int(ln[6:11].strip())
                atomname = ln[12:16].strip()
                # atomid = ln[11:26]
                resname = _read_reserved_resname(ln)
                chain = ln[21:22].strip()
                resnum = ln[22:26].strip()
                atomtype = ""
                segid = ln[72:76].strip()

                resnum = int(resnum) if resnum else RES_NUMBER_UNKNOWN

                atom_data = np.zeros(LEN_ATOMFIELDS, dtype=delphi_real)
                atom_data[ATOMFIELD_X] = delphi_real(ln[30:38].strip())
                atom_data[ATOMFIELD_Y] = delphi_real(ln[38:46].strip())
                atom_data[ATOMFIELD_Z] = delphi_real(ln[46:54].strip())
                atom_data[ATOMFIELD_GRID_X:ATOMFIELD_GRID_END] = [0.0, 0.0, 0.0]
                atom_data[ATOMFIELD_CHARGE] = delphi_real(ln[54:62].strip())
                atom_data[ATOMFIELD_RADIUS] = delphi_real(ln[62:70].strip())
                atom_data[ATOMFIELD_GAUSS_SIGMA] = delphi_real(1.0)
                atom_data[ATOMFIELD_RES_KEY] = _resname_to_reskey(resname)
                element_symbol = _guess_element_from_atom_name(atomname)
                element_enum = get_element_by_symbol(element_symbol)
                atomic_number = element_enum.value
                atom_data[ATOMFIELD_ATOMIC_NUMBER] = atomic_number
                atom_data[ATOMFIELD_LJ_SIGMA] = 0.0
                atom_data[ATOMFIELD_LJ_EPSILON] = 0.0
                atom_data[ATOMFIELD_LJ_GAMMA] = 0.0
                atom_data[ATOMFIELD_MEDIA_ID] = object_media_number

                atom_key = (
                    record,
                    atomserial,
                    atomindex,
                    atomname,
                    resname,
                    chain,
                    resnum,
                    atomtype,
                    segid,
                    atomic_number,
                )  # record is first element of key
                # print("atom_key=", atom_key)
                atoms[atom_key] = atom_data
                atomindex += 1
    # print(np.array(list(atoms.values()))[:,ATOMFIELD_ATOMIC_NUMBER])
    return atoms, objects


AtomKey = Tuple[
    Any, ...
]  # (record, atomnum, atomid, atomname, resname, chain, resnum, atomtype)


def default_sort_key(k: AtomKey) -> int:
    return int(k[1])  # atomnum


def compute_sort_perm(
    atom_keys: Sequence[AtomKey],
    sort_key: Optional[Callable[[AtomKey], Any]] = None,
) -> List[int]:
    """
    Compute a permutation (list of indices) that sorts atom_keys by sort_key.
    Do this once, reuse for all frames.
    """
    keyfunc = sort_key or default_sort_key
    return sorted(range(len(atom_keys)), key=lambda i: keyfunc(atom_keys[i]))


def _infer_format(filename: str) -> str:
    _, ext = path.splitext(filename)
    ext = (ext or "").lower().lstrip(".")
    if ext in ("pqr", "pdb"):
        return ext
    raise ValueError(
        f"format='auto' requires filename extension '.pqr' or '.pdb' (got: {filename!r})."
    )


def write_atoms(
    filename: str,
    atoms: Optional[dict] = None,
    objects: Optional[Iterable[str]] = None,
    atom_keys: Optional[Sequence[Any]] = None,
    atom_data: Optional[Sequence[Sequence[float]]] = None,
    # Sorting controls for trajectory mode:
    sort: bool = True,
    sort_key: Optional[Callable[[Any], Any]] = None,
    sort_perm: Optional[Sequence[int]] = None,
    # NEW:
    format: str = "auto",  # "auto" | "pqr" | "pdb"
):
    """
    Write structure in PQR or PDB format using either:
      - atoms dict (legacy): mapping is inherent, sorting sorts items
      - atom_keys + atom_data (trajectory): mapping is positional; sorting uses a permutation

    Parameters
    ----------
    filename : str
        Output file path. If format='auto', extension must be .pqr or .pdb.
    format : str
        'auto' (default) infers from filename extension, else 'pqr' or 'pdb'.

    Notes
    -----
    - PQR writes charge and radius.
    - PDB does not write charge/radius; uses occupancy/tempFactor placeholders.
    - Residue names use legacy columns 18-20 plus optional reserved column 21
      for a meaningful 4th residue-name character.
    """
    if objects is None:
        objects = []

    fmt = (format or "auto").strip().lower()
    if fmt == "auto":
        fmt = _infer_format(filename)
    if fmt not in ("pqr", "pdb"):
        raise ValueError(
            f"Unsupported format={format!r}. Expected 'auto', 'pqr', or 'pdb'."
        )

    with open(filename, "w") as fout:
        if atoms is not None:
            # dict mode: mapping is explicit, safe to sort items directly
            items = atoms.items()
            if sort:
                keyfunc = sort_key or default_sort_key
                items = sorted(items, key=lambda kv: keyfunc(kv[0]))

            for atom_key, data in items:
                (
                    record,
                    atomnum,
                    atomid,
                    atomname,
                    resname,
                    chain,
                    resnum,
                    atomtype,
                    segid,
                    atomic_number,
                ) = atom_key

                x = data[ATOMFIELD_X]
                y = data[ATOMFIELD_Y]
                z = data[ATOMFIELD_Z]
                atomic_number = data[ATOMFIELD_ATOMIC_NUMBER]
                element = get_element_symbol_by_atomic_number(atomic_number)

                record_out = record if record else "ATOM"

                if fmt == "pqr":
                    charge = data[ATOMFIELD_CHARGE]
                    radius = data[ATOMFIELD_RADIUS]
                    line = (
                        f"{record_out:<6}{int(atomnum):>5} {atomname:<4}{_format_pdb_residue_fields(resname, chain, resnum)}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{charge:>8.4f}{radius:>8.4f}"
                        f"{'':7s}{element:<3s}\n"
                    )
                else:
                    # PDB: no q/r
                    occupancy = 1.00
                    tempfactor = 0.00
                    line = (
                        f"{record_out:<6}{int(atomnum):>5} {atomname:<4}{_format_pdb_residue_fields(resname, chain, resnum)}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                        f"{occupancy:>6.2f}{tempfactor:>6.2f}"
                        f"{'':10s}{element:>2s}\n"
                    )

                fout.write(line)

        else:
            # trajectory mode: mapping is by index; sorting must use an index permutation
            if atom_keys is None or atom_data is None:
                raise ValueError(
                    "Provide either 'atoms' OR both 'atom_keys' and 'atom_data'."
                )
            if len(atom_keys) != len(atom_data):
                raise ValueError("atom_keys and atom_data must have the same length.")

            if not sort:
                perm = range(len(atom_keys))
            else:
                if sort_perm is not None:
                    perm = sort_perm
                else:
                    perm = compute_sort_perm(atom_keys, sort_key=sort_key)

            for i in perm:
                atom_key = atom_keys[i]
                data = atom_data[i]

                # your existing trajectory AtomKey parsing
                record_out = "ATOM"
                chain, segid, resname, resnum, atomname, atomnum = [
                    k.strip() for k in atom_key.split(":")
                ]

                x = data[ATOMFIELD_X]
                y = data[ATOMFIELD_Y]
                z = data[ATOMFIELD_Z]
                atomic_number = data[ATOMFIELD_ATOMIC_NUMBER]
                element = get_element_symbol_by_atomic_number(atomic_number)

                if fmt == "pqr":
                    charge = data[ATOMFIELD_CHARGE]
                    radius = data[ATOMFIELD_RADIUS]
                    line = (
                        f"{record_out:<6}{int(atomnum):>5} {atomname:<4}{_format_pdb_residue_fields(resname, chain, resnum)}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{charge:>8.4f}{radius:>8.4f}"
                        f"{'':7s}{element:<3s}\n"
                    )
                else:
                    occupancy = 1.00
                    tempfactor = 0.00
                    line = (
                        f"{record_out:<6}{int(atomnum):>5} {atomname:<4}{_format_pdb_residue_fields(resname, chain, resnum)}    "
                        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                        f"{occupancy:>6.2f}{tempfactor:>6.2f}"
                        f"{'':10s}{element:>2s}\n"
                    )

                fout.write(line)

        for obj in objects:
            fout.write(str(obj) + "\n")


# Optional: backward-compatible wrapper
def write_pqr(*args, **kwargs):
    """
    Backward-compatible wrapper. Forces PQR unless caller passes format explicitly.
    """
    if "format" not in kwargs:
        kwargs["format"] = "pqr"
    return write_atoms(*args, **kwargs)


def write_selection(
    filename: str,
    format: str,
    all_atoms_keys: Sequence[Any],  # Sequence[AtomKey]
    all_atoms_dict: Mapping[Any, Sequence[float]],  # dict[AtomKey, data]
    sel_atoms_key_indices: Sequence[int],  # indices into all_atoms_keys
    objects: Optional[Iterable[str]] = None,
    # optional: sort selection indices by atom_key metadata
    sort: bool = False,
    sort_key: Optional[Callable[[Any], Any]] = None,
):
    """
    Write a named selection to a structure file.

    Parameters
    ----------
    filename : str
        Output file path.
    format : str
        Output format: 'pqr' or 'pdb'.
    all_atoms_keys : sequence
        List-like container of AtomKey such that all_atoms_keys[i] -> AtomKey.
    all_atoms_dict : mapping
        AtomKey -> atom data array (x,y,z,q,r,atomic_number,...).
    sel_atoms_key_indices : sequence[int]
        Indices selecting atoms from all_atoms_keys.
    objects : iterable[str], optional
        Extra records appended at end (TER/END/REMARK/...).
    sort : bool
        If True, sorts selected atoms using sort_key(atom_key).
    sort_key : callable, optional
        Custom sort key for AtomKey. If None and sort=True, keeps input order.

    Notes
    -----
    - PQR: writes charge and radius.
    - PDB: does NOT write charge and radius.
    - This function assumes filenames/labels preserve case upstream; it does not modify filename.
    """
    if objects is None:
        objects = []

    format = (format or "").strip().lower()
    if format not in ("pqr", "pdb"):
        raise ValueError(
            f"write_selection: unsupported fmt='{format}'. Expected 'pqr' or 'pdb'."
        )

    # Validate indices quickly
    n_all = len(all_atoms_keys)
    for idx in sel_atoms_key_indices:
        if idx < 0 or idx >= n_all:
            raise IndexError(
                f"write_selection: selection index {idx} out of range [0, {n_all})."
            )

    # Optional sorting: sort by AtomKey (not by index)
    if sort and sort_key is not None:
        # stable sort
        sel_indices = sorted(
            sel_atoms_key_indices, key=lambda i: sort_key(all_atoms_keys[i])
        )
    else:
        sel_indices = sel_atoms_key_indices

    with open(filename, "w") as fout:
        for i in sel_indices:
            atom_k = all_atoms_keys[i]
            data = all_atoms_dict.get(atom_k, None)
            if data is None:
                raise KeyError(
                    f"write_selection: AtomKey not found in all_atoms_dict: {atom_k!r}"
                )

            # AtomKey unpacking: match your dict-mode template
            # (record, atomnum, atomid, atomname, resname, chain, resnum, atomtype) = atom_k
            (
                record,
                atomnum,
                atomid,
                atomname,
                resname,
                chain,
                resnum,
                atomtype,
                segid,
                atomic_number,
            ) = atom_k

            x = data[ATOMFIELD_X]
            y = data[ATOMFIELD_Y]
            z = data[ATOMFIELD_Z]

            # element symbol (reuse your existing helper)
            atomic_number = data[ATOMFIELD_ATOMIC_NUMBER]
            element = get_element_symbol_by_atomic_number(atomic_number)

            record_out = record if record else "ATOM"

            if format == "pqr":
                charge = data[ATOMFIELD_CHARGE]
                radius = data[ATOMFIELD_RADIUS]
                line = (
                    f"{record_out:<6}{int(atomnum):>5} {atomname:<4}{_format_pdb_residue_fields(resname, chain, resnum)}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}{charge:>8.4f}{radius:>8.4f}"
                    f"{'':7s}{element:<3s}\n"
                )
            else:
                # PDB: no q/r; use occupancy/tempFactor placeholders
                # Keep formatting simple and stable; element in columns 77-78-ish with padding.
                occupancy = 1.00
                tempfactor = 0.00
                # atomid/atomtype are available but not required; keep minimal
                line = (
                    f"{record_out:<6}{int(atomnum):>5} {atomname:<4}{_format_pdb_residue_fields(resname, chain, resnum)}    "
                    f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"
                    f"{occupancy:>6.2f}{tempfactor:>6.2f}"
                    f"{'':10s}{element:>2s}\n"
                )

            fout.write(line)

        for obj in objects:
            fout.write(str(obj) + "\n")
