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

from __future__ import annotations

"""pdb_reader_lite.py

Dependency-light fixed-width PDB topology reader for pyDelPhi.

The reader returns ``TopologyLite`` and conforms to the lite topology contract:
all topology fields are NumPy arrays before construction, and element identity
is stored only as ``atom_atomic_number``. No atom symbol/element string is
stored because it is redundant and can drift from atomic number.
"""

from pathlib import Path
from typing import List, Tuple, Union

import numpy as np

try:
    from pydelphi.utils.io.lite.topology_lite import (
        TopologyLite,
        make_default_chain_seg,
        guess_element_from_atom_name,
        normalize_element_symbol,
        atomic_number_from_element_symbol,
    )
except Exception:
    from topology_lite import (  # type: ignore
        TopologyLite,
        make_default_chain_seg,
        guess_element_from_atom_name,
        normalize_element_symbol,
        atomic_number_from_element_symbol,
    )


_PDB_ATOM_RECORDS = {"ATOM", "HETATM"}


def _malformed(path: Path, line_no: int, line_text: str, reason: str) -> ValueError:
    return ValueError(
        "\n".join(
            [
                "Malformed PDB atom line.",
                f"File: {path}",
                f"Line: {line_no}",
                f"Reason: {reason}",
                f"Line text: {line_text.rstrip()}",
            ]
        )
    )


def _require_min_len(
    path: Path, line_no: int, line: str, n: int, field_name: str
) -> None:
    if len(line) < n:
        raise _malformed(
            path,
            line_no,
            line,
            f"line is too short to parse {field_name}; expected at least {n} characters, got {len(line)}",
        )


def _parse_required_int(
    text: str,
    *,
    path: Path,
    line_no: int,
    line: str,
    field_name: str,
) -> int:
    value = text.strip()
    if not value:
        raise _malformed(path, line_no, line, f"missing integer field: {field_name}")
    try:
        return int(value)
    except Exception as exc:
        raise _malformed(
            path, line_no, line, f"invalid integer field {field_name}: {value!r}"
        ) from exc


def _parse_required_float(
    text: str,
    *,
    path: Path,
    line_no: int,
    line: str,
    field_name: str,
) -> float:
    value = text.strip()
    if not value:
        raise _malformed(path, line_no, line, f"missing numeric field: {field_name}")
    try:
        return float(value)
    except Exception as exc:
        raise _malformed(
            path, line_no, line, f"invalid numeric field {field_name}: {value!r}"
        ) from exc


def _read_reserved_resname(line: str) -> str:
    """
    Read residue name using pyDelPhi's reserved 4th-character convention.

    Base residue name uses columns 18-20. Column 21 is treated as an optional
    reserved residue-name extension only when it is nonspace. Chain remains
    column 22.
    """
    res_name = line[17:20].strip()
    res_name4 = line[20:21].strip()
    if res_name4:
        res_name = f"{res_name}{res_name4}"
    return res_name


def _atomic_number_from_pdb_line(line: str) -> int:
    """
    Return authoritative atomic number from element columns 77-78 if present,
    otherwise guess from atom name columns 13-16.

    The element symbol is intentionally not stored in TopologyLite.
    """
    elem_field = line[76:78].strip() if len(line) >= 78 else ""
    elem = normalize_element_symbol(elem_field)
    if not elem:
        elem = guess_element_from_atom_name(line[12:16])
    return atomic_number_from_element_symbol(elem)


def read_pdb_lite(
    file_path: Union[str, Path],
    *,
    read_meta: bool = True,
) -> TopologyLite:
    """
    Read a fixed-width PDB file as a lite topology.

    ATOM/HETATM lines must be long enough to parse segid columns 73-76. The
    element field at columns 77-78 is optional; if absent or blank, atomic
    number is guessed from atom name columns 13-16.

    PDB does not carry pyDelPhi-ready charge/size data in this lite path:
        - atom_charge is initialized to 0.0
        - atom_radius is initialized to NaN
        - has_charge is False
        - has_size is False
    """
    _ = read_meta

    path = Path(file_path)
    if not file_path:
        raise ValueError("PDB topology input missing file path.")

    if not path.is_file():
        raise FileNotFoundError(f"PDB topology file not found: {path}")

    atom_serial: List[int] = []
    atom_names: List[str] = []
    atom_atomic_numbers: List[int] = []
    atom_res_index: List[int] = []

    residue_keys: List[Tuple[str, int, str, str, str]] = []
    residue_index_by_key: dict[Tuple[str, int, str, str, str], int] = {}

    res_seq_values: List[int] = []
    chain_values: List[str] = []
    seg_values: List[str] = []
    res_name_values: List[str] = []
    residue_pointer_values: List[int] = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            record = line[0:6].strip().upper()
            if record not in _PDB_ATOM_RECORDS:
                continue

            _require_min_len(
                path, line_no, line, 76, "PDB atom fields through segid column"
            )

            serial = _parse_required_int(
                line[6:11],
                path=path,
                line_no=line_no,
                line=line,
                field_name="atom serial columns 7-11",
            )
            atom_name = line[12:16].strip()
            if not atom_name:
                raise _malformed(path, line_no, line, "missing atom name columns 13-16")

            res_name = _read_reserved_resname(line)
            if not res_name:
                raise _malformed(
                    path, line_no, line, "missing residue name columns 18-20"
                )

            chain_id = line[21:22].strip()
            res_seq = _parse_required_int(
                line[22:26],
                path=path,
                line_no=line_no,
                line=line,
                field_name="residue sequence columns 23-26",
            )
            icode = line[26:27].strip()

            # Coordinates are validated here even though they are not stored,
            # because short/misaligned fixed-width records should fail early.
            _parse_required_float(
                line[30:38],
                path=path,
                line_no=line_no,
                line=line,
                field_name="x coordinate columns 31-38",
            )
            _parse_required_float(
                line[38:46],
                path=path,
                line_no=line_no,
                line=line,
                field_name="y coordinate columns 39-46",
            )
            _parse_required_float(
                line[46:54],
                path=path,
                line_no=line_no,
                line=line,
                field_name="z coordinate columns 47-54",
            )

            seg_id = line[72:76].strip()
            atomic_number = _atomic_number_from_pdb_line(line)

            residue_key = (chain_id, res_seq, icode, seg_id, res_name)
            if residue_key not in residue_index_by_key:
                residue_index_by_key[residue_key] = len(residue_keys)
                residue_keys.append(residue_key)
                res_seq_values.append(res_seq)
                chain_values.append(chain_id)
                seg_values.append(seg_id)
                res_name_values.append(res_name)
                residue_pointer_values.append(len(atom_serial) + 1)

            atom_serial.append(serial)
            atom_names.append(atom_name)
            atom_atomic_numbers.append(atomic_number)
            atom_res_index.append(residue_index_by_key[residue_key])

    natoms = len(atom_serial)
    if natoms == 0:
        raise ValueError(f"No ATOM/HETATM records found in PDB topology: {path}")

    nres = len(residue_keys)

    chain_id_arr, seg_id_arr = make_default_chain_seg(nres)
    chain_id_arr[:] = np.asarray(chain_values, dtype=chain_id_arr.dtype)
    seg_id_arr[:] = np.asarray(seg_values, dtype=seg_id_arr.dtype)

    return TopologyLite(
        natoms=int(natoms),
        nres=int(nres),
        atom_serial=np.asarray(atom_serial, dtype=np.int64),
        atom_res_index=np.asarray(atom_res_index, dtype=np.int32),
        res_seq=np.asarray(res_seq_values, dtype=np.int32),
        chain_id=chain_id_arr,
        seg_id=seg_id_arr,
        atom_charge=np.zeros(natoms, dtype=np.float64),
        atom_radius=np.full(natoms, np.nan, dtype=np.float64),
        atom_name=np.asarray(atom_names, dtype="<U8"),
        atom_atomic_number=np.asarray(atom_atomic_numbers, dtype=np.int16),
        res_name=np.asarray(res_name_values, dtype="<U8"),
        residue_pointer=np.asarray(residue_pointer_values, dtype=np.int32),
        pointers=None,
        ifbox=0,
        format="pdb",
        has_charge=False,
        has_size=False,
    )


read_pdb_topology_lite = read_pdb_lite
read_topology_lite = read_pdb_lite
read = read_pdb_lite
