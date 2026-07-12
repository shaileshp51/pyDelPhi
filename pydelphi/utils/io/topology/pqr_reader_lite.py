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

"""
Dependency-light PQR topology reader for pyDelPhi.

This reader is intentionally narrow and only returns ``TopologyLite`` fields.
It is designed for pyDelPhi's fixed-width PQR/PDB-style input contract and
does not rely on whitespace tokenization.

Fixed-width parsing policy
--------------------------
PQR records are parsed using the same fixed-width column contract as the
static PDB/PQR reader:

    record:        columns 1-6
    atom serial:   columns 7-11
    atom name:     columns 13-16
    residue name:  columns 18-20
    residue char4: column 21, optional/reserved when nonspace
    chain:         column 22
    residue seq:   columns 23-26
    x:             columns 31-38
    y:             columns 39-46
    z:             columns 47-54
    charge:        columns 55-62
    radius:        columns 63-70
    segid:         columns 73-76
    element:       columns 77-78, optional; converted to atomic number

If a line is too short or a field cannot be parsed, the reader raises a
clear ValueError containing file name, line number, and the original line.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Union

import numpy as np

try:  # package import
    from pydelphi.utils.io.lite.topology_lite import (
        TopologyLite,
        make_default_chain_seg,
        guess_element_from_atom_name,
        normalize_element_symbol,
        atomic_number_from_element_symbol,
    )
except Exception:  # standalone / local fallback
    from topology_lite import (  # type: ignore
        TopologyLite,
        make_default_chain_seg,
        guess_element_from_atom_name,
        normalize_element_symbol,
        atomic_number_from_element_symbol,
    )


@dataclass(frozen=True)
class _PqrAtom:
    serial: int
    atom_name: str
    res_name: str
    res_seq: int
    chain_id: str
    seg_id: str
    atomic_number: int
    charge: float
    radius: float


def _malformed(path: Path, line_no: int, line_text: str, reason: str) -> ValueError:
    return ValueError(
        "\n".join(
            [
                "Malformed PQR atom line.",
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


def _atomic_number_from_pqr_line(line: str) -> int:
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


def _parse_pqr_atom_line(path: Path, line_no: int, line: str) -> _PqrAtom:
    """
    Parse a single fixed-width PQR atom line.

    This intentionally does not fall back to whitespace tokenization. Loose
    token-based parsing can silently misread charge/radius when spacing drifts.
    """
    _require_min_len(path, line_no, line, 76, "PQR atom fields through segid column")

    record = line[0:6].strip().upper()
    if record not in {"ATOM", "HETATM"}:
        raise _malformed(path, line_no, line, f"not an ATOM/HETATM record: {record!r}")

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
        raise _malformed(path, line_no, line, "missing residue name columns 18-20")

    chain_id = line[21:22].strip()

    res_seq = _parse_required_int(
        line[22:26],
        path=path,
        line_no=line_no,
        line=line,
        field_name="residue sequence columns 23-26",
    )

    # Coordinates are validated even though TopologyLite stores topology only.
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

    charge = _parse_required_float(
        line[54:62],
        path=path,
        line_no=line_no,
        line=line,
        field_name="charge columns 55-62",
    )

    radius = _parse_required_float(
        line[62:70],
        path=path,
        line_no=line_no,
        line=line,
        field_name="radius columns 63-70",
    )

    if radius <= 0.0:
        raise _malformed(
            path, line_no, line, f"radius must be positive; parsed radius={radius}"
        )

    return _PqrAtom(
        serial=serial,
        atom_name=atom_name,
        res_name=res_name,
        res_seq=res_seq,
        chain_id=chain_id,
        seg_id=line[72:76].strip(),
        atomic_number=_atomic_number_from_pqr_line(line),
        charge=charge,
        radius=radius,
    )


def read_pqr_lite(
    pqr_path: Union[str, Path],
    *,
    preserve_resseq: bool = True,
) -> TopologyLite:
    """
    Read a fixed-width PQR file into ``TopologyLite``.

    Notes
    -----
    - PQR supplies both charge and size, so returned topology has
      ``has_charge=True`` and ``has_size=True``.
    - Atom serials preserve the PQR atom serial column.
    - Residue names use columns 18-20 plus optional reserved column 21.
    - Segid is read from columns 73-76.
    - Atomic number is derived from element columns 77-78 if present, else
      guessed from atom name.
    """
    path = Path(pqr_path)
    if not path.exists():
        raise FileNotFoundError(f"PQR file not found: {path}")

    atoms: List[_PqrAtom] = []

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, raw in enumerate(fh, start=1):
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            rec = line[:6].strip().upper()
            if rec not in {"ATOM", "HETATM"}:
                continue
            atoms.append(_parse_pqr_atom_line(path, line_no, line))

    if not atoms:
        raise ValueError(f"No ATOM/HETATM records found in PQR file: {path}")

    natoms = len(atoms)

    atom_serial = np.empty(natoms, dtype=np.int64)
    atom_charge = np.empty(natoms, dtype=np.float64)
    atom_radius = np.empty(natoms, dtype=np.float64)
    atom_name = np.empty(natoms, dtype="<U8")
    atom_atomic_number = np.zeros(natoms, dtype=np.int16)

    residue_pointer: List[int] = []
    atom_res_index = np.empty(natoms, dtype=np.int32)
    res_seq_vals: List[int] = []
    res_name_vals: List[str] = []
    chain_vals: List[str] = []
    seg_vals: List[str] = []

    prev_key = None
    res_idx = -1
    for i, a in enumerate(atoms):
        atom_serial[i] = a.serial
        atom_name[i] = a.atom_name
        atom_atomic_number[i] = a.atomic_number
        atom_charge[i] = a.charge
        atom_radius[i] = a.radius

        key = (a.chain_id, a.seg_id, a.res_seq, a.res_name)
        if key != prev_key:
            res_idx += 1
            residue_pointer.append(i + 1)
            res_seq_vals.append(a.res_seq if preserve_resseq else res_idx + 1)
            res_name_vals.append(a.res_name)
            chain_vals.append(a.chain_id)
            seg_vals.append(a.seg_id)
            prev_key = key
        atom_res_index[i] = np.int32(res_idx)

    nres = res_idx + 1
    res_seq = np.asarray(res_seq_vals, dtype=np.int32)
    res_name = np.asarray(res_name_vals, dtype="<U8")
    chain_id, seg_id = make_default_chain_seg(nres)
    chain_id[:] = np.asarray(chain_vals, dtype=chain_id.dtype)
    seg_id[:] = np.asarray(seg_vals, dtype=seg_id.dtype)

    return TopologyLite(
        natoms=int(natoms),
        nres=int(nres),
        atom_serial=atom_serial,
        atom_res_index=atom_res_index,
        res_seq=res_seq,
        chain_id=chain_id,
        seg_id=seg_id,
        atom_charge=atom_charge,
        atom_radius=atom_radius,
        atom_name=atom_name,
        atom_atomic_number=atom_atomic_number,
        res_name=res_name,
        residue_pointer=np.asarray(residue_pointer, dtype=np.int32),
        pointers=None,
        ifbox=0,
        format="pqr",
        has_charge=True,
        has_size=True,
    )


if __name__ == "__main__":
    import sys

    top = read_pqr_lite(sys.argv[1])
    print("natoms:", top.natoms)
    print("nres:", top.nres)
    print("q[:5]:", top.atom_charge[:5])
    print("r[:5]:", top.atom_radius[:5])
    print("res_seq[:5]:", top.res_seq[:5])
    print("atom_name[:5]:", top.atom_name[:5] if top.atom_name is not None else None)
    print(
        "atom_atomic_number[:5]:",
        top.atom_atomic_number[:5] if top.atom_atomic_number is not None else None,
    )
    print("res_name[:5]:", top.res_name[:5] if top.res_name is not None else None)
