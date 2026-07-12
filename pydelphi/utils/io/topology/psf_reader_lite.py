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
Lite CHARMM/NAMD PSF reader for pyDelPhi.

Design goals
------------
- Minimal dependencies.
- Return a format-agnostic TopologyLite.
- Preserve the common lite conventions used by the Amber readers:
    - atom order is implicit (0..N-1)
    - atom_serial preserves PSF serials
    - residue indices are contiguous and zero-based
    - residue numbering is provided through res_seq
    - chain_id / seg_id are residue-level arrays
    - atom_charge is in e
    - atom_atomic_number is guessed from atom name/type

Notes
-----
PSF files generally do NOT contain PB radii. This reader therefore accepts an
optional `atom_radius` override. If none is provided, the radius array is filled
with `default_radius` (default: NaN). Downstream code can then decide whether to
accept, replace, or reject the topology.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:  # package import
    from pydelphi.utils.io.lite.topology_lite import (
        TopologyLite,
        atomic_number_from_element_symbol,
        guess_element_from_atom_name,
    )
except Exception:  # standalone / local import fallback
    from topology_lite import (  # type: ignore
        TopologyLite,
        atomic_number_from_element_symbol,
        guess_element_from_atom_name,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_lines(path: Union[str, Path]) -> Tuple[Path, List[str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PSF file not found: {p}")
    with p.open("r", encoding="utf-8", errors="replace") as f:
        return p, [ln.rstrip("\n") for ln in f]


def _first_nonempty_line(lines: Sequence[str], path: Path) -> Tuple[int, str]:
    for i, ln in enumerate(lines):
        if ln.strip():
            return i, ln
    raise ValueError(f"Empty PSF file: {path}")


def _malformed(path: Path, line_no: int, line_text: str, reason: str) -> ValueError:
    return ValueError(
        "\n".join(
            [
                "Malformed PSF atom line.",
                f"File: {path}",
                f"Line: {line_no}",
                f"Reason: {reason}",
                f"Line text: {line_text.rstrip()}",
            ]
        )
    )


def _parse_required_int(
    text: str, *, path: Path, line_no: int, line: str, field_name: str
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
    text: str, *, path: Path, line_no: int, line: str, field_name: str
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


def _normalize_radius_array(
    atom_radius: Optional[Union[float, Sequence[float], np.ndarray]],
    natoms: int,
    *,
    default_radius: float,
) -> np.ndarray:
    if atom_radius is None:
        return np.full(natoms, default_radius, dtype=np.float64)

    if np.isscalar(atom_radius):
        return np.full(natoms, float(atom_radius), dtype=np.float64)

    arr = np.asarray(atom_radius, dtype=np.float64)
    if arr.ndim != 1 or arr.shape[0] != natoms:
        raise ValueError(f"atom_radius must have shape ({natoms},)")
    return arr


def _resid_text_to_resseq(text: str, fallback: int) -> int:
    s = str(text or "").strip()
    if not s:
        return int(fallback)
    try:
        return int(s)
    except Exception:
        # PSF resid can be alphanumeric in some files. Keep TopologyLite.res_seq
        # numeric and deterministic rather than passing a string downstream.
        return int(fallback)


def _atomic_number_from_psf_names(atomname: str, atomtype: str) -> int:
    elem = guess_element_from_atom_name(atomname)
    anum = atomic_number_from_element_symbol(elem)
    if anum:
        return anum

    elem = guess_element_from_atom_name(atomtype)
    return atomic_number_from_element_symbol(elem)


# ---------------------------------------------------------------------------
# Parsed atom record (internal only)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PsfAtom:
    serial: int
    segid: str
    resid_text: str
    resname: str
    atomname: str
    atomtype: str
    charge: float
    atomic_number: int


def _parse_psf_atom_line(
    line: str, *, path: Path, line_no: int, ext: bool, xplor: bool
) -> _PsfAtom:
    """
    Parse one PSF atom record.

    Extracted lite fields:
        serial, segid, resid, resname, atomname, atomtype, charge, atomic_number.

    The atom type is not stored in TopologyLite, but it is used as a fallback
    for atomic-number guessing.
    """
    if ext:
        # EXT format:
        # I10,1X,A8,1X,A8,1X,A8,1X,A8,1X,I4/A4,1X,2G14.6,...
        _require_min_len(path, line_no, line, 66, "EXT PSF atom charge field")
        serial = _parse_required_int(
            line[0:10],
            path=path,
            line_no=line_no,
            line=line,
            field_name="serial columns 1-10",
        )
        segid = line[11:19].strip()
        resid_text = line[20:28].strip()
        resname = line[29:37].strip()
        atomname = line[38:46].strip()
        atomtype = line[47:51].strip()
        charge = _parse_required_float(
            line[52:66],
            path=path,
            line_no=line_no,
            line=line,
            field_name="charge columns 53-66",
        )
    else:
        # Standard format:
        # I8,1X,A4,1X,A4,1X,A4,1X,A4,1X,I4/A4,1X,2G14.6,...
        _require_min_len(path, line_no, line, 48, "standard PSF atom charge field")
        serial = _parse_required_int(
            line[0:8],
            path=path,
            line_no=line_no,
            line=line,
            field_name="serial columns 1-8",
        )
        segid = line[9:13].strip()
        resid_text = line[14:18].strip()
        resname = line[19:23].strip()
        atomname = line[24:28].strip()
        atomtype = line[29:33].strip()
        charge = _parse_required_float(
            line[34:48],
            path=path,
            line_no=line_no,
            line=line,
            field_name="charge columns 35-48",
        )

    if not resname:
        raise _malformed(path, line_no, line, "missing residue name")
    if not atomname:
        raise _malformed(path, line_no, line, "missing atom name")

    _ = xplor
    return _PsfAtom(
        serial=serial,
        segid=segid,
        resid_text=resid_text,
        resname=resname,
        atomname=atomname,
        atomtype=atomtype,
        charge=charge,
        atomic_number=_atomic_number_from_psf_names(atomname, atomtype),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_psf_lite(
    psf_path: Union[str, Path],
    *,
    atom_radius: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    default_radius: float = np.nan,
    default_chain_id: str = "",
) -> TopologyLite:
    """
    Read a lite topology from a CHARMM/NAMD PSF file.
    """
    path, lines = _read_lines(psf_path)
    _, first = _first_nonempty_line(lines, path)

    if not first.lstrip().startswith("PSF"):
        raise ValueError(
            f"Not a valid PSF file (first non-empty line must start with 'PSF'): {path}"
        )

    header = first.strip().upper()
    ext = "EXT" in header
    xplor = "XPLOR" in header
    cheq = "CHEQ" in header

    natom_idx = None
    natoms = None
    for i, ln in enumerate(lines):
        if "!NATOM" in ln.upper():
            natom_idx = i
            head = ln.split("!", 1)[0].strip()
            try:
                natoms = int(head)
            except Exception as exc:
                raise ValueError(
                    f"Invalid !NATOM declaration. File: {path}; line {i + 1}: {ln!r}"
                ) from exc
            break

    if natom_idx is None or natoms is None or natoms <= 0:
        raise ValueError(f"PSF file is missing a valid !NATOM block: {path}")

    atom_start = natom_idx + 1
    atom_end = atom_start + natoms
    if atom_end > len(lines):
        raise ValueError(
            f"PSF atom block is truncated: expected {natoms} atom lines after !NATOM in {path}"
        )

    atoms: List[_PsfAtom] = []
    for offset, ln in enumerate(lines[atom_start:atom_end], start=atom_start + 1):
        if not ln.strip():
            raise _malformed(
                path, offset, ln, "blank line inside declared !NATOM block"
            )
        atoms.append(
            _parse_psf_atom_line(ln, path=path, line_no=offset, ext=ext, xplor=xplor)
        )

    if len(atoms) != natoms:
        raise ValueError(
            f"Parsed {len(atoms)} PSF atom records, but !NATOM declared {natoms}: {path}"
        )

    residue_keys: List[Tuple[str, str, str]] = []
    atom_res_index = np.empty(natoms, dtype=np.int32)

    key_to_resindex: dict[Tuple[str, str, str], int] = {}
    for i, at in enumerate(atoms):
        key = (at.segid, at.resid_text, at.resname)
        if key not in key_to_resindex:
            key_to_resindex[key] = len(residue_keys)
            residue_keys.append(key)
        atom_res_index[i] = key_to_resindex[key]

    nres = len(residue_keys)

    res_seq = np.asarray(
        [
            _resid_text_to_resseq(k[1], fallback=i + 1)
            for i, k in enumerate(residue_keys)
        ],
        dtype=np.int32,
    )
    chain_id = np.full(nres, default_chain_id, dtype="<U4")
    seg_id = np.asarray([k[0] for k in residue_keys], dtype="<U8")
    res_name = np.asarray([k[2] for k in residue_keys], dtype="<U8")

    residue_pointer = np.empty(nres, dtype=np.int32)
    first_atom_for_res = np.full(nres, -1, dtype=np.int32)
    for i, ridx in enumerate(atom_res_index):
        ridx = int(ridx)
        if first_atom_for_res[ridx] < 0:
            first_atom_for_res[ridx] = i + 1
    residue_pointer[:] = first_atom_for_res

    atom_serial = np.asarray([a.serial for a in atoms], dtype=np.int64)
    atom_charge = np.asarray([a.charge for a in atoms], dtype=np.float64)
    atom_name = np.asarray([a.atomname for a in atoms], dtype="<U8")
    atom_atomic_number = np.asarray([a.atomic_number for a in atoms], dtype=np.int16)

    atom_radius_arr = _normalize_radius_array(
        atom_radius, natoms, default_radius=default_radius
    )

    flags = (1 if ext else 0) | (2 if xplor else 0) | (4 if cheq else 0)
    pointers = np.column_stack(
        [
            atom_serial.astype(np.int32, copy=False),
            atom_res_index.astype(np.int32, copy=False),
            np.rint(atom_charge * 1000.0).astype(np.int32, copy=False),
            atom_atomic_number.astype(np.int32, copy=False),
            np.full(natoms, flags, dtype=np.int32),
        ]
    )

    return TopologyLite(
        natoms=int(natoms),
        nres=int(nres),
        atom_serial=atom_serial,
        atom_res_index=atom_res_index,
        res_seq=res_seq,
        chain_id=chain_id,
        seg_id=seg_id,
        atom_charge=atom_charge,
        atom_radius=atom_radius_arr,
        atom_name=atom_name,
        atom_atomic_number=atom_atomic_number,
        res_name=res_name,
        residue_pointer=residue_pointer,
        pointers=pointers,
        ifbox=0,
        format="psf",
        has_charge=True,
        has_size=bool(np.all(np.isfinite(atom_radius_arr))),
    )


open_psf_lite = read_psf_lite


if __name__ == "__main__":
    import sys

    top = read_psf_lite(sys.argv[1])
    print("natoms:", top.natoms)
    print("nres:", top.nres)
    print("atom_charge[:5]:", top.atom_charge[:5])
    print("atom_radius[:5]:", top.atom_radius[:5])
    print("atom_atomic_number[:5]:", top.atom_atomic_number[:5])
    print("res_seq[:5]:", top.res_seq[:5])
