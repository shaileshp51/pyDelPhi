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
prmtop_reader_lite.py  (refactored to return io.lite.TopologyLite)

Lite AMBER PRMTOP reader for pyDelPhi.

Goals:
- Minimal dependencies (no ParmEd).
- Robust parsing of common Amber PRMTOP variants.
- Extract only what PB/SA workflows need:
    - NATOM, NRES, IFBOX (from POINTERS)
    - CHARGE (converted to e)
    - RADII (PB/GB radii in Angstrom) if present
    - Optional: ATOM_NAME, RESIDUE_LABEL, RESIDUE_POINTER
    - Optional: ATOMIC_NUMBER when present; otherwise guessed from ATOM_NAME
- Preserve the raw POINTERS array for provenance/debug/future extensions.

Frozen conventions (pyDelPhi lite):
- atom_index is implicit (0..N-1)
- atom_serial is always 1..N (monotonic, never wraps)
- res_index is implicit (0..NRES-1)
- res_seq is external residue numbering; for PRMTOP default = 1..NRES
- chain_id/seg_id are residue-level arrays; PRMTOP provides none, so filled with "".
- atom_atomic_number is the authoritative element identity.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from pydelphi.utils.io.lite.topology_lite import (
    TopologyLite,
    make_default_chain_seg,
    atom_res_index_from_amber_residue_pointer,
    atomic_number_array_from_atom_names,
)

# Amber charge scaling: q(e) = q_prmtop / 18.2223
AMBER_CHARGE_SCALE = 18.2223


# -----------------------------
# POINTERS documentation
# -----------------------------
AMBER_POINTER_NAMES: Tuple[str, ...] = (
    "NATOM",
    "NTYPES",
    "NBONH",
    "MBONA",
    "NTHETH",
    "MTHETA",
    "NPHIH",
    "MPHIA",
    "NHPARM",
    "NPARM",
    "NNB",
    "NRES",
    "NBONA2",
    "NTHETA2",
    "NPHIA2",
    "NUMBND",
    "NUMANG",
    "NPTRA",
    "NATYP",
    "NPHB",
    "IFPERT",
    "NBPER",
    "NGPER",
    "NDPER",
    "MBPER",
    "MGPER",
    "MDPER",
    "IFBOX",
    "NMXRS",
    "IFCAP",
)

AMBER_PTR = {
    "NATOM": 0,
    "IFBOX": 27,
    "NRES": 11,
}


_FORMAT_RE = re.compile(r"\(\s*(\d+)\s*([aAiIeEfFgG])\s*(\d+)(?:\s*\.\s*(\d+))?\s*\)")


def _parse_fortran_format(fmt_text: str) -> Tuple[int, str, int]:
    """
    Parse AMBER %FORMAT strings like '(10I8)' '(20a4)' '(5E16.8)'.

    Returns:
        (values_per_line, type_char, field_width)
    """
    m = _FORMAT_RE.search(fmt_text.strip())
    if not m:
        raise ValueError(f"Unrecognized %FORMAT spec: {fmt_text!r}")
    n = int(m.group(1))
    t = m.group(2).upper()
    w = int(m.group(3))
    return n, t, w


def _iter_fixed_width_fields(line: str, width: int):
    """Yield fixed-width fields from a line."""
    for i in range(0, len(line), width):
        yield line[i : i + width].strip()


def _read_section_tokens(
    lines: List[str], start_idx: int, fmt: Tuple[int, str, int]
) -> Tuple[List[str], int]:
    """
    Read tokens starting at start_idx until the next %FLAG or EOF.
    Uses field_width from fmt (fixed-width parsing).
    """
    _, _, width = fmt
    tokens: List[str] = []
    i = start_idx

    while i < len(lines):
        raw = lines[i].rstrip("\n")
        s = raw.lstrip()

        if s.startswith("%FLAG"):
            break
        if (not raw) or s.startswith("%COMMENT"):
            i += 1
            continue

        for tok in _iter_fixed_width_fields(raw, width):
            if tok:
                tokens.append(tok)
        i += 1

    return tokens, i


def _coerce_tokens(tokens: List[str], type_char: str) -> Union[np.ndarray, List[str]]:
    """
    Convert tokens to appropriate dtype by section type.
    """
    if type_char == "A":
        return tokens
    if type_char == "I":
        return np.asarray([int(x) for x in tokens], dtype=np.int64)
    if type_char in ("E", "F", "G"):

        def _to_float(s: str) -> float:
            return float(s.replace("D", "E").replace("d", "E"))

        return np.asarray([_to_float(x) for x in tokens], dtype=np.float64)
    raise ValueError(f"Unsupported format type: {type_char!r}")


def _read_sections(prmtop_path: Path) -> Dict[str, Dict[str, object]]:
    with prmtop_path.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    sections: Dict[str, Dict[str, object]] = {}
    current_flag: Optional[str] = None
    i = 0

    while i < len(lines):
        raw = lines[i].rstrip("\n")
        s = raw.lstrip()

        if s.startswith("%FLAG"):
            parts = s.split(None, 1)
            if len(parts) < 2:
                raise ValueError(
                    f"Malformed PRMTOP %FLAG line. File: {prmtop_path}; line {i + 1}: {raw!r}"
                )
            current_flag = parts[1].strip()
            sections[current_flag] = {"fmt": None, "tokens": []}
            i += 1
            continue

        if s.startswith("%FORMAT"):
            if current_flag is None:
                raise ValueError(
                    f"Found PRMTOP %FORMAT before any %FLAG. File: {prmtop_path}; line {i + 1}: {raw!r}"
                )

            fmt_text = s[len("%FORMAT") :].strip()
            if not fmt_text:
                raise ValueError(
                    f"Empty PRMTOP %FORMAT. File: {prmtop_path}; line {i + 1}: {raw!r}"
                )

            fmt = _parse_fortran_format(fmt_text)
            sections[current_flag]["fmt"] = fmt

            tokens, next_i = _read_section_tokens(lines, i + 1, fmt)
            sections[current_flag]["tokens"] = tokens
            i = next_i
            continue

        i += 1

    return sections


def _require_section(
    sections: Dict[str, Dict[str, object]], name: str, path: Path
) -> Tuple[Tuple[int, str, int], List[str]]:
    if name not in sections or sections[name].get("fmt") is None:
        raise ValueError(f"PRMTOP missing required %FLAG {name} / %FORMAT: {path}")
    return sections[name]["fmt"], sections[name]["tokens"]  # type: ignore[return-value]


def _optional_string_section(
    sections: Dict[str, Dict[str, object]],
    name: str,
    expected: int,
) -> Optional[np.ndarray]:
    if name not in sections or sections[name].get("fmt") is None:
        return None
    fmt = sections[name]["fmt"]  # type: ignore[assignment]
    tokens = sections[name]["tokens"]  # type: ignore[assignment]
    _, typ, _ = fmt  # type: ignore[misc]
    values = _coerce_tokens(tokens, typ)
    if not isinstance(values, list) or len(values) < expected:
        return None
    return np.asarray([x.strip() for x in values[:expected]], dtype="<U8")


def _optional_int_section(
    sections: Dict[str, Dict[str, object]],
    name: str,
    expected: int,
) -> Optional[np.ndarray]:
    if name not in sections or sections[name].get("fmt") is None:
        return None
    fmt = sections[name]["fmt"]  # type: ignore[assignment]
    tokens = sections[name]["tokens"]  # type: ignore[assignment]
    _, typ, _ = fmt  # type: ignore[misc]
    values = _coerce_tokens(tokens, typ)
    if not isinstance(values, np.ndarray) or values.size < expected:
        return None
    return values[:expected].astype(np.int16, copy=False)


def read_prmtop_lite(
    prmtop_path: str,
    *,
    require_radii: bool = True,
    require_residue_pointer: bool = True,
    charge_scale: float = AMBER_CHARGE_SCALE,
    read_meta: bool = True,
) -> TopologyLite:
    """
    Read a lite view of an AMBER PRMTOP.

    Args:
        prmtop_path: path to .prmtop
        require_radii: raise if %FLAG RADII missing (True by default)
        require_residue_pointer: require RESIDUE_POINTER (recommended for traj workflows)
        charge_scale: scale factor for CHARGE; atom_charge(e) = q_raw / charge_scale
        read_meta: if True, attempt to read ATOM_NAME / RESIDUE_LABEL / RESIDUE_POINTER

    Returns:
        TopologyLite
    """
    path = Path(prmtop_path)
    if not path.is_file():
        raise FileNotFoundError(f"PRMTOP file not found: {path}")

    sections = _read_sections(path)

    # POINTERS
    ptr_fmt, ptr_tokens = _require_section(sections, "POINTERS", path)
    _, ptr_type, _ = ptr_fmt
    pointers64 = _coerce_tokens(ptr_tokens, ptr_type)

    if not isinstance(pointers64, np.ndarray) or pointers64.size == 0:
        raise ValueError(f"POINTERS section is empty or invalid: {path}")

    natoms = int(pointers64[AMBER_PTR["NATOM"]])
    nres = (
        int(pointers64[AMBER_PTR["NRES"]]) if pointers64.size > AMBER_PTR["NRES"] else 0
    )
    if natoms <= 0:
        raise ValueError(f"Invalid NATOM in POINTERS: {natoms} ({path})")
    if nres <= 0:
        raise ValueError(f"Invalid NRES in POINTERS: {nres} ({path})")

    ifbox = (
        int(pointers64[AMBER_PTR["IFBOX"]])
        if pointers64.size > AMBER_PTR["IFBOX"]
        else 0
    )
    pointers = pointers64.astype(np.int32, copy=False)

    # CHARGE
    chg_fmt, chg_tokens = _require_section(sections, "CHARGE", path)
    _, chg_type, _ = chg_fmt
    chg_raw = _coerce_tokens(chg_tokens, chg_type)

    if (not isinstance(chg_raw, np.ndarray)) or chg_raw.size < natoms:
        raise ValueError(
            f"CHARGE has {0 if not isinstance(chg_raw, np.ndarray) else chg_raw.size} values, expected {natoms}: {path}"
        )

    atom_charge = (chg_raw[:natoms] / float(charge_scale)).astype(
        np.float64, copy=False
    )

    # RADII
    if "RADII" in sections and sections["RADII"].get("fmt") is not None:
        rad_fmt = sections["RADII"]["fmt"]  # type: ignore[assignment]
        rad_tokens = sections["RADII"]["tokens"]  # type: ignore[assignment]
        _, rad_type, _ = rad_fmt  # type: ignore[misc]
        rad_raw = _coerce_tokens(rad_tokens, rad_type)

        if (not isinstance(rad_raw, np.ndarray)) or rad_raw.size < natoms:
            raise ValueError(
                f"RADII has {0 if not isinstance(rad_raw, np.ndarray) else rad_raw.size} values, expected {natoms}: {path}"
            )

        atom_radius = rad_raw[:natoms].astype(np.float64, copy=False)
    else:
        if require_radii:
            raise ValueError(f"PRMTOP missing %FLAG RADII (required): {path}")
        atom_radius = np.full(natoms, np.nan, dtype=np.float64)

    atom_name: Optional[np.ndarray] = None
    res_name: Optional[np.ndarray] = None
    residue_pointer: Optional[np.ndarray] = None
    atom_atomic_number: Optional[np.ndarray] = None

    if read_meta:
        atom_name = _optional_string_section(sections, "ATOM_NAME", natoms)

        res_name = _optional_string_section(sections, "RESIDUE_LABEL", nres)
        if res_name is not None and res_name.size < nres:
            raise ValueError(
                f"RESIDUE_LABEL has {res_name.size} entries, expected {nres}: {path}"
            )

        # AMBER PRMTOP can contain ATOMIC_NUMBER in some variants, but it is not guaranteed.
        atom_atomic_number = _optional_int_section(sections, "ATOMIC_NUMBER", natoms)
        if atom_atomic_number is None and atom_name is not None:
            atom_atomic_number = atomic_number_array_from_atom_names(atom_name)

        if (
            "RESIDUE_POINTER" in sections
            and sections["RESIDUE_POINTER"].get("fmt") is not None
        ):
            rp_fmt = sections["RESIDUE_POINTER"]["fmt"]  # type: ignore[assignment]
            rp_tokens = sections["RESIDUE_POINTER"]["tokens"]  # type: ignore[assignment]
            _, rp_type, _ = rp_fmt  # type: ignore[misc]
            rp = _coerce_tokens(rp_tokens, rp_type)
            if isinstance(rp, np.ndarray) and rp.size > 0:
                residue_pointer = rp.astype(np.int32, copy=False)
                if residue_pointer.size < nres:
                    raise ValueError(
                        f"RESIDUE_POINTER has {residue_pointer.size} entries, expected at least {nres}: {path}"
                    )
                if residue_pointer.size > nres:
                    residue_pointer = residue_pointer[:nres]

    if require_residue_pointer and residue_pointer is None:
        raise ValueError(
            f"PRMTOP missing RESIDUE_POINTER (required for residue mapping / traj workflows): {path}"
        )

    # Build frozen-lite fields; all are NumPy arrays before TopologyLite.
    atom_serial = np.arange(natoms, dtype=np.int64) + 1
    res_seq = np.arange(1, nres + 1, dtype=np.int32)
    chain_id, seg_id = make_default_chain_seg(nres)

    if residue_pointer is None:
        atom_res_index = np.zeros(natoms, dtype=np.int32)
    else:
        atom_res_index = atom_res_index_from_amber_residue_pointer(
            residue_pointer, natoms
        )

    if atom_name is None:
        atom_name = np.asarray([""] * natoms, dtype="<U8")
    if atom_atomic_number is None:
        atom_atomic_number = np.zeros(natoms, dtype=np.int16)
    if res_name is None:
        res_name = np.asarray([""] * nres, dtype="<U8")
    if residue_pointer is None:
        residue_pointer = (
            np.asarray([1], dtype=np.int32)
            if nres == 1
            else np.ones(nres, dtype=np.int32)
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
        atom_radius=atom_radius,
        atom_name=atom_name,
        atom_atomic_number=atom_atomic_number,
        res_name=res_name,
        residue_pointer=residue_pointer,
        pointers=pointers,
        ifbox=ifbox,
        format="prmtop",
        has_charge=True,
        has_size=bool(np.all(np.isfinite(atom_radius))),
    )


if __name__ == "__main__":
    import sys

    top = read_prmtop_lite(sys.argv[1], require_radii=False, read_meta=True)
    print("natoms:", top.natoms)
    print("nres:", top.nres)
    print("ifbox:", top.ifbox)
    print("pointers[:30]:", top.pointers[:30] if top.pointers is not None else None)
    print("q[:5]:", top.atom_charge[:5])
    print("r_pb[:5]:", top.atom_radius[:5])
    print("atom_serial[:5]:", top.atom_serial[:5])
    print("res_seq[:5]:", top.res_seq[:5])
    print("atom_res_index[:10]:", top.atom_res_index[:10])
    print(
        "atom_atomic_number[:5]:",
        top.atom_atomic_number[:5] if top.atom_atomic_number is not None else None,
    )
    if top.atom_name is not None:
        print("atom_name[:5]:", top.atom_name[:5])
    if top.res_name is not None:
        print("res_name[:5]:", top.res_name[:5])
