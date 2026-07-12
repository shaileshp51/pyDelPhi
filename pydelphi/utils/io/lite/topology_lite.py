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

from dataclasses import dataclass
from typing import Optional
from os import path
import numpy as np

from pydelphi.constants import (
    ConstChemElement,
    get_element_by_symbol,
    get_element_symbol_by_atomic_number,
)


def normalize_element_symbol(symbol: str) -> str:
    """
    Normalize an element symbol through pydelphi.constants.

    Returns "" for empty/unknown symbols. The periodic table and symbol mapping
    remain authoritative in pydelphi.constants; this helper only applies the
    same capitalization and lookup policy used by the static PDB/PQR path.
    """
    s = str(symbol or "").strip()
    if not s:
        return ""
    normalized = s.capitalize()
    return normalized if ConstChemElement.has_member(normalized) else ""


def guess_element_from_atom_name(atom_name: str) -> str:
    """
    Guess a chemical element symbol from a PDB/PQR/PSF/PRMTOP atom name.

    This mirrors the static PDB/PQR fallback behavior:
    - remove digits
    - if the second character is lowercase, allow a two-letter element
    - otherwise use the first uppercase character, so "CA" is carbon alpha,
      not calcium, unless an explicit element field says otherwise.
    """
    name_no_digits = "".join(
        ch for ch in str(atom_name or "") if not ch.isdigit()
    ).strip()

    if not name_no_digits:
        return ""

    if len(name_no_digits) == 1:
        return normalize_element_symbol(name_no_digits.upper())

    first_char = name_no_digits[0].upper()
    second_char = name_no_digits[1]
    if second_char.islower() and ConstChemElement.has_member(first_char + second_char):
        return first_char + second_char

    return normalize_element_symbol(first_char)


def atomic_number_from_element_symbol(symbol: str) -> int:
    """
    Return atomic number for an element symbol using pydelphi.constants.

    Returns 0 for unknown symbols.
    """
    return int(get_element_by_symbol(symbol).value)


def atomic_number_array_from_atom_names(
    atom_names: np.ndarray | list[str],
) -> np.ndarray:
    """
    Build atom_atomic_number from atom names using pydelphi.constants.

    Readers for formats without an explicit atomic-number field should use this
    helper so all lite topology paths provide the same downstream selection
    metadata without storing redundant atom-symbol strings.
    """
    names = list(atom_names)
    atomic_numbers = np.zeros(len(names), dtype=np.int16)
    for i, name in enumerate(names):
        elem = guess_element_from_atom_name(str(name))
        atomic_numbers[i] = atomic_number_from_element_symbol(elem)
    return atomic_numbers


def element_symbol_from_atomic_number(atomic_number: int) -> str:
    """Return element symbol for atomic number using pydelphi.constants."""
    return str(get_element_symbol_by_atomic_number(float(atomic_number)))


@dataclass(frozen=True)
class TopologyLite:
    """
    Format-agnostic topology needed by pyDelPhi PB/SA workflows.

    Conventions (frozen):
    - atom_index is implicit: 0..N-1 (array indices)
    - atom_serial is always present: 1..N (monotonic, never wraps)
    - res_index is 0-based contiguous: 0..NRES-1
    - res_seq is external/user-facing residue numbering (may have gaps); for PRMTOP default = res_index+1
    - chain_id / seg_id exist as residue-level labels; absent formats provide "".
    - atom_atomic_number is the authoritative atom-level element identity used
      by selection/materialization logic; symbols can be derived from it when
      needed and are not stored redundantly.
    - Units:
        atom_charge: e
        atom_radius: Angstrom (PB/GB radius)
    """

    # sizes
    natoms: int
    nres: int

    # stable identifiers (always present)
    atom_serial: np.ndarray  # (N,) int64, 1-based, monotonic
    atom_res_index: np.ndarray  # (N,) int32, 0-based res_index per atom
    res_seq: np.ndarray  # (NRES,) int32/int64 external residue number

    # optional grouping ids (normalized to arrays of "" if unknown)
    chain_id: np.ndarray  # (NRES,) '<U2' or '<U4' (can be all "")
    seg_id: np.ndarray  # (NRES,) '<U4' (can be all "")

    # PB-relevant atom properties
    atom_charge: np.ndarray  # (N,) float64 in e
    atom_radius: np.ndarray  # (N,) float64 in Angstrom

    # optional metadata
    atom_name: Optional[np.ndarray] = None  # (N,) '<U4'
    atom_atomic_number: Optional[np.ndarray] = (
        None  # (N,) int16 atomic number; 0 if unknown
    )
    res_name: Optional[np.ndarray] = None  # (NRES,) '<U4' (residue_label)
    residue_pointer: Optional[np.ndarray] = (
        None  # (NRES,) int32, 1-based atom starts (Amber provenance)
    )

    # provenance/debug (format-specific raw pointers etc.)
    pointers: Optional[np.ndarray] = None  # int32, raw POINTERS if available
    ifbox: int = 0  # Amber convenience; safe default 0

    # normalized reader provenance/capability flags
    format: str = ""
    has_charge: bool = False
    has_size: bool = False

    def __post_init__(self) -> None:
        """
        Normalize all topology containers to NumPy arrays at construction time.

        The trajectory/materializer path operates on NumPy arrays, not Python
        lists. Readers should normally pass arrays explicitly, but this
        constructor-level normalization is a defensive boundary so no list-like
        containers leak into downstream trajectory operations.
        """
        natoms = int(self.natoms)
        nres = int(self.nres)
        object.__setattr__(self, "natoms", natoms)
        object.__setattr__(self, "nres", nres)

        def _as_1d_array(
            name: str,
            dtype,
            *,
            expected_len: int | None,
            required: bool,
        ) -> None:
            value = getattr(self, name)
            if value is None:
                if required:
                    raise ValueError(f"TopologyLite.{name} is required.")
                return

            arr = np.asarray(value, dtype=dtype)
            if arr.ndim != 1:
                raise ValueError(
                    f"TopologyLite.{name} must be a 1D NumPy array; got shape {arr.shape}."
                )
            if expected_len is not None and arr.shape[0] != expected_len:
                raise ValueError(
                    f"TopologyLite.{name} length mismatch: expected {expected_len}, "
                    f"got {arr.shape[0]}."
                )
            object.__setattr__(self, name, arr)

        _as_1d_array("atom_serial", np.int64, expected_len=natoms, required=True)
        _as_1d_array("atom_res_index", np.int32, expected_len=natoms, required=True)
        _as_1d_array("res_seq", np.int32, expected_len=nres, required=True)
        _as_1d_array("chain_id", str, expected_len=nres, required=True)
        _as_1d_array("seg_id", str, expected_len=nres, required=True)
        _as_1d_array("atom_charge", np.float64, expected_len=natoms, required=True)
        _as_1d_array("atom_radius", np.float64, expected_len=natoms, required=True)

        _as_1d_array("atom_name", "<U8", expected_len=natoms, required=False)
        _as_1d_array(
            "atom_atomic_number", np.int16, expected_len=natoms, required=False
        )
        _as_1d_array("res_name", "<U8", expected_len=nres, required=False)
        _as_1d_array("residue_pointer", np.int32, expected_len=nres, required=False)

        if self.pointers is not None:
            pointers_arr = np.asarray(self.pointers, dtype=np.int32)
            object.__setattr__(self, "pointers", pointers_arr)

        if natoms < 0 or nres < 0:
            raise ValueError("TopologyLite natoms/nres must be non-negative.")

        if self.atom_res_index.size:
            min_res = int(np.min(self.atom_res_index))
            max_res = int(np.max(self.atom_res_index))
            if min_res < 0 or max_res >= nres:
                raise ValueError(
                    "TopologyLite.atom_res_index contains residue indices outside "
                    f"[0, {nres})."
                )


def make_default_chain_seg(nres: int) -> tuple[np.ndarray, np.ndarray]:
    """Create default empty chain/seg arrays."""
    chain_id = np.full(nres, "", dtype="<U2")
    seg_id = np.full(nres, "", dtype="<U4")
    return chain_id, seg_id


def atom_res_index_from_amber_residue_pointer(
    residue_pointer_1based: np.ndarray, natoms: int
) -> np.ndarray:
    """
    Build atom->res_index mapping from Amber RESIDUE_POINTER (1-based atom start indices).
    O(NATOM + NRES)
    """
    if residue_pointer_1based is None or residue_pointer_1based.size == 0:
        raise ValueError("RESIDUE_POINTER is required to build atom_res_index")

    starts = residue_pointer_1based.astype(np.int64) - 1
    nres = int(starts.size)

    if nres <= 0:
        raise ValueError("Invalid RESIDUE_POINTER length")
    if starts[0] != 0:
        raise ValueError(
            f"Unexpected RESIDUE_POINTER[0]={residue_pointer_1based[0]} (expected 1)."
        )
    if starts[-1] >= natoms:
        raise ValueError("RESIDUE_POINTER last start is beyond NATOM")

    ends = np.empty(nres, dtype=np.int64)
    ends[:-1] = starts[1:]
    ends[-1] = natoms

    atom_res_index = np.empty(natoms, dtype=np.int32)
    for r in range(nres):
        atom_res_index[starts[r] : ends[r]] = np.int32(r)
    return atom_res_index


_SUPPORTED_TOPOLOGY_FORMATS = {
    "pdb": "pdb",
    "ent": "pdb",
    "pqr": "pqr",
    "psf": "psf",
    "prmtop": "prmtop",
    "amber": "prmtop",
    "parm7": "prmtop",
}


def normalize_topology_format(fmt: str) -> str:
    """Normalize a user-facing lite topology format name."""
    fmt = str(fmt or "").strip().lower()

    if fmt not in _SUPPORTED_TOPOLOGY_FORMATS:
        raise ValueError(
            f"Unsupported topology format in traj mode: {fmt!r}. "
            "Supported lite topology formats are: pdb, pqr, psf, prmtop."
        )

    return _SUPPORTED_TOPOLOGY_FORMATS[fmt]


def _reader_capabilities(top_fmt: str) -> tuple[bool, bool]:
    """Return (has_charge, has_size) for normalized lite topology format."""
    if top_fmt == "pdb":
        return False, False
    if top_fmt == "psf":
        return True, False
    if top_fmt in {"pqr", "prmtop"}:
        return True, True
    raise ValueError(f"Unsupported topology format: {top_fmt!r}")


def _with_topology_metadata(top: TopologyLite, top_fmt: str) -> TopologyLite:
    """
    Attach normalized format/capability metadata without mutating readers.

    Readers may eventually populate these directly; the dispatcher still
    normalizes them here so callers can rely on the fields.
    """
    from dataclasses import replace

    has_charge, has_size = _reader_capabilities(top_fmt)
    return replace(
        top,
        format=top_fmt,
        has_charge=has_charge,
        has_size=has_size,
    )


def open_topology_lite(file_path: str, fmt: str | None = None) -> TopologyLite:
    """
    Open a topology file through the lite topology interface.

    This is the format-dispatch boundary for topology readers. Higher-level
    callers should not import individual pdb/pqr/psf/prmtop readers directly.
    """
    if not file_path:
        raise ValueError("Topology input missing required attribute: file=...")

    if not path.isfile(file_path):
        raise FileNotFoundError(f"Topology file not found: {file_path}")

    top_fmt = normalize_topology_format(fmt or "prmtop")

    if top_fmt == "pdb":
        from pydelphi.utils.io.topology.pdb_reader_lite import read_pdb_lite

        top = read_pdb_lite(file_path)
        return _with_topology_metadata(top, top_fmt)

    if top_fmt == "pqr":
        from pydelphi.utils.io.topology.pqr_reader_lite import read_pqr_lite

        top = read_pqr_lite(file_path)
        return _with_topology_metadata(top, top_fmt)

    if top_fmt == "psf":
        from pydelphi.utils.io.topology.psf_reader_lite import read_psf_lite

        top = read_psf_lite(file_path)
        return _with_topology_metadata(top, top_fmt)

    if top_fmt == "prmtop":
        from pydelphi.utils.io.topology.prmtop_reader_lite import read_prmtop_lite

        top = read_prmtop_lite(file_path, require_radii=False, read_meta=True)
        return _with_topology_metadata(top, top_fmt)

    raise ValueError(f"Unsupported topology format in traj mode: {top_fmt!r}")


def _topology_array_names() -> tuple[str, ...]:
    return (
        "atom_serial",
        "atom_res_index",
        "res_seq",
        "chain_id",
        "seg_id",
        "atom_charge",
        "atom_radius",
        "atom_name",
        "atom_atomic_number",
        "res_name",
        "residue_pointer",
        "pointers",
    )


def freeze_topology_lite(top: TopologyLite) -> TopologyLite:
    """
    Finalize a TopologyLite by marking all NumPy arrays read-only.

    Call this after charge, size, and other topology-level properties are
    finalized and before storing the topology in an EnsembleEntry.
    """
    for name in _topology_array_names():
        arr = getattr(top, name, None)
        if arr is not None:
            arr.setflags(write=False)
    return top


def is_topology_lite_frozen(top: TopologyLite) -> bool:
    """Return True when all TopologyLite arrays are read-only."""
    for name in _topology_array_names():
        arr = getattr(top, name, None)
        if arr is not None and arr.flags.writeable:
            return False
    return True


def thaw_topology_lite(top: TopologyLite) -> TopologyLite:
    """
    Return a mutable-copy TopologyLite.

    This is intentionally not part of the normal open -> assign -> freeze path;
    use it only when mutation is required for a topology that is already frozen
    or shared.
    """
    from dataclasses import replace

    def _copy(arr):
        if arr is None:
            return None
        out = np.array(arr, copy=True)
        out.setflags(write=True)
        return out

    return replace(
        top,
        atom_serial=_copy(top.atom_serial),
        atom_res_index=_copy(top.atom_res_index),
        res_seq=_copy(top.res_seq),
        chain_id=_copy(top.chain_id),
        seg_id=_copy(top.seg_id),
        atom_charge=_copy(top.atom_charge),
        atom_radius=_copy(top.atom_radius),
        atom_name=_copy(top.atom_name),
        atom_atomic_number=_copy(top.atom_atomic_number),
        res_name=_copy(top.res_name),
        residue_pointer=_copy(top.residue_pointer),
        pointers=_copy(top.pointers),
    )
