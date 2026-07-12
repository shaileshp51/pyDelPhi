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


import struct
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
    get_element_by_atomic_number,
    ConstChemElement,
    ConstDelPhiInts,
    ConstDelPhiFloats,
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


# ==============================================================================
# ZPHI sparse offset-surface potential reader
# ==============================================================================

from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import sys


ZPHI_VERSION = 1
ZPHI_DEFAULT_DENSE_MISSING_VALUE = 1.0e30
ZPHI_ORIGIN_CHECK_ATOL = 1.0e-5
ZPHI_AVG_CHECK_ATOL = 5.0e-6


class ZphiFormatError(ValueError):
    """Raised when a .zphi file is malformed or internally inconsistent."""


@dataclass(frozen=True)
class ZphiData:
    """
    In-memory representation of a canonical sparse .zphi file.

    Data rows are index-only:
        ix iy iz phi_kT_per_e

    Coordinates are reconstructed from:
        GRID_ORIGIN_ANG + index * GRID_SPACING_ANG
    """

    metadata: Dict[str, str]
    indices: np.ndarray
    potentials: np.ndarray

    @property
    def num_points(self) -> int:
        return int(self.indices.shape[0])


def _zphi_as_3_int_array(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _zphi_as_3_float_array(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _zphi_parse_vec(
    value: str,
    dtype=float,
    expected_len: Optional[int] = None,
) -> np.ndarray:
    parts = value.split()
    if expected_len is not None and len(parts) != expected_len:
        raise ZphiFormatError(
            f"Expected {expected_len} values, got {len(parts)} in metadata value {value!r}."
        )
    return np.asarray(parts, dtype=dtype)


def compute_zphi_grid_origin_from_center(
    grid_center_ang: Union[np.ndarray, Tuple[float, float, float]],
    grid_shape: Union[np.ndarray, Tuple[int, int, int]],
    grid_spacing_ang: float,
) -> np.ndarray:
    """
    Compute grid origin from center, shape, and spacing.

    origin = center - spacing * (shape - 1) / 2
    """
    grid_center_ang = _zphi_as_3_float_array(grid_center_ang, "grid_center_ang")
    grid_shape = _zphi_as_3_int_array(grid_shape, "grid_shape")
    return (
        grid_center_ang - grid_spacing_ang * (grid_shape.astype(np.float64) - 1.0) / 2.0
    )


def compute_zphi_grid_center_from_origin(
    grid_origin_ang: Union[np.ndarray, Tuple[float, float, float]],
    grid_shape: Union[np.ndarray, Tuple[int, int, int]],
    grid_spacing_ang: float,
) -> np.ndarray:
    """Compute grid center from origin, shape, and spacing."""
    grid_origin_ang = _zphi_as_3_float_array(grid_origin_ang, "grid_origin_ang")
    grid_shape = _zphi_as_3_int_array(grid_shape, "grid_shape")
    return (
        grid_origin_ang + grid_spacing_ang * (grid_shape.astype(np.float64) - 1.0) / 2.0
    )


def validate_zphi_indices_in_grid(indices: np.ndarray, grid_shape: np.ndarray) -> None:
    """Validate sparse indices against parent grid shape."""
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"indices must have shape (N, 3), got {indices.shape}.")

    if np.any(indices < 0):
        bad = indices[np.any(indices < 0, axis=1)][0]
        raise ValueError(f"Negative surface grid index found: {bad.tolist()}.")

    upper = grid_shape.reshape((1, 3))
    if np.any(indices >= upper):
        bad = indices[np.any(indices >= upper, axis=1)][0]
        raise ValueError(
            f"Surface grid index {bad.tolist()} is out of bounds for grid shape {grid_shape.tolist()}."
        )


def zphi_indices_to_coords_ang(
    indices: np.ndarray,
    *,
    grid_origin_ang: Union[np.ndarray, Tuple[float, float, float]],
    grid_spacing_ang: float,
) -> np.ndarray:
    """
    Reconstruct coordinates from indices and grid metadata.

    Returns shape (N, 3), dtype float64.
    """
    indices = np.asarray(indices, dtype=np.float64)
    origin = _zphi_as_3_float_array(grid_origin_ang, "grid_origin_ang")
    return origin.reshape((1, 3)) + indices * float(grid_spacing_ang)


def read_zphi(
    zphi_filename: str,
    *,
    strict: bool = True,
    check_origin: bool = True,
) -> ZphiData:
    """
    Read a canonical sparse .zphi file.

    In strict mode, validates required metadata, version, row schema,
    point count, index bounds, origin consistency, and average consistency.
    """
    metadata: Dict[str, str] = {}
    data_rows = []
    in_data = False

    with open(zphi_filename, "r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()

            if not line:
                continue

            if line.startswith("#"):
                body = line[1:].strip()

                if body == "BEGIN_DATA":
                    in_data = True
                    continue
                if body == "END_DATA":
                    in_data = False
                    continue

                if not in_data and ":" in body:
                    key, value = body.split(":", 1)
                    metadata[key.strip().upper()] = value.strip()

                continue

            if in_data:
                parts = line.split()
                if len(parts) != 4:
                    raise ZphiFormatError(
                        f"Malformed data row at line {line_no}: "
                        f"expected 4 columns, got {len(parts)}."
                    )
                try:
                    ix, iy, iz = map(int, parts[0:3])
                    phi = float(parts[3])
                except ValueError as e:
                    raise ZphiFormatError(
                        f"Could not parse data row at line {line_no}: {line!r}"
                    ) from e
                data_rows.append((ix, iy, iz, phi))
            elif strict:
                raise ZphiFormatError(
                    f"Unexpected non-comment line outside data block at line {line_no}: {line!r}"
                )

    required = [
        "ZPHI_VERSION",
        "CONTENT",
        "UNITS_COORD",
        "UNITS_POTENTIAL",
        "GRID_SHAPE",
        "GRID_SCALE_PER_ANG",
        "GRID_SPACING_ANG",
        "GRID_CENTER_ANG",
        "GRID_ORIGIN_ANG",
        "INDEX_BASE",
        "INDEX_ORDER",
        "DATA_COLUMNS",
        "CONTEXT_TYPE",
        "CONTEXT_LABEL",
        "NUM_SURFACE_POINTS",
        "SIMPLE_AVERAGE_SURFACE_POTENTIAL",
    ]

    missing = [key for key in required if key not in metadata]
    if missing and strict:
        raise ZphiFormatError(f"Missing required ZPHI metadata fields: {missing}")

    version = int(metadata.get("ZPHI_VERSION", "-1"))
    if version != ZPHI_VERSION and strict:
        raise ZphiFormatError(
            f"Unsupported ZPHI_VERSION {version}; expected {ZPHI_VERSION}."
        )

    if metadata.get("INDEX_BASE", "0") != "0" and strict:
        raise ZphiFormatError("Only INDEX_BASE: 0 is currently supported.")

    if metadata.get("INDEX_ORDER", "ix iy iz") != "ix iy iz" and strict:
        raise ZphiFormatError("Only INDEX_ORDER: ix iy iz is currently supported.")

    if (
        metadata.get("DATA_COLUMNS", "ix iy iz phi_kT_per_e") != "ix iy iz phi_kT_per_e"
        and strict
    ):
        raise ZphiFormatError(
            "Only DATA_COLUMNS: ix iy iz phi_kT_per_e is currently supported."
        )

    if not data_rows:
        raise ZphiFormatError("No data rows found in .zphi file.")

    arr = np.asarray(data_rows, dtype=np.float64)
    indices = np.ascontiguousarray(arr[:, 0:3].astype(np.int64), dtype=np.int64)
    potentials = np.ascontiguousarray(arr[:, 3].astype(np.float64), dtype=np.float64)

    n_expected = int(metadata.get("NUM_SURFACE_POINTS", indices.shape[0]))
    if indices.shape[0] != n_expected:
        msg = (
            f"NUM_SURFACE_POINTS says {n_expected}, "
            f"but file contains {indices.shape[0]} data rows."
        )
        if strict:
            raise ZphiFormatError(msg)
        print(f"Warning: {msg}", file=sys.stderr)

    if "GRID_SHAPE" in metadata:
        grid_shape = _zphi_parse_vec(
            metadata["GRID_SHAPE"], dtype=np.int64, expected_len=3
        )
        try:
            validate_zphi_indices_in_grid(indices, grid_shape)
        except ValueError as e:
            raise ZphiFormatError(str(e)) from e

    if check_origin and all(
        key in metadata
        for key in [
            "GRID_SHAPE",
            "GRID_SCALE_PER_ANG",
            "GRID_SPACING_ANG",
            "GRID_CENTER_ANG",
            "GRID_ORIGIN_ANG",
        ]
    ):
        grid_shape = _zphi_parse_vec(
            metadata["GRID_SHAPE"], dtype=np.int64, expected_len=3
        )
        grid_scale = float(metadata["GRID_SCALE_PER_ANG"])
        computed_spacing = 1.0 / grid_scale
        stored_spacing = float(metadata["GRID_SPACING_ANG"])
        grid_center = _zphi_parse_vec(
            metadata["GRID_CENTER_ANG"], dtype=np.float64, expected_len=3
        )
        stored_origin = _zphi_parse_vec(
            metadata["GRID_ORIGIN_ANG"], dtype=np.float64, expected_len=3
        )
        computed_origin = compute_zphi_grid_origin_from_center(
            grid_center, grid_shape, computed_spacing
        )

        if not np.isclose(
            stored_spacing, computed_spacing, atol=ZPHI_ORIGIN_CHECK_ATOL
        ):
            msg = (
                "GRID_SPACING_ANG is inconsistent with GRID_SCALE_PER_ANG. "
                f"stored={stored_spacing}, computed={computed_spacing}"
            )
            if strict:
                raise ZphiFormatError(msg)
            print(f"Warning: {msg}", file=sys.stderr)

        if not np.allclose(stored_origin, computed_origin, atol=ZPHI_ORIGIN_CHECK_ATOL):
            msg = (
                "GRID_ORIGIN_ANG is inconsistent with GRID_CENTER_ANG, "
                "GRID_SCALE_PER_ANG, and GRID_SHAPE. "
                f"stored={stored_origin}, computed={computed_origin}"
            )
            if strict:
                raise ZphiFormatError(msg)
            print(f"Warning: {msg}", file=sys.stderr)

    if "SIMPLE_AVERAGE_SURFACE_POTENTIAL" in metadata:
        stored_avg = float(metadata["SIMPLE_AVERAGE_SURFACE_POTENTIAL"].split()[0])
        computed_avg = float(np.mean(potentials))
        if not np.isclose(
            stored_avg, computed_avg, atol=ZPHI_AVG_CHECK_ATOL, rtol=ZPHI_AVG_CHECK_ATOL
        ):
            msg = (
                "SIMPLE_AVERAGE_SURFACE_POTENTIAL is inconsistent with data rows. "
                f"stored={stored_avg}, computed={computed_avg}"
            )
            if strict:
                raise ZphiFormatError(msg)
            print(f"Warning: {msg}", file=sys.stderr)

    return ZphiData(metadata=metadata, indices=indices, potentials=potentials)


def get_zphi_coords_ang(zphi: ZphiData) -> np.ndarray:
    """Return reconstructed coordinates for all sparse .zphi points."""
    if (
        "GRID_ORIGIN_ANG" not in zphi.metadata
        or "GRID_SPACING_ANG" not in zphi.metadata
    ):
        raise ZphiFormatError(
            "Cannot reconstruct coordinates: missing grid origin or spacing metadata."
        )

    origin = _zphi_parse_vec(
        zphi.metadata["GRID_ORIGIN_ANG"], dtype=np.float64, expected_len=3
    )
    spacing = float(zphi.metadata["GRID_SPACING_ANG"])
    return zphi_indices_to_coords_ang(
        zphi.indices, grid_origin_ang=origin, grid_spacing_ang=spacing
    )


def zphi_to_dense_grid(
    zphi: Union[ZphiData, str],
    *,
    missing_value: Optional[float] = None,
    dtype=np.float64,
) -> np.ndarray:
    """
    Convert sparse .zphi data into a dense 3D grid.

    Non-surface points are filled with `missing_value`. If `missing_value` is None,
    uses DENSE_EXPORT_MISSING_VALUE from metadata, falling back to 1.0e30.
    """
    if not isinstance(zphi, ZphiData):
        zphi = read_zphi(zphi)

    if "GRID_SHAPE" not in zphi.metadata:
        raise ZphiFormatError(
            "Cannot build dense grid: GRID_SHAPE metadata is missing."
        )

    grid_shape = _zphi_parse_vec(
        zphi.metadata["GRID_SHAPE"], dtype=np.int64, expected_len=3
    )

    if missing_value is None:
        missing_value = float(
            zphi.metadata.get(
                "DENSE_EXPORT_MISSING_VALUE", ZPHI_DEFAULT_DENSE_MISSING_VALUE
            )
        )

    dense = np.full(tuple(grid_shape), missing_value, dtype=dtype)
    dense[
        zphi.indices[:, 0],
        zphi.indices[:, 1],
        zphi.indices[:, 2],
    ] = zphi.potentials.astype(dtype)

    return dense


def summarize_zphi(zphi: Union[ZphiData, str]) -> Dict[str, Union[str, int, float]]:
    """Return a compact summary dictionary for logs and tests."""
    if not isinstance(zphi, ZphiData):
        zphi = read_zphi(zphi)

    pot = zphi.potentials
    summary: Dict[str, Union[str, int, float]] = {
        "version": int(zphi.metadata.get("ZPHI_VERSION", "-1")),
        "content": zphi.metadata.get("CONTENT", ""),
        "context_type": zphi.metadata.get("CONTEXT_TYPE", ""),
        "context_label": zphi.metadata.get("CONTEXT_LABEL", ""),
        "num_points": int(zphi.num_points),
        "potential_mean": float(np.mean(pot)),
        "potential_min": float(np.min(pot)),
        "potential_max": float(np.max(pot)),
        "potential_std": float(np.std(pot)),
    }

    if "SURFACE_DISTANCE_ANG" in zphi.metadata:
        summary["surface_distance_ang"] = float(zphi.metadata["SURFACE_DISTANCE_ANG"])
    if "CONTEXT_NUM_ATOMS" in zphi.metadata:
        summary["context_num_atoms"] = int(zphi.metadata["CONTEXT_NUM_ATOMS"])

    return summary


def _read_reserved_resname(line: str, start: int = 6) -> str:
    """
    Read a legacy 3-character residue name with an optional reserved 4th
    character.

    For CRG/SIZ/SIGMA-style fixed-width records, residue name occupies
    columns 7-9 (0-based 6:9). Column 10 (0-based 9:10) is normally a
    spacer/reserved column. If it is nonspace, treat it as the 4th residue
    name character. This preserves legacy 3-character records while allowing
    records such as TIP3, TIP4, NADP, etc. without shifting chain/residue
    number columns.
    """
    resname = line[start : start + 3].strip()
    resname4 = line[start + 3 : start + 4].strip()
    if resname4:
        return f"{resname}{resname4}"
    return resname


def read_siz(filename):
    """
    Reads a size configuration file (.siz) and returns a dictionary of atomic sizes.

    Parses a custom size configuration file format to define atomic radii.
    Each line in the file defines a size for an atom based on atom name,
    residue name, and chain.

    File format is as follows:
        - Atom name: columns 1-4
        - Residue name: columns 7-9, with optional 4th character in column 10
        - Chain: column 11
        - Size value: columns 13-end
        - Lines starting with '!', '#', or 'ATOM' are ignored as comments.
        - Inline comments starting with '!' or '#' are also ignored.

    Args:
        filename (str): The path to the size configuration file (.siz).

    Returns:
        dict: A dictionary of atomic sizes.
              Keys are tuples of (atom name, residue name, chain).
              Values are tuples of (ignore flags, size value).
              - ignore flags (tuple): A tuple of 4 booleans indicating whether
                each field (atom name, residue name, chain, size key) is ignored
                in a particular size definition.
              - size value (float): The atomic size value defined in the file.

    Raises:
        ValueError: If no identifying fields (atom name, residue name, chain)
                    are provided for a size record.
        ValueError: If the size value is not a positive number (<= 0.0).
    """
    sizes = {}
    with open(filename) as fin:
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue
            if not (ln.upper().startswith(("!", "#", "ATOM"))):
                # Remove inline comments
                ln_data = ln.split("!", 1)[0].split("#", 1)[0].strip()

                atomname = ln_data[0:4].strip()
                resname = _read_reserved_resname(ln_data, 6)
                chain = ln_data[10:11].strip()
                size_value = float(ln_data[12:].strip())

                # Ignore flags for atomname, resname, chain, size key
                ignore_k = [atomname == "", resname == "", chain == "", True]

                # Check if any field is supplied
                has_size_key = not all(ignore_k[:3])
                ignore_k[3] = has_size_key

                if not has_size_key:
                    raise ValueError(
                        "At least one of the fields (atom name, residue name, chain) "
                        "must be supplied for size record. "
                        f"Line: {ln}"
                    )

                if size_value <= 0.0:
                    raise ValueError(
                        "\n".join(
                            [
                                "Atom radius must be a positive number.",
                                f"Invalid size record in line: {ln}",
                            ]
                        )
                    )

                sizes[(atomname, resname, chain)] = (tuple(ignore_k), size_value)

    return sizes


def read_crg(filename):
    """
    Reads a charge configuration file (.crg) and returns a dictionary of atomic charges.

    Parses a custom charge configuration file format to define atomic charges.
    Each line in the file defines a charge for an atom based on atom name,
    residue name, chain, and residue number.

    File format is as follows:
        - Atom name: columns 1-4
        - Residue name: columns 7-9, with optional 4th character in column 10
        - Chain: column 11
        - Residue number: columns 12-15
        - Charge value: columns 16-end
        - Lines starting with '!', '#', or 'ATOM' are ignored as comments.
        - Inline comments starting with '!' or '#' are also ignored.

    Args:
        filename (str): The path to the charge configuration file (.crg).

    Returns:
        dict: A dictionary of atomic charges.
              Keys are tuples of (atom name, residue name, chain, residue number).
              Values are tuples of (ignore flags, charge value).
              - ignore flags (tuple): A tuple of 5 booleans indicating whether
                each field (atom name, residue name, chain, residue number, charge key)
                is ignored in a particular charge definition.
              - charge value (float): The atomic charge value defined in the file.

    Raises:
        ValueError: If no identifying fields (atom name, residue name, chain,
                    residue number) are provided for a charge record.
    """
    charges = {}
    with open(filename) as fin:
        for line in fin:
            line = line.strip()
            if not line or line.upper().startswith(("!", "#", "ATOM")):
                continue

            # Remove inline comments
            if "!" in line:
                line = line.split("!", 1)[0].strip()
            elif "#" in line:
                line = line.split("#", 1)[0].strip()

            atom_name = line[0:4].strip()
            residue_name = _read_reserved_resname(line, 6)
            chain = line[10:11].strip()
            residue_number_str = line[11:15].strip()

            residue_number = (
                int(residue_number_str) if residue_number_str else RES_NUMBER_UNKNOWN
            )
            charge_value = float(line[15:].strip())

            # Ignore flags: Determine if each field is supplied
            ignore_flags = [
                not atom_name,
                not residue_name,
                not chain,
                residue_number == RES_NUMBER_UNKNOWN,
            ]

            has_size_key = not all(ignore_flags[:4])
            ignore_flags.append(has_size_key)

            if not has_size_key:
                raise ValueError(
                    "At least one of the fields (atom name, residue name, chain, "
                    "residue number) must be supplied for charge record"
                )

            charges[(atom_name, residue_name, chain, residue_number)] = (
                tuple(ignore_flags),
                charge_value,
            )

    return charges


def read_vdw(filename):
    """
    Reads a van der Waals parameter file and returns a dictionary of vdW parameters.

    Parses a fixed-width formatted van der Waals (vdW) parameter file where each line
    defines sigma, epsilon, and gamma values for an atom type.

    File format specification:
        - Comment lines begin with '!' and are ignored.
        - The header line starts with 'atom_' (case-insensitive) and marks the start of data.
        - Each subsequent line contains the following fixed-width fields:

            Columns (1-based)   Field       Type    Units         Description
            ------------------  ----------  ------  ------------  -----------------------------
            1–6                atom name   str     —             Atom type name (left-aligned)
            7–14               sigma       float   Å             Lennard-Jones sigma
            15–22              epsilon     float   kT (298 K)    Lennard-Jones epsilon
            23–30              gamma       float   kT/Å²         Lennard-Jones gamma parameter

        - All fields are mandatory.
        - Field widths are respected; values are extracted via string slicing.

        Example:
            ! Comment line
            atom__sigma___epsilon_gamma___
            N       3.2500  0.2871  1.0000
            H       1.0691  0.0265  1.0000

    Args:
        filename (str): Path to the van der Waals parameter file.

    Returns:
        dict: A dictionary mapping atom names (str) to tuples:
              (sigma: float, epsilon: float, gamma: float)

    Raises:
        ValueError: If a data line is malformed or contains non-numeric values.
    """
    vdw_data = {}
    data_started = False

    with open(filename, "r") as file:
        for line_num, line in enumerate(file, start=1):
            if not line.strip() or line.lstrip().startswith("!"):
                continue

            if not data_started:
                if line.lower().startswith("atom_"):
                    data_started = True
                continue

            # Ensure the line is long enough to contain all fields
            if len(line) < 30:
                raise ValueError(f"Line {line_num} too short: {line.rstrip()}")

            atom_name = line[0:6].strip()
            try:
                sigma = float(line[6:15].strip())
                epsilon = float(line[15:23].strip())
                gamma = float(line[23:30].strip())
            except ValueError:
                raise ValueError(
                    f"Invalid numeric values on line {line_num}: {line.rstrip()}"
                )

            vdw_data[atom_name] = (sigma, epsilon, gamma)
    # print("vdw_data>>", vdw_data)
    return vdw_data


def read_gaussian_sigma(filename):
    """
    Reads a Gaussian sigma configuration file and returns a dictionary of sigma values.

    Parses a custom Gaussian sigma configuration file format where each line defines
    a Gaussian sigma value (positive float) for an atom based on atom name,
    residue name, chain, and residue number.

    File format is as follows:
        - Atom name:        columns 1–4   (left-aligned)
        - Residue name:     columns 7–9   (left-aligned), with optional 4th character in column 10
        - Chain:            column 11     (single character)
        - Residue number:   columns 12–15 (right-aligned)
        - Sigma value:      columns 16–end (float, must be positive)
        - Lines starting with '!', '#', or 'ATOM' are ignored as comments.
        - Inline comments starting with '!' or '#' are also stripped.

    Args:
        filename (str): Path to the Gaussian sigma configuration file.

    Returns:
        dict: A dictionary of Gaussian sigma values.
              Keys are tuples of (atom name, residue name, chain, residue number).
              Values are tuples of (ignore flags, sigma value).
              - ignore flags (tuple): A tuple of 5 booleans indicating whether
                each field (atom name, residue name, chain, residue number, size key)
                is ignored in a particular sigma record.
              - sigma value (float): The Gaussian sigma value (must be positive).

    Raises:
        ValueError: If no identifying fields are provided, or if sigma value is invalid.
    """
    sigmas = {}

    with open(filename) as fin:
        for line_num, line in enumerate(fin, start=1):
            line = line.strip()
            if not line or line.upper().startswith(("!", "#", "ATOM")):
                continue

            # Remove inline comments
            if "!" in line:
                line = line.split("!", 1)[0].strip()
            elif "#" in line:
                line = line.split("#", 1)[0].strip()

            atom_name = line[0:4].strip()
            residue_name = _read_reserved_resname(line, 6)
            chain = line[10:11].strip()
            residue_number_str = line[11:15].strip()

            residue_number = (
                int(residue_number_str) if residue_number_str else RES_NUMBER_UNKNOWN
            )

            sigma_str = line[15:].strip()
            if not sigma_str:
                raise ValueError(f"Missing sigma value on line {line_num}: {line}")

            try:
                sigma_value = float(sigma_str)
                if sigma_value <= 0.0:
                    raise ValueError
            except ValueError:
                raise ValueError(
                    f"Invalid or non-positive sigma value on line {line_num}: {line}"
                )

            # Ignore flags
            ignore_flags = [
                not atom_name,
                not residue_name,
                not chain,
                residue_number == RES_NUMBER_UNKNOWN,
            ]

            has_size_key = not all(ignore_flags[:4])
            ignore_flags.append(has_size_key)

            if not has_size_key:
                raise ValueError(
                    f"Line {line_num}: At least one of the fields (atom name, residue name, "
                    f"chain, residue number) must be supplied for sigma record"
                )

            sigmas[(atom_name, residue_name, chain, residue_number)] = (
                tuple(ignore_flags),
                sigma_value,
            )

    return sigmas


@dataclass(frozen=True)
class ZphiCompareResult:
    """Structured result returned by compare_zphi()."""

    passed: bool
    report: Dict[str, Union[str, int, float, bool, None]]


def _as_zphi_data(value: Union[ZphiData, str]) -> ZphiData:
    """Accept either a ZphiData object or a filename."""
    if isinstance(value, ZphiData):
        return value
    return read_zphi(value)


def _zphi_float_diff_record(
    *,
    field: str,
    ref_value: float,
    out_value: float,
    rtol: float,
    atol: float,
    index: Optional[Tuple[int, int, int]] = None,
) -> Dict[str, Union[str, int, float, bool, None]]:
    """
    Build a float-comparison record.

    Pass rule:
        abs(out - ref) < abs(ref) * rtol + atol
    """
    ref_value = float(ref_value)
    out_value = float(out_value)
    delta = out_value - ref_value
    abs_diff = abs(delta)
    allowed = abs(ref_value) * float(rtol) + float(atol)
    denom = max(abs(ref_value), 1.0e-300)

    return {
        "field": field,
        "index_ix": None if index is None else int(index[0]),
        "index_iy": None if index is None else int(index[1]),
        "index_iz": None if index is None else int(index[2]),
        "ref_value": ref_value,
        "out_value": out_value,
        "delta": delta,
        "abs_diff": abs_diff,
        "relative_error": abs_diff / denom,
        "allowed_diff": allowed,
        "pass": abs_diff < allowed,
    }


def _zphi_metadata_int(metadata: Dict[str, str], key: str) -> Optional[int]:
    if key not in metadata:
        return None
    return int(str(metadata[key]).split()[0])


def _zphi_metadata_float(metadata: Dict[str, str], key: str) -> Optional[float]:
    if key not in metadata:
        return None
    return float(str(metadata[key]).split()[0])


def _zphi_metadata_int_vec(metadata: Dict[str, str], key: str) -> Optional[np.ndarray]:
    if key not in metadata:
        return None
    return _zphi_parse_vec(metadata[key], dtype=np.int64, expected_len=3)


def _zphi_metadata_float_vec(
    metadata: Dict[str, str], key: str
) -> Optional[np.ndarray]:
    if key not in metadata:
        return None
    return _zphi_parse_vec(metadata[key], dtype=np.float64, expected_len=3)


def compare_zphi(
    ref: Union[ZphiData, str],
    out: Union[ZphiData, str],
    *,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-4,
    compare_metadata_floats: bool = True,
    compare_context: bool = True,
) -> Tuple[bool, Dict[str, Union[str, int, float, bool, None]]]:
    """
    Compare two canonical sparse .zphi objects or files.

    Parameters
    ----------
    ref, out
        Either filenames or already-read ZphiData objects. The first argument is
        treated as the reference.

    rtol, atol
        Floating tolerance. For every compared float, pass criterion is:

            abs(out - ref) < abs(ref) * rtol + atol

    compare_metadata_floats
        If True, also compare floating metadata such as grid scale, spacing,
        center, origin, surface distance, average, min, and max.

    compare_context
        If True, require CONTEXT_TYPE and CONTEXT_LABEL to match exactly.
        CONTEXT_NUM_ATOMS is compared as an integer when present in either file.

    Integer fields
    --------------
    The following must match exactly:
        - sparse point count
        - GRID_SHAPE
        - INDEX_BASE
        - ZPHI_VERSION
        - sparse grid indices
        - CONTEXT_NUM_ATOMS, when present and compare_context=True

    Returns
    -------
    passed, report
        A boolean pass/fail flag and a report dictionary suitable for regression
        test CSV rows and one-line [AR] summaries.

    Report fields include the largest pointwise potential difference:
        - zphi_worst_ix / iy / iz
        - zphi_worst_ref_phi
        - zphi_worst_out_phi
        - zphi_worst_delta_phi
        - zphi_worst_abs_diff
        - zphi_worst_relative_error
        - zphi_worst_allowed_diff
    """
    ref_zphi = _as_zphi_data(ref)
    out_zphi = _as_zphi_data(out)

    report: Dict[str, Union[str, int, float, bool, None]] = {
        "zphi_pass": True,
        "zphi_rtol": float(rtol),
        "zphi_atol": float(atol),
        "zphi_error": "",
        "zphi_ref_points": int(ref_zphi.num_points),
        "zphi_out_points": int(out_zphi.num_points),
        "zphi_worst_field": None,
        "zphi_worst_ix": None,
        "zphi_worst_iy": None,
        "zphi_worst_iz": None,
        "zphi_worst_ref_phi": None,
        "zphi_worst_out_phi": None,
        "zphi_worst_delta_phi": None,
        "zphi_worst_abs_diff": None,
        "zphi_worst_relative_error": None,
        "zphi_worst_allowed_diff": None,
        "zphi_num_potential_failures": 0,
        "zphi_num_metadata_float_failures": 0,
    }

    errors = []

    def fail(message: str) -> None:
        report["zphi_pass"] = False
        errors.append(message)

    # Exact integer/schema checks.
    if ref_zphi.num_points != out_zphi.num_points:
        fail(
            f"point count mismatch: ref={ref_zphi.num_points}, out={out_zphi.num_points}"
        )

    for key in ["INDEX_BASE", "ZPHI_VERSION"]:
        ref_val = _zphi_metadata_int(ref_zphi.metadata, key)
        out_val = _zphi_metadata_int(out_zphi.metadata, key)
        if ref_val != out_val:
            fail(f"{key} mismatch: ref={ref_val}, out={out_val}")

    ref_shape = _zphi_metadata_int_vec(ref_zphi.metadata, "GRID_SHAPE")
    out_shape = _zphi_metadata_int_vec(out_zphi.metadata, "GRID_SHAPE")
    if ref_shape is None or out_shape is None:
        fail("GRID_SHAPE missing from one or both zphi metadata blocks")
    elif not np.array_equal(ref_shape, out_shape):
        fail(f"GRID_SHAPE mismatch: ref={ref_shape.tolist()}, out={out_shape.tolist()}")

    if compare_context:
        for key in ["CONTEXT_TYPE", "CONTEXT_LABEL"]:
            ref_val = ref_zphi.metadata.get(key)
            out_val = out_zphi.metadata.get(key)
            if ref_val != out_val:
                fail(f"{key} mismatch: ref={ref_val!r}, out={out_val!r}")

        if (
            "CONTEXT_NUM_ATOMS" in ref_zphi.metadata
            or "CONTEXT_NUM_ATOMS" in out_zphi.metadata
        ):
            ref_atoms = _zphi_metadata_int(ref_zphi.metadata, "CONTEXT_NUM_ATOMS")
            out_atoms = _zphi_metadata_int(out_zphi.metadata, "CONTEXT_NUM_ATOMS")
            if ref_atoms != out_atoms:
                fail(f"CONTEXT_NUM_ATOMS mismatch: ref={ref_atoms}, out={out_atoms}")

    # If point counts differ, exact index/potential comparison is not meaningful.
    if ref_zphi.num_points != out_zphi.num_points:
        report["zphi_error"] = "; ".join(errors)
        return bool(report["zphi_pass"]), report

    # Grid indices must be identical and in the same order.
    if not np.array_equal(ref_zphi.indices, out_zphi.indices):
        mismatch_mask = np.any(ref_zphi.indices != out_zphi.indices, axis=1)
        first = int(np.flatnonzero(mismatch_mask)[0])
        ref_idx = ref_zphi.indices[first]
        out_idx = out_zphi.indices[first]
        fail(
            "grid index mismatch at row "
            f"{first}: ref={ref_idx.tolist()}, out={out_idx.tolist()}"
        )
        report["zphi_worst_field"] = "grid_indices"
        report["zphi_worst_ix"] = int(ref_idx[0])
        report["zphi_worst_iy"] = int(ref_idx[1])
        report["zphi_worst_iz"] = int(ref_idx[2])
        report["zphi_error"] = "; ".join(errors)
        return bool(report["zphi_pass"]), report

    # Pointwise potential comparison.
    ref_pot = ref_zphi.potentials.astype(np.float64)
    out_pot = out_zphi.potentials.astype(np.float64)

    delta = out_pot - ref_pot
    abs_diff = np.abs(delta)
    allowed = np.abs(ref_pot) * float(rtol) + float(atol)
    pass_mask = abs_diff < allowed

    num_failures = int(np.count_nonzero(~pass_mask))
    report["zphi_num_potential_failures"] = num_failures
    if num_failures:
        fail(f"{num_failures} potential value(s) exceed tolerance")

    if ref_pot.size:
        worst_i = int(np.argmax(abs_diff))
        worst_idx = ref_zphi.indices[worst_i]
        denom = max(abs(float(ref_pot[worst_i])), 1.0e-300)

        report["zphi_worst_field"] = "phi_kT_per_e"
        report["zphi_worst_ix"] = int(worst_idx[0])
        report["zphi_worst_iy"] = int(worst_idx[1])
        report["zphi_worst_iz"] = int(worst_idx[2])
        report["zphi_worst_ref_phi"] = float(ref_pot[worst_i])
        report["zphi_worst_out_phi"] = float(out_pot[worst_i])
        report["zphi_worst_delta_phi"] = float(delta[worst_i])
        report["zphi_worst_abs_diff"] = float(abs_diff[worst_i])
        report["zphi_worst_relative_error"] = float(abs_diff[worst_i] / denom)
        report["zphi_worst_allowed_diff"] = float(allowed[worst_i])

    # Optional floating metadata comparison.
    worst_meta_record = None
    metadata_float_failures = 0

    if compare_metadata_floats:
        scalar_float_keys = [
            "GRID_SCALE_PER_ANG",
            "GRID_SPACING_ANG",
            "SURFACE_DISTANCE_ANG",
            "SIMPLE_AVERAGE_SURFACE_POTENTIAL",
            "POTENTIAL_MIN",
            "POTENTIAL_MAX",
            "DENSE_EXPORT_MISSING_VALUE",
        ]
        vector_float_keys = [
            "GRID_CENTER_ANG",
            "GRID_ORIGIN_ANG",
        ]

        for key in scalar_float_keys:
            if key not in ref_zphi.metadata and key not in out_zphi.metadata:
                continue
            if key not in ref_zphi.metadata or key not in out_zphi.metadata:
                fail(f"{key} missing from one zphi file")
                metadata_float_failures += 1
                continue

            rec = _zphi_float_diff_record(
                field=key,
                ref_value=_zphi_metadata_float(ref_zphi.metadata, key),
                out_value=_zphi_metadata_float(out_zphi.metadata, key),
                rtol=rtol,
                atol=atol,
            )
            if not rec["pass"]:
                metadata_float_failures += 1
                fail(
                    f"{key} exceeds tolerance: "
                    f"ref={rec['ref_value']}, out={rec['out_value']}, "
                    f"diff={rec['abs_diff']}, allowed={rec['allowed_diff']}"
                )

            if (
                worst_meta_record is None
                or rec["abs_diff"] > worst_meta_record["abs_diff"]
            ):
                worst_meta_record = rec

        for key in vector_float_keys:
            if key not in ref_zphi.metadata and key not in out_zphi.metadata:
                continue
            if key not in ref_zphi.metadata or key not in out_zphi.metadata:
                fail(f"{key} missing from one zphi file")
                metadata_float_failures += 1
                continue

            ref_vec = _zphi_metadata_float_vec(ref_zphi.metadata, key)
            out_vec = _zphi_metadata_float_vec(out_zphi.metadata, key)

            for axis, ref_value, out_value in zip(["x", "y", "z"], ref_vec, out_vec):
                rec = _zphi_float_diff_record(
                    field=f"{key}_{axis}",
                    ref_value=float(ref_value),
                    out_value=float(out_value),
                    rtol=rtol,
                    atol=atol,
                )
                if not rec["pass"]:
                    metadata_float_failures += 1
                    fail(
                        f"{key}_{axis} exceeds tolerance: "
                        f"ref={rec['ref_value']}, out={rec['out_value']}, "
                        f"diff={rec['abs_diff']}, allowed={rec['allowed_diff']}"
                    )

                if (
                    worst_meta_record is None
                    or rec["abs_diff"] > worst_meta_record["abs_diff"]
                ):
                    worst_meta_record = rec

    report["zphi_num_metadata_float_failures"] = int(metadata_float_failures)

    if worst_meta_record is not None:
        report["zphi_worst_metadata_float_field"] = worst_meta_record["field"]
        report["zphi_worst_metadata_ref"] = worst_meta_record["ref_value"]
        report["zphi_worst_metadata_out"] = worst_meta_record["out_value"]
        report["zphi_worst_metadata_delta"] = worst_meta_record["delta"]
        report["zphi_worst_metadata_abs_diff"] = worst_meta_record["abs_diff"]
        report["zphi_worst_metadata_relative_error"] = worst_meta_record[
            "relative_error"
        ]
        report["zphi_worst_metadata_allowed_diff"] = worst_meta_record["allowed_diff"]

    report["zphi_error"] = "; ".join(errors)
    return bool(report["zphi_pass"]), report


def read_zeta_phi(zeta_filename):
    """
    Reads the zeta surface potentials file and reconstructs:
      - grid_center
      - surf_grid_coords (flat list: [px1, py1, pz1, ...])
      - surf_grid_index (flat list: [ix1, iy1, iz1, ...]) with placeholder -1 (unknown)
      - num_surf_grid_coords
      - surf_grid_potentials
    """
    coords = []
    potentials = []
    grid_center = None
    simple_avg_potential = None

    with open(zeta_filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if line.startswith("# REMARK SIMPLE AVERAGE SURFACE POTENTIAL"):
                    try:
                        simple_avg_potential = float(line.split("=")[1].split()[0])
                    except Exception:
                        pass
                elif line.startswith("# REMARK GRIDBOX GEOMETRIC CENTER (ANG)"):
                    try:
                        parts = line.split("=")[1].split()
                        grid_center = tuple(map(float, parts))
                    except Exception:
                        pass
                continue

            try:
                px, py, pz, potential = map(float, line.split(","))
                coords.append((px, py, pz))
                potentials.append(potential)
            except ValueError:
                raise ValueError(f"Malformed data line in {zeta_filename}: {line}")

    coords = np.array(coords, dtype=float)
    potentials = np.array(potentials, dtype=float)

    surf_grid_coords = coords.flatten().tolist()
    surf_grid_index = [-1] * len(surf_grid_coords)  # Unknown from file
    num_surf_grid_coords = coords.shape[0]

    return (
        grid_center,
        surf_grid_coords,
        surf_grid_index,
        num_surf_grid_coords,
        potentials,
        simple_avg_potential,
    )


def read_grid_charges(filename):
    """
    Reads grid charges file and reconstructs:
      - scale
      - grid_origin
      - grid_shape
      - unique_charged_gridpoints (Nx5 array: [index_1d, total_charge, ix, iy, iz])

    Correct linear index:
        index_1d = (ix * ny + iy) * nz + iz
    """
    meta = {}
    data = []
    include_indices = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "Grid Scale" in line:
                    meta["scale"] = float(line.split(":")[1])
                elif "Grid Spacing" in line:
                    meta["grid_spacing"] = float(line.split(":")[1])
                elif "Grid Origin" in line:
                    vals = line.split(":")[1].split()
                    meta["grid_origin"] = tuple(map(float, vals))
                elif "Grid Shape" in line:
                    vals = line.split(":")[1].split()
                    meta["grid_shape"] = tuple(map(int, vals))
                continue

            parts = line.split("\t")
            if include_indices is None:
                include_indices = len(parts) == 7

            data.append(
                [float(x) if "." in x or "e" in x.lower() else int(x) for x in parts]
            )

    data = np.array(data)

    if include_indices:
        ix = data[:, 0].astype(int)
        iy = data[:, 1].astype(int)
        iz = data[:, 2].astype(int)
        charges = data[:, 6]
        nx, ny, nz = meta["grid_shape"]
        index_1d = (ix * ny + iy) * nz + iz
        unique_charged_gridpoints = np.column_stack([index_1d, charges, ix, iy, iz])
    else:
        raise ValueError("Cannot reconstruct ix, iy, iz from file without indices.")

    return (
        meta["scale"],
        np.array(meta["grid_origin"]),
        np.array(meta["grid_shape"]),
        unique_charged_gridpoints,
    )


def read_induced_surface_charges(filename):
    """
    Reads induced surface charges file and reconstructs:
      - scale
      - grid_origin
      - grid_shape
      - induced_surf_charges_flat (1D array: [ix, iy, iz, charge, ix2, iy2, iz2, charge2, ...])
    """
    meta = {}
    data = []
    include_indices = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if "Grid Scale" in line:
                    meta["scale"] = float(line.split(":")[1])
                elif "Grid Spacing" in line:
                    meta["grid_spacing"] = float(line.split(":")[1])
                elif "Grid Origin" in line:
                    vals = line.split(":")[1].split()
                    meta["grid_origin"] = tuple(map(float, vals))
                elif "Grid Shape" in line:
                    vals = line.split(":")[1].split()
                    meta["grid_shape"] = tuple(map(int, vals))
                continue

            parts = line.split("\t")
            if include_indices is None:
                include_indices = len(parts) == 7

            data.append(
                [float(x) if "." in x or "e" in x.lower() else int(x) for x in parts]
            )

    data = np.array(data)

    if include_indices:
        ix = data[:, 0].astype(int)
        iy = data[:, 1].astype(int)
        iz = data[:, 2].astype(int)
        charges = data[:, 6]
        induced_surf_charges_flat = np.column_stack([ix, iy, iz, charges]).flatten()
    else:
        raise ValueError("Cannot reconstruct ix, iy, iz from file without indices.")

    return (
        meta["scale"],
        np.array(meta["grid_origin"]),
        np.array(meta["grid_shape"]),
        induced_surf_charges_flat,
    )
