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


def read_siz(filename):
    """
    Reads a size configuration file (.siz) and returns a dictionary of atomic sizes.

    Parses a custom size configuration file format to define atomic radii.
    Each line in the file defines a size for an atom based on atom name,
    residue name, and chain.

    File format is as follows:
        - Atom name: columns 1-4
        - Residue name: columns 7-9
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
                resname = ln_data[6:9].strip()
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
        - Residue name: columns 7-9
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
            residue_name = line[6:9].strip()
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
        - Residue name:     columns 7–9   (left-aligned)
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
            residue_name = line[6:9].strip()
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
