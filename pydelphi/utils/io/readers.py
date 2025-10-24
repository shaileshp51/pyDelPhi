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

"""
This module provides functions for reading various file formats commonly used
in molecular modeling and bioinformatics, particularly within the Delphi software
suite. It includes readers for:

- Gaussian cube files (.cube, .phi): Scalar data grids for visualizing properties
  like electrostatic potential. Supports both text and binary formats with
  endianness and marker size auto-detection for binary files.
- Protein Data Bank files (.pdb): Atomic coordinates and basic structural information.
- PQR files (.pqr): PDB files with added charge and radius information for each atom.
- Size configuration files (.siz): Custom atom size definitions.
- Charge configuration files (.crg): Custom atom charge definitions.
- Van der Waals parameter files (.vdw): (To be implemented)
- Sigma parameter files (.sigma): (To be implemented)
- Force center files (.frc): Atomic coordinate files used for force field calculations.

These functions are designed to parse specific file formats and return structured
data, often in the form of dictionaries or numpy arrays, suitable for further
processing within the pydelphi package or other scientific applications.
"""

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


from pydelphi.utils.io.format.assorted.custom_reader import (
    read_crg,
    read_siz,
    read_vdw,
    read_gaussian_sigma,
)

from pydelphi.utils.io.format.pdb_pqr import read_pdb, read_pqr
from pydelphi.utils.io.format.cube.cube_io import read_cube


def calculate_center_of_frc_atoms(filename, fallback_center_offset, scale):
    """
    Calculates the domain center for focusing run calculations from a force center file (.frc).

    Parses a force center file (.frc), specifically used in focusing run calculations
    to define the calculation domain, to determine the central position of atoms.
    This central position, referred to as the domain center or focus center,
    is typically the mean coordinate of all ATOM/HETATM entries. It is crucial
    for setting up the grid and defining the focus of these calculations.

    If no atoms are found in the FRC file, the function can utilize a fallback
    offset vector under specific mode conditions dictated by `fallback_center_offset`.

    Args:
        filename (str): Path to the force center file (.frc). In the context of
            focusing run calculations, FRC files are used to specify the domain
            and atomic positions for these calculations.
        fallback_center_offset (np.ndarray): A 3D offset vector (shape (3,))
            that dictates fallback behavior and domain center calculation modes.
            The behavior is controlled by specific values within this array:

            - **Box-Centered Atom Modes (Exception if no atoms):**
              - When `fallback_center_offset[0]` is set to
                `BOX_CENTERED_ATOM_ANG` or `BOX_CENTERED_ATOM_GRID`, the function
                **requires** atom entries in the FRC file to calculate the domain center.
                If no atoms are found, an `Exception` is raised, as a domain center
                cannot be determined from atoms in this mode.
                - `BOX_CENTERED_ATOM_ANG` (999.0):  Domain center calculation is based on the
                  *average position of all atoms* in the FRC file, typically in Angstrom units.
                - `BOX_CENTERED_ATOM_GRID` (777.0): Domain center calculation is based on the
                  *average position of all atoms* in the FRC file, typically in grid units.
            - **First Atom Center Mode (Specialized for Focusing Runs):**
              - If `fallback_center_offset[1]` is set to
                `FIRST_ATOM_CENTER_ANG` or `FIRST_ATOM_CENTER_GRID`, the function
                immediately terminates reading atom data after processing the *first*
                ATOM/HETATM entry. The domain center is then exclusively determined
                by the coordinates of this initial atom. This mode is intended for
                specific focusing run setups where the first atom's position is sufficient
                to define the domain center.
                - `FIRST_ATOM_CENTER_ANG` (999.0) and `FIRST_ATOM_CENTER_GRID` (777.0)
                  when used in `fallback_center_offset[1]` act as flags to activate
                  this first-atom domain center behavior.
            - **Fallback Offset Mode (No Atom Exception):**
              - If neither of the above domain-centered atom modes are triggered
                (i.e., `fallback_center_offset[0]` is not `BOX_CENTERED_ATOM_ANG`
                or `BOX_CENTERED_ATOM_GRID`), and *no* atoms are found in the FRC file,
                the function will *not* raise an exception. Instead, it will gracefully
                use the `fallback_center_offset` vector, scaled by the `scale` factor,
                as the domain center. This provides a fallback mechanism when atomic data
                is absent or not required for domain centering.

        scale (float): Scaling factor applied to the `fallback_center_offset`
            vector, but **only** when the fallback offset is used as the domain
            center (i.e., when no atoms are found and box-centered atom modes
            are not active). This scaling is relevant when providing a pre-defined
            offset in grid or Angstrom units.

    Returns:
        np.ndarray: A numpy array of shape (3,) representing the calculated
                     domain center coordinates [x, y, z] for focusing run calculations.
                     This is derived as:
                     - The average position of atoms from the FRC file
                       (in box-centered atom mode),
                     - The position of the *first atom* from the FRC file
                       (in first-atom center mode), or
                     - The `fallback_center_offset` vector scaled by `scale`
                       (in fallback offset mode, when no atoms are found and
                        box-centered atom modes are not active).

    Raises:
        FileNotFoundError: If the specified `filename` does not exist or cannot be read.
        Exception: If no atoms are found in the FRC file, and the `fallback_center_offset[0]`
                   value indicates either `BOX_CENTERED_ATOM_ANG` or `BOX_CENTERED_ATOM_GRID` mode,
                   signaling that a domain center calculation from atoms was explicitly
                   required but could not be performed due to the absence of atom data.
    """
    position_center = np.array([0.0, 0.0, 0.0])
    center_position_sum = np.array([0.0, 0.0, 0.0])
    atom_position = np.array([0.0, 0.0, 0.0])
    atom_count = 0

    # Mode flags - Shortest, domain-specific names:
    BOX_CENTERED_ATOM_ANG = (
        999.0  # Domain-centered calculation from average atom position, Angstrom units
    )
    BOX_CENTERED_ATOM_GRID = (
        777.0  # Domain-centered calculation from average atom position, Grid units
    )
    FIRST_ATOM_CENTER_ANG = 999.0  # First atom domain center mode, Angstrom context
    FIRST_ATOM_CENTER_GRID = 777.0  # First atom domain center mode, Grid unit context

    try:
        with open(filename, "r") as infilestream:
            for line in infilestream:
                line = line.strip().upper()

                if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                    continue

                atom_count += 1

                str_sub_line = line[30:54]

                atom_position[0] = float(str_sub_line[0:8])
                atom_position[1] = float(str_sub_line[8:16])
                atom_position[2] = float(str_sub_line[16:24])

                center_position_sum += atom_position

                if (
                    abs(fallback_center_offset[1] - FIRST_ATOM_CENTER_ANG)
                    < ConstDelPhiFloats.ApproxZero.value
                    or abs(fallback_center_offset[1] - FIRST_ATOM_CENTER_GRID)
                    < ConstDelPhiFloats.ApproxZero.value
                ):
                    break

    except FileNotFoundError:
        raise FileNotFoundError(f"FRC file not found: {filename}")

    if atom_count > 0:
        position_center = center_position_sum / atom_count
    else:
        if (
            abs(fallback_center_offset[0] - BOX_CENTERED_ATOM_ANG)
            < ConstDelPhiFloats.ApproxZero.value
        ):
            raise Exception(
                f"No atoms found in FRC file: {filename}. "
                f"Domain center calculation from atoms was enforced (offset mode {BOX_CENTERED_ATOM_ANG})."
            )
        if (
            abs(fallback_center_offset[0] - BOX_CENTERED_ATOM_GRID)
            < ConstDelPhiFloats.ApproxZero.value
        ):
            raise Exception(
                f"No atoms found in FRC file: {filename}. "
                f"Domain center calculation from atoms was enforced (offset mode {BOX_CENTERED_ATOM_GRID})."
            )
        position_center += fallback_center_offset / scale

    return position_center


def _read_atom_data_frc(ifile_stream, format="frc"):
    """Reads atom data from a formatted FRC/PQR/PDB input stream,
    and extracts atom properties based on the specified format.
    Supported formats: "frc", "pqr", "pdb"."""
    str_line = ifile_stream.readline()
    if not str_line:
        return None
    str_line = str_line.strip()
    str_head = str_line[:6].replace("\xa0", " ").strip().upper()
    if str_head not in ("ATOM", "HETATM"):
        return None

    atom_data = np.zeros(LEN_ATOMFIELDS, dtype=delphi_real)
    try:
        crdstr = str_line[30:54]
        atom_data[ATOMFIELD_X] = delphi_real(crdstr[0:8].strip())
        atom_data[ATOMFIELD_Y] = delphi_real(crdstr[8:16].strip())
        atom_data[ATOMFIELD_Z] = delphi_real(crdstr[16:24].strip())
        atom_data[ATOMFIELD_GRID_X:ATOMFIELD_GRID_END] = [0.0, 0.0, 0.0]

        atom_name = str_line[12:16].strip()
        if not atom_name:
            atom_name = "ATOM"

        residue_name = str_line[17:20].strip()
        if not residue_name:
            residue_name = "UNK"

        chain_name = str_line[21:22].strip()
        if not chain_name:
            chain_name = ""

        residue_number_str = str_line[22:26].strip()
        residue_number = (
            int(residue_number_str) if residue_number_str else RES_NUMBER_UNKNOWN
        )

        if format in ("pqr", "frc"):
            chargestr = str_line[54:62].strip()
            if chargestr:
                atom_data[ATOMFIELD_CHARGE] = delphi_real(chargestr)
            else:
                atom_data[ATOMFIELD_CHARGE] = delphi_real(0.0)

            radiusstr = str_line[62:68].strip()
            if radiusstr:
                atom_data[ATOMFIELD_RADIUS] = delphi_real(radiusstr)
            else:
                atom_data[ATOMFIELD_RADIUS] = delphi_real(1.08)

        elif format == "pdb":
            atom_data[ATOMFIELD_CHARGE] = delphi_real(0.0)
            atom_data[ATOMFIELD_RADIUS] = delphi_real(1.08)

        else:
            error_message = f"Error: Unknown format '{format}' provided. Supported formats are 'frc', 'pqr', and 'pdb'."
            vprint(ERROR, _VERBOSITY, error_message)
            raise ValueError(error_message)

        atom_data[ATOMFIELD_GAUSS_SIGMA] = delphi_real(1.0)
        atom_data[ATOMFIELD_RES_KEY] = ResNameToResKey.get(
            residue_name.upper(), ResNameToResKey["UNK"]
        )
        atom_data[ATOMFIELD_ATOMIC_NUMBER] = delphi_real(ConstChemElement.UNK.value)
        atom_data[ATOMFIELD_LJ_SIGMA] = 0.0
        atom_data[ATOMFIELD_LJ_EPSILON] = 0.0
        atom_data[ATOMFIELD_LJ_GAMMA] = 0.0
        atom_data[ATOMFIELD_MEDIA_ID] = 1.0

        return (
            atom_data,
            str_line,
            atom_name,
            residue_name,
            chain_name,
            str(residue_number),
        )

    except ValueError as e:
        vprint(
            WARNING,
            _VERBOSITY,
            f"Could not parse coordinate or other numeric value from line: {str_line.strip()}. Error: {e}",
        )
        return None, None, None, None, None, None


def _get_atom_descriptor(atom_name, residue_name, chain_name, residue_number):
    """Creates formatted atom descriptor string."""
    atom_descriptor = " " * 16
    atom_descriptor = atom_descriptor[:0] + atom_name.ljust(5) + atom_descriptor[5:]
    atom_descriptor = atom_descriptor[:5] + residue_name.ljust(4) + atom_descriptor[9:]
    atom_descriptor = atom_descriptor[:9] + chain_name.ljust(2) + atom_descriptor[11:]
    atom_descriptor = (
        atom_descriptor[:11] + residue_number.ljust(5) + atom_descriptor[16:]
    )
    return atom_descriptor


def read_frc(frc_filepath, format="frc"):
    """
    Reads an FRC file (or related format) to extract atom data efficiently.

    This function processes files in FRC, PQR, or PDB formats to extract
    atom information such as coordinates, charge, and radius.  Minor
    optimizations are included for efficiency.

    Args:
        frc_filepath (str): Path to the input file (FRC, PQR, or PDB).
        format (str, optional): Format of the input file ("frc", "pqr", "pdb").
                                     Defaults to "frc".

    Returns:
        dict: A dictionary containing atom data keyed by atom descriptors.
              Returns an empty dictionary if there's an error or file not found.
    """
    atoms_dict = {}
    format = format.lower()

    try:
        with open(frc_filepath, "r") as ifile_stream:
            atom_num = 0
            try:
                while True:
                    atom_data_tuple = _read_atom_data_frc(ifile_stream, format=format)
                    if atom_data_tuple is None or atom_data_tuple[0] is None:
                        if atom_data_tuple is None:
                            break
                        else:
                            continue
                    (
                        atom_data,
                        str_line,
                        atom_name,
                        residue_name,
                        chain_name,
                        residue_number,
                    ) = atom_data_tuple
                    atom_coords = atom_data[ATOMFIELD_X : ATOMFIELD_Z + 1]
                    str_head = str_line[:6].strip().upper()

                    atom_key = (
                        str_head,
                        "",
                        "",
                        atom_name,
                        residue_name,
                        chain_name,
                        int(residue_number),
                    )
                    atoms_dict[atom_key] = atom_data
                    atom_num += 1

                    atom_descriptor_str = _get_atom_descriptor(
                        atom_name, residue_name, chain_name, residue_number
                    )
                    vprint(
                        TRACE,
                        _VERBOSITY,
                        f"Atom {atom_num}: Coords = {atom_coords}, Atom Data = {atom_data}, Descriptor Components: Name={atom_name}, ResName={residue_name}, Chain={chain_name}, ResNum={residue_number}, Descriptor String = {atom_descriptor_str}, Line = {str_line}, Atom Key = {atom_key}",
                    )
                    vprint(
                        TRACE,
                        _VERBOSITY,
                        f"  Extracted Atom Charge: {atom_data[ATOMFIELD_CHARGE]}, Radius: {atom_data[ATOMFIELD_RADIUS]}",
                    )
            except ValueError as e:
                vprint(ERROR, _VERBOSITY, f"\nError processing FRC file: {e}")
                return {}
    except FileNotFoundError:
        vprint(ERROR, _VERBOSITY, f"Error File not found: {frc_filepath}")
        return atoms_dict
    return atoms_dict
