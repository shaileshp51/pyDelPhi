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

from os import path

from pydelphi.constants import (
    ATOMFIELD_CHARGE,
    ATOMFIELD_RADIUS,
    ATOMFIELD_GAUSS_SIGMA,
    ATOMFIELD_ATOMIC_NUMBER,
    ATOMFIELD_LJ_SIGMA,
    ATOMFIELD_LJ_EPSILON,
    ATOMFIELD_LJ_GAMMA,
)
from pydelphi.config.logging_config import NOTICE, WARNING, get_effective_verbosity
from pydelphi.config.global_runtime import vprint

MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(MODULE_NAME)

from pydelphi.utils.io.readers import read_pdb, read_pqr, read_siz, read_crg, read_vdw


def _read_atomic_data(in_modpdb4, in_pdb, in_siz, in_crg):
    """
    Reads atomic data from modpdb4 (PQR) or raw PDB input files, and assigns
    size and charge if using PDB.

    Args:
        in_modpdb4: Input handle for preprocessed PQR file.
        in_pdb: Input handle for raw PDB file.
        in_siz: Input handle for atomic radii (.siz).
        in_crg: Input handle for atomic charges (.crg).

    Returns:
        atoms (dict): Dictionary of atom records with size and charge fields.
        objects (list): List of auxiliary objects (e.g., geometric-shaped beads).
    """
    atoms = {}
    objects = []

    if in_modpdb4.issupplied:
        file_format = in_modpdb4.get_attribute("format").lower()
        file_path = in_modpdb4.get_attribute("file")

        if not path.isfile(file_path):
            raise FileNotFoundError(f"❌ modpdb4 file not found: {file_path}")

        if file_format != "pqr":
            raise ValueError(
                f"❌ Unsupported modpdb4 format: '{file_format}'. Expected 'pqr'."
            )

        atoms, objects = read_pqr(file_path)

    elif in_pdb.issupplied:
        file_path = in_pdb.get_attribute("file")

        if not path.isfile(file_path):
            raise FileNotFoundError(f"❌ PDB file not found: {file_path}")

        if not (in_siz.issupplied and in_crg.issupplied):
            raise ValueError(
                "❌ InputError: 'in_pdb' requires both 'in_siz' and 'in_crg' to assign atomic sizes and charges."
            )

        atoms, objects = read_pdb(file_path)
        _assign_size(atoms, in_siz)
        _assign_charge(atoms, in_crg)

    else:
        raise ValueError(
            "❌ InputError: Neither 'in_modpdb4' nor 'in_pdb' was supplied."
        )

    return atoms, objects


def _assign_size(atoms, in_size):
    if not (in_size.issupplied and path.isfile(in_size.get_attribute("file"))):
        msg = "With pdb required size param is missing. Check inputs and retry."
        if in_size.issupplied:
            msg = f"siz file: {in_size.get_attribute('file')}"
        raise FileNotFoundError(msg)

    sizes = read_siz(in_size.get_attribute("file"))
    for atom_key, atom_data in atoms.items():
        found = False
        for size_key, (ignore_match, vdw_radius) in sizes.items():
            if (
                ignore_match[3]
                and (ignore_match[0] or atom_key[3] == size_key[0])
                and (ignore_match[1] or atom_key[4] == size_key[1])
                and (ignore_match[2] or atom_key[5] == size_key[2])
            ):
                atom_data[ATOMFIELD_RADIUS] = vdw_radius
                found = True
                break
        if not found:
            vprint(
                WARNING,
                _VERBOSITY,
                f"WARNING>> unassigned size: atom({atom_key}, {atom_data})",
            )


def _assign_charge(atoms, in_charge):
    if not (in_charge.issupplied and path.isfile(in_charge.get_attribute("file"))):
        msg = "With pdb required charge param is missing. Check inputs and retry."
        if in_charge.issupplied:
            msg = f"crg file: {in_charge.get_attribute('file')}"
        raise FileNotFoundError(msg)

    charges = read_crg(in_charge.get_attribute("file"))
    for atom_key, atom_data in atoms.items():
        found = False
        for charge_key, (ignore_match, charge) in charges.items():
            if (
                ignore_match[4]
                and (ignore_match[0] or atom_key[3] == charge_key[0])
                and (ignore_match[1] or atom_key[4] == charge_key[1])
                and (ignore_match[2] or atom_key[5] == charge_key[2])
                and (ignore_match[3] or atom_key[6] == charge_key[3])
            ):
                atom_data[ATOMFIELD_CHARGE] = charge
                found = True
                break
        if not found:
            vprint(
                WARNING,
                _VERBOSITY,
                f"WARNING>> unassigned charge: atom({atom_key}, {atom_data})",
            )


def _assign_vdw(atoms, in_vdw):
    if not (in_vdw.issupplied and path.isfile(in_vdw.get_attribute("file"))):
        msg = "For VDW energy required param in(vdw,file='filename') is missing. Check inputs and retry."
        if in_vdw.issupplied:
            msg = f"vdw file: {in_vdw.get_attribute('file')}"
        raise FileNotFoundError(msg)

    vdw_values = read_vdw(in_vdw.get_attribute("file"))
    for atom_key, atom_data in atoms.items():
        found = False

        for vdw_key, vdw_par_values in vdw_values.items():
            if atom_key[3] == vdw_key:
                atom_data[ATOMFIELD_LJ_SIGMA] = vdw_par_values[0]
                atom_data[ATOMFIELD_LJ_EPSILON] = vdw_par_values[1]
                atom_data[ATOMFIELD_LJ_GAMMA] = vdw_par_values[2]
                found = True
                break
        if not found:
            vprint(
                WARNING,
                _VERBOSITY,
                f"WARNING>> unassigned vdw: atom({atom_key[3]}: {'|'.join([str(t) for t in atom_key])})",
            )


def _set_param_func_attributes(
    param_obj, attributes_list, expected_names=None, is_float=True, file_check=None
):
    """
    Parses and sets function-style parameter attributes on a param object.

    Args:
        param_obj: The parameter object to update.
        attributes_list: List of "key=value" strings or just values.
        expected_names: If provided, zips with positional values.
        is_float: Whether to cast values to float.
        file_check: 'in', 'out', or None — validates 'file' attribute accordingly.

    Raises:
        ValueError or FileNotFoundError
    """
    if expected_names:
        for atb_name, attribute in zip(expected_names, attributes_list):
            try:
                if "=" in attribute:
                    atbs = [a.strip() for a in attribute.split("=")]
                    value = float(atbs[1]) if is_float else atbs[1]
                    param_obj.set_attribute(atbs[0].lower(), value)
                else:
                    value = float(attribute) if is_float else attribute
                    param_obj.set_attribute(atb_name, value)
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid attribute for '{atb_name}': {attribute}"
                ) from e
    else:
        for attribute in attributes_list:
            atbs = [a.strip() for a in attribute.split("=")]
            if len(atbs) == 2:
                key = atbs[0].lower()
                value = atbs[1].replace("'", "").replace('"', "")

                if key == "file":
                    value = "./" + value if not value.startswith(("./", "/")) else value

                    if file_check == "in":
                        if not path.isfile(value):
                            raise FileNotFoundError(
                                f"Input file '{value}' does not exist."
                            )
                    elif file_check == "out":
                        out_dir = path.dirname(value) or "."
                        if not path.isdir(out_dir):
                            raise FileNotFoundError(
                                f"File directory '{out_dir}' does not exist."
                            )

                param_obj.set_attribute(key, value)

            elif len(atbs) == 1:
                param_obj.set_attribute(atbs[0].lower())
            else:
                raise ValueError(f"Ambiguous attribute values: {atbs}")

    param_obj.supplied()
