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

from pydelphi.utils.io.inproc_helpers.param_definitions.parameters import (
    ParameterGroup,
    ParamFunction,
    ParamFunctionAttribute,
)


def get_group_definition():
    """Defines and returns the 'pb' ParameterGroup."""
    return ParameterGroup(
        "infile",
        "The set of parameters for specifying the input files to read.",
        "The set of parameters for specifying the input files to read.",
    )


def get_param_definitions():
    """Defines and returns PB-related ParamStatement objects."""
    params = {}

    in_pdb = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi input function for PDB structure",
        desc_long="Delphi input function for PDB structure",
    )
    in_pdb.add_attribute(
        ParamFunctionAttribute(
            name="pdb",
            alias="pdb",
            desc="read pdb structure",
            required=True,
            nameonly=True,
        )
    )
    in_pdb.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("in_pdb", "in_pdb", "in_pdb")] = in_pdb

    in_siz = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi input function for size",
        desc_long="Delphi input function for size",
    )
    in_siz.add_attribute(
        ParamFunctionAttribute(
            name="siz",
            alias="siz",
            desc="read size file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    in_siz.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("in_siz", "in_size", "in_size")] = in_siz

    in_crg = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi input function for charge",
        desc_long="Delphi input function for charge",
    )
    in_crg.add_attribute(
        ParamFunctionAttribute(
            name="crg",
            alias="crg",
            desc="read charge file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    in_crg.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("in_crg", "in_charge", "in_charge")] = in_crg

    in_vdw = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi input function for Lennard-Jones paramters",
        desc_long="Delphi input function for Lennard-Jones paramters",
    )
    in_vdw.add_attribute(
        ParamFunctionAttribute(
            name="vdw",
            alias="vdw",
            desc="read vdw parameters file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    in_vdw.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("in_vdw", "in_vdw", "in_vdw")] = in_vdw

    in_frc = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi input function for Lennard-Jones paramters",
        desc_long="Delphi input function for Lennard-Jones paramters",
    )
    in_frc.add_attribute(
        ParamFunctionAttribute(
            name="frc",
            alias="frc",
            desc="read frc file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    in_frc.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("in_frc", "in_frc", "in_frc")] = in_frc

    in_modpdb4 = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi input function for PQR",
        desc_long="Delphi input function for PQR",
    )
    in_modpdb4.add_attribute(
        ParamFunctionAttribute(
            name="modpdb4",
            alias="mobpdb4",
            desc="read modifiled pdb format structure",
            required=True,
            nameonly=True,
        )
    )
    in_modpdb4.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    in_modpdb4.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc="file format",
            required=True,
            nameonly=False,
            value="pqr",
        )
    )
    params[("in_modpdb4", "in_modpdb4", "in_modpdb4")] = in_modpdb4

    in_phi = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi function to input",
        desc_long="Delphi function to input",
    )
    in_phi.add_attribute(
        ParamFunctionAttribute(
            name="phi",
            alias="phi",
            desc="read phimap file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    in_phi.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    in_phi.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc="file format",
            required=True,
            nameonly=False,
            value="cube",
        )
    )
    params[("in_phi", "in_phi", "in_phi")] = in_phi

    return params
