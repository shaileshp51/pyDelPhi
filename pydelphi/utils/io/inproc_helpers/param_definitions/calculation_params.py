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
        "calculation",
        "The set of parameters for specifying the set of calculation to perform.",
        "The set of parameters for specifying the set of calculation to perform.",
    )


def get_param_definitions():
    """Defines and returns PB-related ParamStatement objects."""
    params = {}

    fun_acenter = ParamFunction(
        name="acent", alias="ac", attributes=[], active=False, required=False
    )
    fun_acenter.add_attribute(
        ParamFunctionAttribute(
            name="x",
            alias="x",
            desc="x-coordinate",
            required=True,
            nameonly=False,
            value=0.0,
        )
    )
    fun_acenter.add_attribute(
        ParamFunctionAttribute(
            name="y",
            alias="y",
            desc="y-coordinate",
            required=True,
            nameonly=False,
            value=0.0,
        )
    )
    fun_acenter.add_attribute(
        ParamFunctionAttribute(
            name="z",
            alias="z",
            desc="z-coordinate",
            required=True,
            nameonly=False,
            value=0.0,
        )
    )
    params[("acenter", "acent", "ac")] = fun_acenter

    fun_energy = ParamFunction(
        name="calculate_energies",
        alias="energy",
        attributes=[],
        desc_short="Delphi function to specify the energy components to be computed and output in energy file",
    )
    fun_energy.add_attribute(
        ParamFunctionAttribute(
            name="grid",
            alias="g",
            desc="grid energy",
            required=False,
            nameonly=True,
        )
    )
    fun_energy.add_attribute(
        ParamFunctionAttribute(
            name="polar",
            alias="p",
            desc="polar solvation/reaction field energy",
            required=False,
            nameonly=True,
        )
    )
    fun_energy.add_attribute(
        ParamFunctionAttribute(
            name="coulombic",
            alias="c",
            desc="coulombic energy",
            required=False,
            nameonly=True,
        )
    )
    fun_energy.add_attribute(
        ParamFunctionAttribute(
            name="ionic",
            alias="i",
            desc="coulombic energy",
            required=False,
            nameonly=True,
        )
    )
    fun_energy.add_attribute(
        ParamFunctionAttribute(
            name="nonpolar",
            alias="np",
            desc="non-polar energy",
            required=False,
            nameonly=True,
        )
    )
    fun_energy.add_attribute(
        ParamFunctionAttribute(
            name="lj", alias="lj", desc="vdW energy", required=False, nameonly=True
        )
    )
    params[("calculate_energies", "energies", "energy")] = fun_energy

    fun_site = ParamFunction(
        name="site",
        alias="site",
        attributes=[],
        desc_short=(
            "Delphi function to reports the potentials and electrostatic "
            "field components at the positions of the subset of atoms "
            "specified in the frc output file."
        ),
    )
    fun_site.add_attribute(
        ParamFunctionAttribute(
            name="atom",
            alias="a",
            desc="atoms description",
            required=False,
            nameonly=True,
        )
    )
    fun_site.add_attribute(
        ParamFunctionAttribute(
            name="charge",
            alias="q",
            desc="charge positions",
            required=False,
            nameonly=True,
        )
    )
    fun_site.add_attribute(
        ParamFunctionAttribute(
            name="potential",
            alias="p",
            desc="potential",
            required=False,
            nameonly=True,
        )
    )
    fun_site.add_attribute(
        ParamFunctionAttribute(
            name="field", alias="f", desc="field", required=False, nameonly=True
        )
    )
    fun_site.add_attribute(
        ParamFunctionAttribute(
            name="reaction",
            alias="r",
            desc="reaction-field",
            required=False,
            nameonly=True,
        )
    )
    fun_site.add_attribute(
        ParamFunctionAttribute(
            name="coulombic",
            alias="c",
            desc="coulombic",
            required=False,
            nameonly=True,
        )
    )
    fun_site.add_attribute(
        ParamFunctionAttribute(
            name="coordinates",
            alias="x",
            desc="coordinates",
            required=False,
            nameonly=True,
        )
    )
    fun_site.add_attribute(
        ParamFunctionAttribute(
            name="salt", alias="i", desc="salt", required=False, nameonly=True
        )
    )
    fun_site.add_attribute(
        ParamFunctionAttribute(
            name="total",
            alias="t",
            desc="total energy",
            required=False,
            nameonly=True,
        )
    )
    fun_site.add_attribute(
        ParamFunctionAttribute(
            name="reactionforce",
            alias="rf",
            desc="reaction-force",
            required=False,
            nameonly=True,
        )
    )
    fun_site.add_attribute(
        ParamFunctionAttribute(
            name="coulombforce",
            alias="cf",
            desc="coulomb-force",
            required=False,
            nameonly=True,
        )
    )
    fun_site.add_attribute(
        ParamFunctionAttribute(
            name="surfacecharge",
            alias="sf",
            desc="surface-charge",
            required=False,
            nameonly=True,
        )
    )
    fun_site.add_attribute(
        ParamFunctionAttribute(
            name="totalforce",
            alias="tf",
            desc="total-force",
            required=False,
            nameonly=True,
        )
    )
    #  *    ATOMICPOT         or ATPO
    #  *    DEBYEFRACTION     or DEB
    #  *    MOLECULARDYNAMICS or MDF
    params[("site", "site", "site")] = fun_site

    return params
