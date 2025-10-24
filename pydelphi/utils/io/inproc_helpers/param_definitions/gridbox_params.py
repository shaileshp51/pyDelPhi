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

from pydelphi.foundation.enums import (
    SoluteExtremaRule,
    GridBoxType,
    GridboxSize,
)

from pydelphi.utils.io.inproc_helpers.param_definitions.parameters import (
    ParameterGroup,
    ParamStatement,
)


def get_group_definition():
    """Defines and returns the 'pb' ParameterGroup."""
    return ParameterGroup(
        "gridbox",
        "The set of parameters related to define the grixbox type and its dimensions for solving PBE.",
        "The set of parameters related to define the grixbox type and its dimensions for solving PBE.",
    )


def get_param_definitions():
    """Defines and returns PB-related ParamStatement objects."""
    params = {}

    params[("solute_extrema", "soluteextrema", "sextrm")] = ParamStatement(
        full_name="solute_extrema",
        long_name="soluteextrema",
        short_name="sextrm",
        units=None,
        dtype=SoluteExtremaRule,
        default=SoluteExtremaRule.COORDLIMITS,
        min_value=None,
        max_value=None,
        desc_short='Solute extrema rule: choose from: {"COORDINATE", "COORDLIMITS"}. (default: "COORDLIMITS")',
        desc_long='Solute extrema rule: choose from: {"COORDINATE", "COORDLIMITS"}. (default: "COORDLIMITS")',
        override=False,
        required=False,
    )

    params[("gridbox_type", "gridboxtype", "gbtype")] = ParamStatement(
        full_name="gridbox_type",
        long_name="gridboxtype",
        short_name="gbtype",
        units=None,
        dtype=GridBoxType,
        default=GridBoxType.CUBIC,
        min_value=None,
        max_value=None,
        desc_short='Tpye of the gridbox: choose from: {"CUBIC", "CUBOIDAL"}. (default: "CUBIC")',
        desc_long='Tpye of the gridbox: choose from: {"CUBIC", "CUBOIDAL"}. (default: "CUBIC")',
        override=False,
        required=False,
    )

    params[("grid_scale", "scale", "scale")] = ParamStatement(
        full_name="grid_scale",
        long_name="scale",
        short_name="scale",
        units=r"per angstrom: $1.0/\AA$",
        dtype=float,
        default=2,
        min_value=1.0,
        max_value=20.0,
        desc_short="Number of grids per angstrom",
        desc_long="Number of grids per angstrom",
        override=True,
        required=True,
    )

    params[("percent_fill", "percentfill", "perfil")] = ParamStatement(
        full_name="percent_fill",
        long_name="percentfill",
        short_name="perfil",
        units="percent: (%)",
        dtype=float,
        default=70,
        min_value=1.0,
        max_value=99.0,
        desc_short="Percent of gridbox occupied by solute along its X-, Y-, Z-dimentions.",
        desc_long="""Percent of gridbox occupied by solute along its X-, Y-, Z-dimentions.
                    if gridbox_type is CUBIC:
                        the cube side length is decided by the longest dimension
                    if gridbox_type is CUBOID:
                        cuboids side lengths are decided by solutes X-, Y-, Z-dimentions
        """,
        override=True,
        required=False,
    )

    params[("gridbox_margin", "solute_padding", "gbmargin")] = ParamStatement(
        full_name="gridbox_margin",
        long_name="solute_padding",
        short_name="gbmargin",
        units=r"angstrom: $\AA$",
        dtype=float,
        default=15.0,
        min_value=1.0,
        max_value=100.0,
        desc_short="Minimum space (in angstroms) between boundary-grids and any solute coordinates.",
        desc_long="Minimum space (in angstroms) between boundary-grids and any solute coordinates.",
        override=True,
        required=False,
    )

    params[("grid_size", "gridsize", "gsize")] = ParamStatement(
        full_name="grid_size",
        long_name="gridsize",
        short_name="gsize",
        units=None,
        dtype=GridboxSize,
        default=GridboxSize(0),
        min_value=GridboxSize(1),
        max_value=GridboxSize(1000),
        desc_short="Number of grids along each of X-, Y-, Z-dimensions.",
        desc_long="Number of grids along each of X-, Y-, Z-dimensions.",
        override=True,
        required=False,
    )

    return params
