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
    ParamStatement,
)


def get_group_definition():
    """Defines and returns the 'pb' ParameterGroup."""
    return ParameterGroup(
        "zeta",
        "The set of parameters related to calculation of surface potential at a distance from vdW surface.",
        "The set of parameters related to calculation of surface potential at a distance from vdW surface.",
    )


def get_param_definitions():
    """Defines and returns PB-related ParamStatement objects."""
    params = {}

    params[("zeta_potential", "zetapotential", "zp")] = ParamStatement(
        full_name="zeta_potential",
        long_name="surfacepotential",
        short_name="surfpot",
        units=None,
        dtype=bool,
        default=False,
        min_value=False,
        max_value=True,
        desc_short="Whether to compute zeta-surface potential.",
        desc_long="""When zeta_potential is set to True, the surface potential 
                is calculated at a distance (set by zeta_distance) from the molecular surface.""",
        override=True,
        required=False,
    )

    params[("zeta_distance", "zetadistance", "zd")] = ParamStatement(
        full_name="zeta_distance",
        long_name="zetadistance",
        short_name="zd",
        units="Angstrom",
        dtype=float,
        default=0,
        min_value=0,
        max_value=10,
        desc_short="The distance of the zeta-surface for potential calculation from molecular surface.",
        desc_long="The distance of the zeta-surface for potential calculation from molecular surface.",
        override=True,
        required=False,
    )

    return params
