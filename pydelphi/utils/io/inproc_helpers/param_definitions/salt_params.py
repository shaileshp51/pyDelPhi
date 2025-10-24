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
        "salt",
        "The set of parameters related to salt concentration vallance and radii.",
        "The set of parameters related to salt concentration vallance and radii.",
    )


def get_param_definitions():
    """Defines and returns PB-related ParamStatement objects."""
    params = {}

    params[("salt_concentration", "saltconcentration", "salt")] = ParamStatement(
        full_name="salt_concentration",
        long_name="saltconcentration",
        short_name="salt",
        units="Molar: [M]",
        dtype=float,
        default=0.0,
        min_value=0.0,
        max_value=100.0,
        override=True,
        desc_short="Concentaration of salt in Molar units.",
        desc_long="Concentaration of salt in Molar units.",
        required=True,
    )

    params[("ions_radii", "ionsradii", "ionrad")] = ParamStatement(
        full_name="ions_radii",
        long_name="ionsradii",
        short_name="ionrad",
        units=r"angstrom: $\AA$",
        dtype=float,
        default=2.0,
        min_value=0.1,
        max_value=100.0,
        override=True,
        desc_short="Radius of the salt ions in angstrom units.",
        desc_long="Radius of the salt ions in angstrom units.",
        required=True,
    )

    params[("salt_valance", "saltvalance", "saltvalance")] = ParamStatement(
        full_name="salt_valance",
        long_name="saltvalance",
        short_name="saltvalance",
        units=None,
        dtype=int,
        default=1,
        min_value=1,
        max_value=6,
        desc_short="Concentaration of salt in Molar units.",
        desc_long="Concentaration of salt in Molar units.",
        override=True,
        required=True,
    )

    return params
