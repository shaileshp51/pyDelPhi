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
        "solvent",
        "The set of parameters for specifying the solvent properties.",
        "The set of parameters for specifying the solvent properties.",
    )


def get_param_definitions():
    """Defines and returns PB-related ParamStatement objects."""
    params = {}

    params[("probe_radius", "proberadius", "prbrad")] = ParamStatement(
        full_name="probe_radius",
        long_name="proberadius",
        short_name="prbrad",
        units=r"angstrom: $\AA$",
        dtype=float,
        default=1.4,
        min_value=0.0,
        max_value=20.0,
        override=True,
        desc_short="Probe radius for defining solvent accessible surface.",
        desc_long="Probe radius for defining solvent accessible surface.",
        required=True,
    )

    params[("probe_radius2", "proberadius2", "prbrad2")] = ParamStatement(
        full_name="probe_radius2",
        long_name="proberadius2",
        short_name="prbrad2",
        units=r"angstrom: $\AA$",
        dtype=float,
        default=1.4,
        min_value=0.0,
        max_value=20.0,
        override=True,
        desc_short="Probe radius2 for defining solvent accessible surface.",
        desc_long="Probe radius2 for defining solvent accessible surface.",
        required=True,
    )

    return params
