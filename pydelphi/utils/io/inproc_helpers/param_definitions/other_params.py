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
    MemoryState,
)

from pydelphi.utils.io.inproc_helpers.param_definitions.parameters import (
    ParameterGroup,
    ParamStatement,
)


def get_group_definition():
    """Defines and returns the 'pb' ParameterGroup."""
    return ParameterGroup(
        "other",
        "The set of un-grouped parameters.",
        "The set of un-grouped parameters.",
    )


def get_param_definitions():
    """Defines and returns PB-related ParamStatement objects."""
    params = {}

    params[("absolute_temperature", "temperature", "temper")] = ParamStatement(
        full_name="absolute_temperature",
        long_name="temperature",
        short_name="temper",
        units="Kelvin: K",
        dtype=float,
        default=297.15,
        min_value=1.0,
        max_value=1000.0,
        desc_short="Temeperature of the system",
        desc_long="Temeperature of the system",
        override=True,
        required=True,
    )

    params[("memory_state", "memorystate", "memstate")] = ParamStatement(
        full_name="memory_state",
        long_name="memorystate",
        short_name="memstate",
        units=None,
        dtype=MemoryState,
        default=MemoryState.MINIMAL,
        min_value=None,
        max_value=None,
        desc_short="Memory state of maps during the execution.",
        desc_long="Memory state of maps during the execution.",
        override=True,
        required=False,
    )

    return params
