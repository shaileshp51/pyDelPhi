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
    BioModel,
    PBApproximation,
)

from pydelphi.utils.io.inproc_helpers.param_definitions.parameters import (
    ParameterGroup,
    ParamStatement,
)


def get_group_definition():
    """Defines and returns the 'pb' ParameterGroup."""
    return ParameterGroup(
        "pb",
        "The formulation of Poisson Boltzmann Equation to solve.",
        "The formulation of Poisson Boltzmann Equation to solve.",
    )


def get_param_definitions():
    """Defines and returns PB-related ParamStatement objects."""
    params = {}

    params[("bio_model", "biomodel", "biomodel")] = ParamStatement(
        full_name="bio_model",
        long_name="biomodel",
        short_name="biomodel",
        units=None,
        dtype=BioModel,
        default=BioModel.PBE,
        min_value=None,
        max_value=None,
        desc_short='Poisson Boltzmann form to solve: choices {"PBE", "RPBE"}, (default: PBE)',
        desc_long='Poisson Boltzmann form to solve: choices {"PBE", "RPBE"}, (default: PBE)',
        override=False,
        required=False,
    )

    params[("pb_approximation", "pbapproximation", "pbapprox")] = ParamStatement(
        full_name="pb_approximation",
        long_name="pbapproximation",
        short_name="pbapprox",
        units=None,
        dtype=PBApproximation,
        default=PBApproximation.LINEAR,
        min_value=None,
        max_value=None,
        desc_short='Poisson Boltzmann approximation to solve: choices {"LINEAR", "NONLINEAR"}, (default: LINEAR)',
        desc_long="""Poisson Boltzmann approximation to solve: choices {"LINEAR", "NONLINEAR"}, (default: LINEAR),
        NOTE: if max_nonlinear_iteration is set non-zero, this param will be overridden to NONLINEAR.
        """,
        override=False,
        required=False,
    )

    return params
