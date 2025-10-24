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
        "convergence",
        "The formulation of Poisson Boltzmann Equation to solve.",
        "The formulation of Poisson Boltzmann Equation to solve.",
    )


def get_param_definitions():
    """Defines and returns PB-related ParamStatement objects."""
    params = {}

    params[("max_delta_phi", "maxdeltaphi", "maxc")] = ParamStatement(
        full_name="max_delta_phi",
        long_name="maxdeltaphi",
        short_name="maxc",
        units="kT/e",
        dtype=float,
        default=1.0e-3,
        min_value=1.0e-6,
        max_value=1.0e0,
        desc_short="Maximum change allowed for any grid in two succesive iterations to decide convergence.",
        desc_long="Maximum change allowed for any grid in two succesive iterations to decide convergence.",
        override=True,
        required=False,
    )

    params[("max_rmsd", "maxrmsd", "maxr")] = ParamStatement(
        full_name="max_rmsd",
        long_name="maxrmsd",
        short_name="maxr",
        units="kT/e",
        dtype=float,
        default=5.0e-5,
        min_value=1.0e-14,
        max_value=1.0e0,
        desc_short="Maximum rsmd change allowed in two succesive iterations to decide convergence.",
        desc_long="Maximum rsmd change allowed in two succesive iterations to decide convergence.",
        override=True,
        required=False,
    )

    # nonlinear_convergence_tolerance
    params[
        (
            "max_nonlinear_coupling_delta_phi",
            "maxnonlinearcouplingdeltaphi",
            "maxnlcpldphi",
        )
    ] = ParamStatement(
        full_name="max_nonlinear_coupling_delta_phi",
        long_name="maxnonlinearcouplingdeltaphi",
        short_name="maxnlcpldphi",
        units="kT/e",
        dtype=float,
        default=1.0e-2,
        min_value=1.0e-6,
        max_value=1.0e0,
        desc_short="Maximum change allowed for any grid in two successive non-linear iterations to decide convergence.",
        desc_long="Maximum change allowed for any grid in two successive non-linear iterations to decide convergence.",
        override=True,
        required=False,
    )

    return params
