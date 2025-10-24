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

import numpy as np
from pydelphi.foundation.enums import (
    DielectricModel,
)

from pydelphi.utils.io.inproc_helpers.param_definitions.parameters import (
    ParameterGroup,
    ParamStatement,
)


def get_group_definition():
    """Defines and returns the 'pb' ParameterGroup."""
    return ParameterGroup(
        "dielectric",
        "The set of parameters related to choosing the dielectric model and setting values of relevant paramters.",
        "The set of parameters related to choosing the dielectric model and setting values of relevant paramters.",
    )


def get_param_definitions():
    """Defines and returns PB-related ParamStatement objects."""
    params = {}

    params[("dielectric_model", "dielectricmodel", "gaussian")] = ParamStatement(
        full_name="dielectric_model",
        long_name="dielectricmodel",
        short_name="gaussian",
        units=None,
        dtype=DielectricModel,
        default=DielectricModel.TWODIELECTRIC,
        min_value=None,
        max_value=None,
        desc_short='Whether to use Gaussian model: {"TWODIELECTRIC", "GAUSSIAN"}, (default: TWODIELECTRIC)',
        desc_long="",
        override=False,
        required=False,
    )

    params[("gaussian_exponent", "gaussianmultiplier", "gm")] = ParamStatement(
        full_name="gaussian_exponent",
        long_name="gaussianmultiplier",
        short_name="gm",
        units=None,
        dtype=int,
        default=1,
        min_value=1,
        max_value=100,
        desc_short="Exponent of the gaussian function in the gaussian dielectric model",
        desc_long="Exponent of the gaussian function in the gaussian dielectric model",
        override=True,
        required=False,
    )

    params[("gaussian_sigma", "gaussiansigma", "sigma")] = ParamStatement(
        full_name="gaussian_sigma",
        long_name="gaussiansigma",
        short_name="sigma",
        units=None,
        dtype=float,
        default=1.0,
        min_value=0.1,
        max_value=10.0,
        desc_short="Variance of the Gaussian model",
        desc_long="Variance of the Gaussian model",
        override=True,
        required=False,
    )

    params[("midpoint_dielectric_gaussian", "midpointdielmethod", "middim")] = (
        ParamStatement(
            full_name="midpoint_dielectric_gaussian",
            long_name="midpointdielmethod",
            short_name="middim",
            units=None,
            dtype=np.uint8,
            default=True,
            min_value=False,
            max_value=True,
            desc_short="Whether to use gaussian density dependent analytical dielcetric at midpoints.",
            desc_long="""Whether to use gaussian density dependent analytical dielcetric 
                at midpoints between two grids (bool: True/False; default: True)""",
            override=True,
            required=False,
        )
    )

    params[("internal_dielectric", "indi", "indi")] = ParamStatement(
        full_name="internal_dielectric",
        long_name="indi",
        short_name="indi",
        units=None,
        dtype=float,
        default=1,
        min_value=1.0,
        max_value=100.0,
        desc_short="Internal dielectric (range: 1-80)",
        desc_long="",
        override=True,
        required=True,
    )

    params[("external_dielectric", "externaldielectric", "exdi")] = ParamStatement(
        full_name="external_dielectric",
        long_name="externaldielectric",
        short_name="exdi",
        units=None,
        dtype=float,
        default=80,
        min_value=1.0,
        max_value=100.0,
        desc_short="Solvent dielectric",
        override=True,
        desc_long="",
        required=True,
    )
    # Note: external dielectric value is a solvent property, so also added to solvent param_group

    params[("gap_dielectric", "gapdielectric", "gapdi")] = ParamStatement(
        full_name="gap_dielectric",
        long_name="gapdielectric",
        short_name="gapdi",
        units=None,
        dtype=float,
        default=80,
        min_value=1.0,
        max_value=100.0,
        desc_short="Maximum solute or Gap dielectric (range: 1-80)",
        desc_long="Maximum solute or Gap dielectric (range: 1-80)",
        override=True,
        required=False,
    )

    return params
