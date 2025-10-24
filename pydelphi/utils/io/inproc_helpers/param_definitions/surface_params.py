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
    SurfaceMethod,
    PBApproximation,
)

from pydelphi.utils.io.inproc_helpers.param_definitions.parameters import (
    ParameterGroup,
    ParamStatement,
)


def get_group_definition():
    """Defines and returns the 'pb' ParameterGroup."""
    return ParameterGroup(
        "surface",
        "The set of parameters related to define the surface describing the solute solvent interface.",
        "The set of parameters related to define the surface describing the solute solvent interface.",
    )


def get_param_definitions():
    """Defines and returns PB-related ParamStatement objects."""
    params = {}

    params[("surface_method", "surfacemethod", "surfmethod")] = ParamStatement(
        full_name="surface_method",
        long_name="surfacemethod",
        short_name="surfmethod",
        units="",
        dtype=SurfaceMethod,
        default=SurfaceMethod.VDW,
        min_value=None,
        max_value=None,
        desc_short='Method for defining solute and solvent regions: choices {"GCS", "GAUSSIAN", "VDW"}, (default: GAUSSIAN)',
        desc_long='Method for defining solute and solvent regions: choices {"GCS", "GAUSSIAN", "VDW"}, (default: GAUSSIAN)',
        override=False,
        required=False,
    )

    params[("surface_offset", "surfaceoffset", "surfoff")] = ParamStatement(
        full_name="surface_offset",
        long_name="surfaceoffset",
        short_name="surfoff",
        units=r"Angstrom: $\AA$",
        dtype=float,
        default=0.0,
        min_value=0.0,
        max_value=10.0,
        desc_short="extend the vdW radii of atoms by this value and use it for gaussian denity "
        "calculation for surface generation using gaussian surface method.",
        desc_long="extend the vdW radii of atoms by this value and use it for gaussian denity "
        "calculation for surface generation using gaussian surface method.",
        override=True,
        required=False,
    )

    params[("surface_density_exponent", "surfdenexp", "surfexp")] = ParamStatement(
        full_name="surface_density_exponent",
        long_name="surfdenexp",
        short_name="surfexp",
        units=None,
        dtype=int,
        default=4,
        min_value=3,
        max_value=10,
        desc_short="extend the vdW of atoms by this value and use it for gaussian denity "
        "calculation for surface generation using gaussian surface method",
        desc_long="",
        override=True,
        required=False,
    )

    params[("density_cutoff", "densitycutoff", "dencut")] = ParamStatement(
        full_name="density_cutoff",
        long_name="densitycutoff",
        short_name="dencut",
        units=None,
        dtype=float,
        default=0.759,
        min_value=0.001,
        max_value=1.0,
        desc_short="Density cutoff used in PBE/Gaussian to define solute bounday for vacuum run",
        desc_long="Density cutoff used in PBE/Gaussian to define solute bounday for vacuum run",
        override=True,
        required=False,
    )

    params[("surface_cutoff", "surfacecutoff", "surfcut")] = ParamStatement(
        full_name="surface_cutoff",
        long_name="surfacecutoff",
        short_name="surfcut",
        units=None,
        dtype=float,
        default=1.0,
        min_value=1.0,
        max_value=80.0,
        desc_short="Dielectric cutoff used in PBE/Gaussian to define solute bounday for vacuum run",
        desc_long="Dielectric cutoff used in PBE/Gaussian to define solute bounday for vacuum run",
        override=True,
        required=False,
    )

    return params
