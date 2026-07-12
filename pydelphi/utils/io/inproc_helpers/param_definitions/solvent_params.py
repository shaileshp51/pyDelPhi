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

from pydelphi.foundation.bib_manager import cite

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

    # Developer note (radius_offset):
    # - We use R_eff = R_vdw + radius_offset when building atom_props, so
    #   radius_offset flows into α = K / R_eff² and thus changes both:
    #   (i) order-1 Gaussian volumes and (ii) all higher-order overlap volumes
    #   and radii-derivative surface terms.
    # - This mirrors the calibration strategy in Chakravorty et al. (2019),
    #   where Gaussian widths are tuned so that Gaussian volume/area match
    #   hard-sphere/geometric references (see Fig. 6).
    # - Use probe_radius to control neighbor/interstitial detection (geometry/
    #   solvent resolution). Use radius_offset only to tune the underlying
    #   Gaussian model, not to mimic grid-based DelPhi quirks.
    params[("radius_offset", "radiusoffset", "ro")] = ParamStatement(
        full_name="radius_offset",
        long_name="radiusoffset",
        short_name="ro",
        units=r"angstrom: $\AA$",
        dtype=float,
        default=0.0,
        min_value=0.0,
        max_value=10.0,
        override=True,
        desc_short=(
            "Additive offset to van der Waals radii used in the Gaussian-atom "
            "nonpolar volume/surface model."
        ),
        desc_long=(
            "Additive offset to the van der Waals radius of each atom used in the "
            "Gaussian-based Grant & Pickup-like model for molecular volume and "
            "surface area.\n\n"
            "The effective radius is defined as R_eff = R_vdw + radius_offset and "
            "enters the Gaussian width via α = K / R_eff². As a result, "
            "radius_offset affects the integrated Gaussian volume, overlap "
            "volumes, and radii-derivative-based surface areas, and therefore the "
            "nonpolar solvation energy.\n\n"
            "This parameter can be used to calibrate the Gaussian model so that "
            "Gaussian-derived volumes and areas better match hard-sphere or "
            "reference geometric methods (see Chakravorty et al., J. Comput. "
            f"Chem. 40, 1290–1304 (2019) {cite('Chakravorty2019')}, especially Fig. 6). "
            "The default (0.0 Å) leaves radii unchanged; small positive values "
            "(∼0.1–0.2 Å) may improve agreement for some systems but should be "
            "treated as a model parameter, not a fitted constant.\n\n"
            "Note that radius_offset modifies the Gaussian shape itself; it is "
            "distinct from probe_radius, which controls how much solvent-accessible "
            "or interstitial volume is counted via neighbor and surface filtering."
        ),
        required=True,
    )

    params[("pressure_coefficient", "pressurecoeff", "pc")] = ParamStatement(
        full_name="pressure_coefficient",
        long_name="pressurecoeff",
        short_name="pc",
        units=r"kT·Å⁻³",
        dtype=float,
        default=1.0,
        min_value=0.01,
        max_value=10.0,
        override=True,
        desc_short="Pressure-like coefficient p in the nonpolar term G_np = γA + pV.",
        desc_long=(
            "Pressure-like coefficient p in the nonpolar solvation free energy model "
            "G_np = γA + pV. Units are kT·Å⁻³. Values around 1.0 are typical for "
            "kT-based nonpolar models."
        ),
        required=True,
    )

    return params
