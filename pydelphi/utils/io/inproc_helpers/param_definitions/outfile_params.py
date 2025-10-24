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
    ParamFunction,
    ParamFunctionAttribute,
)


def get_group_definition():
    """Defines and returns the 'pb' ParameterGroup."""
    return ParameterGroup(
        "outfile",
        "The set of parameters for specifying the output files to read.",
        "The set of parameters for specifying the output files to read.",
    )


def get_param_definitions():
    """Defines and returns PB-related ParamStatement objects."""
    params = {}

    out_energy = ParamFunction(
        "out",
        "write",
        [],
        "Delphi function to output calculated energy components to a file",
    )
    out_energy.add_attribute(
        ParamFunctionAttribute(
            name="energy",
            alias="energy",
            desc="write energy file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    out_energy.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("out_energy", "out_energy", "out_energy")] = out_energy

    out_surf = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Delphi function to output",
        desc_long="Delphi function to output",
    )
    out_surf.add_attribute(
        ParamFunctionAttribute(
            name="surf",
            alias="surf",
            desc="write solute-surface map file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    out_surf.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    out_surf.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc="output file format.\n options: {cube, phi}. default: cube",
            required=True,
            nameonly=False,
            value="cube",
        )
    )
    out_surf.add_attribute(
        ParamFunctionAttribute(
            name="precision",
            alias="prec",
            desc="precision of the output phi format.\n options: {single, double}. default single.\n "
            "\tNOTE: This could be different than precision of calculation. usually single precision is sufficient for data output thus set default. "
            "\n However, one can override it by asking for double precision which may be useful for parentrun of focusing.",
            required=True,
            nameonly=False,
            value="single",
        )
    )
    params[("out_surf", "out_surf", "out_surf")] = out_surf

    out_density = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Delphi function to output",
        desc_long="Delphi function to output",
    )
    out_density.add_attribute(
        ParamFunctionAttribute(
            name="density",
            alias="density",
            desc="write Gaussian-density file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    out_density.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    out_density.add_attribute(
        ParamFunctionAttribute(
            name="point",
            alias="point",
            desc="choose the point for which density to write.\n options: {grid, mid, both}. default: grid",
            required=True,
            nameonly=False,
            value="grid",
        )
    )
    out_density.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc="output file format.\n options: {cube, phi}. default: cube",
            required=True,
            nameonly=False,
            value="cube",
        )
    )
    out_density.add_attribute(
        ParamFunctionAttribute(
            name="precision",
            alias="prec",
            desc="precision of the output phi format.\n options: {single, double}. default single.\n "
            "\tNOTE: This could be different than precision of calculation. usually single precision is sufficient for data output thus set default. "
            "\n However, one can override it by asking for double precision which may be useful for parentrun of focusing.",
            required=True,
            nameonly=False,
            value="single",
        )
    )
    params[("out_density", "out_density", "out_density")] = out_density

    out_phi = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Delphi function to output",
        desc_long="Delphi function to output",
    )
    out_phi.add_attribute(
        ParamFunctionAttribute(
            name="phi",
            alias="phi",
            desc="write phimap file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    out_phi.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    out_phi.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc="file format",
            required=True,
            nameonly=False,
            value="cube",
        )
    )
    out_phi.add_attribute(
        ParamFunctionAttribute(
            name="precision",
            alias="prec",
            desc="precision of the output phi format.\n options: {single, double}. default single.\n "
            "\tNOTE: This could be different than precision of calculation. usually single precision is sufficient for data output thus set default. "
            "\n However, one can override it by asking for double precision which may be useful for parentrun of focusing.",
            required=True,
            nameonly=False,
            value="single",
        )
    )
    out_phi.add_attribute(
        ParamFunctionAttribute(
            name="media",
            alias="phase",
            desc="choose the media for which phimap to write.\n options: {water,vacuum,both}. default: water",
            required=True,
            nameonly=False,
            value="water",
        )
    )
    params[("out_phi", "out_phi", "out_phi")] = out_phi

    out_zphi = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Delphi function to output",
        desc_long="Delphi function to output",
    )
    out_zphi.add_attribute(
        ParamFunctionAttribute(
            name="zphi",
            alias="zphi",
            desc="write zeta-potential map to file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    out_zphi.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("out_zphi", "out_zphi", "out_zphi")] = out_zphi

    out_eps = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Delphi function to output",
        desc_long="Delphi function to output",
    )
    out_eps.add_attribute(
        ParamFunctionAttribute(
            name="eps",
            alias="eps",
            desc="write epsmap file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    out_eps.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    out_eps.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc="file format",
            required=True,
            nameonly=False,
            value="cube",
        )
    )
    out_eps.add_attribute(
        ParamFunctionAttribute(
            name="media",
            alias="phase",
            desc="choose the media for whch epsmap to write.\n options: {water,vacuum,both}. default: water",
            required=True,
            nameonly=False,
            value="water",
        )
    )
    out_eps.add_attribute(
        ParamFunctionAttribute(
            name="point",
            alias="point",
            desc="choose the point for whch epsmap to write.\n options: {grid,mid,both}. default: grid",
            required=True,
            nameonly=False,
            value="water",
        )
    )
    out_eps.add_attribute(
        ParamFunctionAttribute(
            name="precision",
            alias="prec",
            desc="precision of the output phi format.\n options: {single, double}. default single.\n "
            "\tNOTE: This could be different than precision of calculation. usually single precision is sufficient for data output thus set default. "
            "\n However, one can override it by asking for double precision which may be useful for parentrun of focusing.",
            required=True,
            nameonly=False,
            value="single",
        )
    )
    params[("out_eps", "out_eps", "out_eps")] = out_eps

    out_modpdb4 = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Delphi output function for PQR",
        desc_long="Delphi output function for PQR",
    )
    out_modpdb4.add_attribute(
        ParamFunctionAttribute(
            name="modpdb4",
            alias="mobpdb4",
            desc="write modifiled pdb format structure",
            required=True,
            nameonly=True,
        )
    )
    out_modpdb4.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    out_modpdb4.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc="file format",
            required=True,
            nameonly=False,
            value="pqr",
        )
    )
    params[("out_modpdb4", "out_modpdb4", "out_modpdb4")] = out_modpdb4

    out_frc = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Delphi output function for frc files",
        desc_long="Delphi input function for frc files",
    )
    out_frc.add_attribute(
        ParamFunctionAttribute(
            name="frc",
            alias="frc",
            desc="write frc file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    out_frc.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("out_frc", "out_frc", "out_frc")] = out_frc

    return params
