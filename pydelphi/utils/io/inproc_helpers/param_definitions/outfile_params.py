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


from pydelphi.utils.io.inproc_helpers.param_definitions.parameters import (
    ParameterGroup,
    ParamFunction,
    ParamFunctionAttribute,
)


def get_group_definition():
    """Defines and returns the 'outfile' ParameterGroup."""
    return ParameterGroup(
        "outfile",
        "Parameters for specifying output files to write.",
        "Parameters for specifying output files to write.",
    )


def get_param_definitions():
    """Defines and returns output-related ParamFunction objects."""
    params = {}

    # ----------------------------
    # out(energy, file=...)
    # ----------------------------
    out_energy = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Output energy components to a file.",
        desc_long="Output function to write calculated energy components to a file.",
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
            desc="output file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("out__energy", "out__energy", "out__energy")] = out_energy

    # ----------------------------
    # out(surf, file=..., format=..., precision=...)
    # ----------------------------
    out_surf = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Output solute-surface map to a file.",
        desc_long="Output function to write the solute-surface map.",
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
            desc="output file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    out_surf.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc=(
                "output file format. options: {auto, cube, phi}. "
                "Default: auto. When auto, infer from file extension; explicit format is authoritative."
            ),
            required=False,
            nameonly=False,
            value="auto",
        )
    )
    out_surf.add_attribute(
        ParamFunctionAttribute(
            name="precision",
            alias="prec",
            desc=(
                "precision for phi-format output. options: {single, double}. default: single.\n"
                "\tNOTE: This can differ from calculation precision. Single is usually sufficient for output."
            ),
            required=False,
            nameonly=False,
            value="single",
        )
    )
    params[("out__surf", "out__surf", "out__surf")] = out_surf

    # ----------------------------
    # out(density, file=..., point=..., format=..., precision=...)
    # ----------------------------
    out_density = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Output Gaussian density to a file.",
        desc_long="Output function to write Gaussian density data.",
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
            desc="output file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    out_density.add_attribute(
        ParamFunctionAttribute(
            name="point",
            alias="point",
            desc="choose the point for which density to write. options: {grid, mid, both}. default: grid",
            required=False,
            nameonly=False,
            value="grid",
        )
    )
    out_density.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc=(
                "output file format. options: {auto, cube, phi}. "
                "Default: auto. When auto, infer from file extension; explicit format is authoritative."
            ),
            required=False,
            nameonly=False,
            value="auto",
        )
    )
    out_density.add_attribute(
        ParamFunctionAttribute(
            name="precision",
            alias="prec",
            desc=(
                "precision for phi-format output. options: {single, double}. default: single.\n"
                "\tNOTE: This can differ from calculation precision. Single is usually sufficient for output."
            ),
            required=False,
            nameonly=False,
            value="single",
        )
    )
    params[("out__density", "out__density", "out__density")] = out_density

    # ----------------------------
    # out(phi, file=..., format=..., precision=..., media=...)
    # ----------------------------
    out_phi = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Output potential (phi) map to a file.",
        desc_long="An output function to write potential (phi) map data.",
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
            desc="output file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    out_phi.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc=(
                "output file format. options: {auto, cube, phi}. "
                "Default: auto. When auto, infer from file extension; explicit format is authoritative."
            ),
            required=False,
            nameonly=False,
            value="auto",
        )
    )
    out_phi.add_attribute(
        ParamFunctionAttribute(
            name="precision",
            alias="prec",
            desc=(
                "precision for phi-format output. options: {single, double}. default: single.\n"
                "\tNOTE: This can differ from calculation precision. Single is usually sufficient for output.\n"
                "\tUse double for cases like parent runs of focusing."
            ),
            required=False,
            nameonly=False,
            value="single",
        )
    )
    out_phi.add_attribute(
        ParamFunctionAttribute(
            name="media",
            alias="phase",
            desc="choose the media for which phimap to write. options: {water, vacuum, both}. default: water",
            required=False,
            nameonly=False,
            value="water",
        )
    )
    params[("out__phi", "out__phi", "out__phi")] = out_phi

    # ----------------------------
    # out(zphi, file=...)
    # ----------------------------
    out_zphi = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Output zeta-potential map to a file.",
        desc_long="Output function to write zeta-potential map data.",
    )
    out_zphi.add_attribute(
        ParamFunctionAttribute(
            name="zphi",
            alias="zphi",
            desc="write zeta-potential map file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    out_zphi.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="output file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("out__zphi", "out__zphi", "out__zphi")] = out_zphi

    # ----------------------------
    # out(eps, file=..., format=..., media=..., point=..., precision=...)
    # ----------------------------
    out_eps = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Output dielectric (eps) map to a file.",
        desc_long="Output function to write dielectric (eps) map data.",
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
            desc="output file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    out_eps.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc=(
                "output file format. options: {auto, cube, phi}. "
                "Default: auto. When auto, infer from file extension; explicit format is authoritative."
            ),
            required=False,
            nameonly=False,
            value="auto",
        )
    )
    out_eps.add_attribute(
        ParamFunctionAttribute(
            name="media",
            alias="phase",
            desc="choose the media for which epsmap to write. options: {water, vacuum, both}. default: water",
            required=False,
            nameonly=False,
            value="water",
        )
    )
    out_eps.add_attribute(
        ParamFunctionAttribute(
            name="point",
            alias="point",
            desc="choose the point for which epsmap to write. options: {grid, mid, both}. default: grid",
            required=False,
            nameonly=False,
            value="grid",
        )
    )
    out_eps.add_attribute(
        ParamFunctionAttribute(
            name="precision",
            alias="prec",
            desc=(
                "precision for phi-format output. options: {single, double}. default: single.\n"
                "\tNOTE: This can differ from calculation precision. Single is usually sufficient for output.\n"
                "\tUse double for cases like parent runs of focusing."
            ),
            required=False,
            nameonly=False,
            value="single",
        )
    )
    params[("out__eps", "out__eps", "out__eps")] = out_eps

    # ----------------------------
    # out(modpdb4, file=..., format=...)
    # ----------------------------
    out_modpdb4 = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Output modified structure file (PDB/PQR).",
        desc_long="Output function to write a modified structure file (typically PQR).",
    )
    out_modpdb4.add_attribute(
        ParamFunctionAttribute(
            name="modpdb4",
            alias="modpdb4",
            desc="write modified structure file",
            required=True,
            nameonly=True,
        )
    )
    out_modpdb4.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="output file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    out_modpdb4.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc=(
                "output file format. options: {auto, pqr, pdb}. "
                "Default: auto. When auto, infer from file extension; explicit format is authoritative."
            ),
            required=False,
            nameonly=False,
            value="auto",
        )
    )
    params[("out__modpdb4", "out__modpdb4", "out__modpdb4")] = out_modpdb4

    # ----------------------------
    # out(selection, name=..., file=..., format=...)
    # ----------------------------
    out_selection = ParamFunction(
        name="out",
        alias="write",
        attributes=[],
        desc_short="Output a named selection as a structure file (PDB/PQR).",
        desc_long=(
            "Output function to write a named selection (defined via select(name=...)) "
            "to a structure file (typically PQR)."
        ),
        multicall=True,
    )

    out_selection.add_attribute(
        ParamFunctionAttribute(
            name="selection",
            alias="sel",
            desc="write a named selection as a structure file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )

    out_selection.add_attribute(
        ParamFunctionAttribute(
            name="selname",
            alias="name",
            desc="selection name (must match a previously defined select(name=...))",
            required=True,
            nameonly=False,
            value="",
        )
    )

    out_selection.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="output file name (default: '<name>.<fmt>')",
            required=False,  # <-- changed
            nameonly=False,
            value="",
        )
    )

    out_selection.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc=(
                "output file format. options: {auto, pqr, pdb}. "
                "Default: auto. When auto, infer from file extension; explicit format is authoritative."
            ),
            required=False,
            nameonly=False,
            value="auto",
        )
    )

    params[("out__selection", "out__selection", "out__sel")] = out_selection

    return params
