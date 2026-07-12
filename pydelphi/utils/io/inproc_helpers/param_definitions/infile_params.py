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


from pydelphi.foundation.enums import ParamStatus
from pydelphi.utils.io.inproc_helpers.param_definitions.parameters import (
    ParameterGroup,
    ParamFunction,
    ParamFunctionAttribute,
)


def get_group_definition():
    """Defines and returns the 'infile' ParameterGroup."""
    return ParameterGroup(
        "infile",
        "The set of parameters for specifying the input files to read.",
        "The set of parameters for specifying the input files to read.",
    )


def get_param_definitions():
    """Defines and returns input-file related ParamStatement objects."""
    params = {}

    in_pdb = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi input function for PDB structure",
        desc_long="Delphi input function for PDB structure",
        help_topic="in__pdb",
    )
    in_pdb.add_attribute(
        ParamFunctionAttribute(
            name="pdb",
            alias="pdb",
            desc="read pdb structure",
            required=True,
            nameonly=True,
        )
    )
    in_pdb.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("in__pdb", "in__pdb", "in__pdb")] = in_pdb

    in_crgsiz = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Charge/size set input",
        desc_long=(
            "\nCharge/size set input for topology processing. This is the preferred route for\n"
            "assigning charges and sizes to topology/PDB-derived atoms when a prepared PQR file\n"
            "is not used. Bundled presets are exact residue-name and atom-name lookup tables;\n"
            "they do not perform structure preparation, residue-name translation, protonation-state\n"
            "inference, terminal patching, ligand parameterization, or AMBER/CHARMM conversion.\n"
            "For nonstandard chemistry, modified residues, ligands, cofactors, PTMs, custom \n"
            "states, CHARMM-prepared complex systems, or cross-force-field comparisons, provide\n"
            "a prepared PQR file or protonation use setname=custom with matching charge/size files.\n"
            "The ambiguous preset names amber and charmm are intentionally not accepted; \n"
            "use amber-legacy or charmm-legacy only when reproducing older workflows."
        ),
        multicall=False,
        help_topic="in__crgsiz",
    )

    in_crgsiz.add_attribute(
        ParamFunctionAttribute(
            name="charge_size",
            alias="crgsiz",
            desc=(
                "read charge/size set. Supported bundled presets: "
                "amber-ff99sb-mbondi-set, amber-ff99sb-mbondi2-set, "
                "amber-ff99sb-mbondi3-set, amber-ff14sb-mbondi-set, "
                "amber-ff14sb-mbondi2-set, amber-ff14sb-mbondi3-set, "
                "amber-ff19sb-mbondi-set, amber-ff19sb-mbondi2-set, "
                "amber-ff19sb-mbondi3-set, charmm-c36m-prot-na-pbeq-set, "
                "amber-legacy, charmm-legacy, custom. Use custom only with "
                "user-supplied charge_file and size_file."
            ),
            required=True,
            nameonly=True,
            inuse=True,
        )
    )

    in_crgsiz.add_attribute(
        ParamFunctionAttribute(
            name="setname",
            alias="set",
            desc=(
                "charge/size preset name. Preferred AMBER presets are "
                "amber-ff99sb-{mbondi,mbondi2,mbondi3}-set, "
                "amber-ff14sb-{mbondi,mbondi2,mbondi3}-set, and "
                "amber-ff19sb-{mbondi,mbondi2,mbondi3}-set. "
                "amber-ff99sb-* uses the legacy ff10-era AMBER source stack: "
                "ff99SB protein, ff99bsc0 DNA, ff99sbsc_chiOL3 RNA, Lipid14, TIP3P-compatible ions."
                "amber-ff14sb-* uses ff14SB protein, bsc1 DNA, OL3 RNA, Lipid17, TIP3P-compatible ions. "
                "amber-ff19sb-* uses ff19SB protein, bsc1 DNA, OL3 RNA, Lipid21, TIP3P-compatible ions. "
                "charmm-c36m-prot-na-pbeq-set uses native CHARMM36m/C36 protein and nucleic-acid names "
                "with PBEQ-style protein/nucleic-acid radii; it is for strict CHARMM-conformant "
                "PDB/topology input only and does not include lipids, carbohydrates, "
                "CGenFF, ligands, cofactors, or arbitrary patches. amber-legacy and "
                "charmm-legacy are explicit compatibility presets for old workflows "
                "and are not recommended for new analyses. The names amber, charmm, "
                "and parse are ambiguous and should not be accepted. Default: "
                "amber-ff19sb-mbondi3-set."
            ),
            required=False,
            nameonly=False,
            value="amber-ff19sb-mbondi3-set",
        )
    )

    in_crgsiz.add_attribute(
        ParamFunctionAttribute(
            name="mode",
            alias="mode",
            desc=(
                "how to apply the charge/size set. acquire: fill missing charge/size "
                "values from the selected bundled preset or supplied custom files. "
                "override: replace topology/PQR-provided charge/size values with the "
                "selected set. Use override carefully; for prepared PQR files, embedded "
                "charges/radii are normally authoritative. Default: acquire."
            ),
            required=False,
            nameonly=False,
            value="acquire",
        )
    )

    in_crgsiz.add_attribute(
        ParamFunctionAttribute(
            name="charge_file",
            alias="qfile",
            desc=(
                "custom charge file full path. Allowed only with setname=custom. "
                "The custom charge file must use the same strict residue-name and "
                "atom-name conventions as the input structure/topology. Default: "
                "empty (not supplied)."
            ),
            required=False,
            nameonly=False,
            value="",
        )
    )

    in_crgsiz.add_attribute(
        ParamFunctionAttribute(
            name="size_file",
            alias="rfile",
            desc=(
                "custom size file full path. Allowed only with setname=custom. "
                "The custom size file must use the same strict residue-name and "
                "atom-name conventions as the input structure/topology. Default: "
                "empty (not supplied)."
            ),
            required=False,
            nameonly=False,
            value="",
        )
    )

    params[("in__charge_size", "in__crgsiz", "in__qr")] = in_crgsiz

    in_siz = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi input function for size",
        desc_long="Delphi input function for size",
        status=ParamStatus.DEPRECATED,
        status_desc=(
            "Legacy direct SIZ input is accepted for compatibility and uses "
            "lenient atom-wise assignment.\n"
            "               "
            "Prefer in(crgsiz, setname='custom', qfile='...', rfile='...') "
            "for paired charge/size assignment."
        ),
        help_topic="in__siz",
    )
    in_siz.add_attribute(
        ParamFunctionAttribute(
            name="siz",
            alias="siz",
            desc="read size file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    in_siz.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("in__siz", "in__siz", "in__size")] = in_siz

    in_crg = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi input function for charge",
        desc_long="Delphi input function for charge",
        status=ParamStatus.DEPRECATED,
        status_desc=(
            "Legacy direct CRG input is accepted for compatibility and uses "
            "lenient atom-wise assignment.\n"
            "               "
            "Prefer in(crgsiz, setname='custom', qfile='...', rfile='...') "
            "for paired charge/size assignment."
        ),
        help_topic="in__crg",
    )
    in_crg.add_attribute(
        ParamFunctionAttribute(
            name="crg",
            alias="crg",
            desc="read charge file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    in_crg.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("in__crg", "in__crg", "in__charge")] = in_crg

    in_vdw = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi input function for Lennard-Jones parameters",
        desc_long="Delphi input function for Lennard-Jones parameters",
        status=ParamStatus.DEPRECATED,
        status_desc=(
            "Legacy direct VDW input is accepted for compatibility and uses "
            "a simplified pairwise treatment.\n"
            "               "
            "Future VDW/Lennard-Jones support should be handled through a "
            "dedicated molecular-mechanics backend."
        ),
        help_topic="in__vdw",
    )
    in_vdw.add_attribute(
        ParamFunctionAttribute(
            name="vdw",
            alias="vdw",
            desc="read VDW parameters file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    in_vdw.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("in__vdw", "in__vdw", "in__vdw")] = in_vdw

    in_frc = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi input function for frc files",
        desc_long="Delphi input function for frc files",
        status=ParamStatus.DEPRECATED,
        status_desc=(
            "Legacy direct FRC input is accepted for compatibility.\n"
            "               "
            "Prefer select(...) with frc(source='...', target='...', outfile='...') "
            "for explicit field/response calculations."
        ),
        help_topic="in__frc",
    )
    in_frc.add_attribute(
        ParamFunctionAttribute(
            name="frc",
            alias="frc",
            desc="read frc file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    in_frc.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    params[("in__frc", "in__frc", "in__frc")] = in_frc

    in_modpdb4 = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi input function for PQR",
        desc_long="Delphi input function for PQR",
        help_topic="in__modpdb4",
    )
    in_modpdb4.add_attribute(
        ParamFunctionAttribute(
            name="modpdb4",
            alias="mobpdb4",
            desc="read modifiled pdb format structure",
            required=True,
            nameonly=True,
        )
    )
    in_modpdb4.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    in_modpdb4.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc="file format. options: {auto, pqr, pdb}. default: auto. "
            "When auto, infer from file extension; explicit format is authoritative.",
            required=False,
            nameonly=False,
            value="auto",
        )
    )
    params[("in__modpdb4", "in__modpdb4", "in__modpdb4")] = in_modpdb4

    in_phi = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Delphi function to input",
        desc_long="Delphi function to input",
        help_topic="in__phi",
    )
    in_phi.add_attribute(
        ParamFunctionAttribute(
            name="phi",
            alias="phi",
            desc="read phimap file",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    in_phi.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="file name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    in_phi.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc="file format. options: {auto, cube, phi}. default: auto. "
            "When auto, infer from file extension; explicit format is authoritative.",
            required=False,
            nameonly=False,
            value="auto",
        )
    )
    params[("in__phi", "in__phi", "in__phi")] = in_phi

    in_topol = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Topology input",
        desc_long="Topology input",
        multicall=True,
        help_topic="in__topology",
    )
    in_topol.add_attribute(
        ParamFunctionAttribute(
            name="topology",
            alias="top",
            desc="read topology",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    in_topol.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="topology file",
            required=True,
            nameonly=False,
            value="",
        )
    )
    in_topol.add_attribute(
        ParamFunctionAttribute(
            name="label",
            alias="label",
            desc="ensemble label (e.g., system, complex, receptor, ligand, A, B, AB)",
            required=False,
            nameonly=False,
            value="system",
        )
    )
    in_topol.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc=(
                "topology format. options: {auto, prmtop, pqr, psf, pdb}. "
                "Default: auto. When auto, infer from file extension; explicit format is authoritative. "
                "PSF requires size assignment; PDB requires charge and size assignment."
            ),
            required=False,
            nameonly=False,
            value="auto",
        )
    )
    params[("in__topology", "in__topology", "in__top")] = in_topol

    in_traj = ParamFunction(
        name="in",
        alias="read",
        attributes=[],
        desc_short="Trajectory input",
        desc_long="Trajectory input",
        multicall=True,
        help_topic="in__trajectory",
    )
    in_traj.add_attribute(
        ParamFunctionAttribute(
            name="trajectory",
            alias="traj",
            desc="read trajectory",
            required=True,
            nameonly=True,
            inuse=True,
        )
    )
    in_traj.add_attribute(
        ParamFunctionAttribute(
            name="file",
            alias="file",
            desc="trajectory file",
            required=True,
            nameonly=False,
            value="",
        )
    )
    in_traj.add_attribute(
        ParamFunctionAttribute(
            name="label",
            alias="label",
            desc="ensemble label (e.g., system, complex, receptor, ligand, A, B, AB)",
            required=False,
            nameonly=False,
            value="system",
        )
    )
    in_traj.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc=(
                "trajectory format. options: {auto, netcdf, dcd, trr}. \n"
                "Default: auto. When auto, infer from file extension; explicit format is authoritative."
            ),
            required=False,
            nameonly=False,
            value="auto",
        )
    )
    in_traj.add_attribute(
        ParamFunctionAttribute(
            name="firstframe",
            alias="first",
            desc="first frame index",
            required=False,
            nameonly=False,
            value=None,
        )
    )
    in_traj.add_attribute(
        ParamFunctionAttribute(
            name="lastframe",
            alias="last",
            desc="last frame index",
            required=False,
            nameonly=False,
            value=None,
        )
    )
    in_traj.add_attribute(
        ParamFunctionAttribute(
            name="stride",
            alias="stride",
            desc="frame stride",
            required=False,
            nameonly=False,
            value=1,
        )
    )
    params[("in__trajectory", "in__trajectory", "in__traj")] = in_traj

    return params
