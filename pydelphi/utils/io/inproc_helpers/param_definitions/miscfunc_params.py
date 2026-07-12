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
    """Defines and returns the 'miscfunc' ParameterGroup."""
    return ParameterGroup(
        "miscfunc",
        "Feature/utility functions (e.g., select, frc).",
        "Feature/utility functions (e.g., select, frc).",
    )


def get_param_definitions():
    """Defines and returns misc function ParamFunction objects."""
    params = {}

    # -------------------------------------------------------------------------
    # select(name=..., condition=..., description=...)
    # -------------------------------------------------------------------------
    #
    # Example:
    #   select(name="REC", condition="{chain A} and not {resname HOH}")
    #
    select_fn = ParamFunction(
        name="select",
        alias="sel",
        attributes=[],
        desc_short="Select a named subset of atoms using a boolean condition.",
        desc_long=(
            "Define a named atom selection using a minimal, property-only selection language.\n"
            "Selections can be reused by later functions (e.g., frc, out(selection,...)).\n"
            "\n"
            "Syntax summary:\n"
            "  - Grouping markers: { ... }\n"
            "      Braces are structural only (nesting allowed). Literal braces are not supported.\n"
            "  - Boolean operators: not, and, or\n"
            "      Precedence: not > and > or\n"
            "  - Predicates: <key> <values...>\n"
            "      For string keys, values are a space-separated list of tokens (exact match).\n"
            "      For numeric keys, values are integers and/or inclusive ranges:\n"
            "          a to b [step c]\n"
            "      'step' is optional (default 1) and must be a positive integer (>= 1).\n"
            "      Hyphen ranges (a-b) are NOT supported (avoids ambiguity with negative numbers).\n"
            "\n"
            "Supported keys:\n"
            "  Numeric:\n"
            "    - index      : 0-based position in atom_keys list (file read order)\n"
            "    - index1     : 1-based position in atom_keys list (normalized to index)\n"
            "    - serial     : atom serial from file (alias: atom_serial)\n"
            "    - resid      : residue id from file (alias: resnum)\n"
            "  String (case-sensitive exact match):\n"
            "    - name       : atom name (atomname)\n"
            "    - resname    : residue name\n"
            "    - chain      : chain identifier\n"
            "    - segid      : segment identifier\n"
            "    - element    : element symbol (case-sensitive; e.g., H, C, N, O, Cl, Zn)\n"
            "\n"
            "Examples:\n"
            '  select(name=protein, cond="not {resname HOH} and not {resname NA CL}")\n'
            '  select(name=chainA,  cond="chain A")\n'
            '  select(name=site,    cond="{chain A and resid 10 to 20} and not {resname HOH}")\n'
            '  select(name=bb,      cond="name N CA C O")\n'
            '  select(name=oddres,  cond="resid 1 to 100 step 2")\n'
            '  select(name=zinc,    cond="element Zn")\n'
            "\n"
            "Notes:\n"
            "  - index/index1 refer to the atom_keys list order as read from the structure file.\n"
            "  - String fields are case-sensitive: tokens must match exactly as in the input.\n"
            "  - The optional 'desc' attribute is metadata only and does not affect selection logic."
        ),
        multicall=True,
    )

    select_fn.add_attribute(
        ParamFunctionAttribute(
            name="name",
            alias="name",
            desc="unique selection name",
            required=True,
            nameonly=False,
            value="",
        )
    )
    select_fn.add_attribute(
        ParamFunctionAttribute(
            name="condition",
            alias="cond",
            desc="selection condition string (uses '{' and '}' for grouping)",
            required=True,
            nameonly=False,
            value="",
        )
    )
    select_fn.add_attribute(
        ParamFunctionAttribute(
            name="description",
            alias="desc",
            desc="optional description (metadata only; not used for selection logic)",
            required=False,
            nameonly=False,
            value="",
        )
    )
    select_fn.add_attribute(
        ParamFunctionAttribute(
            name="on",
            alias="on",
            desc=(
                "ensemble/system this selection applies to "
                "(must match one of the topology labels or 'system'; default: system)"
            ),
            required=False,
            nameonly=False,
            value="system",
        )
    )

    params[("select", "select", "sel")] = select_fn

    # -------------------------------------------------------------------------
    # frc(source=..., target=..., target_mode=..., outfile=..., format=...)
    # -------------------------------------------------------------------------
    #
    # Example:
    #   frc(source="REC", target="LIG", target_mode="charge", outfile="frc.tsv")
    #
    # Meaning (high-level):
    #   Use the electrostatic field generated by 'source' and evaluate energies/potentials
    #   on atoms in 'target'. The target can optionally be "uncharge" or "ignore" as per mode.
    #
    frc_fn = ParamFunction(
        name="frc",
        alias="frc",
        attributes=[],
        desc_short="Compute field/response quantities between two named selections.",
        desc_long=(
            "Compute FRC quantities using two named selections defined by select().\n"
            "Typical use: compute field due to a source subset and evaluate on target subset.\n"
            "This is a single-call function to keep execution flow unambiguous."
        ),
        multicall=False,
    )

    frc_fn.add_attribute(
        ParamFunctionAttribute(
            name="source",
            alias="source",
            desc="named selection used as the field-generating source",
            required=True,
            nameonly=False,
            value="",
        )
    )
    frc_fn.add_attribute(
        ParamFunctionAttribute(
            name="target",
            alias="target",
            desc="named selection used as the evaluation target (required if target_file is empty)",
            required=True,
            nameonly=False,
            value="",
        )
    )
    frc_fn.add_attribute(
        ParamFunctionAttribute(
            name="target_file",
            alias="tfile",
            desc=(
                "file defining evaluation points used for field interpolation "
                "(coordinates; optional charges enable interaction energy output). "
                "Overrides target selection. Supported formats: pdb, pqr, frc."
            ),
            required=False,
            nameonly=False,
            value="",
        )
    )
    frc_fn.add_attribute(
        ParamFunctionAttribute(
            name="target_mode",
            alias="tmode",
            desc=(
                "target handling mode (tmode) controlling how target atoms are treated \n"
                "within the field-generating source.\n"
                "    options: {\n"
                "        uncharge: target atoms remain in the field-generating source but with charges set to zero; \n"
                "        ignore: target atoms are discarded from the field-generating source\n"
                "    }. default: uncharge"
            ),
            required=False,
            nameonly=False,
            value="uncharge",
        )
    )
    frc_fn.add_attribute(
        ParamFunctionAttribute(
            name="outfile",
            alias="ofile",
            desc="output file name for frc results",
            required=True,
            nameonly=False,
            value="",
        )
    )
    frc_fn.add_attribute(
        ParamFunctionAttribute(
            name="format",
            alias="fmt",
            desc=(
                "output format. options include auto. \n"
                "Default: auto. When auto, infer from outfile extension; explicit format is authoritative.\n"
            ),
            required=False,
            nameonly=False,
            value="auto",
        )
    )

    params[("frc", "frc", "frc")] = frc_fn

    grid_offset_fn = ParamFunction(
        name="gridoffset",
        alias="go",
        attributes=[],
        desc_short="Shift the grid relative to pmid in grid units.",
        desc_long=(
            "Applies a translation in grid units (OFFSET) in the mapping between real "
            "coordinates and grid coordinates.\n Useful for probing discretization sensitivity "
            "by varying offsets within [0,1)."
        ),
        multicall=False,
    )

    for nm in ("x", "y", "z"):
        grid_offset_fn.add_attribute(
            ParamFunctionAttribute(
                name=nm,
                alias=nm,
                desc=f"{nm}-offset in grid units; default: 0.0",
                required=True,
                nameonly=False,
                value=0.0,
            )
        )

    params[("gridoffset", "grid_offset", "go")] = grid_offset_fn

    fun_acenter = ParamFunction(
        name="acent",
        alias="ac",
        attributes=[],
        active=False,
        required=False,
        desc_short="Define the real-space center mapped to the grid center.",
        desc_long=(
            "Defines the real-space reference point (Å) that is mapped to the grid center. \n"
            "Exactly one of: (x,y,z), sel, or infile must be provided."
        ),
    )

    # --- explicit coordinates (Å) ---
    for nm in ("x", "y", "z"):
        fun_acenter.add_attribute(
            ParamFunctionAttribute(
                name=nm,
                alias=nm,
                desc=f"{nm}-coordinate of the grid center in real space (Å)",
                required=False,
                nameonly=False,
                value=None,  # avoid silent origin-centering
            )
        )

    # --- named selection (named only) ---
    fun_acenter.add_attribute(
        ParamFunctionAttribute(
            name="selection_name",  # canonical (clear in code)
            alias="sel",  # user-friendly
            desc="Named selection whose geometric center defines the grid center",
            required=False,
            nameonly=False,
            value="",
        )
    )

    # --- external file ---
    fun_acenter.add_attribute(
        ParamFunctionAttribute(
            name="infile",  # canonical
            alias="file",  # user-friendly
            desc="File whose atoms' geometric center defines the grid center",
            required=False,
            nameonly=False,
            value="",
        )
    )

    params[("acenter", "acent", "ac")] = fun_acenter

    return params
