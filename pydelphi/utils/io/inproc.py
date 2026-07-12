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


"""
Parameter definition and input handling module for Delphi simulations.

This module defines classes and functions for managing input parameters
used in Delphi Poisson-Boltzmann equation (PBE) calculations. It provides
mechanisms for parameter validation, organization into groups, and
representation in a structured format suitable for Delphi simulations.

The module includes:
    - Enums for Delphi parameter types (DelphiPrecision, DelphiBioModel, etc.).
    - Configuration constants (PRECISION, delphi_bool, delphi_int, etc.).
    - Input/Output utility functions (readers and writers for various file formats).
    - Parameter validation function (param_typecheck).
    - Classes to represent Delphi parameters:
        - DelphiParameter: Base class for Delphi parameters.
        - DelphiParamStatement: Represents a simple parameter statement with a value.
        - DelphiParamFunctionAttribute: Represents an attribute of a Delphi function parameter.
        - DelphiParamFunction: Represents a Delphi function parameter with attributes.
        - DelphiParameterGroup: Represents a group of Delphi parameters.
    - Inputs: Class to manage and organize all input parameters for Delphi simulations.

This module aims to enhance code quality by providing clear documentation,
improving readability through better naming and structure, increasing
maintainability via modular design, and laying the groundwork for potential
performance optimizations by ensuring data type correctness and validation.
"""
import sys
from os import path
from typing import List, Tuple

import numpy as np

# Import Delphi-specific configurations and enums
from pydelphi.config.global_runtime import (
    delphi_bool,
    delphi_int,
    delphi_real,
)
from pydelphi.foundation.enums import (
    PBSolver,
    BioModel,
    BoundaryCondition,
    DielectricModel,
    GridboxSize,
    ParamType,
    SurfaceMethod,
    PBApproximation,
)

from pydelphi.constants import (
    ATOMFIELD_GAUSS_SIGMA,
)

# Import input/output utility functions
from pydelphi.utils.io.readers import (
    read_gaussian_sigma,
    calculate_center_of_frc_atoms,
)

from pydelphi.utils.io.inproc_helpers.param_definitions.parameters import (
    param_typecheck,
    ParamStatement,
    ParamParseError,
)

from pydelphi.utils.io.inproc_helpers.param_definitions import (
    pb_params,
    surface_params,
    dielectric_params,
    gridbox_params,
    solvent_params,
    salt_params,
    iterativesolver_params,
    convergence_params,
    other_params,
    calculation_params,
    zeta_params,
    infile_params,
    outfile_params,
    miscfunc_params,
)

from pydelphi.utils.io.inproc_helpers.props_assigner import (
    _read_atomic_data,
    _assign_vdw,
    _set_param_func_attributes,
)

MODE_POLICY = {
    "static": {
        "deny": {"in__trajectory", "in__traj", "in__topology", "in__top"},
    },
    "trajectory": {
        "deny": {"in__pdb", "in__modpdb4", "in__crg", "in__frc", "in__phi"},
    },
}


def bool_to_str(val: bool) -> str:
    return "TRUE" if val else "FALSE"


class Inputs:
    """
    Manages and organizes input parameters for Delphi Poisson–Boltzmann (PB) simulations.

    The `Inputs` class is the central container for all user-specified input data,
    including scalar parameters, function-like parameters, topology-related objects,
    and atom selections. It supports both static and trajectory-based workflows.

    Responsibilities:
        - Store Delphi parameters (statements and functions), grouped logically.
        - Track protocol- and ensemble-level metadata.
        - Parse and temporarily store atom-selection specifications.
        - Hold evaluated atom selections after input processing.
        - Provide a single structured object consumed by downstream setup and solvers.

    Attributes:
        app_mode (str):
            Application mode, e.g. "static" or "trajectory".

        protocol_mode (str | None):
            Active protocol mode, if any (set during input processing).

        ensemble (str | None):
            Active ensemble label (e.g., system, complex, receptor, ligand).

        strict (bool):
            If True, unknown or malformed input specifications raise errors.

        ignored_specs (list[str]):
            List of input lines or specifications that were intentionally ignored.

        gridbox_offset (np.ndarray):
            Optional grid box translation vector of shape (3,).

        params (dict):
            Dictionary of all parsed Delphi parameters.
            Keys are (full_name, long_name, short_name) tuples.
            Values are DelphiParamStatement or DelphiParamFunction objects.

        atoms (dict):
            Reserved for atom-level data or atom-associated objects.

        selections_spec (dict[str, str]):
            Parsed but unevaluated atom-selection specifications.
            Maps selection name → selection expression (string).
            Populated during input parsing.

        selections_idx (dict[str, np.ndarray]):
            Evaluated atom selections.
            Maps selection name → sorted array of atom indices.
            Populated during input processing after topology is available.

        objects (list):
            Miscellaneous simulation-related objects created during setup.

        param_groups (dict[str, DelphiParameterGroup]):
            Parameter groups (e.g., "pb", "dielectric", "boundary") organizing
            related Delphi parameters.
    """

    def __init__(self, app_mode="static", strict=True):
        """
        Initializes the `Inputs` object and sets up the parameter groups and their associated
        default parameters. Each group represents a category of parameters used for PBE simulations.
        """
        self.app_mode = app_mode  # app_mode: options (static or trajectory)
        self.protocol_mode = None
        self.ensemble = None
        self.strict = strict
        self.ignored_specs = []
        self.gridbox_offset = np.zeros(3, dtype=float)
        self.params = {}
        self.atoms = {}
        # selections
        self.selections_spec: dict[str, str] = (
            {}
        )  # name -> condition; parsed but unevaluated
        self.selections_idx: dict[str, np.ndarray] = (
            {}
        )  # name -> sorted indices; evaluated later
        self.out_sel_calls: dict[str, dict] = {}
        self.objects = []
        self.param_groups = {}

        param_group_modules = [
            pb_params,
            surface_params,
            dielectric_params,
            gridbox_params,
            solvent_params,
            salt_params,
            iterativesolver_params,
            convergence_params,
            other_params,
            calculation_params,
            zeta_params,
            infile_params,
            outfile_params,
            miscfunc_params,
        ]

        for module in param_group_modules:
            group = module.get_group_definition()
            self.param_groups[group.name] = group

        for module in param_group_modules:
            group = module.get_group_definition()
            params = module.get_param_definitions()
            for key, param_obj in params.items():
                self.params[key] = param_obj
                self.add_to_group(group.name, param_obj)

                # Special handling for external_dielectric (as it belongs to two groups)
                if group.name == "dielectric" and key == (
                    "external_dieclectric",
                    "exdi",
                    "exdi",
                ):
                    self.add_to_group("solvent", param_obj)

        self.param_name_lookup = {}  # for efficient lookup of param names
        for prm_key_tuple, prm_obj in self.params.items():
            for prm_name in prm_key_tuple:
                self.param_name_lookup[prm_name] = (prm_key_tuple, prm_obj)

        # Define valid combinations with readable constraints in the last column
        self.valid_model_combinations: List[Tuple[str, str, str, str, str]] = [
            ("PBE", "TWODIELECTRIC", "VDW", "TRUE", "≥1 (unused)"),
            ("PBE", "TWODIELECTRIC", "VDW", "FALSE", "≥1 (unused)"),
            ("PBE", "GAUSSIAN", "GCS", "FALSE", "≥1"),
            ("PBE", "GAUSSIAN", "GAUSSIAN", "TRUE", "≥1"),
            ("PBE", "GAUSSIAN", "GAUSSIAN", "FALSE", "≥1"),
            ("PBE", "GAUSSIAN", "GAUSSIANCUTOFF", "TRUE", "≥1"),
            ("PBE", "GAUSSIAN", "GAUSSIANCUTOFF", "FALSE", "≥1"),
            ("RPBE", "GAUSSIAN", "GCS", "FALSE", ">1"),
            ("RPBE", "GAUSSIAN", "GAUSSIAN", "FALSE", ">1"),
        ]

        # Headers
        self.model_combinations_header = [
            "biomodel",
            "dielectric_model",
            "surface_method",
            "is_focusing",
            "gaussian_exponent",
        ]

    def _is_valid_combination(
        self,
        biomodel: str,
        dielectric_model: str,
        surface_method: str,
        is_focusing: bool,
        gaussian_exponent: int,
    ) -> bool:
        # Basic validity of gaussian_multiplier
        if gaussian_exponent < 1:
            return False

        for combo in self.valid_model_combinations:
            if (
                combo[0] == biomodel
                and combo[1] == dielectric_model
                and combo[2] == surface_method
                and combo[3] == bool_to_str(is_focusing)
            ):
                constraint = combo[4]
                if constraint.startswith("≥"):
                    return True  # already checked ≥1 above
                elif constraint.startswith(">"):
                    return gaussian_exponent > 1
        return False

    def _print_valid_combinations(self):
        print("\nValid parameter combinations:")
        print("  (Note: 'gaussian_exponent' must be an integer ≥ 1)")
        print(
            "        - '≥1 (unused)' means the parameter is ignored and need not be set meaningfully"
        )
        print("        - '>1' means it must be strictly greater than 1")
        print("  - Column 'is_focusing' is derived from boundary_condition == FOCUSING")
        print("          (actual parameter is 'boundary_condition', not 'is_focusing')")
        print()

        # Calculate max widths for columns (header + data)
        columns = list(
            zip(*([self.model_combinations_header] + self.valid_model_combinations))
        )
        col_widths = [max(len(str(item)) for item in col) for col in columns]

        # Prepare format string for fixed-width columns with 2 spaces padding
        row_format = "  " + "  ".join(f"{{:<{w}}}" for w in col_widths)

        # Print header
        print(row_format.format(*self.model_combinations_header))

        # Print separator line matching header width
        total_width = (
            sum(col_widths) + 2 * (len(col_widths) - 1) + 2
        )  # 2 spaces padding at start
        print(" " * 2 + "-" * total_width)

        # Print each valid combination row
        for combo in self.valid_model_combinations:
            print(row_format.format(*combo))

    def add_to_group(self, grpname, param_member):
        """
        Adds a parameter to the specified group.

        Args:
            grpname (str): The name of the parameter group.
            param_member: The parameter to add to the group.

        Raises:
            Exception: If the specified group name is not recognized.
        """
        if grpname in self.param_groups:
            self.param_groups[grpname].add_member(param_member)
        else:
            raise Exception(f"Unknown group: '{grpname}'")

    def get_param(self, prmname):
        """
        Retrieve a parameter object by name.

        Args:
            prmname (str): The name or partial name of the parameter to retrieve.

        Returns:
            object: The parameter object corresponding to the given name.

        Raises:
            KeyError: If the parameter name is not found.
        """
        if prmname in self.param_name_lookup:
            return self.param_name_lookup[prmname][1]  # return only prm_obj
        else:
            raise KeyError(f"Unknown parameter: {prmname}.")

    def get_param_pair(self, prmname):
        """
        Retrieve a parameter (prm_key_tuple, prm_object) by name.

        Args:
            prmname (str): The name or partial name of the parameter to retrieve.

        Returns:
            object: The parameter object corresponding to the given name.

        Raises:
            KeyError: If the parameter name is not found.
        """
        if prmname in self.param_name_lookup:
            return self.param_name_lookup[prmname]
        else:
            raise KeyError(f"Unknown parameter: {prmname}.")

    def get_param_value(self, prmname):
        """
        Retrieve the value of a parameter by name.

        Args:
            prmname (str): The name or partial name of the parameter.

        Returns:
            any: The value of the parameter.

        Raises:
            KeyError: If the parameter name is not found.
        """
        prm_obj = self.get_param(prmname)
        return prm_obj.value

    def set_param_value(self, prmname, value):
        """
        Set the value of a parameter, with special handling for certain parameters.

        Args:
            prmname (str): The name or partial name of the parameter.
            value (any): The value to set for the parameter.

        Raises:
            KeyError: If the parameter name is not found.
        """
        prm_key, prm_obj = self.get_param_pair(prmname)

        # Check for special parameter handling
        if "indi" in prm_key:
            prm_obj.value = param_typecheck(
                prm_obj.full_name,
                value,
                prm_obj.dtype,
                1,
                self.get_param_value("exdi"),
                prm_obj.override,
            )
        elif "exdi" in prm_key:
            prm_obj.value = param_typecheck(
                prm_obj.full_name,
                value,
                prm_obj.dtype,
                self.get_param_value("indi"),
                prm_obj.max_value,
                prm_obj.override,
            )
        elif "gapdi" in prm_key:
            prm_obj.value = param_typecheck(
                prm_obj.full_name,
                value,
                prm_obj.dtype,
                self.get_param_value("indi"),
                self.get_param_value("exdi"),
                prm_obj.override,
            )
        elif "max_delta_phi" in prm_key:
            prm_obj.value = param_typecheck(
                prm_obj.full_name,
                value,
                prm_obj.dtype,
                prm_obj.min_value,
                prm_obj.max_value,
                prm_obj.override,
            )
            prm_obj.activate()
        else:
            prm_obj.value = param_typecheck(
                prm_obj.full_name,
                value,
                prm_obj.dtype,
                prm_obj.min_value,
                prm_obj.max_value,
                prm_obj.override,
            )

        prm_obj.supplied()

    def _add_atom(self, a_key, a_data):
        """
        Adds an atom to the atom dictionary.

        Args:
            a_key (str): The atom key.
            a_data (np.ndarray): The atom data.
        """
        self.atoms[a_key] = a_data

    def list_param_groups(self):
        """
        Lists all parameter groups.

        Returns:
            str: A comma-separated list of parameter group names.
        """
        return ", ".join(self.param_groups.keys())

    def list_params(self):
        """
        Lists all parameter names.

        Returns:
            str: A newline-separated string of parameter names.
        """
        return "\n".join([" OR ".join(k) for k in self.params.keys()])

    def _fatal_input_error(self, exc, exit_code: int = 1) -> None:
        """
        Terminate input parsing/validation cleanly for user-caused parameter-file
        errors. This avoids noisy Python tracebacks for invalid input syntax.
        """
        if isinstance(exc, ParamParseError):
            print(str(exc), file=sys.stderr)
        else:
            print(
                "ERROR: Invalid parameter file.\n\nProblem:\n  " + str(exc),
                file=sys.stderr,
            )
        raise SystemExit(exit_code)

    def help(
        self,
        groups=None,
        params=None,
        detailed=False,
        grpindent=2,
        indent=2,
        fieldwidth=12,
        linewidth=90,
    ):
        """
        Displays help information for parameter groups or individual parameters.

        Args:
            groups (list or None): The parameter groups to display help for. If None, displays all groups.
            params (list or None): The parameters to display help for. If None, displays all parameters.
            detailed (bool): Whether to show detailed help information.
            grpindent (int): Indentation level for groups.
            indent (int): Indentation level for parameters.
            fieldwidth (int): Width of the help field.
            linewidth (int): Maximum line width for the help display.
        """
        if groups:
            if "all" in groups:
                for grpn, grpv in self.param_groups.items():
                    print(
                        grpv.help(
                            detailed=detailed,
                            grpindent=grpindent,
                            fieldwidth=fieldwidth,
                            linewidth=linewidth,
                        )
                    )
            else:
                for grpn in groups:
                    if grpn in self.param_groups:
                        print(
                            self.param_groups[grpn].help(
                                detailed=detailed,
                                grpindent=grpindent,
                                fieldwidth=fieldwidth,
                                linewidth=linewidth,
                            )
                        )
                    else:
                        print(
                            f"Unknown group: {grpn}. Options are: {self.list_param_groups()}"
                        )
        elif not groups:
            if (not params) or "all" in params:
                for prmn, prmv in self.params.items():
                    print(prmv.help(detailed=detailed))
                    print(f"{'.' * linewidth}")
            else:
                for param in params:
                    prm = self.get_param(param)
                    if prm:
                        print(
                            prm.help(
                                detailed=detailed,
                                indent=indent,
                                fieldwidth=fieldwidth,
                                linewidth=linewidth,
                            )
                        )
                        print(f"{'.' * linewidth}")
                    else:
                        print(
                            f"Unknown parameter: '{param}'. Valid options are:\n{self.list_params()}"
                        )

    def _clean_line(self, line):
        """Removes leading/trailing whitespace and comments from a line."""
        line = line.strip()
        if not line or line.startswith(("!", "#")):
            return None
        if "!" in line:
            line = line.split("!", 1)[0].strip()
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        return line

    def _determine_line_type(self, line):
        """Determines if a line is a function call or a statement."""
        if "(" in line:
            return "function"
        elif "=" in line:
            return "statement"
        return "unrecognized"

    def _all_positional(attribs) -> bool:
        # attribs is a list of strings like ["1.0","2.0","3.0"] OR ["sel=\"SITE\""]
        return all(("=" not in a) for a in attribs)

    def _function_selector_candidates(self, func_name: str) -> list[tuple[str, str]]:
        """
        Return public selector candidates for a selector-style function.

        Each item is (selector_token, help_topic), e.g. ("crgsiz", "in__crgsiz").
        """
        func_name = str(func_name).strip().lower()
        candidates: list[tuple[str, str]] = []

        for prm_key_tuple, prm_obj in self.params.items():
            if getattr(prm_obj, "partype", None) is None:
                continue
            if prm_obj.partype.int_value != ParamType.FUNCTION.int_value:
                continue
            if (
                getattr(prm_obj, "name", None) != func_name
                and getattr(prm_obj, "alias", None) != func_name
            ):
                continue

            first_attr = (
                prm_obj.attributes[0]
                if getattr(prm_obj, "attributes", None)
                and getattr(prm_obj.attributes[0], "nameonly", False)
                else None
            )
            if first_attr is None:
                continue

            public_tokens: list[str] = []

            for key in prm_key_tuple:
                key = str(key).strip().lower()
                if "__" in key:
                    prefix, selector = key.split("__", 1)
                    if prefix == getattr(prm_obj, "name", None):
                        public_tokens.append(selector)
                elif key.startswith(func_name + "_"):
                    public_tokens.append(key[len(func_name) + 1 :])

            public_tokens.extend([first_attr.name, first_attr.alias])

            seen_selectors = set()
            for selector in public_tokens:
                selector = str(selector).strip().lower()
                if not selector or selector in seen_selectors:
                    continue
                seen_selectors.add(selector)
                if hasattr(prm_obj, "effective_help_topic"):
                    help_topic = prm_obj.effective_help_topic(selector)
                else:
                    help_topic = f"{func_name}__{selector}"
                candidates.append((selector, help_topic))

        unique: list[tuple[str, str]] = []
        seen_pairs = set()
        for item in candidates:
            if item in seen_pairs:
                continue
            seen_pairs.add(item)
            unique.append(item)
        return unique

    def _selector_help_topics(self, func_name: str) -> list[str]:
        """Return unique canonical help topics for a selector-style function."""
        topics: list[str] = []
        seen = set()
        for _selector, topic in self._function_selector_candidates(func_name):
            if topic not in seen:
                topics.append(topic)
                seen.add(topic)
        return topics

    def _resolve_selector_param(self, func_name: str, selector: str):
        """
        Resolve function(selector, ...) to its ParamFunction.

        Accepted during transition:
          - new registry/help convention: function__selector
          - old registry convention: function_selector
          - first name-only attribute name/alias
        """
        func_name = str(func_name).strip().lower()
        selector = str(selector).strip().lower()

        for key in (f"{func_name}__{selector}", f"{func_name}_{selector}"):
            if key in self.param_name_lookup:
                return self.get_param(key)

        for prm_key_tuple, prm_obj in self.params.items():
            if getattr(prm_obj, "partype", None) is None:
                continue
            if prm_obj.partype.int_value != ParamType.FUNCTION.int_value:
                continue
            if (
                getattr(prm_obj, "name", None) != func_name
                and getattr(prm_obj, "alias", None) != func_name
            ):
                continue
            if not getattr(prm_obj, "attributes", None):
                continue

            first_attr = prm_obj.attributes[0]
            if not getattr(first_attr, "nameonly", False):
                continue

            accepted = {
                str(first_attr.name).strip().lower(),
                str(first_attr.alias).strip().lower(),
            }

            for key in prm_key_tuple:
                key = str(key).strip().lower()
                if "__" in key:
                    prefix, key_selector = key.split("__", 1)
                    if prefix == func_name:
                        accepted.add(key_selector)
                elif key.startswith(func_name + "_"):
                    accepted.add(key[len(func_name) + 1 :])

            if selector in accepted:
                return prm_obj

        raise ParamParseError(
            f"Unknown selector '{selector}' for function '{func_name}'.",
            function_name=func_name,
            selector=selector,
            available_help_topics=self._selector_help_topics(func_name),
        )

    def _make_param_parse_error(
        self,
        message: str,
        *,
        record: str,
        function_name: str | None = None,
        selector: str | None = None,
        attribute: str | None = None,
        help_topic: str | None = None,
        available_help_topics: list[str] | None = None,
    ) -> ParamParseError:
        return ParamParseError(
            message,
            record=record,
            function_name=function_name,
            selector=selector,
            attribute=attribute,
            help_topic=help_topic,
            available_help_topics=available_help_topics,
        )

    def _parse_function(self, line):
        """
        Parses a function-style input line and sets the corresponding parameter attributes.

        NOTE: Function-like params must be single-line.
        """
        func_name_end = line.find("(")
        if func_name_end == -1:
            raise ValueError(f"Invalid function format: {line}")

        func_name = line[:func_name_end].lower()

        close_paren = line.rfind(")")
        if close_paren == -1:
            raise ValueError(f"Invalid function format (missing ')'): {line}")

        attribs_str = line[func_name_end + 1 : close_paren].strip()

        # IMPORTANT: supports commas inside quoted strings (e.g., condition="...")
        attribs = self._split_args_csv_quoted(attribs_str)

        try:
            # Dedicated handlers first (clarity; avoids ParamFunction plumbing for these)
            if func_name in ("select", "sel"):
                self._parse_select_function(line, attribs)
                prm = self.get_param(func_name)
                prm.activate()
                return

            if func_name == "frc":
                self._parse_frc_function(line, attribs)
                prm = self.get_param(func_name)
                prm.activate()
                return

            if func_name == "out" and attribs:
                what = attribs[0].lower()
                if what in ("selection", "sel"):
                    self._parse_out_selection_function(line, attribs)
                    prm = self.get_param("out__selection") or self.get_param("out__sel")
                    if prm:
                        prm.activate()
                    return

            if len(attribs) > 0:
                selector_for_policy = attribs[0].strip().lower()
                parm_name = func_name + "_" + selector_for_policy
                parm_topic = func_name + "__" + selector_for_policy

                deny = MODE_POLICY.get(self.app_mode, {}).get("deny", set())
                if parm_name in deny or parm_topic in deny:
                    msg = (
                        f"{func_name}({selector_for_policy}, ...) is not applicable "
                        f"in `{self.app_mode}` mode; ignoring: {line}"
                    )
                    if self.strict:
                        raise ValueError(msg)
                    self.ignored_specs.append(msg)
                    return

            if func_name in ("acenter", "acent", "ac"):
                prm = self.get_param(func_name)

                # Decide positional vs named based on presence of '=' in any token
                has_named = any("=" in a for a in attribs)

                if not has_named:
                    # Positional form: only allowed as acenter(x, y, z)
                    if len(attribs) != 3:
                        raise ValueError(
                            "acenter positional form requires exactly 3 values: acenter(x, y, z). "
                            "Use named attributes for selection/file modes "
                            "(e.g., acenter(sel='SITE'), acenter(file='site.pqr'))."
                        )
                    _set_param_func_attributes(
                        prm,
                        attribs,
                        expected_names=("x", "y", "z"),
                        is_float=True,
                        file_check=None,
                    )
                else:
                    # Named form: allow x=..., y=..., z=..., sel=.../selection_name=..., file=.../infile=...
                    # Use is_float=False because this list can contain strings (sel/file);
                    # x/y/z (if supplied) can be cast to float later in validation/resolution.
                    _set_param_func_attributes(
                        prm,
                        attribs,
                        expected_names=None,
                        is_float=False,
                        file_check="in",  # keep deferred; acenter(file=...) can be checked later
                    )

            elif func_name in ("grid_offset",):
                prm = self.get_param(func_name)
                has_named = any("=" in a for a in attribs)

                if has_named:
                    # Named form: grid_offset(x=..., y=..., z=...)
                    # Allow any order; require only known names.
                    _set_param_func_attributes(
                        prm,
                        attribs,
                        expected_names=("x", "y", "z"),
                        is_float=True,
                        file_check=None,
                    )
                else:
                    # Positional form: grid_offset(x, y, z)
                    if len(attribs) != 3:
                        raise ValueError(
                            "grid_offset positional form requires exactly 3 values: grid_offset(x, y, z)."
                        )
                    _set_param_func_attributes(
                        prm,
                        attribs,
                        expected_names=("x", "y", "z"),
                        is_float=True,
                        file_check=None,
                    )

                self.gridbox_offset[0] = float(prm.get_attribute("x"))
                self.gridbox_offset[1] = float(prm.get_attribute("y"))
                self.gridbox_offset[2] = float(prm.get_attribute("z"))

                if self.gridbox_offset.shape != (3,):
                    raise ValueError(
                        "grid_offset must be a 3-element vector (x, y, z)."
                    )

                if np.any(np.abs(self.gridbox_offset) > 1.0):
                    raise ValueError("grid_offset values must be in range [-1, 1].")

            elif func_name in ("site", "calculate_energies", "energies", "energy"):
                prm = self.get_param(func_name)
                _set_param_func_attributes(prm, attribs, is_float=False)

            elif func_name in ("in", "out"):
                if not attribs:
                    raise self._make_param_parse_error(
                        f"{func_name}(...): missing required selector.",
                        record=line,
                        function_name=func_name,
                        available_help_topics=self._selector_help_topics(func_name),
                    )

                selector = attribs[0].strip().lower()
                if "=" in selector:
                    raise self._make_param_parse_error(
                        f"{func_name}(...): first argument must be a selector, got {attribs[0]!r}.",
                        record=line,
                        function_name=func_name,
                        available_help_topics=self._selector_help_topics(func_name),
                    )

                try:
                    prm = self._resolve_selector_param(func_name, selector)
                except ParamParseError as e:
                    raise ParamParseError(
                        e.message,
                        record=line,
                        function_name=func_name,
                        selector=selector,
                        available_help_topics=e.available_help_topics,
                    )

                _set_param_func_attributes(
                    prm,
                    attribs[1:],
                    is_float=False,
                    file_check=func_name,
                    record=line,
                    selector=selector,
                )
                prm.supplied()

                if prm.multicall:
                    prm.calls.append(prm.snapshot_call())
                    prm.activate()
                    prm.reset_inuse()

                if func_name == "out":
                    prm.activate()
            else:
                raise ValueError(f"Unknown function: {func_name}")

        except (FileNotFoundError, ValueError, ParamParseError):
            raise

    # ----------------------------
    # Private: arg splitting + kv parsing
    # ----------------------------

    def _split_args_csv_quoted(self, s: str) -> list[str]:
        """
        Split a comma-separated argument list, but do not split inside single/double quotes.
        Example: 'name="A", condition="{resid 1 to 10 and name CA}"'
        """
        out: list[str] = []
        buf: list[str] = []
        quote: str | None = None
        esc = False

        for ch in s:
            if esc:
                buf.append(ch)
                esc = False
                continue

            if ch == "\\":
                buf.append(ch)
                esc = True
                continue

            if quote is not None:
                buf.append(ch)
                if ch == quote:
                    quote = None
                continue

            if ch in ("'", '"'):
                buf.append(ch)
                quote = ch
                continue

            if ch == ",":
                item = "".join(buf).strip()
                if item:
                    out.append(item)
                buf = []
                continue

            buf.append(ch)

        tail = "".join(buf).strip()
        if tail:
            out.append(tail)

        return out

    def _parse_kv_args(self, args: list[str]) -> dict[str, str]:
        """
        Parse args like ['name="REC"', 'condition="{...}"'] into a dict.
        Values may be quoted with single or double quotes.
        """
        kv: dict[str, str] = {}
        for a in args:
            if "=" not in a:
                raise ValueError(f"Expected key=value argument, got: {a!r}")
            k, v = a.split("=", 1)
            k = k.strip().lower()
            v = v.strip()

            # Strip surrounding quotes if present
            if len(v) >= 2 and (v[0] == v[-1]) and v[0] in ("'", '"'):
                q = v[0]
                v = v[1:-1]
                # minimal unescape
                v = v.replace("\\\\", "\\")
                if q == '"':
                    v = v.replace('\\"', '"')
                else:
                    v = v.replace("\\'", "'")

            kv[k] = v
        return kv

    # ----------------------------
    # Private: select(...) and frc(...) handlers
    # ----------------------------

    def _parse_select_function(self, line: str, attribs: list[str]) -> None:
        """
        Parse: select(name="...", condition="...", description="...")
        Stores into self.selections_spec[name] = {"condition": ..., "description": ...}
        Duplicate names: override-last (unless you change policy).
        """
        kv = self._parse_kv_args(attribs)

        name = kv.get("name", "").strip()
        if not name:
            raise ValueError("select(...): missing required attribute name=...")

        condition = kv.get("condition", kv.get("cond", "")).strip()
        if not condition:
            raise ValueError("select(...): missing required attribute condition=...")

        description = kv.get("description", kv.get("desc", "")).strip()

        if not hasattr(self, "selections_spec") or self.selections_spec is None:
            self.selections_spec = {}

        # Strict duplicate protection:
        if self.strict and name in self.selections_spec:
            raise ValueError(f"select(...): duplicate selection name {name!r}")

        self.selections_spec[name] = {
            "condition": condition,
            "description": description,
        }

    def _parse_out_selection_function(self, line: str, attribs: list[str]) -> None:
        """
        Parse: out(selection|sel, name=..., file=..., format=...)
        Stores into self.out_sel_calls[name] = {...}
        Duplicate names: override-last (unless strict).
        """
        kv = self._parse_kv_args(attribs[1:])

        selname = kv.get("selname", kv.get("name", "")).strip()
        if not selname:
            raise ValueError("out(selection,...): missing required attribute name=...")

        if not hasattr(self, "out_sel_calls") or self.out_sel_calls is None:
            self.out_sel_calls = {}

        if self.strict and selname in self.out_sel_calls:
            raise ValueError(
                f"out(selection,...): duplicate selection name {selname!r}"
            )

        self.out_sel_calls[selname] = {
            "file": kv.get("file", "").strip(),  # raw user input
            "format": kv.get("format", kv.get("fmt", "")).strip(),
        }

    def _validate_out_selection_calls(self) -> None:
        """
        Validate and resolve out(selection, ...) using only selection *names*.
        Does NOT require materialized selection indices.
        """
        if not getattr(self, "out_sel_calls", None):
            return

        # selection names known from select(...)
        sel_spec = getattr(self, "selections_spec", None) or {}
        sel_names = set(sel_spec.keys())

        for selname, spec in self.out_sel_calls.items():

            # 1) selection name must exist
            if selname not in sel_names:
                raise ValueError(
                    f"out(selection,name={selname}): unknown selection '{selname}'. "
                    "Define it first using select(name=..., ...)."
                )

            # 2) format
            fmt = (spec.get("format") or "pqr").strip().lower()
            if fmt not in ("pqr", "pdb"):
                raise ValueError(
                    f"out(selection,name={selname}): invalid format '{fmt}'. "
                    "Allowed values: pqr, pdb."
                )
            spec["format"] = fmt

            # 3) file: validate directory only if user provided file
            outfile = (spec.get("file") or "").strip()
            if outfile:
                outdir = path.dirname(outfile)
                if outdir and not path.isdir(outdir):
                    raise ValueError(
                        f"out(selection,name={selname}): "
                        f"output directory does not exist for file='{outfile}'."
                    )
            else:
                # default to CWD
                spec["file"] = f"{selname}.{fmt}"

    def _parse_frc_function(self, line: str, attribs: list[str]) -> None:
        """
        Parse:
            frc(
                source="SEL1",
                target="SEL2",
                target_file="points.pqr",
                target_mode="uncharge|ignore",
                outfile="out.frc",
                format="frc"
            )

        Stores into params["frc"] via the standard path:
            prm = self.get_param("frc")
            _set_param_func_attributes(prm, attribs_norm, is_float=False, file_check="out")

        Notes:
          - Selection existence validation (source/target names) happens later.
          - Here we validate presence + enums + outfile directory existence.
          - target is required iff target_file is empty.
          - target_file overrides target.
          - Accepts legacy target_mode="purge" as a synonym for "ignore".
        """
        kv = self._parse_kv_args(attribs)

        # --- required fields (using parsed kv) ---
        source = kv.get("source", "").strip()

        target_file = kv.get("target_file", "").strip()
        if not target_file:
            target_file = kv.get("tfile", "").strip()

        target = kv.get("target", "").strip()

        outfile = kv.get("outfile", "").strip()
        if not outfile:
            outfile = kv.get("ofile", "").strip()

        if not source:
            raise ValueError("frc(...): missing required attribute source=...")

        if not outfile:
            raise ValueError(
                "frc(...): missing required attribute outfile=... (or ofile=...)"
            )

        if not target_file and not target:
            raise ValueError(
                "frc(...): missing required attribute target=... "
                "(required when target_file/tfile is not provided)"
            )

        # --- validate enums ---
        target_mode = kv.get("target_mode", "").strip()
        if not target_mode:
            target_mode = kv.get("tmode", "uncharge").strip()
        target_mode = target_mode.lower()

        # legacy synonym
        if target_mode == "purge":
            target_mode = "ignore"

        if target_mode not in ("uncharge", "ignore"):
            raise ValueError("frc(...): target_mode must be one of {uncharge, ignore}")

        fmt = kv.get("format", "").strip()
        if not fmt:
            fmt = kv.get("fmt", "frc").strip()
        fmt = fmt.lower()

        if fmt not in ("frc",):
            raise ValueError("frc(...): format must be one of {frc}")

        # --- normalize aliases in the original attribute strings ---
        # We want the setter to store canonical keys only.
        attribs_norm: list[str] = []
        for a in attribs:
            s = a.strip()
            if "=" not in s:
                attribs_norm.append(s)
                continue

            k, v = [x.strip() for x in s.split("=", 1)]
            kl = k.lower()

            if kl == "ofile":
                k = "outfile"
            elif kl == "tfile":
                k = "target_file"
            elif kl == "tmode":
                k = "target_mode"
            elif kl == "fmt":
                k = "format"

            if kl == "format":
                v = v.lower()

            # normalize legacy purge -> ignore if explicitly specified in attribs
            if k.lower() == "target_mode":
                vv = v.strip().strip('"').strip("'").lower()
                if vv == "purge":
                    # preserve quoting style loosely by rewriting the raw token
                    # (safe because v is treated as an opaque string by setter)
                    v = (
                        '"ignore"'
                        if '"' in v
                        else ("'ignore'" if "'" in v else "ignore")
                    )

            attribs_norm.append(f"{k}={v}")

        prm = self.get_param("frc")
        _set_param_func_attributes(prm, attribs_norm, is_float=False, file_check="out")

    def _collect_multiline_function(self, first_line, file_iter):
        """
        Collect a possibly-multiline function call.
        Statements are always single-line and must not reach here.
        """
        buf = [first_line]
        depth = first_line.count("(") - first_line.count(")")

        while depth > 0:
            try:
                line = next(file_iter)
            except StopIteration:
                raise ValueError("Unterminated function call")

            cleaned = self._clean_line(line)
            if cleaned is None:
                continue

            buf.append(cleaned)
            depth += cleaned.count("(") - cleaned.count(")")

        return " ".join(buf)

    def _parse_statement(self, line):
        """
        Parses a statement-style input line and sets the corresponding parameter value.

        Args:
            line (str): The input line representing a statement, e.g., "param=value".

        Raises:
            ValueError: If the statement format is invalid or contains invalid values.
        """
        tokens = [w.strip() for w in line.split("=")]
        if len(tokens) != 2:
            raise ValueError(f"Unrecognized statement format: {line}")

        param_name, value = tokens[0].lower(), tokens[1].lower()
        prm = self.get_param(param_name)

        if prm:
            try:
                if param_name in ["grid_size", "gridsize", "gsize"]:
                    try:
                        value = GridboxSize(int(tokens[1]))
                    except (ValueError, IndexError):
                        raise ValueError("Invalid grid size format.")

                prm.value = param_typecheck(
                    prm.full_name,
                    value,
                    prm.dtype,
                    prm.min_value,
                    prm.max_value,
                    prm.override,
                )
                prm.supplied()
                prm.activate()
            except ValueError:
                raise
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

    def process_traj_inputs(self):
        """
        Process and validate inputs for trajectory mode.

        Current supported protocol:
            - singletraj-system: exactly one topology + one trajectory, both label='system'

        Format dispatch is delegated to lite interface modules:
            - topology_lite.open_topology_lite(): pdb, pqr, psf, prmtop
            - trajectory_lite.open_trajectory_lite(): nc, dcd, trr

        Context-dependent validation remains here:
            - topology/trajectory call pairing
            - label/protocol constraints
            - atom-count consistency
            - frame selection normalization
            - charge/size policy application before topology freeze

        Produces:
            self.ensemble: dict-like Ensemble with one entry: "system"
            self.protocol_mode: "singletraj-system"
        """
        # Lazy imports to avoid circular deps and keep static mode lightweight.
        from pydelphi.utils.io.lite.ensemble import (
            Ensemble,
            EnsembleEntry,
            canonicalize_label,
        )
        from pydelphi.utils.io.lite.topology_lite import (
            freeze_topology_lite,
            normalize_topology_format,
            open_topology_lite,
        )
        from pydelphi.utils.io.lite.trajectory_lite import (
            normalize_trajectory_format,
            open_trajectory_lite,
        )
        from pydelphi.utils.io.inproc_helpers.props_assigner import (
            _apply_crgsiz_to_topology_lite,
        )

        # Pull multicall ParamFunctions. They must have .calls populated by _parse_function().
        prm_top = self.get_param("in__topology")
        prm_traj = self.get_param("in__trajectory")

        top_calls = getattr(prm_top, "calls", None) or []
        traj_calls = getattr(prm_traj, "calls", None) or []

        if len(top_calls) == 0 or len(traj_calls) == 0:
            raise ValueError(
                "traj mode requires at least one in(topology, ...) and one "
                "in(trajectory, ...) specification."
            )

        # Global charge/size inputs. These are not multicall.
        in_crgsiz = self.get_param("in__crgsiz")
        in_siz = self.get_param("in__siz")
        in_crg = self.get_param("in__crg")

        def _label_of(call: dict) -> str:
            # label should be present due to snapshot_call() storing defaults;
            # keep defensive default for older call dictionaries.
            return canonicalize_label(call.get("label", "system"))

        def _require_marker(call: dict, marker: str, func_name: str):
            # marker is a name-only attribute stored as True when present.
            if not call.get(marker, False):
                raise ValueError(
                    f"Invalid {func_name} spec: missing required marker '{marker}'. "
                    f"Expected syntax like in({marker}, file=..., ...). Got: {call}"
                )

        # Validate markers and normalize labels.
        for c in top_calls:
            _require_marker(c, "topology", "in__topology")
        for c in traj_calls:
            _require_marker(c, "trajectory", "in__trajectory")

        # Protocol detection: singletraj-system only.
        if len(top_calls) != 1 or len(traj_calls) != 1:
            raise ValueError(
                "Only singletraj-system is supported right now: provide exactly "
                "one in(topology, ...) and one in(trajectory, ...) call."
            )

        top_call = top_calls[0]
        traj_call = traj_calls[0]

        top_label = _label_of(top_call)
        traj_label = _label_of(traj_call)

        if top_label != traj_label:
            raise ValueError(
                "singletraj-system requires identical labels for topology and trajectory "
                f"(got top={top_label!r}, traj={traj_label!r})."
            )

        if top_label != "system":
            raise ValueError(
                "singletraj-system currently supports only label='system' "
                f"(got {top_label!r})."
            )

        # Extract files/formats. Defaults are already in call dict when supplied by parser,
        # but keep defaults here for older snapshots.
        top_file = top_call.get("file", "")
        traj_file = traj_call.get("file", "")

        if not top_file:
            raise ValueError("in(topology, ...) missing required attribute: file=...")
        if not traj_file:
            raise ValueError("in(trajectory, ...) missing required attribute: file=...")

        top_fmt = normalize_topology_format(top_call.get("format", "prmtop"))
        traj_fmt = normalize_trajectory_format(traj_call.get("format", "nc"))

        # Open/read via lite interface dispatchers.
        top = open_topology_lite(top_file, top_fmt)
        traj = open_trajectory_lite(traj_file, traj_fmt)

        # Apply context-dependent charge/size policy to the topology before freezing.
        # Policy itself lives in prop_assigner; this function only supplies context.
        top = _apply_crgsiz_to_topology_lite(
            top=top,
            input_kind=top_fmt,
            in_crgsiz=in_crgsiz,
            in_siz=in_siz,
            in_crg=in_crg,
        )
        top = freeze_topology_lite(top)

        # Frame selection normalization:
        # - firstframe is inclusive, 0-based
        # - lastframe is inclusive in prmfile
        # - internal stop is exclusive
        first = traj_call.get("firstframe", None)
        last = traj_call.get("lastframe", None)
        stride = traj_call.get("stride", 1)

        start = 0 if first is None else int(first)
        stop = None if last is None else int(last) + 1
        stride = 1 if stride is None else int(stride)

        if start < 0:
            raise ValueError(f"firstframe must be >= 0 (got {start})")
        if stop is not None and stop <= start:
            raise ValueError(
                "Invalid frame range: lastframe must be >= firstframe "
                f"(got first={start}, last={stop - 1})."
            )
        if stride <= 0:
            raise ValueError(f"stride must be >= 1 (got {stride})")

        # Validate natoms consistency early.
        traj_natoms = getattr(traj, "natoms", None)
        if traj_natoms is not None and int(traj_natoms) != int(top.natoms):
            raise ValueError(
                "Topology/trajectory atom count mismatch: "
                f"top.natoms={top.natoms}, traj.natoms={traj_natoms}."
            )

        # Assemble ensemble.
        ens = Ensemble()
        ens.add(
            "system",
            EnsembleEntry(top=top, traj=traj, start=start, stop=stop, stride=stride),
            overwrite=False,
        )

        self.ensemble = ens
        self.protocol_mode = "singletraj-system"

    def parse_inputs(self, filename):
        """
        Parse an input parameter file and finalize input state.

        User-caused parse/validation errors are caught here and reported once,
        without a Python traceback.
        """
        try:
            with open(filename, "r") as file:
                for line in file:
                    cleaned_line = self._clean_line(line)
                    if cleaned_line is None:
                        continue

                    line_type = self._determine_line_type(cleaned_line)
                    if line_type == "function":
                        record = self._collect_multiline_function(cleaned_line, file)
                        self._parse_function(record)
                    elif line_type == "statement":
                        self._parse_statement(cleaned_line)
                    else:
                        print(
                            f"Warning: Ignoring unrecognized input specification: '{cleaned_line}'"
                        )

            for msg in self.ignored_specs:
                print("Warning:", msg, file=sys.stderr)

            self.process_inputs()

            if self.app_mode == "trajectory":
                self.process_traj_inputs()

        except FileNotFoundError as e:
            if str(e):
                self._fatal_input_error(e)
            self._fatal_input_error(f"Parameter file '{filename}' not found.")

        except IOError as e:
            self._fatal_input_error(
                f"An error occurred while reading the file '{filename}': {e}"
            )

        except ParamParseError as e:
            self._fatal_input_error(e)

        except ValueError as e:
            self._fatal_input_error(e)

    def _get_cached_params(self):
        """Caches frequently used parameter objects for efficiency."""
        return {
            "biomodel": self.get_param("biomodel"),
            "solver": self.get_param("solver"),
            "pb_approximation": self.get_param("pb_approximation"),
            "linit": self.get_param("linit"),
            "nonlinit": self.get_param("nonlinit"),
            "nonlinear_coupling_steps": self.get_param("nonlinear_coupling_steps"),
            "nonlinear_relaxation_param": self.get_param("nonlinear_relaxation_param"),
            "max_nonlinear_coupling_delta_phi": self.get_param(
                "max_nonlinear_coupling_delta_phi"
            ),
            "boundary_condition": self.get_param("boundary_condition"),
            "scale": self.get_param("scale"),
            "grid_size": self.get_param("grid_size"),
            "percent_fill": self.get_param("percent_fill"),
            "gridbox_margin": self.get_param("gridbox_margin"),
            "probe_radius": self.get_param("probe_radius"),
            "probe_radius2": self.get_param("probe_radius2"),
            "dielectricmodel": self.get_param("dielectricmodel"),
            "gap_dielectric": self.get_param("gap_dielectric"),
            "sigma": self.get_param("sigma"),
            "gaussian_sigma": self.get_param("gaussian_sigma"),
            "gaussian_exponent": self.get_param("gaussian_exponent"),
            "surface_cutoff": self.get_param("surface_cutoff"),
            "density_cutoff": self.get_param("density_cutoff"),
            "surface_method": self.get_param("surface_method"),
            "midpoint_dielectric_gaussian": self.get_param(
                "midpoint_dielectric_gaussian"
            ),
            "surface_offset": self.get_param("surface_offset"),
            "surface_density_exponent": self.get_param("surface_density_exponent"),
            "max_rmsd": self.get_param("max_rmsd"),
            "max_delta_phi": self.get_param("max_delta_phi"),
            "in__modpdb4": self.get_param("in__modpdb4"),
            "in__pdb": self.get_param("in__pdb"),
            "in__siz": self.get_param("in__siz"),
            "in__crg": self.get_param("in__crg"),
            "in__crgsiz": self.get_param("in__crgsiz"),
            "acenter": self.get_param("acenter"),
            "in__frc": self.get_param("in__frc"),
            "in__vdw": self.get_param("in__vdw"),
            "in__phi": self.get_param("in__phi"),
            "calculate_energies": self.get_param("calculate_energies"),
            "frc": self.get_param("frc"),
            "site": self.get_param("site"),
        }

    def _process_grid_parameters(self, params):
        """Processes grid-related parameters."""
        if not params["scale"].issupplied:
            params["scale"].deactivate()
        if not params["grid_size"].issupplied:
            params["grid_size"].deactivate()
        if (
            params["percent_fill"].issupplied
            and not params["gridbox_margin"].issupplied
        ):
            params["percent_fill"].activate()
            params["gridbox_margin"].deactivate()
        else:
            params["gridbox_margin"].activate()
            params["percent_fill"].deactivate()

    def _process_solvent_parameters(self, params):
        """Processes solvent-related parameters."""
        if params["probe_radius"].issupplied and (
            not params["probe_radius2"].issupplied
        ):
            params["probe_radius2"].set(params["probe_radius"].get())
            params["probe_radius2"].deactivate()

    def _configure_pbe_solver(self, params, solver_value, nonlinit_value):
        """Configures PBE solver settings."""
        if nonlinit_value == 0:
            params["pb_approximation"].set(PBApproximation.LINEAR)
            params["nonlinit"].deactivate()
            params["nonlinear_coupling_steps"].deactivate()
            params["nonlinear_relaxation_param"].deactivate()
            params["max_nonlinear_coupling_delta_phi"].deactivate()
            if solver_value == PBSolver.NWT:
                print(
                    "NOTE: NWT & SOR use identical iteration formula for Linear PB, thus NWT is overridden to SOR for it."
                )
                params["solver"].set(PBSolver.SOR)
        else:
            params["pb_approximation"].set(PBApproximation.NONLINEAR)
            params["max_nonlinear_coupling_delta_phi"].activate()
            if solver_value == PBSolver.NWT:
                params["nonlinear_coupling_steps"].deactivate()
                params["nonlinear_relaxation_param"].deactivate()
                # params["nonlinear_iteration_block_size"].deactivate()
                params["max_nonlinear_coupling_delta_phi"].deactivate()

    def _configure_pbe_biomodel(
        self, params, dielectricmodel_value, surface_method_value
    ):
        """Configures settings specific to PBE biomodel."""
        if dielectricmodel_value == DielectricModel.TWODIELECTRIC.int_value:
            params["dielectricmodel"].deactivate()
            params["gaussian_sigma"].deactivate()
            params["gaussian_exponent"].deactivate()
            params["surface_cutoff"].deactivate()
            params["density_cutoff"].deactivate()
            params["surface_method"].set(SurfaceMethod.VDW)
        elif dielectricmodel_value == DielectricModel.GAUSSIAN.int_value:
            if surface_method_value == SurfaceMethod.GAUSSIANCUTOFF.int_value:
                params["surface_method"].activate()
                if params["density_cutoff"].issupplied:
                    params["density_cutoff"].activate()
                else:
                    params["density_cutoff"].deactivate()
                if params["surface_cutoff"].issupplied:
                    params["surface_cutoff"].activate()
                else:
                    params["surface_cutoff"].deactivate()
                if not (
                    params["density_cutoff"].active or params["surface_cutoff"].active
                ):
                    params["density_cutoff"].activate()
            else:
                params["density_cutoff"].deactivate()
                params["surface_cutoff"].deactivate()

    def _configure_rpbe_biomodel(
        self, params, surface_method_value, surface_offset_value
    ):
        """Configures settings specific to RPBE biomodel."""
        params["surface_cutoff"].deactivate()
        params["density_cutoff"].deactivate()
        if surface_offset_value == 0.0:
            params["surface_offset"].deactivate()
        if surface_method_value in {
            SurfaceMethod.GCS.int_value,
            SurfaceMethod.GAUSSIAN.int_value,
        }:
            if surface_method_value == SurfaceMethod.GCS.int_value:
                self.set_param_value("midpoint_dielectric_gaussian", False)
        else:
            raise Exception(
                "surfacemethod VDW works only for PBE. Choose from ['GCS', 'GAUSSIAN']"
            )

    def _process_convergence_parameters(self, params):
        """Handles convergence-related parameters."""
        if params["max_rmsd"].issupplied and not params["max_delta_phi"].issupplied:
            params["max_rmsd"].activate()
            params["max_delta_phi"].deactivate()
        else:
            params["max_rmsd"].deactivate()
            params["max_delta_phi"].activate()

    def _check_focusing_run_requirements(self, boundary_condition_value, param_in_phi):
        """Checks if parent phimap is supplied for focusing runs."""
        if boundary_condition_value == BoundaryCondition.FOCUSING.value:
            if not (
                param_in_phi.issupplied
                and path.isfile(param_in_phi.get_attribute("file"))
            ):
                raise ValueError(
                    "FileNotFound: parent phimap required for focusing run must be read"
                )

    def _set_gaussian_sigma(self, atoms, dielectricmodel_value, param_sigma):
        """Sets Gaussian sigma for atoms if the dielectric model is Gaussian."""
        if dielectricmodel_value.value == DielectricModel.GAUSSIAN.value:
            for a_key, a_data in atoms.items():
                a_data[ATOMFIELD_GAUSS_SIGMA] = delphi_real(param_sigma.get())

    def _validate_frc(self):
        prm_frc = self.get_param("frc")
        if not prm_frc.issupplied:
            return

        source = str(prm_frc.get_attribute("source") or "").strip()
        target = str(prm_frc.get_attribute("target") or "").strip()
        target_file = str(prm_frc.get_attribute("target_file") or "").strip()
        tmode = str(prm_frc.get_attribute("target_mode") or "uncharge").strip().lower()
        outfile = str(prm_frc.get_attribute("outfile") or "").strip()

        if not source:
            raise ValueError("frc(...): missing required attribute source=...")

        if not outfile:
            raise ValueError("frc(...): missing required attribute outfile=...")

        # legacy synonym support (defensive; parser may already normalize)
        if tmode == "purge":
            tmode = "ignore"

        if tmode not in ("uncharge", "ignore"):
            raise ValueError("frc(...): target_mode must be one of {uncharge, ignore}")

        # --- outfile directory must exist ---
        out_dir = path.dirname(outfile) or "."
        if not path.isdir(out_dir):
            raise ValueError(f"frc(...): output directory does not exist: '{out_dir}'")

        # --- selection name validation ---
        if source not in self.selections_spec:
            raise ValueError(f"frc(...): source selection {source!r} is not defined")

        # --- target resolution validation ---
        if target_file:
            # file-based evaluation points override target selection
            if not path.isfile(target_file):
                raise ValueError(
                    f"frc(...): target_file does not exist: {target_file!r}"
                )

            # target_mode is only meaningful for selection targets
            if self.strict and tmode != "uncharge":
                raise ValueError(
                    "frc(...): target_mode is only applicable when using a selection target "
                    "(i.e., when target_file is empty)"
                )
        else:
            # selection-based evaluation target
            if not target:
                raise ValueError(
                    "frc(...): missing required attribute target=... "
                    "(required when target_file is not provided)"
                )
            if target not in self.selections_spec:
                raise ValueError(
                    f"frc(...): target selection {target!r} is not defined"
                )

    def _validate_acenter(self):
        prm = self.get_param("acenter")
        if not prm.issupplied:
            return

        x = prm.get_attribute("x")
        y = prm.get_attribute("y")
        z = prm.get_attribute("z")

        sel_name = (prm.get_attribute("selection_name") or "").strip()  # alias: sel
        infile = (prm.get_attribute("infile") or "").strip()  # alias: file

        has_any_xyz = (x is not None) or (y is not None) or (z is not None)
        has_sel = sel_name != ""
        has_file = infile != ""

        # Exactly one mode must be chosen
        if (int(has_any_xyz) + int(has_sel) + int(has_file)) != 1:
            raise ValueError(
                "acenter: specify exactly one of: (x,y,z) OR sel/selection_name OR file/infile."
            )

        # xyz mode: all-or-none (already floats)
        if has_any_xyz:
            if x is None or y is None or z is None:
                raise ValueError("acenter: x, y, z must all be provided together.")

        # selection mode: selection must be declared
        if has_sel:
            if sel_name not in self.selections_spec:
                raise ValueError(
                    f"acenter(sel='{sel_name}'): unknown selection name. "
                    "Define it first using select(name=..., ...)."
                )

        if has_file:
            if not path.isfile(infile):
                raise ValueError(f"acenter(infile='{infile}'), file does not exisy. ")

    def process_inputs(self):
        """
        Finalize and validate all input parameters after parsing.

        This method has TWO responsibilities:

        (A) Common parameter finalization (runs for BOTH modes)
        -------------------------------------------------------
        - Resolve solver / PB approximation settings (linear vs nonlinear)
        - Apply biomodel-dependent constraints (PBE vs RPBE)
        - Apply dielectric/surface constraints (two-dielectric vs gaussian, etc.)
        - Resolve convergence criteria activation (max_delta_phi vs max_rmsd)
        - Validate the global (biomodel, dielectric_model, surface_method, focusing, exponent) combo

        (B) Mode-dependent "materialization"
        -----------------------------------
        - static mode:
            * Reads structure-based inputs (PDB/PQR + SIZ/CRG)
            * Constructs self.atoms and self.objects
            * Applies sigma, focusing requirements, gridbox center (acenter/in_frc)
        - trajectory mode:
            * DOES NOT read structure files or build self.atoms here
            * Topology/trajectory assembly is handled in process_traj_inputs()
            * Some static-only concepts (e.g., focusing parent phi, FRC centering)
              are intentionally unsupported unless explicitly enabled later

        NOTE ON MODE_POLICY
        -------------------
        Parsing-time deny rules prevent mixing of static-only and traj-only inputs.
        This method must respect those policy boundaries to avoid hard-to-debug failures.

        Expected values:
            self.app_mode in {"static", "trajectory"}
        """

        # ------------------------------------------------------------------
        # 0) Cache frequently used parameter objects
        # ------------------------------------------------------------------
        params = self._get_cached_params()
        # print([(k, v.get()) for k, v in params.items() if type(v) is ParamStatement])

        solver_value = params["solver"].get()
        nonlinit_value = params["nonlinit"].get()
        surface_offset_value = params["surface_offset"].get()
        gaussian_exponent_value = params["gaussian_exponent"].get()

        biomodel_obj = params["biomodel"].get()
        dielectricmodel_obj = params["dielectricmodel"].get()
        surface_method_obj = params["surface_method"].get()
        bndcond_obj = params["boundary_condition"].get()
        pb_approx_obj = params["pb_approximation"].get()

        frc_obj = params["frc"]

        # ------------------------------------------------------------------
        # 1) Common: grid/solvent parameter resolution
        # ------------------------------------------------------------------
        self._process_grid_parameters(params)
        self._process_solvent_parameters(params)

        # ------------------------------------------------------------------
        # 2) Common: configure PB approximation and solver behavior
        # ------------------------------------------------------------------
        self._configure_pbe_solver(params, solver_value, nonlinit_value)

        # ------------------------------------------------------------------
        # 3) Common: biomodel-specific and dielectric/surface constraints
        # ------------------------------------------------------------------
        if biomodel_obj.int_value == BioModel.PBE.int_value:
            self._configure_pbe_biomodel(
                params, dielectricmodel_obj.int_value, surface_method_obj.int_value
            )
        elif biomodel_obj.int_value == BioModel.RPBE.int_value:
            self._configure_rpbe_biomodel(
                params, surface_method_obj.int_value, surface_offset_value
            )

        # Two-dielectric disables gaussian params
        if dielectricmodel_obj == DielectricModel.TWODIELECTRIC.int_value:
            params["dielectricmodel"].activate()
            params["gap_dielectric"].deactivate()
            params["surface_offset"].deactivate()
            params["surface_density_exponent"].deactivate()

        # Midpoint dielectric gaussian is currently treated as internal-only param
        params["midpoint_dielectric_gaussian"].deactivate()

        # ------------------------------------------------------------------
        # 4) Common: convergence criteria resolution
        # ------------------------------------------------------------------
        self._process_convergence_parameters(params)

        # ------------------------------------------------------------------
        # 5) Mode-dependent materialization
        # ------------------------------------------------------------------
        if self.app_mode == "static":
            # --------------------------------------------------------------
            # Static mode: read atoms from structure-based inputs
            # --------------------------------------------------------------
            atoms, objects = _read_atomic_data(
                in_modpdb4=params["in__modpdb4"],
                in_pdb=params["in__pdb"],
                in_crgsiz=params["in__crgsiz"],
                in_siz=params["in__siz"],
                in_crg=params["in__crg"],
            )

            # Assign VDW params if requested
            if params["in__vdw"].issupplied:
                _assign_vdw(atoms, params["in__vdw"])

            # Focusing: parent phi is required when BC == FOCUSING
            self._check_focusing_run_requirements(
                bndcond_obj.int_value, params["in__phi"]
            )

            # Gridbox center: acenter has priority; in_frc can override if supplied
            # self._set_gridbox_center(params["acenter"], params["in__frc"])

            # Assign per-atom gaussian sigma if needed
            self._set_gaussian_sigma(atoms, dielectricmodel_obj, params["sigma"])

            # Finalize into self.atoms / self.objects
            for a_key, a_data in atoms.items():
                self._add_atom(a_key, a_data.astype(delphi_real))
            self.objects = objects

        elif self.app_mode == "trajectory":
            # --------------------------------------------------------------
            # Trajectory mode: do NOT build self.atoms here.
            #
            # Topology/trajectory pairing and radii/charge provenance handling
            # are performed in process_traj_inputs(), called after this method.
            # --------------------------------------------------------------

            # self._set_gridbox_center(params["acenter"], params["in__frc"])

            # For now, explicitly block focusing in trajectory mode, since
            # MODE_POLICY denies in_phi and the parent phimap mechanism is not
            # yet designed for trajectories.
            if bndcond_obj.int_value == BoundaryCondition.FOCUSING.int_value:
                raise ValueError(
                    "FOCUSING boundary condition is not supported in trajectory mode "
                    "(parent phimap input is disabled)."
                )

            # Trajectory mode currently relies on acenter or auto-centering that
            # will be defined in process_traj_inputs()/TrajApp.
            # NOTE: in_frc is denied by policy in trajectory mode.

            pass

        else:
            raise ValueError(f"Unknown app_mode: {self.app_mode!r}")

        # ------------------------------------------------------------------
        # 6) Common: cross-parameter validity checks
        # ------------------------------------------------------------------
        if (
            pb_approx_obj.int_value == PBApproximation.NONLINEAR.int_value
            and biomodel_obj.int_value == BioModel.RPBE.int_value
        ):
            raise ValueError(
                "Non-linear PB works only with biomodel = PBE. "
                "Either use nonlinit=0 or change biomodel to PBE, then retry."
            )

        # Validate allowed global combination (applies to both modes)
        is_focusing = bndcond_obj.int_value == BoundaryCondition.FOCUSING.int_value
        input_combo = {
            "biomodel": biomodel_obj.name,
            "dielectric_model": dielectricmodel_obj.name,
            "surface_method": surface_method_obj.name,
            "is_focusing": is_focusing,
            "gaussian_exponent": gaussian_exponent_value,
        }

        if not self._is_valid_combination(**input_combo):
            raise ValueError(
                "Invalid parameter combination: "
                + ", ".join(f"{k}={v!r}" for k, v in input_combo.items())
            )

        # ------------------------------------------------------------------
        # 7) Input-level validation: frc
        # ------------------------------------------------------------------
        self._validate_frc()

        # ------------------------------------------------------------------
        # 8) Input-level validation: acenter
        # ------------------------------------------------------------------
        self._validate_acenter()

        # ------------------------------------------------------------------
        # 8) Input-level validation: out_selection
        # ------------------------------------------------------------------
        self._validate_out_selection_calls()

    def __str__(self):
        """
        Generate a string representation of the object, including the number of atoms
        and active parameters of type 'STATEMENT' or 'FUNCTION'.

        Returns:
        str: A formatted string representation of the object.
        """
        return self._generate_param_output(include_statements=True)

    def info_str(
        self,
        include_statements=True,
        include_functions=True,
        indent_spaces=4,
        field_width=50,
        format_specifier="s",
    ):
        """
        Generate a string containing information about the object, including the number
        of atoms and active parameters of type 'STATEMENT' or 'FUNCTION'.

        Similar to __str__, but may include additional logic if expanded in the future.

        Returns:
        str: A formatted string with object information.
        """
        if not (include_statements or include_functions):
            return ""

        return self._generate_param_output(
            include_statements=include_statements,
            include_functions=include_functions,
            indent_spaces=indent_spaces,
            field_width=field_width,
            format_specifier=format_specifier,
        )

    def _generate_param_output(
        self,
        include_statements,
        include_functions,
        indent_spaces,
        field_width,
        format_specifier,
    ):
        """
        Helper function to generate the parameter output for both __str__ and info_str.

        Parameters:
        include_statements (bool): Whether to include parameters of type 'STATEMENT'.

        Returns:
        str: A formatted string representing the object.
        """
        if not (include_statements or include_functions):
            return ""

        output_lines = []
        indent = " " * indent_spaces
        field_format = f"{{:{field_width}{format_specifier}}}"

        output_lines.append(
            f"{field_format.format('number_of_atoms')} = {len(self.atoms)}"
        )
        ignore_param_print = ["scale", "gridbox_margin", "perfil", "grid_size"]
        for k, prm in self.params.items():
            if prm.partype.value == ParamType.STATEMENT.value:
                if include_statements and prm.active and not str(prm).isspace():
                    ignore = any(param_k in ignore_param_print for param_k in k)
                    if not ignore:
                        output_lines.append(
                            prm.formatted_str(indent, field_width, format_specifier)
                        )

            elif (
                prm.partype.value == ParamType.FUNCTION.value
                and prm.active
                and include_functions
            ):
                output_lines.append(f"{indent}{str(prm)}")

        return "\n".join(output_lines)
