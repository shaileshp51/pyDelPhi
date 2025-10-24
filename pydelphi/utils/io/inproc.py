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
from enum import Enum
from os import path
import textwrap as tw
from os.path import isfile
from typing import List, Tuple
import inspect

import numpy as np

# Import Delphi-specific configurations and enums
from pydelphi.config.global_runtime import (
    delphi_bool,
    delphi_int,
    delphi_real,
)
from pydelphi.foundation.enumbase import BaseInfoEnum
from pydelphi.foundation.enums import (
    PBSolver,
    BioModel,
    BoundaryCondition,
    DielectricModel,
    SoluteExtremaRule,
    GridBoxType,
    GridboxSize,
    ParamType,
    SurfaceMethod,
    MemoryState,
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
)


def bool_to_str(val: bool) -> str:
    return "TRUE" if val else "FALSE"


from pydelphi.utils.io.inproc_helpers.props_assigner import (
    _read_atomic_data,
    _assign_vdw,
    _set_param_func_attributes,
)


class Inputs:
    """
    Manages and organizes input parameters for Delphi Poisson-Boltzmann equation (PBE) simulations.

    The `Inputs` class is responsible for:
        - Storing and retrieving Delphi parameters, organized into logical groups.
        - Providing access to parameter groups and individual parameters.
        - Facilitating parameter initialization with default values.
        - Potentially handling parameter input from configuration files or command-line arguments (future enhancement).

    Attributes:
        params (dict): Dictionary storing all Delphi parameters, keyed by (full_name, long_name, short_name) tuple.
                       Values are DelphiParamStatement or DelphiParamFunction objects.
        atoms (dict):  Currently unused, potentially for storing atom-specific parameter information.
        objects (list): Currently unused, potentially for storing other simulation-related objects.
        param_groups (dict): Dictionary storing DelphiParameterGroup objects, keyed by group name (e.g., "pb", "dielectric").
    """

    def __init__(self):
        """
        Initializes the `Inputs` object and sets up the parameter groups and their associated
        default parameters. Each group represents a category of parameters used for PBE simulations.
        """
        self.gridbox_offset = np.zeros(3, dtype=float)
        self.params = {}
        self.atoms = {}
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
        Retrieve a parameter object by name.

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

    def _set_gridbox_center(self, acenter, in_frc):
        """
        Set the center of the grid box for the simulation.

        The grid box center is determined based on the following priority:
        1. If the 'acenter' parameter is supplied, its coordinates are used.
        2. If 'acenter' is not supplied but a valid force (FRC) file is supplied,
        the center is read from the FRC file and adjusted based on the gridbox offset
        and scale.

        Parameters:
        acenter (object): An object representing the 'acenter' parameter, expected to have
                      `issupplied` and `get_attribute` methods to retrieve x, y, z coordinates.
        in_frc (object): An object representing the input force params and file. It should have an
                        attribute 'file' that specifies the FRC file path.

        Raises:
        FileNotFoundError: If the FRC file specified in `in_frc` does not exist.
        """
        # Initialize gridbox center to a zero vector
        self.gridbox_center = np.zeros(3, dtype=np.float64)

        # Check if 'acenter' parameter is supplied and set the gridbox center accordingly
        if acenter.issupplied:
            self.gridbox_center[0] = acenter.get_attribute("x")
            self.gridbox_center[1] = acenter.get_attribute("y")
            self.gridbox_center[2] = acenter.get_attribute("z")

        # If 'acenter' is not supplied but FRC is provided, attempt to read from the FRC file
        elif in_frc.issupplied:
            frc_file = in_frc.get_attribute("file")
            if path.isfile(frc_file):
                # Read the FRC file and calculate the gridbox center
                self.gridbox_center[:] = calculate_center_of_frc_atoms(
                    frc_file,
                    self.gridbox_offset,
                    self.get_param_value("scale"),
                )
            else:
                raise FileNotFoundError(f"FRC file not found: '{frc_file}'")

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

    def _parse_function(self, line):
        """
        Parses a function-style input line and sets the corresponding parameter attributes.

        Args:
            line (str): The input line representing a function call, e.g., "acenter(x, y, z)".

        Raises:
            ValueError: If the function name or attributes are invalid.
        """
        func_name_end = line.find("(")
        if func_name_end == -1:
            raise ValueError(f"Invalid function format: {line}")

        func_name = line[:func_name_end].lower()
        attribs_str = line[func_name_end + 1 : line.rfind(")")].strip()
        attribs = [a.strip() for a in attribs_str.split(",") if a.strip()]
        try:
            if func_name in ("acenter", "acent", "ac"):
                if len(attribs) != 3:
                    raise ValueError("Exactly 3 values for (x, y, z) are required.")
                prm = self.get_param(func_name)
                _set_param_func_attributes(prm, attribs, expected_names=("x", "y", "z"))

            elif func_name in ("site", "calculate_energies", "energies", "energy"):
                prm = self.get_param(func_name)
                _set_param_func_attributes(prm, attribs, is_float=False)

            elif func_name in ("in", "out"):
                if not attribs:
                    raise ValueError(f"{func_name}(what=?); missing attribute `what`.")
                parm_name = func_name + "_" + attribs[0].lower()
                prm = self.get_param(parm_name)
                if not prm:
                    raise ValueError(f"Unknown function parameter: {parm_name}")
                _set_param_func_attributes(
                    prm, attribs[1:], is_float=False, file_check=func_name
                )
                if func_name == "out":
                    prm.activate()
        except (FileNotFoundError, ValueError) as e:
            print(f"{e}")
            sys.exit(1)

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
            print(f"Unrecognized statement format: {line}")
            return

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
            except ValueError as e:
                print(f"❌ {e}")
                sys.exit(1)
        else:
            raise ValueError(f"Unknown parameter: {param_name}")

    def parse_inputs(self, filename):
        """
        Parses an input file, processes comments, and handles parameter definitions or function calls.

        Args:
            filename (str): The name of the input file to parse.

        Raises:
            FileNotFoundError: If the specified file does not exist.
            IOError: If an error occurs during file reading.
        """
        try:
            with open(filename, "r") as file:
                for line in file:
                    cleaned_line = self._clean_line(line)
                    if cleaned_line is None:
                        continue

                    line_type = self._determine_line_type(cleaned_line)
                    if line_type == "function":
                        self._parse_function(cleaned_line)
                    elif line_type == "statement":
                        self._parse_statement(cleaned_line)
                    else:
                        print(
                            f"Warning: Ignoring unrecognized input specification: '{cleaned_line}'"
                        )

            self.process_inputs()
        except FileNotFoundError:
            print(f"❌ Error: Parameter file '{filename}' not found.")
            sys.exit(1)
        except IOError as e:
            print(
                f"❌ Error: An error occurred while reading the file '{filename}': {e}"
            )
            sys.exit(1)

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
            "in_modpdb4": self.get_param("in_modpdb4"),
            "in_pdb": self.get_param("in_pdb"),
            "in_siz": self.get_param("in_siz"),
            "in_crg": self.get_param("in_crg"),
            "acenter": self.get_param("acenter"),
            "in_frc": self.get_param("in_frc"),
            "in_vdw": self.get_param("in_vdw"),
            "in_phi": self.get_param("in_phi"),
            "calculate_energies": self.get_param("calculate_energies"),
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
                params["max_nonlinear_coupling_delta_phi"].deactivate()

    def _configure_pbe_biomodel(
        self, params, dielectricmodel_value, surface_method_value
    ):
        """Configures settings specific to PBE biomodel."""
        if dielectricmodel_value.int_value == DielectricModel.TWODIELECTRIC.int_value:
            params["dielectricmodel"].deactivate()
            params["gaussian_sigma"].deactivate()
            params["gaussian_exponent"].deactivate()
            params["surface_cutoff"].deactivate()
            params["density_cutoff"].deactivate()
            params["surface_method"].set(SurfaceMethod.VDW)
        elif dielectricmodel_value.int_value == DielectricModel.GAUSSIAN.int_value:
            if surface_method_value.int_value == SurfaceMethod.GAUSSIANCUTOFF.int_value:
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
        if surface_method_value.value in {
            SurfaceMethod.GCS.value,
            SurfaceMethod.GAUSSIAN.value,
        }:
            if surface_method_value.value == SurfaceMethod.GCS.value:
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

    def process_inputs(self):
        """
        Process and validate input parameters for running the Delphi electrostatics simulation.

        This method performs the following tasks:
        1. Initializes Delphi precision settings and related parameters.
        2. Activates or deactivates grid and solvent-related parameters based on user inputs.
        3. Reads input files (PDB) and assigns atomic properties (size and charge).
        4. Handles specific configurations for various biomodels (e.g., PBE and RPBE).
        5. Sets Gaussian sigma values for atoms if the dielectric model is Gaussian.
        6. Assigns grid box center or offset based on input parameters.
        """
        params = self._get_cached_params()

        # Cache parameter statement value
        solver_value = params["solver"].get()
        nonlinit_value = params["nonlinit"].get()
        biomodel_value = params["biomodel"].get()
        dielectricmodel_value = params["dielectricmodel"].get()
        surface_method_value = params["surface_method"].get()
        boundary_condition_value = params["boundary_condition"].get()
        surface_offset_value = params["surface_offset"].get()

        # Grid box parameter checks
        self._process_grid_parameters(params)

        # Solvent parameters check
        self._process_solvent_parameters(params)

        # Configure PBE solver and Approximation
        self._configure_pbe_solver(params, solver_value, nonlinit_value)

        # Configure PBE formalism or biomodel-specific settings
        if biomodel_value == BioModel.PBE:
            self._configure_pbe_biomodel(
                params, dielectricmodel_value, surface_method_value
            )
        elif biomodel_value == BioModel.RPBE.value:
            self._configure_rpbe_biomodel(
                params, surface_method_value, surface_offset_value
            )

        # print("Input: dielectricmodel_value=", dielectricmodel_value)
        if dielectricmodel_value == DielectricModel.TWODIELECTRIC:
            params["dielectricmodel"].activate()
            params["gap_dielectric"].deactivate()
            params["surface_offset"].deactivate()
            params["surface_density_exponent"].deactivate()

        params["midpoint_dielectric_gaussian"].deactivate()

        # max_delta_phi has priority over max_rmsd. max_rmsd is used for convergence only
        # when it is supplied and max_delta_phi not supplied
        self._process_convergence_parameters(params)

        # Read atomic coordinate, size(vdw radii), and charge data from PQR/(PDB+SIZ+CRG)
        try:
            atoms, objects = _read_atomic_data(
                params["in_modpdb4"],
                params["in_pdb"],
                params["in_siz"],
                params["in_crg"],
            )
        except (ValueError, FileNotFoundError) as e:
            print(e)
            sys.exit(1)

        # Assign VDW params if vdw energy requested and vdw params provided
        try:
            if params["in_vdw"].issupplied:
                _assign_vdw(
                    atoms,
                    params["in_vdw"],
                )
        except (ValueError, FileNotFoundError) as e:
            print(e)
            sys.exit(1)

        # Load phimap from parent run required for focusing run
        self._check_focusing_run_requirements(
            boundary_condition_value, params["in_phi"]
        )

        # Handle grid box settings considering acenter and in_frc for focusing runs
        self._set_gridbox_center(params["acenter"], params["in_frc"])

        # Set Gaussian sigma if the dielectric model is Gaussian
        self._set_gaussian_sigma(atoms, dielectricmodel_value, params["sigma"])

        # Assign the finalized atoms and objects data to the inputs object
        for a_key, a_data in atoms.items():
            self._add_atom(a_key, a_data.astype(delphi_real))
        self.objects = objects

        if (
            params["pb_approximation"].get().int_value
            == PBApproximation.NONLINEAR.int_value
            and params["biomodel"].get().int_value == BioModel.RPBE.int_value
        ):
            print(
                "\n\nInputError: Non-linear PB works only with biomodel = PBE.\nEither use nonlinit=0 or change biomodel to PBE, then re-try.\n\n"
            )
            sys.exit(0)

        input_combo = {
            "biomodel": self.get_param_value("biomodel").name,
            "dielectric_model": self.get_param_value("dielectric_model").name,
            "surface_method": self.get_param_value("surface_method").name,
            "is_focusing": self.get_param_value("boundary_condition").int_value
            == BoundaryCondition.FOCUSING.int_value,
            "gaussian_exponent": self.get_param_value("gaussian_exponent"),
        }

        if not self._is_valid_combination(**input_combo):
            from pprint import pprint

            print("❌ Invalid parameter combination:")
            pprint(input_combo)
            self._print_valid_combinations()
            raise SystemExit(1)

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
