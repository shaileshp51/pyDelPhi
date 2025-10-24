#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
Centralized module for managing global and module-specific verbosity levels.
It defines verbosity constants and provides functions to query the effective
verbosity for a given module, taking into account global settings and
module-specific overrides.
"""

from pydelphi.foundation.enums import VerbosityLevel as _VL
from typing import TypeAlias  # Import TypeAlias

# Define a TypeAlias for verbosity level integer values
VerbosityLevelValue: TypeAlias = int
# Note: This TypeAlias signifies that the integer should be one of the
#       VerbosityLevel enum's int_value members (e.g., logging_config.INFO).
#       Runtime validation (using _VALID_VERBOSITY_VALUES) enforces this.


# --- Verbosity Level Constants (using int_value directly for convenience) ---
CRITICAL = _VL.CRITICAL.int_value  # 50
ERROR = _VL.ERROR.int_value  # 40
NOTICE = _VL.NOTICE.int_value  # 35
WARNING = _VL.WARNING.int_value  # 30
INFO = _VL.INFO.int_value  # 20
DEBUG = _VL.DEBUG.int_value  # 10
TRACE = _VL.TRACE.int_value  # 5

# --- Valid Verbosity Values for stricter validation ---
# Create a frozenset of all valid integer values from the VerbosityLevel enum.
_VALID_VERBOSITY_VALUES = frozenset(level.int_value for level in _VL)

# Minimum/Maximum verbosity value for quick range checks (also covered by _VALID_VERBOSITY_VALUES)
MIN_VERBOSITY_VALUE = TRACE  # 5
MAX_VERBOSITY_VALUE = CRITICAL  # 50

# --- Module-specific Verbosity Configuration ---
_MODULE_VERBOSITY_SETTINGS = {
    # app
    "app.delphi": NOTICE,
    # config
    "config.global_runtime": NOTICE,
    "config.logging_config": NOTICE,
    # constants
    "constants.application": NOTICE,
    "constants.elements": NOTICE,
    "constants.physical": NOTICE,
    "constants.residues": NOTICE,
    # energy
    "energy.calculator": NOTICE,
    "energy.coulombic_nl": NOTICE,
    "energy.coulombic": NOTICE,
    "energy.gridenergy": NOTICE,
    "energy.lj": NOTICE,
    "energy.nonlinear": NOTICE,
    "energy.nonpolar": NOTICE,
    "energy.reactionfield": NOTICE,
    # foundation
    "foundation.bib_manager": NOTICE,
    "foundation.context": NOTICE,
    "foundation.data_models": NOTICE,
    "foundation.enumbase": NOTICE,
    "foundation.enums": NOTICE,
    "foundation.platforms": NOTICE,
    # site
    "site.site": NOTICE,
    "site.siteexceptions": NOTICE,
    "site.writesite": NOTICE,
    # solver
    "solver.core": NOTICE,
    "solver.solver": NOTICE,
    "solver.pb.common_pb": NOTICE,
    "solver.pb.nwt.base": NOTICE,
    "solver.pb.nwt.nonlinear_pb": NOTICE,
    "solver.pb.sor.nonlinear_pb": NOTICE,
    "solver.rpb.common_rpb": NOTICE,
    "solver.rpb.sor.helpers": NOTICE,
    "solver.rpb.sor.linear_rpb": NOTICE,
    "solver.shared.sor.base": NOTICE,
    # space.core
    "space.core.gaussian_naive": NOTICE,
    "space.core.gaussian": NOTICE,
    "space.core.grid_charger": NOTICE,
    "space.core.vdw.cuber": NOTICE,
    "space.core.vdw.helper_cpu": NOTICE,
    "space.core.vdw.helper_cuda": NOTICE,
    "space.core.vdw.helper": NOTICE,
    "space.core.vdw.molecular_surface": NOTICE,
    "space.core.vdw.sas": NOTICE,
    "space.core.vdw.scale_boundary": NOTICE,
    "space.core.vdw.vertex_indexer": NOTICE,
    "space.core.vdw_eps_initizer": NOTICE,
    "space.core.voxelizer": NOTICE,
    # space
    "space.epsindex_trad": NOTICE,
    "space.gcs_surface": NOTICE,
    "space.space": NOTICE,
    "space.surface": NOTICE,
    "space.vdwms": NOTICE,
    # utils.cuda
    "utils.cuda.double": NOTICE,
    "utils.cuda.single": NOTICE,
    # utils.io
    "utils.io.format.assorted.custom_reader": NOTICE,
    "utils.io.format.assorted.custom_writer": NOTICE,
    "utils.io.format.cube.cube": NOTICE,
    "utils.io.inproc": NOTICE,
    "utils.io.inproc_helpers.props_assigner": INFO,
    "utils.io.readers": NOTICE,
    "utils.io.writers": NOTICE,
    # utils
    "utils.interpolation": NOTICE,
    "utils.nonlinear": NOTICE,
    "utils.utils": NOTICE,
    # utils.prec
    "utils.prec.any": NOTICE,
    "utils.prec.double": NOTICE,
    "utils.prec.single": NOTICE,
}


_GLOBAL_VERBOSITY_LEVEL = INFO


def set_global_verbosity_level(level_value: VerbosityLevelValue):
    """
    Sets the global verbosity level.
    Higher `level_value` means less verbose output (more severe messages).
    Raises ValueError if `level_value` is not a valid VerbosityLevel integer.
    """
    global _GLOBAL_VERBOSITY_LEVEL
    if not isinstance(level_value, int) or level_value not in _VALID_VERBOSITY_VALUES:
        # Provide more specific error message if not in valid values
        raise ValueError(
            f"Invalid global verbosity level_value: {level_value}. Must be one of {sorted(list(_VALID_VERBOSITY_VALUES))}."
        )
    _GLOBAL_VERBOSITY_LEVEL = level_value


def get_global_verbosity_level() -> VerbosityLevelValue:
    """Returns the current global verbosity level (integer value)."""
    return _GLOBAL_VERBOSITY_LEVEL


def set_module_verbosity(module_name: str, level_value: VerbosityLevelValue):
    """
    Sets the specific verbosity level for a given module.
    Higher `level_value` means less verbose output for that module.
    Raises ValueError if `level_value` is not a valid VerbosityLevel integer.
    """
    if not isinstance(level_value, int) or level_value not in _VALID_VERBOSITY_VALUES:
        raise ValueError(
            f"Invalid module verbosity level_value for {module_name}: {level_value}. Must be one of {sorted(list(_VALID_VERBOSITY_VALUES))}."
        )
    _MODULE_VERBOSITY_SETTINGS[module_name] = level_value


def get_module_verbosity(module_name: str) -> VerbosityLevelValue:
    """
    Returns the explicitly configured verbosity level for a module.
    Returns the global verbosity level if the module is not explicitly configured.
    """
    return _MODULE_VERBOSITY_SETTINGS.get(module_name, _GLOBAL_VERBOSITY_LEVEL)


def get_effective_verbosity(module_name: str) -> VerbosityLevelValue:
    """
    Determines the effective verbosity level for a given module based on
    global settings and module-specific overrides.

    Rule: The most restrictive (highest value) level between global and module-specific wins.
    This means: "display only messages at or above the most restrictive (highest value) level."
    This aligns with Python logging where a higher "effective level" means less output.
    """
    global_level = get_global_verbosity_level()
    module_level = get_module_verbosity(module_name)

    return max(global_level, module_level)
