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
This module provides global configuration settings and utility functions for pyDelphi,
including precision control, verbosity management, and Numba-compatible printing.

It defines:
- `PRECISION`: The numerical precision (single or double) used throughout the application.
- `delphi_int`, `delphi_real`, `delphi_bool`, `delphi_uint`: NumPy data types
  dynamically set based on the chosen `PRECISION`.
- Functions to set and get precision and verbosity levels.
- Custom `print_if_verbose` and `nprint_cpu_if_verbose` functions for conditional printing.
"""

import numpy as np
from numba import njit

from pydelphi.foundation.enums import Precision, VerbosityLevel
import pydelphi.config.logging_config as logging_config
from pydelphi.config.logging_config import (
    VerbosityLevelValue,
)

# --- Configuration Variables and Initialization ---
PRECISION = Precision.DOUBLE  # Default precision

# Data type initialization:
delphi_int: type = None
delphi_real: type = None
delphi_bool: type = None
delphi_uint: type = None

delphi_bool = np.uint8  # Keep bool as uint8 regardless of precision


def _initialize_data_types():
    """Initializes Delphi data types based on the current precision.

    This function sets the appropriate NumPy data types for integers,
    real numbers, and mixed-precision real numbers based on the global `PRECISION` setting.
    It is called internally when the module is imported and whenever the precision level is changed.
    """
    global delphi_int, delphi_real, delphi_uint

    if PRECISION == Precision.SINGLE:
        delphi_int = np.int32
        delphi_uint = np.uint32
        delphi_real = np.float32
    elif PRECISION == Precision.DOUBLE:
        delphi_int = np.int64
        delphi_uint = np.uint64
        delphi_real = np.float64
    else:
        raise ValueError(f"Invalid precision: {PRECISION}")


def set_precision(prec: Precision):
    """Sets the precision level and re-initializes data types.

    Args:
        prec: The desired precision level (Precision enum member).
    """
    global PRECISION
    PRECISION = prec
    _initialize_data_types()


def set_verbosity_level(level: VerbosityLevel):
    """
    Sets the global verbosity level. This function delegates to logging_config.
    Higher `level` value means less verbose output.

    Args:
        level: The desired verbosity level (VerbosityLevel enum member).
    """
    logging_config.set_global_verbosity_level(level.int_value)
    # The message itself is DEBUG level (10)
    print_if_verbose(
        logging_config.DEBUG,  # This message's level (10)
        logging_config.get_effective_verbosity(
            __name__
        ),  # The configured threshold for global_runtime (e.g., 10)
        f"Configured global verbosity level to: {level.name} (value: {level.int_value})",
    )


# --- Print Functions ---
# Prints a message if the message_level is GREATER THAN or EQUAL TO the configured_verbosity_level.
# This means: if configured_verbosity_level is INFO (20), messages at INFO (20), WARNING (30), ERROR (40), CRITICAL (50) will print.
# Messages at DEBUG (10), TRACE (5) will NOT print.


def print_if_verbose(
    message_level: VerbosityLevelValue,
    configured_verbosity_level: VerbosityLevelValue,
    *args,
    sep=" ",
    end="\n",
    file=None,
    flush=False,
):
    """
    Prints a message if its level is GREATER THAN or EQUAL TO the configured verbosity level.
    (Aligned with Python logging's filtering where higher value = less verbose/more severe).

    Args:
        message_level: The verbosity level of the message (an integer value, e.g., logging_config.DEBUG which is 10).
        configured_verbosity_level: The determined effective verbosity level for the current context/module (integer value, e.g., logging_config.INFO which is 20).
        *args: The message arguments (like the standard print function).
        sep, end, file, flush: Same as the standard print function.
    """
    if message_level >= configured_verbosity_level:
        print(*args, sep=sep, end=end, file=file, flush=False)


@njit(cache=True)  # CPU target
def nprint_cpu_if_verbose(
    message_level: VerbosityLevelValue,
    configured_verbosity_level: VerbosityLevelValue,
    *args,
):
    """
    Numba-friendly print function for CPU targets.
    Prints a message if its level is GREATER THAN or EQUAL TO the configured verbosity level.
    (Aligned with Python logging's filtering where higher value = less verbose/more severe).

    Args:
        message_level: The verbosity level of the message (an integer value).
        configured_verbosity_level: The determined effective verbosity level for the current context/module (integer value).
        *args: The message arguments.
    """
    if message_level >= configured_verbosity_level:
        print(*args)


# --- Shorter Aliases for Frequent Use ---
# These aliases refer to the functions defined above.
vprint = print_if_verbose
nprint_cpu = nprint_cpu_if_verbose


def get_global_verbosity_level_value() -> VerbosityLevelValue:
    """Returns the current global verbosity level as an integer value."""
    return logging_config.get_global_verbosity_level()


# --- Initialize data types on module import ---
_initialize_data_types()

# Set initial global verbosity level in logging_config when global_runtime is imported.
logging_config.set_global_verbosity_level(VerbosityLevel.INFO.int_value)
