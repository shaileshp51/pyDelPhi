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


import os
import sys
import time
import argparse
import warnings

from pydelphi.foundation.enums import Precision, VerbosityLevel
from pydelphi.foundation.platforms import Platform

# Convert warnings to errors for debugging
warnings.filterwarnings("error", category=RuntimeWarning)

from pydelphi.utils.energy_terms import (
    ENERGY_TERM_ABBREVIATIONS,
    ENERGY_TERM_DESCRIPTIONS,
    REVERSE_ABBREVIATIONS,
)


def isfile(fname):
    if not os.path.isfile(fname):
        raise argparse.ArgumentTypeError(f"File not found: {fname}")
    return fname


def str_to_verbosity(value: str) -> VerbosityLevel:
    value_upper = value.upper()
    if not hasattr(VerbosityLevel, value_upper):
        raise ValueError(f"Invalid verbosity level: {value}")
    return getattr(VerbosityLevel, value_upper)


def check_output_file(outfile, overwrite):
    """
    Check if output file exists and overwrite is False. If so, exit with error.
    """
    if os.path.isfile(outfile) and not overwrite:
        print(f"Error: Output file '{outfile}' already exists.")
        print("       Use -O or --overwrite to overwrite it.")
        sys.exit(1)


def parse_arguments():
    platform_choices = tuple(p.lower() for p in Platform.get_available_names())
    precision_choices = tuple(p.lower() for p in Precision.list())
    verbosity_choices = tuple(v.lower() for v in VerbosityLevel.list())

    parser = argparse.ArgumentParser(
        prog="pydelphi_static.py",
        description="Run pydelphi for a single coordinate set (static PBE calculation).",
    )

    parser.add_argument(
        "-V",
        "--version",
        action="store_true",
        help="Print version of pydelphi and exit.",
    )
    parser.add_argument(
        "-P",
        "--platform",
        choices=platform_choices,
        default="cpu",
        help="Select platform to run calculation (default: cpu).",
    )
    parser.add_argument(
        "-p",
        "--precision",
        choices=precision_choices,
        default="double",
        help="Select real number precision (default: double).",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help="Number of threads/CPUs to use (default: 1).",
    )
    parser.add_argument(
        "-d",
        "--device-id",
        type=int,
        default=0,
        help="Device ID of the Nvidia GPU to use (default: 0).",
    )
    parser.add_argument(
        "-f",
        "--param-file",
        type=isfile,
        help="Input parameters file Required unless using '-h' or '-V'.",
        required=False,
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        choices=verbosity_choices,
        default="info",
        help="Verbosity level for outputs (default: info).",
    )
    parser.add_argument(
        "-l", "--label", default="pdbid", help="Label for this run (default: pdbid)."
    )
    parser.add_argument(
        "-o",
        "--outfile",
        default="outputs.csv",
        help="Output CSV file (default: outputs.csv).",
    )
    parser.add_argument(
        "-O",
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it exists (default: False).",
    )
    parser.add_argument(
        "-S",
        "--setup-timing",
        action="store_true",
        help="Print setup timings before DelphiApp run (default: False).",
    )
    return parser.parse_args()


def print_pydelphi_version_info():
    """Prints version information for pydelphi and its key dependencies."""
    import pydelphi
    import numpy
    import numba
    import platform

    print(f"\n--- PyDelphi Version Info ---")
    print(f"PyDelphi: v{pydelphi.__version__}")
    print(f"Python:   {platform.python_version()}")
    print(f"OS:       {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"NumPy:    v{numpy.__version__}")
    print(f"Numba:    v{numba.__version__}")
    print(f"-----------------------------\n")


def main():
    tic = time.perf_counter()
    from pydelphi.config.global_runtime import (
        set_precision,
        set_verbosity_level,
        get_global_verbosity_level_value,
    )

    args = parse_arguments()

    if not args.version:
        if not args.param_file:
            print(
                "Error: Parameter file is required but was not provided or does not exist."
            )
            print("Use -h to see usage.")
            exit(1)

    elif args.version:
        import pydelphi

        print_pydelphi_version_info()
        exit(1)

    # Before starting the calculation or opening the file
    check_output_file(args.outfile, args.overwrite)

    # Configure platform & precision
    platform = Platform(platform_name=args.platform, debug=False)
    platform.activate(args.platform, args.threads, args.device_id)
    platform.set_precision(Precision[args.precision.upper()])

    set_precision(platform.precision)
    set_verbosity_level(str_to_verbosity(args.verbosity))
    _VERBOSITY = get_global_verbosity_level_value()

    if args.setup_timing:
        print(f"Time> Setup complete in {time.perf_counter() - tic:.3f} sec")

    # Initialize and run DelphiApp
    from pydelphi.app.delphi import DelphiApp

    app = DelphiApp(args.param_file, platform, user_inputs=None)
    energies = app.run(args.outfile, args.label, args.overwrite)


if __name__ == "__main__":
    main()
