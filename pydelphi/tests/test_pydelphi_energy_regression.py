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
import csv
import time
import uuid
import textwrap
import tempfile
import argparse
import subprocess
from typing import Dict, List, Tuple, Any

# Local utility import expected in pydelphi environment
from pydelphi.utils.utils import seconds_to_hms

REFERENCE_FILE = "example-results-delphicpp-8_5_0.tsv"
TEST_REPORT_FILE = "pydelphi_regression_test_report.csv"

# --- TIER 2 INTERNAL CONSISTENCY CONFIGURATION ---
REFERENCE_CORE_CONFIG = ("cpu", "double", 1)
RPBE_MIN_THREADS = 5

# Tier 2 combined tolerances
PDELPHI_CONSISTENCY_RTOL = 0.0001  # 0.01%
PDELPHI_CONSISTENCY_ATOL_FLOOR = 1e-6

# --- TIER 1 EXTERNAL CONSISTENCY CONFIGURATION ---
TOLERANCES = {
    "E_rxn_corr_tot": 0.2,
    "E_grid_tot": 0.2,
    "E_coul": 0.2,
    "E_stress": 0.2,
    "E_osmotic": 0.2,
    "E_stress+E_osmotic": 0.5,
}

REFERENCE_ENERGY_KEYS = [k for k in TOLERANCES.keys() if k != "E_stress+E_osmotic"]

FIXED_ABS_TOL_FOR_ZERO_REF = 0.001
PERCENT_TOL_MINISCULE_REF = 0.65
PERCENT_TOL_TINY_REF = 0.50
PERCENT_TOL_SMALL_REF = 0.15
PERCENT_TOL_MEDIUM_REF = 0.01

# ---------------------------------------------------------------------------
# --- Controlled logging helper (explicit verbose param)
# ---------------------------------------------------------------------------

def log(msg: str, verbose: bool, always: bool = False) -> None:
    """
    Controlled logging helper.

    :param msg: Message to print.
    :param verbose: If True, verbose messages are printed.
    :param always: If True, message is printed regardless of verbose.
    """
    if always or verbose:
        print(msg)


# ---------------------------------------------------------------------------
# --- Tolerance helpers
# ---------------------------------------------------------------------------


def get_effective_tolerance(energy_key: str, ref_value: float) -> Tuple[float, str]:
    """
    Determines the TIER 1 (External Ref) dynamic tolerance.
    Returns (tolerance_value, tolerance_description).
    """
    abs_ref = abs(ref_value)

    if abs_ref == 0:
        return FIXED_ABS_TOL_FOR_ZERO_REF, "Abs (Ref=0)"

    if abs_ref <= 0.5:
        rtol = PERCENT_TOL_MINISCULE_REF
        tol_type = f"Rel ({rtol * 100:.1f}%, |Ref|≤0.5)"
    elif abs_ref <= 4:
        rtol = PERCENT_TOL_TINY_REF
        tol_type = f"Rel ({rtol * 100:.1f}%, 0.5<|Ref|≤4)"
    elif abs_ref <= 10:
        rtol = PERCENT_TOL_SMALL_REF
        tol_type = f"Rel ({rtol * 100:.1f}%, 4<|Ref|≤10)"
    elif abs_ref <= 100:
        rtol = PERCENT_TOL_MEDIUM_REF
        tol_type = f"Rel ({rtol * 100:.1f}%, 10<|Ref|≤100)"
    else:
        rtol = TOLERANCES.get(energy_key, PERCENT_TOL_MEDIUM_REF) / 100.0
        tol_type = f"Rel ({rtol * 100:.3f}%, |Ref|>100, by key)"

    return rtol * abs_ref, tol_type


def get_test_combinations(
    skip_cuda=False, skip_parallel=False, skip_single=False, skip_double=False
) -> Tuple[List[Tuple[str, str, int]], List[Dict[str, Any]]]:
    """
    Returns:
      - planned: list of (platform, precision, threads) to run
      - configuration_skips: list of dicts describing skipped configurations
    """
    all_combinations_base = [
        ("cpu", "single", 1),
        ("cpu", "double", 1),
        ("cpu", "single", 4),
        ("cpu", "double", 4),
        ("cuda", "single", 1),
        ("cuda", "double", 1),
        ("cuda", "single", 4),
        ("cuda", "double", 4),
    ]

    planned = []
    configuration_skips = []

    for platform, precision, threads in all_combinations_base:
        is_ref_core = (platform, precision, threads) == REFERENCE_CORE_CONFIG
        reason = None

        if skip_cuda and platform == "cuda":
            reason = "Skipped by --no-cuda flag (Hardware/Environment incompatibility)"
        elif skip_parallel and threads > 1:
            reason = "Skipped by --no-parallel flag"
        elif skip_single and precision == "single":
            reason = "Skipped by --no-single flag"
        elif skip_double and precision == "double":
            reason = "Skipped by --no-double flag"

        if reason:
            if is_ref_core:
                reason += " (Affects Reference Core)"

            configuration_skips.append(
                {
                    "platform": platform,
                    "precision": precision,
                    "threads": threads,
                    "reason": reason,
                    "test_type": "SKIPPED (Configuration Flag)",
                }
            )
        else:
            planned.append((platform, precision, threads))

    return planned, configuration_skips


def case_has_reference_energies(case_data: dict) -> bool:
    """Returns True if at least one external reference energy is present in case_data."""
    return any(case_data.get(k) is not None for k in REFERENCE_ENERGY_KEYS)


def parse_reference_data(filepath: str) -> List[dict]:
    """
    Parses the external reference TSV and returns a list of processed case dictionaries.
    """
    data = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            processed_row = {}

            for key in [
                "example",
                "bio_model",
                "dielectric_model",
                "surface_method",
                "solver",
                "boundary_condition",
                "salt",
                "indi",
                "exdi",
                "gapdi",
                "gaussian_exponent",
                "density_cutoff",
                "gaussian_sigma",
                "scale",
                "grid_size",
                "acenter",
                "probe_residue",
                "site",
            ]:
                if key in row:
                    if row[key] == "NA":
                        processed_row[key] = None
                    elif key in [
                        "salt",
                        "indi",
                        "exdi",
                        "gapdi",
                        "density_cutoff",
                        "gaussian_sigma",
                        "scale",
                    ]:
                        processed_row[key] = float(row[key])
                    elif key in ["grid_size", "gaussian_exponent"]:
                        try:
                            processed_row[key] = int(row[key])
                        except Exception:
                            processed_row[key] = None
                    else:
                        processed_row[key] = row[key]

            if "is_non_linear" in row and row["is_non_linear"] != "NA":
                processed_row["is_non_linear"] = row["is_non_linear"].lower() == "true"
            else:
                processed_row["is_non_linear"] = None

            reference_header_to_tolerance_key_map = {
                "E_rxn_corr_tot": "E_rxn_corr_tot",
                "E_grid_tot": "E_grid_tot",
                "E_coul": "E_coul",
                "E_stress": "E_stress",
                "E_osmotic": "E_osmotic",
            }

            for (
                ref_header,
                tolerance_key,
            ) in reference_header_to_tolerance_key_map.items():
                if row.get(ref_header) and row[ref_header] != "NA":
                    try:
                        processed_row[tolerance_key] = float(row[ref_header])
                    except Exception:
                        processed_row[tolerance_key] = None
                else:
                    processed_row[tolerance_key] = None

            data.append(processed_row)
    return data


def generate_param_file_content(case_data: dict, project_root: str) -> str:
    """
    Generate a delphi .prm content for a given test case (absolute paths).
    """
    content = []

    if "bio_model" in case_data and case_data["bio_model"] is not None:
        content.append(f"bio_model = {case_data['bio_model']}")
    if "dielectric_model" in case_data and case_data["dielectric_model"] is not None:
        content.append(f"dielectric_model = {case_data['dielectric_model']}")
    if "surface_method" in case_data and case_data["surface_method"] is not None:
        content.append(f"surface_method = {case_data['surface_method']}")
    if "solver" in case_data and case_data["solver"] is not None:
        content.append(f"solver = {case_data['solver']}")
    if (
        "boundary_condition" in case_data
        and case_data["boundary_condition"] is not None
    ):
        content.append(f"boundary_condition = {case_data['boundary_condition']}")

    if case_data.get("salt") is not None:
        content.append(f"salt_concentration = {case_data['salt']}")
    if case_data.get("indi") is not None:
        content.append(f"internal_dielectric = {case_data['indi']}")
    if case_data.get("exdi") is not None:
        content.append(f"external_dielectric = {case_data['exdi']}")
    if case_data.get("gapdi") is not None:
        content.append(f"gap_dielectric = {case_data['gapdi']}")
    if case_data.get("scale") is not None:
        content.append(f"scale = {case_data['scale']}")
    if case_data.get("grid_size") is not None:
        content.append(f"grid_size = {case_data['grid_size']}")
    if case_data.get("is_non_linear") is not None and case_data["is_non_linear"]:
        content.append(f"nonlinit = 10000")

    if case_data.get("gaussian_exponent") is not None:
        content.append(f"gaussian_exponent = {case_data['gaussian_exponent']}")
    if case_data.get("density_cutoff") is not None:
        content.append(f"density_cutoff = {case_data['density_cutoff']}")
    if case_data.get("gaussian_sigma") is not None:
        content.append(f"gaussian_sigma = {case_data['gaussian_sigma']}")

    example_name = case_data.get("example")

    def get_absolute_example_path(relative_path):
        return os.path.join(
            project_root, "pydelphi", "data", "test_examples", relative_path
        )

    # Example-specific file inclusions (keeps parity with previous implementation)
    if example_name == "sphere":
        content.append(f"in(pdb,file={get_absolute_example_path('sphere/sphere.pdb')})")
        content.append(
            f"in(crg, file={get_absolute_example_path('sphere/sphere.crg')})"
        )
        content.append(
            f"in(siz, file={get_absolute_example_path('sphere/sphere.siz')})"
        )
        content.append(
            f"in(vdw, file={get_absolute_example_path('sphere/amber99sb_sig-eps-gamma-1.vdw')})"
        )
    elif example_name == "twoatoms":
        content.append(
            f"in(modpdb4, file={get_absolute_example_path('twoatoms/two-atoms.pqr')}, format=pqr)"
        )
    elif example_name == "arg":
        content.append(
            f"in(modpdb4, file={get_absolute_example_path('arg/arg.pqr')}, format=pqr)"
        )
        content.append(
            f"in(vdw, file={get_absolute_example_path('arg/amber99sb_sig-eps-gamma.vdw')})"
        )
    elif example_name == "barnase":
        content.append(
            f"in(pdb, file={get_absolute_example_path('barnase/barnase.pdb')})"
        )
        content.append(
            f"in(crg, file={get_absolute_example_path('barnase/amber.crg')})"
        )
        content.append(
            f"in(siz, file={get_absolute_example_path('barnase/amber.siz')})"
        )
        content.append(
            f"in(vdw, file={get_absolute_example_path('barnase/amber99sb_sig-eps-gamma-1.vdw')})"
        )
    elif example_name == "5tif":
        content.append(
            f"in(modpdb4, file={get_absolute_example_path('5tif/5tif.pqr')}, format=pqr)"
        )
    elif example_name == "nonlinear":
        content.append(
            f"in(pdb, file={get_absolute_example_path('nonlinear/1brs.pdb')})"
        )
        content.append(
            f"in(crg, file={get_absolute_example_path('nonlinear/amber.crg')})"
        )
        content.append(
            f"in(siz, file={get_absolute_example_path('nonlinear/amber.siz')})"
        )

    if case_data.get("acenter") is not None:
        content.append(f"acenter({case_data['acenter']})")
    if case_data.get("probe_residue") is not None:
        content.append(f"probe_residue = {case_data['probe_residue']}")
    if case_data.get("site") is not None:
        content.append(f"site({case_data['site']})")

    return "\n".join(content)


def _read_calculated_energies(
    output_csv_path: str, tolerances_keys, test_report_row: dict, verbose: bool
) -> Dict[str, float]:
    """
    Reads calculated energies from the outputs.csv file.
    Returns a dictionary of calculated energies.
    """
    calculated_energies = {}
    if os.path.exists(output_csv_path):
        with open(output_csv_path, "r", newline="") as csvfile:
            data_lines = (line for line in csvfile if not line.strip().startswith("#"))
            reader = csv.DictReader(data_lines, delimiter="\t")
            for row in reader:
                for energy_abbr in tolerances_keys:
                    if energy_abbr in row and row[energy_abbr]:
                        try:
                            calculated_energies[energy_abbr] = float(row[energy_abbr])
                        except ValueError:
                            test_report_row[
                                "error_message"
                            ] += f"WARNING: Could not convert '{row[energy_abbr]}' to float for '{energy_abbr}' in outputs.csv. "
                            # Conversion warnings are noteworthy; print them so user sees possible data format issues.
                            log(
                                f"WARNING: Could not convert '{row[energy_abbr]}' to float for '{energy_abbr}' in outputs.csv.",
                                verbose,
                                always=True,
                            )
                break  # Only first data row is relevant
    else:
        raise FileNotFoundError(
            f"Output CSV file not found after run: {output_csv_path}"
        )
    return calculated_energies


def _compare_single_energy_external_ref(
    energy_abbr: str,
    ref_value: float,
    calc_value: float,
    test_report_row: dict,
    verbose: bool,
) -> bool:
    """
    Compares a single energy term against its External Reference (TIER 1).
    Returns True if test passes Tier 1 for this term.
    """
    test_report_row[f"{energy_abbr}_test"] = calc_value

    if ref_value is None:
        test_report_row[f"{energy_abbr}_pass"] = True
        return True

    if calc_value is None:
        log(
            f"ERROR: Could not find '{energy_abbr}' in pydelphi outputs.csv for {test_report_row.get('example_name')}.",
            verbose,
            always=True,
        )
        test_report_row[f"{energy_abbr}_pass"] = False
        return False

    diff = abs(calc_value - ref_value)
    test_report_row[f"{energy_abbr}_diff"] = diff

    # Exact zero reference
    if ref_value == 0:
        atol = FIXED_ABS_TOL_FOR_ZERO_REF
        pass_condition = diff <= atol
        tolerance_type_description = "Abs (Ref=0)"
        current_effective_tolerance = atol
    else:
        # Sign mismatch check
        if calc_value != 0 and (ref_value * calc_value < 0):
            log(
                f"FAIL (TIER 1 External Ref): {energy_abbr} - Sign mismatch. Ref: {ref_value:.4f}, Calc: {calc_value:.4f}",
                verbose,
                always=True,
            )
            pass_condition = False
            tolerance_type_description = "Sign Mismatch"
            current_effective_tolerance = "N/A"
        else:
            atol, tolerance_type_description = get_effective_tolerance(
                energy_abbr, ref_value
            )
            pass_condition = diff <= atol
            current_effective_tolerance = atol

    test_report_row[f"{energy_abbr}_pass"] = pass_condition
    test_report_row[f"{energy_abbr}_diff_type"] = tolerance_type_description
    test_report_row[f"{energy_abbr}_effective_tol"] = (
        f"{current_effective_tolerance:.4g}"
    )

    if not pass_condition:
        log(
            f"FAIL (TIER 1 External Ref): {energy_abbr} - Ref: {ref_value:.4f}, Calc: {calc_value:.4f}, Diff: {diff:.4f}, "
            f"Type: {tolerance_type_description} (Effective Tol: {test_report_row[f'{energy_abbr}_effective_tol']})",
            verbose,
            always=True,
        )
    return pass_condition


def _compare_pydelphi_consistency(
    energy_abbr: str,
    pydp_ref_value: float,
    calc_value: float,
    test_report_row: dict,
    verbose: bool,
) -> bool:
    """
    Compares calculated value against the PyDelphi Reference Core value (TIER 2 Internal Consistency).
    Returns True if consistent within PDELPHI_CONSISTENCY_RTOL/ATOL combination.
    """

    test_report_row[f"{energy_abbr}_pydp_ref"] = pydp_ref_value
    test_report_row[f"{energy_abbr}_pydp_diff"] = None
    test_report_row[f"{energy_abbr}_pydp_pass"] = "N/A"

    if pydp_ref_value is None or calc_value is None:
        if pydp_ref_value is None:
            test_report_row[f"{energy_abbr}_pydp_pass"] = "REF_CORE_MISSING"
        else:
            test_report_row[f"{energy_abbr}_pydp_pass"] = "CALC_MISSING"
        # Cannot meaningfully assert; treat as non-failing for the overall status
        return True

    diff = abs(calc_value - pydp_ref_value)
    max_allowed_diff = PDELPHI_CONSISTENCY_ATOL_FLOOR + PDELPHI_CONSISTENCY_RTOL * abs(
        pydp_ref_value
    )
    pass_condition = diff <= max_allowed_diff

    test_report_row[f"{energy_abbr}_pydp_diff"] = diff
    test_report_row[f"{energy_abbr}_pydp_pass"] = pass_condition

    if not pass_condition:
        log(
            f"FAIL (TIER 2 Internal Consistency): {energy_abbr} - Pydp Ref: {pydp_ref_value:.6f}, Calc: {calc_value:.6f}, Diff: {diff:.6f}, "
            f"Max Tol: {max_allowed_diff:.6f}",
            verbose,
            always=True,
        )
    return pass_condition


def _perform_lenient_stress_osmotic_test(
    case_data: dict, calculated_energies: dict, test_report_row: dict, verbose: bool
) -> None:
    """
    Performs a lenient sum test for E_stress and E_osmotic if individual TIER 1 tests failed.
    If sum check passes, individual TIER 1 flags are overridden to PASS.
    """
    ref_stress = case_data.get("E_stress")
    ref_osmotic = case_data.get("E_osmotic")
    calc_stress = calculated_energies.get("E_stress")
    calc_osmotic = calculated_energies.get("E_osmotic")

    if (
        ref_stress is None
        or ref_osmotic is None
        or calc_stress is None
        or calc_osmotic is None
    ):
        log(
            "INFO: Skipping lenient sum test for E_stress/E_osmotic due to missing reference or calculated values for sum.",
            verbose,
        )
        test_report_row["E_stress_osmotic_sum_pass"] = "SKIPPED"
        return

    sum_ref = ref_stress + ref_osmotic
    sum_calc = calc_stress + calc_osmotic
    sum_diff = abs(sum_calc - sum_ref)

    sum_same_sign = (sum_ref * sum_calc >= 0) or (sum_ref == 0 and sum_calc == 0)

    if sum_ref == 0:
        allowed_deviation = FIXED_ABS_TOL_FOR_ZERO_REF
        tol_type = "Abs (Ref Sum=0)"
    else:
        allowed_deviation, tol_type = get_effective_tolerance(
            "E_stress+E_osmotic", sum_ref
        )

    lenient_pass_condition = sum_diff <= allowed_deviation

    log(
        f"INFO: Lenient test for E_stress + E_osmotic: Ref Sum={sum_ref:.4f}, Calc Sum={sum_calc:.4f}, Diff={sum_diff:.4f}, Allowed={allowed_deviation:.4f} ({tol_type})",
        verbose,
    )

    if sum_same_sign and lenient_pass_condition:
        test_report_row["E_stress_osmotic_sum_pass"] = True
        if not test_report_row.get("E_stress_pass", False):
            log(
                f"OVERRIDE: E_stress TIER 1 status updated to PASS via lenient sum.",
                verbose,
                always=True,
            )
            test_report_row["E_stress_pass"] = True
        if not test_report_row.get("E_osmotic_pass", False):
            log(
                f"OVERRIDE: E_osmotic TIER 1 status updated to PASS via lenient sum.",
                verbose,
                always=True,
            )
            test_report_row["E_osmotic_pass"] = True
    else:
        test_report_row["E_stress_osmotic_sum_pass"] = False


def get_unique_csv_path(project_root: str) -> str:
    unique_name = f"temp_energies_{uuid.uuid4().hex}.csv"
    return os.path.join(project_root, unique_name)


def generate_skipped_report_row(
    case_data: dict,
    platform: str,
    precision: str,
    threads: int,
    test_type_label: str,
    reason: str,
) -> dict:
    """
    Generates a fully populated report row for a skipped test.
    """
    skipped_row = {
        "example_name": case_data.get("example") or "N/A",
        "salt": case_data.get("salt") or "N/A",
        "platform": platform,
        "precision": precision,
        "boundary_condition": case_data.get("boundary_condition") or "N/A",
        "threads": threads,
        "test_type": test_type_label,
        "status": "SKIPPED",
        "pydp_consistency_passed": "SKIPPED",
        "time_taken": 0.0,
        "E_stress_osmotic_sum_pass": "SKIPPED",
        "error_message": reason,
    }

    for energy_abbr in REFERENCE_ENERGY_KEYS:
        skipped_row[f"{energy_abbr}_ref"] = case_data.get(energy_abbr)
        skipped_row[f"{energy_abbr}_test"] = "SKIPPED"
        skipped_row[f"{energy_abbr}_diff"] = "SKIPPED"
        skipped_row[f"{energy_abbr}_effective_tol"] = "SKIPPED"
        skipped_row[f"{energy_abbr}_diff_type"] = "SKIPPED"
        skipped_row[f"{energy_abbr}_pass"] = "SKIPPED"

        skipped_row[f"{energy_abbr}_pydp_ref"] = "SKIPPED"
        skipped_row[f"{energy_abbr}_pydp_diff"] = "SKIPPED"
        skipped_row[f"{energy_abbr}_pydp_pass"] = "SKIPPED"

    return skipped_row


def run_delphi_subtest(
    case_data: dict,
    platform: str,
    precision: str,
    threads_to_execute: int,
    project_root: str,
    is_reference_core: bool,
    pydelphi_core_ref_values: dict = None,
    timeout: int = 300,
    verbose: bool = False,
) -> dict:
    """
    Runs a single pydelphi_static.py instance and compares against TIER 1 (External)
    and optionally TIER 2 (Internal PyDelphi Ref Core).
    """
    temp_file_path = None
    test_label = "Reference Core" if is_reference_core else "Execution Layer"

    test_report_row = {
        "example_name": case_data.get("example"),
        "salt": case_data.get("salt"),
        "platform": platform,
        "precision": precision,
        "boundary_condition": case_data.get("boundary_condition"),
        "threads": threads_to_execute,
        "test_type": test_label,
        "status": "FAIL",
        "pydp_consistency_passed": False,
        "error_message": "",
    }

    # Initialize fields
    for energy_type_abbr in REFERENCE_ENERGY_KEYS:
        test_report_row[f"{energy_type_abbr}_ref"] = case_data.get(energy_type_abbr)
        test_report_row[f"{energy_type_abbr}_test"] = None
        test_report_row[f"{energy_type_abbr}_diff"] = None
        test_report_row[f"{energy_type_abbr}_pass"] = False

        test_report_row[f"{energy_type_abbr}_pydp_ref"] = None
        test_report_row[f"{energy_type_abbr}_pydp_diff"] = None
        test_report_row[f"{energy_type_abbr}_pydp_pass"] = "N/A"

    test_report_row["E_stress_osmotic_sum_pass"] = False

    try:
        # write temporary parameter file
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".prm"
        ) as temp_file:
            param_content = generate_param_file_content(case_data, project_root)
            temp_file.write(param_content)
            temp_file_path = temp_file.name

        output_csv_path = get_unique_csv_path(project_root)

        # Build command
        command = [
            sys.executable,
            "-m",
            "pydelphi.scripts.pydelphi_static",
            "--platform",
            platform,
            "--precision",
            precision,
            "--threads",
            str(threads_to_execute),
            "--param-file",
            temp_file_path,
            "--label",
            case_data.get("example"),
            "--verbosity",
            "error",
            "--outfile",
            output_csv_path,
            "--overwrite",
        ]

        log(
            f"Running subtest ({test_label}): Platform={platform}, Precision={precision}, Threads={threads_to_execute}",
            verbose,
            always=True,
        )

        # Run pydelphi subprocess
        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=project_root,
            timeout=timeout,
        )

        # Parse outputs.csv
        calculated_energies = _read_calculated_energies(
            output_csv_path, REFERENCE_ENERGY_KEYS, test_report_row, verbose
        )

        # Clean temp output csv
        try:
            if os.path.exists(output_csv_path):
                os.remove(output_csv_path)
        except Exception:
            pass

        # --- TIER 1 (External ref) ---
        e_stress_individual_pass = False
        e_osmotic_individual_pass = False
        external_consistency_passed = True

        for energy_abbr in REFERENCE_ENERGY_KEYS:
            ref_value = case_data.get(energy_abbr)
            calc_value = calculated_energies.get(energy_abbr)

            individual_pass = _compare_single_energy_external_ref(
                energy_abbr, ref_value, calc_value, test_report_row, verbose
            )
            if not individual_pass:
                external_consistency_passed = False

            if energy_abbr == "E_stress":
                e_stress_individual_pass = individual_pass
            elif energy_abbr == "E_osmotic":
                e_osmotic_individual_pass = individual_pass

        # Lenient sum override for E_stress/E_osmotic
        if not e_stress_individual_pass or not e_osmotic_individual_pass:
            _perform_lenient_stress_osmotic_test(
                case_data, calculated_energies, test_report_row, verbose
            )

        tier_1_passed = all(
            test_report_row.get(f"{abbr}_pass") for abbr in REFERENCE_ENERGY_KEYS
        )

        # --- TIER 2 (Internal PyDelphi reference core) ---
        internal_consistency_passed = True

        if pydelphi_core_ref_values:
            log("Performing TIER 2 (Internal) consistency check...", verbose)
            for energy_abbr in REFERENCE_ENERGY_KEYS:
                pydp_ref_value = pydelphi_core_ref_values.get(energy_abbr)
                calc_value = calculated_energies.get(energy_abbr)

                is_consistent = _compare_pydelphi_consistency(
                    energy_abbr, pydp_ref_value, calc_value, test_report_row, verbose
                )
                # consider explicit False as failing
                if (
                    not is_consistent
                    and test_report_row.get(f"{energy_abbr}_pydp_pass") is False
                ):
                    internal_consistency_passed = False
        else:
            # If this run is the reference core, mark Tier2 as N/A
            test_report_row["pydp_consistency_passed"] = "N/A"
            internal_consistency_passed = True

        test_report_row["pydp_consistency_passed"] = internal_consistency_passed

        final_status_tag = (
            "PASS" if tier_1_passed and internal_consistency_passed else "FAIL"
        )
        test_report_row["status"] = final_status_tag
        return test_report_row

    except subprocess.CalledProcessError as e:
        test_report_row["status"] = "ERROR"
        test_report_row["pydp_consistency_passed"] = False
        test_report_row["error_message"] = (
            f"Subprocess Error (Exit Code {e.returncode}): {e.stderr.strip() or e.stdout.strip()}"
        )
        log(f"ERROR: Error running pydelphi: {e.stderr.strip()}", verbose, always=True)
        return test_report_row

    except subprocess.TimeoutExpired as e:
        test_report_row["status"] = "TIMEOUT"
        test_report_row["pydp_consistency_passed"] = False
        test_report_row["error_message"] = (
            f"Timeout Error: Command ran for too long ({e.timeout}s)"
        )
        log(
            f"TIMEOUT: Running pydelphi for {case_data.get('example')} ({platform}/{precision}/{threads_to_execute}) - Timeout: {e.timeout} seconds",
            verbose,
            always=True,
        )
        return test_report_row

    except FileNotFoundError as e:
        test_report_row["status"] = "FILE_ERROR"
        test_report_row["pydp_consistency_passed"] = False
        test_report_row["error_message"] = f"File Error: {str(e)}"
        log(f"FILE_ERROR: {e}", verbose, always=True)
        return test_report_row

    except Exception as e:
        test_report_row["status"] = "ERROR"
        test_report_row["pydp_consistency_passed"] = False
        test_report_row["error_message"] = f"Unexpected Error: {str(e)}"
        log(f"ERROR: An unexpected error occurred: {e}", verbose, always=True)
        return test_report_row

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
        try:
            if "output_csv_path" in locals() and os.path.exists(output_csv_path):
                os.remove(output_csv_path)
        except Exception:
            pass


def run_and_compare_all_combinations(
    case_index: int,
    case_data: dict,
    num_cases: int,
    project_root: str,
    combinations: List[Tuple[str, str, int]],
    timeout: int,
    verbose: bool,
) -> List[dict]:
    """
    For a single case: run the reference core (if present) then run all other combinations.
    Returns the list of report rows for this case.
    """
    case_reports: List[dict] = []
    start_all_combinations_time = time.time()

    # Tier 1 case skip
    if not case_has_reference_energies(case_data):
        skip_reason = "Skipped: None of the required energy columns have a reference value in the TSV (Tier 1 External Ref Missing)."
        log(
            f"CASE SKIP: {case_data.get('example')} - {skip_reason}",
            verbose,
            always=True,
        )

        for platform, precision, intended_threads in combinations:
            case_reports.append(
                generate_skipped_report_row(
                    case_data,
                    platform,
                    precision,
                    intended_threads,
                    "SKIPPED (No External Ref)",
                    skip_reason,
                )
            )
        return case_reports

    is_nonlinear = case_data.get("is_non_linear")
    is_nonlinear = str(is_nonlinear).lower() if is_nonlinear else "false"
    dielectric_model = (
        case_data.get("dielectric_model").upper()
        if case_data.get("dielectric_model")
        else ""
    )
    bio_model = case_data.get("bio_model").upper() if case_data.get("bio_model") else ""

    gaussian_params = ""
    if dielectric_model == "GAUSSIAN":
        gaussian_params = (
            f" \t(indi: {case_data.get('indi')}, exdi: {case_data.get('exdi')}, gapdi: {case_data.get('gapdi')}, "
            f" gaussian_exponent: {case_data.get('gaussian_exponent')}, sigma={case_data.get('gaussian_sigma')}, density_cutoff: {case_data.get('density_cutoff')}) \n"
        )

    log(
        f"  Processing case {case_index + 1}/{num_cases}: {case_data.get('example')} with key parameters: \n"
        f" \t(biomodel: {case_data.get('bio_model')}, dielectric_model: {case_data.get('dielectric_model')}, surface_method: {case_data.get('surface_method')}, \n"
        f"{gaussian_params}"
        f" \tsolver: {case_data.get('solver')}, is_nonlinear={is_nonlinear}, salt: {case_data.get('salt')}, boundary_condition={case_data.get('boundary_condition')})",
        verbose,
        always=True,
    )

    pydelphi_core_reference_results: Dict[str, float] = {}
    subtests_status = []
    # 1. Execute Reference Core First (if available in combinations)
    ref_platform, ref_precision, ref_threads_intended = REFERENCE_CORE_CONFIG
    ref_config_tuple = REFERENCE_CORE_CONFIG

    if ref_config_tuple in combinations:
        threads_to_execute = ref_threads_intended
        if case_data.get("bio_model").upper() == "RPBE" and ref_threads_intended == 1:
            threads_to_execute = RPBE_MIN_THREADS

        log(
            f"Running TIER 2 REFERENCE CORE: {ref_platform}/{ref_precision}/{threads_to_execute} threads",
            verbose,
        )

        start_time = time.time()
        ref_report_row = run_delphi_subtest(
            case_data,
            ref_platform,
            ref_precision,
            threads_to_execute,
            project_root,
            is_reference_core=True,
            pydelphi_core_ref_values=None,
            timeout=timeout,
            verbose=verbose,
        )
        elapsed_time = time.time() - start_time
        ref_report_row["time_taken"] = elapsed_time
        subtests_status.append(ref_report_row["status"])

        case_reports.append(ref_report_row)

        if ref_report_row.get("status") == "PASS":
            for energy_abbr in REFERENCE_ENERGY_KEYS:
                pydp_val = ref_report_row.get(f"{energy_abbr}_test")
                if pydp_val is not None:
                    pydelphi_core_reference_results[energy_abbr] = pydp_val

        # ensure we don't run the reference core again
        try:
            combinations.remove(ref_config_tuple)
        except ValueError:
            pass
    else:
        log(
            "WARNING: Reference Core Configuration was skipped by command-line flags. Tier 2 comparison will be unavailable.",
            verbose,
            always=True,
        )

    # 2. Execute Execution Layers
    for platform, precision, intended_threads in combinations:
        threads_to_execute = intended_threads
        if bio_model == "RPBE" and intended_threads == 1:
            threads_to_execute = RPBE_MIN_THREADS

        log(
            f"Running Execution Layer: {platform}/{precision}/{threads_to_execute} threads",
            verbose,
        )

        start_time = time.time()
        report_row = run_delphi_subtest(
            case_data,
            platform,
            precision,
            threads_to_execute,
            project_root,
            is_reference_core=False,
            pydelphi_core_ref_values=pydelphi_core_reference_results,
            timeout=timeout,
            verbose=verbose,
        )
        elapsed_time = time.time() - start_time

        if report_row:
            report_row["time_taken"] = elapsed_time
            case_reports.append(report_row)
            subtests_status.append(ref_report_row["status"])

    elapsed_all_combinations_time = time.time() - start_all_combinations_time
    log(
        f"Time taken for case {case_data.get('example')} combinations: {elapsed_all_combinations_time:.2f} seconds. \n",
        verbose,
        always=True,
    )
    case_status_unique = list(set(subtests_status))
    case_status = get_case_status(case_status_unique)
    return case_reports, case_status


def get_case_status(case_status_unique):
    """
    Returns a string based on the contents of the case_status_unique list,
    following a specific priority order.

    Args:
        case_status_unique (list): A list of strings representing case statuses.

    Returns:
        str: "FAIL", "ERROR", or "PASS" based on the specified conditions.
    """
    # Priority 1: Check for a "FAIL" status.
    if "FAIL" in case_status_unique:
        return "FAIL"

    # Priority 2: Check for any error keywords.
    # This includes "ERROR", "TIMEOUT", "FILE_ERROR", or "PASS OTHER".
    error_keywords = {"ERROR", "TIMEOUT", "FILE_ERROR"}
    if any(status in case_status_unique for status in error_keywords):
        return "ERRORS/TIMEOUTS"

    # Priority 3: Check if "PASS" is the only value.
    if case_status_unique == ["PASS"]:
        return "PASS"

    # Priority 3: Check if "PASS" is the only value.
    if case_status_unique == ["SKIPPED"]:
        return "SKIPPED"

    # Fallback for any other scenario, though the defined rules should cover most cases.
    return "UNKNOWN"


def print_splash_message(verbose: bool) -> None:
    """
    Prints the explanatory splash and methodology text.
    This is intentionally printed only in verbose mode to keep normal runs quiet.
    """

    def print_wrapped(text, initial="", subsequent=""):
        wrapper = textwrap.TextWrapper(
            width=80, initial_indent=initial, subsequent_indent=subsequent
        )
        log(wrapper.fill(text), verbose)

    import pydelphi as pydp  # local import

    log("=" * 80, verbose)
    log(
        f"PyDelphi-{pydp.__version__} Regression Test Suite (Two-Tier Validation)",
        verbose,
    )
    log("=" * 80, verbose)

    print_wrapped(
        "This suite enforces Two-Tier Validation for PyDelphi's energy calculations: Tier 1 (External Reference) "
        "and Tier 2 (Internal PyDelphi Reference Core).",
        "",
        "  ",
    )

    print_wrapped(
        "Tier 1: Compare each execution to values in the external TSV reference (delphicpp-8.5.0).",
        "  - ",
        "    ",
    )
    print_wrapped(
        "Tier 2: Cross-compare each execution against a designated PyDelphi Reference Core "
        "(CPU double minimal threads) using a combined relative/absolute tolerance.",
        "  - ",
        "    ",
    )

    print_wrapped(
        f"Reference Core (Tier 2): {REFERENCE_CORE_CONFIG[0].upper()}/{REFERENCE_CORE_CONFIG[1].upper()}/{REFERENCE_CORE_CONFIG[2]} nominal thread(s)",
        "  - ",
        "    ",
    )
    print_wrapped(
        f"Internal Tolerance: RTOL={PDELPHI_CONSISTENCY_RTOL}, ATOL floor={PDELPHI_CONSISTENCY_ATOL_FLOOR}",
        "  - ",
        "    ",
    )

    print_wrapped(
        "Test skipping logic: configurations skipped by flags are recorded in the report as SKIPPED.",
        "",
        "  ",
    )

    print_wrapped(
        f"RPBE override: single-thread target is replaced by {RPBE_MIN_THREADS} threads for performance.",
        "",
        "  ",
    )

    log("=" * 80, verbose)
    log("", verbose)


# ---------------------------------------------------------------------------
# --- Formatting for CSV output
# ---------------------------------------------------------------------------


def format_row_for_csv(row_data: dict) -> dict:
    """
    Formats float values in a dictionary to '14.6g' format,
    booleans to PASS/FAIL strings, leaving other types unchanged.
    """
    formatted_row = {}
    for key, value in row_data.items():
        if isinstance(value, float):
            formatted_row[key] = f"{value:14.6g}"
        elif isinstance(value, bool):
            formatted_row[key] = "PASS" if value else "FAIL"
        else:
            formatted_row[key] = value
    return formatted_row


# ---------------------------------------------------------------------------
# --- Main entrypoint
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Run PyDelphi regression tests (quiet by default)."
    )
    parser.add_argument(
        "--no-cuda", action="store_true", help="Skip tests involving CUDA platforms."
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Skip tests with more than 1 thread."
    )
    parser.add_argument(
        "--no-single", action="store_true", help="Skip tests with single precision."
    )
    parser.add_argument(
        "--no-double", action="store_true", help="Skip tests with double precision."
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Per-run timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print detailed stage-by-stage progress."
    )
    args = parser.parse_args()

    if args.no_single and args.no_double:
        log(
            "Error: Cannot skip both single and double precision.",
            args.verbose,
            always=True,
        )
        sys.exit(1)

    total_start_time = time.time()
    # Print splash only in verbose mode (quiet by default)
    print_splash_message(args.verbose)

    # Project root detection (assumes tests live inside pydelphi/tests/)
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    full_reference_file_path = os.path.join(
        project_root, "pydelphi", "data", "test_examples", REFERENCE_FILE
    )

    if not os.path.exists(full_reference_file_path):
        log(
            f"Error: Reference file not found at {full_reference_file_path}",
            args.verbose,
            always=True,
        )
        sys.exit(1)

    log(
        f"Loading external reference data from: {full_reference_file_path}",
        args.verbose,
    )
    reference_data = parse_reference_data(full_reference_file_path)
    num_cases = len(reference_data)
    log(f"Loaded {num_cases} unique test cases.", args.verbose)

    combinations, configuration_skips = get_test_combinations(
        skip_cuda=args.no_cuda,
        skip_parallel=args.no_parallel,
        skip_single=args.no_single,
        skip_double=args.no_double,
    )

    if not combinations and not configuration_skips:
        log(
            "No valid test configurations selected or generated. Exiting.",
            args.verbose,
            always=True,
        )
        sys.exit(1)

    log(
        f"Testing {len(combinations)} execution configuration(s) across {num_cases} test cases.",
        args.verbose,
    )
    log(
        f"Recording {len(configuration_skips)} configuration(s) as skipped per test case.",
        args.verbose,
    )

    case_status_list = []
    all_test_reports: List[dict] = []
    # Run tests for every case
    for case_index, case_data in enumerate(reference_data):
        case_reports, case_status = run_and_compare_all_combinations(
            case_index,
            case_data,
            num_cases,
            project_root,
            combinations.copy(),
            timeout=args.timeout,
            verbose=args.verbose,
        )
        all_test_reports.extend(case_reports)
        case_status_list.append(case_status)

    # Append configuration-skip rows for every case (for reporting completeness)
    for case_data in reference_data:
        for config_skip in configuration_skips:
            all_test_reports.append(
                generate_skipped_report_row(
                    case_data=case_data,
                    platform=config_skip["platform"],
                    precision=config_skip["precision"],
                    threads=config_skip["threads"],
                    test_type_label=config_skip["test_type"],
                    reason=config_skip["reason"],
                )
            )

    # --- Write CSV Report (if any results) ---
    if all_test_reports:
        base_keys = [
            "example_name",
            "salt",
            "platform",
            "precision",
            "boundary_condition",
            "threads",
            "test_type",
            "status",
            "pydp_consistency_passed",
            "time_taken",
        ]

        energy_report_keys = []
        for abbr in REFERENCE_ENERGY_KEYS:
            energy_report_keys.extend(
                [
                    f"{abbr}_ref",
                    f"{abbr}_test",
                    f"{abbr}_diff",
                    f"{abbr}_effective_tol",
                    f"{abbr}_diff_type",
                    f"{abbr}_pass",
                    f"{abbr}_pydp_ref",
                    f"{abbr}_pydp_diff",
                    f"{abbr}_pydp_pass",
                ]
            )

        final_keys = (
            base_keys
            + energy_report_keys
            + ["E_stress_osmotic_sum_pass", "error_message"]
        )

        with open(TEST_REPORT_FILE, "w", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=final_keys, extrasaction="ignore"
            )
            writer.writeheader()
            for row in all_test_reports:
                formatted_row = format_row_for_csv(row)
                writer.writerow(formatted_row)

    total_elapsed_time = time.time() - total_start_time
    total_hms = seconds_to_hms(total_elapsed_time)

    # Final summary (always printed)
    num_failed = sum(1 for row in all_test_reports if row.get("status") == "FAIL")
    num_errors = sum(
        1
        for row in all_test_reports
        if row.get("status") in ("ERROR", "TIMEOUT", "FILE_ERROR")
    )
    num_skipped = sum(1 for row in all_test_reports if row.get("status") == "SKIPPED")
    num_passed = len(all_test_reports) - num_failed - num_errors - num_skipped

    log("\n" + "=" * 80, args.verbose, always=True)
    log("REGRESSION TEST SUITE COMPLETE.", args.verbose, always=True)
    log(f"Total execution time: {total_hms}", args.verbose, always=True)
    log(f"Detailed results written to: {TEST_REPORT_FILE}", args.verbose, always=True)
    log("=" * 80 + "\n", args.verbose, always=True)

    num_subtests = len(all_test_reports)
    log(
        f"Among {num_subtests} subtests (case & configuration): PASS={num_passed}, FAIL={num_failed}, ERRORS/TIMEOUTS={num_errors}, SKIPPED={num_skipped}",
        args.verbose,
        always=True,
    )

    status_counts = {}
    for status in case_status_list:
        if status in status_counts:
            status_counts[status] += 1
        else:
            status_counts[status] = 1

    log(
        f"Test Case Summary: PASS={status_counts.get('PASS', 0)}, FAIL={status_counts.get('FAIL', 0)},"
        f"ERRORS/TIMEOUTS={status_counts.get('ERRORS/TIMEOUTS', 0)}, SKIPPED={status_counts.get('SKIPPED', 0)}",
        args.verbose,
        always=True,
    )

    # maintain previous behavior - nonzero exit for failures / errors
    if num_failed > 0 or num_errors > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
