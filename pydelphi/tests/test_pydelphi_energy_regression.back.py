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

from pydelphi.utils.utils import seconds_to_hms

REFERENCE_FILE = "example-results-delphicpp-8_5_0.tsv"

# Define output CSV report file name
TEST_REPORT_FILE = "pydelphi_regression_test_report.csv"

# Per-term percentage tolerances (used only if |ref| > 100)
TOLERANCES = {
    "E_rxn_corr_tot": 0.2,
    "E_grid_tot": 0.2,
    "E_coul": 0.2,
    "E_stress": 0.2,
    "E_osmotic": 0.2,
    "E_stress+E_osmotic": 0.5,  # More lenient sum tolerance if needed
}

FIXED_ABS_TOL_FOR_ZERO_REF = 0.001  # atol when ref == 0
# Tiered relative tolerances (fractions, not percent)
PERCENT_TOL_MINISCULE_REF = 0.65  # |ref| ≤ 0.5
PERCENT_TOL_TINY_REF = 0.50  # 0.5 < |ref| ≤ 4
PERCENT_TOL_SMALL_REF = 0.15  # 4 < |ref| ≤ 10
PERCENT_TOL_MEDIUM_REF = 0.01  # 10 < |ref| ≤ 100


def get_effective_tolerance(energy_key: str, ref_value: float) -> tuple[float, str]:
    abs_ref = abs(ref_value)

    if abs_ref == 0:
        return FIXED_ABS_TOL_FOR_ZERO_REF, "Abs (Ref=0)"

    if abs_ref <= 0.5:
        rtol = PERCENT_TOL_MINISCULE_REF
        tol_type = f"Rel ({rtol*100:.1f}%, |Ref|≤0.5)"
    elif abs_ref <= 4:
        rtol = PERCENT_TOL_TINY_REF
        tol_type = f"Rel ({rtol*100:.1f}%, 0.5<|Ref|≤4)"
    elif abs_ref <= 10:
        rtol = PERCENT_TOL_SMALL_REF
        tol_type = f"Rel ({rtol*100:.1f}%, 4<|Ref|≤10)"
    elif abs_ref <= 100:
        rtol = PERCENT_TOL_MEDIUM_REF
        tol_type = f"Rel ({rtol*100:.1f}%, 10<|Ref|≤100)"
    else:
        # Use key-specific tolerance or fallback
        rtol = TOLERANCES.get(energy_key, PERCENT_TOL_MEDIUM_REF) / 100.0
        tol_type = f"Rel ({rtol*100:.3f}%, |Ref|>100, by key)"

    return rtol * abs_ref, tol_type


# Define platform, precision, and thread combinations to test
def get_test_combinations(
    skip_cuda=False, skip_parallel=False, skip_single=False, skip_double=False
):
    all_combinations = [
        ("cpu", "single", 1),
        ("cpu", "double", 1),
        ("cpu", "single", 4),
        ("cpu", "double", 4),
        ("cuda", "single", 1),
        ("cuda", "double", 1),
        ("cuda", "single", 4),
        ("cuda", "double", 4),
    ]

    filtered = []
    for platform, precision, threads in all_combinations:
        if skip_cuda and platform == "cuda":
            continue
        if skip_parallel and threads > 1:
            continue
        if skip_single and precision == "single":
            continue
        if skip_double and precision == "double":
            continue
        filtered.append((platform, precision, threads))

    return filtered


def parse_reference_data(filepath):
    """
    Parses the reference results file.
    Maps reference file's column names to the new, abbreviated tolerance/CSV keys.
    """
    data = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            processed_row = {}  # Use a new dictionary to store processed data

            # Convert relevant fields to appropriate types, handling 'NA'
            for key in [
                "example",
                "bio_model",
                "dielectric_model",
                "surface_method",
                "solver",
                "boundary_condition",
                # General identifiers
                "salt",
                "indi",
                "exdi",
                "gapdi",
                "gaussian_exponent",
                "density_cutoff",
                "gaussian_sigma",
                # Numerical/Gaussian
                "scale",
                "grid_size",  # Integer
                "acenter",
                "probe_residue",
                "site",  # Specific to examples
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
                    elif key in [
                        "grid_size",
                        "gaussian_exponent",
                    ]:
                        processed_row[key] = int(row[key])
                    elif (
                        key == "is_non_linear"
                    ):  # Handle boolean separately if not already
                        processed_row[key] = row[key].lower() == "true"
                    else:
                        processed_row[key] = row[key]  # Keep as string for other fields

            # Handle 'is_non_linear' separately if it might be missing or 'NA' from the initial loop
            if "is_non_linear" in row and row["is_non_linear"] != "NA":
                processed_row["is_non_linear"] = row["is_non_linear"].lower() == "true"
            else:
                processed_row["is_non_linear"] = None

            # Mapping from reference file's column headers to the new TOLERANCES keys (which are CSV headers)
            # This is crucial because reference file headers might differ from CSV output headers.
            reference_header_to_tolerance_key_map = {
                "E_rxn_corr_tot": "E_rxn_corr_tot",
                "E_grid_tot": "E_grid_tot",
                "E_coul": "E_coul",
                "E_stress": "E_stress",
                "E_osmotic": "E_osmotic",
                # E_total_probe mapping is removed here.
            }

            for (
                ref_header,
                tolerance_key,
            ) in reference_header_to_tolerance_key_map.items():
                if row.get(ref_header) and row[ref_header] != "NA":
                    processed_row[tolerance_key] = float(row[ref_header])
                else:
                    processed_row[tolerance_key] = (
                        None  # Store under the new tolerance_key
                    )

            data.append(processed_row)
    return data


def generate_param_file_content(case_data, project_root):
    """
    Generates the content for a delphi parameter file based on case_data.
    All file paths included in the parameter file will be absolute.
    """
    content = []

    # Generic parameters, typically required
    # if "example" in case_data and case_data["example"] is not None:
    #     content.append(f"example_name = {case_data['example']}")
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

    # Numerical parameters
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
    if (
        case_data.get("is_non_linear") is not None and case_data["is_non_linear"]
    ):  # Only add if True
        content.append(
            f"nonlinit = 10000"
        )  # Assuming 10000 is default/appropriate for non-linear

    # Gaussian parameters (might be NA)
    if case_data.get("gaussian_exponent") is not None:
        content.append(f"gaussian_exponent = {case_data['gaussian_exponent']}")
    if case_data.get("density_cutoff") is not None:
        content.append(f"density_cutoff = {case_data['density_cutoff']}")
    if case_data.get("gaussian_sigma") is not None:
        content.append(f"gaussian_sigma = {case_data['gaussian_sigma']}")

    # Specific input file paths for each example, made absolute
    example_name = case_data.get("example")

    def get_absolute_example_path(relative_path):
        return os.path.join(
            project_root, "pydelphi", "data", "test_examples", relative_path
        )

    # Note: Updated paths based on your new structure pydelphi/data/test_examples
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

    # These are specific for certain examples in the reference table
    # Using .get() for safety against missing keys and checking for None/NA
    if case_data.get("acenter") is not None:
        content.append(f"acenter({case_data['acenter']})")
    if case_data.get("probe_residue") is not None:
        content.append(f"probe_residue = {case_data['probe_residue']}")
    if case_data.get("site") is not None:
        content.append(f"site({case_data['site']})")
    # print("\n".join(content))
    return "\n".join(content)


def _read_calculated_energies(output_csv_path, tolerances_keys, test_report_row):
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
                            print(
                                f"      WARNING: Could not convert '{row[energy_abbr]}' to float for '{energy_abbr}'."
                            )
                break  # Process only the first data row
    else:
        raise FileNotFoundError(
            f"Output CSV file not found after run: {output_csv_path}"
        )
    return calculated_energies


def _compare_single_energy(
    energy_abbr, ref_value, calc_value, base_absolute_tolerance, test_report_row
):
    """
    Compares a single energy term against its reference value and updates the report row.
    Returns True if the individual energy test passes, False otherwise.
    """
    test_report_row[f"{energy_abbr}_test"] = calc_value

    if ref_value is None:
        test_report_row[f"{energy_abbr}_pass"] = True
        return True  # No comparison if no reference

    if calc_value is None:
        print(
            f"      ERROR: Could not find '{energy_abbr}' in pydelphi outputs.csv for {test_report_row.get('example_name')}. "
            f"Check if this energy term is correctly written by pydelphi_static.py."
        )
        test_report_row[f"{energy_abbr}_pass"] = False
        return False

    pass_condition = False
    tolerance_type_description = ""
    current_effective_tolerance = ""

    diff = abs(calc_value - ref_value)
    test_report_row[f"{energy_abbr}_diff"] = diff

    # Rule 1: Exact zero
    if ref_value == 0:
        atol = FIXED_ABS_TOL_FOR_ZERO_REF
        pass_condition = diff <= atol
        tolerance_type_description = "Abs (Ref=0)"
        current_effective_tolerance = atol

    # Rule 2: Non-zero ref
    else:
        # Rule 2.1: Sign mismatch
        if calc_value != 0 and (ref_value * calc_value < 0):
            print(
                f"      FAIL: {energy_abbr} - Sign mismatch. Ref: {ref_value:.4f}, Calc: {calc_value:.4f}"
            )
            pass_condition = False
            tolerance_type_description = "Sign Mismatch"
            current_effective_tolerance = "N/A"

        else:
            # Rule 2.2: Normal tolerance check
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
        print(
            f"      FAIL: {energy_abbr} - Ref: {ref_value:.4f}, Calc: {calc_value:.4f}, Diff: {diff:.4f}, "
            f"Type: {tolerance_type_description} (Effective Tol: {test_report_row[f'{energy_abbr}_effective_tol']})"
        )
    return pass_condition


def _perform_lenient_stress_osmotic_test(
    case_data, calculated_energies, test_report_row
):
    """
    Performs a lenient sum test for E_stress and E_osmotic if individual tests failed.
    Updates the test_report_row accordingly.
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
        print(
            "      INFO: Skipping lenient sum test for E_stress/E_osmotic due to missing reference or calculated values for sum."
        )
        return

    sum_ref = ref_stress + ref_osmotic
    sum_calc = calc_stress + calc_osmotic
    sum_diff = abs(sum_calc - sum_ref)

    # Check 1: Same sign or both zero
    sum_same_sign = (sum_ref * sum_calc >= 0) or (sum_ref == 0 and sum_calc == 0)

    # Check 2: Tolerance-based check
    if sum_ref == 0:
        allowed_deviation = FIXED_ABS_TOL_FOR_ZERO_REF
        tol_type = "Abs (Ref Sum=0)"
    else:
        # Use dynamic or per-key tolerance
        allowed_deviation, tol_type = get_effective_tolerance(
            "E_stress+E_osmotic", sum_ref
        )

    lenient_pass_condition = sum_diff <= allowed_deviation

    print(
        f"      INFO: Lenient test for E_stress + E_osmotic: Ref Sum={sum_ref:.4f}, Calc Sum={sum_calc:.4f}, Diff={sum_diff:.4f}, Allowed={allowed_deviation:.4f} ({tol_type})"
    )

    if sum_same_sign and lenient_pass_condition:
        test_report_row["E_stress_osmotic_sum_pass"] = True
        test_report_row["E_stress_pass"] = True
        test_report_row["E_osmotic_pass"] = True
        print(f"      PASS: E_stress and E_osmotic lenient sum test passed.")
    else:
        test_report_row["E_stress_osmotic_sum_pass"] = False
        print(
            f"      FAIL: E_stress and E_osmotic lenient sum test failed (Sign check: {sum_same_sign}, Deviation check: {lenient_pass_condition})."
        )


def get_unique_csv_path(project_root):
    unique_name = f"temp_energies_{uuid.uuid4().hex}.csv"
    return os.path.join(project_root, unique_name)


def run_delphi_subtest(case_data, platform, precision, threads, project_root):
    """
    Runs a single pydelphi_static.py instance for a given case, platform, precision, and threads.
    Reads results from outputs.csv.
    Returns a dictionary of results for the CSV report or None if a critical error occurs.
    """
    temp_file_path = None

    test_report_row = {
        "example_name": case_data.get("example"),
        "salt": case_data.get("salt"),
        "platform": platform,
        "precision": precision,
        "boundary_condition": case_data.get("boundary_condition"),
        "threads": threads,
        "status": "FAIL",  # Default to FAIL, set to PASS if all checks pass
        "error_message": "",
    }

    # Initialize reference and calculated values in the report row using TOLERANCES keys
    for energy_type_abbr in TOLERANCES.keys():
        test_report_row[f"{energy_type_abbr}_ref"] = case_data.get(energy_type_abbr)
        test_report_row[f"{energy_type_abbr}_test"] = None  # Will be populated
        test_report_row[f"{energy_type_abbr}_diff"] = None  # Will be populated
        test_report_row[f"{energy_type_abbr}_pass"] = False  # Will be populated

    # Initialize for the lenient sum test
    test_report_row["E_stress_osmotic_sum_pass"] = False

    try:
        # Create a temporary file for parameters with absolute paths
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".prm"
        ) as temp_file:
            param_content = generate_param_file_content(case_data, project_root)
            temp_file.write(param_content)
            temp_file_path = temp_file.name

        output_csv_path = get_unique_csv_path(project_root)

        # Construct the command to run pydelphi_static.py as a module
        command = [
            sys.executable,  # Use the current Python interpreter
            "-m",
            "pydelphi.scripts.pydelphi_static",  # Module path for the main script
            "--platform",
            platform,
            "--precision",
            precision,
            "--threads",
            str(threads),
            "--param-file",
            temp_file_path,
            "--label",
            case_data.get("example"),
            "--verbosity",
            "error",
            "--outfile",
            output_csv_path,
            "--overwrite",  # Ensure it overwrites the file for each run
        ]

        print(
            f"    Running subtest: Platform={platform}, Precision={precision}, Threads={threads} using {temp_file_path}"
        )

        subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            cwd=project_root,
            timeout=600,
        )

        calculated_energies = _read_calculated_energies(
            output_csv_path, TOLERANCES.keys(), test_report_row
        )

        # Delete energy.csv file and keep dir clean.
        os.remove(output_csv_path)

        # --- Compare and populate report for this subtest ---
        e_stress_individual_pass = False
        e_osmotic_individual_pass = False

        for energy_abbr, base_absolute_tolerance in TOLERANCES.items():
            ref_value = case_data.get(energy_abbr)
            calc_value = calculated_energies.get(energy_abbr)

            individual_pass = _compare_single_energy(
                energy_abbr,
                ref_value,
                calc_value,
                base_absolute_tolerance,
                test_report_row,
            )

            if energy_abbr == "E_stress":
                e_stress_individual_pass = individual_pass
            elif energy_abbr == "E_osmotic":
                e_osmotic_individual_pass = individual_pass

        # --- Lenient test for E_stress and E_osmotic ---
        # Only apply the lenient test if at least one of them failed their individual test
        if not e_stress_individual_pass or not e_osmotic_individual_pass:
            _perform_lenient_stress_osmotic_test(
                case_data, calculated_energies, test_report_row
            )

        # Determine overall pass/fail status for the subtest after all checks, including lenient
        sub_test_passed_overall = True
        for energy_abbr in TOLERANCES.keys():
            if not test_report_row[f"{energy_abbr}_pass"]:
                sub_test_passed_overall = False
                break

        if sub_test_passed_overall:
            test_report_row["status"] = "PASS"
        else:
            test_report_row["status"] = "FAIL"

        return test_report_row

    except subprocess.CalledProcessError as e:
        test_report_row["status"] = "ERROR"
        test_report_row["error_message"] = (
            f"Subprocess Error (Exit Code {e.returncode}): {e.stderr.strip() or e.stdout.strip()}"
        )
        print(
            f"    Error running pydelphi for {case_data.get('example')} ({platform}/{precision}/{threads} threads):"
        )
        print(f"      Stderr: {e.stderr.strip()}")
        print(f"      Stdout: {e.stdout.strip()}")
        return test_report_row
    except subprocess.TimeoutExpired as e:
        test_report_row["status"] = "TIMEOUT"
        test_report_row["error_message"] = (
            f"Timeout Error: Command ran for too long ({e.timeout}s)"
        )
        print(
            f"    Timeout running pydelphi for {case_data.get('example')} ({platform}/{precision}/{threads} threads):"
        )
        print(f"      Timeout: {e.timeout} seconds")
        return test_report_row
    except FileNotFoundError as e:
        test_report_row["status"] = "FILE_ERROR"
        test_report_row["error_message"] = (
            f"File Error: {str(e)}. Check output path and pydelphi run."
        )
        print(f"    File Error for {case_data.get('example')}: {e}")
        return test_report_row
    except Exception as e:
        test_report_row["status"] = "ERROR"
        test_report_row["error_message"] = f"Unexpected Error: {str(e)}"
        print(
            f"    An unexpected error occurred for {case_data.get('example')} ({platform}/{precision}/{threads} threads): {e}"
        )
        return test_report_row
    finally:
        # Clean up temporary parameter file
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        # Clean up the outputs.csv file generated by the sub-process
        # This is important to ensure a clean slate for the next sub-test
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)


# It iterates over combinations and calls run_delphi_subtest for each.
def run_and_compare_all_combinations(
    case_index, case_data, num_cases, project_root, combinations
):
    case_reports = []
    is_nonlinear = case_data.get("is_non_linear")
    is_nonlinear = str(is_nonlinear).lower() if is_nonlinear else "false"
    gaussian_params = ""
    if case_data.get("bio_model").upper() == "GAUSSIAN":
        gaussian_params = (
            f" \t(indi: {case_data.get('indi')}, exdi: {case_data.get('exdi')}, gapdi: {case_data.get('gapdi')}, "
            f" gaussian_exponent: {case_data.get('gaussian_exponent')}, sigma={case_data.get('sigma')}, density_cutoff: {case_data.get('density_cutoff')}, \n"
        )

    print(
        f"  Processing case {case_index + 1}/{num_cases}: {case_data.get('example')} with key parameters: \n",
        f" \t(biomodel: {case_data.get('bio_model')}, dielectric_model: {case_data.get('dielectric_model')}, surface_method: {case_data.get('surface_method')}, \n",
        gaussian_params,
        f" \tsolver: {case_data.get('solver')}, is_nonlinear={is_nonlinear}, salt: {case_data.get('salt')}, boundary_condition={case_data.get('boundary_condition')})",
    )
    start_all_combinations_time = time.time()
    for platform, precision, threads in combinations:
        start_time = time.time()
        if case_data.get("bio_model").upper() == "RPBE" and threads == 1:
            print(
                "\t  **Known to be too slow on single thread** thus choosing 5 threads (still threads is odd like 1)."
            )
            threads = 5
        report_row = run_delphi_subtest(
            case_data, platform, precision, threads, project_root
        )
        elapsed_time = time.time() - start_time

        if report_row:
            report_row["time_taken"] = elapsed_time
            case_reports.append(report_row)
    elapsed_all_combinations_time = time.time() - start_all_combinations_time
    print(f"  Time taken: {elapsed_all_combinations_time:.2f} seconds")
    return case_reports


def print_splash_message():
    """Prints a splash message explaining the test comparison methodology."""

    # Wrapping utility
    def print_wrapped(text, initial="", subsequent=""):
        wrapper = textwrap.TextWrapper(
            width=80, initial_indent=initial, subsequent_indent=subsequent
        )
        print(wrapper.fill(text))

    import pydelphi as pydp

    print("=" * 80)
    print(f"PyDelphi-{pydp.__version__} Regression Test Suite")
    print("=" * 80)

    print_wrapped(
        "This suite compares computed energy values from PyDelphi against a reference "
        "dataset of values obtained from delphicpp_v8.5.0 if the model is present else other inhouse reference implementations",
        "",
        "  ",
    )

    print_wrapped(
        "Results for each energy term are assessed using a **dynamic tolerance system**:",
        "",
        "  ",
    )

    print_wrapped(
        f"If the reference value is exactly **0**: An absolute tolerance of **+/- {FIXED_ABS_TOL_FOR_ZERO_REF}** "
        f"is applied.",
        f"  - ",
        f"    ",
    )

    print_wrapped(
        "For **non-zero** reference values, two primary checks are performed:",
        "  - ",
        "    ",
    )

    print_wrapped(
        "**Sign Consistency**: Computed and reference values must have the same sign.",
        "    1. ",
        "       ",
    )

    print_wrapped(
        "**Relative Tolerance** (tiered based on absolute reference value):",
        "    2. ",
        "       ",
    )

    print_wrapped(
        "For |Reference| <= 0.5: Allowed deviation is **65.0%** of reference.",
        "       - ",
        "         ",
    )
    print_wrapped(
        "For 0.5 < |Reference| <= 4: Allowed deviation is **50.0%** of reference.",
        "       - ",
        "             ",
    )
    print_wrapped(
        "For 4 < |Reference| <= 10: Allowed deviation is **15.0%** of reference.",
        "       - ",
        "         ",
    )
    print_wrapped(
        "For 10 < |Reference| <= 100: Allowed deviation is **1.0%** of reference.",
        "       - ",
        "             ",
    )
    print_wrapped(
        "For |Reference| > 100: A **key-specific relative tolerance** is applied "
        "(e.g., 0.2% for `E_rxn_corr_tot`).",
        "       - ",
        "             ",
    )

    print()
    print_wrapped("**Special Handling for E_stress and E_osmotic**:", "", "  ")

    print_wrapped(
        "These terms are often very small and sensitive to iterative methods. If their "
        "individual tests fail, a lenient secondary test is performed:",
        "  ",
        "      ",
    )

    print_wrapped(
        "The **sum** of E_stress and E_osmotic (computed vs. reference) must have the "
        "**same sign**.",
        "  - ",
        "        ",
    )

    print_wrapped(
        "The absolute deviation of their sum is then assessed using the **same dynamic "
        "tolerance system** applied to other individual energies, but with a potentially "
        "more lenient base tolerance for |Ref| > 100 (currently set at **0.5%** for the "
        "sum).",
        "  - ",
        "       ",
    )

    print_wrapped(
        "If this lenient sum test passes, **both E_stress and E_osmotic will be marked "
        "as 'PASS'**, overriding their individual failures.",
        "  ",
        "     ",
    )

    print()
    print_wrapped(
        "Detailed test results are recorded in **pydelphi_regression_test_report.csv**.",
        "",
        "  ",
    )
    print("=" * 80)
    print()


def format_row_for_csv(row_data: dict) -> dict:
    """
    Formats float values in a dictionary to '14.6g' format,
    leaving other types (str, int) unchanged.

    Args:
        row_data: A dictionary representing a single row of data.

    Returns:
        A new dictionary with float values formatted as strings.
    """
    formatted_row = {}
    for key, value in row_data.items():
        if isinstance(value, float):
            # Format float to 14.6g (14 total width, 6 significant figures, general format)
            formatted_row[key] = f"{value:14.6g}"
        else:
            # Keep other types (int, str, None, etc.) as they are
            formatted_row[key] = value
    return formatted_row


def main():
    parser = argparse.ArgumentParser(description="Run PyDelphi regression tests.")
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
    args = parser.parse_args()

    # Validate mutual exclusion
    if args.no_single and args.no_double:
        print("Error: Cannot skip both single and double precision.")
        sys.exit(1)

    total_start_time = time.time()

    print_splash_message()  # Call the new splash message function

    # Get the absolute path to the project root
    script_dir = os.path.dirname(__file__)
    # Assumes test_delphi_energy_regression.py is in pydelphi/tests/
    # And project root is the directory above 'pydelphi'
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    # Updated path for reference file based on typical pydelphi project structure:
    full_reference_file_path = os.path.join(
        project_root, "pydelphi", "data", "test_examples", REFERENCE_FILE
    )

    if not os.path.exists(full_reference_file_path):
        print(f"Error: Reference file not found at {full_reference_file_path}")
        proj_rel_file_path = (
            "pydelphi/data/test_examples/example-results-delphicpp-8_5_0.tsv"
        )
        print(
            f"Please ensure {proj_rel_file_path} is in the correct location relative to your project root."
        )
        sys.exit(1)

    reference_cases = parse_reference_data(full_reference_file_path)
    # print(reference_cases)

    all_test_reports = []  # Collect all report rows here

    overall_pass_count = 0
    overall_fail_count = 0  # This will now count cases where at least one subtest failed/errored/timed out

    num_cases = len(reference_cases)
    combinations = get_test_combinations(skip_cuda=args.no_cuda)

    for case_index, case in enumerate(reference_cases):
        # Skip rows where primary reference energies are 'NA' as these might be focus runs or other types
        # Check against energy types in TOLERANCES to be flexible
        all_refs_none = True
        for energy_abbr in TOLERANCES.keys():
            if case.get(energy_abbr) is not None:
                all_refs_none = False
                break

        if all_refs_none:
            print(
                f"Skipping test case {case_index + 1} ({case.get('example')}) due to missing all primary reference energy values."
            )
            print("-" * 50)
            # Add a skipped row to the report for completeness
            skipped_row_base = {
                "example_name": case.get("example"),
                "salt": case.get("salt"),
                "platform": "N/A",
                "precision": "N/A",
                "threads": "N/A",
                "status": "SKIPPED",
                "error_message": "Missing all primary reference energies for comparison",
            }
            # Fill in N/A for energy details for skipped rows
            for energy_abbr in TOLERANCES.keys():
                skipped_row_base[f"{energy_abbr}_ref"] = case.get(energy_abbr)
                skipped_row_base[f"{energy_abbr}_test"] = None
                skipped_row_base[f"{energy_abbr}_diff"] = None
                skipped_row_base[f"{energy_abbr}_pass"] = False
            skipped_row_base["E_stress_osmotic_sum_pass"] = False  # For skipped cases
            all_test_reports.append(skipped_row_base)
            continue

        case_reports = run_and_compare_all_combinations(
            case_index, case, num_cases, project_root, combinations
        )
        all_test_reports.extend(case_reports)  # Add all subtest reports for this case

        # Determine overall case status from subtest reports for summary count
        case_overall_status = "PASS"
        if case_reports:  # Only check if there were actual subtests run
            for report_row in case_reports:
                if report_row["status"] != "PASS":
                    case_overall_status = (
                        "FAIL"  # Mark as fail if any subtest failed/errored/timed out
                    )
                    break
        else:  # If no subtests were run (e.g., due to an unexpected error before loop)
            case_overall_status = "ERROR_NO_SUBTESTS"

        if case_overall_status == "PASS":
            overall_pass_count += 1
        else:
            overall_fail_count += 1

        print("-" * 50)

    # --- Write the comprehensive test report to CSV ---
    if all_test_reports:
        # Determine all unique headers from all rows
        fieldnames = []
        # Ensure a consistent order for common fields
        common_fields = [
            "example_name",
            "salt",
            "platform",
            "precision",
            "threads",
            "boundary_condition",  # Added boundary_condition
            "status",
            "error_message",
        ]
        fieldnames.extend(common_fields)

        # Add energy-related fields ensuring consistent order
        for energy_abbr in TOLERANCES.keys():
            fieldnames.extend(
                [
                    f"{energy_abbr}_ref",
                    f"{energy_abbr}_test",
                    f"{energy_abbr}_diff",
                    f"{energy_abbr}_pass",
                    f"{energy_abbr}_diff_type",
                    f"{energy_abbr}_effective_tol",
                ]
            )
        fieldnames.append("E_stress_osmotic_sum_pass")  # Add the new field
        fieldnames.append("time_taken")

        # Open in write mode, creating if it doesn't exist
        with open(TEST_REPORT_FILE, "w", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=fieldnames, delimiter="\t"
            )  # Use tab as delimiter
            writer.writeheader()
            # writer.writerows(all_test_reports)
            # Apply the formatting function to each report before writing
            formatted_reports_for_writing = [
                format_row_for_csv(report) for report in all_test_reports
            ]

            # Write the pre-formatted rows
            writer.writerows(formatted_reports_for_writing)

        print(f"\nDetailed test report written to: {TEST_REPORT_FILE}")

    print(f"\n--- Final Test Summary ---")
    print(
        f"Total Unique Test Cases Processed: {overall_pass_count + overall_fail_count}"
    )
    print(f"Passed All Combinations for Cases: {overall_pass_count}")
    print(f"Failed in At Least One Combination for Cases: {overall_fail_count}")
    print()
    total_elapsed_time = time.time() - total_start_time
    formatted_time = seconds_to_hms(total_elapsed_time)
    print(f"\nTotal time taken for all tests: {formatted_time} seconds")
    print(f"--------------------------")


if __name__ == "__main__":
    main()
