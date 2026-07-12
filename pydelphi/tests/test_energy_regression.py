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

from dataclasses import dataclass
from typing import Optional

# Local utility import expected in pydelphi environment
from pydelphi.utils.utils import seconds_to_hms

try:
    from pydelphi.utils.io.readers import compare_zphi
except (
    Exception
):  # pragma: no cover - keeps non-zeta tests usable if reader import changes
    compare_zphi = None

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
PERCENT_TOL_MINISCULE_REF = 0.30
PERCENT_TOL_TINY_REF = 0.20
PERCENT_TOL_SMALL_REF = 0.10
PERCENT_TOL_MEDIUM_REF = 0.01

indent = "  "

# --- ARTIFACT REFERENCE TEST CONFIGURATION ---
# Minimal extension for file-output regression tests such as FRC and focusing FRC.
# TSV convention:
#   parm_files: one parameter file, or two ordered files separated by " AND "
#   output_ref_file: checked-in reference artifact filename under the test case directory
# Paths in parm_files/output_ref_file are resolved under:
#   pydelphi/data/test_cases/<example>/
# Parameter-file templates may use these placeholders:
#   {EXAMPLE_DIR}, {TMPDIR}, {OUTPUT_FILE}, {PARENT_PHI_FILE}
# Artifact rows intentionally bypass TSV-generated parameter validation.
# For those rows, the checked-in .prm file is the source of truth.
ARTIFACT_PARAM_DELIMITER = " AND "
DEFAULT_ARTIFACT_RTOL = 1e-4
DEFAULT_ARTIFACT_ATOL = 1e-4
ARTIFACT_COMPARE_COLUMNS = [
    "X",
    "Y",
    "Z",
    "CHARGE",
    "GRID_PHI",
    "GF_EX",
    "GF_EY",
    "GF_EZ",
]


@dataclass
class SubtestSummary:
    tier: str  # RC or EC
    platform: str  # cpu / cuda
    precision: str  # single / double
    threads: int
    status: str  # PASS, FAIL, SKIPPED, etc.
    tier1_pass: Optional[bool]
    tier2_pass: Optional[bool]

    # --- new fields identifying which energy term caused the worst Δ ---
    worst_tier1_abbr: Optional[str] = None
    worst_tier1_diff: Optional[float] = None
    worst_tier1_ref: Optional[float] = None
    worst_tier2_abbr: Optional[str] = None
    worst_tier2_diff: Optional[float] = None
    worst_tier2_ref: Optional[float] = None

    # --- bookkeeping / metadata ---
    time_taken: float = 0.0
    error: Optional[str] = None


def log(msg: str, verbose: bool, always: bool = False) -> None:
    """
    Controlled logging helper.

    :param msg: Message to print.
    :param verbose: If True, verbose messages are printed.
    :param always: If True, message is printed regardless of verbose.
    """
    if always or verbose:
        print(msg)


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


def case_is_disabled(case_data: dict) -> bool:
    """Return True when a TSV row is temporarily disabled.

    The TSV convention is TRUE/FALSE, but a few simple truthy aliases are
    accepted for quick local editing. FALSE, NA, blank, and missing values run.
    """
    value = case_data.get("disabled")
    if value is None:
        return False
    return str(value).strip().lower() in {
        "true",
        "yes",
        "1",
        "skip",
        "disabled",
    }


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
                "parm_files",
                "output_ref_file",
                "testcase_rtol",
                "testcase_atol",
                "disabled",
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
                    elif key in ["testcase_rtol", "testcase_atol"]:
                        try:
                            processed_row[key] = float(row[key])
                        except Exception:
                            processed_row[key] = None
                    else:
                        processed_row[key] = row[key]

            # Backward-compatible defaults for TSVs without artifact columns.
            for optional_artifact_key in [
                "parm_files",
                "output_ref_file",
                "testcase_rtol",
                "testcase_atol",
                "disabled",
            ]:
                processed_row.setdefault(optional_artifact_key, None)

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
            project_root, "pydelphi", "data", "test_cases", relative_path
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
    elif example_name == "1he8":
        content.append(
            f"in(modpdb4, file={get_absolute_example_path('1he8/1he8.pqr')}, format=pqr)"
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
    # print("\n".join(content))
    return "\n".join(content)


def get_unique_csv_path(project_root: str) -> str:
    unique_name = f"temp_energies_{uuid.uuid4().hex}.csv"
    return os.path.join(project_root, unique_name)


def case_has_artifact_reference(case_data: dict) -> bool:
    """Returns True for TSV rows driven by checked-in output_ref_file artifacts."""
    return case_data.get("parm_files") not in (None, "", "NA") and case_data.get(
        "output_ref_file"
    ) not in (None, "", "NA")


def get_test_example_dir(project_root: str, case_data: dict) -> str:
    """Return the source test-data directory for a TSV row.

    Artifact-test paths are intentionally resolved from the case name so the TSV
    only needs filenames, not repeated directory prefixes.
    """
    example_name = case_data.get("example")
    if not example_name:
        raise ValueError("TSV row is missing required example name")

    return os.path.join(
        project_root,
        "pydelphi",
        "data",
        "test_cases",
        str(example_name),
    )


def resolve_case_file_path(
    project_root: str,
    case_data: dict,
    value: str | None,
) -> str | None:
    """Resolve a TSV artifact filename relative to this case's test directory."""
    if value in (None, "", "NA"):
        return None

    value = str(value).strip()
    if os.path.isabs(value):
        return value

    return os.path.join(get_test_example_dir(project_root, case_data), value)


def resolve_parm_files(case_data: dict, project_root: str) -> List[str]:
    """Parse ordered parm_files and resolve each filename under this case directory."""
    raw_value = case_data.get("parm_files")
    if raw_value in (None, "", "NA"):
        return []

    paths = []
    for part in str(raw_value).split(ARTIFACT_PARAM_DELIMITER):
        part = part.strip()
        if part:
            resolved = resolve_case_file_path(project_root, case_data, part)
            if resolved is not None:
                paths.append(resolved)
    return paths


def generated_artifact_path_from_ref(ref_path: str, temp_dir: str) -> str:
    """Derive temp output path from a checked-in *.ref.* artifact name."""
    basename = os.path.basename(ref_path)
    if ".ref." in basename:
        out_basename = basename.replace(".ref.", ".", 1)
    elif basename.endswith(".ref"):
        out_basename = basename[: -len(".ref")]
    else:
        root, ext = os.path.splitext(basename)
        out_basename = f"{root}.out{ext}"
    return os.path.join(temp_dir, out_basename)


def get_case_artifact_tolerances(case_data: dict) -> Tuple[float, float]:
    """Return (rtol, atol), with TSV values overriding artifact defaults."""
    rtol = case_data.get("testcase_rtol")
    atol = case_data.get("testcase_atol")
    if rtol is None:
        rtol = DEFAULT_ARTIFACT_RTOL
    if atol is None:
        atol = DEFAULT_ARTIFACT_ATOL
    return float(rtol), float(atol)


def render_param_template_to_temp(
    *,
    source_param_file: str,
    temp_dir: str,
    replacements: Dict[str, str],
) -> str:
    """Render a checked-in parameter template to a temporary parameter file."""
    with open(source_param_file, "r", encoding="utf-8") as handle:
        content = handle.read()

    for key, value in replacements.items():
        content = content.replace("{" + key + "}", value)

    fd, temp_param_file = tempfile.mkstemp(
        prefix=os.path.basename(source_param_file) + ".",
        suffix=".prm",
        dir=temp_dir,
        text=True,
    )
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(content)
    return temp_param_file


def parse_frc_artifact(path: str) -> Tuple[List[dict], Optional[float]]:
    """Parse FRC-like tabular artifacts and optional total electrostatic energy."""
    rows: List[dict] = []
    total_energy: Optional[float] = None
    header: Optional[List[str]] = None

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue

            if stripped.startswith("#"):
                if (
                    "total electrostatic energy" in stripped.lower()
                    or "total energy" in stripped.lower()
                ):
                    cleaned = stripped.replace("#", " ").replace("=", " ")
                    for token in cleaned.split():
                        try:
                            total_energy = float(token)
                            break
                        except ValueError:
                            continue
                continue

            parts = stripped.split()
            if not parts:
                continue

            if parts[0] in {"ATOM", "X"}:
                header = parts
                continue

            if header is not None:
                rows.append(dict(zip(header, parts)))

    return rows, total_energy


def compare_frc_artifacts(
    *,
    ref_file: str,
    out_file: str,
    rtol: float,
    atol: float,
) -> Tuple[bool, dict]:
    """Compare numeric columns in generated FRC artifact against checked-in reference."""
    ref_rows, ref_energy = parse_frc_artifact(ref_file)
    out_rows, out_energy = parse_frc_artifact(out_file)

    result = {
        "artifact_ref_file": ref_file,
        "artifact_out_file": out_file,
        "artifact_pass": True,
        "artifact_rows_ref": len(ref_rows),
        "artifact_rows_out": len(out_rows),
        "artifact_worst_column": None,
        "artifact_worst_row": None,
        "artifact_worst_ref": None,
        "artifact_worst_out": None,
        "artifact_worst_diff": None,
        "artifact_rtol": rtol,
        "artifact_atol": atol,
        "artifact_error": "",
        "artifact_energy_ref": ref_energy,
        "artifact_energy_out": out_energy,
        "artifact_energy_diff": None,
    }

    if len(ref_rows) != len(out_rows):
        result["artifact_pass"] = False
        result["artifact_error"] = (
            f"row count mismatch: ref={len(ref_rows)}, out={len(out_rows)}"
        )
        return False, result

    worst_diff = -1.0

    for row_index, (ref_row, out_row) in enumerate(zip(ref_rows, out_rows), start=1):
        for column in ARTIFACT_COMPARE_COLUMNS:
            if column not in ref_row:
                continue
            if column not in out_row:
                result["artifact_pass"] = False
                result["artifact_error"] = (
                    f"missing column {column!r} in output row {row_index}"
                )
                return False, result

            try:
                ref_value = float(ref_row[column])
                out_value = float(out_row[column])
            except ValueError:
                continue

            diff = abs(out_value - ref_value)
            allowed = atol + rtol * abs(ref_value)

            if diff > worst_diff:
                worst_diff = diff
                result["artifact_worst_column"] = column
                result["artifact_worst_row"] = row_index
                result["artifact_worst_ref"] = ref_value
                result["artifact_worst_out"] = out_value
                result["artifact_worst_diff"] = diff

            if diff > allowed:
                result["artifact_pass"] = False

    if ref_energy is not None:
        if out_energy is None:
            result["artifact_pass"] = False
            result["artifact_energy_error"] = (
                "missing total electrostatic energy in output"
            )
        else:
            energy_diff = abs(out_energy - ref_energy)
            energy_allowed = atol + rtol * abs(ref_energy)
            result["artifact_energy_diff"] = energy_diff
            if energy_diff > energy_allowed:
                result["artifact_pass"] = False

    return result["artifact_pass"], result


def artifact_reference_kind(ref_file: str) -> str:
    """Return artifact comparison kind based on reference filename."""
    lower_name = os.path.basename(str(ref_file)).lower()

    if lower_name.endswith(".zphi"):
        return "zphi"

    # Allow .zeta during transition if the checked-in reference contains
    # metadata-rich ZPHI content but keeps the older extension.
    if lower_name.endswith(".zeta"):
        try:
            with open(ref_file, "r", encoding="utf-8") as handle:
                for _ in range(12):
                    line = handle.readline()
                    if not line:
                        break
                    if line.startswith("# ZPHI_VERSION:"):
                        return "zphi"
        except OSError:
            pass

    return "frc"


def compare_zeta_artifacts(
    *,
    ref_file: str,
    out_file: str,
    rtol: float,
    atol: float,
) -> Tuple[bool, dict]:
    """Compare generated ZPHI/zeta artifact against checked-in reference."""
    if compare_zphi is None:
        return False, {
            "artifact_ref_file": ref_file,
            "artifact_out_file": out_file,
            "artifact_pass": False,
            "artifact_kind": "zphi",
            "artifact_error": (
                "compare_zphi is unavailable; check pydelphi.utils.io.custom_reader import"
            ),
            "artifact_rtol": rtol,
            "artifact_atol": atol,
        }

    passed, zphi_result = compare_zphi(
        ref_file,
        out_file,
        rtol=rtol,
        atol=atol,
    )

    result = {
        "artifact_ref_file": ref_file,
        "artifact_out_file": out_file,
        "artifact_pass": passed,
        "artifact_kind": "zphi",
        "artifact_rows_ref": zphi_result.get("zphi_ref_points"),
        "artifact_rows_out": zphi_result.get("zphi_out_points"),
        "artifact_worst_column": zphi_result.get("zphi_worst_field"),
        "artifact_worst_row": None,
        "artifact_worst_ref": zphi_result.get("zphi_worst_ref_phi"),
        "artifact_worst_out": zphi_result.get("zphi_worst_out_phi"),
        "artifact_worst_diff": zphi_result.get("zphi_worst_abs_diff"),
        "artifact_rtol": rtol,
        "artifact_atol": atol,
        "artifact_error": zphi_result.get("zphi_error", ""),
        "artifact_energy_ref": None,
        "artifact_energy_out": None,
        "artifact_energy_diff": None,
        "zphi_pass": zphi_result.get("zphi_pass"),
        "zphi_ref_points": zphi_result.get("zphi_ref_points"),
        "zphi_out_points": zphi_result.get("zphi_out_points"),
        "zphi_worst_field": zphi_result.get("zphi_worst_field"),
        "zphi_worst_ix": zphi_result.get("zphi_worst_ix"),
        "zphi_worst_iy": zphi_result.get("zphi_worst_iy"),
        "zphi_worst_iz": zphi_result.get("zphi_worst_iz"),
        "zphi_worst_ref_phi": zphi_result.get("zphi_worst_ref_phi"),
        "zphi_worst_out_phi": zphi_result.get("zphi_worst_out_phi"),
        "zphi_worst_delta_phi": zphi_result.get("zphi_worst_delta_phi"),
        "zphi_worst_abs_diff": zphi_result.get("zphi_worst_abs_diff"),
        "zphi_worst_relative_error": zphi_result.get("zphi_worst_relative_error"),
        "zphi_worst_allowed_diff": zphi_result.get("zphi_worst_allowed_diff"),
        "zphi_num_potential_failures": zphi_result.get("zphi_num_potential_failures"),
        "zphi_num_metadata_float_failures": zphi_result.get(
            "zphi_num_metadata_float_failures"
        ),
        "zphi_error": zphi_result.get("zphi_error", ""),
    }

    return bool(passed), result


def compare_artifacts_by_kind(
    *,
    ref_file: str,
    out_file: str,
    rtol: float,
    atol: float,
) -> Tuple[bool, dict]:
    """Dispatch artifact comparison based on checked-in reference artifact kind."""
    kind = artifact_reference_kind(ref_file)

    if kind == "zphi":
        return compare_zeta_artifacts(
            ref_file=ref_file,
            out_file=out_file,
            rtol=rtol,
            atol=atol,
        )

    passed, result = compare_frc_artifacts(
        ref_file=ref_file,
        out_file=out_file,
        rtol=rtol,
        atol=atol,
    )
    result["artifact_kind"] = "frc"
    return passed, result


def run_param_file_no_energy_parse(
    *,
    param_file: str,
    project_root: str,
    platform: str,
    precision: str,
    threads: int,
    timeout: int,
    label: str,
    output_csv_path: str,
) -> None:
    """Run pydelphi_static for artifact tests without parsing outputs.csv."""
    command = [
        sys.executable,
        "-m",
        "pydelphi.scripts.pydelphi_static",
        "--platform",
        platform,
        "--precision",
        precision,
        "--threads",
        str(threads),
        "--param-file",
        param_file,
        "--label",
        label,
        "--verbosity",
        "error",
        "--outfile",
        output_csv_path,
        "--overwrite",
    ]

    subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
        cwd=project_root,
        timeout=timeout,
    )


def print_splash_message(
    verbose: bool, has_artifact_reference_cases: bool = False
) -> None:
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
        "This suite enforces Two-Tier Validation for PyDelphi's energy calculations: "
        "Tier 1 (External Reference) and Tier 2 (Internal PyDelphi Reference Core).",
        "",
        "  ",
    )

    print_wrapped(
        "Tier 1 (External Reference): Compare each execution to values in the "
        "external TSV reference (delphicpp-8.5.0).",
        "  - ",
        "    ",
    )
    print_wrapped(
        "Tier 2 (Internal Reference Core, RC): Cross-compare each execution against "
        "a designated PyDelphi Reference Core (CPU double minimal threads) using a "
        "combined relative/absolute tolerance.",
        "  - ",
        "    ",
    )
    print_wrapped(
        "Execution Layer (EC): Represents each tested PyDelphi configuration compared "
        "against the Reference Core (RC) and the external reference (Tier 1 + Tier 2 checks).",
        "  - ",
        "    ",
    )

    print_wrapped(
        f"Reference Core (RC): {REFERENCE_CORE_CONFIG[0].upper()}/"
        f"{REFERENCE_CORE_CONFIG[1].upper()}/"
        f"{REFERENCE_CORE_CONFIG[2]} nominal thread(s)",
        "  - ",
        "    ",
    )
    print_wrapped(
        f"Internal Tolerance: RTOL={PDELPHI_CONSISTENCY_RTOL}, "
        f"ATOL floor={PDELPHI_CONSISTENCY_ATOL_FLOOR}",
        "  - ",
        "    ",
    )

    print_wrapped(
        "Console summary: T1[<energy>] and T2[<energy>] report the energy term "
        "with the largest observed deviation for that subtest. Tiny nonzero "
        "relative differences are shown as <0.0005% instead of 0.000% or "
        "overly precise decimal output.",
        "  - ",
        "    ",
    )

    print_wrapped(
        "Test skipping logic: configurations skipped by flags are recorded in the "
        "report as SKIPPED.",
        "",
        "  ",
    )

    if has_artifact_reference_cases:
        log("", verbose)
        print_wrapped(
            "[AR] Auxiliary Reference cases: For tests marked [AR], the runner reads "
            "the required runtime parameters from the corresponding parameter file(s), "
            "rather than generating the parameter setup from TSV energy fields. This "
            "path is used for cases with specific output-file or multi-step input "
            "requirements, such as FRC, focusing FRC, and zeta/ZPHI checks.",
            "  - ",
            "    ",
        )
        print_wrapped(
            "Computed output-file values are compared against checked-in reference "
            "values stored in the test set. The one-line subtest output reports the "
            "largest observed artifact difference as AR[<column/field>]. Tiny "
            "nonzero relative differences are printed as <0.0005% rather than "
            "rounded to 0.000%. For ZPHI zeta artifacts, the detailed CSV also "
            "records the worst grid index, delta phi, and relative error.",
            "    ",
            "    ",
        )
        print_wrapped(
            "Pass criterion: all compared output-file values must be within the "
            "case-specific preset threshold, allowed difference = ATOL + RTOL * "
            "|reference value|. If total electrostatic energy is present in the "
            "reference file, it must also satisfy the same threshold rule.",
            "    ",
            "    ",
        )

    log("", verbose)
    print_wrapped(
        f"RPBE override: single-thread target is replaced by {RPBE_MIN_THREADS} "
        "threads for performance.",
        "",
        "  ",
    )

    log("=" * 80, verbose)
    log("", verbose)


def _rel_ratio(diff: Optional[float], ref: Optional[float]) -> Optional[float]:
    """
    Returns the signed relative deviation (Δ_rel) = (calc - ref) / max(|ref|, ε).
    Positive means calc > ref, negative means calc < ref.
    """
    if diff is None or ref is None:
        return None
    denom = max(abs(ref), 1e-8)
    return diff / denom  # preserve sign


def format_relative_percent(rel: Optional[float]) -> str:
    """Return a compact relative-percent string for console summaries.

    The one-line console summary is diagnostic, not a high-precision numerical
    report. Tiny nonzero relative differences are therefore reported as
    <0.0005%, matching the rounding boundary for a three-decimal percent
    display.
    """
    tol_limit = 0.0005
    if rel is None:
        return "n/a"
    if rel == 0.0:
        return "0.000%"
    percent = abs(rel) * 100.0
    if percent < tol_limit:
        return f"<{tol_limit:7.4f}%"
    sign = "+" if rel >= 0 else "-"
    return f"{sign}{percent:7.3f}%"


def fmt_pair(
    tag: str,
    abbr: Optional[str],
    diff: Optional[float],
    rel: Optional[float],
    field_width: int = 47,
) -> str:
    """
    Format a (Delta, relative %) pair into a fixed-width table field.

    Tiny nonzero relative differences are shown as <0.0005% instead of being
    rounded to 0.000% or printed with excessive decimal places.
    """
    tag_field = f"{tag}[{abbr or 'n/a'}]".ljust(20)

    if diff is None:
        content = "n/a"
    else:
        sign = "+" if diff >= 0 else "-"
        diff_abs = abs(diff)
        diff_str = f"Δ={sign}{diff_abs:.3e}".strip()
        if rel is not None:
            rel_str = f"({format_relative_percent(rel):>9s})"
            content = f"{diff_str:<16}{rel_str:<11}"
        else:
            content = f"{diff_str:<16}{'(n/a)':<11}"

    return f"{tag_field}{content:<{field_width - 20}}"


def colorize_status(status: str) -> str:
    """Return the same TTY-colored status string used in subtest summaries."""
    use_color = sys.stdout.isatty()

    def colorize(txt, code):
        return f"\033[{code}m{txt}\033[0m" if use_color else txt

    if status == "PASS":
        return colorize(status, "92")
    if status in ("FAIL", "ERROR"):
        return colorize(status, "91")
    if status in ("TIMEOUT", "SKIPPED"):
        return colorize(status, "93")
    return status


def log_subtest_summary(summary: SubtestSummary, verbose: bool):
    """Print one-line, fixed-width subtest summary (TTY-safe, colorless)."""
    status = summary.status
    status_disp = colorize_status(status)

    rel1 = _rel_ratio(summary.worst_tier1_diff, summary.worst_tier1_ref)
    rel2 = _rel_ratio(summary.worst_tier2_diff, summary.worst_tier2_ref)

    tier1_info = fmt_pair(
        "T1", summary.worst_tier1_abbr, summary.worst_tier1_diff, rel1
    )
    tier2_info = fmt_pair(
        "T2", summary.worst_tier2_abbr, summary.worst_tier2_diff, rel2
    )

    # --- widened Status field (12 chars) ---
    msg = (
        f" {indent}[{summary.tier:<2}] "
        f"{summary.platform:<5}/{summary.precision:<7}/{summary.threads:<3} "
        f"→ {status_disp:<12} "
        f"{tier1_info}  {tier2_info} "
        f"in {summary.time_taken:7.2f}s"
    )

    if summary.error:
        msg += f" ⚠ {summary.error[:80]}"
    log(msg, verbose, always=True)


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
        "disabled": case_data.get("disabled") or "NA",
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


def run_artifact_reference_subtest(
    *,
    case_data: dict,
    platform: str,
    precision: str,
    threads_to_execute: int,
    project_root: str,
    timeout: int,
    verbose: bool,
) -> dict:
    """Run one artifact-reference subtest, including ordered parent/child param files."""
    test_report_row = {
        "example_name": case_data.get("example"),
        "salt": case_data.get("salt"),
        "platform": platform,
        "precision": precision,
        "boundary_condition": case_data.get("boundary_condition"),
        "threads": threads_to_execute,
        "test_type": "ARTIFACT_REF",
        "status": "FAIL",
        "pydp_consistency_passed": "N/A",
        "error_message": "",
        "disabled": case_data.get("disabled") or "NA",
    }

    for energy_type_abbr in REFERENCE_ENERGY_KEYS:
        test_report_row[f"{energy_type_abbr}_ref"] = case_data.get(energy_type_abbr)
        test_report_row[f"{energy_type_abbr}_test"] = "N/A"
        test_report_row[f"{energy_type_abbr}_diff"] = "N/A"
        test_report_row[f"{energy_type_abbr}_pass"] = "N/A"
        test_report_row[f"{energy_type_abbr}_pydp_ref"] = "N/A"
        test_report_row[f"{energy_type_abbr}_pydp_diff"] = "N/A"
        test_report_row[f"{energy_type_abbr}_pydp_pass"] = "N/A"

    test_report_row["E_stress_osmotic_sum_pass"] = "N/A"

    try:
        parm_files = resolve_parm_files(case_data, project_root)
        ref_file = resolve_case_file_path(
            project_root, case_data, case_data.get("output_ref_file")
        )

        if not parm_files:
            raise ValueError("artifact reference case has no parm_files")
        if len(parm_files) > 2:
            raise ValueError(
                "artifact reference case supports one parm file, or two ordered "
                f"parm files separated by {ARTIFACT_PARAM_DELIMITER!r}; got {len(parm_files)}"
            )
        if ref_file is None or not os.path.exists(ref_file):
            raise FileNotFoundError(f"output_ref_file not found: {ref_file}")

        with tempfile.TemporaryDirectory(prefix="pydelphi_artifact_test_") as temp_dir:
            out_file = generated_artifact_path_from_ref(ref_file, temp_dir)
            parent_phi_file = os.path.join(temp_dir, "parent.phi")
            example_dir = os.path.join(
                project_root,
                "pydelphi",
                "data",
                "test_cases",
                str(case_data.get("example")),
            )

            replacements = {
                "PROJECT_ROOT": project_root,
                "EXAMPLE_DIR": example_dir,
                "TMPDIR": temp_dir,
                "OUTPUT_FILE": out_file,
                "PARENT_PHI_FILE": parent_phi_file,
            }

            for run_index, source_param_file in enumerate(parm_files, start=1):
                if source_param_file is None or not os.path.exists(source_param_file):
                    raise FileNotFoundError(f"parm file not found: {source_param_file}")

                temp_param_file = render_param_template_to_temp(
                    source_param_file=source_param_file,
                    temp_dir=temp_dir,
                    replacements=replacements,
                )
                output_csv_path = os.path.join(
                    temp_dir, f"outputs_run{run_index}_{uuid.uuid4().hex}.csv"
                )

                run_param_file_no_energy_parse(
                    param_file=temp_param_file,
                    project_root=project_root,
                    platform=platform,
                    precision=precision,
                    threads=threads_to_execute,
                    timeout=timeout,
                    label=f"{case_data.get('example')}_run{run_index}",
                    output_csv_path=output_csv_path,
                )

            if not os.path.exists(out_file):
                raise FileNotFoundError(
                    f"expected artifact was not generated: {out_file}"
                )

            rtol, atol = get_case_artifact_tolerances(case_data)
            passed, artifact_result = compare_artifacts_by_kind(
                ref_file=ref_file,
                out_file=out_file,
                rtol=rtol,
                atol=atol,
            )
            test_report_row.update(artifact_result)
            test_report_row["status"] = "PASS" if passed else "FAIL"
            if not passed and not test_report_row["error_message"]:
                test_report_row["error_message"] = artifact_result.get(
                    "artifact_error", ""
                )

        return test_report_row

    except subprocess.CalledProcessError as e:
        test_report_row["status"] = "ERROR"
        test_report_row["error_message"] = (
            f"Subprocess Error (Exit Code {e.returncode}): "
            f"{e.stderr.strip() or e.stdout.strip()}"
        )
        log(
            f"ERROR: Error running auxiliary-reference case: {test_report_row['error_message']}",
            verbose,
            always=True,
        )
        return test_report_row

    except subprocess.TimeoutExpired as e:
        test_report_row["status"] = "TIMEOUT"
        test_report_row["error_message"] = (
            f"Timeout Error: Command ran for too long ({e.timeout}s)"
        )
        log(
            f"TIMEOUT: Auxiliary-reference case {case_data.get('example')} exceeded {e.timeout}s",
            verbose,
            always=True,
        )
        return test_report_row

    except FileNotFoundError as e:
        test_report_row["status"] = "FILE_ERROR"
        test_report_row["error_message"] = f"File Error: {str(e)}"
        log(f"FILE_ERROR: {e}", verbose, always=True)
        return test_report_row

    except Exception as e:
        test_report_row["status"] = "ERROR"
        test_report_row["error_message"] = f"Unexpected Error: {str(e)}"
        log(
            f"ERROR: An unexpected auxiliary-reference test error occurred: {e}",
            verbose,
            always=True,
        )
        return test_report_row


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
        "disabled": case_data.get("disabled") or "NA",
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
) -> Tuple[List[dict], str]:
    """
    For a single case: run the reference core (if present) then run all other combinations.

    Returns
    -------
    case_reports : list of dict
        Report rows for this case.
    case_status : str
        Aggregated status (PASS, FAIL, SKIPPED, etc.)
    """
    case_reports: List[dict] = []
    start_all_combinations_time = time.time()

    # Artifact-reference cases bypass the energy TSV path but still run all selected
    # platform/precision/thread combinations and report one row per subtest.
    if case_has_artifact_reference(case_data):
        subtests_status = []
        log(
            f"  Processing auxiliary-reference case {case_index + 1}/{num_cases}: "
            f"{case_data.get('example')}",
            verbose,
            always=True,
        )

        bio_model = (
            case_data.get("bio_model").upper() if case_data.get("bio_model") else ""
        )

        for platform, precision, intended_threads in combinations:
            threads_to_execute = intended_threads
            if bio_model == "RPBE" and intended_threads == 1:
                threads_to_execute = RPBE_MIN_THREADS

            start_time = time.time()
            report_row = run_artifact_reference_subtest(
                case_data=case_data,
                platform=platform,
                precision=precision,
                threads_to_execute=threads_to_execute,
                project_root=project_root,
                timeout=timeout,
                verbose=verbose,
            )
            elapsed_time = time.time() - start_time
            report_row["time_taken"] = elapsed_time
            case_reports.append(report_row)
            subtests_status.append(report_row["status"])

            status_disp = colorize_status(report_row["status"])
            artifact_diff = report_row.get("artifact_worst_diff")
            artifact_ref = report_row.get("artifact_worst_ref")
            artifact_info = fmt_pair(
                "AR",
                report_row.get("artifact_worst_column") or "n/a",
                artifact_diff if isinstance(artifact_diff, (float, int)) else None,
                _rel_ratio(
                    artifact_diff if isinstance(artifact_diff, (float, int)) else None,
                    artifact_ref if isinstance(artifact_ref, (float, int)) else None,
                ),
            )

            energy_diff = report_row.get("artifact_energy_diff")
            energy_ref = report_row.get("artifact_energy_ref")
            energy_info = fmt_pair(
                "E",
                "total" if isinstance(energy_diff, (float, int)) else "n/a",
                energy_diff if isinstance(energy_diff, (float, int)) else None,
                _rel_ratio(
                    energy_diff if isinstance(energy_diff, (float, int)) else None,
                    energy_ref if isinstance(energy_ref, (float, int)) else None,
                ),
            )

            msg = (
                f" {indent}[AR] {platform:<5}/{precision:<7}/{threads_to_execute:<3} "
                f"→ {status_disp:<12} "
                f"{artifact_info}  {energy_info} "
                f"in {elapsed_time:7.2f}s"
            )
            if report_row.get("error_message"):
                msg += f" ⚠ {report_row['error_message'][:80]}"
            log(msg, verbose, always=True)

        elapsed_all_combinations_time = time.time() - start_all_combinations_time
        log(
            f" {indent}Time taken for auxiliary-reference case {case_data.get('example')} combinations: "
            f"{elapsed_all_combinations_time:.2f} seconds. \n",
            verbose,
            always=True,
        )
        return case_reports, get_case_status(list(set(subtests_status)))

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
        return case_reports, "SKIPPED"

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
            f" {indent}indi: {case_data.get('indi')}, exdi: {case_data.get('exdi')}, gapdi: {case_data.get('gapdi')}, "
            f" gaussian_exponent: {case_data.get('gaussian_exponent')}, sigma={case_data.get('gaussian_sigma')}, density_cutoff: {case_data.get('density_cutoff')}) \n"
        )

    log(
        f"  Processing case {case_index + 1}/{num_cases}: {case_data.get('example')} with key parameters: \n"
        f" {indent}(biomodel: {case_data.get('bio_model')}, dielectric_model: {case_data.get('dielectric_model')}, surface_method: {case_data.get('surface_method')}, \n"
        f"{gaussian_params}"
        f" {indent}solver: {case_data.get('solver')}, is_nonlinear={is_nonlinear}, salt: {case_data.get('salt')}, boundary_condition={case_data.get('boundary_condition')})",
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

        # --- concise RC summary with relative context ---
        # Find the energy term with the largest absolute difference
        worst_t1_abbr, worst_t1_signed_diff = max(
            (
                (
                    abbr,
                    ref_report_row.get(f"{abbr}_test")
                    - ref_report_row.get(f"{abbr}_ref"),
                )
                for abbr in REFERENCE_ENERGY_KEYS
                if isinstance(ref_report_row.get(f"{abbr}_test"), (float, int))
                and isinstance(ref_report_row.get(f"{abbr}_ref"), (float, int))
            ),
            key=lambda x: abs(x[1]),
            default=(None, None),
        )

        worst_t1_ref = (
            ref_report_row.get(f"{worst_t1_abbr}_ref") if worst_t1_abbr else None
        )

        summary = SubtestSummary(
            tier="RC",
            platform=ref_platform,
            precision=ref_precision,
            threads=threads_to_execute,
            status=ref_report_row["status"],
            tier1_pass=ref_report_row["status"] == "PASS",
            tier2_pass=None,
            worst_tier1_abbr=worst_t1_abbr,
            worst_tier1_diff=(
                worst_t1_signed_diff if worst_t1_signed_diff not in (0, None) else None
            ),
            worst_tier1_ref=worst_t1_ref,
            worst_tier2_abbr="n/a",
            worst_tier2_diff=None,
            worst_tier2_ref=None,
            time_taken=elapsed_time,
            error=ref_report_row.get("error_message"),
        )
        log_subtest_summary(summary, verbose)

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

            # --- concise EC summary ---
            worst_t1_abbr, worst_t1_diff = max(
                (
                    (abbr, report_row.get(f"{abbr}_diff") or 0.0)
                    for abbr in REFERENCE_ENERGY_KEYS
                    if isinstance(report_row.get(f"{abbr}_diff"), (float, int))
                ),
                key=lambda x: x[1],
                default=(None, None),
            )
            worst_t1_ref = (
                report_row.get(f"{worst_t1_abbr}_ref") if worst_t1_abbr else None
            )

            worst_t2_abbr, worst_t2_diff = max(
                (
                    (abbr, report_row.get(f"{abbr}_pydp_diff") or 0.0)
                    for abbr in REFERENCE_ENERGY_KEYS
                    if isinstance(report_row.get(f"{abbr}_pydp_diff"), (float, int))
                ),
                key=lambda x: x[1],
                default=(None, None),
            )
            worst_t2_ref = (
                report_row.get(f"{worst_t2_abbr}_pydp_ref") if worst_t2_abbr else None
            )

            summary = SubtestSummary(
                tier="EC",
                platform=platform,
                precision=precision,
                threads=threads_to_execute,
                status=report_row["status"],
                tier1_pass=all(
                    report_row.get(f"{abbr}_pass") for abbr in REFERENCE_ENERGY_KEYS
                ),
                tier2_pass=report_row.get("pydp_consistency_passed"),
                worst_tier1_abbr=worst_t1_abbr,
                worst_tier1_diff=worst_t1_diff,
                worst_tier1_ref=worst_t1_ref,
                worst_tier2_abbr=worst_t2_abbr,
                worst_tier2_diff=worst_t2_diff,
                worst_tier2_ref=worst_t2_ref,
                time_taken=elapsed_time,
                error=report_row.get("error_message"),
            )
            log_subtest_summary(summary, verbose)

    elapsed_all_combinations_time = time.time() - start_all_combinations_time
    log(
        f" {indent}Time taken for case {case_data.get('example')} combinations: {elapsed_all_combinations_time:.2f} seconds. \n",
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

    # Project root detection (assumes tests live inside pydelphi/tests/)
    script_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    full_reference_file_path = os.path.join(
        project_root, "pydelphi", "data", "test_cases", REFERENCE_FILE
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
    reference_data_all = parse_reference_data(full_reference_file_path)
    disabled_reference_data = [
        case_data for case_data in reference_data_all if case_is_disabled(case_data)
    ]
    reference_data = [
        case_data for case_data in reference_data_all if not case_is_disabled(case_data)
    ]

    has_artifact_reference_cases = any(
        case_has_artifact_reference(case_data) for case_data in reference_data
    )

    # Always show test setup and methodology for clarity and reproducibility.
    # The [AR] explanation is shown only when enabled Artifact Reference cases exist.
    print_splash_message(
        True, has_artifact_reference_cases=has_artifact_reference_cases
    )

    num_cases_total = len(reference_data_all)
    num_disabled_cases = len(disabled_reference_data)
    num_cases = len(reference_data)

    log(
        f"Loaded {num_cases_total} unique test cases: "
        f"{num_cases} enabled, {num_disabled_cases} disabled.",
        args.verbose,
        always=True,
    )

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

        artifact_report_keys = [
            "artifact_ref_file",
            "artifact_out_file",
            "artifact_kind",
            "artifact_pass",
            "artifact_rows_ref",
            "artifact_rows_out",
            "artifact_worst_column",
            "artifact_worst_row",
            "artifact_worst_ref",
            "artifact_worst_out",
            "artifact_worst_diff",
            "artifact_energy_ref",
            "artifact_energy_out",
            "artifact_energy_diff",
            "artifact_rtol",
            "artifact_atol",
            "artifact_error",
            "artifact_energy_error",
            "zphi_pass",
            "zphi_ref_points",
            "zphi_out_points",
            "zphi_worst_field",
            "zphi_worst_ix",
            "zphi_worst_iy",
            "zphi_worst_iz",
            "zphi_worst_ref_phi",
            "zphi_worst_out_phi",
            "zphi_worst_delta_phi",
            "zphi_worst_abs_diff",
            "zphi_worst_relative_error",
            "zphi_worst_allowed_diff",
            "zphi_num_potential_failures",
            "zphi_num_metadata_float_failures",
            "zphi_error",
        ]

        final_keys = (
            base_keys
            + energy_report_keys
            + artifact_report_keys
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

    if num_disabled_cases:
        log(
            f"Disabled test cases ignored by TSV disabled column: {num_disabled_cases}",
            args.verbose,
            always=True,
        )

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
