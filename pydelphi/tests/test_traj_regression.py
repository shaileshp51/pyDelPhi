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

"""Trajectory regression tests for pyDelPhi.

This runner is intentionally separate from the energy regression suite. It uses
case definitions from pydelphi/data/test_traj/test_traj.tsv, generates temporary
parameter files, runs pydelphi_static for each selected platform/precision/thread
configuration, and compares computed frame-wise TSV output against finalized
reference values in traj_ref_values.tsv.

The comparison is literal TSV regression: each enabled case is compared
frame-by-frame and field-by-field. Numeric values must satisfy:

    abs(calc - ref) <= atol + rtol * abs(ref)

Frame slicing follows Python-style semantics: first is inclusive, last is
exclusive, and stride follows range(first, last, stride).
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import textwrap
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from pydelphi.utils.utils import seconds_to_hms
except Exception:  # pragma: no cover - fallback for standalone syntax checks

    def seconds_to_hms(seconds: float) -> str:
        seconds = int(round(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


TEST_TRAJ_DIRNAME = "test_traj"
TEST_TRAJ_FILE = "test_traj.tsv"
TEST_REPORT_FILE = "pydelphi_traj_regression_test_report.tsv"
DEBUG_OUTPUT_DIR = "pydelphi_traj_debug_files"

REFERENCE_CORE_CONFIG = ("cpu", "double", 1)
ALL_COMBINATIONS_BASE = [
    ("cpu", "single", 1),
    ("cpu", "double", 1),
    ("cpu", "single", 4),
    ("cpu", "double", 4),
    ("cuda", "single", 1),
    ("cuda", "double", 1),
    ("cuda", "single", 4),
    ("cuda", "double", 4),
]

DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-6
NA_VALUES = {"", "NA", "N/A", "None", "none", None}
KEY_COLUMNS = {"case_id", "frame"}
NON_NUMERIC_METADATA_COLUMNS = {
    "case_id",
    "label",
    "platform",
    "precision",
    "threads",
    "topology_file",
    "trajectory_file",
    "frame_selection_label",
}


@dataclass
class TrajSubtestSummary:
    case_id: str
    platform: str
    precision: str
    threads: int
    status: str
    worst_field: Optional[str] = None
    worst_frame: Optional[int] = None
    worst_ref: Optional[float] = None
    worst_calc: Optional[float] = None
    worst_diff: Optional[float] = None
    worst_allowed: Optional[float] = None
    time_taken: float = 0.0
    error: str = ""


def log(msg: str, verbose: bool, always: bool = False) -> None:
    if always or verbose:
        print(msg)


def is_na(value: Any) -> bool:
    return value in NA_VALUES or str(value).strip() in NA_VALUES


def normalize_optional(value: Any) -> Optional[str]:
    if is_na(value):
        return None
    return str(value).strip()


def parse_bool(value: Any) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "yes", "1", "skip", "disabled"}


def parse_int(row: dict, key: str) -> int:
    try:
        return int(str(row[key]).strip())
    except Exception as exc:
        raise ValueError(
            f"Invalid integer value for {key!r}: {row.get(key)!r}"
        ) from exc


def parse_float_or_default(row: dict, key: str, default: float) -> float:
    value = row.get(key)
    if is_na(value):
        return default
    try:
        return float(str(value).strip())
    except Exception as exc:
        raise ValueError(f"Invalid float value for {key!r}: {value!r}") from exc


def get_project_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", ".."))


def get_traj_data_dir(project_root: str) -> str:
    return os.path.join(project_root, "pydelphi", "data", TEST_TRAJ_DIRNAME)


def resolve_data_file(project_root: str, filename: Optional[str]) -> Optional[str]:
    if filename is None:
        return None
    if os.path.isabs(filename):
        return filename
    return os.path.join(get_traj_data_dir(project_root), filename)


def get_test_combinations(
    skip_cuda: bool = False,
    skip_parallel: bool = False,
    skip_single: bool = False,
    skip_double: bool = False,
) -> Tuple[List[Tuple[str, str, int]], List[Dict[str, Any]]]:
    planned: List[Tuple[str, str, int]] = []
    configuration_skips: List[Dict[str, Any]] = []

    for platform, precision, threads in ALL_COMBINATIONS_BASE:
        reason = None
        if skip_cuda and platform == "cuda":
            reason = "Skipped by --no-cuda flag"
        elif skip_parallel and threads > 1:
            reason = "Skipped by --no-parallel flag"
        elif skip_single and precision == "single":
            reason = "Skipped by --no-single flag"
        elif skip_double and precision == "double":
            reason = "Skipped by --no-double flag"

        if reason:
            configuration_skips.append(
                {
                    "platform": platform,
                    "precision": precision,
                    "threads": threads,
                    "reason": reason,
                }
            )
        else:
            planned.append((platform, precision, threads))

    return planned, configuration_skips


def read_test_cases(test_list_path: str) -> Tuple[List[dict], List[dict]]:
    enabled: List[dict] = []
    disabled: List[dict] = []

    with open(test_list_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw_row in reader:
            row = {
                key: (value.strip() if isinstance(value, str) else value)
                for key, value in raw_row.items()
            }
            validate_case_row(row)
            if parse_bool(row.get("disabled")):
                disabled.append(row)
            else:
                enabled.append(row)

    return enabled, disabled


def validate_case_row(row: dict) -> None:
    required = [
        "case_id",
        "topology_file",
        "trajectory_file",
        "first",
        "last",
        "stride",
        "frame_selection_label",
        "crgsiz_set",
        "crgsiz_mode",
        "qfile",
        "rfile",
        "dielectric_model",
        "gridbox_type",
        "gridbox_margin",
        "scale",
        "indi",
        "exdi",
        "bndcon",
        "maxc",
        "energy_terms",
        "ref_values_file",
        "rtol",
        "atol",
        "disabled",
    ]
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"test_traj.tsv row is missing required columns: {missing}")

    first = parse_int(row, "first")
    last = parse_int(row, "last")
    stride = parse_int(row, "stride")
    if first < 0:
        raise ValueError(f"{row.get('case_id')}: first must be >= 0")
    if last <= first:
        raise ValueError(f"{row.get('case_id')}: last must be greater than first")
    if stride <= 0:
        raise ValueError(f"{row.get('case_id')}: stride must be > 0")

    crgsiz_set = normalize_optional(row.get("crgsiz_set"))
    crgsiz_mode = normalize_optional(row.get("crgsiz_mode"))

    # Keep validation intentionally light here. The TSV schema always includes
    # crgsiz_set, crgsiz_mode, qfile, and rfile, but prm generation only writes
    # arguments whose cells are not NA. If crgsiz_set is NA, the complete
    # in(crgsiz, ...) command is omitted regardless of the other crgsiz columns.
    # Scientific/model-conformance rules can be enforced by the input layer and by
    # keeping the checked-in TSV rows conformant.
    if crgsiz_set is not None and crgsiz_mode is not None:
        if crgsiz_mode not in {"acquire", "override"}:
            raise ValueError(
                f"{row.get('case_id')}: crgsiz_mode must be acquire, override, or NA"
            )


def selected_frames(row: dict) -> List[int]:
    return list(
        range(parse_int(row, "first"), parse_int(row, "last"), parse_int(row, "stride"))
    )


def quote_param_path(path: str) -> str:
    return path.replace('"', '\\"')


def render_param_file(case_data: dict, project_root: str, temp_dir: str) -> str:
    topology_file = resolve_data_file(
        project_root, normalize_optional(case_data.get("topology_file"))
    )
    trajectory_file = resolve_data_file(
        project_root, normalize_optional(case_data.get("trajectory_file"))
    )
    if topology_file is None or not os.path.exists(topology_file):
        raise FileNotFoundError(f"topology_file not found: {topology_file}")
    if trajectory_file is None or not os.path.exists(trajectory_file):
        raise FileNotFoundError(f"trajectory_file not found: {trajectory_file}")

    lines: List[str] = []
    gridbox_type = str(case_data.get("gridbox_type")).strip()
    if gridbox_type == "gridbox_margin":
        lines.append(f"gridbox_margin={case_data['gridbox_margin']}")
    else:
        lines.append(f"gridbox_type={gridbox_type}")
        if not is_na(case_data.get("gridbox_margin")):
            lines.append(f"gridbox_margin={case_data['gridbox_margin']}")
    lines.append(f"scale={case_data['scale']}")
    lines.append("")

    crgsiz_set = normalize_optional(case_data.get("crgsiz_set"))
    crgsiz_mode = normalize_optional(case_data.get("crgsiz_mode"))
    qfile = normalize_optional(case_data.get("qfile"))
    rfile = normalize_optional(case_data.get("rfile"))

    if crgsiz_set is not None:
        # Use set= for the charge/size set selector. The older name= spelling
        # is intentionally not emitted because current pyDelPhi input parsing
        # treats name as an undefined in(crgsiz, ...) attribute.
        crgsiz_args = [f'set="{crgsiz_set}"']

        if qfile is not None:
            qfile_path = resolve_data_file(project_root, qfile)
            if qfile_path is None or not os.path.exists(qfile_path):
                raise FileNotFoundError(f"qfile not found: {qfile_path}")
            crgsiz_args.append(f'qfile="{quote_param_path(qfile_path)}"')

        if rfile is not None:
            rfile_path = resolve_data_file(project_root, rfile)
            if rfile_path is None or not os.path.exists(rfile_path):
                raise FileNotFoundError(f"rfile not found: {rfile_path}")
            crgsiz_args.append(f'rfile="{quote_param_path(rfile_path)}"')

        if crgsiz_mode is not None:
            crgsiz_args.append(f'mode="{crgsiz_mode}"')

        lines.append("in(crgsiz, " + ", ".join(crgsiz_args) + ")")

    lines.append(f'in(topology,file="{quote_param_path(topology_file)}")')
    lines.append(
        'in(trajectory,file="%s", first=%d, last=%d, stride=%d)'
        % (
            quote_param_path(trajectory_file),
            parse_int(case_data, "first"),
            parse_int(case_data, "last"),
            parse_int(case_data, "stride"),
        )
    )
    lines.append("")
    if not is_na(case_data.get("dielectric_model")):
        lines.append(f"dielectric_model={case_data['dielectric_model']}")
    lines.append(f"bndcon={case_data['bndcon']}")
    lines.append(f"maxc={case_data['maxc']}")
    lines.append("")
    lines.append(f"indi={case_data['indi']}")
    lines.append(f"exdi={case_data['exdi']}")
    lines.append(f"energy({case_data['energy_terms']})")

    param_path = os.path.join(
        temp_dir, f"{case_data['case_id']}_{uuid.uuid4().hex}.prm"
    )
    with open(param_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return param_path


def run_pydelphi_traj(
    *,
    param_file: str,
    output_tsv: str,
    case_id: str,
    platform: str,
    precision: str,
    threads: int,
    project_root: str,
    timeout: int,
) -> None:
    command = [
        sys.executable,
        "-m",
        "pydelphi.scripts.pydelphi_traj",
        "--platform",
        platform,
        "--precision",
        precision,
        "--threads",
        str(threads),
        "--param-file",
        param_file,
        "--label",
        case_id,
        "--verbosity",
        "error",
        "--outfile",
        output_tsv,
        "--overwrite",
    ]
    subprocess.run(
        command,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )


def iter_data_lines(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8", newline="") as handle:
        for line in handle:
            if line.strip() and not line.lstrip().startswith("#"):
                yield line


def read_tsv_rows(path: str) -> List[dict]:
    reader = csv.DictReader(iter_data_lines(path), delimiter="\t")
    if reader.fieldnames is None:
        raise ValueError(f"TSV has no header: {path}")
    return [dict(row) for row in reader]


def try_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text in NA_VALUES:
        return None
    try:
        return float(text)
    except Exception:
        return None


def get_row_value_ci(row: dict, *names: str) -> Any:
    """Return a row value using case-insensitive column-name matching."""
    lowered = {str(key).lower(): key for key in row.keys()}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return row.get(key)
    return None


def has_column_ci(rows: List[dict], column_name: str) -> bool:
    """Return True if any row has a column matching column_name case-insensitively."""
    target = column_name.lower()
    return any(target in {str(key).lower() for key in row.keys()} for row in rows)


def canonical_frame_value(row: dict) -> int:
    value = get_row_value_ci(row, "frame", "frame_index", "iframe")
    if value is not None and not is_na(value):
        return int(float(str(value).strip()))
    raise ValueError(f"computed/reference row has no frame column: {row}")


def filter_rows_for_case(rows: List[dict], case_id: str) -> List[dict]:
    """Filter by case_id only when the file actually carries a case_id column.

    pyDelPhi trajectory outputs may use uppercase headers such as FRAME and LABEL
    and may not contain case_id. In that case, the file is assumed to contain
    only the current case's rows.
    """
    if has_column_ci(rows, "case_id"):
        return [
            row
            for row in rows
            if str(get_row_value_ci(row, "case_id")).strip() == str(case_id)
        ]
    return rows


def canonical_numeric_field_map(rows: List[dict]) -> Dict[str, str]:
    """Map lower-case numeric field names to their original column names."""
    result: Dict[str, str] = {}
    for row in rows:
        for key, value in row.items():
            key_lower = str(key).lower()
            if key_lower in KEY_COLUMNS:
                continue
            if key_lower in NON_NUMERIC_METADATA_COLUMNS:
                continue
            if try_float(value) is not None:
                result.setdefault(key_lower, key)
    return result


def numeric_compare_fields(
    ref_rows: List[dict], calc_rows: List[dict]
) -> List[Tuple[str, str, str]]:
    """Return shared numeric fields as (display_name, ref_key, calc_key).

    Column matching is case-insensitive so FRAME/LABEL-style output headers do
    not break comparisons against lower-case or mixed-case reference files.
    """
    if not ref_rows or not calc_rows:
        return []

    ref_map = canonical_numeric_field_map(ref_rows)
    calc_map = canonical_numeric_field_map(calc_rows)
    shared = sorted(set(ref_map) & set(calc_map))
    return [(ref_map[key], ref_map[key], calc_map[key]) for key in shared]


def compare_traj_tsv(
    *,
    case_id: str,
    expected_frames: List[int],
    ref_values_file: str,
    computed_file: str,
    rtol: float,
    atol: float,
) -> Tuple[bool, Dict[str, Any]]:
    ref_rows_all = read_tsv_rows(ref_values_file)
    calc_rows_all = read_tsv_rows(computed_file)
    ref_rows = filter_rows_for_case(ref_rows_all, case_id)
    calc_rows = filter_rows_for_case(calc_rows_all, case_id)

    result: Dict[str, Any] = {
        "ref_values_file": ref_values_file,
        "computed_values_file": computed_file,
        "frame_count_ref": len(ref_rows),
        "frame_count_calc": len(calc_rows),
        "expected_frames": ",".join(str(frame) for frame in expected_frames),
        "worst_field": None,
        "worst_frame": None,
        "worst_ref": None,
        "worst_calc": None,
        "worst_diff": None,
        "worst_allowed": None,
        "rtol": rtol,
        "atol": atol,
        "comparison_error": "",
    }

    if not ref_rows:
        result["comparison_error"] = f"No reference rows found for case_id={case_id!r}"
        return False, result
    if not calc_rows:
        result["comparison_error"] = f"No computed rows found for case_id={case_id!r}"
        return False, result

    ref_by_frame = {canonical_frame_value(row): row for row in ref_rows}
    calc_by_frame = {canonical_frame_value(row): row for row in calc_rows}
    ref_frames = sorted(ref_by_frame)
    calc_frames = sorted(calc_by_frame)

    # The reference file may intentionally contain all frames for a case. Each
    # TSV row selects the subset to compare through first/last/stride, so extra
    # reference frames are allowed. Missing selected reference frames still fail.
    missing_ref_frames = [
        frame for frame in expected_frames if frame not in ref_by_frame
    ]
    if missing_ref_frames:
        result["comparison_error"] = (
            f"Reference values missing selected frame(s): missing={missing_ref_frames}, "
            f"available_ref={ref_frames}, expected={expected_frames}"
        )
        return False, result

    # Computed output should contain exactly the selected frames for this run.
    if calc_frames != expected_frames:
        result["comparison_error"] = (
            f"Computed frames do not match selected frames: calc={calc_frames}, expected={expected_frames}"
        )
        return False, result

    ref_rows_selected = [ref_by_frame[frame] for frame in expected_frames]
    calc_rows_selected = [calc_by_frame[frame] for frame in expected_frames]

    result["frame_count_ref"] = len(ref_rows_selected)
    result["frame_count_calc"] = len(calc_rows_selected)

    fields = numeric_compare_fields(ref_rows_selected, calc_rows_selected)
    if not fields:
        result["comparison_error"] = "No shared numeric comparison fields found"
        return False, result

    passed = True
    worst_diff = -1.0

    for frame in expected_frames:
        ref_row = ref_by_frame[frame]
        calc_row = calc_by_frame[frame]
        for display_field, ref_key, calc_key in fields:
            ref_value = try_float(ref_row.get(ref_key))
            calc_value = try_float(calc_row.get(calc_key))
            if ref_value is None:
                continue
            if calc_value is None:
                passed = False
                result["comparison_error"] = (
                    f"Missing computed numeric value for frame={frame}, field={display_field}"
                )
                return False, result

            diff = abs(calc_value - ref_value)
            allowed = atol + rtol * abs(ref_value)
            if diff > worst_diff:
                worst_diff = diff
                result.update(
                    {
                        "worst_field": display_field,
                        "worst_frame": frame,
                        "worst_ref": ref_value,
                        "worst_calc": calc_value,
                        "worst_diff": diff,
                        "worst_allowed": allowed,
                    }
                )
            if diff > allowed:
                passed = False

    return passed, result


def format_float(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:14.6g}"
    return value


def colorize_status(status: str) -> str:
    use_color = sys.stdout.isatty()
    if not use_color:
        return status
    if status == "PASS":
        return f"\033[92m{status}\033[0m"
    if status in {"FAIL", "ERROR", "FILE_ERROR"}:
        return f"\033[91m{status}\033[0m"
    if status in {"TIMEOUT", "SKIPPED"}:
        return f"\033[93m{status}\033[0m"
    return status


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


def log_subtest_summary(summary: TrajSubtestSummary, verbose: bool) -> None:
    status_disp = colorize_status(summary.status)
    if summary.worst_diff is None:
        compare_info = "TRJ[n/a]             n/a"
    else:
        rel = summary.worst_diff / max(abs(summary.worst_ref or 0.0), 1e-8)
        rel_text = format_relative_percent(rel)
        compare_info = (
            f"TRJ[{summary.worst_field or 'n/a'}]".ljust(20)
            + f"frame={summary.worst_frame:<4} "
            + f"Δ={summary.worst_diff: .3e} "
            + f"({rel_text:>9})"
        )
    msg = (
        f"  [TRJ] {summary.case_id:<24} "
        f"{summary.platform:<5}/{summary.precision:<7}/{summary.threads:<3} "
        f"→ {status_disp:<12} {compare_info} in {summary.time_taken:7.2f}s"
    )
    if summary.error:
        msg += f" ⚠ {summary.error[:100]}"
    log(msg, verbose, always=True)


def print_splash_message(verbose: bool, has_custom_cases: bool) -> None:
    try:
        import pydelphi as pydp

        version = pydp.__version__
    except Exception:
        version = "unknown"

    def log_wrapped(text: str, initial: str = "", subsequent: str = "") -> None:
        wrapper = textwrap.TextWrapper(
            width=80,
            initial_indent=initial,
            subsequent_indent=subsequent,
            break_long_words=False,
            break_on_hyphens=False,
        )
        log(wrapper.fill(text), verbose, always=True)

    log("=" * 80, verbose, always=True)
    log(f"PyDelphi-{version} Trajectory Regression Test Suite", verbose, always=True)
    log("=" * 80, verbose, always=True)

    log_wrapped(
        "This suite validates trajectory input by comparing computed output "
        "values against finalized reference values on a frame-by-frame and "
        "field-by-field basis."
    )
    log_wrapped(
        "Test list: pydelphi/data/test_traj/test_traj.tsv defines topology, "
        "trajectory, frame slicing, charge/size source, and numerical setup.",
        "  - ",
        "    ",
    )
    log_wrapped(
        "Reference values: traj_ref_values.tsv stores the finalized per-frame "
        "values used for TSV comparison.",
        "  - ",
        "    ",
    )
    log_wrapped(
        "Frame slicing: first is inclusive, last is exclusive, and stride follows "
        "Python range(first, last, stride) semantics.",
        "  - ",
        "    ",
    )
    log_wrapped(
        "Pass criterion: every compared numeric field must satisfy "
        "abs(calc-ref) <= ATOL + RTOL * abs(ref).",
        "  - ",
        "    ",
    )

    log_wrapped(
        "Summary output: TRJ[<field>] reports the field and frame with the "
        "largest observed deviation among all compared values for that subtest. "
        "For example, TRJ[E_grid_tot] frame=6 Δ=2.150e-02 (<0.0005%) means "
        "E_grid_tot at frame 6 had the largest deviation from the reference, "
        "with an absolute difference of 2.150e-02 and a relative difference "
        "below 0.0005% versus the reference value.",
        "  - ",
        "    ",
    )

    log("", verbose, always=True)
    log_wrapped(
        "Charge/size columns: crgsiz_set, crgsiz_mode, qfile, and rfile are "
        "always present in the TSV. crgsiz_set is written as set= in the generated "
        "prm. NA cells are omitted. When crgsiz_set is NA, no in(crgsiz, ...) "
        "command is written.",
        "  - ",
        "    ",
    )
    if has_custom_cases:
        log_wrapped(
            "In acquire mode, topology-provided values are used when available "
            "and missing values are acquired from the selected paired source. "
            "In override mode, topology-provided values are replaced by the "
            "selected paired source.",
            "    ",
            "    ",
        )

    log("=" * 80, verbose, always=True)
    log("", verbose, always=True)


def make_skipped_row(case_data: dict, config_skip: dict) -> dict:
    return {
        "case_id": case_data.get("case_id"),
        "topology_file": case_data.get("topology_file"),
        "trajectory_file": case_data.get("trajectory_file"),
        "frame_selection_label": case_data.get("frame_selection_label"),
        "selected_frames": ",".join(str(frame) for frame in selected_frames(case_data)),
        "platform": config_skip["platform"],
        "precision": config_skip["precision"],
        "threads": config_skip["threads"],
        "status": "SKIPPED",
        "time_taken": 0.0,
        "error_message": config_skip["reason"],
    }


def run_case_configuration(
    *,
    case_data: dict,
    platform: str,
    precision: str,
    threads: int,
    project_root: str,
    timeout: int,
    verbose: bool,
    debug_files: bool = False,
) -> dict:
    start = time.time()
    row: Dict[str, Any] = {
        "case_id": case_data.get("case_id"),
        "topology_file": case_data.get("topology_file"),
        "trajectory_file": case_data.get("trajectory_file"),
        "first": case_data.get("first"),
        "last": case_data.get("last"),
        "stride": case_data.get("stride"),
        "frame_selection_label": case_data.get("frame_selection_label"),
        "selected_frames": ",".join(str(frame) for frame in selected_frames(case_data)),
        "crgsiz_set": case_data.get("crgsiz_set"),
        "crgsiz_mode": case_data.get("crgsiz_mode"),
        "qfile": case_data.get("qfile"),
        "rfile": case_data.get("rfile"),
        "platform": platform,
        "precision": precision,
        "threads": threads,
        "status": "FAIL",
        "time_taken": 0.0,
        "ref_values_file": case_data.get("ref_values_file"),
        "computed_values_file": "TEMPORARY",
        "param_file": "TEMPORARY",
        "temp_dir": "TEMPORARY",
        "error_message": "",
    }

    temp_context = None
    temp_dir: Optional[str] = None

    try:
        if debug_files:
            safe_case = str(case_data["case_id"]).replace(os.sep, "_")
            safe_config = f"{platform}_{precision}_{threads}"
            run_token = uuid.uuid4().hex[:8]
            temp_dir = os.path.abspath(
                os.path.join(
                    DEBUG_OUTPUT_DIR,
                    f"{safe_case}_{safe_config}_{run_token}",
                )
            )
            os.makedirs(temp_dir, exist_ok=False)
        else:
            temp_context = tempfile.TemporaryDirectory(prefix="pydelphi_traj_test_")
            temp_dir = temp_context.name

        param_file = render_param_file(case_data, project_root, temp_dir)
        computed_file = os.path.join(
            temp_dir, f"{case_data['case_id']}_{uuid.uuid4().hex}.tsv"
        )

        run_pydelphi_traj(
            param_file=param_file,
            output_tsv=computed_file,
            case_id=case_data["case_id"],
            platform=platform,
            precision=precision,
            threads=threads,
            project_root=project_root,
            timeout=timeout,
        )

        ref_values_file = resolve_data_file(
            project_root, normalize_optional(case_data.get("ref_values_file"))
        )
        if ref_values_file is None or not os.path.exists(ref_values_file):
            raise FileNotFoundError(f"ref_values_file not found: {ref_values_file}")

        rtol = parse_float_or_default(case_data, "rtol", DEFAULT_RTOL)
        atol = parse_float_or_default(case_data, "atol", DEFAULT_ATOL)
        passed, comparison = compare_traj_tsv(
            case_id=case_data["case_id"],
            expected_frames=selected_frames(case_data),
            ref_values_file=ref_values_file,
            computed_file=computed_file,
            rtol=rtol,
            atol=atol,
        )
        row.update(comparison)

        # Keep normal reports portable. Test-data files are reported using the
        # TSV-provided relative name. Temporary generated files are reported as
        # TEMPORARY unless --debug-files is enabled.
        row["ref_values_file"] = case_data.get("ref_values_file")
        if debug_files:
            row["computed_values_file"] = computed_file
            row["param_file"] = param_file
            row["temp_dir"] = temp_dir
        else:
            row["computed_values_file"] = os.path.basename(computed_file)
            row["param_file"] = os.path.basename(param_file)
            row["temp_dir"] = os.path.basename(temp_dir or "")

        row["status"] = "PASS" if passed else "FAIL"
        if not passed:
            row["error_message"] = comparison.get("comparison_error", "")

    except subprocess.CalledProcessError as exc:
        row["status"] = "ERROR"
        row["error_message"] = (
            f"Subprocess Error (Exit Code {exc.returncode}): "
            f"{exc.stderr.strip() or exc.stdout.strip()}"
        )
    except subprocess.TimeoutExpired as exc:
        row["status"] = "TIMEOUT"
        row["error_message"] = f"Timeout Error: command exceeded {exc.timeout}s"
    except FileNotFoundError as exc:
        row["status"] = "FILE_ERROR"
        row["error_message"] = f"File Error: {exc}"
    except Exception as exc:
        row["status"] = "ERROR"
        row["error_message"] = f"Unexpected Error: {exc}"
    finally:
        if temp_context is not None:
            temp_context.cleanup()
        row["time_taken"] = time.time() - start

    log_subtest_summary(
        TrajSubtestSummary(
            case_id=row.get("case_id", ""),
            platform=platform,
            precision=precision,
            threads=threads,
            status=row.get("status", "UNKNOWN"),
            worst_field=row.get("worst_field"),
            worst_frame=row.get("worst_frame"),
            worst_ref=row.get("worst_ref"),
            worst_calc=row.get("worst_calc"),
            worst_diff=row.get("worst_diff"),
            worst_allowed=row.get("worst_allowed"),
            time_taken=row.get("time_taken", 0.0),
            error=row.get("error_message", ""),
        ),
        verbose,
    )
    return row


def write_report(rows: List[dict], report_path: str) -> None:
    fieldnames = [
        "case_id",
        "topology_file",
        "trajectory_file",
        "first",
        "last",
        "stride",
        "frame_selection_label",
        "selected_frames",
        "crgsiz_set",
        "crgsiz_mode",
        "qfile",
        "rfile",
        "platform",
        "precision",
        "threads",
        "status",
        "time_taken",
        "ref_values_file",
        "computed_values_file",
        "param_file",
        "temp_dir",
        "frame_count_ref",
        "frame_count_calc",
        "expected_frames",
        "worst_field",
        "worst_frame",
        "worst_ref",
        "worst_calc",
        "worst_diff",
        "worst_allowed",
        "rtol",
        "atol",
        "comparison_error",
        "error_message",
    ]
    with open(report_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_float(value) for key, value in row.items()})


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run PyDelphi trajectory regression tests."
    )
    parser.add_argument(
        "--no-cuda", action="store_true", help="Skip CUDA configurations."
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Skip configurations with more than 1 thread.",
    )
    parser.add_argument(
        "--no-single", action="store_true", help="Skip single precision configurations."
    )
    parser.add_argument(
        "--no-double", action="store_true", help="Skip double precision configurations."
    )
    parser.add_argument(
        "--timeout", type=int, default=300, help="Per-run timeout in seconds."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print additional progress information."
    )
    parser.add_argument(
        "--debug-files",
        action="store_true",
        default=False,
        help=(
            "Keep generated parameter files and computed TSV outputs under "
            f"{DEBUG_OUTPUT_DIR}/ and write full paths for those files in the "
            "report. Otherwise, temporary files are deleted and only filenames "
            "are recorded. Default: false."
        ),
    )
    args = parser.parse_args()

    if args.no_single and args.no_double:
        log(
            "Error: Cannot skip both single and double precision.",
            args.verbose,
            always=True,
        )
        sys.exit(1)

    debug_files = bool(args.debug_files)

    total_start = time.time()
    project_root = get_project_root()
    test_list_path = os.path.join(get_traj_data_dir(project_root), TEST_TRAJ_FILE)
    if not os.path.exists(test_list_path):
        log(
            f"Error: trajectory test list not found: {test_list_path}",
            args.verbose,
            always=True,
        )
        sys.exit(1)

    enabled_cases, disabled_cases = read_test_cases(test_list_path)
    has_custom_cases = any(row.get("crgsiz_set") == "custom" for row in enabled_cases)
    print_splash_message(True, has_custom_cases)

    combinations, configuration_skips = get_test_combinations(
        skip_cuda=args.no_cuda,
        skip_parallel=args.no_parallel,
        skip_single=args.no_single,
        skip_double=args.no_double,
    )
    if not combinations and not configuration_skips:
        log(
            "No valid test configurations selected. Exiting.", args.verbose, always=True
        )
        sys.exit(1)

    log(
        f"Loaded {len(enabled_cases) + len(disabled_cases)} trajectory cases: "
        f"{len(enabled_cases)} enabled, {len(disabled_cases)} disabled.",
        args.verbose,
        always=True,
    )
    log(
        f"Testing {len(combinations)} execution configuration(s) per enabled case.",
        args.verbose,
        always=True,
    )
    if debug_files:
        log(
            f"Debug files will be kept under: {os.path.abspath(DEBUG_OUTPUT_DIR)}",
            args.verbose,
            always=True,
        )

    all_rows: List[dict] = []
    for case_index, case_data in enumerate(enabled_cases, start=1):
        crgsiz_desc = (
            "no crgsiz"
            if is_na(case_data.get("crgsiz_set"))
            else f"crgsiz={case_data.get('crgsiz_set')}, mode={case_data.get('crgsiz_mode')}"
        )
        log(
            f"\nProcessing trajectory case {case_index}/{len(enabled_cases)}: "
            f"{case_data['case_id']} "
            f"({case_data['topology_file']} + {case_data['trajectory_file']}; "
            f"frames {','.join(str(frame) for frame in selected_frames(case_data))}; "
            f"{crgsiz_desc})",
            args.verbose,
            always=True,
        )
        for platform, precision, threads in combinations:
            all_rows.append(
                run_case_configuration(
                    case_data=case_data,
                    platform=platform,
                    precision=precision,
                    threads=threads,
                    project_root=project_root,
                    timeout=args.timeout,
                    verbose=args.verbose,
                    debug_files=debug_files,
                )
            )

    for case_data in enabled_cases:
        for config_skip in configuration_skips:
            all_rows.append(make_skipped_row(case_data, config_skip))

    write_report(all_rows, TEST_REPORT_FILE)

    num_failed = sum(1 for row in all_rows if row.get("status") == "FAIL")
    num_errors = sum(
        1 for row in all_rows if row.get("status") in {"ERROR", "TIMEOUT", "FILE_ERROR"}
    )
    num_skipped = sum(1 for row in all_rows if row.get("status") == "SKIPPED")
    num_passed = len(all_rows) - num_failed - num_errors - num_skipped
    total_hms = seconds_to_hms(time.time() - total_start)

    log("\n" + "=" * 80, args.verbose, always=True)
    log("TRAJECTORY REGRESSION TEST SUITE COMPLETE.", args.verbose, always=True)
    log(f"Total execution time: {total_hms}", args.verbose, always=True)
    log(f"Detailed results written to: {TEST_REPORT_FILE}", args.verbose, always=True)
    if disabled_cases:
        log(
            f"Disabled trajectory cases ignored: {len(disabled_cases)}",
            args.verbose,
            always=True,
        )
    log(
        f"Among {len(all_rows)} subtests: PASS={num_passed}, FAIL={num_failed}, "
        f"ERRORS/TIMEOUTS={num_errors}, SKIPPED={num_skipped}",
        args.verbose,
        always=True,
    )
    log("=" * 80 + "\n", args.verbose, always=True)

    if num_failed > 0 or num_errors > 0:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
