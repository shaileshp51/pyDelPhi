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
format_resolver.py

Central format inference constants and helpers for pyDelPhi function-style
parameters.

This module is the single authoritative place for:

    1. extension -> format inference
    2. file-like attribute priority for format inference

Policy
------
- format="auto" means infer from a file-like attribute.
- Explicit user-supplied format is authoritative.
- Ambiguous extensions are intentionally omitted. For example, ".top" is not
  mapped because it can refer to incompatible topology formats.
- in(crgsiz, ...) / in(charge_size, ...) is semantic set selection, not
  file-format inference, and should not use this module for set-name logic.
"""

from __future__ import annotations

from os import path
from typing import Mapping, Optional


FORMAT_AUTO = "auto"

FORMAT_BY_EXT: Mapping[str, str] = {
    ".pdb": "pdb",
    ".pqr": "pqr",
    ".psf": "psf",
    ".prmtop": "prmtop",
    ".parm7": "prmtop",
    ".dcd": "dcd",
    ".trr": "trr",
    ".nc": "netcdf",
    ".netcdf": "netcdf",
    ".cube": "cube",
    ".phi": "phi",
    ".frc": "frc",
    ".tsv": "tsv",
    ".csv": "csv",
}

# Authoritative file-like attribute priority for format inference.
#
# The order matters when a function has more than one file-like attribute.
# For frc(format=auto), "outfile" should win over "target_file" because the
# current frc.format attribute describes output format.
FILE_LIKE_FORMAT_ATTR_PRIORITY: tuple[str, ...] = (
    "outfile",
    "file",
    "infile",
    "target_file",
    "tfile",
)

# Use this for future attributes that describe target/input-file format rather
# than primary output format.
TARGET_FILE_LIKE_FORMAT_ATTR_PRIORITY: tuple[str, ...] = (
    "target_file",
    "tfile",
    "file",
    "infile",
    "outfile",
)

FORMAT_ALIASES: Mapping[str, str] = {
    "auto": "auto",
    "nc": "netcdf",
    "netcdf": "netcdf",
    "cdf": "netcdf",
    "parm7": "prmtop",
    "prmtop": "prmtop",
    "pdb": "pdb",
    "pqr": "pqr",
    "psf": "psf",
    "dcd": "dcd",
    "trr": "trr",
    "cube": "cube",
    "phi": "phi",
    "frc": "frc",
    "tsv": "tsv",
    "csv": "csv",
}


def normalize_format_value(fmt: object) -> str:
    """Normalize a supplied format string without guessing from a file."""
    value = str(fmt or "").strip().lower()
    if not value:
        return FORMAT_AUTO
    return FORMAT_ALIASES.get(value, value)


def infer_format_from_filename(filename: object) -> str:
    """Infer canonical format from a filename extension."""
    filename_s = str(filename or "").strip()
    if not filename_s:
        raise ValueError("format='auto' requires a non-empty file-like attribute.")

    ext = path.splitext(filename_s)[1].lower()
    if not ext:
        raise ValueError(
            f"Cannot infer format from file name {filename_s!r}: no extension. "
            "Specify format explicitly."
        )

    fmt = FORMAT_BY_EXT.get(ext)
    if fmt is None:
        raise ValueError(
            f"Cannot infer format from file extension {ext!r} in {filename_s!r}. "
            "Specify format explicitly."
        )

    return fmt


def find_file_like_value(
    call: Mapping[str, object],
    *,
    priority: tuple[str, ...] = FILE_LIKE_FORMAT_ATTR_PRIORITY,
) -> tuple[Optional[str], Optional[object]]:
    """Return the first non-empty file-like attribute from a call dictionary."""
    for key in priority:
        value = call.get(key)
        if value not in (None, ""):
            return key, value
    return None, None


def resolve_auto_format(
    fmt: object,
    *,
    file_value: object,
) -> str:
    """Normalize an explicit format, or infer when format is auto/empty."""
    normalized = normalize_format_value(fmt)
    if normalized == FORMAT_AUTO:
        return infer_format_from_filename(file_value)
    return normalized


def resolve_call_format_auto(
    call: dict,
    *,
    file_attr_priority: tuple[str, ...] = FILE_LIKE_FORMAT_ATTR_PRIORITY,
) -> dict:
    """Resolve call['format'] or call['fmt'] in-place when it is auto/empty."""
    if "format" in call:
        fmt_key = "format"
    elif "fmt" in call:
        fmt_key = "fmt"
    else:
        return call

    fmt = normalize_format_value(call.get(fmt_key))
    if fmt == FORMAT_AUTO:
        file_key, file_value = find_file_like_value(call, priority=file_attr_priority)
        if file_key is None:
            raise ValueError(
                "format='auto' requires one of these file-like attributes: "
                + ", ".join(file_attr_priority)
            )
        call[fmt_key] = infer_format_from_filename(file_value)
    else:
        call[fmt_key] = fmt

    return call
