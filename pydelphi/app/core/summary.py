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


from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from pydelphi.foundation.enums import BoundaryCondition


def format_coords(coords: NDArray[np.floating]) -> str:
    """
    Formats coordinate arrays of shape (3,).

    Raises:
      - ValueError if coords is not a 1D float ndarray.
    """
    if not isinstance(coords, np.ndarray):
        raise TypeError("coords must be a numpy.ndarray")
    if coords.ndim != 1:
        raise ValueError("coords must be 1D")
    return ", ".join(f"{float(s):12.4f}" for s in coords)


def format_grid_size(grid_shape: NDArray[np.integer]) -> str:
    """
    Formats grid size arrays of shape (3,).

    Raises:
      - ValueError if grid_shape is not a 1D integer ndarray.
    """
    if not isinstance(grid_shape, np.ndarray):
        raise TypeError("grid_shape must be a numpy.ndarray")
    if grid_shape.ndim != 1:
        raise ValueError("grid_shape must be 1D")
    return ", ".join(f"{int(s):12d}" for s in grid_shape)


def summary_str(
    platform,
    inp,
    ctx,
    indent_spaces: int,
    field_width: int,
    format_specifier: str = "s",
) -> str:
    """
    Generates a summary string for a DelphiApp-like run.

    Reads:
      - platform.active, platform.precision, platform.names[...]
      - inp.info_str(...), inp.get_param_value(...)
      - ctx.* fields (if ctx is not None)

    Writes:
      - none

    Returns:
      - A newline-joined summary string (does not print).
    """
    indent = " " * indent_spaces
    field_format = f"{{:{field_width}{format_specifier}}}"
    summary_lines: list[str] = []

    summary_lines.append(
        f"{indent}{field_format.format('platform')} = {platform.active:10s}"
    )

    if platform.active == "cpu":
        summary_lines.append(
            f"{indent}{field_format.format('num_threads')} = {platform.names['cpu']['num_threads']:d}"
        )
    elif platform.active == "cuda":
        selected_dev_id = platform.names["cuda"]["selected_id"]
        summary_lines.append(
            f"{indent}{field_format.format('num_cpus')} = {platform.names['cpu']['num_threads']:d}"
        )
        summary_lines.append(
            f"{indent}{field_format.format('device_index')} = {selected_dev_id:d}"
        )
        summary_lines.append(
            f"{indent}{field_format.format('device_identity')} = "
            f"{platform.names['cuda']['device'][selected_dev_id]['device_identity']}"
        )

    summary_lines.append(
        f"{indent}{field_format.format('precision')} = {platform.precision}"
    )

    inp_str = inp.info_str(
        include_statements=True,
        include_functions=False,
        indent_spaces=indent_spaces,
        field_width=field_width,
        format_specifier=format_specifier,
    )
    summary_lines.append(f"{indent}{inp_str}")

    if ctx:
        if (
            inp.get_param_value("boundary_condition").int_value
            != BoundaryCondition.FOCUSING.int_value
        ):
            summary_lines.append(
                f"{indent}{field_format.format('percent_fill')} = {ctx.perfil:12.04f}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('gridbox_margin')} = {ctx.gridbox_margin:12.04f}"
            )

        summary_lines.append(
            f"{indent}{field_format.format('scale')} = {ctx.scale:12.04f}"
        )
        summary_lines.append(
            f"{indent}{field_format.format('grid_size: (nX, nY, nZ)')} = {format_grid_size(ctx.grid_shape)}"
        )

        summary_lines.append(
            f"{indent}{field_format.format('Sum of (-)-ve charges')} = {ctx.negative_charge:12.6g}"
        )
        summary_lines.append(
            f"{indent}{field_format.format('Sum of (+)-ve charges')} = {ctx.positive_charge:12.6g}"
        )
        summary_lines.append(
            f"{indent}{field_format.format('Sum of all charges')} = {ctx.total_charge:12.6g}"
        )
        summary_lines.append(
            f"{indent}{field_format.format('Number of (-)-ve charged atoms')} = {ctx.num_negative_charge:12.6g}"
        )
        summary_lines.append(
            f"{indent}{field_format.format('Number of (+)-ve charged atoms')} = {ctx.num_positive_charge:12.6g}"
        )

        if ctx.extremas_rule:
            summary_lines.append(
                f"{indent}{field_format.format('Minimum solute coords: (X, Y, Z)')} = {format_coords(ctx.coords_by_axis_min)}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('Maximum solute coords: (X, Y, Z)')} = {format_coords(ctx.coords_by_axis_max)}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('Range of solute coords: (X, Y, Z)')} = {format_coords(ctx.solute_range)}"
            )
        else:
            summary_lines.append(
                f"{indent}{field_format.format('Minimum solute coords (-) radius: (X, Y, Z)')} = {format_coords(ctx.boundary_min)}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('Maximum solute coords (+) radius: (X, Y, Z)')} = {format_coords(ctx.boundary_max)}"
            )
            summary_lines.append(
                f"{indent}{field_format.format('Range of solute coords (-/+) radius: (X, Y, Z)')} = {format_coords(ctx.solute_range)}"
            )

        summary_lines.append(
            f"{indent}{field_format.format('Centroid of (-)-ve charges: (X, Y, Z)')} = {format_coords(ctx.centroid_negative_charge)}"
        )
        summary_lines.append(
            f"{indent}{field_format.format('Centroid of (+)-ve charges: (X, Y, Z)')} = {format_coords(ctx.centroid_positive_charge)}"
        )
        summary_lines.append(
            f"{indent}{field_format.format('Centroid of all the atoms: (X, Y, Z)')} = {format_coords(ctx.centroid)}"
        )
        summary_lines.append(
            f"{indent}{field_format.format('Origin of gridbox: (X, Y, Z)')} = {format_coords(ctx.grid_origin)}"
        )

        center_coords = np.array(
            [
                s + ctx.grid_spacing * (nd // 2)
                for s, nd in zip(ctx.grid_origin, ctx.grid_shape)
            ]
        )
        summary_lines.append(
            f"{indent}{field_format.format('Center of gridbox: (X, Y, Z)')} = {format_coords(center_coords)}"
        )

    return "\n".join(summary_lines)


def summarize_parentrun_str(
    ctx,
    indent_spaces: int,
    field_width: int,
    format_specifier: str = "s",
) -> str:
    indent = " " * indent_spaces
    field_format = f"{{:{field_width}{format_specifier}}}"
    summary_lines: list[str] = []

    summary_lines.append(
        f"{indent}{field_format.format('Scale of parent-run')} = {ctx.scale_parentrun:12.04f}"
    )
    summary_lines.append(
        f"{indent}{field_format.format('Center of parent-run gridbox: (X, Y, Z)')} = {format_coords(ctx.grid_center_parentrun)}"
    )
    summary_lines.append(
        f"{indent}{field_format.format('grid_size parent-run: (nX, nY, nZ)')} = {format_grid_size(ctx.grid_shape_parentrun)}"
    )

    return "\n".join(summary_lines)
