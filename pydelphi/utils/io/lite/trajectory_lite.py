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

from dataclasses import dataclass
from os import path
from typing import Iterator, Optional, Protocol, Tuple

import numpy as np


class TrajectoryLite(Protocol):
    """
    Format-agnostic trajectory interface.

    Conventions (frozen):
    - atom order matches TopologyLite arrays
    - xyz returned in Angstrom
    - xyz shape: (N, 3), dtype float32/float64
    """

    natoms: int
    n_frames: Optional[int]  # may be unknown for streaming readers

    def iter_xyz(
        self, *, start: int = 0, stop: Optional[int] = None, stride: int = 1
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Yield (frame_index, xyz_angstrom).
        frame_index is the absolute frame number in the underlying trajectory.
        """


@dataclass(frozen=True)
class SingleFrameTrajectoryLite:
    """Convenience: treat static coordinates as a 1-frame trajectory."""

    xyz: np.ndarray  # (N,3) Angstrom

    def __post_init__(self):
        if self.xyz.ndim != 2 or self.xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (N,3)")

    @property
    def natoms(self) -> int:
        return int(self.xyz.shape[0])

    @property
    def n_frames(self) -> int:
        return 1

    def iter_xyz(self, *, start: int = 0, stop: Optional[int] = None, stride: int = 1):
        if start > 0:
            return
        if stop is not None and stop <= 0:
            return
        if stride <= 0:
            raise ValueError("stride must be >= 1")
        yield 0, self.xyz


_SUPPORTED_TRAJECTORY_FORMATS = {
    "nc": "nc",
    "netcdf": "nc",
    "amber_nc": "nc",
    "amber-netcdf": "nc",
    "dcd": "dcd",
    "trr": "trr",
}


def normalize_trajectory_format(fmt: str) -> str:
    """
    Normalize a user-facing lite trajectory format name.

    Supported exposed lite trajectory formats:
        nc, dcd, trr
    """
    fmt = str(fmt or "").strip().lower()

    if fmt not in _SUPPORTED_TRAJECTORY_FORMATS:
        raise ValueError(
            f"Unsupported trajectory format in traj mode: {fmt!r}. "
            "Supported lite trajectory formats are: nc, dcd, trr."
        )

    return _SUPPORTED_TRAJECTORY_FORMATS[fmt]


def open_trajectory_lite(file_path: str, fmt: str) -> TrajectoryLite:
    """
    Open a trajectory file through the lite trajectory interface.

    This is the format-dispatch boundary for trajectory readers. Higher-level
    callers should not import individual nc/dcd/trr readers directly.
    """
    if not file_path:
        raise ValueError("Trajectory input missing required attribute: file=...")

    if not path.isfile(file_path):
        raise FileNotFoundError(f"Trajectory file not found: {file_path}")

    traj_fmt = normalize_trajectory_format(fmt)

    if traj_fmt == "nc":
        from pydelphi.utils.io.trajectory.nc_reader_lite import (
            open_amber_netcdf_lite,
        )

        return open_amber_netcdf_lite(file_path)

    if traj_fmt == "dcd":
        from pydelphi.utils.io.trajectory.dcd_reader_lite import open_dcd_lite

        return open_dcd_lite(file_path)

    if traj_fmt == "trr":
        from pydelphi.utils.io.trajectory.trr_reader_lite import open_trr_lite

        return open_trr_lite(file_path)

    raise ValueError(f"Unsupported trajectory format in traj mode: {traj_fmt!r}")
