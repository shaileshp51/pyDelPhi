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

from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import numpy as np

from pydelphi.utils.io.lite.trajectory_lite import TrajectoryLite


def _open_netcdf(path):
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"NetCDF trajectory file not found: {path}")

    try:
        import netCDF4
    except ImportError as e:
        raise ImportError(
            "Reading Amber NetCDF trajectories requires the 'netCDF4' package.\n"
            "Install with: conda install -c conda-forge netcdf4\n"
            "or: pip install netCDF4"
        ) from e

    try:
        ds = netCDF4.Dataset(path, mode="r")
    except Exception as e:
        raise RuntimeError(
            f"Failed to open NetCDF file with netCDF4.\n"
            f"Path: {path}\n"
            f"Error: {type(e).__name__}: {e}"
        ) from e

    return ds, "netCDF4", None


def _get_attr(var, name: str, default=None):
    # netCDF4 variables expose .getncattr; scipy exposes attributes differently
    if hasattr(var, "getncattr"):
        try:
            return var.getncattr(name)
        except Exception:
            return default
    return getattr(var, name, default)


@dataclass
class AmberNetCDFTrajectoryLite(TrajectoryLite):
    """
    Lite AMBER NetCDF trajectory reader.

    Provides TrajectoryLite.iter_xyz yielding xyz in Angstrom.
    """

    path: str
    coord_var: str = "coordinates"  # Amber standard
    dtype: np.dtype = np.float32  # output dtype
    _ds: object = None  # netcdf handle (internal)
    _backend: str = ""
    _xyz_scale: float = 1.0  # to Angstrom

    def __post_init__(self):
        ds, backend, _ = _open_netcdf(self.path)
        self._ds = ds
        self._backend = backend

        if self.coord_var not in ds.variables:
            # common alternatives people sometimes use
            candidates = [
                k for k in ("coords", "coord", "xyz", "positions") if k in ds.variables
            ]
            hint = (
                f" Available coordinate-like vars: {candidates}" if candidates else ""
            )
            raise ValueError(f"NetCDF missing variable {self.coord_var!r}.{hint}")

        v = ds.variables[self.coord_var]
        shape = v.shape
        if len(shape) != 3 or shape[-1] != 3:
            raise ValueError(
                f"{self.coord_var!r} has shape {shape}, expected (nframe, natom, 3)"
            )

        # Units handling: Amber NetCDF is typically Angstrom
        units = _get_attr(v, "units", None)
        if units is None:
            self._xyz_scale = 1.0
        else:
            u = str(units).strip().lower()
            if u in ("angstrom", "angstroms", "a", "å"):
                self._xyz_scale = 1.0
            elif u in ("nm", "nanometer", "nanometers"):
                self._xyz_scale = 10.0  # nm -> Å
            else:
                # Unknown units: default to 1.0 but keep explicit error if you prefer strictness
                self._xyz_scale = 1.0

    def close(self) -> None:
        if self._ds is None:
            return
        try:
            self._ds.close()
        finally:
            self._ds = None

    @property
    def natoms(self) -> int:
        v = self._ds.variables[self.coord_var]
        return int(v.shape[1])

    @property
    def n_frames(self) -> Optional[int]:
        v = self._ds.variables[self.coord_var]
        return int(v.shape[0])

    def iter_xyz(
        self, *, start: int = 0, stop: Optional[int] = None, stride: int = 1
    ) -> Iterator[Tuple[int, np.ndarray]]:
        if stride <= 0:
            raise ValueError("stride must be >= 1")

        v = self._ds.variables[self.coord_var]
        n_frames = int(v.shape[0])

        if start < 0:
            start = max(n_frames + start, 0)
        if stop is None:
            stop = n_frames
        elif stop < 0:
            stop = max(n_frames + stop, 0)

        if start > n_frames:
            return
        stop = min(stop, n_frames)

        # Read one frame at a time (memory-safe for huge trajectories)
        for fi in range(start, stop, stride):
            xyz = v[fi, :, :]
            # ensure numpy array
            xyz = np.asarray(xyz, dtype=self.dtype)
            if self._xyz_scale != 1.0:
                xyz *= self._xyz_scale
            yield fi, xyz


def open_amber_netcdf_lite(
    path: str, *, coord_var: str = "coordinates"
) -> AmberNetCDFTrajectoryLite:
    """
    Convenience constructor.
    """
    return AmberNetCDFTrajectoryLite(path=path, coord_var=coord_var)


# ------------------------------------
# Tiny in-place writer helper (optional)
# ------------------------------------
def write_xyz_into_atom_data(
    atom_data: np.ndarray,
    xyz: np.ndarray,
    *,
    x_col: int,
    y_col: int,
    z_col: int,
) -> None:
    """
    In-place update atom_data coordinate columns from xyz.

    atom_data: (N, M)
    xyz: (N, 3)
    """
    atom_data[:, x_col] = xyz[:, 0]
    atom_data[:, y_col] = xyz[:, 1]
    atom_data[:, z_col] = xyz[:, 2]
