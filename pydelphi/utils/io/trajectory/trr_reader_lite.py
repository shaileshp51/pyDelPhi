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

"""trr_reader_lite.py

Dependency-light GROMACS TRR trajectory reader for pyDelPhi.

The reader is intentionally small and avoids MDAnalysis/mdtraj.  It implements
only the `TrajectoryLite` contract:

- `natoms`
- `n_frames` (optional, cached by a light scan)
- `iter_xyz(start=0, stop=None, stride=1)` yielding `(frame_index, xyz_angstrom)`

Coordinates are returned in Angstrom.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union
import struct

import numpy as np

from pydelphi.utils.io.lite.trajectory_lite import TrajectoryLite


GROMACS_MAGIC = 1993
NM2ANG = 10.0


class _BinaryReader:
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"TRR file not found: {self.path}")
        self.f = self.path.open("rb")
        self.endian = self._detect_endian()

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass

    def tell(self) -> int:
        return int(self.f.tell())

    def seek(self, offset: int, whence: int = 0) -> None:
        self.f.seek(offset, whence)

    def _read_exact(self, n: int) -> bytes:
        data = self.f.read(n)
        if len(data) != n:
            raise EOFError(f"Unexpected EOF while reading {n} bytes from {self.path}")
        return data

    def _detect_endian(self) -> str:
        raw = self.f.read(4)
        if len(raw) != 4:
            raise EOFError(f"TRR file too small: {self.path}")
        le = struct.unpack("<i", raw)[0]
        if le == GROMACS_MAGIC:
            self.f.seek(0, 0)
            return "<"
        be = struct.unpack(">i", raw)[0]
        if be == GROMACS_MAGIC:
            self.f.seek(0, 0)
            return ">"
        # Fall back to little-endian; parsing will fail fast if wrong.
        self.f.seek(0, 0)
        return "<"

    def read_int(self) -> int:
        return struct.unpack(self.endian + "i", self._read_exact(4))[0]

    def read_uint(self) -> int:
        return struct.unpack(self.endian + "I", self._read_exact(4))[0]

    def read_float(self) -> float:
        return struct.unpack(self.endian + "f", self._read_exact(4))[0]

    def read_double(self) -> float:
        return struct.unpack(self.endian + "d", self._read_exact(8))[0]

    def read_string(self) -> str:
        n = self.read_int()
        if n < 0 or n > 10_000_000:
            raise ValueError(f"Unreasonable string length {n} in {self.path}")
        raw = self._read_exact(n)
        pad = (-n) % 4
        if pad:
            self.f.seek(pad, 1)
        return raw.rstrip(b"\x00").decode("utf-8", errors="replace")

    def skip_bytes(self, n: int) -> None:
        if n > 0:
            self.f.seek(n, 1)

    def skip_int(self, n: int = 1) -> None:
        self.skip_bytes(4 * n)

    def skip_real(self, n: int, precision: int) -> None:
        self.skip_bytes(n * precision)


@dataclass(frozen=True)
class _FrameMeta:
    natoms: int
    x_size: int
    v_size: int
    f_size: int
    precision: int


class TRRTrajectoryLite(TrajectoryLite):
    """Lite TRR reader with a dependency-light native parser."""

    def __init__(
        self, path: Union[str, Path], *, dtype=np.float32, convert_units: bool = True
    ):
        self.path = Path(path)
        self.dtype = np.dtype(dtype)
        self.convert_units = bool(convert_units)
        self._natoms: Optional[int] = None
        self._n_frames: Optional[int] = None
        self._validate_and_prime()

    def _validate_and_prime(self) -> None:
        reader = _BinaryReader(self.path)
        try:
            meta = self._read_next_frame_meta(reader)
            self._natoms = int(meta.natoms)
        finally:
            reader.close()

    @property
    def natoms(self) -> int:
        if self._natoms is None:
            self._validate_and_prime()
        return int(self._natoms)

    @property
    def n_frames(self) -> Optional[int]:
        if self._n_frames is None:
            self._n_frames = self._count_frames()
        return self._n_frames

    def _read_next_frame_meta(self, reader: _BinaryReader) -> _FrameMeta:
        # TRR frame header per the reference C++ code.
        magic = reader.read_int()
        if magic != GROMACS_MAGIC:
            # Try byte-swapped interpretation once.
            swapped = struct.unpack(
                ">i" if reader.endian == "<" else "<i",
                struct.pack(reader.endian + "i", magic),
            )[0]
            if swapped != GROMACS_MAGIC:
                raise EOFError("No more TRR frames or invalid TRR magic number")
            reader.endian = ">" if reader.endian == "<" else "<"

        _version = reader.read_int()
        _title = reader.read_string()

        ir_size = reader.read_int()
        e_size = reader.read_int()
        box_size = reader.read_int()
        vir_size = reader.read_int()
        pres_size = reader.read_int()
        top_size = reader.read_int()
        sym_size = reader.read_int()

        x_size = reader.read_int()
        v_size = reader.read_int()
        f_size = reader.read_int()

        natoms = reader.read_int()

        # step / nre
        reader.skip_int(2)

        # time / lambda
        reader.skip_real(1, 4)
        reader.skip_real(1, 4)

        # Skip the seven size blocks.
        for sz in (ir_size, e_size, box_size, vir_size, pres_size, top_size, sym_size):
            reader.skip_bytes(sz)

        return _FrameMeta(
            natoms=natoms,
            x_size=x_size,
            v_size=v_size,
            f_size=f_size,
            precision=(x_size // (3 * natoms)) if natoms > 0 else 4,
        )

    def _skip_frame_payload(self, reader: _BinaryReader, meta: _FrameMeta) -> None:
        reader.skip_bytes(meta.x_size + meta.v_size + meta.f_size)

    def _read_frame_xyz(self, reader: _BinaryReader, meta: _FrameMeta) -> np.ndarray:
        precision = meta.precision
        if precision == 4:
            dtype = np.dtype(self.dtype).newbyteorder(reader.endian)
            raw = reader._read_exact(meta.x_size)
            arr = np.frombuffer(raw, dtype=dtype, count=3 * meta.natoms)
        elif precision == 8:
            dtype = np.dtype("f8").newbyteorder(reader.endian)
            raw = reader._read_exact(meta.x_size)
            arr = np.frombuffer(raw, dtype=dtype, count=3 * meta.natoms)
        else:
            raise ValueError(f"Unsupported TRR coordinate precision: {precision}")

        if arr.size != 3 * meta.natoms:
            raise EOFError(
                f"TRR coordinate block truncated: expected {3 * meta.natoms} values"
            )

        xyz = arr.reshape(meta.natoms, 3).astype(self.dtype, copy=False)
        if self.convert_units:
            xyz = xyz * NM2ANG
        return xyz

    def _count_frames(self) -> int:
        reader = _BinaryReader(self.path)
        count = 0
        try:
            while True:
                try:
                    meta = self._read_next_frame_meta(reader)
                except EOFError:
                    break
                self._skip_frame_payload(reader, meta)
                count += 1
        finally:
            reader.close()
        return count

    def _normalize_bounds(
        self, start: int, stop: Optional[int]
    ) -> tuple[int, Optional[int]]:
        n_frames = self.n_frames
        if start < 0:
            if n_frames is None:
                raise ValueError("negative start requires a known frame count")
            start = max(n_frames + start, 0)
        if stop is not None and stop < 0:
            if n_frames is None:
                raise ValueError("negative stop requires a known frame count")
            stop = max(n_frames + stop, 0)
        return start, stop

    def iter_xyz(
        self, *, start: int = 0, stop: Optional[int] = None, stride: int = 1
    ) -> Iterator[Tuple[int, np.ndarray]]:
        if stride <= 0:
            raise ValueError("stride must be >= 1")

        start, stop = self._normalize_bounds(start, stop)

        reader = _BinaryReader(self.path)
        frame_index = 0
        try:
            while True:
                try:
                    meta = self._read_next_frame_meta(reader)
                except EOFError:
                    break

                record = (
                    frame_index >= start
                    and (stop is None or frame_index < stop)
                    and ((frame_index - start) % stride == 0)
                )
                if record:
                    xyz = self._read_frame_xyz(reader, meta)
                    yield frame_index, xyz
                else:
                    self._skip_frame_payload(reader, meta)

                frame_index += 1
                if stop is not None and frame_index >= stop:
                    break
        finally:
            reader.close()


def open_trr_lite(
    path: Union[str, Path], *, dtype=np.float32, convert_units: bool = True
) -> TRRTrajectoryLite:
    return TRRTrajectoryLite(path=path, dtype=dtype, convert_units=convert_units)


if __name__ == "__main__":
    import sys

    traj = open_trr_lite(sys.argv[1])
    print("natoms:", traj.natoms)
    print("n_frames:", traj.n_frames)
    for i, xyz in traj.iter_xyz(stop=1):
        print(
            "frame",
            i,
            "xyz.shape",
            xyz.shape,
            "min/max",
            float(xyz.min()),
            float(xyz.max()),
        )
        break
