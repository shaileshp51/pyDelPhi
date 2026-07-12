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

"""
Lite DCD trajectory reader for pyDelPhi.

Design goals
------------
- Minimal dependencies.
- Format-agnostic TrajectoryLite interface.
- Memory-safe frame iteration.
- Coordinates returned in Angstrom.
- Supports the standard CHARMM/NAMD style DCD layout with optional unit-cell
  blocks (48-byte record before XYZ blocks).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union
import struct

import numpy as np

from pydelphi.utils.io.lite.trajectory_lite import TrajectoryLite


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _read_exact(f, n: int) -> bytes:
    data = f.read(n)
    if len(data) != n:
        raise EOFError(f"Unexpected end of file while reading {n} bytes")
    return data


def _unpack_i32(raw: bytes, endian: str) -> int:
    return struct.unpack(f"{endian}i", raw)[0]


def _unpack_f32_array(
    raw: bytes, endian: str, natoms: int, *, dtype=np.float32
) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.dtype(f"{endian}f4"), count=natoms)
    if arr.shape[0] != natoms:
        raise EOFError(f"Expected {natoms} float32 coordinates, got {arr.shape[0]}")
    return arr.astype(dtype, copy=False)


def _detect_endian_and_header(f) -> tuple[str, int]:
    """
    Read the first block size and detect endianness.

    Standard DCD header:
    - first record block size = 84
    - payload includes a 4-byte signature plus 20 int32 words
    - trailing block size repeats 84
    """
    start_pos = f.tell()
    raw = _read_exact(f, 4)

    le = struct.unpack("<i", raw)[0]
    be = struct.unpack(">i", raw)[0]

    if le == 84:
        endian = "<"
        block_size = le
    elif be == 84:
        endian = ">"
        block_size = be
    else:
        raise ValueError(
            f"Not a standard DCD header block: block size is neither 84 in little- nor big-endian "
            f"(little={le}, big={be})"
        )

    # rewind to the start so the caller can read the whole record
    f.seek(start_pos, 0)
    return endian, block_size


def _skip_bytes(f, nbytes: int) -> None:
    if nbytes < 0:
        raise ValueError("nbytes must be >= 0")
    f.seek(nbytes, 1)


# ---------------------------------------------------------------------------
# Public reader
# ---------------------------------------------------------------------------


@dataclass
class DCDTrajectoryLite(TrajectoryLite):
    """
    Lite DCD trajectory reader.

    Parameters
    ----------
    path
        Path to the DCD file.
    dtype
        Output dtype for xyz arrays.
    """

    path: Union[str, Path]
    dtype: np.dtype = np.float32

    _fh: object = None
    _endian: str = "<"
    _natoms: int = -1
    _n_frames: Optional[int] = None
    _frame0_offset: int = -1

    def __post_init__(self):
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"DCD file not found: {p}")

        fh = p.open("rb")
        try:
            self._fh = fh
            self._endian, _ = _detect_endian_and_header(fh)
            self._parse_header()
        except Exception:
            fh.close()
            self._fh = None
            raise

    def close(self) -> None:
        fh = self._fh
        if fh is None:
            return
        try:
            fh.close()
        finally:
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def _read_block_size(self) -> int:
        raw = _read_exact(self._fh, 4)
        return _unpack_i32(raw, self._endian)

    def _parse_header(self) -> None:
        """
        Parse the header/title/atom-count sections and store the offset at the
        beginning of the first frame.
        """
        fh = self._fh
        endian = self._endian

        # Header record (84 bytes payload)
        start = self._read_block_size()
        if start != 84:
            raise ValueError(f"Unexpected DCD header block size {start}, expected 84")
        header_payload = _read_exact(fh, start)
        end = self._read_block_size()
        if end != start:
            raise ValueError("DCD header block sizes do not match")

        # Standard payload layout:
        # signature (4 bytes) + 20 x int32
        if len(header_payload) != 84:
            raise ValueError("Corrupt DCD header payload")

        signature = header_payload[:4].decode("ascii", errors="replace").strip()
        ints = np.frombuffer(
            header_payload[4:], dtype=np.dtype(f"{endian}i4"), count=20
        )
        if ints.shape[0] < 1:
            raise ValueError("DCD header missing frame count")

        nset = int(ints[0])
        if nset >= 0:
            self._n_frames = nset
        else:
            self._n_frames = None

        # Title block (variable length)
        title_block = self._read_block_size()
        _skip_bytes(fh, title_block)
        title_end = self._read_block_size()
        if title_end != title_block:
            raise ValueError("DCD title block sizes do not match")

        # Atom-count block
        natom_block = self._read_block_size()
        if natom_block not in (4, 8):
            # Most DCDs store natoms in a 4-byte int; tolerate 8 only if needed.
            raise ValueError(f"Unexpected DCD atom-count block size {natom_block}")
        natoms_raw = _read_exact(fh, natom_block)
        self._natoms = _unpack_i32(natoms_raw[:4], endian)
        natom_end = self._read_block_size()
        if natom_end != natom_block:
            raise ValueError("DCD atom-count block sizes do not match")

        if self._natoms <= 0:
            raise ValueError(f"Invalid NATOM value in DCD header: {self._natoms}")

        # Save the position of the first trajectory frame.
        self._frame0_offset = fh.tell()

        # Optional sanity check on signature; DCD commonly uses CORD or VELD.
        # We do not hard-fail because some writers use variants.
        _ = signature

    @property
    def natoms(self) -> int:
        return int(self._natoms)

    @property
    def n_frames(self) -> Optional[int]:
        return self._n_frames

    def _skip_one_frame(self) -> None:
        """
        Skip a single DCD frame, handling optional unit-cell blocks.
        """
        fh = self._fh

        blk = self._read_block_size()

        # Optional unit cell record (48 bytes)
        if blk == 48:
            _skip_bytes(fh, blk)
            end = self._read_block_size()
            if end != blk:
                raise ValueError("DCD unit-cell block sizes do not match")
            blk = self._read_block_size()

        expected = self._natoms * 4
        if blk != expected:
            raise ValueError(
                f"Unexpected X-block size {blk} (expected {expected}); "
                "this DCD may not match the standard float32 coordinate layout"
            )

        # x, y, z blocks: skip payload and trailing record for each
        _skip_bytes(fh, blk)
        end = self._read_block_size()
        if end != blk:
            raise ValueError("DCD X-block sizes do not match")

        blk = self._read_block_size()
        _skip_bytes(fh, blk)
        end = self._read_block_size()
        if end != blk:
            raise ValueError("DCD Y-block sizes do not match")

        blk = self._read_block_size()
        _skip_bytes(fh, blk)
        end = self._read_block_size()
        if end != blk:
            raise ValueError("DCD Z-block sizes do not match")

    def _read_one_frame(self) -> np.ndarray:
        """
        Read one DCD frame and return xyz as (N,3) array in Angstrom.
        """
        fh = self._fh

        blk = self._read_block_size()

        # Optional unit cell record (48 bytes)
        if blk == 48:
            _skip_bytes(fh, blk)
            end = self._read_block_size()
            if end != blk:
                raise ValueError("DCD unit-cell block sizes do not match")
            blk = self._read_block_size()

        expected = self._natoms * 4
        if blk != expected:
            raise ValueError(
                f"Unexpected X-block size {blk} (expected {expected}); "
                "this DCD may not match the standard float32 coordinate layout"
            )

        x = _unpack_f32_array(
            _read_exact(fh, blk), self._endian, self._natoms, dtype=self.dtype
        )
        end = self._read_block_size()
        if end != blk:
            raise ValueError("DCD X-block sizes do not match")

        blk = self._read_block_size()
        y = _unpack_f32_array(
            _read_exact(fh, blk), self._endian, self._natoms, dtype=self.dtype
        )
        end = self._read_block_size()
        if end != blk:
            raise ValueError("DCD Y-block sizes do not match")

        blk = self._read_block_size()
        z = _unpack_f32_array(
            _read_exact(fh, blk), self._endian, self._natoms, dtype=self.dtype
        )
        end = self._read_block_size()
        if end != blk:
            raise ValueError("DCD Z-block sizes do not match")

        xyz = np.column_stack((x, y, z)).astype(self.dtype, copy=False)
        return xyz

    def iter_xyz(
        self, *, start: int = 0, stop: Optional[int] = None, stride: int = 1
    ) -> Iterator[Tuple[int, np.ndarray]]:
        """
        Yield (frame_index, xyz_angstrom).

        start/stop/stride follow Python slice semantics on frame indices.
        """
        if stride <= 0:
            raise ValueError("stride must be >= 1")

        fh = self._fh
        fh.seek(self._frame0_offset, 0)

        frame_index = 0
        n_frames = self._n_frames

        while True:
            pos = fh.tell()
            # EOF -> stop.
            first = fh.read(4)
            if len(first) == 0:
                break
            if len(first) != 4:
                raise EOFError("Unexpected EOF while reading DCD frame block size")
            fh.seek(pos, 0)

            # Record current frame if it falls within the requested slice.
            in_range = frame_index >= start and (stop is None or frame_index < stop)
            take = in_range and ((frame_index - start) % stride == 0)

            if take:
                xyz = self._read_one_frame()
                yield frame_index, xyz
            else:
                self._skip_one_frame()

            frame_index += 1

            if n_frames is not None and frame_index >= n_frames:
                break

            if stop is not None and frame_index >= stop:
                break


def open_dcd_lite(path: Union[str, Path], *, dtype=np.float32) -> DCDTrajectoryLite:
    """Convenience constructor."""
    return DCDTrajectoryLite(path=path, dtype=dtype)


if __name__ == "__main__":
    import sys

    traj = open_dcd_lite(sys.argv[1])
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
