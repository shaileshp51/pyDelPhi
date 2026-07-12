#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
Atom materialization utilities.

This module adapts topology + trajectory data into the
Delphi internal atom array layout used by both static
and trajectory-based workflows.

The resulting atoms_data array is compatible with the
static execution pipeline; only coordinates are updated
in-place for successive frames.
"""

import numpy as np

import pydelphi.utils.io.writers as wrt

from pydelphi.constants import (
    LEN_ATOMFIELDS,
    ATOMFIELD_MEDIA_ID,
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_CHARGE,
    ATOMFIELD_RADIUS,
    ATOMFIELD_ATOMIC_NUMBER,
)

MODULE_NAME = __name__


def build_resid_from_residue_pointer(
    residue_pointer_1based: np.ndarray, natoms: int
) -> np.ndarray:
    """
    residue_pointer_1based: (NRES,) 1-based atom start indices for each residue
    returns resid0: (N,) 0-based residue index for each atom
    """
    # Convert to 0-based start indices
    starts0 = residue_pointer_1based.astype(np.int64) - 1  # (NRES,)
    atom_idx = np.arange(natoms, dtype=np.int64)
    # right insertion gives "largest start <= atom_idx"
    resid0 = np.searchsorted(starts0, atom_idx, side="right") - 1
    # Safety
    resid0[resid0 < 0] = 0
    return resid0.astype(np.int32)


def build_atoms_from_top_and_frame0(
    top,
    frame_xyz: np.ndarray,
    delphi_real,
    chain_default: str = "",
    segid_default: str = "",
):
    """
    Build atom_keys and atoms_data in the same layout as static mode.

    - atoms_data is allocated once and reused across frames.
    - only X/Y/Z are updated in-place each frame.
    """
    natoms = int(top.natoms)
    object_media_number = 1.0

    # ---- 1) coords from first frame ----
    # Expect traj.get_frame(i) -> (N,3) float (Angstrom)

    if frame_xyz.shape[0] != natoms:
        raise ValueError(
            f"natoms mismatch: top={natoms}, traj frame={frame_xyz.shape[0]}"
        )

    # ---- 2) allocate atoms_data ----
    atoms_data = np.zeros((natoms, LEN_ATOMFIELDS), dtype=delphi_real)

    atoms_data[:, ATOMFIELD_X] = frame_xyz[:, 0].astype(delphi_real, copy=False)
    atoms_data[:, ATOMFIELD_Y] = frame_xyz[:, 1].astype(delphi_real, copy=False)
    atoms_data[:, ATOMFIELD_Z] = frame_xyz[:, 2].astype(delphi_real, copy=False)
    atoms_data[:, ATOMFIELD_MEDIA_ID] = delphi_real(object_media_number)

    atoms_data[:, ATOMFIELD_CHARGE] = top.atom_charge.astype(delphi_real, copy=False)

    # radii may be NaN if missing; can be validated earlier in process_traj_inputs()
    atoms_data[:, ATOMFIELD_RADIUS] = top.atom_radius.astype(delphi_real, copy=False)

    # ---- 3) residue indices (optional but recommended for keys) ----
    if getattr(top, "residue_pointer", None) is not None:
        resid0 = build_resid_from_residue_pointer(top.residue_pointer, natoms)
        resSeq = resid0 + 1  # user-facing
    else:
        resid0 = np.zeros(natoms, dtype=np.int32)
        resSeq = np.ones(natoms, dtype=np.int32)

    # ---- 4) atom serial (1-based increasing) ----
    atom_serial = np.arange(natoms, dtype=np.int32) + 1

    # ---- 5) build keys ----
    atom_name = getattr(top, "atom_name", None)
    residue_label = getattr(top, "res_name", None)
    # print(top)

    # keys built once
    atom_keys = []
    atom_keys_append = atom_keys.append

    for i in range(natoms):
        an = atom_name[i] if atom_name is not None else f"A{i+1}"
        rl = residue_label[resid0[i]] if residue_label is not None else "RES"
        key = f"{chain_default}:{segid_default}:{rl}:{int(resSeq[i])}:{an}:{int(atom_serial[i])}"
        atom_keys_append(key)
        atoms_data[i, ATOMFIELD_ATOMIC_NUMBER] = wrt.get_atomic_number_from_atomname(an)

    return atom_keys, atoms_data, atom_serial, resid0, resSeq


def update_atoms_coords_inplace(
    traj,
    frame_xyz: np.ndarray,
    atoms_data: np.ndarray,
    delphi_real,
):
    atoms_data[:, ATOMFIELD_X] = frame_xyz[:, 0].astype(delphi_real, copy=False)
    atoms_data[:, ATOMFIELD_Y] = frame_xyz[:, 1].astype(delphi_real, copy=False)
    atoms_data[:, ATOMFIELD_Z] = frame_xyz[:, 2].astype(delphi_real, copy=False)
