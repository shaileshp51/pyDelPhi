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

from typing import Any

import numpy as np

from pydelphi.foundation.enums import BioModel, BoundaryCondition, DielectricModel
from pydelphi.constants import LEN_ATOMFIELDS


def _atom_focus_signature(atoms):
    sig = []
    for a in atoms:
        # adjust field names/indexes to your atom representation
        sig.append(
            (
                getattr(a, "serial", None),
                getattr(a, "name", None),
                getattr(a, "resname", None),
                getattr(a, "resid", None),
                round(float(getattr(a, "x", 0.0)), 6),
                round(float(getattr(a, "y", 0.0)), 6),
                round(float(getattr(a, "z", 0.0)), 6),
                round(float(getattr(a, "crg", 0.0)), 6),
                round(float(getattr(a, "rad", 0.0)), 6),
            )
        )
    return hash(tuple(sig)), len(sig)


def _focus_atoms_data_debug(ctx, label):
    import numpy as np
    import hashlib

    arr = np.asarray(ctx.atoms_data)

    # Stable content hash, independent of Python's randomized hash seed.
    arr_c = np.ascontiguousarray(arr)
    digest = hashlib.sha256(arr_c.view(np.uint8)).hexdigest()

    print(
        f"[FOCUS DEBUG] {label} "
        f"ctx.num_atoms={ctx.num_atoms} "
        f"len(ctx.atoms_data)={len(ctx.atoms_data)} "
        f"shape={arr.shape} dtype={arr.dtype} "
        f"sha256={digest}",
        flush=True,
    )

    # Numeric summaries help when hashes differ.
    if arr.size:
        print(
            f"[FOCUS DEBUG] {label} "
            f"min={np.nanmin(arr):.12e} "
            f"max={np.nanmax(arr):.12e} "
            f"sum={np.nansum(arr):.12e} "
            f"mean={np.nanmean(arr):.12e}",
            flush=True,
        )

        nshow = min(3, len(arr))
        print(f"[FOCUS DEBUG] {label} first{nshow}={arr[:nshow]}", flush=True)
        print(f"[FOCUS DEBUG] {label} last{nshow}={arr[-nshow:]}", flush=True)


def prepare_focusing_if_needed(
    *,
    inp: Any,
    ctx: Any,
    rdr: Any,
    frc_target_atoms: Any = None,
    delphi_real: Any = float,
) -> None:
    """
    Prepare focusing inputs for PBE runs.

    Contract:
      - Does NOT compute ctx.grid_origin. Caller must have finalized:
          ctx.scale, ctx.grid_center, ctx.grid_shape, ctx.grid_origin
      - On focusing:
          * validates TWODIELECTRIC + in_phi
          * validates that frc target/evaluation atoms exist
          * filters current source atoms to the focused child-grid region
          * fills ctx.*_parentrun from cube
          * sets ctx.grid_origin_parentrun
          * re-summarizes the focused source subset
      - On non-focusing PBE:
          * sets parentrun fields to current-run values (self is its own parent)
          * creates dummy ctx.phimap_parentrun (required by solver signature)
      - On non-PBE biomodel:
          * no-op (leaves any existing parentrun as-is)

    Important distinction:
      - ctx.atoms_data is the source atom set used by the PB solve.
      - frc_target_atoms are FRC/evaluation sites only.
      - Focusing filters source atoms to the child box; it must not replace
        ctx.atoms_data with frc_target_atoms.
    """
    if inp.get_param_value("biomodel").int_value != BioModel.PBE.int_value:
        return

    bc = inp.get_param_value("boundary_condition")
    is_focusing = bc.int_value == BoundaryCondition.FOCUSING.int_value

    if is_focusing:
        diel = inp.get_param_value("dielectric_model")
        if diel != DielectricModel.TWODIELECTRIC:
            raise ValueError(
                "FOCUSING boundary condition is compatible with only TWODIELECTRIC dielectric model"
            )

        in_phi = inp.get_param("in__phi")
        if not in_phi.issupplied:
            raise ValueError(
                "Parentrun phimap is not provided but required for FOCUSING."
            )

        if frc_target_atoms is None or len(frc_target_atoms) == 0:
            raise ValueError(
                "FOCUSING boundary condition requires non-empty frc target/evaluation atoms. "
                "Provide frc(target=..., ...) or frc(target_file=..., ...)."
            )

        # # Prepare focusing on the current source atom set. This reduces the
        # # child-run PB source to source atoms inside/near the focused box.
        # # It must not use frc_target_atoms, which are evaluation sites only.
        num_atoms_focus, atoms_data_focus, epsdim_focus, focus_start = (
            ctx.prepare_focusing(
                ctx.scale,
                ctx.num_atoms,
                ctx.num_objects,
                ctx.grid_shape,
                ctx.acenter,
                ctx.atoms_data,
            )
        )

        # print(
        #     "[FOCUS DEBUG] prepare_focusing returned "
        #     f"num_atoms_focus={num_atoms_focus} "
        #     f"len_atoms_data_focus={len(atoms_data_focus)} "
        #     f"epsdim_focus={epsdim_focus} "
        #     f"focus_start={focus_start}",
        #     flush=True,
        # )

        atoms_to_focus = {}
        focused_parent_ids = []

        for this_atom in atoms_data_focus:
            parent_idx = int(this_atom[LEN_ATOMFIELDS])
            focused_parent_ids.append(parent_idx)

            if parent_idx < 0:
                raise ValueError(
                    "Internal focusing error: focused source atom has invalid parent index. "
                    "Only source atoms should be passed through ctx.prepare_focusing()."
                )

            atom_k = ctx.atoms_index_to_keys[parent_idx]
            atoms_to_focus[atom_k] = this_atom

        # print(
        #     "[FOCUS DEBUG] focused parent ids "
        #     f"len={len(focused_parent_ids)} "
        #     f"hash={hash(tuple(focused_parent_ids))} "
        #     f"first20={focused_parent_ids[:20]} "
        #     f"last20={focused_parent_ids[-20:]}",
        #     flush=True,
        # )

        (
            ctx.scale_parentrun,
            ctx.grid_center_parentrun,
            ctx.grid_shape_parentrun,
            ctx.phimap_parentrun,
            _read_origin_bohr,
            _read_vectors_bohr,
            ctx.phimap_comment_parentrun,
            _read_data_type_comment,
            ctx.phimap_endianness_parentrun,
            ctx.phimap_marker_parentrun,
        ) = rdr.read_cube(
            in_phi.get_attribute("file"),
            format=in_phi.get_attribute("format"),
        )

        ctx.grid_origin_parentrun = ctx.grid_center_parentrun - (
            ctx.grid_shape_parentrun // 2
        ) * (1.0 / ctx.scale_parentrun)

        h, n = _atom_focus_signature(atoms_to_focus)
        print(f"[FOCUS DEBUG] atoms_to_focus len={n} hash={h}", flush=True)

        arr = np.asarray(atoms_data_focus)
        print(
            "[FOCUS DEBUG] before focused atoms_init "
            f"shape={arr.shape} dtype={arr.dtype}",
            flush=True,
        )

        # Re-summarize the focused source subset, not the FRC target atoms.
        ctx.atoms_init_and_summary(
            atoms_to_focus,
            objects=inp.objects,
            extremas_rule=inp.get_param_value("solute_extrema"),
            acenter=ctx.acenter,
            enforce_acenter=ctx.enforce_acenter,
            is_focusing=True,
        )
        _focus_atoms_data_debug(ctx, "after focused atoms_init")

        return

    # Non-focusing PBE: self is its own parent (Numba-friendly, no None).
    ctx.scale_parentrun = ctx.scale
    ctx.grid_center_parentrun = ctx.grid_center
    ctx.grid_shape_parentrun = ctx.grid_shape
    ctx.grid_origin_parentrun = ctx.grid_origin
    ctx.phimap_parentrun = np.zeros((3, 3, 3), dtype=delphi_real)  # dummy placeholder
