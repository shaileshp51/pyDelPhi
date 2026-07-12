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

from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_Z,
    ATOMFIELD_CHARGE,
    ATOMFIELD_MEDIA_ID,
)


def write_frc_if_requested(
    inp: Any,
    ctx: Any,
    rdr: Any,
    total_iters: Any = None,
    final_rms: Any = None,
    final_dphi: Any = None,
    convergence_status: Any = None,
    frc_outfile: Any = None,
    frc_target_atoms: Any = None,
    delphi_real: Any = float,
) -> None:
    """
    Write .frc output if out_frc is supplied.

    Target resolution priority:
      1) frc_target_atoms argument (already in correct dict format)
      2) frc(target_file=...) / frc(tfile=...) if supplied
      3) Legacy fallback: in__frc(file=...)  [DEPRECATED: remove later]

    Notes:
      - Imports pydelphi.site.writesite lazily to avoid extra deps on runs that don't need it.
      - Assumes ctx fields are populated (phimap_in_water, dielectric maps, induced charges, etc.).
    """
    # --- decide whether we should write at all ---
    output_frc_file = frc_outfile
    if output_frc_file is None:
        prm_out_frc = inp.get_param("frc")
        if prm_out_frc is None or not prm_out_frc.issupplied:
            return
        output_frc_file = prm_out_frc.get_attribute("outfile")

    # --- resolve frc_atoms_dict (read_frc already returns correct dict format) ---
    atoms_frc = frc_target_atoms

    if atoms_frc is None:
        # Prefer modern frc(target_file=...) over legacy in__frc.
        prm_frc = inp.get_param("frc")
        if prm_frc is not None and prm_frc.issupplied:
            target_file = str(prm_frc.get_attribute("target_file") or "").strip()
            if not target_file:
                # Defensive: allow alias if an older path leaks through.
                target_file = str(prm_frc.get_attribute("tfile") or "").strip()

            if target_file:
                atoms_frc = rdr.read_frc(target_file)

        # Legacy fallback (DEPRECATED).
        if atoms_frc is None:
            prm_in_frc = inp.get_param("in__frc")
            if prm_in_frc is not None and prm_in_frc.issupplied:
                atoms_frc = rdr.read_frc(prm_in_frc.get_attribute("file"))

    if atoms_frc is None:
        raise ValueError(
            "out__frc requested but no FRC evaluation points were provided. "
            "Provide frc_target_atoms, or use frc(target_file=...)."
        )

    site_param = inp.get_param("site")

    def _site_inuse(attribute_name: str) -> bool:
        return (
            site_param is not None
            and site_param.issupplied
            and site_param.is_attribute_inuse(attribute_name)
        )

    import pydelphi.site.writesite as wrts

    wrts.write_frc_file(
        output_frc_file=output_frc_file,
        frc_atoms_dict=atoms_frc,
        grid_shape=ctx.grid_shape,
        percentage_fill=ctx.perfil,
        external_dielectric=ctx.external_dielectric_scaled * ctx.epkt,
        media_eps=ctx.media_epsilon,
        gap_dielectric=getattr(ctx, "gap_dielectric", None),
        dielectric_model=inp.get_param_value("dielectric_model"),
        surface_method=inp.get_param_value("surface_method"),
        epkt=ctx.epkt,
        ion_strength=inp.get_param_value("salt"),
        ion_radius=inp.get_param_value("ions_radii"),
        probe_radii=inp.get_param_value("probe_radius"),
        total_iters=total_iters,
        final_rms=final_rms,
        final_dphi=final_dphi,
        convergence_status=convergence_status,
        boundary_type=inp.get_param_value("boundary_condition"),
        file_map_record="frc map",
        potential_upper_bond=np.max(ctx.phimap_in_water),
        box_center=ctx.grid_center,
        grid_offset=np.zeros(3, dtype=delphi_real),
        scale_factor=ctx.scale,
        potential_map=ctx.phimap_in_water,
        dielectric_map_bool=ctx.dielectric_boundary_map_1d,
        surface_charge_pos_array=ctx.induced_surf_charge_positions,
        surface_charge_e_array=ctx.induced_surf_charges[::4],
        boundary_grid_array=ctx.dielectric_boundary_grids,
        charge_grid_num=ctx.charged_gridpoints_1d,
        charge_pos_array=ctx.atoms_data[:, ATOMFIELD_X : ATOMFIELD_Z + 1],
        atomic_charge_list=ctx.atoms_data[:, ATOMFIELD_CHARGE],
        atom_eps_array=ctx.atoms_data[:, ATOMFIELD_MEDIA_ID],
        residue_num=0,
        out_atom_desc=_site_inuse("atom"),
        out_atom_coords=_site_inuse("coordinates"),
        out_charge=_site_inuse("charge"),
        out_grid_pot=_site_inuse("potential"),
        out_field=_site_inuse("field"),
    )
