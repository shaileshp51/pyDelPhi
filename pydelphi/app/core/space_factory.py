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

from numpy import asarray as np_asarray

from pydelphi.foundation.enums import BoundaryCondition


def new_space_obj(space_module, platform, inp, ctx, verbosity: int):
    """
    Initialize and return a Space object using parameters from `inp` and `ctx`.

    Design rules:
      - Core does NOT import global_runtime (no vprint/logging).
      - Space is provided via `space_module` to avoid hard module coupling while
        keeping call-sites simple (App passes its already-imported module).

    Reads:
      - inp.get_param_value/get_param + inp.gridbox_center
      - ctx fields used below
      - platform

    Writes:
      - none

    Returns:
      - space_module.Space instance (mutable; executed later to populate/update ctx maps)
    """
    bc = inp.get_param_value("boundary_condition")
    is_focusing = bc.int_value == BoundaryCondition.FOCUSING.int_value

    # Parent-run fallback: “self is its own parent”
    grid_shape_parentrun = ctx.grid_shape_parentrun
    if grid_shape_parentrun is None:
        grid_shape_parentrun = np_asarray(ctx.grid_shape)

    grid_origin_parentrun = ctx.grid_origin_parentrun
    if grid_origin_parentrun is None:
        grid_origin_parentrun = np_asarray(ctx.grid_origin)

    num_objects = len(ctx.objects_data) // 2
    enabled_nonpolar_energy = inp.get_param("calculate_energies").is_attribute_inuse(
        "np"
    )

    return space_module.Space(
        platform=platform,
        is_surf_midpoints=inp.get_param_value("midpoint_dielectric_gaussian"),
        scale=ctx.scale,
        exdi=inp.get_param_value("exdi"),
        gapdi=inp.get_param_value("gapdi"),
        indi=inp.get_param_value("indi"),
        media_epsilon=ctx.media_epsilon,
        probe_radius=inp.get_param_value("probe_radius"),
        probe_radius2=inp.get_param_value("probe_radius2"),
        debye_length=ctx.debye_length,
        salt_radius=inp.get_param_value("ions_radii"),
        gaussian_sigma=inp.get_param_value("gaussian_sigma"),
        gaussian_exponent=inp.get_param_value("gaussian_exponent"),
        max_atom_radius=ctx.max_atom_radius,
        verbosity=verbosity,
        dielectric_model=inp.get_param("dielectric_model"),
        surface_method=inp.get_param_value("surface_method"),
        surface_density_exponent=inp.get_param_value("surface_density_exponent"),
        surface_offset=inp.get_param_value("surface_offset"),
        r_offset=inp.get_param_value("radius_offset"),
        grid_shape=ctx.grid_shape,
        grid_origin=ctx.grid_origin,
        atoms_data=ctx.atoms_data,
        objects_data=ctx.objects_data,
        is_focusing=is_focusing,
        grid_shape_parentrun=grid_shape_parentrun,
        grid_origin_parentrun=grid_origin_parentrun,
        acenter=ctx.grid_center,
        num_objects=num_objects,
        num_molecules=1,
        use_zeta_surf=inp.get_param_value("zeta_potential"),
        zeta_distance=inp.get_param_value("zeta_distance"),
        coords_by_axis_min=ctx.coords_by_axis_min,
        coords_by_axis_max=ctx.coords_by_axis_max,
        enabled_nonpolar_energy=enabled_nonpolar_energy,
    )
