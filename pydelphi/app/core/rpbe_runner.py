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

from time import perf_counter
from typing import Any, Callable

from pydelphi.foundation.enums import (
    BioModel,
    MemoryState,
    BoundaryCondition,
    IonExclusionRegion,
    SurfaceMethod,
)


from pydelphi.app.core.output_maps import write_spatial_maps, write_phase_dependent_maps
from pydelphi.app.core.maps_memory import reset_phase_dependent_maps
from pydelphi.app.core.cuda_utils import determine_cuda_thread_count
from pydelphi.app.core.space_factory import new_space_obj


def process_phase_rpbe(
    *,
    vacuum: bool,
    n_threads: int,
    space_obj: Any,
    inp: Any,
    ctx: Any,
    platform: Any,
    verbosity: int,
    lvl_debug: int,
    lvl_info: int,
    approx_zero: Any,
    RPBESolverCtor: Callable[..., Any],
    timings: dict[str, Any],
) -> None:
    """Process RPBE for a single phase (vacuum or water)."""

    phase_name = "vacuum" if vacuum else "water"

    # ---- epsilon maps ----
    tic_epscalc = perf_counter()
    space_obj.calc_phase_spatial_epsilon_map_midpoints(
        is_regularized=True, vacuum=vacuum
    )

    ctx.epsilon_map_1d = space_obj.epsilon_map_1d
    if vacuum:
        ctx.epsilon_map_midpoints_vacuum_1d = space_obj.epsilon_map_midpoints_1d
    else:
        ctx.epsilon_map_midpoints_water_1d = space_obj.epsilon_map_midpoints_1d

    timings.update(space_obj.timings)

    # Writes density/surface maps (same as your code)
    write_spatial_maps(inp=inp, ctx=ctx, verbosity=verbosity)

    if verbosity <= lvl_debug:
        eps_map_name = (
            "self.dc.epsilon_map_midpoints_vacuum_1d"
            if vacuum
            else "self.dc.epsilon_map_midpoints_water_1d"
        )
        eps_map = (
            ctx.epsilon_map_midpoints_vacuum_1d
            if vacuum
            else ctx.epsilon_map_midpoints_water_1d
        )
        if ctx.epsilon_map_1d is None:
            print("self.dc.epsilon_map_1d=", None)
        else:
            print("self.dc.epsilon_map_1d.shape=", ctx.epsilon_map_1d.shape)
        if eps_map is None:
            print(f"{eps_map_name}=", None)
        else:
            print(f"{eps_map_name}.shape=", eps_map.shape)

    toc_epscalc = perf_counter()
    timings[f"Calculating epsilon map in {phase_name}"] = (
        f"{toc_epscalc - tic_epscalc:0.3f}"
    )

    # ---- RPBE solver ----
    rpbe_solver = RPBESolverCtor(
        platform,
        verbosity,
        n_threads,
        grid_shape=ctx.grid_shape,
        coords_by_axis_min=ctx.coords_by_axis_min,
        coords_by_axis_max=ctx.coords_by_axis_max,
        num_objects=ctx.num_objects,
        num_molecules=1,
        coulomb_map_1d=ctx.coulomb_map_1d,
        grad_coulomb_map_1d=ctx.grad_coulomb_map_1d,
    )

    if verbosity <= lvl_info:
        print(f"\n    RPBE> run is starting for solute in {phase_name} phase.")

    non_zero_salt = inp.get_param_value("salt") != 0.0

    ion_exclusion_map_1d_args = ctx.surface_map_1d
    surface_method = inp.get_param_value("surface_method")
    if surface_method.int_value == SurfaceMethod.GAUSSIANCUTOFF.int_value:
        ion_exclusion_map_1d_args = ctx.ion_exclusion_map_1d

    output_phimap = rpbe_solver.run(
        vacuum=vacuum,
        non_zero_salt=non_zero_salt,
        bound_cond=BoundaryCondition.COULOMBIC,
        ion_exclusion_method=IonExclusionRegion.SOLUTESURFACE,
        gaussian_exponent=inp.get_param_value("gaussian_exponent"),
        itr_block_size=inp.get_param_value("iteration_block_size"),
        max_linear_iters=inp.get_param_value("linit"),
        scale=ctx.scale,
        exdi=inp.get_param_value("exdi"),
        gapdi=inp.get_param_value("gapdi"),
        indi=inp.get_param_value("indi"),
        probe_radius=inp.get_param_value("probe_radius"),
        salt_radius=inp.get_param_value("ions_radii"),
        debye_length=ctx.debye_length,
        total_pve_charge=ctx.positive_charge,
        total_nve_charge=ctx.negative_charge,
        rms_tol=inp.get_param_value("max_rmsd"),
        dphi_tol=inp.get_param_value("max_delta_phi"),
        check_dphi=inp.get_param("max_delta_phi").active,
        epkt=ctx.epkt,
        approx_zero=approx_zero,
        grid_shape=ctx.grid_shape,
        grid_origin=ctx.grid_origin,
        atoms_data=ctx.atoms_data,
        density_map_1d=ctx.gauss_density_map_1d,
        solute_surface_map_1d=ctx.surface_map_1d,
        ion_exclusion_map_1d=ion_exclusion_map_1d_args,
        epsilon_map_1d=ctx.epsilon_map_1d,
        epsmap_midpoints_1d=(
            ctx.epsilon_map_midpoints_vacuum_1d
            if vacuum
            else ctx.epsilon_map_midpoints_water_1d
        ),
        centroid_pve_charge=ctx.centroid_positive_charge,
        centroid_nve_charge=ctx.centroid_negative_charge,
        grad_surface_map_1d=ctx.grad_surface_map_1d,
    )

    timings.update(rpbe_solver.timings)

    if vacuum:
        ctx.phimap_in_vacuum = output_phimap
        ctx.coulomb_map_1d = rpbe_solver.coulomb_map_1d
        ctx.grad_coulomb_map_1d = rpbe_solver.grad_coulomb_map_1d
    else:
        ctx.phimap_in_water = output_phimap

    # NOTE: preserved exactly from your snippet (even for water phase)
    ctx.grad_epsgauss_map_vacuum_1d = rpbe_solver.grad_epsin_map_1d
    ctx.grad_epsilon_map_vacuum_1d = rpbe_solver.grad_epsmap_1d
    ctx.grad_eps_dot_gad_coul_vacuum_1d = rpbe_solver.eps_dot_coul_map_1d

    toc_rpb_phase = perf_counter()
    timings[f"Solving RPBE in {phase_name}"] = f"{toc_rpb_phase - toc_epscalc:0.3f}"

    # Writes epsilon/phi/zeta-phi for the phase
    write_phase_dependent_maps(inp=inp, ctx=ctx, verbosity=verbosity, isvacuum=vacuum)

    if inp.get_param_value("memory_state").int_value == MemoryState.MINIMAL.int_value:
        reset_phase_dependent_maps(ctx=ctx, vacuum=vacuum)


def run_rpbe(
    *,
    inp: Any,
    ctx: Any,
    platform: Any,
    space_module: Any,
    verbosity: int,
    lvl_debug: int,
    lvl_info: int,
    approx_zero: float,
    timings: dict[str, Any],
    erg_settings: Any,
    calculate_all_energies: Callable[..., None],
    RPBESolverCtor: Callable[..., Any],
    extra_final_water_write: bool = False,
) -> None:
    """
    Orchestrate RPBE run: vacuum phase then water phase.

    `extra_final_water_write` exists only if you want to preserve the legacy
    duplicated final water write/reset behavior from older app code.
    Recommended: keep False.
    """
    if inp.get_param_value("biomodel").int_value != BioModel.RPBE.int_value:
        return

    n_threads = determine_cuda_thread_count(ctx.grid_shape)

    space_obj = new_space_obj(
        space_module=space_module,
        platform=platform,
        inp=inp,
        ctx=ctx,
        verbosity=verbosity,
    )
    space_obj.run(n_threads, ctx)
    space_obj.update_runtime_context(ctx=ctx)

    # ---- vacuum phase ----
    process_phase_rpbe(
        vacuum=True,
        n_threads=n_threads,
        space_obj=space_obj,
        inp=inp,
        ctx=ctx,
        platform=platform,
        verbosity=verbosity,
        lvl_debug=lvl_debug,
        lvl_info=lvl_info,
        approx_zero=approx_zero,
        timings=timings,
        RPBESolverCtor=RPBESolverCtor,
    )
    calculate_all_energies(vacuum=True, final=False, ctx=ctx, erg_settings=erg_settings)

    # ---- water phase ----
    process_phase_rpbe(
        vacuum=False,
        n_threads=n_threads,
        space_obj=space_obj,
        inp=inp,
        ctx=ctx,
        platform=platform,
        verbosity=verbosity,
        lvl_debug=lvl_debug,
        lvl_info=lvl_info,
        approx_zero=approx_zero,
        timings=timings,
        RPBESolverCtor=RPBESolverCtor,
    )
    calculate_all_energies(vacuum=False, final=True, ctx=ctx, erg_settings=erg_settings)

    # Optional legacy duplicate write/reset (off by default)
    if extra_final_water_write:
        write_phase_dependent_maps(
            inp=inp, ctx=ctx, verbosity=verbosity, isvacuum=False
        )
        if (
            inp.get_param_value("memory_state").int_value
            == MemoryState.MINIMAL.int_value
        ):
            reset_phase_dependent_maps(ctx=ctx, vacuum=False)
