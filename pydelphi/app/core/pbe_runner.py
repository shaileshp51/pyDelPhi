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
from typing import Any, Callable, Tuple

from pydelphi.foundation.enums import (
    BioModel,
    MemoryState,
    SurfaceMethod,
    DielectricModel,
    PBApproximation,
)

from pydelphi.app.core.output_maps import write_spatial_maps, write_phase_dependent_maps
from pydelphi.app.core.maps_memory import reset_phase_dependent_maps
from pydelphi.app.core.cuda_utils import determine_cuda_thread_count
from pydelphi.app.core.space_factory import new_space_obj


def process_phase_pbe(
    vacuum: bool,
    n_threads: int,
    space_obj: Any,
    inp: Any,
    ctx: Any,
    platform: Any,
    verbosity: int,
    lvl_debug: int,
    lvl_info: int,
    lvl_trace: int,
    approx_zero: Any,
    PBESolverCtor: Callable[..., Any],
    timings: dict[str, Any],
) -> Tuple[float, float, int, str]:
    """Process PBE for a single phase (vacuum or water)."""

    phase_name = "vacuum" if vacuum else "water"

    # ---- epsilon maps ----
    tic_epscalc = perf_counter()

    has_dencut = inp.get_param("density_cutoff").issupplied
    has_srfcut = inp.get_param("surface_cutoff").issupplied

    gaussian_density_cutoff = 0.0
    gaussian_epsilon_cutoff = inp.get_param_value("indi")

    # Preserve your current logic as-is (even if it looks like a typo).
    if has_dencut or not (has_srfcut or has_srfcut):
        gaussian_density_cutoff = inp.get_param_value("density_cutoff")
    if has_srfcut:
        gaussian_epsilon_cutoff = inp.get_param_value("surface_cutoff")

    space_obj.calc_phase_spatial_epsilon_map_midpoints(
        is_regularized=False,
        vacuum=vacuum,
        gaussian_density_cutoff=gaussian_density_cutoff,
        gaussian_epsilon_cutoff=gaussian_epsilon_cutoff,
    )

    ctx.epsilon_map_1d = space_obj.epsilon_map_1d
    if vacuum:
        ctx.epsilon_map_midpoints_vacuum_1d = space_obj.epsilon_map_midpoints_1d
    else:
        ctx.epsilon_map_midpoints_water_1d = space_obj.epsilon_map_midpoints_1d

    timings.update(space_obj.timings)

    # Original: only write spatial maps in water phase
    if not vacuum:
        write_spatial_maps(inp=inp, ctx=ctx, verbosity=verbosity)

    if verbosity <= lvl_trace:
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

    # ---- PBE solver ----
    solver_name = inp.get_param_value("solver").name.lower()
    pbe_solver = PBESolverCtor(
        platform, verbosity, n_threads, solver_name, ctx.grid_shape
    )

    if verbosity <= lvl_info:
        print(f"\n    PBE> run is starting for solute in {phase_name} phase.")

    # Ion exclusion selection
    ion_exclusion_map_1d_args = ctx.surface_map_1d
    surface_method = inp.get_param_value("surface_method")

    if surface_method.int_value == SurfaceMethod.GAUSSIANCUTOFF.int_value:
        ion_exclusion_map_1d_args = ctx.ion_exclusion_map_1d
    elif surface_method.int_value == SurfaceMethod.VDW.int_value:
        ion_exclusion_map_1d_args = ctx.dielectric_boundary_map_1d == False

    # Nonlinear coupling steps
    non_zero_salt = inp.get_param_value("salt") != 0.0
    nonlinear_coupling_steps = 0
    pb_approximation = inp.get_param_value("pb_approximation")
    if pb_approximation.int_value == PBApproximation.NONLINEAR.int_value:
        nonlinear_coupling_steps = inp.get_param_value("nonlinear_coupling_steps")

    output_phimap_1d = pbe_solver.solve_pbe(
        vacuum=vacuum,
        bound_cond=inp.get_param_value("boundary_condition"),
        dielectric_model=inp.get_param_value("dielectric_model"),
        gaussian_exponent=inp.get_param_value("gaussian_exponent"),
        nonlinear_itr_block_size=inp.get_param_value("nonlinear_iteration_block_size"),
        itr_block_size=inp.get_param_value("iteration_block_size"),
        max_linear_iters=inp.get_param_value("linit"),
        max_nonlinear_iters=inp.get_param_value("nonlinit"),
        max_nonlinear_coupling_dphi=inp.get_param_value(
            "max_nonlinear_coupling_delta_phi"
        ),
        coupling_steps=nonlinear_coupling_steps,
        manual_relaxation_value=inp.get_param_value("nlrelpar"),
        scale=ctx.scale,
        scale_parentrun=ctx.scale_parentrun,
        exdi=inp.get_param_value("exdi"),
        indi=inp.get_param_value("indi"),
        debye_length=ctx.debye_length,
        non_zero_salt=non_zero_salt,
        total_pve_charge=ctx.positive_charge,
        total_nve_charge=ctx.negative_charge,
        max_rms=inp.get_param_value("max_rmsd"),
        max_dphi=inp.get_param_value("max_delta_phi"),
        check_dphi=inp.get_param("max_delta_phi").active,
        epkt=ctx.epkt,
        approx_zero=approx_zero,
        omega_adaptive=inp.get_param_value("nwt_adaptive_omega"),
        grid_shape=ctx.grid_shape,
        grid_origin=ctx.grid_origin,
        grid_shape_parentrun=ctx.grid_shape_parentrun,
        grid_origin_parentrun=ctx.grid_origin_parentrun,
        atoms_data=ctx.atoms_data,
        density_map_1d=ctx.gauss_density_map_1d,
        ion_exclusion_map_1d=ion_exclusion_map_1d_args,
        epsilon_map_1d=ctx.epsilon_map_1d,
        epsmap_midpoints_1d=(
            ctx.epsilon_map_midpoints_vacuum_1d
            if vacuum
            else ctx.epsilon_map_midpoints_water_1d
        ),
        centroid_pve_charge=ctx.centroid_positive_charge,
        centroid_nve_charge=ctx.centroid_negative_charge,
        charged_gridpoints_1d=ctx.charged_gridpoints_1d,
        phimap_parentrun=ctx.phimap_parentrun,
    )

    # Solver returns 1D, reshape for downstream
    output_phimap_3d = output_phimap_1d.reshape(ctx.grid_shape)

    timings.update(pbe_solver.timings)

    if vacuum:
        ctx.phimap_in_vacuum = output_phimap_3d
    else:
        ctx.phimap_in_water = output_phimap_3d

        # Preserve: update ctx ion exclusion for Gaussian salt methods
        if surface_method.int_value in {
            SurfaceMethod.GAUSSIAN.int_value,
            SurfaceMethod.GCS.int_value,
        }:
            ctx.ion_exclusion_map_1d = 1.0 - ion_exclusion_map_1d_args

    toc_pb_phase = perf_counter()
    timings[f"Solving PBE in {phase_name}"] = f"{toc_pb_phase - toc_epscalc:0.3f}"

    return (
        pbe_solver.final_rms,
        pbe_solver.final_dphi,
        pbe_solver.total_iters,
        pbe_solver.convergence_status,
    )


def run_pbe(
    inp: Any,
    ctx: Any,
    platform: Any,
    space_module: Any,
    verbosity: int,
    lvl_debug: int,
    lvl_info: int,
    lvl_trace: int,
    approx_zero: float,
    timings: dict[str, Any],
    erg_settings: Any,
    calculate_all_energies: Callable[..., None],
    PBESolverCtor: Callable[..., Any],
) -> Tuple[float, float, int, str]:
    """Orchestrate PBE run: (vacuum if applicable) then water."""
    if inp.get_param_value("biomodel").int_value != BioModel.PBE.int_value:
        return 0, 0, 0, ""

    n_threads = determine_cuda_thread_count(ctx.grid_shape)

    tic_space_init = perf_counter()
    space_obj = new_space_obj(
        space_module=space_module,
        platform=platform,
        inp=inp,
        ctx=ctx,
        verbosity=verbosity,
    )
    toc_space_init = perf_counter()
    timings["space initialization"] = f"{toc_space_init - tic_space_init:0.3f}"

    space_obj.run(n_threads, ctx)
    space_obj.update_runtime_context(ctx=ctx)

    toc_space_run = perf_counter()
    timings["space running"] = f"{toc_space_run - toc_space_init:0.3f}"

    dielectrc_model_value = inp.get_param_value("dielectric_model")
    if dielectrc_model_value.int_value == DielectricModel.TWODIELECTRIC.int_value:
        ctx.induced_surf_charge_positions = space_obj.induced_surf_charge_positions
        ctx.dielectric_boundary_grids = space_obj.dielectric_boundary_grids

    # ---- vacuum phase (skipped for TWODIELECTRIC) ----
    if dielectrc_model_value.int_value != DielectricModel.TWODIELECTRIC.int_value:
        tic_pb_vacuum = perf_counter()
        (final_rms, final_dphi, total_iters, convergence_status) = process_phase_pbe(
            vacuum=True,
            n_threads=n_threads,
            space_obj=space_obj,
            inp=inp,
            ctx=ctx,
            platform=platform,
            verbosity=verbosity,
            lvl_debug=lvl_debug,
            lvl_info=lvl_info,
            lvl_trace=lvl_trace,
            approx_zero=approx_zero,
            timings=timings,
            PBESolverCtor=PBESolverCtor,
        )
        toc_pb_vacuum = perf_counter()
        timings["Solving PBE in vacuum"] = f"{toc_pb_vacuum - tic_pb_vacuum:0.3f}"

        calculate_all_energies(
            vacuum=True, final=False, ctx=ctx, erg_settings=erg_settings
        )

        # Preserve original: write vacuum maps + optional reset
        write_phase_dependent_maps(inp=inp, ctx=ctx, verbosity=verbosity, isvacuum=True)
        if (
            inp.get_param_value("memory_state").int_value
            == MemoryState.MINIMAL.int_value
        ):
            reset_phase_dependent_maps(ctx=ctx, vacuum=True)

    # ---- water phase ----
    tic_pb_water = perf_counter()
    (final_rms, final_dphi, total_iters, convergence_status) = process_phase_pbe(
        vacuum=False,
        n_threads=n_threads,
        space_obj=space_obj,
        inp=inp,
        ctx=ctx,
        platform=platform,
        verbosity=verbosity,
        lvl_debug=lvl_debug,
        lvl_info=lvl_info,
        lvl_trace=lvl_trace,
        approx_zero=approx_zero,
        timings=timings,
        PBESolverCtor=PBESolverCtor,
    )
    toc_pb_water = perf_counter()
    timings["Solving PBE in water"] = f"{toc_pb_water - tic_pb_water:0.3f}"

    calculate_all_energies(vacuum=False, final=True, ctx=ctx, erg_settings=erg_settings)

    # Preserve original: write water maps + optional reset + timing accumulation
    tic_pb_out_maps = perf_counter()
    write_phase_dependent_maps(inp=inp, ctx=ctx, verbosity=verbosity, isvacuum=False)
    if inp.get_param_value("memory_state").int_value == MemoryState.MINIMAL.int_value:
        reset_phase_dependent_maps(ctx=ctx, vacuum=False)
    toc_pb_out_maps = perf_counter()

    timings["Writing solvent-phase output maps"] = (
        f"{toc_pb_out_maps - tic_pb_out_maps:0.3f}"
    )

    return final_rms, final_dphi, total_iters, convergence_status
