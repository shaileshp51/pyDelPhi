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

#
# pyDelPhi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# pyDelPhi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#

#
# PyDelphi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# PyDelphi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#


"""
This module provides the core functionality for calculating various types of energies
within the Delphi framework. It orchestrates calls to specialized energy calculation
sub-modules, manages the distinction between phase-dependent and phase-independent
calculations, and handles the finalization of total energy results.

The main entry point is `calculate_all_energies`, which iteratively computes
coulombic, Lennard-Jones, nonpolar, reaction field, and grid energies,
potentially across vacuum and solvent phases, and accounts for non-linear PB terms.
It utilizes a `RuntimeContext` to access necessary simulation data and
stores results in an `EnergyResults` object.
"""
MODULE_NAME = __name__
from time import perf_counter

from numba import set_num_threads

from pydelphi.config.global_runtime import vprint
from pydelphi.config.logging_config import WARNING, DEBUG, get_effective_verbosity

_VERBOSITY = get_effective_verbosity(MODULE_NAME)

# Import specific functions
from pydelphi.energy.coulombic import calc_coulombic_energy
from pydelphi.energy.gridenergy import calc_grid_energy
from pydelphi.energy.lj import calc_lj_energy
from pydelphi.energy.nonlinear import energy_nonlinear
from pydelphi.energy.nonpolar import calc_nonpolar_energy
from pydelphi.energy.reactionfield_induced_surf_v2 import (
    calc_reactionfield_energy,
    calc_induced_charge_rf_energy,
)

from pydelphi.foundation.enums import PBApproximation, DielectricModel, SurfaceMethod
from pydelphi.foundation.context import RuntimeContext
from pydelphi.energy.energy_models import EnergySettings, EnergyResults


# --- Helper for Phase-Independent Calculations ---
def _calculate_phase_independent_energies(
    ctx: RuntimeContext,
    erg_settings: EnergySettings,
):
    """
    Performs energy calculations that are independent of the solvent phase.
    These typically use properties available regardless of the active phase in ctx.
    """

    vprint(DEBUG, _VERBOSITY, "Calculating phase-independent energies...")
    results = ctx.energy_results

    # --- Coulombic Energy (Atom) ---
    if erg_settings.calculate_coulombic_energy:
        tic_coul = perf_counter()
        energy_coul_atom = calc_coulombic_energy(
            erg_settings.platform,
            ctx.atoms_data,
            erg_settings.indi,
            ctx.epkt,
        )
        results.add_energy("phase_independent", "coulombic_energy", energy_coul_atom)
        toc_coul = perf_counter()
        results.add_timing("phase_independent", "coulombic_energy", toc_coul - tic_coul)

    # --- LJ Energy ---
    if erg_settings.calculate_lj:
        tic_lj = perf_counter()
        lj_energy = calc_lj_energy(
            erg_settings.platform,
            ctx.atoms_data,
            ctx.temperature,
        )
        results.add_energy("phase_independent", "LJ_energy", lj_energy)
        toc_lj = perf_counter()
        results.add_timing("phase_independent", "LJ_energy", toc_lj - tic_lj)

    # --- Nonpolar Energy (if truly phase-independent in calculation) ---
    if erg_settings.calculate_nonpolar:
        tic_nonpolar = perf_counter()
        nonpolar_energy = calc_nonpolar_energy(
            # ctx.atoms_data,
            # ctx.surface_area,  # Assuming ctx has pre-computed surface area
            # erg_settings.surface_tension,  # From config
        )
        results.add_energy("phase_independent", "nonpolar_energy", nonpolar_energy)
        toc_nonpolar = perf_counter()
        results.add_timing(
            "phase_independent", "nonpolar_energy", toc_nonpolar - tic_nonpolar
        )


# --- Helper for Phase-Dependent Calculations ---
def _calculate_for_phase(
    phase_name: str,
    ctx: RuntimeContext,
    erg_settings: EnergySettings,
):
    """
    Performs all energy calculations for a single phase (vacuum or water)
    using the given RuntimeContext, which is assumed to be appropriately
    configured for or to provide access to data for this specific phase.
    """
    is_vacuum = phase_name == "vacuum"
    results = ctx.energy_results

    # --- Grid Energy ---
    if erg_settings.calculate_grid_energy:
        tic_grid = perf_counter()
        phimap_for_grid = ctx.phimap_in_vacuum if is_vacuum else ctx.phimap_in_water
        energy_grid_phase = calc_grid_energy(
            ctx.charged_gridpoints_1d,
            phimap_for_grid,
        )
        results.add_energy(phase_name, "grid_energy", energy_grid_phase)
        toc_grid = perf_counter()
        results.add_timing(phase_name, "grid_energy", toc_grid - tic_grid)

    # --- Reaction Field Energy ---
    if erg_settings.calculate_reactionfield:
        tic_rf = perf_counter()
        energy_rxn_phase = 0.0
        induced_surf_charges = None

        if (
            erg_settings.dielectric_model.int_value
            == DielectricModel.TWODIELECTRIC.int_value
            and (not is_vacuum)
        ):
            # Induced surface charge method of reaction field energy requires solution of PBE in only water phase
            phimap_for_induced = ctx.phimap_in_water
            energy_rxn_phase, induced_surf_charges = calc_induced_charge_rf_energy(
                erg_settings.platform,
                ctx.charged_gridpoints_1d,
                ctx.dielectric_boundary_grids,
                ctx.dielectric_boundary_map_1d,
                ctx.atoms_data,
                ctx.induced_surf_charge_positions,
                phimap_for_induced,
                ctx.grid_shape,
                ctx.scale,
                ctx.internal_dielectric_scaled,
                ctx.debye_length,
                ctx.epkt,
            )
            results.add_intermediate_data(
                phase_name, "induced_surf_charges", induced_surf_charges
            )
            ctx.induced_surf_charges = induced_surf_charges
        else:
            # Note ctx must provide phimap based on the active phases and specific attributes
            phimap_for_rf = ctx.phimap_in_vacuum if is_vacuum else ctx.phimap_in_water
            energy_rxn_phase = calc_reactionfield_energy(
                ctx.charged_gridpoints_1d,
                phimap_for_rf,
            )
        results.add_energy(phase_name, "reactionfield_energy", energy_rxn_phase)
        toc_rf = perf_counter()
        results.add_timing(phase_name, "reactionfield_energy", toc_rf - tic_rf)

    # --- Non-linear PB Approximation (if applicable and for water phase) ---
    if erg_settings.calculate_nonlinear_pb_terms:
        if (
            erg_settings.pb_approximation.int_value
            == PBApproximation.NONLINEAR.int_value
            and phase_name == "water"  # Nonlinear energy typically applies to solvent
        ):
            tic_nonlinear = perf_counter()
            ion_exclusion_map_1d_args = ctx.surface_map_1d

            if erg_settings.surface_method.int_value == SurfaceMethod.VDW.int_value:
                ion_exclusion_map_1d_args = ctx.dielectric_boundary_map_1d == True
            elif (
                erg_settings.surface_method.int_value
                == SurfaceMethod.GAUSSIANCUTOFF.int_value
            ):
                ion_exclusion_map_1d_args = 1.0 - ctx.ion_exclusion_map_1d
            elif erg_settings.surface_method.int_value in {
                SurfaceMethod.GCS.int_value,
                SurfaceMethod.GAUSSIAN.int_value,
            }:
                ion_exclusion_map_1d_args = 1.0 - ctx.ion_exclusion_map_1d

            set_num_threads(erg_settings.platform.names["cpu"]["num_threads"])
            energy_rho_phi_water, energy_osmotic_water = energy_nonlinear(
                grid_shape=ctx.grid_shape,
                scale=ctx.scale,
                grid_origin=ctx.grid_origin,
                ion_exclusion_map_1d=ion_exclusion_map_1d_args,
                phimap=ctx.phimap_in_water,  # This must be the water phimap
                taylor_coeffs=ctx.taylor_coefficients,
                nthreads=erg_settings.platform.names["cpu"]["num_threads"],
            )
            _note = f"""Finally, negate the signs of energy_rho_times_phi_over_2 and energy_osmotic to absorb the negative sign in
            Equation (4) of \\cite{'Rocchia2001jpcb'} for consistency, note afterwards it will be added to reaction-field
            energy (NOT SUBTRACTED).
            """
            energy_rho_phi_water *= -1
            energy_osmotic_water *= -1

            results.add_energy(
                phase_name, "electrostatic_stress_term", energy_rho_phi_water
            )
            results.add_energy(
                phase_name, "osmotic_pressure_term", energy_osmotic_water
            )
            toc_nonlinear = perf_counter()
            results.add_timing(
                phase_name, "nonlinear_pb_terms", toc_nonlinear - tic_nonlinear
            )
    results.set_phase_calculated(phase_name)


# --- Main Entry Point ---
def calculate_all_energies(
    vacuum: bool,
    final: bool,
    ctx: RuntimeContext,
    erg_settings: EnergySettings,
) -> EnergyResults:
    overall_tic_current_call = perf_counter()

    results = ctx.energy_results

    only_water_phase_needed = False
    if (
        erg_settings.dielectric_model.int_value
        == DielectricModel.TWODIELECTRIC.int_value
        and erg_settings.surface_method.int_value == SurfaceMethod.VDW.int_value
    ):
        only_water_phase_needed = True

    if not final:
        results.set_finalized(False)

    if not results.energies["phase_independent"]:
        _calculate_phase_independent_energies(ctx, erg_settings)

    if vacuum:
        if not results.get_energy("vacuum", "reactionfield_energy"):
            _calculate_for_phase("vacuum", ctx, erg_settings)
        else:
            vprint(
                DEBUG,
                _VERBOSITY,
                "Vacuum phase already calculated in this results object. Skipping.",
            )
    else:  # Implies water phase
        if not results.get_energy("water", "reactionfield_energy"):
            _calculate_for_phase("water", ctx, erg_settings)
        else:
            vprint(
                DEBUG,
                _VERBOSITY,
                "Water phase already calculated in this results object. Skipping.",
            )

    # Record the time taken by this specific call to calculate_all_energies
    time_taken_this_call = perf_counter() - overall_tic_current_call
    results.add_to_overall_timing("total_calculation_time", time_taken_this_call)

    # --- Finalization steps, only if final=True ---
    if final:
        vprint(DEBUG, _VERBOSITY, "Finalizing results: Calculating total energies...")
        vacuum_rf = results.get_energy("vacuum", "reactionfield_energy")
        water_rf = results.get_energy("water", "reactionfield_energy")

        if vacuum_rf is not None and water_rf is not None:
            total_reactionfield_energy = water_rf - vacuum_rf
            results.add_energy(
                "total", "corrected_reaction_field_energy", total_reactionfield_energy
            )
        elif only_water_phase_needed and vacuum_rf is None and water_rf is not None:
            total_reactionfield_energy = water_rf - 0.0
            results.add_energy(
                "total", "corrected_reaction_field_energy", total_reactionfield_energy
            )
        elif only_water_phase_needed and water_rf is None:
            vprint(
                WARNING,
                _VERBOSITY,
                "Warning: Reaction field solvation energy for two-dielectric requires water phases to finalize.",
            )
        elif not only_water_phase_needed:
            vprint(
                WARNING,
                _VERBOSITY,
                "Warning: Reaction field solvation energy requires both vacuum and water phases to finalize.",
            )

        vacuum_grid = results.get_energy("vacuum", "grid_energy")
        water_grid = results.get_energy("water", "grid_energy")
        if vacuum_grid is not None and water_grid is not None:
            total_grid_energy = water_grid - vacuum_grid
            results.add_energy("total", "total_grid_energy", total_grid_energy)
        elif only_water_phase_needed and water_grid is not None and vacuum_grid is None:
            total_grid_energy = water_grid - 0.0
            results.add_energy("total", "total_grid_energy", total_grid_energy)
        elif only_water_phase_needed and water_grid is None:
            vprint(
                WARNING,
                _VERBOSITY,
                "Warning: Total grid energy for two-dielectric requires water phases to finalize.",
            )
        else:
            vprint(
                WARNING,
                _VERBOSITY,
                "Warning: Total grid energy requires both vacuum and water phases to finalize.",
            )

        if (
            erg_settings.pb_approximation.int_value
            == PBApproximation.NONLINEAR.int_value
        ):
            osmotic_pressure_energy = (
                results.get_energy("water", "osmotic_pressure_term")
                if "osmotic_pressure_term" in results.energies["water"]
                else 0.0
            )
            electrostatic_stress_energy = (
                results.get_energy("water", "electrostatic_stress_term")
                if "electrostatic_stress_term" in results.energies["water"]
                else 0.0
            )
            if vacuum_grid is not None and water_grid is not None:
                total_nonlinear_grid_energy = (
                    (water_grid - vacuum_grid)
                    + osmotic_pressure_energy
                    + electrostatic_stress_energy
                )
                results.add_energy(
                    "total", "total_nonlinear_grid_energy", total_nonlinear_grid_energy
                )
            elif (
                only_water_phase_needed
                and water_grid is not None
                and vacuum_grid is None
            ):
                total_nonlinear_grid_energy = (
                    (water_grid - 0.0)
                    + osmotic_pressure_energy
                    + electrostatic_stress_energy
                )
                results.add_energy(
                    "total", "total_nonlinear_grid_energy", total_nonlinear_grid_energy
                )
                results.add_energy(
                    "total",
                    "corrected_reaction_field_energy",
                    results.get_energy("total", "corrected_reaction_field_energy")
                    + osmotic_pressure_energy
                    + electrostatic_stress_energy,
                )

            elif only_water_phase_needed and water_grid is None:
                vprint(
                    WARNING,
                    _VERBOSITY,
                    "Warning: Total non-linear grid energy for two-dielectric requires water phases to finalize.",
                )
            else:
                vprint(
                    WARNING,
                    _VERBOSITY,
                    "Warning: Nonlinear total grid energy requires both vacuum/water grid energies and osmotic term to finalize.",
                )

        if (
            "corrected_reaction_field_energy" in results.energies["total"]
            and "nonpolar_energy" in results.energies["phase_independent"]
        ):
            overall_solvation_energy = results.get_energy(
                "total", "corrected_reaction_field_energy"
            ) + results.get_energy("phase_independent", "nonpolar_energy")
            results.add_energy(
                "total", "overall_solvation_energy", overall_solvation_energy
            )
        elif erg_settings.calculate_reactionfield and erg_settings.calculate_nonpolar:
            vprint(
                WARNING,
                _VERBOSITY,
                "Warning: Overall solvation energy calculation might be incomplete due to missing components needed for finalization.",
            )

        results.set_finalized(True)
    else:
        results.set_finalized(False)  # Explicitly set false for non-final runs

    return results
