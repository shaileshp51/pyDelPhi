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
This module defines data structures for configuring and storing results of energy calculations.

It includes:
- EnergySettings: A dataclass for specifying various parameters for energy calculations,
  such as approximation methods, dielectric models, and components to calculate (e.g., coulombic, LJ, nonpolar).
  Instances of EnergySettings are mutable until the `freeze()` method is called, after which they become immutable
  to ensure configuration stability during a calculation.
- EnergyResults: A dataclass for storing the outcomes of energy calculations, including
  energy values for different phases (vacuum, water, phase-independent),
  timing information for various computational steps, and intermediate data.
  It also provides methods for adding and retrieving results, and generating formatted reports.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json  # Used for a nicer default __repr__

from pydelphi.foundation.platforms import Platform
from pydelphi.foundation.enums import (
    PBApproximation,
    DielectricModel,
    SurfaceMethod,
)

from pydelphi.config.global_runtime import vprint
from pydelphi.config.logging_config import DEBUG, get_effective_verbosity

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)


@dataclass
class EnergySettings:
    """
    Configuration settings for energy calculations.
    Instances are mutable until the 'freeze()' method is called, after which they become immutable.
    """

    platform: Platform = None

    # Default PB-approximation, dielectric, and surface model initialization must be overridden by user inputs
    pb_approximation: PBApproximation = PBApproximation.LINEAR
    dielectric_model: DielectricModel = DielectricModel.TWODIELECTRIC
    surface_method: SurfaceMethod = SurfaceMethod.VDW

    calculate_nonlinear_pb_terms: bool = False

    # General Energy Components
    calculate_reactionfield: bool = True
    calculate_nonlinear_pb_terms: bool = (
        False  # Only applies if PBApproximation is NONLINEAR
    )
    calculate_grid_energy: bool = True
    # Phase independent components
    calculate_coulombic_energy: bool = True
    # Optional: must be set to True before call when needed.
    calculate_lj: bool = False
    calculate_nonpolar: bool = False

    # Parameters for specific energy components
    indi: float = 1.0  # Internal dielectric constant
    lj_cutoff: float = 10.0  # Cutoff distance for Lennard-Jones (example)
    surface_tension: float = (
        0.0  # For nonpolar energy (example, units depend on formula)
    )

    # Output / Reporting
    write_phimaps: bool = False
    write_induced_charges: bool = False
    output_directory: Optional[str] = None

    # Internal flag to manage mutability
    _frozen: bool = field(
        init=False, default=False, repr=False
    )  # Not part of init, not shown in repr

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Intercepts attribute assignments to prevent modification after freezing.
        """
        if hasattr(self, "_frozen") and self._frozen:
            raise TypeError(
                f"Cannot modify attribute '{name}' of a frozen EnergyCalculationConfig instance."
            )
        super().__setattr__(name, value)

    def freeze(self) -> None:
        """
        Freezes the configuration, preventing any further modifications to its attributes.
        This should be called once the inputs are finalized.
        """
        # Perform final validation before freezing
        self._validate_for_freeze()

        # Set the _frozen flag using object.__setattr__ to bypass our custom __setattr__
        object.__setattr__(self, "_frozen", True)

        vprint(
            DEBUG,
            _VERBOSITY,
            "EnergyCalculationConfig instance frozen. No further modifications allowed.",
        )

    def _validate_for_freeze(self):
        """
        Internal method for validation checks performed right before freezing.
        This allows for more complex validation that might depend on multiple
        settings being finalized.
        """
        if self.pb_approximation == PBApproximation.NONLINEAR:
            # If nonlinear PB is chosen, ensure nonlinear terms calculation is enabled.
            object.__setattr__(self, "calculate_nonlinear_pb_terms", True)

        if self.calculate_lj and self.lj_cutoff <= 0:
            raise ValueError("LJ cutoff must be positive if LJ energy is calculated.")

        if (
            self.calculate_nonlinear_pb_terms
            and self.pb_approximation != PBApproximation.NONLINEAR
        ):
            raise ValueError(
                "Nonlinear PB terms calculation requested but PB Approximation is not set to NONLINEAR. "
                "Please correct the configuration."
            )
        # Add any other complex inter-parameter validations here.

    def clone(self) -> "EnergySettings":
        """
        Creates a mutable clone of the current configuration.
        Useful for making variations from a base configuration.
        """
        # Uses self.__class__ for flexibility if subclassing
        cloned_config = self.__class__(
            **{f.name: getattr(self, f.name) for f in field(self)}
        )
        # The _frozen flag should be False in the clone by default due to dataclass init behavior
        return cloned_config


@dataclass
class EnergyResults:
    """
    Stores and manages energy calculation results and timings across different phases.
    """

    energies: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "phase_independent": {},
            "vacuum": {},
            "water": {},
            "total": {},
        }
    )
    timings: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "phase_independent": {},
            "vacuum": {},
            "water": {},
            "overall": {
                "total_calculation_time": 0.0
            },  # Initialize overall time to 0.0
        }
    )
    intermediate_data: Dict[str, Dict[str, Any]] = field(
        default_factory=lambda: {"vacuum": {}, "water": {}}
    )
    calculated_phases: Dict[str, bool] = field(
        default_factory=lambda: {"vacuum": False, "water": False}
    )
    finalized: bool = False  # New: Tracks if final calculations have been run

    def add_energy(self, phase: str, component: str, value: float):
        """Adds an energy component for a specific phase."""
        if phase not in self.energies:
            self.energies[phase] = {}
        self.energies[phase][component] = value

    def get_energy(self, phase: str, component: str) -> Optional[float]:
        """Retrieves an energy component for a specific phase."""
        return self.energies.get(phase, {}).get(component)

    def add_timing(self, phase: str, component: str, value: float):
        """Adds a timing for a specific phase and component."""
        if phase not in self.timings:
            self.timings[phase] = {}
        self.timings[phase][component] = value

    def add_to_overall_timing(self, component: str, value: float):
        """Adds value to an existing overall timing component."""
        if "overall" not in self.timings:
            self.timings["overall"] = {}
        self.timings["overall"][component] = (
            self.timings["overall"].get(component, 0.0) + value
        )

    def get_timing(self, phase: str, component: str) -> Optional[float]:
        """Retrieves a timing for a specific phase and component."""
        return self.timings.get(phase, {}).get(component)

    def add_intermediate_data(self, phase: str, key: str, data: Any):
        """Stores intermediate data, typically for debugging or detailed analysis."""
        if phase not in self.intermediate_data:
            self.intermediate_data[phase] = {}
        self.intermediate_data[phase][key] = data

    def set_phase_calculated(self, phase: str):
        """Marks a specific phase as having completed its calculations."""
        if phase in self.calculated_phases:
            self.calculated_phases[phase] = True

    def is_phase_calculated(self, phase: str) -> bool:
        """Checks if a specific phase has been calculated."""
        return self.calculated_phases.get(phase, False)

    def set_finalized(self, state: bool):
        """Sets the finalized state of the results."""
        self.finalized = state

    def __repr__(self):
        """Provides a detailed string representation of the results."""
        return json.dumps(
            {
                "energies": self.energies,
                "timings": self.timings,
                "calculated_phases": self.calculated_phases,
                "finalized": self.finalized,
            },
            indent=2,
        )

    def get_timing(self, phase: str, component: str):
        """Helper to get timing values."""
        return self.timings.get(phase, {}).get(component)

    def generate_energy_report_strings(
        self, indent_spaces, field_width, format_specifier="s"
    ):
        """
        Generates formatted strings for a report of calculated energies and key timings.
        Follows the summary function's convention by returning strings instead of printing directly.

        Args:
            indent_spaces (int): Number of spaces for indentation.
            field_width (int): Width of the field names in the report.
            format_specifier (str): Format specifier for field names (e.g., 's' for string).

        Returns:
            tuple[str, str]: A tuple containing two strings:
                             - timing_message (str): The full formatted timing report.
                             - energy_message (str): The full formatted energy report.
        """
        indent = " " * indent_spaces
        field_format = f"{{:{field_width}{format_specifier}}}"
        timing_lines = []
        energy_lines = []

        timing_tag = "Time> "
        energy_tag = "Energy> "

        if not self.finalized:
            warning_msg = (
                "\n" + "=" * 50 + "\n"
                "WARNING: Energy Report is not finalized!\n"
                "To get a complete report, ensure `calculate_all_energies` was called with `final=True`.\n"
                "=" * 50 + "\n"
            )
            return warning_msg, ""  # Return warning and empty energy message

        # --- Timings Report ---
        # Helper to calculate total energy calculation time for a phase based on specific components
        def _get_phase_energy_calc_time(phase: str) -> float:
            total_time = 0.0
            nonlinear_time = 0.0
            rf_time = self.get_timing(phase, "reactionfield_energy")
            if rf_time is not None:
                total_time += rf_time
            grid_time = self.get_timing(phase, "grid_energy")
            if grid_time is not None:
                total_time += grid_time
            if phase == "water":  # Nonlinear is specific to water
                nonlinear_time = self.get_timing(phase, "nonlinear_pb_terms")
                if nonlinear_time is not None:
                    total_time += nonlinear_time
                else:
                    nonlinear_time = 0.0
            return total_time, nonlinear_time

        # Print individual phase timings IF BOTH PHASES ARE PRESENT
        vacuum_calculated_for_energy = (
            self.is_phase_calculated("vacuum")
            and self.get_energy("vacuum", "reactionfield_energy") is not None
        )
        water_calculated_for_energy = (
            self.is_phase_calculated("water")
            and self.get_energy("water", "reactionfield_energy") is not None
        )
        both_phases_present = (
            vacuum_calculated_for_energy and water_calculated_for_energy
        )

        if both_phases_present:
            vacuum_phase_time, nonlinear_time = _get_phase_energy_calc_time("vacuum")
            for phase_timings in self.timings.values():
                # Exclude 'overall' timings from this specific sum
                if phase_timings is self.timings["overall"]:
                    continue
                for component_txt, component_time in phase_timings.items():
                    if component_time is not None:
                        timing_lines.append(
                            f"{indent}{field_format.format(timing_tag + component_txt.replace("_", " "))} : {component_time:13.3f} s"
                        )
            if nonlinear_time > 1.0e-1:
                timing_lines.append(
                    f"{indent}{field_format.format(timing_tag + 'Non-linear energy calc in vacuum')} : {nonlinear_time:13.3f} s"
                )
            timing_lines.append(
                f"{indent}{field_format.format(timing_tag + 'Energy calc in vacuum')} : {vacuum_phase_time:13.3f} s"
            )

            water_phase_time, nonlinear_time = _get_phase_energy_calc_time("water")
            if nonlinear_time > 1.0e-1:
                timing_lines.append(
                    f"{indent}{field_format.format(timing_tag + 'Non-linear energy calc in water')} : {nonlinear_time:13.3f} s"
                )
            timing_lines.append(
                f"{indent}{field_format.format(timing_tag + 'Energy calc in water')} : {water_phase_time:13.3f} s"
            )

        # Print the total time spent on individual energy components (sum of specific timed parts)
        total_energy_component_time = 0.0
        for phase_timings in self.timings.values():
            # Exclude 'overall' timings from this specific sum
            if phase_timings is self.timings["overall"]:
                continue
            for component_time in phase_timings.values():
                if component_time is not None:
                    total_energy_component_time += component_time

        if not both_phases_present:
            for phase_timings in self.timings.values():
                # Exclude 'overall' timings from this specific sum
                if phase_timings is self.timings["overall"]:
                    continue
                for component_txt, component_time in phase_timings.items():
                    if component_time is not None:
                        timing_lines.append(
                            f"{indent}{field_format.format(timing_tag + component_txt.replace("_", " "))} : {component_time:13.3f} s"
                        )
            nonlinear_time = self.get_timing("water", "nonlinear_pb_terms")
            if nonlinear_time is not None and nonlinear_time > 1.0e-1:
                timing_lines.append(
                    f"{indent}{field_format.format(timing_tag + 'Non-linear energy calc (ossmos+stress)')} : {nonlinear_time:13.3f} s"
                )

        timing_lines.append(
            f"{indent}{field_format.format(timing_tag + 'Total time for energy calc')} : {total_energy_component_time:13.3f} s"
        )

        # Print the cumulative time of all calls to calculate_all_energies that contributed.
        cumulative_overall_time = self.get_timing("overall", "total_calculation_time")
        if cumulative_overall_time is not None:
            timing_lines.append(
                f"{indent}{field_format.format(timing_tag + 'Cumulative time by energy module')} : {cumulative_overall_time:13.3f} s"
            )

        # --- Energy Report ---
        # Print ALL computed phase-independent energies
        if self.energies["phase_independent"]:
            for component, value in self.energies["phase_independent"].items():
                component_time = self.get_timing("phase_independent", component)
                component_format = component.replace("_", " ").capitalize()

                if component_format.startswith("Lj"):
                    component_format = component_format.replace("Lj ", "LJ ")

                energy_lines.append(
                    f"{indent}{field_format.format(energy_tag+ component_format)} : {value:12.2f} kT"
                )
                timing_lines.append(
                    f"{indent}{field_format.format(timing_tag + component_format)} : {component_time:13.3f} s"
                )

        water_rf_energy = self.get_energy("water", "reactionfield_energy")
        vacuum_rf_energy = self.get_energy("vacuum", "reactionfield_energy")
        total_rf_solvation_energy = self.get_energy(
            "total", "reactionfield_solvation_energy"
        )

        # Conditional printing of solvation energies
        if both_phases_present:
            for component, value in self.energies["water"].items():
                # phase-dependent reactionfield energy is physically meaningless
                if component.startswith("reactionfield"):
                    continue
                display_component = (
                    energy_tag + component.replace("_", " ").capitalize()
                )
                if component not in (
                    "electrostatic_stress_term",
                    "osmotic_pressure_term",
                ):
                    display_component += display_component + " in water"

                energy_lines.append(
                    f"{indent}{field_format.format(display_component)} : {value:12.2f} kT"
                )

            for component, value in self.energies["vacuum"].items():
                # phase-dependent reactionfield energy is physically meaningless
                if component.startswith("reactionfield"):
                    continue
                display_component = (
                    energy_tag + component.replace("_", " ").capitalize() + " in vacuum"
                )
                energy_lines.append(
                    f"{indent}{field_format.format(display_component)} : {value:12.2f} kT"
                )

            # Always print the total reaction-field solvation energy if both phases were available
            if total_rf_solvation_energy is not None:
                energy_lines.append(
                    f"{indent}{field_format.format(energy_tag + 'Total reaction-field energy')} : {total_rf_solvation_energy:12.2f} kT"
                )
        elif water_calculated_for_energy or vacuum_calculated_for_energy:
            if water_calculated_for_energy:
                for component, value in self.energies["water"].items():
                    display_component = (
                        energy_tag + component.replace("_", " ").capitalize()
                    )
                    energy_lines.append(
                        f"{indent}{field_format.format(display_component)} : {value:12.2f} kT"
                    )
                # energy_lines.append(
                #     f"{indent}{field_format.format(energy_tag + 'Total reaction-field energy')} : {water_rf_energy:12.2f} kT"
                # )
            elif vacuum_calculated_for_energy:
                raise Exception(
                    "Only vaccum phase energy calculations are meaningless."
                )
        # Print total energies
        for component, value in self.energies["total"].items():
            display_component = energy_tag + component.replace("_", " ").capitalize()
            energy_lines.append(
                f"{indent}{field_format.format(display_component)} : {value:12.2f} kT"
            )

        # If neither is present (e.g., only phase_independent was calculated), then no solvation energy is printed

        # overall_solvation_energy = self.get_energy("total", "overall_solvation_energy")
        # if overall_solvation_energy is not None:
        #     energy_lines.append(
        #         f"{indent}{field_format.format(energy_tag + 'Overall solvation energy')} : {overall_solvation_energy:12.2f} kT"
        #     )

        # Join lines for final messages
        timing_message = "\n".join(timing_lines)
        energy_message = "\n".join(energy_lines)

        return timing_message, energy_message
