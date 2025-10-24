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
Energy term abbreviations and descriptions for standardized output.

This module defines mappings between energy term keys (used in energy results dictionaries)
and their standardized abbreviations for CSV/TSV output. It also provides human-readable
descriptions for each energy term.

Available features:
- `ENERGY_TERM_ABBREVIATIONS`: maps energy term keys to CSV/TSV-friendly abbreviations.
- `ENERGY_TERM_DESCRIPTIONS`: maps energy term keys to human-readable descriptions.
- `REVERSE_ABBREVIATIONS`: maps abbreviations back to their full energy term keys.

Intended usage:
- CSV/TSV report generation.
- Logging and debugging.
- Ensuring consistent column naming across single-point and trajectory analyses.
- Parsing CSV/TSV outputs back into energy dictionaries (using the reverse mapping).

Future extensions may include:
- Defining term groupings (e.g., phase-independent, vacuum, water, total).
- Supporting aliasing or custom term mappings.
"""

ENERGY_TERM_ABBREVIATIONS = {
    "phase_independent.coulombic_energy": "E_coul",
    "phase_independent.LJ_energy": "E_lj",
    "phase_independent.nonpolar_cavity": "E_np_cav",
    "phase_independent.nonpolar_sa": "E_np_sa",
    "vacuum.grid_energy": "E_grid_v",
    "vacuum.reactionfield_energy": "E_rxn_v",
    "water.grid_energy": "E_grid_w",
    "water.reactionfield_energy": "E_rxn_w",
    "water.electrostatic_stress_term": "E_stress",
    "water.osmotic_pressure_term": "E_osmotic",
    "total.corrected_reaction_field_energy": "E_rxn_corr_tot",
    "total.total_grid_energy": "E_grid_tot",
    "total.total_nonlinear_grid_energy": "E_grid_nl_tot",
}

ENERGY_TERM_DESCRIPTIONS = {
    "phase_independent.coulombic_energy": "Coulombic energy (phase-independent)",
    "phase_independent.LJ_energy": "Lennard-Jones energy (phase-independent)",
    "phase_independent.nonpolar_cavity": "Non-polar cavity solvation energy",
    "phase_independent.nonpolar_sa": "Non-polar surface area solvation energy",
    "vacuum.grid_energy": "Vacuum phase grid energy",
    "vacuum.reactionfield_energy": "Vacuum phase reaction field energy",
    "water.grid_energy": "Water phase grid energy",
    "water.reactionfield_energy": "Water phase reaction field energy",
    "water.electrostatic_stress_term": "Water phase electrostatic stress contribution",
    "water.osmotic_pressure_term": "Water phase osmotic pressure contribution",
    "total.corrected_reaction_field_energy": "Total corrected reaction field energy",
    "total.total_grid_energy": "Total grid energy",
    "total.total_nonlinear_grid_energy": "Total nonlinear grid energy",
}

# Reverse mapping: abbreviation -> full energy term key
REVERSE_ABBREVIATIONS = {
    abbr: full_key for full_key, abbr in ENERGY_TERM_ABBREVIATIONS.items()
}
