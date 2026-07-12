#!/usr/bin/env python
# coding: utf-8
from pydelphi.constants import ATOMFIELD_LJ_GAMMA

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

from pydelphi.geometry.gaussian_overlap import (
    compute_gaussian_overlap_surface_volume,
    SIGMOID_G,
)


def calc_nonpolar_energy(
    atom_data,
    atom_adjacency_csr,
    probe_radius,
    radius_offset,
    temperature,
    pressure_coeff,
    sigmoid_g=SIGMOID_G,
    T_ref=298.0,
    sentinel_adj=-1,
):
    total_volume, cavity_volume, total_SA, per_atom_SA = (
        compute_gaussian_overlap_surface_volume(
            atom_data=atom_data,
            atom_adjacency_csr=atom_adjacency_csr,
            probe_radius=probe_radius,
            radius_offset=radius_offset,
            sigmoid_g=sigmoid_g,
            sentinel_adj=sentinel_adj,
        )
    )

    temperature_factor = temperature / T_ref
    # print(
    #     temperature_factor,
    # )

    # SA term
    solute_surface_area = 0.0
    energy_cavity_SA = 0.0
    num_atoms = per_atom_SA.shape[0]
    for i in range(num_atoms):
        solute_surface_area += per_atom_SA[i]
        energy_cavity_SA += per_atom_SA[i] * atom_data[i, ATOMFIELD_LJ_GAMMA]
    energy_cavity_SA *= temperature_factor

    # Volume term
    energy_cavity_volume = pressure_coeff * cavity_volume * temperature_factor

    return (
        cavity_volume,
        total_volume,
        solute_surface_area,
        energy_cavity_SA,
        energy_cavity_volume,
    )
