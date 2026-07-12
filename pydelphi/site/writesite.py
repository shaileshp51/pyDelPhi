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


from sys import exit as sys_exit
from pydelphi.site.site import *
from pydelphi.utils.interpolation import *
from pydelphi.site.siteexceptions import *

import numpy as np

from pydelphi.foundation.enums import Precision

from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_real,
    vprint,
)

from pydelphi.config.logging_config import INFO, DEBUG, ERROR, get_effective_verbosity

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

# --- Dynamic Precision Handling ---
if PRECISION.value in {Precision.SINGLE.value}:
    from pydelphi.utils.prec.single import (
        or_lt_vector,
    )

    try:
        import pydelphi.utils.cuda.single as size_gpu
    except ImportError:
        size_gpu = None
elif PRECISION.value == Precision.DOUBLE.value:
    from pydelphi.utils.prec.double import (
        or_lt_vector,
    )

    try:
        import pydelphi.utils.cuda.double as size_gpu
    except ImportError:
        size_gpu = None
else:
    raise ValueError(f"Unsupported PRECISION: {PRECISION}")

from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_Z,
    ATOMFIELD_CHARGE,
    ATOMFIELD_RADIUS,
    ConstDelPhiInts,
)

RES_NUMBER_UNKNOWN = ConstDelPhiInts.ResidueNumberUnknown


def find_record(atm, res, rnum, chn, file_type, value):
    """Placeholder for find_record function."""
    pass


# def _write_text_frc_header(
#     outfile_stream,
#     grid_shape,
#     percent_fill,
#     external_dielectric,
#     media_epsilons,
#     gap_dielectric,
#     dielectric_model,
#     surface_method,
#     ion_strength,
#     ion_radius,
#     probe_radius,
#     total_iters,
#     final_rms,
#     final_dphi,
#     convergence_status,
#     boundary_condition,
#     datum,
#     map_title,
#     vrow,
# ):
#     """Writes the header for a text format FRC file."""
#     outfile_stream.write("DELPHI SITE POTENTIAL FILE\n")
#     outfile_stream.write(
#         f"grid size, percent fill:   {grid_shape}    {percent_fill:.3f}\n"
#     )
#     internal_epsilon_first = (
#         media_epsilons[1]
#         if (media_epsilons is not None) and len(media_epsilons) > 1
#         else 1.0
#     )
#     outfile_stream.write(
#         "outer diel. and first one assigned :   "
#         f"{external_dielectric:.2f}    "
#         f"{internal_epsilon_first:.2f}\n"
#     )
#     outfile_stream.write(f"ionic strength (M):   {ion_strength}\n")
#
#     if isinstance(probe_radius, (list, tuple, np.ndarray)):
#         probe_radius_1 = probe_radius[0] if len(probe_radius) > 0 else 1.4
#         probe_radius_2 = probe_radius[1] if len(probe_radius) > 1 else probe_radius_1
#     elif isinstance(probe_radius, (int, float)):
#         probe_radius_1 = probe_radius
#         probe_radius_2 = probe_radius_1
#     else:
#         probe_radius_1 = 0.0
#         probe_radius_2 = 0.0
#
#     outfile_stream.write(
#         f"ion excl., probe radii:   {ion_radius}    {probe_radius_1}    {probe_radius_2}\n"
#     )
#
#     outfile_stream.write(
#         f"linear, nolinear iterations:   {linear_iteration_num}    {non_linear_iteration_num}\n"
#     )
#     outfile_stream.write(f"boundary condition:   {boundary_condition}\n")
#     outfile_stream.write(f"Data Output:   {datum}\n")
#     outfile_stream.write(f"title: {map_title}\n")
#     outfile_stream.write("\n\n")
#     outfile_stream.write(f"{vrow}\n")


def _write_text_frc_header(
    outfile_stream,
    grid_shape,
    percent_fill,
    external_dielectric,
    media_epsilons,
    gap_dielectric,
    dielectric_model,
    surface_method,
    ion_strength,
    ion_radius,
    probe_radius,
    total_iters,
    final_rms,
    final_dphi,
    convergence_status,
    boundary_condition,
    datum,
    map_title,
    vrow,
):
    """Writes the header for a text format FRC file."""

    internal_epsilon_first = (
        media_epsilons[1]
        if (media_epsilons is not None) and len(media_epsilons) > 1
        else 1.0
    )

    if isinstance(probe_radius, (list, tuple, np.ndarray)):
        probe_radius_1 = probe_radius[0] if len(probe_radius) > 0 else 1.4
        probe_radius_2 = probe_radius[1] if len(probe_radius) > 1 else probe_radius_1
    elif isinstance(probe_radius, (int, float)):
        probe_radius_1 = probe_radius
        probe_radius_2 = probe_radius_1
    else:
        probe_radius_1 = 0.0
        probe_radius_2 = 0.0

    outfile_stream.write("# DELPHI SITE POTENTIAL FILE\n")
    outfile_stream.write(f"# grid_size: {grid_shape}\n")
    outfile_stream.write(f"# percent_fill: {percent_fill:.3f}\n")
    outfile_stream.write(f"# dielectric_model: {dielectric_model}\n")
    outfile_stream.write(f"# surface_method: {surface_method}\n")
    outfile_stream.write(f"# exdi: {external_dielectric:.2f}\n")
    outfile_stream.write(f"# indi: {internal_epsilon_first:.2f}\n")
    if gap_dielectric is not None:
        outfile_stream.write(f"# gapdi: {gap_dielectric:.2f}\n")
    outfile_stream.write(f"# ionic_strength_M: {ion_strength}\n")
    outfile_stream.write(f"# ion_exclusion_radius: {ion_radius}\n")
    outfile_stream.write(f"# probe_radius_1: {probe_radius_1}\n")
    outfile_stream.write(f"# probe_radius_2: {probe_radius_2}\n")
    outfile_stream.write(f"# total_iters: {total_iters}\n")
    outfile_stream.write(f"# final_rms: {final_rms:.3e}\n")
    outfile_stream.write(f"# final_dphi: {final_dphi:.3e}\n")
    outfile_stream.write(f"# convergence_status: {convergence_status}\n")
    outfile_stream.write(f"# boundary_condition: {boundary_condition}\n")
    outfile_stream.write(f"# data_output: {datum}\n")
    outfile_stream.write(f"# title: {map_title}\n")
    outfile_stream.write(
        "# GRID_PHI = grid electrostatic potential interpolated at the evaluation point\n"
    )
    outfile_stream.write(
        "# GF_EX/GF_EY/GF_EZ = grid electric field components interpolated at the evaluation point\n"
    )
    outfile_stream.write(f"{vrow}\n")


def _calculate_atom_potential(
    grid_coords,
    atom_radius,
    charge_value,
    potential_upper_bond,
    grid_shape,
    potential_map,
    scale_factor,
):
    """Calculates atom potential value."""
    if abs(charge_value) <= 1e-6:
        return 0.0

    atom_radius_scaled = min(atom_radius, potential_upper_bond * scale_factor)
    atom_potential_value = 0.0
    coords_offsets = [
        np.array([atom_radius_scaled, 0, 0]),
        np.array([-atom_radius_scaled, 0, 0]),
        np.array([0, atom_radius_scaled, 0]),
        np.array([0, -atom_radius_scaled, 0]),
        np.array([0, 0, atom_radius_scaled]),
        np.array([0, 0, -atom_radius_scaled]),
    ]
    for offset in coords_offsets:
        xt = grid_coords + offset
        intrpl_status, intrpl_value = tricubic_interpolation(
            grid_shape, potential_map, xt
        )
        atom_potential_value += intrpl_value
    return atom_potential_value / 6.0


def _calculate_grid_potential_and_salt(
    grid_coords,
    grid_shape,
    potential_map,
    output_salt_concentration,
    non_linear_iteration_num,
    ion_strength,
    taylor_coeff1,
    taylor_coeff2,
    taylor_coeff3,
    taylor_coeff4,
    taylor_coeff5,
):
    """Calculates grid potential and salt concentration."""
    _, potential_value = tricubic_interpolation(grid_shape, potential_map, grid_coords)
    salt_concentration = 0.0

    if output_salt_concentration:
        vprint(INFO, _VERBOSITY, CNoIDebMap())
        if non_linear_iteration_num != 0:
            temp = potential_value * taylor_coeff5 + taylor_coeff4
            temp = potential_value * temp + taylor_coeff3
            temp = potential_value * temp + taylor_coeff2
            temp = potential_value * temp + taylor_coeff1
            salt_concentration = potential_value * temp
        else:
            salt_concentration = -ion_strength * 2.0 * potential_value
    return potential_value, salt_concentration


def _calculate_debye_fraction(
    grid_coords,
    grid_shape,
    dielectric_map_bool,
    output_debye_fraction_value,
    verbose,
):
    """Calculates Debye Fraction."""
    if verbose and output_debye_fraction_value:
        print("Calculating Debye Fraction")
    interpl_status, interpl_value = bool_interpolation(
        grid_shape, dielectric_map_bool, grid_coords
    )
    return interpl_value


def _calculate_field_xyz(
    grid_coords,
    grid_shape,
    potential_map,
    scale_factor,
):
    """Calculates electric field components (Ex, Ey, Ez)."""
    field_xyz = np.zeros(3)
    coords_offsets = [
        np.array([1.0, 0, 0]),
        np.array([-1.0, 0, 0]),
        np.array([0, 1.0, 0]),
        np.array([0, -1.0, 0]),
        np.array([0, 0, 1.0]),
        np.array([0, 0, -1.0]),
    ]
    _, phi_x_plus = tricubic_interpolation(
        grid_shape, potential_map, grid_coords + coords_offsets[0]
    )
    _, phi_x_minus = tricubic_interpolation(
        grid_shape, potential_map, grid_coords + coords_offsets[1]
    )
    _, phi_y_plus = tricubic_interpolation(
        grid_shape, potential_map, grid_coords + coords_offsets[2]
    )
    _, phi_y_minus = tricubic_interpolation(
        grid_shape, potential_map, grid_coords + coords_offsets[3]
    )
    _, phi_z_plus = tricubic_interpolation(
        grid_shape, potential_map, grid_coords + coords_offsets[4]
    )
    _, phi_z_minus = tricubic_interpolation(
        grid_shape, potential_map, grid_coords + coords_offsets[5]
    )

    # Compute -∇φ using central difference
    field_xyz[0] = -(phi_x_plus - phi_x_minus) * 0.5 * scale_factor
    field_xyz[1] = -(phi_y_plus - phi_y_minus) * 0.5 * scale_factor
    field_xyz[2] = -(phi_z_plus - phi_z_minus) * 0.5 * scale_factor
    return field_xyz


def _calculate_total_potential_terms(
    atom_coords,
    grid_coords_box,
    box_center,
    grid_shape,
    scale_factor,
    num_surface_charges,
    surface_charge_pos_array,
    surface_charge_e_array,
    epkt_value,
    boundary_grid_array,
    charge_grid_num,
    charge_pos_array,
    atomic_charge_list,
    atom_eps_array,
    grid_offset,
):
    """Calculates terms for total potential (reaction, surface, coulomb, atomic coulomb)."""
    box_edge = np.array(
        [
            box_center[0] - (grid_shape - 1) * 0.5 / scale_factor,
            box_center[1] - (grid_shape - 1) * 0.5 / scale_factor,
            box_center[2] - (grid_shape - 1) * 0.5 / scale_factor,
        ]
    )
    box_edge_max = np.array(
        [
            box_center[0] + (grid_shape - 1) * 0.5 / scale_factor,
            box_center[1] + (grid_shape - 1) * 0.5 / scale_factor,
            box_center[2] - (grid_shape - 1) * 0.5 / scale_factor,
        ]
    )

    if np.any(atom_coords < box_edge) or np.any(atom_coords > box_edge_max):
        return 0.0, 0.0, 0.0, 0.0, 0

    reaction_potential = 0.0
    surface_potential_term = 0.0
    n_surface_charges_internal = 0
    total_surface_charge_value = 0.0
    sold = [0.0] * 30

    for i in range(num_surface_charges):
        vtemp = atom_coords - surface_charge_pos_array[i]
        dist = np.linalg.norm(vtemp)
        n_surface_charges_internal += 1
        total_surface_charge_value += surface_charge_e_array[i]
        reaction_potential_term_i = surface_charge_e_array[i] / dist
        reaction_potential_term_i *= epkt_value
        reaction_potential += reaction_potential_term_i
        xu2_val = np.array(
            [
                float(boundary_grid_array[i][0]),
                float(boundary_grid_array[i][1]),
                float(boundary_grid_array[i][2]),
            ]
        )
        charge_value = surface_charge_e_array[i]
        surface_potential_term_i = tops(
            xu=xu2_val, xo=grid_coords_box, c=charge_value, r=1.0, n=1
        )
        surface_potential_term_i *= 2.0
        surface_potential_term += surface_potential_term_i
        idist = int(dist)
        if idist < 30:
            sold[idist] += surface_potential_term_i - reaction_potential_term_i

    coulomb_potential = 0.0
    atomic_coulomb_potential = 0.0
    n_charge_grids = 0

    for i in range(charge_grid_num):
        if np.any(charge_pos_array[i] < box_edge) or np.any(
            charge_pos_array[i] > box_edge_max
        ):
            continue
        n_charge_grids += 1
        vtemp = atom_coords - charge_pos_array[i]
        dist_sq = np.dot(vtemp, vtemp)
        if 5.0 > dist_sq:
            if dist_sq > 1e-6:
                temp_atomic_coulomb_potential = atomic_charge_list[
                    i
                ].nValue / math.sqrt(dist_sq)
                coulomb_potential += temp_atomic_coulomb_potential / atom_eps_array[i]
                xu = charge_pos_array[i]
                charge_value = atomic_charge_list[i].nValue
                xu2_val = (xu - box_center) * scale_factor + grid_offset
                eps_val = atom_eps_array[i] * epkt_value
                atomic_coulomb_potential_i = tops(
                    xu=xu2_val, xo=grid_coords_box, c=charge_value, r=eps_val, n=1
                )
                atomic_coulomb_potential += atomic_coulomb_potential_i
    atomic_coulomb_potential *= 2.0

    return (
        reaction_potential,
        surface_potential_term,
        coulomb_potential,
        atomic_coulomb_potential,
        sold,
    )


def _calculate_reaction_potential_only(
    atom_coords,
    num_surface_charges,
    surface_charge_pos_array,
    surface_charge_e_array,
    epkt_value,
):
    """Calculates only reaction potential."""
    reaction_potential = 0.0
    for i in range(num_surface_charges):
        vtemp = atom_coords - surface_charge_pos_array[i]
        dist = np.linalg.norm(vtemp)
        reaction_potential += epkt_value * surface_charge_e_array[i] / dist
    return reaction_potential


# def _write_output_values(
#     output_file_stream,
#     out_react_pot,
#     out_coulomb_pot,
#     out_atom_pot,
#     out_debye_frac,
#     out_field,
#     out_surf_charge,
#     out_total_force,
#     out_react_force,
#     out_coulomb_force,
#     out_total_pot,
#     out_atom_desc,
#     out_atom_coords,
#     out_charge,
#     out_grid_pot,
#     out_salt,
#     epkt_value,
#     reaction_potential,
#     coulomb_potential,
#     atom_potential_value,
#     debye_fraction,
#     field_xyz,
#     total_surface_charge_value,
#     atom_coords,
#     surface_potential_term,
#     total_force_xyz,
#     reaction_force_xyz,
#     coulomb_force_xyz,
#     total_potential,
#     atom_descriptor,
#     charge_value,
#     potential_value,
#     salt_concentration,
# ):
#     """Writes output values in formatted text mode."""
#     if out_atom_desc and atom_descriptor is not None:
#         output_file_stream.write(f"{atom_descriptor}")
#     if out_atom_coords and atom_coords is not None:
#         output_file_stream.write(
#             f"{atom_coords[0]:10.4f}{atom_coords[1]:10.4f}{atom_coords[2]:10.4f}"
#         )
#     if out_charge and charge_value is not None:
#         output_file_stream.write(f"{charge_value:10.4f}")
#
#     # print("site: 432>>> ", potential_value)
#     if out_grid_pot and potential_value is not None:
#         output_file_stream.write(f"{potential_value:10.4f}")
#     if out_salt and salt_concentration is not None:
#         output_file_stream.write(f"{salt_concentration:10.4f}")
#     if out_react_pot and reaction_potential is not None:
#         output_file_stream.write(f"{reaction_potential:10.4f}")
#     if out_coulomb_pot and coulomb_potential is not None:
#         output_file_stream.write(f"{coulomb_potential:10.4f}")
#     if out_atom_pot and atom_potential_value is not None:
#         output_file_stream.write(f"{atom_potential_value:10.4f}")
#
#     if out_debye_frac and debye_fraction is not None:
#         output_file_stream.write(f"{debye_fraction:10.4f}")
#     if out_field and field_xyz is not None:
#         output_file_stream.write(
#             f"{field_xyz[0]:10.4f}{field_xyz[1]:10.4f}{field_xyz[2]:10.4f}"
#         )
#
#     if out_react_force and reaction_force_xyz is not None:
#         output_file_stream.write(
#             f"{reaction_force_xyz[0]:10.4f}{reaction_force_xyz[1]:10.4f}{reaction_force_xyz[2]:10.4f}"
#         )
#     if out_coulomb_force and coulomb_force_xyz is not None:
#         output_file_stream.write(
#             f"{coulomb_force_xyz[0]:10.4f}{coulomb_force_xyz[1]:10.4f}{coulomb_force_xyz[2]:10.4f}"
#         )
#     if out_total_force and total_force_xyz is not None:
#         output_file_stream.write(
#             f"{total_force_xyz[0]:10.4f}{total_force_xyz[1]:10.4f}{total_force_xyz[2]:10.4f}"
#         )
#
#     if out_total_pot and total_potential is not None:
#         output_file_stream.write(f"{total_potential:10.4f}")
#
#     if (
#         out_surf_charge
#         and total_surface_charge_value is not None
#         and atom_coords is not None
#         and surface_potential_term is not None
#     ):
#         output_file_stream.write(
#             f"{total_surface_charge_value:10.4f} {atom_coords[0]:10.4f} {atom_coords[1]:10.4f} {atom_coords[2]:10.4f} {surface_potential_term:10.4f} {surface_potential_term / epkt_value:10.4f}"
#         )
#     output_file_stream.write("\n")


def _write_output_values(
    output_file_stream,
    out_react_pot,
    out_coulomb_pot,
    out_atom_pot,
    out_debye_frac,
    out_field,
    out_surf_charge,
    out_total_force,
    out_react_force,
    out_coulomb_force,
    out_total_pot,
    out_atom_desc,
    out_atom_coords,
    out_charge,
    out_grid_pot,
    out_salt,
    epkt_value,
    reaction_potential,
    coulomb_potential,
    atom_potential_value,
    debye_fraction,
    field_xyz,
    total_surface_charge_value,
    atom_coords,
    surface_potential_term,
    total_force_xyz,
    reaction_force_xyz,
    coulomb_force_xyz,
    total_potential,
    atom_descriptor,
    charge_value,
    potential_value,
    salt_concentration,
):
    """Writes one row of formatted text FRC output.

    Column widths mirror _setup_output_header_strings():
      ATOM/RESNAME/CHAIN/RESID : 35 chars total
      coordinates              : 3 x 12 chars
      scalar potentials/charge : mostly 10 chars
      SALT_CONC                : 12 chars
      DEBYE_FRAC               : 14 chars
      vector fields/forces     : 3 x 10 chars
      surface-charge block     : 6 x 12 chars
    """
    if out_atom_desc and atom_descriptor is not None:
        output_file_stream.write(f"{atom_descriptor:<35.35s}")

    if out_atom_coords and atom_coords is not None:
        output_file_stream.write(
            f"{atom_coords[0]:12.4f}"
            f"{atom_coords[1]:12.4f}"
            f"{atom_coords[2]:12.4f}"
        )

    if out_charge and charge_value is not None:
        output_file_stream.write(f"{charge_value:12.4f}")

    if out_grid_pot and potential_value is not None:
        output_file_stream.write(f"{potential_value:12.4f}")

    if out_salt and salt_concentration is not None:
        output_file_stream.write(f"{salt_concentration:12.4f}")

    if out_react_pot and reaction_potential is not None:
        output_file_stream.write(f"{reaction_potential:12.4f}")

    if out_coulomb_pot and coulomb_potential is not None:
        output_file_stream.write(f"{coulomb_potential:12.4f}")

    if out_atom_pot and atom_potential_value is not None:
        output_file_stream.write(f"{atom_potential_value:12.4f}")

    if out_debye_frac and debye_fraction is not None:
        output_file_stream.write(f"{debye_fraction:14.4f}")

    if out_field and field_xyz is not None:
        output_file_stream.write(
            f"{field_xyz[0]:12.4f}" f"{field_xyz[1]:12.4f}" f"{field_xyz[2]:12.4f}"
        )

    if out_react_force and reaction_force_xyz is not None:
        output_file_stream.write(
            f"{reaction_force_xyz[0]:12.4f}"
            f"{reaction_force_xyz[1]:12.4f}"
            f"{reaction_force_xyz[2]:12.4f}"
        )

    if out_coulomb_force and coulomb_force_xyz is not None:
        output_file_stream.write(
            f"{coulomb_force_xyz[0]:12.4f}"
            f"{coulomb_force_xyz[1]:12.4f}"
            f"{coulomb_force_xyz[2]:12.4f}"
        )

    if out_total_force and total_force_xyz is not None:
        output_file_stream.write(
            f"{total_force_xyz[0]:12.4f}"
            f"{total_force_xyz[1]:12.4f}"
            f"{total_force_xyz[2]:12.4f}"
        )

    if out_total_pot and total_potential is not None:
        output_file_stream.write(f"{total_potential:10.4f}")

    if (
        out_surf_charge
        and total_surface_charge_value is not None
        and atom_coords is not None
        and surface_potential_term is not None
    ):
        output_file_stream.write(
            f"{total_surface_charge_value:12.4f}"
            f"{atom_coords[0]:12.4f}"
            f"{atom_coords[1]:12.4f}"
            f"{atom_coords[2]:12.4f}"
            f"{surface_potential_term:12.4f}"
            f"{surface_potential_term / epkt_value:12.4f}"
        )

    output_file_stream.write("\n")


# def _setup_output_header_strings(
#     out_atom_desc,
#     out_atom_coords,
#     out_charge,
#     out_grid_pot,
#     out_salt,
#     out_react_pot,
#     out_coulomb_pot,
#     out_atom_pot,
#     out_debye_frac,
#     out_field,
#     out_surf_charge,
#     out_total_force,
#     out_react_force,
#     out_coulomb_force,
#     out_total_pot,
# ):
#     """Sets up the column/datum header strings for output frc file based on output flags."""
#     frc_header = " " * 80
#     datum = " " * 65
#     j = 0
#     k = 0
#     output_columns_flags = [
#         (out_atom_desc, "ATOM DESCRIPTOR", "ATOM ", 20, 5, 15),
#         (
#             out_atom_coords,
#             "ATOM COORDINATES (X,Y,Z)",
#             "COORDINATES ",
#             30,
#             12,
#             24,
#         ),
#         (out_charge, "CHARGE", "CHARGE ", 10, 7, 6),
#         (out_grid_pot, "GRID_PHI", "POTENTIALS ", 10, 11, 8),
#         (out_salt, "SALT_CONC", "SALT ", 10, 5, 8),
#         (out_react_pot, " REAC._PHI", "REACTION ", 10, 9, 10),
#         (out_coulomb_pot, " COUL._PHI", "COULOMBIC ", 10, 10, 10),
#         (out_atom_pot, "ATOM_PHI", "ATOMIC PT. ", 10, 11, 8),
#         (out_debye_frac, "DEBFRACTION", "DEBFRACTION ", 14, 12, 11),
#         (out_field, "GRID FIELDS: (Ex, Ey, Ez)", "FIELDS ", 30, 7, 25),
#         (out_react_force, "REAC. FORCE: (Rx, Ry, Rz)", "RFORCE ", 30, 7, 25),
#         (out_coulomb_force, "COUL. FORCE: (Cx, Cy, Cz)", "CFORCE ", 30, 7, 25),
#         (out_total_force, "TOTAL FORCE: (Tx, Ty, Tz)", "TFORCE ", 30, 7, 25),
#         (out_total_pot, " TOTAL", "TOTAL ", 10, 6, 6),
#         (
#             out_surf_charge,
#             "sCharge,    x          y       z       surf.E°n,surf. E[kT/(qA)]",
#             "SCh, x, y, z, surf En, surf. E",
#             50,
#             35,
#             65,
#         ),
#     ]
#
#     for (
#         flag,
#         column_name,
#         datum_name,
#         column_start_index,
#         datum_start_index,
#         column_len,
#     ) in output_columns_flags:
#         if flag:
#             frc_header = frc_header[:j] + column_name + frc_header[j + column_len :]
#             datum = datum[:k] + datum_name + datum[k + datum_start_index :]
#             j += column_start_index
#             k += datum_start_index
#         if (
#             j >= 80
#             and (
#                 out_react_pot
#                 or out_coulomb_pot
#                 or out_atom_pot
#                 or out_debye_frac
#                 or out_field
#                 or out_surf_charge
#                 or out_total_force
#                 or out_react_force
#                 or out_coulomb_force
#                 or out_total_pot
#             )
#             and flag not in [out_surf_charge]
#         ):
#             out_react_pot = out_coulomb_pot = out_atom_pot = out_debye_frac = (
#                 out_field
#             ) = out_surf_charge = out_total_force = out_react_force = (
#                 out_coulomb_force
#             ) = out_total_pot = False
#         if (
#             j >= 60
#             and flag
#             in [
#                 out_field,
#                 out_react_force,
#                 out_coulomb_force,
#                 out_total_force,
#             ]
#             and (
#                 out_field
#                 or out_surf_charge
#                 or out_total_force
#                 or out_react_force
#                 or out_coulomb_force
#                 or out_total_pot
#             )
#         ):
#             out_field = out_surf_charge = out_total_force = out_react_force = (
#                 out_coulomb_force
#             ) = out_total_pot = False
#         if j >= 70 and flag is out_total_pot and out_total_pot:
#             out_total_pot = False
#         if j >= 50 and flag is out_surf_charge and out_surf_charge:
#             out_surf_charge = False
#
#     # print("datum:>>>", datum)
#     return frc_header, datum

# def _setup_output_header_strings(
#     out_atom_desc,
#     out_atom_coords,
#     out_charge,
#     out_grid_pot,
#     out_salt,
#     out_react_pot,
#     out_coulomb_pot,
#     out_atom_pot,
#     out_debye_frac,
#     out_field,
#     out_surf_charge,
#     out_total_force,
#     out_react_force,
#     out_coulomb_force,
#     out_total_pot,
# ):
#     """Sets up column/datum header strings for text FRC output.
#
#     The table header uses parser-friendly column names:
#       - atom descriptor is split into ATOM / RESNAME / CHAIN / RESID
#       - potential columns use *_PHI
#       - grid electric field columns use GF_E*
#       - force columns use *_F*
#     """
#     frc_header = " " * 120
#     datum = " " * 80
#     j = 0
#     k = 0
#
#     output_columns_flags = [
#         (
#             out_atom_desc,
#             "ATOM    RESNAME    CHAIN    RESID",
#             "ATOM ",
#             35,
#             5,
#             31,
#         ),
#         (
#             out_atom_coords,
#             "X          Y          Z",
#             "COORDINATES ",
#             36,
#             12,
#             30,
#         ),
#         (
#             out_charge,
#             "CHARGE",
#             "CHARGE ",
#             10,
#             7,
#             6,
#         ),
#         (
#             out_grid_pot,
#             "GRID_PHI",
#             "POTENTIALS ",
#             10,
#             11,
#             8,
#         ),
#         (
#             out_salt,
#             "SALT_CONC",
#             "SALT ",
#             12,
#             5,
#             9,
#         ),
#         (
#             out_react_pot,
#             "RXN_PHI",
#             "REACTION ",
#             10,
#             9,
#             7,
#         ),
#         (
#             out_coulomb_pot,
#             "COUL_PHI",
#             "COULOMBIC ",
#             10,
#             10,
#             8,
#         ),
#         (
#             out_atom_pot,
#             "ATOM_PHI",
#             "ATOMIC PT. ",
#             10,
#             11,
#             8,
#         ),
#         (
#             out_debye_frac,
#             "DEBYE_FRAC",
#             "DEBFRACTION ",
#             14,
#             12,
#             10,
#         ),
#         (
#             out_field,
#             "GF_EX     GF_EY     GF_EZ",
#             "FIELDS ",
#             30,
#             7,
#             25,
#         ),
#         (
#             out_react_force,
#             "RXN_FX    RXN_FY    RXN_FZ",
#             "RFORCE ",
#             30,
#             7,
#             25,
#         ),
#         (
#             out_coulomb_force,
#             "COUL_FX   COUL_FY   COUL_FZ",
#             "CFORCE ",
#             30,
#             7,
#             25,
#         ),
#         (
#             out_total_force,
#             "TOT_FX    TOT_FY    TOT_FZ",
#             "TFORCE ",
#             30,
#             7,
#             25,
#         ),
#         (
#             out_total_pot,
#             "TOTAL_PHI",
#             "TOTAL ",
#             10,
#             6,
#             9,
#         ),
#         (
#             out_surf_charge,
#             "SURF_CHARGE    SURF_X     SURF_Y     SURF_Z     SURF_EN     SURF_E",
#             "SCh, x, y, z, surf En, surf. E",
#             70,
#             35,
#             65,
#         ),
#     ]
#
#     for (
#         flag,
#         column_name,
#         datum_name,
#         column_step,
#         datum_step,
#         column_len,
#     ) in output_columns_flags:
#         if flag:
#             frc_header = frc_header[:j] + column_name + frc_header[j + column_len :]
#             datum = datum[:k] + datum_name + datum[k + datum_step :]
#             j += column_step
#             k += datum_step
#
#         if (
#             j >= 120
#             and (
#                 out_react_pot
#                 or out_coulomb_pot
#                 or out_atom_pot
#                 or out_debye_frac
#                 or out_field
#                 or out_surf_charge
#                 or out_total_force
#                 or out_react_force
#                 or out_coulomb_force
#                 or out_total_pot
#             )
#             and flag not in [out_surf_charge]
#         ):
#             out_react_pot = out_coulomb_pot = out_atom_pot = out_debye_frac = (
#                 out_field
#             ) = out_surf_charge = out_total_force = out_react_force = (
#                 out_coulomb_force
#             ) = out_total_pot = False
#
#         if (
#             j >= 90
#             and flag
#             in [
#                 out_field,
#                 out_react_force,
#                 out_coulomb_force,
#                 out_total_force,
#             ]
#             and (
#                 out_field
#                 or out_surf_charge
#                 or out_total_force
#                 or out_react_force
#                 or out_coulomb_force
#                 or out_total_pot
#             )
#         ):
#             out_field = out_surf_charge = out_total_force = out_react_force = (
#                 out_coulomb_force
#             ) = out_total_pot = False
#
#         if j >= 105 and flag is out_total_pot and out_total_pot:
#             out_total_pot = False
#
#         if j >= 85 and flag is out_surf_charge and out_surf_charge:
#             out_surf_charge = False
#
#     return frc_header.rstrip(), datum.rstrip()


def _setup_output_header_strings(
    out_atom_desc,
    out_atom_coords,
    out_charge,
    out_grid_pot,
    out_salt,
    out_react_pot,
    out_coulomb_pot,
    out_atom_pot,
    out_debye_frac,
    out_field,
    out_surf_charge,
    out_total_force,
    out_react_force,
    out_coulomb_force,
    out_total_pot,
):
    """Sets up column/datum header strings for text FRC output.

    Header widths are aligned with _write_output_values().
    """
    atom_w = 8
    resname_w = 11
    chain_w = 8
    resid_w = 8
    desc_w = atom_w + resname_w + chain_w + resid_w

    scalar_w = 12
    vector_w = 12
    debye_w = 14
    surf_w = 12

    header_parts = []
    datum_parts = []

    if out_atom_desc:
        header_parts.append(
            f"{'ATOM':<{atom_w}}"
            f"{'RESNAME':<{resname_w}}"
            f"{'CHAIN':<{chain_w}}"
            f"{'RESID':<{resid_w}}"
        )
        datum_parts.append("ATOM")

    if out_atom_coords:
        header_parts.append(
            f"{'X':>{vector_w}}" f"{'Y':>{vector_w}}" f"{'Z':>{vector_w}}"
        )
        datum_parts.append("COORDINATES")

    if out_charge:
        header_parts.append(f"{'CHARGE':>{scalar_w}}")
        datum_parts.append("CHARGE")

    if out_grid_pot:
        header_parts.append(f"{'GRID_PHI':>{scalar_w}}")
        datum_parts.append("POTENTIALS")

    if out_salt:
        header_parts.append(f"{'SALT_CONC':>{scalar_w}}")
        datum_parts.append("SALT")

    if out_react_pot:
        header_parts.append(f"{'RXN_PHI':>{scalar_w}}")
        datum_parts.append("REACTION")

    if out_coulomb_pot:
        header_parts.append(f"{'COUL_PHI':>{scalar_w}}")
        datum_parts.append("COULOMBIC")

    if out_atom_pot:
        header_parts.append(f"{'ATOM_PHI':>{scalar_w}}")
        datum_parts.append("ATOMIC_PT")

    if out_debye_frac:
        header_parts.append(f"{'DEBYE_FRAC':>{debye_w}}")
        datum_parts.append("DEBFRACTION")

    if out_field:
        header_parts.append(
            f"{'GF_EX':>{vector_w}}" f"{'GF_EY':>{vector_w}}" f"{'GF_EZ':>{vector_w}}"
        )
        datum_parts.append("FIELDS")

    if out_react_force:
        header_parts.append(
            f"{'RXN_FX':>{vector_w}}"
            f"{'RXN_FY':>{vector_w}}"
            f"{'RXN_FZ':>{vector_w}}"
        )
        datum_parts.append("RXN_FORCE")

    if out_coulomb_force:
        header_parts.append(
            f"{'COUL_FX':>{vector_w}}"
            f"{'COUL_FY':>{vector_w}}"
            f"{'COUL_FZ':>{vector_w}}"
        )
        datum_parts.append("COUL_FORCE")

    if out_total_force:
        header_parts.append(
            f"{'TOT_FX':>{vector_w}}"
            f"{'TOT_FY':>{vector_w}}"
            f"{'TOT_FZ':>{vector_w}}"
        )
        datum_parts.append("TOT_FORCE")

    if out_total_pot:
        header_parts.append(f"{'TOTAL_PHI':>{scalar_w}}")
        datum_parts.append("TOTAL")

    if out_surf_charge:
        header_parts.append(
            f"{'SURF_CHARGE':>{surf_w}}"
            f"{'SURF_X':>{surf_w}}"
            f"{'SURF_Y':>{surf_w}}"
            f"{'SURF_Z':>{surf_w}}"
            f"{'SURF_EN':>{surf_w}}"
            f"{'SURF_E':>{surf_w}}"
        )
        datum_parts.append("SURFACE_CHARGE")

    frc_header = "".join(header_parts).rstrip()
    datum = " ".join(datum_parts)

    return frc_header, datum


def write_frc_file(
    output_frc_file,
    frc_atoms_dict,
    grid_shape,
    percentage_fill,
    external_dielectric,
    media_eps,
    gap_dielectric,
    dielectric_model,
    surface_method,
    epkt,
    ion_strength,
    ion_radius,
    probe_radii,
    total_iters,
    final_rms,
    final_dphi,
    convergence_status,
    boundary_type,
    file_map_record,
    potential_upper_bond,
    out_atom_desc=False,
    out_salt=False,
    out_md=False,
    out_pot=False,
    out_atom_coords=False,
    out_charge=False,
    out_field=False,
    out_grid_pot=False,
    out_react_pot=False,
    out_coulomb_pot=False,
    out_atom_pot=False,
    out_debye_frac=False,
    out_surf_charge=False,
    out_total_force=False,
    out_react_force=False,
    out_total_pot=False,
    out_coulomb_force=False,
    box_center=np.array([0.0, 0.0, 0.0]),
    grid_offset=np.array([0.0, 0.0, 0.0]),
    scale_factor=1.0,
    potential_map=None,
    dielectric_map_bool=None,
    num_surface_charges=0,
    surface_charge_pos_array=None,
    surface_charge_e_array=None,
    boundary_grid_array=None,
    charge_grid_num=0,
    charge_pos_array=None,
    atomic_charge_list=None,
    atom_eps_array=None,
    residue_num=0,
    taylor_coeffs=np.zeros(5, dtype=delphi_real),
):
    """
    Writes an FRC file containing site potentials and/or fields and/or atom information.
    """
    taylor_coeff1 = taylor_coeffs[0]
    taylor_coeff2 = taylor_coeffs[1]
    taylor_coeff3 = taylor_coeffs[2]
    taylor_coeff4 = taylor_coeffs[3]
    taylor_coeff5 = taylor_coeffs[4]

    # Legacy helper compatibility:
    # _calculate_grid_potential_and_salt() still expects this name. The new
    # FRC header reports solver-level total_iters/final_rms/final_dphi/status
    # instead, so do not reuse total_iters here as a nonlinear iteration count.
    non_linear_iteration_num = 0

    custom_output_specified = (
        out_atom_desc
        or out_charge
        or out_grid_pot
        or out_field
        or out_react_pot
        or out_total_pot
        or out_coulomb_pot
        or out_atom_coords
        or out_salt
        or out_react_force
        or out_coulomb_force
        or out_atom_pot
        or out_total_force
        or out_debye_frac
    )

    if not custom_output_specified:
        out_atom_coords = True
        out_charge = True
        out_field = True
        out_grid_pot = True

    column_header, datum_header = _setup_output_header_strings(
        out_atom_desc=out_atom_desc,
        out_atom_coords=out_atom_coords,
        out_charge=out_charge,
        out_grid_pot=out_grid_pot,
        out_salt=out_salt,
        out_react_pot=out_react_pot,
        out_coulomb_pot=out_coulomb_pot,
        out_atom_pot=out_atom_pot,
        out_debye_frac=out_debye_frac,
        out_field=out_field,
        out_surf_charge=out_surf_charge,
        out_total_force=out_total_force,
        out_react_force=out_react_force,
        out_coulomb_force=out_coulomb_force,
        out_total_pot=out_total_pot,
    )

    is_quality_assurance_step = True
    residue_surface_flags = [False] * residue_num

    if not (out_md or out_pot):
        vprint(DEBUG, _VERBOSITY, "\nwriting potentials at given sites...")

    output_file_stream = None

    try:
        output_mode = "w"
        output_file_stream = open(output_frc_file, output_mode)
        if not output_file_stream:
            raise Exception(f"Could not open output file: {output_frc_file}")

        _write_text_frc_header(
            outfile_stream=output_file_stream,
            grid_shape=grid_shape,
            percent_fill=percentage_fill,
            external_dielectric=external_dielectric,
            media_epsilons=media_eps,
            gap_dielectric=gap_dielectric,
            dielectric_model=dielectric_model,
            surface_method=surface_method,
            ion_strength=ion_strength,
            ion_radius=ion_radius,
            probe_radius=probe_radii,
            total_iters=total_iters,
            final_rms=final_rms,
            final_dphi=final_dphi,
            convergence_status=convergence_status,
            boundary_condition=boundary_type,
            datum=datum_header,
            map_title=file_map_record,
            vrow=column_header,
        )

        if not (out_react_force or out_md or out_total_force):
            out_react_force = out_total_force = out_md = False

        rfield_data = []
        media_num = 1
        if out_react_force or out_md or out_total_force:
            if (
                1 == media_num
                and media_eps
                and len(media_eps) > 1
                and abs(media_eps[1] * epkt - 1.0) < 1e-6
            ):
                rfield_data = rforceeps1()
            else:
                rfield_data = rforce()

        grid_offset = ((grid_shape - 1.0) / 2.0)[:]  # Note: python has 0-based index
        num_atoms_processed = 0
        total_electrostatic_energy = 0.0

        if not out_pot:
            for atom_key, atom_data in frc_atoms_dict.items():
                num_atoms_processed += 1

                (
                    str_head,
                    _,
                    _,
                    atom_name,
                    residue_name,
                    chain_name,
                    residue_number,
                    _,
                    segid,
                    atomic_number,
                ) = atom_key
                atom_coords = atom_data[ATOMFIELD_X : ATOMFIELD_Z + 1]

                grid_coords = (atom_coords - box_center) * scale_factor + grid_offset

                atom_descriptor = (
                    f"{atom_name:<8s}"
                    f"{residue_name:<11s}"
                    f"{chain_name:<8s}"
                    f"{residue_number:<8d}"
                    if out_atom_desc
                    else None
                )

                charge_value = (
                    atom_data[ATOMFIELD_CHARGE]
                    if (out_charge and is_quality_assurance_step)
                    or out_atom_pot
                    or out_grid_pot
                    else None
                )
                atom_radius = atom_data[ATOMFIELD_RADIUS] if out_atom_pot else None

                if out_surf_charge:
                    try:
                        iresnum = int(residue_number)
                        residue_atom_surface_flags = [False] * residue_num
                        residue_atom_surface_flags[num_atoms_processed - 1] = (
                            residue_surface_flags[iresnum - 1]
                            if 0 < iresnum <= len(residue_surface_flags)
                            else False
                        )
                    except ValueError:
                        residue_atom_surface_flags = [False] * residue_num
                        residue_atom_surface_flags[num_atoms_processed - 1] = False

                atom_potential_value = (
                    _calculate_atom_potential(
                        grid_coords=grid_coords,
                        atom_radius=atom_radius,
                        charge_value=charge_value,
                        potential_upper_bond=potential_upper_bond,
                        grid_shape=grid_shape,
                        potential_map=potential_map,
                        scale_factor=scale_factor,
                    )
                    if out_atom_pot
                    else None
                )

                potential_value, salt_concentration = (
                    _calculate_grid_potential_and_salt(
                        grid_coords=grid_coords,
                        grid_shape=grid_shape,
                        potential_map=potential_map,
                        output_salt_concentration=out_salt,
                        non_linear_iteration_num=non_linear_iteration_num,
                        ion_strength=ion_strength,
                        taylor_coeff1=taylor_coeff1,
                        taylor_coeff2=taylor_coeff2,
                        taylor_coeff3=taylor_coeff3,
                        taylor_coeff4=taylor_coeff4,
                        taylor_coeff5=taylor_coeff5,
                    )
                    if out_grid_pot
                    or out_salt
                    or (out_atom_pot and atom_potential_value == 0.0)
                    else (None, None)
                )

                if potential_value is not None and charge_value is not None:
                    total_electrostatic_energy += potential_value * charge_value

                debye_fraction = (
                    _calculate_debye_fraction(
                        grid_coords=grid_coords,
                        grid_shape=grid_shape,
                        dielectric_map_bool=dielectric_map_bool,
                        output_debye_fraction_value=out_debye_frac,
                        verbose=_VERBOSITY <= DEBUG,
                    )
                    if out_debye_frac
                    else None
                )

                field_xyz = (
                    _calculate_field_xyz(
                        grid_coords=grid_coords,
                        grid_shape=grid_shape,
                        potential_map=potential_map,
                        scale_factor=scale_factor,
                    )
                    if out_field
                    else None
                )

                reaction_potential = None
                surface_potential_term = None
                coulomb_potential = None
                atomic_coulomb_potential = None
                sold = None
                total_potential = None
                total_surface_charge_value = None
                total_force_xyz = None
                reaction_force_xyz = None
                coulomb_force_xyz = None

                _write_output_values(
                    output_file_stream=output_file_stream,
                    out_react_pot=out_react_pot,
                    out_coulomb_pot=out_coulomb_pot,
                    out_atom_pot=out_atom_pot,
                    out_debye_frac=out_debye_frac,
                    out_field=out_field,
                    out_surf_charge=out_surf_charge,
                    out_total_force=out_total_force,
                    out_react_force=out_react_force,
                    out_coulomb_force=out_coulomb_force,
                    out_total_pot=out_total_pot,
                    out_atom_desc=out_atom_desc,
                    out_atom_coords=out_atom_coords,
                    out_charge=out_charge,
                    out_grid_pot=out_grid_pot,
                    out_salt=out_salt,
                    epkt_value=epkt,
                    reaction_potential=reaction_potential,
                    coulomb_potential=coulomb_potential,
                    atom_potential_value=atom_potential_value,
                    debye_fraction=debye_fraction,
                    field_xyz=field_xyz,
                    total_surface_charge_value=total_surface_charge_value,
                    atom_coords=atom_coords,
                    surface_potential_term=surface_potential_term,
                    total_force_xyz=total_force_xyz,
                    reaction_force_xyz=reaction_force_xyz,
                    coulomb_force_xyz=coulomb_force_xyz,
                    total_potential=total_potential,
                    atom_descriptor=atom_descriptor,
                    charge_value=charge_value,
                    potential_value=potential_value,
                    salt_concentration=salt_concentration,
                )

            # Electrostatic energy is 0.5 * sum(q_i * phi_i) to avoid double-counting
            # pairwise charge interactions.
            total_electrostatic_energy *= 0.5
            output_file_stream.write(
                f"# Total electrostatic energy = {total_electrostatic_energy:.4f} kT\n"
            )

    except FileNotFoundError as e:
        missing_path = e.filename if e.filename else output_frc_file
        vprint(
            ERROR,
            _VERBOSITY,
            f"FRC output failed: required file or directory was not found: {missing_path}",
        )
        sys_exit(1)

    except PermissionError as e:
        target_path = e.filename if e.filename else output_frc_file
        vprint(
            ERROR,
            _VERBOSITY,
            f"FRC output failed: permission denied: {target_path}",
        )
        sys_exit(1)

    except OSError as e:
        vprint(
            ERROR,
            _VERBOSITY,
            f"FRC output failed: could not write '{output_frc_file}': {e}",
        )
        sys_exit(1)

    except Exception as e:
        vprint(
            ERROR,
            _VERBOSITY,
            f"FRC output failed while writing '{output_frc_file}': {e}",
        )
        sys_exit(1)
    finally:
        if output_file_stream:
            output_file_stream.close()
