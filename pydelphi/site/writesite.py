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

from pydelphi.config.logging_config import INFO, DEBUG, get_effective_verbosity

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


def _write_text_frc_header(
    outfile_stream,
    grid_shape,
    percent_fill,
    external_dielectric,
    media_epsilons,
    epkt_value,
    ion_strength,
    ion_radius,
    probe_radius,
    linear_iteration_num,
    non_linear_iteration_num,
    boundary_condition,
    datum,
    map_title,
    vrow,
):
    """Writes the header for a text format FRC file."""
    outfile_stream.write("DELPHI SITE POTENTIAL FILE\n")
    outfile_stream.write(
        f"grid size, percent fill:   {grid_shape}    {percent_fill:.3f}\n"
    )
    internal_epsilon_first = (
        media_epsilons[1]
        if (media_epsilons is not None) and len(media_epsilons) > 1
        else 1.0
    )
    outfile_stream.write(
        "outer diel. and first one assigned :   "
        f"{external_dielectric:.2f}    "
        f"{internal_epsilon_first:.2f}\n"
    )
    outfile_stream.write(f"ionic strength (M):   {ion_strength}\n")

    if isinstance(probe_radius, (list, tuple, np.ndarray)):
        probe_radius_1 = probe_radius[0] if len(probe_radius) > 0 else 1.4
        probe_radius_2 = probe_radius[1] if len(probe_radius) > 1 else probe_radius_1
    elif isinstance(probe_radius, (int, float)):
        probe_radius_1 = probe_radius
        probe_radius_2 = (
            probe_radius_1  # Or decide what the second radius should be for scalar case
        )
    else:
        probe_radius_1 = 0.0
        probe_radius_2 = 0.0

    outfile_stream.write(
        f"ion excl., probe radii:   {ion_radius}    {probe_radius_1}    {probe_radius_2}\n"
    )

    outfile_stream.write(
        f"linear, nolinear iterations:   {linear_iteration_num}    {non_linear_iteration_num}\n"
    )
    outfile_stream.write(f"boundary condition:   {boundary_condition}\n")
    outfile_stream.write(f"Data Output:   {datum}\n")
    outfile_stream.write(f"title: {map_title}\n")
    outfile_stream.write("\n\n")
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
    """Writes output values in formatted text mode."""
    if out_atom_desc and atom_descriptor is not None:
        output_file_stream.write(f"{atom_descriptor}")
    if out_atom_coords and atom_coords is not None:
        output_file_stream.write(
            f"{atom_coords[0]:10.4f}{atom_coords[1]:10.4f}{atom_coords[2]:10.4f}"
        )
    if out_charge and charge_value is not None:
        output_file_stream.write(f"{charge_value:10.4f}")

    # print("site: 432>>> ", potential_value)
    if out_grid_pot and potential_value is not None:
        output_file_stream.write(f"{potential_value:10.4f}")
    if out_salt and salt_concentration is not None:
        output_file_stream.write(f"{salt_concentration:10.4f}")
    if out_react_pot and reaction_potential is not None:
        output_file_stream.write(f"{reaction_potential:10.4f}")
    if out_coulomb_pot and coulomb_potential is not None:
        output_file_stream.write(f"{coulomb_potential:10.4f}")
    if out_atom_pot and atom_potential_value is not None:
        output_file_stream.write(f"{atom_potential_value:10.4f}")

    if out_debye_frac and debye_fraction is not None:
        output_file_stream.write(f"{debye_fraction:10.4f}")
    if out_field and field_xyz is not None:
        output_file_stream.write(
            f"{field_xyz[0]:10.4f}{field_xyz[1]:10.4f}{field_xyz[2]:10.4f}"
        )

    if out_react_force and reaction_force_xyz is not None:
        output_file_stream.write(
            f"{reaction_force_xyz[0]:10.4f}{reaction_force_xyz[1]:10.4f}{reaction_force_xyz[2]:10.4f}"
        )
    if out_coulomb_force and coulomb_force_xyz is not None:
        output_file_stream.write(
            f"{coulomb_force_xyz[0]:10.4f}{coulomb_force_xyz[1]:10.4f}{coulomb_force_xyz[2]:10.4f}"
        )
    if out_total_force and total_force_xyz is not None:
        output_file_stream.write(
            f"{total_force_xyz[0]:10.4f}{total_force_xyz[1]:10.4f}{total_force_xyz[2]:10.4f}"
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
            f"{total_surface_charge_value:10.4f} {atom_coords[0]:10.4f} {atom_coords[1]:10.4f} {atom_coords[2]:10.4f} {surface_potential_term:10.4f} {surface_potential_term / epkt_value:10.4f}"
        )
    output_file_stream.write("\n")


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
    """Sets up the column/datum header strings for output frc file based on output flags."""
    frc_header = " " * 80
    datum = " " * 65
    j = 0
    k = 0
    output_columns_flags = [
        (out_atom_desc, "ATOM DESCRIPTOR", "ATOM ", 20, 5, 15),
        (
            out_atom_coords,
            "ATOM COORDINATES (X,Y,Z)",
            "COORDINATES ",
            30,
            12,
            24,
        ),
        (out_charge, "CHARGE", "CHARGE ", 10, 7, 6),
        (out_grid_pot, "GRID PT.", "POTENTIALS ", 10, 11, 8),
        (out_salt, "SALT CON", "SALT ", 10, 5, 8),
        (out_react_pot, " REAC. PT.", "REACTION ", 10, 9, 10),
        (out_coulomb_pot, " COUL. POT", "COULOMBIC ", 10, 10, 10),
        (out_atom_pot, "ATOM PT.", "ATOMIC PT. ", 10, 11, 8),
        (out_debye_frac, "DEBFRACTION", "DEBFRACTION ", 14, 12, 11),
        (out_field, "GRID FIELDS: (Ex, Ey, Ez)", "FIELDS ", 30, 7, 25),
        (out_react_force, "REAC. FORCE: (Rx, Ry, Rz)", "RFORCE ", 30, 7, 25),
        (out_coulomb_force, "COUL. FORCE: (Cx, Cy, Cz)", "CFORCE ", 30, 7, 25),
        (out_total_force, "TOTAL FORCE: (Tx, Ty, Tz)", "TFORCE ", 30, 7, 25),
        (out_total_pot, " TOTAL", "TOTAL ", 10, 6, 6),
        (
            out_surf_charge,
            "sCharge,    x          y       z       urf.E°n,surf. E[kT/(qA)]",
            "SCh, x, y, z, surf En, surf. E",
            50,
            35,
            65,
        ),
    ]

    for (
        flag,
        column_name,
        datum_name,
        column_start_index,
        datum_start_index,
        column_len,
    ) in output_columns_flags:
        if flag:
            frc_header = frc_header[:j] + column_name + frc_header[j + column_len :]
            datum = datum[:k] + datum_name + datum[k + datum_start_index :]
            j += column_start_index
            k += datum_start_index
        if (
            j >= 80
            and (
                out_react_pot
                or out_coulomb_pot
                or out_atom_pot
                or out_debye_frac
                or out_field
                or out_surf_charge
                or out_total_force
                or out_react_force
                or out_coulomb_force
                or out_total_pot
            )
            and flag not in [out_surf_charge]
        ):
            out_react_pot = out_coulomb_pot = out_atom_pot = out_debye_frac = (
                out_field
            ) = out_surf_charge = out_total_force = out_react_force = (
                out_coulomb_force
            ) = out_total_pot = False
        if (
            j >= 60
            and flag
            in [
                out_field,
                out_react_force,
                out_coulomb_force,
                out_total_force,
            ]
            and (
                out_field
                or out_surf_charge
                or out_total_force
                or out_react_force
                or out_coulomb_force
                or out_total_pot
            )
        ):
            out_field = out_surf_charge = out_total_force = out_react_force = (
                out_coulomb_force
            ) = out_total_pot = False
        if j >= 70 and flag is out_total_pot and out_total_pot:
            out_total_pot = False
        if j >= 50 and flag is out_surf_charge and out_surf_charge:
            out_surf_charge = False

    # print("datum:>>>", datum)
    return frc_header, datum


def write_frc_file(
    output_frc_file,
    frc_atoms_dict,
    grid_shape,
    percentage_fill,
    external_dielectric,
    media_eps,
    epkt,
    ion_strength,
    ion_radius,
    linear_iteration_num,
    non_linear_iteration_num,
    boundary_type,
    file_map_record,
    probe_radii,
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
            epkt_value=epkt,
            ion_strength=ion_strength,
            ion_radius=ion_radius,
            probe_radius=probe_radii,
            linear_iteration_num=linear_iteration_num,
            non_linear_iteration_num=non_linear_iteration_num,
            boundary_condition=boundary_type,
            datum=datum_header,
            map_title=file_map_record,
            vrow=column_header,
        )

        if not (out_react_force or out_md or out_total_force):
            out_react_force = out_total_force = out_md = False

        rfield_data = []
        if out_react_force or out_md or out_total_force:
            if (
                1 == media_num
                and abs(media_eps[1] * epkt - 1.0) < 1e-6
                and media_eps
                and len(media_eps) > 1
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
                ) = atom_key
                atom_coords = atom_data[ATOMFIELD_X : ATOMFIELD_Z + 1]

                grid_coords = (atom_coords - box_center) * scale_factor + grid_offset
                atom_descriptor = (
                    f"{atom_name:<5s}{residue_name:<5s}{chain_name:<2s}{residue_number:<4d}"
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
            total_electrostatic_energy *= 0.5
            output_file_stream.write(
                f"Total energy = {total_electrostatic_energy:.4f} kT\n"
            )

    # except FileNotFoundError as e:
    #     vprint(PRINT_MANDATORY, f"File not found: {e.filename}")
    # except Exception as e:
    #     vprint(PRINT_MANDATORY, f"Error writing FRC file: {e}")
    finally:
        if output_file_stream:
            output_file_stream.close()
