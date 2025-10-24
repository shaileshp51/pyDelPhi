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

import numpy as np


def write_zeta_phi(
    zeta_filename,
    grid_center,
    surf_grid_coords,
    surf_grid_index,
    num_surf_grid_coords,
    phimap,
):
    """
    Writes the surface potential values to a specified output file in a formatted manner.

    Parameters:
        zeta_filename (str): Name of the output file to write the surface potentials.
        grid_center (list or tuple): A 3-element list or tuple representing the grid's geometric center in Angstroms.
        surf_grid_coords (list): A list of 3D coordinates (x, y, z) representing surface grid points. Must be a multiple of 3.
        surf_grid_index (list): A list of indices for the surface grid points in the `phimap` array. Must be a multiple of 3.
        phimap (numpy.ndarray): A 3D numpy array representing the potential map, indexed by [x][y][z].

    Raises:
        ValueError: If `surf_grid_coords` or `surf_grid_index` are not valid.
    """
    MAXWIDTH = 55
    surf_grid_potentials = []

    if num_surf_grid_coords == 0:
        raise ValueError(
            "Zeta surface potentials must have some zeta surface points defined, found zero. Check Inputs."
        )

    with open(zeta_filename, "w") as out_file:
        # Write initial remark
        out_file.write(
            "# REMARK DATA: POINT-COORDINATES(X,Y,Z) FOLLOWED BY POTENTIAL\n"
        )
        out_file.write(
            "# REMARK DATAFORMAT: {px:>13.6f},{py:>13.6f},{pz:>13.6f},{potential:>15.6f}\n"
        )
        out_file.write(
            "# REMARK ADDITIONAL IMPORTANT INFORMATION AT THE BOTTOM OF THIS FILE\n"
        )

        for id_point in range(0, num_surf_grid_coords):
            idx = id_point * 3
            # Extract indices for potential values
            ix, jy, kz = surf_grid_index[idx : idx + 3]

            # Append the corresponding potential from `phimap`
            try:
                potential = phimap[ix][jy][kz]
                surf_grid_potentials.append(potential)
            except IndexError:
                raise ValueError(
                    f"Index ({ix}, {jy}, {kz}) is out of bounds for the provided phimap array."
                )

            # Extract corresponding coordinates
            px, py, pz = surf_grid_coords[idx : idx + 3]

            # Write coordinates and potential to file
            out_file.write(f"{px:>13.6f},{py:>13.6f},{pz:>13.6f},{potential:>15.6f}\n")

        # Separator line
        out_file.write("# " + "-" * MAXWIDTH + "\n")

        # Calculate the mean surface potential
        simple_avg_surf_potential = (
            np.mean(surf_grid_potentials) if len(surf_grid_potentials) else 0.0
        )

        # Write remarks for average potential and grid center
        out_file.write(
            f"{'# REMARK SIMPLE AVERAGE SURFACE POTENTIAL':<{MAXWIDTH}} = {simple_avg_surf_potential:.6f} kT/e\n"
        )
        out_file.write(
            f"{'# REMARK GRIDBOX GEOMETRIC CENTER (ANG)':<{MAXWIDTH}} = {grid_center[0]:.6f}  {grid_center[1]:.6f}  {grid_center[2]:.6f}\n"
        )


def write_grid_charges(
    filename: str,
    scale: float,
    grid_origin: np.ndarray,
    grid_shape: np.ndarray,
    unique_charged_gridpoints: np.ndarray,
    include_grid_indices: bool = True,  # New optional argument
):
    """
    Write the real indices and coordinates of unique charged grids in pydelphi.

    Args:
        filename (str): A string representing the output filename.
        scale (float): The grid scale factor, representing "grids per angstrom".
                       (e.g., 2.0 means 2 grid points per angstrom).
        grid_origin (np.ndarray): Array representing the origin of the grid (x0, y0, z0).
        grid_shape (np.ndarray): Array representing the shape of the grid (nx, ny, nz).
        unique_charged_gridpoints (np.ndarray): A 2D array of shape (K, 5) containing
                                                [index_1d, total_charge, ix, iy, iz]
                                                of unique charged grid points, sorted by index_1d.
        include_grid_indices (bool, optional): If True (default), include 'ix', 'iy', 'iz'
                                               columns in the output file. If False,
                                               only 'x', 'y', 'z', 'charge' are written.

    Data is written in tab-delimited form.
    If include_grid_indices is True, columns are:
        ix, iy, iz, x (Angstroms), y (Angstroms), z (Angstroms), total_charge.
        Format: {:>5d}\t{>:5d}\t{>:5d}\t{:>10.4f}\t{:>10.4f}\t{:>10.4f}\t{:>13.6g}
    If include_grid_indices is False, columns are:
        x (Angstroms), y (Angstroms), z (Angstroms), total_charge.
        Format: {:>10.4f}\t{:>10.4f}\t{:>10.4f}\t{:>10.4f}\t{:>13.6g}
    """
    try:
        with open(filename, "w") as f:
            # Calculate grid spacing from scale
            if scale <= 0:
                raise ValueError(
                    "Scale must be a positive value ('grids per angstrom')."
                )
            grid_spacing = 1.0 / scale  # Angstroms per grid point

            # Write header information
            f.write(f"# Grid Scale (grids/Angstrom): {scale}\n")
            f.write(f"# Grid Spacing (Angstroms/grid): {grid_spacing:.4f}\n")
            f.write(
                f"# Grid Origin (x0, y0, z0): {grid_origin[0]:.4f}\t{grid_origin[1]:.4f}\t{grid_origin[2]:.4f}\n"
            )
            f.write(
                f"# Grid Shape (nx, ny, nz): {int(grid_shape[0])}\t{int(grid_shape[1])}\t{int(grid_shape[2])}\n"
            )
            f.write("#\n")  # Separator for clarity

            # Determine column headers and format string based on include_grid_indices
            if include_grid_indices:
                header_line = f"{'ix':>5s}\t{'iy':>5s}\t{'iz':>5s}\t{'x':>10s}\t{'y':>10s}\t{'z':>10s}\t{'charge':>13s}\n"
                data_format_string = (
                    f"{{:>5d}}\t"  # ix
                    f"{{:>5d}}\t"  # iy
                    f"{{:>5d}}\t"  # iz
                    f"{{:>10.4f}}\t"  # x
                    f"{{:>10.4f}}\t"  # y
                    f"{{:>10.4f}}\t"  # z
                    f"{{:>13.6g}}\n"  # charge
                )
            else:
                header_line = f"{'x':>10s}\t{'y':>10s}\t{'z':>10s}\t{'charge':>13s}\n"
                data_format_string = (
                    f"{{:>10.4f}}\t"  # x
                    f"{{:>10.4f}}\t"  # y
                    f"{{:>10.4f}}\t"  # z
                    f"{{:>13.6g}}\n"  # charge
                )
            f.write(header_line)

            # Iterate through each unique charged grid point
            for grid_point in unique_charged_gridpoints:
                # Skip index_1d, extract: total_charge, ix, iy, iz
                total_charge = grid_point[1]
                ix = int(grid_point[2])
                iy = int(grid_point[3])
                iz = int(grid_point[4])

                # Convert grid indices to real-space coordinates using grid_spacing
                x = grid_origin[0] + ix * grid_spacing
                y = grid_origin[1] + iy * grid_spacing
                z = grid_origin[2] + iz * grid_spacing

                # Prepare values for formatting based on include_grid_indices
                if include_grid_indices:
                    values_to_format = (ix, iy, iz, x, y, z, total_charge)
                else:
                    values_to_format = (x, y, z, total_charge)

                f.write(data_format_string.format(*values_to_format))

        print(f"Successfully wrote grid charges to {filename}")

    except ValueError as e:
        print(f"Input Error: {e}")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def write_induced_surface_charges(
    filename: str,
    scale: float,
    grid_origin: np.ndarray,
    grid_shape: np.ndarray,
    induced_surf_charges_flat: np.ndarray,
    include_grid_indices: bool = True,  # New optional argument
):
    """
    Write the real indices and coordinates of induced surface charges.

    Args:
        filename (str): A string representing the output filename.
        scale (float): The grid scale factor, representing "grids per angstrom".
                       (e.g., 2.0 means 2 grid points per angstrom).
        grid_origin (np.ndarray): Array representing the origin of the grid (x0, y0, z0).
        grid_shape (np.ndarray): Array representing the shape of the grid (nx, ny, nz).
        induced_surf_charges_flat (np.ndarray): A 1D array of shape (M * 4) containing
                                                [ix, iy, iz, charge] for each boundary grid point.
                                                This array is assumed to be flat, e.g.,
                                                [ix1, iy1, iz1, q1, ix2, iy2, iz2, q2, ...].
        include_grid_indices (bool, optional): If True (default), include 'ix', 'iy', 'iz'
                                               columns in the output file. If False,
                                               only 'x', 'y', 'z', 'charge' are written.

    Data is written in tab-delimited form.
    If include_grid_indices is True, columns are:
        ix, iy, iz, x (Angstroms), y (Angstroms), z (Angstroms), total_charge.
        Format: {:>5d}\t{>:5d}\t{>:5d}\t{:>10.4f}\t{:>10.4f}\t{:>10.4f}\t{:>13.6g}
    If include_grid_indices is False, columns are:
        x (Angstroms), y (Angstroms), z (Angstroms), total_charge.
        Format: {:>10.4f}\t{:>10.4f}\t{:>10.4f}\t{:>13.6g}
    """
    try:
        with open(filename, "w") as f:
            # Calculate grid spacing from scale
            if scale <= 0:
                raise ValueError(
                    "Scale must be a positive value ('grids per angstrom')."
                )
            grid_spacing = 1.0 / scale  # Angstroms per grid point

            # Write header information
            f.write(f"# Grid Scale (grids/Angstrom): {scale}\n")
            f.write(f"# Grid Spacing (Angstroms/grid): {grid_spacing:.4f}\n")
            f.write(
                f"# Grid Origin (x0, y0, z0): {grid_origin[0]:.4f}\t{grid_origin[1]:.4f}\t{grid_origin[2]:.4f}\n"
            )
            f.write(
                f"# Grid Shape (nx, ny, nz): {int(grid_shape[0])}\t{int(grid_shape[1])}\t{int(grid_shape[2])}\n"
            )
            f.write("#\n")  # Separator for clarity

            # Determine column headers and format string based on include_grid_indices
            if include_grid_indices:
                header_line = f"{'ix':>5s}\t{'iy':>5s}\t{'iz':>5s}\t{'x':>10s}\t{'y':>10s}\t{'z':>10s}\t{'charge':>13s}\n"
                data_format_string = (
                    f"{{:>5d}}\t"  # ix
                    f"{{:>5d}}\t"  # iy
                    f"{{:>5d}}\t"  # iz
                    f"{{:>10.4f}}\t"  # x
                    f"{{:>10.4f}}\t"  # y
                    f"{{:>10.4f}}\t"  # z
                    f"{{:>13.6g}}\n"  # charge
                )
            else:
                header_line = f"{'x':>10s}\t{'y':>10s}\t{'z':>10s}\t{'charge':>13s}\n"
                data_format_string = (
                    f"{{:>10.4f}}\t"  # x
                    f"{{:>10.4f}}\t"  # y
                    f"{{:>10.4f}}\t"  # z
                    f"{{:>13.6g}}\n"  # charge
                )
            f.write(header_line)

            # Reshape the flat array for easier iteration
            # Ensure it's empty if original is empty to avoid reshape errors
            if induced_surf_charges_flat.size == 0:
                reshaped_charges = np.array([]).reshape(0, 4)
            else:
                if induced_surf_charges_flat.size % 4 != 0:
                    raise ValueError(
                        "induced_surf_charges_flat must have a size divisible by 4 (ix, iy, iz, charge)."
                    )
                reshaped_charges = induced_surf_charges_flat.reshape(-1, 4)

            # Iterate through each boundary grid point with its charge
            for grid_point_info in reshaped_charges:
                ix = int(grid_point_info[0])
                iy = int(grid_point_info[1])
                iz = int(grid_point_info[2])
                charge = grid_point_info[3]

                # Convert grid indices to real-space coordinates
                x = grid_origin[0] + ix * grid_spacing
                y = grid_origin[1] + iy * grid_spacing
                z = grid_origin[2] + iz * grid_spacing

                # Prepare values for formatting
                if include_grid_indices:
                    values_to_format = (ix, iy, iz, x, y, z, charge)
                else:
                    values_to_format = (x, y, z, charge)

                f.write(data_format_string.format(*values_to_format))

        print(f"Successfully wrote induced surface charges to {filename}")

    except ValueError as e:
        print(f"Input Error: {e}")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def extract_nested(d, keys):
    """Safely extract a nested dictionary value given a dotted key. Returns None if missing."""
    for key in keys.split("."):
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def write_energies_to_tsv(
    energies,
    energy_outfile,
    run_label,
    key_mapping,
    frame=None,
    write_mode="a",
    only_phase=False,
    write_header=False,
):
    """
    Writes energy terms to a TSV file, dynamically omitting missing terms and optionally
    excluding phase-specific terms.

    Parameters:
        energies: dict — Nested energy results.
        args: argparse.Namespace — CLI args with outfile, label, etc.
        key_mapping: dict — Maps nested keys (dot notation) to column names.
        frame: Optional[int] — Frame number for trajectory output.
        write_mode: str — 'w' or 'a'; controls file open mode.
        only_phase: bool — If True, write only total terms.
        write_header: bool — If True, writes the column header and key comment.
    """
    available_items = []
    for full_key, column_label in key_mapping.items():
        if only_phase and not full_key.startswith("total."):
            continue

        value = extract_nested(energies, full_key)
        if value is not None:
            available_items.append((full_key, column_label, value))

    with open(energy_outfile, write_mode) as fout:
        if write_header:
            columns = []
            if frame is not None:
                columns.append("FRAME")
            columns.append("LABEL")
            columns.extend(col for _, col, _ in available_items)

            # Comment with full keys for clarity
            comment = "# " + " | ".join(
                f"{col}: {key}" for key, col, _ in available_items
            )
            fout.write(comment + "\n")
            fout.write("\t".join(columns) + "\n")

        # Write the data row
        row = []
        if frame is not None:
            row.append(str(frame))
        row.append(run_label)
        row.extend(f"{v:.4f}" for _, _, v in available_items)

        fout.write("\t".join(row) + "\n")

    # print(f"[Frame {frame if frame is not None else 0}] Results written to {args.outfile}")
