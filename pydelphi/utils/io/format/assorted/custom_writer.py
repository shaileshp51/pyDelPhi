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


import numpy as np


# ==============================================================================
# ZPHI sparse offset-surface potential writer
# ==============================================================================

from typing import Dict, Iterable, Optional, Tuple, Union
from collections import OrderedDict


ZPHI_VERSION = 1
ZPHI_DEFAULT_CONTENT = "offset_surface_potential"
ZPHI_DEFAULT_COORD_UNITS = "angstrom"
ZPHI_DEFAULT_POTENTIAL_UNITS = "kT/e"
ZPHI_DEFAULT_DENSE_MISSING_VALUE = 1.0e30


def _zphi_as_3_int_array(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.int64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _zphi_as_3_float_array(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {arr.shape}.")
    return arr


def _zphi_format_vec(values: Iterable[Union[int, float]], fmt: str = "{:.6f}") -> str:
    parts = []
    for value in values:
        if isinstance(value, (int, np.integer)):
            parts.append(str(int(value)))
        else:
            parts.append(fmt.format(float(value)))
    return " ".join(parts)


def _zphi_format_potential(value: float) -> str:
    """
    Format potential compactly for .zphi rows and scalar metadata.

    Prefer fixed-width 10.6f for readability. If the value does not fit,
    fall back to scientific notation.
    """
    fixed = f"{float(value):10.6f}"
    if len(fixed) <= 10:
        return fixed
    return f"{float(value):10.6e}"


def compute_zphi_grid_origin_from_center(
    grid_center_ang: Union[np.ndarray, Tuple[float, float, float]],
    grid_shape: Union[np.ndarray, Tuple[int, int, int]],
    grid_spacing_ang: float,
) -> np.ndarray:
    """
    Compute grid origin from center, shape, and spacing.

    origin = center - spacing * (shape - 1) / 2
    """
    grid_center_ang = _zphi_as_3_float_array(grid_center_ang, "grid_center_ang")
    grid_shape = _zphi_as_3_int_array(grid_shape, "grid_shape")
    return (
        grid_center_ang - grid_spacing_ang * (grid_shape.astype(np.float64) - 1.0) / 2.0
    )


def normalize_zphi_surface_indices(
    surf_grid_indices: Union[np.ndarray, Iterable[int]],
    num_surf_grid_points: Optional[int] = None,
) -> np.ndarray:
    """
    Normalize flattened or (N, 3) surface indices to shape (N, 3), dtype int64.
    """
    indices = np.asarray(surf_grid_indices, dtype=np.int64)

    if indices.ndim == 1:
        if indices.size % 3 != 0:
            raise ValueError(
                f"Flattened surf_grid_indices size must be a multiple of 3, got {indices.size}."
            )
        n_available = indices.size // 3
        if num_surf_grid_points is None:
            n = n_available
        else:
            n = int(num_surf_grid_points)
            if n < 0 or n > n_available:
                raise ValueError(
                    f"num_surf_grid_points={n} is invalid for {n_available} available points."
                )
        indices = indices[: 3 * n].reshape((n, 3))

    elif indices.ndim == 2:
        if indices.shape[1] != 3:
            raise ValueError(
                f"surf_grid_indices must have shape (N, 3), got {indices.shape}."
            )
        if num_surf_grid_points is None:
            n = indices.shape[0]
        else:
            n = int(num_surf_grid_points)
            if n < 0 or n > indices.shape[0]:
                raise ValueError(
                    f"num_surf_grid_points={n} is invalid for array with {indices.shape[0]} rows."
                )
        indices = indices[:n, :]

    else:
        raise ValueError(
            "surf_grid_indices must be a flattened 1D array or a 2D (N, 3) array."
        )

    return np.ascontiguousarray(indices, dtype=np.int64)


def validate_zphi_indices_in_grid(indices: np.ndarray, grid_shape: np.ndarray) -> None:
    """Validate sparse indices against parent grid shape."""
    if indices.ndim != 2 or indices.shape[1] != 3:
        raise ValueError(f"indices must have shape (N, 3), got {indices.shape}.")

    if np.any(indices < 0):
        bad = indices[np.any(indices < 0, axis=1)][0]
        raise ValueError(f"Negative surface grid index found: {bad.tolist()}.")

    upper = grid_shape.reshape((1, 3))
    if np.any(indices >= upper):
        bad = indices[np.any(indices >= upper, axis=1)][0]
        raise ValueError(
            f"Surface grid index {bad.tolist()} is out of bounds for grid shape {grid_shape.tolist()}."
        )


def zphi_potentials_from_phimap(
    *,
    grid_shape: np.ndarray,
    surf_grid_indices: np.ndarray,
    phimap: np.ndarray,
) -> np.ndarray:
    """
    Return one potential value per surface index from the full potential map.

    `phimap` may be 1D C-ravelled or 3D shape (nx, ny, nz).
    """
    phimap_arr = np.asarray(phimap)
    expected_size = int(np.prod(grid_shape))

    if phimap_arr.ndim == 1:
        if phimap_arr.size != expected_size:
            raise ValueError(
                f"1D phimap size {phimap_arr.size} does not match grid size {expected_size}."
            )
        phimap_3d = phimap_arr.reshape(tuple(grid_shape), order="C")
    elif phimap_arr.ndim == 3:
        if phimap_arr.shape != tuple(grid_shape):
            raise ValueError(
                f"3D phimap shape {phimap_arr.shape} does not match grid_shape {tuple(grid_shape)}."
            )
        phimap_3d = phimap_arr
    else:
        raise ValueError("phimap must be either 1D or 3D.")

    return np.ascontiguousarray(
        phimap_3d[
            surf_grid_indices[:, 0],
            surf_grid_indices[:, 1],
            surf_grid_indices[:, 2],
        ].astype(np.float64),
        dtype=np.float64,
    )


def write_zphi(
    zphi_filename: str,
    *,
    grid_shape: Union[np.ndarray, Tuple[int, int, int]],
    grid_scale: float,
    grid_center: Union[np.ndarray, Tuple[float, float, float]],
    surf_grid_indices: Union[np.ndarray, Iterable[int]],
    num_surf_grid_points: Optional[int] = None,
    phimap: np.ndarray,
    surface_distance: Optional[float] = None,
    context_type: str = "all_solute",
    context_label: str = "all",
    context_num_atoms: Optional[int] = None,
    content: str = ZPHI_DEFAULT_CONTENT,
    potential_units: str = ZPHI_DEFAULT_POTENTIAL_UNITS,
    coord_units: str = ZPHI_DEFAULT_COORD_UNITS,
    dense_missing_value: float = ZPHI_DEFAULT_DENSE_MISSING_VALUE,
    extra_metadata: Optional[Dict[str, Union[str, int, float]]] = None,
) -> None:
    """
    Write a canonical sparse .zphi file.

    Canonical row format:
        ix iy iz phi_kT_per_e

    Coordinates are not stored. They are reconstructed from:
        GRID_ORIGIN_ANG + indices * GRID_SPACING_ANG

    To avoid silent divergence, this writer accepts only one source of truth
    for each derived quantity:
        - potentials are extracted from `phimap`
        - GRID_ORIGIN_ANG is computed from center, shape, and scale
    """
    grid_shape = _zphi_as_3_int_array(grid_shape, "grid_shape")
    grid_center = _zphi_as_3_float_array(grid_center, "grid_center_ang")

    if grid_scale <= 0:
        raise ValueError("grid_scale_per_ang must be positive.")

    grid_spacing_ang = 1.0 / float(grid_scale)
    grid_origin_ang = compute_zphi_grid_origin_from_center(
        grid_center, grid_shape, grid_spacing_ang
    )

    indices = normalize_zphi_surface_indices(surf_grid_indices, num_surf_grid_points)
    n = int(indices.shape[0])

    if n <= 0:
        raise ValueError(
            "No zeta/offset-surface points were found. "
            "Check context selection, atom filtering, atom radii, surface distance, "
            "grid size, and whether the selected atoms are exposed."
        )

    validate_zphi_indices_in_grid(indices, grid_shape)

    pot = zphi_potentials_from_phimap(
        grid_shape=grid_shape,
        surf_grid_indices=indices,
        phimap=phimap,
    )

    simple_avg = float(np.mean(pot))
    pot_min = float(np.min(pot))
    pot_max = float(np.max(pot))

    metadata = OrderedDict()
    metadata["ZPHI_VERSION"] = str(ZPHI_VERSION)
    metadata["CONTENT"] = str(content)
    metadata["UNITS_COORD"] = str(coord_units)
    metadata["UNITS_POTENTIAL"] = str(potential_units)

    metadata["GRID_SHAPE"] = _zphi_format_vec(grid_shape)
    metadata["GRID_SCALE_PER_ANG"] = f"{float(grid_scale):.6f}"
    metadata["GRID_SPACING_ANG"] = f"{float(grid_spacing_ang):.6f}"
    metadata["GRID_CENTER_ANG"] = _zphi_format_vec(grid_center)
    metadata["GRID_ORIGIN_ANG"] = _zphi_format_vec(grid_origin_ang)
    metadata["GRID_METADATA_ROLE"] = (
        "origin_is_derived_from_center_shape_and_scale_for_integrity_check"
    )

    metadata["INDEX_BASE"] = "0"
    metadata["INDEX_ORDER"] = "ix iy iz"
    metadata["DATA_COLUMNS"] = "ix iy iz phi_kT_per_e"
    metadata["DATA_FORMAT"] = "fixed_width_indices_5d whitespace phi_10.6f_or_10.6e"

    if surface_distance is not None:
        metadata["SURFACE_DISTANCE_ANG"] = f"{float(surface_distance):.6f}"

    metadata["CONTEXT_TYPE"] = str(context_type)
    metadata["CONTEXT_LABEL"] = str(context_label)
    if context_num_atoms is not None:
        metadata["CONTEXT_NUM_ATOMS"] = str(int(context_num_atoms))

    metadata["NUM_SURFACE_POINTS"] = str(n)
    metadata["SIMPLE_AVERAGE_SURFACE_POTENTIAL"] = _zphi_format_potential(
        simple_avg
    ).strip()
    metadata["POTENTIAL_MIN"] = _zphi_format_potential(pot_min).strip()
    metadata["POTENTIAL_MAX"] = _zphi_format_potential(pot_max).strip()

    metadata["DENSE_EXPORT_MISSING_VALUE"] = f"{float(dense_missing_value):.6e}"
    metadata["DENSE_EXPORT_MISSING_VALUE_MEANING"] = "not_a_surface_point"

    if extra_metadata:
        for key, value in extra_metadata.items():
            key_norm = str(key).strip().upper().replace(" ", "_")
            if not key_norm:
                raise ValueError("extra_metadata contains an empty key.")
            metadata[key_norm] = str(value)

    with open(zphi_filename, "w") as out:
        out.write("# ZPHI sparse offset-surface electrostatic potential file\n")
        out.write("# Metadata lines use '# KEY: VALUE'\n")
        out.write("# Canonical data rows: ix iy iz phi_kT_per_e\n")

        for key, value in metadata.items():
            out.write(f"# {key}: {value}\n")

        out.write("# BEGIN_DATA\n")
        out.write("# ix iy iz phi_kT_per_e\n")

        for (ix, iy, iz), phi in zip(indices, pot):
            out.write(
                f"{int(ix):5d} {int(iy):5d} {int(iz):5d} "
                f"{_zphi_format_potential(phi)}\n"
            )

        out.write("# END_DATA\n")


def write_zeta_phi_zphi(
    zeta_filename,
    grid_center,
    grid_shape,
    scale,
    surf_grid_index,
    num_surf_grid_coords,
    phimap,
    surface_distance_ang=None,
    context_type="all_solute",
    context_label="all",
    context_num_atoms=None,
    extra_metadata=None,
):
    """
    Compatibility-oriented wrapper for writing the new canonical .zphi format
    from the existing zeta writer call path.

    Unlike legacy `write_zeta_phi`, this writes index-only sparse rows plus
    full grid metadata for reliable conversion to dense .cube/.phi formats.

    To avoid redundant inputs, grid origin is not accepted here; it is derived
    from grid_center, grid_shape, and scale.
    """
    return write_zphi(
        zphi_filename=zeta_filename,
        grid_shape=grid_shape,
        grid_scale=scale,
        grid_center=grid_center,
        surf_grid_indices=surf_grid_index,
        num_surf_grid_points=num_surf_grid_coords,
        phimap=phimap,
        surface_distance=surface_distance_ang,
        context_type=context_type,
        context_label=context_label,
        context_num_atoms=context_num_atoms,
        extra_metadata=extra_metadata,
    )


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
    include_grid_indices: bool = True,
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
    include_grid_indices: bool = True,
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
    ordered_keys=None,
    float_fmt="{:.4f}",
    header_meta=None,
):
    """
    Writes energy terms to a TSV file in a streaming, frame-by-frame manner.

    On the first frame (write_header=True), the function determines the ordered list
    of energy terms present in the input, writes the header and column–key mapping,
    and returns this ordered key list. On subsequent frames, the same ordered key list
    must be provided to ensure consistent column ordering while appending rows.

    Missing energy terms in later frames are written as empty fields. The function
    optionally omits phase-specific terms and supports writing one-time static
    metadata as comment lines in the header.

    Parameters:
        energies: dict
            Nested energy results for the current frame.
        energy_outfile: str
            Path to the TSV output file.
        run_label: str
            Label identifying the run or ensemble; written to each row.
        key_mapping: dict
            Maps nested energy keys (dot notation) to column names.
        frame: Optional[int]
            Frame index for trajectory output. If provided, a FRAME column is written.
        write_mode: str
            File open mode, typically 'w' for the first frame and 'a' for subsequent frames.
        only_phase: bool
            If True, writes only phase-independent / total energy terms.
        write_header: bool
            If True, writes the header and key-mapping comments. Should be True only
            for the first frame.
        ordered_keys: Optional[list[str]]
            Ordered list of energy keys defining the column schema. If None, the schema
            is inferred from the current frame (intended for first-frame use).
        float_fmt: str
            Format string for floating-point values.
        header_meta: Optional[dict]
            Mapping of static, frame-invariant metadata to be written once as a
            comment line in the header.

    Returns:
        ordered_keys: list[str] or None
            The ordered list of energy keys if write_header is True; otherwise None.
    """
    if ordered_keys is None:
        ordered_keys = []
        for full_key in key_mapping:
            if only_phase and not full_key.startswith(
                (
                    "total.",
                    "phase_independent.",
                    "water.electrostatic_stress",
                    "water.osmotic",
                )
            ):
                continue
            v = extract_nested(energies, full_key)
            if v is not None:
                ordered_keys.append(full_key)

    with open(energy_outfile, write_mode) as fout:
        if write_header:
            # optional one-time static metadata comment
            if header_meta:
                meta_str = " ".join(f"{k}={v}" for k, v in header_meta.items())
                fout.write(f"# {meta_str}\n")

            # comment mapping columns->keys
            comment = "# " + " | ".join(f"{key_mapping[k]}: {k}" for k in ordered_keys)
            fout.write(comment + "\n")

            columns = []
            if frame is not None:
                columns.append("FRAME")
            columns.append("LABEL")
            columns.extend(key_mapping[k] for k in ordered_keys)
            fout.write("\t".join(columns) + "\n")

        row = []
        if frame is not None:
            row.append(str(frame))
        row.append(run_label)

        for k in ordered_keys:
            v = extract_nested(energies, k)
            if v is None:
                row.append("")
            else:
                try:
                    row.append(float_fmt.format(float(v)))
                except Exception:
                    row.append(str(v))

        fout.write("\t".join(row) + "\n")

    return ordered_keys if write_header else None
