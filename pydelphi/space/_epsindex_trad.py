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

from Cython.Shadow import boundscheck
#
# PyDelphi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# PyDelphi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#

from numba import njit, prange
import numpy as np

# -- Constants assumed --
ATOMFIELD_X = 0
ATOMFIELD_Y = 1
ATOMFIELD_Z = 2
ATOMFIELD_RADIUS = 3
ATOMFIELD_MEDIA_ID = 4


@njit(cache=True)
def get_flat_index(ix, iy, iz, cell_shape):
    return ix * cell_shape[1] * cell_shape[2] + iy * cell_shape[2] + iz


@njit(boundscheck=False,cache=True)
def count_atoms_per_cell(atoms_data, cell_shape, cell_origin, cell_size):
    """
    Count the number of atoms in each spatial cell.
    Returns an array where each index corresponds to the number of atoms in that cell.
    """
    n_cells = cell_shape[0] * cell_shape[1] * cell_shape[2]
    cell_counts = np.zeros(n_cells, dtype=np.int32)

    for idx in range(atoms_data.shape[0]):
        x, y, z = atoms_data[idx, ATOMFIELD_X : ATOMFIELD_Z + 1]
        ix = int((x - cell_origin[0]) // cell_size)
        iy = int((y - cell_origin[1]) // cell_size)
        iz = int((z - cell_origin[2]) // cell_size)
        if (
            0 <= ix < cell_shape[0]
            and 0 <= iy < cell_shape[1]
            and 0 <= iz < cell_shape[2]
        ):
            flat = get_flat_index(ix, iy, iz, cell_shape)
            cell_counts[flat] += 1

    return cell_counts


@njit(boundscheck=False,cache=True)
def exclusive_prefix_sum(counts):
    """
    Compute exclusive prefix sum (scan) of counts array.
    The result[i] is the sum of counts[0] to counts[i-1].
    Used to compute starting indices for storing atom indices per cell.
    """
    result = np.empty(counts.shape[0] + 1, dtype=counts.dtype)
    result[0] = 0
    for i in range(counts.shape[0]):
        result[i + 1] = result[i] + counts[i]
    return result


@njit(boundscheck=False,cache=True)
def fill_cell_atoms(atoms_data, cell_shape, cell_origin, cell_size, cell_starts):
    """
    Fill a flattened array (cell_atoms) with atom indices grouped by spatial cell.
    cell_starts provides where to start writing for each cell.
    """
    n_atoms = atoms_data.shape[0]
    cell_atoms = np.empty(n_atoms, dtype=np.int32)
    cell_pos = np.copy(cell_starts)

    for idx in range(n_atoms):
        x, y, z = atoms_data[idx, ATOMFIELD_X : ATOMFIELD_Z + 1]
        ix = int((x - cell_origin[0]) // cell_size)
        iy = int((y - cell_origin[1]) // cell_size)
        iz = int((z - cell_origin[2]) // cell_size)
        if (
            0 <= ix < cell_shape[0]
            and 0 <= iy < cell_shape[1]
            and 0 <= iz < cell_shape[2]
        ):
            flat = get_flat_index(ix, iy, iz, cell_shape)
            cell_atoms[cell_pos[flat]] = idx
            cell_pos[flat] += 1

    return cell_atoms


@njit(boundscheck=False,cache=True)
def setup_dual_cell_maps(
    atoms_data, grid_origin, max_atom_radius, max_probe_radius, zeta_distance
):
    """
    Set up spatial hash structures (cell maps) for both epsilon and zeta distance maps.

    Returns:
        - epsilon: (cell_shape, cell_origin, cell_size, cell_counts, cell_starts, cell_atoms)
        - zeta: (cell_shape, cell_origin, cell_size, cell_counts, cell_starts, cell_atoms)
    """

    def _setup_cell_map(cell_size):
        min_corner = np.min(atoms_data[:, :3], axis=0)
        max_corner = np.max(atoms_data[:, :3], axis=0)
        bounds = max_corner - min_corner
        cell_origin = (
            grid_origin - 0.5 * cell_size
        )  # offset cell origin by half cell_size for symmetric coverage
        cell_shape = np.ceil(bounds / cell_size).astype(np.int32) + 1
        cell_counts = count_atoms_per_cell(
            atoms_data, cell_shape, cell_origin, cell_size
        )
        cell_starts = exclusive_prefix_sum(cell_counts)
        cell_atoms = fill_cell_atoms(
            atoms_data, cell_shape, cell_origin, cell_size, cell_starts
        )
        return cell_shape, cell_origin, cell_size, cell_counts, cell_starts, cell_atoms

    # Cell size must be large enough to contain any influence radius
    epsilon_cell_size = 2.0 * (max_atom_radius + max_probe_radius)
    zeta_influence = max(zeta_distance, max_probe_radius)
    zeta_cell_size = 2.0 * (zeta_influence + max_probe_radius)

    return (_setup_cell_map(epsilon_cell_size), _setup_cell_map(zeta_cell_size))


@njit(inline="always")
def squared_distance(dx, dy, dz):
    return dx * dx + dy * dy + dz * dz


@njit(inline="always")
def update_zeta_and_dielectric(
    zeta_surface_map,
    dielectric_map,
    ijk1d,
    dist2,
    atom_radius,
    salt_radius,
    zeta_distance,
    use_zeta,
):
    """
    Update exclusion flags based on zeta and dielectric influence radii.
    """
    if use_zeta:
        zeta_r2 = (atom_radius + zeta_distance) ** 2
        if dist2 < zeta_r2:
            zeta_surface_map[ijk1d] = False
    stern_r2 = (atom_radius + salt_radius) ** 2
    if dist2 < stern_r2:
        dielectric_map[ijk1d] = False


@njit(inline="always")
def update_epsilon_index_if_needed(
    index_map,
    base_index,
    dx,
    dy,
    dz,
    grid_spacing_half,
    atom_radius2,
    epsilon_index,
    valid_x,
    valid_y,
    valid_z,
):
    """
    Update epsilon index map if a neighboring face center is within atom radius.
    No failure expected: this is a deterministic update based on radius check.
    """
    if valid_x:
        dx_half = dx + grid_spacing_half
        if squared_distance(dx_half, dy, dz) < atom_radius2:
            index_map[base_index] = epsilon_index
    if valid_y:
        dy_half = dy + grid_spacing_half
        if squared_distance(dx, dy_half, dz) < atom_radius2:
            index_map[base_index + 1] = epsilon_index
    if valid_z:
        dz_half = dz + grid_spacing_half
        if squared_distance(dx, dy, dz_half) < atom_radius2:
            index_map[base_index + 2] = epsilon_index


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_calc_vdw_discrete_epsilon_map(
    epsdim,
    scale,
    probe_radius,
    salt_radius,
    zeta_distance,
    use_zeta_surf,
    grid_shape,
    grid_origin,
    atoms_data,
    index_discrete_epsilon_map_1d,
    dielectric_boundary_map_1d,
    zeta_surface_map_1d,
    cell_shape,
    cell_origin,
    cell_size,
    cell_starts,
    cell_atoms,
):
    """
    Compute epsilon index map, dielectric boundary, and zeta surface flags for each grid point.

    Spatial hash cell map is precomputed to allow efficient lookup of atoms near each grid point.
    Assumes cell size is large enough to encompass all possibly influencing atoms (i.e. double of max influence radius).
    """
    grid_spacing = 1.0 / scale
    grid_spacing_half = 0.5 * grid_spacing
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    n_grid_points = grid_shape[0] * x_stride

    for ijk1d in prange(n_grid_points):
        # Convert 1D index to 3D grid coordinates
        i = ijk1d // x_stride
        j = (ijk1d % x_stride) // y_stride
        k = ijk1d % y_stride
        base_index = 3 * ijk1d

        # Real-space coordinates of current grid point
        gx = i * grid_spacing + grid_origin[0]
        gy = j * grid_spacing + grid_origin[1]
        gz = k * grid_spacing + grid_origin[2]

        # Determine if current point is interior (to avoid accessing out of bounds neighbors)
        valid_x = i < grid_shape[0] - 1
        valid_y = j < grid_shape[1] - 1
        valid_z = k < grid_shape[2] - 1

        # Find which spatial hash cell this grid point lies in
        ix = int((gx - cell_origin[0]) // cell_size)
        iy = int((gy - cell_origin[1]) // cell_size)
        iz = int((gz - cell_origin[2]) // cell_size)
        if not (
            0 <= ix < cell_shape[0]
            and 0 <= iy < cell_shape[1]
            and 0 <= iz < cell_shape[2]
        ):
            continue

        flat_cell = get_flat_index(ix, iy, iz, cell_shape)
        start, end = cell_starts[flat_cell], cell_starts[flat_cell + 1]

        # Loop through all atoms in the current spatial hash cell
        for idx in range(start, end):
            ia = cell_atoms[idx]
            ax, ay, az = atoms_data[ia, ATOMFIELD_X : ATOMFIELD_Z + 1]
            ar = atoms_data[ia, ATOMFIELD_RADIUS]
            media_id = atoms_data[ia, ATOMFIELD_MEDIA_ID]

            dx = gx - ax
            dy = gy - ay
            dz = gz - az

            dist2 = squared_distance(dx, dy, dz)
            ar2 = ar * ar
            epsilon_index = ia + 2 + int(media_id) * epsdim

            update_zeta_and_dielectric(
                zeta_surface_map_1d,
                dielectric_boundary_map_1d,
                ijk1d,
                dist2,
                ar,
                salt_radius,
                zeta_distance,
                use_zeta_surf,
            )

            update_epsilon_index_if_needed(
                index_discrete_epsilon_map_1d,
                base_index,
                dx,
                dy,
                dz,
                grid_spacing_half,
                ar2,
                epsilon_index,
                valid_x,
                valid_y,
                valid_z,
            )
