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
This module provides functions for calculating discrete dielectric maps and zeta surface maps
based on atomic Van der Waals (VDW) radii. These maps are critical for defining the
dielectric environment and solvent-accessible surface in electrostatic calculations,
such as those performed by Poisson-Boltzmann solvers.

The module offers both CPU-optimized (using Numba's `@njit` for parallel processing)
and CUDA-optimized (using Numba's `@cuda.jit` for GPU acceleration) implementations
for the core calculation routines. This allows for flexible and performant execution
depending on the available hardware.

Key functionalities include:

1.  **`_cpu_calc_vdw_discrete_epsilon_map` and `_cuda_calc_vdw_discrete_epsilon_map`**:
    These functions determine the dielectric constant for each grid point and its
    associated half-grid points. They also identify grid points that are excluded
    from the dielectric region due to overlap with ion exclusion zones around atoms.
    The calculations take into account:
    -   Atomic positions, radii, and media IDs.
    -   A global grid scale and salt radius for ion exclusion.
    -   A voxelized atom map for efficient nearest-neighbor searches.
    The output is an epsilon index map (representing different dielectric regions)
    and a boolean dielectric boundary map (marking ion-excluded regions).

2.  **`_cpu_calc_vdw_zeta_surf_map` and `_cuda_calc_vdw_zeta_surf_map`**:
    These functions compute a "zeta surface map," which indicates whether each grid point
    is considered part of the solvent-accessible surface. This is typically determined
    by checking if a grid point lies within an extended radius (VDW radius + zeta distance)
    of any protein atom. Grid points within this extended radius are excluded from the
    zeta surface. This map is crucial for defining boundary conditions in certain
    electrostatic models.

The module dynamically imports precision-specific utility functions (e.g., `is_atom_res_protein`)
based on the global `PRECISION` setting (single or double), ensuring type compatibility and
numerical accuracy. It also leverages constants defined in `pydelphi.constants` for
atom field indices and other physical parameters.
"""

import numpy as np
from numba import njit, prange, cuda

from pydelphi.foundation.enums import Precision

from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
    vprint,
)
from pydelphi.config.logging_config import (
    DEBUG,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_RADIUS,
    ATOMFIELD_MEDIA_ID,
    NEIGHBOR_VOXEL_RELATIVE_COORDINATES as NEIGHBOR_VOXEL_REL_COORDS,
    ConstDelPhiFloats as ConstDelPhi,
)

APPROX_ZERO = ConstDelPhi.ApproxZero.value
GAUSSIAN_INFLUENCE_RADIUS_FACTOR = ConstDelPhi.GaussianInfluenceRadiusFactor.value


# --- Dynamic Precision Handling ---
if PRECISION.int_value in {
    Precision.SINGLE.int_value,
}:
    import pydelphi.utils.prec.single as size_cpu

    try:
        import pydelphi.utils.cuda.single as size_gpu
    except ImportError:
        size_gpu = None
elif PRECISION.int_value == Precision.DOUBLE.int_value:
    import pydelphi.utils.prec.double as size_cpu

    try:
        import pydelphi.utils.cuda.double as size_gpu
    except ImportError:
        size_gpu = None
else:
    raise ValueError(f"Unsupported PRECISION: {PRECISION}")


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_calc_vdw_discrete_epsilon_map(
    epsilon_dimension: delphi_int,
    scale: delphi_real,
    salt_radius: delphi_real,
    grid_shape: np.ndarray,  # shape: (3,), dtype=delphi_int
    grid_origin: np.ndarray,  # shape: (3,), dtype=delphi_real
    atoms_data: np.ndarray,  # shape: (n_atoms, n_fields), dtype=delphi_real
    index_discrete_epsilon_map_1d: np.ndarray,  # shape: (n_grid_points * 3,), dtype=delphi_int
    dielectric_boundary_map_1d: np.ndarray,  # shape: (n_grid_points,), dtype=delphi_bool
    voxel_atom_ids: np.ndarray,  # 1D, dtype=delphi_int
    voxel_atom_start_index: np.ndarray,  # 3D, dtype=delphi_int
    voxel_atom_end_index: np.ndarray,  # 3D, dtype=delphi_int
    voxel_map_origin: np.ndarray,  # shape: (3,), dtype=delphi_real
    voxel_map_shape: np.ndarray,  # shape: (3,), dtype=delphi_int
    voxel_map_scale: delphi_real,
) -> None:
    """
    For each grid point, this function determines the dielectric constant based on the van der Waals radii of nearby
        atoms and identifies grid points excluded due to ion overlap. It updates the following output arrays in place:

    - `dielectric_boundary_map_1d`: A boolean array where `False` indicates grid points within the ion exclusion zone
            around any atom (defined by the atom's radius plus the salt radius). `True` indicates the point is outside
            this zone.
    - `index_discrete_epsilon_map_1d`: An integer array storing atom-specific dielectric region identifiers for the
            three half-grid points (x, y, z) associated with each grid point. The value represents the index of the
            epsilon region.

    The function utilizes a voxelized atom map to efficiently find nearby atoms for each grid point. It operates in parallel across all grid points.

    Notes:
        - **Half-Grid Points:** The epsilon map is defined at half-grid points between the regular grid points.
                This is common in finite difference methods. The function only checks the validity of these half-grid
                points within the grid boundaries (i.e., `i < grid_shape[0] - 1`).
        - **Parallel Processing:** The function is decorated with `@njit(parallel=True)`, meaning each grid point is
                processed independently in parallel. Updates to the output arrays are handled with a "last-writer-wins"
                strategy, which is acceptable here because if multiple atoms influence the same half-grid point, the
                result is based on the last atom that satisfies the condition.
        - **Ion Exclusion:** For each grid point, the function checks if it lies within the salt-inflated radius of any
                atom. If it does, the corresponding entry in `dielectric_boundary_map_1d` is set to `False`.
        - **Epsilon Index Assignment:** For each half-grid point, the function checks if it lies within the van der Waals
                radius of any atom. If it does, an epsilon index specific to that atom and its media type is calculated
                and stored in `index_discrete_epsilon_map_1d`.
        - **Neighboring Voxels:** The function iterates through the neighboring voxels (including the current voxel) of
                the grid point to find potentially close atoms, using the `NEIGHBOR_VOXEL_REL_COORDS` constant.
        - **Atom Media ID:** The media ID of each atom is retrieved using `this_atom[ATOMFIELD_MEDIA_ID]` and is used
                to calculate the final epsilon index.
        - **Early Exit:** The inner loop over neighboring voxels and atoms breaks early for a given grid point once all
                four updates (ion exclusion and epsilon index for x, y, and z half-grid directions) have been performed.

    Parameters:
        epsilon_dimension (delphi_int): Total number of epsilon regions defined per atom. This is used to calculate a
                unique global epsilon index.
        scale (delphi_real): The grid scale factor (inverse of the grid spacing).
        salt_radius (delphi_real): The radius of a salt ion. Grid points within this distance (plus the atom's radius)
                from an atom are considered excluded from the dielectric region.
        grid_shape (np.ndarray): A 3-element array (nx, ny, nz) defining the dimensions of the grid.
        grid_origin (np.ndarray): A 3-element array defining the physical coordinates of the grid's origin.
        atoms_data (np.ndarray): A 2D array where each row represents an atom and contains its properties (including
                position, radius, and media ID). The specific columns are assumed to be accessible via constants like
                `ATOMFIELD_X`, `ATOMFIELD_Y`, `ATOMFIELD_Z`, `ATOMFIELD_RADIUS`, and `ATOMFIELD_MEDIA_ID`.
        index_discrete_epsilon_map_1d (np.ndarray): A 1D array to store the epsilon index for each half-grid point.
                For a grid of size (nx, ny, nz), this array has a shape of (nx * ny * nz * 3).
        dielectric_boundary_map_1d (np.ndarray): A 1D boolean array where each element corresponds to a grid point.
        voxel_atom_ids (np.ndarray): A 1D array containing the indices (1-based) of the atoms stored in the voxels.
        voxel_atom_start_index (np.ndarray): A 3D array with the same shape as the voxel map, where each element stores
                the starting index in `voxel_atom_ids` for the atoms in that voxel.
        voxel_atom_end_index (np.ndarray): A 3D array with the same shape as the voxel map, where each element stores
                the ending index in `voxel_atom_ids` for the atoms in that voxel.
        voxel_map_origin (np.ndarray): A 3-element array defining the physical coordinates of the origin of the voxel map.
        voxel_map_shape (np.ndarray): A 3-element array (vx, vy, vz) defining the dimensions of the voxel map.
        voxel_map_scale (delphi_real): The scaling factor to convert physical coordinates to voxel indices (inverse of the voxel size).

    Returns:
        None: This function modifies the `index_discrete_epsilon_map_1d` and `dielectric_boundary_map_1d` arrays in place.
    """
    grid_spacing = 1.0 / scale
    grid_spacing_half = 0.5 * grid_spacing
    y_stride: delphi_int = grid_shape[2]
    x_stride: delphi_int = grid_shape[1] * y_stride
    num_grid_points: delphi_int = grid_shape[0] * x_stride

    last_grid_indices = grid_shape - 1  # (nx-1, ny-1, nz-1)

    v_origin = voxel_map_origin
    v_shape = voxel_map_shape
    v_scale = voxel_map_scale

    for ijk1d in prange(num_grid_points):
        i: delphi_int = ijk1d // x_stride
        j: delphi_int = (ijk1d - i * x_stride) // y_stride
        k: delphi_int = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3: delphi_int = 3 * ijk1d

        # Real-space position of the grid point
        grid_pos_x = i * grid_spacing + grid_origin[0]
        grid_pos_y = j * grid_spacing + grid_origin[1]
        grid_pos_z = k * grid_spacing + grid_origin[2]

        # Check if half-grid spacing is valid in each direction
        valid_half_h_x = i < last_grid_indices[0]
        valid_half_h_y = j < last_grid_indices[1]
        valid_half_h_z = k < last_grid_indices[2]

        # Get voxel indices for this grid point
        central_vx = max(
            0, min(delphi_int((grid_pos_x - v_origin[0]) * v_scale), v_shape[0])
        )
        central_vy = max(
            0, min(delphi_int((grid_pos_y - v_origin[1]) * v_scale), v_shape[1])
        )
        central_vz = max(
            0, min(delphi_int((grid_pos_z - v_origin[2]) * v_scale), v_shape[2])
        )

        # Track how many of the 4 updates are complete
        updated_ion_exclusion = False
        updated_half_x = not valid_half_h_x
        updated_half_y = not valid_half_h_y
        updated_half_z = not valid_half_h_z

        dielectric_boundary_map_1d[ijk1d] = (
            True  # Initialize here, will be changed to False if inside stern layer
        )

        for neighbor_offset in NEIGHBOR_VOXEL_REL_COORDS:
            vx = central_vx + neighbor_offset[0]
            vy = central_vy + neighbor_offset[1]
            vz = central_vz + neighbor_offset[2]

            if (
                0 <= vx <= v_shape[0]
                and 0 <= vy <= v_shape[1]
                and 0 <= vz <= v_shape[2]
            ):
                start = voxel_atom_start_index[vx, vy, vz]
                end = voxel_atom_end_index[vx, vy, vz]
                if start <= end:
                    for atom_list_idx in range(start, end + 1):
                        atom_idx = voxel_atom_ids[atom_list_idx] - 1
                        this_atom = atoms_data[atom_idx]

                        r_dx = grid_pos_x - this_atom[ATOMFIELD_X]
                        r_dy = grid_pos_y - this_atom[ATOMFIELD_Y]
                        r_dz = grid_pos_z - this_atom[ATOMFIELD_Z]

                        atom_radius = this_atom[ATOMFIELD_RADIUS]
                        atom_radius_sq = atom_radius * atom_radius
                        atom_ion_radii_sq = (atom_radius + salt_radius) ** 2
                        dist_sq = r_dx * r_dx + r_dy * r_dy + r_dz * r_dz

                        # Ion exclusion check
                        if not updated_ion_exclusion and dist_sq < atom_ion_radii_sq:
                            if dielectric_boundary_map_1d[ijk1d]:
                                dielectric_boundary_map_1d[ijk1d] = False
                            updated_ion_exclusion = True

                        # Compute epsilon index for this atom
                        media_id = delphi_int(this_atom[ATOMFIELD_MEDIA_ID])
                        atom_epsilon_index = atom_idx + 2 + media_id * epsilon_dimension

                        # Half-grid checks
                        if not updated_half_x and valid_half_h_x:
                            dist_sq_hx = (
                                (r_dx + grid_spacing_half) ** 2 + r_dy**2 + r_dz**2
                            )
                            if dist_sq_hx < atom_radius_sq:
                                index_discrete_epsilon_map_1d[ijk1d_x_3 + 0] = (
                                    atom_epsilon_index
                                )
                                updated_half_x = True

                        if not updated_half_y and valid_half_h_y:
                            dist_sq_hy = (
                                r_dx**2 + (r_dy + grid_spacing_half) ** 2 + r_dz**2
                            )
                            if dist_sq_hy < atom_radius_sq:
                                index_discrete_epsilon_map_1d[ijk1d_x_3 + 1] = (
                                    atom_epsilon_index
                                )
                                updated_half_y = True

                        if not updated_half_z and valid_half_h_z:
                            dist_sq_hz = (
                                r_dx**2 + r_dy**2 + (r_dz + grid_spacing_half) ** 2
                            )
                            if dist_sq_hz < atom_radius_sq:
                                index_discrete_epsilon_map_1d[ijk1d_x_3 + 2] = (
                                    atom_epsilon_index
                                )
                                updated_half_z = True

                        # Early exit: all 4 updates are complete
                        if (
                            updated_ion_exclusion
                            and updated_half_x
                            and updated_half_y
                            and updated_half_z
                        ):
                            break


@cuda.jit(cache=True)
def _cuda_calc_vdw_discrete_epsilon_map(
    epsilon_dimension,
    scale,
    salt_radius,
    grid_shape,
    grid_origin,
    atoms_data,
    index_discrete_epsilon_map_1d,
    dielectric_boundary_map_1d,
    voxel_atom_ids,
    voxel_atom_start_index,
    voxel_atom_end_index,
    voxel_map_origin,
    voxel_map_shape,
    voxel_map_scale,
):
    """
    CUDA kernel to assign van der Waals-based epsilon indices and update the
    dielectric boundary exclusion map for each grid point.

    For each grid point, it finds nearby atoms using a voxelized atom map
    and computes distances to determine contributions to:

    - `dielectric_boundary_map_1d`: Marks grid points excluded by ion radius overlap.
    - `index_discrete_epsilon_map_1d`: Stores atom-dielectric pair identifiers for
      x/y/z half-grid directions.

    Notes:
        - Only valid half-grid directions are checked (i.e., i < grid_shape[0] - 1).
        - Each grid point is processed by one CUDA thread.
        - Updates to `index_discrete_epsilon_map_1d` use atomic exchange for thread safety
          ("last-writer-wins" behavior).
        - Exits early when all 4 updates (x, y, z directions + ion exclusion)
          have been made for a grid point.

    Parameters:
        epsilon_dimension (int): Total number of epsilon regions per atom.
        scale (float): Grid scale (reciprocal of spacing).
        salt_radius (float): Radius for ion exclusion zone.
        grid_shape (array of int (3,)): Shape of the full grid (nx, ny, nz).
        grid_origin (array of float (3,)): Physical origin of the grid.
        atoms_data (array of float (n_atoms, n_fields)): Per-atom field array
            (position, radius, media ID).
        index_discrete_epsilon_map_1d (array of int (n_grid_points * 3,)): Output
            array for epsilon index at half-grid positions (3 per point).
        dielectric_boundary_map_1d (array of bool (n_grid_points,)): Output array
            marking ion-excluded grid points.
        voxel_atom_ids (array of int (1D)): Flat array of atom indices placed into voxels.
        voxel_atom_start_index (array of int (3D)): Start indices in voxel_atom_ids
            for each voxel.
        voxel_atom_end_index (array of int (3D)): End indices in voxel_atom_ids
            for each voxel.
        voxel_map_origin (array of float (3,)): Origin of voxel space.
        voxel_map_shape (array of int (3,)): Shape of voxel space.
        voxel_map_scale (float): Inverse size of voxel in real space.
    """
    grid_spacing = 1.0 / scale
    grid_spacing_half = 0.5 * grid_spacing
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    num_grid_points = grid_shape[0] * x_stride

    last_grid_x, last_grid_y, last_grid_z = (
        grid_shape[0] - 1,
        grid_shape[1] - 1,
        grid_shape[2] - 1,
    )  # (nx-1, ny-1, nz-1)

    v_origin = voxel_map_origin
    v_shape = voxel_map_shape
    v_scale = voxel_map_scale

    ijk1d = cuda.grid(1)
    if ijk1d < num_grid_points:
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride
        ijk1d_x_3 = 3 * ijk1d

        # Real-space position of the grid point
        grid_pos_x = i * grid_spacing + grid_origin[0]
        grid_pos_y = j * grid_spacing + grid_origin[1]
        grid_pos_z = k * grid_spacing + grid_origin[2]

        # Check if half-grid spacing is valid in each direction
        valid_half_h_x = i < last_grid_x
        valid_half_h_y = j < last_grid_y
        valid_half_h_z = k < last_grid_z

        # Get voxel indices for this grid point
        central_vx = max(0, min(int((grid_pos_x - v_origin[0]) * v_scale), v_shape[0]))
        central_vy = max(0, min(int((grid_pos_y - v_origin[1]) * v_scale), v_shape[1]))
        central_vz = max(0, min(int((grid_pos_z - v_origin[2]) * v_scale), v_shape[2]))

        # Track how many of the 4 updates are complete
        updated_ion_exclusion = False
        updated_half_x = not valid_half_h_x
        updated_half_y = not valid_half_h_y
        updated_half_z = not valid_half_h_z

        dielectric_boundary_map_1d[ijk1d] = (
            True  # Initialize here, will be changed to False if inside stern layer
        )

        for neighbor_offset in NEIGHBOR_VOXEL_REL_COORDS:
            vx = central_vx + neighbor_offset[0]
            vy = central_vy + neighbor_offset[1]
            vz = central_vz + neighbor_offset[2]

            if (
                0 <= vx <= v_shape[0]
                and 0 <= vy <= v_shape[1]
                and 0 <= vz <= v_shape[2]
            ):
                start = voxel_atom_start_index[vx, vy, vz]
                end = voxel_atom_end_index[vx, vy, vz]
                if start <= end:
                    for atom_list_idx in range(start, end + 1):
                        atom_idx = voxel_atom_ids[atom_list_idx] - 1
                        this_atom = atoms_data[atom_idx]

                        r_dx = grid_pos_x - this_atom[ATOMFIELD_X]
                        r_dy = grid_pos_y - this_atom[ATOMFIELD_Y]
                        r_dz = grid_pos_z - this_atom[ATOMFIELD_Z]

                        atom_radius = this_atom[ATOMFIELD_RADIUS]
                        atom_radius_sq = atom_radius * atom_radius
                        atom_ion_radii_sq = (atom_radius + salt_radius) ** 2
                        dist_sq = r_dx * r_dx + r_dy * r_dy + r_dz * r_dz

                        # Ion exclusion check
                        if not updated_ion_exclusion and dist_sq < atom_ion_radii_sq:
                            if dielectric_boundary_map_1d[ijk1d]:
                                dielectric_boundary_map_1d[ijk1d] = False
                            updated_ion_exclusion = True

                        # Compute epsilon index for this atom
                        media_id = int(this_atom[ATOMFIELD_MEDIA_ID])
                        atom_epsilon_index = atom_idx + 2 + media_id * epsilon_dimension

                        # Half-grid checks
                        if not updated_half_x and valid_half_h_x:
                            dist_sq_hx = (
                                (r_dx + grid_spacing_half) ** 2 + r_dy**2 + r_dz**2
                            )
                            if dist_sq_hx < atom_radius_sq:
                                cuda.atomic.exch(
                                    index_discrete_epsilon_map_1d,
                                    ijk1d_x_3 + 0,
                                    atom_epsilon_index,
                                )
                                updated_half_x = True

                        if not updated_half_y and valid_half_h_y:
                            dist_sq_hy = (
                                r_dx**2 + (r_dy + grid_spacing_half) ** 2 + r_dz**2
                            )
                            if dist_sq_hy < atom_radius_sq:
                                cuda.atomic.exch(
                                    index_discrete_epsilon_map_1d,
                                    ijk1d_x_3 + 1,
                                    atom_epsilon_index,
                                )
                                updated_half_y = True

                        if not updated_half_z and valid_half_h_z:
                            dist_sq_hz = (
                                r_dx**2 + r_dy**2 + (r_dz + grid_spacing_half) ** 2
                            )
                            if dist_sq_hz < atom_radius_sq:
                                cuda.atomic.exch(
                                    index_discrete_epsilon_map_1d,
                                    ijk1d_x_3 + 2,
                                    atom_epsilon_index,
                                )
                                updated_half_z = True

                        # Early exit: all 4 updates are complete
                        if (
                            updated_ion_exclusion
                            and updated_half_x
                            and updated_half_y
                            and updated_half_z
                        ):
                            break


def calculate_vdw_discrete_epsilon_map(
    platform,
    num_cuda_threads,
    epsilon_dimension,
    scale,
    salt_radius,
    grid_shape,
    grid_origin,
    atoms_data,
    index_discrete_epsilon_map_1d,
    dielectric_boundary_map_1d,
    voxel_atom_ids,
    voxel_atom_start_index,
    voxel_atom_end_index,
    voxel_map_origin,
    voxel_map_shape,
    voxel_map_scale,
):
    if platform.active == "cpu":
        vprint(
            DEBUG,
            _VERBOSITY,
            "Calling OPTIMIZED _cpu_calc_vdw_discrete_epsilon_map",
        )
        # Call the CPU function for VDW epsilon/boundary
        _cpu_calc_vdw_discrete_epsilon_map(
            epsilon_dimension=epsilon_dimension,
            scale=scale,
            salt_radius=salt_radius,
            grid_shape=grid_shape,
            grid_origin=grid_origin,
            atoms_data=atoms_data,
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
            dielectric_boundary_map_1d=dielectric_boundary_map_1d,
            voxel_atom_ids=voxel_atom_ids,
            voxel_atom_start_index=voxel_atom_start_index,
            voxel_atom_end_index=voxel_atom_end_index,
            voxel_map_origin=voxel_map_origin,
            voxel_map_shape=voxel_map_shape,
            voxel_map_scale=voxel_map_scale,
        )
    elif platform.active == "cuda":
        vprint(
            DEBUG,
            _VERBOSITY,
            "Calling OPTIMIZED _cuda_calc_vdw_discrete_epsilon_map",
        )
        # Transfer data to GPU
        grid_shape_dev = cuda.to_device(grid_shape.astype(delphi_int))
        grid_origin_dev = cuda.to_device(grid_origin.astype(delphi_real))
        atoms_data_dev = cuda.to_device(atoms_data.astype(delphi_real))
        index_discrete_epsilon_map_1d_dev = cuda.to_device(
            index_discrete_epsilon_map_1d
        )
        dielectric_boundary_map_1d_dev = cuda.to_device(dielectric_boundary_map_1d)
        voxel_ids_dev = cuda.to_device(voxel_atom_ids.astype(delphi_int))
        voxel_start_dev = cuda.to_device(voxel_atom_start_index.astype(delphi_int))
        voxel_end_dev = cuda.to_device(voxel_atom_end_index.astype(delphi_int))
        voxel_origin_dev = cuda.to_device(voxel_map_origin.astype(delphi_real))
        voxel_shape_dev = cuda.to_device(voxel_map_shape.astype(delphi_int))

        # Configure kernel launch
        threads_per_block = num_cuda_threads
        num_grid_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
        blocks_per_grid = (
            num_grid_points + (threads_per_block - 1)
        ) // threads_per_block

        # Launch CUDA kernel
        _cuda_calc_vdw_discrete_epsilon_map[blocks_per_grid, threads_per_block](
            delphi_real(epsilon_dimension),
            delphi_real(scale),
            delphi_real(salt_radius),
            grid_shape_dev,
            grid_origin_dev,
            atoms_data_dev,
            index_discrete_epsilon_map_1d_dev,
            dielectric_boundary_map_1d_dev,
            voxel_ids_dev,
            voxel_start_dev,
            voxel_end_dev,
            voxel_origin_dev,
            voxel_shape_dev,
            delphi_real(voxel_map_scale),
        )
        cuda.synchronize()

        # Copy results back to host
        index_discrete_epsilon_map_1d_dev.copy_to_host(index_discrete_epsilon_map_1d)
        dielectric_boundary_map_1d_dev.copy_to_host(dielectric_boundary_map_1d)

    else:
        raise RuntimeError(f"Unsupported platform: {platform.active}")

    return index_discrete_epsilon_map_1d, dielectric_boundary_map_1d


@njit(nogil=True, boundscheck=False, parallel=True, cache=True)
def _cpu_calc_vdw_zeta_surf_map(
    scale: delphi_real,
    zeta_distance: delphi_real,
    grid_shape: np.ndarray,  # (3,)
    grid_origin: np.ndarray,  # (3,)
    atoms_data: np.ndarray,  # (n_atoms, n_fields)
    zeta_surface_map_1d: np.ndarray,  # (n,) bool
    voxel_zeta_atom_ids: np.ndarray,  # (n,) int (1-based indices)
    voxel_zeta_atom_start_index: np.ndarray,  # (vx, vy, vz)
    voxel_zeta_atom_end_index: np.ndarray,  # (vx, vy, vz)
    voxel_zeta_map_origin: np.ndarray,  # (3,)
    voxel_zeta_map_shape: np.ndarray,  # (3,)
    voxel_zeta_map_scale: delphi_real,
) -> None:
    """
    For each grid point that is initially marked as part of the zeta surface,
    this function checks whether the point is within a zeta-inflated radius
    of any nearby protein atom. If it is, the grid point is marked as not being
    part of the zeta surface.

    Parameters
    ----------
    scale : delphi_real
        Grid scale (1 / grid spacing).
    zeta_distance : delphi_real
        Additional distance added to atom radii for defining zeta surface exclusion.
    grid_shape : np.ndarray of shape (3,)
        Shape of the full 3D grid (nx, ny, nz).
    grid_origin : np.ndarray of shape (3,)
        Origin coordinates (x, y, z) of the grid.
    atoms_data : np.ndarray of shape (n_atoms, n_fields)
        Atom data including coordinates and radii.
    zeta_surface_map_1d : np.ndarray of bool
        Flattened 1D boolean array representing which grid points are part of the zeta surface.
    voxel_zeta_atom_ids : np.ndarray
        1D array of atom indices mapped to coarse voxels (1-based indexing).
    voxel_zeta_atom_start_index : np.ndarray
        3D array marking start index of atoms in voxel_zeta_atom_ids per voxel.
    voxel_zeta_atom_end_index : np.ndarray
        3D array marking end index of atoms in voxel_zeta_atom_ids per voxel.
    voxel_zeta_map_origin : np.ndarray of shape (3,)
        Origin of the coarse voxel map.
    voxel_zeta_map_shape : np.ndarray of shape (3,)
        Shape of the coarse voxel map.
    voxel_zeta_map_scale : delphi_real
        Scale factor for mapping world coordinates to voxel indices.
    """
    grid_spacing = 1.0 / scale
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    num_grid_points = grid_shape[0] * x_stride

    vz_origin = voxel_zeta_map_origin
    vz_shape = voxel_zeta_map_shape
    vz_scale = voxel_zeta_map_scale

    for ijk1d in prange(num_grid_points):
        if not zeta_surface_map_1d[ijk1d]:
            continue  # Already excluded

        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = ijk1d - i * x_stride - j * y_stride

        grid_x = i * grid_spacing + grid_origin[0]
        grid_y = j * grid_spacing + grid_origin[1]
        grid_z = k * grid_spacing + grid_origin[2]

        vz_x = min(max(0, int((grid_x - vz_origin[0]) * vz_scale)), vz_shape[0])
        vz_y = min(max(0, int((grid_y - vz_origin[1]) * vz_scale)), vz_shape[1])
        vz_z = min(max(0, int((grid_z - vz_origin[2]) * vz_scale)), vz_shape[2])

        for offset in NEIGHBOR_VOXEL_REL_COORDS:
            neighbor_vx = vz_x + offset[0]
            neighbor_vy = vz_y + offset[1]
            neighbor_vz = vz_z + offset[2]

            if (
                0 <= neighbor_vx <= vz_shape[0]
                and 0 <= neighbor_vy <= vz_shape[1]
                and 0 <= neighbor_vz <= vz_shape[2]
            ):
                start_voxel_id = voxel_zeta_atom_start_index[
                    neighbor_vx, neighbor_vy, neighbor_vz
                ]
                end_voxel_id = voxel_zeta_atom_end_index[
                    neighbor_vx, neighbor_vy, neighbor_vz
                ]

                if start_voxel_id <= end_voxel_id:
                    for atom_list_idx in range(start_voxel_id, end_voxel_id + 1):
                        atom_idx = voxel_zeta_atom_ids[atom_list_idx] - 1
                        this_atom = atoms_data[atom_idx]

                        if size_cpu.is_atom_res_protein(this_atom.astype(delphi_real)):
                            dx = grid_x - this_atom[ATOMFIELD_X]
                            dy = grid_y - this_atom[ATOMFIELD_Y]
                            dz = grid_z - this_atom[ATOMFIELD_Z]
                            dist_square = dx * dx + dy * dy + dz * dz
                            radius_effective = (
                                this_atom[ATOMFIELD_RADIUS] + zeta_distance
                            )

                            if dist_square < radius_effective * radius_effective:
                                zeta_surface_map_1d[ijk1d] = False
                                break  # Found one atom that excludes this grid point

                if not zeta_surface_map_1d[ijk1d]:
                    break
            if not zeta_surface_map_1d[ijk1d]:
                break


@cuda.jit(device=True)
def get_neighbor_offset(offset_idx):
    return (
        NEIGHBOR_VOXEL_REL_COORDS[offset_idx][0],
        NEIGHBOR_VOXEL_REL_COORDS[offset_idx][1],
        NEIGHBOR_VOXEL_REL_COORDS[offset_idx][2],
    )


@cuda.jit(cache=True)
def _cuda_calc_vdw_zeta_surf_map(
    scale,
    zeta_distance,
    grid_shape,
    grid_origin,
    atoms_data,
    zeta_surface_map_1d,
    voxel_zeta_atom_ids,
    voxel_zeta_atom_start_index,
    voxel_zeta_atom_end_index,
    voxel_zeta_map_origin,
    voxel_zeta_map_shape,
    voxel_zeta_map_scale,
):
    """
    CUDA kernel to compute a zeta-surface map indicating whether each grid point is
    outside the extended radius of nearby protein atoms (used to define van der Waals surface).

    Parameters
    ----------
    scale : float
        Grid scaling factor (reciprocal of spacing).
    zeta_distance : float
        Additional distance added to atomic radius to define the zeta surface.
    grid_shape : 1D array of int
        Shape of the 3D grid as [nx, ny, nz].
    grid_origin : 1D array of float
        Origin coordinates of the grid.
    atoms_data : 2D array of float
        Atom data, each row corresponds to an atom with fields defined by ATOMFIELD_* constants.
    zeta_surface_map_1d : 1D array of bool
        Output map marking which grid points are outside the zeta-surface.
    voxel_zeta_atom_ids : 1D array of int
        Atom indices per voxel (flattened).
    voxel_zeta_atom_start_index : 3D array of int
        Start index in voxel_zeta_atom_ids for each voxel.
    voxel_zeta_atom_end_index : 3D array of int
        End index in voxel_zeta_atom_ids for each voxel.
    voxel_zeta_map_origin : 1D array of float
        Origin of the voxel grid.
    voxel_zeta_map_shape : 1D array of int
        Shape of the voxel grid in x, y, z.
    voxel_zeta_map_scale : float
        Inverse spacing of the voxel grid.
    """
    i = cuda.grid(1)
    grid_nx, grid_ny, grid_nz = grid_shape[0], grid_shape[1], grid_shape[2]
    y_stride = grid_nz
    x_stride = grid_ny * y_stride
    num_grid_points = grid_nx * x_stride

    if i >= num_grid_points:
        return

    ix = i // x_stride
    iy = (i - ix * x_stride) // y_stride
    iz = i - ix * x_stride - iy * y_stride

    grid_pos_x = ix / scale + grid_origin[0]
    grid_pos_y = iy / scale + grid_origin[1]
    grid_pos_z = iz / scale + grid_origin[2]

    vzx = int((grid_pos_x - voxel_zeta_map_origin[0]) * voxel_zeta_map_scale)
    vzy = int((grid_pos_y - voxel_zeta_map_origin[1]) * voxel_zeta_map_scale)
    vzz = int((grid_pos_z - voxel_zeta_map_origin[2]) * voxel_zeta_map_scale)

    if vzx < 0 or vzx >= voxel_zeta_map_shape[0]:
        return
    if vzy < 0 or vzy >= voxel_zeta_map_shape[1]:
        return
    if vzz < 0 or vzz >= voxel_zeta_map_shape[2]:
        return

    if not zeta_surface_map_1d[i]:
        return

    zeta_surface_map_1d[i] = (
        True  # Initialize values to default expected value True: defensive programming
    )
    for offset_idx in range(27):
        dvx, dvy, dvz = get_neighbor_offset(offset_idx)
        nx = vzx + dvx
        ny = vzy + dvy
        nz = vzz + dvz

        if (
            0 <= nx < voxel_zeta_map_shape[0]
            and 0 <= ny < voxel_zeta_map_shape[1]
            and 0 <= nz < voxel_zeta_map_shape[2]
        ):
            start_voxel_id = voxel_zeta_atom_start_index[nx, ny, nz]
            end_voxel_id = voxel_zeta_atom_end_index[nx, ny, nz]

            if start_voxel_id <= end_voxel_id:
                for atom_list_idx in range(start_voxel_id, end_voxel_id + 1):
                    atom_idx = voxel_zeta_atom_ids[atom_list_idx] - 1
                    this_atom = atoms_data[atom_idx]

                    dx = grid_pos_x - this_atom[ATOMFIELD_X]
                    dy = grid_pos_y - this_atom[ATOMFIELD_Y]
                    dz = grid_pos_z - this_atom[ATOMFIELD_Z]
                    dist_square = dx * dx + dy * dy + dz * dz

                    if size_gpu.cu_is_atom_res_protein(this_atom):
                        radius_effective_square = (
                            this_atom[ATOMFIELD_RADIUS] + zeta_distance
                        ) ** 2
                        if dist_square < radius_effective_square:
                            zeta_surface_map_1d[i] = False
                            return


def calculate_vdw_zeta_surf_map(
    platform,
    num_cuda_threads,
    scale,
    zeta_distance,
    grid_shape,
    grid_origin,
    atoms_data,
    zeta_surface_map_1d,
    # Pass chosen voxel args (VDW or Zeta specific)
    voxel_zeta_atom_ids,
    voxel_zeta_atom_start_index,
    voxel_zeta_atom_end_index,
    voxel_zeta_map_origin,
    voxel_zeta_map_shape,
    voxel_zeta_map_scale,
):
    if platform.active == "cpu":
        vprint(DEBUG, _VERBOSITY, "Calling _cpu_calc_vdw_discrete_zeta_surf_map")
        # Call the NEW CPU function for Zeta surface
        _cpu_calc_vdw_zeta_surf_map(
            scale=scale,
            zeta_distance=zeta_distance,
            grid_shape=grid_shape,
            grid_origin=grid_origin,
            atoms_data=atoms_data,
            zeta_surface_map_1d=zeta_surface_map_1d,
            # Pass chosen voxel args (VDW or Zeta specific)
            voxel_zeta_atom_ids=voxel_zeta_atom_ids,
            voxel_zeta_atom_start_index=voxel_zeta_atom_start_index,
            voxel_zeta_atom_end_index=voxel_zeta_atom_end_index,
            voxel_zeta_map_origin=voxel_zeta_map_origin,
            voxel_zeta_map_shape=voxel_zeta_map_shape,
            voxel_zeta_map_scale=voxel_zeta_map_scale,
        )
    elif platform.active == "cuda" and hasattr(cuda, "jit"):
        vprint(DEBUG, _VERBOSITY, "Calling _cuda_calc_vdw_discrete_zeta_surf_map")
        # Transfer data to GPU
        grid_shape_dev = cuda.to_device(grid_shape.astype(delphi_int))
        grid_origin_dev = cuda.to_device(grid_origin.astype(delphi_real))
        atoms_data_dev = cuda.to_device(atoms_data.astype(delphi_real))
        zeta_surface_map_1d_dev = cuda.to_device(zeta_surface_map_1d)
        voxel_ids_dev = cuda.to_device(voxel_zeta_atom_ids.astype(delphi_int))
        voxel_start_dev = cuda.to_device(voxel_zeta_atom_start_index.astype(delphi_int))
        voxel_end_dev = cuda.to_device(voxel_zeta_atom_end_index.astype(delphi_int))
        voxel_origin_dev = cuda.to_device(voxel_zeta_map_origin.astype(delphi_real))
        voxel_shape_dev = cuda.to_device(voxel_zeta_map_shape.astype(delphi_int))

        # Configure kernel launch
        num_grid_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
        threads_per_block = num_cuda_threads
        blocks_per_grid = (
            num_grid_points + (threads_per_block - 1)
        ) // threads_per_block

        # Launch CUDA kernel
        _cuda_calc_vdw_zeta_surf_map[blocks_per_grid, threads_per_block](
            delphi_real(scale),
            delphi_real(zeta_distance),
            grid_shape_dev,
            grid_origin_dev,
            atoms_data_dev,
            zeta_surface_map_1d_dev,
            voxel_ids_dev,
            voxel_start_dev,
            voxel_end_dev,
            voxel_origin_dev,
            voxel_shape_dev,
            delphi_real(voxel_zeta_map_scale),
        )
        cuda.synchronize()

        # Copy results back to host
        zeta_surface_map_1d_dev.copy_to_host(zeta_surface_map_1d)

    else:
        raise RuntimeError(f"Unsupported platform: {platform.active}")


@njit(nogil=True, boundscheck=False, parallel=True)
def calculate_vdw_dielectric_map_midpoints(
    vacuum: delphi_bool,
    epsilon_dim: delphi_int,
    media_epsilon: np.ndarray[delphi_real],
    index_discrete_epsilon_map_1d: np.ndarray[delphi_int],
    discrete_epsilon_map_1d: np.ndarray[delphi_real],
):
    num_grid_points_x_3 = index_discrete_epsilon_map_1d.size
    media_epsilon_temp = np.copy(media_epsilon)
    num_media = media_epsilon_temp.shape[0]
    epsilon_out = 1.0 if vacuum else media_epsilon[0]
    media_epsilon_temp[0] = epsilon_out

    for ijk1d_x_3 in prange(num_grid_points_x_3):
        media_id = abs(index_discrete_epsilon_map_1d[ijk1d_x_3]) // epsilon_dim
        if 0 <= media_id < num_media:
            discrete_epsilon_map_1d[ijk1d_x_3] = media_epsilon_temp[media_id]
        else:
            discrete_epsilon_map_1d[ijk1d_x_3] = epsilon_out  # Default
