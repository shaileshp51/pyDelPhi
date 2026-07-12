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
# PyDelphi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# PyDelphi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#

import gc
import time
import math
import numpy as np
from numba import njit, prange, set_num_threads
from numba import cuda, float64, int32


from pydelphi.foundation.enums import Precision

from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
    nprint_cpu_if_verbose as nprint_cpu,
    vprint,
)
from pydelphi.config.logging_config import (
    DEBUG,
    VERBOSE,
    get_effective_verbosity,
)
from pydelphi.constants import (
    ConstDelPhiFloats,
    ConstDelPhiInts,
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_CRD_END,
    ATOMFIELD_RADIUS,
    ATOMFIELD_MEDIA_ID,
)

from pydelphi.space.core.vdw.scale_bgp.initial_bgp import (
    scale_initial_vdw_surface_boundary_points,
)

from pydelphi.space.core.vdw.scale_bgp.reentrant_bgp import (
    _cpu_scale_reentrant_boundary_points,
    _cuda_scale_reentrant_boundary_points_gridstride,
)
import pydelphi.space.core.voxelizer as voxelizer
import pydelphi.space.core.vdw.sas_builder.sas_parallel as sas

# Define a small constant used for comparisons with zero.
APPROX_ZERO = ConstDelPhiFloats.ApproxZero.value
"""
A small constant used for comparisons with zero. Values smaller than this can be treated as numeric zero.
"""

# Define the maximum size of the neighbor list.
SPACE_NBRA_MAX_SIZE = ConstDelPhiInts.SpaceNBRASize.value
# Define the return value indicating an error in njit functions.
EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value

SHARED_SIZE = 256  # max threads per block we'll support in shared reduction
THREADS_PER_BLOCK = 256

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)


def scale_vdw_surface_boundary_points(
    use_cuda: delphi_bool,
    num_threads: delphi_int,
    num_atoms: delphi_int,
    num_molecules: delphi_int,
    num_objects: delphi_int,
    max_atom_radius: delphi_real,
    probe_radius: delphi_real,
    probe_radius_2: delphi_real,
    is_focusing_run: delphi_bool,
    grid_spacing: delphi_real,
    grid_origin: np.ndarray,
    grid_dimensions: tuple,
    atom_data: np.ndarray,
    min_coords_by_axis: np.ndarray,
    max_coords_by_axis: np.ndarray,
    num_exposed_grid_points: delphi_int,
    num_boundary_points: delphi_int,
    num_external_boundary_points: delphi_int,
    surface_charge_positions: np.ndarray,
    discrete_epsilon_index_map_1d: np.ndarray,
    scaled_surface_normal_vectors: np.ndarray,
    exposed_grid_point_coords: np.ndarray,
    atom_accessibility: np.ndarray,
    atom_surface_index: np.ndarray,
    atom_index_for_boundary: np.ndarray,
    atom_plus_probe_radii: np.ndarray,
    atom_plus_probe_radii_squared: np.ndarray,
    atom_plus_probe_radii_squared_shrunk: np.ndarray,
    system_min_coords: np.ndarray,
    cube_side_indver_inverse: delphi_real,
    cube_shape_indver: np.ndarray,
    cube_voxel_atom_index_start: np.ndarray,
    cube_voxel_atom_index_end: np.ndarray,
    cube_voxel_atom_index_cumulative: np.ndarray,
) -> tuple[int, float]:
    """Scales the initial set of boundary points to accurately represent the molecular van der Waals surface.

    This function iterates through the boundary points and adjusts their positions to lie on the solvent-accessible surface.
    It handles both external and internal boundaries, as well as solvent exposure.

    Args:
        num_atoms (delphi_int): The number of atoms in the system.
        num_molecules (delphi_int): The number of molecules in the system.
        num_objects (delphi_int): The number of other objects in the system.
        max_atom_radius (delphi_real): The maximum radius of any atom.
        probe_radius (delphi_real): The radius of the solvent probe.
        probe_radius_2 (delphi_real): The radius of the second solvent probe (used in some cases).
        is_focusing_run (delphi_bool): Flag indicating if this is a focusing run.
        grid_spacing (delphi_real): The spacing between grid points.
        grid_origin (np.ndarray): The coordinates of the origin of the grid.
        grid_dimensions (tuple): The dimensions of the grid (nx, ny, nz).
        atom_data (np.ndarray): Array containing atom properties.
        object_data (np.ndarray): Array containing object properties.
        min_coords_by_axis (np.ndarray): Minimum coordinates of the system along each axis.
        max_coords_by_axis (np.ndarray): Maximum coordinates of the system along each axis.
        voxel_atom_count (np.ndarray): Array storing the starting index of atoms in each voxel.
        voxel_atom_count_cumulative (np.ndarray): Array storing the ending index of atoms in each voxel.
        num_exposed_grid_points (delphi_int): The number of solvent-exposed grid points.
        num_boundary_points (delphi_int): The total number of boundary points.
        num_external_boundary_points (delphi_int): The number of external boundary points.
        surface_charge_positions (np.ndarray): Array storing the positions of surface charge points.
        discrete_epsilon_index_map_1d (np.ndarray): 1D array representing the discrete epsilon map.
        scaled_surface_normal_vectors (np.ndarray): Array to store the scaled surface normal vectors.
        exposed_grid_point_coords (np.ndarray): Array containing coordinates of all solvent-exposed grid points.
        atom_accessibility (np.ndarray): Array indicating the accessibility of each atom.
        atom_surface_index (np.ndarray): Array to store the index of the closest atom to each boundary point.
        atom_index_for_boundary (np.ndarray): Array to store the index of the atom associated with each boundary point.
        atom_plus_probe_radii (np.ndarray): Array containing the sum of atom radii and probe radius.
        atom_plus_probe_radii_squared (np.ndarray): Array containing the squared sum of atom radii and probe radius.
        atom_plus_probe_radii_squared_shrunk (np.ndarray): Array containing the squared sum of shrunk atom radii and probe radius.
        system_min_coords (np.ndarray): Minimum coordinates of the system.
        cube_side_indver_inverse (delphi_real): Inverse of the cube side length used for indexing.
        cube_voxel_atom_index_start (np.ndarray): Array storing the starting index of atoms in each cube.
        cube_voxel_atom_index_end (np.ndarray): Array storing the ending index of atoms in each cube.
        cube_voxel_atom_index_cumulative (np.ndarray): Array storing the cumulative index of atoms in each cube.

    Returns:
        int: An exit flag (0 for success).
        float: The inverse of the cube side length.
    """
    temp_grid_coords = np.zeros(3, dtype=delphi_real)
    # Initialize a boolean array to track if sub-cubes are outside the main cube.
    is_outside_cube = np.ones((5, 5, 5), dtype=delphi_bool)
    # Mark the central 3x3x3 sub-cubes as inside.
    for bgp_index in range(3):
        for j in range(3):
            for k in range(3):
                is_outside_cube[bgp_index + 1][j + 1][k + 1] = False

    epsilon_dimension = num_atoms + num_objects + 2
    half_grid_spacing = grid_spacing / 2.0
    is_internal_bgp = False

    # Determine the current grid origin based on whether it's a focusing run.
    current_grid_origin = grid_origin if is_focusing_run else grid_origin
    max_probe_radius = max(probe_radius_2, probe_radius)

    # Calculate the solvent-accessible surface if no exposed grid points are found and probe radius is greater than zero.
    if (
        num_exposed_grid_points == 0
        and max_probe_radius > 0.0
        and (num_objects > 1 or num_atoms > 1)
    ):
        nprint_cpu(DEBUG, _VERBOSITY, "Scaling routine in action")
        (
            num_exposed_grid_points,
            exposed_grid_point_coords,
            atom_accessibility,
            voxel_atom_count,
            voxel_atom_count_cumulative,
            voxel_space_origin,
            voxel_space_shape,
        ) = sas.solvent_accessible_surface(
            grid_spacing=grid_spacing,
            probe_radius=probe_radius,
            probe_radius2=probe_radius_2,
            max_atom_radius=max_atom_radius,
            min_coords_by_axis=min_coords_by_axis,
            max_coords_by_axis=max_coords_by_axis,
            num_atoms=num_atoms,
            num_objects=num_objects,
            num_molecules=num_molecules,
            atoms_data=atom_data,
            atom_plus_probe_radii_1d=atom_plus_probe_radii,
            atom_plus_probe_radii_shrink_1d=atom_plus_probe_radii_squared_shrunk,
            use_cuda=use_cuda,
            num_threads=num_threads,
            num_vertices=delphi_int(520),
            num_edges=delphi_int(1040),
        )

    # Determine the side length of the spatial partitioning cube
    delta = max(max_probe_radius, half_grid_spacing)
    voxel_side_length = max_atom_radius + delta
    voxel_space_scale = 1.0 / voxel_side_length

    # Set up the spatial partitioning into voxels
    (voxel_space_origin, voxel_space_shape) = (
        voxelizer.calculate_voxel_space_parameters(
            voxel_side_length,
            min_coords_by_axis,
            max_coords_by_axis,
            scaling_factor=2.0,
        )
    )
    voxels_extended_flat_size = (
        (voxel_space_shape[0] + 1)
        * (voxel_space_shape[1] + 1)
        * (voxel_space_shape[2] + 1)
    )
    voxels_per_entity = 27
    # Adjust the number of voxels per entity if there are non-molecule objects.
    if (num_objects - num_molecules) > 0:
        voxels_per_entity = voxels_extended_flat_size
        if voxels_per_entity < 27:
            voxels_per_entity = delphi_int(27)
    # Initialize an array to store atom indices for each voxel.
    voxel_atom_indices = np.zeros(
        voxels_per_entity * (num_atoms + num_objects - num_molecules) + 1,
        dtype=delphi_int,
    )

    # Populate the spatial partitioning cube with atom indices.
    (
        voxel_atom_indices,
        voxel_atom_count,
        voxel_atom_count_cumulative,
    ) = voxelizer.build_atom_voxel_map(
        voxel_side_length,
        num_atoms,
        num_objects,
        num_molecules,
        voxel_space_origin,
        voxel_space_shape,
        atom_data,
        voxel_atom_indices,
    )

    scaled_boundary_point_count = 0
    y_stride = grid_dimensions[2]
    x_stride = grid_dimensions[1] * grid_dimensions[2]
    x_stride_x_3 = x_stride * 3
    y_stride_x_3 = y_stride * 3

    atom_surface_index = np.zeros(num_boundary_points, dtype=delphi_int)
    atom_index_for_boundary = np.zeros(num_boundary_points, dtype=delphi_int)
    scaled_surface_normal_vectors = np.zeros_like(
        surface_charge_positions, dtype=delphi_real
    )

    init_scale_tic = time.perf_counter()
    # Iterate through each boundary grid point.
    voxel_space_shape_tuple = tuple([int(i) for i in voxel_space_shape.tolist()])
    threads_per_block = 256
    (bgp_loop1_status, scaled_boundary_point_count) = (
        scale_initial_vdw_surface_boundary_points(
            use_cuda,
            num_threads,
            threads_per_block,
            num_boundary_points,
            surface_charge_positions,
            atom_data,
            atom_plus_probe_radii,
            atom_plus_probe_radii_squared_shrunk,
            voxel_atom_indices,
            voxel_atom_count,
            voxel_atom_count_cumulative,
            voxel_space_origin,
            voxel_space_scale,
            voxel_space_shape_tuple,
            atom_accessibility,
            num_atoms,
            num_molecules,
            discrete_epsilon_index_map_1d,
            epsilon_dimension,
            x_stride,
            y_stride,
            x_stride_x_3,
            y_stride_x_3,
            current_grid_origin,
            grid_spacing,
            max_probe_radius,
            atom_surface_index,
            atom_index_for_boundary,
            scaled_surface_normal_vectors,
            debug_level=DEBUG,
            verbosity_level=_VERBOSITY,
            exit_status_flag=EXIT_NJIT_FLAG,
            delphi_real=delphi_real,
            delphi_int=delphi_int,
        )
    )

    init_scale_toc = time.perf_counter()
    # print("atom_index_for_boundary=", atom_index_for_boundary)
    # np.savetxt(
    #     f"atom_index_for_boundary_{num_threads}.txt", atom_index_for_boundary, fmt="%d"
    # )

    vprint(
        VERBOSE,
        _VERBOSITY,
        f"Time>> initial boundary-grid-point scaling: {init_scale_toc-init_scale_tic:.3f}s",
    )

    vprint(
        DEBUG,
        _VERBOSITY,
        "SCALE-BGP>> n_boundary_points:",
        num_boundary_points,
    )
    vprint(
        DEBUG,
        _VERBOSITY,
        "SCALE-BGP>> num_exposed_grid_points:",
        num_exposed_grid_points,
    )

    # Handle re-entrant surfaces by finding the closest exposed point for buried boundary points.
    if max_probe_radius > 0.0:
        num_fallthrough_bgps = 0
        cube_side_length_index_vertices = 1.0 / cube_side_indver_inverse

        renetrant_tic = time.perf_counter()
        # Iterate through boundary points again to handle re-entrant cases
        if use_cuda:
            # setup GPU
            blocks = 8  # or tune based on GPU

            # 1. Copy arrays to device
            d_surface_charge_positions = cuda.to_device(surface_charge_positions)
            d_scaled_surface_normal_vectors = cuda.to_device(
                scaled_surface_normal_vectors
            )
            d_atom_index_for_boundary = cuda.to_device(atom_index_for_boundary)
            d_current_grid_origin = cuda.to_device(current_grid_origin)
            d_system_min_coords = cuda.to_device(system_min_coords)
            d_exposed_grid_point_coords = cuda.to_device(exposed_grid_point_coords)
            d_cube_shape_indver = cuda.to_device(cube_shape_indver)
            d_cube_voxel_atom_index_start = cuda.to_device(cube_voxel_atom_index_start)
            d_cube_voxel_atom_index_end = cuda.to_device(cube_voxel_atom_index_end)
            d_cube_voxel_atom_index_cumulative = cuda.to_device(
                cube_voxel_atom_index_cumulative
            )
            d_is_outside_cube = cuda.to_device(is_outside_cube)

            d_num_fallthrough_bgps = cuda.device_array(1, dtype=np.int32)
            d_num_fallthrough_bgps[0] = 0

            # 2. Launch kernel
            threads_per_block = THREADS_PER_BLOCK
            blocks_per_grid = (
                num_boundary_points + threads_per_block - 1
            ) // threads_per_block
            _cuda_scale_reentrant_boundary_points_gridstride[
                int(blocks_per_grid), int(threads_per_block)
            ](
                num_boundary_points,
                d_surface_charge_positions,
                d_scaled_surface_normal_vectors,
                d_atom_index_for_boundary,
                grid_spacing,
                d_current_grid_origin,
                num_exposed_grid_points,
                d_system_min_coords,
                cube_side_length_index_vertices,
                d_cube_shape_indver,
                d_cube_voxel_atom_index_start,
                d_cube_voxel_atom_index_end,
                d_cube_voxel_atom_index_cumulative,
                d_exposed_grid_point_coords,
                d_is_outside_cube,
                probe_radius,
                probe_radius_2,
                d_num_fallthrough_bgps,
            )

            cuda.synchronize()

            # 3. Copy back results
            surface_charge_positions[:] = d_surface_charge_positions.copy_to_host()
            scaled_surface_normal_vectors[:] = (
                d_scaled_surface_normal_vectors.copy_to_host()
            )
            num_fallthrough_bgps = d_num_fallthrough_bgps.copy_to_host()[0]

            cuda.synchronize()
            d_surface_charge_positions = None
            d_scaled_surface_normal_vectors = None
            d_atom_index_for_boundary = None
            d_current_grid_origin = None
            d_system_min_coords = None
            d_exposed_grid_point_coords = None
            d_cube_shape_indver = None
            d_cube_voxel_atom_index_start = None
            d_cube_voxel_atom_index_end = None
            d_cube_voxel_atom_index_cumulative = None
            d_is_outside_cube = None
            d_num_fallthrough_bgps = None

            gc.collect()
        else:
            # CPU path
            set_num_threads(num_threads)
            num_fallthrough_bgps = _cpu_scale_reentrant_boundary_points(
                num_threads,
                num_boundary_points,
                surface_charge_positions,
                scaled_surface_normal_vectors,
                atom_index_for_boundary,
                grid_spacing,
                current_grid_origin,
                num_exposed_grid_points,
                system_min_coords,
                cube_side_length_index_vertices,
                cube_shape_indver,
                cube_voxel_atom_index_start,
                cube_voxel_atom_index_end,
                cube_voxel_atom_index_cumulative,
                exposed_grid_point_coords,
                is_outside_cube,
                probe_radius,
                probe_radius_2,
                delphi_real,
                delphi_int,
            )

        renetrant_toc = time.perf_counter()
        vprint(
            VERBOSE,
            _VERBOSITY,
            f"Time>> reentrant boundary-grid-point scaling: {renetrant_toc - renetrant_tic:.3f}s",
        )

    nprint_cpu(
        DEBUG,
        _VERBOSITY,
        "SCALE-BGP fall-through points>> num_fallthrough_bgps:",
        num_fallthrough_bgps,
    )

    return 0, voxel_space_scale
