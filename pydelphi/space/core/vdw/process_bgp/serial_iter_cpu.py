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


import numpy as np

from math import sqrt
from numba import njit

from pydelphi.foundation.enums import (
    Precision,
)

from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
    vprint,
    nprint_cpu_if_verbose as nprint_cpu,
)

from pydelphi.config.logging_config import (
    CRITICAL,
    ERROR,
    VERBOSE,
    DEBUG,
    TRACE,
    get_effective_verbosity,
)
from pydelphi.foundation.platforms import Platform

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

# # --- Dynamic Precision Handling ---
if PRECISION.int_value in {Precision.SINGLE.int_value}:
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

from pydelphi.constants import (
    ConstDelPhiInts,
    # ATOMFIELD_X,
    # ATOMFIELD_CRD_END,
    # ATOMFIELD_RADIUS,
    # ATOMFIELD_MEDIA_ID,
    NON_SOLUTE_BOUNDARY,
    SOLUTE_BOUNDARY_ANY,
    SOLUTE_BOUNDARY_EXTERNAL,
    ConstDelPhiFloats as ConstDelPhi,
    # NEIGHBOR_VOXEL_RELATIVE_COORDINATES as NEIGHBOR_VOXEL_REL_COORDS,
)

# Initialize module level constants based on global constants.
APPROX_ZERO = ConstDelPhi.ApproxZero.value
RADII_SQUARED_SHRINK_FACTOR = ConstDelPhi.SASSquaredRadiiShrinkFactor.value
MAX_NEIGHBR_BOUNDARY_ARRAY_LEN = ConstDelPhiInts.SpaceNBRASize.value
RESIZE_FACTOR = ConstDelPhi.ZetaArrayResizeFactor.value
INITIAL_SIZE_PERCENT = ConstDelPhi.ZetaArrayInitialSizePercent.value


EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value

# from pydelphi.space.core.vdw import helper as helpers
from pydelphi.space.core.vdw.internals import (
    _calculate_strides,
)
from pydelphi.space.core.vdw.process_bgp.helpers_cpu import (
    _vdw_to_ms_all_voxels,
    _scan_candidates_and_find_closest_neighbor,
    _check_contact_region,
    _get_media_id,
    _remap_epsilon_map,
    _remap_possibly_sign_updated_epsilon_map,
)


@njit(nogil=True, boundscheck=False, cache=True)
def _process_bgp_one_iteration(
    num_threads: int,  # unused but allows unified interface with parallel
    probe_radius_squared_1: float,
    probe_radius_squared_2: float,
    boundary_point_start_index: int,
    boundary_point_end_index: int,
    x_stride: int,
    y_stride: int,
    z_stride: int,
    grid_origin_current: np.ndarray,
    max_boundary_grid_points: int,
    grid_neighbor_coords_offsets: np.ndarray,
    boundary_grid_indices: np.ndarray,
    grid_spacing: float,
    grid_shape: np.ndarray,
    exposed_grids_coords: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
    index_map: np.ndarray,
    solute_bgp_type_1d: np.ndarray,
    epsilon_dimension: int,
    rm_boundary_pt_condition: bool,
    min_xyz: np.ndarray,
    cube_side_indver_inverse: float,
    cube_shape_indver: np.ndarray,
    cube_voxel_start_indices: np.ndarray,
    cube_voxel_end_indices: np.ndarray,
    cube_vertex_lowest_xyz: np.ndarray,
    cube_side_length_inverse: float,
    cube_shape: np.ndarray,
    voxel_grid_point_indices: np.ndarray,
    voxel_atom_count: np.ndarray,
    voxel_atom_count_cumulative: np.ndarray,
    voxel_atom_ids: np.ndarray,
    atom_surface_flags: np.ndarray,
    num_atoms: int,
    atoms_data: np.ndarray,
    atom_plus_probe_radii_1d: np.ndarray,
    atom_plus_probe_radii_shrink_1d: np.ndarray,
    num_external_boundary_points: int,
    dtype_int=int,
    dtype_real=float,
    dtype_bool=bool,
):
    """
    Processes the boundary grid points to determine the molecular surface.

    This function iterates through a range of identified boundary grid points and
    refines their status based on their local environment, specifically checking
    for overlap with atoms using the probe radius. It can remove boundary points
    that are deemed to be inside atoms (based on the `rm_boundary_pt_condition`)
    and updates the `solute_bgp_type_1d` array accordingly. It also manages
    counters for added and removed boundary points and the number of external
    boundary points.

    Args:
        probe_radius_squared_1 (float): The square of the probe radius.
        probe_radius_squared_2 (float): Another value related to the probe radius squared (purpose might vary).
        boundary_point_start_index (int): The starting index (1-based) in the `boundary_grid_indices_1d` array
            to begin processing.
        boundary_point_end_index (int): The ending index (inclusive, 1-based) in the `boundary_grid_indices_1d`
            array to stop processing.
        x_stride (int): The stride in the 1D grid array along the x-axis.
        y_stride (int): The stride in the 1D grid array along the y-axis.
        z_stride (int): The stride in the 1D grid array along the z-axis.
        grid_origin_current (np.ndarray): A NumPy array containing the coordinates of the origin of the current grid.
        max_boundary_grid_points (int): The maximum number of boundary grid points allowed.
        grid_neighbor_coords_offsets (np.ndarray): A NumPy array containing the coordinate offsets
            for the six neighboring midpoints of a grid point.
        boundary_grid_indices (np.ndarray): A NumPy array storing the 3D indices of the identified
            boundary grid points.
        grid_spacing (float): The spacing between grid points.
        grid_shape (np.ndarray): A NumPy array representing the dimensions of the grid (nx, ny, nz).
        exposed_grids_coords (np.ndarray): A NumPy array containing the coordinates of grid points
            that are considered 'exposed' (e.g., on the surface).
        index_discrete_epsilon_map_1d (np.ndarray): A 1D array representing the discrete
            dielectric constant map.
        index_map (np.ndarray): A 5x7 array used for mapping indices to neighbor offsets.
        solute_bgp_type_1d (np.ndarray): A 1D array used to mark grid points as being on
            the dielectric boundary. Stores additional information as well.
        epsilon_dimension (int): The number of different epsilon media.
        rm_boundary_pt_condition (bool): A boolean flag indicating whether to apply the condition
            for removing boundary points (e.g., based on the number of molecules).
        min_xyz (np.ndarray): A NumPy array containing the minimum x, y, and z coordinates of the system.
        cube_side_indver_inverse (float): The inverse of the side length of the indexing cube used for vertices.
        cube_shape_indver (np.ndarray): A NumPy array representing the shape of the indexing cube used for vertices.
        cube_voxel_start_indices (np.ndarray): A NumPy array containing the start indices of grid points
            within each voxel of the vertex indexing cube.
        cube_voxel_end_indices (np.ndarray): A NumPy array containing the end indices of grid points
            within each voxel of the vertex indexing cube.
        cube_vertex_lowest_xyz (np.ndarray): A NumPy array containing the coordinates of the lowest vertex
            of the indexing cube used for vertices.
        cube_side_length_inverse (float): The inverse of the side length of the main indexing cube.
        cube_shape (np.ndarray): A NumPy array representing the shape of the main indexing cube.
        voxel_grid_point_indices (np.ndarray): A NumPy array containing the 1D indices of exposed grid points
            that fall within each voxel of the vertex indexing cube.
        voxel_atom_count (np.ndarray): A NumPy array storing the number of atoms in each voxel of the
            main indexing cube.
        voxel_atom_count_cumulative (np.ndarray): A NumPy array storing the cumulative count of atoms
            in the voxels of the main indexing cube.
        voxel_atom_ids (np.ndarray): A NumPy array storing the IDs (1-based) of the atoms present in each
            voxel of the main indexing cube.
        atom_surface_flags (np.ndarray): A NumPy array of boolean flags indicating if an atom is on the surface.
        num_atoms (int): The total number of atoms in the system.
        atoms_data (np.ndarray): A NumPy array containing the data for each atom (including coordinates and radii).
        atom_plus_probe_radii_1d (np.ndarray): A 1D array storing the sum of the radius of each atom and the probe radius.
        atom_plus_probe_radii_shrink_1d (np.ndarray): A 1D array storing the squared sum of the radius of each atom
            and the probe radius, shrunk by a factor for contact detection.
        num_external_boundary_points (int): The current number of boundary grid points that are on the
            exterior of the solute. This count will be updated.
        dtype_int (type): The integer data type used in the calculations (e.g., `int` or `np.int32`).
        dtype_real (type): The floating-point data type used in the calculations (e.g., `float` or `np.float64`).
        dtype_bool (type): The boolean data type used in the calculations (e.g., `bool` or `np.bool_`).

    Returns:
        tuple[int, bool, int, int, int]:
            - neigh_update_exec_status (int): An execution status flag. Returns `EXIT_NJIT_FLAG` if an
              error occurs in the neighbor update process, otherwise 0.
            - cycle_flag (bool): A boolean flag indicating if a cycle was detected during the process (purpose might vary).
            - added_boundary_points_increment (int): The number of boundary points added during this loop iteration.
            - removed_boundary_points_increment (int): The number of boundary points removed during this loop iteration.
            - num_external_boundary_points (int): The updated number of external boundary points.
    """
    added_boundary_points_increment = 0
    removed_boundary_points_increment = 0
    grid_neighs_entity_ids = np.zeros(7, dtype=dtype_int)  # Renamed from eps
    grid_neighs_media_ids = np.zeros(7, dtype=dtype_int)  # Renamed from eps2

    x_stride_x_3 = x_stride * 3
    y_stride_x_3 = y_stride * 3

    grid_neighs_1d_offsets = np.zeros(7, dtype=dtype_int)
    # Leading by h/2 neighbors in x, y, z directions
    grid_neighs_1d_offsets[1:4] = 0, 1, 2
    # Lagging by h/2 neighbors in x, y, z directions
    grid_neighs_1d_offsets[4:7] = -x_stride_x_3, -y_stride_x_3 + 1, -z_stride * 3 + 2

    probe_radius = sqrt(probe_radius_squared_1)
    cycle_flag = False
    num_cavity_points = 0

    for this_bgp_index in range(
        boundary_point_start_index, boundary_point_end_index + 1
    ):
        ixyz = boundary_grid_indices[
            this_bgp_index - 1
        ]  # Adjust index for 0-based array
        grid_index_x = ixyz[0]
        grid_index_y = ixyz[1]
        grid_index_z = ixyz[2]
        grid_index_1d = (
            grid_index_x * x_stride + grid_index_y * y_stride + grid_index_z * z_stride
        )
        grid_index_1d_times_3 = grid_index_1d * 3
        if solute_bgp_type_1d[grid_index_1d] != NON_SOLUTE_BOUNDARY:
            rm_boundary_pt = False
            for neigh_id in range(1, 7):
                grid_neighs_entity_ids[neigh_id] = (
                    index_discrete_epsilon_map_1d[
                        grid_index_1d_times_3 + grid_neighs_1d_offsets[neigh_id]
                    ]
                    % epsilon_dimension
                )
                grid_neighs_media_ids[neigh_id] = (
                    index_discrete_epsilon_map_1d[
                        grid_index_1d_times_3 + grid_neighs_1d_offsets[neigh_id]
                    ]
                    // epsilon_dimension
                )

            rm_boundary_pt = (
                (1 < grid_neighs_entity_ids[1] <= num_atoms + 1)
                or (1 < grid_neighs_entity_ids[2] <= num_atoms + 1)
                or (
                    (1 < grid_neighs_entity_ids[3] <= num_atoms + 1)
                    or (1 < grid_neighs_entity_ids[4] <= num_atoms + 1)
                )
                or (
                    (1 < grid_neighs_entity_ids[5] <= num_atoms + 1)
                    or (1 < grid_neighs_entity_ids[6] <= num_atoms + 1)
                )
                or rm_boundary_pt
            )

            rm_boundary_pt = rm_boundary_pt and rm_boundary_pt_condition

            (
                neigh_update_exec_status,
                grid_neighs_entity_ids,
                grid_neighs_media_ids,
                num_cavity_points,
                solute_bgp_type_1d,
                num_external_boundary_points,
                boundary_grid_indices,
                added_boundary_points_increment,
                removed_boundary_points_increment,
            ) = _process_boundary_point_midpoint(
                boundary_point_index=this_bgp_index,
                grid_index_x=grid_index_x,
                grid_index_y=grid_index_y,
                grid_index_z=grid_index_z,
                grid_spacing=grid_spacing,
                grid_shape=grid_shape,
                grid_origin=grid_origin_current,
                probe_radius_squared_1=probe_radius_squared_1,
                probe_radius_squared_2=probe_radius_squared_2,
                index_map=index_map,
                grid_neighbor_coords_offsets=grid_neighbor_coords_offsets,
                grid_neighs_entity_ids=grid_neighs_entity_ids,
                grid_neighs_media_ids=grid_neighs_media_ids,
                rm_boundary_pt=rm_boundary_pt,
                min_xyz=min_xyz,
                cube_side_indver_inverse=cube_side_indver_inverse,
                cube_shape_indver=cube_shape_indver,
                index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
                cycle_flag=cycle_flag,
                cube_voxel_start_indices=cube_voxel_start_indices,
                cube_voxel_end_indices=cube_voxel_end_indices,
                grid_point_indices_in_voxels=voxel_grid_point_indices,
                exposed_grids_coords=exposed_grids_coords,
                cube_vertex_lowest_xyz=cube_vertex_lowest_xyz,
                cube_side_length_inverse=cube_side_length_inverse,
                cube_shape=cube_shape,
                voxel_atom_count=voxel_atom_count,
                voxel_atom_count_cumulative=voxel_atom_count_cumulative,
                voxel_atom_ids=voxel_atom_ids,
                atom_surface_flags=atom_surface_flags,
                num_atoms=num_atoms,
                atoms_data=atoms_data,
                atom_plus_probe_radii_1d=atom_plus_probe_radii_1d,
                atom_plus_probe_radii_shrink_1d=atom_plus_probe_radii_shrink_1d,
                epsilon_dimension=epsilon_dimension,
                boundary_point_end_index=boundary_point_end_index,
                max_boundary_grid_points=max_boundary_grid_points,
                num_cavity_points=num_cavity_points,
                solute_bgp_type_1d=solute_bgp_type_1d,
                num_external_boundary_points=num_external_boundary_points,
                boundary_grid_indices=boundary_grid_indices,
                added_boundary_points_increment=added_boundary_points_increment,
                removed_boundary_points_increment=removed_boundary_points_increment,
                dtype_int=dtype_int,
                dtype_real=dtype_real,
                dtype_bool=dtype_bool,
            )

            if neigh_update_exec_status != EXIT_NJIT_FLAG:
                # remap discrete_epsilon_map_1d in case there have been changes..
                for neighbor_index in range(1, 7):
                    index_discrete_epsilon_map_1d = (
                        _remap_possibly_sign_updated_epsilon_map(
                            grid_index_x=grid_index_x,
                            grid_index_y=grid_index_y,
                            grid_index_z=grid_index_z,
                            stride_x_3=x_stride_x_3,
                            stride_y_3=y_stride_x_3,
                            index_map=index_map,
                            boundary_point_index=this_bgp_index,
                            neighbor_index=neighbor_index,
                            epsilon_dimension=epsilon_dimension,
                            grid_neighs_entity_ids=grid_neighs_entity_ids,
                            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
                            dtype_int=dtype_int,
                            dtype_real=dtype_real,
                        )
                    )

                # NOTE: this is final neighbor update and boundary point marking, here
                # Neighbors are not tested rather the point (ix, iy, iz) itself is processed.
                (
                    neigh_update_exec_status,
                    solute_bgp_type_1d,
                    num_external_boundary_points,
                    removed_boundary_points_increment,
                ) = _update_gridpoint_boundary_status(
                    grid_index_x,
                    grid_index_y,
                    grid_index_z,
                    epsilon_dimension,
                    grid_shape,
                    index_discrete_epsilon_map_1d,
                    solute_bgp_type_1d,
                    num_external_boundary_points,
                    removed_boundary_points_increment,
                    dtype_int=dtype_int,
                )
    return (
        neigh_update_exec_status,
        cycle_flag,
        added_boundary_points_increment,
        removed_boundary_points_increment,
        num_external_boundary_points,
        boundary_grid_indices,
        solute_bgp_type_1d,
        index_discrete_epsilon_map_1d,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def _process_boundary_point_midpoint(
    boundary_point_index: int,
    grid_index_x: int,
    grid_index_y: int,
    grid_index_z: int,
    grid_spacing: float,
    grid_shape: np.ndarray,
    grid_origin: np.ndarray,
    probe_radius_squared_1: float,
    probe_radius_squared_2: float,
    index_map: np.ndarray,
    grid_neighbor_coords_offsets: np.ndarray,
    grid_neighs_entity_ids: np.ndarray,
    grid_neighs_media_ids: np.ndarray,
    rm_boundary_pt: bool,
    min_xyz: np.ndarray,
    cube_side_indver_inverse: float,
    cube_shape_indver: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
    cycle_flag: bool,
    cube_voxel_start_indices: np.ndarray,
    cube_voxel_end_indices: np.ndarray,
    grid_point_indices_in_voxels: np.ndarray,
    exposed_grids_coords: np.ndarray,
    cube_vertex_lowest_xyz: np.ndarray,
    cube_side_length_inverse: float,
    cube_shape: np.ndarray,
    voxel_atom_count: np.ndarray,
    voxel_atom_count_cumulative: np.ndarray,
    voxel_atom_ids: np.ndarray,
    atom_surface_flags: np.ndarray,
    num_atoms: int,
    atoms_data: np.ndarray,
    atom_plus_probe_radii_1d: np.ndarray,
    atom_plus_probe_radii_shrink_1d: np.ndarray,
    epsilon_dimension: int,
    boundary_point_end_index: int,
    max_boundary_grid_points: int,
    num_cavity_points: int,
    solute_bgp_type_1d: np.ndarray,
    num_external_boundary_points: int,
    boundary_grid_indices: np.ndarray,
    added_boundary_points_increment: int,
    removed_boundary_points_increment: int,
    dtype_int: type,
    dtype_real: type,
    dtype_bool: type,
) -> tuple[
    int,
    np.ndarray,
    np.ndarray,
    int,
    np.ndarray,
    int,
    np.ndarray,
    int,
    int,
]:
    """
    Processes a boundary grid point by checking its neighboring midpoints and updating
    dielectric boundary conditions based on atom proximity.

    This function determines whether a given midpoint lies inside or outside the indexing
    cube, finds the closest atom or object, and updates dielectric boundary values accordingly.

    Args:
        boundary_point_index (int): Index of the boundary point being processed.
        grid_index_x (int): X-coordinate index in the grid.
        grid_index_y (int): Y-coordinate index in the grid.
        grid_index_z (int): Z-coordinate index in the grid.
        grid_spacing (float): Spacing between grid points in the computational domain.
        grid_shape (np.ndarray): Dimensions (shape) of the grid.
        grid_origin (np.ndarray): Origin coordinates of the grid.
        probe_radius_squared_1 (float): Squared probe radius for boundary calculations.
        probe_radius_squared_2 (float): Alternative squared probe radius for different cases.
        grid_neighbor_coords_offsets (np.ndarray): Coordinate offsets for neighboring midpoints of a grid points.
        index_map (np.ndarray): Mapping from grid to atom indices.
        grid_neighs_entity_ids (np.ndarray): Epsilon entiry ids assigned to midpoints points.
        grid_neighs_media_ids (np.ndarray): Epsilon media ids for neighboring midpoints points.
        rm_boundary_pt (bool): Flag to indicate whether the boundary point should be removed.
        num_atoms (int): Total number of atoms in the system.
        min_xyz (np.ndarray): Minimum coordinates for the computational domain.
        cube_side_indver_inverse (float): Inverse cube side length factor for indexing.
        cube_shape_indver (np.ndarray): Shape of the indexing cube.
        index_discrete_epsilon_map_1d (np.ndarray): 1D mapping of epsilon values.
        cycle_flag (bool): Flag for iteration control in boundary processing.
        lower_atom_index_bounds (np.ndarray): Lower index bounds for atoms in the grid.
        upper_atom_index_bounds (np.ndarray): Upper index bounds for atoms in the grid.
        cumulative_atom_indices (np.ndarray): Cumulative indices for atoms in voxel storage.
        exposed_grids_coords (np.ndarray): Coordinates of exposed grid points.
        cube_vertex_lowest_xyz (np.ndarray): Lowest vertex coordinate of the cube.
        cube_side_length_inverse (float): Inverse length of a cube side.
        cube_shape (np.ndarray): Shape of the indexing cube.
        voxel_atom_count (np.ndarray): Number of atoms per voxel.
        voxel_atom_count_cumulative (np.ndarray): Cumulative count of atoms in voxels.
        voxel_atom_ids (np.ndarray): IDs of atoms in each voxel.
        atom_surface_flags (np.ndarray): Flags indicating which atoms are part of a surface.
        atoms_data (np.ndarray): Atomic data such as positions and radii.
        epsilon_dimension (int): Dimensionality of the epsilon array.
        num_cavity_points (int): Counter for cavity points detected. Defaults to 0.
        solute_bgp_type_1d (np.ndarray): Dielectric boundary mapping. Defaults to None.
        num_external_boundary_points (int): Count of external boundary points. Defaults to 0.
        boundary_grid_indices (np.ndarray): Grid indices marking boundaries. Defaults to None.
        added_boundary_points_increment (int): Loop index tracking media updates.
        removed_boundary_points_increment (int): Loop index tracking neighbor updates.
        dtype_int (type): Integer type for Delphi calculations.
        dtype_real (type): Floating-point type for Delphi calculations.
        dtype_bool (type): Boolean type for Delphi calculations.

    Returns:
        tuple:
            - Run status: neigh_update_exec_status, 0 on success EXIT_NJIT_FLAG on error
            - Updated `grid_neighs_entity_ids` array.
            - Updated `grid_neighs_media_ids` array.
            - Updated `num_cavity_points` count.
            - Updated `solute_bgp_type_1d` array.
            - Updated `num_external_boundary_points` count.
            - Updated `boundary_grid_indices_1d` array.
            - Updated `added_boundary_points_increment` count.
            - Updated `removed_boundary_points_increment` count.
    """
    probe_radius = sqrt(probe_radius_squared_1)
    last_cube_indver_indices = (cube_shape_indver - 1).astype(dtype_int)
    neigh_update_exec_status = (
        0  # Set it to success, it will be set to EXIT_NJIT_FLAG on error
    )

    grid_point_coords = (
        (
            grid_spacing
            * np.array([grid_index_x, grid_index_y, grid_index_z], dtype=dtype_int)
        )
        + grid_origin
    ).astype(dtype_real)

    z_stride = 1
    y_stride = dtype_int(grid_shape[2])
    x_stride = dtype_int(grid_shape[1] * y_stride)
    grid_index_1d = dtype_int(
        grid_index_x * x_stride + grid_index_y * y_stride + grid_index_z
    )
    grid_index_1d_times_3 = dtype_int(grid_index_1d * 3)

    for neighbor_index in range(1, 7):
        if (
            grid_neighs_entity_ids[neighbor_index] == 0
            or (
                rm_boundary_pt
                and grid_neighs_entity_ids[neighbor_index] > num_atoms + 1
            )
            or (
                grid_neighs_media_ids[neighbor_index] == 0
                and grid_neighs_entity_ids[neighbor_index] > 0
            )
        ):
            probe_radius_squared = probe_radius_squared_2
            if (
                grid_neighs_entity_ids[neighbor_index] == 0
                or grid_neighs_media_ids[neighbor_index] == 0
            ):
                probe_radius_squared = probe_radius_squared_1

            # add midpoint offset to grid point..
            midpoint_coords = (
                grid_point_coords + grid_neighbor_coords_offsets[neighbor_index]
            )
            # determine if this virgin midpoint is in or out of the indexing cube
            relative_coords = (midpoint_coords - min_xyz) * cube_side_indver_inverse
            midpoint_indices = relative_coords.astype(dtype_int)
            midpoint_index_x = midpoint_indices[0]
            midpoint_index_y = midpoint_indices[1]
            midpoint_index_z = midpoint_indices[2]

            if size_cpu.or_le_scalar(
                midpoint_indices, dtype_int(0)
            ) or size_cpu.or_ge_vector(midpoint_indices, last_cube_indver_indices):
                nprint_cpu(TRACE, _VERBOSITY, " VdMS> midpoint out of cube")
                nprint_cpu(
                    TRACE,
                    _VERBOSITY,
                    index_discrete_epsilon_map_1d[grid_index_1d_times_3],
                )

            min_distance_squared = dtype_real(1000.0)
            closest_atom_or_object_index_voxel = dtype_int(0)

            (
                cycle_flag,
                closest_atom_or_object_index_voxel,
                min_distance_squared,
            ) = _vdw_to_ms_all_voxels(
                cycle_flag,
                neighbor_index,
                midpoint_index_x,
                midpoint_index_y,
                midpoint_index_z,
                probe_radius_squared,
                midpoint_coords,
                grid_neighs_entity_ids,
                grid_neighs_media_ids,
                closest_atom_or_object_index_voxel,
                min_distance_squared,
                cube_voxel_start_indices,
                cube_voxel_end_indices,
                grid_point_indices_in_voxels,
                exposed_grids_coords,
            )
            if cycle_flag:
                continue

            # it might be in the contact region find the closest atom surface
            indexing_cube_indices = (
                (midpoint_coords - cube_vertex_lowest_xyz) * cube_side_length_inverse
            ).astype(dtype_int)
            min_distance_squared = dtype_real(100.0)
            closest_atom_or_object_index = dtype_int(0)
            num_neighbors_found = dtype_int(0)

            indexing_cube_dimensions = cube_shape.astype(dtype_int)

            if size_cpu.or_lt_scalar(
                indexing_cube_indices, dtype_int(0)
            ) or size_cpu.or_gt_vector(indexing_cube_indices, indexing_cube_dimensions):
                pass  # Logic for objects might go here
            else:
                lower_limit = voxel_atom_count[indexing_cube_indices[0]][
                    indexing_cube_indices[1]
                ][indexing_cube_indices[2]]
                upper_limit = voxel_atom_count_cumulative[indexing_cube_indices[0]][
                    indexing_cube_indices[1]
                ][indexing_cube_indices[2]]

            indexing_cube_dimensions = cube_shape.astype(dtype_int)

            if size_cpu.or_lt_scalar(
                indexing_cube_indices, dtype_int(0)
            ) or size_cpu.or_gt_vector(indexing_cube_indices, indexing_cube_dimensions):
                # Logic for objects might go here, currently implies empty neighbors
                lower_limit = dtype_int(1)
                upper_limit = dtype_int(0)  # Empty range
            else:
                lower_limit = voxel_atom_count[indexing_cube_indices[0]][
                    indexing_cube_indices[1]
                ][indexing_cube_indices[2]]
                upper_limit = voxel_atom_count_cumulative[indexing_cube_indices[0]][
                    indexing_cube_indices[1]
                ][indexing_cube_indices[2]]

            (
                closest_atom_or_object_index,
                min_distance_squared,
                atom_or_object_index,
            ) = _scan_candidates_and_find_closest_neighbor(
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                voxel_atom_ids=voxel_atom_ids,
                atom_surface_flags=atom_surface_flags,
                neighbor_entity_id=grid_neighs_entity_ids[neighbor_index],
                neighbor_index=neighbor_index,
                min_distance_squared=min_distance_squared,
                closest_atom_or_object_index=closest_atom_or_object_index,
                num_atoms=num_atoms,
                atoms_data=atoms_data,
                midpoint_coords=midpoint_coords,
                dtype_int=dtype_int,
                dtype_real=dtype_real,
            )

            if closest_atom_or_object_index == 0:
                num_cavity_points += 1  # possibly a cavity point
            else:
                (in_contact, neighbor_atom_or_object_index) = _check_contact_region(
                    midpoint_entity_id=grid_neighs_entity_ids[neighbor_index],
                    midpoint_coords=midpoint_coords,
                    closest_atom_or_object_index=closest_atom_or_object_index,
                    num_atoms=num_atoms,
                    atoms_data=atoms_data,
                    atom_plus_probe_radii_1d=atom_plus_probe_radii_1d,
                    atom_plus_probe_radii_shrink_1d=atom_plus_probe_radii_shrink_1d,
                    cube_side_length_inverse=cube_side_length_inverse,
                    cube_shape=cube_shape,
                    cube_vertex_lowest_xyz=cube_vertex_lowest_xyz,
                    voxel_atom_count=voxel_atom_count,
                    voxel_atom_count_cumulative=voxel_atom_count_cumulative,
                    voxel_atom_ids=voxel_atom_ids,
                    dtype_int=dtype_int,
                    dtype_real=dtype_real,
                    dtype_bool=dtype_bool,
                )
                if in_contact:
                    grid_neighs_entity_ids[neighbor_index] = (
                        -closest_atom_or_object_index
                    )
                    grid_neighs_media_ids[neighbor_index] = (
                        -closest_atom_or_object_index
                    )
                    continue

            grid_neighs_entity_ids[neighbor_index] = (
                1  # //eps = 1 means cavity or reentrant;
            )

            contact_region_media_id = _get_media_id(
                closest_atom_or_object_index,
                atom_or_object_index,
                num_atoms,
                atoms_data,
                dtype_int=dtype_int,
            )

            index_discrete_epsilon_map_1d = _remap_epsilon_map(
                grid_index_x=grid_index_x,
                grid_index_y=grid_index_y,
                grid_index_z=grid_index_z,
                stride_x_3=x_stride * 3,
                stride_y_3=y_stride * 3,
                index_map=index_map,
                boundary_point_index=boundary_point_index,
                neighbor_index=neighbor_index,
                epsilon_dimension=epsilon_dimension,
                contact_region_media_id=contact_region_media_id,
                grid_neighs_entity_ids=grid_neighs_entity_ids,
                index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
                dtype_int=dtype_int,
                dtype_real=dtype_real,
            )
            grid_neighs_media_ids[neighbor_index] = contact_region_media_id

            (
                neigh_update_exec_status,
                solute_bgp_type_1d,
                num_external_boundary_points,
                added_boundary_points_increment,
                removed_boundary_points_increment,
            ) = _update_neighbor_status(
                grid_index_x,
                grid_index_y,
                grid_index_z,
                neighbor_index,
                epsilon_dimension,
                grid_shape,
                index_discrete_epsilon_map_1d,
                boundary_grid_indices,
                solute_bgp_type_1d,
                num_external_boundary_points,
                added_boundary_points_increment,
                removed_boundary_points_increment,
                boundary_point_end_index,
                max_boundary_grid_points,
                dtype_int=dtype_int,
            )

    return (
        neigh_update_exec_status,
        grid_neighs_entity_ids,
        grid_neighs_media_ids,
        num_cavity_points,
        solute_bgp_type_1d,
        num_external_boundary_points,
        boundary_grid_indices,
        added_boundary_points_increment,
        removed_boundary_points_increment,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def _update_gridpoint_boundary_status(
    grid_index_x: int,
    grid_index_y: int,
    grid_index_z: int,
    epsilon_dimension: int,
    grid_shape: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
    solute_bgp_type_1d: np.ndarray,
    num_external_boundary_points: int,
    removed_boundary_points_increment: int,
    dtype_int,
) -> tuple[int, np.ndarray, int, int]:
    """
    Checks and updates the status of neighboring grid points based on epsilon values.

    This function examines a specific neighbor of a given grid point and determines
    if it lies on the dielectric boundary. It updates the `solute_bgp_type_1d`
    array to mark boundary points and keeps track of the number of external boundary
    points and the number of added or removed boundary points.

    Args:
        grid_index_x (int): The x-index of the central grid point.
        grid_index_y (int): The y-index of the central grid point.
        grid_index_z (int): The z-index of the central grid point.
        epsilon_dimension (int): The number of different epsilon media.
        grid_shape (np.ndarray): A NumPy array representing the dimensions of the grid (nx, ny, nz).
        index_discrete_epsilon_map_1d (np.ndarray): A 1D array representing the discrete
            dielectric constant map.
        solute_bgp_type_1d (np.ndarray): A 1D array used to mark grid points as being on
            the dielectric boundary. Stores additional information as well.
        num_external_boundary_points (int): The current number of boundary grid points that
            are on the exterior of the solute. This count is updated.
        removed_boundary_points_increment (int): A counter for the number of boundary points
            removed in the current iteration. This is updated.

    Returns:
        tuple[int, np.ndarray, int, int, int]:
            - exec_status (int): An execution status flag. Returns `EXIT_NJIT_FLAG` if an
              error occurs (e.g., exceeding the maximum number of boundary points), otherwise 0.
            - solute_bgp_type_1d (np.ndarray): The updated 1D array marking dielectric
              boundary points.
            - num_external_boundary_points (int): The updated count of external boundary points.
            - removed_boundary_points_increment (int): The updated count of removed boundary points.
    """
    exec_status = 0  # Assume function run is successful, on error return EXIT_NJIT_FLAG
    epsilon_dimension = dtype_int(epsilon_dimension)

    # Calculate strides to convert 3D indices to 1D index
    x_stride, y_stride, z_stride, x_stride_x_3, y_stride_x_3, z_stride_x_3 = (
        _calculate_strides(grid_shape, dtype_int=dtype_int)
    )
    neighbor_x, neighbor_y, neighbor_z = grid_index_x, grid_index_y, grid_index_z
    process_neighbor = True

    is_external = dtype_int(0)
    is_boundary = dtype_int(0)
    neighbor_index_1d = neighbor_x * x_stride + neighbor_y * y_stride + neighbor_z
    neighbor_index_1d_x_3 = neighbor_index_1d * 3
    neighbor_media_ids = np.zeros(7, dtype=dtype_int)

    grid_neighs_1d_offsets = np.zeros(7, dtype=dtype_int)
    # Offsets for the six neighboring midpoints in the 1D epsilon map array.
    # Leading by h/2 neighbors in x, y, z directions.
    grid_neighs_1d_offsets[1:4] = 0, 1, 2
    # Lagging by h/2 neighbors in x, y, z directions.
    grid_neighs_1d_offsets[4:7] = -x_stride_x_3, -y_stride_x_3 + 1, -z_stride_x_3 + 2

    # Get the media IDs of the six neighboring midpoints of the current neighbor.
    for neigh_index in range(1, 7):
        neighbor_media_ids[neigh_index] = (
            abs(
                index_discrete_epsilon_map_1d[
                    neighbor_index_1d_x_3 + grid_neighs_1d_offsets[neigh_index]
                ]
            )
            // epsilon_dimension
        )

    # Determine if the neighbor is external (solvent) based on the first midpoint.
    if neighbor_media_ids[1] == 0:
        is_external = 1
    # Check if there's a change in media ID between the first and last midpoints
    if neighbor_media_ids[1] != neighbor_media_ids[6]:
        is_boundary = 1
    # Check for changes in media ID across all adjacent midpoints
    for midpoint_index in range(2, 7):
        if neighbor_media_ids[midpoint_index] == 0:
            is_external = 1
        if neighbor_media_ids[midpoint_index] != neighbor_media_ids[midpoint_index - 1]:
            is_boundary = 1

    old_mark = solute_bgp_type_1d[neighbor_index_1d]
    old_ext = (
        solute_bgp_type_1d[neighbor_index_1d] & SOLUTE_BOUNDARY_EXTERNAL
    ) == SOLUTE_BOUNDARY_EXTERNAL

    # If the neighbor is no longer a boundary point
    if (is_external == 0) or (is_boundary == 0):
        # Decrement the external boundary point count if it was external
        num_external_boundary_points -= old_ext
        if (
            (is_boundary == 1)
            and (old_mark != NON_SOLUTE_BOUNDARY)
            and (old_ext == 1)
            and (is_external == 0)
        ):
            nprint_cpu(
                TRACE,
                _VERBOSITY,
                "GRIDPT EXTCLR (boundary stays):",
                (neighbor_x, neighbor_y, neighbor_z),
            )

        # Update the boundary point information
        if is_boundary == 1:
            solute_bgp_type_1d[neighbor_index_1d] = (
                SOLUTE_BOUNDARY_ANY | SOLUTE_BOUNDARY_EXTERNAL
            )

        if is_boundary == 0 and (old_mark != 0):
            nprint_cpu(
                TRACE,
                _VERBOSITY,
                "GRIDPT REMOVE:",
                (neighbor_x, neighbor_y, neighbor_z),
            )
        # Reset the boundary point information
        if is_boundary == 0:
            solute_bgp_type_1d[neighbor_index_1d] = NON_SOLUTE_BOUNDARY
            removed_boundary_points_increment += 1
            if is_external == 1:
                nprint_cpu(
                    TRACE,
                    _VERBOSITY,
                    "born a new external point on gridpoint",
                    (neighbor_x, neighbor_y, neighbor_z),
                )

    return (
        exec_status,
        solute_bgp_type_1d,
        num_external_boundary_points,
        removed_boundary_points_increment,
    )


@njit(nogil=True, boundscheck=False, cache=True)
def _update_neighbor_status(
    grid_index_x: int,
    grid_index_y: int,
    grid_index_z: int,
    neighbor_index: int,
    epsilon_dimension: int,
    grid_shape: np.ndarray,
    index_discrete_epsilon_map_1d: np.ndarray,
    boundary_grid_indices: np.ndarray,
    solute_bgp_type_1d: np.ndarray,
    num_external_boundary_points: int,
    added_boundary_points_increment: int,
    removed_boundary_points_increment: int,
    num_discovered_bndy_grid_points: int,
    max_boundary_grid_points: int,
    dtype_int,
) -> tuple[int, np.ndarray, int, int, int]:
    """
    Checks and updates the status of neighboring grid points based on epsilon values.

    This function examines a specific neighbor of a given grid point and determines
    if it lies on the dielectric boundary. It updates the `solute_bgp_type_1d`
    array to mark boundary points and keeps track of the number of external boundary
    points and the number of added or removed boundary points.

    Args:
        grid_index_x (int): The x-index of the central grid point.
        grid_index_y (int): The y-index of the central grid point.
        grid_index_z (int): The z-index of the central grid point.
        neighbor_index (int): The index of the neighbor being considered (1-6).
            Passing 7 skips neighbor marking.
            1: +x, 2: +y, 3: +z, 4: -x, 5: -y, 6: -z.
        epsilon_dimension (int): The number of different epsilon media.
        grid_shape (np.ndarray): A NumPy array representing the dimensions of the grid (nx, ny, nz).
        index_discrete_epsilon_map_1d (np.ndarray): A 1D array representing the discrete
            dielectric constant map.
        boundary_grid_indices (np.ndarray): A NumPy array to store the 3D indices of boundary
            grid points. This array is updated if a new boundary point is found.
        solute_bgp_type_1d (np.ndarray): A 1D array used to mark grid points as being on
            the dielectric boundary. Stores additional information as well.
        num_external_boundary_points (int): The current number of boundary grid points that
            are on the exterior of the solute. This count is updated.
        added_boundary_points_increment (int): A counter for the number of boundary points
            added in the current iteration. This is updated.
        removed_boundary_points_increment (int): A counter for the number of boundary points
            removed in the current iteration. This is updated.
        num_discovered_bndy_grid_points (int): The total count of boundary grid points
            discovered so far.
        max_boundary_grid_points (int): The maximum allowed number of boundary grid points.
        dtype_int (type): The integer data type used in the calculations (e.g., np.int32).

    Returns:
        tuple[int, np.ndarray, int, int, int]:
            - exec_status (int): An execution status flag. Returns `EXIT_NJIT_FLAG` if an
              error occurs (e.g., exceeding the maximum number of boundary points), otherwise 0.
            - solute_bgp_type_1d (np.ndarray): The updated 1D array marking dielectric
              boundary points.
            - num_external_boundary_points (int): The updated count of external boundary points.
            - added_boundary_points_increment (int): The updated count of added boundary points.
            - removed_boundary_points_increment (int): The updated count of removed boundary points.
    """
    exec_status = 0  # Assume function run is successful, on error return EXIT_NJIT_FLAG
    epsilon_dimension = dtype_int(epsilon_dimension)

    # Calculate strides to convert 3D indices to 1D index
    x_stride, y_stride, z_stride, x_stride_x_3, y_stride_x_3, z_stride_x_3 = (
        _calculate_strides(grid_shape, dtype_int=dtype_int)
    )
    neighbor_x, neighbor_y, neighbor_z = grid_index_x, grid_index_y, grid_index_z

    # Adjust neighbor indices based on the neighbor_index
    if 1 <= neighbor_index <= 6:
        if neighbor_index == 1:
            neighbor_x += 1
            if neighbor_x >= grid_shape[0] - 1:
                return (
                    exec_status,
                    solute_bgp_type_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                )
        elif neighbor_index == 2:
            neighbor_y += 1
            if neighbor_y >= grid_shape[1] - 1:
                return (
                    exec_status,
                    solute_bgp_type_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                )
        elif neighbor_index == 3:
            neighbor_z += 1
            if neighbor_z >= grid_shape[2] - 1:
                return (
                    exec_status,
                    solute_bgp_type_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                )
        elif neighbor_index == 4:
            neighbor_x -= 1
            if neighbor_x <= 0:
                return (
                    exec_status,
                    solute_bgp_type_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                )
        elif neighbor_index == 5:
            neighbor_y -= 1
            if neighbor_y <= 0:
                return (
                    exec_status,
                    solute_bgp_type_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                )
        elif neighbor_index == 6:
            neighbor_z -= 1
            if neighbor_z <= 0:
                return (
                    exec_status,
                    solute_bgp_type_1d,
                    num_external_boundary_points,
                    added_boundary_points_increment,
                    removed_boundary_points_increment,
                )

    is_external = dtype_int(0)
    is_boundary = dtype_int(0)
    neighbor_index_1d = neighbor_x * x_stride + neighbor_y * y_stride + neighbor_z
    neighbor_index_1d_x_3 = neighbor_index_1d * 3
    neighbor_media_ids = np.zeros(7, dtype=dtype_int)

    grid_neighs_1d_offsets = np.zeros(7, dtype=dtype_int)
    # Offsets for the six neighboring midpoints in the 1D epsilon map array.
    # Leading by h/2 neighbors in x, y, z directions.
    grid_neighs_1d_offsets[1:4] = 0, 1, 2
    # Lagging by h/2 neighbors in x, y, z directions.
    grid_neighs_1d_offsets[4:7] = -x_stride_x_3, -y_stride_x_3 + 1, -z_stride_x_3 + 2

    # Get the media IDs of the six neighboring midpoints of the current neighbor.
    for neigh_index in range(1, 7):
        neighbor_media_ids[neigh_index] = (
            abs(
                index_discrete_epsilon_map_1d[
                    neighbor_index_1d_x_3 + grid_neighs_1d_offsets[neigh_index]
                ]
            )
            // epsilon_dimension
        )

    # Determine if the neighbor is external (solvent) based on the first midpoint.
    if neighbor_media_ids[1] == 0:
        is_external = 1
    # Check if there's a change in media ID between the first and last midpoints
    if neighbor_media_ids[1] != neighbor_media_ids[6]:
        is_boundary = 1
    # Check for changes in media ID across all adjacent midpoints
    for midpoint_index in range(2, 7):
        if neighbor_media_ids[midpoint_index] == 0:
            is_external = 1
        if neighbor_media_ids[midpoint_index] != neighbor_media_ids[midpoint_index - 1]:
            is_boundary = 1

    # If the neighbor is no longer a boundary point
    if (is_boundary == 0) and (
        solute_bgp_type_1d[neighbor_index_1d] != NON_SOLUTE_BOUNDARY
    ):
        # Decrement the external boundary point count if it was external
        old_is_external = (
            solute_bgp_type_1d[neighbor_index_1d] & SOLUTE_BOUNDARY_EXTERNAL
        ) != 0
        num_external_boundary_points -= old_is_external
        # Reset the boundary point information
        solute_bgp_type_1d[neighbor_index_1d] = NON_SOLUTE_BOUNDARY
        removed_boundary_points_increment += 1
        nprint_cpu(
            TRACE,
            _VERBOSITY,
            "removing boundary point: ",
            (neighbor_x, neighbor_y, neighbor_z),
        )
    else:
        # If it's a boundary and was marked as external, but shouldn't be anymore
        if (
            is_boundary == 1
            and is_external == 0
            and solute_bgp_type_1d[neighbor_index_1d] != NON_SOLUTE_BOUNDARY
        ):
            num_external_boundary_points -= 1
            solute_bgp_type_1d[neighbor_index_1d] = NON_SOLUTE_BOUNDARY
            nprint_cpu(
                TRACE,
                _VERBOSITY,
                "changing ext_bgp into int_bgp: ",
                (neighbor_x, neighbor_y, neighbor_z),
            )

    # If the neighbor is a boundary point and not already marked as one
    if (
        is_boundary == 1
        and solute_bgp_type_1d[neighbor_index_1d] == NON_SOLUTE_BOUNDARY
    ):
        added_boundary_points_increment += 1

        if (
            num_discovered_bndy_grid_points + added_boundary_points_increment
            > max_boundary_grid_points
        ):
            nprint_cpu(
                CRITICAL,
                _VERBOSITY,
                " ERROR> This case is too big, ibmx need to be increased.",
            )
            exec_status = EXIT_NJIT_FLAG

        if exec_status != EXIT_NJIT_FLAG:
            # If the limit is not exceeded, store the boundary point's index
            boundary_grid_indices[
                num_discovered_bndy_grid_points + added_boundary_points_increment - 1
            ][:] = np.array([neighbor_x, neighbor_y, neighbor_z], dtype=dtype_int)
            solute_bgp_type_1d[neighbor_index_1d] = (
                SOLUTE_BOUNDARY_ANY | SOLUTE_BOUNDARY_EXTERNAL
            )
            num_external_boundary_points += is_external
            nprint_cpu(
                TRACE,
                _VERBOSITY,
                "creating boundary_point: ",
                (neighbor_x, neighbor_y, neighbor_z),
            )
    return (
        exec_status,
        solute_bgp_type_1d,
        num_external_boundary_points,
        added_boundary_points_increment,
        removed_boundary_points_increment,
    )
