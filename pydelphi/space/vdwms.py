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


"""
This module provides utility functions and a class (`SurfaceMolecularVdW`)
for setting up and processing the computational grid used in DelPhi calculations,
specifically for generating the Van der Waals molecular surface.

The module includes functionalities for:

- Defining grid properties and boundaries.
- Identifying boundary grid points based on dielectric constant changes.
- Handling the molecular surface using probe radius.
- Setting up an indexing cube for efficient neighbor searching.
- Remapping the dielectric constant map based on molecular surface information.
- Updating the status of neighboring grid points.
- Handling the zeta surface for advanced calculations.

The `SurfaceMolecularVdW` class orchestrates these functionalities to create the
molecular surface representation.

These functions and the class methods are often decorated with `@njit`
for performance optimization using Numba. The module also includes
precision-dependent imports to handle single, double, and mixed-precision
calculations.
"""

import time
import numpy as np

from numba import njit

from pydelphi.config.global_runtime import (
    delphi_bool,
    delphi_int,
    delphi_real,
    vprint,
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

from pydelphi.constants import (
    ConstDelPhiInts,
    ConstDelPhiFloats,
)

# Initialize module level constants based on global constants.
EXIT_NJIT_FLAG = ConstDelPhiInts.ExitNjitReturnValue.value

APPROX_ZERO = ConstDelPhiFloats.ApproxZero.value
ZERO_MOLAR_DEBYE_LENGHT = ConstDelPhiFloats.ZeroMolarSaltDebyeLength.value
RADII_SQUARED_SHRINK_FACTOR = ConstDelPhiFloats.SASSquaredRadiiShrinkFactor.value
RESIZE_FACTOR = ConstDelPhiFloats.ZetaArrayResizeFactor.value
INITIAL_SIZE_PERCENT = ConstDelPhiFloats.ZetaArrayInitialSizePercent.value

import pydelphi.space.core.voxelizer as voxelizer
from pydelphi.space.core.vdw.internals import (
    _calculate_strides,
    _calculate_grid_properties,
    _calculate_solute_grid_boundaries,
    _calculate_atom_probe_radii,
    _calculate_rms,
    _calculate_cube_voxels_per_entity,
    _handle_zero_probe_radius,
    _set_constant_values,
    _setup_grid_neighbor_coords_offsets,
)
from pydelphi.space.core.vdw.init_boundary_finder import find_boundary_grid_points
from pydelphi.space.core.vdw import helper as helpers
from pydelphi.space.core.vdw.sas_builder import sas_parallel as sas
from pydelphi.space.core.vdw.scale_bgp import scale_boundary as sclbp

from pydelphi.utils.io.writers import write_nparray_to_npy


# Deferred imports for precision-sensitive components
def configure_precision_dependent_imports():
    # Global identifiers (variables, imported-module-aliases and data_types) declaration
    global delphi_bool, delphi_int, delphi_real
    global helpers, sas, sclbp

    from pydelphi.space.core.vdw import helper as helpers
    from pydelphi.space.core.vdw.sas_builder import sas_parallel as sas
    from pydelphi.space.core.vdw.scale_bgp import scale_boundary as sclbp


def _dbg_epsmap(tag, arr, epsdim=None, sample=10):
    if arr is None:
        print(f"[{tag}] epsmap: None")
        return

    a = np.asarray(arr)
    print(f"[{tag}] epsmap dtype={a.dtype}, shape={a.shape}, nbytes={a.nbytes:,}")

    # quick stats (safe for ints/floats)
    try:
        amin = a.min()
        amax = a.max()
        print(f"[{tag}] epsmap min={amin}, max={amax}")
    except Exception as e:
        print(f"[{tag}] epsmap min/max failed: {e}")

    head = a.ravel()[:sample]
    print(f"[{tag}] epsmap head[{sample}] = {head}")

    if epsdim is not None:
        # media id = abs(eps) // epsdim, entity id = abs(eps) % epsdim
        # This is the exact semantic your CPU uses (abs() first!).
        abs_head = np.abs(head.astype(np.int64, copy=False))
        media = abs_head // int(epsdim)
        ent = abs_head % int(epsdim)
        print(f"[{tag}] head media = {media}")
        print(f"[{tag}] head entity= {ent}")


@njit(nogil=True, boundscheck=False, cache=True)
def _perform_cube_calculation(
    num_atoms: int,
    num_objects: int,
    num_molecules: int,
    voxels_per_entity: int,
    cube_side_length: float,
    cube_origin: np.ndarray,
    cube_shape: np.ndarray,
    atoms_data: np.ndarray,
    dtype_int,
    dtype_real,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs the calculation to populate the indexing cube with atom and object IDs (standalone version).

    Args:
        num_atoms (int): The total number of atoms.
        num_objects (int): The total number of objects.
        num_molecules (int): The total number of molecules.
        voxels_per_entity (int): The number of voxels per entity.
        cube_side_length (float): The length of each side of the cube.
        cube_origin (np.ndarray): The coordinates of the lowest vertex of the cube.
        cube_shape (np.ndarray): The dimensions (shape) of the cube grid.
        atoms_data (np.ndarray): Array containing atom properties.
        dtype_int (type): Integer data type. Defaults to int.
        dtype_real (type): Real data type. Defaults to float.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
               - voxel_atom_ids (np.ndarray): 1D array mapping voxel indices to atom/object IDs.
               - voxel_atom_start_indices (np.ndarray): Array storing the count of atoms/objects in each voxel.
               - voxel_atom_end_indices (np.ndarray): Array storing the cumulative count.
    """
    voxel_atom_ids = np.zeros(
        voxels_per_entity * (num_atoms + num_objects - num_molecules) + 1,
        dtype=dtype_int,
    )
    voxel_atom_ids, voxel_atom_start_indices, voxel_atom_end_indices = (
        voxelizer.build_atom_voxel_map(
            dtype_real(cube_side_length),
            dtype_int(num_atoms),
            dtype_int(num_objects),
            dtype_int(num_molecules),
            cube_origin.astype(dtype_real),
            cube_shape.astype(dtype_int),
            atoms_data.astype(dtype_real),
            voxel_atom_ids,
        )
    )
    return voxel_atom_ids, voxel_atom_start_indices, voxel_atom_end_indices


def _setup_cube_for_indexing(
    grid_scale: float,
    max_probe_radius: float,
    max_atom_radius: float,
    min_coords_by_axis: np.ndarray,
    max_coords_by_axis: np.ndarray,
    dtype_real,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Sets up a cubic grid for efficient indexing of atoms and objects (standalone version).

    Args:
        grid_scale (float): Number of grids per angstrom.
        max_probe_radius (float): The maximum radius of the probe molecule.
        max_atom_radius (float): The maximum radius of any atom in the system.
        min_coords_by_axis (np.ndarray): Minimum coordinates of the system along each axis (x, y, z).
        max_coords_by_axis (np.ndarray): Maximum coordinates of the system along each axis (x, y, z).
        dtype_real (type): Real data type.

    Returns:
        tuple[np.ndarray, np.ndarray, float, float]: A tuple containing:
               - voxel_space_origin (np.ndarray): The coordinates of the lowest vertex of the cube.
               - voxel_space_shape (np.ndarray): The dimensions (shape) of the cube grid.
               - voxel_side_length (float): The length of each side of the cube.
               - inverse_cube_side_length (float): The inverse of the cube side length.
    """
    delta = dtype_real(1.0 / grid_scale)
    delta = max(delta, max_probe_radius)
    voxel_side_length = dtype_real(max_atom_radius + delta)
    inverse_cube_side_length = dtype_real(1.0 / voxel_side_length)
    min_coords_array = np.asarray(min_coords_by_axis, dtype=dtype_real)
    max_coords_array = np.asarray(max_coords_by_axis, dtype=dtype_real)

    voxel_space_origin, voxel_space_shape = voxelizer.calculate_voxel_space_parameters(
        dtype_real(voxel_side_length),
        min_coords_array,
        max_coords_array,
        scaling_factor=dtype_real(2.0),
        voxel_space_offset=dtype_real(0.1),
    )
    return (
        voxel_space_origin,
        voxel_space_shape,
        voxel_side_length,
        inverse_cube_side_length,
    )


def _handle_zeta_surface(
    use_zeta_surface: bool,
    grid_spacing: float,
    grid_shape: tuple,
    gridbox_center: np.ndarray,
    mid_grid_indices: np.ndarray,
    zeta_surface_map_1d: np.ndarray,
    epsilon_dimension: int,
    index_discrete_epsilon_map_1d: np.ndarray,
    dtype_int,
    dtype_real,
) -> tuple[np.ndarray, np.ndarray, int, int, int, int]:
    """
    Builds the zeta surface map if use_zeta_surface is True (standalone version).

    Args:
        use_zeta_surface (bool): Flag indicating whether to build the zeta surface map.
        grid_spacing (float): The spacing between grid points.
        grid_shape (tuple): The dimensions of the grid (nx, ny, nz).
        gridbox_center (np.ndarray): The coordinates of the center of the grid.
        mid_grid_indices (np.ndarray): The indices of the middle grid point in each dimension.
        zeta_surface_map_1d (np.ndarray): 1D array representing the zeta surface.
        epsilon_dimension (int): The number of discrete dielectric constants.
        epsilon_map_1d (np.ndarray): 1D array representing the discrete dielectric constant map.
        dtype_int (type): Integer data type. Defaults to int.
        dtype_real (type): Real data type. Defaults to float.

    Returns:
        tuple[np.ndarray, np.ndarray, int, int, int, int]: A tuple containing the updated zeta surface grid coordinates, indices,
               number of points, and capacities.
    """
    zeta_surface_grid_coords = np.array(3, dtype=dtype_real)
    zeta_surface_grid_indices = np.array(3, dtype=dtype_int)
    num_zeta_point_coords = dtype_int(0)
    num_zeta_point_indices = dtype_int(0)
    zeta_coords_capacity = dtype_int(0)
    zeta_indices_capacity = dtype_int(0)

    if use_zeta_surface:
        max_possible_points = dtype_int(grid_shape[0] * grid_shape[1] * grid_shape[2])
        initial_size_coords = dtype_int(max_possible_points * INITIAL_SIZE_PERCENT) * 3
        initial_size_indices = dtype_int(max_possible_points * INITIAL_SIZE_PERCENT) * 3

        current_zeta_surf_grid_coords = np.zeros(initial_size_coords, dtype=dtype_real)
        current_zeta_surf_grid_indices = np.zeros(initial_size_indices, dtype=dtype_int)
        current_num_zeta_point_coords = dtype_int(0)
        current_num_zeta_point_indices = dtype_int(0)
        current_zeta_coords_capacity = dtype_int(initial_size_coords)
        current_zeta_indices_capacity = dtype_int(initial_size_indices)

        (
            zeta_surface_grid_coords,
            zeta_surface_grid_indices,
            num_zeta_point_coords,
            num_zeta_point_indices,
            zeta_coords_capacity,
            zeta_indices_capacity,
        ) = helpers.build_zeta_surface_map(
            grid_spacing=grid_spacing,
            grid_shape=grid_shape,
            gridbox_center=gridbox_center,
            indices_mid_grid=mid_grid_indices,
            zeta_surface_map_1d=zeta_surface_map_1d,
            index_discrete_epsilon_map_1d=index_discrete_epsilon_map_1d,
            epsdim=epsilon_dimension,
            zeta_surf_grid_coords=current_zeta_surf_grid_coords,
            zeta_surf_grid_indices=current_zeta_surf_grid_indices,
            zeta_coords_capacity=current_zeta_coords_capacity,
            zeta_indices_capacity=current_zeta_indices_capacity,
            num_zeta_surf_grid_coords=current_num_zeta_point_coords,
            num_zeta_surf_grid_indices=current_num_zeta_point_indices,
        )
    return (
        zeta_surface_grid_coords,
        zeta_surface_grid_indices,
        num_zeta_point_coords,
        num_zeta_point_indices,
        zeta_coords_capacity,
        zeta_indices_capacity,
    )


def _check_boundary_point_limit(
    num_boundary_grid_points: int,
    max_boundary_grid_points: int,
):
    """
    Checks if the number of boundary grid points exceeds the allowed maximum (standalone version).

    Args:
        num_boundary_grid_points (int): The current number of boundary grid points.
        max_boundary_grid_points (int): The maximum allowed number of boundary grid points.
    """
    if num_boundary_grid_points > max_boundary_grid_points:
        vprint(
            CRITICAL,
            _VERBOSITY,
            " WARNING iBoundNum= ",
            num_boundary_grid_points,
            " is greater than ibmx = ",
            max_boundary_grid_points,
        )
        vprint(CRITICAL, _VERBOSITY, " CRITICAL> Increase `max_boundary_grid_points`")
        exit(0)


def _print_dielectric_boundary(
    solute_bgp_type_1d: np.ndarray,
    grid_shape: np.ndarray,
):
    """
    Prints the dielectric boundary information if the verbosity level is high enough (standalone version).

    Args:
        solute_bgp_type_1d (np.ndarray): 1D array containing information about the dielectric boundary.
        grid_shape (np.ndarray): The dimensions of the grid (nx, ny, nz).
    """
    if _VERBOSITY <= TRACE:
        helpers.print_3d_array(
            "self.solute_bgp_type_1d bndeps: (ijk1d, iix, iiy, iiz, iext)",
            solute_bgp_type_1d,
            list(grid_shape),
        )


class SurfaceMolecularVdW:
    """
    A class responsible for creating the Van der Waals molecular surface
    representation used in DelPhi calculations.

    This class manages the grid, atom, and object data, and orchestrates the
    process of identifying boundary points and determining the molecular surface
    based on probe radius and atom/object radii.
    """

    def __init__(
        self,
        platform,
        grid_spacing,
        probe_radius,
        probe_radius_second,
        debye_length,
        salt_radius,
        radius_offset,
        max_radius,
        max_atom_radius,
        surface_offset,
        grid_shape,
        grid_origin,
        grid_scale,
        min_coords_by_axis,
        max_coords_by_axis,
        grid_shape_parentrun,
        grid_origin_parentrun,
        atoms_data,
        atom_index_array,
        objects_data,
        n_objects,
        n_molecules,
        surface_charge_positions,
        is_focusing,
        use_zeta_surface_calculation,
        index_discrete_epsilon_map_1d,
        dielectric_boundary_map_1d,
        zeta_surface_map_1d,
        verbosity,
        approx_zero,
    ):
        """
        Initializes the SurfaceMolecularVdW object.

        Args:
            platform (str): The computational platform (e.g., 'cpu', 'cuda').
            grid_spacing (float): The spacing between grid points in Angstroms.
            probe_radius (float): The radius of the probe molecule in Angstroms.
            probe_radius_second (float): A second probe radius for specific cases.
            debye_length (float): The Debye length of the solution.
            salt_radius (float): The radius of the salt ions.
            radius_offset (float): An offset applied to atomic radii.
            max_radius (float): The maximum radius considered in the system.
            max_atom_radius (float): The maximum radius of any atom in the system.
            surface_offset (float): An offset for surface calculations.
            grid_shape (tuple): A tuple (nx, ny, nz) representing the dimensions of the grid.
            grid_origin (np.ndarray): A NumPy array [x, y, z] representing the origin of the grid.
            grid_scale (float): The number of grid points per Angstrom.
            min_coords_by_axis (np.ndarray): Minimum coordinates of the system along each axis.
            max_coords_by_axis (np.ndarray): Maximum coordinates of the system along each axis.
            grid_shape_parentrun (tuple): Grid shape of the parent run (for focusing).
            grid_origin_parentrun (np.ndarray): Grid origin of the parent run (for focusing).
            atoms_data (np.ndarray): A NumPy array containing atomic data (coordinates, radii, etc.).
            atom_index_array (np.ndarray): An array mapping atom indices.
            objects_data (np.ndarray): A NumPy array containing data for geometric objects.
            n_objects (int): The number of geometric objects.
            n_molecules (int): The number of molecules in the system.
            surface_charge_positions (np.ndarray): Positions of surface charges.
            is_focusing (bool): A flag indicating if this is a focusing run.
            use_zeta_surface_calculation (bool): A flag to use zeta surface calculation.
            index_discrete_epsilon_map_1d (np.ndarray): 1D array of the discrete epsilon map.
            dielectric_boundary_map_1d (np.ndarray): 1D array of the dielectric boundary map at grid-points.
            zeta_surface_map_1d (np.ndarray): 1D array of the zeta surface map.
            verbosity (int): The verbosity level for output.
            approx_zero (float): A small value considered as zero.
        """
        # Lazy global precision dependent module imports
        configure_precision_dependent_imports()
        # set members
        self.num_cuda_threads = 256
        self.surface_method = None
        self.calculation_platform = platform
        self.grid_spacing = grid_spacing
        self.probe_radius = probe_radius
        self.probe_radius_second = probe_radius_second
        self.debye_length = debye_length
        self.salt_radius = salt_radius
        self.radius_offset = radius_offset
        self.surface_offset = surface_offset
        self.approx_zero = approx_zero
        self.grid_shape = grid_shape
        self.grid_origin = grid_origin
        self.grid_scale = grid_scale
        self.min_coords_by_axis = min_coords_by_axis
        self.max_coords_by_axis = max_coords_by_axis
        self.grid_shape_parentrun = grid_shape_parentrun
        self.grid_origin_parentrun = grid_origin_parentrun
        self.atoms_data = atoms_data
        self.atom_index_array = atom_index_array
        self.objects_data = objects_data
        self.maximum_radius = max_radius
        self.max_atom_radius = max_atom_radius
        self.num_molecules = n_molecules
        # positions of induced surface charges
        self.surface_charge_positions = surface_charge_positions
        self.use_zeta_surface_calculation = use_zeta_surface_calculation
        self.zeta_surface_grid_coords = None
        self.zeta_surface_grid_indices = None
        self.discrete_epsilon_index_map_1d = index_discrete_epsilon_map_1d
        self.dielectric_boundary_map_1d = dielectric_boundary_map_1d
        self.zeta_surface_map_1d = zeta_surface_map_1d
        self.surface_map_midpoints_1d = None
        self.is_focusing = is_focusing
        self.verbosity = verbosity
        # local fields initialized here
        self.num_grid_points = 0
        self.num_exposed_grids = 0
        self.num_atoms = len(self.atoms_data) if self.atoms_data is not None else 0
        self.num_objects = n_objects
        self.cube_side_length_inverse = None
        self.voxel_space_origin = np.zeros(3, dtype=delphi_int)
        self.cube_vertex_highest_xyz = np.zeros(3, dtype=delphi_int)
        self.voxel_space_shape = np.zeros(3, dtype=delphi_int)
        self.num_boundary_grid_points = None
        self.num_external_boundary_points = None
        self.epsilon_dimension = self.num_atoms + self.num_objects + 2

        # Boundary points related info
        self.max_boundary_grid_points = None
        self.boundary_grid_indices = None
        self.boundary_grid_points = None
        self.solute_bgp_type_1d = None
        self.point_indices_by_voxel = None

        self.atom_plus_probe_radii_1d = None
        self.atom_plus_probe_radii_square_1d = None
        self.atom_plus_probe_radii_square_shrunk_1d = None

        # Zeta surface fields
        self.zeta_surface_grid_coords = None
        self.zeta_surface_grid_indices = None
        self.num_zeta_surface_point_coords = 0
        self.num_zeta_surface_point_indices = 0
        self.zeta_surface_coords_capacity = 0
        self.zeta_surface_indices_capacity = 0

        self.sLimObject = None
        self.voxel_atom_count = np.zeros(1, dtype=delphi_int)
        self.voxel_atom_count_cumulative = np.zeros(1, dtype=delphi_int)
        self.atom_accessibility = None
        self.limeps_min = np.zeros(3, dtype=delphi_int)
        self.limeps_max = np.zeros(3, dtype=delphi_int)
        self.exposed_grids_coords = np.zeros((0, 3), dtype=delphi_real)
        self.zeta_surfmap = None
        self.boundary_grid_points = np.zeros((0, 3), dtype=delphi_int)
        # Atoms-neighbor lookup utility fields
        self.indexing_voxel_scale = None
        self.indexing_voxel_origin = None
        self.indexing_voxel_shape = None
        self.voxel_point_start_indices = np.zeros(1, dtype=delphi_int)
        self.voxel_point_end_indices = np.zeros(1, dtype=delphi_int)
        self.scaled_surface_normal_vectors = None

    def _initialize_arrays(self, num_atoms):
        """Initializes NumPy arrays used in the VdwToMs method."""
        index_map = np.zeros((5, 7), dtype=delphi_int)  # Initialize index_map
        neighbor_exists_array = np.zeros(
            7, dtype=np.bool_
        )  # Initialize neighbor_exists_array
        neighbor_grids_offset = np.zeros(
            (7, 3), dtype=delphi_real
        )  # Initialize neighbor_grids_offset

        return (
            index_map,
            neighbor_grids_offset,
            neighbor_exists_array,
        )

    def create_vdw_molecular_surfaces(
        self,
        platform: Platform,
        use_zeta_surface=True,
        solve_pbe=True,
        read_rxn_from_frc=True,
        calc_solvation_energy=True,
        calc_nonlinear_energy=False,
        calc_surface_energy=False,
        calc_surface_charge=False,
        only_molecule=True,
        profile_timings=False,
    ):
        """
        Creates the Van der Waals molecular surface.

        This method orchestrates the sequence of steps required to generate the
        molecular surface, including finding boundary grid points, handling the
        zeta surface (if enabled), and elaborating the boundary grid points.

        Args:
            platform (Platform): The platfrom object with user choices and configurations.
            use_zeta_surface (bool): Whether to use zeta surface calculation. Defaults to True.
            solve_pbe (bool): Flag for solvation energy calculation. Defaults to True.
            read_rxn_from_frc (bool): Flag related to reaction field energy. Defaults to True.
            calc_solvation_energy (bool): Flag for generating log information. Defaults to True.
            calc_nonlinear_energy (bool): Flag for non-linear Poisson-Boltzmann log. Defaults to False.
            calc_surface_energy (bool): Flag related to entropy calculation. Defaults to False.
            calc_surface_charge (bool): Flag related to enthalpy calculation. Defaults to False.
            only_molecule (bool): Flag to consider only the molecule. Defaults to True.
            profile_timings (bool): Flag to enable time-profiling of key steps. Defaults to False.

        Returns:
            int: 0 on success, or an error code (EXIT_NJIT_FLAG) on failure.
        """

        use_cuda = platform.active == "cuda"
        num_threads = platform.names["cpu"]["num_threads"]

        if profile_timings:
            tic_vdw_setup = time.perf_counter()
        # Initialization
        (
            index_map,
            grid_neighbor_coords_offsets,
            neighbor_exists_array,
        ) = self._initialize_arrays(self.num_atoms)

        # Calculate grid properties
        (
            grid_spacing_half,
            mid_grid_point_indices,
            gridbox_center,
            n_grid_points,
            n_grid_points_x_3,
        ) = _calculate_grid_properties(
            self.grid_spacing,
            self.grid_shape,
            self.grid_origin,
            dtype_int=delphi_int,
            dtype_real=delphi_real,
        )
        grid_spacing = self.grid_spacing
        self.num_grid_points = n_grid_points
        self.num_grid_points_x_3 = n_grid_points_x_3

        cycle_flag = False

        # Set constant values
        index_map, neighbor_exists_array = _set_constant_values(
            dtype_int=delphi_int, dtype_bool=delphi_bool
        )

        # Set neighbor offsets
        grid_neighbor_coords_offsets = _setup_grid_neighbor_coords_offsets(
            grid_spacing_half=grid_spacing_half, dtype_real=delphi_real
        )

        # Determine maximum probe radius
        max_probe_radius = max(self.probe_radius, self.probe_radius_second)

        # Determine grid origin, grid_origin already accounts for parent or focusing. NO update is needed.
        grid_origin_current = (
            self.grid_origin if self.is_focusing else self.grid_origin
        )  # - 0.5 * self.grid_spacing

        # Determine exclusion radius
        exclusion_radius = (
            max(self.max_atom_radius, self.salt_radius)
            if self.debye_length != delphi_real(ZERO_MOLAR_DEBYE_LENGHT)
            else self.max_atom_radius
        )

        # Calculate grid boundaries
        (
            min_solute_grid_index,
            max_solute_grid_index,
        ) = _calculate_solute_grid_boundaries(
            max_atom_radius=self.max_atom_radius,
            grid_spacing=self.grid_spacing,
            grid_shape=self.grid_shape,
            grid_origin=self.grid_origin,
            coords_by_axis_min=self.min_coords_by_axis,
            coords_by_axis_max=self.max_coords_by_axis,
            dtype_int=delphi_int,
        )

        # Calculate strides
        (
            x_stride,
            y_stride,
            z_stride,
            x_stride_x_3,
            y_stride_x_3,
            z_stride_x_3,
        ) = _calculate_strides(self.grid_shape, dtype_int=delphi_int)

        # Initialize boundary grid points arrays if not already initialized
        if self.max_boundary_grid_points is None:
            self.max_boundary_grid_points = 1000000
            small_size_threshold = delphi_int(300)

            if (
                self.grid_shape[0] > small_size_threshold
                or self.grid_shape[1] > small_size_threshold
                or self.grid_shape[2] > small_size_threshold
            ):
                self.max_boundary_grid_points = 50000000
        if self.boundary_grid_indices is None:
            self.boundary_grid_indices = np.zeros(
                (self.max_boundary_grid_points, 3), dtype=delphi_int
            )

        vprint(VERBOSE, _VERBOSITY, " Info> Drawing MS from vdW surface")
        iarv = 0
        if profile_timings:
            toc_vdw_setup = time.perf_counter()
            vprint(
                VERBOSE,
                _VERBOSITY,
                f">> Time: vdw to ms setup: {(toc_vdw_setup - tic_vdw_setup):.3f}s",
            )
        # Find initial set of boundary grid points based on dielectric constant map.
        (
            num_boundary_grid_points_found,
            num_external_boundary_points,
            self.solute_bgp_type_1d,
            boundary_grid_indices,
        ) = find_boundary_grid_points(
            epsilon_dimension=self.epsilon_dimension,
            grid_shape=self.grid_shape,
            min_solute_grid_index=min_solute_grid_index,
            max_solute_grid_index=max_solute_grid_index,
            index_discrete_epsilon_map_1d=self.discrete_epsilon_index_map_1d,
            dtype_int=delphi_int,
            use_cuda=use_cuda,
            num_threads=num_threads,
        )
        self.boundary_grid_indices[:num_external_boundary_points, :] = (
            boundary_grid_indices[:num_external_boundary_points, :]
        )

        if profile_timings:
            toc_vdw_init_bgp = time.perf_counter()
            vprint(
                VERBOSE,
                _VERBOSITY,
                f"vdw to ms find initial boundary grid points: {(toc_vdw_init_bgp - toc_vdw_setup):.3f}s",
            )

        # Handle zeta surface calculation if enabled
        (
            self.zeta_surface_grid_coords,
            self.zeta_surface_grid_indices,
            self.num_zeta_surface_point_coords,
            self.num_zeta_surface_point_indices,
            self.zeta_surface_coords_capacity,
            self.zeta_surface_indices_capacity,
        ) = _handle_zeta_surface(
            use_zeta_surface=use_zeta_surface,
            grid_spacing=self.grid_spacing,
            grid_shape=self.grid_shape,
            gridbox_center=gridbox_center,
            mid_grid_indices=mid_grid_point_indices,
            zeta_surface_map_1d=self.zeta_surface_map_1d,
            epsilon_dimension=self.epsilon_dimension,
            index_discrete_epsilon_map_1d=self.discrete_epsilon_index_map_1d,
            dtype_int=delphi_int,
            dtype_real=delphi_real,
        )
        # print(
        #     "vdwms.create_vdw_molecular_surface: ",
        #     "self.use_zeta_surf = ",
        #     use_zeta_surface,
        #     "self.zeta_surf_grid_coords = ",
        #     self.zeta_surface_grid_coords,
        #     "self.zeta_surf_grid_indices = ",
        #     self.zeta_surface_grid_indices,
        #     "self.num_zeta_surface_point_coords = ",
        #     self.num_zeta_surface_point_coords,
        #     "self.num_zeta_surface_point_indices = ",
        #     self.num_zeta_surface_point_indices,
        # )

        self.num_boundary_grid_points = num_boundary_grid_points_found
        self.num_external_boundary_points = num_external_boundary_points

        vprint(
            DEBUG,
            _VERBOSITY,
            " VdMS> Boundary points facing continuum solvent= ",
            self.num_external_boundary_points,
        )
        vprint(
            DEBUG,
            _VERBOSITY,
            " VdMS> Total number of boundary points before elab.= ",
            self.num_boundary_grid_points,
        )

        # Check if the number of boundary grid points exceeds the limit.
        _check_boundary_point_limit(
            self.num_boundary_grid_points, self.max_boundary_grid_points
        )

        # Print the dielectric boundary information if verbosity is high enough.
        _print_dielectric_boundary(self.solute_bgp_type_1d, self.grid_shape)
        if profile_timings:
            toc_vdw_zeta = time.perf_counter()
            vprint(
                VERBOSE,
                _VERBOSITY,
                f"vdw to ms _handle_zeta_surface: {(toc_vdw_zeta - toc_vdw_init_bgp):.3f}s",
            )

        # Handle the case where the probe radius is zero.
        self.boundary_grid_points = _handle_zero_probe_radius(
            max_probe_radius,
            self.num_boundary_grid_points,
            self.boundary_grid_indices,
            dtype_int=delphi_int,
            dtype_real=delphi_real,
        )
        if max_probe_radius > APPROX_ZERO and self.boundary_grid_points.shape[0] == 0:
            # Proceed if probe radius is non-zero
            if profile_timings:
                toc_vdw_zero_probe = time.perf_counter()
                vprint(
                    VERBOSE,
                    _VERBOSITY,
                    f"vdw to ms _handle_zero_probe_radius: {(toc_vdw_zero_probe - toc_vdw_zeta):.3f}s",
                )
            if profile_timings:
                toc_vdw_zero_probe = time.perf_counter()
            # Calculate atom plus probe radii and their squares (for contact detection).
            (
                self.atom_plus_probe_radii_1d,
                self.atom_plus_probe_radii_square_1d,
                self.atom_plus_probe_radii_square_shrunk_1d,
            ) = _calculate_atom_probe_radii(
                self.probe_radius,
                RADII_SQUARED_SHRINK_FACTOR,
                self.num_atoms,
                self.atoms_data,
                dtype_real=delphi_real,
            )

            probe_radius_squared_1 = self.probe_radius * self.probe_radius
            probe_radius_squared_2 = self.probe_radius_second * self.probe_radius_second

            # Calculate root mean square of atomic positions (used in some surface calculations).
            rms = _calculate_rms(
                only_molecule, self.num_atoms, self.atoms_data, dtype_real=delphi_real
            )

            surface_generation_flag = True
            if profile_timings:
                toc_vdw_radii_acc = time.perf_counter()
                vprint(
                    VERBOSE,
                    _VERBOSITY,
                    f"vdw to ms _calculate_atom_probe_radii: {(toc_vdw_radii_acc - toc_vdw_zero_probe):.3f}s",
                )
            # Calculate solvent accessible surface area (SAS) related information.
            (
                self.num_exposed_grids,
                self.exposed_grids_coords,
                self.atom_accessibility,
                self.voxel_atom_count,
                self.voxel_atom_count_cumulative,
                self.voxel_space_origin,
                self.voxel_space_shape,
            ) = sas.solvent_accessible_surface(
                grid_spacing=self.grid_spacing,
                probe_radius=self.probe_radius,
                probe_radius2=self.probe_radius_second,
                max_atom_radius=self.max_atom_radius,
                min_coords_by_axis=self.min_coords_by_axis,
                max_coords_by_axis=self.max_coords_by_axis,
                num_atoms=self.num_atoms,
                num_objects=self.num_objects,
                num_molecules=self.num_molecules,
                atoms_data=self.atoms_data,
                atom_plus_probe_radii_1d=self.atom_plus_probe_radii_1d,
                atom_plus_probe_radii_shrink_1d=self.atom_plus_probe_radii_square_shrunk_1d,
                use_cuda=use_cuda,
                num_threads=num_threads,
                num_vertices=520,
                num_edges=1040,
            )
            if profile_timings:
                toc_vdw_sas = time.perf_counter()
                print(
                    f"vdw to ms solvent_accessible_surface: {(toc_vdw_sas - toc_vdw_radii_acc):.3f}s"
                )

            # Setup a cubic grid for efficient indexing of vertices.
            (
                cube_vertex_lowest_xyz,
                cube_shape,
                cube_side_length,
                self.cube_side_length_inverse,
            ) = _setup_cube_for_indexing(
                grid_scale=self.grid_scale,
                max_probe_radius=max_probe_radius,
                max_atom_radius=self.max_atom_radius,
                min_coords_by_axis=self.min_coords_by_axis,
                max_coords_by_axis=self.max_coords_by_axis,
                dtype_real=delphi_real,
            )
            self.voxel_space_origin = cube_vertex_lowest_xyz
            self.voxel_space_shape = cube_shape

            # Calculate the number of voxels per entity (atom, object, molecule).
            num_voxel_per_entity = _calculate_cube_voxels_per_entity(
                num_objects=self.num_objects,
                num_molecules=self.num_molecules,
                cube_shape=self.voxel_space_shape,
                dtype_int=delphi_int,
            )
            if profile_timings:
                toc_vdw_cube_setup = time.perf_counter()
                vprint(
                    VERBOSE,
                    _VERBOSITY,
                    f"vdw to ms _setup_cube_for_indexing: {(toc_vdw_cube_setup - toc_vdw_sas):.3f}s",
                )

            # Perform the cube calculation to assign atoms to voxels.
            (
                voxel_atom_ids,
                voxel_atom_count,
                voxel_atom_count_cumulative,
            ) = _perform_cube_calculation(
                self.num_atoms,
                self.num_objects,
                self.num_molecules,
                num_voxel_per_entity,
                cube_side_length,
                self.voxel_space_origin,
                self.voxel_space_shape,
                self.atoms_data,
                delphi_int,
                delphi_real,
            )
            self.voxel_atom_ids = voxel_atom_ids
            self.voxel_atom_count = voxel_atom_count
            self.voxel_atom_count_cumulative = voxel_atom_count_cumulative
            if profile_timings:
                toc_vdw_cube_calc = time.perf_counter()
                vprint(
                    VERBOSE,
                    _VERBOSITY,
                    f"vdw to ms _perform_cube_calculation: {(toc_vdw_cube_calc - toc_vdw_cube_setup):.3f}s",
                )
            voxel_space_boundary_extension = self.max_atom_radius + self.probe_radius
            # voxels_per_dim must be small fraction of grids, we use maximum of 50 and 20% of largest grid_shape
            # to balance the voxelation size and very large-voxel to avoid very large neighborhood search.
            max_voxels_per_dimension_value = int(
                max(50, int(max(self.grid_shape) * 0.25))
            )
            # Setup vertex indexing data structures.
            (
                indexing_voxel_side,
                self.indexing_voxel_origin,
                self.indexing_voxel_shape,
            ) = voxelizer.calculate_indexing_voxel_parameters(
                self.grid_spacing,
                self.probe_radius,
                voxel_space_boundary_extension,
                self.min_coords_by_axis,
                self.max_coords_by_axis,
                max_voxels_per_dimension=max_voxels_per_dimension_value,
            )

            self.indexing_voxel_scale = 1.0 / indexing_voxel_side
            self.point_indices_by_voxel = np.zeros(
                self.num_exposed_grids + 1, dtype=delphi_int
            )

            vprint(
                DEBUG,
                _VERBOSITY,
                " VdMS> grid for indexing accessible points =  ",
                indexing_voxel_side,
            )
            if profile_timings:
                toc_vdw_index_setup = time.perf_counter()
                vprint(
                    VERBOSE,
                    _VERBOSITY,
                    f"vdw to ms setup_index_vertices: {(toc_vdw_index_setup - toc_vdw_cube_calc):.3f}s",
                )

            # Perform vertex indexing to find which atoms are accessible to the solvent
            (
                self.voxel_point_start_indices,
                self.voxel_point_end_indices,
                self.point_indices_by_voxel,
            ) = voxelizer.build_point_voxel_index_map(
                self.num_exposed_grids,
                self.indexing_voxel_scale,
                self.indexing_voxel_shape,
                self.indexing_voxel_origin,
                self.exposed_grids_coords,
                self.point_indices_by_voxel,
            )
            if profile_timings:
                toc_vdw_index_calc = time.perf_counter()
                vprint(
                    VERBOSE,
                    _VERBOSITY,
                    f"vdw to ms index_vertices: {(toc_vdw_index_calc - toc_vdw_index_setup):.3f}s",
                )

            from pydelphi.space.core.vdw.process_bgp import process_bgp_orchestrate

            rm_boundary_pt_condition = self.num_molecules > 0
            num_cavity_midpoints = 0  # tracked elsewhere if needed

            # Initial wavefront state
            num_boundary_grid_points = self.num_boundary_grid_points
            num_external_boundary_points = self.num_external_boundary_points

            # Convergence / safety controls
            max_points_added_in_iteration = 100_000
            max_no_divergence_count = 50
            max_total_iterations = None  # or an int if you want a hard cap

            if _VERBOSITY <= DEBUG:
                _dbg_epsmap(
                    "PRE-BGP host",
                    self.discrete_epsilon_index_map_1d,
                    epsdim=self.epsilon_dimension,
                    sample=12,
                )

            backend_kwargs = dict(
                num_threads=num_threads,  # CPU uses; CUDA ignores
                probe_radius_squared_1=probe_radius_squared_1,
                probe_radius_squared_2=probe_radius_squared_2,
                x_stride=x_stride,
                y_stride=y_stride,
                z_stride=z_stride,
                grid_origin_current=grid_origin_current,
                max_boundary_grid_points=self.max_boundary_grid_points,  # REQUIRED for CUDA workspace sizing
                grid_neighbor_coords_offsets=grid_neighbor_coords_offsets,
                grid_spacing=grid_spacing,
                grid_shape=self.grid_shape,
                exposed_grids_coords=self.exposed_grids_coords,
                epsilon_dimension=self.epsilon_dimension,
                index_map=index_map,
                rm_boundary_pt_condition=rm_boundary_pt_condition,
                min_xyz=self.indexing_voxel_origin,
                cube_side_indver_inverse=self.indexing_voxel_scale,
                cube_shape_indver=self.indexing_voxel_shape,
                cube_voxel_start_indices=self.voxel_point_start_indices,
                cube_voxel_end_indices=self.voxel_point_end_indices,
                cube_vertex_lowest_xyz=self.voxel_space_origin,
                cube_side_length_inverse=self.cube_side_length_inverse,
                cube_shape=self.voxel_space_shape,
                voxel_grid_point_indices=self.point_indices_by_voxel,
                voxel_atom_count=self.voxel_atom_count,
                voxel_atom_count_cumulative=self.voxel_atom_count_cumulative,
                voxel_atom_ids=self.voxel_atom_ids,
                atom_surface_flags=self.atom_accessibility,
                num_atoms=self.num_atoms,
                atoms_data=self.atoms_data,
                atom_plus_probe_radii_1d=self.atom_plus_probe_radii_1d,
                atom_plus_probe_radii_shrink_1d=self.atom_plus_probe_radii_square_shrunk_1d,
                dtype_int=delphi_int,
                dtype_real=delphi_real,
                dtype_bool=delphi_bool,
            )
            tic_broc_bgp = time.perf_counter()
            (
                exec_status,
                boundary_grid_indices,
                solute_bgp_type_1d,
                index_discrete_epsilon_map_1d,
                num_discovered_bndy_grid_points,
                num_external_boundary_points,
                num_iterations,
            ) = process_bgp_orchestrate(
                use_cuda=False,  # just flip this
                num_threads=num_threads,
                profile_timings=profile_timings,
                max_points_added_in_iteration=max_points_added_in_iteration,
                max_no_divergence_count=max_no_divergence_count,
                max_total_iterations=max_total_iterations,
                num_boundary_grid_points=num_boundary_grid_points,
                num_external_boundary_points=num_external_boundary_points,
                boundary_grid_indices=self.boundary_grid_indices,
                solute_bgp_type_1d=self.solute_bgp_type_1d,
                index_discrete_epsilon_map_1d=self.discrete_epsilon_index_map_1d,
                backend_kwargs=backend_kwargs,
                exit_njit_flag=EXIT_NJIT_FLAG,
            )
            toc_broc_bgp = time.perf_counter()
            vprint(
                DEBUG,
                _VERBOSITY,
                f"    Total time for process_bgp_orchestrate: {toc_broc_bgp - tic_broc_bgp:0.6f}",
            )
            # Re-bind updated state explicitly
            self.boundary_grid_indices = boundary_grid_indices
            self.solute_bgp_type_1d = solute_bgp_type_1d
            self.discrete_epsilon_index_map_1d = index_discrete_epsilon_map_1d

            self.num_boundary_grid_points = num_discovered_bndy_grid_points
            self.num_external_boundary_points = num_external_boundary_points
            boundary_point_end_index = num_discovered_bndy_grid_points

            if boundary_point_end_index > self.max_boundary_grid_points:
                vprint(
                    ERROR,
                    _VERBOSITY,
                    " Error> ibnd upper bound ",
                    boundary_point_end_index,
                    " exceeds ibmx",
                )
                exit(0)

            vprint(
                DEBUG,
                _VERBOSITY,
                " VdMS> Number of cavity mid-points inaccessible to solvent = ",
                num_cavity_midpoints,
            )
            if profile_timings:
                tic_vdw_elab_bgp = time.perf_counter()

            if _VERBOSITY <= TRACE:
                np.savetxt(
                    f"boundary_grid_indices_nt{num_threads}_before_srfelab.txt",
                    self.boundary_grid_indices[:boundary_point_end_index],
                    fmt="%d",
                )
                np.savetxt(
                    f"solute_bgp_type_1d{num_threads}_before_srfelab.txt",
                    self.solute_bgp_type_1d,
                    fmt="%d",
                )
                np.savetxt(
                    f"index_discrete_epsilon_map_1d{num_threads}_before_srfelab.txt",
                    self.discrete_epsilon_index_map_1d,
                    fmt="%.2f",
                )

            bnd_gp_index_shape = self.boundary_grid_indices.shape
            new_bnd_gp_index_shape = (bnd_gp_index_shape[0], bnd_gp_index_shape[1] + 1)
            vprint(VERBOSE, _VERBOSITY, "bnd_gp_index_shape:", bnd_gp_index_shape)
            boundary_grid_flagged_indices = np.zeros(
                new_bnd_gp_index_shape, dtype=self.boundary_grid_indices.dtype
            )
            boundary_grid_flagged_indices[:boundary_point_end_index, :3] = (
                self.boundary_grid_indices[:boundary_point_end_index, 0:3]
            )
            for i in range(boundary_point_end_index):
                ix, iy, iz = boundary_grid_flagged_indices[i][:3]
                ind1d = ix * x_stride + iy * y_stride + iz
                boundary_grid_flagged_indices[i, 3] = solute_bgp_type_1d[ind1d]

            # Elaborate the boundary grid points to finalize the surface.
            (
                self.num_boundary_grid_points,
                self.boundary_grid_points,
            ) = helpers.surface_elaborate_boundary_gridpoints(
                num_boundary_grid_indices=boundary_point_end_index,
                epsilon_dimension=self.epsilon_dimension,
                max_boundary_grid_points=self.max_boundary_grid_points,
                grid_shape=self.grid_shape,
                boundary_grid_points=self.boundary_grid_points,
                boundary_grid_indices=self.boundary_grid_indices,
                boundary_grid_flagged_indices=boundary_grid_flagged_indices,
                # solute_bgp_type_1d=self.solute_bgp_type_1d,
                index_discrete_epsilon_map_1d=self.discrete_epsilon_index_map_1d,
                index_map=index_map,
            )

            if _VERBOSITY <= TRACE:
                np.savetxt(
                    f"boundary_grid_indices_nt{num_threads}_after_srfelab.txt",
                    self.boundary_grid_indices[:boundary_point_end_index],
                    fmt="%d",
                )
                np.savetxt(
                    f"solute_bgp_type_1d{num_threads}_after_srfelab.txt",
                    self.solute_bgp_type_1d,
                    fmt="%d",
                )
                np.savetxt(
                    f"index_discrete_epsilon_map_1d{num_threads}_after_srfelab.txt",
                    self.discrete_epsilon_index_map_1d,
                    fmt="%.2f",
                )
            if profile_timings:
                toc_vdw_elab_bgp = time.perf_counter()
                vprint(
                    VERBOSE,
                    _VERBOSITY,
                    f"vdw to ms surface_elaborate_boundary_gridpoints: {(toc_vdw_elab_bgp - tic_vdw_elab_bgp):.3f}s",
                )
            if self.num_boundary_grid_points == EXIT_NJIT_FLAG:
                exit(0)

        # Scale boundary grid point positions relative to accessible data.
        if solve_pbe and (
            read_rxn_from_frc
            or calc_solvation_energy
            or calc_nonlinear_energy
            or calc_surface_energy
            or calc_surface_charge
        ):
            vprint(DEBUG, _VERBOSITY, " VdMS> Scaling boundary grid points ...")

            self.surface_charge_positions = np.zeros(
                (self.num_boundary_grid_points, 3), dtype=delphi_real
            )

            for boundary_point_index in range(self.num_boundary_grid_points):
                self.surface_charge_positions[boundary_point_index] = (
                    self.boundary_grid_points[boundary_point_index]
                ).astype(delphi_real)

            self.scaled_surface_normal_vectors = np.zeros(
                (self.num_boundary_grid_points, 3), dtype=delphi_real
            )
            self.atom_surface_index = np.zeros(
                self.num_boundary_grid_points, dtype=delphi_int
            )
            self.atom_index_array = np.zeros(
                self.num_boundary_grid_points, dtype=delphi_int
            )
            if _VERBOSITY <= DEBUG:
                write_nparray_to_npy(
                    f"before_sacale_scspos_{num_threads}.npy",
                    self.surface_charge_positions,
                    decimals=4,
                )

            if profile_timings:
                tic_vdw_scale_bgp = time.perf_counter()

            # Call the function to scale the Van der Waals surface boundary points.
            return_status, _ = sclbp.scale_vdw_surface_boundary_points(
                use_cuda=use_cuda,
                num_threads=num_threads,
                num_atoms=self.num_atoms,
                num_molecules=self.num_molecules,
                num_objects=self.num_objects,
                max_atom_radius=delphi_real(self.max_atom_radius),
                probe_radius=delphi_real(self.probe_radius),
                probe_radius_2=delphi_real(self.probe_radius_second),
                is_focusing_run=self.is_focusing,
                grid_spacing=delphi_real(self.grid_spacing),
                grid_origin=self.grid_origin.astype(delphi_real),
                grid_dimensions=self.grid_shape.astype(delphi_int),
                atom_data=self.atoms_data,
                min_coords_by_axis=self.min_coords_by_axis,
                max_coords_by_axis=self.max_coords_by_axis,
                num_external_boundary_points=self.num_external_boundary_points,
                num_exposed_grid_points=self.num_exposed_grids,
                num_boundary_points=self.num_boundary_grid_points,
                surface_charge_positions=self.surface_charge_positions,
                discrete_epsilon_index_map_1d=self.discrete_epsilon_index_map_1d,
                scaled_surface_normal_vectors=self.scaled_surface_normal_vectors,
                exposed_grid_point_coords=self.exposed_grids_coords,
                atom_accessibility=self.atom_accessibility,
                atom_surface_index=self.atom_surface_index,
                atom_index_for_boundary=self.atom_index_array,
                atom_plus_probe_radii=self.atom_plus_probe_radii_1d,
                atom_plus_probe_radii_squared=self.atom_plus_probe_radii_square_1d,
                atom_plus_probe_radii_squared_shrunk=self.atom_plus_probe_radii_square_shrunk_1d,
                system_min_coords=self.indexing_voxel_origin,
                cube_side_indver_inverse=self.indexing_voxel_scale,
                cube_shape_indver=self.indexing_voxel_shape,
                cube_voxel_atom_index_start=self.voxel_point_start_indices,
                cube_voxel_atom_index_end=self.voxel_point_end_indices,
                cube_voxel_atom_index_cumulative=self.point_indices_by_voxel,
            )
            vprint(
                VERBOSE,
                _VERBOSITY,
                "self.num_external_boundary_points=",
                num_external_boundary_points,
            )
            if profile_timings:
                toc_vdw_scale_bgp = time.perf_counter()
                vprint(
                    VERBOSE,
                    _VERBOSITY,
                    f"vdw to ms scale_vdw_surface_boundary_points: {(toc_vdw_scale_bgp - tic_vdw_scale_bgp):.3f}s",
                )
            if return_status == EXIT_NJIT_FLAG:
                return return_status

        # Print surface charge positions if verbosity is high enough
        if _VERBOSITY <= DEBUG:
            write_nparray_to_npy(
                f"scspos_{num_threads}.npy", self.surface_charge_positions, decimals=4
            )

        vprint(DEBUG, _VERBOSITY, " VdMS> MS creation done")
        return 0
