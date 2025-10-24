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


"""
This module defines the `RuntimeContext` class, a central repository for managing
all parameters, data, and state variables required during a pyDelphi calculation.

The `RuntimeContext` class consolidates information ranging from global physical constants,
dielectric settings, and grid dimensions to atomic properties, charge distributions,
and various intermediate and final potential/energy maps. It is designed to
provide a coherent and accessible structure for all components of the pyDelphi
pipeline, ensuring data consistency and facilitating the flow of information
between different calculation stages (e.g., parsing, space setup, solving PBE,
and energy calculation).

The module also includes Numba-jitted helper functions (`_njit_prepare_focusing_loop`,
`_njit_atoms_summary_loop`) to accelerate critical numerical operations on atom data.
"""

import math
import numpy as np
from numba import njit

from pydelphi.foundation.enums import (
    Precision,
    SoluteExtremaRule,
    GridboxSize,
    GridBoxType,
)
from pydelphi.energy.energy_models import EnergyResults
from pydelphi.constants import ConstPhysical as Constants
from pydelphi.constants import ConstDelPhiFloats as ConstDelPhi

from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
    vprint,
)

from pydelphi.constants import (
    ATOMFIELD_X,
    ATOMFIELD_Y,
    ATOMFIELD_Z,
    ATOMFIELD_CRD_END,
    ATOMFIELD_RADIUS,
    ATOMFIELD_CHARGE,
    LEN_ATOMFIELDS,
)

from pydelphi.config.logging_config import (
    WARNING,
    DEBUG,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

if PRECISION.value == Precision.SINGLE.value:
    from pydelphi.utils.prec.single import (
        or_gt_vector,
        or_lt_vector,
    )

elif PRECISION.value == Precision.DOUBLE.value:
    from pydelphi.utils.prec.double import (
        or_gt_vector,
        or_lt_vector,
    )

# Lets check the currently loaded precision
vprint(DEBUG, _VERBOSITY, "context>> precision:", PRECISION)


@njit(nogil=True, boundscheck=False, cache=True)
def _njit_prepare_focusing_loop(
    atoms_data_parentrun,
    edge_high,
    edge_low,
    LEN_ATOMFIELDS_val,
    dtype_real,
):
    """Numba-jitted loop to select atoms within the focusing box."""
    n_atoms_focus = 0
    atoms_data_temp = np.zeros(
        (atoms_data_parentrun.shape[0], LEN_ATOMFIELDS_val + 1),
        dtype=dtype_real,
    )  # Maximum possible size

    for this_index in range(atoms_data_parentrun.shape[0]):
        this_atom = atoms_data_parentrun[this_index]
        xyz_temp = this_atom[
            ATOMFIELD_X:ATOMFIELD_CRD_END
        ]  # get_atom_coords(this_atom)
        # Check if atom coordinate is within the focusing box (not outside lower or higher edge)
        if not (or_gt_vector(xyz_temp, edge_high) or or_lt_vector(xyz_temp, edge_low)):
            atoms_data_temp[n_atoms_focus][:LEN_ATOMFIELDS_val] = this_atom[
                :
            ]  # Copy atom data
            atoms_data_temp[n_atoms_focus][LEN_ATOMFIELDS_val] = dtype_real(
                this_index
            )  # Store original atom index
            n_atoms_focus += 1

    atoms_data_focus = np.zeros(
        (n_atoms_focus, LEN_ATOMFIELDS_val + 1), dtype=dtype_real
    )
    for a_index in range(n_atoms_focus):
        atoms_data_focus[a_index][:] = atoms_data_temp[a_index][:]

    return n_atoms_focus, atoms_data_focus


@njit(nogil=True, boundscheck=False, cache=True)
def _njit_atoms_summary_loop(
    atoms_data,
    max_atom_radius_val,
    extremas_rule_value,
    acenter,
    enforce_acenter,
    dtype_real,
):
    """Numba-jitted loop for atom summarization calculations."""
    total_charge = 0.0
    positive_charge = 0.0
    negative_charge = 0.0
    n_positive_charge = 0
    n_negative_charge = 0

    centroid_positive_charge = np.zeros(3, dtype=dtype_real)
    centroid_negative_charge = np.zeros(3, dtype=dtype_real)

    coords_by_axis_min = np.zeros(3, dtype=dtype_real)
    coords_by_axis_max = np.zeros(3, dtype=dtype_real)
    boundary_min = np.zeros(3, dtype=dtype_real)
    boundary_max = np.zeros(3, dtype=dtype_real)
    coords_sum = np.zeros(3, dtype=dtype_real)

    for ia in range(atoms_data.shape[0]):
        atom = atoms_data[ia]
        atom_charge = atom[ATOMFIELD_CHARGE]
        atom_x = atom[ATOMFIELD_X]
        atom_y = atom[ATOMFIELD_Y]
        atom_z = atom[ATOMFIELD_Z]
        atom_radius = atom[ATOMFIELD_RADIUS]

        # Charge Summary Calculations
        absolute_charge = abs(atom_charge)
        if atom_charge > 0.0:
            if n_positive_charge == 0:
                centroid_positive_charge[0] = atom_x * absolute_charge
                centroid_positive_charge[1] = atom_y * absolute_charge
                centroid_positive_charge[2] = atom_z * absolute_charge
            else:
                centroid_positive_charge[0] += atom_x * absolute_charge
                centroid_positive_charge[1] += atom_y * absolute_charge
                centroid_positive_charge[2] += atom_z * absolute_charge
            n_positive_charge += 1
            positive_charge += atom_charge
        elif atom_charge < 0:
            if n_negative_charge == 0:
                centroid_negative_charge[0] = atom_x * absolute_charge
                centroid_negative_charge[1] = atom_y * absolute_charge
                centroid_negative_charge[2] = atom_z * absolute_charge
            else:
                centroid_negative_charge[0] += atom_x * absolute_charge
                centroid_negative_charge[1] += atom_y * absolute_charge
                centroid_negative_charge[2] += atom_z * absolute_charge
            n_negative_charge += 1
            negative_charge += atom_charge

        # Coordinate Summary Calculations
        atom_x_minus_radius = atom_x - atom_radius
        atom_y_minus_radius = atom_y - atom_radius
        atom_z_minus_radius = atom_z - atom_radius
        atom_x_plus_radius = atom_x + atom_radius
        atom_y_plus_radius = atom_y + atom_radius
        atom_z_plus_radius = atom_z + atom_radius
        if ia == 0:
            coords_by_axis_min[0] = atom_x
            coords_by_axis_min[1] = atom_y
            coords_by_axis_min[2] = atom_z
            coords_by_axis_max[0] = atom_x
            coords_by_axis_max[1] = atom_y
            coords_by_axis_max[2] = atom_z
            boundary_min[0] = atom_x_minus_radius
            boundary_min[1] = atom_y_minus_radius
            boundary_min[2] = atom_z_minus_radius
            boundary_max[0] = atom_x_plus_radius
            boundary_max[1] = atom_y_plus_radius
            boundary_max[2] = atom_z_plus_radius
        else:
            # Update minimum coordinates along each axis
            if atom_x < coords_by_axis_min[0]:
                coords_by_axis_min[0] = atom_x
            if atom_y < coords_by_axis_min[1]:
                coords_by_axis_min[1] = atom_y
            if atom_z < coords_by_axis_min[2]:
                coords_by_axis_min[2] = atom_z

            # Update min boundary coordinates along each axis
            if atom_x_minus_radius < boundary_min[0]:
                boundary_min[0] = atom_x_minus_radius
            if atom_y_minus_radius < boundary_min[1]:
                boundary_min[1] = atom_y_minus_radius
            if atom_z_minus_radius < boundary_min[2]:
                boundary_min[2] = atom_z_minus_radius

            # Update maximum coordinates along each axis
            if atom_x > coords_by_axis_max[0]:
                coords_by_axis_max[0] = atom_x
            if atom_y > coords_by_axis_max[1]:
                coords_by_axis_max[1] = atom_y
            if atom_z > coords_by_axis_max[2]:
                coords_by_axis_max[2] = atom_z

            # Update max boundary coordinates along each axis
            if atom_x_plus_radius > boundary_max[0]:
                boundary_max[0] = atom_x_plus_radius
            if atom_y_plus_radius > boundary_max[1]:
                boundary_max[1] = atom_y_plus_radius
            if atom_z_plus_radius > boundary_max[2]:
                boundary_max[2] = atom_z_plus_radius

        # Accumulate coordinates sum for geometric center calculation
        coords_sum[0] += atom_x
        coords_sum[1] += atom_y
        coords_sum[2] += atom_z

    total_charge = positive_charge + negative_charge

    return (
        total_charge,
        positive_charge,
        negative_charge,
        n_positive_charge,
        n_negative_charge,
        centroid_positive_charge,
        centroid_negative_charge,
        coords_by_axis_min,
        coords_by_axis_max,
        boundary_min,
        boundary_max,
        coords_sum,
    )


class RuntimeContext:
    """
    A container class to hold and manage runtime-context/state for pyDelphi calculations.

    This class centralizes and organizes all necessary data for
    Poisson-Boltzmann electrostatics calculations using the Delphi method.
    It stores parameters, atomic information, grid settings, and calculation
    results at various stages.

    Attributes:
        epkt (delphi_real): Conversion factor to kT/e, based on temperature.
        external_dielectric_scaled (delphi_real): Scaled external dielectric constant.
        gap_dielectric_scaled (delphi_real): Scaled gap dielectric constant.
        internal_dielectric_scaled (delphi_real): Scaled internal dielectric constant.
        debye_length (delphi_real): Debye length of the ionic solution.
        debye_factor (delphi_real): Debye factor constant.
        _gridbox_size (GridboxSize): Dimensions of the grid box (nx, ny, nz).
        scale (delphi_real): Scaling factor for grid spacing.
        perfil (delphi_real): Percentage fill of the solute in the grid box.
        gridbox_margin (delphi_real): Margin around the solute in the grid box.
        grid_spacing (delphi_real): Grid spacing in Angstroms or (1 / scale).
        grid_shape (np.ndarray[delphi_int]): Shape of the grid (nx, ny, nz) as integer array.
        grid_origin (np.ndarray[delphi_real]): Origin of the grid in Angstroms (x, y, z).
        grid_center (np.ndarray[delphi_real]): Center of the grid in Angstroms (x, y, z).

        gridbox_size_parentrun (GridboxSize): Gridbox size from parent run (for focusing).
        scale_parentrun (delphi_real): Scale from parent run (for focusing).
        perfil_parentrun (delphi_real): Perfil from parent run (for focusing).
        gridbox_margin_parentrun (delphi_real): Gridbox margin from parent run (for focusing).
        grid_spacing_parentrun (delphi_real): Grid spacing from parent run (for focusing).
        grid_shape_parentrun (np.ndarray[delphi_int]): Grid shape from parent run (for focusing).
        grid_origin_parentrun (np.ndarray[delphi_real]): Grid origin from parent run (for focusing).
        atoms_data_parentrun (np.ndarray): Atom data from parent run (for focusing).
        n_atoms_parentrun (delphi_int): Number of atoms from parent run (for focusing).
        epsdim_parentrun (delphi_int): Epsilon dimension from parent run (for focusing).
        focus_start (np.ndarray[delphi_int]): Focus start indices.

        atoms_data (np.ndarray): Array holding atom properties (coordinates, charge, radius, etc.).
        atoms_keys (dict[str, int]): Dictionary mapping atom keys to their indices in `atoms_data`.
        atoms_index_to_keys (dict[int, str]): Dictionary mapping atom indices to their keys.
        objects_data (Any): Data for objects (if any), type depends on the application.
        objects_keys (dict): Keys for objects data.
        max_atom_radius (delphi_real): Maximum atomic radius among all atoms.
        num_atoms (delphi_int): Number of atoms in the system.
        num_objects (int): Number of objects in the system.
        num_molecules (int): Number of molecules (currently always 0, might be used in future).
        extremas_rule (SoluteExtremaRule): Rule to define solute extrema.
        coords_by_axis_min (np.ndarray[delphi_real]): Minimum coordinates of atoms along each axis (x, y, z).
        coords_by_axis_max (np.ndarray[delphi_real]): Maximum coordinates of atoms along each axis (x, y, z).
        boundary_min (np.ndarray[delphi_real]): Minimum boundary coordinates (x, y, z), considering max atom radius.
        boundary_max (np.ndarray[delphi_real]): Maximum boundary coordinates (x, y, z), considering max atom radius.
        solute_range (np.ndarray[delphi_real]): Range of solute dimensions (x, y, z).
        solute_range_max (delphi_real): Maximum solute range among x, y, z.
        total_charge (delphi_real): Total charge of all atoms.
        positive_charge (delphi_real): Total positive charge of all atoms.
        negative_charge (delphi_real): Total negative charge of all atoms.
        num_positive_charge (delphi_int): Number of positive charges.
        num_negative_charge (delphi_int): Number of negative charges.
        charged_gridpoints_1d (np.ndarray): 1D array of charged grid point indices (if applicable).
        num_charged_gridpoints (int): Number of charged grid points.
        centroid_positive_charge (np.ndarray[delphi_real]): Centroid of positive charges (x, y, z).
        centroid_negative_charge (np.ndarray[delphi_real]): Centroid of negative charges (x, y, z).
        media_epsilon (np.ndarray[delphi_real]): Array of [external_dielectric_scaled * epkt, internal_dielectric_scaled * epkt].

        gauss_density_map_1d (np.ndarray): 1D array of Gaussian density map values.
        gauss_density_map_midpoints_1d (np.ndarray): 1D array of Gaussian density map midpoints.
        epsilon_map_1d (np.ndarray): 1D array of dielectric map values.
        epsilon_map_midpoints_water_1d (np.ndarray): 1D array of dielectric map midpoints (water region).
        epsilon_map_midpoints_vacuum_1d (np.ndarray): 1D array of dielectric map midpoints (vacuum region).
        bool_debmap_1d (np.ndarray): 1D array of boolean Debye map values.
        zeta_surf_grid_coords (np.ndarray): Coordinates of zeta surface grid points.
        zeta_surf_grid_indices (np.ndarray): Indices of zeta surface grid points.

        acenter (np.ndarray[np.float64]): Center of the atom system (for focusing runs).

        surface_map_1d (np.ndarray): 1D surface map (for RPBE).
        surface_map_midpoints_1d (np.ndarray): 1D surface map midpoints (for RPBE).
        coulomb_map_1d (np.ndarray): 1D Coulomb potential map (for RPBE).
        grad_coulomb_map_1d (np.ndarray): 1D gradient of Coulomb potential map (for RPBE).
        grad_surface_map_1d (np.ndarray): 1D gradient of surface map (for RPBE).
        grad_epsgauss_map_water_1d (np.ndarray): 1D gradient of Gaussian epsilon map (water, for RPBE).
        grad_epsgauss_map_vacuum_1d (np.ndarray): 1D gradient of Gaussian epsilon map (vacuum, for RPBE).
        grad_epsilon_map_water_1d (np.ndarray): 1D gradient of epsilon map (water, for RPBE).
        grad_epsilon_map_vacuum_1d (np.ndarray): 1D gradient of epsilon map (vacuum, for RPBE).
        grad_eps_dot_gad_coul_water_1d (np.ndarray): 1D gradient of epsilon dot gradient of Coulomb (water, for RPBE).
        grad_eps_dot_gad_coul_vacuum_1d (np.ndarray): 1D gradient of epsilon dot gradient of Coulomb (vacuum, for RPBE).

        surf_heavyside_map_1d (np.ndarray): 1D Heaviside surface map (for RPBE with GCS).
        solute_inside_map_1d (np.ndarray): 1D solute inside map (for RPBE with GCS).
        solute_outside_map_1d (np.ndarray): 1D solute outside map (for RPBE with GCS).

        dielectric_boundary_map_1d (np.ndarray): 1D dielectric boundary map (for PBE with TWODIELECTRIC and VDW).
        dielectric_boundary_grids (np.ndarray): Grid coordinates of dielectric boundary (for PBE with TWODIELECTRIC and VDW).
        induced_surf_charge_positions (np.ndarray): Positions of induced surface charges (for PBE with TWODIELECTRIC and VDW).

        ion_exclusion_map_1d (np.ndarray): 1D ion exclusion map (for PBE with GAUSSIAN dielectric and GAUSSIANCUTOFF surface).

        phimap_in_vacuum (np.ndarray): Potential map in vacuum.
        phimap_in_water (np.ndarray): Potential map in water/solution.

        energies (dict[str, delphi_real]): Dictionary to store calculated energies.
    """

    def __init__(
        self,
        temperature,
        exdi,
        gapdi,
        indi,
        precision: Precision,
        dtype_int,
        dtype_real,
    ) -> None:
        """
        Initializes the context with basic parameters.

        Args:
            temperature (delphi_real): Temperature of the system in Kelvin.
            exdi (delphi_real): External dielectric constant.
            gapdi (delphi_real): Gap dielectric constant.
            indi (delphi_real): Internal dielectric constant.
        """

        self.epsilon_dimension = 1
        self.PRECISION = precision
        self.delphi_real = dtype_real
        self.delphi_int = dtype_int
        self.temperature = self.delphi_real(temperature)
        self.epkt = (
            self.delphi_real(Constants.EPK.value) / self.temperature
        )  # conversion factor to kT/e
        self.external_dielectric_scaled = exdi / self.epkt
        self.gap_dielectric_scaled = gapdi / self.epkt
        self.internal_dielectric_scaled = indi / self.epkt

        # Salt dependent parameters
        self.ion_conc = None  # Concentration of Ions
        self.debye_length = None
        self.debye_factor = None
        self.num_taylor_coeff = 5
        self.taylor_coefficients = None

        # Grid settings - will be set by setup_grid method.
        self._gridbox_size = None  # This is supposted to be a typed DelphiGridboxSize for intermediate internal use only.
        self.scale = None
        self.perfil = None
        self.gridbox_margin = None
        self.grid_spacing = None
        self.grid_shape = np.zeros(3, dtype=self.delphi_int)
        self.grid_origin = None
        self.grid_center = None
        self.geometric_center = None
        self.centroid = None

        # Parent run grid settings for focusing runs
        self.scale_parentrun = None
        self.perfil_parentrun = None
        self.gridbox_margin_parentrun = None
        self.grid_spacing_parentrun = None
        self.grid_shape_parentrun = None
        self.grid_origin_parentrun = None
        self.atoms_data_parentrun = None
        self.num_atoms_parentrun = None
        self.epsdim_parentrun = None
        self.grid_center_parentrun = None
        self.phimap_parentrun = None
        self.phimap_comment_parentrun = None
        self.phimap_endianness_parentrun = None
        self.phimap_marker_parentrun = None
        self.focus_start = None

        # Atom and object related data - will be set or updated by self.atoms_summary(...)
        self.atoms_data = None
        self.atoms_keys = {}
        self.atoms_index_to_keys = {}
        self.objects_data = None

        self.max_atom_radius = None
        self.num_atoms = 0
        self.objects_keys = {}
        self.num_objects = 1  # Default value, might be updated later
        self.num_molecules = 0
        self.extremas_rule = None
        self.coords_by_axis_min = None
        self.coords_by_axis_max = None
        self.boundary_min = None
        self.boundary_max = None
        self.solute_range = None
        self.solute_range_max = None
        self.total_charge = 0.0
        self.positive_charge = 0.0
        self.negative_charge = 0.0
        self.num_positive_charge = 0
        self.num_negative_charge = 0
        self.charged_gridpoints_1d = None
        self.num_charged_gridpoints = 0
        self.centroid_positive_charge = None
        self.centroid_negative_charge = None
        self.media_epsilon = None

        # Space module related data - will be generated by Space module
        self.gauss_density_map_1d = None
        self.gauss_density_map_midpoints_1d = None
        self.epsilon_map_1d = None
        self.epsilon_map_midpoints_water_1d = None
        self.epsilon_map_midpoints_vacuum_1d = None
        self.bool_debmap_1d = None
        self.num_zeta_surf_grid_coords = 0
        self.zeta_surf_grid_coords = None
        self.zeta_surf_grid_indices = None

        # Space module data for focusing runs
        self.acenter = np.zeros(3, dtype=np.float64)
        self.enforce_acenter = False

        # RPBE related data - generated only for RPBE calculations
        self.surface_map_1d = None
        self.surface_map_midpoints_1d = None
        self.coulomb_map_1d = None
        self.grad_coulomb_map_1d = None
        self.grad_surface_map_1d = None
        self.grad_epsgauss_map_water_1d = None
        self.grad_epsgauss_map_vacuum_1d = None
        self.grad_epsilon_map_water_1d = None
        self.grad_epsilon_map_vacuum_1d = None
        self.grad_eps_dot_gad_coul_water_1d = None
        self.grad_eps_dot_gad_coul_vacuum_1d = None

        # RPBE with GCS surface related data
        self.surf_heavyside_map_1d = None
        self.solute_inside_map_1d = None
        self.solute_outside_map_1d = None

        # PBE with TWODIELECTRIC and VDW surface related data
        self.dielectric_boundary_map_1d = None
        self.dielectric_boundary_grids = None
        self.induced_surf_charge_positions = None

        # PBE with GAUSSIAN dielectric and GAUSSIANCUTOFF surface related data
        self.ion_exclusion_map_1d = None

        # Solution data - generated after solving PBE/RPBE
        self.phimap_in_vacuum = None
        self.phimap_in_water = None

        # Calculation results - will be stored upon completion
        self.energies = {}
        self.energy_results = EnergyResults()

    def _reset_maps(self) -> None:
        """Resets data maps related to space and potential calculations.

        This method clears out the grids and maps that are generated by the
        `Space` module and during the PBE/RPBE solving process. These maps
        include density maps, dielectric maps, potential maps, surface maps,
        and related gradient maps.

        This method is typically called when:
            - Starting a new Delphi calculation for a different frame in a trajectory.
            - Performing parameter sweeps where only the calculation-dependent maps need refreshing.
            - Setting up a focusing (child) run after a parent run.
        """
        # Reset the fields which are used by Space module
        self.gauss_density_map_1d = None
        self.gauss_density_map_midpoints_1d = None
        self.epsilon_map_1d = None
        self.epsilon_map_midpoints_water_1d = None
        self.epsilon_map_midpoints_vacuum_1d = None
        self.bool_debmap_1d = None
        # Reset the fields which are used only for RPBE
        self.surface_map_1d = None
        self.surface_map_midpoints_1d = None
        self.coulomb_map_1d = None
        self.grad_coulomb_map_1d = None
        self.grad_surface_map_1d = None
        self.grad_epsgauss_map_water_1d = None
        self.grad_epsgauss_map_vacuum_1d = None
        self.grad_epsilon_map_water_1d = None
        self.grad_epsilon_map_vacuum_1d = None
        self.grad_eps_dot_gad_coul_water_1d = None
        self.grad_eps_dot_gad_coul_vacuum_1d = None
        # Following information are generated only for RPBE with GCS surface
        self.surf_heavyside_map_1d = None
        self.solute_inside_map_1d = None
        self.solute_outside_map_1d = None
        # Information generated by solving the PBE/RPBE
        self.phimap_in_vacuum = None
        self.phimap_in_water = None
        # Final information as result of PBE/RPBE solution
        self.energies = {}
        self.energy_results = EnergyResults()

    def _reset_atoms_info(self) -> None:
        """Resets atom-related summary information.

        This method clears attributes derived from the atom data, such as:
            - Atom keys and indices mappings (`atoms_keys`, `atoms_index_to_keys`)
            - Maximum atom radius (`max_atom_radius`)
            - Coordinate ranges and boundaries (`coords_by_axis_min`, `boundary_min`, etc.)
            - Charge summaries (`total_charge`, `positive_charge`, `centroid_positive_charge`, etc.)

        This method is typically called when:
            - Loading a completely new molecular system with different atoms.
            - Re-summarizing atom data after modifications to atom properties (e.g., charges).
        """
        # Reset fields set by method: self.atoms_summary(...)
        self.atoms_data = None
        self.atoms_keys = {}
        self.atoms_index_to_keys = {}
        self.max_atom_radius = None
        self.num_atoms = None
        self.extremas_rule = None
        self.coords_by_axis_min = None
        self.coords_by_axis_max = None
        self.boundary_min = None
        self.boundary_max = None
        self.solute_range = None
        self.solute_range_max = None
        self.total_charge = 0.0
        self.positive_charge = 0.0
        self.negative_charge = 0.0
        self.num_positive_charge = 0
        self.num_negative_charge = 0
        self.charged_gridpoints_1d = None
        self.num_charged_gridpoints = 0
        self.centroid_positive_charge = None
        self.centroid_negative_charge = None

    def _reset_grids(self) -> None:
        """Resets grid-related parameters and settings.

        This method clears attributes defining the computational grid, including:
            - Grid shape and size (`grid_shape`, `gridbox_size`)
            - Grid origin and spacing (`grid_origin`, `grid_spacing`)
            - Scale and perfil (`scale`, `perfil`, `gridbox_margin`)

        This method is typically called when:
            - Changing the grid resolution (scale, gridbox size).
            - Switching between cubic and cuboidal grid types.
            - Setting up a new grid configuration for adaptive refinement (if implemented).
        """
        # Reset fields set by setup_grid method.
        self.grid_shape = None
        self.grid_origin = None
        self.grid_spacing = None
        self.grid_shape = np.zeros(3, dtype=self.delphi_int)
        self._gridbox_size = None
        self.scale = None
        self.perfil = None

    def reset(self) -> None:
        """Resets all relevant fields in the context to initial states.

        This method performs a comprehensive reset by calling `_reset_grids()`,
        `_reset_atoms_info()`, and `_reset_maps()` sequentially. It is intended
        to be used between independent Delphi calculations or at the start of a
        new simulation to ensure a clean context state.

        This method is suitable for:
            - Starting a completely new, independent Delphi calculation.
            - After processing a series of trajectory frames or a batch of calculations.
            - Situations where you want to guarantee no data carry-over from previous runs.
        """
        # Reset all the fields. This is supposed to be used after every frame
        # while running a trajectory
        self._reset_grids()
        self._reset_atoms_info()
        self._reset_maps()

    def set_debyelength(
        self,
        salt_conc_molar: delphi_real,
        temperature: delphi_real,
        exdi: delphi_real,
    ) -> delphi_real:
        """
        Calculates and sets the Debye length based on salt concentration,
        temperature, and external dielectric constant.

        Args:
            salt_conc_molar (delphi_real): Salt concentration in molar units.
            temperature (delphi_real): Temperature in Kelvin.
            exdi (delphi_real): External dielectric constant.

        Returns:
            delphi_real: The calculated Debye length.
                                    Returns a predefined large value if salt concentration is very low or zero.
        """
        if salt_conc_molar > self.delphi_real(ConstDelPhi.ApproxZero.value):
            # Calculate Debye factor and Debye length if salt concentration is non-zero
            ion_concentration = self.delphi_real(salt_conc_molar)
            taylor_coeffs = np.zeros(self.num_taylor_coeff, dtype=self.delphi_real)
            z_plus, z_minus = 1.0, 1.0  # Valances of (+)-ve/(-)-ve charges in salt
            (
                z_plus_conc,
                z_minus_conc,
            ) = (
                ion_concentration,
                ion_concentration,
            )  # Equal +/- charges for monovalent salt

            """
            Coefficients in Taylor series of the charge concentration apart from n! (order >= 1)
            Correct coefficients in Taylor series: NOT in compact form just for clean math formula
            Note: here coeffs. are saved by index thus ith entry represents (i+1)th coeff.
            """

            taylor_coeffs[0] = -2.0 * ion_concentration
            taylor_coeffs[1] = (
                z_plus_conc * z_plus * z_plus**2 - z_plus_conc * z_minus * z_minus**2
            ) / 2.0
            taylor_coeffs[2] = (
                -(z_plus_conc * z_plus * z_plus**3 + z_plus_conc * z_minus * z_minus**3)
                / 6.0
            )
            taylor_coeffs[3] = (
                z_plus_conc * z_plus * z_plus**4 - z_plus_conc * z_minus * z_minus**3
            ) / 24.0
            taylor_coeffs[4] = (
                -(z_plus_conc * z_plus * z_plus**5 + z_plus_conc * z_minus * z_minus**5)
                / 120.0
            )
            self.taylor_coefficients = taylor_coeffs
            self.debye_factor = self.delphi_real(
                Constants.DebyConstant.value
            ) * math.sqrt(temperature * exdi)
            self.ion_conc = ion_concentration
            self.debye_length = self.debye_factor / math.sqrt(salt_conc_molar)

            vprint(
                DEBUG, _VERBOSITY, "(ctx) taylor_coeffs:>>", self.taylor_coefficients
            )
        else:
            # If salt concentration is very low or zero, set Debye length to a predefined large value
            self.debye_length = self.delphi_real(
                ConstDelPhi.ZeroMolarSaltDebyeLength.value
            )
            taylor_coeffs = np.zeros(self.num_taylor_coeff, dtype=self.delphi_real)
            self.taylor_coefficients = taylor_coeffs
            self.ion_conc = 0.0
        return self.debye_length

    def atoms_summary(
        self,
        atoms: dict,
        objects: any,
        extremas_rule: SoluteExtremaRule,
        acenter: np.ndarray[delphi_real],
        enforce_acenter: delphi_bool,
        is_focusing: delphi_bool = False,
    ) -> None:
        """Processes and summarizes atom data, populating `atoms_data` and related attributes.

        This method takes a dictionary of atom data, object data, and a solute extrema rule
        to populate the `atoms_data` array, `atoms_keys`, `max_atom_radius`, and other
        atom-related summary attributes in the context. It iterates through the atom
        dictionary, extracts relevant properties, calculates the maximum atom radius,
        and calls `_atoms_charge_summary` and `_atoms_coordinates_summary` to compute
        charge and coordinate summaries.

        Args:
            atoms (dict[str, np.ndarray]): Dictionary of atom data, where keys are atom identifiers
                                                and values are numpy arrays containing atom properties.
            objects (Any): Data for objects in the system. The type is application-dependent.
            extremas_rule (SoluteExtremaRule): Rule to determine solute extrema (COORDINATE or COORDLIMITS).
            acenter (np.ndarray[delphi_real]): The supplied center for the grid box.
            enforce_acenter (delphi_bool): Whether to enforce the grid box to be centered at the supplied acenter.
            is_focusing (delphi_bool: Whether this is a focusing run.

        Attributes updated:
            extremas_rule (DelphiSoluteExtremaRule): Stores the provided extremas rule.
            atoms_data (np.ndarray): 2D numpy array containing atom properties.
            objects_data (Any): Stores the provided objects data.
            atoms_keys (dict[str, int]): Maps atom keys to their indices in `atoms_data`.
            atoms_index_to_keys (dict[int, str]): Maps atom indices to their keys.
            max_atom_radius (delphi_real): Maximum atomic radius found in the atom data.
            num_atoms (int): Number of atoms processed.
            (and all attributes previously updated by _atoms_charge_summary and _atoms_coordinates_summary)
        """
        vprint(
            DEBUG,
            _VERBOSITY,
            "ctx>>",
            PRECISION,
            _VERBOSITY,
            delphi_bool,
            self.delphi_int,
            self.delphi_real,
        )
        self.extremas_rule = extremas_rule
        num_atoms = len(atoms)
        atomfield_focusing = 1 if is_focusing else 0
        atoms_data = np.zeros(
            (num_atoms, LEN_ATOMFIELDS + atomfield_focusing), dtype=self.delphi_real
        )
        self.objects_data = objects
        self.num_objects = len(objects)

        atoms_keys = self.atoms_keys
        atoms_index_to_keys = self.atoms_index_to_keys
        max_atom_radius_val = 0.0

        atom_index = 0
        # Iterate through the atoms dictionary and populate atoms_data array, and find max_atom_radius
        for a_key, a_data in atoms.items():
            # Assign atom key to index mapping, note focusing summary should not update keys
            if not is_focusing:
                atoms_keys[a_key] = atom_index
                atoms_index_to_keys[atom_index] = a_key
            # Copy atom data to atoms_data array, converting to delphi_real type
            atoms_data[atom_index, :] = a_data.astype(self.delphi_real)[:]

            # Determine maximum atom radius among all atoms
            this_atom_radius = atoms_data[atom_index, ATOMFIELD_RADIUS]
            # Assign to max_atom_radius_val for first atom as it must be +ve value
            if this_atom_radius > max_atom_radius_val:
                max_atom_radius_val = this_atom_radius  # Update max_atom_radius_val

            atom_index += 1

        self.atoms_keys = atoms_keys
        self.atoms_index_to_keys = atoms_index_to_keys
        self.atoms_data = atoms_data
        self.num_atoms = num_atoms
        self.max_atom_radius = max_atom_radius_val
        # Call Numba-jitted loop and unpack results directly into self attributes
        (
            self.total_charge,
            self.positive_charge,
            self.negative_charge,
            self.num_positive_charge,
            self.num_negative_charge,
            self.centroid_positive_charge,
            self.centroid_negative_charge,
            self.coords_by_axis_min,
            self.coords_by_axis_max,
            self.boundary_min,
            self.boundary_max,
            coords_sum,
        ) = _njit_atoms_summary_loop(
            self.atoms_data,
            max_atom_radius_val,
            extremas_rule.value,
            acenter,
            enforce_acenter,
            self.delphi_real,
        )

        # Normalize centroid of positive charge by total positive charge
        if self.positive_charge > 0:
            self.centroid_positive_charge = (
                self.centroid_positive_charge / self.positive_charge
            )
        # Normalize centroid of negative charge by total negative charge (absolute value)
        if abs(self.negative_charge) > 0:
            self.centroid_negative_charge = self.centroid_negative_charge / abs(
                self.negative_charge
            )

        # Calculate centroid of the boundary box
        self.centroid = (self.boundary_max + self.boundary_min) / 2.0
        # Calculate geometric center of all atoms
        self.geometric_center = coords_sum / self.num_atoms

        # Determine solute range and grid center based on extremas_rule
        if extremas_rule.value == SoluteExtremaRule.COORDINATE.value:
            # Solute range based on coordinate extrema (min/max atom coords)
            self.solute_range = self.coords_by_axis_max - self.coords_by_axis_min
            self.grid_center = (
                acenter.astype(self.delphi_real)
                if enforce_acenter
                else self.geometric_center
            )

        else:  # DelphiSoluteExtremaRule.COORDLIMITS.value
            # Solute range based on boundary box (considering atom radii)
            self.solute_range = self.boundary_max - self.boundary_min
            self.grid_center = (
                acenter.astype(self.delphi_real) if enforce_acenter else self.centroid
            )

        # Maximum solute range among x, y, z dimensions
        self.solute_range_max = np.max(self.solute_range)

        self.media_epsilon = np.array(
            [
                self.external_dielectric_scaled * self.epkt,
                self.internal_dielectric_scaled * self.epkt,
            ]
        )

        vprint(
            DEBUG,
            _VERBOSITY,
            "solute_range> min> ",
            self.coords_by_axis_min,
            "max> ",
            self.coords_by_axis_max,
        )
        vprint(
            DEBUG,
            _VERBOSITY,
            "bnd> min> ",
            self.boundary_min,
            "max> ",
            self.boundary_max,
        )

    def prepare_focusing(
        self,
        scale: delphi_real,
        num_atoms: int,
        num_objects: int,
        grid_shape: np.ndarray[delphi_int],
        acenter: np.ndarray[np.float64],
        atoms_data: np.ndarray,
    ) -> tuple[int, np.ndarray, int, np.ndarray[np.intc]]:
        """Prepares context for focusing run, calls Numba helper for atom selection."""
        # Calculate half-length of the grid box based on grid shape and scale
        half_gridbox_length = (max(grid_shape) - 1) / (2.0 * scale)
        # Define lower and higher edges of the focusing box, adding a margin of 3.5 Angstroms
        edge_low = (acenter - half_gridbox_length - 3.5).astype(self.delphi_real)
        edge_high = (acenter + half_gridbox_length + 3.5).astype(self.delphi_real)

        vprint(
            DEBUG,
            _VERBOSITY,
            "datacenter> half_gridbox_length =",
            half_gridbox_length,
        )
        vprint(
            DEBUG,
            _VERBOSITY,
            f"datacenter> acenter = ( {acenter[0]:.6g}, {acenter[1]:.6g}, {acenter[2]:.6g})",
        )
        vprint(
            DEBUG,
            _VERBOSITY,
            "datacenter>> edge_high = ",
            edge_high,
            edge_high.dtype,
        )
        vprint(DEBUG, _VERBOSITY, "datacenter>> edge_low = ", edge_low, edge_low.dtype)
        vprint(
            DEBUG,
            _VERBOSITY,
            "datacenter>> get_atom_coords(this_atom)",
            atoms_data[0][ATOMFIELD_X:ATOMFIELD_CRD_END],
            atoms_data[0][0].dtype,
        )

        # Call Numba-jitted loop for atom selection
        num_atoms_focus, atoms_data_focus = _njit_prepare_focusing_loop(
            atoms_data,
            edge_high,
            edge_low,
            LEN_ATOMFIELDS,
            self.delphi_real,  # Pass numpy dtype for Numba compatibility
        )

        # Epsilon dimension for focusing run: atoms + objects + 2
        epsdim_focus = num_atoms_focus + num_objects + 2
        focus_start = np.zeros(
            3, dtype=np.intc
        )  # Initialize focus start indices (always [0,0,0])

        # Save parent run information to context
        self.num_atoms_parentrun = num_atoms
        self.atoms_data_parentrun = atoms_data
        self.epsdim_parentrun = self.epsilon_dimension
        self.grid_center_parentrun = self.grid_center
        self.grid_origin_parentrun = self.grid_origin_parentrun

        # Update context attributes with focusing run information
        self.num_atoms = num_atoms_focus
        self.atoms_data = atoms_data_focus
        self.epsilon_dimension = epsdim_focus
        self.focus_start = focus_start

        return num_atoms_focus, atoms_data_focus, epsdim_focus, focus_start

    def grid_params(
        self,
        scale: delphi_real,
        perfil: delphi_real,
        gridbox_margin: delphi_real,
        gridbox_size: GridboxSize,
        gridbox_type: GridBoxType,
    ) -> tuple[delphi_real, delphi_real, np.ndarray[delphi_int]]:
        """Calculates and sets grid parameters based on input or solute properties.

        This method determines the grid scale, perfil (percentage fill), grid shape,
        and grid margin based on the combination of input parameters provided.
        It prioritizes `scale` and `gridbox_size` if both are given, otherwise, it
        calculates the missing parameters based on available inputs and solute dimensions.
        It also ensures that the grid box dimensions are odd numbers, incrementing by 1
        if necessary to maintain grid centering.

        Args:
            scale (delphi_real): Desired grid scale (grid points per Angstrom).
            perfil (delphi_real): Desired percentage fill of the solute in the grid box.
            gridbox_margin (delphi_real): Desired margin around the solute in Angstroms.
            gridbox_size (GridboxSize): Desired grid box dimensions (nx, ny, nz).
                                            Use DelphiGridboxSize(0) or 0 to indicate auto-determination.
            gridbox_type (GridBoxType): Type of grid box (CUBIC or CUBOIDAL).

        Returns:
            tuple[delphi_real, delphi_real, np.ndarray[delphi_int]]:
                A tuple containing:
                    - scale (delphi_real): The calculated or provided grid scale.
                    - perfil (delphi_real): The calculated or provided percentage fill.
                    - grid_shape (np.ndarray[delphi_int]): The determined grid shape (nx, ny, nz) as integer array.

        Attributes updated:
            scale (delphi_real): Stores the calculated or provided scale.
            grid_spacing (delphi_real): Grid spacing (1/scale).
            gridbox_size (DelphiGridboxSize): Stores the determined grid box size, ensuring odd dimensions.
            perfil (delphi_real): Stores the calculated or provided perfil.
            gridbox_margin (delphi_real): Stores the calculated or provided gridbox_margin.
            grid_shape (np.ndarray[delphi_int]): Stores the determined grid shape.
        """
        warnings = []
        # Ensure gridbox_size dimensions are odd by incrementing if even
        if (not gridbox_size == 0) and gridbox_size.nx % 2 == 0:
            gridbox_size_new = GridboxSize(
                gridbox_size.nx + 1, gridbox_size.ny, gridbox_size.nz
            )
            warnings.append(
                (
                    "WARNING> only zero or odd ngrid accepted!",
                    gridbox_size,
                    "will be replaced with:",
                    gridbox_size_new,
                )
            )
            gridbox_size = gridbox_size_new
        if (not gridbox_size == 0) and gridbox_size.ny % 2 == 0:
            gridbox_size_new = GridboxSize(
                gridbox_size.nx, gridbox_size.ny + 1, gridbox_size.nz
            )
            warnings.append(
                (
                    "WARNING> only zero or odd ngrid accepted!",
                    gridbox_size,
                    "will be replaced with:",
                    gridbox_size_new,
                )
            )
            gridbox_size = gridbox_size_new
        if (not gridbox_size == 0) and gridbox_size.nz % 2 == 0:
            gridbox_size_new = GridboxSize(
                gridbox_size.nx, gridbox_size.ny, gridbox_size.nz + 1
            )
            warnings.append(
                (
                    "WARNING> only zero or odd ngrid accepted!",
                    gridbox_size,
                    "will be replaced with:",
                    gridbox_size_new,
                )
            )
            gridbox_size = gridbox_size_new

        # Determine grid parameters based on input priority and availability
        if scale > 0 and gridbox_size != 0:
            self.scale = scale
            self.grid_spacing = 1.0 / self.scale
            self._gridbox_size = gridbox_size
            gs_max = self._gridbox_size.nx  # Start with nx, then compare with ny, nz
            gs_max = self._gridbox_size.ny if self._gridbox_size.ny > gs_max else gs_max
            gs_max = self._gridbox_size.nz if self._gridbox_size.nz > gs_max else gs_max
            self.perfil = float(self.solute_range_max * 100 * scale / (gs_max - 1))
            self.gridbox_margin = float(
                ((gs_max - 1) * self.grid_spacing - self.solute_range_max) / 2.0
            )
        elif scale != 0 and (not gridbox_size == 0):
            # Scale and gridbox_size are provided, calculate perfil and gridbox_margin
            self.scale = scale
            self.grid_spacing = 1.0 / self.scale
            self._gridbox_size = gridbox_size
            gs_max = self._gridbox_size.nx  # Start with nx, then compare with ny, nz
            gs_max = self._gridbox_size.ny if self._gridbox_size.ny > gs_max else gs_max
            gs_max = self._gridbox_size.nz if self._gridbox_size.nz > gs_max else gs_max
            self.perfil = float(self.solute_range_max * 100 * scale / (gs_max - 1))
            self.gridbox_margin = float(
                ((gs_max - 1) * self.grid_spacing - self.solute_range_max) / 2.0
            )
        elif perfil != 0 and (not gridbox_size == 0):
            # Perfil and gridbox_size are provided, calculate scale and gridbox_margin
            self.perfil = perfil
            self._gridbox_size = gridbox_size
            gs_max = self._gridbox_size.nx  # Start with nx, then compare with ny, nz
            gs_max = self._gridbox_size.ny if self._gridbox_size.ny > gs_max else gs_max
            gs_max = self._gridbox_size.nz if self._gridbox_size.nz > gs_max else gs_max
            self.scale = float((gs_max - 1) * perfil / (self.solute_range_max * 100))
            self.grid_spacing = 1.0 / self.scale
            self.gridbox_margin = float(
                ((gs_max - 1) * self.grid_spacing - self.solute_range_max) / 2.0
            )
        elif scale != 0 and gridbox_margin != 0:
            # Scale and gridbox_margin are provided, calculate gridbox_size and perfil
            self.scale = scale
            self.grid_spacing = 1.0 / self.scale
            self.gridbox_margin = gridbox_margin
            if gridbox_type.value == GridBoxType.CUBOIDAL.value:
                # Cuboidal gridbox: size is determined by solute range and margin for each dimension
                gs_nx = int((self.solute_range[0] + 2 * gridbox_margin) * scale + 1)
                gs_ny = int((self.solute_range[1] + 2 * gridbox_margin) * scale + 1)
                gs_nz = int((self.solute_range[2] + 2 * gridbox_margin) * scale + 1)
                # Ensure grid dimensions are odd
                if gs_nx % 2 == 0:
                    gs_nx += 1
                if gs_ny % 2 == 0:
                    gs_ny += 1
                if gs_nz % 2 == 0:
                    gs_nz += 1
                gs_max = gs_nx  # Start with nx, then compare with ny, nz to find max dimension
                gs_max = gs_ny if gs_ny > gs_max else gs_max
                gs_max = gs_nz if gs_nz > gs_max else gs_max
                self._gridbox_size = GridboxSize(gs_nx, gs_ny, gs_nz)
                self.perfil = float(self.solute_range_max * 100 * scale / (gs_max - 1))
            else:  # DelphiGridBoxType.CUBIC.value:
                # Cubic gridbox: size is determined by maximum solute range and margin
                gs_nlargest = int(
                    (self.solute_range_max + 2 * gridbox_margin) * scale + 1
                )
                # Ensure grid dimension is odd
                if gs_nlargest % 2 == 0:
                    gs_nlargest += 1
                self._gridbox_size = GridboxSize(gs_nlargest, gs_nlargest, gs_nlargest)
                self.perfil = float(
                    self.solute_range_max * 100 * scale / (gs_nlargest - 1)
                )
        elif scale != 0 and perfil != 0 and gridbox_margin == 0:
            # Scale and perfil are provided, calculate gridbox_size and gridbox_margin
            self.scale = scale
            self.grid_spacing = 1.0 / self.scale
            self.perfil = perfil
            if gridbox_type.value == GridBoxType.CUBOIDAL.value:
                # Cuboidal gridbox: size determined by solute range and perfil for each dimension
                gs_nx = int((scale * 100 / perfil) * self.solute_range[0])
                gs_ny = int((scale * 100 / perfil) * self.solute_range[1])
                gs_nz = int((scale * 100 / perfil) * self.solute_range[2])
                # Ensure grid dimensions are odd
                if gs_nx % 2 == 0:
                    gs_nx += 1
                if gs_ny % 2 == 0:
                    gs_ny += 1
                if gs_nz % 2 == 0:
                    gs_nz += 1
                gs_max = gs_nx  # Start with nx, then compare with ny, nz to find max dimension
                gs_max = gs_ny if gs_ny > gs_max else gs_max
                gs_max = gs_nz if gs_nz > gs_max else gs_max
                self._gridbox_size = GridboxSize(gs_nx, gs_ny, gs_nz)
                self.gridbox_margin = float(
                    ((gs_max - 1) * self.grid_spacing - self.solute_range_max) / 2.0
                )
            else:  # DelphiGridBoxType.CUBIC.value:
                # Cubic gridbox: size determined by maximum solute range and perfil
                gs_nlargest = int((scale * 100 / perfil) * self.solute_range_max)
                # Ensure grid dimension is odd
                if gs_nlargest % 2 == 0:
                    gs_nlargest += 1
                self._gridbox_size = GridboxSize(gs_nlargest, gs_nlargest, gs_nlargest)
                self.gridbox_margin = float(
                    ((gs_nlargest - 1) * self.grid_spacing - self.solute_range_max)
                    / 2.0
                )
        # Print warnings if any and verbosity is higher than mandatory
        for warning in warnings:
            vprint(WARNING, _VERBOSITY, warning)

        # Store grid shape as numpy array
        self.grid_shape[0] = self._gridbox_size.nx
        self.grid_shape[1] = self._gridbox_size.ny
        self.grid_shape[2] = self._gridbox_size.nz
        return (self.scale, self.perfil, self.grid_shape)

    def setup_gridmap_3d(
        self,
        grid_center: np.ndarray[delphi_real],
        grid_shape: np.ndarray[delphi_int],
        scale: delphi_real,
    ) -> np.ndarray[delphi_real]:
        """Sets up the 3D grid origin based on grid_center, grid shape, and scale.

        Calculates the origin of the 3D grid in Angstrom coordinates based on the
        provided centroid of the solute, grid shape (dimensions), and grid scale.
        It determines the grid spacing and handles grid centering for both even and odd
        grid dimensions.

        Args:
            grid_center (np.ndarray[delphi_real]): Center coordinates of the gridbox.
            grid_shape (np.ndarray[delphi_int]): Shape of the grid (nx, ny, nz) as integer array.
            scale (delphi_real): Grid scale (grid points per Angstrom).

        Returns:
            np.ndarray[delphi_real]: The calculated grid origin in Angstroms (x, y, z).

        Attributes updated:
            grid_spacing (delphi_real): Grid spacing (1/scale).
            grid_origin (np.ndarray[delphi_real]): Stores the calculated grid origin.
        """
        # Calculate grid spacing from scale
        grid_spacing = 1.0 / scale
        grid_spacing_half = 0.5 * grid_spacing

        # Determine midpoint indices of the grid along each dimension
        mid_grid_x = (grid_shape[0]) // 2
        mid_grid_y = (grid_shape[1]) // 2
        mid_grid_z = (grid_shape[2]) // 2

        # Initialize offset and grid origin arrays
        offset_com = np.array([0, 0, 0], dtype=self.delphi_real)
        grid_origin = np.array([0, 0, 0], dtype=self.delphi_real)

        # Apply offset if grid dimensions are even to center grid properly
        if (
            grid_shape[0] % 2 == 0
        ):  # Checking only grid_shape[0] is sufficient for cubic grids in DELPHI
            offset_com = (grid_spacing_half, grid_spacing_half, grid_spacing_half)

        vprint(
            DEBUG, _VERBOSITY, "dc|offset_com, offset_com>> ", grid_shape, offset_com
        )
        # Calculate grid origin based on midpoint, spacing, offset and centroid
        grid_origin = np.array(
            [
                -mid_grid_x * grid_spacing + offset_com[0] + grid_center[0],
                -mid_grid_y * grid_spacing + offset_com[1] + grid_center[1],
                -mid_grid_z * grid_spacing + offset_com[2] + grid_center[2],
            ],
            dtype=self.delphi_real,
        )

        # Update context attributes with calculated grid parameters
        self.grid_spacing = grid_spacing
        self.grid_origin = grid_origin

        return grid_origin

    def gridbox_size_to_shape_array(self):
        grid_shape = np.zeros(3, dtype=np.int64)
        if self._gridbox_size is not None:
            grid_shape[0] = self._gridbox_size.nx
            grid_shape[1] = self._gridbox_size.ny
            grid_shape[2] = self._gridbox_size.nz
        else:
            raise ValueError(
                "Uninitialized `ctx._gridbox_size` to grid_shape conversion attempted."
            )
        return grid_shape

    def dump_text(self, filename: str) -> None:
        """Dumps the contents of the context to a text file for debugging.

        Writes out all attributes of the context and their values
        to a text file. This is primarily used for debugging and inspection
        of the context's state.

        Args:
            filename (str): Path to the file where the context's data will be dumped.
        """
        fout = open(filename, "w")
        for fn, fv in self.__dict__.items():
            fout.write(f"___{fn}___: type={type(fv)}")
            if not fv is None:
                if type(fv) not in [bool, int, float, str, dict, GridboxSize]:
                    shape = [v for v in fv.shape]
                    fout.write(
                        f"; size={fv.size}; shape={fv.shape}; dtype={type(fv.reshape(-1)[0])}\n"
                    )
                    fv = fv.reshape(-1)
                    if fv.size > 0:
                        for iv, val in enumerate(fv):
                            # print(val)
                            if isinstance(val, np.ndarray):
                                fout.write(
                                    "["
                                    + ", ".join(
                                        [f"{i:10.06g}" for i in val.reshape(-1)]
                                    )
                                    + "]\n"
                                )
                            else:
                                fout.write(f"{val:13.06g}")
                                if (iv + 1) % 6 == 0 or iv == fv.size - 1:
                                    fout.write("\n")
                    else:
                        fout.write(f"{fv:13.08g}")
                    fout.write("\n")
                else:
                    if type(fv) in [dict]:
                        for kv, vv in fv.items():
                            fout.write(f"{kv}:{vv}")
                        fout.write("\n")
                    elif isinstance(fv, GridboxSize):
                        pass  # Do not dump DelphiGridboxSize info
                    else:
                        fout.write("\n")
                        fout.write(f"{fv:13.08g}\n")
            else:
                fout.write("\n")
        fout.flush()
