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


import time
import math
import numpy as np
from numba import cuda

from pydelphi.foundation.enums import Precision, SurfaceMethod

from pydelphi.constants import (
    ConstDelPhiFloats as ConstDelPhi,
)

APPROX_ZERO = ConstDelPhi.ApproxZero.value
GAUSSIAN_INFLUENCE_RADIUS_FACTOR = ConstDelPhi.GaussianInfluenceRadiusFactor.value

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

# --- Dynamic Precision Handling ---
if PRECISION.int_value in {Precision.SINGLE.int_value}:

    try:
        import pydelphi.utils.cuda.single as size_gpu
    except ImportError:
        size_gpu = None
elif PRECISION.int_value == Precision.DOUBLE.int_value:

    try:
        import pydelphi.utils.cuda.double as size_gpu
    except ImportError:
        size_gpu = None
else:
    raise ValueError(f"Unsupported PRECISION: {PRECISION}")

from pydelphi.space.core.voxelizer import (
    build_consolidated_atoms_space_voxel_map,
)

from pydelphi.foundation.context import RuntimeContext
from pydelphi.space import surface
from pydelphi.space import vdwms

from pydelphi.space.core.grid_charger import set_grid_charges_sorted_by_index1d
from pydelphi.space.core.gaussian import (
    calc_atom_gaussian_influence_radius,
    calc_spatial_epsilon_map_midpoints,
    calc_gaussian_cutoff_spatial_epsilon_map_midpoints,
    calc_gaussian_density_map,
)

from pydelphi.space.core.vdw_eps_initizer import (
    calculate_vdw_zeta_surf_map,
    calculate_vdw_discrete_epsilon_map,
    calculate_vdw_dielectric_map_midpoints,
)


# ==============================================================================
# Space Class Definition
# ==============================================================================
class Space:
    def __init__(
        self,
        platform,
        is_surf_midpoints,
        scale,
        exdi,
        gapdi,
        indi,
        media_epsilon,
        probe_radius,
        probe_radius2,
        debye_length,
        salt_radius,
        gaussian_sigma,
        gaussian_exponent,
        max_atom_radius,
        verbosity,
        dielectric_model,
        surface_method,
        surface_offset,
        surface_density_exponent,
        grid_shape,
        grid_origin,
        atoms_data,
        is_focusing,
        grid_shape_parentrun,
        grid_origin_parentrun,
        acenter,
        coords_by_axis_min,
        coords_by_axis_max,
        objects_data=None,
        num_objects=1,
        num_molecules=None,
        use_zeta_surf=True,
        zeta_distance=0.0,
        enabled_nonpolar_energy=False,
    ):
        self.num_cuda_threads = 128
        self.platform = platform
        self.is_surf_midpoints = is_surf_midpoints
        self.scale = delphi_real(scale)
        self.exdi = exdi
        self.gapdi = gapdi
        self.indi = indi
        self.media_epsilon = media_epsilon.astype(delphi_real)
        self.probe_radius = probe_radius
        self.probe_radius2 = probe_radius2
        self.debye_length = debye_length
        self.salt_radius = delphi_real(salt_radius)
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_exponent = gaussian_exponent
        self.max_atom_radius = max_atom_radius
        self.verbosity = verbosity
        self.dielectric_model = dielectric_model
        self.surface_method = surface_method
        self.surface_offset = surface_offset
        self.surface_density_exponent = surface_density_exponent
        self.grid_shape = grid_shape.astype(delphi_int)
        self.grid_origin = grid_origin.astype(delphi_real)
        self.atoms_data = atoms_data.astype(delphi_real)
        self.is_focusing = is_focusing
        self.grid_shape_parentrun = grid_shape_parentrun
        self.grid_origin_parentrun = grid_origin_parentrun
        self.acenter = acenter
        if coords_by_axis_min is None or coords_by_axis_max is None:
            raise ValueError("coords_by_axis_min/max must be provided.")
        self.coords_by_axis_min = coords_by_axis_min.astype(delphi_real)
        self.coords_by_axis_max = coords_by_axis_max.astype(delphi_real)
        self.objects_data = objects_data
        self.num_objects = num_objects if num_objects is not None else 1
        self.num_molecules = (
            num_molecules if num_molecules is not None else self.num_objects
        )
        self.atom_influence_radius = 0.0  # Should be computed if gausian method
        self.use_zeta_surf = use_zeta_surf
        self.zeta_distance = delphi_real(zeta_distance)
        self.enabled_nonpolar = enabled_nonpolar_energy
        self.num_atoms = self.atoms_data.shape[0]
        self.grid_spacing = 1.0 / self.scale
        self.num_grid_points = np.prod(self.grid_shape)
        self.epsilon_dimension = self.num_atoms + self.num_objects + 2

        # Object local fields used to facilitate calculations
        self.zeta_surf_grid_coords = None
        self.zeta_surf_grid_indices = None
        self.num_zeta_surf_grid_coords = 0
        self.atom_voxel_map_params = None
        self.atom_voxel_map_data = None
        self.atom_voxel_zeta_map_params = None
        self.atom_voxel_zeta_map_data = None
        self.atom_voxel_gauss_map_params = None
        self.atom_voxel_gauss_map_data = None

        # Initialize all outputs
        self.num_charged_gridpoints = 0
        self.num_positive_charge = 0
        self.num_negative_charge = 0
        self.index_discrete_epsilon_map_1d = None
        self.dielectric_boundary_map_1d = None
        self.discrete_epsilon_map_midpoints_1d = None
        self.dielectric_boundary_grids = None

        # Gaussian dielectric model output fields
        self.gauss_density_map_1d = None
        self.gauss_density_map_midpoints_1d = None
        self.surface_map_1d = None
        self.grad_surface_map_1d = None

        # Diffused interface surface models related fields
        self.solute_outside_map_1d = None  # for GCS
        self.solute_inside_map_1d = None  # for GCS
        self.surf_heavyside_map_1d = None  # for GCS
        self.surface_map_midpoints_1d = None  # for Gaussian surface

        # Dielectric and ion-exclusion maps
        self.epsilon_map_1d = None
        self.epsilon_map_midpoints_1d = None
        self.ion_exclusion_map_1d = None

        self.charged_gridpoints_1d = None
        self.zeta_surface_map_1d = np.ones(
            0, dtype=delphi_bool
        )  # init to enable type inference
        self.induced_surf_charge_positions = None
        self.timings = {}

    def _calc_atoms_voxel_space_aware_gaussian_density_map(
        self, side_length_gauss_voxel, generate_ion_exclusion_map=False
    ):
        """Calculates Gaussian density map, using a voxel map for efficiency."""
        vprint(DEBUG, _VERBOSITY, "Initializing Gaussian Density Map...")

        # Determine side length for Gaussian density context
        vprint(
            DEBUG,
            _VERBOSITY,
            f"Gaussian Density Context: Calculating map with voxel side length {side_length_gauss_voxel:.3f}",
        )

        # Calculate Params & Build Map for Gaussian density context if not already built
        if self.atom_voxel_map_params is None or self.atom_voxel_map_data is None:
            vprint(DEBUG, _VERBOSITY, "Building voxel map for Gaussian density...")
            voxel_params_gauss, voxel_data_gauss, time_elapsed = (
                build_consolidated_atoms_space_voxel_map(
                    side_length_gauss_voxel,
                    self.coords_by_axis_min,
                    self.coords_by_axis_max,
                    1.0,
                    0.1,
                    self.num_atoms,
                    self.num_objects,
                    self.num_molecules,
                    self.atoms_data,
                )
            )
            self.timings["space| build specific voxel map"] = round(
                self.timings.get("space| build specific voxel map", 0) + time_elapsed,
                3,
            )
            if voxel_data_gauss is None or voxel_params_gauss is None:
                raise RuntimeError("Failed to build Gaussian density voxel map.")
            self.atom_voxel_map_params = voxel_params_gauss
            self.atom_voxel_map_data = voxel_data_gauss
        else:
            vprint(
                DEBUG, _VERBOSITY, "Reusing existing voxel map for Gaussian density."
            )

        voxel_ids, voxel_start, voxel_end = self.atom_voxel_map_data
        voxel_origin, voxel_shape, voxel_scale, _ = self.atom_voxel_map_params

        # Allocate output arrays
        self.gauss_density_map_1d = np.zeros(self.num_grid_points, dtype=delphi_real)
        self.gauss_density_map_midpoints_1d = np.zeros(
            self.num_grid_points * 3, dtype=delphi_real
        )
        self.ion_exclusion_map_1d = np.zeros(self.num_grid_points, dtype=delphi_bool)

        calc_gaussian_density_map(
            self.platform,
            self.num_cuda_threads,
            generate_ion_exclusion_map=delphi_bool(generate_ion_exclusion_map),
            scale=self.scale,
            gaussian_exponent=self.gaussian_exponent,
            gaussian_influence_radius_factor=delphi_real(
                GAUSSIAN_INFLUENCE_RADIUS_FACTOR
            ),
            surface_offset=self.surface_offset,
            atom_influence_radius=self.atom_influence_radius,
            salt_radius=self.salt_radius,
            grid_shape=self.grid_shape.astype(delphi_int),
            grid_origin=self.grid_origin.astype(delphi_real),
            atoms_data=self.atoms_data.astype(delphi_real),
            gauss_density_map_1d=self.gauss_density_map_1d,
            gauss_density_map_midpoints_1d=self.gauss_density_map_midpoints_1d,
            ion_exclusion_map_1d=self.ion_exclusion_map_1d,
            voxel_atom_ids=voxel_ids.astype(delphi_int),
            voxel_atom_start_index=voxel_start.astype(delphi_int),
            voxel_atom_end_index=voxel_end.astype(delphi_int),
            voxel_map_origin=voxel_origin.astype(delphi_real),
            voxel_map_shape=voxel_shape.astype(delphi_int),
            voxel_map_scale=voxel_scale,
        )

        vprint(DEBUG, _VERBOSITY, " Finished Gaussian Density Map ")

    def _init_discrete_epsilon_index_map(self, activate_polar_energy_calc=False):
        """MODIFIED: Calculates VDW map (epsidx, boundary) using VDW-context voxel map."""
        if self.num_atoms == 0:
            vprint(DEBUG, _VERBOSITY, "No atoms, skipping VDW map init.")
            return

        # Determine side length for VDW context
        grid_spacing_extension_factor = 0.5 * self.grid_spacing * math.sqrt(3)
        side_length_vdw = (
            self.max_atom_radius + self.salt_radius + grid_spacing_extension_factor
        )
        vprint(
            DEBUG,
            _VERBOSITY,
            f"VDW Context: Calculating map with voxel side length {side_length_vdw:.3f}",
        )

        # Calculate Params & Build Map for VDW context
        # voxel_params_vdw = self._setup_voxel_params(side_length_vdw)
        # voxel_data_vdw = self._build_specific_voxel_map(voxel_params_vdw)
        voxel_params_vdw, voxel_data_vdw, time_elapsed = (
            build_consolidated_atoms_space_voxel_map(
                side_length_vdw,
                self.coords_by_axis_min,
                self.coords_by_axis_max,
                1.0,
                0.1,
                self.num_atoms,
                self.num_objects,
                self.num_molecules,
                self.atoms_data,
            )
        )
        self.timings["space| build specific voxel map"] = round(
            self.timings.get("space| build specific voxel map", 0) + time_elapsed,
            3,
        )
        if voxel_data_vdw is None or voxel_params_vdw is None:
            raise RuntimeError("Failed to build VDW voxel map.")

        # Store map for potential reuse (e.g., by Zeta)
        self.atom_voxel_map_params = voxel_params_vdw
        self.atom_voxel_map_data = voxel_data_vdw

        # Allocate output arrays for this function
        self.index_discrete_epsilon_map_1d = np.zeros(
            self.num_grid_points * 3, dtype=np.int32
        )
        self.dielectric_boundary_map_1d = np.empty(self.num_grid_points, dtype=np.bool_)

        # Retrieve data/params for the call
        voxel_ids, voxel_start, voxel_end = self.atom_voxel_map_data
        voxel_origin, voxel_shape, voxel_scale, _ = self.atom_voxel_map_params

        (self.index_discrete_epsilon_map_1d, self.dielectric_boundary_map_1d) = (
            calculate_vdw_discrete_epsilon_map(
                platform=self.platform,
                num_cuda_threads=self.num_cuda_threads,
                epsilon_dimension=self.epsilon_dimension,
                scale=self.scale,
                salt_radius=self.salt_radius,
                grid_shape=self.grid_shape,
                grid_origin=self.grid_origin,
                atoms_data=self.atoms_data,
                index_discrete_epsilon_map_1d=self.index_discrete_epsilon_map_1d,
                dielectric_boundary_map_1d=self.dielectric_boundary_map_1d,
                voxel_atom_ids=voxel_ids,
                voxel_atom_start_index=voxel_start,
                voxel_atom_end_index=voxel_end,
                voxel_map_origin=voxel_origin,
                voxel_map_shape=voxel_shape,
                voxel_map_scale=voxel_scale,
            )
        )

        vprint(DEBUG, _VERBOSITY, " Finished VDW Epsilon/Boundary Map (Optimized) ")

    def _init_zeta_surface_map(self):
        """Calculates zeta surface map, potentially reusing or building a new voxel map."""
        if not self.use_zeta_surf:
            vprint(DEBUG, _VERBOSITY, "Zeta surface not requested, skipping map init.")
            return
        if self.num_atoms == 0:
            vprint(DEBUG, _VERBOSITY, "No atoms, skipping zeta surface map init.")
            return

        vprint(DEBUG, _VERBOSITY, "Initializing Zeta Surface Map...")
        voxel_params_to_use = None
        voxel_data_to_use = None

        # Decide whether to reuse VDW map or build a new one
        # Condition: Build new if zeta_distance > salt_radius AND VDW map exists
        build_separate_zeta_map = (
            self.zeta_distance > self.salt_radius
            and self.atom_voxel_map_params is not None
            and self.atom_voxel_map_data is not None
        )

        if build_separate_zeta_map:
            vprint(
                DEBUG,
                _VERBOSITY,
                "Zeta distance > salt radius, building separate zeta voxel map.",
            )
            grid_spacing_extension_factor = 0.5 * self.grid_spacing * math.sqrt(3)
            side_length_zeta = (
                self.max_atom_radius
                + self.zeta_distance
                + grid_spacing_extension_factor
            )
            vprint(
                DEBUG,
                _VERBOSITY,
                f"Zeta Context: Calculating map with voxel side length {side_length_zeta:.3f}",
            )
            # voxel_params_zeta = self._setup_voxel_params(side_length_zeta)
            # voxel_data_zeta = self._build_specific_voxel_map(voxel_params_zeta)
            voxel_params_zeta, voxel_data_zeta, time_elapsed = (
                build_consolidated_atoms_space_voxel_map(
                    side_length_zeta,
                    self.coords_by_axis_min,
                    self.coords_by_axis_max,
                    1.0,
                    0.1,
                    self.num_atoms,
                    self.num_objects,
                    self.num_molecules,
                    self.atoms_data,
                )
            )
            self.timings["space| build specific voxel map"] = round(
                self.timings.get("space| build specific voxel map", 0) + time_elapsed,
                3,
            )
            if voxel_data_zeta is None or voxel_params_zeta is None:
                raise RuntimeError("Failed to build Zeta voxel map.")
            # Store separately
            self.atom_voxel_zeta_map_params = voxel_params_zeta
            self.atom_voxel_zeta_map_data = voxel_data_zeta
            voxel_params_to_use = voxel_params_zeta
            voxel_data_to_use = voxel_data_zeta
        else:
            # Reuse VDW map if available, otherwise build it first
            if self.atom_voxel_map_data is None or self.atom_voxel_map_params is None:
                vprint(
                    DEBUG,
                    _VERBOSITY,
                    "VDW voxel map not pre-built for Zeta reuse. Building now...",
                )
                # For now, assume VDW init is called before Zeta init
                raise RuntimeError(
                    "VDW voxel map needed for Zeta context but not found."
                )
            vprint(DEBUG, _VERBOSITY, "Reusing VDW voxel map for Zeta context.")
            voxel_params_to_use = self.atom_voxel_map_params
            voxel_data_to_use = self.atom_voxel_map_data
            # Optionally store references if needed for clarity elsewhere
            self.atom_voxel_zeta_map_params = self.atom_voxel_map_params
            self.atom_voxel_zeta_map_data = self.atom_voxel_map_data

        # Allocate output array
        self.zeta_surface_map_1d = np.ones(self.num_grid_points, dtype=delphi_bool)

        # Retrieve data/params for the call
        voxel_ids, voxel_start, voxel_end = voxel_data_to_use
        voxel_origin, voxel_shape, voxel_scale, _ = voxel_params_to_use

        calculate_vdw_zeta_surf_map(
            platform=self.platform,
            num_cuda_threads=self.num_cuda_threads,
            scale=self.scale,
            zeta_distance=self.zeta_distance,
            grid_shape=self.grid_shape,
            grid_origin=self.grid_origin,
            atoms_data=self.atoms_data,
            zeta_surface_map_1d=self.zeta_surface_map_1d,
            # Pass chosen voxel args (VDW or Zeta specific)
            voxel_zeta_atom_ids=voxel_ids,
            voxel_zeta_atom_start_index=voxel_start,
            voxel_zeta_atom_end_index=voxel_end,
            voxel_zeta_map_origin=voxel_origin,
            voxel_zeta_map_shape=voxel_shape,
            voxel_zeta_map_scale=voxel_scale,
        )

        vprint(DEBUG, _VERBOSITY, " Finished Zeta Surface Map ")

    def _calculate_vdw_molecular_surface(self):
        # --- VDW Surface Path ---
        tic_vdw_total = time.perf_counter()
        # 1. Initialize VDW epsilon index and dielectric boundary maps
        self._init_discrete_epsilon_index_map()  # Builds/stores VDW voxel map

        # 2. Initialize Zeta surface map (conditionally builds/reuses map)
        self._init_zeta_surface_map()

        # 3. VDW Surface Calculation (Passes calculated maps)
        tic_vdw_srf = time.perf_counter()
        molsurf_vdw = vdwms.SurfaceMolecularVdW(
            platform=self.platform,
            grid_spacing=self.grid_spacing,
            probe_radius=self.probe_radius,
            probe_radius_second=self.probe_radius2,
            debye_length=self.debye_length,
            salt_radius=self.salt_radius,
            radius_offset=0.0,
            max_radius=self.max_atom_radius,
            max_atom_radius=self.max_atom_radius,
            grid_shape=self.grid_shape,
            grid_origin=self.grid_origin,
            grid_scale=self.scale,
            atoms_data=self.atoms_data,
            atom_index_array=None,
            objects_data=self.objects_data,
            n_objects=1,
            n_molecules=1,
            min_coords_by_axis=self.coords_by_axis_min,
            max_coords_by_axis=self.coords_by_axis_max,
            surface_offset=self.surface_offset,
            surface_charge_positions=None,
            is_focusing=self.is_focusing,
            grid_shape_parentrun=self.grid_shape_parentrun,
            grid_origin_parentrun=self.grid_origin_parentrun,
            use_zeta_surface_calculation=self.use_zeta_surf,
            index_discrete_epsilon_map_1d=self.index_discrete_epsilon_map_1d,
            dielectric_boundary_map_1d=self.dielectric_boundary_map_1d,
            zeta_surface_map_1d=self.zeta_surface_map_1d,
            verbosity=self.verbosity,
            approx_zero=delphi_real(ConstDelPhi.ApproxZero.value),
        )
        molsurf_vdw.create_vdw_molecular_surfaces(
            use_zeta_surface=self.use_zeta_surf,
            solve_pbe=True,
            read_rxn_from_frc=True,
            calc_solvation_energy=True,
            calc_nonlinear_energy=False,
            calc_surface_energy=False,
            calc_surface_charge=False,
            only_molecule=True,
            profile_timings=_VERBOSITY <= DEBUG,
        )
        toc_vdw_srf = time.perf_counter()
        self.timings["space| build vdw surfaces"] = f"{toc_vdw_srf - tic_vdw_srf:.3f}"
        # 4. Store results from vdwms
        self.zeta_surf_grid_coords = molsurf_vdw.zeta_surface_grid_coords
        self.zeta_surf_grid_indices = molsurf_vdw.zeta_surface_grid_indices
        self.num_zeta_surf_grid_coords = molsurf_vdw.num_zeta_surface_point_coords
        self.dielectric_boundary_grids = molsurf_vdw.boundary_grid_points[
            : molsurf_vdw.num_boundary_grid_points
        ].astype(delphi_int)
        self.induced_surf_charge_positions = molsurf_vdw.surface_charge_positions

    def _init_adjacency_map(self):
        """Calculates adjacency list, building a new voxel map."""
        if not self.enabled_nonpolar:
            vprint(
                DEBUG,
                _VERBOSITY,
                "Non-polar energy not not requested, skipping map init.",
            )
            return
        if self.num_atoms == 0:
            vprint(DEBUG, _VERBOSITY, "No atoms, skipping adjacency list init.")
            return

        vprint(DEBUG, _VERBOSITY, "Initializing Overlap Adjacency List...")

        from pydelphi.space.core.adjacency_builder import (
            calculate_atom_overlap_adjacency,
        )

        voxel_params_to_use = None
        voxel_data_to_use = None

        grid_spacing_extension_factor = 0.5 * self.grid_spacing * math.sqrt(3)
        side_length_overlap = self.max_atom_radius + grid_spacing_extension_factor

        voxel_params_overlap, voxel_data_overlap, time_elapsed = (
            build_consolidated_atoms_space_voxel_map(
                side_length_overlap,
                self.coords_by_axis_min,
                self.coords_by_axis_max,
                1.0,
                0.1,
                self.num_atoms,
                self.num_objects,
                self.num_molecules,
                self.atoms_data,
            )
        )
        vprint(
            DEBUG,
            _VERBOSITY,
            f"Overlap Context: Calculating map with voxel side length {side_length_overlap:.3f}, completed.",
        )

        if voxel_data_overlap is None or voxel_params_overlap is None:
            raise RuntimeError("Failed to build Overlap voxel map.")
        # Store separately
        voxel_params_to_use = voxel_params_overlap
        voxel_data_to_use = voxel_data_overlap

        # Retrieve data/params for the call
        voxel_ids, voxel_start, voxel_end = voxel_data_to_use
        voxel_origin, voxel_shape, voxel_scale, _ = voxel_params_to_use

        self.adjacency_map = calculate_atom_overlap_adjacency(
            platform=self.platform,
            # num_cuda_threads=self.num_cuda_threads,
            atoms_data=self.atoms_data,
            # Pass chosen voxel args
            voxel_atom_ids=voxel_ids,
            voxel_atom_start_index=voxel_start,
            voxel_atom_end_index=voxel_end,
            voxel_map_origin=voxel_origin,
            voxel_map_shape=voxel_shape,
            voxel_map_scale=voxel_scale,
        )

        vprint(DEBUG, _VERBOSITY, " Finished Overlap Adjacency Map ")

    def run(self, num_cuda_threads, dc):
        """MODIFIED: Calculates context-specific voxel maps before use."""
        if not isinstance(dc, RuntimeContext):
            raise TypeError(
                "dc must be an object of class `foundation.context.RuntimeContext`"
            )
        self.num_cuda_threads = num_cuda_threads
        if self.platform.active == "cuda" and size_gpu is not None:
            cuda.select_device(self.platform.names["cuda"]["selected_id"])
        tic_grdcrg = time.perf_counter()

        (
            num_charged_gridpoints,
            num_positive_charge,
            num_negative_charge,
            self.charged_gridpoints_1d,  # dtype: np.float64 (always to avoid index1d overflow)
        ) = set_grid_charges_sorted_by_index1d(
            is_focusing=self.is_focusing,
            atoms_data=self.atoms_data.astype(np.float64),
            grid_shape=self.grid_shape.astype(np.int64),
        )
        toc_grdcrg = time.perf_counter()
        self.timings["space| set grid charges"] = f"{toc_grdcrg - tic_grdcrg:.3f}"

        if self.enabled_nonpolar:
            tic_adjmap = time.perf_counter()
            self._init_adjacency_map()
            toc_adjmap = time.perf_counter()
            self.timings["space| build atom-overlap adjacency map"] = (
                f"{toc_adjmap - tic_adjmap:.3f}"
            )

        is_gaussian_surf = self.surface_method.int_value in {
            SurfaceMethod.GAUSSIANCUTOFF.int_value,
            SurfaceMethod.GAUSSIAN.int_value,
            SurfaceMethod.GCS.int_value,
        }
        voxel_data_for_gauss = None
        voxel_params_for_gauss = None

        if is_gaussian_surf:
            # --- Gaussian Surface Path ---
            tic_gauss_total = time.perf_counter()
            self.gauss_density_map_1d = np.zeros(
                self.num_grid_points, dtype=delphi_real
            )
            self.gauss_density_map_midpoints_1d = np.zeros(
                self.num_grid_points * 3, dtype=delphi_real
            )
            generate_ion_exclusion_map = False
            generate_ion_exclusion_map = (
                self.surface_method.int_value == SurfaceMethod.GAUSSIANCUTOFF.int_value
            )
            self.ion_exclusion_map_1d = np.zeros(
                self.num_grid_points, dtype=delphi_bool
            )
            grid_spacing_extension_factor = 0.5 * self.grid_spacing * math.sqrt(3)
            self.atom_influence_radius = (
                calc_atom_gaussian_influence_radius(
                    self.probe_radius,
                    self.salt_radius,
                    0.0,
                    self.max_atom_radius,
                    self.atoms_data,
                )
                + grid_spacing_extension_factor
            )
            vprint(
                DEBUG,
                _VERBOSITY,
                f"(Gaussian Influence Radius: {self.atom_influence_radius:.3f})",
            )

            # --- Voxelization for Gaussian Context ---
            side_length_gauss = self.atom_influence_radius
            vprint(
                DEBUG,
                _VERBOSITY,
                f"Gaussian Context: Calculating map with voxel side length {side_length_gauss:.3f}",
            )

            voxel_params_for_gauss, voxel_data_for_gauss, time_elapsed = (
                build_consolidated_atoms_space_voxel_map(
                    side_length_gauss,
                    self.coords_by_axis_min,
                    self.coords_by_axis_max,
                    1.0,
                    0.1,
                    self.num_atoms,
                    self.num_objects,
                    self.num_molecules,
                    self.atoms_data,
                )
            )

            self.atom_voxel_map_params = voxel_params_for_gauss
            self.atom_voxel_map_data = voxel_data_for_gauss
            self.timings["space| build specific voxel map"] = round(
                self.timings.get("space| build specific voxel map", 0) + time_elapsed,
                3,
            )
            if voxel_data_for_gauss is None or voxel_params_for_gauss is None:
                raise RuntimeError("Failed to build Gaussian voxel map.")
            # --- End Gaussian Voxelization ---

            use_voxel_opt = (
                self.platform.active == "cpu" and voxel_data_for_gauss is not None
            )
            if use_voxel_opt:
                vprint(
                    DEBUG, _VERBOSITY, "Calling OPTIMIZED _calc_gaussian_density_map"
                )

            tic_gauss_calc = time.perf_counter()
            self._calc_atoms_voxel_space_aware_gaussian_density_map(
                side_length_gauss_voxel=side_length_gauss,
                generate_ion_exclusion_map=generate_ion_exclusion_map,
            )
            toc_gauss_calc = time.perf_counter()
            opt_tag = (
                "(Optimized)"
                if use_voxel_opt
                else "" if self.platform.active == "cpu" else "(CUDA)"
            )
            self.timings[f"space| calc. gaussian density {opt_tag}"] = (
                f"{toc_gauss_calc - tic_gauss_calc:.3f}"
            )

            # Surface Calculations ...
            voxel_ids, voxel_start, voxel_end = self.atom_voxel_map_data
            voxel_origin, voxel_shape, voxel_scale, _ = self.atom_voxel_map_params

            # Rebuild voxel map before using for surface if surface_cutoff > 0
            if self.surface_offset > 0:
                side_length_gauss_surf = side_length_gauss + self.surface_offset
                voxel_params_for_gauss, voxel_data_for_gauss, time_elapsed = (
                    build_consolidated_atoms_space_voxel_map(
                        side_length_gauss_surf,
                        self.coords_by_axis_min,
                        self.coords_by_axis_max,
                        1.0,
                        0.1,
                        self.num_atoms,
                        self.num_objects,
                        self.num_molecules,
                        self.atoms_data,
                    )
                )
                voxel_ids, voxel_start, voxel_end = voxel_data_for_gauss
                voxel_origin, voxel_shape, voxel_scale, _ = voxel_params_for_gauss

            surf_obj = surface.Surface(
                self.platform,
                self.grid_spacing,
                self.probe_radius,
                self.salt_radius,
                self.gaussian_sigma,
                self.gaussian_exponent,
                self.surface_offset,
                APPROX_ZERO,
                self.grid_shape,
                self.grid_origin,
                self.num_atoms,
                self.num_objects,
                self.num_molecules,
                self.coords_by_axis_min,
                self.coords_by_axis_max,
                self.atoms_data,
                voxel_atom_ids=voxel_ids.astype(delphi_int),
                voxel_atom_start_index=voxel_start.astype(delphi_int),
                voxel_atom_end_index=voxel_end.astype(delphi_int),
                voxel_map_origin=voxel_origin.astype(delphi_real),
                voxel_map_shape=voxel_shape.astype(delphi_int),
                voxel_map_scale=voxel_scale,
            )
            surf_obj.run(
                num_cuda_threads,
                self.surface_method,
                self.surface_density_exponent,
                self.is_surf_midpoints,
                self.gauss_density_map_1d,
                self.gauss_density_map_midpoints_1d,
            )
            self.surface_map_1d = surf_obj.surface_map_1d
            self.grad_surface_map_1d = surf_obj.grad_surface_map_1d
            if self.surface_method.int_value == SurfaceMethod.GCS.int_value:
                if self.verbosity <= DEBUG:
                    print("setting: solute_inside_map_1d")
                self.solute_inside_map_1d = surf_obj.solute_inside_map_1d
                self.solute_outside_map_1d = surf_obj.solute_outside_map_1d
                self.surf_heavyside_map_1d = surf_obj.surf_heavyside_map_1d
            elif self.is_surf_midpoints and self.surface_method.int_value in (
                SurfaceMethod.GAUSSIAN.int_value,
                SurfaceMethod.GAUSSIANCUTOFF.int_value,
            ):
                self.surface_map_midpoints_1d = surf_obj.surface_map_midpoints_1d
            toc_srf = time.perf_counter()
            self.timings["space| calc. surface_map and it's gradient"] = (
                "{:0.3f}".format(toc_srf - toc_grdcrg)
            )

            # ATTENTION: If zeta-surface is enabled `vdw_molecular_surface` must also be
            # calculated to satisfy downstream usage.
            if self.use_zeta_surf:
                self._calculate_vdw_molecular_surface()

        elif self.surface_method.int_value == SurfaceMethod.VDW.int_value:
            self._calculate_vdw_molecular_surface()

        # Finally format (from float kept to allow multiple voxelation timing summation) to string
        self.timings["space| build specific voxel map"] = "{:0.3f}".format(
            self.timings.get("space| build specific voxel map", 0)
        )

    def update_runtime_context(self, ctx):
        if not isinstance(ctx, RuntimeContext):
            raise TypeError(
                "ctx must be an object of class `foundation.context.RuntimeContext`"
            )
        ctx.num_charged_gridpoints = self.num_charged_gridpoints
        ctx.num_positive_charge = self.num_positive_charge
        ctx.num_negative_charge = self.num_negative_charge
        ctx.charged_gridpoints_1d = self.charged_gridpoints_1d
        ctx.gauss_density_map_1d = self.gauss_density_map_1d
        ctx.gauss_density_map_midpoints_1d = self.gauss_density_map_midpoints_1d
        ctx.surface_map_1d = self.surface_map_1d
        ctx.grad_surface_map_1d = self.grad_surface_map_1d
        if self.surface_method.int_value == SurfaceMethod.GCS.int_value:
            ctx.solute_inside_map_1d = self.solute_inside_map_1d
            ctx.solute_outside_map_1d = self.solute_outside_map_1d
            ctx.surf_heavyside_map_1d = self.surf_heavyside_map_1d
        elif (
            self.is_surf_midpoints
            and self.surface_method.int_value == SurfaceMethod.GAUSSIAN.int_value
        ):
            ctx.surface_map_midpoints_1d = self.surface_map_midpoints_1d
        elif self.surface_method.int_value == SurfaceMethod.GAUSSIANCUTOFF.int_value:
            ctx.ion_exclusion_map_1d = self.ion_exclusion_map_1d
        elif self.surface_method.int_value == SurfaceMethod.VDW.int_value:
            ctx.dielectric_boundary_map_1d = self.dielectric_boundary_map_1d
            if self.use_zeta_surf:
                ctx.zeta_surf_grid_coords = self.zeta_surf_grid_coords
                ctx.zeta_surf_grid_indices = self.zeta_surf_grid_indices
                ctx.num_zeta_surf_grid_coords = self.num_zeta_surf_grid_coords
                # Note: always num_zeta_surf_grid_coords and num_zeta_surf_grid_index must be equal.
                ctx.num_zeta_surf_grid_index = self.num_zeta_surf_grid_coords

        # ATTENTION: If zeta-surface is enabled `vdw_molecular_surface` must also be
        # saved to satisfy downstream usage.
        if (
            self.surface_method.int_value != SurfaceMethod.VDW.int_value
            and self.use_zeta_surf
        ):
            ctx.zeta_surf_grid_coords = self.zeta_surf_grid_coords
            ctx.zeta_surf_grid_indices = self.zeta_surf_grid_indices
            ctx.num_zeta_surf_grid_coords = self.num_zeta_surf_grid_coords
            # Note: always num_zeta_surf_grid_coords and num_zeta_surf_grid_index must be equal.
            ctx.num_zeta_surf_grid_index = self.num_zeta_surf_grid_coords
        ctx.epsilon_map_1d = self.epsilon_map_1d
        ctx.epsilon_map_midpoints_1d = self.epsilon_map_midpoints_1d

    def calc_phase_spatial_epsilon_map_midpoints(
        self,
        is_regularized,
        vacuum,
        gaussian_density_cutoff=0.0,
        gaussian_epsilon_cutoff=0.0,
        gaussian_solute_density_threshold=0.02,
        gaussian_eps_maxmin_ratio_threshold=1.02,
    ):
        tic_epscalc = time.perf_counter()
        num_cpu_threads = self.platform.names["cpu"]["num_threads"]
        if (
            self.surface_method.int_value == SurfaceMethod.GAUSSIAN.int_value
            or self.surface_method.int_value == SurfaceMethod.GCS.int_value
        ):
            epsilon_map_1d_local = None
            epsilon_r_map_1d_local = None
            # ATTENTION: GCS, surface_map is known only at grid points, at midpoints is approximated from neighbors.
            # So, is_surf_midpoints_arg must be False to enable approximation at midpoints for GCS.
            is_surf_midpoints_arg = (
                False
                if self.surface_method.int_value == SurfaceMethod.GCS.int_value
                else self.is_surf_midpoints
            )
            (
                epsilon_map_1d_local,
                epsilon_r_map_1d_local,
                self.epsilon_map_midpoints_1d,
            ) = calc_spatial_epsilon_map_midpoints(
                num_cpu_threads=num_cpu_threads,
                is_surf_midpoints=is_surf_midpoints_arg,
                vacuum=vacuum,
                exdi=self.exdi,
                gapdi=self.gapdi,
                indi=self.indi,
                grid_shape=self.grid_shape,
                gauss_density_map_1d=self.gauss_density_map_1d,
                solute_surface_map_1d=self.surface_map_1d,
                gauss_density_map_midpoints_1d=self.gauss_density_map_midpoints_1d,
                surface_map_midpoints_1d=self.surface_map_midpoints_1d,
            )
            if is_regularized:
                self.epsilon_map_1d = epsilon_map_1d_local
            else:
                self.epsilon_map_1d = epsilon_r_map_1d_local
        elif self.surface_method.int_value == SurfaceMethod.GAUSSIANCUTOFF.int_value:
            (
                self.epsilon_map_1d,
                self.epsilon_map_midpoints_1d,
            ) = calc_gaussian_cutoff_spatial_epsilon_map_midpoints(
                num_cpu_threads=num_cpu_threads,
                vaccum=vacuum,
                exdi=self.exdi,
                gapdi=self.gapdi,
                indi=self.indi,
                density_cutoff=gaussian_density_cutoff,
                epsilon_cutoff=gaussian_epsilon_cutoff,
                solute_density_threshold=gaussian_solute_density_threshold,
                eps_maxmin_ratio_threshold=gaussian_eps_maxmin_ratio_threshold,
                grid_shape=self.grid_shape,
                gauss_density_map_1d=self.gauss_density_map_1d,
                gauss_density_map_midpoints_1d=self.gauss_density_map_midpoints_1d,
                ion_exclusion_map_1d=self.ion_exclusion_map_1d,
            )
        elif self.surface_method.int_value == SurfaceMethod.VDW.int_value:
            vprint(DEBUG, _VERBOSITY, "media_epsilon:", self.media_epsilon)
            if self.discrete_epsilon_map_midpoints_1d is None:
                self.discrete_epsilon_map_midpoints_1d = np.zeros(
                    self.num_grid_points * 3, dtype=delphi_real
                )
                # Must be initialized to enable type-deduction by numba
                self.epsilon_map_1d = np.full(
                    self.num_grid_points,
                    fill_value=self.media_epsilon[-1],
                    dtype=delphi_real,
                )
            if self.index_discrete_epsilon_map_1d is None:
                raise RuntimeError("VDW index map not initialized.")
            if self.dielectric_boundary_grids is None:
                raise RuntimeError("VDW boundary grids not available.")
            calculate_vdw_dielectric_map_midpoints(
                vacuum=vacuum,
                epsilon_dim=self.epsilon_dimension,
                media_epsilon=self.media_epsilon,
                index_discrete_epsilon_map_1d=self.index_discrete_epsilon_map_1d,
                discrete_epsilon_map_1d=self.discrete_epsilon_map_midpoints_1d,
            )
            self.epsilon_map_midpoints_1d = self.discrete_epsilon_map_midpoints_1d
        toc_epscalc = time.perf_counter()
        phase = "vacuum" if vacuum else "water"
        self.timings[f"space| calc. epsilon map in {phase}"] = (
            f"{toc_epscalc - tic_epscalc:.3f}"
        )
