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

from numba import set_num_threads, njit, cuda, prange

from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
    vprint,
)

from pydelphi.foundation.enums import Precision, SurfaceMethod
from pydelphi.config.logging_config import INFO, get_effective_verbosity

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

from pydelphi.constants import ConstDelPhiFloats as ConstDelPhi

APPROX_ZERO = ConstDelPhi.ApproxZero.value
GAUSSIAN_INFLUENCE_RADIUS_FACTOR = ConstDelPhi.GaussianInfluenceRadiusFactor.value

if PRECISION.int_value == Precision.SINGLE.int_value:
    from pydelphi.utils.prec.single import *  # May contain other useful functions

    try:
        from pydelphi.utils.cuda.single import *  # May contain other useful functions
    except ImportError:
        pass
        # print("No Cuda")

elif PRECISION.int_value == Precision.DOUBLE.int_value:
    from pydelphi.utils.prec.double import *  # May contain other useful functions

    try:
        from pydelphi.utils.cuda.double import *  # May contain other useful functions
    except ImportError:
        pass
        # print("No Cuda")

use_gaussian_naive = False
if use_gaussian_naive:
    from pydelphi.space.core.gaussian_naive import (
        calc_grad_surface_map_analytical,
        calc_gaussian_like_surface,
    )
else:
    from pydelphi.space.core.gaussian import (
        calc_grad_surface_map_analytical,
        calc_gaussian_like_surface,
    )


@njit(nogil=True, boundscheck=False, parallel=True)
def _cpu_grad_surface_map(
    grid_spacing: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    surface_map_1d: np.ndarray[delphi_real],
    grad_surface_map_1d: np.ndarray[delphi_real],
) -> None:
    # expression: 1 / (2 * h) = (1/2) * (1/h) = 0.5 / h
    reciprocal_interval_width = 0.5 / grid_spacing
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    num_grid_points = grid_shape[0] * x_stride
    num_grid_points_x_3 = 3 * num_grid_points

    last_grid_x, last_grid_y, last_grid_z = (
        grid_shape[0] - 1,
        grid_shape[1] - 1,
        grid_shape[2] - 1,
    )

    for ijk1d in prange(num_grid_points):
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = (ijk1d - i * x_stride) - j * y_stride
        ijk1d_x_3 = 3 * ijk1d
        if (0 < i < last_grid_x) and (0 < j < last_grid_y) and (0 < k < last_grid_z):
            grad_surface_map_1d[ijk1d_x_3] = (
                surface_map_1d[ijk1d + x_stride] - surface_map_1d[ijk1d - x_stride]
            ) * reciprocal_interval_width
            grad_surface_map_1d[ijk1d_x_3 + 1] = (
                surface_map_1d[ijk1d + y_stride] - surface_map_1d[ijk1d - y_stride]
            ) * reciprocal_interval_width
            grad_surface_map_1d[ijk1d_x_3 + 2] = (
                surface_map_1d[ijk1d + 1] - surface_map_1d[ijk1d - 1]
            ) * reciprocal_interval_width
        else:
            grad_surface_map_1d[ijk1d_x_3] = 0.0
            if ijk1d_x_3 + 1 < num_grid_points_x_3:
                grad_surface_map_1d[ijk1d_x_3 + 1] = 0.0
            if ijk1d_x_3 + 2 < num_grid_points_x_3:
                grad_surface_map_1d[ijk1d_x_3 + 2] = 0.0


@cuda.jit(cache=True)
def _cuda_grad_surface_map(
    grid_spacing: delphi_real,
    grid_shape: np.ndarray[delphi_int],
    surface_map_1d: np.ndarray[delphi_real],
    grad_surface_map_1d: np.ndarray[delphi_real],
) -> None:
    # expression: 1 / (2 * h) = (1/2) * (1/h) = 0.5 / h
    reciprocal_interval_width = 0.5 / grid_spacing
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    num_grid_points = grid_shape[0] * x_stride
    num_grid_points_x_3 = 3 * num_grid_points

    last_grid_x, last_grid_y, last_grid_z = (
        grid_shape[0] - 1,
        grid_shape[1] - 1,
        grid_shape[2] - 1,
    )

    ijk1d = cuda.grid(1)
    if ijk1d < num_grid_points:
        i = ijk1d // x_stride
        j = (ijk1d - i * x_stride) // y_stride
        k = (ijk1d - i * x_stride) - j * y_stride
        ijk1d_x_3 = 3 * ijk1d
        if (0 < i < last_grid_x) and (0 < j < last_grid_y) and (0 < k < last_grid_z):
            grad_surface_map_1d[ijk1d_x_3] = (
                surface_map_1d[ijk1d + x_stride] - surface_map_1d[ijk1d - x_stride]
            ) * reciprocal_interval_width
            grad_surface_map_1d[ijk1d_x_3 + 1] = (
                surface_map_1d[ijk1d + y_stride] - surface_map_1d[ijk1d - y_stride]
            ) * reciprocal_interval_width
            grad_surface_map_1d[ijk1d_x_3 + 2] = (
                surface_map_1d[ijk1d + 1] - surface_map_1d[ijk1d - 1]
            ) * reciprocal_interval_width
        else:
            grad_surface_map_1d[ijk1d_x_3] = 0.0
            if ijk1d_x_3 + 1 < num_grid_points_x_3:
                grad_surface_map_1d[ijk1d_x_3 + 1] = 0.0
            if ijk1d_x_3 + 2 < num_grid_points_x_3:
                grad_surface_map_1d[ijk1d_x_3 + 2] = 0.0


class Surface:
    def __init__(
        self,
        platform,
        grid_spacing,
        probe_radius,
        salt_radius,
        gaussian_sigma,
        gaussian_exponent,
        surface_offset,
        approx_zero,
        grid_shape,
        grid_origin,
        num_atoms,
        num_objects,
        num_molecules,
        coords_by_axis_min,
        coords_by_axis_max,
        atoms_data,
        # --- Added Voxel Map Parameters to __init__ ---
        voxel_atom_ids,
        voxel_atom_start_index,
        voxel_atom_end_index,
        voxel_map_origin,
        voxel_map_shape,
        voxel_map_scale,
        # -----------------------------------------------
    ):
        self.platform = platform
        self.grid_spacing = grid_spacing
        self.probe_radius = probe_radius
        self.salt_radius = salt_radius
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_exponent = gaussian_exponent
        self.surface_offset = surface_offset
        self.gcs_tolerance = 0.0025
        self.approx_zero = approx_zero
        self.grid_shape = grid_shape
        self.grid_origin = grid_origin

        self.num_atoms = num_atoms
        self.num_objects = num_objects
        self.num_molecules = num_molecules
        self.coords_by_axis_min = coords_by_axis_min
        self.coords_by_axis_max = coords_by_axis_max
        self.atoms_data = atoms_data

        self.solute_inside_map_1d = None
        self.solute_outside_map_1d = None
        self.surf_heavyside_map_1d = None
        self.surface_map_1d = None
        self.surface_map_midpoints_1d = None
        self.grad_surface_map_1d = None
        self.surface_method = None

        # --- Store Voxel Map Parameters ---
        self.initial_voxel_atom_ids = voxel_atom_ids
        self.initial_voxel_atom_start_index = voxel_atom_start_index
        self.initial_voxel_atom_end_index = voxel_atom_end_index
        self.initial_voxel_map_origin = voxel_map_origin
        self.initial_voxel_map_shape = voxel_map_shape
        self.initial_voxel_map_scale = voxel_map_scale

        # Local object fields
        self.num_grid_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.num_grad_map_points = None
        self.num_cuda_threads = 1024

    def _use_initial_voxel_map(self):
        self._current_voxel_atom_ids = self.initial_voxel_atom_ids
        self._current_voxel_atom_start_index = self.initial_voxel_atom_start_index
        self._current_voxel_atom_end_index = self.initial_voxel_atom_end_index
        self._current_voxel_map_origin = self.initial_voxel_map_origin
        self._current_voxel_map_shape = self.initial_voxel_map_shape
        self._current_voxel_map_scale = self.initial_voxel_map_scale

    def _use_new_voxel_map(self, ids, start_idx, end_idx, origin, shape, scale):
        self._current_voxel_atom_ids = ids
        self._current_voxel_atom_start_index = start_idx
        self._current_voxel_atom_end_index = end_idx
        self._current_voxel_map_origin = origin
        self._current_voxel_map_shape = shape
        self._current_voxel_map_scale = scale

    def _calc_grad_surface_map(
        self,
        grid_spacing,
        grid_shape,
        surface_map_1d,
        grad_surface_map_1d,
    ):
        if self.platform.active == "cpu":
            set_num_threads(self.platform.names["cpu"]["num_threads"])
            _cpu_grad_surface_map(
                grid_spacing, grid_shape, surface_map_1d, grad_surface_map_1d
            )
        elif self.platform.active == "cuda":
            num_blocks = (
                surface_map_1d.size + self.num_cuda_threads - 1
            ) // self.num_cuda_threads
            grid_shape_device = cuda.to_device(self.grid_shape)
            surface_map_1d_device = cuda.to_device(self.surface_map_1d)
            grad_surface_map_1d_device = cuda.to_device(grad_surface_map_1d)
            _cuda_grad_surface_map[num_blocks, self.num_cuda_threads](
                grid_spacing,
                grid_shape_device,
                surface_map_1d_device,
                grad_surface_map_1d_device,
            )
            grad_surface_map_1d_device.copy_to_host(grad_surface_map_1d)
            grid_shape_device = None
            surface_map_1d_device = None
            grad_surface_map_1d_device = None

    def run(
        self,
        num_cuda_threads,
        surface_method,
        surf_den_exponent,
        is_surf_midpoints,
        gauss_density_map_1d,  # Input density map 1D
        gauss_density_map_midpoints_1d,  # Input density map midpoints 1D
    ):
        if self.platform.active == "cuda":
            cuda.select_device(self.platform.names["cuda"]["selected_id"])
        self.num_cuda_threads = num_cuda_threads
        self.surface_method = surface_method
        self.num_grid_points = np.prod(self.grid_shape)
        self.num_grad_map_points = self.num_grid_points * 3

        self.surface_map_1d = np.zeros(self.num_grid_points, dtype=delphi_real)
        self.grad_surface_map_1d = np.zeros(self.num_grad_map_points, dtype=delphi_real)

        # Local variables to hold the density maps used for surface calculation
        density_for_surface_1d = None
        density_for_surface_midpoints_1d = None

        # Variables to hold the voxel map to use for the analytical gradient calculation
        voxel_ids_to_use = None
        voxel_start_idx_to_use = None
        voxel_end_idx_to_use = None
        voxel_origin_to_use = None
        voxel_shape_to_use = None
        voxel_scale_to_use = None

        if self.surface_method.int_value in (
            SurfaceMethod.GCS.int_value,
            SurfaceMethod.GAUSSIAN.int_value,
        ):
            from pydelphi.space.core.gaussian import (
                calc_atom_gaussian_influence_radius,
                calc_gaussian_density_map,
            )

            # Determine the influence radius for density calculation
            max_original_atom_radius = (
                np.max(self.atoms_data[:, ATOMFIELD_RADIUS])
                if self.atoms_data.shape[0] > 0
                else 0.0
            )

            required_influence_radius = calc_atom_gaussian_influence_radius(
                self.probe_radius,
                self.salt_radius,
                self.surface_offset,
                max_original_atom_radius,
                self.atoms_data,
                GAUSSIAN_INFLUENCE_RADIUS_FACTOR,
            )

            # Determine if re-voxelation and density recalculation are needed.
            # Simple logic: Recalculate if surface_offset is non-zero.
            # A more sophisticated check could be added if needed.
            recalculate_density_and_revoxelate = self.surface_offset > 0.0

            if recalculate_density_and_revoxelate:
                from pydelphi.space.core.voxelizer import (
                    build_consolidated_atoms_space_voxel_map,
                )

                vprint(
                    INFO,
                    _VERBOSITY,
                    f"    SURFACE> Surface offset {self.surface_offset} > 0. Recalculating density and re-voxelating...",
                )
                # Re-build the voxel map using the inflated influence radius
                (
                    new_voxel_params,
                    new_voxel_data,
                    time_elapsed,
                ) = build_consolidated_atoms_space_voxel_map(
                    required_influence_radius,
                    self.coords_by_axis_min,
                    self.coords_by_axis_max,
                    1.0,
                    0.1,
                    self.num_atoms,
                    self.num_objects,
                    self.num_molecules,
                    self.atoms_data,
                )
                (
                    new_voxel_map_origin,
                    new_voxel_map_shape,
                    new_voxel_map_scale,
                    new_voxel_map_side,
                ) = new_voxel_params
                (
                    new_voxel_atom_ids,
                    new_voxel_start_index,
                    new_voxel_end_index,
                ) = new_voxel_data

                # Update the variables holding the voxel map to use
                voxel_ids_to_use = new_voxel_atom_ids
                voxel_start_idx_to_use = new_voxel_start_index
                voxel_end_idx_to_use = new_voxel_end_index
                voxel_origin_to_use = new_voxel_map_origin
                voxel_shape_to_use = new_voxel_map_shape
                voxel_scale_to_use = new_voxel_map_scale

                # Create *new* local arrays for the recalculated density
                density_for_surface_1d = np.zeros(
                    self.num_grid_points, dtype=delphi_real
                )
                density_for_surface_midpoints_1d = np.zeros(
                    self.num_grad_map_points, dtype=delphi_real
                )
                ion_exclusion_map_1d_srf = np.zeros(
                    self.num_grid_points, dtype=delphi_bool
                )

                # Call the main density calculation function from PART 1 context
                calc_gaussian_density_map(
                    self.platform,
                    self.num_cuda_threads,
                    False,
                    1.0 / self.grid_spacing,
                    self.gaussian_exponent,
                    GAUSSIAN_INFLUENCE_RADIUS_FACTOR,
                    self.surface_offset,
                    required_influence_radius,
                    self.salt_radius,
                    self.grid_shape,
                    self.grid_origin,
                    self.atoms_data,
                    density_for_surface_1d,  # Populate the *new* local array
                    density_for_surface_midpoints_1d,  # Populate the *new* local array
                    ion_exclusion_map_1d_srf,
                    voxel_ids_to_use,  # Pass the new voxel map
                    voxel_start_idx_to_use,
                    voxel_end_idx_to_use,
                    voxel_origin_to_use,
                    voxel_shape_to_use,
                    voxel_scale_to_use,
                )
            else:
                vprint(
                    INFO,
                    _VERBOSITY,
                    "    SURFACE> Surface offset is zero. Using input density maps and initial voxel map.",
                )
                # If surface_offset is 0, use the input density maps directly
                # These are references to the input arrays, not copies
                density_for_surface_1d = gauss_density_map_1d
                density_for_surface_midpoints_1d = gauss_density_map_midpoints_1d

                # Use the initial voxel map stored in self
                if (
                    self.initial_voxel_atom_ids is None
                    or self.initial_voxel_atom_start_index is None
                    or self.initial_voxel_atom_end_index is None
                    or self.initial_voxel_map_origin is None
                    or self.initial_voxel_map_shape is None
                    or self.initial_voxel_map_scale is None
                ):
                    raise ValueError(
                        "    SURFACE> Initial voxel map data must be provided when surface_offset is zero."
                    )

                voxel_ids_to_use = self.initial_voxel_atom_ids
                voxel_start_idx_to_use = self.initial_voxel_atom_start_index
                voxel_end_idx_to_use = self.initial_voxel_atom_end_index
                voxel_origin_to_use = self.initial_voxel_map_origin
                voxel_shape_to_use = self.initial_voxel_map_shape
                voxel_scale_to_use = self.initial_voxel_map_scale

            if self.surface_method.int_value == SurfaceMethod.GCS.int_value:
                from pydelphi.space.gcs_surface import (
                    generate_gaussian_convolution_surface,
                )

                self.solute_inside_map_1d = np.zeros(
                    self.num_grid_points, dtype=delphi_bool
                )
                self.solute_outside_map_1d = np.zeros(
                    self.num_grid_points, dtype=delphi_bool
                )
                self.surf_heavyside_map_1d = np.zeros(
                    self.num_grid_points, dtype=delphi_real
                )

                generate_gaussian_convolution_surface(
                    self.grid_spacing,
                    self.probe_radius,
                    self.gaussian_sigma,
                    self.gcs_tolerance,
                    self.grid_shape,
                    self.grid_origin,
                    self.atoms_data,
                    self.solute_inside_map_1d,
                    self.solute_outside_map_1d,
                    # self.surf_heavyside_map_1d,
                    self.surface_map_1d,
                    # --- Pass the Voxel Map Parameters determined above ---
                    voxel_ids_to_use,
                    voxel_start_idx_to_use,
                    voxel_end_idx_to_use,
                    voxel_origin_to_use,
                    voxel_shape_to_use,
                    voxel_scale_to_use,
                )
                set_num_threads(self.platform.names["cpu"]["num_threads"])
                self._calc_grad_surface_map(  # Finite difference gradient
                    self.grid_spacing,
                    self.grid_shape,
                    self.surface_map_1d,
                    self.grad_surface_map_1d,
                )
            if self.surface_method.int_value == SurfaceMethod.GAUSSIAN.int_value:
                surf_den_exp_scaled = (1.0 * surf_den_exponent) / self.gaussian_exponent

                # Calculate surface map from density map (no atom iteration)
                calc_gaussian_like_surface(
                    self.platform,
                    self.num_cuda_threads,
                    surf_den_exp_scaled,
                    delphi_real(APPROX_ZERO),
                    density_for_surface_1d,  # Use the appropriate local density map
                    self.surface_map_1d,
                )

                if is_surf_midpoints:
                    self.surface_map_midpoints_1d = np.zeros(
                        self.num_grad_map_points, dtype=delphi_real
                    )
                    # Calculate surface map on midpoints (no atom iteration)
                    calc_gaussian_like_surface(
                        self.platform,
                        self.num_cuda_threads,
                        surf_den_exp_scaled,
                        delphi_real(APPROX_ZERO),
                        density_for_surface_midpoints_1d,  # Use the appropriate local midpoint density map
                        self.surface_map_midpoints_1d,
                    )

                # --- Call the Analytical Gradient Calculation ---
                # This call now uses the local variables pointing to the correct voxel map
                if (
                    voxel_ids_to_use is None
                    or voxel_start_idx_to_use is None
                    or voxel_end_idx_to_use is None
                    or voxel_origin_to_use is None
                    or voxel_shape_to_use is None
                    or voxel_scale_to_use is None
                ):
                    # This check should technically be redundant if the logic above is correct,
                    # but kept for safety.
                    raise ValueError(
                        "Voxel map data for gradient calculation is missing after density handling."
                    )

                calc_grad_surface_map_analytical(
                    self.platform,
                    self.num_cuda_threads,
                    self.gaussian_exponent,
                    self.grid_spacing,
                    surf_den_exp_scaled,
                    self.approx_zero,
                    self.grid_shape,
                    self.grid_origin,
                    self.atoms_data,
                    density_for_surface_1d,
                    self.surface_map_1d,
                    self.grad_surface_map_1d,
                    # --- Pass the Voxel Map Parameters determined above ---
                    voxel_ids_to_use,
                    voxel_start_idx_to_use,
                    voxel_end_idx_to_use,
                    voxel_origin_to_use,
                    voxel_shape_to_use,
                    voxel_scale_to_use,
                )
