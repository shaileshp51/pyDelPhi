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
This module implements the Regularized Poisson-Boltzmann Equation (RPBE) solver,
designed for molecular electrostatics calculations. It extends the standard
Poisson-Boltzmann equation to handle singularities and improve numerical stability,
especially in regions with sharp dielectric interfaces or point charges.

The solver supports both CPU and CUDA (GPU) computation platforms, leveraging
Numba for optimized performance. It includes functionalities for:
- Initializing solver parameters and data structures, adapting to grid dimensions
  and chosen precision (single or double).
- Calculating and assigning charges to grid points, incorporating charge spreading
  techniques (e.g., Gaussian).
- Setting up various boundary conditions, including Coulombic, Dipolar, and Focusing,
  to define the potential at the computational domain's edges.
- Preparing grid-related maps, such as dielectric constant distributions,
  ion exclusion regions, and their gradients, which are crucial for the RPBE formulation.
- Iteratively solving the RPBE using the Successive Over-Relaxation (SOR) method,
  which efficiently converges to the electrostatic potential.
- Computing intermediate values like the Coulombic potential, its gradient, and
  the dot product of the dielectric gradient and Coulombic potential gradient,
  essential for the regularized term.
- Providing detailed timing information and verbosity control for debugging and
  performance monitoring.

The module is integrated with PyDelphi's global runtime configuration to ensure
consistency in precision and platform selection across the application.
"""

import time
import math
import numpy as np

from numba import set_num_threads
from numba import cuda

from pydelphi.foundation.enums import (
    BoundaryCondition,
    VerbosityLevel,
    IonExclusionRegion,
)
from pydelphi.config.global_runtime import (
    delphi_bool,
    delphi_int,
    delphi_real,
    nprint_cpu,
    vprint,
)

from pydelphi.constants import ConstDelPhiFloats as ConstDelPhi

# if (
#     PRECISION.value == Precision.SINGLE.value
#     or PRECISION.value == Precision.MIXED.value
# ):
#     from pydelphi.utils.prec.single import *
#
#     try:
#         from pydelphi.utils.cuda.single import *
#     except ImportError:
#         pass
#         # print("No Cuda")
#
# elif PRECISION.value == Precision.DOUBLE.value:
#     from pydelphi.utils.prec.double import *
#
#     try:
#         from pydelphi.utils.cuda.double import *
#     except ImportError:
#         pass
#         # print("No Cuda")

from pydelphi.config.logging_config import (
    INFO,
    DEBUG,
)

from pydelphi.solver.core import (
    _copy_to_sample,
    _copy_to_full,
    _sum_of_product_sample,
    _calculate_phi_map_sample_rmsd,
)

from pydelphi.solver.shared.sor.base import (
    _prepare_to_init_relaxfactor_phimap,
    _cpu_init_relaxfactor_phimap,
    _cuda_init_relaxfactor_phimap,
    _cpu_iterate_relaxation_factor,
    _cuda_iterate_relaxation_factor,
    _cpu_iterate_block_SOR,
    _cuda_iterate_SOR,
)

from pydelphi.solver.rpb.common_rpb import (
    _cpu_calc_coulomb_map,
    _cuda_calc_coulomb_map,
    _cpu_calc_grad_coulomb_map,
    _cuda_calc_grad_coulomb_map,
    _cpu_grad_epsilon_dot_coulomb_map,
    _cuda_grad_epsilon_dot_coulomb_map,
    _cpu_calc_grad_epsilon_in_map,
    _cuda_calc_grad_epsilon_in_map,
    _cpu_setup_coulombic_boundary_condition,
    _cuda_setup_coulombic_boundary_condition,
    _cpu_setup_dipolar_boundary_condition,
    _cuda_setup_dipolar_boundary_condition,
    _cpu_prepare_charge_neigh_eps_sum_to_iterate,
    _cuda_prepare_charge_neigh_eps_sum_to_iterate,
)

from pydelphi.solver.rpb.sor.helpers import (
    _cpu_helper_calc_spatial_epsilon_map,
    _cuda_helper_calc_spatial_epsilon_map,
)


class RPBESolver:
    def __init__(
        self,
        platform,
        verbosity,
        num_cuda_threads,
        grid_shape,
        coulomb_map_1d=None,
        grad_coulomb_map_1d=None,
        debug=False,
    ):
        self.debug = debug
        self.phase = None
        self.platform = platform
        self.verbosity = verbosity
        self.timings = {}
        self.num_cpu_threads = 1  # default to 1, if not-configured.
        if "cpu" in platform.names and "num_threads" in platform.names["cpu"]:
            self.num_cpu_threads = platform.names["cpu"]["num_threads"]
        # Set the scalar variables used in the class
        self.num_cuda_threads = num_cuda_threads
        self.grid_shape = grid_shape
        self.num_grid_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.num_grad_map_points = self.num_grid_points * 3
        # Allocate and init with zero maps_3d and grad_maps_4d
        self.grad_epsmap_1d = np.zeros(self.num_grad_map_points, dtype=delphi_real)
        self.grad_epsin_map_1d = np.zeros(self.num_grad_map_points, dtype=delphi_real)
        self.eps_dot_coul_map_1d = np.zeros(self.num_grid_points, dtype=delphi_real)
        if not coulomb_map_1d is None:
            self.coulomb_map_1d = coulomb_map_1d
        else:
            self.coulomb_map_1d = np.zeros(self.num_grid_points, dtype=delphi_real)
        if not grad_coulomb_map_1d is None:
            self.grad_coulomb_map_1d = grad_coulomb_map_1d
        else:
            self.grad_coulomb_map_1d = np.zeros(
                self.num_grad_map_points, dtype=delphi_real
            )

    def _prepare_to_iterate(
        self,
        vacuum: delphi_bool,
        non_zero_salt: delphi_bool,
        ion_exclusion_method_int_value: delphi_int,
        exdi: delphi_real,
        ion_radius: delphi_real,
        ions_valance: delphi_real,
        grid_spacing: delphi_real,
        debye_length: delphi_real,
        epkt: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        solute_surface_map_1d: np.ndarray[delphi_real],
        ion_exclusion_map_1d: np.ndarray[delphi_real],
        epsilon_map_1d: np.ndarray[delphi_real],
        epsmap_midpoints_1d: np.ndarray[delphi_real],
        coulomb_map_1d: np.ndarray[delphi_real],
        charge_map_1d: np.ndarray[delphi_real],
        eps_midpoint_neighs_sum_plus_salt_screening_1d: np.ndarray[delphi_real],
        boundary_gridpoints_1d: np.ndarray[delphi_bool],
    ):
        if self.platform.active == "cpu":
            set_num_threads(self.platform.names["cpu"]["num_threads"])
            _cpu_prepare_charge_neigh_eps_sum_to_iterate(
                vacuum=vacuum,
                non_zero_salt=non_zero_salt,
                ion_exclusion_method_int_value=ion_exclusion_method_int_value,
                exdi=exdi,
                ion_radius=ion_radius,
                ions_valance=ions_valance,
                grid_spacing=grid_spacing,
                debye_length=debye_length,
                epkt=epkt,
                grid_shape=grid_shape,
                solute_surface_map_1d=solute_surface_map_1d,
                ion_exclusion_map_1d=ion_exclusion_map_1d,
                epsilon_map_1d=epsilon_map_1d,
                epsmap_midpoints_1d=epsmap_midpoints_1d,
                coulomb_map_1d=coulomb_map_1d,
                charge_map_1d=charge_map_1d,
                eps_midpoint_neighs_sum_plus_salt_screening_1d=eps_midpoint_neighs_sum_plus_salt_screening_1d,
                boundary_gridpoints_1d=boundary_gridpoints_1d,
            )
        elif self.platform.active == "cuda":
            # BEGIN: CUDA call section for function: <<_prepare_to_iterate>>
            num_blocks = (self.num_grid_points + self.num_cuda_threads - 1) // (
                self.num_cuda_threads
            )
            grid_shape_device = cuda.to_device(grid_shape)
            surface_map_1d_device = cuda.to_device(solute_surface_map_1d)
            epsmap_midpoints_1d_device = cuda.to_device(epsmap_midpoints_1d)
            coulomb_map_1d_device = cuda.to_device(self.coulomb_map_1d)
            charge_map_1d_device = cuda.to_device(charge_map_1d)
            eps_nd_midpoint_neighs_sum_1d_device = cuda.device_array_like(
                eps_midpoint_neighs_sum_plus_salt_screening_1d
            )
            boundary_gridpoints_1d_device = cuda.device_array_like(
                boundary_gridpoints_1d
            )
            # CALL: CUDA kernel for the computation
            _cuda_prepare_charge_neigh_eps_sum_to_iterate[
                num_blocks, self.num_cuda_threads
            ](
                vacuum,
                exdi,
                grid_spacing,
                debye_length,
                grid_shape_device,
                surface_map_1d_device,
                epsmap_midpoints_1d_device,
                coulomb_map_1d_device,
                charge_map_1d_device,
                eps_nd_midpoint_neighs_sum_1d_device,
                boundary_gridpoints_1d_device,
            )
            # FETCH RESULTS TO HOST FROM DEVICE
            charge_map_1d_device.copy_to_host(charge_map_1d)
            eps_nd_midpoint_neighs_sum_1d_device.copy_to_host(
                eps_midpoint_neighs_sum_plus_salt_screening_1d
            )
            boundary_gridpoints_1d_device.copy_to_host(boundary_gridpoints_1d)
            # CLEAR: mark CUDA memory for garbage collection
            grid_shape_device = None
            surface_map_1d_device = None
            epsmap_midpoints_1d_device = None
            coulomb_map_1d_device = None
            charge_map_1d_device = None
            eps_nd_midpoint_neighs_sum_1d_device = None
            boundary_gridpoints_1d_device = None
            # END: CUDA call section for function: <<_prepare_to_iterate>>

    def _helper_calc_spatial_epsilon_map(
        self,
        epsout,
        grid_shape,
        surface_map_1d,
        epsmap_gridpoints_1d,
        grad_surface_map_1d,
        grad_epsin_map_1d,
        grad_epsmap_1d,
    ):
        if self.platform.active == "cpu":
            set_num_threads(self.platform.names["cpu"]["num_threads"])
            _cpu_helper_calc_spatial_epsilon_map(
                epsout,
                grid_shape,
                surface_map_1d,
                epsmap_gridpoints_1d,
                grad_surface_map_1d,
                grad_epsin_map_1d,
                grad_epsmap_1d,
            )
        elif self.platform.active == "cuda":
            grid_shape_device = cuda.to_device(grid_shape)
            surface_map_1d_device = cuda.to_device(surface_map_1d)
            epsmap_gridpoints_1d_device = cuda.to_device(epsmap_gridpoints_1d)
            grad_surface_map_1d_device = cuda.to_device(grad_surface_map_1d)
            grad_epsin_map_1d_device = cuda.to_device(grad_epsin_map_1d)
            grad_epsmap_1d_device = cuda.device_array_like(grad_epsmap_1d)

            n_blocks = (
                grad_epsmap_1d.size + self.num_cuda_threads - 1
            ) // self.num_cuda_threads
            _cuda_helper_calc_spatial_epsilon_map[n_blocks, self.num_cuda_threads](
                epsout,
                grid_shape_device,
                surface_map_1d_device,
                epsmap_gridpoints_1d_device,
                grad_surface_map_1d_device,
                grad_epsin_map_1d_device,
                grad_epsmap_1d_device,
            )
            grad_epsmap_1d_device.copy_to_host(grad_epsmap_1d)
            # CLEAR: mark CUDA memory for garbage collection
            grid_shape_device = None
            surface_map_1d_device = None
            epsmap_gridpoints_1d_device = None
            grad_surface_map_1d_device = None
            grad_epsin_map_1d_device = None
            grad_epsmap_1d_device = None

    def _calc_grad_epsilon_in_map(
        self,
        gaussian_exponent,
        grid_spacing,
        diff_gap_indi,
        approx_zero,
        grid_shape,
        grid_origin,
        atoms_data,
        density_gridpoint_map_1d,
        grad_epsin_map_1d,
    ):
        if self.platform.active == "cpu":
            set_num_threads(self.platform.names["cpu"]["num_threads"])
            _cpu_calc_grad_epsilon_in_map(
                gaussian_exponent,
                grid_spacing,
                diff_gap_indi,
                approx_zero,
                grid_shape,
                grid_origin,
                atoms_data,
                density_gridpoint_map_1d,
                grad_epsin_map_1d,
            )
        elif self.platform.active == "cuda":
            # Allocating memory to call the cuda kernel for calculating the coulombic potential
            grid_shape_device = cuda.to_device(grid_shape)
            grid_origin_device = cuda.to_device(grid_origin)
            atoms_data_device = cuda.to_device(atoms_data)
            density_gridpoint_map_1d_device = cuda.to_device(density_gridpoint_map_1d)
            grad_epsin_map_1d_device = cuda.device_array_like(grad_epsin_map_1d)

            n_blocks = (
                self.num_grid_points + self.num_cuda_threads - 1
            ) // self.num_cuda_threads
            _cuda_calc_grad_epsilon_in_map[n_blocks, self.num_cuda_threads](
                gaussian_exponent,
                grid_spacing,
                diff_gap_indi,
                approx_zero,
                grid_shape_device,
                grid_origin_device,
                atoms_data_device,
                density_gridpoint_map_1d_device,
                grad_epsin_map_1d_device,
            )
            grad_epsin_map_1d_device.copy_to_host(grad_epsin_map_1d)
            # CLEAR: mark CUDA memory for garbage collection
            grid_shape_device = None
            grid_origin_device = None
            atoms_data_device = None
            density_gridpoint_map_1d_device = None
            grad_epsin_map_1d_device = None

    def _calc_grad_spatial_epsilon_map(
        self,
        gaussian_exponent: delphi_int,
        scale: delphi_real,
        epsout: delphi_real,
        gapdi: delphi_real,
        indi: delphi_real,
        approx_zero: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        grid_origin: np.ndarray[delphi_real],
        atoms_data: np.ndarray[delphi_real],
        density_gridpoint_map_1d: np.ndarray[delphi_real],
        epsmap_gridpoints_1d: np.ndarray[delphi_real],
        surface_map_1d: np.ndarray[delphi_real],
        grad_surface_map_1d: np.ndarray[delphi_real],
        grad_epsmap_1d: np.ndarray[delphi_real],
        grad_epsin_map_1d: np.ndarray[delphi_real],
    ) -> None:
        grid_spacing = 1 / scale
        diff_gap_indi = gapdi - indi

        self._calc_grad_epsilon_in_map(
            gaussian_exponent,
            grid_spacing,
            diff_gap_indi,
            approx_zero,
            grid_shape,
            grid_origin,
            atoms_data,
            density_gridpoint_map_1d,
            grad_epsin_map_1d,
        )

        self._helper_calc_spatial_epsilon_map(
            epsout,
            grid_shape,
            surface_map_1d,
            epsmap_gridpoints_1d,
            grad_surface_map_1d,
            grad_epsin_map_1d,
            grad_epsmap_1d,
        )

    def _calc_coulomb_map(
        self,
        grid_spacing: delphi_real,
        indi_scaled: delphi_real,
        approx_zero: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        grid_origin: np.ndarray[delphi_real],
        atoms_data: np.ndarray[delphi_real],
        coulomb_map_1d: np.ndarray[delphi_real],
    ) -> None:
        if self.platform.active == "cpu":
            set_num_threads(self.platform.names["cpu"]["num_threads"])
            _cpu_calc_coulomb_map(
                grid_spacing,
                indi_scaled,
                approx_zero,
                grid_shape,
                grid_origin,
                atoms_data,
                coulomb_map_1d,
            )
        elif self.platform.active == "cuda":
            n_blocks = (
                self.num_grid_points + self.num_cuda_threads - 1
            ) // self.num_cuda_threads
            # BEGIN: CUDA call section for function: <<_cuda_calc_coulomb_map>>
            grid_shape_device = cuda.to_device(grid_shape)
            grid_origin_device = cuda.to_device(grid_origin)
            atoms_data_device = cuda.to_device(atoms_data)
            coulomb_map_1d_device = cuda.device_array_like(coulomb_map_1d)
            # CALL: CUDA kernel for the computation
            _cuda_calc_coulomb_map[n_blocks, self.num_cuda_threads](
                grid_spacing,
                indi_scaled,
                approx_zero,
                grid_shape_device,
                grid_origin_device,
                atoms_data_device,
                coulomb_map_1d_device,
            )
            # FETCH RESULTS TO HOST FROM DEVICE
            coulomb_map_1d_device.copy_to_host(coulomb_map_1d)

            vprint(
                DEBUG,
                self.verbosity,
                "_cuda_calc_coulomb_map>> coulomb_map_1d.shape=",
                coulomb_map_1d.shape,
            )
            # CLEAR: mark CUDA memory for garbage collection
            grid_shape_device = None
            grid_origin_device = None
            atoms_data_device = None
            coulomb_map_1d_device = None
            # END: CUDA call section for function: <<_cuda_calc_coulomb_map>>

    def _calc_grad_coulomb_map(
        self,
        grid_spacing: delphi_real,
        indi_scaled: delphi_real,
        approx_zero: delphi_real,
        grid_shape: np.ndarray[delphi_real],
        grid_origin: np.ndarray[delphi_real],
        atoms_data: np.ndarray[delphi_real],
        grad_coulomb_map_1d: np.ndarray[delphi_real],
    ) -> None:
        if self.platform.active == "cpu":
            set_num_threads(self.platform.names["cpu"]["num_threads"])
            _cpu_calc_grad_coulomb_map(
                grid_spacing,
                indi_scaled,
                approx_zero,
                grid_shape,
                grid_origin,
                atoms_data,
                grad_coulomb_map_1d,
            )
        elif self.platform.active == "cuda":
            # BEGIN: CUDA call section for function: <<_cuda_calc_grad_coulomb_map>>
            n_blocks = (
                self.num_grid_points + self.num_cuda_threads - 1
            ) // self.num_cuda_threads
            grid_shape_device = cuda.to_device(grid_shape)
            grid_origin_device = cuda.to_device(grid_origin)
            atoms_data_device = cuda.to_device(atoms_data)
            # coulomb_map_1d_device = cuda.to_device(self.coulomb_map_1d)
            grad_coulomb_map_1d_device = cuda.device_array_like(grad_coulomb_map_1d)
            # CALL: CUDA kernel for the computation
            _cuda_calc_grad_coulomb_map[n_blocks, self.num_cuda_threads](
                grid_spacing,
                indi_scaled,
                approx_zero,
                grid_shape_device,
                grid_origin_device,
                atoms_data_device,
                grad_coulomb_map_1d_device,
            )
            # FETCH RESULTS TO HOST FROM DEVICE
            grad_coulomb_map_1d_device.copy_to_host(grad_coulomb_map_1d)
            # CLEAR: mark CUDA memory for garbage collection
            grid_shape_device = None
            grid_origin_device = None
            atoms_data_device = None
            grad_coulomb_map_1d_device = None
            # END: CUDA call section for function: <<_cuda_calc_grad_coulomb_map>>

    def _grad_epsilon_dot_coulomb_map(
        self,
        grad_epsmap_1d: np.ndarray[delphi_real],
        grad_coulomb_map_1d: np.ndarray[delphi_real],
        eps_dot_coul_map_1d: np.ndarray[delphi_real],
    ) -> None:
        if self.platform.active == "cpu":
            set_num_threads(self.platform.names["cpu"]["num_threads"])
            _cpu_grad_epsilon_dot_coulomb_map(
                grad_epsmap_1d,
                grad_coulomb_map_1d,
                eps_dot_coul_map_1d,
            )
        elif self.platform.active == "cuda":
            # BEGIN: CUDA call section for function: <<_cuda_grad_epsilon_dot_coulomb_map>>
            n_blocks = (
                self.num_grid_points + self.num_cuda_threads - 1
            ) // self.num_cuda_threads
            grad_epsmap_1d_device = cuda.to_device(grad_epsmap_1d)
            grad_coulomb_map_1d_device = cuda.to_device(grad_coulomb_map_1d)
            eps_dot_coul_map_1d_device = cuda.device_array_like(eps_dot_coul_map_1d)
            # CALL: CUDA kernel for the computation
            _cuda_grad_epsilon_dot_coulomb_map[n_blocks, self.num_cuda_threads](
                grad_epsmap_1d_device,
                grad_coulomb_map_1d_device,
                eps_dot_coul_map_1d_device,
            )
            # FETCH RESULTS TO HOST FROM DEVICE
            eps_dot_coul_map_1d_device.copy_to_host(eps_dot_coul_map_1d)
            # CLEAR: mark CUDA memory for garbage collection
            grad_epsmap_1d_device = None
            grad_coulomb_map_1d_device = None
            eps_dot_coul_map_1d_device = None
            # END: CUDA call section for function: <<_cuda_grad_epsilon_dot_coulomb_map>>

    def _set_gridpoint_charges(
        self,
        vacuum: delphi_bool,
        gaussian_exponent: delphi_int,
        verbosity: VerbosityLevel,
        scale: delphi_real,
        exdi: delphi_real,
        gapdi: delphi_real,
        indi: delphi_real,
        epkt: delphi_real,
        approx_zero: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        grid_origin: np.ndarray[delphi_real],
        atoms_data: np.ndarray[delphi_real],
        density_gridpoint_map_1d: np.ndarray[delphi_real],
        epsmap_gridpoint_1d: np.ndarray[delphi_real],
        solute_surface_map_1d: np.ndarray[delphi_real],
        grad_surface_map_1d: np.ndarray[delphi_real],
    ) -> None:
        epsout = 1 if vacuum else exdi
        grid_spacing = 1 / scale
        indi_scaled = indi / epkt
        num_grid_points = np.prod(grid_shape)

        tic_gradeps = time.perf_counter()
        n_blocks = (
            num_grid_points + self.num_cuda_threads - 1
        ) // self.num_cuda_threads
        self._calc_grad_spatial_epsilon_map(
            gaussian_exponent=gaussian_exponent,
            scale=scale,
            epsout=epsout,
            gapdi=gapdi,
            indi=indi,
            approx_zero=approx_zero,
            grid_shape=grid_shape,
            grid_origin=grid_origin,
            atoms_data=atoms_data,
            density_gridpoint_map_1d=density_gridpoint_map_1d,
            epsmap_gridpoints_1d=epsmap_gridpoint_1d,
            surface_map_1d=solute_surface_map_1d,
            grad_surface_map_1d=grad_surface_map_1d,
            grad_epsmap_1d=self.grad_epsmap_1d,
            grad_epsin_map_1d=self.grad_epsin_map_1d,
        )
        toc_gradeps = time.perf_counter()
        self.timings[f"rpbe, {self.phase}| calc. gradeps map"] = "{:0.3f}".format(
            toc_gradeps - tic_gradeps
        )

        if self.debug:
            np.save(
                f"new_crgsrc_density_gridpoint_map_1d_vacuum-{vacuum}.npy",
                density_gridpoint_map_1d,
            )
            np.save(
                f"new_crgsrc_epsmap_gridpoints_1d_vacuum-{vacuum}.npy",
                epsmap_gridpoint_1d,
            )
            np.save(
                f"new_crgsrc_solute_surface_map_1d_vacuum-{vacuum}.npy",
                solute_surface_map_1d,
            )
            np.save(
                f"new_crgsrc_grad_surface_map_1d_vacuum-{vacuum}.npy",
                grad_surface_map_1d,
            )
            np.save(
                f"new_crgsrc_grad_epsmap_1d_vacuum-{vacuum}.npy", self.grad_epsmap_1d
            )
            np.save(
                f"new_crgsrc_grad_epsin_map_1d_vacuum-{vacuum}.npy",
                self.grad_epsin_map_1d,
            )
        if vacuum:
            self._calc_coulomb_map(
                grid_spacing,
                indi_scaled,
                approx_zero,
                grid_shape,
                grid_origin,
                atoms_data,
                self.coulomb_map_1d,
            )
            if self.debug:
                np.save(
                    f"new_crgsrc_coulomb_map_1d_vacuum-{vacuum}.npy",
                    self.coulomb_map_1d,
                )
            toc_coul = time.perf_counter()
            self.timings[f"rpbe, {self.phase}| calc. coulombic map"] = "{:0.3f}".format(
                toc_coul - toc_gradeps
            )
            self._calc_grad_coulomb_map(
                grid_spacing,
                indi_scaled,
                approx_zero,
                grid_shape,
                grid_origin,
                atoms_data,
                self.grad_coulomb_map_1d,
            )
            if self.debug:
                np.save(
                    f"new_crgsrc_grad_coulomb_map_1d_vacuum-{vacuum}.npy",
                    self.grad_coulomb_map_1d,
                )
            toc_gradcoul = time.perf_counter()
            self.timings[f"rpbe, {self.phase}| calc. grad-coulombic map"] = (
                "{:0.3f}".format(toc_gradcoul - toc_coul)
            )
        toc_gradcoul = time.perf_counter()
        self._grad_epsilon_dot_coulomb_map(
            self.grad_epsmap_1d,
            self.grad_coulomb_map_1d,
            self.eps_dot_coul_map_1d,
        )
        if self.debug:
            np.save(
                f"new_crgsrc_eps_dot_coul_map_1d_vacuum-{vacuum}.npy",
                self.eps_dot_coul_map_1d,
            )
        toc_epsdotphi = time.perf_counter()
        self.timings[f"rpbe, {self.phase}| calc. grad-eps.grad-coul map"] = (
            "{:0.3f}".format(toc_epsdotphi - toc_gradcoul)
        )

    def _calc_relaxation_factor(
        self,
        itr_block_size: delphi_int,
        grid_shape: np.ndarray[delphi_int],
        periodic_boundary_xyz: np.ndarray[delphi_bool],
        epsmap_midpoints_1d: np.ndarray[delphi_real],
        eps_nd_midpoint_neighs_sum_1d: np.ndarray[delphi_real],
        boundary_gridpoints_1d: np.ndarray[delphi_bool],
    ) -> delphi_real:
        """Calculates the spectral radius for SOR iteration.

        Estimates the optimal spectral radius using a power iteration method with a sine function
        as the initial guess for the eigenvector corresponding to the largest eigenvalue.

        Args:
            itr_block_size (delphi_int): Block size for iteration.
            grid_shape (np.ndarray[delphi_int]): Shape of the grid.
            periodic_boundary_xyz (np.ndarray[delphi_bool]): Periodic boundary condition flags.
            epsmap_midpoints_1d (np.ndarray[delphi_real]): 1D midpoint dielectric map.
            eps_nd_midpoint_neighs_sum_1d (np.ndarray[delphi_real]): 1D array of neighbor epsilon sums.
            boundary_gridpoints_1d (np.ndarray[delphi_bool]): 1D boundary points array.

        Returns:
            delphi_real: The calculated spectral radius.
        """
        # Initialize sine values and phimap arrays for relaxation computation
        sin_values_x = np.zeros(grid_shape[0], dtype=delphi_real)
        sin_values_y = np.zeros(grid_shape[1], dtype=delphi_real)
        sin_values_z = np.zeros(grid_shape[2], dtype=delphi_real)
        phimap_current_1d = np.zeros(self.num_grid_points, dtype=delphi_real)

        # Prepare initial sine-based phimap values
        _prepare_to_init_relaxfactor_phimap(
            grid_shape, periodic_boundary_xyz, sin_values_x, sin_values_y, sin_values_z
        )

        # Map sine values to phimap_current_1d based on execution platform
        if self.platform.active == "cpu":
            _cpu_init_relaxfactor_phimap(
                grid_shape, sin_values_x, sin_values_y, sin_values_z, phimap_current_1d
            )
        elif self.platform.active == "cuda":
            # Transfer necessary data to CUDA device
            grid_shape_device = cuda.to_device(grid_shape)
            sn1_device = cuda.to_device(sin_values_x)
            sn2_device = cuda.to_device(sin_values_y)
            sn3_device = cuda.to_device(sin_values_z)
            phimap_current_1d_device = cuda.to_device(phimap_current_1d)

            # Compute required number of CUDA blocks
            num_blocks = (
                self.num_grid_points + self.num_cuda_threads - 1
            ) // self.num_cuda_threads

            # Initialize phimap values on GPU
            _cuda_init_relaxfactor_phimap[num_blocks, self.num_cuda_threads](
                grid_shape_device,
                sn1_device,
                sn2_device,
                sn3_device,
                phimap_current_1d_device,
            )

            # Copy results back to host
            phimap_current_1d_device.copy_to_host(phimap_current_1d)

        # Split phimap into even and odd indexed elements for efficient updates
        self.num_grid_points_half = (self.num_grid_points + 1) // 2
        phimap_even_1d = np.zeros(self.num_grid_points_half, dtype=delphi_real)
        phimap_odds_1d = np.zeros(self.num_grid_points_half, dtype=delphi_real)

        # Extract even and odd indexed elements for processing
        _copy_to_sample(phimap_even_1d, phimap_current_1d, 0, 2)
        _copy_to_sample(phimap_odds_1d, phimap_current_1d, 1, 2)

        if self.platform.active == "cpu":
            # Perform block iterations for CPU execution
            # Two separate calls are made: one with itr_block_size - 1 iterations, and another single iteration.
            # This enables computing the RMSD between the last and its previous iteration.
            if itr_block_size > 1:
                _cpu_iterate_relaxation_factor(
                    itr_block_size - 1,
                    grid_shape,
                    phimap_odds_1d,
                    phimap_even_1d,
                    epsmap_midpoints_1d,
                    eps_nd_midpoint_neighs_sum_1d,
                    boundary_gridpoints_1d,
                )
                _copy_to_full(phimap_current_1d, phimap_odds_1d, 1, 2)
            _cpu_iterate_relaxation_factor(
                1,
                grid_shape,
                phimap_odds_1d,
                phimap_even_1d,
                epsmap_midpoints_1d,
                eps_nd_midpoint_neighs_sum_1d,
                boundary_gridpoints_1d,
            )
        elif self.platform.active == "cuda":
            num_blocks = (
                self.num_grid_points_half + self.num_cuda_threads - 1
            ) // self.num_cuda_threads
            grid_shape_device = cuda.to_device(grid_shape)
            epsmap_midpoints_1d_device = cuda.to_device(epsmap_midpoints_1d)
            eps_nd_midpoint_neighs_sum_1d_device = cuda.to_device(
                eps_nd_midpoint_neighs_sum_1d
            )
            boundary_gridpoints_1d_device = cuda.to_device(boundary_gridpoints_1d)
            phimap_even_1d_device = cuda.to_device(phimap_even_1d)
            phimap_odds_1d_device = cuda.to_device(phimap_odds_1d)

            for itrid in range(1, itr_block_size + 1):
                for even_odd in [0, 1]:
                    _cuda_iterate_relaxation_factor[num_blocks, self.num_cuda_threads](
                        even_odd,
                        grid_shape_device,
                        (
                            phimap_odds_1d_device
                            if even_odd == 0
                            else phimap_even_1d_device
                        ),
                        (
                            phimap_even_1d_device
                            if even_odd == 0
                            else phimap_odds_1d_device
                        ),
                        epsmap_midpoints_1d_device,
                        eps_nd_midpoint_neighs_sum_1d_device,
                        boundary_gridpoints_1d_device,
                    )
                if itrid == itr_block_size - 1:
                    phimap_odds_1d_device.copy_to_host(phimap_odds_1d)
                    _copy_to_full(phimap_current_1d, phimap_odds_1d, 1, 2)

            # After the full block_size iterations, both odd and even phimap arrays are copied to host
            # to enable RMSD calculation.
            phimap_odds_1d_device.copy_to_host(phimap_odds_1d)
            phimap_even_1d_device.copy_to_host(phimap_even_1d)

        temp = _sum_of_product_sample(
            phimap_current_1d, phimap_odds_1d, 1, 2, self.num_grid_points
        )
        spectral_radius = 2.0 * temp

        if spectral_radius > 1.0:
            spectral_radius = 1.0

        return spectral_radius

    def run(
        self,
        vacuum: delphi_bool,
        non_zero_salt: delphi_bool,
        bound_cond: BoundaryCondition,
        ion_exclusion_method: IonExclusionRegion,
        gaussian_exponent: delphi_int,
        itr_block_size: delphi_int,
        max_linear_iters: delphi_int,
        scale: delphi_real,
        exdi: delphi_real,
        gapdi: delphi_real,
        indi: delphi_real,
        debye_length: delphi_real,
        total_pve_charge: delphi_real,
        total_nve_charge: delphi_real,
        max_rms: delphi_real,
        max_dphi: delphi_real,
        check_dphi: delphi_bool,
        epkt: delphi_real,
        approx_zero: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        grid_origin: np.ndarray[delphi_real],
        atoms_data: np.ndarray[delphi_real],
        density_map_1d: np.ndarray[delphi_real],
        solute_surface_map_1d: np.ndarray[delphi_real],
        ion_exclusion_map_1d: np.ndarray[delphi_real],
        epsilon_map_1d: np.ndarray[delphi_real],
        epsmap_midpoints_1d: np.ndarray[delphi_real],
        centroid_pve_charge: np.ndarray[delphi_real],
        centroid_nve_charge: np.ndarray[delphi_real],
        grad_surface_map_1d: np.ndarray[delphi_real],
    ) -> np.ndarray[delphi_real]:
        if self.platform.active == "cuda":
            cuda.select_device(self.platform.names["cuda"]["selected_id"])
        self.phase = "vacuum" if vacuum else "water"
        grid_spacing = 1.0 / scale
        grid_spacing_square = grid_spacing**2
        kappa_square = exdi / debye_length**2
        kappa_x_grid_spacing_wholesquare = kappa_square * grid_spacing_square

        tic_gchrg = time.perf_counter()
        self._set_gridpoint_charges(
            vacuum=vacuum,
            gaussian_exponent=gaussian_exponent,
            verbosity=self.verbosity,
            scale=scale,
            exdi=exdi,
            gapdi=gapdi,
            indi=indi,
            epkt=epkt,
            approx_zero=approx_zero,
            grid_shape=grid_shape,
            grid_origin=grid_origin,
            atoms_data=atoms_data,
            density_gridpoint_map_1d=density_map_1d,
            epsmap_gridpoint_1d=epsilon_map_1d,
            solute_surface_map_1d=solute_surface_map_1d,
            grad_surface_map_1d=grad_surface_map_1d,
        )
        toc_gcrg = time.perf_counter()
        self.timings[f"rpbe, {self.phase}| calc. charge source"] = "{:0.3f}".format(
            toc_gcrg - tic_gchrg
        )
        tic_bndcon = time.perf_counter()
        phimap_current_1d = np.zeros(self.num_grid_points, dtype=delphi_real)
        self._setup_boundary_condition(
            vacuum=vacuum,
            bound_cond=bound_cond,
            grid_spacing=1 / scale,
            exdi=exdi,
            indi=indi,
            total_nve_charge=total_pve_charge,
            total_pve_charge=total_nve_charge,
            debye_length=debye_length,
            epkt=epkt,
            grid_shape=grid_shape,
            grid_origin=grid_origin,
            atoms_data=atoms_data,
            coulomb_map_1d=self.coulomb_map_1d,
            centroid_pve_charge=centroid_pve_charge,
            centroid_nve_charge=centroid_nve_charge,
            phimap_1d=phimap_current_1d,
        )

        toc_bndcon = time.perf_counter()
        self.timings[f"rpbe, {self.phase}| set boundary condition"] = "{:0.3f}".format(
            toc_bndcon - tic_bndcon
        )

        vprint(
            DEBUG,
            self.verbosity,
            "phimap_current_1d after bc:",
            phimap_current_1d[:100],
        )
        eps_nd_midpoint_neighs_sum_1d = np.empty(
            grid_shape[0] * grid_shape[1] * grid_shape[2], dtype=delphi_real
        )
        boundary_gridpoints_1d = np.empty(
            grid_shape[0] * grid_shape[1] * grid_shape[2], dtype=delphi_bool
        )
        charge_map_1d = np.copy(self.eps_dot_coul_map_1d)
        self._prepare_to_iterate(
            vacuum=vacuum,
            non_zero_salt=non_zero_salt,
            ion_exclusion_method_int_value=ion_exclusion_method.int_value,
            exdi=exdi,
            ion_radius=2.0,
            ions_valance=1.0,
            grid_spacing=grid_spacing,
            debye_length=debye_length,
            epkt=epkt,
            grid_shape=grid_shape,
            solute_surface_map_1d=solute_surface_map_1d,
            ion_exclusion_map_1d=ion_exclusion_map_1d,
            epsilon_map_1d=epsilon_map_1d,
            epsmap_midpoints_1d=epsmap_midpoints_1d,
            coulomb_map_1d=self.coulomb_map_1d,
            charge_map_1d=charge_map_1d,
            eps_midpoint_neighs_sum_plus_salt_screening_1d=eps_nd_midpoint_neighs_sum_1d,
            boundary_gridpoints_1d=boundary_gridpoints_1d,
        )
        toc_prepitr = time.perf_counter()
        self.timings[f"rpbe, {self.phase}| prepare for iteration"] = "{:0.3f}".format(
            toc_prepitr - toc_bndcon
        )
        if vacuum == False and non_zero_salt:
            vprint(
                DEBUG,
                self.verbosity,
                f"     <<RPBE>> kappa_square={kappa_square:0.6f}, kappa_sq_times_h_sq={kappa_x_grid_spacing_wholesquare:0.6f}",
            )

        self.num_grid_points_half = (self.num_grid_points + 1) // 2
        phimap_even_1d = np.zeros(self.num_grid_points_half, dtype=delphi_real)
        phimap_odds_1d = np.zeros(self.num_grid_points_half, dtype=delphi_real)
        _copy_to_sample(phimap_even_1d, phimap_current_1d, 0, 2)
        _copy_to_sample(phimap_odds_1d, phimap_current_1d, 1, 2)

        relax_factor = self._calc_relaxation_factor(
            3,
            grid_shape,
            (0, 0, 0),
            epsmap_midpoints_1d,
            eps_nd_midpoint_neighs_sum_1d,
            boundary_gridpoints_1d,
        )

        num_iters_estimated = (int)(7.8 / math.log(1.0 + math.sqrt(1 - relax_factor)))
        # print("Print estimated number of iterations: ", num_iters_estimated)
        omega = 1 - (2.0 / (1.0 + math.sqrt(1.0 - relax_factor)))
        print(
            f"    RPBE> Relaxation-factor = {relax_factor:10.6f} and SOR-omega = {omega:10.6f}"
        )

        toc_calrelpar = time.perf_counter()
        self.timings[f"rpbe, {self.phase}| calc. relaxation factor"] = "{:0.3f}".format(
            toc_calrelpar - toc_prepitr
        )

        if self.platform.active == "cuda":
            # Number of blocks to cover the entire vector depending on its length
            n_blocks = (self.num_grid_points_half + self.num_cuda_threads - 1) // (
                self.num_cuda_threads
            )
            # BEGIN: send common data to CUDA for all the iterations
            grid_shape_device = cuda.to_device(grid_shape)
            epsmap_midpoints_1d_device = cuda.to_device(epsmap_midpoints_1d)
            eps_nd_midpoint_neighs_sum_1d_device = cuda.to_device(
                eps_nd_midpoint_neighs_sum_1d
            )
            boundary_gridpoints_1d_device = cuda.to_device(boundary_gridpoints_1d)
            charge_map_1d_device = cuda.to_device(charge_map_1d)
            # END: send common data to CUDA for all the iterations

        do_iterate, itr_num = True, 0
        while do_iterate:
            tic_itr = time.perf_counter()
            if itr_num == 0:
                vprint(
                    INFO,
                    self.verbosity,
                    f"    RPBE> | #Iteration |    RMSD    |  Max(dPhi) | Time (seconds) |",
                )
            if self.platform.active == "cuda":
                # BEGIN: send data to CUDA which is updated every iterations
                # and do the iteration.
                phimap_even_1d_device = cuda.to_device(phimap_even_1d)
                phimap_odds_1d_device = cuda.to_device(phimap_odds_1d)
                for itrid in range(itr_num + 1, itr_num + itr_block_size + 1):
                    for even_odd in np.array([0, 1], dtype=delphi_int):
                        if even_odd == 0:
                            # Iterate for gridpoints where indices_sum is even
                            _cuda_iterate_SOR[n_blocks, self.num_cuda_threads](
                                even_odd,
                                omega,
                                approx_zero,
                                grid_shape_device,
                                phimap_odds_1d_device,
                                phimap_even_1d_device,
                                epsmap_midpoints_1d_device,
                                eps_nd_midpoint_neighs_sum_1d_device,
                                boundary_gridpoints_1d_device,
                                charge_map_1d_device,
                            )
                        elif even_odd == 1:
                            _cuda_iterate_SOR[n_blocks, self.num_cuda_threads](
                                even_odd,
                                omega,
                                approx_zero,
                                grid_shape_device,
                                phimap_even_1d_device,
                                phimap_odds_1d_device,
                                epsmap_midpoints_1d_device,
                                eps_nd_midpoint_neighs_sum_1d_device,
                                boundary_gridpoints_1d_device,
                                charge_map_1d_device,
                            )
                            # END: CUDA call every iteration
                        if itrid == itr_num + itr_block_size - 1:
                            phimap_odds_1d_device.copy_to_host(phimap_odds_1d)
                            _copy_to_full(phimap_current_1d, phimap_odds_1d, 1, 2)
                # FETCH RESULTS TO HOST FROM DEVICE
                phimap_odds_1d_device.copy_to_host(phimap_odds_1d)
                phimap_even_1d_device.copy_to_host(phimap_even_1d)
                # CLEAR: mark CUDA memory for this iteration_block for garbage collection
                phimap_odds_1d_device = None
                phimap_even_1d_device = None
            elif self.platform.active == "cpu":
                set_num_threads(self.num_cpu_threads)
                _cpu_iterate_block_SOR(
                    itr_num,
                    itr_block_size - 1,
                    omega,
                    approx_zero,
                    grid_shape,
                    phimap_odds_1d,
                    phimap_even_1d,
                    epsmap_midpoints_1d,
                    eps_nd_midpoint_neighs_sum_1d,
                    boundary_gridpoints_1d,
                    charge_map_1d,
                )
                _copy_to_full(phimap_current_1d, phimap_odds_1d, 1, 2)
                _cpu_iterate_block_SOR(
                    itr_num + itr_block_size - 1,
                    1,
                    omega,
                    approx_zero,
                    grid_shape,
                    phimap_odds_1d,
                    phimap_even_1d,
                    epsmap_midpoints_1d,
                    eps_nd_midpoint_neighs_sum_1d,
                    boundary_gridpoints_1d,
                    charge_map_1d,
                )

            itr_num += itr_block_size
            if itr_num % itr_block_size == 0:
                rmsd, max_delta_phi = _calculate_phi_map_sample_rmsd(
                    phimap_current_1d,
                    phimap_odds_1d,
                    offset=1,
                    stride=2,
                    num_cpu_threads=self.num_cpu_threads,
                    dtype=delphi_real,
                )
                max_delta_phi = abs(max_delta_phi)
                if itr_num >= max_linear_iters:
                    do_iterate = False
                elif (not check_dphi) and rmsd < max_rms:
                    do_iterate = False
                elif check_dphi and max_delta_phi < max_dphi:
                    do_iterate = False
                toc_itr = time.perf_counter()

                vprint(
                    INFO,
                    self.verbosity,
                    f"    RPBE> | {itr_num:>10d} | {rmsd:>9.04e} | {max_delta_phi:>9.04e} | {toc_itr - tic_itr:14.06f} |",
                )
                _copy_to_full(phimap_current_1d, phimap_even_1d, 0, 2)
                _copy_to_full(phimap_current_1d, phimap_odds_1d, 1, 2)
        if self.platform.active == "cuda":
            # BEGIN: clear the CUDA memory after the iterations
            grid_shape_device = None
            epsmap_midpoints_1d_device = None
            eps_nd_midpoint_neighs_sum_1d_device = None
            boundary_gridpoints_1d_device = None
            charge_map_1d_device = None
            # END: clear the cuda memory after the iterations
        phimap_current_1d = phimap_current_1d.reshape(grid_shape)
        return phimap_current_1d

    def _setup_boundary_condition(
        self,
        vacuum: np.bool_,
        bound_cond: BoundaryCondition,
        grid_spacing: delphi_real,
        exdi: delphi_real,
        indi: delphi_real,
        total_pve_charge: delphi_real,
        total_nve_charge: delphi_real,
        debye_length: delphi_real,
        epkt: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        grid_origin: np.ndarray[delphi_real],
        atoms_data: np.ndarray[delphi_real],
        coulomb_map_1d: np.ndarray[delphi_real],
        centroid_pve_charge: np.ndarray[delphi_real],
        centroid_nve_charge: np.ndarray[delphi_real],
        phimap_1d: np.ndarray[delphi_real],
    ) -> None:
        exdi_scaled = exdi / epkt
        indi_scaled = indi / epkt
        if vacuum:
            debye_length = ConstDelPhi.ZeroMolarSaltDebyeLength.value

        if bound_cond.int_value == BoundaryCondition.COULOMBIC.int_value:
            if self.platform.active == "cpu":
                set_num_threads(self.platform.names["cpu"]["num_threads"])
                _cpu_setup_coulombic_boundary_condition(
                    vacuum,
                    grid_spacing,
                    exdi_scaled,
                    indi_scaled,
                    debye_length,
                    grid_shape,
                    atoms_data,
                    coulomb_map_1d,
                    phimap_1d,
                )
            if self.platform.active == "cuda":
                # BEGIN: CUDA call section for function: <<_cuda_setup_coulombic_boundary_condition>>
                n_blocks = (
                    self.num_grid_points + self.num_cuda_threads - 1
                ) // self.num_cuda_threads
                grid_shape_device = cuda.to_device(grid_shape)
                atoms_data_device = cuda.to_device(atoms_data)
                coulomb_map_1d_device = cuda.to_device(coulomb_map_1d)
                phimap_1d_device = cuda.device_array_like(phimap_1d)
                # CALL: CUDA kernel for the computation
                _cuda_setup_coulombic_boundary_condition[
                    n_blocks, self.num_cuda_threads
                ](
                    vacuum,
                    grid_spacing,
                    exdi_scaled,
                    indi_scaled,
                    debye_length,
                    grid_shape_device,
                    atoms_data_device,
                    coulomb_map_1d_device,
                    phimap_1d_device,
                )
                # FETCH RESULTS TO HOST FROM DEVICE
                phimap_1d_device.copy_to_host(phimap_1d)
                # CLEAR: mark CUDA memory for garbage collection
                grid_shape_device = None
                atoms_data_device = None
                coulomb_map_1d_device = None
                phimap_1d_device = None
                # END: CUDA call section for function: <<_cuda_setup_coulombic_boundary_condition>>
        elif bound_cond.int_value == BoundaryCondition.DIPOLAR.int_value:
            if self.platform.active == "cpu":
                set_num_threads(self.platform.names["cpu"]["num_threads"])
                _cpu_setup_dipolar_boundary_condition(
                    vacuum,
                    grid_spacing,
                    exdi_scaled,
                    indi_scaled,
                    debye_length,
                    total_pve_charge,
                    total_nve_charge,
                    grid_shape,
                    grid_origin,
                    centroid_pve_charge,
                    centroid_nve_charge,
                    coulomb_map_1d,
                    phimap_1d,
                )
            if self.platform.active == "cuda":
                # BEGIN: CUDA call section for function: <<_cuda_setup_dipolar_boundary_condition>>
                n_blocks = (
                    self.num_grid_points + self.num_cuda_threads - 1
                ) // self.num_cuda_threads
                grid_shape_device = cuda.to_device(grid_shape)
                grid_origin_device = cuda.to_device(grid_origin)
                centroid_pve_charge_device = cuda.to_device(centroid_pve_charge)
                centroid_nve_charge_device = cuda.to_device(centroid_nve_charge)
                coulomb_map_1d_device = cuda.to_device(coulomb_map_1d)
                phimap_1d_device = cuda.device_array_like(phimap_1d)
                # CALL: CUDA kernel for the computation
                _cuda_setup_dipolar_boundary_condition[n_blocks, self.num_cuda_threads](
                    vacuum,
                    grid_spacing,
                    exdi_scaled,
                    indi_scaled,
                    debye_length,
                    total_pve_charge,
                    total_nve_charge,
                    grid_shape_device,
                    grid_origin_device,
                    centroid_pve_charge_device,
                    centroid_nve_charge_device,
                    coulomb_map_1d_device,
                    phimap_1d_device,
                )
                # FETCH RESULTS TO HOST FROM DEVICE
                phimap_1d_device.copy_to_host(phimap_1d)
                # CLEAR: mark CUDA memory for garbage collection
                grid_shape_device = None
                grid_origin_device = None
                centroid_pve_charge_device = None
                centroid_nve_charge_device = None
                coulomb_map_1d_device = None
                phimap_1d_device = None
                # END: CUDA call section for function: <<_cuda_setup_dipolar_boundary_condition>>
        if self.debug:
            np.save(f"new_bndcon_phimap_1d_vacuum-{vacuum}.npy", phimap_1d)
