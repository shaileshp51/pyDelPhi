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
Poisson-Boltzmann Equation Solver Module for PyDelphi

This module implements a class `PBESolver` to solve the Poisson-Boltzmann
Equation (PBE) for electrostatic potential calculations. It supports both
CPU and CUDA platforms for accelerated computation. The solver is designed
to handle various boundary conditions such as Coulombic and Dipolar, and
is optimized for performance using Numba and CUDA.

The module includes functions for:
    - Setting up boundary conditions (Coulombic, Dipolar)
    - Preparing data for iterative solving
    - Calculating relaxation factors for SOR
    - Performing iterative Successive Over-Relaxation (SOR) to solve PBE
    - Managing platform-specific (CPU/CUDA) execution and memory operations
    - Supporting vacuum and aqueous solution phases

Classes:
    PBESolver: Main class for setting up and running the PBE solver.

This module is a core component of the PyDelphi software package for
electrostatic calculations in molecular biophysics and related fields.
"""

import time
import numpy as np

from numba import set_num_threads, njit, prange

from pydelphi.foundation.enums import (
    Precision,
    BoundaryCondition,
    DielectricModel,
    VerbosityLevel,
)
from pydelphi.config.global_runtime import (
    PRECISION,
    delphi_bool,
    delphi_int,
    delphi_real,
    vprint,
)

from pydelphi.config.logging_config import (
    WARNING,
    INFO,
    DEBUG,
    TRACE,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

from pydelphi.constants import ConstPhysical as Constants
from pydelphi.constants import ConstDelPhiFloats as ConstDelPhi
from pydelphi.utils.nonlinear import (
    sinh_taylor_safe,
)

if PRECISION.value == Precision.SINGLE.value:
    from pydelphi.utils.prec.single import *

    try:
        from pydelphi.utils.cuda.single import *
    except ImportError:
        pass
        # print("No Cuda")

elif PRECISION.value == Precision.DOUBLE.value:
    from pydelphi.utils.prec.double import *

    try:
        from pydelphi.utils.cuda.double import *
    except ImportError:
        pass
        # print("No Cuda")

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

from pydelphi.solver.pb.common_pb import (
    _set_gridpoint_charges,
    _cpu_setup_focusing_boundary_condition,
    _cpu_setup_coulombic_boundary_condition,
    _cuda_setup_coulombic_boundary_condition,
    _cpu_setup_dipolar_boundary_condition,
    _cuda_setup_dipolar_boundary_condition,
    _cpu_prepare_charge_neigh_eps_sum_to_iterate,
    _cuda_prepare_charge_neigh_eps_sum_to_iterate,
    _cpu_salt_ions_solvation_penalty,
)


@njit(nogil=True, boundscheck=False, cache=True)
def _add_salt_penalty_to_neigh_eps_sum(
    vacuum: delphi_bool,
    kappa_x_grid_spacing_wholesquare: delphi_real,
    salt_ions_solvation_penalty_map_1d: np.ndarray[delphi_real],
    eps_midpoint_neighs_sum_plus_salt_screening_1d: np.ndarray[delphi_real],
    effective_ions_exclusion_map_1d: np.ndarray[delphi_real],
):
    if not vacuum:
        num_grid_points = eps_midpoint_neighs_sum_plus_salt_screening_1d.shape[0]
        for ijk1d in prange(num_grid_points):
            eps_midpoint_neighs_sum_plus_salt_screening_1d[
                ijk1d
            ] += salt_ions_solvation_penalty_map_1d[ijk1d]
            effective_ions_exclusion_map_1d[ijk1d] = (
                salt_ions_solvation_penalty_map_1d[ijk1d]
                / kappa_x_grid_spacing_wholesquare
            )


@njit(nogil=True, boundscheck=False, cache=True)
def _update_effective_charge_map_1d(
    grid_shape,
    epkt,
    excess_charge_strength,
    excess_charge_scale_factor,
    charge_map_1d,
    non_linear_difference_term,
    effective_charge_map_1d,
    ion_exclusion_map_1d,
    boundary_gridpoints_1d,
    phimap_1d,
):
    y_stride = grid_shape[2]
    x_stride = grid_shape[1] * y_stride
    num_grid_points = grid_shape[0] * x_stride
    effective_scaling_factor = excess_charge_strength * excess_charge_scale_factor

    for ijk1d in prange(num_grid_points):
        not_ion_exclusion_region = not ion_exclusion_map_1d[ijk1d]
        is_not_grid_boundary = not boundary_gridpoints_1d[ijk1d]
        if not_ion_exclusion_region and is_not_grid_boundary:
            phi_at_ijk1d = phimap_1d[ijk1d]
            excess_charge_ijk1d = effective_scaling_factor * (
                sinh_taylor_safe(phi_at_ijk1d, 4.0, 5.0, 5, 0.2) - phi_at_ijk1d
            )

            non_linear_difference_term[ijk1d] = excess_charge_ijk1d
            effective_charge_map_1d[ijk1d] = charge_map_1d[ijk1d] - excess_charge_ijk1d
        else:
            effective_charge_map_1d[ijk1d] = charge_map_1d[ijk1d]
            non_linear_difference_term[ijk1d] = 0.0


class NLPBESolver:
    """
    Linearized Poisson-Boltzmann Equation (LPBE) Solver class.

    This class manages the setup and execution of the PBE solver,
    handling both CPU and CUDA implementations. It initializes necessary
    data structures, prepares for iterations, calculates relaxation factors,
    and runs the iterative solver to compute the electrostatic potential.
    """

    def __init__(
        self,
        platform,
        verbosity,
        num_cuda_threads,
        grid_shape,
    ):
        """
        Initializes the PBESolver.

        Args:
            platform (Platform):  Platform object indicating CPU or CUDA execution and related properties.
            verbosity (VerbosityLevel): Verbosity level for output control.
            num_cuda_threads (int): Number of threads to use for CPU or CUDA kernels.
            grid_shape (tuple): Shape of the 3D grid (nx, ny, nz) as a tuple of ints.
        """
        self.phase = None
        self.platform = platform
        self.verbosity = verbosity
        self.timings = {}
        # Set the scalar variables used in the class
        self.num_cuda_threads = num_cuda_threads
        self.grid_shape = grid_shape
        self.num_grid_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
        self.n_grad_map_points = self.num_grid_points * 3
        # Allocate and init with zero maps_3d and grad_maps_4d
        self.grid_charge_map_1d = np.zeros(self.num_grid_points, dtype=delphi_real)

    def _prepare_to_iterate(
        self,
        vacuum: delphi_bool,
        exdi: delphi_real,
        is_gaussian_diel_model: delphi_bool,
        grid_spacing: delphi_real,
        debye_length: delphi_real,
        non_zero_salt: delphi_bool,
        four_pi: delphi_real,
        epkt: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        ion_exclusion_map_1d: np.ndarray[delphi_real],
        epsilon_map_1d: np.ndarray[delphi_real],
        epsmap_midpoints_1d: np.ndarray[delphi_real],
        charge_map_1d: np.ndarray[delphi_real],
        eps_midpoint_neighs_sum_plus_salt_screening_1d: np.ndarray[delphi_real],
        boundary_gridpoints_1d: np.ndarray[delphi_bool],
    ):
        """
        Prepares the necessary parameters and data structures for the iteration process in the PBE solver.

        This method is called before the main iteration loop and handles platform-specific setup
        (CPU or CUDA). It calculates key derived parameters such as `grid_spacing_square`, `kappa_square`,
        `kappa_x_grid_spacing_wholesquare`, and `four_pi_epkt_grid_spacing`. The function also determines
        boundary grid points and initializes arrays like `eps_nd_midpoint_neighs_sum_1d` and
        `boundary_gridpoints_1d`. It performs all these tasks while considering the active platform.

        Args:
            vacuum (delphi_bool): Flag indicating if the system is in vacuum (non-zero for vacuum).
            exdi (delphi_real): Exterior dielectric constant.
            is_gaussian_diel_model: Whether a Gaussian dielectric model is in use.
            grid_spacing (delphi_real): Distance between consecutive grid points.
            debye_length (delphi_real): Debye length in the medium.
            non_zero_salt (delphi_bool): Whether the salt concentration is non-zero.
            four_pi (delphi_real): Constant 4 * pi.
            epkt (delphi_real): Constant related to scaling factor `r * EPKT`.
            grid_shape (np.ndarray[delphi_int]): Shape of the grid (in x, y, z dimensions).
            ion_exclusion_map_1d (np.ndarray[delphi_real]): 1D array representing the surface of the grid.
            epsilon_map_1d (np.ndarray[delphi_real]): 1D array of dielectric values the grid.
            epsmap_midpoints_1d (np.ndarray[delphi_real]): 1D array of dielectric values at grid midpoints.
            charge_map_1d (np.ndarray[delphi_real]): 1D array holding the charge distribution.
            eps_midpoint_neighs_sum_plus_salt_screening_1d (np.ndarray[delphi_real]): 1D array to hold the sum of dielectric values
                                                                from neighboring grid midpoints.
            boundary_gridpoints_1d (np.ndarray[delphi_bool]): 1D boolean array marking the boundary grid points.

        Returns:
            None
        """
        debug_me = False
        if debug_me:
            charge_map_1d_back = charge_map_1d.copy()
            eps_midpoint_neighs_sum_plus_salt_screening_1d_back = (
                eps_midpoint_neighs_sum_plus_salt_screening_1d.copy()
            )
            boundary_gridpoints_1d_back = boundary_gridpoints_1d.copy()
        # Platform-specific setup
        if self.platform.active == "cpu":
            # Set the number of threads for CPU execution based on platform settings
            set_num_threads(self.platform.names["cpu"]["num_threads"])

            # Call the CPU-specific preparation function to set up data for iteration
            _cpu_prepare_charge_neigh_eps_sum_to_iterate(
                vacuum,
                exdi,
                grid_spacing,
                four_pi,
                epkt,
                grid_shape,
                epsmap_midpoints_1d,
                charge_map_1d,
                eps_midpoint_neighs_sum_plus_salt_screening_1d,
                boundary_gridpoints_1d,
            )

        elif self.platform.active == "cuda":
            # CUDA-specific execution: determine the number of blocks required for grid processing
            n_blocks = (
                self.num_grid_points + self.num_cuda_threads - 1
            ) // self.num_cuda_threads

            # Allocate memory on the GPU for input and output arrays
            grid_shape_device = cuda.to_device(grid_shape)
            epsmap_midpoints_1d_device = cuda.to_device(epsmap_midpoints_1d)
            charge_map_1d_device = cuda.to_device(charge_map_1d)
            eps_midpoint_neighs_sum_plus_salt_screening_1d_device = cuda.to_device(
                eps_midpoint_neighs_sum_plus_salt_screening_1d
            )
            boundary_gridpoints_1d_device = cuda.to_device(boundary_gridpoints_1d)

            # Launch the CUDA kernel with appropriate grid and block configuration
            _cuda_prepare_charge_neigh_eps_sum_to_iterate[
                n_blocks, self.num_cuda_threads
            ](
                vacuum,
                exdi,
                grid_spacing,
                four_pi,
                epkt,
                grid_shape_device,
                epsmap_midpoints_1d_device,
                charge_map_1d_device,
                eps_midpoint_neighs_sum_plus_salt_screening_1d_device,
                boundary_gridpoints_1d_device,
            )

            # Transfer the computed results back from GPU to host memory
            charge_map_1d_device.copy_to_host(charge_map_1d)
            eps_midpoint_neighs_sum_plus_salt_screening_1d_device.copy_to_host(
                eps_midpoint_neighs_sum_plus_salt_screening_1d
            )
            boundary_gridpoints_1d_device.copy_to_host(boundary_gridpoints_1d)

            # Clear GPU memory by setting references to None
            grid_shape_device = None
            surface_map_1d_device = None
            epsmap_midpoints_1d_device = None
            charge_map_1d_device = None
            eps_midpoint_neighs_sum_plus_salt_screening_1d_device = None
            boundary_gridpoints_1d_device = None

        num_grid_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
        salt_ions_solvation_penalty_map_1d = np.zeros(
            num_grid_points, dtype=delphi_real
        )

        if (not vacuum) and non_zero_salt:
            grid_spacing_square = grid_spacing**2
            kappa_square = exdi / debye_length**2  # Related to ionic screening

            kappa_x_grid_spacing_wholesquare = (
                kappa_square * grid_spacing_square
            )  # Screening term
            # Compute salt ions solvation penalty/screening factor
            _cpu_salt_ions_solvation_penalty(
                vacuum=vacuum,
                non_zero_salt=non_zero_salt,
                is_gaussian_diel_model=is_gaussian_diel_model,
                exdi=exdi,
                ion_radius=2.0,
                ions_valance=1.0,
                debye_length=debye_length,
                epkt=epkt,
                grid_spacing=grid_spacing,
                grid_shape=grid_shape,
                epsilon_map_1d=epsilon_map_1d,
                ion_exclusion_map_1d=ion_exclusion_map_1d,
                salt_ions_solvation_penalty_map_1d=salt_ions_solvation_penalty_map_1d,
            )
            if debug_me:
                np.save(
                    f"epsilon_map_1d_gaussian-{is_gaussian_diel_model}.npy",
                    epsilon_map_1d,
                )
                np.save(
                    f"salt_ions_solvation_penalty_map_1d_gaussian-{is_gaussian_diel_model}.npy",
                    salt_ions_solvation_penalty_map_1d,
                )

            # Add salt ions solvation penalty to the neighbor midpoints dielectric sum to prepare denominator of iter formula
            _add_salt_penalty_to_neigh_eps_sum(
                vacuum=vacuum,
                kappa_x_grid_spacing_wholesquare=kappa_x_grid_spacing_wholesquare,
                salt_ions_solvation_penalty_map_1d=salt_ions_solvation_penalty_map_1d,
                eps_midpoint_neighs_sum_plus_salt_screening_1d=eps_midpoint_neighs_sum_plus_salt_screening_1d,
                effective_ions_exclusion_map_1d=ion_exclusion_map_1d,
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

            if itr_block_size > 1:
                for itrid_inner in range(
                    1, itr_block_size
                ):  # Runs for 1 to itr_block_size - 1
                    for even_odd in [0, 1]:
                        _cuda_iterate_relaxation_factor[
                            num_blocks, self.num_cuda_threads
                        ](
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
                # After (itr_block_size - 1) iterations on GPU, copy back the relevant part
                # to set phimap_current_1d for the spectral radius calculation.
                phimap_odds_1d_device.copy_to_host(
                    phimap_odds_1d
                )  # Get the odd part from iteration (itr_block_size - 1)
                _copy_to_full(
                    phimap_current_1d, phimap_odds_1d, 1, 2
                )  # Set phimap_current_1d

                # Perform the final (itr_block_size)th iteration
            for even_odd in [0, 1]:
                _cuda_iterate_relaxation_factor[num_blocks, self.num_cuda_threads](
                    even_odd,
                    grid_shape_device,
                    (phimap_odds_1d_device if even_odd == 0 else phimap_even_1d_device),
                    (phimap_even_1d_device if even_odd == 0 else phimap_odds_1d_device),
                    epsmap_midpoints_1d_device,
                    eps_nd_midpoint_neighs_sum_1d_device,
                    boundary_gridpoints_1d_device,
                )

                # Copy back the final phimap_odds_1d (from itr_block_size) for the spectral radius calculation
            phimap_odds_1d_device.copy_to_host(phimap_odds_1d)
            phimap_even_1d_device.copy_to_host(
                phimap_even_1d
            )  # Also copy even if needed elsewhere, though not for spectral radius

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
        # print("calculated spectral_radius:", spectral_radius)

        return spectral_radius

    def _solve_linear_pb_sor(
        self,
        phimap_current_1d: np.ndarray[delphi_real],
        omega: delphi_real,
        approx_zero: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        epsmap_midpoints_1d: np.ndarray[delphi_real],
        eps_midpoint_neighs_sum_plus_salt_screening_1d: np.ndarray[delphi_real],
        boundary_gridpoints_1d: np.ndarray[delphi_bool],
        charge_map_1d: np.ndarray[delphi_real],
        itr_block_size: delphi_int,
        max_linear_iters: delphi_int,
        max_rms: delphi_real,
        effective_max_dphi: delphi_real,
        check_dphi: delphi_bool,
        platform_active: str,  # "cpu" or "cuda"
        num_cuda_threads: delphi_int,
        num_cpu_threads: delphi_int,
        verbosity_level: int,
        num_grid_points: delphi_int,
    ) -> np.ndarray[delphi_real]:
        """
        Performs the core iterative solution loop for the Linearized Poisson-Boltzmann
        Equation using the SOR method.

        This method encapsulates the allocation/initialization of working arrays
        (even/odd potentials), the iteration loop (Steps 7), and associated CUDA
        memory cleanup (Step 8). It operates on the provided potential array
        `phimap_current_1d`, updating it in-place. The convergence check logic
        is replicated as found in the source.

        Args:
            phimap_current_1d (np.ndarray[delphi_real]): 1D array for the full potential map.
                Used as input initial guess and updated in-place to the final solution.
                Note: The state of this array *before* the last block iteration is
                used in the RMSD calculation.
            omega (delphi_real): The SOR relaxation factor.
            approx_zero (delphi_real): Value considered approximately zero.
            grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid.
            epsmap_midpoints_1d (np.ndarray[delphi_real]): Dielectric map at midpoints.
            eps_midpoint_neighs_sum_plus_salt_screening_1d (np.ndarray[delphi_real]):
                Neighbor epsilon sums including salt screening.
            boundary_gridpoints_1d (np.ndarray[delphi_bool]): Boundary points array.
            charge_map_1d (np.ndarray[delphi_real]): Charge distribution map.
            itr_block_size (delphi_int): Number of iterations per block.
            max_linear_iters (delphi_int): Maximum total iterations.
            max_rms (delphi_real): RMSD convergence tolerance.
            effective_max_dphi (delphi_real): Max dPhi convergence tolerance.
            check_dphi (delphi_bool): Flag to use max dPhi for convergence.
            platform_active (str): "cpu" or "cuda".
            num_cuda_threads (delphi_int): Threads per block for CUDA.
            num_cpu_threads (delphi_int): Threads for CPU parallelization.
            verbosity_level (int): Verbosity level.
            num_grid_points (delphi_int): Total grid points.

        Returns:
            np.ndarray[delphi_real]: The updated `phimap_current_1d` array containing the converged potential.
        """
        # --- 5. Iterative Solver Setup ---
        num_grid_points_half = (num_grid_points + 1) // 2
        phimap_even_1d = np.zeros(num_grid_points_half, dtype=delphi_real)
        phimap_odds_1d = np.zeros(num_grid_points_half, dtype=delphi_real)
        _copy_to_sample(phimap_even_1d, phimap_current_1d, offset=0, skip=2)
        _copy_to_sample(phimap_odds_1d, phimap_current_1d, offset=1, skip=2)

        # --- 6. CUDA Setup ---
        if platform_active == "cuda":
            n_blocks = (num_grid_points_half + num_cuda_threads - 1) // (
                num_cuda_threads
            )
            grid_shape_device = cuda.to_device(grid_shape)
            epsmap_midpoints_1d_device = cuda.to_device(epsmap_midpoints_1d)
            eps_midpoint_neighs_sum_plus_salt_screening_1d_device = cuda.to_device(
                eps_midpoint_neighs_sum_plus_salt_screening_1d
            )
            boundary_gridpoints_1d_device = cuda.to_device(boundary_gridpoints_1d)
            charge_map_1d_device = cuda.to_device(charge_map_1d)

        vprint(TRACE, _VERBOSITY, "phimap_current_1d before itr:", phimap_current_1d)

        # --- 7. Iteration Loop (LIFTED DIRECTLY FROM ORIGINAL CODE) ---
        do_iterate, itr_num = True, 0
        while do_iterate:
            tic_itr = time.perf_counter()
            if itr_num == 0:
                vprint(
                    INFO,
                    _VERBOSITY,
                    f"    PBE> | #Iteration |    RMSD    |  Max(dPhi) | Time (seconds) |",
                )

            if platform_active == "cuda":
                # Transfer iteration-specific data to CUDA device for computation (already done in moved Step 6)
                # phimap_even_1d_device and phimap_odds_1d_device need to be transferred *inside* the loop block
                phimap_even_1d_device = cuda.to_device(phimap_even_1d)
                phimap_odds_1d_device = cuda.to_device(phimap_odds_1d)

                for itrid in range(itr_num + 1, itr_num + itr_block_size + 1):
                    for even_odd in np.array([0, 1], dtype=delphi_int):
                        if even_odd == 0:
                            # 7.1. Iterate for even indexed grid points using CUDA kernel
                            _cuda_iterate_SOR[n_blocks, num_cuda_threads](
                                even_odd,
                                omega,
                                approx_zero,
                                grid_shape_device,
                                phimap_odds_1d_device,
                                phimap_even_1d_device,
                                epsmap_midpoints_1d_device,
                                eps_midpoint_neighs_sum_plus_salt_screening_1d_device,
                                boundary_gridpoints_1d_device,
                                charge_map_1d_device,
                            )
                        elif even_odd == 1:
                            # 7.2. Iterate for odd indexed grid points using CUDA kernel
                            _cuda_iterate_SOR[n_blocks, num_cuda_threads](
                                even_odd,
                                omega,
                                approx_zero,
                                grid_shape_device,
                                phimap_even_1d_device,
                                phimap_odds_1d_device,
                                epsmap_midpoints_1d_device,
                                eps_midpoint_neighs_sum_plus_salt_screening_1d_device,
                                boundary_gridpoints_1d_device,
                                charge_map_1d_device,
                            )
                        if itrid == itr_num + itr_block_size - 1:
                            # For the second to last iteration in the block, copy results back to host for RMSD check
                            phimap_odds_1d_device.copy_to_host(phimap_odds_1d)
                            # Update full potential map with the latest odd potential values
                            _copy_to_full(
                                phimap_current_1d,
                                phimap_odds_1d,
                                1,
                                2,
                            )

                # Fetch final results from CUDA device to host after the block
                phimap_odds_1d_device.copy_to_host(phimap_odds_1d)
                phimap_even_1d_device.copy_to_host(phimap_even_1d)
                # Release iteration-specific CUDA memory
                phimap_odds_1d_device = None
                phimap_even_1d_device = None

            elif platform_active == "cpu":
                set_num_threads(num_cpu_threads)
                # 7.3. Iterate block of grid points (excluding the last iteration) on CPU
                _cpu_iterate_block_SOR(
                    itr_num,  # Global iteration number counter
                    itr_block_size - 1,
                    omega,
                    approx_zero,
                    grid_shape,
                    phimap_odds_1d,
                    phimap_even_1d,
                    epsmap_midpoints_1d,
                    eps_midpoint_neighs_sum_plus_salt_screening_1d,
                    boundary_gridpoints_1d,
                    charge_map_1d,
                )
                # Update full potential map with odd potentials after block iteration
                _copy_to_full(
                    phimap_current_1d,
                    phimap_odds_1d,
                    1,
                    2,
                )
                # 7.4. Iterate the last iteration of the block separately on CPU
                _cpu_iterate_block_SOR(
                    itr_num
                    + itr_block_size
                    - 1,  # Global iteration number counter for the last iter
                    1,
                    omega,
                    approx_zero,
                    grid_shape,
                    phimap_odds_1d,
                    phimap_even_1d,
                    epsmap_midpoints_1d,
                    eps_midpoint_neighs_sum_plus_salt_screening_1d,
                    boundary_gridpoints_1d,
                    charge_map_1d,
                )

            # After the block iterations (CPU or CUDA), update the global iteration counter
            itr_num += itr_block_size

            # 7.5. Calculate RMSD and Max dPhi for convergence check
            rmsd, max_delta_phi = _calculate_phi_map_sample_rmsd(
                phimap_current_1d,  # State before the last iteration of the block
                phimap_odds_1d,  # State after the last-1 iteration of the block
                offset=1,
                stride=2,
                num_cpu_threads=num_cpu_threads,
                dtype=delphi_real,
            )
            max_delta_phi = abs(max_delta_phi)

            # --- Convergence Check ---
            # Check if maximum iterations limit is reached (for the total iterations completed)
            if itr_num >= max_linear_iters:
                do_iterate = False
            # Check for convergence based on RMSD or Max dPhi
            elif (not check_dphi) and rmsd < max_rms:
                do_iterate = False
            elif check_dphi and max_delta_phi < effective_max_dphi:
                do_iterate = False

            # After convergence check, update phimap_current_1d for the *next* block's starting state
            # Need to copy the final even and odd results after the block into phimap_current_1d.
            # The original code copies even results here. Odd results were copied earlier.
            _copy_to_full(
                phimap_current_1d,
                phimap_even_1d,
                0,
                2,
            )
            toc_itr = time.perf_counter()

            vprint(
                INFO,
                _VERBOSITY,
                f"    PBE> | {itr_num:>10d} | {rmsd:>9.04e} | {max_delta_phi:>9.04e} | {toc_itr - tic_itr:14.06f} |",
            )

        # --- 8. CUDA Memory Cleanup (LIFTED DIRECTLY FROM ORIGINAL CODE) ---
        if platform_active == "cuda":
            grid_shape_device = None
            epsmap_midpoints_1d_device = None
            eps_midpoint_neighs_sum_plus_salt_screening_1d_device = None
            boundary_gridpoints_1d_device = None
            charge_map_1d_device = None

        return phimap_current_1d

    def solve_nonlinear_pb(
        self,
        vacuum: delphi_bool,
        bound_cond: BoundaryCondition,
        dielectric_model: DielectricModel,
        gaussian_exponent: delphi_int,
        itr_block_size: delphi_int,
        max_linear_iters: delphi_int,
        max_nonlinear_iters: delphi_int,
        nonlinear_coupling_max_dphi: delphi_real,
        coupling_steps: delphi_int,
        manual_relaxation_value: delphi_real,
        scale: delphi_real,
        scale_parentrun: delphi_real,
        exdi: delphi_real,
        indi: delphi_real,
        debye_length: delphi_real,
        non_zero_salt: delphi_bool,
        total_pve_charge: delphi_real,
        total_nve_charge: delphi_real,
        max_rms: delphi_real,
        max_dphi: delphi_real,
        check_dphi: delphi_bool,
        epkt: delphi_real,
        approx_zero: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        grid_origin: np.ndarray[delphi_real],
        grid_shape_parentrun: np.ndarray[delphi_int],
        grid_origin_parentrun: np.ndarray[delphi_real],
        atoms_data: np.ndarray[delphi_real],
        density_map_1d: np.ndarray[delphi_real],
        ion_exclusion_map_1d: np.ndarray[delphi_real],
        epsilon_map_1d: np.ndarray[delphi_real],
        epsmap_midpoints_1d: np.ndarray[delphi_real],
        centroid_pve_charge: np.ndarray[delphi_real],
        centroid_nve_charge: np.ndarray[delphi_real],
        charged_gridpoints_1d: np.ndarray[delphi_real],  # Original fixed charges
        phimap_parentrun: np.ndarray[delphi_real],
    ) -> np.ndarray[delphi_real]:
        """
        Solves the Poisson-Boltzmann Equation (PBE), handling both linear and non-linear cases.

        This method employs an iterative Successive Over-Relaxation (SOR) solver.
        For the non-linear PBE, it uses an iterative coupling approach where a parameter
        'chi' is gradually increased from 0 to 1. This parameter controls the extent to
        which the non-linear (Boltzmann) term for mobile ions contributes to the effective
        charge source. At each 'coupling_step', a modified linear Poisson equation is solved
        with an updated effective charge.

        The linear PBE is solved as a special case when 'coupling_steps' is set to 0.
        In this scenario, 'chi' remains 1.0 (or is effectively ignored), the non-linear
        charge update is skipped, and a single linear PBE solve is performed.

        The main steps are:
        1.  Platform Selection and Initialization.
        2.  Charge Assignment to grid points (fixed charges).
        3.  Boundary Condition Setup (initial potential).
        4.  Preparation for Linear Iteration (calculating constant operator terms like
            dielectric sums and linear salt screening coefficients).
        5.  Iterative Solver Setup (calculating the SOR relaxation factor/omega) for
            the *linear* part of the operator.
        6.  Iterative Nonlinear Solving Loop:
            - If `coupling_steps` > 0: The loop iterates from `nonlinear_iter = 0` up to `coupling_steps`.
              - `current_chi` is gradually increased from 0.0 to 1.0.
              - For `nonlinear_iter > 0`, an `effective_charge_map_1d` is computed by adding the fixed
                solute charges and the *nonlinear* mobile ion charge term (scaled by `current_chi`),
                which is dependent on the current potential.
              - A linear SOR solve (`_solve_linear_pb_sor`) is performed using this `effective_charge_map_1d`.
              - Convergence of the non-linear iteration is checked based on the change in potential.
              - The loop can terminate early if non-linear convergence is achieved or `max_nonlinear_iters` is reached.
            - If `coupling_steps` == 0: The loop runs only for `nonlinear_iter = 0`.
              - `current_chi` is effectively 1.0.
              - The non-linear charge update is skipped.
              - A single linear SOR solve is performed using only the fixed solute charges.

        Args:
            vacuum (delphi_bool): True if solving in vacuum, False for water.
            bound_cond (BoundaryCondition): Type of boundary condition.
            gaussian_exponent (delphi_int): Exponent for Gaussian charge spreading.
            itr_block_size (delphi_int): Number of iterations per block for status checks/RMSD calculation.
            max_linear_iters (delphi_int): Maximum total iterations for each *linear* sub-solve within the non-linear loop.
                                           Also the maximum for the single linear solve when coupling_steps is 0.
            max_nonlinear_iters (delphi_int): Maximum total iterations for the non-linear solve. This limits the total
                                              number of coupling steps if convergence isn't reached earlier.
            nonlinear_coupling_max_dphi (delphi_real): Tolerance for non-linear convergence (e.g., change in potential)
                                                       during intermediate coupling steps.
            coupling_steps (delphi_int): Number of steps to increase the coupling parameter from 0 to 1.
                                         Set to 0 for a purely linear PBE solve.
            manual_relaxation_value (delphi_real): Manual relaxation parameter for non-linear iterations (0.0 to disable).
            scale (delphi_real): Grid scale (points per Angstrom).
            scale_parentrun (delphi_real): Grid scale of a parent run (if applicable for boundary conditions).
            exdi (delphi_real): Exterior dielectric constant.
            indi (delphi_real): Interior dielectric constant.
            debye_length (delphi_real): Debye length (related to salt concentration).
            non_zero_salt (delphi_bool): True if salt concentration is non-zero.
            total_pve_charge (delphi_real): Total positive charge from atoms.
            total_nve_charge (delphi_real): Total negative charge from atoms.
            max_rms (delphi_real): Maximum RMSD tolerance for convergence of the linear sub-solves.
            max_dphi (delphi_real): Maximum potential change tolerance for convergence of the linear sub-solves.
                                    This specific tolerance is used at the start (lambda=0) and end (lambda=1 or coupling_steps)
                                    of the overall non-linear process.
            check_dphi (delphi_bool): Flag to use max_dphi instead of max_rms for linear sub-solve convergence.
            epkt (delphi_real): kT/e in the appropriate units.
            approx_zero (delphi_real): A value considered close to zero.
            grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid (nx, ny, nz).
            grid_origin (np.ndarray[delphi_real]): Origin coordinates of the grid.
            grid_shape_parentrun (np.ndarray[delphi_int]): Shape of the parent grid (if applicable).
            grid_origin_parentrun (np.ndarray[delphi_real]): Origin of the parent grid (if applicable).
            atoms_data (np.ndarray[delphi_real]): Array containing atom data (coords, radii, charges).
            density_map_1d (np.ndarray[delphi_real]): 1D map of atom density/exclusion volumes.
            ion_exclusion_map_1d (np.ndarray[delphi_real]): 1D boolean map marking ion exclusion regions.
            epsilon_map_1d (np.ndarray[delphi_real]): 1D map of dielectric constants at grid points.
            epsmap_midpoints_1d (np.ndarray[delphi_real]): 1D map of dielectric constants at cell midpoints.
            centroid_pve_charge (np.ndarray[delphi_real]): Centroid coordinates of positive charges.
            centroid_nve_charge (np.ndarray[delphi_real]): Centroid coordinates of negative charges.
            charged_gridpoints_1d (np.ndarray[delphi_real]): 1D array of charges assigned to grid points.
            phimap_parentrun (np.ndarray[delphi_real]): Potential map from a parent run (if applicable for boundary conditions).

        Returns:
            np.ndarray[delphi_real]: The computed electrostatic potential map as a 1D array.
        """
        tic_total = time.perf_counter()
        # --- 1. Platform Selection and Initialization ---
        if self.platform.active == "cuda":
            try:
                cuda.select_device(self.platform.names["cuda"]["selected_id"])
            except Exception as e:
                vprint(
                    WARNING,
                    _VERBOSITY,
                    f"Warning: Could not select CUDA device {self.platform.names['cuda']['selected_id']}: {e}",
                )
                self.platform.active = "cpu"
                vprint(INFO, _VERBOSITY, "Falling back to CPU.")
                set_num_threads(self.platform.names["cpu"]["num_threads"])

        self.phase = "vacuum" if vacuum else "water"
        grid_spacing = 1.0 / scale
        grid_spacing_square = grid_spacing**2
        kappa_square = (exdi / debye_length**2) if non_zero_salt else delphi_real(0.0)
        kappa_x_grid_spacing_wholesquare = kappa_square * grid_spacing_square
        is_gaussian_diel_model = dielectric_model.int_value in (
            DielectricModel.GAUSSIAN.int_value,
        )

        # --- 2. Initial Charge Assignment (Fixed Charges) ---
        tic_gchrg = time.perf_counter()
        if (
            self.grid_charge_map_1d is None
            or self.grid_charge_map_1d.shape[0] != self.num_grid_points
        ):
            self.grid_charge_map_1d = np.zeros(self.num_grid_points, dtype=delphi_real)
        else:
            self.grid_charge_map_1d.fill(0.0)

        _set_gridpoint_charges(
            grid_shape=grid_shape,
            charged_gridpoints_1d=charged_gridpoints_1d,
            grid_charge_map_1d=self.grid_charge_map_1d,
        )
        toc_gchrg = time.perf_counter()
        self.timings[f"pb, {self.phase}| calc. charge source"] = "{:0.3f}".format(
            toc_gchrg - tic_total
        )

        vprint(
            TRACE,
            _VERBOSITY,
            f"     <<PBE>> total_pve_charge={total_pve_charge:0.6f}, total_nve_charge={total_nve_charge:0.6f}",
        )

        # --- 3. Boundary Condition Setup (Initial Potential) ---
        tic_bndcon = time.perf_counter()
        # Start with potential initialized by boundary conditions. This serves as the
        # initial guess for the non-linear solver (and the solution at lambda=0).
        phimap_current_1d = np.zeros(self.num_grid_points, dtype=delphi_real)

        # Initialize boundary condition will be updated while solving
        self._setup_boundary_condition(
            vacuum=vacuum,
            bound_cond=bound_cond,
            grid_spacing=grid_spacing,
            grid_spacing_parentrun=1.0 / scale_parentrun,
            exdi=exdi,
            indi=indi,
            total_nve_charge=total_nve_charge,
            total_pve_charge=total_pve_charge,
            debye_length=debye_length,
            non_zero_salt=non_zero_salt,
            epkt=epkt,
            grid_shape=grid_shape,
            grid_origin=grid_origin,
            grid_shape_parentrun=grid_shape_parentrun,
            grid_origin_parentrun=grid_origin_parentrun,
            atoms_data=atoms_data,
            centroid_pve_charge=centroid_pve_charge,
            centroid_nve_charge=centroid_nve_charge,
            phimap_1d=phimap_current_1d,
            phimap_parentrun=phimap_parentrun,
        )
        toc_bndcon = time.perf_counter()
        self.timings[f"pb, {self.phase}| set boundary condition"] = "{:0.3f}".format(
            toc_bndcon - toc_gchrg
        )

        # --- 4. Prepare for Linear Iteration (Constant Parts) ---
        # Calculate arrays needed for the linear SOR solver that depend only on the
        # linear operator (Laplacian + Linear Salt) and grid structure.
        # This populates the diagonal coefficients including the linear salt term,
        # and identifies boundary grid points for the SOR mask.
        tic_prepitr_const = time.perf_counter()

        # These arrays represent the properties of the linear system being solved at each step
        eps_midpoint_neighs_sum_plus_salt_screening_1d = np.empty(
            self.num_grid_points, dtype=delphi_real
        )
        boundary_gridpoints_1d = np.zeros(self.num_grid_points, dtype=delphi_bool)

        # Call _prepare_to_iterate with a dummy charge map. The diagonal coefficient array
        # and boundary array calculation within _prepare_to_iterate should only depend
        # on dielectric, grid, and linear salt term, not the charge source term (RHS).
        charge_map_1d = np.copy(self.grid_charge_map_1d)
        self._prepare_to_iterate(
            vacuum=vacuum,
            exdi=exdi,
            is_gaussian_diel_model=is_gaussian_diel_model,
            grid_spacing=grid_spacing,
            debye_length=debye_length,
            non_zero_salt=non_zero_salt,
            four_pi=delphi_real(Constants.FourPi.value),
            epkt=epkt,  # Used for linear salt term calc in prepare?
            grid_shape=grid_shape,
            ion_exclusion_map_1d=ion_exclusion_map_1d,
            epsilon_map_1d=epsilon_map_1d,
            epsmap_midpoints_1d=epsmap_midpoints_1d,
            charge_map_1d=charge_map_1d,  # This is the RHS, not needed for operator setup
            eps_midpoint_neighs_sum_plus_salt_screening_1d=eps_midpoint_neighs_sum_plus_salt_screening_1d,
            boundary_gridpoints_1d=boundary_gridpoints_1d,
        )
        toc_prepitr_const = time.perf_counter()
        self.timings[f"pb, {self.phase}| prepare for linear operator"] = (
            "{:0.3f}".format(toc_prepitr_const - toc_bndcon)
        )
        # Make a copy of constant solute charge as effective charge to start with
        effective_charge_map_1d = np.copy(charge_map_1d)
        if vacuum == False and non_zero_salt and self.verbosity >= DEBUG:
            vprint(
                DEBUG,
                _VERBOSITY,
                f"     <<PBE>> kappa_square={kappa_square:0.6f}, kappa_sq_times_h_sq={kappa_x_grid_spacing_wholesquare:0.6f}",
            )

        # --- 5. Calculate Omega (Constant) ---
        # Calculate relaxation factor and SOR omega for the linear operator.
        # This should be done once as the linear operator is fixed.
        periodic_boundary_xyz = np.zeros(3, dtype=delphi_bool)
        tic_calrelpar = time.perf_counter()
        relax_factor = self._calc_relaxation_factor(
            itr_block_size=1,  # Calculation typically based on spectral radius estimate
            grid_shape=grid_shape,
            periodic_boundary_xyz=periodic_boundary_xyz,
            epsmap_midpoints_1d=epsmap_midpoints_1d,
            eps_nd_midpoint_neighs_sum_1d=eps_midpoint_neighs_sum_plus_salt_screening_1d,
            # Using calculated sum with linear salt
            boundary_gridpoints_1d=boundary_gridpoints_1d,
        )

        omega = delphi_real(1 - (2.0 / (1.0 + math.sqrt(1.0 - relax_factor))))

        vprint(
            INFO,
            _VERBOSITY,
            f"    PBE> Relaxation-factor (linear operator) = {relax_factor:10.6f} and SOR-omega = {omega:10.6f}",
        )
        vprint(
            DEBUG,
            _VERBOSITY,
            "Non-Linear coupling max delta phi",
            nonlinear_coupling_max_dphi,
        )

        toc_calrelpar = time.perf_counter()
        self.timings[f"pb, {self.phase}| calc. relaxation factor"] = "{:0.3f}".format(
            toc_calrelpar - toc_prepitr_const
        )

        # --- 6. Iterative Nonlinear Solving Loop ---
        vprint(TRACE, _VERBOSITY, "    PBE> Starting non-linear iterations...")

        previous_phimap_1d = np.copy(phimap_current_1d)
        nonlinear_converged = False

        effective_omega = omega  # Note: use calculated omega for linear part and ``nlrelpar`` if supplied.
        previous_chi = 0.0

        for nonlinear_iter in range(coupling_steps + 1):
            current_chi = (
                min(1.0, 0.0 + nonlinear_iter * 1.0 / coupling_steps)
                if coupling_steps > 0
                else 1.0
            )
            effective_max_dphi = (
                max_dphi
                if nonlinear_iter in (0, coupling_steps)
                else nonlinear_coupling_max_dphi
            )
            effective_check_dphi = (
                check_dphi if nonlinear_iter in (0, coupling_steps) else True
            )
            if nonlinear_iter != 0:
                if manual_relaxation_value != 0.0:
                    effective_omega = (
                        1.0 - manual_relaxation_value
                    )  # Note here omega represents 1- omega in derivation

                # --- Calculate Effective Charge Map for current lambda and potential ---
                if non_zero_salt:
                    vprint(
                        INFO,
                        _VERBOSITY,
                        f"\n\n    PBE> Non-linear coupling: iteration {nonlinear_iter} of {coupling_steps}, strength (chi): {current_chi:0.4f},"
                        f" effective_max_dphi: {effective_max_dphi:0.4e}",
                    )

                    # Start with fixed charges
                    non_linear_difference_term = np.zeros_like(
                        effective_charge_map_1d, dtype=delphi_real
                    )
                    _update_effective_charge_map_1d(
                        grid_shape=grid_shape,
                        epkt=epkt,
                        excess_charge_strength=current_chi,
                        excess_charge_scale_factor=kappa_x_grid_spacing_wholesquare,
                        charge_map_1d=charge_map_1d,
                        non_linear_difference_term=non_linear_difference_term,
                        effective_charge_map_1d=effective_charge_map_1d,
                        ion_exclusion_map_1d=ion_exclusion_map_1d,
                        boundary_gridpoints_1d=boundary_gridpoints_1d,
                        phimap_1d=phimap_current_1d,
                    )
                    # np.save(
                    #     f"phimap_current_1d_{nonlinear_iter}.npy", phimap_current_1d
                    # )
                    # np.save(
                    #     f"non_linear_difference_term_{nonlinear_iter}.npy",
                    #     non_linear_difference_term,
                    # )
                    # np.save(
                    #     f"effective_charge_map_1d_{nonlinear_iter}.npy",
                    #     effective_charge_map_1d,
                    # )

            else:  # effective_charge_map_1d remains just the fixed charges (no non-linear term)
                effective_charge_map_1d[:] = charge_map_1d[:]

            # Store previous potential for convergence check
            previous_phimap_1d[:] = phimap_current_1d[:]
            # np.save(
            #     f"phimap_current_1d_before{nonlinear_iter}.npy",
            #     phimap_current_1d,
            # )

            # --- Perform Linear Iterations with Effective Charge ---
            # Call the SOR solver with the current potential as the starting guess
            # and the effective charge map as the RHS source term.
            tic_linear_solve = time.perf_counter()
            self._solve_linear_pb_sor(
                phimap_current_1d=phimap_current_1d,  # Potential is updated in-place
                omega=effective_omega,
                approx_zero=approx_zero,
                grid_shape=grid_shape,
                epsmap_midpoints_1d=epsmap_midpoints_1d,
                eps_midpoint_neighs_sum_plus_salt_screening_1d=eps_midpoint_neighs_sum_plus_salt_screening_1d,
                # Use constant linear operator terms
                boundary_gridpoints_1d=boundary_gridpoints_1d,  # Use constant boundary flags
                charge_map_1d=effective_charge_map_1d,  # Pass the effective charge map (RHS)
                itr_block_size=itr_block_size,
                max_linear_iters=max_linear_iters,  # Max linear iters per non-linear step
                max_rms=max_rms,
                effective_max_dphi=effective_max_dphi,
                check_dphi=effective_check_dphi,
                platform_active=self.platform.active,
                num_cuda_threads=self.num_cuda_threads,
                num_cpu_threads=self.platform.names["cpu"]["num_threads"],
                verbosity_level=_VERBOSITY,
                num_grid_points=self.num_grid_points,
            )
            # np.save(
            #     f"phimap_current_1d_after{nonlinear_iter}.npy",
            #     phimap_current_1d,
            # )
            toc_linear_solve = time.perf_counter()
            # More accurate timing for the linear solve within the loop
            if nonlinear_iter == 0:
                self.timings[
                    f"pb, {self.phase}| linear solve iter {nonlinear_iter + 1}"
                ] = "{:0.3f}".format(
                    toc_linear_solve - toc_calrelpar  # Time from omega calculation
                )
            else:
                pass  # Timing is already done inside _solve_linear_pb_sor if it logs

            # --- Check for Nonlinear Convergence ---
            if non_zero_salt and nonlinear_iter > 0:
                # Calculate the change in potential
                max_nonlinear_dphi = np.max(
                    np.abs(phimap_current_1d - previous_phimap_1d)
                )

                vprint(
                    INFO,
                    _VERBOSITY,
                    f"    PBE> Non-linear iteration {nonlinear_iter} potential change: {max_nonlinear_dphi:0.6g}",
                )

                # Convergence is achieved if potential change is small AND lambda has reached 1.0
                # Check if current_chi is close to 1.0 using approx_zero
                if (
                    max_nonlinear_dphi < nonlinear_coupling_max_dphi
                    and abs(current_chi - 1.0) < approx_zero
                ):
                    nonlinear_converged = True
                    vprint(INFO, _VERBOSITY, "    PBE> Non-linear solver converged.")
                    break  # Exit the nonlinear loop

                if nonlinear_iter == max_nonlinear_iters - 1:
                    vprint(
                        INFO,
                        _VERBOSITY,
                        "    PBE> Non-linear solver did not converge within max iterations.",
                    )

        toc_total = time.perf_counter()
        self.timings[f"pb, {self.phase}| total time"] = "{:0.3f}".format(
            toc_total - tic_total
        )

        return phimap_current_1d

    def _setup_boundary_condition(
        self,
        vacuum: np.bool_,
        bound_cond: BoundaryCondition,
        grid_spacing: delphi_real,
        grid_spacing_parentrun: delphi_real,
        exdi: delphi_real,
        indi: delphi_real,
        total_pve_charge: delphi_real,
        total_nve_charge: delphi_real,
        debye_length: delphi_real,
        non_zero_salt: delphi_bool,
        epkt: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        grid_origin: np.ndarray[delphi_real],
        grid_shape_parentrun: np.ndarray[delphi_int],
        grid_origin_parentrun: np.ndarray[delphi_real],
        atoms_data: np.ndarray[delphi_real],
        centroid_pve_charge: np.ndarray[delphi_real],
        centroid_nve_charge: np.ndarray[delphi_real],
        phimap_1d: np.ndarray[delphi_real],
        phimap_parentrun: np.ndarray[delphi_real],
    ) -> None:
        """
        Sets up the boundary condition for the Poisson-Boltzmann Equation (PBE) solver.

        This method initializes the electrostatic potential (phimap_1d) at the boundaries of the grid
        based on the specified boundary condition type (DelphiBoundaryCondition). It supports Coulombic
        and Dipolar boundary conditions and dispatches execution to platform-specific (CPU or CUDA)
        functions for actual computation.

        Args:
            vacuum (np.bool_): Boolean flag indicating if the calculation is in vacuum.
            bound_cond (BoundaryCondition): Enumeration specifying the type of boundary condition
                                                    (COULOMBIC or DIPOLAR).
            grid_spacing (delphi_real): Spacing between grid points.
            exdi (delphi_real): Exterior dielectric constant.
            indi (delphi_real): Interior dielectric constant.
            total_pve_charge (delphi_real): Total positive mobile ion charge.
            total_nve_charge (delphi_real): Total negative mobile ion charge.
            debye_length (delphi_real): Debye length of the solution.
            epkt (delphi_real): Value of r * EPKT, a scaling factor related to thermal energy.
            grid_shape (np.ndarray[delphi_int]): 3D shape of the grid (nx, ny, nz).
            grid_origin (np.ndarray[delphi_real]): Origin of the grid in 3D space.
            atoms_data (np.ndarray[delphi_real]): Array of atomic data (charge, coordinates, etc.).
            centroid_pve_charge (np.ndarray[delphi_real]): Centroid coordinates of positive mobile ions.
            centroid_nve_charge (np.ndarray[delphi_real]): Centroid coordinates of negative mobile ions.
            phimap_1d (np.ndarray[delphi_real]): 1D array to store and initialize the potential map.

        Returns:
            None
        """
        exdi_scaled = exdi / epkt
        indi_scaled = indi / epkt
        if vacuum:
            debye_length = delphi_real(ConstDelPhi.ZeroMolarSaltDebyeLength.value)

        if bound_cond.value == BoundaryCondition.COULOMBIC.value:
            if self.platform.active == "cpu":
                set_num_threads(self.platform.names["cpu"]["num_threads"])
                _cpu_setup_coulombic_boundary_condition(
                    vacuum,
                    grid_spacing,
                    exdi_scaled,
                    indi_scaled,
                    debye_length,
                    non_zero_salt,
                    grid_shape,
                    atoms_data,
                    phimap_1d,
                )
            if self.platform.active == "cuda":
                # BEGIN: CUDA call section for function: <<_cuda_setup_coulombic_boundary_condition>>
                n_blocks = (
                    self.num_grid_points + self.num_cuda_threads - 1
                ) // self.num_cuda_threads
                grid_shape_device = cuda.to_device(grid_shape)
                atoms_data_device = cuda.to_device(atoms_data)
                phimap_1d_device = cuda.to_device(phimap_1d)
                # CALL: CUDA kernel for the computation
                _cuda_setup_coulombic_boundary_condition[
                    n_blocks, self.num_cuda_threads
                ](
                    vacuum,
                    grid_spacing,
                    exdi_scaled,
                    indi_scaled,
                    debye_length,
                    non_zero_salt,
                    grid_shape_device,
                    atoms_data_device,
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
        elif bound_cond.value == BoundaryCondition.DIPOLAR.value:
            has_pve_charges, has_nve_charges = False, False
            grid_centroid_pve_charge = np.zeros(3, dtype=delphi_real)
            grid_centroid_nve_charge = np.zeros(3, dtype=delphi_real)

            if not centroid_pve_charge is None:
                has_pve_charges = True
            if not centroid_nve_charge is None:
                has_nve_charges = True
            if has_pve_charges:
                grid_centroid_pve_charge = to_grid_coords(
                    centroid_pve_charge, grid_origin, grid_spacing
                )
            if has_nve_charges:
                grid_centroid_nve_charge = to_grid_coords(
                    centroid_nve_charge, grid_origin, grid_spacing
                )
            if has_pve_charges or has_nve_charges:
                if self.platform.active == "cpu":
                    set_num_threads(self.platform.names["cpu"]["num_threads"])
                    _cpu_setup_dipolar_boundary_condition(
                        delphi_bool(vacuum),
                        delphi_bool(has_pve_charges),
                        delphi_bool(has_nve_charges),
                        grid_spacing,
                        exdi_scaled,
                        indi_scaled,
                        debye_length,
                        non_zero_salt,
                        total_pve_charge,
                        total_nve_charge,
                        grid_shape,
                        grid_centroid_pve_charge,
                        grid_centroid_nve_charge,
                        phimap_1d,
                    )
                if self.platform.active == "cuda":
                    # BEGIN: CUDA call section for function: <<_cuda_setup_dipolar_boundary_condition>>
                    n_blocks = (
                        self.num_grid_points + self.num_cuda_threads - 1
                    ) // self.num_cuda_threads
                    grid_shape_device = cuda.to_device(grid_shape)
                    grid_centroid_pve_charge_device = cuda.to_device(
                        grid_centroid_pve_charge
                    )
                    grid_centroid_nve_charge_device = cuda.to_device(
                        grid_centroid_nve_charge
                    )
                    phimap_1d_device = cuda.to_device(phimap_1d)
                    # CALL: CUDA kernel for the computation
                    _cuda_setup_dipolar_boundary_condition[
                        n_blocks, self.num_cuda_threads
                    ](
                        delphi_bool(vacuum),
                        delphi_bool(has_pve_charges),
                        delphi_bool(has_nve_charges),
                        grid_spacing,
                        exdi_scaled,
                        indi_scaled,
                        debye_length,
                        non_zero_salt,
                        total_pve_charge,
                        total_nve_charge,
                        grid_shape_device,
                        grid_centroid_pve_charge_device,
                        grid_centroid_nve_charge_device,
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
        elif bound_cond.value == BoundaryCondition.FOCUSING.value:
            scale = 1.0 / grid_spacing
            scale_parentrun = 1.0 / grid_spacing_parentrun
            grid_center = grid_origin + (grid_shape // 2) * grid_spacing
            grid_center_parentrun = (
                grid_origin_parentrun
                + (grid_shape_parentrun // 2) * grid_spacing_parentrun
            )

            _cpu_setup_focusing_boundary_condition(
                scale_parentrun=scale_parentrun,
                scale=scale,
                grid_shape_parentrun=grid_shape_parentrun,
                grid_shape=grid_shape,
                grid_center_parentrun=grid_center_parentrun,
                grid_center=grid_center,
                approx_zero=ConstDelPhi.ApproxZero.value,
                phimap_parentrun=phimap_parentrun,
                phimap_1d=phimap_1d,
            )
