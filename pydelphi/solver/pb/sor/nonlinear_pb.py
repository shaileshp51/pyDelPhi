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

from numba import set_num_threads, njit, prange, cuda

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
    ERROR,
    WARNING,
    INFO,
    DEBUG,
    TRACE,
    get_effective_verbosity,
)

_MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(_MODULE_NAME)

from pydelphi.constants import BOX_BOUNDARY, BOX_ION_ACCESSIBLE, BOX_NONE
from pydelphi.constants import ConstPhysical as Constants
from pydelphi.constants import ConstDelPhiFloats as ConstDelPhi
from pydelphi.utils.nonlinear import (
    sinh_taylor_safe,
    cu_sinh_taylor_safe,
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
    _iteration_control_check,
)

from pydelphi.solver.shared.sor.base import (
    _prepare_to_init_relaxfactor_phimap,
    _cpu_init_relaxfactor_phimap,
    _cuda_init_relaxfactor_phimap,
    _cpu_iterate_relaxation_factor,
    _cuda_iterate_relaxation_factor,
    _cpu_iterate_SOR,
    _cpu_iterate_SOR_odd_with_dphi_rmsd,
    _cuda_iterate_SOR,
    _cuda_reset_rmsd_and_dphi,
    _cuda_iterate_SOR_odd_with_dphi_rmsd,
)

from pydelphi.solver.pb.common_pb import (
    _set_gridpoint_charges,
    _cpu_mark_ion_accessible_in_boundary_flags_1d,
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


@njit(nogil=True, boundscheck=False, parallel=False, cache=True)
def _cpu_update_effective_charge_map_1d(
    nonlin_coupling_factor: float,
    grid_shape: np.ndarray,
    boundary_flags_1d: np.ndarray,  # uint8 bitmask
    phi_even_half_1d: np.ndarray,
    phi_odd_half_1d: np.ndarray,
    effective_charge_map_1d: np.ndarray,  # updated in-place
):
    """
    Nonlinear effective-charge update (faithful to v1 behavior):

        ρ_eff(r) = -κ² [sinh(φ(r)) − φ(r)]
    applied only in ion-accessible solvent voxels (κ² > 0),
    with fixed charges already initialized elsewhere.

    Boundary voxels (flag == BOX_BOUNDARY) are excluded.
    """
    nx, ny, nz = grid_shape
    num_grid_points = nx * ny * nz
    num_grid_points_half = (num_grid_points + 1) // 2

    for even_odd in (0, 1):
        phi_half_1d = phi_even_half_1d if even_odd == 0 else phi_odd_half_1d
        for ijk1d_half in prange(num_grid_points_half):
            ijk1d = ijk1d_half * 2 + even_odd
            if ijk1d < num_grid_points:
                flag = boundary_flags_1d[ijk1d]

                # skip boundary and non-solvent voxels
                if ((flag & BOX_BOUNDARY) != BOX_BOUNDARY) and (
                    flag & BOX_ION_ACCESSIBLE
                ) == BOX_ION_ACCESSIBLE:
                    phi = phi_half_1d[ijk1d_half]
                    sinh_term = sinh_taylor_safe(phi, 4.0, 5.0, 5, 0.2)
                    rho_nl = nonlin_coupling_factor * (sinh_term - phi)

                    # overwrite solvent contribution only
                    effective_charge_map_1d[ijk1d] = -rho_nl
                else:
                    pass
                    # effective_charge_map_1d[ijk1d] = charge_map_1d[ijk1d]


@cuda.jit(cache=True)
def _cuda_update_effective_charge_map_1d(
    eff_scale,
    nx,
    ny,
    nz,
    boundary_flags_1d,
    phi_even_half_1d,
    phi_odd_half_1d,
    effective_charge_map_1d,
):
    num_grid_points = nx * ny * nz
    num_grid_points_half = (num_grid_points + 1) // 2

    tid = cuda.grid(1)
    stride = cuda.gridsize(1)

    for even_odd in (0, 1):
        phi_half_1d = phi_even_half_1d if even_odd == 0 else phi_odd_half_1d
        for ijk1d_half in range(tid, num_grid_points_half, stride):
            ijk1d = ijk1d_half * 2 + even_odd
            if ijk1d < num_grid_points:
                flag = boundary_flags_1d[ijk1d]
                if ((flag & BOX_BOUNDARY) != BOX_BOUNDARY) and (
                    (flag & BOX_ION_ACCESSIBLE) == BOX_ION_ACCESSIBLE
                ):
                    phi = phi_half_1d[ijk1d_half]
                    sinh_term = sinh_taylor_safe(phi, 4.0, 5.0, 5, 0.2)
                    rho_nl = eff_scale * (sinh_term - phi)
                    effective_charge_map_1d[ijk1d] = -rho_nl


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

        # Encode the ION_EXCLUSION Binary state in boundary_gridpoints_1d
        _cpu_mark_ion_accessible_in_boundary_flags_1d(
            ion_exclusion_map_1d,
            boundary_gridpoints_1d,
        )
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
    ) -> tuple[delphi_real, delphi_real]:
        """
        Compute the spectral radius (ρ) and corresponding optimal SOR relaxation
        factor (ω_SOR) for the linear Poisson–Boltzmann iteration operator.

        Steps:
            1. Estimate the spectral radius ρ using a sine-based power iteration.
            2. Convert ρ to ω_SOR using:
                   ω_SOR = 2 / (1 + sqrt(1 - ρ²))

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
            tuple[delphi_real, delphi_real]:
                (spectral_radius, omega_sor)

        Notes:
            - ω_SOR directly controls over-relaxation in the SOR update rule:
                  φ_new = (1 - ω_SOR) * φ_old + ω_SOR * φ_GS
            - The caller is responsible for logging or printing both ρ and ω_SOR.
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

        omega_sor = delphi_real(2.0 / (1.0 + math.sqrt(1.0 - spectral_radius)))

        return spectral_radius, omega_sor

    def _cuda_solve_nonlinear_pb(
        self,
        num_cuda_threads,
        nonlinear_itr_block_size,
        itr_block_size,
        max_linear_iters,
        max_nonlinear_iters,
        nonlinear_coupling_max_dphi,
        coupling_steps,
        manual_relaxation_value,
        non_zero_salt,
        max_rms,
        max_dphi,
        check_dphi,
        omega_sor,
        kappa_x_h2,
        approx_zero,
        grid_shape,
        epsmap_midpoints_1d,
        eps_mid_sum_plus_salt_1d,
        boundary_flags_1d,
        effective_charge_map_1d,
        phi_even_half_1d,
        phi_odd_half_1d,
        phimap_current_1d,
    ):
        """
        Nonlinear PBE solver (CUDA v2).
        Mirrors CPU version but operates entirely on device-resident arrays.
        """
        if coupling_steps > 0:
            vprint(INFO, _VERBOSITY, "    PBE> Running nonlinear CUDA solver...")

        # --- Device setup ---
        # Read only device arrays
        grid_shape_device = cuda.to_device(grid_shape)
        epsmap_midpoints_1d_device = cuda.to_device(epsmap_midpoints_1d)
        eps_mid_sum_plus_salt_1d_device = cuda.to_device(eps_mid_sum_plus_salt_1d)
        boundary_flags_1d_device = cuda.to_device(boundary_flags_1d)

        # Read write device arrays
        effective_charge_map_1d_device = cuda.to_device(effective_charge_map_1d)
        phi_even_half_1d_device = cuda.to_device(phi_even_half_1d)
        phi_odd_half_1d_device = cuda.to_device(phi_odd_half_1d)

        nx, ny, nz = grid_shape
        num_grid_points = nx * ny * nz
        num_grid_points_half = (num_grid_points + 1) // 2
        n_blocks = (num_grid_points_half + num_cuda_threads - 1) // num_cuda_threads

        effective_omega_sor = omega_sor
        converged_nonlinear = False

        for nonlinear_iter in range(coupling_steps + 1):
            effective_check_dphi = (
                check_dphi if nonlinear_iter in (0, coupling_steps) else True
            )
            current_chi = (
                min(1.0, nonlinear_iter / coupling_steps) if coupling_steps > 0 else 1.0
            )
            effective_max_dphi = (
                max_dphi
                if nonlinear_iter in (0, coupling_steps)
                else nonlinear_coupling_max_dphi
            )
            effective_itr_block = (
                itr_block_size
                if nonlinear_iter in (0, coupling_steps)
                else nonlinear_itr_block_size
            )
            effective_max_iter = (
                max_linear_iters
                if nonlinear_iter in (0, coupling_steps)
                else max_nonlinear_iters
            )
            if nonlinear_iter == 0:
                if coupling_steps > 0:
                    vprint(
                        INFO,
                        _VERBOSITY,
                        f"\n    PBE> Nonlinear running initial linear step",
                    )
                else:
                    vprint(
                        INFO,
                        _VERBOSITY,
                        f"\n    PBE> Running SOR solver for linear PB",
                    )

            # --- Nonlinear update (on GPU) ---
            if nonlinear_iter > 0:
                effective_omega_sor = omega_sor
                omega_source = "auto"

                if manual_relaxation_value != 0.0:
                    effective_omega_sor = manual_relaxation_value
                    omega_source = "manual"

                vprint(
                    INFO,
                    _VERBOSITY,
                    f"\n    PBE> Nonlinear step {nonlinear_iter}, χ={current_chi:.3f}, "
                    f"Δφ_tol={effective_max_dphi:10.4e}, enforce_Δφ={effective_check_dphi}, ω_SOR={effective_omega_sor:.6f} ({omega_source})",
                )

                if non_zero_salt:
                    nonlin_coupling_factor = current_chi * kappa_x_h2
                    _cuda_update_effective_charge_map_1d[n_blocks, num_cuda_threads](
                        nonlin_coupling_factor,
                        grid_shape[0],
                        grid_shape[1],
                        grid_shape[2],
                        boundary_flags_1d_device,
                        phi_even_half_1d_device,
                        phi_odd_half_1d_device,
                        effective_charge_map_1d_device,
                    )

            # --- Linear subsolve (CUDA SOR) ---
            rmsd, max_change, num_iter = self._cuda_solve_linear_pb_sor(
                num_cuda_threads,
                phi_even_half_1d_device,
                phi_odd_half_1d_device,
                grid_shape_device,
                epsmap_midpoints_1d_device,
                eps_mid_sum_plus_salt_1d_device,
                boundary_flags_1d_device,
                effective_charge_map_1d_device,
                effective_omega_sor,
                approx_zero,
                max_rms,
                effective_max_dphi,
                effective_check_dphi,
                effective_max_iter,
                effective_itr_block,
                verbose=True,
            )

            if (
                max_change < nonlinear_coupling_max_dphi
                and abs(current_chi - 1.0) < approx_zero
            ):
                if coupling_steps > 0:
                    vprint(INFO, _VERBOSITY, "    PBE> Nonlinear solver converged.")
                converged_nonlinear = True
                break

        # --- Finalize ---
        phi_even_half_1d_device.copy_to_host(phi_even_half_1d)
        phi_odd_half_1d_device.copy_to_host(phi_odd_half_1d)
        _copy_to_full(phimap_current_1d, phi_even_half_1d, offset=0, skip=2)
        _copy_to_full(phimap_current_1d, phi_odd_half_1d, offset=1, skip=2)

        if not converged_nonlinear:
            vprint(
                WARNING,
                _VERBOSITY,
                "    PBE> WARNING: Nonlinear solver did not converge (max steps reached).",
            )

        return phimap_current_1d, converged_nonlinear

    def _cpu_solve_nonlinear_pb(
        self,
        num_cpu_threads,
        nonlinear_itr_block_size,
        itr_block_size,
        max_linear_iters,
        max_nonlinear_iters,
        nonlinear_coupling_max_dphi,
        coupling_steps,
        manual_relaxation_value,
        non_zero_salt,
        max_rms,
        max_dphi,
        enforce_dphi,
        omega_sor,
        kappa_x_h2,
        approx_zero,
        grid_shape,
        epsmap_midpoints_1d,
        eps_mid_sum_plus_salt_1d,
        boundary_flags_1d,
        effective_charge_map_1d,
        phi_even_half_1d,
        phi_odd_half_1d,
        phimap_current_1d,
    ):
        effective_omega_sor = omega_sor
        # --- 5. Nonlinear coupling loop ---
        for nonlinear_iter in range(coupling_steps + 1):
            effective_enforce_dphi = (
                enforce_dphi if nonlinear_iter in (0, coupling_steps) else True
            )
            current_chi = (
                min(1.0, nonlinear_iter / coupling_steps) if coupling_steps > 0 else 1.0
            )
            effective_max_dphi = (
                max_dphi
                if nonlinear_iter in (0, coupling_steps)
                else nonlinear_coupling_max_dphi
            )
            effective_itr_block = (
                itr_block_size
                if nonlinear_iter in (0, coupling_steps)
                else nonlinear_itr_block_size
            )
            effective_max_iter = (
                max_linear_iters
                if nonlinear_iter in (0, coupling_steps)
                else max_nonlinear_iters
            )
            if nonlinear_iter == 0:
                vprint(
                    INFO,
                    _VERBOSITY,
                    f"\n    PBE> Nonlinear running initial linear step",
                )

            if nonlinear_iter > 0:
                effective_omega_sor = omega_sor
                omega_source = "auto"

                if manual_relaxation_value != 0.0:
                    effective_omega_sor = manual_relaxation_value
                    omega_source = "manual"

                vprint(
                    INFO,
                    _VERBOSITY,
                    f"\n    PBE> Nonlinear step {nonlinear_iter}, χ={current_chi:.3f}, "
                    f"ω_SOR={effective_omega_sor:.6f} ({omega_source})",
                )

                if non_zero_salt:
                    _cpu_update_effective_charge_map_1d(
                        nonlin_coupling_factor=current_chi * kappa_x_h2,
                        grid_shape=grid_shape,
                        boundary_flags_1d=boundary_flags_1d,
                        phi_even_half_1d=phi_even_half_1d,
                        phi_odd_half_1d=phi_odd_half_1d,
                        effective_charge_map_1d=effective_charge_map_1d,
                    )
                    # np.save(
                    #     f"effective_charge_map_1d_v2_{nonlinear_iter}.npy",
                    #     effective_charge_map_1d,
                    # )

            # --- Linear subsolve (on half arrays) ---
            rmsd, max_change, num_iter = self._cpu_solve_linear_pb_sor(
                num_cpu_threads,
                phi_even_half_1d,
                phi_odd_half_1d,
                grid_shape,
                epsmap_midpoints_1d,
                eps_mid_sum_plus_salt_1d,
                boundary_flags_1d,
                effective_charge_map_1d,
                effective_omega_sor,
                approx_zero,
                max_rms,
                effective_max_dphi,
                enforce_dphi=effective_enforce_dphi,
                max_linear_iters=effective_max_iter,
                itr_block_size=effective_itr_block,
                verbose=True,
            )

            # np.save(
            #     f"phi_even_half_1d_v2_after{nonlinear_iter}.npy",
            #     phi_even_half_1d,
            # )
            # np.save(
            #     f"phi_odd_half_1d_v2_after{nonlinear_iter}.npy",
            #     phi_odd_half_1d,
            # )
            #
            # np.save(
            #     f"phimap_current_1d_v2_after{nonlinear_iter}.npy",
            #     phimap_current_1d,
            # )

            if (
                max_change < nonlinear_coupling_max_dphi
                and abs(current_chi - 1.0) < approx_zero
            ):
                vprint(INFO, _VERBOSITY, "    PBE> Nonlinear solver converged.")
                break

        # --- 6. Copy half → full once at the end ---
        _copy_to_full(phimap_current_1d, phi_even_half_1d, offset=0, skip=2)
        _copy_to_full(phimap_current_1d, phi_odd_half_1d, offset=1, skip=2)

        return phimap_current_1d

    def _cuda_solve_linear_pb_sor(
        self,
        num_cuda_threads,
        phi_even_half_1d_device,
        phi_odd_half_1d_device,
        grid_shape_device,
        epsilon_map_midpoints_1d_device,
        epsilon_sum_neighbors_plus_salt_screening_1d_device,
        boundary_flags_1d_device,
        charge_map_1d_device,
        omega_sor,
        approx_zero,
        rmsd_tol,
        dphi_tol,
        enforce_dphi,
        max_linear_iters,
        itr_block_size,
        verbose=True,
    ):
        """
        Faithful linear SOR solver (CUDA v2).
        Mirrors CPU orchestration exactly using phi_half_read / phi_half_write aliases.
        - Alternates even/odd half-grid updates.
        - RMSD/Δφ computed *only* on odd iterations that are also the last in a block.
        - Each iteration uses consistent read/write buffers to match CPU version.
        """
        # --- Setup and allocations ---
        nx, ny, nz = grid_shape_device.copy_to_host()
        num_grid_points_half = (nx * ny * nz + 1) // 2
        n_blocks = (num_grid_points_half + num_cuda_threads - 1) // num_cuda_threads

        sum_squared_host = np.zeros(1, dtype=np.float64)
        max_delta_phi_host = np.zeros(1, dtype=np.float64)
        sum_squared_device = cuda.to_device(sum_squared_host)
        max_delta_phi_device = cuda.to_device(max_delta_phi_host)

        rmsd_buffer = np.zeros(10, dtype=np.float64)  # rolling RMSD buffer
        ptr = 0
        wrapped = False

        total_iter = 0

        vprint(
            INFO,
            _VERBOSITY,
            "    PBE> | #Iteration |    RMSD    |  Max(dPhi) | Time (seconds) |",
        )

        # --- Iteration blocks ---
        for iter_block_start in range(0, max_linear_iters, itr_block_size):
            tic_block = time.perf_counter()

            for itr_in_block in range(itr_block_size):
                for even_odd in (0, 1):
                    is_last_iter_of_block = itr_in_block == itr_block_size - 1
                    is_last_overall = (
                        iter_block_start + itr_in_block + 1 >= max_linear_iters
                    )

                    # --- Select read/write halves (identical to CPU version) ---
                    if even_odd == 0:
                        phi_half_read = phi_odd_half_1d_device
                        phi_half_write = phi_even_half_1d_device
                    else:
                        phi_half_read = phi_even_half_1d_device
                        phi_half_write = phi_odd_half_1d_device

                    # Determine if this iteration should compute RMSD
                    is_last_odd_iter_of_block = (even_odd == 1) and (
                        is_last_iter_of_block or is_last_overall
                    )

                    # --- Perform iteration ---
                    if is_last_odd_iter_of_block:
                        # Reset accumulators on device before RMSD reduction
                        _cuda_reset_rmsd_and_dphi[1, 1](
                            sum_squared_device, max_delta_phi_device
                        )

                        _cuda_iterate_SOR_odd_with_dphi_rmsd[
                            n_blocks, num_cuda_threads
                        ](
                            even_odd,
                            omega_sor,
                            approx_zero,
                            grid_shape_device,
                            phi_half_read,
                            phi_half_write,
                            epsilon_map_midpoints_1d_device,
                            epsilon_sum_neighbors_plus_salt_screening_1d_device,
                            boundary_flags_1d_device,
                            charge_map_1d_device,
                            sum_squared_device,
                            max_delta_phi_device,
                        )
                        cuda.synchronize()

                        # Retrieve block RMSD/Δφ only after last odd iteration
                        sum_squared_device.copy_to_host(sum_squared_host)
                        max_delta_phi_device.copy_to_host(max_delta_phi_host)

                    else:
                        _cuda_iterate_SOR[n_blocks, num_cuda_threads](
                            even_odd,
                            omega_sor,
                            approx_zero,
                            grid_shape_device,
                            phi_half_read,
                            phi_half_write,
                            epsilon_map_midpoints_1d_device,
                            epsilon_sum_neighbors_plus_salt_screening_1d_device,
                            boundary_flags_1d_device,
                            charge_map_1d_device,
                        )
                        cuda.synchronize()

                total_iter += 1  # Match CPU per-half-pair increment

            # --- End of block: compute timing and report ---
            toc_block = time.perf_counter()
            rmsd = math.sqrt(sum_squared_host[0] / num_grid_points_half)
            max_delta_phi = abs(max_delta_phi_host[0])

            vprint(
                INFO,
                _VERBOSITY,
                f"    PBE> | {total_iter:10d} | {rmsd:10.4e} | "
                f"{max_delta_phi:10.4e} | {toc_block - tic_block:14.6f} |",
            )

            # --- unified iteration control ---
            stop, status, ptr, wrapped = _iteration_control_check(
                rmsd_buffer,
                ptr,
                wrapped,
                rmsd,
                max_delta_phi,
                rmsd_tol,
                dphi_tol,
                enforce_dphi,
                max_linear_iters,
                total_iter,
                disable_stagnation_check=False,
            )

            if stop:
                if status == 1:
                    vprint(
                        INFO,
                        _VERBOSITY,
                        "    PBE> Convergence reached (RMSD/ΔΦ thresholds satisfied)",
                    )
                elif status == 2:
                    vprint(
                        INFO,
                        _VERBOSITY,
                        "    PBE> Convergence reached (stagnation plateau, relaxed criterion)",
                    )
                elif status == 3:
                    vprint(
                        INFO,
                        _VERBOSITY,
                        "    PBE> Divergence detected (non-finite residuals)",
                    )
                break

        return rmsd, max_delta_phi, total_iter

    def _cpu_solve_linear_pb_sor(
        self,
        num_cpu_threads: delphi_int,
        phi_map_even_half_1d: np.ndarray,
        phi_map_odd_half_1d: np.ndarray,
        grid_shape: np.ndarray,
        epsilon_map_midpoints_1d: np.ndarray,
        epsilon_sum_neighbors_plus_salt_screening_1d: np.ndarray,
        is_boundary_gridpoint_1d: np.ndarray,
        charge_map_1d: np.ndarray,
        omega: delphi_real,
        approx_zero: delphi_real,
        rmsd_tol: delphi_real,
        max_change_tol: delphi_real,
        enforce_dphi: delphi_bool,
        max_linear_iters: delphi_int,
        itr_block_size: delphi_int,
        verbose: bool = True,
    ):
        """
        Faithful linear SOR solver:
        - Alternates even/odd half-grid updates.
        - RMSD/Δφ computed *only* on odd passes that are also the last in a block.
        - Preserves original timing/printing structure.
        """
        total_sum_sq = 0.0
        global_max_change = 0.0
        num_iter = 0

        rmsd_buffer = np.zeros(10, dtype=np.float64)  # rolling RMSD buffer
        ptr = 0
        wrapped = False

        vprint(
            INFO,
            _VERBOSITY,
            "    PBE> | #Iteration |    RMSD    |  Max(dPhi) | Time (seconds) |",
        )

        # --- Iteration blocks ---
        for iter_block_start in range(0, max_linear_iters, itr_block_size):
            tic_block = time.perf_counter()

            for itr_in_block in range(itr_block_size):
                for even_odd in (0, 1):
                    is_last_iter_of_block = itr_in_block == itr_block_size - 1
                    is_last_overall = (
                        iter_block_start + itr_in_block + 1 >= max_linear_iters
                    )
                    # Select read/write buffers
                    # Important: even iterations update even points using odd neighbor half; do not invert.
                    if even_odd == 0:
                        read_half = phi_map_odd_half_1d
                        write_half = phi_map_even_half_1d
                    else:
                        # Note: not even is odd
                        read_half = phi_map_even_half_1d
                        write_half = phi_map_odd_half_1d

                    is_last_odd_iter_of_block = (
                        is_last_iter_of_block or is_last_overall
                    ) and (even_odd == 1)

                    # --- Perform iteration ---
                    if is_last_odd_iter_of_block:
                        # Odd + boundary of block → compute RMSD/Δφ
                        total_sum_sq, global_max_change = (
                            _cpu_iterate_SOR_odd_with_dphi_rmsd(
                                num_cpu_threads,
                                even_odd,
                                omega,
                                approx_zero,
                                grid_shape,
                                read_half,
                                write_half,
                                epsilon_map_midpoints_1d,
                                epsilon_sum_neighbors_plus_salt_screening_1d,
                                is_boundary_gridpoint_1d,
                                charge_map_1d,
                            )
                        )
                    else:
                        _cpu_iterate_SOR(
                            even_odd,
                            omega,
                            approx_zero,
                            grid_shape,
                            read_half,
                            write_half,
                            epsilon_map_midpoints_1d,
                            epsilon_sum_neighbors_plus_salt_screening_1d,
                            is_boundary_gridpoint_1d,
                            charge_map_1d,
                        )

                num_iter += 1

            # --- End of block: compute timing and report ---
            toc_block = time.perf_counter()
            block_time = toc_block - tic_block
            num_grid_points_half = (
                grid_shape[0] * grid_shape[1] * grid_shape[2] + 1
            ) // 2
            rmsd = math.sqrt(total_sum_sq / num_grid_points_half)

            vprint(
                INFO,
                _VERBOSITY,
                f"    PBE> | {num_iter:10d} | {rmsd:10.4e} | "
                f"{global_max_change:10.4e} | {block_time:14.6f} |",
            )

            # --- unified iteration control ---
            stop, status, ptr, wrapped = _iteration_control_check(
                rmsd_buffer,
                ptr,
                wrapped,
                rmsd,
                global_max_change,
                rmsd_tol,
                max_change_tol,
                enforce_dphi,
                max_linear_iters,
                num_iter,
                disable_stagnation_check=False,
            )

            if stop:
                if status == 1:
                    vprint(
                        INFO,
                        _VERBOSITY,
                        "    PBE> Convergence reached (RMSD/ΔΦ thresholds satisfied)",
                    )
                elif status == 2:
                    vprint(
                        INFO,
                        _VERBOSITY,
                        "    PBE> Convergence reached (stagnation plateau, relaxed criterion)",
                    )
                elif status == 3:
                    vprint(
                        INFO,
                        _VERBOSITY,
                        "    PBE> Divergence detected (non-finite residuals)",
                    )
                break

        # --- Finalization ---
        return rmsd, global_max_change, num_iter

    def solve_nonlinear_pb(
        self,
        vacuum,
        bound_cond,
        dielectric_model,
        gaussian_exponent,
        nonlinear_itr_block_size,
        itr_block_size,
        max_linear_iters,
        max_nonlinear_iters,
        nonlinear_coupling_max_dphi,
        coupling_steps,
        manual_relaxation_value,
        scale,
        scale_parentrun,
        exdi,
        indi,
        debye_length,
        non_zero_salt,
        total_pve_charge,
        total_nve_charge,
        max_rms,
        max_dphi,
        check_dphi,
        epkt,
        approx_zero,
        grid_shape,
        grid_origin,
        grid_shape_parentrun,
        grid_origin_parentrun,
        atoms_data,
        density_map_1d,
        ion_exclusion_map_1d,
        epsilon_map_1d,
        epsmap_midpoints_1d,
        centroid_pve_charge,
        centroid_nve_charge,
        charged_gridpoints_1d,
        phimap_parentrun,
    ):
        """
        CPU nonlinear PBE orchestrator.
        Copies data to half-arrays once at start and back once at the end.
        Uses _solve_linear_pb_sor_cpu_v2 for all linear sub-solves.
        """
        # --- 1. Setup constants ---
        grid_spacing = 1.0 / scale
        grid_spacing_sq = grid_spacing * grid_spacing
        kappa_sq = (exdi / debye_length**2) if non_zero_salt else delphi_real(0.0)
        kappa_x_h2 = kappa_sq * grid_spacing_sq
        is_gaussian = dielectric_model.int_value == DielectricModel.GAUSSIAN.int_value

        num_cpu_threads = self.platform.names["cpu"]["num_threads"]

        # --- 2. Charge map and boundary setup ---
        charge_map_1d = np.zeros(self.num_grid_points, dtype=delphi_real)
        _set_gridpoint_charges(grid_shape, charged_gridpoints_1d, charge_map_1d)

        phimap_current_1d = np.zeros(self.num_grid_points, dtype=delphi_real)
        self._setup_boundary_condition(
            vacuum=vacuum,
            bound_cond=bound_cond,
            grid_spacing=grid_spacing,
            grid_spacing_parentrun=1.0 / scale_parentrun,
            exdi=exdi,
            indi=indi,
            total_pve_charge=total_pve_charge,
            total_nve_charge=total_nve_charge,
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

        # --- 3. Linear operator setup ---
        eps_mid_sum_plus_salt_1d = np.zeros(self.num_grid_points, dtype=delphi_real)
        # boundary_flags_1d = np.zeros(self.num_grid_points, dtype=delphi_bool)
        boundary_flags_1d = np.full(self.num_grid_points, BOX_NONE, dtype=np.uint8)

        self._prepare_to_iterate(
            vacuum,
            exdi,
            is_gaussian,
            grid_spacing,
            debye_length,
            non_zero_salt,
            delphi_real(Constants.FourPi.value),
            epkt,
            grid_shape,
            ion_exclusion_map_1d,
            epsilon_map_1d,
            epsmap_midpoints_1d,
            charge_map_1d,
            eps_mid_sum_plus_salt_1d,
            boundary_flags_1d,
        )

        spectral_radius, omega_sor = self._calc_relaxation_factor(
            1,
            grid_shape,
            np.zeros(3, dtype=delphi_bool),
            epsmap_midpoints_1d,
            eps_mid_sum_plus_salt_1d,
            boundary_flags_1d,
        )

        vprint(
            INFO,
            _VERBOSITY,
            f"\n    PBE> Spectral radius (ρ) = {spectral_radius:.6f}, Relaxation factor (ω_SOR) = {omega_sor:.6f}",
        )

        # --- 4. Persistent buffers ---
        effective_charge_map_1d = np.copy(charge_map_1d)

        # Allocate half-grid buffers only once
        num_points_half = (self.num_grid_points + 1) // 2
        phi_even_half_1d = np.zeros(num_points_half, dtype=delphi_real)
        phi_odd_half_1d = np.zeros(num_points_half, dtype=delphi_real)

        # Copy full → half safely via helper  once at the start
        _copy_to_sample(phi_even_half_1d, phimap_current_1d, offset=0, skip=2)
        _copy_to_sample(phi_odd_half_1d, phimap_current_1d, offset=1, skip=2)

        # np.save(
        #     f"boundary_flags_1d_v2.npy",
        #     boundary_flags_1d,
        # )
        # np.save(
        #     f"charge_map_1d_v2.npy",
        #     charge_map_1d,
        # )

        if self.platform.active == "cpu":
            self._cpu_solve_nonlinear_pb(
                num_cpu_threads,
                nonlinear_itr_block_size,
                itr_block_size,
                max_linear_iters,
                max_nonlinear_iters,
                nonlinear_coupling_max_dphi,
                coupling_steps,
                manual_relaxation_value,
                non_zero_salt,
                max_rms,
                max_dphi,
                check_dphi,
                omega_sor,
                kappa_x_h2,
                approx_zero,
                grid_shape,
                epsmap_midpoints_1d,
                eps_mid_sum_plus_salt_1d,
                boundary_flags_1d,
                effective_charge_map_1d,
                phi_even_half_1d,
                phi_odd_half_1d,
                phimap_current_1d,
            )
        elif self.platform.active == "cuda":
            self._cuda_solve_nonlinear_pb(
                self.num_cuda_threads,
                nonlinear_itr_block_size,
                itr_block_size,
                max_linear_iters,
                max_nonlinear_iters,
                nonlinear_coupling_max_dphi,
                coupling_steps,
                manual_relaxation_value,
                non_zero_salt,
                max_rms,
                max_dphi,
                check_dphi,
                omega_sor,
                kappa_x_h2,
                approx_zero,
                grid_shape,
                epsmap_midpoints_1d,
                eps_mid_sum_plus_salt_1d,
                boundary_flags_1d,
                effective_charge_map_1d,
                phi_even_half_1d,
                phi_odd_half_1d,
                phimap_current_1d,
            )
        else:
            vprint(ERROR, _VERBOSITY, f"Unknown platform: {self.platform.active}")

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
