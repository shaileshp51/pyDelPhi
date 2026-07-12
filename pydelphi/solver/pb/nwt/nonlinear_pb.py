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

"""
This module implements the Non-linear Poisson-Boltzmann Equation (NLPBE) solver,
supporting both CPU and CUDA (GPU) computation platforms. It is designed to
calculate the electrostatic potential within a molecular system.

The solver utilizes a Newton-like iterative method for solving the NLPBE,
which accounts for charge-charge interactions, dielectric boundaries, and
mobile ion distributions in an implicit solvent model.

Key functionalities include:
- Initialization of solver parameters and data structures based on grid
  shape, precision, and chosen computation platform (CPU or CUDA).
- Preparation of grid-related data, such as charge maps, dielectric maps,
  and boundary conditions, optimizing for the selected platform.
- Execution of the iterative solver, which updates the electrostatic
  potential across the grid until convergence criteria are met.
- Support for various boundary conditions: Coulombic, Dipolar, and Focusing.
- Integration with Numba for high-performance CPU execution and Numba CUDA
  for GPU acceleration.
- Verbosity control to provide detailed output during the solving process.

The module handles precision (single or double) dynamically based on the
global runtime configuration.
"""

import time
import numpy as np

from numba import set_num_threads, njit

from pydelphi.foundation.enums import (
    Precision,
    BoundaryCondition,
    VerbosityLevel,
    DielectricModel,
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
    _iteration_control_check,
)

from pydelphi.solver.pb.nwt.base import (
    _cpu_iterate_nwt,
    _cpu_iterate_nwt_with_dphi_rmsd,
    _cuda_iterate_nwt,
    _cuda_reset_rmsd_and_dphi,
    _cuda_iterate_nwt_with_dphi_rmsd,
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


class NLNewtonPBESolver:
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

    def _prepare_charge_neighbor_epsmids_sum_to_iterate(
        self,
        vacuum: delphi_bool,
        exdi: delphi_real,
        grid_spacing: delphi_real,
        debye_length: delphi_real,
        non_zero_salt: delphi_bool,
        four_pi: delphi_real,
        epkt: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        epsmap_midpoints_1d: np.ndarray[delphi_real],
        charge_map_1d: np.ndarray[delphi_real],
        eps_midpoint_neighs_sum_only_1d: np.ndarray[delphi_real],
        ion_exclusion_map_1d: np.ndarray[delphi_real],
        boundary_flags_1d: np.ndarray[delphi_bool],
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
            grid_spacing (delphi_real): Distance between consecutive grid points.
            debye_length (delphi_real): Debye length in the medium.
            non_zero_salt (delphi_bool): Whether the salt concentration is non-zero.
            four_pi (delphi_real): Constant 4 * pi.
            epkt (delphi_real): Constant related to scaling factor `r * EPKT`.
            grid_shape (np.ndarray[delphi_int]): Shape of the grid (in x, y, z dimensions).
            epsmap_midpoints_1d (np.ndarray[delphi_real]): 1D array of dielectric constants at grid midpoints.
            charge_map_1d (np.ndarray[delphi_real]): 1D array holding the charge distribution.
            eps_midpoint_neighs_sum_only_1d (np.ndarray[delphi_real]): 1D array to hold the sum of dielectric constants
                                                                from neighboring grid midpoints.
            ion_exclusion_map_1d (np.ndarray[delphi_bool]): 1D real array data range [0,1] marking the ion excluded regions.
            boundary_flags_1d (np.ndarray[delphi_bool]): 1D boolean array marking the boundary grid points.

        Returns:
            None
        """
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
                eps_midpoint_neighs_sum_only_1d,
                boundary_flags_1d,
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
            eps_midpoint_neighs_sum_only_1d_device = cuda.to_device(
                eps_midpoint_neighs_sum_only_1d
            )
            boundary_gridpoints_1d_device = cuda.to_device(boundary_flags_1d)

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
                eps_midpoint_neighs_sum_only_1d_device,
                boundary_gridpoints_1d_device,
            )

            # Transfer the computed results back from GPU to host memory
            charge_map_1d_device.copy_to_host(charge_map_1d)
            eps_midpoint_neighs_sum_only_1d_device.copy_to_host(
                eps_midpoint_neighs_sum_only_1d
            )
            boundary_gridpoints_1d_device.copy_to_host(boundary_flags_1d)

            # Clear GPU memory by setting references to None
            grid_shape_device = None
            surface_map_1d_device = None
            epsmap_midpoints_1d_device = None
            charge_map_1d_device = None
            eps_midpoint_neighs_sum_only_1d_device = None
            boundary_gridpoints_1d_device = None

        # Encode the ION_EXCLUSION Binary state in boundary_gridpoints_1d
        _cpu_mark_ion_accessible_in_boundary_flags_1d(
            ion_exclusion_map_1d,
            boundary_flags_1d,
        )

    def _cpu_solve_nonlinear_pb_nwt(
        self,
        vacuum: delphi_bool,
        phimap_current_1d: np.ndarray[delphi_real],
        non_zero_salt: delphi_bool,
        approx_zero: delphi_real,
        omega_adaptive: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        salt_ions_solvation_penalty_map_1d: np.ndarray[delphi_real],
        epsilon_map_midpoints_1d: np.ndarray[delphi_real],
        epsilon_sum_neighbors_sum_only_1d: np.ndarray[delphi_real],
        boundary_flags_1d: np.ndarray[delphi_bool],
        charge_map_1d: np.ndarray[delphi_real],
        ion_exclusion_map_1d: np.ndarray[delphi_real],
        itr_block_size: delphi_int,
        max_nonlinear_iters: delphi_int,
        rms_tol: delphi_real,
        dphi_tol: delphi_real,
        check_dphi: delphi_bool,
        num_cpu_threads: delphi_int,
        verbose: bool = True,
    ):
        """
        CPU orchestrator for nonlinear Newton-like PB solver (v2).

        Uses even/odd half-grid alternation identical to CUDA version.
        RMSD/Δφ are computed only on the last odd iteration of each block.

        Args:
            All arguments mirror those of the CUDA version.

        Returns:
            (rmsd, max_dphi, total_iter)
        """
        nx, ny, nz = grid_shape
        num_grid_points_half = (nx * ny * nz + 1) // 2
        total_iter = 0

        rmsd_buffer = np.zeros(10, dtype=np.float64)  # rolling RMSD buffer
        ptr = 0
        wrapped = False
        stop_iters = False

        # Half maps
        phi_even_1d = np.zeros(num_grid_points_half, dtype=delphi_real)
        phi_odd_1d = np.zeros(num_grid_points_half, dtype=delphi_real)
        _copy_to_sample(phi_even_1d, phimap_current_1d, 0, 2)
        _copy_to_sample(phi_odd_1d, phimap_current_1d, 1, 2)

        vprint(
            INFO,
            _VERBOSITY,
            f"\n    PBE> Running nonlinear CPU solver (NWT) with ω_adaptive={omega_adaptive:.3f}",
        )
        vprint(
            INFO,
            _VERBOSITY,
            "    PBE> | #Iteration |    RMSD    |  Max(dPhi) | Time (seconds) |",
        )

        while not stop_iters and total_iter < max_nonlinear_iters:
            tic_block = time.perf_counter()
            total_sum_sq = 0.0
            global_max_dphi = 0.0

            for itr_in_block in range(itr_block_size):
                for even_odd in (0, 1):
                    is_last_iter = itr_in_block == itr_block_size - 1
                    is_last_overall = (
                        total_iter + itr_in_block + 1 >= max_nonlinear_iters
                    )
                    read_half = phi_odd_1d if even_odd == 0 else phi_even_1d
                    write_half = phi_even_1d if even_odd == 0 else phi_odd_1d

                    if (even_odd == 1) and (is_last_iter or is_last_overall):
                        total_sum_sq, global_max_dphi = _cpu_iterate_nwt_with_dphi_rmsd(
                            vacuum,
                            even_odd,
                            non_zero_salt,
                            approx_zero,
                            omega_adaptive,
                            grid_shape,
                            read_half,
                            write_half,
                            salt_ions_solvation_penalty_map_1d,
                            epsilon_map_midpoints_1d,
                            epsilon_sum_neighbors_sum_only_1d,
                            boundary_flags_1d,
                            charge_map_1d,
                            ion_exclusion_map_1d,
                            1,  # num_cpu_threads,
                        )
                    else:
                        _cpu_iterate_nwt(
                            vacuum,
                            even_odd,
                            non_zero_salt,
                            approx_zero,
                            omega_adaptive,
                            grid_shape,
                            read_half,
                            write_half,
                            salt_ions_solvation_penalty_map_1d,
                            epsilon_map_midpoints_1d,
                            epsilon_sum_neighbors_sum_only_1d,
                            boundary_flags_1d,
                            charge_map_1d,
                            ion_exclusion_map_1d,
                        )

            total_iter += itr_block_size
            toc_block = time.perf_counter()
            rmsd = math.sqrt(total_sum_sq / num_grid_points_half)
            max_dphi = abs(global_max_dphi)

            vprint(
                INFO,
                _VERBOSITY,
                f"    PBE> | {total_iter:10d} | {rmsd:10.4e} | {max_dphi:10.4e} | {toc_block - tic_block:14.6f} |",
            )

            # --- unified iteration control ---
            stop_iters, status, ptr, wrapped = _iteration_control_check(
                rmsd_buffer,
                ptr,
                wrapped,
                rmsd,
                max_dphi,
                rms_tol,
                dphi_tol,
                check_dphi,
                max_nonlinear_iters,
                total_iter,
                disable_stagnation_check=False,
            )

            if stop_iters:
                if status == 1 and verbose:
                    vprint(
                        INFO,
                        _VERBOSITY,
                        "    PBE> Convergence reached (RMSD/ΔΦ thresholds satisfied)",
                    )
                elif status == 2 and verbose:
                    vprint(
                        INFO,
                        _VERBOSITY,
                        "    PBE> Convergence reached (stagnation plateau, relaxed criterion)",
                    )
                elif status == 3 and verbose:
                    vprint(
                        INFO,
                        _VERBOSITY,
                        "    PBE> Divergence detected (non-finite residuals)",
                    )
                break

        # Reconstruct full phimap from half phimaps
        _copy_to_full(phimap_current_1d, phi_even_1d, 0, 2)
        _copy_to_full(phimap_current_1d, phi_odd_1d, 1, 2)

        return rmsd, dphi_tol, total_iter

    def _cuda_solve_nonlinear_pb_nwt(
        self,
        vacuum: delphi_bool,
        phimap_current_1d: np.ndarray[delphi_real],
        non_zero_salt: delphi_bool,
        approx_zero: delphi_real,
        omega_adaptive: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        salt_ions_solvation_penalty_map_1d: np.ndarray[delphi_real],
        epsmap_midpoints_1d: np.ndarray[delphi_real],
        eps_midpoint_neighs_sum_only_1d: np.ndarray[delphi_real],
        boundary_flags_1d: np.ndarray[delphi_bool],
        charge_map_1d: np.ndarray[delphi_real],
        ion_exclusion_map_1d: np.ndarray[delphi_real],
        itr_block_size: delphi_int,
        max_nonlinear_iters: delphi_int,
        rms_tol: delphi_real,
        dphi_tol: delphi_real,
        check_dphi: delphi_bool,
        num_cuda_threads: delphi_int,
        verbosity_level: int,
    ):
        """
        CUDA orchestrator for nonlinear Newton-like PB solver (v2).
        Mirrors the flow of the CPU version exactly.
        Uses persistent device arrays and computes RMSD/Δφ only on
        the last odd iteration of each iteration block.
        """
        nx, ny, nz = grid_shape
        num_grid_points_half = (nx * ny * nz + 1) // 2
        n_blocks = (num_grid_points_half + num_cuda_threads - 1) // num_cuda_threads

        # --- Device-side setup ---
        grid_shape_device = cuda.to_device(grid_shape)
        epsmap_midpoints_1d_device = cuda.to_device(epsmap_midpoints_1d)
        eps_midpoint_neighs_sum_only_1d_device = cuda.to_device(
            eps_midpoint_neighs_sum_only_1d
        )
        boundary_flags_1d_device = cuda.to_device(boundary_flags_1d)
        charge_map_1d_device = cuda.to_device(charge_map_1d)
        ion_exclusion_map_1d_device = cuda.to_device(ion_exclusion_map_1d)
        salt_ions_solvation_penalty_map_1d_device = cuda.to_device(
            salt_ions_solvation_penalty_map_1d
        )
        # print(
        #     "CUDA:",
        #     boundary_flags_1d.dtype,
        #     boundary_flags_1d.min(),
        #     boundary_flags_1d.max(),
        # )
        # print("CUDA, dtype:", boundary_flags_1d_device.dtype)
        # print("CUDA, shape:", boundary_flags_1d_device.shape)

        # --- Initialize phi halves ---
        phi_even_1d = np.zeros(num_grid_points_half, dtype=delphi_real)
        phi_odd_1d = np.zeros(num_grid_points_half, dtype=delphi_real)
        _copy_to_sample(phi_even_1d, phimap_current_1d, 0, 2)
        _copy_to_sample(phi_odd_1d, phimap_current_1d, 1, 2)

        phi_even_device = cuda.to_device(phi_even_1d)
        phi_odd_device = cuda.to_device(phi_odd_1d)

        # --- Allocations for convergence accumulators ---
        sum_sq_host = np.zeros(1, dtype=np.float64)
        max_dphi_host = np.zeros(1, dtype=np.float64)
        sum_sq_device = cuda.to_device(sum_sq_host)
        max_dphi_device = cuda.to_device(max_dphi_host)

        rmsd_buffer = np.zeros(10, dtype=np.float64)  # rolling RMSD buffer
        ptr = 0
        wrapped = False
        stop_iters = False

        total_iter = 0

        vprint(
            INFO,
            _VERBOSITY,
            f"\n    PBE> Running nonlinear CUDA solver (NWT) with ω_adaptive={omega_adaptive:.3f}",
        )
        vprint(
            INFO,
            _VERBOSITY,
            "    PBE> | #Iteration |    RMSD    |  Max(dPhi) | Time (seconds) |",
        )

        # ======================
        # Main iteration blocks
        # ======================
        while not stop_iters and total_iter < max_nonlinear_iters:
            tic_block = time.perf_counter()
            for itr_in_block in range(itr_block_size):
                for even_odd in (0, 1):
                    is_last_iter = itr_in_block == itr_block_size - 1
                    is_last_overall = (
                        total_iter + itr_in_block + 1
                    ) >= max_nonlinear_iters

                    # Select half maps
                    read_half_device = (
                        phi_odd_device if even_odd == 0 else phi_even_device
                    )
                    write_half_device = (
                        phi_even_device if even_odd == 0 else phi_odd_device
                    )

                    # ----------------------
                    # Main iteration switch
                    # ----------------------
                    if (even_odd == 1) and (is_last_iter or is_last_overall):
                        # Reset accumulators each block
                        _cuda_reset_rmsd_and_dphi[1, 1](sum_sq_device, max_dphi_device)
                        cuda.synchronize()

                        # Last odd iteration → fused RMSD kernel
                        _cuda_iterate_nwt_with_dphi_rmsd[n_blocks, num_cuda_threads](
                            vacuum,
                            even_odd,
                            non_zero_salt,
                            approx_zero,
                            omega_adaptive,
                            grid_shape_device,
                            read_half_device,
                            write_half_device,
                            salt_ions_solvation_penalty_map_1d_device,
                            epsmap_midpoints_1d_device,
                            eps_midpoint_neighs_sum_only_1d_device,
                            boundary_flags_1d_device,
                            charge_map_1d_device,
                            ion_exclusion_map_1d_device,
                            sum_sq_device,
                            max_dphi_device,
                        )
                        cuda.synchronize()
                    else:
                        # Regular iteration
                        _cuda_iterate_nwt[n_blocks, num_cuda_threads](
                            vacuum,
                            even_odd,
                            non_zero_salt,
                            approx_zero,
                            omega_adaptive,
                            grid_shape_device,
                            read_half_device,
                            write_half_device,
                            salt_ions_solvation_penalty_map_1d_device,
                            epsmap_midpoints_1d_device,
                            eps_midpoint_neighs_sum_only_1d_device,
                            boundary_flags_1d_device,
                            charge_map_1d_device,
                            ion_exclusion_map_1d_device,
                        )
                        cuda.synchronize()

            # --- Fetch RMSD / maxΔφ after block ---
            sum_sq_device.copy_to_host(sum_sq_host)
            max_dphi_device.copy_to_host(max_dphi_host)

            total_iter += itr_block_size
            rmsd = math.sqrt(sum_sq_host[0] / num_grid_points_half)
            max_delta_phi = abs(max_dphi_host[0])
            toc_block = time.perf_counter()

            vprint(
                INFO,
                _VERBOSITY,
                f"    PBE> | {total_iter:10d} | {rmsd:10.4e} | {max_delta_phi:10.4e} | {toc_block - tic_block:14.6f} |",
            )

            # --- unified iteration control ---
            stop_iters, status, ptr, wrapped = _iteration_control_check(
                rmsd_buffer,
                ptr,
                wrapped,
                rmsd,
                max_delta_phi,
                rms_tol,
                dphi_tol,
                check_dphi,
                max_nonlinear_iters,
                total_iter,
                disable_stagnation_check=False,
            )

            if stop_iters:
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

        # --- Final copy back ---
        phi_even_device.copy_to_host(phi_even_1d)
        phi_odd_device.copy_to_host(phi_odd_1d)
        _copy_to_full(phimap_current_1d, phi_even_1d, 0, 2)
        _copy_to_full(phimap_current_1d, phi_odd_1d, 1, 2)

        return rmsd, max_delta_phi, total_iter

    def _solve_nonlinear_pb_nwt(
        self,
        vacuum: delphi_bool,
        phimap_current_1d: np.ndarray[delphi_real],
        non_zero_salt: delphi_bool,
        approx_zero: delphi_real,
        omega_adaptive: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        salt_ions_solvation_penalty_map_1d: np.ndarray[delphi_real],
        epsmap_midpoints_1d: np.ndarray[delphi_real],
        eps_midpoint_neighs_sum_only_1d: np.ndarray[delphi_real],
        boundary_flags_1d: np.ndarray[delphi_bool],
        charge_map_1d: np.ndarray[delphi_real],
        ion_exclusion_map_1d: np.ndarray[delphi_real],
        itr_block_size: delphi_int,
        max_nonlinear_iters: delphi_int,
        max_rms: delphi_real,
        max_dphi: delphi_real,
        check_dphi: delphi_bool,
        platform_active: str,
        num_cuda_threads: delphi_int,
        num_cpu_threads: delphi_int,
        verbosity_level: int,
    ):
        """
        Unified Newton-like PB solver orchestrator.
        Dispatches to CPU or CUDA backend according to `platform_active`.
        """
        # print(
        #     "ORCHES:",
        #     boundary_flags_1d.dtype,
        #     boundary_flags_1d.min(),
        #     boundary_flags_1d.max(),
        # )
        if platform_active == "cuda":
            # print(
            #     "CUDA: nonzero salt penalty count:",
            #     np.count_nonzero(salt_ions_solvation_penalty_map_1d),
            # )
            # print(
            #     "CUDA: salt penalty mean:", np.mean(salt_ions_solvation_penalty_map_1d)
            # )

            rmsd, max_change, total_iter = self._cuda_solve_nonlinear_pb_nwt(
                vacuum=vacuum,
                phimap_current_1d=phimap_current_1d,
                non_zero_salt=non_zero_salt,
                approx_zero=approx_zero,
                omega_adaptive=omega_adaptive,
                grid_shape=grid_shape,
                salt_ions_solvation_penalty_map_1d=salt_ions_solvation_penalty_map_1d,
                epsmap_midpoints_1d=epsmap_midpoints_1d,
                eps_midpoint_neighs_sum_only_1d=eps_midpoint_neighs_sum_only_1d,
                boundary_flags_1d=boundary_flags_1d,
                charge_map_1d=charge_map_1d,
                ion_exclusion_map_1d=ion_exclusion_map_1d,
                itr_block_size=itr_block_size,
                max_nonlinear_iters=max_nonlinear_iters,
                rms_tol=max_rms,
                dphi_tol=max_dphi,
                check_dphi=check_dphi,
                num_cuda_threads=num_cuda_threads,
                verbosity_level=verbosity_level,
            )
        else:
            if DEBUG >= _VERBOSITY:
                nzsc = (np.count_nonzero(salt_ions_solvation_penalty_map_1d),)
                vprint(
                    DEBUG,
                    _VERBOSITY,
                    f"CPU: nonzero salt penalty count: {nzsc}",
                )
                sionmean = np.mean(salt_ions_solvation_penalty_map_1d)
                vprint(
                    DEBUG,
                    _VERBOSITY,
                    f"CPU: salt penalty mean: {sionmean}",
                )

            rmsd, max_change, total_iter = self._cpu_solve_nonlinear_pb_nwt(
                vacuum=vacuum,
                phimap_current_1d=phimap_current_1d,
                non_zero_salt=non_zero_salt,
                approx_zero=approx_zero,
                omega_adaptive=omega_adaptive,
                grid_shape=grid_shape,
                salt_ions_solvation_penalty_map_1d=salt_ions_solvation_penalty_map_1d,
                epsilon_map_midpoints_1d=epsmap_midpoints_1d,
                epsilon_sum_neighbors_sum_only_1d=eps_midpoint_neighs_sum_only_1d,
                boundary_flags_1d=boundary_flags_1d,
                charge_map_1d=charge_map_1d,
                ion_exclusion_map_1d=ion_exclusion_map_1d,
                itr_block_size=itr_block_size,
                max_nonlinear_iters=max_nonlinear_iters,
                rms_tol=max_rms,
                dphi_tol=max_dphi,
                check_dphi=check_dphi,
                num_cpu_threads=num_cpu_threads,
                verbose=verbosity_level,
            )

        vprint(
            INFO,
            _VERBOSITY,
            f"    PBE> Nonlinear solver finished after {total_iter} iterations",
        )

        return rmsd, max_change, total_iter

    def solve_pbe(
        self,
        vacuum: delphi_bool,
        bound_cond: BoundaryCondition,
        dielectric_model: DielectricModel,
        gaussian_exponent: delphi_int,
        itr_block_size: delphi_int,
        max_nonlinear_iters: delphi_int,
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
        omega_adaptive: delphi_real,
        grid_shape: np.ndarray[delphi_int],
        grid_origin: np.ndarray[delphi_real],
        grid_shape_parentrun: np.ndarray[delphi_int],
        grid_origin_parentrun: np.ndarray[delphi_real],
        atoms_data: np.ndarray[delphi_real],
        ion_exclusion_map_1d: np.ndarray[delphi_real],
        epsilon_map_1d: np.ndarray[delphi_real],
        epsmap_midpoints_1d: np.ndarray[delphi_real],
        centroid_pve_charge: np.ndarray[delphi_real],
        centroid_nve_charge: np.ndarray[delphi_real],
        charged_gridpoints_1d: np.ndarray[delphi_real],
        phimap_parentrun: np.ndarray[delphi_real],
    ) -> np.ndarray[delphi_real]:
        """
        Solves the Linearized Poisson-Boltzmann Equation (LPBE) using the SOR method.

        This method orchestrates the setup and calls the core iterative solver.
        The main steps are:
        1.  Platform Selection and Initialization.
        2.  Charge Assignment to grid points.
        3.  Boundary Condition Setup.
        4.  Preparation for Iteration (calculating epsilon sums, boundary maps, etc.).
        5.  Iterative Solver Setup (calculating the SOR relaxation factor/omega).
        6.  Perform Iterations (delegated to _solve_linear_pb_sor).

        Args:
            vacuum (delphi_bool): True if solving in vacuum, False for water.
            bound_cond (BoundaryCondition): Type of boundary condition.
            gaussian_exponent (delphi_int): Exponent for Gaussian charge spreading.
            itr_block_size (delphi_int): Number of iterations per block for status checks/RMSD calculation.
            max_nonlinear_iters (delphi_int): Maximum total iterations for the linear solve.
            scale (delphi_real): Grid scale (points per Angstrom).
            scale_parentrun (delphi_real): Grid scale of a parent run (if applicable for boundary conditions).
            exdi (delphi_real): Exterior dielectric constant.
            indi (delphi_real): Interior dielectric constant.
            debye_length (delphi_real): Debye length (related to salt concentration).
            non_zero_salt (delphi_bool): True if salt concentration is non-zero.
            total_pve_charge (delphi_real): Total positive charge from atoms.
            total_nve_charge (delphi_real): Total negative charge from atoms.
            max_rms (delphi_real): Maximum RMSD tolerance for convergence.
            max_dphi (delphi_real): Maximum potential change tolerance for convergence.
            check_dphi (delphi_bool): Flag to use max_dphi instead of max_rms for convergence check.
            epkt (delphi_real): kT/e in the appropriate units.
            approx_zero (delphi_real): A value considered close to zero.
            omega_adaptive (delphi_real): Adaptive damping factor to use in NWT.
            grid_shape (np.ndarray[delphi_int]): Shape of the 3D grid (nx, ny, nz).
            grid_origin (np.ndarray[delphi_real]): Origin coordinates of the grid.
            grid_shape_parentrun (np.ndarray[delphi_int]): Shape of the parent grid (if applicable).
            grid_origin_parentrun (np.ndarray[delphi_real]): Origin of the parent grid (if applicable).
            atoms_data (np.ndarray[delphi_real]): Array containing atom data (coords, radii, charges).
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
        # --- 1. Platform Selection and Initialization ---
        tic_total = time.perf_counter()

        is_gaussian_diel_model = dielectric_model in (
            DielectricModel.GAUSSIAN.int_value,
        )
        if self.platform.active == "cuda":
            # Ensure device is selected
            try:
                cuda.select_device(self.platform.names["cuda"]["selected_id"])
            except Exception as e:
                vprint(
                    WARNING,
                    _VERBOSITY,
                    f"Warning: Could not select CUDA device {self.platform.names['cuda']['selected_id']}: {e}",
                )
                # Fallback to CPU or raise error? Reverting to CPU for robustness example
                self.platform.active = "cpu"
                vprint(WARNING, _VERBOSITY, "Falling back to CPU.")
                set_num_threads(
                    self.platform.names["cpu"]["num_threads"]
                )  # Set CPU threads if falling back

        self.phase = "vacuum" if vacuum else "water"

        grid_spacing = 1.0 / scale
        grid_spacing_square = grid_spacing**2
        kappa_square = (
            (exdi / debye_length**2) if non_zero_salt else delphi_real(0.0)
        )  # Ensure kappa_square is 0 if no salt
        kappa_x_grid_spacing_wholesquare = kappa_square * grid_spacing_square

        # --- 2. Charge Assignment Phase ---
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
        toc_gcrg = time.perf_counter()
        self.timings[f"pb, {self.phase}| calc. charge source"] = "{:0.3f}".format(
            toc_gcrg - tic_gchrg
        )

        vprint(
            DEBUG,
            _VERBOSITY,
            f"     <<PBE>> total_pve_charge={total_pve_charge:0.6f}, total_nve_charge={total_nve_charge:0.6f}",
        )

        # --- 3. Boundary Condition Setup Phase ---
        tic_bndcon = time.perf_counter()
        # phimap_current_1d will be the input/output for the iterative solver method
        # Initialize phimap_current_1d if not already done
        phimap_current_1d = np.zeros(self.num_grid_points, dtype=delphi_real)

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
            phimap_1d=phimap_current_1d,  # Pass the 1D map
            phimap_parentrun=phimap_parentrun,
        )
        toc_bndcon = time.perf_counter()
        self.timings[f"pb, {self.phase}| set boundary condition"] = "{:0.3f}".format(
            toc_bndcon - tic_bndcon
        )

        vprint(
            TRACE,
            _VERBOSITY,
            "phimap_current_1d after bc:",
            phimap_current_1d[: min(100, self.num_grid_points)],
        )

        # --- 4. Preparation for Iteration Phase ---
        # Allocate/get eps_midpoint_neighs_sum_only_1d and boundary_gridpoints_1d
        eps_midpoint_neighs_sum_only_1d = np.zeros(
            self.num_grid_points,
            dtype=delphi_real,
        )

        boundary_gridpoints_1d = np.zeros(
            self.num_grid_points,
            dtype=delphi_bool,
        )

        charge_map_1d = np.copy(self.grid_charge_map_1d)

        tic_prepitr = time.perf_counter()
        self._prepare_charge_neighbor_epsmids_sum_to_iterate(
            vacuum=vacuum,
            exdi=exdi,
            grid_spacing=grid_spacing,
            debye_length=debye_length,
            non_zero_salt=non_zero_salt,
            four_pi=delphi_real(Constants.FourPi.value),
            epkt=epkt,
            grid_shape=grid_shape,
            epsmap_midpoints_1d=epsmap_midpoints_1d,  # Assuming epsmap_midpoints_1d is pre-calculated
            charge_map_1d=charge_map_1d,
            eps_midpoint_neighs_sum_only_1d=eps_midpoint_neighs_sum_only_1d,
            ion_exclusion_map_1d=ion_exclusion_map_1d,
            boundary_flags_1d=boundary_gridpoints_1d,
        )
        toc_prepitr = time.perf_counter()
        self.timings[f"pb, {self.phase}| prepare for iteration"] = "{:0.3f}".format(
            toc_prepitr - toc_bndcon
        )

        num_grid_points = grid_shape[0] * grid_shape[1] * grid_shape[2]
        salt_ions_solvation_penalty_map_1d = np.zeros(
            num_grid_points, dtype=delphi_real
        )
        if vacuum == False and non_zero_salt:
            vprint(
                DEBUG,
                _VERBOSITY,
                f"     <<PBE>> kappa_square={kappa_square:0.6f}, kappa_sq_times_h_sq={kappa_x_grid_spacing_wholesquare:0.6f}",
            )
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

        # --- 6. Perform Iterations (Delegated to helper) ---
        # The helper method _solve_linear_pb_sor handles the iteration loop
        # It also handles the allocation and initialization of its working even/odd arrays.
        # We pass the initial potential (phimap_current_1d) and all necessary parameters.
        # The potential is updated in-place within phimap_current_1d.
        self._solve_nonlinear_pb_nwt(
            vacuum=vacuum,
            phimap_current_1d=phimap_current_1d,
            non_zero_salt=non_zero_salt,
            approx_zero=approx_zero,
            omega_adaptive=omega_adaptive,
            grid_shape=grid_shape,
            salt_ions_solvation_penalty_map_1d=salt_ions_solvation_penalty_map_1d,
            epsmap_midpoints_1d=epsmap_midpoints_1d,
            eps_midpoint_neighs_sum_only_1d=eps_midpoint_neighs_sum_only_1d,
            boundary_flags_1d=boundary_gridpoints_1d,
            charge_map_1d=charge_map_1d,
            ion_exclusion_map_1d=ion_exclusion_map_1d,
            itr_block_size=itr_block_size,
            max_nonlinear_iters=max_nonlinear_iters,
            max_rms=max_rms,
            max_dphi=max_dphi,
            check_dphi=check_dphi,
            platform_active=self.platform.active,
            num_cuda_threads=self.num_cuda_threads,
            num_cpu_threads=self.platform.names["cpu"]["num_threads"],
            verbosity_level=self.verbosity,
        )

        toc_total = time.perf_counter()
        self.timings[f"pb, {self.phase}| total time"] = "{:0.3f}".format(
            toc_total - tic_total
        )

        # Note: When Gaussian_salt is used the ion_exclusion must be updated to align with the one usd in calculation
        if is_gaussian_diel_model and non_zero_salt and (not vacuum):
            grid_spacing_square = grid_spacing**2
            kappa_square = exdi / debye_length**2  # Related to ionic screening

            kappa_x_grid_spacing_wholesquare = (
                kappa_square * grid_spacing_square
            )  # Screening term
            ion_exclusion_map_1d[:] = (
                salt_ions_solvation_penalty_map_1d[:] / kappa_x_grid_spacing_wholesquare
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

        if bound_cond.int_value == BoundaryCondition.COULOMBIC.int_value:
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
        elif bound_cond.int_value == BoundaryCondition.DIPOLAR.int_value:
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
        elif bound_cond.int_value == BoundaryCondition.FOCUSING.int_value:
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
