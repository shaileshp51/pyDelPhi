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


"""
This module provides a high-level interface (`PBESolver` class) for solving
the Poisson-Boltzmann Equation (PBE), including both linear and non-linear
forms. It acts as a dispatcher, selecting and initializing the appropriate
solver implementation (e.g., Successive Over-Relaxation (SOR) or Newton-based
methods) based on the user's configuration.

The `PBESolver` abstracts away the underlying numerical solver details,
offering a consistent API for setting up and executing PBE calculations.
It integrates with other `pydelphi` modules for handling grid data,
dielectric maps, charge distributions, and boundary conditions.

The module supports:
- Selection between different non-linear PBE solvers (currently SOR and Newton).
- Management of solver-specific parameters, including convergence criteria,
  relaxation values, and iteration limits.
- Integration with `pydelphi`'s global runtime configuration for data types
  and platform specifics.
- Collection of timing information from the chosen solver.
"""

import numpy as np

from pydelphi.foundation.enums import (
    BoundaryCondition,
    VerbosityLevel,
    DielectricModel,
)
from pydelphi.config.global_runtime import (
    delphi_bool,
    delphi_int,
    delphi_real,
)

from pydelphi.solver.pb.sor.nonlinear_pb import NLPBESolver
from pydelphi.solver.pb.nwt.nonlinear_pb import NLNewtonPBESolver


class PBESolver:
    def __init__(
        self,
        platform,
        verbosity,
        num_cuda_threads,
        solver_name,
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
        self.solver = None
        self.solver_name = solver_name.lower()
        self.timings = {}

        if self.solver_name == "nwt":
            self.solver = NLNewtonPBESolver(
                platform,
                verbosity,
                num_cuda_threads,
                grid_shape,
            )
        elif self.solver_name == "sor":
            self.solver = NLPBESolver(
                platform,
                verbosity,
                num_cuda_threads,
                grid_shape,
            )
        else:
            raise ValueError("Unknown solver: " + self.solver_name)

    def solve_pbe(
        self,
        vacuum: delphi_bool,
        bound_cond: BoundaryCondition,
        dielectric_model: DielectricModel,
        gaussian_exponent: delphi_int,
        itr_block_size: delphi_int,
        max_linear_iters: delphi_int,
        max_nonlinear_iters: delphi_int,  # New parameter
        max_nonlinear_coupling_dphi: delphi_real,
        coupling_steps: delphi_int,  # New parameter for lambda schedule
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
        Solves the Non-Linear/Linear Poisson-Boltzmann Equation ([N/L]PBE) using a finite difference SOR or NWT method.

        Args:
            vacuum (delphi_bool): True if solving in vacuum, False for water.
            bound_cond (BoundaryCondition): Type of boundary condition.
            dielectric_model (DielectricModel): Chosen dielectric model.
            gaussian_exponent (delphi_int): Exponent for Gaussian charge spreading.
            itr_block_size (delphi_int): Number of iterations per block for status checks/RMSD calculation.
            max_linear_iters (delphi_int): Maximum total iterations for the linear solve.
            max_nonlinear_iters (delphi_int): Maximum total iterations for the non-linear solve.
            max_nonlinear_coupling_dphi (delphi_real): Tolerance for non-linear convergence (e.g., change in potential).
            coupling_steps (delphi_int): Number of steps to increase the coupling parameter from 0 to 1.
            manual_relaxation_value (delphi_real): The user supplied relaxation value to use instead of calculated one.
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
        phimap_current_1d = None
        if self.solver_name == "sor":

            phimap_current_1d = self.solver.solve_nonlinear_pb(
                vacuum,
                bound_cond,
                dielectric_model,
                gaussian_exponent,
                itr_block_size,
                max_linear_iters,
                max_nonlinear_iters,
                max_nonlinear_coupling_dphi,
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
                charged_gridpoints_1d,  # Original fixed charges
                phimap_parentrun,
            )
            self.timings.update(self.solver.timings)
        elif self.solver_name == "nwt":
            phimap_current_1d = self.solver.solve_pbe(
                vacuum,
                bound_cond,
                dielectric_model,
                gaussian_exponent,
                itr_block_size,
                max_nonlinear_iters,
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
            )
            self.timings.update(self.solver.timings)

        return phimap_current_1d
