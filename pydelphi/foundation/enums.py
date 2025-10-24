#!/usr/bin/env python3
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
Module defining enumeration classes and a gridbox dimension class for pydelphi configuration.

This module contains various Enum classes that define the valid options
for configuring different aspects of pydelphi calculations. These enums
cover settings such as:

    - PBE solver types (PBSolver)
    - Calculation precision (Precision)
    - Verbosity levels (VerbosityLevel)
    - Memory management states (MemoryState)
    - Boundary conditions (BoundaryCondition)
    - Solute extrema rules (SoluteExtremaRule)
    - Grid box types (GridBoxType)
    - Parameter types (ParamType)
    - Bio-model types (BioModel)
    - PB approximation forms (PBApproximation)
    - Surface methods (SurfaceMethod)
    - Ion exclusion region methods (IonExclusionRegion)
    - Dielectric models (DielectricModel)

These enums are designed to provide type-safe and self-documenting
configuration choices throughout the pydelphi project.

This module is intentionally kept lightweight and avoids dependencies
on libraries that are not essential for defining configuration enums.

It also includes the `GridboxSize` class for representing grid box
dimensions, which is used in conjunction with some of these enums.
"""

from pydelphi.foundation.enumbase import BaseInfoEnum
from pydelphi.foundation.bib_manager import cite
from numpy import array as np_array


class PBSolver(BaseInfoEnum):
    """Enumerates all supported options for PBE solver in PyDelphi."""

    SOR = 1, (
        f"Use successive over-relaxation solver for both linearized and non-linear PBE. (see: {cite('Nicholls1991')})"
    )
    NWT = 2, (
        f"Use Newton like solver for non-linear PBE. For linearized PBE, both NWT and SOR use the same formula. (see: {cite('Li2020')})"
    )


class Precision(BaseInfoEnum):
    """Enumerates all supported options for setting precision in PyDelphi calculations."""

    SINGLE = 1, "Use single precision (4-byte) for real numbers."
    DOUBLE = 2, "Use double precision (8-byte) for real numbers."


class VerbosityLevel(BaseInfoEnum):
    """Enumerates all supported verbosity levels for logging in PyDelphi."""

    CRITICAL = (
        50,
        "Log only critical failures or mandatory final results (e.g., unrecoverable errors).",
    )
    ERROR = (
        40,
        "Log errors preventing an operation from completing (e.g., invalid input, file not found).",
    )
    NOTICE = (
        35,
        "Log final results, excluding warnings, timings, and progress details.",
    )
    WARNING = (
        30,
        "Log warnings for potential issues or unexpected conditions (e.g., deprecated usage, non-optimal path).",
    )
    INFO = (
        20,
        "Log general application progress, informational messages, and significant events (formerly APPLICATION and VERBOSE).",
    )
    DEBUG = (
        10,
        "Enable all general debug messages, including detailed internal states, values, and minor computations.",
    )
    TRACE = (
        5,
        "Enable extremely fine-grained tracing, usually for performance analysis or deep debugging.",
    )


class MemoryState(BaseInfoEnum):
    """Enumerates supported memory usage modes for PyDelphi simulations."""

    FULL = 1, "Retain all simulation maps and data after runs."
    MINIMAL = 2, "Keep only necessary maps/data during run to reduce memory."


class BoundaryCondition(BaseInfoEnum):
    """Enumerates supported electrostatic boundary condition options."""

    COULOMBIC = (
        1,
        f"Use Coulombic potential from total charge. (see: {cite('Nicholls1991')})",
    )
    DIPOLAR = (
        2,
        f"Use dipole approximation from total +/- charges. (see: {cite('Panday2024')})",
    )
    FOCUSING = 4, "Use potential from prior run for focusing calculation."


class SoluteExtremaRule(BaseInfoEnum):
    """Enumerates methods to determine solute box bounds from atoms."""

    COORDINATE = 1, "Use min/max atom centers only."
    COORDLIMITS = 2, "Extend min/max coordinates by max atom radius."


class GridBoxType(BaseInfoEnum):
    """Enumerates geometric shapes of the grid box."""

    CUBIC = 1, f"Cubic grid box (equal sides). (see: {cite('Nicholls1991')})"
    CUBOIDAL = 2, f"Cuboidal grid box (unequal sides). (see: {cite('Panday2024')})"


class ParamType(BaseInfoEnum):
    """Enumerates parameter types used in input configuration."""

    STATEMENT = 1, "Simple parameter like 'param=value'."
    FUNCTION = 2, "Function-style parameter like 'param(category, attr=value)'."


class BioModel(BaseInfoEnum):
    """Enumerates formulations of the Poisson-Boltzmann equation."""

    PBE = 1, f"Standard Poisson-Boltzmann equation. (see: {cite('Nicholls1991')})"
    RPBE = 2, f"Regularized Poisson-Boltzmann equation. (see: {cite('Panday2024')})"


class PBApproximation(BaseInfoEnum):
    """Enumerates linear and non-linear PB equation forms."""

    LINEAR = 1, "Linearized PB approximation."
    NONLINEAR = 2, "Nonlinear full PB equation."


class SurfaceMethod(BaseInfoEnum):
    """Enumerates solute-solvent surface generation methods."""

    VDW = 1, f"Van der Waals surface. (see: {cite('Rocchia2001jcc')})"
    GAUSSIAN = 2, f"Gaussian smoothed surface. (see: {cite('Panday2024')})"
    GAUSSIANCUTOFF = (
        4,
        f"Cutoff-based Gaussian surface (for vacuum-phase). (see: {cite('Chakravorty2018')})",
    )
    GCS = 8, f"Gaussian Convolution Surface for RPBE. (see: {cite('Wang2021')})"


class IonExclusionRegion(BaseInfoEnum):
    """Enumerates methods/options for the placement of stern-layer around solute."""

    STERNLAYER = (
        1,
        f"Ions cant reach the ion-exclusion region (a layer offset by ion-radius), supported for (PB). (see: {cite('Nicholls1991')})",
    )
    GAUSSIANLAYER = (
        2,
        f"The ion exclusion layer is function of dielectric values thus Gaussian density, supported for (PB). (see: {cite('Jia2017')})",
    )
    SOLUTESURFACE = (
        4,
        f"A Gaussian density dependent solute surface defines the ion-exclusion region, supported for (RPB). (see: {cite('Panday2024')})",
    )


class DielectricModel(BaseInfoEnum):
    """Enumerates dielectric constant assignment models."""

    TWODIELECTRIC = (
        1,
        f"Two-region model with constant internal/external dielectric. (see: {cite('Rocchia2001jcc')})",
    )
    GAUSSIAN = (
        2,
        f"Gaussian-density weighted dielectric distribution. (see: {cite(['Li2013', 'Chakravorty2020', 'Panday2024'])})",
    )


class GridboxSize:
    """Represents gridbox dimensions (nx, ny, nz).

    This class encapsulates the dimensions of a 3D gridbox used in Delphi calculations.
    It stores the dimensions as a NumPy array for efficient numerical operations and
    supports initialization with either a single value for cubic gridboxes or three
    values for cuboidal gridboxes.

    The class also provides methods for:
        - Equality, less than, and greater than comparisons between DelphiGridboxSize objects
          or with tuples/integers, allowing for dimension-wise comparisons.
        - Modulo operation with an integer, applied to each dimension.
        - String representation of the dimensions in the format "(nx, ny, nz)".

    This class is designed for use in performance-critical Numba-compiled code,
    providing efficient access to grid dimensions.
    """

    def __init__(self, nx, ny=None, nz=None):
        """Initializes DelphiGridboxSize with dimensions.

        Args:
            nx (int): Number of grid points in x-direction (or all directions if ny, nz are None).
            ny (int, optional): Number of grid points in y-direction. Defaults to None.
            nz (int, optional): Number of grid points in z-direction. Defaults to None.

        Raises:
            ValueError: If an invalid number of arguments (not one or three) is provided.
        """
        if ny is None and nz is None:
            self.dims = np_array([nx, nx, nx], dtype=int)
        elif ny is not None and nz is not None:
            self.dims = np_array([nx, ny, nz], dtype=int)
        else:
            raise ValueError("Supply one or all three values.")

    @property
    def nx(self):
        return self.dims[0]

    @property
    def ny(self):
        return self.dims[1]

    @property
    def nz(self):
        return self.dims[2]

    def __eq__(self, other):
        """Checks if two DelphiGridboxSize objects or with tuple/int are equal."""
        return self._compare(other, lambda a, b: a == b)

    def __lt__(self, other):
        """Checks if one DelphiGridboxSize is less than another or tuple/int (component-wise)."""
        return self._compare(other, lambda a, b: a < b)

    def __gt__(self, other):
        """Checks if one DelphiGridboxSize is greater than another or tuple/int (component-wise)."""
        return self._compare(other, lambda a, b: a > b)

    def _compare(self, other, op):
        """Internal helper for comparisons."""
        if isinstance(other, GridboxSize):
            return (
                op(self.dims[0], other.dims[0])
                and op(self.dims[1], other.dims[1])
                and op(self.dims[2], other.dims[2])
            )
        elif isinstance(other, tuple) and len(other) == 3:
            return (
                op(self.dims[0], other[0])
                and op(self.dims[1], other[1])
                and op(self.dims[2], other[2])
            )
        elif isinstance(other, int):
            return (
                op(self.dims[0], other)
                and op(self.dims[1], other)
                and op(self.dims[2], other)
            )
        return False

    def __mod__(self, other):
        """Calculates modulo of each dimension with an integer."""
        if not isinstance(other, int):
            raise TypeError("Modulo operation must be with integer.")
        return GridboxSize(*(self.dims % other))

    def __str__(self):
        """Returns string representation as '(nx, ny, nz)'."""
        return f"({self.dims[0]}, {self.dims[1]}, {self.dims[2]})"
