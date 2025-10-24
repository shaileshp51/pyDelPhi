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
This module defines various constants and enumerations used throughout the Delphi program.

It includes:
- `NEIGHBOR_VOXEL_RELATIVE_COORDINATES`: A NumPy array of relative coordinates for neighboring voxels,
  used in surface building modules.
- `AtomFields`: An enumeration detailing the fields present in the atom data array,
  providing a structured and readable way to access atom properties.
- Derived constants (e.g., `ATOMFIELD_X`, `LEN_ATOMFIELDS`) for convenient access to `AtomFields` members.
- General integer constants like `MAX_KERNEL_SHARED_MEM_THREADS`, `GRID_POINT_NEIGHBOR_COUNT`, etc.
- `ConstDelPhiFloats`: An enumeration of floating-point constants, including practical zeros,
  factors for Gaussian models, and resizing/estimation parameters for SAS calculations.
- `ConstDelPhiInts`: An enumeration of integer constants, such as array sizes and special sentinel values.
"""


from enum import Enum
from numpy import uint8, int32, array
from pydelphi.foundation.enumbase import BaseInfoEnum
from .physical import ConstPhysical  # Import from the local constants package

"""
Relative cube-coordinates of all immediate neighbor-cubes containing the atom/object (placed at 0,0,0)
It is used in van der Walls molecular surface building modules helper and cuber sub-modules.
"""
NEIGHBOR_VOXEL_RELATIVE_COORDINATES = array(
    [
        (0, 0, 0),
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
        (1, 0, 1),  # changes two of x-,y-,z-directions at a time
        (-1, 0, 1),
        (0, 1, 1),
        (0, -1, 1),
        (-1, -1, 0),
        (1, -1, 0),
        (1, 1, 0),
        (-1, 1, 0),
        (-1, 0, -1),
        (1, 0, -1),
        (0, 1, -1),
        (0, -1, -1),
        (-1, -1, -1),  # changes all three x-,y-,z-directions
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, 1, 1),
        (1, -1, 1),
        (-1, -1, 1),
    ],
    dtype=int32,
)
# NOTE: relative-cube-voxel coordinates of neighbor of atom/object are not mutable
NEIGHBOR_VOXEL_RELATIVE_COORDINATES.setflags(write=False)


class AtomFields(BaseInfoEnum):
    """
    Enumerates fields available in the atom_data array used by Delphi.

    Each field is represented by its integer ID and a descriptive string.
    This provides a descriptive and accessible way to refer to atom data fields
    instead of using raw indices.

    Fields:
        CoordX (int): Input X-coordinate in angstrom.
        CoordY (int): Input Y-coordinate in angstrom.
        CoordZ (int): Input Z-coordinate in angstrom.
        GridX (int): Grid X-coordinate in angstrom/grid_scale.
        GridY (int): Grid Y-coordinate in angstrom/grid_scale.
        GridZ (int): Grid Z-coordinate in angstrom/grid_scale.
        Charge (int): Partial charge on the atom in multiples of proton charge.
        Radius (int): Van der Waals radius or size in angstrom.
        GaussianSigma (int): Gaussian variance parameter used in Gaussian model.
        ResidueKey (int): Residue-key of the atom; unique per residue type.
                            Refer to `ResNameToResKey` and `ResKeyToResName`.
        AtomicNumber (int): Atomic number of the atom's chemical element.
                            Refer to `ConstChemElement`.
        LJ_Sigma (int): Sigma value of Lennard-Jones interaction.
        LJ_Epsilon (int): Epsilon value of Lennard-Jones interaction.
        LJ_Gamma (int): Gamma value of Lennard-Jones interaction.
        MediaID (int): ID of the media in which the atom is present.
    """

    CoordX = 0, "Input X-coordinate in angstrom"
    CoordY = 1, "Input Y-coordinate in angstrom"
    CoordZ = 2, "Input Z-coordinate in angstrom"
    GridX = 3, "Grid X-coordinate in angstrom/grid_scale"
    GridY = 4, "Grid Y-coordinate in angstrom/grid_scale"
    GridZ = 5, "Grid Z-coordinate in angstrom/grid_scale"
    Charge = 6, "Partial charge on the atom in multiples of proton charge"
    Radius = 7, "The van der Waals radius or size in angstrom"
    GaussianSigma = 8, "Gaussian variance parameter used in gaussian model"
    ResidueKey = 9, "The residue-key of the atom"
    AtomicNumber = 10, "The atomic number of atom's chemical element"
    LJ_Sigma = 11, "The sigma value of Lennard-Jones interaction"
    LJ_Epsilon = 12, "The epsilon value of Lennard-Jones interaction"
    LJ_Gamma = 13, "The gamma value of Lennard-Jones interaction"
    MediaID = 14, "The ID of the media in which this atom is present"

    @property
    def id(self):
        # Alias for backward compatibility with .value
        return self.value


# Define ATOMFIELD constants for efficient field access using integer indices.
# These are derived from DelphiAtomsFields enum for consistency and readability.
ATOMFIELD_X = AtomFields.CoordX.id
ATOMFIELD_Y = AtomFields.CoordY.id
ATOMFIELD_Z = AtomFields.CoordZ.id

# Index of the field after Z coordinate for use in range based coords access
ATOMFIELD_CRD_END = ATOMFIELD_Z + 1

ATOMFIELD_GRID_X = AtomFields.GridX.id
ATOMFIELD_GRID_Y = AtomFields.GridY.id
ATOMFIELD_GRID_Z = AtomFields.GridZ.id

# Index of the field after Grid coordinate for use in range based Grid coords access
ATOMFIELD_GRID_END = ATOMFIELD_GRID_Z + 1

ATOMFIELD_CHARGE = AtomFields.Charge.id
ATOMFIELD_RADIUS = AtomFields.Radius.id
ATOMFIELD_GAUSS_SIGMA = AtomFields.GaussianSigma.id
ATOMFIELD_RES_KEY = AtomFields.ResidueKey.id
ATOMFIELD_ATOMIC_NUMBER = AtomFields.AtomicNumber.id
ATOMFIELD_LJ_SIGMA = AtomFields.LJ_Sigma.id
ATOMFIELD_LJ_EPSILON = AtomFields.LJ_Epsilon.id
ATOMFIELD_LJ_GAMMA = AtomFields.LJ_Gamma.id
ATOMFIELD_MEDIA_ID = AtomFields.MediaID.id
LEN_ATOMFIELDS = len(AtomFields)  # Total number of atom fields

# Constant for CUDA kernel shared memory compilation size
MAX_KERNEL_SHARED_MEM_THREADS = 1024

# Constants - Define constants for better readability and maintainability
GRID_POINT_NEIGHBOR_COUNT = 6  # For 3D grid, each point has 6 midpoint neighbors
XYZ_COMPONENTS = 3  # x, y, z components in 3D
HALF_GRID_OFFSET_LAGGING = 1
HALF_GRID_OFFSET_LEADING = 0

# Define named constants for grid point charge fields
GPCHRGFIELD_INDX_1D = 0  # Index of the 1D grid point index
GPCHRGFIELD_CHARGE = 1  # Index of the charge at the grid point
GPCHRGFIELD_INDX_X = 2  # Index of the x-coordinate of the grid point
GPCHRGFIELD_INDX_Y = 3  # Index of the y-coordinate of the grid point
GPCHRGFIELD_INDX_Z = 4  # Index of the z-coordinate of the grid point

# Define named constant for neighbor offsets
GRID_NEIGHBOR_OFFSETS = (0, 1)

# Define constant for the number of dimensions
NUM_DIMENSIONS = 3

# LJ-pair inclusion cutoffs
LJCUTOFF_MIN = 1.0
LJCUTOFF_MAX = 7.0

# Constants used for circle-of-intersection in .space.core.sas
CCOI_X = 0
CCOI_Y = 1
CCOI_Z = 2
CCOI_CRD_END = 3
CCOI_RADIANS = 3
CCOI_COUNT = 4
CCOI_FIELDS_COUNT = 5


class ConstDelPhiFloats(Enum):
    r"""
    Delphi numerical constants (double precision).

    Constants:
        ApproxZero (float): Practical zero for comparing Delphi real numbers.

        ZeroDensity (float): Practical zero for comparing Gaussian density values.

        ZeroMolarSaltDebyeLength (float): Represents infinite Debye length (zero salt conc.).

        GaussianInfluenceRadiusFactor (float): Factor controlling the extent of the
            Gaussian influence radius. This factor is multiplied by $(\text{atom\_radius} \times \text{atom\_sigma})$
            to determine a radius beyond which the Gaussian contribution is considered
            negligible. Higher values result in a larger radius and a smaller minimal
            density accounted for.

            For a value of 2.0, the minimal density at the boundary is approximately
            $\exp(-4) \approx 0.018$ (1.8%).

            For a value of $3 / \sqrt{2} \approx 2.1213$ (approximately 3 standard deviations),
            the minimal density is approximately $\exp(-4.5) \approx 0.0111$ (1.11%).

            For a value of 3.0, the minimal density at the boundary is approximately
            $\exp(-9) \approx 0.000123$ (0.0123%).

            For a value of 4.0, the minimal density at the boundary is approximately
            $\exp(-16) \approx 1.13 \times 10^{-7}$.

        GaussianDensityPruneThreshold (float): A density threshold below which Gaussian
            contributions are considered insignificant and treated as zero. This improves
            performance by ignoring negligible values that do not contribute meaningfully
            to electrostatics. This is intentionally set smaller than the smoothing threshold
            and is used for pruning during density map generation.

        GaussianDensitySmoothingThreshold (float): Density threshold below which the Gaussian
            density is smoothed out during epsilon map modeling. This is used only in specific
            models to ensure a smooth dielectric transition at molecular surfaces.

        ZetaArrayResizeFactor (float): Factor to increase zeta-surface-array size when resizing.

        ZetaArrayInitialSizePercent (float): Initial size for zeta surface grid coords as a
            fraction of max possible $(n\_grid\_points)$.

        SASSquaredRadiiShrinkFactor (float): Slightly shrink squared radii to avoid ambiguous
            grid-boundary cases.

        SASAtomPairsMinimalCount (float): The minimum initial number (an educated guess based
            on domain knowledge) of atom contact pairs to allocate for SAS calculation.

        SASLinearPairsFactorPerAtom (float): The factor (an educated guess based on domain
            knowledge) multiplied by the number of atoms to estimate the initial number of
            contact pairs for SAS.

        SASQuadraticPairsFactorOfNSquared (float): The factor (an educated guess based on domain
            knowledge) multiplied by the square of the number of atoms to estimate the initial
            number of contact pairs for SAS.

        SASExposedGridsMinimalCount (float): The minimum initial number (an educated guess based
            on domain knowledge) of exposed grid points to allocate for SAS calculation.

        SASExposedGridsSurfaceAreaFactor (float): Factor used to estimate the initial number of
            exposed grid points based on the surface area scaling of the grid.

            The number of exposed points is expected to scale with the surface area of the
            molecule, which is proportional to $N^{2/3}$ where $N^3$ is the total number of grid
            points. The value is chosen based on the surface area of a sphere ($4\pi r^2$) and
            considering that a sphere has the minimal surface area for a given volume.

            For a grid representing a molecule, the shape might be more elongated (like a cylinder
            with a higher surface area to volume ratio). For a sphere, the surface area is
            $4\pi r^2$ and if we relate the volume to $(2r)^3 \approx N^3$, then $r \approx N/2$,
            giving surface area $\approx \pi N^2 = \pi (\text{grid\_dimension})^{2/3}$.

            To be more liberal and account for non-spherical shapes, we use a factor of
            $2 \times 4\pi \approx 25.13$. The final initial size is taken as the maximum of this
            estimate and SASExposedGridsMinimalCount.
    """

    ApproxZero = 1.0e-13
    ZeroDensity = 1.0e-3
    ZeroMolarSaltDebyeLength = 1.0e6
    GaussianInfluenceRadiusFactor = 3.0
    GaussianDensityPruneThreshold = 1.0e-6
    GaussianDensitySmoothingThreshold = 0.02
    ZetaArrayResizeFactor = 1.5
    ZetaArrayInitialSizePercent = 0.1
    SASSquaredRadiiShrinkFactor = 0.99999
    SASAtomPairsMinimalCount = 10000.0
    SASLinearPairsFactorPerAtom = 200.0
    SASQuadraticPairsFactorOfNSquared = 0.02
    SASExposedGridsMinimalCount = 5000.0
    SASExposedGridsSurfaceAreaFactor = 2 * ConstPhysical.FourPi.value


class ConstDelPhiInts(Enum):
    """
    Delphi integer constants.

    This enumeration defines integer constants used in Delphi, often for array sizing
    or representing special integer values within the program.

    Constants:
        SpaceNBRASize (int): Size of the space NBR array.
        ResidueNumberUnknown (int): Integer representing an unknown residue number.
    """

    SpaceNBRASize = 50001
    ResidueNumberUnknown = -999999
    ExitNjitReturnValue = -9999


class BoxGridPointType(Enum):
    """
    Integer labels assigned to grid points in the simulation box.

    These labels are stored in a uint8 property map to categorize grid points
    for electrostatic calculations.

    - BOUNDARY (255): Box boundary grid point.
    - INTERIOR (0): Solute/interior grid point.
    - HOMO_EPSILON (1): Grid point is in a homogeneous dielectric region all
                          neighboring midpoints have same epsilon.
    """

    BOUNDARY = uint8(0)
    INTERIOR = uint8(1)
    HOMO_EPSILON = uint8(255)


# Constants for fast lightweight access in performance-sensitive code
BOX_BOUNDARY: uint8 = BoxGridPointType.BOUNDARY.value
BOX_INTERIOR: uint8 = BoxGridPointType.INTERIOR.value
BOX_HOMO_EPSILON: uint8 = BoxGridPointType.HOMO_EPSILON.value
