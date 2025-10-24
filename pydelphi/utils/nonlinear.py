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
import math

from numba import set_num_threads, njit, prange

# Assuming these are defined elsewhere in the user's code
# For demonstration, I'll define them here with example values
MAX_SINH_TAYLOR_TERMS = 7
SINH_TAYLOR_COEFFS = np.array(
    [
        1 / 6,  # for i=0, corresponds to phi^3 / 3!
        1 / 120,  # for i=1, corresponds to phi^5 / 5!
        1 / 5040,  # for i=2, corresponds to phi^7 / 7!
        1 / 362880,  # for i=3, corresponds to phi^9 / 9!
        1 / 39916800,  # for i=4, corresponds to phi^11 / 11!
        1 / 6227020800,  # for i=5, corresponds to phi^13 / 13!
        1 / 1307674368000,  # for i=6, corresponds to phi^15 / 15!
    ],
    dtype=np.float64,
)


@njit(nogil=True, boundscheck=False, cache=True)
def sinh_taylor_safe(
    phi: float,
    taylor_cutoff: float = 0.1,
    clip_cutoff: float = 6.0,
    n_terms: int = 5,
    linear_extension_slope: float = 0.1,
) -> float:
    """
    Safe approximation of sinh(phi) using Taylor expansion for intermediate values,
    native sinh for very small values, and a controlled linear extension for large values.

    Args:
        phi: Input value.
        taylor_cutoff: Threshold below which native sinh is used.
        clip_cutoff: Maximum |phi| value allowed for Taylor approximation.
        n_terms: Number of Taylor terms to use (up to MAX_SINH_TAYLOR_TERMS).
        linear_extension_slope: The slope for the linear extension beyond clip_cutoff.

    Returns:
        Approximated sinh(phi), safe for CPU/CUDA use.
    """
    abs_phi = abs(phi)

    if abs_phi < taylor_cutoff:
        # Return native sinh(phi) for very small values
        return np.sinh(phi)
    elif abs_phi < clip_cutoff:
        # Use Taylor approximation for values between taylor_cutoff and clip_cutoff
        phi_sq = phi * phi
        term = phi
        result = phi

        max_terms = n_terms
        if max_terms > MAX_SINH_TAYLOR_TERMS:
            max_terms = MAX_SINH_TAYLOR_TERMS

        # Taylor series for sinh(x) = x + x^3/3! + x^5/5! + ...
        # The loop calculates terms phi^(2i+3) and multiplies by 1/(2i+3)!
        # SINH_FACTOR_TAYLOR_COEFFS_ARRAY should contain 1/3!, 1/5!, etc.
        for i in prange(
            max_terms
        ):  # Use prange for Numba parallelization if applicable
            term *= phi_sq
            result += SINH_TAYLOR_COEFFS[i] * term

        return result
    else:
        # For |phi| >= clip_cutoff, extend linearly from the value at clip_cutoff.
        # First, calculate the Taylor approximation at the positive clip_cutoff.
        # This ensures continuity at the clip_cutoff point.
        phi_clipped_for_taylor = clip_cutoff
        phi_sq_clipped = phi_clipped_for_taylor * phi_clipped_for_taylor
        term_clipped = phi_clipped_for_taylor
        result_clipped = phi_clipped_for_taylor

        max_terms = n_terms
        if max_terms > MAX_SINH_TAYLOR_TERMS:
            max_terms = MAX_SINH_TAYLOR_TERMS

        for i in prange(
            max_terms
        ):  # Use prange for Numba parallelization if applicable
            term_clipped *= phi_sq_clipped
            result_clipped += SINH_TAYLOR_COEFFS[i] * term_clipped

        value_at_clip_cutoff_positive = result_clipped

        if phi >= 0.0:
            # Linear extension for positive phi: starting value + slope * distance from cutoff
            return value_at_clip_cutoff_positive + linear_extension_slope * (
                phi - clip_cutoff
            )
        else:
            # Linear extension for negative phi, maintaining odd function symmetry:
            # -(starting value + slope * distance from cutoff (using abs_phi))
            return -(
                value_at_clip_cutoff_positive
                + linear_extension_slope * (abs_phi - clip_cutoff)
            )


@njit(nogil=True, boundscheck=False, cache=True)
def calc_sinh(phi: float, phi_cutoff: float):
    if abs(phi) > phi_cutoff:
        phi = phi_cutoff if phi > 0 else -phi_cutoff

    return math.sinh(phi)
