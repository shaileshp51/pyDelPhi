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

import numpy as np


def write_nparray_to_npy(
    filename,
    np_ndarray,
    decimals=6,
):
    """
    Dump surface charge positions to .npy with controlled precision.

    Parameters
    ----------
    decimals : int
        Number of decimal places to keep (e.g. 6 ≈ 1e-6 Å).
    """
    arr = np.asarray(np_ndarray, dtype=np.float64)
    arr = np.round(arr, decimals=decimals)
    np.save(filename, arr)
