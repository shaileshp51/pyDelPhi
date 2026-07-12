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


from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def determine_cuda_thread_count(grid_shape: NDArray[np.integer]) -> int:
    if not isinstance(grid_shape, np.ndarray):
        raise TypeError("grid_shape must be a numpy.ndarray")
    if grid_shape.ndim != 1 or grid_shape.size != 3:
        raise ValueError("grid_shape must be a 1D ndarray of length 3")

    n_grid_points = int(grid_shape[0]) * int(grid_shape[1]) * int(grid_shape[2])

    n_threads = 512
    n_blocks = (n_grid_points + 2 * n_threads - 1) // (2 * n_threads)
    while n_blocks < n_threads:
        n_threads //= 2
        if n_threads == 0:
            break
        n_blocks = (n_grid_points + 2 * n_threads - 1) // (2 * n_threads)

    return 1 if n_threads == 0 else int(n_threads)
