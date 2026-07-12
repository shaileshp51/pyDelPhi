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

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AtomAdjacencyCSR:
    """
    CSR adjacency model for atoms.
    Arrays are automatically frozen (write-protected) on construction.
    """

    row_ptr: np.ndarray
    col_idx: np.ndarray
    degree: np.ndarray
    sorted_rows: bool = True
    deduped_rows: bool = True

    def __post_init__(self):
        # Freeze underlying arrays to prevent accidental writes
        self.row_ptr.flags.writeable = False
        self.col_idx.flags.writeable = False
        self.degree.flags.writeable = False

    def neighbors(self, i: int) -> np.ndarray:
        start = self.row_ptr[i]
        end = self.row_ptr[i + 1]
        return self.col_idx[start:end]
