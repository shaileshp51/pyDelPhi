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

import numpy as np
import time

from pydelphi.constants import ConstPhysical
from pydelphi.foundation.enums import Precision, VerbosityLevel
from pydelphi.config.global_runtime import set_precision, set_verbosity_level

set_precision(Precision.DOUBLE)
set_verbosity_level(VerbosityLevel.DEBUG)

from pydelphi.utils.io.inproc import Inputs
from pydelphi.foundation.platforms import Platform
from pydelphi.energy.coulombic import calc_coulombic_energy

epkt = ConstPhysical.EPK.value / 297.15
print("epkt:", epkt)

platform = Platform("cuda", True)

inp = Inputs()
inp.parse_inputs("param_1he8_b-a_py.prm")
atoms_data = np.array(list(inp.atoms.values()), dtype=np.float64)

# Run CUDA version
platform.activate("cuda")
start_cuda = time.perf_counter()
coul_energy_cuda = calc_coulombic_energy(platform, atoms_data, 1.0, epkt)
end_cuda = time.perf_counter()
print(f"CUDA Coulombic energy: {coul_energy_cuda:.6f}")
print(f"CUDA execution time: {end_cuda - start_cuda:.6f} seconds")

# Run CPU version
platform.activate("cpu")
start_cpu = time.perf_counter()
coul_energy_cpu = calc_coulombic_energy(platform, atoms_data, 1.0, epkt)
end_cpu = time.perf_counter()
print(f"CPU Coulombic energy: {coul_energy_cpu:.6f}")
print(f"CPU execution time: {end_cpu - start_cpu:.6f} seconds")

# Compare energies
if np.abs(coul_energy_cpu - coul_energy_cuda) < 1.0e-4:
    print("Coulombic energy on both platforms is the same.")
else:
    print("Coulombic energy differs between CPU and CUDA.")
