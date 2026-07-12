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


def reset_phase_dependent_maps(ctx, vacuum: bool) -> None:
    """
    Reset phase-dependent maps in `ctx` to None (memory release hint).

    Reads:
      - vacuum

    Writes:
      - ctx.{epsilon_map_midpoints_*, grad_*_*} = None
    """
    if vacuum:
        ctx.epsilon_map_midpoints_vacuum_1d = None
        ctx.grad_eps_dot_gad_coul_vacuum_1d = None
        ctx.grad_epsgauss_map_vacuum_1d = None
        ctx.grad_epsilon_map_vacuum_1d = None
    else:
        ctx.epsilon_map_midpoints_water_1d = None
        ctx.grad_eps_dot_gad_coul_water_1d = None
        ctx.grad_epsgauss_map_water_1d = None
        ctx.grad_epsilon_map_water_1d = None
