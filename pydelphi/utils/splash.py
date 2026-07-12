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
Splash and citation text for pyDelPhi logs.

Design principles:
- Return formatted strings (no printing).
- App controls verbosity and when to print.
- Consistent banner width: 90 characters.
- Delimiter semantics:
    '*' : lifecycle timestamps (start / finish) — handled in app
    '#' : splash banner
    '-' : citation footer
"""

from __future__ import annotations

_BANNER_WIDTH = 90


def _rule(ch: str, width: int = _BANNER_WIDTH) -> str:
    if not ch or len(ch) != 1:
        raise ValueError("Delimiter must be a single character")
    return ch * width


def format_splash(width: int = _BANNER_WIDTH) -> str:
    """Return the pyDelPhi splash banner."""
    import pydelphi  # local import to avoid import cycles

    version = getattr(pydelphi, "__version__", "unknown")

    lines = [
        # _rule("#", width),
        "",
        f"pyDelPhi v{version}".center(width),
        "Accurate and Scalable Continuum Electrostatics for Biomolecular Systems".center(
            width
        ),
        "",
        "Authors : Shailesh Kumar Panday, Shan Zhao, Emil Alexov",
        "License : AGPL-3.0",
        "",
        _rule("#", width),
    ]
    return "\n".join(lines)


def format_citation(width: int = _BANNER_WIDTH) -> str:
    """Return the pyDelPhi citation footer."""
    lines = [
        _rule("-", width),
        "If you use pyDelPhi in published work, please cite:",
        "",
        "  Panday, S. K.; Zhao, S.; Alexov, E.",
        "  Accurate and Scalable Continuum Electrostatics for Large Biomolecular Systems:",
        "  The pyDelPhi Poisson–Boltzmann Framework",
        "  J. Chem. Inf. Model. 2026 66 (1), 488-502",
        "  DOI: 10.1021/acs.jcim.5c02818",
        # _rule("-", width),
    ]
    return "\n".join(lines)
