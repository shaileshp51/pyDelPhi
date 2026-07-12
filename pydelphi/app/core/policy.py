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

from dataclasses import dataclass
from typing import Any, Optional

from pydelphi.config.logging_config import WARNING
from pydelphi.config.global_runtime import vprint
from pydelphi.foundation.enums import DielectricModel, SurfaceMethod


@dataclass(frozen=True)
class TrajPolicy:
    """
    Trajectory-mode feature gates.

    Notes:
      - "disallow_*" raises an error if the feature is requested.
      - "warn_*" only emits a warning (does not block execution).
    """

    disallow_focusing: bool = True
    disallow_rpbe: bool = True
    disallow_gaussian_dielectric: bool = True
    disallow_gaussian_surface: bool = False
    warn_fixed_grid_if_multiframe: bool = True


def _io_warn(io: Any, msg: str) -> None:
    """Best-effort warning emitter with a safe fallback."""
    if io is not None:
        warn = getattr(io, "warn", None)
        if callable(warn):
            warn(msg)
            return
    vprint(msg, level=WARNING)


def _get_attr_any(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _truthy(x: Any) -> bool:
    return bool(x)


def _is_enum_value(x: Any, enum_value: Any) -> bool:
    # Supports direct enum compare, string compare, and .name/.value compare.
    if x is None:
        return False
    if x == enum_value:
        return True
    sx = str(x)
    return sx == str(enum_value) or getattr(x, "name", None) == getattr(
        enum_value, "name", None
    )


def _infer_n_frames(inp: Any, traj: Any = None, state: Any = None) -> Optional[int]:
    # Try explicit/obvious names first.
    n = _get_attr_any(inp, ("n_frames", "nframes", "num_frames", "frames"), None)
    if n is not None:
        try:
            return int(n)
        except Exception:
            return None

    # Try traj/state containers.
    for obj in (traj, state, getattr(state, "traj", None)):
        if obj is None:
            continue
        n = _get_attr_any(
            obj, ("n_frames", "nframes", "num_frames", "frames", "nframe"), None
        )
        if n is not None:
            try:
                return int(n)
            except Exception:
                return None
    return None


def enforce_traj_policy(
    inp: Any,
    policy: TrajPolicy,
    *,
    traj: Any = None,
    state: Any = None,
    io: Any = None,
) -> None:
    """
    Enforce trajectory-mode restrictions.

    Reads:
      - inp: feature flags + model choices
      - policy: TrajPolicy
      - traj/state: frame count (optional)

    Writes:
      - none

    Raises:
      - ValueError if a disallowed feature is requested
    """

    # ---- focusing ----
    focusing_requested = _truthy(
        _get_attr_any(
            inp,
            ("focusing", "do_focusing", "focus", "focus_on", "enable_focusing"),
            False,
        )
    )
    if policy.disallow_focusing and focusing_requested:
        raise ValueError(
            "Trajectory mode policy: focusing is disallowed (disable focusing)."
        )

    # ---- RPBE / regularized PB ----
    rpbe_requested = _truthy(
        _get_attr_any(
            inp,
            ("rpbe", "do_rpbe", "regularized_pb", "use_rpbe", "enable_rpbe"),
            False,
        )
    )
    if policy.disallow_rpbe and rpbe_requested:
        raise ValueError("Trajectory mode policy: RPBE/regularized PB is disallowed.")

    # ---- gaussian dielectric ----
    diel_model = _get_attr_any(
        inp, ("dielectric_model", "diel_model", "eps_model"), None
    )
    gaussian_diel_requested = _is_enum_value(diel_model, DielectricModel.GAUSSIAN)
    if policy.disallow_gaussian_dielectric and gaussian_diel_requested:
        raise ValueError(
            "Trajectory mode policy: Gaussian dielectric model is disallowed."
        )

    # ---- gaussian surface ----
    surf_method = _get_attr_any(inp, ("surface_method", "surf_method", "surface"), None)
    gaussian_surf_requested = _is_enum_value(surf_method, SurfaceMethod.GAUSSIAN)
    if policy.disallow_gaussian_surface and gaussian_surf_requested:
        raise ValueError(
            "Trajectory mode policy: Gaussian surface method is disallowed."
        )

    # ---- warn: fixed grid with multiframe ----
    n_frames = _infer_n_frames(inp, traj=traj, state=state)
    if (
        policy.warn_fixed_grid_if_multiframe
        and (n_frames is not None)
        and (n_frames > 1)
    ):
        # Heuristic: "fixed grid" if focusing is not requested and grid center/size looks constant.
        # We keep this intentionally lightweight and non-blocking.
        fixed_grid_likely = not focusing_requested
        if fixed_grid_likely:
            _io_warn(
                io,
                f"Trajectory mode warning: {n_frames} frames requested with a likely fixed grid "
                f"(no focusing). Ensure the grid adequately covers all frames or enable an "
                f"appropriate grid-updating strategy.",
            )
