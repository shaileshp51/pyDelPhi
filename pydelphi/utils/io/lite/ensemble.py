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
from typing import Dict, Iterator, Mapping, Optional, Tuple

import numpy as np

from .topology_lite import TopologyLite
from .trajectory_lite import TrajectoryLite


# -----------------------------
# Canonical labels (frozen)
# -----------------------------
_CANON_LABELS = ("system", "complex", "receptor", "ligand")
_ALIAS_TO_CANON = {
    "ab": "complex",
    "a": "receptor",
    "b": "ligand",
}


def canonicalize_label(label: str) -> str:
    """
    Canonicalize ensemble labels.

    Frozen rules:
    - case-insensitive
    - aliases: AB->complex, A->receptor, B->ligand
    - other labels preserved (lowercased stripped)
    """
    if label is None:
        raise ValueError("label cannot be None")
    s = label.strip()
    if not s:
        raise ValueError("label cannot be empty")
    k = s.lower()
    return _ALIAS_TO_CANON.get(k, k)


# -----------------------------
# Entry and container
# -----------------------------
@dataclass(frozen=True)
class EnsembleEntry:
    """
    A single (top, traj) pair with optional frame slicing.

    Frame slicing is metadata only; readers/adapters may use it,
    and Apps can decide whether to respect it.
    """

    top: TopologyLite
    traj: TrajectoryLite
    start: int = 0
    stop: Optional[int] = None
    stride: int = 1

    def validate(self) -> None:
        if self.top.natoms != self.traj.natoms:
            raise ValueError(
                f"Topology/Trajectory NATOM mismatch: top.natoms={self.top.natoms} "
                f"traj.natoms={self.traj.natoms}"
            )
        if self.start < 0:
            raise ValueError("start must be >= 0")
        if self.stop is not None and self.stop < 0:
            raise ValueError("stop must be >= 0 or None")
        if self.stride <= 0:
            raise ValueError("stride must be >= 1")


class Ensemble:
    """
    A labeled mapping of EnsembleEntry.

    This is the app-facing container that Inputs should build.
    """

    def __init__(self):
        self._entries: Dict[str, EnsembleEntry] = {}
        self._insertion_order: list[str] = []

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[str]:
        return iter(self._insertion_order)

    def items(self) -> Iterator[Tuple[str, EnsembleEntry]]:
        for k in self._insertion_order:
            yield k, self._entries[k]

    def keys(self) -> Iterator[str]:
        return iter(self._insertion_order)

    def values(self) -> Iterator[EnsembleEntry]:
        for k in self._insertion_order:
            yield self._entries[k]

    def get(self, label: str) -> Optional[EnsembleEntry]:
        return self._entries.get(canonicalize_label(label))

    def __getitem__(self, label: str) -> EnsembleEntry:
        return self._entries[canonicalize_label(label)]

    def add(self, label: str, entry: EnsembleEntry, *, overwrite: bool = False) -> str:
        """
        Add an entry under a canonicalized label.

        Returns the canonical label used.
        """
        lab = canonicalize_label(label)
        entry.validate()

        if (not overwrite) and lab in self._entries:
            raise ValueError(f"Duplicate ensemble label: {lab!r}")

        if lab not in self._entries:
            self._insertion_order.append(lab)

        self._entries[lab] = entry
        return lab

    @property
    def default_label(self) -> str:
        """
        Default label for convenience access.
        Frozen choice: first inserted label.
        Inputs may choose to insert 'system' or 'complex' first depending on mode.
        """
        if not self._insertion_order:
            raise ValueError("Ensemble is empty")
        return self._insertion_order[0]

    @property
    def top(self) -> TopologyLite:
        return self._entries[self.default_label].top

    @property
    def traj(self) -> TrajectoryLite:
        return self._entries[self.default_label].traj


# -----------------------------
# Masks (singletraj-binding)
# -----------------------------
_MASK_CANON = {"rec": "receptor", "rl": "receptor", "lig": "ligand"}


def canonicalize_mask_name(name: str) -> str:
    s = name.strip().lower()
    if not s:
        raise ValueError("mask name cannot be empty")
    s = _MASK_CANON.get(s, s)
    if s not in ("complex", "receptor", "ligand"):
        raise ValueError(f"Unsupported mask name: {name!r}")
    return s


def validate_mask(mask: np.ndarray, natoms: int, *, name: str) -> np.ndarray:
    """
    Ensure mask is boolean of shape (N,). Returns a boolean view/copy.
    """
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask {name!r} must be a numpy array")
    if mask.ndim != 1 or mask.shape[0] != natoms:
        raise ValueError(f"mask {name!r} must have shape ({natoms},)")
    if mask.dtype != np.bool_:
        mask = mask.astype(np.bool_, copy=False)
    return mask


# -----------------------------
# Protocol mode detection
# -----------------------------
def detect_protocol_mode(
    *,
    ensemble: Mapping[str, EnsembleEntry],
    masks: Optional[Mapping[str, np.ndarray]] = None,
) -> str:
    """
    Decide protocol mode from ensemble labels and masks.

    Frozen modes:
    - 'singletraj-system'
    - 'singletraj-binding'
    - 'multitraj-binding'
    - 'batch'
    """
    canon_labels = {canonicalize_label(k) for k in ensemble.keys()}
    has_masks = masks is not None and len(masks) > 0

    # If user supplies receptor/ligand masks -> singletraj-binding
    if has_masks:
        # Expect single top/traj universe under 'complex' (or 'system' but will be normalized by Inputs)
        mkeys = {canonicalize_mask_name(k) for k in masks.keys()}
        if ("receptor" in mkeys) != ("ligand" in mkeys):
            raise ValueError("Both receptor and ligand masks must be supplied together")
        if "receptor" in mkeys and "ligand" in mkeys:
            return "singletraj-binding"
        # masks present but not binding masks -> still singletraj-system
        return "singletraj-system"

    # Multi-traj binding if all three present
    if {"complex", "receptor", "ligand"}.issubset(canon_labels):
        return "multitraj-binding"

    # Singletraj-system if only one entry (any label)
    if len(canon_labels) == 1:
        return "singletraj-system"

    return "batch"


# -----------------------------
# Binding consistency checks
# -----------------------------
def validate_singletraj_binding(
    *,
    entry: EnsembleEntry,
    masks: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Validate and normalize masks for singletraj-binding mode.

    Returns:
        dict with canonical keys: receptor, ligand, and complex (always provided).
    """
    entry.validate()
    natoms = entry.top.natoms

    # canonicalize and validate masks
    out: Dict[str, np.ndarray] = {}
    for k, v in masks.items():
        ck = canonicalize_mask_name(k)
        out[ck] = validate_mask(v, natoms, name=ck)

    if ("receptor" in out) != ("ligand" in out):
        raise ValueError("Both receptor and ligand masks must be supplied together")

    # default complex is all atoms unless user supplies it
    if "complex" not in out:
        out["complex"] = np.ones(natoms, dtype=np.bool_)

    # Optional: ensure receptor/ligand are subsets of complex
    if "receptor" in out and "ligand" in out:
        if np.any(out["receptor"] & ~out["complex"]):
            raise ValueError("receptor mask must be subset of complex mask")
        if np.any(out["ligand"] & ~out["complex"]):
            raise ValueError("ligand mask must be subset of complex mask")

    return out


def validate_multitraj_binding(ensemble: Mapping[str, EnsembleEntry]) -> None:
    """
    Validate multitraj-binding mode: require complex/receptor/ligand labels.
    """
    need = ("complex", "receptor", "ligand")
    canon = {canonicalize_label(k) for k in ensemble.keys()}
    missing = [k for k in need if k not in canon]
    if missing:
        raise ValueError(f"Missing required binding entries: {missing}")
