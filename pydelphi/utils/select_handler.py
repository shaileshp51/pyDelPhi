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

from pydelphi.utils.select_lang import select_atom_indices, SelectionError


def materialize_selections(selections_spec: dict, atom_keys) -> dict:
    """
    Materialize all selections against atom_keys.

    Args:
        selections_spec:
            {
                name: {
                    "condition": <selection string>,
                    "desc": <human-readable description>,
                    "on": <topology label this selection applies to (default: system)>
                }
            }
        atom_keys:
            topology atom keys

    Returns:
        selections_idx:
            { name: indices }
    """
    selections_idx = {}

    for name, spec in selections_spec.items():
        try:
            cond = spec.get("condition")
            if not cond:
                raise SelectionError("missing 'condition'")

            selections_idx[name] = select_atom_indices(cond, atom_keys)

        except SelectionError as e:
            desc = spec.get("desc", "")
            msg = f"select(name={name!r})"
            if desc:
                msg += f" [{desc}]"
            msg += f": {e}"
            raise SelectionError(msg) from e

    return selections_idx
