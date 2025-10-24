#!/usr/bin/env python3
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


from enum import Enum


class BaseInfoEnum(Enum):
    """
    Base class for Delphi-style enums that include both an integer value and a descriptive info string.

    This module defines the `BaseInfoEnum` class, which extends Python's
    built-in `enum.Enum` to include an additional informational string
    (`info`) alongside the standard enum value. This design allows for
    self-documenting enumeration members, where each option not only has
    an associated integer identifier but also a human-readable description.

    Key features of `BaseInfoEnum` include:

    - **Dual Representation**: Each enum member stores both an integer
      value (accessible via `int_value` or `value`) and a descriptive
      `info` string.
    - **`info` Property**: Provides access to the descriptive string,
      useful for help messages, UI tooltips, and documentation.
    - **`int_value` Property**: Returns the underlying integer constant,
      suitable for serialization, configuration, or direct use in logic
      where integer identifiers are required.
    - **`list()` Class Method**: Returns a list of all public enum member
      names, useful for dynamic UI generation or input validation.
    - **`help()` Class Method**: Generates a formatted list of help
      descriptions for each enum option, combining the name and info
      string.

    This base class is intended to be inherited by other enumeration
    classes within the pydelphi project to provide consistent and
    informative configuration options.
    """

    def __new__(cls, int_value, info):
        obj = object.__new__(cls)
        obj._value_ = int_value
        obj._info = info
        return obj

    @property
    def info(self):
        """
        Returns the descriptive string associated with the enum member.

        This text can be used in help messages, GUI tooltips, or documentation to clarify
        the purpose of each option.
        """
        return self._info

    @property
    def int_value(self):
        """
        Returns the underlying primitive integer value of the enum.

        This is especially useful for serialization, storage, or logic that depends
        on raw integer constants rather than the enum object itself.
        """
        return self.value

    @classmethod
    def list(cls):
        """
        Returns a list of all enum member names defined in the class.

        Useful for constructing UI dropdowns, input validation, or introspection.
        """
        return [c.name for c in cls if not c.name.startswith("_")]

    @classmethod
    def help(cls):
        """
        Returns a list of help descriptions for each enum option.

        Each entry is formatted as '<NAME>: <description>' for quick reference.
        """
        return [f"{c.name}: {c.info}" for c in cls if not c.name.startswith("_")]
