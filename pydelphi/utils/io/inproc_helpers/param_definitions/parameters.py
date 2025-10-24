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

#
# PyDelphi is free software: you can redistribute it and/or modify
# (at your option) any later version.
#
# PyDelphi is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#

import sys
from enum import Enum
from os import path
import textwrap as tw
import inspect

import numpy as np

# Import Delphi-specific configurations and enums
from pydelphi.config.global_runtime import (
    delphi_bool,
    delphi_int,
    delphi_real,
)
from pydelphi.foundation.enumbase import BaseInfoEnum
from pydelphi.foundation.enums import (
    ParamType,
)


def param_typecheck(
    prm_name, value, dtype, min_value=None, max_value=None, override=False
):
    """
    Checks and validates a parameter value against a specified data type, range, and optional constraints.

    This function ensures that the provided `value` conforms to the expected `dtype`
    and falls within the optional `min_value` and `max_value` range. It also supports
    overriding out-of-range values by clipping them to the boundary values.

    Args:
        prm_name (str): The name of the parameter being validated. Used in error messages.
        value: The value to be validated. Can be of various types, will be cast to `dtype`.
        dtype (type or Enum): The expected data type for the parameter value.
                               Supported types: int, float, bool, str, or Enum classes.
        min_value (optional): The minimum allowable value for numeric types (int, float).
                              If specified, the value must be greater than or equal to this.
        max_value (optional): The maximum allowable value for numeric types (int, float).
                              If specified, the value must be less than or equal to this.
        override (bool, optional): If True, out-of-range numeric values will be clipped
                                    to the nearest boundary value (min_value or max_value).
                                    If False (default), ValueError is raised for out-of-range values.

    Returns:
        The validated and potentially corrected value, cast to the appropriate `dtype`.

    Raises:
        ValueError: If the value is invalid (cannot be cast to `dtype`, out of range,
                    or an invalid Enum choice is provided) and `override` is False.
        TypeError: If an unsupported `dtype` is provided.
    """
    value_obj = None

    # Handle integer types
    if dtype in [int, delphi_int]:
        try:
            value_obj = int(value)
            if min_value is not None and value_obj < min_value:
                if override:
                    value_obj = min_value
                else:
                    raise ValueError(f"`{prm_name}` must be >= {min_value}.")
            if max_value is not None and value_obj > max_value:
                if override:
                    value_obj = max_value
                else:
                    raise ValueError(f"`{prm_name}` must be <= {max_value}.")
        except ValueError:
            raise ValueError(
                f"Invalid value for `{prm_name}`. Expected an integer in the range {min_value} to {max_value}."
            )

    # Handle float types
    elif dtype in [float, delphi_real]:
        try:
            value_obj = float(value)
            if min_value is not None and value_obj < min_value:
                if override:
                    value_obj = min_value
                else:
                    raise ValueError(f"`{prm_name}` must be >= {min_value}.")
            if max_value is not None and value_obj > max_value:
                if override:
                    value_obj = max_value
                else:
                    raise ValueError(f"`{prm_name}` must be <= {max_value}.")
        except ValueError:
            raise ValueError(
                f"Invalid value for `{prm_name}`. Expected a float in the range {min_value} to {max_value}."
            )

    # Handle boolean types
    elif dtype in [bool, delphi_bool]:
        value_obj = str(value).strip().lower() in ["yes", "1", "true", "on"]

    # Handle Enum types
    elif issubclass(dtype, Enum):
        try:
            if isinstance(value, dtype):
                value_obj = value
            else:
                value_str = str(value).upper().split(".")[-1]
                value_obj = dtype[value_str]
        except (KeyError, ValueError):
            raise ValueError(
                f"Invalid choice `{value}` for `{prm_name}`. Options are: {', '.join([e.name for e in dtype])}."
            )

    # Handle string types
    elif dtype == str:
        value_obj = str(value)

    # Handle cases where the value is already of the correct type
    elif isinstance(value, dtype):
        value_obj = value

    # Handle unsupported data types
    else:
        raise TypeError(f"Unexpected data type `{dtype}` for `{prm_name}`.")

    return value_obj


class Parameter:
    """
    Base class representing a generic Delphi parameter.

    This class serves as a foundation for different types of Delphi parameters
    (statements, functions, groups) by providing common attributes and an initializer.

    Attributes:
        full_name (str): Full descriptive name of the parameter (e.g., "dielectric_constant").
        long_name (str): Long alias or alternative name of the parameter (e.g., "dielectricconstant").
        short_name (str): Short alias of the parameter for concise referencing (e.g., "diel").
        partype (ParamType): Enum indicating the type of the parameter (STATEMENT, FUNCTION).
        description_short (str): Short, concise description of the parameter's purpose.
        description_long (str): Detailed description of the parameter, including usage and implications.
        required (bool): Boolean flag indicating if the parameter is mandatory for a Delphi simulation.
    """

    def __init__(self):
        self.full_name = None
        self.long_name = None
        self.short_name = None
        self.partype = None
        self.description_short = None
        self.description_long = None
        self.required = None


class ParamStatement(Parameter):
    """
    Represents a Delphi parameter statement - a simple parameter with a value.

    This class inherits from DelphiParameter and extends it to include attributes
    specific to parameter statements, such as units, data type, default value,
    value range, override behavior, and activity status.

    Attributes (inherits from DelphiParameter):
        full_name (str): Full descriptive name of the parameter.
        long_name (str): Long alias of the parameter.
        short_name (str): Short alias of the parameter.
        partype (DelphiParamType): Set to DelphiParamType.STATEMENT.
        description_short (str): Short description of the parameter.
        description_long (str): Detailed description of the parameter.
        required (bool): Whether the parameter is required.

    Attributes (specific to DelphiParamStatement):
        units (str): Unit of measurement for the parameter (e.g., "Angstrom", "dimensionless").
        dtype (type or Enum): Data type of the parameter's value (e.g., int, float, bool, Enum).
        default: Default value for the parameter if not explicitly specified.
        min_value: Minimum allowed value for the parameter (if applicable).
        max_value: Maximum allowed value for the parameter (if applicable).
        override (bool): Whether to override (clip) out-of-bound values to min/max.
        active (bool): Flag indicating if the parameter statement is currently active in the simulation.
        issupplied (bool): Flag indicating if the parameter value has been explicitly supplied by the user.
        value: The current value of the parameter. Initialized to the default value.
    """

    def __init__(
        self,
        full_name,
        long_name,
        short_name,
        units,
        dtype,
        default,
        min_value,
        max_value,
        desc_short="",
        desc_long="",
        override=True,
        required=False,
    ):
        """
        Initializes a DelphiParamStatement object.

        Args:
            full_name (str): Full descriptive name.
            long_name (str): Long alias.
            short_name (str): Short alias.
            units (str): Unit of measurement.
            dtype (type or Enum): Data type of the parameter.
            default: Default value.
            min_value: Minimum allowed value.
            max_value: Maximum allowed value.
            desc_short (str, optional): Short description. Defaults to "".
            desc_long (str, optional): Long description. Defaults to "".
            override (bool, optional): Override out-of-bound values. Defaults to True.
            required (bool, optional): Parameter is required. Defaults to False.
        """
        super().__init__()
        self.partype = ParamType.STATEMENT
        self.full_name = full_name
        self.long_name = long_name
        self.short_name = short_name
        self.units = units
        self.dtype = dtype
        self.default = default
        self.value = default
        self.min_value = min_value
        self.max_value = max_value
        self.description_short = desc_short
        self.description_long = desc_long
        self.override = override
        self.required = required
        self.active = True
        self.issupplied = False

    def activate(self):
        """Activate the parameter."""
        self.active = True

    def deactivate(self):
        """Deactivate the parameter."""
        self.active = False

    def supplied(self):
        """Mark the parameter as supplied."""
        self.issupplied = True

    def get(self):
        """Return the current value of the parameter."""
        return self.value

    def set(self, param_value):
        """Set a new value for the parameter."""
        self.value = param_value

    def __str__(self):
        """Return a formatted string representation of the parameter."""
        if self.value is not None:
            return f"    {self.full_name:<50s} = {self.value}"
        return ""

    def formatted_str(self, indent, field_width, format_specifier):
        if self.value is not None:
            field_format = f"{{:{field_width}{format_specifier}}}"
            return f"{indent}{field_format.format(self.full_name)} = {self.value}"
        return ""

    def help(self, detailed=False, indent=0, fieldwidth=12, linewidth=90):
        """
        Returns detailed help information for the parameter statement.

        Args:
            detailed (bool, optional): If True, returns the long description; otherwise, short description. Defaults to False.
            indent (int, optional): Number of spaces to indent the help output. Defaults to 0.
            fieldwidth (int, optional): Width of the attribute name field in the help output. Defaults to 12.
            linewidth (int, optional): Maximum line width for the help output. Defaults to 90.

        Returns:
            str: A formatted string containing help information for the parameter statement.
        """
        import textwrap

        outs = [
            f"{'':{indent}s}{'full_name:':{fieldwidth}s} {self.full_name}",
            f"{'':{indent}s}{'long_name:':{fieldwidth}s} {self.long_name}",
            f"{'':{indent}s}{'short_name:':{fieldwidth}s} {self.short_name}",
        ]

        if self.units is not None:
            outs.append(f"{'':{indent}s}{'unit:':{fieldwidth}s} {self.units}")

        # Check if self.dtype is a class and is a subclass of BaseInfoEnum
        # --- UPDATED LOGIC FOR ENUM DTYPE OPTIONS ---
        if inspect.isclass(self.dtype) and issubclass(self.dtype, BaseInfoEnum):
            outs.append(
                f"{'':{indent}s}{'data_type:':{fieldwidth}s} {self.dtype.__name__}"
            )
            outs.append(f"{'':{indent}s}{'options:':{fieldwidth}s}")

            enum_options_raw = []
            max_option_name_len = 0
            for option_line in self.dtype.help():
                # Split "NAME: description" into "NAME" and "description"
                if ": " in option_line:
                    option_name, option_desc = option_line.split(": ", 1)
                else:  # Fallback, though BaseInfoEnum.help() should always have a colon
                    option_name, option_desc = option_line, ""
                enum_options_raw.append((option_name, option_desc))
                if len(option_name) > max_option_name_len:
                    max_option_name_len = len(option_name)

            # Calculate the absolute indentation for the start of the option name
            option_listing_start_indent = indent + fieldwidth + 4

            # Calculate the starting column for the wrapped description text
            # This is option_listing_start_indent + (length of name_part including padding and ": ")
            description_text_start_col = (
                option_listing_start_indent + max_option_name_len + 2
            )

            # The effective width available for wrapping the description text
            effective_wrap_width = max(
                1, linewidth - description_text_start_col
            )  # Ensure width is at least 1

            for option_name, option_desc in enum_options_raw:
                # Format the option name part with left-justified padding
                name_label = f"{option_name:<{max_option_name_len}s}: "

                # Wrap the description, ensuring no initial indent as we'll apply it manually
                wrapped_desc_lines = textwrap.fill(
                    option_desc,
                    width=effective_wrap_width,
                    break_long_words=False,  # Prevents breaking words like DOIs/URLs
                    replace_whitespace=True,
                ).splitlines()

                # Add the first line of the option output (name + first part of description)
                # Apply the overall indentation for the option listing block
                first_line_content = (
                    f"{name_label}{wrapped_desc_lines[0]}"
                    if wrapped_desc_lines
                    else name_label
                )
                outs.append(f"{'':{option_listing_start_indent}s}{first_line_content}")

                # Add subsequent lines of the wrapped description, indented to align with the start of the description
                # The indentation for these lines is relative to the start of the entire option block
                subsequent_line_offset_from_option_start = (
                    max_option_name_len + 2
                )  # Offset to align with description
                for sub_line_idx in range(1, len(wrapped_desc_lines)):
                    outs.append(
                        f"{'':{option_listing_start_indent}s}{'':{subsequent_line_offset_from_option_start}s}"
                        f"{wrapped_desc_lines[sub_line_idx]}"
                    )
        else:
            # For non-Enum dtypes, just print the type name
            outs.append(
                f"{'':{indent}s}{'data_type:':{fieldwidth}s} {self.dtype.__name__ if inspect.isclass(self.dtype) else str(self.dtype)}"
            )
        # --- END UPDATED LOGIC ---

        outs.append(f"{'':{indent}s}{'default:':{fieldwidth}s} {self.default}")

        if not detailed:
            outs.append(
                f"{'':{indent}s}{'description:':{fieldwidth}s} {self.description_short}"
            )
        else:
            wrapped_long_desc = textwrap.fill(
                self.description_long,
                width=linewidth - (indent + fieldwidth + 2),
                initial_indent=" " * (indent + fieldwidth + 2),
                subsequent_indent=" " * (indent + fieldwidth + 2),
            )
            outs.append(
                f"{'':{indent}s}{'description:':{fieldwidth}s} {wrapped_long_desc.lstrip()}"
            )

        return "\n".join(outs) + "\n"


class ParamFunctionAttribute:
    """
    Represents an attribute of a Delphi parameter function.

    Attributes:
        name (str): Name of the attribute (e.g., "file", "x", "radius").
        alias (str): Alias or short form of the attribute name (e.g., "f", "x_coord", "rad").
        desc (str): Description of the attribute's purpose and usage.
        required (bool): Whether the attribute is mandatory for the function to operate.
        nameonly (bool): True if the attribute is a flag (name present implies True) and does not hold a value.
        inuse (bool): Flag indicating if the attribute is currently used or set for the function call.
        default: Default value of the attribute if not explicitly set (can be None).
        value: The currently assigned value of the attribute (can be None).
    """

    def __init__(
        self,
        name,
        alias,
        desc="",
        required=False,
        nameonly=False,
        inuse=False,
        default=None,
        value=None,
    ):
        """
        Initializes a DelphiParamFunctionAttribute object.

        Args:
            name (str): Attribute name.
            alias (str): Attribute alias.
            desc (str, optional): Description of the attribute. Defaults to "".
            required (bool, optional): Attribute is required. Defaults to False.
            nameonly (bool, optional): Attribute is a name-only flag. Defaults to False.
            inuse (bool, optional): Attribute is currently in use. Defaults to False.
            default (optional): Default value of the attribute. Defaults to None.
            value (optional): Assigned value of the attribute. Defaults to None.
        """
        self.name = name
        self.alias = alias
        self.required = required
        self.nameonly = nameonly
        self.inuse = inuse
        self.default = default
        self.value = value
        self.description = desc

    def set(self, value):
        """Set the value of the attribute."""
        self.value = value

    def set_description(self, desc):
        """Set the description of the attribute."""
        self.description = desc

    def get_description(self):
        """Get the description of the attribute."""
        return self.description

    def __str__(self):
        """Return a string representation of the attribute."""
        if self.nameonly:
            return self.name
        return '{}="{}"'.format(self.name, self.value)

    def help(self):
        """Return a help string for the attribute."""
        if self.nameonly:
            return f"{self.name}: {self.description}"
        return f'{self.name}="{self.value}": {self.description}'


class ParamFunction(Parameter):
    """
    Represents a Delphi parameter function, which is a function call with attributes as arguments.

    Attributes (inherits from DelphiParameter):
        full_name (str): Not typically used directly for functions.
        long_name (str): Not typically used directly for functions.
        short_name (str): Not typically used directly for functions.
        partype (DelphiParamType): Set to DelphiParamType.FUNCTION.
        description_short (str): Short description of the function's purpose.
        description_long (str): Detailed description of the function's functionality and attributes.
        required (bool): Whether the function is required in the input.

    Attributes (specific to DelphiParamFunction):
        attributes (list): List of DelphiParamFunctionAttribute objects associated with this function.
        active (bool): Flag indicating if this function is currently active (called) in the simulation.
    """

    def __init__(
        self,
        name,
        alias,
        attributes,
        desc_short="",
        desc_long="",
        active=False,
        required=False,
    ):
        """
        Initializes a DelphiParamFunction object.

        Args:
            name (str): Function name (e.g., "focus", "read").
            alias (str): Function alias (e.g., "foc", "in").
            attributes (list): List of DelphiParamFunctionAttribute objects for this function.
            desc_short (str, optional): Short description. Defaults to "".
            desc_long (str, optional): Long description. Defaults to "".
            active (bool, optional): Function is initially active. Defaults to False.
            required (bool, optional): Function is required. Defaults to False.
        """
        super().__init__()
        self.partype = ParamType.FUNCTION
        self.attributes = attributes
        self.name = name
        self.alias = alias
        self.description_short = desc_short
        self.description_long = desc_long
        self.active = active
        self.required = required
        self.issupplied = False

    def add_attribute(self, attrib):
        """Add a new attribute to the function."""
        if isinstance(attrib, ParamFunctionAttribute):
            self.attributes.append(attrib)
        else:
            raise AttributeError(f"Unknown attribute {attrib}")

    def set_attribute(self, name, value=""):
        """Set the value of an attribute by name or alias."""
        for attr in self.attributes:
            if attr.name == name or attr.alias == name:
                attr.value = value if not attr.nameonly else None
                attr.inuse = True
                return
        raise Exception(f"Undefined attribute: {name}")

    def get_attribute(self, name):
        """Retrieve the value of an attribute by name or alias."""
        for attr in self.attributes:
            if attr.name == name or attr.alias == name:
                return attr.value
        raise Exception(f"Undefined attribute: {name}")

    def is_attribute_inuse(self, name):
        """Retrieve the value of an attribute by name or alias."""
        for attr in self.attributes:
            if attr.name == name or attr.alias == name:
                return attr.inuse
        raise Exception(f"Undefined attribute: {name}")

    def activate(self):
        """Activate the function."""
        self.active = True

    def deactivate(self):
        """Deactivate the function."""
        self.active = False

    def supplied(self):
        """Mark the function input supplied by user."""
        self.issupplied = True

    def __str__(self):
        """Return a string representation of the function."""
        if self.active:
            return f"    {self.name}({', '.join(str(a) for a in self.attributes)})"
        return ""

    def help(self, detailed=False, indent=0, fieldwidth=12, linewidth=90):
        """
        Returns detailed help information for the parameter function.

        Args:
            detailed (bool, optional): If True, returns the long description; otherwise, short description. Defaults to False.
            indent (int, optional): Indentation level for the help output. Defaults to 0.
            fieldwidth (int, optional): Width of the field names in the help output. Defaults to 12.
            linewidth (int, optional): Maximum line width for the help output. Defaults to 90.

        Returns:
            str: Formatted help string for the parameter function, including attributes.
        """
        outs = [
            f"{'':{indent}s}{'name:':{fieldwidth}s} {self.name}",
            f"{'':{indent}s}{'alias:':{fieldwidth}s} {self.alias}",
        ]

        if self.attributes:
            outs.append(
                f"{'':{indent}s}{'attributes:':{fieldwidth}s}\n"
                + "\n".join(
                    [f"{'':{indent}s}* {attr.help()}" for attr in self.attributes]
                )
            )

        description = self.description_short if not detailed else self.description_long
        outs.append(
            f"{'':{indent}s}{'description:':{fieldwidth}s} {self.description_long}"
        )

        return "\n".join(outs) + "\n"


class ParameterGroup:
    """
    Represents a group of Delphi parameters, used to organize parameters by category.

    Attributes (inherits from DelphiParameter):
        full_name (str): Name of the group (for consistency, although group names are primarily used).
        long_name (str): Not typically used for groups.
        short_name (str): Not typically used for groups.
        partype (DelphiParamType): Not explicitly set, but conceptually represents a GROUP.
        description_short (str): Short description of the parameter group's purpose.
        description_long (str): Detailed description of the parameter group and its members.
        required (bool): Not typically used for groups.

    Attributes (specific to DelphiParameterGroup):
        name (str): Name of the parameter group (e.g., "dielectric", "gridbox").
        members (dict): Dictionary storing member parameters, keyed by (full_name, long_name, short_name) tuple.
    """

    def __init__(self, name, desc_short, desc_long):
        """
        Initializes a DelphiParameterGroup object.

        Args:
            name (str): Name of the parameter group.
            desc_short (str): Short description of the group.
            desc_long (str): Long description of the group.
        """
        super().__init__()  # Initialize superclass
        self.name = name
        self.description_short = desc_short
        self.description_long = desc_long
        self.members = {}  # Dictionary to hold member parameters

    def add_member(self, member):
        """
        Adds a new member to the parameter group if it is not already present.

        Args:
            member (object): The member object to be added. The object should have
                             attributes `full_name`, `long_name`, and `short_name`.

        Note:
            The member is added only if its (full_name, long_name, short_name) tuple is not
            already a key in the `members` dictionary.
        """
        if member.partype.int_value == ParamType.STATEMENT.int_value:
            key = (member.full_name, member.long_name, member.short_name)
        elif member.partype.int_value == ParamType.FUNCTION.int_value:
            func_name_attrib = ""
            if len(member.attributes) and member.attributes[0].nameonly:
                func_name_attrib = member.attributes[0].name
            key1 = (
                f"{member.name}" + f".{func_name_attrib}"
                if func_name_attrib
                else f"{member.name}"
            )
            key2 = (
                f"{member.alias}" + f".{func_name_attrib}"
                if func_name_attrib
                else f"{member.alias}"
            )
            key = (key1, key2, key2)
        if key not in self.members:
            self.members[key] = member

    def help(self, detailed=True, grpindent=0, fieldwidth=20, linewidth=80):
        """
        Generates a formatted help string for the parameter group and its members.

        Args:
            detailed (bool, optional): Use detailed descriptions for members if True, else short descriptions. Defaults to True.
            grpindent (int, optional): Indentation level for the group header. Defaults to 0.
            fieldwidth (int, optional): Field width for member help formatting. Defaults to 20.
            linewidth (int, optional): Maximum line width for the help output. Defaults to 80.

        Returns:
            str: Formatted help string describing the parameter group and its members.
        """
        description = self.description_long if detailed else self.description_short
        group_header = "\n".join(
            tw.wrap(
                f"DelphiParameterGroup = {self.name}: {description}",
                initial_indent=f"{'':{grpindent}s}",
                subsequent_indent=f"{'':{grpindent + 2}s}",
                width=linewidth,
            )
        )
        output = [group_header, "-" * linewidth]

        # Add help for each member
        num_members = len(self.members)
        for idx, member in enumerate(self.members.values()):
            output.append(
                member.help(
                    detailed=detailed,
                    indent=grpindent + 2,
                    fieldwidth=fieldwidth,
                    linewidth=linewidth,
                )
            )
            if idx != num_members - 1:
                output.append(f"{'.' * linewidth}")

        output.append("=" * linewidth)
        return "\n".join(output)
