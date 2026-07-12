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
    ParamStatus,
)
from pydelphi.utils.io.format.format_resolver import (
    resolve_call_format_auto,
)


def _format_param_status(status, status_desc=None):
    """
    Format lifecycle status for parameter help output.

    ACTIVE parameters do not print a status line. DEPRECATED and RETIRED
    parameters print explicit user-facing lifecycle information.
    """
    if status is None:
        return None

    if status.int_value == ParamStatus.SUPPORTED.int_value:
        return None

    if status.int_value == ParamStatus.DEPRECATED.int_value:
        label = "[DEPRECATED]"
    elif status.int_value == ParamStatus.RETIRED.int_value:
        label = "[RETIRED] IGNORED, no longer supported"
    else:
        label = f"[{status.name}]"

    if status_desc:
        return f"{label} {status_desc}"

    return label


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


class ParamParseError(ValueError):
    """
    User-facing parameter-file parse/validation error.

    This exception is for invalid user input parameter files. CLI/front-end
    entry points should catch it and print str(exc), without a traceback.
    """

    def __init__(
        self,
        message,
        *,
        record=None,
        line_no=None,
        function_name=None,
        function_alias=None,
        selector=None,
        attribute=None,
        help_topic=None,
        available_help_topics=None,
    ):
        self.message = str(message)
        self.record = record
        self.line_no = line_no
        self.function_name = function_name
        self.function_alias = function_alias
        self.selector = selector
        self.attribute = attribute
        self.help_topic = help_topic
        self.available_help_topics = list(available_help_topics or [])
        super().__init__(self.render())

    def render(self):
        if self.line_no is not None:
            parts = [f"ERROR: Invalid parameter file at line {self.line_no}."]
        else:
            parts = ["ERROR: Invalid parameter file."]

        if self.record:
            parts.extend(["", "Function call:", f"  {self.record}"])

        parts.extend(["", "Problem:", f"  {self.message}"])

        if self.available_help_topics:
            parts.extend(
                [
                    "",
                    "Available selector help topics:",
                    "  " + ", ".join(self.available_help_topics),
                ]
            )

        if self.help_topic:
            parts.extend(["", "Help:", f"  Run: pydelphi-help -n {self.help_topic}"])
        elif self.available_help_topics:
            parts.extend(["", "Help:", "  Run: pydelphi-help --list-names"])

        return "\n".join(parts)


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
        self.status = ParamStatus.SUPPORTED
        self.status_desc = None


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
        status=ParamStatus.SUPPORTED,
        status_desc=None,
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
        self.status = status
        self.status_desc = status_desc
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

        outs = [
            f"{'':{indent}s}{'full_name:':{fieldwidth}s} {self.full_name}",
            f"{'':{indent}s}{'long_name:':{fieldwidth}s} {self.long_name}",
            f"{'':{indent}s}{'short_name:':{fieldwidth}s} {self.short_name}",
        ]

        status_line = _format_param_status(self.status, self.status_desc)
        if status_line:
            outs.append(f"{'':{indent}s}{'status:':{fieldwidth}s} {status_line}")

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
                # wrapped_desc_lines = option_desc
                wrapped_desc_lines = tw.fill(
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
            wrapped_long_desc = tw.fill(
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
        status=ParamStatus.SUPPORTED,
        status_desc=None,
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
        self.status = status
        self.status_desc = status_desc

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

    def help(self, width: int = 100):
        """
        Return a wrapped help string for the attribute.

        The canonical attribute name is shown first. If an alias exists and is
        different from the canonical name, it is shown explicitly so shorthand
        input forms are not hidden from users.
        """
        alias = getattr(self, "alias", None)

        if self.nameonly:
            if alias and alias != self.name:
                head = f"{self.name} or {alias}: "
            else:
                head = f"{self.name}: "
        else:
            value = self.value
            if value is None:
                value = ""

            canonical = f'{self.name}="{value}"'
            if alias and alias != self.name:
                alias_form = f'{alias}="{value}"'
                head = f"{canonical} or {alias_form}: "
            else:
                head = f"{canonical}: "

        body = self.description

        status_line = _format_param_status(self.status, self.status_desc)
        if status_line:
            body = f"{status_line}\n        {body}"

        indent = " " * 8

        return tw.fill(
            head + body,
            width=width,
            subsequent_indent=indent,
            replace_whitespace=False,
            drop_whitespace=False,
        )

    # def help(self, width: int = 100):
    #     """
    #     Return a wrapped help string for the attribute.
    #
    #     Parameters
    #     ----------
    #     width : int, default 100
    #         Maximum line width for wrapping.
    #     """
    #     if self.nameonly:
    #         head = f"{self.name}: "
    #         body = self.description
    #     else:
    #         alias_str = (
    #             f' or {self.alias}="{self.value}"' if self.name != self.alias else ""
    #         )
    #         head = f'{self.name}="{self.value}"{alias_str}: '
    #         body = self.description
    #
    #     status_line = _format_param_status(self.status, self.status_desc)
    #     if status_line:
    #         body = f"{status_line} {body}"
    #
    #     indent = " " * 8
    #
    #     return tw.fill(
    #         head + body,
    #         width=width,
    #         subsequent_indent=indent,
    #         replace_whitespace=False,
    #         drop_whitespace=False,
    #     )


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
        multicall=False,
        status=ParamStatus.SUPPORTED,
        status_desc=None,
        help_topic=None,
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
            multicall (bool, optional): Whether multiple occurances of function is supported. Defaults to False.
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
        self.multicall = bool(multicall)
        self.calls = []  # list[dict[str, Any]]; used when multicall=True
        self.issupplied = False
        self.status = status
        self.status_desc = status_desc
        self.help_topic = help_topic

    def first_nameonly_attribute(self):
        """Return the first attribute if it is the name-only selector."""
        if self.attributes and getattr(self.attributes[0], "nameonly", False):
            return self.attributes[0]
        return None

    def effective_help_topic(self, selector=None):
        """
        Return the canonical help topic for this function-style parameter.

        Selector functions use:
            function__selector

        The explicit self.help_topic wins because internal selector names can
        differ from public selector tokens.
        """
        if self.help_topic:
            return self.help_topic

        if selector:
            return f"{self.name}__{selector}"

        first = self.first_nameonly_attribute()
        if first is not None:
            public_selector = getattr(first, "alias", None) or getattr(
                first, "name", None
            )
            if public_selector:
                return f"{self.name}__{public_selector}"

        return self.name

    def _undefined_attribute_error(
        self,
        name,
        *,
        record=None,
        line_no=None,
        selector=None,
    ):
        first = self.first_nameonly_attribute()
        selector = selector or (
            getattr(first, "alias", None) or getattr(first, "name", None)
            if first is not None
            else None
        )
        func_label = (
            f"{self.name}({selector}, ...)" if selector else f"{self.name}(...)"
        )
        return ParamParseError(
            f"Undefined attribute '{name}' for {func_label}.",
            record=record,
            line_no=line_no,
            function_name=self.name,
            function_alias=self.alias,
            selector=selector,
            attribute=name,
            help_topic=self.effective_help_topic(selector),
        )

    def add_attribute(self, attrib):
        """Add a new attribute to the function."""
        if isinstance(attrib, ParamFunctionAttribute):
            self.attributes.append(attrib)
        else:
            raise AttributeError(f"Unknown attribute {attrib}")

    def set_attribute(
        self,
        name,
        value="",
        *,
        record=None,
        line_no=None,
        selector=None,
    ):
        """Set the value of an attribute by name or alias."""
        for attr in self.attributes:
            if attr.name == name or attr.alias == name:
                attr.value = value if not attr.nameonly else None
                attr.inuse = True
                return
        raise self._undefined_attribute_error(
            name,
            record=record,
            line_no=line_no,
            selector=selector,
        )

    def get_attribute(self, name):
        """Retrieve the value of an attribute by name or alias."""
        for attr in self.attributes:
            if attr.name == name or attr.alias == name:
                return attr.value
        raise self._undefined_attribute_error(name)

    def is_attribute_inuse(self, name):
        """Return whether an attribute is in use by name or alias."""
        for attr in self.attributes:
            if attr.name == name or attr.alias == name:
                return attr.inuse
        raise self._undefined_attribute_error(name)

    def activate(self):
        """Activate the function."""
        self.active = True

    def deactivate(self):
        """Deactivate the function."""
        self.active = False

    def supplied(self):
        """Mark the function input supplied by user."""
        self.issupplied = True

    def normalize_call(self, call: dict) -> dict:
        return resolve_call_format_auto(call)

    def current_call(self, *, normalize: bool = True) -> dict:
        call = {}
        for a in self.attributes:
            if a.nameonly:
                if a.inuse:
                    call[a.name] = True
            else:
                call[a.name] = a.value
        if normalize:
            return self.normalize_call(call)
        return call

    def normalize_current_attributes(self) -> None:
        call = self.current_call(normalize=True)
        for a in self.attributes:
            if a.nameonly:
                continue
            if a.name in call:
                a.value = call[a.name]
            elif a.alias in call:
                a.value = call[a.alias]

    def snapshot_call(self) -> dict:
        """
        Snapshot a single function-call spec.

        Rules:
        - nameonly attributes are markers: include them only if inuse (store True)
        - normal attributes: always include their current value (default or overridden)
          so that label/file/fmt/etc are always present in the call record.
        """
        call = {}
        for a in self.attributes:
            if a.nameonly:
                if a.inuse:
                    call[a.name] = True
            else:
                call[a.name] = a.value
        return self.normalize_call(call)

    def reset_inuse(self):
        for a in self.attributes:
            a.inuse = False

    def __str__(self):
        """Return a string representation of the function."""
        if self.active:
            return f"    {self.name}({', '.join(str(a) for a in self.attributes)})"
        return ""

    def _attribute_usage_token(self, attr):
        """
        Return one attribute's canonical syntax token for function usage examples.

        The usage header should show canonical parameter-file syntax. Shorthand
        aliases are documented below in attr.help(), not in the usage header.
        """
        attr_name = getattr(attr, "name", "")

        if getattr(attr, "nameonly", False):
            return str(attr_name)

        value = getattr(attr, "value", None)
        if value is None or value == "" or value == "0.0":
            value = "value"

        return f'{attr_name}="{value}"'

    def usage_signature(self, indent=0, linewidth=90):
        """
        Return a user-facing function-style syntax example.
        """
        pad = " " * indent
        inner_pad = " " * (indent + 2)

        tokens = [self._attribute_usage_token(attr) for attr in self.attributes]

        if not tokens:
            return f"{pad}{self.name}()"

        if len(tokens) == 1:
            candidate = f"{pad}{self.name}({tokens[0]})"
            if len(candidate) <= linewidth:
                return candidate

        lines = [f"{pad}{self.name}("]
        for i, token in enumerate(tokens):
            comma = "," if i < len(tokens) - 1 else ""
            lines.append(f"{inner_pad}{token}{comma}")
        lines.append(f"{pad})")
        return "\n".join(lines)

    def help(self, detailed=False, indent=0, fieldwidth=20, linewidth=90):
        """
        Returns detailed help information for the parameter function.

        The first block shows the expected input syntax. Attribute details,
        aliases, defaults, and descriptions follow below the syntax header.
        """
        usage = self.usage_signature(indent=indent, linewidth=linewidth)

        outs = [
            f"{'':{indent}s}{'input format:':{fieldwidth}s}",
            usage,
            "",
            f"{'':{indent}s}For valid values of attributes see below:",
            "",
            f"{'':{indent}s}{'function_name:':{fieldwidth}s} {self.name}",
            f"{'':{indent}s}{'function_alias:':{fieldwidth}s} {self.alias}",
        ]

        help_topic = getattr(self, "help_topic", None)
        if help_topic:
            outs.append(f"{'':{indent}s}{'help_topic:':{fieldwidth}s} {help_topic}")

        status_line = _format_param_status(self.status, self.status_desc)
        if status_line:
            outs.append(f"{'':{indent}s}{'status:':{fieldwidth}s} {status_line}")

        if self.attributes:
            outs.append(
                f"{'':{indent}s}{'attributes:':{fieldwidth}s}\n"
                + "\n".join(
                    [f"{'':{indent}s}* {attr.help()}" for attr in self.attributes]
                )
            )

        description = self.description_short if not detailed else self.description_long
        desc_prefix = f"{'':{indent}s}{'description:':{fieldwidth}s} "
        desc_subseq = " " * len(desc_prefix)
        wrapped_description = str(description)
        # wrapped_description = tw.fill(
        #     str(description),
        #     width=linewidth,
        #     initial_indent=desc_prefix,
        #     subsequent_indent=desc_subseq,
        # )
        outs.append(wrapped_description)

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
