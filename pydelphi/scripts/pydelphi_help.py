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


import argparse
import sys
import textwrap

from pydelphi.utils.io.inproc import Inputs

# Define a desired linewidth for help text
HELP_LINEWIDTH = 100  # You can adjust this value


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Detailed help for pydelphi input parameters",
        usage="%(prog)s [-g group] [-n param_name]",
    )
    parser.add_argument(
        "-g",
        "--group",
        help="Print help for parameters in the specified group",
        choices=["none", "all"]  + list(Inputs().param_groups.keys()),
        default=None,
        metavar="",
    )

    # --- Prepare necessary data structures ---
    original_param_tuples = list(Inputs().params.keys())

    alias_to_primary_map = {}
    all_valid_aliases_set = set()

    for param_name_tuple in original_param_tuples:
        primary_name = param_name_tuple[0]
        for alias in param_name_tuple:
            all_valid_aliases_set.add(alias)
            alias_to_primary_map[alias] = primary_name

    # --- Prepare parts for the wrapped HELP message with blank lines ---
    user_friendly_param_strings = [" OR ".join(t) for t in original_param_tuples]

    # Build the options section with blank lines after every 5 entries
    options_section_lines = []

    initial_options_line_prefix = "Valid parameter options include: all,"
    options_section_lines.extend(
        textwrap.wrap(
            initial_options_line_prefix,
            width=HELP_LINEWIDTH,
            initial_indent="  ",
            subsequent_indent="  ",
        )
    )

    for i in range(0, len(user_friendly_param_strings), 5):
        current_block = user_friendly_param_strings[i : i + 5]
        block_text = ", ".join(current_block)

        wrapped_block_lines = textwrap.wrap(
            block_text,
            width=HELP_LINEWIDTH,
            initial_indent="  ",
            subsequent_indent="  ",
        )
        options_section_lines.extend(wrapped_block_lines)

        if (i + 5) < len(user_friendly_param_strings):
            options_section_lines.append("")  # Blank line

    # Full wrapped help message
    intro_text = (
        "Print help for the specified parameter. "
        "Parameters may have multiple interchangeable names (aliases) "
        "separated by 'OR'. You can use any of these names to specify the parameter. "
    )
    wrapped_intro_text = textwrap.fill(intro_text, width=HELP_LINEWIDTH)
    wrapped_help_message = (
        wrapped_intro_text + "\n\n" + "\n".join(options_section_lines)
    )

    parser.add_argument(
        "-n",
        "--param-name",
        help=wrapped_help_message,
        default=None,
        metavar="PARAM_NAME",
    )

    return (
        parser.parse_args(),
        all_valid_aliases_set,
        alias_to_primary_map,
        original_param_tuples,
    )


def main():
    args, all_valid_aliases_set, alias_to_primary_map, original_param_tuples = (
        parse_arguments()
    )
    inp = Inputs()

    if args.group is not None:
        if args.group.lower() == "none":
            # Trigger argparse's built-in help display
            parser = argparse.ArgumentParser(
                description="Detailed help for pydelphi input parameters",
                usage="%(prog)s [-g group] [-n param_name]",
            )
            parser.print_help()
            sys.exit(0)
        elif args.group == "all":
            inp.help(groups=list(inp.param_groups.keys()), detailed=True)
        else:
            inp.help(groups=[args.group], detailed=True)
    elif args.param_name is not None:
        if args.param_name == "all":
            primary_param_names = [t[0] for t in Inputs().params.keys()]
            inp.help(params=primary_param_names, detailed=True)
        elif args.param_name not in all_valid_aliases_set:
            # --- Format the error message like the help message ---
            ERROR_LINEWIDTH = HELP_LINEWIDTH - 10
            ERROR_INITIAL_INDENT = "      "
            ERROR_SUBSEQUENT_INDENT = "      "

            user_friendly_param_strings = [
                " OR ".join(t) for t in original_param_tuples
            ]

            error_message_lines = []

            # Add the initial "Choose from: all," line
            initial_error_prefix = "all,"
            error_message_lines.extend(
                textwrap.wrap(
                    initial_error_prefix,
                    width=ERROR_LINEWIDTH,
                    initial_indent=ERROR_INITIAL_INDENT,
                    subsequent_indent=ERROR_SUBSEQUENT_INDENT,
                )
            )

            # Add parameters in blocks of 5, with blank lines between
            for i in range(0, len(user_friendly_param_strings), 5):
                current_block = user_friendly_param_strings[i : i + 5]
                block_text = ", ".join(current_block)

                wrapped_block_lines = textwrap.wrap(
                    block_text,
                    width=ERROR_LINEWIDTH,
                    initial_indent=ERROR_INITIAL_INDENT,
                    subsequent_indent=ERROR_SUBSEQUENT_INDENT,
                )
                error_message_lines.extend(wrapped_block_lines)

                if (i + 5) < len(user_friendly_param_strings):
                    error_message_lines.append("")  # Blank line after each block

            # Print the final error message
            print(
                f"pydelphi-help: error: argument -n/--param-name: invalid choice: '{args.param_name}'.\n"
                f"Parameter names may have multiple interchangeable names (aliases) separated by 'OR'.\n"
                f"Choose from:",
                file=sys.stderr,
            )
            print("\n".join(error_message_lines), file=sys.stderr)
            sys.exit(2)
        else:
            resolved_param_name = alias_to_primary_map.get(
                args.param_name, args.param_name
            )
            inp.help(params=[resolved_param_name], detailed=True)
    else:
        inp.help(detailed=True)


if __name__ == "__main__":
    main()
