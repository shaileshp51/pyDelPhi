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

"""
Command-line help for pyDelPhi input parameters.
"""

import argparse
import sys
import textwrap

from pydelphi.utils.io.inproc import Inputs


HELP_LINEWIDTH = 100


def _dedupe_keep_order(items):
    out = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _format_name_groups(
    name_groups,
    *,
    width=HELP_LINEWIDTH,
    initial_indent="  ",
    subsequent_indent="  ",
    block_size=5,
    include_all=False,
):
    groups = [" OR ".join(_dedupe_keep_order(group)) for group in name_groups]

    if include_all:
        groups.insert(0, "all")

    lines = []
    for i in range(0, len(groups), block_size):
        block = groups[i : i + block_size]
        block_text = ", ".join(block)
        if (i + block_size) < len(groups):
            block_text += ","

        lines.extend(
            textwrap.wrap(
                block_text,
                width=width,
                initial_indent=initial_indent,
                subsequent_indent=subsequent_indent,
            )
        )

        if (i + block_size) < len(groups):
            lines.append("")

    return "\n".join(lines)


def _build_param_maps(inp):
    original_param_tuples = list(inp.params.keys())

    alias_to_primary_map = {}
    all_valid_aliases_set = set()

    for param_name_tuple in original_param_tuples:
        primary_name = param_name_tuple[0]
        for alias in param_name_tuple:
            all_valid_aliases_set.add(alias)
            alias_to_primary_map[alias] = primary_name

    return original_param_tuples, alias_to_primary_map, all_valid_aliases_set


def _print_help_topic_convention(file=sys.stdout):
    print(
        "Help topic convention:\n"
        "  name                  statement-style parameter, e.g. grid_size\n"
        "  function              selector-free function, e.g. zeta for zeta(...)\n"
        "  function__namedattr   function-style parameter construct, e.g. in__crgsiz for in(crgsiz, ...)",
        file=file,
    )


def _print_param_names(original_param_tuples, *, include_all=True, file=sys.stdout):
    _print_help_topic_convention(file=file)
    print("", file=file)
    print(
        "Note: topics like func__namedattr represent function-style parameter "
        "constructs such as func(namedattr, ...).",
        file=file,
    )
    print("", file=file)
    print("Valid parameter help topics:", file=file)
    print(
        _format_name_groups(
            original_param_tuples,
            width=HELP_LINEWIDTH,
            initial_indent="  ",
            subsequent_indent="  ",
            include_all=include_all,
        ),
        file=file,
    )


def _print_groups(inp, *, include_all=True, file=sys.stdout):
    groups = list(inp.param_groups.keys())
    if include_all:
        groups = ["all"] + groups

    print("Valid parameter groups:", file=file)
    print(
        _format_name_groups(
            [(group,) for group in groups],
            width=HELP_LINEWIDTH,
            initial_indent="  ",
            subsequent_indent="  ",
            include_all=False,
        ),
        file=file,
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description="Detailed help for pydelphi input parameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Help topic convention:\n"
            "  name                  statement-style parameter, e.g. grid_size\n"
            "  function              selector-free function, e.g. zeta for zeta(...)\n"
            "  function__namedattr   function-style parameter construct, e.g. in__crgsiz for in(crgsiz, ...)\n\n"
            "Examples:\n"
            "  pydelphi-help -n grid_size\n"
            "  pydelphi-help -n in__crgsiz\n"
            "  pydelphi-help -g infile\n"
            "  pydelphi-help --list-param-names\n"
            "  pydelphi-help --list-groups"
        ),
    )

    parser.add_argument(
        "-g",
        "--group",
        metavar="GROUP",
        help=(
            "Print help for parameters in the specified group. "
            "Use --list-groups to show valid groups."
        ),
    )

    parser.add_argument(
        "-n",
        "--param-name",
        metavar="PARAM_NAME",
        help=(
            "Print help for the specified parameter/function help topic. "
            "Use --list-param-names to show valid names."
        ),
    )

    parser.add_argument(
        "-ln",
        "--list-param-names",
        action="store_true",
        help="List valid parameter/function help topics.",
    )

    parser.add_argument(
        "-lg",
        "--list-groups",
        action="store_true",
        help="List valid parameter groups.",
    )

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    inp = Inputs()
    original_param_tuples, alias_to_primary_map, all_valid_aliases_set = (
        _build_param_maps(inp)
    )

    if not any(
        (
            args.group,
            args.param_name,
            args.list_param_names,
            args.list_groups,
        )
    ):
        parser.print_help()
        return 0

    if args.list_param_names:
        _print_param_names(original_param_tuples, include_all=True)
        return 0

    if args.list_groups:
        _print_groups(inp, include_all=True)
        return 0

    if args.group is not None:
        group = args.group.strip()

        if group == "all":
            inp.help(groups=list(inp.param_groups.keys()), detailed=True)
            return 0

        if group not in inp.param_groups:
            print(
                f"pydelphi-help: error: unknown group: '{group}'.\n\n"
                "Run:\n"
                "  pydelphi-help --list-groups",
                file=sys.stderr,
            )
            return 2

        inp.help(groups=[group], detailed=True)
        return 0

    if args.param_name is not None:
        param_name = args.param_name.strip()

        if param_name == "all":
            primary_param_names = [t[0] for t in original_param_tuples]
            inp.help(params=primary_param_names, detailed=True)
            return 0

        if param_name not in all_valid_aliases_set:
            print(
                f"pydelphi-help: error: unknown parameter help topic: '{param_name}'.\n\n"
                "Run:\n"
                "  pydelphi-help --list-param-names\n\n"
                "For selector-style function help, use:\n"
                "  function__selector\n\n"
                "Example:\n"
                "  pydelphi-help -n in__crgsiz",
                file=sys.stderr,
            )
            return 2

        resolved_param_name = alias_to_primary_map[param_name]
        inp.help(params=[resolved_param_name], detailed=True)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
