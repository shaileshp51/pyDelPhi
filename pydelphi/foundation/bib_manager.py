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

"""
Minimal BibTeX reference manager for in-house use.

- Automatically loads pydelphi/data/references.bib
- Use `cite("Key")` to insert citation strings
"""

import os
import re
from typing import Dict, Union, List


class BibManager:
    def __init__(self):
        self.references: Dict[str, Dict[str, str]] = {}

    def load_bib_file(self, filepath: str) -> None:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()

        entries = re.findall(r"@(\w+)\s*{\s*([^,]+),(.*?)\n}", content, re.DOTALL)
        for entry_type, key, body in entries:
            fields = self._parse_entry_fields(body)
            fields["entry_type"] = entry_type
            self.references[key.strip()] = fields

    def _parse_entry_fields(self, body: str) -> Dict[str, str]:
        field_pattern = re.compile(r"(\w+)\s*=\s*[{\"](.*?)[}\"]\s*,?", re.DOTALL)
        return {
            field.lower(): value.strip().replace("\n", " ")
            for field, value in field_pattern.findall(body)
        }

    def get_entry(self, key: str) -> Dict[str, str]:
        return self.references.get(key, {})

    def format_citation(self, key: str) -> str:
        entry = self.get_entry(key)
        if not entry:
            return f"[{key}]"

        author = entry.get("author", "Unknown author")
        year = entry.get("year", "n.d.")
        title = entry.get("title", "Untitled")
        journal = entry.get("journal") or entry.get("booktitle", "Unknown source")
        doi = entry.get("doi", "").strip()

        authors_short = self._shorten_authors(author)
        citation = f"{authors_short} ({year}). {title}. {journal}."
        if doi:
            citation += f" https://doi.org/{doi}"
        return citation

    def _shorten_authors(self, author_field: str) -> str:
        authors = [a.strip() for a in author_field.split(" and ")]
        if len(authors) == 0:
            return "Unknown"
        elif len(authors) == 1:
            return self._last_name(authors[0])
        else:
            return f"{self._last_name(authors[0])} et al."

    def _last_name(self, name: str) -> str:
        if "," in name:
            return name.split(",")[0].strip()
        else:
            return name.strip().split()[-1]


# --- Load reference library once ---
reflib = BibManager()
_bib_path = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "references.bib")
)
if os.path.exists(_bib_path):
    reflib.load_bib_file(_bib_path)
else:
    raise FileNotFoundError(f"Missing expected references.bib at {_bib_path}")


# --- Public function to cite by key ---
def cite(keys: Union[str, List[str]]) -> str:
    """
    Return a formatted citation string for one or more BibTeX keys.

    Args:
        keys (str or list of str): One or more BibTeX citation keys.

    Returns:
        str: Formatted citation(s), separated by semicolons if multiple.

    Examples:
        >>> cite("Einstein1905")
        'Einstein (1905). On the Electrodynamics of Moving Bodies. Annalen der Physik. https://doi.org/...'

        >>> cite(["Einstein1905", "Newton1687"])
        'Einstein (1905). ...; Newton (1687). ...'
    """
    if isinstance(keys, str):
        return reflib.format_citation(keys)
    elif isinstance(keys, list):
        return "; ".join(reflib.format_citation(key) for key in keys)
    else:
        raise TypeError("`keys` must be a string or a list of strings.")
