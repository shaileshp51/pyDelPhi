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
Self-contained, NumPy-only selection language module for pyDelPhi atom filtering.

Selection language (minimal, property-only):
  - Grouping markers: { ... }  (braces are structural only; literal braces are disallowed)
  - Boolean ops: not, and, or  (precedence: not > and > or)
  - Predicates: key values...
  - Numeric range syntax (inclusive endpoints):
        a to b [step c]
    where step is optional (default 1), and c must be a positive integer (>= 1).
  - Hyphen-based ranges (a-b) are NOT supported (avoids ambiguity with negative numbers).

Supported keys:
  Numeric:
    - index   : 0-based position in atom_keys list
    - index1  : 1-based position in atom_keys list (normalized to index)
    - serial  : atom serial (from atom_key)
    - resid   : residue id (from atom_key) (alias: resnum)
    - atom_serial is accepted as alias for serial
  String:
    - name, resname, chain, segid, element

Public API:
  - select_atom_indices(condition, atom_keys) -> np.ndarray[int32] sorted increasing

Unified atom_key layout expected by default (v2):
  atom_key = (
      record,        # 0
      serial,        # 1  (string or int)
      atomindex,     # 2  (0-based int, unique, monotonic)
      atomname,      # 3
      resname,       # 4
      chain,         # 5
      resid,         # 6
      atomtype,      # 7
      segid,         # 8
      atomic_number, # 9  (int; guessed if element missing)
  )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from pydelphi.constants import ConstChemElement


# ----------------------------
# Public constants (atom_key layout)
# ----------------------------

from pydelphi.utils.io.atomkey_fields import (
    AK_ATOMNUM,
    AK_ATOMINDEX,
    AK_NAME,
    AK_RESNAME,
    AK_CHAIN,
    AK_RESNUM,
    AK_SEGID,
    AK_ATOMIC_NUMBER,
)


# ----------------------------
# Public error type
# ----------------------------


class SelectionError(ValueError):
    """Raised when selection condition parsing or evaluation fails."""


# ----------------------------
# Public API
# ----------------------------


def select_atom_indices(
    condition: str,
    atom_keys: Sequence[tuple],
    atomindex_field: int = AK_ATOMINDEX,
    serial_field: int = AK_ATOMNUM,
    name_field: int = AK_NAME,
    resname_field: int = AK_RESNAME,
    chain_field: int = AK_CHAIN,
    segid_field: int = AK_SEGID,
    resid_field: int = AK_RESNUM,
    atomic_number_field: int = AK_ATOMIC_NUMBER,
    unique: bool = True,
) -> np.ndarray:
    """
    Evaluate selection condition against atom_keys and return selected atom indices
    (atom_key[atomindex_field]) sorted increasing.

    Notes:
      - condition uses '{' '}' for grouping.
      - numeric range uses 'to' and optional 'step': a to b [step c]
      - endpoints are inclusive; step selects discrete values
      - hyphen ranges (a-b) are not supported (negative resid safe)
    """
    n = len(atom_keys)
    if n == 0:
        return np.empty(0, dtype=np.int32)

    ast = _parse_condition(condition)

    # list-position index selectors
    arr_index = np.arange(n, dtype=np.int32)

    # stable returned indices (atomindex)
    try:
        arr_atomindex = np.fromiter(
            (int(k[atomindex_field]) for k in atom_keys), count=n, dtype=np.int32
        )
    except Exception as e:
        raise SelectionError(
            f"Failed reading atomindex_field={atomindex_field}: {e}"
        ) from e

    # numeric fields
    arr_serial = _build_int_array(
        atom_keys, serial_field, n, field_name="serial", dtype=np.int32
    )
    arr_resid = _build_int_array(
        atom_keys, resid_field, n, field_name="resid", dtype=np.int32
    )
    arr_Z = _build_int_array(
        atom_keys, atomic_number_field, n, field_name="atomic_number", dtype=np.int16
    )

    # string fields (object arrays; switch to code arrays later if desired)
    arr_name = _build_str_array(atom_keys, name_field, n, field_name="name")
    arr_resname = _build_str_array(atom_keys, resname_field, n, field_name="resname")
    arr_chain = _build_str_array(atom_keys, chain_field, n, field_name="chain")
    arr_segid = _build_str_array(atom_keys, segid_field, n, field_name="segid")

    mask = _eval_ast(
        ast,
        n=n,
        arr_index=arr_index,
        arr_serial=arr_serial,
        arr_resid=arr_resid,
        arr_name=arr_name,
        arr_resname=arr_resname,
        arr_chain=arr_chain,
        arr_segid=arr_segid,
        arr_Z=arr_Z,
    )

    out = arr_atomindex[mask]
    out.sort()
    if unique:
        out = np.unique(out)
    return out.astype(np.int32, copy=False)


# ----------------------------
# Private: AST nodes
# ----------------------------


@dataclass(frozen=True)
class _Node:
    pass


@dataclass(frozen=True)
class _And(_Node):
    a: _Node
    b: _Node


@dataclass(frozen=True)
class _Or(_Node):
    a: _Node
    b: _Node


@dataclass(frozen=True)
class _Not(_Node):
    x: _Node


# Numeric term: either singleton or stepped range
@dataclass(frozen=True)
class _NumTerm:
    kind: str  # "singleton" | "range"
    a: int
    b: int
    step: int = 1  # for kind=="range"; must be >= 1


@dataclass(frozen=True)
class _Pred(_Node):
    key: str
    num_terms: Optional[List[_NumTerm]] = None  # numeric keys
    tokens: Optional[List[str]] = None  # string keys


# ----------------------------
# Private: condition normalization
# ----------------------------


def _normalize_braces(condition: str) -> str:
    """
    Enforce braces are grouping markers only (no literal braces / escaping).
    Validate balance; replace { -> ( and } -> ).
    """
    depth = 0
    for pos, ch in enumerate(condition):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                raise SelectionError(f"Unexpected '}}' at position {pos} in condition")
    if depth != 0:
        raise SelectionError("Unmatched '{' in condition (brace imbalance)")
    return condition.replace("{", "(").replace("}", ")")


# ----------------------------
# Private: lexer
# ----------------------------

_Token = Tuple[str, str]  # (type, value)

_BOOL_WORDS = {"and": "AND", "or": "OR", "not": "NOT"}
_RANGE_WORD = "to"
_STEP_WORD = "step"

_KEY_ALIASES = {
    "resnum": "resid",
    "atom_serial": "serial",
}
_NUM_KEYS = {"index", "index1", "serial", "resid"}
_STR_KEYS = {"name", "resname", "chain", "segid", "element"}


def _lex(s: str) -> List[_Token]:
    """
    Tokens:
      LPAREN '('
      RPAREN ')'
      AND/OR/NOT
      INT  (supports optional leading sign: -5, +10)
      IDENT
      TO    'to'
      STEP  'step'
    """
    out: List[_Token] = []
    i, n = 0, len(s)

    while i < n:
        ch = s[i]

        if ch.isspace():
            i += 1
            continue

        if ch == "(":
            out.append(("LPAREN", ch))
            i += 1
            continue

        if ch == ")":
            out.append(("RPAREN", ch))
            i += 1
            continue

        # INT with optional leading sign
        if ch.isdigit() or (ch in "+-" and (i + 1) < n and s[i + 1].isdigit()):
            j = i + 1
            while j < n and s[j].isdigit():
                j += 1
            out.append(("INT", s[i:j]))
            i = j
            continue

        if ch.isalpha() or ch == "_":
            j = i + 1
            while j < n and (s[j].isalnum() or s[j] == "_"):
                j += 1
            word = s[i:j]
            lw = word.lower()

            if lw in _BOOL_WORDS:
                out.append((_BOOL_WORDS[lw], lw))
            elif lw == _RANGE_WORD:
                out.append(("TO", lw))
            elif lw == _STEP_WORD:
                out.append(("STEP", lw))
            else:
                out.append(("IDENT", word))
            i = j
            continue

        # No hyphen-range support; '-' is only allowed as a sign on INT.
        raise SelectionError(f"Invalid character {ch!r} at position {i} in condition")

    return out


def _fmt_token_context(
    toks: List[_Token],
    i: int,
    show_braces: bool = True,
) -> str:
    """
    Return a short token context snippet with a caret at token index i.
    Displays grouping with '{}' to match user-facing syntax.
    """
    if not toks:
        return ""

    def _display(tok):
        if not show_braces:
            return tok
        if tok == "(":
            return "{"
        if tok == ")":
            return "}"
        return tok

    lo = max(0, i - 6)
    hi = min(len(toks), i + 7)

    parts = [_display(t[1]) for t in toks[lo:hi]]
    snippet = " ".join(parts)

    caret_pos = 0
    for j in range(lo, min(i, hi)):
        caret_pos += len(_display(toks[j][1])) + 1

    return snippet + "\n" + (" " * caret_pos) + "^"


# ----------------------------
# Private: parser (recursive descent)
# precedence: NOT > AND > OR
# ----------------------------


class _Parser:
    def __init__(self, toks: List[_Token]):
        self._toks = toks
        self._i = 0

    def _peek(self) -> Optional[_Token]:
        if self._i >= len(self._toks):
            return None
        return self._toks[self._i]

    def _pop(self, expected: Optional[str] = None) -> _Token:
        tok = self._peek()
        if tok is None:
            raise SelectionError("Unexpected end of condition")
        if expected is not None and tok[0] != expected:
            ctx = _fmt_token_context(self._toks, self._i)
            raise SelectionError(
                f"Syntax error: expected {expected}, got {tok[0]} ({tok[1]!r}).\n\n{ctx}"
            )
        self._i += 1
        return tok

    def parse(self) -> _Node:
        node = self._parse_or()
        tok = self._peek()
        if tok is not None:
            ttype, tval = tok
            ctx = _fmt_token_context(self._toks, self._i)

            # Most common user mistake: "{A} not {B}" (missing 'and' before not)
            if ttype == "NOT":
                raise SelectionError(
                    "Syntax error: missing boolean operator before 'not'.\n"
                    "Hint: use 'and not' or 'or not'. For example:\n"
                    "  {A} and not {B}\n\n"
                    f"{ctx}"
                )

            # More general message
            raise SelectionError(
                "Syntax error: unexpected token after a complete expression.\n"
                f"Unexpected token: {ttype} ({tval!r})\n\n"
                f"{ctx}"
            )
        return node

    def _parse_or(self) -> _Node:
        node = self._parse_and()
        while True:
            tok = self._peek()
            if tok and tok[0] == "OR":
                self._pop("OR")
                rhs = self._parse_and()
                node = _Or(node, rhs)
            else:
                break
        return node

    def _parse_and(self) -> _Node:
        node = self._parse_not()
        while True:
            tok = self._peek()
            if tok and tok[0] == "AND":
                self._pop("AND")
                rhs = self._parse_not()
                node = _And(node, rhs)
            else:
                break
        return node

    def _parse_not(self) -> _Node:
        tok = self._peek()
        if tok and tok[0] == "NOT":
            self._pop("NOT")
            return _Not(self._parse_not())
        return self._parse_primary()

    def _parse_primary(self) -> _Node:
        tok = self._peek()
        if tok is None:
            raise SelectionError("Unexpected end of condition")

        if tok[0] == "LPAREN":
            self._pop("LPAREN")
            node = self._parse_or()
            self._pop("RPAREN")
            return node

        if tok[0] == "IDENT":
            raw_key = self._pop("IDENT")[1]
            key = _KEY_ALIASES.get(raw_key.lower(), raw_key.lower())

            if key in _NUM_KEYS:
                terms = self._parse_numeric_values(key)
                if key == "index1":
                    terms = _convert_index1_terms(terms)
                    key = "index"
                return _Pred(key=key, num_terms=terms)

            if key in _STR_KEYS:
                tokens = self._parse_string_values()
                if key == "element":
                    tokens = [t.upper() for t in tokens]
                return _Pred(key=key, tokens=tokens)

            raise SelectionError(f"Unknown selector field: {raw_key!r}")

        raise SelectionError(f"Unexpected token: {tok}")

    def _parse_string_values(self) -> List[str]:
        vals: List[str] = []
        while True:
            tok = self._peek()
            if tok is None or tok[0] in ("AND", "OR", "NOT", "RPAREN", "LPAREN"):
                break
            if tok[0] == "IDENT":
                vals.append(self._pop("IDENT")[1])
                continue
            if tok[0] == "INT":
                vals.append(self._pop("INT")[1])
                continue
            if tok[0] in ("TO", "STEP"):
                raise SelectionError(f"Unexpected {tok[1]!r} in string value list")
            raise SelectionError(f"Unexpected token in values: {tok}")

        if not vals:
            raise SelectionError("Missing values after string selector field")
        return vals

    def _parse_numeric_values(self, key: str) -> List[_NumTerm]:
        """
        Parse numeric sequences composed of:
          - singletons: 121 123 -5
          - ranges:     a to b [step c]
        Endpoints inclusive; step optional, default 1; step must be >= 1.

        Hyphen-based ranges are not supported.
        """
        terms: List[_NumTerm] = []
        consumed = False

        while True:
            tok = self._peek()
            if tok is None or tok[0] in ("AND", "OR", "NOT", "RPAREN", "LPAREN"):
                break
            if tok[0] != "INT":
                raise SelectionError(f"Expected integer after {key}, got {tok}")

            a = int(self._pop("INT")[1])
            consumed = True

            tok2 = self._peek()
            if tok2 and tok2[0] == "TO":
                self._pop("TO")
                b_tok = self._pop("INT")
                b = int(b_tok[1])

                # optional: step c
                step = 1
                tok3 = self._peek()
                if tok3 and tok3[0] == "STEP":
                    self._pop("STEP")
                    c_tok = self._pop("INT")
                    step = int(c_tok[1])
                    if step <= 0:
                        raise SelectionError("step must be a positive integer (>= 1)")

                # canonicalize a<=b for simplicity (no descending ranges in v1)
                if a <= b:
                    terms.append(_NumTerm(kind="range", a=a, b=b, step=step))
                else:
                    terms.append(_NumTerm(kind="range", a=b, b=a, step=step))
            else:
                terms.append(_NumTerm(kind="singleton", a=a, b=a, step=1))

        if not consumed:
            raise SelectionError(f"Missing numeric values after {key}")
        return terms


def _parse_condition(condition: str) -> _Node:
    norm = _normalize_braces(condition)
    toks = _lex(norm)
    return _Parser(toks).parse()


def _convert_index1_terms(terms: List[_NumTerm]) -> List[_NumTerm]:
    out: List[_NumTerm] = []
    for t in terms:
        if t.kind == "singleton":
            if t.a <= 0:
                raise SelectionError("index1 must be >= 1")
            out.append(_NumTerm(kind="singleton", a=t.a - 1, b=t.a - 1, step=1))
        else:
            # range
            if t.a <= 0 or t.b <= 0:
                raise SelectionError("index1 must be >= 1")
            out.append(_NumTerm(kind="range", a=t.a - 1, b=t.b - 1, step=t.step))
    return out


# ----------------------------
# Private: evaluation
# ----------------------------


def _eval_ast(
    node: _Node,
    n: int,
    arr_index: np.ndarray,
    arr_serial: np.ndarray,
    arr_resid: np.ndarray,
    arr_name: np.ndarray,
    arr_resname: np.ndarray,
    arr_chain: np.ndarray,
    arr_segid: np.ndarray,
    arr_Z: np.ndarray,
) -> np.ndarray:
    if isinstance(node, _And):
        return _eval_ast(
            node.a,
            n=n,
            arr_index=arr_index,
            arr_serial=arr_serial,
            arr_resid=arr_resid,
            arr_name=arr_name,
            arr_resname=arr_resname,
            arr_chain=arr_chain,
            arr_segid=arr_segid,
            arr_Z=arr_Z,
        ) & _eval_ast(
            node.b,
            n=n,
            arr_index=arr_index,
            arr_serial=arr_serial,
            arr_resid=arr_resid,
            arr_name=arr_name,
            arr_resname=arr_resname,
            arr_chain=arr_chain,
            arr_segid=arr_segid,
            arr_Z=arr_Z,
        )

    if isinstance(node, _Or):
        return _eval_ast(
            node.a,
            n=n,
            arr_index=arr_index,
            arr_serial=arr_serial,
            arr_resid=arr_resid,
            arr_name=arr_name,
            arr_resname=arr_resname,
            arr_chain=arr_chain,
            arr_segid=arr_segid,
            arr_Z=arr_Z,
        ) | _eval_ast(
            node.b,
            n=n,
            arr_index=arr_index,
            arr_serial=arr_serial,
            arr_resid=arr_resid,
            arr_name=arr_name,
            arr_resname=arr_resname,
            arr_chain=arr_chain,
            arr_segid=arr_segid,
            arr_Z=arr_Z,
        )

    if isinstance(node, _Not):
        return ~_eval_ast(
            node.x,
            n=n,
            arr_index=arr_index,
            arr_serial=arr_serial,
            arr_resid=arr_resid,
            arr_name=arr_name,
            arr_resname=arr_resname,
            arr_chain=arr_chain,
            arr_segid=arr_segid,
            arr_Z=arr_Z,
        )

    if isinstance(node, _Pred):
        return _eval_pred(
            node,
            n=n,
            arr_index=arr_index,
            arr_serial=arr_serial,
            arr_resid=arr_resid,
            arr_name=arr_name,
            arr_resname=arr_resname,
            arr_chain=arr_chain,
            arr_segid=arr_segid,
            arr_Z=arr_Z,
        )

    raise TypeError(f"Unknown AST node: {type(node)}")


def _eval_pred(
    p: _Pred,
    n: int,
    arr_index: np.ndarray,
    arr_serial: np.ndarray,
    arr_resid: np.ndarray,
    arr_name: np.ndarray,
    arr_resname: np.ndarray,
    arr_chain: np.ndarray,
    arr_segid: np.ndarray,
    arr_Z: np.ndarray,
) -> np.ndarray:
    key = p.key

    # Numeric predicates
    if key in ("index", "serial", "resid"):
        terms = p.num_terms or []
        if not terms:
            raise SelectionError(
                f"Internal error: missing numeric terms for key {key!r}"
            )

        if key == "index":
            arr = arr_index
        elif key == "serial":
            arr = arr_serial
        else:
            arr = arr_resid

        mask = np.zeros(n, dtype=np.bool_)
        for t in terms:
            if t.kind == "singleton":
                mask |= arr == t.a
            else:
                # inclusive endpoints; step selects discrete values
                a, b, step = t.a, t.b, t.step
                if step == 1:
                    mask |= (arr >= a) & (arr <= b)
                else:
                    # Fast stepped range without allocating values:
                    # valid iff a<=arr<=b and (arr-a) % step == 0
                    mask |= (arr >= a) & (arr <= b) & (((arr - a) % step) == 0)
        return mask

    # String predicates
    toks = p.tokens or []
    if not toks:
        raise SelectionError(f"Internal error: missing tokens for string key {key!r}")

    if key == "name":
        return np.isin(arr_name, toks)
    if key == "resname":
        return np.isin(arr_resname, toks)
    if key == "chain":
        return np.isin(arr_chain, toks)
    if key == "segid":
        return np.isin(arr_segid, toks)
    if key == "element":
        nums = np.array([_element_symbol_to_Z(t) for t in toks], dtype=arr_Z.dtype)
        return np.isin(arr_Z, nums)

    raise SelectionError(f"Unknown selector key: {key!r}")


# ----------------------------
# Private: element mapping
# ----------------------------


def _element_symbol_to_Z(sym: str) -> int:
    s = sym.strip()
    # strict: require exact symbol spelling/case (e.g., "Zn", "Cl", "H")
    if s in ConstChemElement.__members__:
        # enum values are floats in your module
        return int(ConstChemElement[s].value)
    raise SelectionError(
        f"Unknown element symbol: {sym!r}. "
        "Element symbols are case-sensitive (e.g., H, C, N, O, Cl, Zn)."
    )


# ----------------------------
# Private: atom_keys -> arrays
# ----------------------------


def _build_int_array(
    atom_keys: Sequence[tuple],
    field_idx: int,
    n: int,
    field_name: str,
    dtype: np.dtype = np.int32,
) -> np.ndarray:
    try:
        return np.fromiter((int(k[field_idx]) for k in atom_keys), count=n, dtype=dtype)
    except Exception as e:
        raise SelectionError(
            f"Failed building int array for field '{field_name}' at index {field_idx}: {e}"
        ) from e


def _build_str_array(
    atom_keys: Sequence[tuple],
    field_idx: int,
    n: int,
    field_name: str,
) -> np.ndarray:
    try:
        return np.array([str(k[field_idx]) for k in atom_keys], dtype=object)
    except Exception as e:
        raise SelectionError(
            f"Failed building str array for field '{field_name}' at index {field_idx}: {e}"
        ) from e
