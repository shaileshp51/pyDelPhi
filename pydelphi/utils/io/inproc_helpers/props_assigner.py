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

from importlib import resources
from importlib.resources import files as resource_files
from os import path

from pydelphi.constants import (
    ATOMFIELD_CHARGE,
    ATOMFIELD_RADIUS,
    ATOMFIELD_GAUSS_SIGMA,
    ATOMFIELD_ATOMIC_NUMBER,
    ATOMFIELD_LJ_SIGMA,
    ATOMFIELD_LJ_EPSILON,
    ATOMFIELD_LJ_GAMMA,
)
from pydelphi.config.logging_config import NOTICE, WARNING, get_effective_verbosity
from pydelphi.config.global_runtime import vprint

from pydelphi.utils.io.atomkey_fields import (
    AK_RECORD,
    AK_ATOMNUM,
    AK_ATOMINDEX,
    AK_NAME,
    AK_RESNAME,
    AK_CHAIN,
    AK_RESNUM,
    AK_ATOMTYPE,
    AK_SEGID,
    AK_ATOMIC_NUMBER,
    AK_LEN_V1,
    AK_LEN_V2,
)

from pydelphi.utils.io.readers import read_pdb, read_pqr, read_siz, read_crg, read_vdw

_CRGSIZ_PACKAGE = "pydelphi.data.crgsiz"

_ASSIGN_RULE_STRICT = "strict"
_ASSIGN_RULE_LENIENT = "lenient"

_LEGACY_CRGSIZ_PRESETS = {
    "amber-legacy",
    "charmm-legacy",
}
_RETIRED_BARE_CRGSIZ_NAMES = {"amber", "charmm", "parse"}

MODULE_NAME = __name__
_VERBOSITY = get_effective_verbosity(MODULE_NAME)


class _ResolvedFileInput:
    """
    Minimal ParamFunction-like adapter for internally resolved files.

    This keeps _assign_size() and _assign_charge() unchanged in spirit: they
    still consume an object with issupplied and get_attribute("file").
    """

    def __init__(self, file_path, label):
        self.issupplied = bool(file_path)
        self._file_path = file_path
        self._label = label

    def get_attribute(self, name):
        if name == "file":
            return self._file_path
        raise KeyError(f"{self._label} has no attribute '{name}'")


def _normalize_crgsiz_mode(in_crgsiz):
    """
    Return normalized in(crgsiz, ...) mode.

    Public contract:
        mode="acquire" | "override"

    Default when in(crgsiz, ...) is not supplied:
        acquire
    """
    if in_crgsiz is None or not in_crgsiz.issupplied:
        return "acquire"

    mode = in_crgsiz.get_attribute("mode").lower()
    if mode not in {"acquire", "override"}:
        raise ValueError(
            "❌ InputError: in(crgsiz, ...) mode must be 'acquire' or 'override'."
        )

    return mode


def _normalize_internal_required(required):
    """
    Normalize internal charge/size requirement.

    This is deliberately not a user-facing in(crgsiz, ...) attribute.

    Internal values:
        None -> no external charge/size assignment
        "q"  -> charge only
        "r"  -> size only
        "qr" -> charge and size
    """
    if required in (None, ""):
        return None

    required = str(required).lower()
    if required not in {"q", "r", "qr"}:
        raise ValueError(
            "InternalError: required must be one of None, 'q', 'r', or 'qr'."
        )

    return required


def _resolve_package_crgsiz_file(setname, suffix):
    """
    Resolve a packaged charge/size preset file.

    Presets are expected under:
        pydelphi/data/crgsiz/{amber,charmm,parse}.crg
        pydelphi/data/crgsiz/{amber,charmm,parse}.siz
    """
    setname = str(setname).lower()
    suffix = str(suffix).lower()

    if setname not in {"amber", "charmm", "parse"}:
        raise ValueError(
            f"❌ InputError: unsupported charge/size preset '{setname}'. "
            "Expected one of: amber, charmm, parse."
        )

    if suffix not in {"crg", "siz"}:
        raise ValueError(
            f"InternalError: unsupported charge/size suffix '{suffix}'. "
            "Expected 'crg' or 'siz'."
        )

    file_path = str(
        resource_files("pydelphi").joinpath("data", "crgsiz", f"{setname}.{suffix}")
    )

    if not path.isfile(file_path):
        raise FileNotFoundError(f"❌ packaged charge/size file not found: {file_path}")

    return file_path


def _determine_required_crgsiz(input_kind, mode):
    """
    Determine which charge/size data must be assigned.

    input_kind:
        "pdb", "pqr", "psf", "prmtop"

    mode="acquire":
        pdb     -> "qr"
        psf     -> "r"
        pqr     -> None
        prmtop  -> None

    mode="override":
        pdb     -> "qr"
        psf     -> "qr"
        pqr     -> "qr"
        prmtop  -> "qr"
    """
    input_kind = str(input_kind).lower()
    mode = str(mode or "acquire").lower()

    if mode not in {"acquire", "override"}:
        raise ValueError(
            "❌ InputError: in(crgsiz, ...) mode must be 'acquire' or 'override'."
        )

    if mode == "override":
        return "qr"

    if input_kind == "pdb":
        return "qr"

    if input_kind == "psf":
        return "r"

    if input_kind in {"pqr", "prmtop"}:
        return None

    raise ValueError(
        "❌ InputError: unsupported input kind for charge/size assignment: "
        f"'{input_kind}'."
    )


def _init_crgsiz_presets():
    """
    Discover bundled charge/size presets from package data/crgsiz.

    A preset is available only when both files exist:

        <preset>.crg
        <preset>.siz

    The package data directory is the source of truth.
    """
    try:
        crgsiz_root = resources.files(_CRGSIZ_PACKAGE)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Could not locate bundled charge/size package: {_CRGSIZ_PACKAGE}"
        ) from exc

    crg_names = set()
    siz_names = set()

    for item in crgsiz_root.iterdir():
        if not item.is_file():
            continue

        filename = item.name
        if filename.endswith(".crg"):
            crg_names.add(filename[:-4].lower())
        elif filename.endswith(".siz"):
            siz_names.add(filename[:-4].lower())

    return sorted(crg_names & siz_names)


def _format_available_crgsiz_presets():
    presets = _init_crgsiz_presets()
    return ", ".join(presets) if presets else "(none found)"


def _resolve_package_crgsiz_file(name, ext):
    """
    Resolve a bundled charge/size preset file.

    Keeps the existing helper contract:
        _resolve_package_crgsiz_file(name, "crg") -> path-like string
        _resolve_package_crgsiz_file(name, "siz") -> path-like string
    """
    normalized = (name or "").strip().lower()

    if normalized in _RETIRED_BARE_CRGSIZ_NAMES:
        raise ValueError(
            f"❌ InputError: in(crgsiz, name='{normalized}') is ambiguous and "
            "is no longer accepted. Use an explicit preset such as "
            "'amber-ff19sb-mbondi3-set', 'amber-legacy', "
            "'charmm-c36m-prot-na-pbeq-set', or 'charmm-legacy'. "
            "Use setname='custom' with qfile/rfile for user-supplied files."
            "Run pydelphi-help -n in__crgsiz to check all acceptable preset names."
        )

    if ext not in {"crg", "siz"}:
        raise ValueError(
            f"Internal error: unsupported charge/size file extension '{ext}'."
        )

    available = set(_init_crgsiz_presets())
    if normalized not in available:
        raise ValueError(
            f"❌ InputError: unsupported in(crgsiz, ...) name '{name}'. "
            f"Available bundled presets are: {_format_available_crgsiz_presets()}. "
            "Use setname='custom' with qfile/rfile for user-supplied files."
        )

    return str(resources.files(_CRGSIZ_PACKAGE).joinpath(f"{normalized}.{ext}"))


def _resolve_crgsiz_inputs(in_crgsiz, in_siz, in_crg, required):
    """
    Resolve charge/size inputs according to internal requirement.

    Preferred user-facing input:
        in(crgsiz,
           name="<bundled-preset>" | "custom",
           mode="acquire" | "override",
           qfile="path/to/file.crg",
           rfile="path/to/file.siz")

    Bundled presets are discovered from package data/crgsiz and require both:
        <preset>.crg
        <preset>.siz

    Legacy direct inputs:
        in(crg, file="path/to/file.crg")
        in(siz, file="path/to/file.siz")

    Args:
        required:
            Internal requirement. One of None, "q", "r", "qr".
            This is not exposed in user input.

    Returns:
        size_input, charge_input
    """
    required = _normalize_internal_required(required)

    if required is None:
        return None, None

    if in_crgsiz is not None and in_crgsiz.issupplied:
        name = in_crgsiz.get_attribute("setname").lower()

        # Validate, even if the caller already used the mode to choose policy.
        _normalize_crgsiz_mode(in_crgsiz)

        charge_file = None
        size_file = None

        if name == "custom":
            if "q" in required:
                charge_file = in_crgsiz.get_attribute("qfile")
                if not charge_file:
                    raise ValueError(
                        "❌ InputError: in(crgsiz, setname='custom') requires "
                        "qfile when charge assignment is needed."
                    )

            if "r" in required:
                size_file = in_crgsiz.get_attribute("rfile")
                if not size_file:
                    raise ValueError(
                        "❌ InputError: in(crgsiz, setname='custom') requires "
                        "rfile when size assignment is needed."
                    )

        else:
            if "q" in required:
                charge_file = _resolve_package_crgsiz_file(name, "crg")
            if "r" in required:
                size_file = _resolve_package_crgsiz_file(name, "siz")

        size_input = (
            _ResolvedFileInput(size_file, "in(crgsiz) size file") if size_file else None
        )
        charge_input = (
            _ResolvedFileInput(charge_file, "in(crgsiz) charge file")
            if charge_file
            else None
        )

        return size_input, charge_input

    size_input = in_siz if "r" in required else None
    charge_input = in_crg if "q" in required else None

    return size_input, charge_input


def _infer_crgsiz_assignment_rule(in_crgsiz, in_siz=None, in_crg=None):
    """
    Decide CRG/SIZ assignment behavior without adding a user-facing rule.

    Strict assignment uses all-or-none residue-template matching. Lenient
    assignment preserves the historical atom-wise first-match behavior with
    warnings. During the transition period, explicit legacy presets and direct
    legacy in(crg)/in(siz) inputs remain lenient; all other in(crgsiz, ...)
    inputs are strict.
    """
    if in_crgsiz is not None and in_crgsiz.issupplied:
        name = str(in_crgsiz.get_attribute("setname") or "").strip().lower()
        if name in _LEGACY_CRGSIZ_PRESETS:
            return _ASSIGN_RULE_LENIENT
        return _ASSIGN_RULE_STRICT

    if (in_siz is not None and in_siz.issupplied) or (
        in_crg is not None and in_crg.issupplied
    ):
        return _ASSIGN_RULE_LENIENT

    return _ASSIGN_RULE_STRICT


def _require_param_file(param_input, label):
    """
    Validate a resolved or user-supplied CRG/SIZ input object.
    """
    if not (
        param_input
        and param_input.issupplied
        and path.isfile(param_input.get_attribute("file"))
    ):
        msg = f"Required {label} parameter file is missing."
        if param_input and param_input.issupplied:
            msg = f"{label} file not found: {param_input.get_attribute('file')}"
        raise FileNotFoundError(msg)

    return param_input.get_attribute("file")


def _candidate_residue_templates(res_name):
    """
    Return candidate CRG/SIZ residue-template names for one observed residue.
    """
    res_name = str(res_name).strip()
    candidates = [res_name]

    # AMBER terminal residue templates are four-character names such as NTHR
    # and CLEU. Keep this local and deterministic; do not infer chain breaks.
    candidates.append(("N" + res_name)[:4])
    candidates.append(("C" + res_name)[:4])

    out = []
    for candidate in candidates:
        if candidate and candidate not in out:
            out.append(candidate)
    return out


def _build_crg_template_table(charge_file):
    """
    Build an exact residue-template charge table from read_crg() output.

    Strict CRG assignment accepts only records with explicit atom and residue
    names. Wildcard-style records remain supported only by the lenient path.
    """
    charges = read_crg(charge_file)
    table = {}

    for charge_key, (ignore_match, charge) in charges.items():
        if not ignore_match[4]:
            continue

        # Strict residue-template matching requires exact atom and residue.
        if ignore_match[0] or ignore_match[1]:
            continue

        atom_name = str(charge_key[0]).strip()
        res_name = str(charge_key[1]).strip()
        table.setdefault(res_name, {})[atom_name] = float(charge)

    return table


def _build_siz_template_table(size_file):
    """
    Build an exact residue-template size table from read_siz() output.

    Strict SIZ assignment accepts only records with explicit atom and residue
    names. Wildcard-style records remain supported only by the lenient path.
    """
    sizes = read_siz(size_file)
    table = {}

    for size_key, (ignore_match, vdw_radius) in sizes.items():
        if not ignore_match[3]:
            continue

        # Strict residue-template matching requires exact atom and residue.
        if ignore_match[0] or ignore_match[1]:
            continue

        atom_name = str(size_key[0]).strip()
        res_name = str(size_key[1]).strip()
        table.setdefault(res_name, {})[atom_name] = float(vdw_radius)

    return table


def _exact_template_matches(res_name, observed_atoms, template_table):
    """
    Find templates whose atom set exactly equals the observed atom set.
    """
    observed = set(str(a).strip() for a in observed_atoms)
    matches = []
    diagnostics = []

    for template in _candidate_residue_templates(res_name):
        template_atoms = set(template_table.get(template, {}))
        missing_from_template = observed - template_atoms
        missing_from_input = template_atoms - observed

        if observed == template_atoms:
            matches.append(template)

        diagnostics.append(
            {
                "template": template,
                "missing_from_template": sorted(missing_from_template),
                "missing_from_input": sorted(missing_from_input),
            }
        )

    return matches, diagnostics


def _format_template_diagnostics(diagnostics):
    """
    Format residue-template mismatch details for strict assignment errors.
    """
    lines = []
    for item in diagnostics or []:
        template = item["template"]
        missing_from_template = item["missing_from_template"]
        missing_from_input = item["missing_from_input"]

        lines.append(f"  {template}:")
        lines.append(
            "    atoms in input but not template: "
            + (", ".join(missing_from_template) if missing_from_template else "none")
        )
        lines.append(
            "    atoms in template but not input: "
            + (", ".join(missing_from_input) if missing_from_input else "none")
        )

    return "\n".join(lines)


def _resolve_exact_residue_template(
    *,
    res_name,
    observed_atoms,
    required,
    crg_table=None,
    siz_table=None,
    residue_label="residue",
):
    """
    Resolve one residue to a single exact CRG/SIZ template.

    Assignment is all-or-none at residue level. Missing atoms, extra atoms,
    ambiguous terminal forms, unmatched atom names, and CRG/SIZ template
    disagreement are rejected.
    """
    required = _normalize_internal_required(required)
    if required not in {"q", "r", "qr"}:
        raise ValueError(
            f"InternalError: invalid residue-template requirement: {required}"
        )

    observed = sorted(set(str(a).strip() for a in observed_atoms))

    crg_matches = None
    siz_matches = None
    crg_diag = None
    siz_diag = None

    if "q" in required:
        crg_matches, crg_diag = _exact_template_matches(
            res_name, observed, crg_table or {}
        )

    if "r" in required:
        siz_matches, siz_diag = _exact_template_matches(
            res_name, observed, siz_table or {}
        )

    if required == "q":
        if len(crg_matches) == 1:
            return crg_matches[0]

        detail = _format_template_diagnostics(crg_diag)
        raise ValueError(
            "❌ InputError: strict all-or-none charge assignment failed for "
            f"{residue_label}. Observed residue name '{res_name}', atoms: "
            f"{', '.join(observed)}.\n"
            f"Tried templates: {', '.join(_candidate_residue_templates(res_name))}.\n"
            f"CRG exact matches: {', '.join(crg_matches) if crg_matches else 'none'}.\n"
            f"{detail}\n"
            "The input residue must exactly match one required CRG residue "
            "template from the selected source. Missing atoms, extra atoms, "
            "ambiguous terminal forms, and unmatched atom names are rejected."
        )

    if required == "r":
        if len(siz_matches) == 1:
            return siz_matches[0]

        detail = _format_template_diagnostics(siz_diag)
        raise ValueError(
            "❌ InputError: strict all-or-none size assignment failed for "
            f"{residue_label}. Observed residue name '{res_name}', atoms: "
            f"{', '.join(observed)}.\n"
            f"Tried templates: {', '.join(_candidate_residue_templates(res_name))}.\n"
            f"SIZ exact matches: {', '.join(siz_matches) if siz_matches else 'none'}.\n"
            f"{detail}\n"
            "The input residue must exactly match one required SIZ residue "
            "template from the selected source. Missing atoms, extra atoms, "
            "ambiguous terminal forms, and unmatched atom names are rejected."
        )

    # required == "qr"
    if len(crg_matches) == 1 and len(siz_matches) == 1:
        if crg_matches[0] == siz_matches[0]:
            return crg_matches[0]

        raise ValueError(
            "❌ InputError: charge/size residue-template disagreement for "
            f"{residue_label}. Observed residue name '{res_name}', atoms: "
            f"{', '.join(observed)}. CRG matched '{crg_matches[0]}', "
            f"but SIZ matched '{siz_matches[0]}'. Charge and size assignment "
            "must resolve to the same force-field template."
        )

    crg_detail = _format_template_diagnostics(crg_diag)
    siz_detail = _format_template_diagnostics(siz_diag)

    raise ValueError(
        "❌ InputError: strict all-or-none charge/size assignment failed for "
        f"{residue_label}. Observed residue name '{res_name}', atoms: "
        f"{', '.join(observed)}.\n"
        f"Tried templates: {', '.join(_candidate_residue_templates(res_name))}.\n"
        f"CRG exact matches: {', '.join(crg_matches) if crg_matches else 'none'}.\n"
        f"SIZ exact matches: {', '.join(siz_matches) if siz_matches else 'none'}.\n"
        "CRG diagnostics:\n"
        f"{crg_detail}\n"
        "SIZ diagnostics:\n"
        f"{siz_detail}\n"
        "The input residue must exactly match one required CRG/SIZ residue "
        "template from the selected source. Missing atoms, extra atoms, "
        "ambiguous terminal forms, unmatched atom names, and CRG/SIZ template "
        "disagreement are rejected."
    )


def _group_atom_dict_by_residue(atoms):
    """
    Group read_pdb/read_pqr atom dictionary records by observed residue.
    """
    residues = {}
    for atom_key, atom_data in atoms.items():
        residue_id = (
            atom_key[AK_RESNAME],
            atom_key[AK_CHAIN],
            atom_key[AK_RESNUM],
        )
        residues.setdefault(residue_id, []).append((atom_key, atom_data))
    return residues


def _assign_crgsiz_to_atoms_strict(atoms, size_input, charge_input, required):
    """
    Apply CRG/SIZ to atom dictionaries by strict residue-template matching.
    """
    required = _normalize_internal_required(required)

    crg_table = None
    siz_table = None

    if "q" in required:
        charge_file = _require_param_file(charge_input, "charge")
        crg_table = _build_crg_template_table(charge_file)

    if "r" in required:
        size_file = _require_param_file(size_input, "size")
        siz_table = _build_siz_template_table(size_file)

    for residue_id, records in _group_atom_dict_by_residue(atoms).items():
        res_name, chain, res_num = residue_id
        observed_atoms = [atom_key[AK_NAME] for atom_key, _ in records]
        residue_label = f"residue(name={res_name}, chain={chain}, resnum={res_num})"

        template = _resolve_exact_residue_template(
            res_name=res_name,
            observed_atoms=observed_atoms,
            required=required,
            crg_table=crg_table,
            siz_table=siz_table,
            residue_label=residue_label,
        )

        for atom_key, atom_data in records:
            atom_name = str(atom_key[AK_NAME]).strip()

            if "q" in required:
                atom_data[ATOMFIELD_CHARGE] = crg_table[template][atom_name]

            if "r" in required:
                atom_data[ATOMFIELD_RADIUS] = siz_table[template][atom_name]


def _assign_charge_size_by_policy(atoms, input_kind, in_crgsiz, in_siz, in_crg):
    """
    Assign charge and/or size according to input kind and in(crgsiz, ...) mode.

    This is the reusable property-assignment entry point for static input now
    and topology/trajectory materialization later.
    """
    mode = _normalize_crgsiz_mode(in_crgsiz)
    required = _determine_required_crgsiz(input_kind, mode)

    if required is None:
        return

    size_input, charge_input = _resolve_crgsiz_inputs(
        in_crgsiz=in_crgsiz,
        in_siz=in_siz,
        in_crg=in_crg,
        required=required,
    )

    rule = _infer_crgsiz_assignment_rule(in_crgsiz, in_siz, in_crg)

    if rule == _ASSIGN_RULE_STRICT:
        _assign_crgsiz_to_atoms_strict(
            atoms=atoms,
            size_input=size_input,
            charge_input=charge_input,
            required=required,
        )
        return

    if "r" in required:
        _assign_size(atoms, size_input)

    if "q" in required:
        _assign_charge(atoms, charge_input)


def _require_topology_identity_arrays(top, required):
    """
    Validate that TopologyLite has the identity fields needed to apply CRG/SIZ
    pattern matching.

    CRG/SIZ assignment uses atom name, residue name, chain id, and residue
    sequence. Readers that participate in acquire/override must provide enough
    normalized identity data for those matches.
    """
    if required is None:
        return

    missing = []

    if getattr(top, "atom_name", None) is None:
        missing.append("atom_name")
    if getattr(top, "res_name", None) is None:
        missing.append("res_name")
    if getattr(top, "chain_id", None) is None:
        missing.append("chain_id")
    if getattr(top, "atom_res_index", None) is None:
        missing.append("atom_res_index")
    if getattr(top, "res_seq", None) is None:
        missing.append("res_seq")

    if missing:
        raise ValueError(
            "TopologyLite is missing fields required for charge/size assignment: "
            + ", ".join(missing)
        )


def _group_topology_lite_by_residue(top):
    """
    Group TopologyLite atom indices by residue index.
    """
    residues = {}
    for atom_i in range(int(top.natoms)):
        res_i = int(top.atom_res_index[atom_i])
        residues.setdefault(res_i, []).append(atom_i)
    return residues


def _assign_crgsiz_to_topology_lite_strict(top, size_input, charge_input, required):
    """
    Apply CRG/SIZ to TopologyLite by strict residue-template matching.
    """
    required = _normalize_internal_required(required)
    _require_topology_identity_arrays(top, required)

    crg_table = None
    siz_table = None

    if "q" in required:
        charge_file = _require_param_file(charge_input, "charge")
        crg_table = _build_crg_template_table(charge_file)

    if "r" in required:
        size_file = _require_param_file(size_input, "size")
        siz_table = _build_siz_template_table(size_file)

    for res_i, atom_indices in _group_topology_lite_by_residue(top).items():
        res_name = str(top.res_name[res_i]).strip()
        chain = str(top.chain_id[res_i]).strip()
        res_num = int(top.res_seq[res_i])
        observed_atoms = [str(top.atom_name[atom_i]).strip() for atom_i in atom_indices]
        residue_label = (
            f"residue(index={res_i}, name={res_name}, "
            f"chain={chain}, resnum={res_num})"
        )

        template = _resolve_exact_residue_template(
            res_name=res_name,
            observed_atoms=observed_atoms,
            required=required,
            crg_table=crg_table,
            siz_table=siz_table,
            residue_label=residue_label,
        )

        for atom_i in atom_indices:
            atom_name = str(top.atom_name[atom_i]).strip()

            if "q" in required:
                top.atom_charge[atom_i] = crg_table[template][atom_name]

            if "r" in required:
                top.atom_radius[atom_i] = siz_table[template][atom_name]


def _assign_size_to_topology_lite(top, in_size):
    """
    Assign size values directly onto TopologyLite.atom_radius.

    This mirrors _assign_size() matching behavior but writes into the topology
    array instead of an atom dictionary.
    """
    if not (
        in_size and in_size.issupplied and path.isfile(in_size.get_attribute("file"))
    ):
        msg = (
            "Required size parameter file is missing. Use in(crgsiz, ...), "
            "or legacy in(siz, file='...')."
        )
        if in_size and in_size.issupplied:
            msg = f"size file not found: {in_size.get_attribute('file')}"
        raise FileNotFoundError(msg)

    _require_topology_identity_arrays(top, "r")

    sizes = read_siz(in_size.get_attribute("file"))
    for i in range(int(top.natoms)):
        res_i = int(top.atom_res_index[i])
        atom_name = str(top.atom_name[i])
        res_name = str(top.res_name[res_i])
        chain = str(top.chain_id[res_i])

        found = False
        for size_key, (ignore_match, vdw_radius) in sizes.items():
            if (
                ignore_match[3]
                and (ignore_match[0] or atom_name == size_key[0])
                and (ignore_match[1] or res_name == size_key[1])
                and (ignore_match[2] or chain == size_key[2])
            ):
                top.atom_radius[i] = vdw_radius
                found = True
                break

        if not found:
            vprint(
                WARNING,
                _VERBOSITY,
                "WARNING>> unassigned size: "
                f"atom(index={i}, name={atom_name}, res={res_name}, chain={chain})",
            )


def _assign_charge_to_topology_lite(top, in_charge):
    """
    Assign charge values directly onto TopologyLite.atom_charge.

    This mirrors _assign_charge() matching behavior but writes into the topology
    array instead of an atom dictionary.
    """
    if not (
        in_charge
        and in_charge.issupplied
        and path.isfile(in_charge.get_attribute("file"))
    ):
        msg = (
            "Required charge parameter file is missing. Use in(crgsiz, ...), "
            "or legacy in(crg, file='...')."
        )
        if in_charge and in_charge.issupplied:
            msg = f"charge file not found: {in_charge.get_attribute('file')}"
        raise FileNotFoundError(msg)

    _require_topology_identity_arrays(top, "q")

    charges = read_crg(in_charge.get_attribute("file"))
    for i in range(int(top.natoms)):
        res_i = int(top.atom_res_index[i])
        atom_name = str(top.atom_name[i])
        res_name = str(top.res_name[res_i])
        chain = str(top.chain_id[res_i])
        res_num = int(top.res_seq[res_i])

        found = False
        for charge_key, (ignore_match, charge) in charges.items():
            if (
                ignore_match[4]
                and (ignore_match[0] or atom_name == charge_key[0])
                and (ignore_match[1] or res_name == charge_key[1])
                and (ignore_match[2] or chain == charge_key[2])
                and (ignore_match[3] or res_num == charge_key[3])
            ):
                top.atom_charge[i] = charge
                found = True
                break

        if not found:
            vprint(
                WARNING,
                _VERBOSITY,
                "WARNING>> unassigned charge: "
                f"atom(index={i}, name={atom_name}, res={res_name}, "
                f"chain={chain}, resnum={res_num})",
            )


def _apply_crgsiz_to_topology_lite(top, input_kind, in_crgsiz, in_siz, in_crg):
    """
    Apply charge/size policy to a TopologyLite object.

    Intended internal use only. This is the topology-side counterpart of
    _assign_charge_size_by_policy(), and is used by trajectory preparation
    after the lite topology has been read but before it is frozen into the
    ensemble.

    Policy:
        mode="acquire":
            pdb     -> assign q/r
            psf     -> assign r
            pqr     -> assign nothing
            prmtop  -> assign nothing

        mode="override":
            pdb, psf, pqr, prmtop -> assign q/r

    Returns:
        The same TopologyLite object, after in-place assignment when needed.
    """
    mode = _normalize_crgsiz_mode(in_crgsiz)
    required = _determine_required_crgsiz(input_kind, mode)

    if required is None:
        return top

    size_input, charge_input = _resolve_crgsiz_inputs(
        in_crgsiz=in_crgsiz,
        in_siz=in_siz,
        in_crg=in_crg,
        required=required,
    )

    if mode == "override":
        if "q" in required:
            top.atom_charge[:] = 0.0

        if "r" in required:
            top.atom_radius[:] = 0.0

    rule = _infer_crgsiz_assignment_rule(in_crgsiz, in_siz, in_crg)

    if rule == _ASSIGN_RULE_STRICT:
        _assign_crgsiz_to_topology_lite_strict(
            top=top,
            size_input=size_input,
            charge_input=charge_input,
            required=required,
        )
        return top

    if "r" in required:
        _assign_size_to_topology_lite(top, size_input)

    if "q" in required:
        _assign_charge_to_topology_lite(top, charge_input)

    return top


def _read_atomic_data(in_modpdb4, in_pdb, in_crgsiz, in_siz, in_crg):
    """
    Reads atomic data from modpdb4/PQR or raw PDB input files, and assigns
    charge/size properties through the prop_assigner policy.

    PDB:
        Not self-contained. In acquire or override mode, both charge and size
        are required. Preferred input is in(crgsiz, ...); legacy fallback is
        in(crg, ...) plus in(siz, ...).

    PQR:
        Self-contained in acquire mode. In override mode, charge and size are
        reassigned from in(crgsiz, ...) or the legacy inputs.

    Returns:
        atoms (dict): Dictionary of atom records with size and charge fields.
        objects (list): List of auxiliary objects (e.g., geometric-shaped beads).
    """
    atoms = {}
    objects = []

    if in_modpdb4.issupplied:
        file_format = in_modpdb4.get_attribute("format").lower()
        file_path = in_modpdb4.get_attribute("file")

        if not path.isfile(file_path):
            raise FileNotFoundError(f"❌ modpdb4 file not found: {file_path}")

        if file_format != "pqr":
            raise ValueError(
                f"❌ Unsupported modpdb4 format: '{file_format}'. Expected 'pqr'."
            )

        atoms, objects = read_pqr(file_path)
        _assign_charge_size_by_policy(atoms, "pqr", in_crgsiz, in_siz, in_crg)

    elif in_pdb.issupplied:
        file_path = in_pdb.get_attribute("file")

        if not path.isfile(file_path):
            raise FileNotFoundError(f"❌ PDB file not found: {file_path}")

        atoms, objects = read_pdb(file_path)
        _assign_charge_size_by_policy(atoms, "pdb", in_crgsiz, in_siz, in_crg)

    else:
        raise ValueError(
            "❌ InputError: Neither 'in(pqr, ...)' nor 'in(pdb, ...)' was supplied."
        )

    return atoms, objects


def _assign_size(atoms, in_size):
    """
    Legacy atom-wise size assignment with warning-only misses.
    """
    if not (in_size.issupplied and path.isfile(in_size.get_attribute("file"))):
        msg = "With pdb required size param is missing. Check inputs and retry."
        if in_size.issupplied:
            msg = f"siz file: {in_size.get_attribute('file')}"
        raise FileNotFoundError(msg)

    sizes = read_siz(in_size.get_attribute("file"))
    for atom_key, atom_data in atoms.items():
        found = False
        for size_key, (ignore_match, vdw_radius) in sizes.items():
            if (
                ignore_match[3]
                and (ignore_match[0] or atom_key[AK_NAME] == size_key[0])
                and (ignore_match[1] or atom_key[AK_RESNAME] == size_key[1])
                and (ignore_match[2] or atom_key[AK_CHAIN] == size_key[2])
            ):
                atom_data[ATOMFIELD_RADIUS] = vdw_radius
                found = True
                break
        if not found:
            vprint(
                WARNING,
                _VERBOSITY,
                f"WARNING>> unassigned size: atom({atom_key}, {atom_data})",
            )


def _assign_charge(atoms, in_charge):
    """
    Legacy atom-wise charge assignment with warning-only misses.
    """
    if not (in_charge.issupplied and path.isfile(in_charge.get_attribute("file"))):
        msg = "With pdb required charge param is missing. Check inputs and retry."
        if in_charge.issupplied:
            msg = f"crg file: {in_charge.get_attribute('file')}"
        raise FileNotFoundError(msg)

    charges = read_crg(in_charge.get_attribute("file"))
    for atom_key, atom_data in atoms.items():
        found = False
        for charge_key, (ignore_match, charge) in charges.items():
            if (
                ignore_match[4]
                and (ignore_match[0] or atom_key[AK_NAME] == charge_key[0])
                and (ignore_match[1] or atom_key[AK_RESNAME] == charge_key[1])
                and (ignore_match[2] or atom_key[AK_CHAIN] == charge_key[2])
                and (ignore_match[3] or atom_key[AK_RESNUM] == charge_key[3])
            ):
                atom_data[ATOMFIELD_CHARGE] = charge
                found = True
                break
        if not found:
            vprint(
                WARNING,
                _VERBOSITY,
                f"WARNING>> unassigned charge: atom({atom_key}, {atom_data})",
            )


# Constants for Atomic Numbers (for easy reference in the code)
# 6: Carbon, 7: Nitrogen, 8: Oxygen, 1: Hydrogen
C_ATOMIC_NUM = 6
N_ATOMIC_NUM = 7
O_ATOMIC_NUM = 8
H_ATOMIC_NUM = 1


# Private helper function for parameter guessing
def _guess_vdw_parameters(atomic_number):
    # 1. Use the generic element defaults based on ATOMIC NUMBER
    if atomic_number == C_ATOMIC_NUM:  # Carbon (C)
        sigma, epsilon, gamma = 3.3997, 0.1452, 1.0000
        element_symbol = "C"
    elif atomic_number == N_ATOMIC_NUM:  # Nitrogen (N)
        sigma, epsilon, gamma = 3.2500, 0.2871, 1.0000
        element_symbol = "N"
    elif atomic_number == O_ATOMIC_NUM:  # Oxygen (O)
        sigma, epsilon, gamma = 2.9599, 0.3546, 1.0000
        element_symbol = "O"
    elif atomic_number == H_ATOMIC_NUM:  # Hydrogen (H)
        sigma, epsilon, gamma = 1.0691, 0.0265, 1.0000
        element_symbol = "H"
    else:
        # Fallback for truly unusual elements (S, P, metals, etc.)
        element_symbol = f"Atomic Num {atomic_number}"
        sigma, epsilon, gamma = 0.0, 0.0, 0.0

    return sigma, epsilon, gamma


def _assign_vdw(atoms, in_vdw):
    if not (in_vdw.issupplied and path.isfile(in_vdw.get_attribute("file"))):
        msg = "For VDW energy required param in(vdw,file='filename') is missing. Check inputs and retry."
        if in_vdw.issupplied:
            msg = f"vdw file: {in_vdw.get_attribute('file')}"
        raise FileNotFoundError(msg)

    vdw_values = read_vdw(in_vdw.get_attribute("file"))
    for atom_key, atom_data in atoms.items():
        found = False

        for vdw_key, vdw_par_values in vdw_values.items():
            if atom_key[AK_NAME] == vdw_key:
                atom_data[ATOMFIELD_LJ_SIGMA] = vdw_par_values[0]
                atom_data[ATOMFIELD_LJ_EPSILON] = vdw_par_values[1]
                atom_data[ATOMFIELD_LJ_GAMMA] = vdw_par_values[2]
                found = True
                break
        if not found:
            sigma, epsilon, gamma = _guess_vdw_parameters(
                atom_data[ATOMFIELD_ATOMIC_NUMBER]
            )
            atom_data[ATOMFIELD_LJ_SIGMA] = sigma
            atom_data[ATOMFIELD_LJ_EPSILON] = epsilon
            atom_data[ATOMFIELD_LJ_GAMMA] = gamma
            # print("atom_key=", atom_key)
            vprint(
                WARNING,
                _VERBOSITY,
                f"WARNING>> unassigned LJ params: atom({atom_key[AK_NAME]}: {'|'.join([str(t) for t in atom_key])})",
            )
            vprint(
                WARNING,
                _VERBOSITY,
                f"         Using generic guess: Sigma={sigma:.4f}, Epsilon={epsilon:.4f}, Gamma={gamma:.4f}",
            )


def _strip_wrapping_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        return s[1:-1]
    return s


def _set_param_func_attributes(
    param_obj,
    attributes_list,
    expected_names=None,
    is_float=True,
    file_check=None,
    record=None,
    line_no=None,
    selector=None,
    case_insensitive_attribs=(
        "format",
        "fmt",
        "media",
        "phase",
        "point",
        "precision",
        "prec",
        "target_mode",
        "tmode",
        "mode",
    ),
):
    """
    Parse and assign function-style attributes to a ParamFunction instance.

    This helper processes attributes of the form:
        - key=value
        - positional values (when expected_names is provided)
        - name-only flags (e.g., out(phi))

    Attribute keys are always normalized to lowercase before storage.
    Attribute values are preserved as provided, except for attributes explicitly
    listed in `case_insensitive_attribs`, whose values are stored in lowercase.

    Quoting behavior:
        - Single and double quotes in values are stripped unconditionally.
        - Quoted values containing spaces must already be preserved by the
          upstream argument splitter (e.g., for condition="...").
        - No attempt is made to preserve or interpret quotes here.

    File handling:
        - For key in {"file", "infile"}, relative paths are normalized to "./<path>".
        - If file_check == "in", the file must exist.
        - If file_check == "out", the parent directory must exist.
        - File paths are never lowercased.

    Parameters
    ----------
    param_obj :
        ParamFunction instance to receive parsed attributes.
    attributes_list : list[str]
        Attribute tokens (already split at commas, with quoted strings preserved).
    expected_names : tuple[str] or None
        If provided, attributes_list is treated as positional and zipped with these names.
    is_float : bool, default True
        If True, positional values are cast to float; otherwise kept as strings.
    file_check : {"in", "out", None}
        Enable input/output file validation for file/infile attributes.
    case_insensitive_attribs : iterable[str]
        Attribute names whose values are case-insensitive and stored in lowercase
        (e.g., format/fmt).

    Raises
    ------
    ValueError
        On malformed attributes or incorrect positional usage.
    FileNotFoundError
        If file_check validation fails.

    Notes
    -----
    - This function does not perform semantic validation; it only parses and stores values.
    - Canonicalization (e.g., mapping aliases, resolving defaults) is expected to occur
      in a later validation/finalization phase.
    - String-valued attributes not listed in case_insensitive_attribs are case-sensitive
      and preserved as provided.
    """
    ci = set(a.lower() for a in (case_insensitive_attribs or ()))

    def _maybe_normalize_value(key: str, value: str) -> str:
        # normalize values for specific case-insensitive attributes
        if key in ci:
            return value.lower()
        return value

    def _normalize_file_value(value: str) -> str:
        # keep exact case; just normalize relative path prefix if you want that behavior
        if not value.startswith(("./", "/")):
            return "./" + value
        return value

    if expected_names:
        if len(attributes_list) != len(expected_names):
            raise ValueError(
                f"Expected {len(expected_names)} positional values {expected_names}, "
                f"got {len(attributes_list)}: {attributes_list}"
            )

        for atb_name, attribute in zip(expected_names, attributes_list):
            try:
                if "=" in attribute:
                    k, v = [a.strip() for a in attribute.split("=", 1)]
                    key = k.lower()
                    v = _strip_wrapping_quotes(v)
                    v = _maybe_normalize_value(key, v)
                    value = float(v) if is_float else v
                    param_obj.set_attribute(
                        key, value, record=record, line_no=line_no, selector=selector
                    )
                else:
                    v = _strip_wrapping_quotes(attribute)
                    # positional uses the expected name as key
                    key = atb_name.lower()
                    v = _maybe_normalize_value(key, v)
                    value = float(v) if is_float else v
                    param_obj.set_attribute(
                        key, value, record=record, line_no=line_no, selector=selector
                    )
            except (ValueError, TypeError) as e:
                raise ValueError(
                    f"Invalid attribute for '{atb_name}': {attribute}"
                ) from e

    else:
        for attribute in attributes_list:
            parts = [a.strip() for a in attribute.split("=", 1)]

            if len(parts) == 2:
                key = parts[0].lower()
                value = _strip_wrapping_quotes(parts[1])

                # normalize case-insensitive attributes
                value = _maybe_normalize_value(key, value)

                # file/infile handling (case-sensitive value; do NOT lower)
                if key in ("file", "infile"):
                    value = _normalize_file_value(value)

                    if file_check == "in":
                        if not path.isfile(value):
                            raise FileNotFoundError(
                                f"Input file '{value}' does not exist."
                            )
                    elif file_check == "out":
                        out_dir = path.dirname(value) or "."
                        if not path.isdir(out_dir):
                            raise FileNotFoundError(
                                f"File directory '{out_dir}' does not exist."
                            )

                param_obj.set_attribute(
                    key, value, record=record, line_no=line_no, selector=selector
                )

            elif len(parts) == 1:
                # name-only flag like out(phi) etc.
                param_obj.set_attribute(
                    parts[0].lower(), record=record, line_no=line_no, selector=selector
                )
            else:
                raise ValueError(f"Ambiguous attribute values: {parts}")

    param_obj.supplied()
    if not param_obj.multicall:
        param_obj.normalize_current_attributes()
