#!/usr/bin/env python
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


r"""
Lightweight, application-level constants and mappings for biomolecular residue identifiers.
Supports bidirectional lookup from name-to-key and key-to-name.
Different biomolecular kinds are assigned exclusive residue key spaces.

Keying Scheme:
The integer key for a residue is structured to encode its kind, base residue, and variant status.
The structure is:
$$ \text{Key} = (\text{Kind Index} \times \text{Block Size}) + (\text{Base Index within Kind} \times \text{Variant Span}) + \text{Variant Offset} $$

Where:
- Kind Index: A unique integer for the biomolecule type (Protein=1, Nucleic=2, etc.).
- Block Size (`RES_KIND_BLOCK_SIZE`): The total range of keys allocated per kind.
- Base Index within Kind: An index assigned sequentially to each base residue type within a kind (0 for the first base,
                1 for the second, etc.).
- Variant Span (`RES_VARIANT_SPAN`): The number of key slots allocated for a base residue and its variants. Base residues
                get the first key in their span, and variants/non-standard residues use the subsequent keys within that span.
- Variant Offset: The position within the `RES_VARIANT_SPAN` (0 for the base, 1 to `RES_VARIANT_SPAN` - 1 for variants or
                non-standard residues).

Extracting Information from a Key:
Given a residue key `key`:
1. To find the **base key** for the residue (the key of the standard, unmodified residue):
   $$ \text{Base Key} = \text{Key} - (\text{Key} \pmod{\text{RES\_VARIANT\_SPAN}}) $$
   This works because the Kind Offset and the Base Index part of the key are designed to be divisible by `RES_VARIANT_SPAN`.
2. To find the **variant offset**:
   $$ \text{Variant Offset} = \text{Key} \pmod{\text{RES\_VARIANT\_SPAN}} $$
   If `Variant Offset` is 0, the key represents the base residue. If `Variant Offset` is > 0, it represents a variant or
                non-standard residue of that base type. Specific values of the offset can distinguish known variants.
   The first `RES_VARIANT_SPAN - 1` slots after a base key are reserved for variants and other non-standard residues related
                to that base type.
3. To find the **kind offset** (and thus infer the kind):
   $$ \text{Kind Offset} = \text{Base Key} - (\text{Base Key} \pmod{\text{RES\_KIND\_BLOCK\_SIZE}}) $$
   By comparing the `Kind Offset` to the values in the `KIND_BASE_OFFSET` dictionary, the biomolecule kind can be identified.

This scheme ensures unique keys, allows for easy lookup of base residues and variants using modulo arithmetic, and clearly
                separates key spaces by biomolecule kind.
"""

import numpy as np
from numpy import int32, array

# -----------------------------------
# Kind Indexing and Key Structure Parameters
# -----------------------------------

# Integer identifiers for different kinds of biomolecules.
# Each kind is assigned a unique integer index to structure the residue key space.
# These indices are used to calculate the base offset for each kind's key block.
RES_KIND_UNKNOWN = 0
"""Integer index for un-categorized residues."""
RES_KIND_PROTEIN = 1
"""Integer index for Protein residues."""
RES_KIND_NUCLEIC = 2
"""Integer index for Nucleic Acid residues."""
RES_KIND_LIPID = 3
"""Integer index for Lipid residues."""
RES_KIND_CARBOHYDRATE = 4
"""Integer index for Carbohydrate residues."""

RES_KIND_BLOCK_SIZE = 10000
"""
The size of the key block allocated for each biomolecular kind.
This determines the maximum range of keys available per kind (excluding the base offset).
The Kind Offset for a given kind is calculated as `Kind Index * RES_KIND_BLOCK_SIZE`.
This constant is chosen to be a multiple of `RES_VARIANT_SPAN` to facilitate the modulo logic for base key retrieval.
"""

RES_VARIANT_SPAN = 50
"""
The number of key slots reserved for a base residue and its variants/non-standard forms.
Base residues are assigned keys at intervals determined by this span (`base_key % RES_VARIANT_SPAN == 0`).
Variants and non-standard residues derived from a base are assigned keys sequentially
within the span immediately following the base key (keys with a variant offset from 1
to `RES_VARIANT_SPAN` - 1). This constant determines the maximum number of distinct
variants/non-standard residues supported per base residue.
"""

# Derived maximum number of base residues that can be defined within a single kind's block.
RES_MAX_BASES_PER_KIND = RES_KIND_BLOCK_SIZE // RES_VARIANT_SPAN
"""
The maximum number of unique base residue types that can be defined within a single kind
based on the allocated block size and the variant span.
"""

# Dictionary mapping residue kind indices to the starting key offset for that kind's block.
# The offset is calculated by multiplying the kind index by the block size.
# This ensures non-overlapping key ranges for different kinds.
_CHEM_KIND_BASE_OFFSET_DICT = {
    RES_KIND_UNKNOWN: 1,
    RES_KIND_PROTEIN: RES_KIND_BLOCK_SIZE * RES_KIND_PROTEIN,  # Starts at 10000
    RES_KIND_NUCLEIC: RES_KIND_BLOCK_SIZE * RES_KIND_NUCLEIC,  # Starts at 20000
    RES_KIND_LIPID: RES_KIND_BLOCK_SIZE * RES_KIND_LIPID,  # Starts at 30000
    RES_KIND_CARBOHYDRATE: RES_KIND_BLOCK_SIZE
    * RES_KIND_CARBOHYDRATE,  # Starts at 40000
}
"""
Dictionary mapping residue kind index (e.g., `RES_KIND_PROTEIN`) to the base integer
key offset for that kind's allocated block.
Keys for residues of a given kind will start from this offset.
Example: Protein keys start from 10000, Nucleic from 20000, etc.
"""
CHEM_KIND_BASE_OFFSET_VALUES = np.array(
    list(_CHEM_KIND_BASE_OFFSET_DICT.values()), dtype=np.int32
)
# Global dictionaries to store the bidirectional mappings between residue names (string)
# and their assigned integer keys (stored as float). These dictionaries are populated
# by the logic in this section, implementing the described keying scheme.
ResNameToResKey = {
    # Default key for unknown residue name. This key is chosen outside the
    # standard kind blocks to represent an unambiguously unknown type.
    "UNK": -99999.0,
}
"""
Dictionary mapping residue name (string, typically 3-letter code) to its unique integer key.
The key is stored as a float type. Populated by `assign_keys` and variant processing.
Includes a default entry for "UNK" (Unknown) outside the standard kind key ranges.
This dictionary and its population logic replace the definition from Part 1.
"""
ResKeyToResName = {
    # Default name for unknown residue key.
    -99999.0: "UNK",
}
"""
Dictionary mapping residue integer key (stored as float) back to its standard residue name (string).
This is the reverse mapping of `ResNameToResKey`. Populated by `assign_keys` and variant processing.
Includes a default entry for the "UNK" key.
This dictionary and its population logic replace the definition from Part 1.
"""

# -----------------------------------
# Base Residue Names Definitions
# -----------------------------------
# List of standard base protein residue names (3-letter codes).
# These names will be assigned base keys within the protein kind's block.
BaseResNameProtein = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLU",
    "GLN",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
"""
List of standard 3-letter codes for the 20 proteinogenic amino acids.
These are used as the base names for assigning keys in the protein kind block
(using `RES_KIND_PROTEIN`). This data structure is part of the keying logic from Part 2.
"""

# Placeholder for other kinds' base names lists (e.g., BaseResNameNucleic = ["A", "C", "G", "T", "U"])
# BaseResNameNucleic = [...]
# BaseResNameLipid = [...]
# BaseResNameCarbohydrate = [...]


# -----------------------------------
# Key Assignment Logic
# -----------------------------------
# Function to assign base keys to a list of residue base names for a specific kind.
# It calculates keys based on the kind's offset and variant span, and populates
# the global `ResNameToResKey` and `ResKeyToResName` dictionaries.
# Keys are assigned as float values to align with data storage conventions.
# Uses numba's jit decorator for potential performance improvement on loop.
def assign_keys(base_names, chemical_kind_index):
    """
    Assigns unique base keys to a list of residue names for a specific biomolecular kind.

    Calculates the base key for each residue name using the kind's offset
    (`CHEM_KIND_BASE_OFFSET_VALUES[kind_index]`) and assigns keys at intervals of
    `RES_VARIANT_SPAN`. Populates the global `ResNameToResKey` and `ResKeyToResName`
    dictionaries with these base keys. The assigned base key `k` satisfies
    `k % RES_VARIANT_SPAN == 0`. Keys are assigned as float values.

    Parameters
    ----------
    base_names : list of str
        A list of standard residue names (strings) that serve as bases for this kind.
    chemical_kind_index : int
        The integer index representing the biomolecular kind (e.g., `RES_KIND_PROTEIN`).
        Must be a valid index for `CHEM_KIND_BASE_OFFSET_VALUES`.
    """
    base_offset = CHEM_KIND_BASE_OFFSET_VALUES[chemical_kind_index]
    for i, base_name in enumerate(base_names):
        # Calculate the base key: Kind Offset + (Index within kind * Variant Span)
        # This ensures the base key is a multiple of RES_VARIANT_SPAN relative to the Kind Offset,
        # and the Kind Offset is also a multiple of RES_VARIANT_SPAN.
        # Thus, the base key itself is a multiple of RES_VARIANT_SPAN.
        base_key = base_offset + i * RES_VARIANT_SPAN
        # Assign the base key to the base name in both dictionaries (as float)
        ResNameToResKey[base_name] = float(base_key)
        ResKeyToResName[float(base_key)] = base_name


# Assign base keys for protein residues using the defined protein kind index.
# This populates ResNameToResKey and ResKeyToResName with standard protein residues
# at keys like 10000.0, 10050.0, 10100.0, etc.
assign_keys(BaseResNameProtein, RES_KIND_PROTEIN)

# Call assign_keys for other kinds as their base name lists are defined...
# assign_keys(BaseResNameNucleic, RES_KIND_NUCLEIC)
# assign_keys(BaseResNameLipid, RES_KIND_LIPID)
# assign_keys(BaseResNameCarbohydrate, RES_KIND_CARBOHYDRATE)

# -----------------------------------
# Variant Definitions and Assignment
# -----------------------------------

# Dictionary defining common variant residue names and their corresponding base residue names.
# This is used to assign keys to variants based on their base residue's key.
# Variants are assigned keys sequentially starting from the base key + 1,
# utilizing the slots within the `RES_VARIANT_SPAN`.
ProteinVariants = {
    "HSD": "HIS",  # CHARMM neutral histidine delta protonation state
    "HSE": "HIS",  # CHARMM neutral histidine epsilon protonation state
    "HSP": "HIS",  # CHARMM positively charged histidine
    "HID": "HIS",  # AMBER neutral histidine delta protonation state
    "HIE": "HIS",  # AMBER neutral histidine epsilon protonation state
    "HIP": "HIS",  # AMBER positively charged histidine
    "CYX": "CYS",  # Disulfide-bonded cysteine
    "ASX": "ASP",  # Ambiguous Asp/Asn (often assigned during initial PDB processing)
    "GLX": "GLU",  # Ambiguous Glu/Gln (often assigned during initial PDB processing)
}
"""
Dictionary mapping common variant residue names (string, e.g., "HSD") to their
standard base residue name (string, e.g., "HIS"). This mapping is used to find
 the base key for assigning keys to variants. This data structure is part of
 the keying logic from Part 2.
"""

# Placeholder for other kinds' variant definitions
# NucleicVariants = {...}
# LipidVariants = {...}
# CarbohydrateVariants = {...}

# Internal counter to track the next available key slot (variant offset) for variants
# of a specific base key. This ensures variants get sequential keys starting from
# the base key + 1. Resets for each base key.
_variant_counter = {}
"""
Internal dictionary used during the population of variant keys to track the next
available key offset (1 to `RES_VARIANT_SPAN` - 1) within the variant span for each
base residue key. This is part of the keying logic from Part 2.
"""

# Iterate through the defined protein variants.
# For each variant, find its base residue's key, assign a sequential key within
# the variant span, and add the mapping to the global dictionaries.
for variant_name, base_name in ProteinVariants.items():
    # Get the base key for the variant's base residue.
    # This assumes the base name already exists in ResNameToResKey from assign_keys.
    if base_name not in ResNameToResKey:
        print(
            f"Warning: Base residue '{base_name}' for variant '{variant_name}' not found in ResNameToResKey. Skipping variant assignment."
        )
        continue

    base_key = ResNameToResKey[base_name]

    # Initialize the variant counter for this base key if it's the first variant encountered.
    # The counter tracks the offset from the base key. We start from 1.
    if base_key not in _variant_counter:
        _variant_counter[base_key] = 1  # Start assigning variants from base_key + 1

    # Get the next available offset for this base key. This offset will be 1, 2, 3, ...
    offset = _variant_counter[base_key]

    # Calculate the variant key: Base Key + Variant Offset
    # Ensure the offset does not exceed the reserved variant span.
    if offset >= RES_VARIANT_SPAN:
        print(
            f"Warning: Exceeded maximum variant span ({RES_VARIANT_SPAN-1}) for base residue {base_name} (key {base_key}).",
            f"Cannot assign key for variant {variant_name}. Consider increasing RES_VARIANT_SPAN.",
        )
        continue  # Skip assigning this variant if span is full

    variant_key = float(base_key + offset)

    # Add the variant name and key to the global mapping dictionaries.
    ResNameToResKey[variant_name] = variant_key
    ResKeyToResName[variant_key] = variant_name

    # Increment the counter for the next variant of this base key.
    _variant_counter[base_key] += 1

# Process variants for other kinds similarly...
# for variant_name, base_name in NucleicVariants.items(): ...
# for variant_name, base_name in LipidVariants.items(): ...
# for variant_name, base_name in CarbohydrateVariants.items(): ...


# Dictionaries for standard protein residue names and residue keys.
# This dictionary is from Part 1 and maps 3-letter codes to 1-letter codes.
# It is NOT part of the integer keying scheme defined in Part 2.
# Source: PDB (Protein Data Bank) standard residue names.
ResNameProtein = {
    "ALA": "A",  # Alanine
    "ARG": "R",  # Arginine
    "ASN": "N",  # Asparagine
    "ASP": "D",  # Aspartic Acid
    "CYS": "C",  # Cysteine
    "GLN": "Q",  # Glutamine
    "GLU": "E",  # Glutamic Acid
    "GLY": "G",  # Glycine
    "HIS": "H",  # Histidine
    "ILE": "I",  # Isoleucine
    "LEU": "L",  # Leucine
    "LYS": "K",  # Lysine
    "MET": "M",  # Methionine
    "PHE": "F",  # Phenylalanine
    "PRO": "P",  # Proline
    "SER": "S",  # Serine
    "THR": "T",  # Threonine
    "TRP": "W",  # Tryptophan
    "TYR": "Y",  # Tyrosine
    "VAL": "V",  # Valine
    # Non-standard/modified residues might be added as needed.
}
"""
Dictionary mapping standard 3-letter protein residue codes to their
corresponding 1-letter codes. This is a separate mapping from the
integer keying scheme defined above.
"""
