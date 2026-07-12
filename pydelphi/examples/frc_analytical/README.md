# FRC analytical regression example

This directory contains a minimal analytical FRC regression example for PyDelPhi.

The example is based on the original DelPhi FRC toy problem: two charged source atoms generate an electrostatic potential and electric field, and one target atom is used as the evaluation point.

## Physics setup

The merged system contains three atoms:

```text
N1  SRC  at (5.0, 5.0, 0.0), charge +10e
N2  SRC  at (5.0, 0.0, 0.0), charge +20e
N3  TGT  at (0.0, 0.0, 0.0), evaluation point
```

The source selection is `SRC`, containing atoms `N1` and `N2`.

The target selection is `TGT`, containing atom `N3`.

The FRC output reports the grid electrostatic potential and grid electric-field components at `N3` due to the source atoms `N1` and `N2`.

The analytical reference treats `N1` and `N2` as point charges. DelPhi distributes charge on the grid, so small differences between analytical values and DelPhi output are expected.

Approximate analytical values are:

```text
GRID_PHI ~= 38.0
GFEx     ~= -6.6
GFEy     ~= -1.0
GFEz     ~=  0.0
```

## What is tested

This regression set tests the FRC analytical case through two input and charge/size paths.

### PQR input

The PQR case is self-contained. Coordinates, charges, and sizes are read directly from:

```text
system.pqr
```

This validates the PQR path for the modern `select(...)` + `frc(...)` workflow.

### PDB + custom charge/size input

The PDB case reads coordinates from:

```text
system.pdb
```

Charges and sizes are assigned through the paired charge/size interface:

```text
in(crgsiz, name="custom", qfile="custom.crg", rfile="custom.siz")
```

This validates that the paired `in(crgsiz, ...)` path works for PDB-based FRC runs.

## FRC target mode

This analytical example uses:

```text
tmode="ignore"
```

This is intentional.

In the original FRC toy problem, `N3` is an external evaluation point. It should not contribute source charge, excluded volume, or dielectric-boundary effects to the field-generating model.

Therefore, in the merged-system version, the target atom must be removed from the field-generating model before solving:

```text
select(name="SRC", cond="resname SRC")
select(name="TGT", cond="resname TGT")
frc(source="SRC", target="TGT", tmode="ignore", outfile="...")
```

The alternative mode, `tmode="uncharge"`, is not used in this analytical fixture because it would keep `N3` in the dielectric/excluded-volume model with zero charge. That changes the physics relative to the original external-probe-point example.

`tmode="uncharge"` should be tested in a separate molecular interaction example, such as a barnase/barstar-style case, where the target is a real molecular subset and retaining its dielectric/excluded-volume contribution is meaningful.

## Why the test matrix is small

This fixture intentionally avoids unnecessary combinations.

The analytical regression cases are:

```text
PQR input          + tmode="ignore"
PDB + custom crgsiz + tmode="ignore"
```

This is enough to validate:

- the self-contained PQR path,
- the paired custom charge/size path through `in(crgsiz, ...)`,
- the modern named-selection FRC workflow,
- the original analytical FRC physics where the target is an external evaluation point.

## What is intentionally not tested here

This fixture does not test `tmode="uncharge"`.

This fixture also does not test:

```text
frc(target_file=...)
```

The `target_file` path is retained for compatibility and advanced use, but it is not part of this core analytical regression set. Manually prepared target files can drift from the source system, especially in high-throughput workflows, and may introduce atom-order, coordinate-frame, charge/size, residue-name, or chain-name mismatches across many runs.

A separate narrow test may be added later for `target_file` parsing only.

## File naming

The cleaned fixture uses compact canonical names:

```text
system.pqr
system.pdb
custom.crg
custom.siz
```

The residue names clarify the source/target roles:

```text
SRC  source atoms N1 and N2
TGT  target/evaluation atom N3
```

In the PDB file, atom serial numbers are monotonic, and the target atom `N3` is placed after the source atoms.

## Regression outputs

Each parameter file writes an FRC output file that is compared against a matching `.ref.frc` file with numeric tolerance.

FRC metadata/comment lines begin with `# `.

The first non-comment line is the table header. Remaining non-comment lines are data rows.

`GRID_PHI` is the grid electrostatic potential interpolated at the evaluation point.

`GFE` denotes the grid electric-field components interpolated at the evaluation point: `GFEx`, `GFEy`, and `GFEz`.
