# pyDelPhi Example: PDB 5TIF

This directory contains a complete, ready-to-run example for the protein system **PDB 5TIF**, distributed with the `pyDelPhi` package under `examples/5tif/`.

It includes pre-generated PQR and parameter (`.prm`) files for both **Gaussian** and **traditional** Poisson–Boltzmann (PB) calculations, allowing you to run pyDelPhi immediately without additional preprocessing. This example illustrates the basic workflow and use of the `pydelphi-static` command-line interface.  
Comprehensive numerical verification is covered separately by the automated regression-test suite.

---

## Getting Started

From the root of the repository:

```bash
cd examples/5tif
```

To inspect all available command-line options:

```bash
pydelphi-static --help
```

This lists configurable settings grouped by category, including:

- Platform selection (`cpu`, `cuda`)
- Numerical precision (`single`, `double`)
- Solver and grid configuration
- File handling and output settings

All example commands below use:

- `--platform cpu`
- `--precision double`

for broad hardware compatibility. CUDA-enabled systems may substitute `--platform cuda`.

---

## Provided Input Files

The directory includes parameter files for several PB formulations:

- `param_5tif_linear_trad.prm`  
  Linear PB, **classical two-dielectric model**.

- `param_5tif_linear_gaussian.prm`  
  Linear PB, **Gaussian dielectric model**.

- `param_5tif_nonlinear_trad.prm`  
  **Nonlinear** PB, classical two-dielectric model.

Pre-generated PQR and auxiliary files required to run these examples are also provided.

---

## Running the Examples

Each command executes a **single static electrostatic potential calculation** using the specified parameter file.

The `--overwrite` flag replaces any existing output files without asking for confirmation.

> **Important:**  
> **pyDelPhi does *not* write potential maps (`.phimap`) by default.**  
> Maps must be explicitly requested in the `.prm` file using the appropriate output directives.

### 1. Linear PBE — Traditional Two-Dielectric Model

```bash
pydelphi-static   --platform cpu   --precision double   --param-file param_5tif_linear_trad.prm   --overwrite
```

This solves the linearized Poisson–Boltzmann equation using the classical DelPhi-style two-dielectric interface model.

---

### 2. Linear PBE — Gaussian Dielectric Model

```bash
pydelphi-static   --platform cpu   --precision double   --param-file param_5tif_linear_gaussian.prm   --overwrite
```

This run uses the **Gaussian dielectric representation**, enabling a smooth permittivity transition across the solute–solvent boundary.

---

### 3. Nonlinear PBE — Traditional Two-Dielectric Model

```bash
pydelphi-static   --platform cpu   --precision double   --param-file param_5tif_nonlinear_trad.prm   --overwrite
```

This activates the **nonlinear Poisson–Boltzmann equation**, incorporating charge-dependent dielectric response and finite-ion effects.

---

## GPU Usage

If your installation supports CUDA, replace `--platform cpu` with:

```bash
--platform cuda
```

For example:

```bash
pydelphi-static   --platform cuda   --precision double   --param-file param_5tif_linear_trad.prm   --overwrite
```

Any of the provided parameter files can be used on GPU.

---

## Output Files

Upon completion, pyDelPhi writes results to the current directory. Depending on the `.prm` configuration, outputs may include:

- **Energy summaries** (`*.tsv`)
- **Log files** with runtime details
- **Potential maps** (`*.phimap`) — *only if explicitly enabled in the parameter file*
- **Auxiliary diagnostic files**

Refer to the parameter file reference for enabling map/output generation.

---

## Additional Resources

- Use `pydelphi-static --help` to browse all options.
- Detailed validation, performance benchmarks, and methodology descriptions are provided in the project documentation and the Supporting Information of the associated publication.

