# 🧩 pyDelPhi: A Modern, High-Performance Poisson–Boltzmann Solver

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-yellow.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Numba](https://img.shields.io/badge/Accelerated%20by-Numba-lightgrey.svg)](https://numba.pydata.org/)

---

**pyDelPhi** is a high-performance, Python-based reimplementation and extension of the classic **DelPhi** electrostatics solver.  
It provides accurate and efficient solutions to the **Poisson–Boltzmann (PB)** equation for biomolecular systems,  
with both **CPU** and **GPU (CUDA)** acceleration.

---

## ✨ Key Features

- **Physics-Faithful Reimplementation**
  - Fully compatible with DelPhi 8.5 reference outputs.
  - Validated across protein, nucleic-acid, and viral capsid benchmarks.

- **High-Performance Backends**
  - CPU parallelization via `Numba` and `prange`.
  - GPU acceleration through custom CUDA kernels optimized for A100-class devices.

- **Model Support**
  - Linear and nonlinear PB formulations.
  - Traditional two-dielectric and Gaussian dielectric models.
  - Cubic and cuboidal grid geometries with automatic padding control.

- **Precision and Solvers**
  - Single / double precision arithmetic.
  - Successive Over-Relaxation (SOR) and Newton-like (NWT) iterative solvers.

- **Modular and Extensible Architecture**
  - Designed for scientific transparency, benchmarking, and reproducibility.
  
---

## ⚙️ Installation

### Requirements
- Python ≥ 3.10  
- CUDA Toolkit ≥ 12.0 (optional for GPU backend)

Core dependencies:
```bash
numpy
numba
scipy
pandas
matplotlib
```

Optional (profiling / plots):
```bash
seaborn
psutil
```

### From Source
```bash
git clone https://github.com/<your-org>/pyDelPhi.git
cd pyDelPhi
pip install -e .
```

Verify installation:
```bash
python -m pydelphi --version
```

---

## 🚀 Quick Start

### 🔹 Command-Line Usage (Recommended for End Users)

pyDelPhi provides three primary executables:

| Command | Purpose |
|----------|----------|
| `pydelphi-static` | Run a single Poisson–Boltzmann (PB) electrostatics calculation |
| `pydelphi-test` | Execute regression and consistency tests |
| `pydelphi-help` | Access built-in documentation and parameter reference |

---

#### 🧮 `pydelphi-static` — Main Solver

Run the solver on a biomolecular system:
```bash
pydelphi-static --param-file params.inp --platform cuda --precision double --threads 32
```

**Usage**
```
usage: pydelphi_static.py [-h] [-V] [-P {cpu,cuda}] [-p {single,double}]
                          [-t THREADS] [-d DEVICE_ID] [-f PARAM_FILE]
                          [-v {critical,error,notice,warning,info,debug,trace}]
                          [-l LABEL] [-o OUTFILE] [-O] [-S]
```

| Flag | Description | Default |
|------|--------------|----------|
| `-h`, `--help` | Show help and exit | — |
| `-V`, `--version` | Print version and exit | — |
| `-P`, `--platform {cpu,cuda}` | Compute platform | `cpu` |
| `-p`, `--precision {single,double}` | Real precision | `double` |
| `-t`, `--threads` | Number of CPU threads | `1` |
| `-d`, `--device-id` | GPU device ID | `0` |
| `-f`, `--param-file` | Input parameter file (required) | — |
| `-v`, `--verbosity` | Output verbosity (`critical`→`trace`) | `info` |
| `-l`, `--label` | Label for run | `pdbid` |
| `-o`, `--outfile` | Output CSV filename | `outputs.csv` |
| `-O`, `--overwrite` | Overwrite output file | `False` |
| `-S`, `--setup-timing` | Print setup timing | `False` |

Example:
```bash
pydelphi-static -f examples/1CRN_params.inp -P cuda -p double -t 64 -l 1CRN
```

---

#### 🧪 `pydelphi-test` — Regression and Validation Suite

Run automated regression tests to verify consistency and numerical accuracy.

```bash
pydelphi-test --help
```

**Usage**
```
usage: pydelphi-test [-h] [--no-cuda] [--no-parallel] [--no-single] [--no-double]
```

| Flag | Description |
|------|--------------|
| `-h`, `--help` | Show help and exit |
| `--no-cuda` | Skip tests involving CUDA platforms |
| `--no-parallel` | Skip tests with more than one thread |
| `--no-single` | Skip tests using single precision |
| `--no-double` | Skip tests using double precision |

Example:
```bash
pydelphi-test --no-cuda --no-parallel
```

Use these flags to suppress specific test categories when certain platforms or configurations (e.g., GPU hardware) are unavailable.  
This ensures clean, reproducible regression runs across heterogeneous environments.

---

#### 📘 `pydelphi-help` — Built-in Parameter Documentation

Interactive access to parameter definitions, defaults, and references.

```bash
pydelphi-help -h
```

**Usage**
```
usage: pydelphi-help [-g group] [-n param_name]
```

| Flag | Description |
|------|--------------|
| `-g`, `--group` | Print help for all parameters in a group |
| `-n`, `--param-name` | Show detailed help for one parameter (supports aliases) |
| `-h`, `--help` | Show this message and exit |

Examples:
```bash
pydelphi-help -g grid
pydelphi-help -n surfmethod
```

**Sample Output**
```
full_name:   surface_method
long_name:   surfacemethod
short_name:  surfmethod
unit:
data_type:   SurfaceMethod
options:
    VDW           : Van der Waals surface. (Rocchia et al. 2001, JCC https://doi.org/10.1002/jcc.1161)
    GAUSSIAN      : Gaussian smoothed surface. (Panday et al. 2024, JCC https://doi.org/10.1002/jcc.27496)
    GAUSSIANCUTOFF: Cutoff-based Gaussian surface (vacuum). (Chakravorty et al. 2018, JCTC https://doi.org/10.1021/acs.jctc.7b00756)
    GCS           : Gaussian Convolution Surface for RPBE. (Wang et al. 2021, MBE https://doi.org/10.3934/mbe.2021072)
default:     SurfaceMethod.VDW
description: Method for defining solute and solvent regions:
             choices {"GCS","GAUSSIAN","VDW"} (default: GAUSSIAN)
```

This command enumerates valid parameter names & aliases, data types, default values, and key references—serving as an in-terminal manual for all pyDelPhi inputs.

---

### 🔹 Python Interface (For Programmatic Use)

While end users typically run `pyDelPhi` via the command-line interface (`pydelphi-static`),  
developers and advanced users can invoke the solver directly through the **DelphiApp** API.

This mirrors the same flow as the CLI driver — parsing parameters, configuring runtime,  
and executing the solver under a specified platform and precision context.

**Example:**
```python
from pydelphi.app.delphi import DelphiApp
from pydelphi.foundation.platforms import Platform
from pydelphi.foundation.enums import Precision

# --- Configure platform and precision ---
platform = Platform("cuda", debug=False)
platform.activate("cuda", threads=64, device_id=0)
platform.set_precision(Precision.DOUBLE)

# --- Initialize and run DelphiApp ---
app = DelphiApp(param_file="examples/1CRN_params.inp", platform=platform)
energies = app.run(outfile="outputs.csv", label="1CRN", overwrite=True)

print(f"Reaction Field Energy (kT): {energies['E_rxn_kT_tot']:.6f}")
```

This entry point allows pyDelPhi to be embedded in Python workflows —  
for instance, within MD post-processing pipelines, automated electrostatics scans,  
or large-scale benchmarking scripts — while maintaining full compatibility  
with CLI-based parameterization.

---

## 📊 Benchmark Summary

| Dataset | Description | Grid Range | Validation Target | Speedup |
|----------|--------------|-------------|------------------|----------|
| pm74 | Monomeric proteins | 129³–321³ | RMSD < 1×10⁻⁵ | 10–64× CPU |
| pd66 | Protein–DNA complexes | 161³–401³ | ΔE_rxn < 0.1 % | 25–80× CPU |
| pp46 | Protein–protein complexes | 257³–513³ | Identical energies | 40–100× GPU |
| capsid | Viral capsid (> 5 M atoms) | up to 1029³ | Scaling test | > 100× GPU |

All benchmarks were performed under identical parameters to DelPhi 8.5; results are available in `tsv-reports/` and `plots/`.

---

## 📁 Repository Layout

```
pydelphi/
 ├── app/            # High-level API (DelPhiApp entry point)
 ├── config/         # Global runtime configuration and logging
 ├── constants/      # Physical constants, elements, and residue data
 ├── data/           # Reference datasets and test examples (1he8, 5tif, sphere, etc.)
 ├── energy/         # Energy term calculators (Coulombic, Reaction Field, Nonpolar)
 ├── foundation/     # Core enums, context management, and platform abstractions
 ├── scripts/        # CLI tools (pydelphi-static, pydelphi-help)
 ├── site/           # Site generation and file writing utilities
 ├── solver/         # PB solvers (linear, nonlinear, SOR, NWT, RPBE)
 ├── space/          # Dielectric and grid-space generation (VDW, SAS, Gaussian)
 ├── tests/          # Regression and unit test suite
 └── utils/          # Supporting utilities (I/O, precision, CUDA helpers)

Ancillary files:
 ├── LICENSE          # GNU AGPLv3 license
 ├── pyproject.toml   # Build and dependency metadata
 ├── PKG-INFO         # Distribution metadata (auto-generated)
 └── README.md        # Documentation file (this document)
```

---

## 🧩 For Developers

### 🔹 Design Overview

pyDelPhi follows a modular, layered architecture designed for both clarity and extensibility:

```
CLI Entry (scripts/)
   ↓
Runtime Configuration (config/, foundation/)
   ↓
Compute Backend (Platform, Precision, Verbosity)
   ↓
Application Layer (app/DelphiApp)
   ↓
Numerical Solvers (solver/, space/, energy/)
```

This structure enables reproducible command-line execution while allowing developers to extend components independently — for example, adding new solvers, backends, or dielectric models without modifying user-facing tools.

---

### 🔹 CLI Execution Flow (`pydelphi-static`)

The main entry point for static PB calculations is defined in  
`pydelphi/scripts/pydelphi_static.py`.

**Simplified execution flow:**

```python
def main():
    # 1. Parse command-line arguments
    args = parse_arguments()

    # 2. Handle version or input validation
    if args.version:
        print_pydelphi_version_info(); exit(1)
    if not args.param_file:
        print("Error: Parameter file required."); exit(1)

    # 3. Configure output and runtime
    check_output_file(args.outfile, args.overwrite)
    platform = Platform(args.platform, debug=False)
    platform.activate(args.platform, args.threads, args.device_id)
    platform.set_precision(Precision[args.precision.upper()])

    set_precision(platform.precision)
    set_verbosity_level(str_to_verbosity(args.verbosity))

    # 4. Initialize and run application
    from pydelphi.app.delphi import DelphiApp
    app = DelphiApp(args.param_file, platform, user_inputs=None)
    energies = app.run(args.outfile, args.label, args.overwrite)
```

**Key principles:**
- **Single entry point:** CLI handles argument parsing and safety checks only.  
- **Explicit configuration:** Platform, precision, and verbosity are globally set before solver execution.  
- **Encapsulation:** All computational logic resides inside `DelphiApp`.  
- **Reproducibility:** Every calculation is parameter-driven and version-logged.

---

### 🔹 Extensibility Points

| Area | Module Path | Description |
|------|--------------|-------------|
| **New solvers** | `pydelphi/solver/` | Add new iterative schemes or nonlinear models. |
| **Surface / dielectric models** | `pydelphi/space/core/` | Implement Gaussian or hybrid boundary schemes. |
| **Energy components** | `pydelphi/energy/` | Add analytical or empirical energy terms. |
| **Platform abstraction** | `pydelphi/foundation/platforms.py` | Extend to new backends or accelerators. |
| **Configuration system** | `pydelphi/utils/io/inproc_helpers/param_definitions/` | Define new `.prm` keywords with aliases and validation. |

Each component is self-contained and unit-tested, ensuring that scientific accuracy and reproducibility are preserved during extension.

---

### 🔹 Recommended Development Workflow

1. **Run validation suite locally:**
   ```bash
   pydelphi-test --no-cuda
   ```
   (use `--no-double` or `--no-single` to isolate precision tests)

2. **Profile a new feature:**
   ```bash
   python -m pydelphi.app.delphi my_params.prm
   ```

3. **Inspect runtime configuration:**
   ```bash
   pydelphi-help -n <param_name>
   ```

4. **Benchmark changes:**  
   Compare solver time and energy RMSD against DelPhi reference TSVs in  
   `pydelphi/data/test_examples/`.

---

### 🔹 Guiding Philosophy

> **“Accuracy is physics; performance is engineering.”**  
>  
> pyDelPhi separates physical formulation (solver & dielectric model)  
> from computational engineering (platform, precision, threading) —  
> ensuring transparent, scientifically rigorous, and performance-portable implementations.

---

## 🧩 Citation

If you use pyDelPhi in your work, please cite:

> **Pandey, S. et al. (2025)**  
> *pyDelPhi: A Python-based, High-Performance Poisson–Boltzmann Solver for Biomolecular Electrostatics*  
> *(manuscript in preparation)*

---

## 🤝 Contributing

Contributions are welcome — see `CONTRIBUTING.md` for style, testing, and PR guidelines.  
Bug reports and enhancement requests can be submitted via GitHub Issues.

---

## 🧾 License

**pyDelPhi** is released under the  
[GNU Affero General Public License v3 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0).

> This file is part of **pyDelPhi**.  
> pyDelPhi is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.  
> pyDelPhi is distributed in the hope that it will be useful, but **without any warranty**; without even the implied warranty of **merchantability or fitness for a particular purpose**.  
> See the full license text in [`LICENSE`](LICENSE) or visit <https://www.gnu.org/licenses/>.

---

## 🧭 Acknowledgments

- **DelPhi (C++)** developers for foundational algorithms  
- **Numba** and **CUDA Python** communities for enabling hybrid acceleration  
- Computational resources provided by the **Clemson Palmetto HPC Cluster**

---

> _“Accuracy is physics; performance is engineering — pyDelPhi unites both.”_
