# 🧩 pyDelPhi: A Modern, High-Performance Poisson–Boltzmann Solver

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.13-green.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-yellow.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Numba](https://img.shields.io/badge/Accelerated%20by-Numba-lightgrey.svg)](https://numba.pydata.org/)

---

**pyDelPhi** is a high-performance, Python-based reimplementation and extension of the classic **DelPhi** electrostatics solver. It provides accurate and efficient solutions to the **Poisson–Boltzmann (PB)** equation for biomolecular systems, with both **CPU** and **GPU (CUDA)** acceleration.

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
- Python >= 3.13,<3.14
- NumPy >= 2.3.5,<2.4
- Numba >= 0.62.1,<0.63
- CUDA Toolkit >= 12.0 and an NVIDIA GPU/driver (optional for GPU backend)

NetCDF trajectory support uses `netCDF4`. To avoid native-library and ABI conflicts, both `netCDF4` and the optional `numba-cuda` backend are managed through Conda in the supplied environment files.

Optional (profiling / plots):
```bash
scipy
pandas
matplotlib
seaborn
psutil
```

## 🧩 Recommended Environment Setup

A dedicated Conda or Miniconda environment is **strongly recommended**. The source distribution includes environment files for CPU/NetCDF and CUDA installations.

### CPU and NetCDF
```bash
conda env create -f environment.yml
conda activate pydelphi
python -m pip install . --no-deps
```

### CUDA and NetCDF
```bash
conda env create -f environment-cuda.yml
conda activate pydelphi-cuda
python -m pip install . --no-deps
```

The CUDA environment installs `numba-cuda` through Conda. A compatible NVIDIA driver is still required on the host system.

### From a Git Checkout
```bash
git clone https://github.com/delphi001/pyDelPhi.git
cd pyDelPhi
conda env create -f environment.yml
conda activate pydelphi
python -m pip install -e . --no-deps
```

Use editable installation (`-e`) only for development. For a released source distribution, use the normal non-editable installation shown above.

Verify installation:
```bash
pydelphi-static --version
```

---

## 🚀 Quick Start

### 🔹 Command-Line Usage (Recommended for End Users)

pyDelPhi provides five primary executables:

| Command | Purpose |
|----------|----------|
| `pydelphi-static` | Run a single Poisson–Boltzmann (PB) electrostatics calculation |
| `pydelphi-trajectory` | Run PB calculations over a molecular trajectory |
| `pydelphi-test` | Execute static-mode regression and consistency tests |
| `pydelphi-test-traj` | Execute trajectory-mode regression tests |
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
pydelphi-static -f examples/5tif/param_5tif_linear_trad.prm -P cpu -p double -t 4 -l 5TIF -O
```

---

#### 🎞️ `pydelphi-trajectory` — Trajectory PB Calculations

Run PB calculations over frames from a molecular trajectory:
```bash
pydelphi-trajectory -f trajectory_params.inp -P cpu -p double -t 4 -O
```

**Usage**
```
usage: pydelphi-trajectory.py [-h] [-V] [-P {cpu,cuda}] [-p {single,double}]
                              [-t THREADS] [-d DEVICE_ID] [-f PARAM_FILE]
                              [-v {critical,error,notice,warning,info,verbose,debug,trace}]
                              [-l LABEL] [-o OUTFILE] [-O] [-S]
```

The command uses the same platform, precision, threading, output, and logging options as `pydelphi-static`. The parameter file additionally defines the trajectory input and its associated topology. Use the built-in help to inspect the relevant parameter definitions and aliases:

```bash
pydelphi-trajectory --help
pydelphi-help --list-param-names
pydelphi-help -g infile
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

#### 🧪 `pydelphi-test-traj` — Trajectory Regression Suite

Run the dedicated trajectory regression tests:

```bash
pydelphi-test-traj --help
```

**Usage**
```
usage: pydelphi-test-traj [-h] [--no-cuda] [--no-parallel] [--no-single]
                           [--no-double] [--timeout TIMEOUT] [--verbose]
                           [--debug-files]
```

| Flag | Description |
|------|-------------|
| `-h`, `--help` | Show help and exit |
| `--no-cuda` | Skip CUDA configurations |
| `--no-parallel` | Skip configurations using more than one thread |
| `--no-single` | Skip single-precision configurations |
| `--no-double` | Skip double-precision configurations |
| `--timeout TIMEOUT` | Set the per-run timeout in seconds |
| `--verbose` | Print additional progress information |
| `--debug-files` | Preserve generated parameter files and computed TSV outputs under `pydelphi_traj_debug_files/` and record their full paths in the report |

Example:
```bash
pydelphi-test-traj --no-cuda --timeout 300
```

By default, temporary trajectory-test files are deleted and only their filenames are recorded. Use `--debug-files` when diagnosing failed cases or inspecting generated inputs and outputs.

---

#### 📘 `pydelphi-help` — Built-in Parameter Documentation

Interactive access to parameter definitions, function-style input constructs, defaults, and references.

```bash
pydelphi-help -h
```

**Usage**
```
usage: pydelphi-help [-h] [-g GROUP] [-n PARAM_NAME] [-ln] [-lg]
```

| Flag | Description |
|------|-------------|
| `-h`, `--help` | Show this message and exit |
| `-g`, `--group GROUP` | Print help for parameters in a group |
| `-n`, `--param-name PARAM_NAME` | Print help for a parameter or function-style help topic |
| `-ln`, `--list-param-names` | List valid parameter and function help topics |
| `-lg`, `--list-groups` | List valid parameter groups |

Help topics use the following convention:

- `name` for a statement-style parameter, such as `grid_size`.
- `function` for a selector-free function, such as `zeta` for `zeta(...)`.
- `function__namedattr` for a function-style construct, such as `in__crgsiz` for `in(crgsiz, ...)`.

Examples:
```bash
pydelphi-help -n grid_size
pydelphi-help -n in__crgsiz
pydelphi-help -g infile
pydelphi-help --list-param-names
pydelphi-help --list-groups
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

Comprehensive benchmarking of **pyDelPhi** — including accuracy validation,  
runtime scaling, and memory efficiency across datasets **pm74**, **pp46**, **pd66**,  
and viral capsid systems — is detailed in the accompanying publication:

> **Panday, S. K.; Zhao, S.; Alexov, E.**  
> *Accurate and Scalable Continuum Electrostatics for Large Biomolecular Systems: The pyDelPhi Poisson–Boltzmann Framework.*  
> **J. Chem. Inf. Model.** 2026, **66** (1), 488–502.  
> DOI: [10.1021/acs.jcim.5c02818](https://doi.org/10.1021/acs.jcim.5c02818)

The benchmarks compare pyDelPhi against the original **DelPhi (C++)** implementation,  
demonstrating numerical equivalence and substantial acceleration on both CPU and GPU platforms.

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
 ├── scripts/        # CLI tools (static, trajectory, help, and tests)
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
        print_pydelphi_version_info();
        exit(1)
    if not args.param_file:
        print("Error: Parameter file required.");
        exit(1)

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
   pydelphi-test-traj --no-cuda
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

4. **Validate numerical changes:**  
   Run the regression suites against the reference data in  
   `pydelphi/data/test_cases/`. These files support correctness and reproducibility checks; they are not the published performance-benchmark dataset. For runtime, scaling, and large-system benchmark results, see the pyDelPhi publication cited below.

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

> **Panday, S. K.; Zhao, S.; Alexov, E.**  
> *Accurate and Scalable Continuum Electrostatics for Large Biomolecular Systems: The pyDelPhi Poisson–Boltzmann Framework.*  
> **J. Chem. Inf. Model.** 2026, **66** (1), 488–502.  
> DOI: [10.1021/acs.jcim.5c02818](https://doi.org/10.1021/acs.jcim.5c02818)

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
