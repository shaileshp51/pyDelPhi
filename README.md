# 🧩 pyDelPhi: A Modern, High-Performance Poisson–Boltzmann Solver

[![License: AGPL v3+](https://img.shields.io/badge/License-AGPL_v3%2B-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.12-green.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.0%2B-yellow.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Numba](https://img.shields.io/badge/Accelerated%20by-Numba-lightgrey.svg)](https://numba.pydata.org/)

---

**pyDelPhi** is a high-performance, Python/Numba/CUDA reimplementation of the classic **DelPhi** Poisson–Boltzmann (PB) electrostatics solver. It enables accurate, reproducible computation of **electrostatic potentials** and **polar solvation free energies** for biomolecules, with efficient execution on both **CPU** and **GPU** platforms.

---

## ✨ Key Features

- **Faithful Reimplementation of DelPhi**
  - Numerically consistent with DelPhi 8.5 reference outputs.  
  - Validated across proteins, protein–DNA complexes, and viral capsids.

- **High-Performance Backends**
  - CPU parallelism via `Numba` and `prange`.  
  - GPU acceleration using custom CUDA kernels.

- **Flexible Models**
  - Linear / nonlinear PB solvers.  
  - Traditional two-dielectric and Gaussian dielectric formulations.  
  - Cubic and cuboidal grid geometries with automatic margin control.

- **Precision & Solvers**
  - Single and double precision arithmetic.  
  - Successive-Over-Relaxation (SOR) and Newton-like (NWT) iterative schemes.

- **Reproducible & Extensible**
  - Modular architecture separating physics, numerics, and platform logic.

---

## ⚙️ Installation

### Requirements
- **Python ≥ 3.12**  
- **NumPy ≥ 1.26**, **Numba ≥ 0.61**  
- *(Optional)* CUDA Toolkit ≥ 12.0 for GPU acceleration  

### From Source
```bash
git clone https://github.com/shaileshp51/pyDelPhi.git
cd pyDelPhi
pip install -e .
```

Verify installation:
```bash
pydelphi-static --version
```

---

## 🚀 Quick Start

### 🔹 Command-Line Interface

pyDelPhi installs three primary executables:

| Command | Purpose |
|----------|----------|
| `pydelphi-static` | Run a Poisson–Boltzmann electrostatics calculation |
| `pydelphi-test` | Execute regression and consistency tests |
| `pydelphi-help` | Display built-in parameter documentation |

---

#### 🧮 `pydelphi-static` — Main Solver

Run a full PB calculation:
```bash
pydelphi-static -f examples/1CRN_params.inp -P cuda -p double -t 64 -l 1CRN
```

| Flag | Description | Default |
|------|--------------|----------|
| `-P`, `--platform {cpu,cuda}` | Execution platform | `cpu` |
| `-p`, `--precision {single,double}` | Numeric precision | `double` |
| `-t`, `--threads` | Number of CPU threads | `1` |
| `-f`, `--param-file` | Input parameter file | *required* |
| `-o`, `--outfile` | Output CSV filename | `outputs.csv` |
| `-O`, `--overwrite` | Overwrite existing output | `False` |

---

#### 🧪 `pydelphi-test` — Validation Suite

Run automated regression tests to verify numerical consistency:

```bash
pydelphi-test --no-cuda
```

Optional flags allow skipping specific backends or precisions  
(e.g., `--no-parallel`, `--no-single`, `--no-double`).

---

#### 📘 `pydelphi-help` — Parameter Reference

Interactive in-terminal help for parameters, groups, and aliases:

```bash
pydelphi-help -g grid
pydelphi-help -n surfmethod
```

Displays definitions, units, defaults, and key references.

---

## 🔹 Python API (Advanced Use)

Developers can embed pyDelPhi in Python workflows via the `DelphiApp` interface.

```python
from pydelphi.app.delphi import DelphiApp
from pydelphi.foundation.platforms import Platform
from pydelphi.foundation.enums import Precision

platform = Platform("cuda", debug=False)
platform.activate("cuda", threads=64, device_id=0)
platform.set_precision(Precision.DOUBLE)

app = DelphiApp(param_file="examples/1CRN_params.inp", platform=platform)
energies = app.run(outfile="outputs.csv", label="1CRN", overwrite=True)

print(f"Reaction Field Energy (kT): {energies['E_rxn_kT_tot']:.6f}")
```

---

## 📊 Benchmark Summary

| Dataset | Type | Grid Range | Validation Target | Speedup |
|----------|------|-------------|------------------|----------|
| pm74 | Monomeric proteins | 129³–321³ | RMSD < 1×10⁻⁵ | 10–64× CPU |
| pd66 | Protein–DNA complexes | 161³–401³ | ΔE_rxn < 0.1 % | 25–80× CPU |
| pp46 | Protein–protein complexes | 257³–513³ | Identical energies | 40–100× GPU |
| capsid | Viral capsid (> 5 M atoms) | up to 1029³ | Scaling test | > 100× GPU |

All tests were performed under identical parameters to DelPhi 8.5.  
Benchmark reports and plots are available in `tsv-reports/` and `plots/`.

---

## 📁 Repository Layout

```
pydelphi/
 ├── app/          # High-level application layer (DelphiApp)
 ├── config/       # Runtime and logging configuration
 ├── constants/    # Physical constants, elements, residues
 ├── data/         # Reference examples and test cases
 ├── energy/       # Energy component calculators
 ├── foundation/   # Enums, context, platform abstractions
 ├── scripts/      # CLI tools (pydelphi-static, pydelphi-help)
 ├── solver/       # PB solvers (linear, nonlinear, SOR, NWT)
 ├── space/        # Grid and dielectric surface models
 ├── tests/        # Unit / regression tests
 └── utils/        # I/O, CUDA, precision helpers
```

---

## 🧩 Citation

If you use **pyDelPhi** in your work, please cite:

> **Pandey, S. K.** (2025)  
> *pyDelPhi: A Python-based, High-Performance Poisson–Boltzmann Solver for Biomolecular Electrostatics*  
> *(manuscript in preparation)*

---

## 🤝 Contributing

Formal contribution guidelines will be published in a future release.  
For now, please report issues or enhancement suggestions through the  
[GitHub issue tracker](https://github.com/shaileshp51/pyDelPhi/issues).

All contributions are licensed under the **AGPL-3.0-or-later**.

---

## 🧾 License

**pyDelPhi** is distributed under the  
[GNU Affero General Public License v3 or later (AGPL-3.0-or-later)](https://www.gnu.org/licenses/agpl-3.0).

> © 2025 The pyDelPhi Project and contributors.  
> pyDelPhi is provided *as is*, without any warranty of merchantability or fitness for a particular purpose.  
> See the [LICENSE](LICENSE) file for full details.

---

## 🧭 Acknowledgments

- **DelPhi (C++)** developers for foundational algorithms  
- **Numba** and **CUDA Python** communities for enabling hybrid acceleration  
- Computational resources provided by the **Clemson Palmetto HPC Cluster**

---

> _“Accuracy is physics; performance is engineering — pyDelPhi unites both.”_
