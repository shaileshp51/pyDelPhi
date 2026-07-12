# Changelog

All notable changes to **pyDelPhi** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and adheres to semantic versioning with date-stamped dev builds:
`vMAJOR.MINOR.PATCH`.

# 🧾 **pyDelPhi v0.3.0 — Trajectory Electrostatics, Modular I/O, and Expanded Regression Coverage**

**Release date:** Unreleased

This release expands pyDelPhi from single-structure calculations to trajectory-based Poisson–Boltzmann workflows, while refactoring core application, input, geometry, surface, and output components for improved modularity and maintainability.

The release also introduces dedicated trajectory regression testing, packaged force-field charge/size resources, Conda-managed NetCDF support, and updated installation and documentation workflows.

---

## 🎞️ Trajectory Mode

- Added trajectory-mode Poisson–Boltzmann calculations through:
  
  ```bash
  pydelphi-trajectory
  ```
- Added lightweight topology readers for:
  - PDB
  - PQR
  - PSF
  - PRMTOP
- Added lightweight trajectory readers for:
  - DCD
  - TRR
  - NetCDF
- Added topology/trajectory ensemble handling with frame-wise coordinate updates.
- Added trajectory-specific parameter processing and validation.
- Added packaged trajectory examples, test inputs, and reference results.

---

## 🧬 Charge and Size Parameter Support

- Added packaged Amber and CHARMM charge/size parameter sets under:
  
  ```text
  pydelphi/data/crgsiz/
  ```
- Expanded input processing for charge/size acquisition and assignment.
- Improved topology-aware handling of structures that require external charge and size data.
- Added supporting examples and regression cases for custom and packaged charge/size inputs.

---

## 🧱 Application and Architecture Refactoring

- Refactored the main application workflow into reusable components for:
  - atom materialization
  - CUDA runtime handling
  - focusing preparation
  - frc-output writing
  - output-map handling
  - PB and RPBE execution
  - runtime policy
  - space construction
  - run summaries
- Added a dedicated trajectory application path.
- Expanded foundation models, enums, and runtime context handling.
- Improved separation between input parsing, application control, solver execution, and output generation.

---

## 🧮 Solver, Surface, and Energy Updates

- Revised PB, nonlinear PB, RPBE, and shared SOR execution paths.
- Updated molecular-surface, dielectric, voxelization, and boundary-generation components.
- Added and reorganized CPU, parallel, and CUDA boundary-processing implementations.
- Expanded geometry and nonpolar calculation components.
- Updated reaction-field, Coulombic, Lennard-Jones, and nonpolar energy handling.
- Improved focusing and force-output workflows.

---

## 📥 Input, Output, Selection, and Help-System Updates

- Expanded parameter definitions for:
  - topology and trajectory input
  - charge/size configuration
  - calculation settings
  - solvent settings
  - output controls
- Added a molecular selection language for selecting atoms and structural subsets.
- Added selection support for FRC force-output workflows.
- Added improved `acenter` handling for atom-centered grid and calculation setup.
- Updated PDB/PQR and custom-format readers and writers.
- Added force-file and trajectory logging utilities.
- Updated the help command with:
  - parameter groups
  - named function-style topics
  - `--list-param-names`
  - `--list-groups`
- Added help-topic conventions such as:
  
  ```text
  name
  function
  function__namedattr
  ```

---

## 🧪 Regression Testing and Verification

- Added static regression coverage through:
  
  ```bash
  pydelphi-test
  ```
- Added dedicated trajectory regression coverage through:
  
  ```bash
  pydelphi-test-traj
  ```
- Renamed packaged regression data to:
  
  ```text
  pydelphi/data/test_cases/
  ```
- Added trajectory regression data under:
  
  ```text
  pydelphi/data/test_traj/
  ```
- Added source-distribution and wheel build verification.
- Added practical installation verification using:
  
  ```bash
  pydelphi-test --no-cuda --no-single --no-parallel
  pydelphi-test-traj --no-cuda --no-single --no-parallel
  ```
  
  These reduced configurations are intended to verify environment build and basic usability within a reasonable runtime. Broader validation can be performed later without the skip flags.
- Added checks that CPU mode remains usable when CUDA is absent, disabled, or misconfigured.

---

## 📦 Packaging and Environment Updates

- Updated supported runtime dependencies to:
  - Python 3.13
  - NumPy 2.3.x
  - Numba 0.62.x
- Kept CUDA and NetCDF outside the mandatory core dependency set.
- Added:
  
  ```text
  environment.yml
  environment-traj.yml
  ```
- Added Conda-managed trajectory support with the tested:
  
  ```text
  netCDF4 1.7.3
  ```
- Ensured CPU-only installations do not require a working CUDA device or driver.
- Updated source-distribution contents and excluded backup and temporary files from release artifacts.

---

## 📚 Documentation and Publication

- Updated installation, trajectory-mode, regression-suite, help-system, and repository documentation. The public repository URL is:
  
  ```text
  https://github.com/delphi001/pyDelPhi
  ```

- Updated the project citation to:
  
  **Panday, S. K.; Zhao, S.; Alexov, E.**  
  *Accurate and Scalable Continuum Electrostatics for Large Biomolecular Systems: The pyDelPhi Poisson–Boltzmann Framework.*  
  **J. Chem. Inf. Model.** 2026, **66** (1), 488–502.  
  DOI: `10.1021/acs.jcim.5c02818`

- Clarified that packaged test cases support regression validation; performance benchmarking is reported in the publication.

---

## 🏁 Summary

**v0.3.0** extends pyDelPhi into trajectory-based continuum-electrostatics workflows while preserving static CPU execution as the base installation path.

The release combines trajectory I/O, packaged force-field resources, modularized application components, expanded regression coverage, and clearer environment management to support reproducible static and trajectory calculations across CPU and CUDA-capable systems.

---

### 📦 Banner identifier

```text
PyDelPhi-0.3.0 — Trajectory Electrostatics, Modular I/O, and Expanded Regression Coverage
```

---

### 🔹 Release-Summary

**pyDelPhi v0.3.0 — Trajectory Electrostatics, Modular I/O, and Expanded Regression Coverage**

- Added trajectory-mode PB calculations with PDB, PQR, PSF, PRMTOP, DCD, TRR, and NetCDF support.
- Added packaged Amber and CHARMM charge/size parameter sets.
- Added molecular selection language support, including selections in FRC workflows.
- Improved `acenter` handling.
- Refactored the main application into reusable execution and I/O components.
- Added dedicated static and trajectory regression suites.
- Added Conda-managed trajectory/NetCDF installation support.
- Preserved CPU usability when CUDA is absent or unavailable.
- Updated documentation and the published pyDelPhi citation.

---

# 🧾 **pyDelPhi v0.2.0 — Optimized Iteration Core with Adaptive Convergence Control**

**Release date:** 2025-11-06

This release introduces a major internal upgrade to the solver core with fused-kernel iteration, adaptive convergence control, improved cross-platform consistency, and expanded parallelism in surface and dielectric computations.  
It remains fully backward-compatible with v0.1.x parameter files and APIs.

---

## 🔧 Solver Core and Iteration Control

- **Standardized relaxation factor:**  
  The solver now explicitly reports `omega_SOR`, matching the conventional Successive Over-Relaxation (SOR) coefficient.  
  Earlier versions printed `1 − ω` under this name; the new notation aligns with standard numerical-analysis conventions.

- **Fused iteration kernel for RMSD / ΔΦ computation:**  
  RMSD and maximum potential change (ΔΦ) are now computed within the same update kernel.  
  
  - Eliminates reconstruction of `phimap_half_even` and `phimap_half_odd` each block.  
  - Avoids per-block host ↔ device transfers.  
  - Reduces iteration-loop overhead on both CPU and GPU backends.

- **Precision-adaptive stagnation detection:**  
  The controller now tracks *consecutive sign-flip streaks* of ΔRMSD to detect oscillatory plateaus caused by floating-point hysteresis.  
  
  - Terminates automatically once oscillation confidence exceeds a threshold.  
  - Prevents prolonged or iteration-bounded micro-oscillations once RMSD reaches the precision limit.  
  - Differentiates clearly between:  
    - **True convergence** (RMSD / ΔΦ below tolerance)  
    - **Stagnation plateau** (precision-limited hysteresis)

- **Enhanced numerical safety:**  
  
  - Explicit divergence detection on non-finite residuals (NaN / Inf).  
  - Deterministic fallback termination for all solver paths.

---

## ⚡ Performance and Parallelism

- **Partial parallelization of VDW surface generation:**  
  Surface-voxel traversal and marking routines now use `prange` parallel loops, improving scalability for medium-to-large molecular systems without altering surface topology or electrostatic accuracy.

- **Voxel-based neighbourhood search in RPB (Regularized Poisson–Boltzmann) dielectric gradients:**  
  The Regularized Poisson–Boltzmann (RPB) formalism now employs a voxel-based spatial partitioning scheme to accelerate dielectric-gradient evaluation.  
  
  - Restricts atomic-neighbour searches to local voxels instead of the full domain.  
  - Reduces distance checks from O(N²) to near-linear complexity.  
  - Preserves dielectric-gradient fidelity within machine precision, consistent with the Gaussian-regularized framework.

---

## 🧠 Numerical Fidelity and Stability

- **Precision-limit handling:**  
  Single-precision solvers terminate predictably at the hysteresis plateau, avoiding wasted iterations while maintaining consistent electrostatic energy.

- **Improved reporting:**  
  Log output distinguishes convergence modes:  
  
  - `Convergence reached (RMSD/ΔΦ thresholds satisfied)` — strict convergence  
  - `Convergence reached (stagnation plateau, relaxed criterion)` — precision-limited stop  

---

## 🧰 Developer and Infrastructure Updates

- Unified solver-control functions:  
  - `_iteration_control_check` adds consecutive flip-streak logic and divergence detection.  
  - `_calculate_phi_map_sample_rmsd` fuses RMSD reduction with consistent dtype casting and Numba caching.
- Relaxation factor (`_calc_relaxation_factor`) now computed once host-side and reused across CPU / CUDA phases.
- Verified compatibility with:  
  - Python 3.12  
  - Numba 0.61.2 (< 0.62)  
  - NumPy 2.2.x (< 2.3)  
  - CUDA 11.8 – 12.x (SM ≥ 7.0)

---

## 🧮 Performance Summary

| Platform | Precision | Primary Improvement Source                       |
|:-------- |:--------- |:------------------------------------------------ |
| CPU      | Double    | Fused RMSD kernel, reduced transfer overhead     |
| CPU      | Single    | Improved cache reuse, reduced iteration overhead |
| CUDA     | Double    | Unified kernel, more consistent memory access    |
| CUDA     | Single    | Early hysteresis detection and lower sync cost   |

---

## 🏁 Summary

**v0.2.0** represents a major optimization milestone for pyDelPhi’s solver architecture.  
The transition from transfer-bound block iteration to a fused, adaptive, and precision-aware iteration core delivers measurable stability and computational-efficiency gains across supported platforms, without altering the validated physics.

---

### 📦 Recommended splash / banner identifier

```
PyDelPhi-0.2.0  —  Optimized Iteration Core with Adaptive Convergence Control
```

---

## 📘 Licensing

Released under the **GNU Affero General Public License v3 (or later)**.  
© 2025 The pyDelPhi Project and contributors.

---

### 🔹 Release-Summary (for GitHub)

**pyDelPhi v0.2.0 — Optimized Iteration Core with Adaptive Convergence Control**  

- Fused iteration kernel eliminates redundant host/device transfers.  
- New RMSD-flip stagnation detector prevents precision-bound oscillations.  
- Partial parallelization of VDW surface generation.  
- Voxel-based acceleration for Regularized Poisson–Boltzmann (RPB) dielectric gradients.  
- Improved cross-platform numerical consistency and solver reporting.
