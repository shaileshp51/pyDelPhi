# Changelog
All notable changes to **pyDelPhi** are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and adheres to semantic versioning with date-stamped dev builds:
`vMAJOR.MINOR.PATCH`.

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

| Platform | Precision | Primary Improvement Source |
|:--|:--|:--|
| CPU | Double | Fused RMSD kernel, reduced transfer overhead |
| CPU | Single | Improved cache reuse, reduced iteration overhead |
| CUDA | Double | Unified kernel, more consistent memory access |
| CUDA | Single | Early hysteresis detection and lower sync cost |

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
