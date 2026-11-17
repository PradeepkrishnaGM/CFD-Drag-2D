# CFD-Drag-2D: Aerodynamic Dataset Generator & Flow Studio

![C++](https://img.shields.io/badge/C++-17-blue) ![Python](https://img.shields.io/badge/Python-3.8+-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview
This repository contains a high-performance 2D Computational Fluid Dynamics (CFD) solver based on the D2Q9 Lattice Boltzmann Method (LBM), implemented from scratch in C++ (a CUDA port is planned). 

Designed as a complete aerodynamic "data factory," the project features a custom interactive GUI for real-time flow visualization and a Python automation pipeline. It is purpose-built to rapidly generate large-scale, geometry-aware fluid flow datasets and compute precise aerodynamic metrics (Drag Coefficient, Pressure Loss) for machine learning and surrogate model training.

## Features
* **C++ Compute Engine:** A dependency-free D2Q9 LBM solver driven from the command line (CUDA port planned).
* **Interactive GUI Studio:** A responsive, dark-themed PyQt6 desktop application to manually design obstacles, tune physics parameters, and visualize flow fields instantly.
* **Custom Geometry Handling:** Simulates fluid interaction around circles, rectangles, and triangles with dynamic sizes and variable Angles of Attack (AoA).
* **Native Physics Calculation:** Accurately calculates the Drag Coefficient ($C_d$) via the Momentum Exchange Method directly inside the C++ simulation loop.
* **Batch Automation:** Features a robust Python wrapper (`auto.py`) to procedurally generate balanced datasets of thousands of `(Image, Metric)` samples with built-in numerical stability checks.

## Solver
* D2Q9 lattice Boltzmann with TRT collision, half-way bounce-back walls, Zou-He velocity inlet
  (uniform or parabolic) and pressure outlet. The inlet ramps up smoothly from rest.
* Inputs: shape, sizes, angle and **Reynolds number** `Re = U·D/ν`, where `D` is the projected
  frontal height of the rotated obstacle and `U` the mean inlet velocity (0.05 lattice units).
* Outputs (`results.csv` in `--out-dir`): time-averaged **Cd**, **Cl**, lift fluctuation `Cl_rms`,
  pressure loss, a steady/periodic flag, and optionally the mask and flow fields as `.npy`.
* Validated in `tests/`: determinism, mass conservation, channel (Poiseuille) flow, and the
  Schäfer–Turek 2D-1 cylinder benchmark (Cd = 5.58 at Re = 20).

## Status
Work in progress. Planned next: a CUDA solver,
large-scale dataset generation, and a CNN surrogate that predicts drag from geometry.
See `SPEC.md` for the full plan and `DECISIONS.md` for design choices.

## Prerequisites
* **C++ Compiler:** GCC/G++ with C++17 support, and `make`.
* **Python:** 3.10+ (3.12 recommended). Dependencies are listed in `requirements.txt`.

## Quick Start
All commands run from the repository root.

```bash
# 1. Build the solver (-> build/solver)
make solver

# 2. Python environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -e .

# 3. Interactive Flow Studio (GUI)
python gui/app.py

# 4. CLI dataset generation (writes to ./dataset by default)
python scripts/auto.py --samples 100 --shapes circle rect tri --out dataset

# 5. Tests (quick; use `make test-all` to include slow validation tests)
make test
```

All tunable defaults (domain size, steps, sampling ranges) live in `cfd_drag/params.py`.

## Output Format
Whether running via the GUI's batch mode or the CLI automation script, the pipeline generates a dataset directory containing:
1. `training_data.csv`: A master record tracking the geometric inputs (shape, size1, size2, angle, velocity) and the corresponding aerodynamic targets ($C_d$ and $\Delta P$).
2. `images/`: A folder containing the .png images of the steady-state velocity fields for each sample, acting as the visual input for downstream CNN/Computer Vision tasks.
