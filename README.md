# CFD-Drag-2D: Aerodynamic Dataset Generator & Flow Studio

![C++](https://img.shields.io/badge/C++-17-blue) ![CUDA](https://img.shields.io/badge/CUDA-batch%20solver-green) ![Python](https://img.shields.io/badge/Python-3.8+-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview
This repository contains a high-performance 2D Computational Fluid Dynamics (CFD) solver based on the D2Q9 Lattice Boltzmann Method (LBM), implemented from scratch in C++ and CUDA. 

Designed as a complete aerodynamic "data factory," the project features a custom interactive GUI for real-time flow visualization and a Python automation pipeline. It is purpose-built to rapidly generate large-scale, geometry-aware fluid flow datasets and compute precise aerodynamic metrics (Drag Coefficient, Pressure Loss) for machine learning and surrogate model training.

## Features
* **CPU and CUDA engines:** Both share one per-node update (`cfd_engine/lbm_common.hpp`). The CUDA solver simulates hundreds of cases per launch and matches the CPU solver to 10 digits.
* **Interactive GUI Studio:** A responsive, dark-themed PyQt6 desktop application to manually design obstacles, tune physics parameters, and visualize flow fields instantly.
* **Custom Geometry Handling:** Simulates fluid interaction around circles, rectangles, and triangles with dynamic sizes and variable Angles of Attack (AoA).
* **Native Physics Calculation:** Accurately calculates the Drag Coefficient ($C_d$) via the Momentum Exchange Method directly inside the C++ simulation loop.
* **Batch Automation:** `scripts/generate.py` samples valid cases from a seed, runs them in parallel, resumes interrupted runs, and stores obstacle masks with their aerodynamic targets.

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
Work in progress. Planned next:
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

# 4. CLI dataset generation (resumable; writes to ./dataset by default)
python scripts/generate.py --n 100 --seed 0 --out dataset --images

# 5. Tests (quick; use `make test-all` to include slow validation tests;
#    GPU tests run only where build/solver_cuda and an NVIDIA GPU exist)
make test

# 6. GPU generation (needs nvcc): build, then use the cuda back-end
make cuda
python scripts/generate.py --n 1000 --seed 0 --backend cuda --out dataset
```

All tunable defaults (domain size, steps, sampling ranges) live in `cfd_drag/params.py`.

## Output Format
A dataset directory contains:
1. `data.csv`: one row per case with the inputs (`shape`, `size1`, `size2`, `angle`, `re`, frontal height `D`, `blockage`) and the targets (`Cd`, `Cl`, `Cl_rms`, `pressure_loss`), plus `status` (steady/periodic) and `ok`.
2. `masks.npy`: `(N, 64, 64)` uint8 obstacle masks cropped around the obstacle, aligned with `data.csv` rows. These are the CNN inputs.
3. `images/`: optional velocity-magnitude PNGs, for visualisation only.

Load it with `cfd_drag.dataset.load("dataset")`. `example_data/` holds 24 cases.

Throughput: about 6.75 s per case on the development machine (i5-10500T, 6 workers) and 1.4 s per case per Tesla T4 on Kaggle. `notebooks/m3_cuda_check.ipynb` verifies the GPU solver on Kaggle.
