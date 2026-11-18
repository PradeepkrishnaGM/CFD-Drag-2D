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
Work in progress. Done: validated solver, CPU and CUDA dataset pipeline, a 5000-case dataset,
a CNN surrogate that predicts Cd and Cl from geometry. Planned next: the GUI update.
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

# 7. Check a dataset (writes summary.json; exits 1 if a check fails)
python scripts/check_dataset.py --data dataset

# 8. CNN surrogate (optional [ml] extra; CPU torch shown)
uv pip install torch --index-url https://download.pytorch.org/whl/cpu && uv pip install -e ".[ml]"
python scripts/train.py --data dataset --out runs/v1        # ~15 s/epoch on CPU, ~1 s on a T4
python scripts/evaluate.py --data dataset --run runs/v1     # metrics.json, predictions.csv, parity.png
```

All tunable defaults (domain size, steps, sampling ranges) live in `cfd_drag/params.py`.

## Output Format
A dataset directory contains:
1. `data.csv`: one row per case with the inputs (`shape`, `size1`, `size2`, `angle`, `re`, frontal height `D`, `blockage`) and the targets (`Cd`, `Cl`, `Cl_rms`, `pressure_loss`), plus `status` (steady/periodic) and `ok`.
2. `masks.npy`: `(N, 64, 64)` uint8 obstacle masks cropped around the obstacle, aligned with `data.csv` rows. These are the CNN inputs.
3. `images/`: optional velocity-magnitude PNGs, for visualisation only.

Load it with `cfd_drag.dataset.load("dataset")`. `example_data/` holds 24 cases.

Throughput: about 6.75 s per case on the development machine (i5-10500T, 6 workers) and 1.4 s per case per Tesla T4 on Kaggle. `notebooks/m3_cuda_check.ipynb` verifies the GPU solver on Kaggle.

## Dataset
`notebooks/m4_generate_dataset.ipynb` generates the full dataset on a Kaggle GPU notebook (2x Tesla T4):
it clones the repository at a fixed commit, runs the tests, solves 5000 cases (seed 0) with the CUDA
solver, re-solves 8 random cases on the CPU as a reproducibility check, and runs `check_dataset.py`.

| | |
|---|---|
| Cases | 5000, all solved cleanly (1671 circles, 1645 rectangles, 1684 triangles) |
| Flow regime | 2757 steady, 2243 vortex shedding |
| Re / frontal height D | 20 to 100 (median 54) / 5 to 24.5 cells |
| Cd (min / median / max) | 0.82 / 2.16 / 5.37 |
| GPU vs CPU re-runs | max relative difference 4e-10 |
| Generation time | 59 min on 2x T4 (0.70 s per case) |
| Size | 21 MB (`data.csv`, `masks.npy`, config and check files) |

The dataset is a private Kaggle dataset for now (link to follow when it is made public).

## Surrogate model
A small CNN (4 conv blocks, 0.26 M parameters) predicts Cd and Cl from the 64x64 obstacle mask;
log(Re) joins in the dense head. It is trained on 3500 cases of the dataset above, the best epoch is
chosen on 750 validation cases, and it is scored once on 750 held-out test cases.
`notebooks/m5_train_surrogate.ipynb` trains it on a Kaggle T4 (about 160 s for 150 epochs).

Test set, CNN as mean ± standard deviation over 3 training seeds (same split):

| Model | Cd MAE | Cd R² | Cd mean rel. error | Cd 95th pct rel. error | Cl MAE | Cl R² |
|---|---|---|---|---|---|---|
| Mean predictor | 0.429 | 0.00 | 20.0 % | 49.9 % | 0.458 | 0.00 |
| Gradient boosting on shape parameters | 0.073 | 0.966 | 3.4 % | 10.2 % | 0.101 | 0.937 |
| **CNN on mask + Re** | **0.051 ± 0.002** | **0.981** | **2.3 ± 0.1 %** | **6.8 %** | **0.060 ± 0.007** | **0.973** |

The CNN sees only pixels and Re, while the gradient-boosting baseline gets the exact shape
parameters; the CNN is still better for every shape and in both flow regimes
(Cd mean relative error: circles 0.8 %, rectangles 2.8 %, triangles 3.4 %; steady 1.8 %,
vortex shedding 2.9 %).

Limits:
- The surrogate error is about as large as the solver's own discretisation error (Cd 2.6 % below
  the Schäfer–Turek benchmark at D = 20 cells), so lower numbers would mostly fit solver artefacts.
- Trained on one channel setup (no-slip walls, blockage up to 25 %, so Cd is above free-stream
  values), three shape families and Re 20 to 100. Other shapes or flow conditions are untested.
- Cl of circles (true value about 0): CNN MAE 0.018 against 0.005 for gradient boosting, which is
  told the shape is a circle; a pixelated circle is not perfectly symmetric.
