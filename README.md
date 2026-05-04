# CFD-Drag-2D: Aerodynamic Dataset Generator & Flow Studio

![C++](https://img.shields.io/badge/C++-17-blue) ![CUDA](https://img.shields.io/badge/CUDA-Enabled-green) ![Python](https://img.shields.io/badge/Python-3.8+-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview
This repository contains a high-performance 2D Computational Fluid Dynamics (CFD) solver based on the D2Q9 Lattice Boltzmann Method (LBM), implemented "from scratch" in both standard C++ and CUDA. 

Designed as a complete aerodynamic "data factory," the project features a custom interactive GUI for real-time flow visualization and a Python automation pipeline. It is purpose-built to rapidly generate large-scale, geometry-aware fluid flow datasets and compute precise aerodynamic metrics (Drag Coefficient, Pressure Loss) for machine learning and surrogate model training.

## Features
* **Hybrid Compute Engine:** Includes a standard C++ solver for high compatibility, and a memory-efficient CUDA solver for ultra-fast, PCIe-optimized GPU batch generation.
* **Interactive GUI Studio:** A responsive, dark-themed PyQt6 desktop application to manually design obstacles, tune physics parameters, and visualize flow fields instantly.
* **Custom Geometry Handling:** Simulates fluid interaction around circles, rectangles, and triangles with dynamic sizes and variable Angles of Attack (AoA).
* **Native Physics Calculation:** Accurately calculates the Drag Coefficient ($C_d$) via the Momentum Exchange Method directly inside the C++/CUDA simulation loop.
* **Batch Automation:** Features a robust Python wrapper (`auto.py`) to procedurally generate balanced datasets of thousands of `(Image, Metric)` samples with built-in numerical stability checks.

## Prerequisites
* **C++ Compiler:** GCC/G++ with C++17 support.
* **CUDA Support (Optional):** NVIDIA CUDA Toolkit (requires `nvcc`).
* **Python Environment:** Python 3.8+
* **Dependencies:** `PyQt6`, `Pillow`, `pandas`, `tqdm` (Can be installed via `pip install -r requirements.txt`)

## Quick Start

### 1. Compile the Solver(s)
You can compile the standard CPU solver or the GPU-accelerated version depending on your hardware.

```bash
g++ -O3 -std=c++17 solver.cpp -o solver
```

### 2. Launch the Interactive Flow Studio (GUI)
Explore the physics interactively by running the desktop application. The GUI allows for both single manual runs and automated batch generation.

```bash
python app.py
```
(Note: Ensure the compiled executable is in the same directory as the script.)

### 3. CLI Dataset Generation
To bypass the GUI and autonomously generate a large-scale training dataset using the terminal:

```bash
python auto.py --samples 1000 --shapes circle rect tri
```

## Output Format
Whether running via the GUI's batch mode or the CLI automation script, the pipeline will generate a dataset/ directory containing:
1. `training_data.csv`: A master record tracking the geometric inputs (shape, size1, size2, angle, velocity) and the corresponding aerodynamic targets ($C_d$ and $\Delta P$).
2. `images/`: A folder containing the high-resolution .png representations of the steady-state velocity fields for each sample, acting as the visual input for downstream CNN/Computer Vision tasks.
