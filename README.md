# CFD-Drag-2D: Aerodynamic Dataset Generator

![C++](https://img.shields.io/badge/C++-17-blue) ![Python](https://img.shields.io/badge/Python-3.8+-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

## Overview
This repository contains a high-performance, "from scratch" 2D Computational Fluid Dynamics (CFD) solver based on the D2Q9 Lattice Boltzmann Method (LBM). It is specifically designed as a "data factory" to rapidly generate fluid flow images and compute precise aerodynamic coefficients (Drag, Pressure Loss) for arbitrary geometries.

## Features
* **Custom Geometry Handling:** Simulates flow around circles, rectangles, and triangles with variable Angle of Attack (AoA).
* **Momentum Exchange:** Accurately calculates the Drag Coefficient ($C_d$) natively in C++.
* **Python Automation:** Includes a wrapper script to automatically generate balanced datasets of thousands of samples.

## Prerequisites
* GCC/G++ (C++17 support)
* Python 3.8+
* `pandas`, `Pillow`, `tqdm`

## Quick Start
### 1. Compile the Solver
```bash
g++ -O3 main.cpp -o cfd_solver
```

### 2. Run a Single Simulation
```bash
./cfd_solver --shape rectangle --width 10 --height 20 --angle 45 --velocity 0.05
```

### 3. Generate a Dataset
```bash
python generate_dataset.py --samples 1000 --output_dir ./data/
```

## Output Format
The pipeline outputs a `results.csv` containing the aerodynamic metrics and a folder of corresponding `.png` velocity fields representing the wind tunnel states.
