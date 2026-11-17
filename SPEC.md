# CFD-Drag-2D — Specification

## Goal
An end-to-end pipeline: a validated 2D Lattice Boltzmann (LBM) solver (CPU C++ and CUDA)
generates a dataset of obstacle shapes and their aerodynamic coefficients, and a CNN
surrogate learns to predict those coefficients from the shape alone, orders of
magnitude faster than the solver.

## Physics
- D2Q9 LBM, BGK collision, half-way bounce-back on obstacle and channel walls,
  Zou-He velocity inlet / pressure outlet.
- Inputs per case: shape (circle, rect, tri), size1, size2, angle of attack, Reynolds number Re.
- Re = U * D / nu, with D = projected frontal height of the (rotated) obstacle.
  Velocity U is fixed (low Mach); nu, and hence tau, is derived from Re.
- Forces via momentum exchange, **time-averaged** over a window after warm-up.
- Outputs per case: Cd (mean), Cl (mean), Cl_rms, pressure loss, convergence flag.
- Blockage ratio D / channel height is limited (default max 0.25, see params).
- Validation: channel (Poiseuille) flow vs analytic profile; cylinder Cd vs the
  Schäfer–Turek 2D-1 benchmark (Re = 20, Cd ≈ 5.58), within a documented tolerance.

## Dataset
- Per sample: geometry mask (solid/fluid, fixed resolution), Re, shape params, targets.
  Optional: velocity-magnitude image (visualisation only, never a CNN input).
- Seeded and reproducible; generation runs in parallel on CPU, or on GPU via Kaggle.
- Large datasets live outside git (Kaggle dataset); the repo keeps a tiny example set.

## Surrogate model
- Input: geometry mask + Re (injected as an extra channel or into the dense head).
- Targets: Cd (primary), Cl (secondary).
- Baselines to beat: mean predictor and a gradient-boosting model on the shape parameters.
- Split train/val/test by sample; report MAE, relative error and R² on the test set.
- Runs on CPU with a tiny dataset; full training on a Kaggle GPU via a notebook that clones the repo.

## Apps
- CLI: solver binary, dataset generator, train/evaluate scripts.
- GUI (PyQt6): design a shape, run the solver, see the flow field and the CNN prediction side by side.

## Non-goals
3D flow, turbulence models, high-Re (> ~200) flows, real-time GPU visualisation.
