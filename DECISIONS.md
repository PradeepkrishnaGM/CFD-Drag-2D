# Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-11-17 | Scope: full pipeline (validated solver, CUDA, dataset, CNN surrogate, GUI) | Owner's choice |
| 2026-11-17 | CNN input is geometry mask + Re, not the solved flow field | Flow-field input leaks the answer; a surrogate must work before solving |
| 2026-11-17 | Re is a solver input; U fixed, tau derived from Re | Controls flow regime and Mach number; makes samples comparable |
| 2026-11-17 | Cd/Cl time-averaged after warm-up; Cl_rms recorded | Wakes shed vortices above Re ≈ 47; last-step values are noise |
| 2026-11-17 | Reference length = projected frontal height of rotated shape | size2 is wrong for rotated rects/triangles |
| 2026-11-17 | Blockage ratio capped (default 0.25) | High blockage inflates Cd and is unlike free-stream flow |
| 2026-11-17 | Targets: Cd primary, Cl secondary; pressure loss recorded only | Pressure loss depends on domain length, not just the shape |
| 2026-11-17 | Keep the PyQt6 GUI, update in M6 | Portfolio demo value |
| 2026-11-17 | Existing example_data will be regenerated after M1 | Old values come from the uncorrected solver |
| 2026-11-17 | M1 defaults approved: U = 0.05, Re 20–150, blockage cap 0.25, benchmark tolerance 5% | Owner approval |
| 2026-11-17 | Collision changed from BGK to TRT (Λ = 3/16) | BGK diverged at Re 100 with sharp corners (tau ≈ 0.53); TRT is stable there and places half-way walls exactly |
| 2026-11-17 | Solid nodes are inset half a cell from the nominal shape boundary | Half-way bounce-back puts the wall half a cell outside the last solid node; without the inset the cylinder acts ~1 cell wider (Cd +5.4% at D = 20) |
| 2026-11-17 | min_tau raised from 0.51 (planned) to 0.52 | tau = 0.51 still diverged with TRT; 0.515 ran; 0.52 adds margin |
| 2026-11-17 | Open channel starts at rest; inlet ramps up with sin² over warmup/4 steps | Impulsive start caused divergence at low tau |
| 2026-11-17 | Poiseuille test compares profile shape and flux-corrected pressure gradient | LBM is weakly compressible: density drops ~5% along the test channel, raising u by the same amount |
| 2026-11-17 | Solver outputs now go to --out-dir; results.csv is overwritten, not appended | Parallel runs (M2) need isolated outputs |
| 2026-11-17 | Benchmark convergence: Cd = 5.434 (−2.6%) at D = 20, 5.555 (−0.4%) at D = 40 vs 5.5795 | Error shrinks with resolution, confirming the scheme; the test uses D = 20 for speed |
