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
| 2026-11-17 | Re range 20–100 (option a) | At Re 150 the tau and blockage guards force D into 20–24.5 cells, coupling Re with size; Re 100 still spans steady and shedding regimes |
| 2026-11-17 | Solver core: fused pull-stream + TRT collide, SoA layout, branch-free interior nodes, -march=native | 1.6x faster (18 -> 29 MLUPS), same Cd to 6 digits |
| 2026-11-17 | Steady/periodic threshold Cl_rms < 0.02 (was 1e-4) | Low-Re steady cases keep Cl_rms ~4e-3 from slow transients; shedding cases are > 0.1 |
| 2026-11-17 | (Re, D) sampled uniformly over the feasible region, then shape/aspect/angle; sizes scaled to hit D | Guards hold by construction; no hidden rejection bias in the solver |
| 2026-11-17 | Per-case RNG seeded by (seed, index); failed cases are kept as rows with ok = False, not redrawn | Dataset is identical for any worker count and resumable |
| 2026-11-17 | CNN input stored as 64x64 obstacle mask cropped around the obstacle centre | Obstacle position is fixed; crop keeps files small (~4 KB/case) |
| 2026-11-17 | Default workers = physical cores | 6 workers: 6.75 s/case; 12 workers: 8 s/case on the i5-10500T |
| 2026-11-17 | scripts/auto.py replaced by scripts/generate.py | New sampler, parallelism, resume and mask output |
| 2026-11-18 | CPU and CUDA solvers share one per-node update (lbm_common.hpp) | One physics implementation, tested on the CPU; CPU cost +6% (8.2 -> 8.7 s per benchmark run) |
| 2026-11-18 | CUDA solver runs a batch of cases per launch, double precision, fixed-order force reduction | A 400x100 case cannot fill a GPU; deterministic results for any batch size |
| 2026-11-18 | Kaggle T4 (P100 retired): GPU results equal CPU results to the 10 digits stored, for steady and shedding cases | Verified on 16 cases (Kaggle run, 2026-11-18) |
| 2026-11-18 | GPU throughput 1.38 s/case on one T4 (128-case batch) vs 6.75 s/case locally | ~5x per GPU; Kaggle's 2 T4s give ~0.7 s/case when there are >= 2 batches |
| 2026-11-18 | Kaggle notebooks clone the public repo and build with the image's nvcc; scripts/kaggle_setup_cuda.sh falls back to NVIDIA's redistributable | No GPU or CUDA toolkit on the dev machine |
