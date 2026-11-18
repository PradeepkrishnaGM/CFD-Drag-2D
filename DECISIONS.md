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
| 2026-11-18 | M4 dataset: N = 5000 cases, seed 0, CUDA back-end on Kaggle 2x T4, no velocity images | ~1 h of GPU; images are visualisation only and large |
| 2026-11-18 | Kaggle dataset private until M6; dataset check fails below 95 % ok cases | Owner approval |
| 2026-11-18 | Dataset check (cfd_drag/check.py) plus a CPU spot check of 8 random cases run inside the generation notebook | Catches broken data before training; proves GPU results are reproducible from the code |
| 2026-11-18 | Dataset v1: 5000 cases from commit 96556fb, 100 % ok, spot check max rel diff 4e-10; 59 min on 2x T4 (0.70 s/case) | Kaggle run 2026-11-18; private Kaggle dataset doffice2026/cfd-drag-2d-lbm |
| 2026-11-18 | M5 surrogate: Re (standardised log Re) joins the CNN in the dense head; split 70/15/15 by sample, seed 0 | Mask stays binary; 750 test cases give stable metrics |
| 2026-11-18 | M5 metrics: MAE + R² for Cd and Cl, relative error for Cd only; pass = CNN beats mean and gradient-boosting baselines on test Cd MAE | Cl median ~0 makes relative error meaningless; a fixed % target invites tuning to the test set |
| 2026-11-18 | ML dependencies (torch, scikit-learn, matplotlib) as optional extra `[ml]`; checkpoints stay out of git until M6 | Keeps the solver/dataset install light |
| 2026-11-18 | Training seed is a CLI option (`--seed`, default 0); the split seed stays fixed | Seed spread measured on the same test set |
| 2026-11-18 | Surrogate v1 (commits 38ce07f / 38db73e, Kaggle T4): CNN test Cd MAE 0.051 ± 0.002, Cd rel 2.3 ± 0.1 %, Cl MAE 0.060 ± 0.007 (3 seeds) vs gradient boosting 0.073 / 3.4 % / 0.101 -> passes | Owner reviewed; README reports mean ± std and the limits |
| 2026-11-18 | M6 run without per-step approval | Owner: "go full speed and non stop and complete the project" |
| 2026-11-18 | GUI builds the CNN input with a one-step solver run (--warmup 1 --avg 1 --dump), ~20 ms | Exactly the training mask; no second rasteriser to keep in sync |
| 2026-11-18 | Trained surrogate committed as models/surrogate_v1.pt (1 MB, seed 0) | GUI works right after cloning; small enough for git |
| 2026-11-18 | GUI inputs are shape, sizes, angle and Re (inlet velocity and legacy --steps removed); invalid cases disable RUN SOLVER, out-of-range cases get a warning but still a CNN prediction | Same settings as the dataset, so CNN and solver are comparable |
| 2026-11-18 | GUI dataset mode calls cfd_drag.dataset.generate (progress callback added); old GUI batch code removed | One pipeline, same seeds and checks as the CLI |
| 2026-11-18 | Dataset v1 published as a GitHub release asset (v1.0, zip + model), not committed to git; Kaggle copy stays private | Owner: dataset on GitHub, not Kaggle; keeps the git history small |
| 2026-11-18 | v2 = flow-field prediction (M7-M9): time-averaged ux, uy, p in a 256 x 96 window, same 5000 cases, U-Net vs mean-field and nearest-neighbour baselines, release v2.0 | Owner approval (PLAN_M7.md) |
| 2026-11-18 | M7: solver flag `--mean-fields` writes fields_mean.npy (ux, uy, rho summed in the kernel over the --avg window); off by default, so v1 runs are unchanged | Averaging matches how Cd/Cl are averaged; ~+1 MB GPU memory per case |
| 2026-11-18 | M7: stored channels ux / U, uy / U, Cp = (rho - 1) / 3 / (0.5 U^2) (outlet rho = 1), float16, window 256 x 96 (x from cx - 32, rows 2..97); obstacle cells u = 0, Cp = 0; window mask rebuilt from the 64 x 64 crop (no extra file) | Dimensionless, ~147 kB per case; crop fits exactly inside the window |
| 2026-11-18 | M7 checks: no-slip on obstacle cells, zeros for failed rows, volume flux of every window column within 5 % of the first column; spot check compares fields (abs diff <= 1e-2, float16 rounding) | Local 12-case run: flux deviation <= 0.54 %, Cd/Cl/masks identical to v1 rows 0-11 |
| 2026-11-18 | M8: U-Net (4 levels, base 32, 7.8 M parameters); inputs mask, signed distance / 32 and standardised log Re as channels; loss = MSE on fluid cells of per-channel standardised fields; AdamW + cosine, 100 epochs, fp16 mixed precision on CUDA; best epoch by validation (velocity + Cp relative L2) / 2 | Standard choices for steady-flow field surrogates (signed distance input as in DeepCFD); same split as the M5 CNN so both are tested on the same 750 cases |
| 2026-11-18 | M8 metrics: relative L2 error of the velocity vector field and of Cp on fluid cells, also in the wake (x >= obstacle centre), plus MAE; baselines: mean training field and nearest neighbour in shape-parameter space; pass = U-Net beats both on velocity and Cp | The mean field is a strong velocity baseline (free stream dominates), so Cp and wake errors are reported too |
| 2026-11-18 | M9: Flow Studio views (shape preview, U-Net, solver mean field, difference) with fixed colour ranges; the solver now runs with --mean-fields in the GUI; U-Net CPU inference ~35-40 ms | Solver view shows the same quantity the U-Net predicts |
| 2026-11-18 | M9: U-Net weights (31 MB) are a release asset fetched by scripts/fetch_field_model.py, not tracked in git; the 1 MB CNN stays in git | Keep the repository small (owner rule: large outputs out of git) |
| 2026-11-18 | M7 result (Kaggle T4, ~67 min, commit 1c4f8eb): 5000/5000 ok; Cd, Cl, status and masks identical to v1; flux deviation median 0.42 %, max 1.02 %; no-slip and failed-row checks pass; 8-case CPU spot check matches | Field dataset accepted (dataset/fields_v1, gitignored) |
