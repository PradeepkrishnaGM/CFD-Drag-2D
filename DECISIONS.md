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
