/**
 * CPU LBM solver: one case per run. Physics and I/O live in lbm_common.hpp.
 * Build with: make solver  (from the repo root; output: build/solver)
 *
 * Exit codes: 0 ok, 2 invalid configuration (rejected before running), 3 diverged.
 */
#include "lbm_common.hpp"

using namespace lbm;

static int fail(const std::string& msg) {
    std::cerr << "Invalid configuration: " << msg << "\n";
    return 2;
}

int main(int argc, char** argv) {
    Config cfg;
    try {
        parse_args(argc, argv, cfg);
    } catch (const std::exception& e) {
        return fail(e.what());
    }
    if (!cfg.batch.empty()) return fail("--batch is only supported by the CUDA solver");
    std::string err = validate(cfg);
    if (!err.empty()) return fail(err);
    finalize(cfg);

    const double tau = tau_of(cfg);
    std::cout << "LBM D2Q9 | domain " << cfg.nx << "x" << cfg.ny << " | " << cfg.shape << " | Re "
              << cfg.re << " | U " << cfg.u << " | D " << frontal_height(cfg) << " | tau " << tau
              << " | blockage " << frontal_height(cfg) / (cfg.ny - 2.0) << "\n";

    Geometry g = build_geometry(cfg);
    std::vector<double> A(9 * g.N), B(9 * g.N);
    init_populations(cfg, g, A.data());
    Fields f;
    f.rho.assign(g.N, 1.0);
    f.ux.assign(g.N, 0.0);
    f.uy.assign(g.N, 0.0);
    if (cfg.mean_fields) f.mean.assign(3 * g.N, 0.0);

    StepParams p = make_step_params(cfg, g, tau);
    const double mass0 = fluid_mass(g, A.data());
    const int total = cfg.warmup + cfg.avg;
    ForceStats stats;
    bool diverged = false;
    for (int step = 0; step < total; ++step) {
        p.ramp = cfg.box ? 1.0 : ramp_factor(step, cfg.ramp);
        const bool last = step == total - 1;
        double* rho = last ? f.rho.data() : nullptr;
        double* acc = cfg.mean_fields && step >= cfg.warmup ? f.mean.data() : nullptr;
        double fx = 0.0, fy = 0.0;
        for (int y = 1; y < cfg.ny - 1; ++y)
            for (int x = 0; x < cfg.nx; ++x)
                node_update(x, y, p, g.cell.data(), g.interior.data(), g.u_in.data(), A.data(),
                            B.data(), fx, fy, rho, f.ux.data(), f.uy.data(), acc);
        A.swap(B);
        accumulate(stats, step, cfg.warmup, cfg.avg, fx, fy);
        if (step % 500 == 0) {
            std::cout << "Step " << step << " / " << total << "\r" << std::flush;
            if (!std::isfinite(fluid_mass(g, A.data()))) { diverged = true; break; }
        }
    }
    std::cout << "\n";
    std::string status = write_outputs(cfg, g, f, stats, mass0, fluid_mass(g, A.data()), diverged);
    return status == "diverged" ? 3 : 0;
}
