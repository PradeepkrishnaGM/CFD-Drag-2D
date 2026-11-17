/**
 * Shared core of the CPU (solver.cpp) and CUDA (solver.cu) LBM solvers.
 *
 * Model: D2Q9 lattice Boltzmann, TRT collision, half-way bounce-back, Zou-He velocity
 * inlet and pressure outlet. Conventions (lattice units, dx = dt = 1, reference density 1):
 *  - Rows y = 0 and y = ny-1 are wall nodes; with half-way bounce-back the effective
 *    walls sit at y = 0.5 and y = ny - 1.5, so the fluid channel width is H = ny - 2.
 *  - U is the MEAN inlet velocity (uniform or parabolic profile).
 *  - D is the projected frontal height of the rotated obstacle (H for --shape none).
 *  - Re = U * D / nu, nu = (tau - 0.5) / 3, so tau is derived from Re.
 *  - Cd = 2 <Fx> / (U^2 D), Cl = 2 <Fy> / (U^2 D), averaged over the averaging window.
 *
 * node_update() is the single per-node kernel (stream + boundaries + collide) used by both
 * back-ends, so the physics is tested once on the CPU.
 *
 * Defaults mirror cfd_drag/params.py, which is authoritative; the Python scripts always
 * pass values explicitly.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef __CUDACC__
#define LBM_HD __host__ __device__
#define LBM_UNROLL _Pragma("unroll")
#else
#define LBM_HD
#define LBM_UNROLL _Pragma("GCC unroll 9")
#endif

namespace lbm {

enum Cell : uint8_t { FLUID = 0, OBSTACLE = 1, WALL = 2 };
constexpr double TRT_LAMBDA = 3.0 / 16.0;  // TRT magic parameter (3/16: exact half-way walls)
constexpr double HALF_CELL = 0.5;          // solid-node inset from the nominal boundary

// D2Q9 lattice (host copies; device code uses local constexpr copies in node_update)
const int CX[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int CY[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const double W[9] = {4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0, 1.0 / 9.0,
                     1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};

struct Config {
    int nx = 400;
    int ny = 100;
    std::string shape = "circle";  // circle | rect | tri | none
    double size1 = 10.0;           // circle: radius; rect: width (x); tri: base (y)
    double size2 = 10.0;           // rect: height (y); tri: length (x)
    double angle = 0.0;            // rotation [deg], counter-clockwise
    double re = 100.0;
    double u = 0.05;               // mean inlet velocity
    std::string inlet = "uniform"; // uniform | parabolic
    double cx = -1.0, cy = -1.0;   // obstacle centre; default (nx/4, ny/2)
    int warmup = -1;               // steps before averaging; default 2 flow-throughs
    int avg = -1;                  // averaging steps; default 1 flow-through
    int ramp = -1;                 // inlet ramp-up steps; default warmup / 4
    int legacy_steps = -1;         // --steps (old interface): total, 2/3 warm-up
    double min_tau = 0.52;         // stability guard
    double max_blockage = 0.25;    // D / H guard
    double steady_cl_rms = 0.02;   // Cl_rms below this is reported as 'steady'
    bool box = false;              // closed box (all walls), for conservation tests
    bool dump = false;             // write mask.npy and fields.npy
    std::string out_dir = ".";
    std::string batch;             // CUDA only: CSV of cases (out_dir,shape,size1,size2,angle,re)
};

// ---------------------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------------------
inline void parse_args(int argc, char** argv, Config& cfg) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return argv[++i];
        };
        if (a == "--width") cfg.nx = std::stoi(next());
        else if (a == "--height") cfg.ny = std::stoi(next());
        else if (a == "--shape") cfg.shape = next();
        else if (a == "--size1") cfg.size1 = std::stod(next());
        else if (a == "--size2") cfg.size2 = std::stod(next());
        else if (a == "--angle") cfg.angle = std::stod(next());
        else if (a == "--re") cfg.re = std::stod(next());
        else if (a == "--u" || a == "--velocity") cfg.u = std::stod(next());
        else if (a == "--inlet") cfg.inlet = next();
        else if (a == "--cx") cfg.cx = std::stod(next());
        else if (a == "--cy") cfg.cy = std::stod(next());
        else if (a == "--warmup") cfg.warmup = std::stoi(next());
        else if (a == "--avg") cfg.avg = std::stoi(next());
        else if (a == "--ramp") cfg.ramp = std::stoi(next());
        else if (a == "--steps") cfg.legacy_steps = std::stoi(next());
        else if (a == "--min-tau") cfg.min_tau = std::stod(next());
        else if (a == "--max-blockage") cfg.max_blockage = std::stod(next());
        else if (a == "--steady-cl-rms") cfg.steady_cl_rms = std::stod(next());
        else if (a == "--out-dir") cfg.out_dir = next();
        else if (a == "--batch") cfg.batch = next();
        else if (a == "--box") cfg.box = true;
        else if (a == "--dump") cfg.dump = true;
        else throw std::runtime_error("unknown argument " + a);
    }
}

// Fill in derived defaults (obstacle centre, step counts). Call after parsing.
inline void finalize(Config& cfg) {
    if (cfg.cx < 0) cfg.cx = cfg.nx / 4.0;
    if (cfg.cy < 0) cfg.cy = cfg.ny / 2.0;
    double flow_through = cfg.nx / cfg.u;
    if (cfg.legacy_steps > 0 && cfg.warmup < 0 && cfg.avg < 0) {
        cfg.warmup = cfg.legacy_steps * 2 / 3;
        cfg.avg = cfg.legacy_steps - cfg.warmup;
    }
    if (cfg.warmup < 0) cfg.warmup = (int)(2.0 * flow_through);
    if (cfg.avg <= 0) cfg.avg = std::max(1, (int)flow_through);
    if (cfg.ramp <= 0) cfg.ramp = std::max(1, cfg.warmup / 4);
}

inline double frontal_height(const Config& c) {
    if (c.shape == "none") return c.ny - 2.0;
    if (c.shape == "circle") return 2.0 * c.size1;
    std::vector<std::pair<double, double>> v;  // local vertices (lx, ly)
    if (c.shape == "rect") {
        v = {{-c.size1 / 2, -c.size2 / 2}, {c.size1 / 2, -c.size2 / 2},
             {c.size1 / 2, c.size2 / 2}, {-c.size1 / 2, c.size2 / 2}};
    } else {  // tri
        v = {{c.size2 / 2, 0.0}, {-c.size2 / 2, c.size1 / 2}, {-c.size2 / 2, -c.size1 / 2}};
    }
    double t = c.angle * M_PI / 180.0, lo = 1e300, hi = -1e300;
    for (auto& p : v) {
        double y = p.first * std::sin(t) + p.second * std::cos(t);
        lo = std::min(lo, y);
        hi = std::max(hi, y);
    }
    return hi - lo;
}

inline double tau_of(const Config& c) { return 3.0 * c.u * frontal_height(c) / c.re + 0.5; }

// Returns an empty string if the configuration is valid, else the reason it is rejected.
inline std::string validate(const Config& c) {
    if (c.shape != "circle" && c.shape != "rect" && c.shape != "tri" && c.shape != "none")
        return "unknown shape " + c.shape;
    if (c.inlet != "uniform" && c.inlet != "parabolic") return "unknown inlet " + c.inlet;
    if (c.nx < 8 || c.ny < 5 || c.re <= 0 || c.u <= 0) return "non-positive size, Re or U";
    double tau = tau_of(c), blockage = frontal_height(c) / (c.ny - 2.0);
    if (tau < c.min_tau) return "tau " + std::to_string(tau) + " < min_tau";
    if (c.shape != "none" && blockage > c.max_blockage)
        return "blockage " + std::to_string(blockage) + " > max_blockage";
    return "";
}

// ---------------------------------------------------------------------------------------
// Geometry and initial state
// ---------------------------------------------------------------------------------------
inline bool inside_obstacle(const Config& c, int x, int y) {
    double dx = x - c.cx, dy = y - c.cy;
    double t = c.angle * M_PI / 180.0;
    double lx = dx * std::cos(t) + dy * std::sin(t);
    double ly = -dx * std::sin(t) + dy * std::cos(t);
    // Half-way bounce-back places the wall half a cell outside the last solid node,
    // so a node is solid only if it lies at least HALF_CELL inside the shape boundary.
    // This makes the effective obstacle match its nominal size (see DECISIONS.md).
    const double h = HALF_CELL;
    if (c.shape == "circle") return lx * lx + ly * ly <= (c.size1 - h) * (c.size1 - h);
    if (c.shape == "rect") return std::abs(lx) <= c.size1 / 2 - h && std::abs(ly) <= c.size2 / 2 - h;
    if (c.shape == "tri") {
        // Isosceles triangle: flat base on the left, tip on the right at 0 deg.
        // Vertices counter-clockwise, so the interior is left of every edge.
        const double vx[3] = {c.size2 / 2, -c.size2 / 2, -c.size2 / 2};
        const double vy[3] = {0.0, c.size1 / 2, -c.size1 / 2};
        for (int e = 0; e < 3; ++e) {
            double ex = vx[(e + 1) % 3] - vx[e], ey = vy[(e + 1) % 3] - vy[e];
            double dist = (ex * (ly - vy[e]) - ey * (lx - vx[e])) / std::hypot(ex, ey);
            if (dist < h) return false;
        }
        return true;
    }
    return false;
}

LBM_HD inline double equilibrium(int i, double r, double u_x, double u_y, double w, int ex, int ey) {
    double cu = ex * u_x + ey * u_y;
    double u2 = u_x * u_x + u_y * u_y;
    return w * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
}

struct Geometry {
    int nx = 0, ny = 0, N = 0;
    std::vector<uint8_t> cell;      // FLUID / OBSTACLE / WALL
    std::vector<uint8_t> interior;  // fluid node whose 9 pull sources are all fluid
    std::vector<double> u_in;       // inlet velocity per row (mean = u)
    std::vector<int> force_nodes;   // fluid nodes with an obstacle neighbour (in order)
    int off[9];                     // node-index offset of the pull source per direction
};

inline Geometry build_geometry(const Config& c) {
    Geometry g;
    g.nx = c.nx;
    g.ny = c.ny;
    g.N = c.nx * c.ny;
    g.cell.assign(g.N, FLUID);
    g.interior.assign(g.N, 0);
    g.u_in.assign(c.ny, 0.0);
    for (int i = 0; i < 9; ++i) g.off[i] = -(CX[i] + CY[i] * c.nx);
    double H = c.ny - 2.0;
    for (int y = 1; y < c.ny - 1; ++y) {
        double s = (y - 0.5) / H;
        g.u_in[y] = (c.inlet == "parabolic") ? 6.0 * c.u * s * (1.0 - s) : c.u;
    }
    for (int y = 0; y < c.ny; ++y)
        for (int x = 0; x < c.nx; ++x) {
            bool wall = y == 0 || y == c.ny - 1 || (c.box && (x == 0 || x == c.nx - 1));
            if (wall) g.cell[y * c.nx + x] = WALL;
            else if (inside_obstacle(c, x, y)) g.cell[y * c.nx + x] = OBSTACLE;
        }
    for (int y = 1; y < c.ny - 1; ++y)
        for (int x = 0; x < c.nx; ++x) {
            int k = y * c.nx + x;
            if (g.cell[k] != FLUID) continue;
            bool ok = x > 0 && x < c.nx - 1, near_obstacle = false;
            for (int i = 1; i < 9; ++i) {
                int sx = x - CX[i], sy = y - CY[i];
                if (sx < 0 || sx >= c.nx) continue;
                uint8_t s = g.cell[sy * c.nx + sx];
                ok = ok && s == FLUID;
                near_obstacle = near_obstacle || s == OBSTACLE;
            }
            g.interior[k] = ok;
            if (near_obstacle) g.force_nodes.push_back(k);
        }
    return g;
}

// Initial post-collision populations (structure-of-arrays, f[i * N + k]). Equilibrium is
// unchanged by collision, so it is a valid post-collision state. The open channel starts
// at rest (the inlet ramps up); the closed box starts with the inlet velocity everywhere.
inline void init_populations(const Config& c, const Geometry& g, double* f) {
    for (int k = 0; k < g.N; ++k) {
        int y = k / g.nx;
        double u = (c.box && g.cell[k] == FLUID) ? g.u_in[y] : 0.0;
        for (int i = 0; i < 9; ++i) f[i * g.N + k] = equilibrium(i, 1.0, u, 0.0, W[i], CX[i], CY[i]);
    }
}

inline double fluid_mass(const Geometry& g, const double* f) {
    double m = 0.0;
    for (int k = 0; k < g.N; ++k)
        if (g.cell[k] == FLUID)
            for (int i = 0; i < 9; ++i) m += f[i * g.N + k];
    return m;
}

// ---------------------------------------------------------------------------------------
// The per-node step, shared by CPU and GPU
// ---------------------------------------------------------------------------------------
struct StepParams {
    int nx, ny, N;
    int box;
    double om_p, om_m;  // TRT relaxation rates
    double ramp;        // inlet ramp factor for this step, in [0, 1]
    int off[9];
};

inline StepParams make_step_params(const Config& c, const Geometry& g, double tau) {
    StepParams p;
    p.nx = c.nx;
    p.ny = c.ny;
    p.N = g.N;
    p.box = c.box;
    // TRT: symmetric part relaxes with tau (sets viscosity), antisymmetric part with
    // tau_minus chosen so that (tau - 1/2)(tau_minus - 1/2) = TRT_LAMBDA.
    p.om_p = 1.0 / tau;
    p.om_m = 1.0 / (0.5 + TRT_LAMBDA / (tau - 0.5));
    p.ramp = 1.0;
    for (int i = 0; i < 9; ++i) p.off[i] = g.off[i];
    return p;
}

// Smooth sin^2 ramp of the inlet velocity avoids a start-up shock.
inline double ramp_factor(int n, int ramp) {
    return n >= ramp ? 1.0 : std::pow(std::sin(0.5 * M_PI * n / ramp), 2);
}

/**
 * Update node (x, y): pull-stream from post-collision populations `a` (with half-way
 * bounce-back and Zou-He boundaries), then TRT-collide into `b`. Adds the node's
 * momentum-exchange force on the obstacle to (fx, fy). If rho is non-null, stores the
 * macroscopic fields. Non-fluid and wall-row nodes are left untouched.
 */
LBM_HD inline void node_update(int x, int y, const StepParams& p, const uint8_t* cell,
                               const uint8_t* interior, const double* u_in,
                               const double* __restrict__ a, double* __restrict__ b,
                               double& fx, double& fy, double* rho, double* ux, double* uy) {
    constexpr int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    constexpr int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    constexpr int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    constexpr double w[9] = {4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0, 1.0 / 9.0,
                             1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};
    const int N = p.N, k = y * p.nx + x;
    if (y == 0 || y == p.ny - 1 || cell[k] != FLUID) return;

    double g[9];
    if (interior[k]) {
LBM_UNROLL
        for (int i = 0; i < 9; ++i) g[i] = a[i * N + k + p.off[i]];
    } else {
LBM_UNROLL
        for (int i = 0; i < 9; ++i) {
            const int sx = x - cx[i], sy = y - cy[i];
            if (sx < 0 || sx >= p.nx) { g[i] = 0.0; continue; }  // set by Zou-He below
            const int ks = sy * p.nx + sx;
            if (cell[ks] == FLUID) {
                g[i] = a[i * N + ks];
            } else {  // half-way bounce-back
                const double v = a[opp[i] * N + k];
                g[i] = v;
                if (cell[ks] == OBSTACLE) {
                    fx += 2.0 * cx[opp[i]] * v;
                    fy += 2.0 * cy[opp[i]] * v;
                }
            }
        }
        if (!p.box && x == 0) {  // Zou-He velocity inlet: unknown 1, 5, 8
            const double u = p.ramp * u_in[y];
            const double r = (g[0] + g[2] + g[4] + 2.0 * (g[3] + g[6] + g[7])) / (1.0 - u);
            const double d = 0.5 * (g[2] - g[4]);
            g[1] = g[3] + (2.0 / 3.0) * r * u;
            g[5] = g[7] - d + (1.0 / 6.0) * r * u;
            g[8] = g[6] + d + (1.0 / 6.0) * r * u;
        }
        if (!p.box && x == p.nx - 1) {  // Zou-He pressure outlet (rho = 1): unknown 3, 6, 7
            const double u = -1.0 + g[0] + g[2] + g[4] + 2.0 * (g[1] + g[5] + g[8]);
            const double d = 0.5 * (g[2] - g[4]);
            g[3] = g[1] - (2.0 / 3.0) * u;
            g[6] = g[8] - d - (1.0 / 6.0) * u;
            g[7] = g[5] + d - (1.0 / 6.0) * u;
        }
    }

    double r = 0.0;
LBM_UNROLL
    for (int i = 0; i < 9; ++i) r += g[i];
    const double u_x = (g[1] - g[3] + g[5] - g[6] - g[7] + g[8]) / r;
    const double u_y = (g[2] - g[4] + g[5] + g[6] - g[7] - g[8]) / r;
    if (rho) { rho[k] = r; ux[k] = u_x; uy[k] = u_y; }

    double feq[9];
LBM_UNROLL
    for (int i = 0; i < 9; ++i) feq[i] = equilibrium(i, r, u_x, u_y, w[i], cx[i], cy[i]);
    b[k] = g[0] - p.om_p * (g[0] - feq[0]);
    constexpr int pair_lo[4] = {1, 2, 5, 6};  // opposite pairs (1,3), (2,4), (5,7), (6,8)
LBM_UNROLL
    for (int q = 0; q < 4; ++q) {
        const int i = pair_lo[q], j = i + 2;
        const double sp = p.om_p * 0.5 * ((g[i] + g[j]) - (feq[i] + feq[j]));
        const double sm = p.om_m * 0.5 * ((g[i] - g[j]) - (feq[i] - feq[j]));
        b[i * N + k] = g[i] - sp - sm;
        b[j * N + k] = g[j] - sp + sm;
    }
}

// ---------------------------------------------------------------------------------------
// Force statistics and outputs
// ---------------------------------------------------------------------------------------
struct ForceStats {
    double sx = 0, sy = 0, syy = 0, sx1 = 0, sx2 = 0;  // sums over the averaging window
};

// Accumulate one step's force; window halves are used for the drift estimate.
LBM_HD inline void accumulate(ForceStats& s, int step, int warmup, int avg, double fx, double fy) {
    if (step < warmup) return;
    s.sx += fx;
    s.sy += fy;
    s.syy += fy * fy;
    if (step < warmup + avg / 2) s.sx1 += fx; else s.sx2 += fx;
}

struct Fields {  // macroscopic fields of one case
    std::vector<double> rho, ux, uy;
};

inline double pressure_loss(const Config& c, const Geometry& g, const Fields& f) {
    double in = 0, out = 0;
    int n_in = 0, n_out = 0;
    for (int y = 1; y < c.ny - 1; ++y) {
        int ki = y * c.nx + 1, ko = y * c.nx + c.nx - 2;
        if (g.cell[ki] == FLUID) { in += f.rho[ki]; ++n_in; }
        if (g.cell[ko] == FLUID) { out += f.rho[ko]; ++n_out; }
    }
    return (in / n_in - out / n_out) / 3.0;  // p = rho * cs^2
}

template <typename T>
void write_npy(const std::string& path, const std::vector<T>& data, const std::vector<size_t>& shape,
               const std::string& descr) {
    std::ostringstream h;
    h << "{'descr': '" << descr << "', 'fortran_order': False, 'shape': (";
    for (size_t i = 0; i < shape.size(); ++i) h << shape[i] << (shape.size() == 1 || i + 1 < shape.size() ? ", " : "");
    h << "), }";
    std::string header = h.str();
    size_t total = 10 + header.size() + 1;
    header.append((64 - total % 64) % 64, ' ');
    header.push_back('\n');
    std::ofstream out(path, std::ios::binary);
    out.write("\x93NUMPY\x01\x00", 8);
    uint16_t len = static_cast<uint16_t>(header.size());
    out.put(static_cast<char>(len & 0xff));
    out.put(static_cast<char>(len >> 8));
    out << header;
    out.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(T));
}

inline void save_image(const Config& c, const Geometry& g, const Fields& f, const std::string& path) {
    std::ofstream ppm(path, std::ios::binary);
    ppm << "P6\n" << c.nx << " " << c.ny << "\n255\n";
    double umax = 1.5 * (c.inlet == "parabolic" ? 1.5 * c.u : c.u);
    for (int y = c.ny - 1; y >= 0; --y) {
        for (int x = 0; x < c.nx; ++x) {
            int k = y * c.nx + x;
            if (g.cell[k] != FLUID) { ppm << (char)0 << (char)0 << (char)0; continue; }
            double v = std::min(1.0, std::sqrt(f.ux[k] * f.ux[k] + f.uy[k] * f.uy[k]) / umax);
            auto ch = [&](double o) { return (char)std::clamp((int)(255.0 * (1.5 - std::abs(4.0 * v - o))), 0, 255); };
            ppm << ch(3.0) << ch(2.0) << ch(1.0);  // jet colormap
        }
    }
}

// Write results.csv, flow_field.ppm and (with --dump) mask.npy / fields.npy to cfg.out_dir.
// Returns the status string.
inline std::string write_outputs(const Config& c, const Geometry& g, const Fields& f, const ForceStats& s,
                                 double mass0, double mass1, bool diverged) {
    const double D = frontal_height(c), tau = tau_of(c), blockage = D / (c.ny - 2.0);
    const double q = 0.5 * c.u * c.u * D;  // dynamic pressure * reference length
    const int n = c.avg, n1 = c.avg / 2, n2 = c.avg - c.avg / 2;
    const double Fx_mean = s.sx / n, Fy_mean = s.sy / n;
    const double Cd = Fx_mean / q, Cl = Fy_mean / q;
    const double Cl_rms = std::sqrt(std::max(0.0, s.syy / n - Fy_mean * Fy_mean)) / q;
    const double cd_drift = (n1 > 0 && n2 > 0 && Fx_mean != 0.0)
                                ? std::abs(s.sx2 / n2 - s.sx1 / n1) / std::abs(Fx_mean) : 0.0;
    diverged = diverged || !std::isfinite(mass1);
    const std::string status = diverged ? "diverged" : (Cl_rms < c.steady_cl_rms ? "steady" : "periodic");

    std::ofstream csv(c.out_dir + "/results.csv");
    csv << std::setprecision(10)
        << "shape_type,size1,size2,angle,re,u,tau,D,blockage,Cd,Cl,Cl_rms,cd_drift,"
           "pressure_loss,mass_rel_change,steps,status\n"
        << c.shape << "," << c.size1 << "," << c.size2 << "," << c.angle << "," << c.re << "," << c.u
        << "," << tau << "," << D << "," << blockage << "," << Cd << "," << Cl << "," << Cl_rms << ","
        << cd_drift << "," << pressure_loss(c, g, f) << "," << (mass1 - mass0) / mass0 << ","
        << c.warmup + c.avg << "," << status << "\n";
    csv.close();
    save_image(c, g, f, c.out_dir + "/flow_field.ppm");
    if (c.dump) {
        size_t ny = c.ny, nx = c.nx;
        write_npy(c.out_dir + "/mask.npy", g.cell, {ny, nx}, "|u1");
        std::vector<double> all;
        all.reserve(3 * nx * ny);
        all.insert(all.end(), f.ux.begin(), f.ux.end());
        all.insert(all.end(), f.uy.begin(), f.uy.end());
        all.insert(all.end(), f.rho.begin(), f.rho.end());
        write_npy(c.out_dir + "/fields.npy", all, {3, ny, nx}, "<f8");
    }
    std::cout << "=> Cd " << Cd << " | Cl " << Cl << " | Cl_rms " << Cl_rms << " | status " << status << "\n";
    return status;
}

}  // namespace lbm
