/**
 * 2D Lattice Boltzmann solver (D2Q9, TRT collision) for flow past an obstacle in a channel.
 * Build with: make solver  (from the repo root; output: build/solver)
 *
 * Conventions (lattice units, dx = dt = 1, reference density 1):
 *  - Rows y = 0 and y = ny-1 are wall nodes. With half-way bounce-back the effective
 *    walls sit at y = 0.5 and y = ny - 1.5, so the fluid channel width is H = ny - 2.
 *  - U is the MEAN inlet velocity (uniform or parabolic profile).
 *  - D is the projected frontal height of the rotated obstacle (H for --shape none).
 *  - Re = U * D / nu, nu = (tau - 0.5) / 3, so tau is derived from Re.
 *  - Cd = 2 <Fx> / (U^2 D), Cl = 2 <Fy> / (U^2 D), averaged over the averaging window.
 *
 * Defaults mirror cfd_drag/params.py, which is authoritative; the Python scripts always
 * pass values explicitly.
 *
 * Exit codes: 0 ok, 2 invalid configuration (rejected before running), 3 diverged.
 */

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

// D2Q9 lattice constants
const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const double w[9] = {4.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0, 1.0 / 9.0,
                     1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0};
const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

enum Cell : uint8_t { FLUID = 0, OBSTACLE = 1, WALL = 2 };
const double TRT_LAMBDA = 3.0 / 16.0;  // TRT magic parameter (3/16: exact half-way walls)
const double HALF_CELL = 0.5;  // solid-node inset from the nominal boundary

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
    double min_tau = 0.52;         // stability guard
    double max_blockage = 0.25;    // D / H guard
    double steady_cl_rms = 0.02;   // Cl_rms below this is reported as 'steady'
    bool box = false;              // closed box (all walls), for conservation tests
    bool dump = false;             // write mask.npy and fields.npy
    std::string out_dir = ".";
};

static double frontal_height(const Config& c) {
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

template <typename T>
static void write_npy(const std::string& path, const std::vector<T>& data,
                      const std::vector<size_t>& shape, const std::string& descr) {
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

class LBMSolver {
   public:
    Config cfg;
    double D = 0.0, tau = 0.0, nu = 0.0;
    int N = 0;
    // Post-collision populations, structure-of-arrays: A[i * N + k] for direction i, node k.
    // Each step pulls from A, collides, and writes B (fused stream + collide), then swaps.
    std::vector<double> A, B;
    std::vector<double> rho, ux, uy, u_in;  // macroscopic fields (stored on the last step)
    std::vector<uint8_t> cell;
    std::vector<uint8_t> interior;          // fluid node whose 9 pull sources are all fluid
    int off[9];                             // node-index offset of the pull source per direction
    double Fx = 0.0, Fy = 0.0;

    inline int idx(int x, int y) const { return y * cfg.nx + x; }

    static double equilibrium(int i, double r, double u_x, double u_y) {
        double cu = cx[i] * u_x + cy[i] * u_y;
        double u2 = u_x * u_x + u_y * u_y;
        return w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }

    bool inside_obstacle(int x, int y) const {
        double dx = x - cfg.cx, dy = y - cfg.cy;
        double t = cfg.angle * M_PI / 180.0;
        double lx = dx * std::cos(t) + dy * std::sin(t);
        double ly = -dx * std::sin(t) + dy * std::cos(t);
        // Half-way bounce-back places the wall half a cell outside the last solid node,
        // so a node is solid only if it lies at least HALF_CELL inside the shape boundary.
        // This makes the effective obstacle match its nominal size (see DECISIONS.md).
        const double h = HALF_CELL;
        if (cfg.shape == "circle") return lx * lx + ly * ly <= (cfg.size1 - h) * (cfg.size1 - h);
        if (cfg.shape == "rect") return std::abs(lx) <= cfg.size1 / 2 - h && std::abs(ly) <= cfg.size2 / 2 - h;
        if (cfg.shape == "tri") {
            // Isosceles triangle: flat base on the left, tip on the right at 0 deg.
            // Vertices counter-clockwise, so the interior is left of every edge.
            const double vx[3] = {cfg.size2 / 2, -cfg.size2 / 2, -cfg.size2 / 2};
            const double vy[3] = {0.0, cfg.size1 / 2, -cfg.size1 / 2};
            for (int e = 0; e < 3; ++e) {
                double ex = vx[(e + 1) % 3] - vx[e], ey = vy[(e + 1) % 3] - vy[e];
                double dist = (ex * (ly - vy[e]) - ey * (lx - vx[e])) / std::hypot(ex, ey);
                if (dist < h) return false;
            }
            return true;
        }
        return false;
    }

    explicit LBMSolver(const Config& c) : cfg(c) {
        D = frontal_height(cfg);
        nu = cfg.u * D / cfg.re;
        tau = 3.0 * nu + 0.5;

        N = cfg.nx * cfg.ny;
        A.assign(9 * N, 0.0);
        B.assign(9 * N, 0.0);
        rho.assign(N, 1.0);
        ux.assign(N, 0.0);
        uy.assign(N, 0.0);
        cell.assign(N, FLUID);
        interior.assign(N, 0);
        for (int i = 0; i < 9; ++i) off[i] = -(cx[i] + cy[i] * cfg.nx);

        // Inlet profile per row (mean velocity = u)
        u_in.assign(cfg.ny, 0.0);
        double H = cfg.ny - 2.0;
        for (int y = 1; y < cfg.ny - 1; ++y) {
            double s = (y - 0.5) / H;
            u_in[y] = (cfg.inlet == "parabolic") ? 6.0 * cfg.u * s * (1.0 - s) : cfg.u;
        }

        for (int y = 0; y < cfg.ny; ++y) {
            for (int x = 0; x < cfg.nx; ++x) {
                int k = idx(x, y);
                bool wall = y == 0 || y == cfg.ny - 1 || (cfg.box && (x == 0 || x == cfg.nx - 1));
                if (wall) cell[k] = WALL;
                else if (inside_obstacle(x, y)) cell[k] = OBSTACLE;
                else if (cfg.box) ux[k] = u_in[y];  // open channel starts at rest (inlet ramps up)
                // Equilibrium is unchanged by collision, so it is a valid post-collision state.
                for (int i = 0; i < 9; ++i) A[i * N + k] = equilibrium(i, rho[k], ux[k], uy[k]);
            }
        }
        for (int y = 1; y < cfg.ny - 1; ++y)
            for (int x = 1; x < cfg.nx - 1; ++x) {
                int k = idx(x, y);
                bool ok = cell[k] == FLUID;
                for (int i = 1; i < 9 && ok; ++i) ok = cell[k + off[i]] == FLUID;
                interior[k] = ok;
            }
    }

    double fluid_mass() const {
        double m = 0.0;
        for (int k = 0; k < N; ++k)
            if (cell[k] == FLUID)
                for (int i = 0; i < 9; ++i) m += A[i * N + k];
        return m;
    }

    // Zou-He velocity inlet (x = 0): unknown populations 1, 5, 8.
    static void zou_he_inlet(double* g, double u) {
        double r = (g[0] + g[2] + g[4] + 2.0 * (g[3] + g[6] + g[7])) / (1.0 - u);
        double d = 0.5 * (g[2] - g[4]);
        g[1] = g[3] + (2.0 / 3.0) * r * u;
        g[5] = g[7] - d + (1.0 / 6.0) * r * u;
        g[8] = g[6] + d + (1.0 / 6.0) * r * u;
    }

    // Zou-He pressure outlet (x = nx - 1, rho = 1): unknown populations 3, 6, 7.
    static void zou_he_outlet(double* g) {
        double u = -1.0 + g[0] + g[2] + g[4] + 2.0 * (g[1] + g[5] + g[8]);
        double d = 0.5 * (g[2] - g[4]);
        g[3] = g[1] - (2.0 / 3.0) * u;
        g[6] = g[8] - d - (1.0 / 6.0) * u;
        g[7] = g[5] + d - (1.0 / 6.0) * u;
    }

    // One fused LBM step: pull-stream (with half-way bounce-back and Zou-He boundaries),
    // then TRT collision. Sets Fx, Fy to the momentum-exchange force on the obstacle.
    // Macroscopic fields are written only when store_fields is true.
    void step(int n, bool store_fields) {
        // TRT: symmetric part relaxes with tau (sets viscosity), antisymmetric part with
        // tau_minus chosen so that (tau - 1/2)(tau_minus - 1/2) = TRT_LAMBDA.
        const double om_p = 1.0 / tau;
        const double om_m = 1.0 / (0.5 + TRT_LAMBDA / (tau - 0.5));
        // Smooth sin^2 ramp of the inlet velocity avoids a start-up shock.
        const double ramp = n >= cfg.ramp ? 1.0 : std::pow(std::sin(0.5 * M_PI * n / cfg.ramp), 2);
        const double* __restrict a = A.data();
        double* __restrict b = B.data();
        double fx = 0.0, fy = 0.0;

        for (int y = 1; y < cfg.ny - 1; ++y) {
            for (int x = 0; x < cfg.nx; ++x) {
                const int k = idx(x, y);
                if (cell[k] != FLUID) continue;
                double g[9];
                if (interior[k]) {
                    for (int i = 0; i < 9; ++i) g[i] = a[i * N + k + off[i]];
                } else {
                    for (int i = 0; i < 9; ++i) {
                        const int sx = x - cx[i], sy = y - cy[i];
                        if (sx < 0 || sx >= cfg.nx) { g[i] = 0.0; continue; }  // set by Zou-He
                        const int ks = idx(sx, sy);
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
                    if (!cfg.box && x == 0) zou_he_inlet(g, ramp * u_in[y]);
                    if (!cfg.box && x == cfg.nx - 1) zou_he_outlet(g);
                }

                double r = 0.0;
                for (int i = 0; i < 9; ++i) r += g[i];
                const double u_x = (g[1] - g[3] + g[5] - g[6] - g[7] + g[8]) / r;
                const double u_y = (g[2] - g[4] + g[5] + g[6] - g[7] - g[8]) / r;
                if (store_fields) { rho[k] = r; ux[k] = u_x; uy[k] = u_y; }

                double feq[9];
                for (int i = 0; i < 9; ++i) feq[i] = equilibrium(i, r, u_x, u_y);
                b[k] = g[0] - om_p * (g[0] - feq[0]);
                for (int i = 1; i <= 2; ++i) {  // opposite pairs (1,3), (2,4)
                    const int j = i + 2;
                    const double sp = om_p * 0.5 * ((g[i] + g[j]) - (feq[i] + feq[j]));
                    const double sm = om_m * 0.5 * ((g[i] - g[j]) - (feq[i] - feq[j]));
                    b[i * N + k] = g[i] - sp - sm;
                    b[j * N + k] = g[j] - sp + sm;
                }
                for (int i = 5; i <= 6; ++i) {  // opposite pairs (5,7), (6,8)
                    const int j = i + 2;
                    const double sp = om_p * 0.5 * ((g[i] + g[j]) - (feq[i] + feq[j]));
                    const double sm = om_m * 0.5 * ((g[i] - g[j]) - (feq[i] - feq[j]));
                    b[i * N + k] = g[i] - sp - sm;
                    b[j * N + k] = g[j] - sp + sm;
                }
            }
        }
        Fx = fx;
        Fy = fy;
        A.swap(B);
    }

    double pressure_loss() const {
        double in = 0, out = 0;
        int n_in = 0, n_out = 0;
        for (int y = 1; y < cfg.ny - 1; ++y) {
            if (cell[idx(1, y)] == FLUID) { in += rho[idx(1, y)]; ++n_in; }
            if (cell[idx(cfg.nx - 2, y)] == FLUID) { out += rho[idx(cfg.nx - 2, y)]; ++n_out; }
        }
        return (in / n_in - out / n_out) / 3.0;  // p = rho * cs^2
    }

    void save_image(const std::string& path) const {
        std::ofstream ppm(path, std::ios::binary);
        ppm << "P6\n" << cfg.nx << " " << cfg.ny << "\n255\n";
        double umax = 1.5 * (cfg.inlet == "parabolic" ? 1.5 * cfg.u : cfg.u);
        for (int y = cfg.ny - 1; y >= 0; --y) {
            for (int x = 0; x < cfg.nx; ++x) {
                int k = idx(x, y);
                if (cell[k] != FLUID) { ppm << (char)0 << (char)0 << (char)0; continue; }
                double v = std::min(1.0, std::sqrt(ux[k] * ux[k] + uy[k] * uy[k]) / umax);
                auto ch = [&](double o) { return (char)std::clamp((int)(255.0 * (1.5 - std::abs(4.0 * v - o))), 0, 255); };
                ppm << ch(3.0) << ch(2.0) << ch(1.0);  // jet colormap
            }
        }
    }

    void dump(const std::string& dir) const {
        size_t ny = cfg.ny, nx = cfg.nx;
        write_npy(dir + "/mask.npy", cell, {ny, nx}, "|u1");
        std::vector<double> fields;
        fields.reserve(3 * nx * ny);
        fields.insert(fields.end(), ux.begin(), ux.end());
        fields.insert(fields.end(), uy.begin(), uy.end());
        fields.insert(fields.end(), rho.begin(), rho.end());
        write_npy(dir + "/fields.npy", fields, {3, ny, nx}, "<f8");
    }
};

static int fail(const std::string& msg) {
    std::cerr << "Invalid configuration: " << msg << "\n";
    return 2;
}

int main(int argc, char** argv) {
    Config cfg;
    int legacy_steps = -1;
    try {
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
            else if (a == "--steps") legacy_steps = std::stoi(next());  // total; 2/3 warm-up
            else if (a == "--min-tau") cfg.min_tau = std::stod(next());
            else if (a == "--max-blockage") cfg.max_blockage = std::stod(next());
            else if (a == "--steady-cl-rms") cfg.steady_cl_rms = std::stod(next());
            else if (a == "--out-dir") cfg.out_dir = next();
            else if (a == "--box") cfg.box = true;
            else if (a == "--dump") cfg.dump = true;
            else throw std::runtime_error("unknown argument " + a);
        }
    } catch (const std::exception& e) {
        return fail(e.what());
    }

    if (cfg.shape != "circle" && cfg.shape != "rect" && cfg.shape != "tri" && cfg.shape != "none")
        return fail("unknown shape " + cfg.shape);
    if (cfg.inlet != "uniform" && cfg.inlet != "parabolic") return fail("unknown inlet " + cfg.inlet);
    if (cfg.nx < 8 || cfg.ny < 5 || cfg.re <= 0 || cfg.u <= 0) return fail("non-positive size, Re or U");
    if (cfg.cx < 0) cfg.cx = cfg.nx / 4.0;
    if (cfg.cy < 0) cfg.cy = cfg.ny / 2.0;
    double flow_through = cfg.nx / cfg.u;
    if (legacy_steps > 0 && cfg.warmup < 0 && cfg.avg < 0) {
        cfg.warmup = legacy_steps * 2 / 3;
        cfg.avg = legacy_steps - cfg.warmup;
    }
    if (cfg.warmup < 0) cfg.warmup = (int)(2.0 * flow_through);
    if (cfg.avg <= 0) cfg.avg = std::max(1, (int)flow_through);
    if (cfg.ramp <= 0) cfg.ramp = std::max(1, cfg.warmup / 4);

    LBMSolver s(cfg);
    double H = cfg.ny - 2.0, blockage = s.D / H;
    std::cout << "LBM D2Q9 | domain " << cfg.nx << "x" << cfg.ny << " | " << cfg.shape
              << " | Re " << cfg.re << " | U " << cfg.u << " | D " << s.D << " | tau " << s.tau
              << " | blockage " << blockage << "\n";
    if (s.tau < cfg.min_tau) return fail("tau " + std::to_string(s.tau) + " < min_tau");
    if (cfg.shape != "none" && blockage > cfg.max_blockage)
        return fail("blockage " + std::to_string(blockage) + " > max_blockage");

    const double mass0 = s.fluid_mass();
    const int total = cfg.warmup + cfg.avg, half = cfg.warmup + cfg.avg / 2;
    double sx = 0, sy = 0, syy = 0, sx1 = 0, sx2 = 0;
    bool diverged = false;
    for (int step = 0; step < total; ++step) {
        s.step(step, step == total - 1);
        if (step >= cfg.warmup) {
            sx += s.Fx;
            sy += s.Fy;
            syy += s.Fy * s.Fy;
            (step < half ? sx1 : sx2) += s.Fx;
        }
        if (step % 500 == 0) {
            std::cout << "Step " << step << " / " << total << "\r" << std::flush;
            if (!std::isfinite(s.fluid_mass())) { diverged = true; break; }
        }
    }
    std::cout << "\n";
    if (!std::isfinite(s.fluid_mass())) diverged = true;

    const double q = 0.5 * cfg.u * cfg.u * s.D;  // dynamic pressure * reference length
    const int n = cfg.avg, n1 = half - cfg.warmup, n2 = total - half;
    double Fx_mean = sx / n, Fy_mean = sy / n;
    double Cd = Fx_mean / q, Cl = Fy_mean / q;
    double Cl_rms = std::sqrt(std::max(0.0, syy / n - Fy_mean * Fy_mean)) / q;
    double cd_drift = (n1 > 0 && n2 > 0 && Fx_mean != 0.0)
                          ? std::abs(sx2 / n2 - sx1 / n1) / std::abs(Fx_mean) : 0.0;
    double mass_change = (s.fluid_mass() - mass0) / mass0;
    std::string status = diverged ? "diverged" : (Cl_rms < cfg.steady_cl_rms ? "steady" : "periodic");

    std::ofstream csv(cfg.out_dir + "/results.csv");
    csv << std::setprecision(10)
        << "shape_type,size1,size2,angle,re,u,tau,D,blockage,Cd,Cl,Cl_rms,cd_drift,"
           "pressure_loss,mass_rel_change,steps,status\n"
        << cfg.shape << "," << cfg.size1 << "," << cfg.size2 << "," << cfg.angle << "," << cfg.re
        << "," << cfg.u << "," << s.tau << "," << s.D << "," << blockage << "," << Cd << "," << Cl
        << "," << Cl_rms << "," << cd_drift << "," << s.pressure_loss() << "," << mass_change << ","
        << total << "," << status << "\n";
    csv.close();
    s.save_image(cfg.out_dir + "/flow_field.ppm");
    if (cfg.dump) s.dump(cfg.out_dir);

    std::cout << "=> Cd " << Cd << " | Cl " << Cl << " | Cl_rms " << Cl_rms << " | status " << status << "\n";
    return diverged ? 3 : 0;
}
