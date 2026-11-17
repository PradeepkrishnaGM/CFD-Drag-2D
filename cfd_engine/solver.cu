/**
 * CUDA LBM solver: simulates a batch of cases in one launch. A single 400x100 case is too
 * small to fill a GPU, so all cases advance together (grid z = case index). Physics and
 * I/O are shared with the CPU solver via lbm_common.hpp (node_update is the same code).
 *
 * Usage: solver_cuda --batch cases.csv [global options as for the CPU solver]
 *   cases.csv columns: out_dir,shape,size1,size2,angle,re   (header line required)
 *   All cases share the global options (domain, U, inlet, warmup, avg, guards, --dump).
 *   Each case writes the same files as the CPU solver into its out_dir; a case rejected
 *   by the guards writes rejected.txt instead.
 *
 * Determinism: node updates are independent, and per-step forces are summed by one block
 * per case in a fixed order, so results do not depend on batch size or composition.
 *
 * Build with: make cuda  (from the repo root; output: build/solver_cuda)
 * Exit codes: 0 ran (per-case status in results.csv), 2 invalid arguments, 4 CUDA error.
 */
#include <cuda_runtime.h>

#include <chrono>

#include "lbm_common.hpp"

using namespace lbm;

static void cuda_check(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) {
        std::cerr << "CUDA error at line " << line << " (" << what << "): " << cudaGetErrorString(e) << "\n";
        std::exit(4);
    }
}
#define CK(x) cuda_check((x), #x, __LINE__)

constexpr int BLOCK_X = 128;    // threads per block along x in the step kernel
constexpr int FORCE_BLOCK = 256;  // threads per block in the force reduction

__global__ void step_kernel(StepParams base, size_t N, const double* om_p, const double* om_m,
                            const uint8_t* cell, const uint8_t* interior, const double* u_in,
                            const double* A, double* B, double* fnode, int store, double* rho,
                            double* ux, double* uy) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    const size_t c = blockIdx.z;
    if (x >= base.nx) return;
    StepParams p = base;
    p.om_p = om_p[c];
    p.om_m = om_m[c];
    const uint8_t* cl = cell + c * N;
    const uint8_t* in = interior + c * N;
    double fx = 0.0, fy = 0.0;
    node_update(x, y, p, cl, in, u_in, A + c * 9 * N, B + c * 9 * N, fx, fy,
                store ? rho + c * N : nullptr, ux + c * N, uy + c * N);
    const size_t k = (size_t)y * base.nx + x;
    if (!in[k]) {
        fnode[2 * (c * N + k)] = fx;
        fnode[2 * (c * N + k) + 1] = fy;
    }
}

// One block per case: sum the per-node forces over the case's force nodes in a fixed
// order (strided partial sums + tree reduction), then accumulate the statistics.
__global__ void force_kernel(size_t N, const int* fstart, const int* fcount, const int* fidx,
                             const double* fnode, ForceStats* stats, int step, int warmup, int avg) {
    __shared__ double sx[FORCE_BLOCK], sy[FORCE_BLOCK];
    const size_t c = blockIdx.x;
    const int t = threadIdx.x;
    double ax = 0.0, ay = 0.0;
    for (int j = t; j < fcount[c]; j += FORCE_BLOCK) {
        const size_t k = fidx[fstart[c] + j];
        ax += fnode[2 * (c * N + k)];
        ay += fnode[2 * (c * N + k) + 1];
    }
    sx[t] = ax;
    sy[t] = ay;
    __syncthreads();
    for (int s = FORCE_BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) {
            sx[t] += sx[t + s];
            sy[t] += sy[t + s];
        }
        __syncthreads();
    }
    if (t == 0) accumulate(stats[c], step, warmup, avg, sx[0], sy[0]);
}

struct HostCase {
    Config cfg;
    Geometry geo;
    double tau = 0.0;
};

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(item);
    return out;
}

int main(int argc, char** argv) {
    Config base;
    try {
        parse_args(argc, argv, base);
    } catch (const std::exception& e) {
        std::cerr << "Invalid arguments: " << e.what() << "\n";
        return 2;
    }
    if (base.batch.empty()) {
        std::cerr << "Invalid arguments: --batch cases.csv is required\n";
        return 2;
    }

    // ---- Read and validate cases ------------------------------------------------------
    std::ifstream in(base.batch);
    if (!in) {
        std::cerr << "Invalid arguments: cannot open " << base.batch << "\n";
        return 2;
    }
    std::vector<HostCase> cases;
    std::string line;
    std::getline(in, line);  // header
    int n_rejected = 0;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        auto f = split_csv(line);
        if (f.size() != 6) {
            std::cerr << "Invalid arguments: bad batch line: " << line << "\n";
            return 2;
        }
        Config c = base;
        c.batch.clear();
        c.out_dir = f[0];
        c.shape = f[1];
        c.size1 = std::stod(f[2]);
        c.size2 = std::stod(f[3]);
        c.angle = std::stod(f[4]);
        c.re = std::stod(f[5]);
        std::string err = validate(c);
        if (!err.empty()) {
            std::ofstream(c.out_dir + "/rejected.txt") << err << "\n";
            ++n_rejected;
            continue;
        }
        finalize(c);
        HostCase hc;
        hc.cfg = c;
        hc.tau = tau_of(c);
        cases.push_back(std::move(hc));
    }
    if (cases.empty()) {
        std::cout << "No valid cases (" << n_rejected << " rejected)\n";
        return 0;
    }
    const Config& c0 = cases[0].cfg;  // shared settings (identical for all cases)
    const int nx = c0.nx, ny = c0.ny, C = (int)cases.size();
    const size_t N = (size_t)nx * ny;
    const int total = c0.warmup + c0.avg;

    // ---- Host-side geometry and initial state -----------------------------------------
    std::vector<uint8_t> h_cell(C * N), h_interior(C * N);
    std::vector<double> h_omp(C), h_omm(C);
    std::vector<int> h_fstart(C), h_fcount(C), h_fidx;
    std::vector<double> h_A(9 * N);
    std::vector<double> mass0(C);

    double *d_A, *d_B, *d_fnode, *d_rho, *d_ux, *d_uy, *d_uin, *d_omp, *d_omm;
    uint8_t *d_cell, *d_interior;
    int *d_fstart, *d_fcount, *d_fidx;
    ForceStats* d_stats;
    CK(cudaMalloc(&d_A, 9 * N * C * sizeof(double)));
    CK(cudaMalloc(&d_B, 9 * N * C * sizeof(double)));
    CK(cudaMalloc(&d_fnode, 2 * N * C * sizeof(double)));
    CK(cudaMalloc(&d_rho, N * C * sizeof(double)));
    CK(cudaMalloc(&d_ux, N * C * sizeof(double)));
    CK(cudaMalloc(&d_uy, N * C * sizeof(double)));
    CK(cudaMalloc(&d_uin, ny * sizeof(double)));
    CK(cudaMalloc(&d_omp, C * sizeof(double)));
    CK(cudaMalloc(&d_omm, C * sizeof(double)));
    CK(cudaMalloc(&d_cell, N * C));
    CK(cudaMalloc(&d_interior, N * C));
    CK(cudaMalloc(&d_stats, C * sizeof(ForceStats)));
    CK(cudaMemset(d_stats, 0, C * sizeof(ForceStats)));
    CK(cudaMemset(d_fnode, 0, 2 * N * C * sizeof(double)));

    StepParams base_params{};
    for (int c = 0; c < C; ++c) {
        HostCase& hc = cases[c];
        hc.geo = build_geometry(hc.cfg);
        StepParams p = make_step_params(hc.cfg, hc.geo, hc.tau);
        if (c == 0) base_params = p;
        h_omp[c] = p.om_p;
        h_omm[c] = p.om_m;
        std::copy(hc.geo.cell.begin(), hc.geo.cell.end(), h_cell.begin() + c * N);
        std::copy(hc.geo.interior.begin(), hc.geo.interior.end(), h_interior.begin() + c * N);
        h_fstart[c] = (int)h_fidx.size();
        h_fcount[c] = (int)hc.geo.force_nodes.size();
        h_fidx.insert(h_fidx.end(), hc.geo.force_nodes.begin(), hc.geo.force_nodes.end());
        init_populations(hc.cfg, hc.geo, h_A.data());
        mass0[c] = fluid_mass(hc.geo, h_A.data());
        CK(cudaMemcpy(d_A + c * 9 * N, h_A.data(), 9 * N * sizeof(double), cudaMemcpyHostToDevice));
        hc.geo.interior.clear();  // no longer needed on the host
        hc.geo.interior.shrink_to_fit();
    }
    CK(cudaMalloc(&d_fstart, C * sizeof(int)));
    CK(cudaMalloc(&d_fcount, C * sizeof(int)));
    CK(cudaMalloc(&d_fidx, std::max<size_t>(1, h_fidx.size()) * sizeof(int)));
    CK(cudaMemcpy(d_cell, h_cell.data(), N * C, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_interior, h_interior.data(), N * C, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_uin, cases[0].geo.u_in.data(), ny * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_omp, h_omp.data(), C * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_omm, h_omm.data(), C * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_fstart, h_fstart.data(), C * sizeof(int), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(d_fcount, h_fcount.data(), C * sizeof(int), cudaMemcpyHostToDevice));
    if (!h_fidx.empty())
        CK(cudaMemcpy(d_fidx, h_fidx.data(), h_fidx.size() * sizeof(int), cudaMemcpyHostToDevice));
    h_cell.clear();
    h_interior.clear();

    int dev = 0;
    cudaDeviceProp prop;
    CK(cudaGetDevice(&dev));
    CK(cudaGetDeviceProperties(&prop, dev));
    std::cout << "CUDA LBM | " << prop.name << " | " << C << " cases (" << n_rejected << " rejected) | "
              << nx << "x" << ny << " | " << total << " steps\n";

    // ---- Time loop ----------------------------------------------------------------------
    const dim3 grid((nx + BLOCK_X - 1) / BLOCK_X, ny, C), block(BLOCK_X);
    auto t0 = std::chrono::steady_clock::now();
    for (int step = 0; step < total; ++step) {
        base_params.ramp = c0.box ? 1.0 : ramp_factor(step, c0.ramp);
        const int store = step == total - 1;
        step_kernel<<<grid, block>>>(base_params, N, d_omp, d_omm, d_cell, d_interior, d_uin, d_A, d_B,
                                     d_fnode, store, d_rho, d_ux, d_uy);
        force_kernel<<<C, FORCE_BLOCK>>>(N, d_fstart, d_fcount, d_fidx, d_fnode, d_stats, step,
                                         c0.warmup, c0.avg);
        std::swap(d_A, d_B);
        if (step % 2000 == 0) {
            CK(cudaGetLastError());
            std::cout << "Step " << step << " / " << total << "\r" << std::flush;
        }
    }
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "\nTime " << secs << " s | " << (double)C * N * total / secs / 1e9 << " GLUPS (all nodes)\n";

    // ---- Outputs ----------------------------------------------------------------------
    std::vector<ForceStats> h_stats(C);
    CK(cudaMemcpy(h_stats.data(), d_stats, C * sizeof(ForceStats), cudaMemcpyDeviceToHost));
    Fields f;
    f.rho.resize(N);
    f.ux.resize(N);
    f.uy.resize(N);
    int n_diverged = 0;
    for (int c = 0; c < C; ++c) {
        CK(cudaMemcpy(h_A.data(), d_A + c * 9 * N, 9 * N * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(f.rho.data(), d_rho + c * N, N * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(f.ux.data(), d_ux + c * N, N * sizeof(double), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(f.uy.data(), d_uy + c * N, N * sizeof(double), cudaMemcpyDeviceToHost));
        // Non-fluid nodes keep the host defaults used by the CPU solver.
        for (size_t k = 0; k < N; ++k)
            if (cases[c].geo.cell[k] != FLUID) { f.rho[k] = 1.0; f.ux[k] = 0.0; f.uy[k] = 0.0; }
        const double m1 = fluid_mass(cases[c].geo, h_A.data());
        if (write_outputs(cases[c].cfg, cases[c].geo, f, h_stats[c], mass0[c], m1, false) == "diverged")
            ++n_diverged;
    }
    std::cout << "Done: " << C << " cases, " << n_diverged << " diverged, " << n_rejected << " rejected\n";

    for (void* p : {(void*)d_A, (void*)d_B, (void*)d_fnode, (void*)d_rho, (void*)d_ux, (void*)d_uy,
                    (void*)d_uin, (void*)d_omp, (void*)d_omm, (void*)d_cell, (void*)d_interior,
                    (void*)d_fstart, (void*)d_fcount, (void*)d_fidx, (void*)d_stats})
        cudaFree(p);
    return 0;
}
