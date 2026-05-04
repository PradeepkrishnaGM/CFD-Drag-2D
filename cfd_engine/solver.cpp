/**
 * Standalone 2D Lattice Boltzmann Method (LBM) Solver
 * Compile with: g++ -O3 -std=c++17 lbm_solver.cpp -o lbm_solver
 */

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm>

// D2Q9 Lattice Constants
const int cx[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
const int cy[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
const double w[9] = {
    4.0 / 9.0, 
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};
// Opposite directions for bounce-back
const int opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

struct Config {
    int nx = 400;
    int ny = 100;
    std::string shape = "circle";
    double size1 = 10.0; // radius, width, or base
    double size2 = 10.0; // height
    double angle = 0.0;
    double velocity = 0.1;
    int steps = 10000;
    
    // Internal placements
    double cx, cy;
};

class LBMSolver {
private:
    Config cfg;
    double tau = 0.6; // Relaxation time (determines kinematic viscosity)

    std::vector<double> f;
    std::vector<double> f_post;
    std::vector<double> rho;
    std::vector<double> ux;
    std::vector<double> uy;
    std::vector<double> prev_u;
    
    std::vector<bool> is_solid;    // True for any wall/obstacle
    std::vector<bool> is_obstacle; // True ONLY for the obstacle (for drag calc)

    double Fx_drag = 0.0;
    double Fy_lift = 0.0;

    inline int idx(int x, int y) const { return y * cfg.nx + x; }
    inline int f_idx(int x, int y, int i) const { return (y * cfg.nx + x) * 9 + i; }

    double equilibrium(int i, double r, double u_x, double u_y) const {
        double cu = cx[i] * u_x + cy[i] * u_y;
        double u2 = u_x * u_x + u_y * u_y;
        return w[i] * r * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }

    bool checkObstacleGeometry(int x, int y) {
        double dx = x - cfg.cx;
        double dy = y - cfg.cy;

        // Apply rotation matrix
        double theta = cfg.angle * M_PI / 180.0;
        double lx = dx * cos(theta) + dy * sin(theta);
        double ly = -dx * sin(theta) + dy * cos(theta);

        if (cfg.shape == "circle") {
            return (lx * lx + ly * ly) <= (cfg.size1 * cfg.size1);
        } 
        else if (cfg.shape == "rect") {
            return (std::abs(lx) <= cfg.size1 / 2.0) && (std::abs(ly) <= cfg.size2 / 2.0);
        } 
        else if (cfg.shape == "tri") {
            // Isosceles triangle. Flat base faces left, tip faces right by default (0 deg).
            // Vertices: tip=(size2/2, 0), bottom=(-size2/2, -size1/2), top=(-size2/2, size1/2)
            double v1x = cfg.size2 / 2.0,  v1y = 0.0;
            double v2x = -cfg.size2 / 2.0, v2y = cfg.size1 / 2.0;
            double v3x = -cfg.size2 / 2.0, v3y = -cfg.size1 / 2.0;

            // Barycentric coordinates for inside test
            double denom = (v2y - v3y) * (v1x - v3x) + (v3x - v2x) * (v1y - v3y);
            double w1 = ((v2y - v3y) * (lx - v3x) + (v3x - v2x) * (ly - v3y)) / denom;
            double w2 = ((v3y - v1y) * (lx - v3x) + (v1x - v3x) * (ly - v3y)) / denom;
            double w3 = 1.0 - w1 - w2;

            return (w1 >= -1e-6 && w2 >= -1e-6 && w3 >= -1e-6);
        }
        return false;
    }

    void apply_zou_he_boundaries() {
        // Left Inlet (Velocity boundary)
        for (int y = 1; y < cfg.ny - 1; ++y) {
            if (is_solid[idx(0, y)]) continue;
            double r = f[f_idx(0,y,0)] + f[f_idx(0,y,2)] + f[f_idx(0,y,4)] +
                       2.0 * (f[f_idx(0,y,3)] + f[f_idx(0,y,6)] + f[f_idx(0,y,7)]);
            r /= (1.0 - cfg.velocity);

            f[f_idx(0,y,1)] = f[f_idx(0,y,3)] + (2.0/3.0) * r * cfg.velocity;
            f[f_idx(0,y,5)] = f[f_idx(0,y,7)] - 0.5 * (f[f_idx(0,y,2)] - f[f_idx(0,y,4)]) + (1.0/6.0) * r * cfg.velocity;
            f[f_idx(0,y,8)] = f[f_idx(0,y,6)] + 0.5 * (f[f_idx(0,y,2)] - f[f_idx(0,y,4)]) + (1.0/6.0) * r * cfg.velocity;
        }

        // Right Outlet (Constant pressure / density boundary)
        for (int y = 1; y < cfg.ny - 1; ++y) {
            int x = cfg.nx - 1;
            if (is_solid[idx(x, y)]) continue;
            double r = 1.0;
            double u = -1.0 + (f[f_idx(x,y,0)] + f[f_idx(x,y,2)] + f[f_idx(x,y,4)] +
                               2.0 * (f[f_idx(x,y,1)] + f[f_idx(x,y,5)] + f[f_idx(x,y,8)])) / r;

            f[f_idx(x,y,3)] = f[f_idx(x,y,1)] - (2.0/3.0) * r * u;
            f[f_idx(x,y,6)] = f[f_idx(x,y,8)] - 0.5 * (f[f_idx(x,y,2)] - f[f_idx(x,y,4)]) - (1.0/6.0) * r * u;
            f[f_idx(x,y,7)] = f[f_idx(x,y,5)] + 0.5 * (f[f_idx(x,y,2)] - f[f_idx(x,y,4)]) - (1.0/6.0) * r * u;
        }
    }

public:
    LBMSolver(const Config& configuration) : cfg(configuration) {
        int N = cfg.nx * cfg.ny;
        f.resize(N * 9);
        f_post.resize(N * 9);
        rho.resize(N, 1.0);
        ux.resize(N, cfg.velocity);
        uy.resize(N, 0.0);
        prev_u.resize(N, cfg.velocity);
        is_solid.resize(N, false);
        is_obstacle.resize(N, false);

        cfg.cx = cfg.nx / 4.0;
        cfg.cy = cfg.ny / 2.0;

        // Initialize Geometry & Boundary Arrays
        for (int y = 0; y < cfg.ny; ++y) {
            for (int x = 0; x < cfg.nx; ++x) {
                // Top and Bottom walls
                if (y == 0 || y == cfg.ny - 1) {
                    is_solid[idx(x, y)] = true;
                }
                // Central Obstacle
                else if (checkObstacleGeometry(x, y)) {
                    is_solid[idx(x, y)] = true;
                    is_obstacle[idx(x, y)] = true;
                    ux[idx(x, y)] = 0.0;
                    uy[idx(x, y)] = 0.0;
                }

                // Initialize F with equilibrium
                for (int i = 0; i < 9; ++i) {
                    f[f_idx(x, y, i)] = equilibrium(i, rho[idx(x,y)], ux[idx(x,y)], uy[idx(x,y)]);
                }
            }
        }
    }

    bool step(int current_step) {
        // COLLISION & MACROSCOPIC VARS
        for (int y = 1; y < cfg.ny - 1; ++y) {
            for (int x = 0; x < cfg.nx; ++x) {
                if (is_solid[idx(x, y)]) continue;

                double r = 0, ru = 0, rv = 0;
                for (int i = 0; i < 9; ++i) {
                    double val = f[f_idx(x, y, i)];
                    r += val;
                    ru += cx[i] * val;
                    rv += cy[i] * val;
                }
                
                double u_x = ru / r;
                double u_y = rv / r;

                rho[idx(x, y)] = r;
                ux[idx(x, y)] = u_x;
                uy[idx(x, y)] = u_y;

                for (int i = 0; i < 9; ++i) {
                    double feq = equilibrium(i, r, u_x, u_y);
                    f_post[f_idx(x, y, i)] = f[f_idx(x, y, i)] - (1.0 / tau) * (f[f_idx(x, y, i)] - feq);
                }
            }
        }

        // Convergence Check (every 100 steps)
        bool converged = false;
        if (current_step % 100 == 0) {
            double error = 0.0;
            double sum_u = 0.0;
            for (int i = 0; i < cfg.nx * cfg.ny; ++i) {
                if (!is_solid[i]) {
                    double u_mag = std::sqrt(ux[i]*ux[i] + uy[i]*uy[i]);
                    error += std::abs(u_mag - prev_u[i]);
                    sum_u += u_mag;
                    prev_u[i] = u_mag;
                }
            }
            
            // A heuristic of `nx * 5` ensures the flow crosses the domain 5 times.
            int min_steps_for_convergence = cfg.nx * 5; 
            if (current_step > min_steps_for_convergence && sum_u > 1e-9 && (error / sum_u) < 1e-5) {
                converged = true;
            }
        }

        // STREAMING & DRAG CALCULATION
        Fx_drag = 0.0;
        Fy_lift = 0.0;

        for (int y = 0; y < cfg.ny; ++y) {
            for (int x = 0; x < cfg.nx; ++x) {
                if (is_solid[idx(x, y)]) continue;

                for (int i = 0; i < 9; ++i) {
                    int nx = x - cx[i];
                    int ny = y - cy[i];

                    if (nx < 0 || nx >= cfg.nx || ny < 0 || ny >= cfg.ny) {
                        f[f_idx(x, y, i)] = f_post[f_idx(x, y, i)];
                        continue;
                    }

                    if (is_solid[idx(nx, ny)]) {
                        // Mid-way bounce back
                        int opp_dir = opp[i];
                        f[f_idx(x, y, i)] = f_post[f_idx(x, y, opp_dir)];
                        
                        // Momentum Exchange Method for forces on obstacle
                        if (is_obstacle[idx(nx, ny)]) {
                            Fx_drag += 2.0 * cx[opp_dir] * f_post[f_idx(x, y, opp_dir)];
                            Fy_lift += 2.0 * cy[opp_dir] * f_post[f_idx(x, y, opp_dir)];
                        }
                    } else {
                        // Standard pull stream
                        f[f_idx(x, y, i)] = f_post[f_idx(nx, ny, i)];
                    }
                }
            }
        }

        // APPLY BOUNDARIES
        apply_zou_he_boundaries();

        return converged;
    }

    void saveResultsAndImage() {
        // --- Calculate final Aerodynamics metrics ---
        double L = cfg.size2; 
        if (cfg.shape == "circle") L = 2.0 * cfg.size1;

        double Cd = (2.0 * Fx_drag) / (1.0 * cfg.velocity * cfg.velocity * L);

        // Pressure Drop (average rho at inlet vs outlet)
        double rho_in = 0, rho_out = 0;
        int count_in = 0, count_out = 0;
        for (int y = 1; y < cfg.ny - 1; ++y) {
            if (!is_solid[idx(1, y)]) { rho_in += rho[idx(1, y)]; count_in++; }
            if (!is_solid[idx(cfg.nx - 2, y)]) { rho_out += rho[idx(cfg.nx - 2, y)]; count_out++; }
        }
        double dP = ((rho_in / count_in) - (rho_out / count_out)) / 3.0; // P = rho * cs^2

        // --- Save to CSV ---
        std::ifstream infile("results.csv");
        bool write_header = !infile.good();
        infile.close();

        std::ofstream csv("results.csv", std::ios::app);
        if (write_header) {
            csv << "shape_type,size1,size2,angle,velocity,Cd,pressure_loss\n";
        }
        csv << cfg.shape << "," << cfg.size1 << "," << cfg.size2 << ","
            << cfg.angle << "," << cfg.velocity << "," << Cd << "," << dP << "\n";
        csv.close();

        std::cout << "=> Final Cd: " << Cd << ", Pressure Loss: " << dP << std::endl;

        // --- Save Flow Field PPM ---
        std::ofstream ppm("flow_field.ppm", std::ios::binary);
        ppm << "P6\n" << cfg.nx << " " << cfg.ny << "\n255\n";
        
        for (int y = cfg.ny - 1; y >= 0; --y) { // Top to bottom for image formats
            for (int x = 0; x < cfg.nx; ++x) {
                if (is_solid[idx(x, y)]) {
                    ppm << (char)0 << (char)0 << (char)0;
                    continue;
                }

                double u = std::sqrt(ux[idx(x, y)] * ux[idx(x, y)] + uy[idx(x, y)] * uy[idx(x, y)]);
                double v = std::min(1.0, u / (1.5 * cfg.velocity)); // Map [0, 1.5 * U]

                // Standard Jet Colormap
                int r = std::max(0, std::min(255, (int)(255.0 * (1.5 - std::abs(4.0 * v - 3.0)))));
                int g = std::max(0, std::min(255, (int)(255.0 * (1.5 - std::abs(4.0 * v - 2.0)))));
                int b = std::max(0, std::min(255, (int)(255.0 * (1.5 - std::abs(4.0 * v - 1.0)))));
                
                ppm << (char)r << (char)g << (char)b;
            }
        }
        ppm.close();
        std::cout << "=> Saved 'results.csv' and 'flow_field.ppm'\n";
    }
};

int main(int argc, char** argv) {
    Config cfg;

    // Headless CLI parser
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--width" && i + 1 < argc) cfg.nx = std::stoi(argv[++i]);
        else if (arg == "--height" && i + 1 < argc) cfg.ny = std::stoi(argv[++i]);
        else if (arg == "--shape" && i + 1 < argc) cfg.shape = argv[++i];
        else if (arg == "--size1" && i + 1 < argc) cfg.size1 = std::stod(argv[++i]);
        else if (arg == "--size2" && i + 1 < argc) cfg.size2 = std::stod(argv[++i]);
        else if (arg == "--angle" && i + 1 < argc) cfg.angle = std::stod(argv[++i]);
        else if (arg == "--velocity" && i + 1 < argc) cfg.velocity = std::stod(argv[++i]);
        else if (arg == "--steps" && i + 1 < argc) cfg.steps = std::stoi(argv[++i]);
    }

    std::cout << "Initializing 2D LBM Solver...\n";
    std::cout << "Domain: " << cfg.nx << "x" << cfg.ny << ", Velocity: " << cfg.velocity << "\n";
    std::cout << "Obstacle: " << cfg.shape << " (s1:" << cfg.size1 << ", s2:" << cfg.size2 
              << ", angle:" << cfg.angle << " deg)\n";

    LBMSolver solver(cfg);

    int min_steps_for_convergence = cfg.nx * 5;
    std::cout << "Minimum warm-up steps required: " << min_steps_for_convergence << "\n";

    for (int step = 0; step < cfg.steps; ++step) {
        bool converged = solver.step(step);
        
        if (step % 500 == 0) {
            std::cout << "Step " << step << " / " << cfg.steps << "\r" << std::flush;
        }

        if (converged) {
            std::cout << "\nSteady-state convergence reached at step " << step << "!" << std::endl;
            break;
        }
    }
    std::cout << "\nSimulation Complete. Calculating and exporting metrics..." << std::endl;
    
    solver.saveResultsAndImage();

    return 0;
}
