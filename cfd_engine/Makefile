# Build the LBM solvers into ../build
#   make            CPU solver   -> ../build/solver
#   make cuda       CUDA solver  -> ../build/solver_cuda  (needs nvcc; Turing sm_75 + PTX for newer GPUs)
CXX       ?= g++
CXXFLAGS  ?= -O3 -march=native -std=c++17 -Wall
NVCC      ?= nvcc
NVCCFLAGS ?= -O3 -std=c++17 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75
EXTRA_NVCCFLAGS ?=
BUILD     := ../build

$(BUILD)/solver: solver.cpp lbm_common.hpp
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BUILD)/solver_cuda: solver.cu lbm_common.hpp
	mkdir -p $(BUILD)
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $< -o $@

cuda: $(BUILD)/solver_cuda

clean:
	rm -rf $(BUILD)

.PHONY: cuda clean
