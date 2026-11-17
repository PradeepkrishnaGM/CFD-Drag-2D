# Build the CPU LBM solver into ../build/solver
CXX      ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall
BUILD    := ../build

$(BUILD)/solver: solver.cpp
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(BUILD)

.PHONY: clean
