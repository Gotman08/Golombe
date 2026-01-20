# Golomb Ruler Solver - Makefile
# Three versions: Sequential, OpenMP, Hybrid MPI+OpenMP

CXX = g++
MPICXX = mpicxx

# Base flags
CXXFLAGS_BASE = -std=c++17 -Wall -Wextra

# Performance optimization flags
OPT_FLAGS = -O3 -flto=auto -march=native -funroll-loops \
            -falign-functions=32 -falign-loops=32 -fno-rtti

# Production flags
CXXFLAGS = $(CXXFLAGS_BASE) $(OPT_FLAGS)

# OpenMP and SIMD flags
OPENMP_FLAGS = -fopenmp
AVX_FLAGS = -mavx2 -mfma -DUSE_AVX2

# Directories
SRC_DIR = src
INCLUDE_DIR = include
COMMON_DIR = $(SRC_DIR)/common
BUILD_DIR = build
RESULTS_DIR = results

# Common source files
COMMON_SRCS = $(COMMON_DIR)/validation.cpp $(COMMON_DIR)/timing.cpp
COMMON_HDRS = $(INCLUDE_DIR)/golomb/golomb.hpp $(INCLUDE_DIR)/golomb/greedy.hpp $(INCLUDE_DIR)/golomb/bitset256.hpp

# Executables
V1 = golomb_v1
V2 = golomb_v2
V3 = golomb_v3
V4 = golomb_v4
V5 = golomb_v5

# Default target
all: v1 v2

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ===== MAIN TARGETS =====

# v1: Sequential (single-threaded, with optional AVX2)
v1: $(SRC_DIR)/v1_sequential.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(AVX_FLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V1) $< $(COMMON_SRCS)

# v1 without AVX2 (for older CPUs)
v1_noavx: $(SRC_DIR)/v1_sequential.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V1)_noavx $< $(COMMON_SRCS)

# v2: OpenMP (multi-threaded with AVX2)
v2: $(SRC_DIR)/v2_openmp.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(AVX_FLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V2) $< $(COMMON_SRCS)

# v2 without AVX2
v2_noavx: $(SRC_DIR)/v2_openmp.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V2)_noavx $< $(COMMON_SRCS)

# v3: Hybrid MPI+OpenMP (distributed + multi-threaded with AVX2)
v3: $(SRC_DIR)/v3_hybrid.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(AVX_FLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V3) $< $(COMMON_SRCS)

# v3 without AVX2
v3_noavx: $(SRC_DIR)/v3_hybrid.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(OPENMP_FLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V3)_noavx $< $(COMMON_SRCS)

# v4: Pure Hypercube MPI+OpenMP (decentralized, all ranks equal)
v4: $(SRC_DIR)/v4_hypercube.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(AVX_FLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V4) $< $(COMMON_SRCS)

# v4 without AVX2
v4_noavx: $(SRC_DIR)/v4_hypercube.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(OPENMP_FLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V4)_noavx $< $(COMMON_SRCS)

# v5: Pure MPI (no OpenMP, hypercube topology)
v5: $(SRC_DIR)/v5_pure_mpi.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(AVX_FLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V5) $< $(COMMON_SRCS)

# v5 without AVX2
v5_noavx: $(SRC_DIR)/v5_pure_mpi.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) -I$(INCLUDE_DIR) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V5)_noavx $< $(COMMON_SRCS)

# ===== BUILD ALL =====

sequential: v1
openmp: v2
hybrid: v3 v4
parallel: v3 v4 v5
hypercube: v4
mpi: v3 v4 v5
purempi: v5

# ===== TEST TARGETS =====

# Unit tests
TEST_DIR = tests
TEST_UNIT = test_correctness
TEST_OMP = test_openmp
TEST_MPI = test_mpi_unit

test_unit: $(TEST_DIR)/test_correctness.cpp $(COMMON_SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BASE) -O2 -I$(INCLUDE_DIR) -o $(BUILD_DIR)/$(TEST_UNIT) $< $(COMMON_SRCS)
	./$(BUILD_DIR)/$(TEST_UNIT)

test_openmp_unit: $(TEST_DIR)/test_openmp.cpp $(COMMON_SRCS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BASE) -O2 $(OPENMP_FLAGS) -I$(INCLUDE_DIR) -o $(BUILD_DIR)/$(TEST_OMP) $< $(COMMON_SRCS)
	OMP_NUM_THREADS=4 ./$(BUILD_DIR)/$(TEST_OMP)

test_mpi_unit: $(TEST_DIR)/test_mpi.cpp $(COMMON_SRCS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS_BASE) -O2 -I$(INCLUDE_DIR) -o $(BUILD_DIR)/$(TEST_MPI) $< $(COMMON_SRCS)
	mpirun --oversubscribe -np 4 ./$(BUILD_DIR)/$(TEST_MPI)

# Integration tests
test: v1 v2
	@echo "=== Testing v1 (Sequential) ==="
	./$(BUILD_DIR)/$(V1) 7
	@echo ""
	@echo "=== Testing v2 (OpenMP, 4 threads) ==="
	OMP_NUM_THREADS=4 ./$(BUILD_DIR)/$(V2) 7

test_mpi: v3
	@echo "=== Testing v3 (Hybrid MPI+OpenMP) ==="
	OMP_NUM_THREADS=2 mpirun --oversubscribe -np 2 ./$(BUILD_DIR)/$(V3) 8 --threads 2

test_v4: v4
	@echo "=== Testing v4 (Pure Hypercube MPI+OpenMP) ==="
	OMP_NUM_THREADS=2 mpirun --oversubscribe -np 4 ./$(BUILD_DIR)/$(V4) 8 --threads 2

test_v5: v5
	@echo "=== Testing v5 (Pure MPI, no OpenMP) ==="
	mpirun --oversubscribe -np 4 ./$(BUILD_DIR)/$(V5) 8

# Run all tests
test_all: test_unit test_openmp_unit test test_mpi test_v4 test_v5

# ===== BENCHMARK TARGETS =====

benchmark: v1 v2
	@echo "=== Sequential Benchmark ==="
	./$(BUILD_DIR)/$(V1) 10 --benchmark
	@echo ""
	@echo "=== OpenMP Benchmark (8 threads) ==="
	OMP_NUM_THREADS=8 ./$(BUILD_DIR)/$(V2) 11 --benchmark

benchmark_v1: v1
	@echo "=== v1 Sequential Benchmark ==="
	./$(BUILD_DIR)/$(V1) 11 --benchmark

benchmark_v2: v2
	@echo "=== v2 OpenMP Benchmark ==="
	@for t in 1 2 4 8 16 32; do \
		echo "--- $$t threads ---"; \
		OMP_NUM_THREADS=$$t ./$(BUILD_DIR)/$(V2) 12 2>&1 | grep -E "Time|Length"; \
	done

benchmark_v3: v3
	@echo "=== v3 Hybrid Benchmark ==="
	@for np in 2 4; do \
		for t in 4 8; do \
			echo "--- $$np ranks x $$t threads ---"; \
			OMP_NUM_THREADS=$$t mpirun --oversubscribe -np $$np ./$(BUILD_DIR)/$(V3) 12 --threads $$t 2>&1 | grep -E "Time|Length"; \
		done; \
	done

benchmark_v4: v4
	@echo "=== v4 Pure Hypercube Benchmark ==="
	@for np in 2 4 8; do \
		for t in 4 8; do \
			echo "--- $$np ranks x $$t threads (hypercube) ---"; \
			OMP_NUM_THREADS=$$t mpirun --oversubscribe -np $$np ./$(BUILD_DIR)/$(V4) 12 --threads $$t 2>&1 | grep -E "Time|Length"; \
		done; \
	done

benchmark_v5: v5
	@echo "=== v5 Pure MPI Benchmark (no OpenMP) ==="
	@for np in 2 4 8 16 32; do \
		echo "--- $$np MPI ranks ---"; \
		mpirun --oversubscribe -np $$np ./$(BUILD_DIR)/$(V5) 11 2>&1 | grep -E "Time|Length"; \
	done

# ===== CLEAN =====

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(RESULTS_DIR)/sequential/*.csv
	rm -f $(RESULTS_DIR)/parallel/*.csv

# Clean all CSV files (local + romeo)
clean-csv:
	rm -f $(RESULTS_DIR)/sequential/*.csv
	rm -f $(RESULTS_DIR)/parallel/*.csv
	rm -f $(RESULTS_DIR)/openmp/*.csv
	rm -f $(RESULTS_DIR)/romeo/*.csv
	rm -f $(RESULTS_DIR)/romeo/sequential/*.csv
	rm -f $(RESULTS_DIR)/romeo/parallel/*.csv
	rm -f $(RESULTS_DIR)/romeo/openmp/*.csv

# Clean all PNG plot files
clean-plots:
	rm -f $(RESULTS_DIR)/plots/*.png
	rm -f $(RESULTS_DIR)/romeo/plots/*.png
	rm -f $(RESULTS_DIR)/romeo/plots_fixed/*.png

# Clean all results (CSV + plots + logs)
clean-results: clean-csv clean-plots
	rm -f $(RESULTS_DIR)/romeo/*.out
	rm -f $(RESULTS_DIR)/romeo/*.err
	rm -f $(RESULTS_DIR)/romeo/logs/*

# ===== DOCUMENTATION =====

# Generate Doxygen documentation
docs:
	@echo "=== Generating Doxygen documentation ==="
	doxygen docs/Doxyfile
	@echo ""
	@echo "Documentation generated: docs/html/index.html"

# Clean documentation
clean-docs:
	rm -rf docs/html

# ===== HELP =====

help:
	@echo "Golomb Ruler Solver - Makefile"
	@echo ""
	@echo "Main targets:"
	@echo "  v1         - Sequential version (single-threaded + AVX2)"
	@echo "  v2         - OpenMP version (multi-threaded + AVX2)"
	@echo "  v3         - Hybrid MPI+OpenMP version (master/worker)"
	@echo "  v4         - Hypercube MPI+OpenMP (decentralized, O(log P) comm)"
	@echo "  v5         - Pure MPI (no OpenMP, hypercube, for MPI scaling)"
	@echo ""
	@echo "Variants without AVX2:"
	@echo "  v1_noavx   - Sequential without AVX2"
	@echo "  v2_noavx   - OpenMP without AVX2"
	@echo "  v3_noavx   - Hybrid without AVX2"
	@echo "  v4_noavx   - Hypercube without AVX2"
	@echo "  v5_noavx   - Pure MPI without AVX2"
	@echo ""
	@echo "Test targets:"
	@echo "  test_unit      - Run unit tests (test_correctness.cpp)"
	@echo "  test_openmp_unit - Run OpenMP unit tests"
	@echo "  test_mpi_unit  - Run MPI unit tests (requires mpirun)"
	@echo "  test           - Integration test of v1 and v2"
	@echo "  test_mpi       - Integration test of v3 (MPI master/worker)"
	@echo "  test_v4        - Integration test of v4 (MPI+OpenMP hypercube)"
	@echo "  test_v5        - Integration test of v5 (Pure MPI hypercube)"
	@echo "  test_all       - Run all tests"
	@echo ""
	@echo "Benchmark targets:"
	@echo "  benchmark      - Run benchmarks"
	@echo "  benchmark_v2   - Benchmark v2 with various thread counts"
	@echo "  benchmark_v3   - Benchmark v3 with various configurations"
	@echo "  benchmark_v4   - Benchmark v4 with various hypercube sizes"
	@echo "  benchmark_v5   - Benchmark v5 with various MPI process counts"
	@echo ""
	@echo "Romeo HPC targets:"
	@echo "  romeo-setup    - First-time setup: copy SSH key to Romeo"
	@echo "  romeo          - Full workflow: deploy + submit all benchmarks"
	@echo "  romeo-deploy   - Deploy code to Romeo and compile"
	@echo "  romeo-bench    - Submit all benchmarks (~200 jobs)"
	@echo "  romeo-bench-quick - Submit essential benchmarks (42 jobs)"
	@echo "  romeo-status   - Check job status on Romeo"
	@echo "  romeo-fetch    - Download results from Romeo"
	@echo "  romeo-wait     - Wait for jobs to complete, then fetch"
	@echo ""
	@echo "  Arguments (defaults: ROMEO_USER=nimarano, ROMEO_HOST=romeo1):"
	@echo "    make romeo-setup ROMEO_USER=dupont   # Setup for another user"
	@echo "    make romeo ROMEO_USER=dupont"
	@echo ""
	@echo "Documentation:"
	@echo "  docs           - Generate Doxygen HTML documentation"
	@echo "  clean-docs     - Remove generated documentation"
	@echo ""
	@echo "Clean targets:"
	@echo "  clean          - Remove build directory and local CSV files"
	@echo "  clean-csv      - Remove all CSV files (local + romeo)"
	@echo "  clean-plots    - Remove all PNG plot files"
	@echo "  clean-results  - Remove all results (CSV + plots + logs)"
	@echo ""
	@echo "Usage examples:"
	@echo "  make v1 && ./build/golomb_v1 12"
	@echo "  make v2 && OMP_NUM_THREADS=32 ./build/golomb_v2 12"
	@echo "  make v3 && OMP_NUM_THREADS=8 mpirun -np 4 ./build/golomb_v3 14 --threads 8"
	@echo "  make v4 && OMP_NUM_THREADS=8 mpirun -np 8 ./build/golomb_v4 14 --threads 8"
	@echo "  make v5 && mpirun -np 64 ./build/golomb_v5 13"
	@echo ""
	@echo "Romeo workflow:"
	@echo "  make romeo-setup                 # First time only (copies SSH key)"
	@echo "  make romeo && make romeo-wait    # Deploy, run, wait, fetch"

# ===== ROMEO HPC TARGETS =====
# Configuration via environment variables (override with: ROMEO_USER=prof make romeo)

ROMEO_HOST ?= romeo1
ROMEO_USER ?= nimarano
ROMEO_DIR ?= ~/golomb
ROMEO_FULL_HOST ?= $(ROMEO_HOST).univ-reims.fr

# First-time setup: generate SSH key if needed and copy to Romeo
romeo-setup:
	@echo "=== Romeo SSH Setup ==="
	@if [ ! -f ~/.ssh/id_rsa ] && [ ! -f ~/.ssh/id_ed25519 ]; then \
		echo "No SSH key found, generating one..."; \
		ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "$(ROMEO_USER)@romeo"; \
	fi
	@echo "Copying SSH key to Romeo (you'll need to enter your password once)..."
	@ssh-copy-id -o StrictHostKeyChecking=accept-new $(ROMEO_USER)@$(ROMEO_FULL_HOST) || \
		(echo "If ssh-copy-id failed, try: cat ~/.ssh/id_ed25519.pub | ssh $(ROMEO_USER)@$(ROMEO_FULL_HOST) 'mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys'")
	@echo ""
	@echo "Testing connection..."
	@ssh -o BatchMode=yes $(ROMEO_USER)@$(ROMEO_FULL_HOST) "echo 'SSH OK! Connected as $$(whoami) on $$(hostname)'"
	@echo ""
	@echo "Setup complete! You can now use: make romeo"

# Deploy code to Romeo and compile
romeo-deploy:
	@echo "=== Deploying to Romeo ==="
	@bash scripts/hpc/deploy.sh

# Submit all benchmarks (~200 jobs)
romeo-bench:
	@echo "=== Submitting all benchmark jobs ==="
	@ssh $(ROMEO_USER)@$(ROMEO_FULL_HOST) "cd $(ROMEO_DIR) && bash scripts/hpc/run_benchmarks.sh --submit"

# Submit only essential benchmarks (42 jobs - faster)
romeo-bench-quick:
	@echo "=== Submitting essential benchmark jobs ==="
	@ssh $(ROMEO_USER)@$(ROMEO_FULL_HOST) "cd $(ROMEO_DIR) && bash scripts/hpc/run_essential.sh"

# Check job status
romeo-status:
	@ssh $(ROMEO_USER)@$(ROMEO_FULL_HOST) "squeue -u $(ROMEO_USER) --format='%.10i %.20j %.8T %.10M %.6D %R'"

# Fetch results from Romeo
romeo-fetch:
	@echo "=== Fetching results ==="
	@mkdir -p results/romeo
	@rsync -avz --progress $(ROMEO_USER)@$(ROMEO_FULL_HOST):$(ROMEO_DIR)/results/romeo/ results/romeo/
	@echo "Results saved to results/romeo/"

# Wait for jobs and fetch results
romeo-wait:
	@bash scripts/hpc/wait_and_fetch.sh

# Full workflow: deploy + bench
romeo: romeo-deploy romeo-bench
	@echo ""
	@echo "Jobs submitted! Next steps:"
	@echo "  make romeo-status  - Check job progress"
	@echo "  make romeo-fetch   - Download results (when jobs complete)"
	@echo "  make romeo-wait    - Auto-wait and fetch"

.PHONY: all v1 v2 v3 v4 v5 v1_noavx v2_noavx v3_noavx v4_noavx v5_noavx
.PHONY: sequential openmp hybrid parallel hypercube mpi purempi
.PHONY: test test_unit test_openmp_unit test_mpi_unit test_mpi test_v4 test_v5 test_all
.PHONY: benchmark benchmark_v1 benchmark_v2 benchmark_v3 benchmark_v4 benchmark_v5
.PHONY: romeo romeo-setup romeo-deploy romeo-bench romeo-bench-quick romeo-status romeo-fetch romeo-wait
.PHONY: clean clean-csv clean-plots clean-results clean-docs docs help
