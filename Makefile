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
COMMON_DIR = $(SRC_DIR)/common
BUILD_DIR = build
RESULTS_DIR = results

# Common source files
COMMON_SRCS = $(COMMON_DIR)/validation.cpp $(COMMON_DIR)/timing.cpp
COMMON_HDRS = $(COMMON_DIR)/golomb.hpp $(COMMON_DIR)/greedy.hpp $(COMMON_DIR)/bitset256.hpp

# Executables
V1 = golomb_v1
V2 = golomb_v2
V3 = golomb_v3
V4 = golomb_v4

# Default target
all: v1 v2

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ===== MAIN TARGETS =====

# v1: Sequential (single-threaded, with optional AVX2)
v1: $(SRC_DIR)/v1_sequential.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(AVX_FLAGS) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V1) $< $(COMMON_SRCS)

# v1 without AVX2 (for older CPUs)
v1_noavx: $(SRC_DIR)/v1_sequential.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V1)_noavx $< $(COMMON_SRCS)

# v2: OpenMP (multi-threaded with AVX2)
v2: $(SRC_DIR)/v2_openmp.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(AVX_FLAGS) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V2) $< $(COMMON_SRCS)

# v2 without AVX2
v2_noavx: $(SRC_DIR)/v2_openmp.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V2)_noavx $< $(COMMON_SRCS)

# v3: Hybrid MPI+OpenMP (distributed + multi-threaded with AVX2)
v3: $(SRC_DIR)/v3_hybrid.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(AVX_FLAGS) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V3) $< $(COMMON_SRCS)

# v3 without AVX2
v3_noavx: $(SRC_DIR)/v3_hybrid.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(OPENMP_FLAGS) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V3)_noavx $< $(COMMON_SRCS)

# v4: Pure Hypercube MPI+OpenMP (decentralized, all ranks equal)
v4: $(SRC_DIR)/v4_hypercube.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(OPENMP_FLAGS) $(AVX_FLAGS) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V4) $< $(COMMON_SRCS)

# v4 without AVX2
v4_noavx: $(SRC_DIR)/v4_hypercube.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(OPENMP_FLAGS) -I$(SRC_DIR) -o $(BUILD_DIR)/$(V4)_noavx $< $(COMMON_SRCS)

# ===== BUILD ALL =====

sequential: v1
openmp: v2
hybrid: v3 v4
parallel: v3 v4
hypercube: v4

# ===== TEST TARGETS =====

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

# ===== CLEAN =====

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(RESULTS_DIR)/sequential/*.csv
	rm -f $(RESULTS_DIR)/parallel/*.csv

# ===== HELP =====

help:
	@echo "Golomb Ruler Solver - Makefile"
	@echo ""
	@echo "Main targets:"
	@echo "  v1         - Sequential version (single-threaded + AVX2)"
	@echo "  v2         - OpenMP version (multi-threaded + AVX2)"
	@echo "  v3         - Hybrid MPI+OpenMP version (master/worker)"
	@echo "  v4         - Pure Hypercube MPI+OpenMP (all ranks equal, O(log P) comm)"
	@echo ""
	@echo "Variants without AVX2:"
	@echo "  v1_noavx   - Sequential without AVX2"
	@echo "  v2_noavx   - OpenMP without AVX2"
	@echo "  v3_noavx   - Hybrid without AVX2"
	@echo "  v4_noavx   - Hypercube without AVX2"
	@echo ""
	@echo "Test and benchmark:"
	@echo "  test       - Quick test of v1 and v2"
	@echo "  test_mpi   - Quick test of v3 (MPI master/worker)"
	@echo "  test_v4    - Quick test of v4 (MPI hypercube)"
	@echo "  benchmark  - Run benchmarks"
	@echo "  benchmark_v2 - Benchmark v2 with various thread counts"
	@echo "  benchmark_v3 - Benchmark v3 with various configurations"
	@echo "  benchmark_v4 - Benchmark v4 with various hypercube sizes"
	@echo ""
	@echo "  clean      - Remove all built files"
	@echo ""
	@echo "Usage examples:"
	@echo "  make v1 && ./build/golomb_v1 12"
	@echo "  make v2 && OMP_NUM_THREADS=32 ./build/golomb_v2 12"
	@echo "  make v3 && OMP_NUM_THREADS=8 mpirun -np 4 ./build/golomb_v3 14 --threads 8"
	@echo "  make v4 && OMP_NUM_THREADS=8 mpirun -np 8 ./build/golomb_v4 14 --threads 8"

.PHONY: all v1 v2 v3 v4 v1_noavx v2_noavx v3_noavx v4_noavx
.PHONY: sequential openmp hybrid parallel hypercube
.PHONY: test test_mpi test_v4 benchmark benchmark_v1 benchmark_v2 benchmark_v3 benchmark_v4
.PHONY: clean help
