# Golomb Ruler Solver - Makefile
# Sequential and Parallel versions

CXX = g++
MPICXX = mpicxx

# Base flags
CXXFLAGS_BASE = -std=c++17 -Wall -Wextra

# Performance optimization flags
# -O3: Aggressive optimization
# -flto: Link-Time Optimization (cross-module inlining)
# -march=native: Use CPU-specific instructions
# -funroll-loops: Unroll small loops for better pipelining
OPT_FLAGS = -O3 -flto -march=native -funroll-loops

# Production flags (optimized)
CXXFLAGS = $(CXXFLAGS_BASE) $(OPT_FLAGS)

# Debug flags
DEBUG_FLAGS = -g -O0 -DDEBUG -std=c++17 -Wall -Wextra

# Directories
SRC_DIR = src
COMMON_DIR = $(SRC_DIR)/common
SEQ_DIR = $(SRC_DIR)/sequential
PAR_DIR = $(SRC_DIR)/parallel
BUILD_DIR = build
RESULTS_DIR = results

# Common source files
COMMON_SRCS = $(COMMON_DIR)/validation.cpp $(COMMON_DIR)/timing.cpp
COMMON_HDRS = $(COMMON_DIR)/golomb.hpp $(COMMON_DIR)/greedy.hpp

# Sequential executables
SEQ_V1 = golomb_v1
SEQ_V2 = golomb_v2
SEQ_V3 = golomb_v3
SEQ_V4 = golomb_v4
SEQ_V5 = golomb_v5
SEQ_V6 = golomb_v6

# v6 Hardware optimization flags
OPENMP_FLAGS = -fopenmp
AVX_FLAGS = -mavx2 -mfma
V6_FLAGS = $(OPENMP_FLAGS) $(AVX_FLAGS) -march=native -DUSE_AVX2

# Parallel executables
PAR_V1 = golomb_mpi_v1
PAR_V2 = golomb_mpi_v2
PAR_V3 = golomb_mpi_v3
PAR_V4 = golomb_mpi_v4
PAR_V5 = golomb_mpi_v5

# Hybrid MPI+OpenMP flags (combines MPI with OpenMP threading)
HYBRID_FLAGS = $(OPENMP_FLAGS) $(AVX_FLAGS) -march=native -DUSE_AVX2

# Default target
all: sequential

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# ===== SEQUENTIAL TARGETS =====

sequential: $(SEQ_V1) $(SEQ_V2) $(SEQ_V3) $(SEQ_V4) $(SEQ_V5) $(SEQ_V6)

$(SEQ_V1): $(SEQ_DIR)/v1_bruteforce.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/$@ $< $(COMMON_SRCS)

$(SEQ_V2): $(SEQ_DIR)/v2_backtracking.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/$@ $< $(COMMON_SRCS)

$(SEQ_V3): $(SEQ_DIR)/v3_branch_bound.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/$@ $< $(COMMON_SRCS)

$(SEQ_V4): $(SEQ_DIR)/v4_optimized.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/$@ $< $(COMMON_SRCS)

$(SEQ_V5): $(SEQ_DIR)/v5_final_seq.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/$@ $< $(COMMON_SRCS)

$(SEQ_V6): $(SEQ_DIR)/v6_hardware.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(V6_FLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/$@ $< $(COMMON_SRCS)

# v6 without AVX2 (fallback for older CPUs)
v6_noavx: $(SEQ_DIR)/v6_hardware.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OPENMP_FLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/golomb_v6_noavx $< $(COMMON_SRCS)

# Individual sequential targets
v1: $(SEQ_V1)
v2: $(SEQ_V2)
v3: $(SEQ_V3)
v4: $(SEQ_V4)
v5: $(SEQ_V5)
v6: $(SEQ_V6)

# ===== PARALLEL TARGETS =====

parallel: $(PAR_V1) $(PAR_V2) $(PAR_V3) $(PAR_V4) $(PAR_V5)

$(PAR_V1): $(PAR_DIR)/v1_basic_mpi.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/$@ $< $(COMMON_SRCS)

$(PAR_V2): $(PAR_DIR)/v2_hypercube.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/$@ $< $(COMMON_SRCS)

$(PAR_V3): $(PAR_DIR)/v3_optimized_mpi.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/$@ $< $(COMMON_SRCS)

# Hybrid MPI+OpenMP version
$(PAR_V4): $(PAR_DIR)/v4_hybrid_mpi_omp.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) $(HYBRID_FLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/$@ $< $(COMMON_SRCS)

# Work Stealing version
$(PAR_V5): $(PAR_DIR)/v5_work_stealing.cpp $(COMMON_SRCS) $(COMMON_HDRS) $(COMMON_DIR)/grasp.hpp | $(BUILD_DIR)
	$(MPICXX) $(CXXFLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/$@ $< $(COMMON_SRCS)

# Individual parallel targets
mpi_v1: $(PAR_V1)
mpi_v2: $(PAR_V2)
mpi_v3: $(PAR_V3)
mpi_v4: $(PAR_V4)
mpi_v5: $(PAR_V5)

# ===== TEST TARGETS =====

test: test_validation

test_validation: tests/test_correctness.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/test_correctness $< $(COMMON_SRCS)
	./$(BUILD_DIR)/test_correctness

# ===== BENCHMARK TARGETS =====

benchmark_seq: sequential
	@echo "Running sequential benchmarks..."
	@for v in 1 2 3 4 5; do \
		echo "=== Version $$v ==="; \
		for n in 4 5 6 7 8; do \
			timeout 300 ./$(BUILD_DIR)/golomb_v$$v $$n || echo "Timeout for G$$n"; \
		done; \
	done

benchmark_par: parallel
	@echo "Running parallel benchmarks..."
	@for np in 2 4 8; do \
		echo "=== $$np processes ==="; \
		for n in 7 8 9; do \
			timeout 600 mpirun --oversubscribe -np $$np ./$(BUILD_DIR)/golomb_mpi_v3 $$n || echo "Timeout"; \
		done; \
	done

benchmark_v6: v6
	@echo "Running v6 hardware-optimized benchmarks..."
	@echo ""
	@for t in 1 2 4 8; do \
		echo "=== $$t threads ==="; \
		OMP_NUM_THREADS=$$t ./$(BUILD_DIR)/golomb_v6 10 --benchmark || echo "Error"; \
		echo ""; \
	done

benchmark_hybrid: mpi_v4
	@echo "Running hybrid MPI+OpenMP benchmarks..."
	@echo ""
	@for np in 2 4; do \
		for t in 2 4; do \
			echo "=== $$np MPI ranks x $$t OpenMP threads ==="; \
			OMP_NUM_THREADS=$$t mpirun --oversubscribe -np $$np ./$(BUILD_DIR)/golomb_mpi_v4 10 --threads $$t || echo "Error"; \
			echo ""; \
		done; \
	done

# ===== DEBUG TARGETS =====

debug_v1: $(SEQ_DIR)/v1_bruteforce.cpp $(COMMON_SRCS) $(COMMON_HDRS) | $(BUILD_DIR)
	$(CXX) $(DEBUG_FLAGS) -I$(COMMON_DIR) -o $(BUILD_DIR)/golomb_v1_debug $< $(COMMON_SRCS)

# ===== CLEAN =====

clean:
	rm -rf $(BUILD_DIR)
	rm -f $(RESULTS_DIR)/sequential/*.csv
	rm -f $(RESULTS_DIR)/parallel/*.csv

clean_results:
	@bash scripts/clean_results.sh --all --force

# ===== HELP =====

help:
	@echo "Golomb Ruler Solver - Available targets:"
	@echo ""
	@echo "  all          - Build all sequential versions (default)"
	@echo "  sequential   - Build all sequential versions (v1-v6)"
	@echo "  parallel     - Build all parallel MPI versions"
	@echo ""
	@echo "  v1, v2, v3, v4, v5  - Build individual sequential version"
	@echo "  v6           - Build hardware-optimized version (OpenMP + AVX2)"
	@echo "  v6_noavx     - Build v6 without AVX2 (for older CPUs)"
	@echo "  mpi_v1, mpi_v2, mpi_v3 - Build individual parallel version"
	@echo "  mpi_v4       - Build hybrid MPI+OpenMP version"
	@echo ""
	@echo "  test         - Run correctness tests"
	@echo "  benchmark_seq - Run sequential benchmarks"
	@echo "  benchmark_par - Run parallel benchmarks"
	@echo "  benchmark_v6  - Run v6 multi-threaded benchmarks"
	@echo "  benchmark_hybrid - Run hybrid MPI+OpenMP benchmarks"
	@echo ""
	@echo "  clean        - Remove all built files"
	@echo "  clean_results - Remove result CSV and plot files"
	@echo ""
	@echo "Usage examples:"
	@echo "  make v6 && ./build/golomb_v6 10 --threads 8"
	@echo "  make mpi_v2 && mpirun -np 4 ./build/golomb_mpi_v2 9"
	@echo "  OMP_NUM_THREADS=16 ./build/golomb_v6 11 --benchmark"
	@echo "  make mpi_v4 && OMP_NUM_THREADS=4 mpirun -np 2 ./build/golomb_mpi_v4 11 --threads 4"

.PHONY: all sequential parallel clean clean_results help test
.PHONY: v1 v2 v3 v4 v5 v6 v6_noavx mpi_v1 mpi_v2 mpi_v3 mpi_v4 mpi_v5
.PHONY: benchmark_seq benchmark_par benchmark_v6 benchmark_hybrid debug_v1 test_validation
