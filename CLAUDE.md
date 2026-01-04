# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Golomb Ruler Solver - A parallel computing project for finding optimal Golomb rulers (sets of n integer positions where all pairwise differences are unique). Implements 6 sequential versions and 4 MPI parallel versions (including hybrid MPI+OpenMP) demonstrating progressive algorithmic optimization.

## Build Commands

```bash
# Build all sequential versions (v1-v6)
make sequential

# Build all parallel (MPI) versions
make parallel

# Build individual versions
make v1          # Brute force
make v2          # Backtracking
make v3          # Branch & Bound
make v4          # Optimized (bitset + symmetry)
make v5          # Final sequential
make v6          # Hardware optimized (OpenMP + AVX2)
make v6_noavx    # v6 without AVX2 (older CPUs)

make mpi_v1      # Basic master/worker
make mpi_v2      # Hypercube topology
make mpi_v3      # Dynamic load balancing
make mpi_v4      # Hybrid MPI+OpenMP

# Clean builds
make clean
```

## Running

```bash
# Sequential (order is required argument)
./build/golomb_v5 <order> [--benchmark] [--csv]

# Hardware-optimized v6
./build/golomb_v6 <order> [--threads N] [--no-simd] [--benchmark] [--verbose]
OMP_NUM_THREADS=8 ./build/golomb_v6 11

# Parallel (requires MPI)
mpirun -np <processes> ./build/golomb_mpi_v3 <order> [--depth N] [--csv FILE]

# Hybrid MPI+OpenMP (requires MPI + OpenMP)
OMP_NUM_THREADS=8 mpirun -np 4 ./build/golomb_mpi_v4 <order> --threads 8 [--depth N]
```

## Testing and Benchmarking

```bash
make test              # Run correctness validation (58 tests)
make benchmark_seq     # Benchmark sequential versions
make benchmark_par     # Benchmark parallel versions
make benchmark_v6      # Benchmark v6 with different thread counts
make benchmark_hybrid  # Benchmark hybrid MPI+OpenMP version

# Generate CSV data for analysis
bash scripts/generate_csv.sh           # Sequential benchmarks
bash scripts/generate_parallel_csv.sh  # Parallel benchmarks
```

## Architecture

### Core Components (`src/common/`)
- **golomb.hpp**: `GolombRuler` struct, `SearchStats` for tracking, `Timer` class, known optimal lengths for G2-G14, `MAX_LENGTH=256` constant
- **greedy.hpp**: Template-based greedy heuristic `computeGreedySolution()` compatible with both `std::bitset` and custom `BitSet256`
- **validation.cpp**: Fast validation using bitset for O(1) difference lookup
- **timing.cpp**: CSV export with `writeResultCSV()`, `printStats()`

### Sequential Versions (`src/sequential/`)
Progressive optimization demonstrating algorithm evolution:
1. **v1_bruteforce**: Baseline - enumerates all combinations
2. **v2_backtracking**: Incremental construction with early termination (~6x faster)
3. **v3_branch_bound**: Greedy upper bound pruning (~2,500x faster than v1)
4. **v4_optimized**: Bitset + symmetry breaking (~28,000x faster than v1)
5. **v5_final_seq**: Production version with CLI options
6. **v6_hardware**: OpenMP task parallelism + AVX2 SIMD (~320,000x faster than v1)

### Parallel Versions (`src/parallel/`)
- **v1_basic_mpi**: Master/worker with static round-robin distribution
- **v2_hypercube**: Dynamic bound sharing via hypercube topology
- **v3_optimized_mpi**: Adaptive depth + work stealing + hypercube reduce
- **v4_hybrid_mpi_omp**: Hybrid MPI+OpenMP combining inter-node MPI with intra-node OpenMP tasks

### Key Implementation Details
- Bitset-based difference tracking replaces set for O(1) lookups
- Symmetry breaking constraint: `marks[1] <= best_length/2` halves search space
- MPI tags: `TAG_WORK=1`, `TAG_RESULT=2`, `TAG_DONE=3`, `TAG_BOUND=4`, `TAG_REQUEST_WORK=5`
- Prefix depth controls work granularity in parallel versions (auto-selected based on order)
- v6 uses `alignas(64)` cache-aligned structures, `BitSet256` for AVX2 compatibility, and `std::atomic` for thread-safe bound updates
- v4_hybrid uses MPI_THREAD_FUNNELED for thread-safe MPI calls, OpenMP tasks for subtree parallelism, and combines the optimizations of v3 (MPI) with v6 (OpenMP+AVX2)

## Results

Results are stored in `results/sequential/` and `results/parallel/` as CSV files. Visualization scripts in `src/visualization/generate_plots.py`.
