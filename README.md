# Golomb Ruler Solver

A high-performance solver for finding optimal Golomb rulers, featuring 6 progressively optimized sequential versions (including hardware-optimized v6 with OpenMP/AVX2) and 3 MPI parallel implementations with hypercube topology communication.

```
         Golomb Ruler of Order 4
    ────────────────────────────────
    0        1        3           6
    │        │        │           │
    ●────────●────────●───────────●
         1        2         3
              ┌───────────┘
              │    5
              └────────────────┘
    All differences are unique: {1, 2, 3, 5, 6}
```

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Performance Results](#performance-results)
- [Architecture](#architecture)
- [Building](#building)
- [Usage](#usage)
- [Testing](#testing)
- [Documentation](#documentation)

## Overview

A **Golomb ruler** of order *n* is a set of *n* integers (marks) such that all pairwise differences are distinct. Finding the shortest (optimal) Golomb ruler for a given order is an NP-hard problem.

This project implements multiple algorithmic approaches, from brute force to highly optimized branch-and-bound with parallelization, demonstrating progressive optimization techniques and parallel computing concepts.

### Known Optimal Lengths

| Order | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|-------|---|---|---|---|---|---|----|----|-----|
| Length | 6 | 11 | 17 | 25 | 34 | 44 | 55 | 72 | 85 |

## Features

### Sequential Versions (Progressive Optimization)

| Version | Algorithm | Key Optimization | Speedup vs v1 |
|---------|-----------|------------------|---------------|
| v1 | Brute Force | Baseline | 1x |
| v2 | Backtracking | Early termination | ~6x |
| v3 | Branch & Bound | Upper bound pruning | ~2,500x |
| v4 | Optimized B&B | Bitset + Symmetry breaking | ~28,000x |
| v5 | Production | CLI options + CSV export | ~28,000x |
| v6 | Hardware Optimized | OpenMP + AVX2 + Cache alignment | ~320,000x |

### Parallel Versions (MPI)

| Version | Architecture | Communication |
|---------|--------------|---------------|
| MPI v1 | Master/Worker | Static round-robin |
| MPI v2 | Hypercube | Dynamic bound sharing |
| MPI v3 | Optimized | Adaptive depth + work stealing |

## Quick Start

```bash
# Build everything
make all

# Run sequential solver
./build/golomb_v5 8

# Run parallel solver (requires MPI)
mpirun -np 4 ./build/golomb_mpi_v3 9

# Run tests
make test

# Generate benchmarks
bash scripts/generate_csv.sh
bash scripts/generate_parallel_csv.sh
```

## Performance Results

### Sequential Performance (G4-G11)

| Order | v1 (ms) | v2 (ms) | v3 (ms) | v4 (ms) | v5 (ms) | v6 (ms)* |
|-------|---------|---------|---------|---------|---------|----------|
| G4 | 0.04 | 0.03 | 0.00 | 0.00 | 0.00 | 90 |
| G5 | 1.92 | 0.75 | 0.01 | 0.01 | 0.01 | 51 |
| G6 | 96 | 21 | 0.03 | 0.02 | 0.03 | 31 |
| G7 | 6,206 | 865 | 0.37 | 0.22 | 0.22 | 17 |
| G8 | - | 25,454 | 5.6 | 2.4 | 2.4 | 23 |
| G9 | - | - | 74 | 21 | 25 | 20 |
| G10 | - | - | - | 188 | 182 | **38** |
| G11 | - | - | - | 3,274 | 3,596 | **312** |

*v6 with 20 threads (OpenMP). For small orders, thread overhead dominates.

### v6 Hardware Optimized Scaling (G11)

| Threads | Time (ms) | Speedup vs 1T |
|---------|-----------|---------------|
| 1 | 3,612 | 1.0x |
| 4 | 881 | 4.1x |
| 8 | 569 | 6.3x |
| 20 | 312 | 11.6x |

### Parallel Scalability (MPI v3)

| Order | 1 proc | 2 procs | 4 procs | 8 procs |
|-------|--------|---------|---------|---------|
| G9 | 21.6 ms | 48.5 ms | 25.0 ms | 38.5 ms |
| G10 | 205 ms | 244 ms | 136 ms | 93.6 ms |

*Note: MPI overhead is significant for small problems. Best speedup achieved for G10+ on cluster.*

## Architecture

```
golomb/
├── src/
│   ├── sequential/           # 6 sequential versions
│   │   ├── v1_bruteforce.cpp
│   │   ├── v2_backtracking.cpp
│   │   ├── v3_branch_bound.cpp
│   │   ├── v4_optimized.cpp
│   │   ├── v5_final_seq.cpp
│   │   └── v6_hardware.cpp   # OpenMP + AVX2 optimized
│   ├── parallel/             # 3 MPI versions
│   │   ├── v1_basic_mpi.cpp
│   │   ├── v2_hypercube.cpp
│   │   └── v3_optimized_mpi.cpp
│   ├── common/               # Shared code
│   │   ├── golomb.hpp        # Data structures
│   │   ├── validation.cpp    # Ruler validation
│   │   └── timing.cpp        # Performance timing
│   └── visualization/
│       └── generate_plots.py # Performance graphs
├── tests/
│   └── test_correctness.cpp  # Unit tests (58 tests)
├── scripts/
│   ├── generate_csv.sh       # Sequential benchmarks
│   └── generate_parallel_csv.sh # Parallel benchmarks
├── results/
│   ├── sequential/           # CSV data
│   ├── parallel/
│   └── plots/                # Generated graphs
├── docs/
│   └── journal.md            # Development journal
└── Makefile
```

### Key Optimizations

1. **Bitset for Differences** - O(1) lookup instead of O(log n) with std::set
2. **Symmetry Breaking** - Constraint `marks[1] <= best/2` halves search space
3. **Greedy Initial Bound** - Better pruning from the start
4. **Hypercube Communication** - Efficient bound sharing in O(log p) steps
5. **OpenMP Task Parallelism** (v6) - Task-based parallel search at shallow depths
6. **AVX2 SIMD** (v6) - Vectorized difference computation (8 diffs per instruction)
7. **Cache Alignment** (v6) - 64-byte aligned structures, explicit prefetching

## Building

### Prerequisites

- C++17 compatible compiler (g++ >= 7.0)
- MPI implementation (OpenMPI or MPICH) for parallel versions
- Python 3 + matplotlib/pandas for visualization

### Build Commands

```bash
make all          # Build everything
make sequential   # Build sequential versions only
make parallel     # Build parallel versions only
make v5           # Build specific version
make clean        # Clean build artifacts
```

## Usage

### Sequential Solver

```bash
# Basic usage
./build/golomb_v5 <order>

# With options
./build/golomb_v5 10 --benchmark    # Benchmark mode
./build/golomb_v5 9 --csv           # Export to CSV
./build/golomb_v5 8 --verbose       # Detailed output
```

### Hardware Optimized Solver (v6)

```bash
# Automatic thread detection
./build/golomb_v6 11

# Specify thread count
./build/golomb_v6 11 --threads 8

# Disable SIMD (for compatibility)
./build/golomb_v6 10 --no-simd

# Benchmark with custom threads
OMP_NUM_THREADS=4 ./build/golomb_v6 11 --benchmark
```

### Parallel Solver

```bash
# Local execution
mpirun -np 4 ./build/golomb_mpi_v3 10

# Cluster execution (SLURM)
sbatch scripts/run_parallel.slurm
```

## Testing

```bash
# Run all tests (58 unit tests)
make test

# Manual test run
./build/test_correctness
```

### Test Coverage

- Validation logic (valid/invalid rulers)
- Optimal length verification (G4-G11)
- GolombRuler struct operations
- Edge cases (empty, single mark, shifted)
- Timer functionality

## Documentation

- **[docs/journal.md](docs/journal.md)** - Development journal with detailed analysis
- **[PROJET_GOLOMB_SPEC.md](PROJET_GOLOMB_SPEC.md)** - Project specifications
- **[CLAUDE.md](CLAUDE.md)** - AI assistant configuration

## Visualization

Generate performance plots:

```bash
python3 src/visualization/generate_plots.py
```

Generated plots in `results/plots/`:
- `sequential_times.png` - Execution time comparison
- `nodes_explored.png` - Search space reduction
- `speedup_vs_v1.png` - Optimization speedup
- `parallel_speedup.png` - MPI scalability
- `parallel_efficiency.png` - Parallel efficiency

## Algorithm Details

### Branch and Bound Pruning

```
If current_length + min_remaining >= best_length:
    PRUNE (cannot improve)
```

### Hypercube Broadcast

```
For p = 2^d processes:
- Neighbor i of process r = r XOR 2^i
- Broadcast reaches all in d = log2(p) steps
```

## References

- [OEIS A003022](https://oeis.org/A003022) - Optimal Golomb ruler lengths
- Shearer, J.B. "Some new optimum Golomb rulers"
- MPI Standard 3.1 - Message Passing Interface

## License

This project was developed as an educational exercise for parallel computing.

---

*Generated with progressive optimization from brute force to parallel MPI implementation.*
