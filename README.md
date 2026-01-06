# Golomb Ruler Solver

A high-performance solver for finding optimal Golomb rulers, featuring 4 progressively parallelized versions: sequential, OpenMP, hybrid MPI+OpenMP, and pure hypercube MPI+OpenMP.

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

## Overview

A **Golomb ruler** of order *n* is a set of *n* integers (marks) such that all pairwise differences are distinct. Finding the shortest (optimal) Golomb ruler for a given order is an NP-hard problem.

This project implements 4 versions demonstrating progressive parallelization techniques, from single-threaded to distributed memory parallel computing.

### Known Optimal Lengths

| Order | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|-------|---|---|---|---|---|---|----|----|-----|
| Length | 6 | 11 | 17 | 25 | 34 | 44 | 55 | 72 | 85 |

## Quick Start

```bash
# Build all versions
make all

# Run sequential solver
./build/golomb_v1 10

# Run OpenMP solver (8 threads)
OMP_NUM_THREADS=8 ./build/golomb_v2 11

# Run hybrid MPI+OpenMP solver
OMP_NUM_THREADS=4 mpirun -np 4 ./build/golomb_v3 12

# Run hypercube MPI+OpenMP solver
OMP_NUM_THREADS=4 mpirun -np 8 ./build/golomb_v4 12

# Run tests
make test
```

## Version Comparison

| Version | Type | Parallelism | Description |
|---------|------|-------------|-------------|
| v1 | Sequential | Single-threaded | Optimized branch & bound with AVX2 |
| v2 | OpenMP | Multi-threaded | Task parallelism + AVX2 |
| v3 | MPI+OpenMP | Distributed | Master/worker with load balancing |
| v4 | MPI+OpenMP | Distributed | Pure hypercube topology (O(log P) comm) |

## Project Structure

```
golomb/
├── LICENSE                  # MIT License
├── README.md
├── CHANGELOG.md             # Version history
├── Makefile
│
├── include/golomb/          # Public headers
│   ├── golomb.hpp           # Core data structures
│   ├── bitset256.hpp        # AVX2-optimized bitset
│   └── greedy.hpp           # Greedy heuristic
│
├── src/                     # Source files
│   ├── v1_sequential.cpp    # Sequential version
│   ├── v2_openmp.cpp        # OpenMP version
│   ├── v3_hybrid.cpp        # MPI+OpenMP (master/worker)
│   ├── v4_hypercube.cpp     # MPI+OpenMP (hypercube)
│   └── common/              # Shared implementation
│       ├── validation.cpp
│       └── timing.cpp
│
├── tests/                   # Test suite
│   └── test_correctness.cpp
│
├── tools/                   # Analysis tools
│   └── visualization/       # Plot generation scripts
│
├── scripts/                 # Utility scripts
│   └── hpc/                 # HPC deployment scripts
│
├── jobs/                    # SLURM job templates
│
├── docs/                    # Documentation
│   ├── ARCHITECTURE.md      # System design
│   ├── ALGORITHMS.md        # Algorithm details
│   └── HPC_GUIDE.md         # Cluster deployment
│
└── examples/                # Usage examples
    ├── basic_usage.sh
    └── mpi_cluster.sh
```

## Building

### Prerequisites

- C++17 compatible compiler (g++ >= 7.0)
- MPI implementation (OpenMPI or MPICH) for v3/v4
- Python 3 + matplotlib for visualization

### Build Commands

```bash
make all          # Build v1 and v2
make parallel     # Build v3 and v4
make v1           # Build v1 only
make v2           # Build v2 only
make v3           # Build v3 only
make v4           # Build v4 only
make clean        # Clean build artifacts
```

## Usage

See `examples/` directory for complete examples, or `docs/HPC_GUIDE.md` for cluster deployment.

### v1: Sequential
```bash
./build/golomb_v1 <order> [--benchmark] [--csv FILE]
```

### v2: OpenMP
```bash
OMP_NUM_THREADS=8 ./build/golomb_v2 <order> [--threads N] [--benchmark]
```

### v3: Hybrid MPI+OpenMP (Master/Worker)
```bash
OMP_NUM_THREADS=4 mpirun -np 4 ./build/golomb_v3 <order> --threads 4
```

### v4: Pure Hypercube MPI+OpenMP
```bash
OMP_NUM_THREADS=4 mpirun -np 8 ./build/golomb_v4 <order> --threads 4
```

## Testing

```bash
make test         # Test v1 and v2
make test_mpi     # Test v3
make test_v4      # Test v4
```

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design and components
- [docs/ALGORITHMS.md](docs/ALGORITHMS.md) - Algorithm details and optimizations
- [docs/HPC_GUIDE.md](docs/HPC_GUIDE.md) - HPC cluster deployment guide

## Key Optimizations

1. **Bitset Difference Tracking** - O(1) lookup instead of O(n)
2. **Symmetry Breaking** - Halves the search space
3. **Greedy Initial Bound** - Better pruning from the start
4. **AVX2 SIMD** - Vectorized difference computation
5. **Cache Alignment** - 64-byte aligned structures
6. **Hypercube Communication** - O(log P) bound propagation

## License

MIT License - Copyright (c) 2025 Nicolas Marano

See [LICENSE](LICENSE) for details.
