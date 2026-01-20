# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Golomb Ruler Solver - A parallel computing project for finding optimal Golomb rulers. Implements 4 versions: sequential, OpenMP, hybrid MPI+OpenMP (master/worker), and pure hypercube MPI+OpenMP.

## Build Commands

```bash
# Build all versions
make all          # v1 and v2
make parallel     # v3 and v4

# Build individual versions
make v1           # Sequential (single-threaded + AVX2)
make v2           # OpenMP (multi-threaded + AVX2)
make v3           # Hybrid MPI+OpenMP (master/worker)
make v4           # Pure Hypercube MPI+OpenMP

# Variants without AVX2
make v1_noavx
make v2_noavx
make v3_noavx
make v4_noavx

# Clean
make clean
```

## Running

```bash
# v1: Sequential
./build/golomb_v1 <order> [--benchmark] [--csv FILE]

# v2: OpenMP
OMP_NUM_THREADS=8 ./build/golomb_v2 <order> [--threads N] [--benchmark]

# v3: Hybrid MPI+OpenMP (master/worker)
OMP_NUM_THREADS=4 mpirun -np 4 ./build/golomb_v3 <order> --threads 4

# v4: Pure Hypercube MPI+OpenMP
OMP_NUM_THREADS=4 mpirun -np 8 ./build/golomb_v4 <order> --threads 4
```

## Testing

```bash
make test          # Test v1 and v2
make test_mpi      # Test v3
make test_v4       # Test v4
```

## Project Structure

```
include/golomb/       # Public headers
├── golomb.hpp        # Core data structures
├── bitset256.hpp     # AVX2-optimized bitset
└── greedy.hpp        # Greedy heuristic

src/
├── common/           # Shared implementation
├── v1_sequential.cpp
├── v2_openmp.cpp
├── v3_hybrid.cpp
└── v4_hypercube.cpp

tests/                # Test suite
tools/visualization/  # Plot generation
scripts/hpc/          # HPC deployment
docs/                 # Documentation
examples/             # Usage examples
```

## Key Components

### Data Structures
- **GolombRuler**: Mark positions, length, order
- **SearchState**: Cache-aligned (64 bytes) search state
- **BitSet256**: AVX2-optimized 256-bit bitset

### Parallelization
- **v2**: OpenMP tasks at shallow depths
- **v3**: Master distributes work, workers explore subtrees
- **v4**: Decentralized, hypercube bound sharing O(log P)

### MPI Tags
- `TAG_WORK=1`: Work distribution
- `TAG_RESULT=2`: Result reporting
- `TAG_BOUND=4`: Bound propagation
