# Golomb Ruler Solver

<div align="center">

**High-Performance Computing Implementation for Finding Optimal Golomb Rulers**

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![MPI](https://img.shields.io/badge/MPI-OpenMPI%20%7C%20MPICH-green.svg)](https://www.open-mpi.org/)
[![OpenMP](https://img.shields.io/badge/OpenMP-4.5%2B-orange.svg)](https://www.openmp.org/)
[![AVX2](https://img.shields.io/badge/SIMD-AVX2-red.svg)](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)

[Quick Start](#quick-start) | [Installation](#installation) | [Usage](#usage) | [Documentation](#documentation) | [HPC Deployment](#hpc-deployment)

</div>

---

## What is a Golomb Ruler?

A **Golomb ruler** of order *n* is a set of *n* integers (marks) such that all pairwise differences are distinct. The challenge is finding the shortest (optimal) ruler for a given order - an NP-hard problem that has applications in radio astronomy, information theory, and combinatorics.

```
         Golomb Ruler of Order 4 (Optimal Length: 6)
    ------------------------------------------------
    0        1             3                 6
    |        |             |                 |
    *--------*-------------*-----------------*
         1         2               3
              |___________|
                    5
    |__________________________________|
                    6

    All pairwise differences are unique: {1, 2, 3, 5, 6}
```

### Why This Project?

This solver demonstrates progressive parallelization techniques - from single-threaded execution to distributed computing across HPC clusters. It serves as both a practical tool for finding optimal Golomb rulers and an educational resource for parallel algorithm design.

### Key Features

- **Four Solver Versions**: Sequential, OpenMP, Hybrid MPI+OpenMP, and Pure Hypercube architectures
- **AVX2 SIMD Optimizations**: Vectorized collision detection using 256-bit operations
- **Scalable Architecture**: From laptops to supercomputers with thousands of cores
- **O(log P) Bound Propagation**: Hypercube topology for efficient distributed pruning
- **HPC-Ready**: Integrated SLURM job templates and deployment scripts for Romeo HPC cluster

---

## Known Optimal Golomb Rulers

| Order (n) | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|-----------|---|---|---|---|---|---|---|---|----|----|----|----|-----|
| **Length** | 1 | 3 | 6 | 11 | 17 | 25 | 34 | 44 | 55 | 72 | 85 | 106 | 127 |

*Source: [OEIS A003022](https://oeis.org/A003022)*

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/golomb.git
cd golomb

# Build all versions
make all          # Sequential (v1) and OpenMP (v2)
make parallel     # MPI versions (v3 and v4)

# Run the sequential solver for order 10
./build/golomb_v1 10

# Run with OpenMP (8 threads) for order 11
OMP_NUM_THREADS=8 ./build/golomb_v2 11

# Run MPI hybrid solver (4 processes x 4 threads)
OMP_NUM_THREADS=4 mpirun -np 4 ./build/golomb_v3 12 --threads 4

# Run tests to verify correctness
make test
```

---

## Installation

### Prerequisites

| Requirement | Minimum Version | Notes |
|-------------|-----------------|-------|
| C++ Compiler | g++ 7.0+ or clang++ 6.0+ | C++17 support required |
| MPI | OpenMPI 3.0+ or MPICH 3.2+ | Required for v3 and v4 |
| CMake | 3.10+ | Optional (Makefile provided) |
| Python | 3.6+ | For visualization tools |

### Build from Source

```bash
# Build all sequential and OpenMP versions
make all

# Build MPI versions (requires MPI)
make parallel

# Build individual versions
make v1           # Sequential (single-threaded + AVX2)
make v2           # OpenMP (multi-threaded + AVX2)
make v3           # Hybrid MPI+OpenMP (master/worker)
make v4           # Pure Hypercube MPI+OpenMP

# Build without AVX2 (for older CPUs)
make v1_noavx
make v2_noavx
make v3_noavx
make v4_noavx
```

### Verify Installation

```bash
# Run unit tests
make test_unit        # Core algorithm tests
make test_openmp_unit # OpenMP parallelization tests
make test_mpi_unit    # MPI communication tests

# Run integration tests
make test             # Test v1 and v2
make test_mpi         # Test v3
make test_v4          # Test v4

# Run all tests
make test_all
```

---

## Usage

### Version Overview

| Version | Command | Best For |
|---------|---------|----------|
| **v1** | `golomb_v1` | Single-core machines, debugging, baseline benchmarks |
| **v2** | `golomb_v2` | Multi-core workstations (2-64 threads) |
| **v3** | `golomb_v3` | Small clusters (2-16 nodes), good load balancing |
| **v4** | `golomb_v4` | Large clusters (16+ nodes), best scalability |

### v1: Sequential Solver

Pure single-threaded implementation with AVX2 SIMD optimizations.

```bash
./build/golomb_v1 <order> [options]

Options:
  --no-simd       Disable AVX2 optimizations
  --csv FILE      Save results to CSV file
  --verbose       Show progress during search
  --benchmark     Run benchmarks for orders 4 to <order>

Examples:
  ./build/golomb_v1 10                    # Find optimal G10
  ./build/golomb_v1 11 --csv results.csv  # Save to CSV
  ./build/golomb_v1 12 --benchmark        # Benchmark G4-G12
```

### v2: OpenMP Solver

Multi-threaded solver using OpenMP task parallelism.

```bash
OMP_NUM_THREADS=<N> ./build/golomb_v2 <order> [options]

Options:
  --threads N     Set thread count (overrides OMP_NUM_THREADS)
  --no-simd       Disable AVX2 optimizations
  --csv FILE      Save results to CSV file
  --verbose       Show progress during search
  --benchmark     Run benchmarks for orders 4 to <order>
  --info          Show hardware info and exit

Examples:
  OMP_NUM_THREADS=8 ./build/golomb_v2 11              # 8 threads
  ./build/golomb_v2 12 --threads 32 --csv results.csv # 32 threads
  ./build/golomb_v2 10 --info                         # Show CPU info
```

### v3: Hybrid MPI+OpenMP (Master/Worker)

Distributed solver with master process distributing work to workers.

```bash
OMP_NUM_THREADS=<T> mpirun -np <P> ./build/golomb_v3 <order> [options]

Options:
  --threads N     OpenMP threads per MPI rank
  --depth N       Prefix depth for work distribution
  --csv FILE      Save results to CSV file
  --trace FILE    Save MPI trace for timeline visualization
  --no-simd       Disable AVX2 optimizations

Examples:
  # 4 MPI processes x 8 threads = 32 workers
  OMP_NUM_THREADS=8 mpirun -np 4 ./build/golomb_v3 12 --threads 8

  # With CSV output
  mpirun -np 8 ./build/golomb_v3 13 --threads 4 --csv hybrid.csv

  # Generate MPI timeline trace
  mpirun -np 4 ./build/golomb_v3 11 --trace timeline.csv
```

### v4: Pure Hypercube MPI+OpenMP

Decentralized architecture where all MPI ranks are equal peers.

```bash
OMP_NUM_THREADS=<T> mpirun -np <P> ./build/golomb_v4 <order> [options]

Options:
  --threads N     OpenMP threads per MPI rank
  --depth N       Prefix depth for work distribution
  --csv FILE      Save results to CSV file
  --trace FILE    Save MPI trace for timeline visualization
  --no-simd       Disable AVX2 optimizations

Examples:
  # 8 equal peers x 8 threads = 64 workers (power of 2 recommended)
  OMP_NUM_THREADS=8 mpirun -np 8 ./build/golomb_v4 13 --threads 8

  # Large-scale run
  mpirun -np 32 ./build/golomb_v4 14 --threads 4 --csv hypercube.csv
```

**Note:** For optimal hypercube performance, use a power of 2 for the process count (4, 8, 16, 32, 64, etc.).

---

## Architecture

### Version Comparison

| Feature | v1 Sequential | v2 OpenMP | v3 Hybrid | v4 Hypercube |
|---------|---------------|-----------|-----------|--------------|
| **Parallelism** | None | Shared memory | Distributed | Distributed |
| **Communication** | N/A | Atomic ops | Master/Worker | Peer-to-peer |
| **Bound Propagation** | Immediate | Atomic + cache | Centralized | O(log P) |
| **Load Balancing** | N/A | Task-based | Dynamic | Static |
| **Scalability** | 1 core | ~64 threads | ~100 processes | 1000+ processes |
| **Best Use Case** | Baseline | Workstation | Small cluster | Large cluster |

### Project Structure

```
golomb/
|-- Makefile                    # Build system
|-- README.md                   # This file
|-- LICENSE                     # MIT License
|-- CLAUDE.md                   # AI assistant guidelines
|
|-- include/golomb/             # Public headers
|   |-- golomb.hpp              # Core data structures (GolombRuler, SearchStats)
|   |-- bitset256.hpp           # AVX2-optimized 256-bit bitset
|   |-- greedy.hpp              # Greedy heuristic for initial bound
|   |-- config.hpp              # Centralized configuration constants
|   +-- difference.hpp          # Difference tracking utilities
|
|-- src/                        # Source implementations
|   |-- v1_sequential.cpp       # Sequential solver
|   |-- v2_openmp.cpp           # OpenMP parallel solver
|   |-- v3_hybrid.cpp           # MPI+OpenMP master/worker
|   |-- v4_hypercube.cpp        # MPI+OpenMP pure hypercube
|   +-- common/                 # Shared utilities
|       |-- validation.cpp      # Ruler validation
|       +-- timing.cpp          # High-resolution timing
|
|-- tests/                      # Test suite
|   |-- test_correctness.cpp    # Algorithm correctness tests
|   |-- test_openmp.cpp         # OpenMP parallelization tests
|   +-- test_mpi.cpp            # MPI communication tests
|
|-- scripts/hpc/                # HPC deployment automation
|   |-- config.sh               # Cluster configuration
|   |-- deploy.sh               # Deploy to cluster
|   |-- run_benchmarks.sh       # Submit benchmark jobs
|   +-- wait_and_fetch.sh       # Wait for jobs and fetch results
|
|-- jobs/                       # SLURM job templates
|
|-- tools/visualization/        # Analysis and plotting tools
|
|-- docs/                       # Documentation
|   |-- ARCHITECTURE.md         # System design details
|   |-- ALGORITHMS.md           # Algorithm explanations
|   |-- HPC_GUIDE.md            # HPC deployment guide
|   +-- Doxyfile                # Doxygen configuration
|
+-- results/                    # Benchmark results directory
```

### Key Data Structures

#### GolombRuler
```cpp
struct GolombRuler {
    std::vector<int> marks;  // Mark positions [0, a2, a3, ..., an]
    int length;              // Ruler length (last mark position)
    int order;               // Number of marks
};
```

#### BitSet256 (AVX2-optimized)
```cpp
struct alignas(32) BitSet256 {
    uint64_t words[4];       // 256 bits as 4 x 64-bit words

    void set(int bit);       // O(1) set
    bool test(int bit);      // O(1) test
    bool hasCollisionAVX2(const BitSet256& mask);  // SIMD collision check
};
```

#### ThreadState (Cache-aligned)
```cpp
struct alignas(64) ThreadState {
    int marks[MAX_ORDER];    // Current mark positions
    BitSet256 usedDiffs;     // Used differences
    int markCount;           // Number of marks placed
    // ... counters and cache padding
};
```

### Parallelization Strategies

#### v2: OpenMP Task Parallelism
- Tasks spawned at shallow tree depths (1-3)
- Thread-local bound caching (refresh every 16K nodes)
- Atomic bound updates with relaxed memory ordering
- Adaptive cutoff depth based on problem size

#### v3: Master/Worker MPI
- Master (rank 0) generates and distributes subtrees
- Workers explore subtrees with OpenMP
- Dynamic load balancing via work stealing
- Hypercube topology for O(log P) bound propagation

#### v4: Pure Hypercube
- All ranks are equal (no master bottleneck)
- Static work distribution (deterministic subtree assignment)
- Decentralized bound sharing via hypercube neighbors
- Best scalability for large clusters

---

## Optimizations

### Algorithm Optimizations

1. **Branch and Bound with Greedy Initial Bound**
   - Greedy heuristic provides initial upper bound
   - Enables aggressive pruning from the start

2. **Symmetry Breaking**
   - First mark limited to [1, bestLength/2]
   - Eliminates symmetric solutions, halving search space

3. **Triangular Pruning**
   - Remaining k marks need at least k(k+1)/2 additional length
   - Prunes infeasible branches early

### Implementation Optimizations

4. **BitSet256 with AVX2 SIMD**
   - 256-bit vectorized collision detection
   - Single instruction tests all differences at once

5. **Cache-Aligned Structures**
   - 64-byte alignment prevents false sharing
   - Thread-local state minimizes synchronization

6. **Bound Caching**
   - Local cache refreshed every 16K nodes
   - Reduces atomic operation overhead by ~90%

7. **Hypercube Communication (v4)**
   - O(log P) bound propagation latency
   - Fire-and-forget MPI_Isend for non-blocking updates

---

## Benchmarks

### Running Benchmarks

```bash
# Quick benchmark on local machine
make benchmark            # v1 and v2
make benchmark_v2         # v2 scaling test (1-32 threads)
make benchmark_v3         # v3 MPI scaling
make benchmark_v4         # v4 hypercube scaling

# Generate CSV results
./build/golomb_v1 12 --benchmark --csv results/v1_bench.csv
./build/golomb_v2 12 --benchmark --csv results/v2_bench.csv
```

### Expected Performance (Reference)

Performance on AMD EPYC 7763 (128 cores, 2.45 GHz):

| Order | v1 (1 thread) | v2 (32 threads) | v3 (4x8) | v4 (8x8) |
|-------|---------------|-----------------|----------|----------|
| G10   | ~2s | ~0.1s | ~0.05s | ~0.04s |
| G11   | ~30s | ~1.5s | ~0.5s | ~0.3s |
| G12   | ~12min | ~30s | ~8s | ~5s |

*Note: Actual performance varies by hardware and MPI implementation.*

---

## HPC Deployment

### Romeo HPC Cluster Integration

This project includes full automation for the Romeo HPC cluster at University of Reims.

#### First-Time Setup

```bash
# Configure SSH key for passwordless login
make romeo-setup ROMEO_USER=yourusername
```

#### Workflow

```bash
# Deploy code and compile on cluster
make romeo-deploy

# Submit all benchmark jobs (~200 jobs)
make romeo-bench

# Or submit essential benchmarks only (42 jobs)
make romeo-bench-quick

# Monitor job status
make romeo-status

# Fetch results when complete
make romeo-fetch

# Full workflow: deploy, submit, wait, fetch
make romeo && make romeo-wait
```

#### Configuration

Edit `scripts/hpc/config.sh` to customize:

```bash
ROMEO_USER="yourusername"           # Your cluster username
ROMEO_HOST="romeo1.univ-reims.fr"   # Login node
ARCHITECTURE="x64cpu"               # x64cpu or armgpu
SLURM_ACCOUNT="your-account"        # SLURM account/project
```

### Generic SLURM Deployment

For other clusters, use the SLURM templates in `jobs/`:

```bash
# Sequential job
sbatch jobs/seq_G12_v2_t32.slurm

# MPI job
sbatch jobs/mpi_G12_v4_p8.slurm
```

---

## API Reference

### Command-Line Interface

All versions support these common options:

| Option | Description |
|--------|-------------|
| `<order>` | Golomb ruler order (2-20) |
| `--csv FILE` | Append results to CSV file |
| `--no-simd` | Disable AVX2 optimizations |
| `--verbose` | Show progress during search |
| `--benchmark` | Run benchmarks for orders 4 to <order> |

MPI versions (v3, v4) add:

| Option | Description |
|--------|-------------|
| `--threads N` | OpenMP threads per MPI rank |
| `--depth N` | Prefix depth for work distribution |
| `--trace FILE` | Save MPI timeline trace to CSV |

### CSV Output Format

```csv
version,order,threads,time_ms,nodes_explored,nodes_pruned,solution,length
2,12,32,28543.21,1234567890,987654321,"[0,1,3,7,...]",85
```

For MPI versions, additional columns:
```csv
version,order,mpi_procs,omp_threads,total_workers,time_ms,...
```

---

## Documentation

### Generated Documentation

```bash
# Generate Doxygen HTML documentation
make docs

# View documentation
open docs/html/index.html
```

### Additional Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - System design and component details
- [docs/ALGORITHMS.md](docs/ALGORITHMS.md) - Algorithm explanations and complexity analysis
- [docs/HPC_GUIDE.md](docs/HPC_GUIDE.md) - Complete HPC deployment guide

---

## Troubleshooting

### Common Issues

**MPI version fails to compile**
```bash
# Check MPI installation
which mpicc mpicxx
mpirun --version

# Try MPICH instead of OpenMPI
apt install mpich  # Debian/Ubuntu
brew install mpich # macOS
```

**AVX2 not supported**
```bash
# Build without AVX2
make v1_noavx
make v2_noavx
```

**Permission denied on cluster**
```bash
# Ensure SSH key is properly configured
make romeo-setup ROMEO_USER=yourusername
```

**Wrong solution found**
```bash
# Run validation tests
make test_unit
```

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with appropriate tests
4. Ensure all tests pass (`make test_all`)
5. Submit a pull request

### Code Style

- C++17 standard
- 4-space indentation
- Doxygen-style documentation for public APIs
- Cache-aligned structures for shared data

---

## License

MIT License - Copyright (c) 2025 Nicolas Marano

See [LICENSE](LICENSE) for the full license text.

---

## Acknowledgments

- **OEIS A003022** - Reference for optimal Golomb ruler lengths
- **Romeo HPC Center** - University of Reims computing resources
- **OpenMPI/MPICH** - MPI implementations
- **Intel Intrinsics** - AVX2 SIMD documentation

---

## References

1. Golomb, S.W. (1972). *How to Number a Graph*. Graph Theory and Computing.
2. Distributed.net OGR Project - Distributed search for optimal Golomb rulers
3. OEIS Foundation. *A003022 - Length of optimal Golomb ruler with n marks*.

---

<div align="center">

**[Back to Top](#golomb-ruler-solver)**

</div>
