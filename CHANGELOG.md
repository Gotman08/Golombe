# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-01-06

### Added

#### Solver Versions
- **v1 - Sequential**: Single-threaded optimized solver with AVX2 SIMD support
- **v2 - OpenMP**: Multi-threaded solver using OpenMP task parallelism
- **v3 - Hybrid MPI+OpenMP**: Distributed solver with master/worker architecture
- **v4 - Hypercube MPI+OpenMP**: Decentralized solver with O(log P) communication

#### Core Features
- Branch and Bound algorithm with greedy initial bound
- BitSet256 for O(1) difference lookup with AVX2 optimization
- Symmetry breaking optimization (halves search space)
- Cache-aligned data structures (64-byte alignment)
- Comprehensive test suite (58 tests)

#### HPC Support
- SLURM job templates for cluster deployment
- Romeo HPC cluster integration scripts
- Weak and strong scaling benchmarks
- CSV output for result analysis

#### Documentation
- Architecture documentation
- Algorithm explanation
- HPC deployment guide
- Usage examples

### Performance
- Verified optimal solutions for G4-G14
- Near-linear speedup with OpenMP (up to 16 threads)
- Efficient load balancing in MPI versions
- Hypercube topology for scalable bound propagation
