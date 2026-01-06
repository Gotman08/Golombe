# Architecture Overview

## Version Progression

| Version | Type | Parallelism | Communication |
|---------|------|-------------|---------------|
| v1 | Sequential | Single-threaded | None |
| v2 | OpenMP | Multi-threaded | Shared memory |
| v3 | MPI+OpenMP | Distributed | Master/worker |
| v4 | MPI+OpenMP | Distributed | Hypercube O(log P) |

## Directory Structure

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
```

## Key Components

### Data Structures
- **GolombRuler**: Mark positions, length, order
- **SearchState**: Cache-aligned (64 bytes) search state
- **BitSet256**: AVX2-optimized 256-bit bitset

### Parallelization
- **v2**: OpenMP tasks at shallow depths, sequential at deeper levels
- **v3**: Master distributes work, workers explore subtrees
- **v4**: Decentralized, all ranks equal, hypercube bound sharing

### Communication (MPI)
- `TAG_WORK=1`: Work distribution
- `TAG_RESULT=2`: Result reporting
- `TAG_BOUND=4`: Bound propagation

## Performance
- Cache-aligned structures (64-byte)
- AVX2 SIMD for difference checking
- Non-blocking MPI communication
- Thread-local bound caching
