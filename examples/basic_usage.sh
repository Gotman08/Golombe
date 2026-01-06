#!/bin/bash
# =============================================================================
# Basic Usage Examples - Golomb Ruler Solver
#
# Copyright (c) 2025 Nicolas Marano
# Licensed under the MIT License. See LICENSE file for details.
# =============================================================================

set -e

echo "=== Golomb Ruler Solver - Basic Examples ==="
echo ""

# Check if executables exist
if [ ! -f "./build/golomb_v1" ]; then
    echo "Error: Executables not found. Run 'make all' first."
    exit 1
fi

# Example 1: Sequential solver
echo "--- Example 1: Sequential (v1) ---"
echo "Command: ./build/golomb_v1 8"
./build/golomb_v1 8
echo ""

# Example 2: OpenMP with 4 threads
echo "--- Example 2: OpenMP (v2) with 4 threads ---"
echo "Command: OMP_NUM_THREADS=4 ./build/golomb_v2 9"
OMP_NUM_THREADS=4 ./build/golomb_v2 9
echo ""

# Example 3: Benchmark mode
echo "--- Example 3: Benchmark mode ---"
echo "Command: ./build/golomb_v1 7 --benchmark"
./build/golomb_v1 7 --benchmark
echo ""

echo "=== All examples completed ==="
