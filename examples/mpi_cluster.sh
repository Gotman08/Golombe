#!/bin/bash
# =============================================================================
# MPI Cluster Examples - Golomb Ruler Solver
#
# Copyright (c) 2025 Nicolas Marano
# Licensed under the MIT License. See LICENSE file for details.
# =============================================================================

set -e

echo "=== Golomb Ruler Solver - MPI Examples ==="
echo ""

# Check if MPI executables exist
if [ ! -f "./build/golomb_v3" ]; then
    echo "Error: MPI executables not found. Run 'make v3 v4' first."
    exit 1
fi

# Check if mpirun is available
if ! command -v mpirun &> /dev/null; then
    echo "Error: mpirun not found. Install OpenMPI or MPICH."
    exit 1
fi

# Example 1: Hybrid MPI+OpenMP (v3)
echo "--- Example 1: Hybrid MPI+OpenMP (v3) ---"
echo "Command: OMP_NUM_THREADS=2 mpirun --oversubscribe -np 2 ./build/golomb_v3 9 --threads 2"
OMP_NUM_THREADS=2 mpirun --oversubscribe -np 2 ./build/golomb_v3 9 --threads 2
echo ""

# Example 2: Hypercube MPI+OpenMP (v4)
echo "--- Example 2: Hypercube MPI+OpenMP (v4) ---"
echo "Command: OMP_NUM_THREADS=2 mpirun --oversubscribe -np 4 ./build/golomb_v4 9 --threads 2"
OMP_NUM_THREADS=2 mpirun --oversubscribe -np 4 ./build/golomb_v4 9 --threads 2
echo ""

echo "=== All MPI examples completed ==="
echo ""
echo "For cluster deployment, see: docs/HPC_GUIDE.md"
