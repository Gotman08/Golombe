#!/bin/bash
# =============================================================================
# Deploy Golomb Solver to Romeo HPC Cluster
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# =============================================================================
# Main deployment
# =============================================================================

main() {
    echo "=============================================="
    echo "  Golomb Ruler Solver - Romeo Deployment"
    echo "=============================================="
    echo ""

    # Check SSH connection
    if ! check_ssh; then
        exit 1
    fi

    # Create remote directories
    log_info "Creating remote directories..."
    remote_exec "mkdir -p ${REMOTE_BASE_DIR}/{src,include,build,results/romeo,scripts}"

    # Sync source code and headers
    log_info "Syncing source code to Romeo..."
    remote_copy "${LOCAL_PROJECT_DIR}/src/" "${REMOTE_BASE_DIR}/src/"
    remote_copy "${LOCAL_PROJECT_DIR}/include/" "${REMOTE_BASE_DIR}/include/"
    remote_copy "${LOCAL_PROJECT_DIR}/Makefile" "${REMOTE_BASE_DIR}/"
    remote_copy "${SCRIPT_DIR}/" "${REMOTE_BASE_DIR}/scripts/romeo/"

    # Create environment setup script for Romeo 2025 (Spack 1.0.1)
    log_info "Creating environment setup script for ${ARCHITECTURE} architecture..."

    local env_command=$(get_env_command)

    remote_exec "cat > ${REMOTE_BASE_DIR}/setup_env.sh << 'EOF'
#!/bin/bash
# Environment setup for Golomb solver on Romeo 2025
# Architecture: ${ARCHITECTURE}
# Generated: $(date)

# Load Romeo 2025 environment (Spack 1.0.1)
# This replaces the old 'module load' system
${env_command}

# Load OpenMPI for parallel builds (use hash for reliability)
spack load /kfjcqqr 2>/dev/null || spack load openmpi 2>/dev/null || true

# Verify environment is loaded
echo \"=== Romeo 2025 Environment ===\"
echo \"Architecture: ${ARCHITECTURE}\"

# Show compiler and MPI versions
if command -v g++ &> /dev/null; then
    echo \"Compiler: \$(g++ --version | head -1)\"
fi

if command -v mpirun &> /dev/null; then
    echo \"MPI: \$(mpirun --version 2>&1 | head -1)\"
else
    echo \"Warning: mpirun not found in PATH\"
fi

# Show spack-loaded packages (if spack is available)
if command -v spack &> /dev/null; then
    echo \"\"
    echo \"Spack packages:\"
    spack find --loaded 2>/dev/null || true
fi
EOF"

    # Compile on Romeo (v1-v4)
    log_info "Compiling on Romeo (v1, v2, v3, v4)..."
    remote_exec "cd ${REMOTE_BASE_DIR} && source setup_env.sh && make clean && make v1 v2 v3 v4"

    log_success "Deployment complete!"
    echo ""
    log_info "Versions compiled:"
    echo "  - v1: Sequential (single-threaded)"
    echo "  - v2: OpenMP (multi-threaded)"
    echo "  - v3: MPI+OpenMP (master/worker)"
    echo "  - v4: MPI+OpenMP (hypercube)"
    echo ""
    log_info "Next steps:"
    echo "  1. SSH to Romeo: ssh ${ROMEO_SSH}"
    echo "  2. Go to project: cd ${REMOTE_BASE_DIR}"
    echo "  3. Submit G12 benchmark: bash scripts/romeo/benchmark_g12.sh"
    echo ""
    log_info "Or run benchmark from here:"
    echo "  ssh ${ROMEO_SSH} 'cd ${REMOTE_BASE_DIR} && bash scripts/romeo/benchmark_g12.sh'"
}

# Run main
main "$@"
