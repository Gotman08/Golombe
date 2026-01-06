#!/bin/bash
# =============================================================================
# Check Storage Quotas on Romeo HPC Cluster
# =============================================================================
# Romeo 2025 storage quotas:
#   /home     : Soft 15 Go, Hard 20 Go (persistent storage)
#   /scratch_p: Soft 15 Go, Hard 20 Go (fast temporary storage)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# =============================================================================
# Functions
# =============================================================================

check_local_space() {
    echo "=== Local Storage Status ==="
    echo ""
    df -h "${LOCAL_PROJECT_DIR}" 2>/dev/null || echo "Could not check local storage"
    echo ""

    local project_size=$(du -sh "${LOCAL_PROJECT_DIR}" 2>/dev/null | cut -f1)
    echo "Project size: ${project_size:-unknown}"

    local results_size=$(du -sh "${LOCAL_RESULTS_DIR}" 2>/dev/null | cut -f1)
    echo "Results size: ${results_size:-unknown}"
    echo ""
}

check_remote_quotas() {
    echo "=== Romeo Storage Quotas ==="
    echo ""

    # Check quota using GPFS command
    remote_exec "mmlsquota --block-size auto gpfs 2>/dev/null" || {
        echo "Note: mmlsquota command not available or permission denied"
        echo "Trying alternative methods..."
        echo ""

        # Fallback: check disk usage manually
        remote_exec "echo '--- /home usage ---' && du -sh \$HOME 2>/dev/null || echo 'Could not check'"
        remote_exec "echo '--- /scratch_p usage ---' && du -sh /scratch_p/\$USER 2>/dev/null || echo 'Not available'"
    }

    echo ""
}

check_remote_project() {
    echo "=== Remote Project Status ==="
    echo ""

    remote_exec "
        if [[ -d ${REMOTE_BASE_DIR} ]]; then
            echo 'Project directory: ${REMOTE_BASE_DIR}'
            echo 'Total size:' \$(du -sh ${REMOTE_BASE_DIR} 2>/dev/null | cut -f1)
            echo ''
            echo 'Breakdown:'
            du -sh ${REMOTE_BASE_DIR}/* 2>/dev/null | sort -h || echo '  (empty or inaccessible)'
            echo ''
            echo 'Results files:'
            ls -lh ${REMOTE_RESULTS_DIR}/*.csv 2>/dev/null | wc -l | xargs echo '  CSV files:'
            ls -lh ${REMOTE_RESULTS_DIR}/*.out 2>/dev/null | wc -l | xargs echo '  Output logs:'
        else
            echo 'Project not deployed yet: ${REMOTE_BASE_DIR}'
        fi
    "
    echo ""
}

check_scratch_usage() {
    echo "=== Scratch Usage ==="
    echo ""

    remote_exec "
        SCRATCH_DIR='/scratch_p/\$USER'
        if [[ -d \"\$SCRATCH_DIR\" ]]; then
            echo 'Scratch directory: '\$SCRATCH_DIR
            echo 'Total size:' \$(du -sh \$SCRATCH_DIR 2>/dev/null | cut -f1)
            echo ''
            echo 'Active job directories:'
            ls -la \$SCRATCH_DIR 2>/dev/null || echo '  (empty)'
        else
            echo 'No scratch directory found'
        fi
    "
    echo ""
}

show_recommendations() {
    echo "=== Recommendations ==="
    echo ""
    echo "Romeo 2025 storage limits:"
    echo "  /home     : 15 GB soft / 20 GB hard"
    echo "  /scratch_p: 15 GB soft / 20 GB hard"
    echo ""
    echo "Best practices:"
    echo "  - Use /scratch_p for job data (faster, auto-cleaned)"
    echo "  - Keep compiled binaries in /home"
    echo "  - Download results regularly to local machine"
    echo "  - Clean old job outputs: rm \$HOME/golomb/results/romeo/*.out"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  local     Check local storage status"
    echo "  remote    Check remote Romeo quotas"
    echo "  project   Check remote project size"
    echo "  scratch   Check scratch usage"
    echo "  all       Run all checks (default)"
    echo ""
}

echo "=============================================="
echo "  Romeo Storage Quota Checker"
echo "=============================================="
echo ""

case "${1:-all}" in
    local)
        check_local_space
        ;;
    remote)
        check_ssh && check_remote_quotas
        ;;
    project)
        check_ssh && check_remote_project
        ;;
    scratch)
        check_ssh && check_scratch_usage
        ;;
    all)
        check_local_space
        if check_ssh; then
            check_remote_quotas
            check_remote_project
            check_scratch_usage
        fi
        show_recommendations
        ;;
    -h|--help)
        usage
        exit 0
        ;;
    *)
        echo "Unknown command: $1"
        usage
        exit 1
        ;;
esac

echo "=============================================="
