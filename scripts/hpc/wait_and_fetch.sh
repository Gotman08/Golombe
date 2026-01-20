#!/bin/bash
# =============================================================================
# Wait for SLURM jobs to complete, then fetch results
# =============================================================================
# Usage: bash scripts/hpc/wait_and_fetch.sh [--interval SECONDS]
#
# Polls Romeo squeue until all jobs are complete, then fetches results.
# =============================================================================

set -e

# Load configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

# Configuration
POLL_INTERVAL="${1:-60}"  # Default: 60 seconds

echo "=============================================="
echo "Waiting for SLURM jobs to complete on Romeo"
echo "=============================================="
echo "Host: $ROMEO_HOST"
echo "User: $ROMEO_USER"
echo "Poll interval: ${POLL_INTERVAL}s"
echo ""

# Check SSH connectivity first
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$ROMEO_SSH" "echo OK" &>/dev/null; then
    echo "ERROR: Cannot connect to Romeo via SSH"
    echo "Check your SSH key configuration"
    exit 1
fi

# Wait loop
while true; do
    # Count running/pending jobs
    JOB_COUNT=$(ssh "$ROMEO_SSH" "squeue -u $ROMEO_USER -h 2>/dev/null | wc -l" || echo "0")

    if [ "$JOB_COUNT" -eq 0 ]; then
        echo ""
        echo "All jobs complete!"
        break
    fi

    # Show progress with timestamp
    TIMESTAMP=$(date '+%H:%M:%S')
    echo "[$TIMESTAMP] $JOB_COUNT job(s) remaining..."

    # Optional: show job names
    if [ "$JOB_COUNT" -le 5 ]; then
        ssh "$ROMEO_SSH" "squeue -u $ROMEO_USER --format='  %.20j %.8T %.10M' -h 2>/dev/null" || true
    fi

    sleep "$POLL_INTERVAL"
done

# Fetch results
echo ""
echo "=============================================="
echo "Fetching results from Romeo"
echo "=============================================="

mkdir -p "$LOCAL_RESULTS_DIR"
rsync -avz --progress "$ROMEO_SSH:$REMOTE_RESULTS_DIR/" "$LOCAL_RESULTS_DIR/"

# Count results
CSV_COUNT=$(find "$LOCAL_RESULTS_DIR" -name "*.csv" 2>/dev/null | wc -l)
echo ""
echo "=============================================="
echo "Done! $CSV_COUNT CSV files in results/romeo/"
echo "=============================================="
