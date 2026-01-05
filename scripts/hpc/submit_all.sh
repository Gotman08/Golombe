#!/bin/bash
# =============================================================================
# Submit All Jobs in Batches (Respects Romeo's 40 job limit)
# =============================================================================

JOBS_DIR="${HOME}/golomb/jobs"
MAX_JOBS=40
SLEEP_TIME=60

# Get list of all job files
JOB_FILES=("${JOBS_DIR}"/*.slurm)
TOTAL_JOBS=${#JOB_FILES[@]}
SUBMITTED=0

echo "=== Batch Job Submitter ==="
echo "Total jobs to submit: ${TOTAL_JOBS}"
echo "Max concurrent jobs: ${MAX_JOBS}"
echo ""

for job_file in "${JOB_FILES[@]}"; do
    # Check current queue size
    while true; do
        CURRENT=$(squeue -u $USER -h 2>/dev/null | wc -l)
        if [ "$CURRENT" -lt "$MAX_JOBS" ]; then
            break
        fi
        echo "Queue full (${CURRENT}/${MAX_JOBS}). Waiting ${SLEEP_TIME}s..."
        sleep $SLEEP_TIME
    done

    # Submit job
    JOB_NAME=$(basename "$job_file")
    if sbatch "$job_file" >/dev/null 2>&1; then
        SUBMITTED=$((SUBMITTED + 1))
        echo "[${SUBMITTED}/${TOTAL_JOBS}] Submitted: ${JOB_NAME}"
    else
        echo "[${SUBMITTED}/${TOTAL_JOBS}] FAILED: ${JOB_NAME}"
    fi
done

echo ""
echo "=== Submission Complete ==="
echo "Total submitted: ${SUBMITTED}/${TOTAL_JOBS}"
echo ""
echo "Monitor with: squeue -u $USER"
