#!/bin/bash
#
# Golomb Ruler Solver - Benchmark Script
# Generates CSV data for visualization
#
# Usage:
#   ./scripts/run_benchmark.sh              # Run all benchmarks
#   ./scripts/run_benchmark.sh --quick      # Quick test (G7-G8 only)
#   ./scripts/run_benchmark.sh --seq-only   # Sequential only (v1, v2)
#   ./scripts/run_benchmark.sh --mpi-only   # MPI only (v3, v4)
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
RESULTS_DIR="$PROJECT_DIR/results"

# Default orders to benchmark
SEQ_ORDERS=(7 8 9 10)
MPI_ORDERS=(8 9 10)

# Default parallelism
THREADS=(1 2 4 8)
MPI_PROCS=(2 4)
OMP_PER_PROC=2

# Parse arguments
QUICK_MODE=false
SEQ_ONLY=false
MPI_ONLY=false
WITH_TRACE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            SEQ_ORDERS=(7 8)
            MPI_ORDERS=(7 8)
            THREADS=(1 2 4)
            MPI_PROCS=(2)
            shift
            ;;
        --seq-only)
            SEQ_ONLY=true
            shift
            ;;
        --mpi-only)
            MPI_ONLY=true
            shift
            ;;
        --trace)
            WITH_TRACE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick      Quick mode (G7-G8 only)"
            echo "  --seq-only   Run sequential benchmarks only (v1, v2)"
            echo "  --mpi-only   Run MPI benchmarks only (v3, v4)"
            echo "  --trace      Generate MPI traces for timeline visualization"
            echo "  --help       Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check binaries exist
check_binaries() {
    local missing=false

    if [[ "$MPI_ONLY" != "true" ]]; then
        if [[ ! -x "$BUILD_DIR/golomb_v1" ]]; then
            log_error "golomb_v1 not found. Run 'make v1' first."
            missing=true
        fi
        if [[ ! -x "$BUILD_DIR/golomb_v2" ]]; then
            log_error "golomb_v2 not found. Run 'make v2' first."
            missing=true
        fi
    fi

    if [[ "$SEQ_ONLY" != "true" ]]; then
        if [[ ! -x "$BUILD_DIR/golomb_v3" ]]; then
            log_warn "golomb_v3 not found. Skipping MPI v3 benchmarks."
        fi
        if [[ ! -x "$BUILD_DIR/golomb_v4" ]]; then
            log_warn "golomb_v4 not found. Skipping MPI v4 benchmarks."
        fi
    fi

    if [[ "$missing" == "true" ]]; then
        echo ""
        echo "Build with: make all && make parallel"
        exit 1
    fi
}

# Create results directories
setup_directories() {
    mkdir -p "$RESULTS_DIR/sequential"
    mkdir -p "$RESULTS_DIR/parallel"
    mkdir -p "$RESULTS_DIR/traces"

    log_info "Results will be saved to: $RESULTS_DIR"
}

# Run V1 Sequential benchmarks
run_v1_benchmarks() {
    log_info "Running V1 Sequential benchmarks..."

    local csv_file="$RESULTS_DIR/sequential/v1.csv"

    # Clear previous results
    rm -f "$csv_file"

    for order in "${SEQ_ORDERS[@]}"; do
        echo -n "  G$order... "
        "$BUILD_DIR/golomb_v1" "$order" --csv "$csv_file" > /dev/null 2>&1
        echo "done"
    done

    log_info "V1 results saved to: $csv_file"
}

# Run V2 OpenMP benchmarks
run_v2_benchmarks() {
    log_info "Running V2 OpenMP benchmarks..."

    local csv_file="$RESULTS_DIR/sequential/v2.csv"

    # Clear previous results
    rm -f "$csv_file"

    for order in "${SEQ_ORDERS[@]}"; do
        for threads in "${THREADS[@]}"; do
            echo -n "  G$order (${threads}T)... "
            OMP_NUM_THREADS=$threads "$BUILD_DIR/golomb_v2" "$order" \
                --threads "$threads" \
                --csv "$csv_file" > /dev/null 2>&1
            echo "done"
        done
    done

    log_info "V2 results saved to: $csv_file"
}

# Run V3 MPI Hybrid benchmarks
run_v3_benchmarks() {
    if [[ ! -x "$BUILD_DIR/golomb_v3" ]]; then
        log_warn "Skipping V3 (binary not found)"
        return
    fi

    log_info "Running V3 MPI Hybrid benchmarks..."

    local csv_file="$RESULTS_DIR/parallel/v3.csv"
    local trace_file="$RESULTS_DIR/traces/v3_trace.csv"

    # Clear previous results
    rm -f "$csv_file"
    rm -f "$trace_file"

    for order in "${MPI_ORDERS[@]}"; do
        for procs in "${MPI_PROCS[@]}"; do
            echo -n "  G$order (${procs}P x ${OMP_PER_PROC}T)... "

            local cmd="OMP_NUM_THREADS=$OMP_PER_PROC mpirun --oversubscribe -np $procs"
            cmd="$cmd $BUILD_DIR/golomb_v3 $order"
            cmd="$cmd --threads $OMP_PER_PROC"
            cmd="$cmd --csv $csv_file"

            if [[ "$WITH_TRACE" == "true" ]]; then
                cmd="$cmd --trace $trace_file"
            fi

            eval "$cmd" > /dev/null 2>&1 || {
                echo "failed"
                continue
            }
            echo "done"
        done
    done

    log_info "V3 results saved to: $csv_file"
    if [[ "$WITH_TRACE" == "true" ]]; then
        log_info "V3 traces saved to: $trace_file"
    fi
}

# Run V4 Hypercube benchmarks
run_v4_benchmarks() {
    if [[ ! -x "$BUILD_DIR/golomb_v4" ]]; then
        log_warn "Skipping V4 (binary not found)"
        return
    fi

    log_info "Running V4 Hypercube MPI benchmarks..."

    local csv_file="$RESULTS_DIR/parallel/v4.csv"
    local trace_file="$RESULTS_DIR/traces/v4_trace.csv"

    # Clear previous results
    rm -f "$csv_file"
    rm -f "$trace_file"

    # V4 requires power-of-2 process counts
    local v4_procs=(2 4)

    for order in "${MPI_ORDERS[@]}"; do
        for procs in "${v4_procs[@]}"; do
            echo -n "  G$order (${procs}P x ${OMP_PER_PROC}T)... "

            local cmd="OMP_NUM_THREADS=$OMP_PER_PROC mpirun --oversubscribe -np $procs"
            cmd="$cmd $BUILD_DIR/golomb_v4 $order"
            cmd="$cmd --threads $OMP_PER_PROC"
            cmd="$cmd --csv $csv_file"

            if [[ "$WITH_TRACE" == "true" ]]; then
                cmd="$cmd --trace $trace_file"
            fi

            eval "$cmd" > /dev/null 2>&1 || {
                echo "failed"
                continue
            }
            echo "done"
        done
    done

    log_info "V4 results saved to: $csv_file"
    if [[ "$WITH_TRACE" == "true" ]]; then
        log_info "V4 traces saved to: $trace_file"
    fi
}

# Print summary
print_summary() {
    echo ""
    log_info "Benchmark complete!"
    echo ""
    echo "Generated files:"

    if [[ -f "$RESULTS_DIR/sequential/v1.csv" ]]; then
        local count=$(wc -l < "$RESULTS_DIR/sequential/v1.csv")
        echo "  - sequential/v1.csv ($((count-1)) results)"
    fi

    if [[ -f "$RESULTS_DIR/sequential/v2.csv" ]]; then
        local count=$(wc -l < "$RESULTS_DIR/sequential/v2.csv")
        echo "  - sequential/v2.csv ($((count-1)) results)"
    fi

    if [[ -f "$RESULTS_DIR/parallel/v3.csv" ]]; then
        local count=$(wc -l < "$RESULTS_DIR/parallel/v3.csv")
        echo "  - parallel/v3.csv ($((count-1)) results)"
    fi

    if [[ -f "$RESULTS_DIR/parallel/v4.csv" ]]; then
        local count=$(wc -l < "$RESULTS_DIR/parallel/v4.csv")
        echo "  - parallel/v4.csv ($((count-1)) results)"
    fi

    if [[ "$WITH_TRACE" == "true" ]]; then
        if [[ -f "$RESULTS_DIR/traces/v3_trace.csv" ]]; then
            echo "  - traces/v3_trace.csv"
        fi
        if [[ -f "$RESULTS_DIR/traces/v4_trace.csv" ]]; then
            echo "  - traces/v4_trace.csv"
        fi
    fi

    echo ""
    echo "Generate visualizations with:"
    echo "  python tools/visualization/generate_all.py --data $RESULTS_DIR --output $RESULTS_DIR/plots"
}

# Main
main() {
    echo "========================================"
    echo "Golomb Ruler Solver - Benchmark Suite"
    echo "========================================"
    echo ""

    check_binaries
    setup_directories

    if [[ "$MPI_ONLY" != "true" ]]; then
        run_v1_benchmarks
        run_v2_benchmarks
    fi

    if [[ "$SEQ_ONLY" != "true" ]]; then
        run_v3_benchmarks
        run_v4_benchmarks
    fi

    print_summary
}

main
