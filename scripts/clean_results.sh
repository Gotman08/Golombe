#!/bin/bash
# Clean results script for Golomb Ruler project
# Usage: ./scripts/clean_results.sh [OPTIONS]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_DIR/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Flags
DRY_RUN=false
FORCE=false
CLEAN_CSV=false
CLEAN_PLOTS=false
CLEAN_LOCAL=false
CLEAN_ROMEO=false
CLEAN_ALL=false

show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Clean generated results from the Golomb Ruler project."
    echo ""
    echo "Options:"
    echo "  --all       Clean all results (CSV, plots, logs)"
    echo "  --csv       Clean CSV files only"
    echo "  --plots     Clean PNG plot files only"
    echo "  --local     Clean results/local/ directory"
    echo "  --romeo     Clean results/romeo/ directory"
    echo "  --dry-run   Show what would be deleted without deleting"
    echo "  --force     Skip confirmation prompt"
    echo "  -h, --help  Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all              # Clean everything (with confirmation)"
    echo "  $0 --all --force      # Clean everything without confirmation"
    echo "  $0 --csv --plots      # Clean only CSV and PNG files"
    echo "  $0 --romeo --dry-run  # Preview Romeo cleanup"
}

clean_files() {
    local pattern="$1"
    local description="$2"
    local count=0
    local files

    # Use compgen to safely expand glob patterns
    files=$(compgen -G "$pattern" 2>/dev/null) || true

    if [ -n "$files" ]; then
        while IFS= read -r f; do
            if [ -f "$f" ]; then
                if [ "$DRY_RUN" = true ]; then
                    echo "  Would delete: $f"
                else
                    rm -f "$f"
                fi
                ((count++)) || true
            fi
        done <<< "$files"
    fi

    if [ $count -gt 0 ]; then
        if [ "$DRY_RUN" = true ]; then
            echo -e "${YELLOW}$description: $count file(s) would be deleted${NC}"
        else
            echo -e "${GREEN}$description: $count file(s) deleted${NC}"
        fi
    fi
}

clean_csv() {
    echo "Cleaning CSV files..."
    clean_files "$RESULTS_DIR/sequential/*.csv" "Sequential CSV"
    clean_files "$RESULTS_DIR/parallel/*.csv" "Parallel CSV"
    clean_files "$RESULTS_DIR/local/*.csv" "Local CSV"
    clean_files "$RESULTS_DIR/romeo/*.csv" "Romeo CSV"
}

clean_plots() {
    echo "Cleaning plot files..."
    clean_files "$RESULTS_DIR/plots/*.png" "Plots"
    clean_files "$RESULTS_DIR/analysis/*.png" "Analysis"
    clean_files "$RESULTS_DIR/comparison/*.png" "Comparison"
    clean_files "$RESULTS_DIR/romeo/plots/*.png" "Romeo plots"
}

clean_local() {
    echo "Cleaning local results..."
    clean_files "$RESULTS_DIR/local/*.csv" "Local CSV"
    clean_files "$RESULTS_DIR/local/*.txt" "Local logs"
}

clean_romeo() {
    echo "Cleaning Romeo results..."
    clean_files "$RESULTS_DIR/romeo/*.csv" "Romeo CSV"
    clean_files "$RESULTS_DIR/romeo/*.txt" "Romeo logs"
    clean_files "$RESULTS_DIR/romeo/*.out" "Romeo stdout"
    clean_files "$RESULTS_DIR/romeo/*.err" "Romeo stderr"
    clean_files "$RESULTS_DIR/romeo/plots/*.png" "Romeo plots"
}

clean_logs() {
    echo "Cleaning log files..."
    clean_files "$RESULTS_DIR/parallel/*.txt" "Parallel logs"
    clean_files "$RESULTS_DIR/sequential/*.txt" "Sequential logs"
}

# Parse arguments
if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --all)
            CLEAN_ALL=true
            ;;
        --csv)
            CLEAN_CSV=true
            ;;
        --plots)
            CLEAN_PLOTS=true
            ;;
        --local)
            CLEAN_LOCAL=true
            ;;
        --romeo)
            CLEAN_ROMEO=true
            ;;
        --dry-run)
            DRY_RUN=true
            ;;
        --force)
            FORCE=true
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
    shift
done

# Check if results directory exists
if [ ! -d "$RESULTS_DIR" ]; then
    echo -e "${YELLOW}Results directory does not exist: $RESULTS_DIR${NC}"
    exit 0
fi

# Confirmation for --all
if [ "$CLEAN_ALL" = true ] && [ "$FORCE" = false ] && [ "$DRY_RUN" = false ]; then
    echo -e "${YELLOW}This will delete ALL results in $RESULTS_DIR${NC}"
    read -p "Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Execute cleaning
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}=== DRY RUN MODE ===${NC}"
fi

if [ "$CLEAN_ALL" = true ]; then
    clean_csv
    clean_plots
    clean_logs
    clean_local
    clean_romeo
else
    [ "$CLEAN_CSV" = true ] && clean_csv
    [ "$CLEAN_PLOTS" = true ] && clean_plots
    [ "$CLEAN_LOCAL" = true ] && clean_local
    [ "$CLEAN_ROMEO" = true ] && clean_romeo
fi

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}=== No files were actually deleted ===${NC}"
else
    echo -e "${GREEN}Done!${NC}"
fi
