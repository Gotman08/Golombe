#!/bin/bash
# =============================================================================
# Collect and Analyze Results from Romeo HPC Cluster
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

# =============================================================================
# Functions
# =============================================================================

fetch_results() {
    log_info "Fetching results from Romeo..."

    mkdir -p "${LOCAL_RESULTS_DIR}"

    # Fetch CSV results
    remote_fetch "${REMOTE_RESULTS_DIR}/*.csv" "${LOCAL_RESULTS_DIR}/" 2>/dev/null || true

    # Fetch SLURM output logs
    remote_fetch "${REMOTE_RESULTS_DIR}/*.out" "${LOCAL_RESULTS_DIR}/" 2>/dev/null || true
    remote_fetch "${REMOTE_RESULTS_DIR}/*.err" "${LOCAL_RESULTS_DIR}/" 2>/dev/null || true

    # Fetch raw output files
    remote_fetch "${REMOTE_RESULTS_DIR}/*_raw.txt" "${LOCAL_RESULTS_DIR}/" 2>/dev/null || true

    log_success "Results downloaded to: ${LOCAL_RESULTS_DIR}"

    # Count files
    local csv_count=$(ls -1 "${LOCAL_RESULTS_DIR}"/*.csv 2>/dev/null | wc -l || echo 0)
    log_info "Downloaded ${csv_count} CSV files"
}

cleanup_remote() {
    log_info "Cleaning up remote files on Romeo..."

    # Show what will be cleaned
    echo ""
    echo "=== Files to be cleaned ==="

    remote_exec "
        echo 'Old SLURM job scripts (>7 days):'
        find ${REMOTE_BASE_DIR}/jobs -name '*.slurm' -mtime +7 2>/dev/null | wc -l | xargs echo '  Count:'

        echo ''
        echo 'Output logs (*.out, *.err):'
        ls -1 ${REMOTE_RESULTS_DIR}/*.out ${REMOTE_RESULTS_DIR}/*.err 2>/dev/null | wc -l | xargs echo '  Count:'

        echo ''
        echo 'Raw output files (*_raw.txt):'
        ls -1 ${REMOTE_RESULTS_DIR}/*_raw.txt 2>/dev/null | wc -l | xargs echo '  Count:'

        echo ''
        echo 'Orphaned scratch directories:'
        ls -1d /scratch_p/\$USER/* 2>/dev/null | wc -l | xargs echo '  Count:'
    "

    echo ""
    read -p "Proceed with cleanup? [y/N] " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Performing cleanup..."

        remote_exec "
            # Remove old job scripts (>7 days)
            find ${REMOTE_BASE_DIR}/jobs -name '*.slurm' -mtime +7 -delete 2>/dev/null || true
            echo 'Cleaned old job scripts'

            # Remove output logs (keep CSV for data)
            rm -f ${REMOTE_RESULTS_DIR}/*.out ${REMOTE_RESULTS_DIR}/*.err 2>/dev/null || true
            echo 'Cleaned output logs'

            # Remove raw output files
            rm -f ${REMOTE_RESULTS_DIR}/*_raw.txt 2>/dev/null || true
            echo 'Cleaned raw output files'

            # Clean orphaned scratch directories (be careful!)
            # Only clean directories that don't correspond to running jobs
            for dir in /scratch_p/\$USER/*/; do
                if [[ -d \"\$dir\" ]]; then
                    job_id=\$(basename \"\$dir\")
                    if ! squeue -j \"\$job_id\" &>/dev/null; then
                        rm -rf \"\$dir\"
                        echo \"Cleaned orphaned scratch: \$job_id\"
                    fi
                fi
            done
        "

        log_success "Cleanup complete!"
    else
        log_info "Cleanup cancelled"
    fi
}

check_job_status() {
    log_info "Checking job status on Romeo..."
    echo ""

    remote_exec "squeue -u \$USER -o '%.10i %.20j %.8T %.10M %.6D %R' 2>/dev/null" || {
        log_warn "Could not fetch job status"
    }

    echo ""
    log_info "Completed jobs (CSV files):"
    remote_exec "ls -1 ${REMOTE_RESULTS_DIR}/*.csv 2>/dev/null | wc -l" || echo "0"
}

generate_summary() {
    log_info "Generating results summary..."

    local summary_file="${LOCAL_RESULTS_DIR}/summary_$(date +%Y%m%d_%H%M%S).md"

    cat > "${summary_file}" << EOF
# Golomb Ruler Benchmark Results - Romeo HPC

**Generated:** $(date)
**Cluster:** Romeo (Université de Reims)

## Test Configuration

| Version Type | Versions | Orders |
|-------------|----------|--------|
| Sequential v1 | Brute force | G5-G7 |
| Sequential v2 | Backtracking | G6-G8 |
| Sequential v3 | Branch & Bound | G7-G9 |
| Sequential v4 | Optimized | G8-G11 |
| Sequential v5 | Production | G10-G13 |
| Sequential v6 | Hardware (OpenMP) | G11-G14 |
| MPI v1 | Basic master/worker | G9-G11 |
| MPI v2 | Hypercube | G9-G11 |
| MPI v3 | Optimized | G10-G14 |

---

## Sequential Results

EOF

    # Parse sequential results (v1-v5 format)
    echo "### Standard Sequential (v1-v5)" >> "${summary_file}"
    echo "" >> "${summary_file}"
    echo "| Version | Order | Time (ms) | Nodes Explored | Nodes Pruned | Length |" >> "${summary_file}"
    echo "|---------|-------|-----------|----------------|--------------|--------|" >> "${summary_file}"

    for csv in "${LOCAL_RESULTS_DIR}"/seq_*_v[1-5]_*.csv; do
        if [[ -f "$csv" ]]; then
            tail -n +2 "$csv" 2>/dev/null | while IFS=, read -r version order time nodes pruned solution length; do
                echo "| v${version} | G${order} | ${time} | ${nodes} | ${pruned} | ${length} |" >> "${summary_file}"
            done
        fi
    done

    # Parse v6 results (includes threads)
    echo "" >> "${summary_file}"
    echo "### Hardware Optimized (v6)" >> "${summary_file}"
    echo "" >> "${summary_file}"
    echo "| Order | Threads | Time (ms) | Nodes Explored | Length |" >> "${summary_file}"
    echo "|-------|---------|-----------|----------------|--------|" >> "${summary_file}"

    for csv in "${LOCAL_RESULTS_DIR}"/seq_*_v6_*.csv; do
        if [[ -f "$csv" ]]; then
            tail -n +2 "$csv" 2>/dev/null | while IFS=, read -r version order threads time nodes pruned solution length; do
                echo "| G${order} | ${threads} | ${time} | ${nodes} | ${length} |" >> "${summary_file}"
            done
        fi
    done

    # Parse MPI results
    echo "" >> "${summary_file}"
    echo "---" >> "${summary_file}"
    echo "" >> "${summary_file}"
    echo "## MPI Parallel Results" >> "${summary_file}"
    echo "" >> "${summary_file}"
    echo "| Version | Order | Procs | Time (ms) | Nodes | Length |" >> "${summary_file}"
    echo "|---------|-------|-------|-----------|-------|--------|" >> "${summary_file}"

    for csv in "${LOCAL_RESULTS_DIR}"/mpi_*.csv; do
        if [[ -f "$csv" ]]; then
            tail -n +2 "$csv" 2>/dev/null | while IFS=, read -r version order procs time speedup efficiency nodes solution length; do
                echo "| MPI v${version} | G${order} | ${procs} | ${time} | ${nodes} | ${length} |" >> "${summary_file}"
            done
        fi
    done

    # Add hypercube comparison section
    echo "" >> "${summary_file}"
    echo "---" >> "${summary_file}"
    echo "" >> "${summary_file}"
    echo "## Hypercube Communication Analysis" >> "${summary_file}"
    echo "" >> "${summary_file}"
    echo "Comparison between MPI v2 (hypercube bounds) and MPI v3 (optimized hypercube + work stealing):" >> "${summary_file}"
    echo "" >> "${summary_file}"

    log_success "Summary written to: ${summary_file}"
    echo ""
    cat "${summary_file}"
}

generate_plots() {
    log_info "Generating performance plots..."

    # Create Python script for visualization
    local plot_script="${LOCAL_RESULTS_DIR}/plot_results.py"

    cat > "${plot_script}" << 'PYTHON'
#!/usr/bin/env python3
"""
Golomb Ruler Benchmark Results Visualization
Handles different CSV formats from various solver versions
"""
import os
import glob
import sys
from pathlib import Path

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("Error: Required packages not found.")
    print("Install with: pip install pandas matplotlib numpy")
    sys.exit(1)

RESULTS_DIR = Path(__file__).parent

def load_sequential_results():
    """Load sequential CSV results (v1-v5 and v6 formats)."""
    data = []

    # v1-v5 format: version,order,time_ms,nodes_explored,nodes_pruned,solution,length
    for f in glob.glob(str(RESULTS_DIR / "seq_*_v[1-5]_*.csv")):
        try:
            df = pd.read_csv(f)
            df['threads'] = 1  # Single threaded
            data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    # v6 format: version,order,threads,time_ms,nodes_explored,nodes_pruned,solution,length
    for f in glob.glob(str(RESULTS_DIR / "seq_*_v6_*.csv")):
        try:
            df = pd.read_csv(f)
            data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    if data:
        return pd.concat(data, ignore_index=True)
    return pd.DataFrame()

def load_mpi_results():
    """Load MPI CSV results."""
    data = []

    # MPI format: version,order,procs,time_ms,speedup,efficiency,nodes,solution,length
    for f in glob.glob(str(RESULTS_DIR / "mpi_*.csv")):
        try:
            df = pd.read_csv(f)
            data.append(df)
        except Exception as e:
            print(f"Warning: Could not load {f}: {e}")

    if data:
        return pd.concat(data, ignore_index=True)
    return pd.DataFrame()

def plot_version_comparison(seq_df):
    """Compare sequential versions performance."""
    if seq_df.empty:
        print("No sequential data for version comparison")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by version and order, get minimum time
    grouped = seq_df.groupby(['version', 'order'])['time_ms'].min().reset_index()

    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    markers = ['o', 's', '^', 'D', 'v', 'p']

    for idx, version in enumerate(sorted(grouped['version'].unique())):
        v_data = grouped[grouped['version'] == version].sort_values('order')
        ax.plot(v_data['order'], v_data['time_ms'],
               marker=markers[idx % len(markers)],
               color=colors[idx],
               linewidth=2,
               markersize=8,
               label=f'v{version}')

    ax.set_xlabel('Order (G)', fontsize=12)
    ax.set_ylabel('Time (ms)', fontsize=12)
    ax.set_title('Sequential Version Comparison', fontsize=14)
    ax.set_yscale('log')
    ax.legend(title='Version')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'seq_version_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR / 'seq_version_comparison.png'}")

def plot_v6_scaling(seq_df):
    """Plot v6 OpenMP thread scaling."""
    v6_df = seq_df[seq_df['version'] == 6]
    if v6_df.empty:
        print("No v6 data for thread scaling")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Speedup plot
    for order in sorted(v6_df['order'].unique()):
        order_df = v6_df[v6_df['order'] == order].sort_values('threads')
        if len(order_df) > 1:
            base_time = order_df[order_df['threads'] == 1]['time_ms'].values
            if len(base_time) > 0:
                speedups = base_time[0] / order_df['time_ms']
                axes[0].plot(order_df['threads'], speedups,
                           marker='o', label=f'G{order}')

    max_threads = v6_df['threads'].max()
    axes[0].plot([1, max_threads], [1, max_threads], 'k--', alpha=0.3, label='Ideal')
    axes[0].set_xlabel('Threads')
    axes[0].set_ylabel('Speedup')
    axes[0].set_title('v6 OpenMP Thread Scaling')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Efficiency plot
    for order in sorted(v6_df['order'].unique()):
        order_df = v6_df[v6_df['order'] == order].sort_values('threads')
        if len(order_df) > 1:
            base_time = order_df[order_df['threads'] == 1]['time_ms'].values
            if len(base_time) > 0:
                efficiency = (base_time[0] / order_df['time_ms']) / order_df['threads'] * 100
                axes[1].plot(order_df['threads'], efficiency,
                           marker='s', label=f'G{order}')

    axes[1].axhline(y=100, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Threads')
    axes[1].set_ylabel('Efficiency (%)')
    axes[1].set_title('v6 Parallel Efficiency')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'v6_thread_scaling.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR / 'v6_thread_scaling.png'}")

def plot_mpi_scaling(mpi_df):
    """Plot MPI strong scaling for all versions."""
    if mpi_df.empty:
        print("No MPI data for scaling plot")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {'1': 'blue', '2': 'green', '3': 'red'}
    linestyles = {'1': '-', '2': '--', '3': '-.'}

    # Strong scaling
    for version in sorted(mpi_df['version'].unique()):
        for order in sorted(mpi_df['order'].unique()):
            v_df = mpi_df[(mpi_df['version'] == version) & (mpi_df['order'] == order)]
            v_df = v_df.sort_values('procs')
            if len(v_df) > 1:
                base_time = v_df.iloc[0]['time_ms']
                if base_time > 0:
                    speedups = base_time / v_df['time_ms']
                    axes[0].plot(v_df['procs'], speedups,
                               marker='o',
                               color=colors.get(str(version), 'gray'),
                               linestyle=linestyles.get(str(version), '-'),
                               label=f'MPI v{version} G{order}')

    max_procs = mpi_df['procs'].max()
    axes[0].plot([1, max_procs], [1, max_procs], 'k--', alpha=0.3, label='Ideal')
    axes[0].set_xlabel('Processes')
    axes[0].set_ylabel('Speedup')
    axes[0].set_title('MPI Strong Scaling')
    axes[0].set_xscale('log', base=2)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Efficiency
    for version in sorted(mpi_df['version'].unique()):
        for order in sorted(mpi_df['order'].unique()):
            v_df = mpi_df[(mpi_df['version'] == version) & (mpi_df['order'] == order)]
            v_df = v_df.sort_values('procs')
            if len(v_df) > 1:
                base_time = v_df.iloc[0]['time_ms']
                if base_time > 0:
                    efficiency = (base_time / v_df['time_ms']) / v_df['procs'] * 100
                    axes[1].plot(v_df['procs'], efficiency,
                               marker='s',
                               color=colors.get(str(version), 'gray'),
                               linestyle=linestyles.get(str(version), '-'),
                               label=f'MPI v{version} G{order}')

    axes[1].axhline(y=100, color='k', linestyle='--', alpha=0.3)
    axes[1].set_xlabel('Processes')
    axes[1].set_ylabel('Efficiency (%)')
    axes[1].set_title('MPI Parallel Efficiency')
    axes[1].set_xscale('log', base=2)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'mpi_scaling.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR / 'mpi_scaling.png'}")

def plot_hypercube_comparison(mpi_df):
    """Compare hypercube versions (v2 vs v3)."""
    hyper_df = mpi_df[mpi_df['version'].isin([2, 3])]
    if hyper_df.empty:
        print("No hypercube version data (v2, v3)")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for order in sorted(hyper_df['order'].unique()):
        for version in [2, 3]:
            v_df = hyper_df[(hyper_df['order'] == order) & (hyper_df['version'] == version)]
            if not v_df.empty:
                v_df = v_df.sort_values('procs')
                marker = 'o' if version == 2 else 's'
                linestyle = '-' if version == 2 else '--'
                color = 'green' if version == 2 else 'red'
                ax.plot(v_df['procs'], v_df['time_ms'],
                       marker=marker,
                       linestyle=linestyle,
                       color=color,
                       label=f'G{order} MPI v{version}')

    ax.set_xlabel('Processes')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Hypercube Communication: v2 (basic) vs v3 (optimized)')
    ax.legend()
    ax.set_xscale('log', base=2)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'hypercube_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {RESULTS_DIR / 'hypercube_comparison.png'}")

if __name__ == '__main__':
    print("=" * 50)
    print("Golomb Ruler Benchmark - Results Visualization")
    print("=" * 50)
    print(f"Results directory: {RESULTS_DIR}")
    print("")

    print("Loading results...")
    seq_df = load_sequential_results()
    mpi_df = load_mpi_results()

    print(f"  Sequential results: {len(seq_df)} entries")
    print(f"  MPI results: {len(mpi_df)} entries")

    if seq_df.empty and mpi_df.empty:
        print("\nNo results found. Make sure CSV files are in the results directory.")
        sys.exit(1)

    print("\nGenerating plots...")

    if not seq_df.empty:
        plot_version_comparison(seq_df)
        plot_v6_scaling(seq_df)

    if not mpi_df.empty:
        plot_mpi_scaling(mpi_df)
        plot_hypercube_comparison(mpi_df)

    print("\nDone! Check the results directory for PNG files.")
PYTHON

    chmod +x "${plot_script}"
    log_success "Plot script created: ${plot_script}"
    log_info "Run with: python3 ${plot_script}"
}

# =============================================================================
# Main
# =============================================================================

usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  fetch     Download results from Romeo"
    echo "  status    Check job status on Romeo"
    echo "  summary   Generate markdown summary"
    echo "  plots     Generate visualization plots"
    echo "  cleanup   Clean up old files on Romeo (interactive)"
    echo "  all       Run fetch, status, summary, plots"
    echo ""
}

case "${1:-all}" in
    fetch)
        check_ssh && fetch_results
        ;;
    status)
        check_ssh && check_job_status
        ;;
    summary)
        generate_summary
        ;;
    plots)
        generate_plots
        ;;
    cleanup)
        check_ssh && cleanup_remote
        ;;
    all)
        check_ssh
        check_job_status
        fetch_results
        generate_summary
        generate_plots
        ;;
    *)
        usage
        exit 1
        ;;
esac
