#!/usr/bin/env python3
"""
Generate All Visualizations for Golomb Ruler Solver

Main script to generate all performance visualizations including:
- Performance comparison charts
- Scaling analysis (strong/weak)
- Roofline model
- MPI timeline (if trace data available)
- Flamegraph/treemap (if profiling data available)

Usage:
    python generate_all.py --data results/ --output plots/ --theme publication

Prerequisites:
    Run benchmarks first to generate CSV data:
    ./scripts/run_benchmark.sh
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from golomb_viz import apply_style, COLORS, EXPORT
from golomb_viz.data_loader import DataLoader, BenchmarkData
from golomb_viz.plots import (
    PerformancePlots,
    ScalingPlots,
    RooflinePlot,
    FlamegraphPlot,
    MPITimelinePlot,
)
from golomb_viz.plots.base import PlotConfig


def generate_performance_plots(
    data: BenchmarkData,
    config: PlotConfig,
    verbose: bool = True
) -> None:
    """Generate performance comparison plots."""
    if data.sequential is None:
        if verbose:
            print("  Skipping performance plots (no sequential data)")
        return

    if verbose:
        print("  Generating performance plots...")

    perf = PerformancePlots(config)

    # Time comparison
    fig, _ = perf.time_comparison(data.sequential)
    perf.save('performance_time_comparison')
    perf.close()

    # Speedup chart (if multiple versions)
    if data.sequential['version'].nunique() > 1:
        fig, ax = perf.speedup_chart(data.sequential)
        # Skip saving if no data (only text message)
        if len(ax.lines) > 0:
            perf.save('performance_speedup')
        elif verbose:
            print("    Skipping performance_speedup (no data)")
        perf.close()

    # Heatmap
    fig, _ = perf.version_heatmap(data.sequential)
    perf.save('performance_heatmap')
    perf.close()

    # Throughput (if nodes data available)
    if 'nodes_explored' in data.sequential.columns:
        fig, _ = perf.throughput_comparison(data.sequential)
        perf.save('performance_throughput')
        perf.close()

    if verbose:
        print("    -> performance_*.png generated")


def generate_scaling_plots(
    data: BenchmarkData,
    config: PlotConfig,
    verbose: bool = True
) -> None:
    """Generate scaling analysis plots."""
    if data.parallel is None and data.scaling is None:
        if verbose:
            print("  Skipping scaling plots (no parallel/scaling data)")
        return

    if verbose:
        print("  Generating scaling plots...")

    scaling = ScalingPlots(config)

    # Use parallel or scaling data
    scaling_data = data.scaling if data.scaling is not None else data.parallel

    if scaling_data is not None and 'procs' in scaling_data.columns:
        # Get T_seq baselines from sequential data
        t_seq_map = {}
        if data.sequential is not None and 'time_ms' in data.sequential.columns:
            seq = data.sequential
            # Filter for single-thread baseline (version 1 or lowest thread count)
            if 'omp_threads' in seq.columns:
                baseline = seq[seq['omp_threads'] == 1]
            elif 'version' in seq.columns:
                baseline = seq[seq['version'] == 1]
            else:
                baseline = seq

            for order in baseline['order'].unique():
                order_time = baseline[baseline['order'] == order]['time_ms'].values
                if len(order_time) > 0:
                    t_seq_map[int(order)] = float(order_time[0])

        # Generate plots per version and meaningful orders (G11+)
        versions = sorted(scaling_data['version'].dropna().unique())
        orders = sorted([o for o in scaling_data['order'].dropna().unique() if o >= 11])

        for version in versions:
            version_int = int(version)
            for order in orders:
                order_int = int(order)

                # Filter data for this version and order
                mask = (scaling_data['version'] == version) & (scaling_data['order'] == order)
                subset = scaling_data[mask].sort_values('procs')

                if len(subset) < 2:
                    continue  # Need at least 2 points for a line

                # Get T_seq for this order
                t_seq = t_seq_map.get(order_int)
                if t_seq is None and verbose:
                    print(f"    Warning: No T_seq for G{order_int}, using min procs time")

                # Generate strong scaling plot
                try:
                    fig, _ = scaling.strong_scaling(subset, t_seq=t_seq)
                    scaling.save(f'scaling_strong_v{version_int}_G{order_int}')
                    scaling.close()
                except Exception as e:
                    if verbose:
                        print(f"    Warning: v{version_int} G{order_int} scaling plot failed: {e}")

        # Generate combined efficiency plot for best version (v3)
        if 3 in versions:
            v3_data = scaling_data[scaling_data['version'] == 3]
            for order in orders:
                order_int = int(order)
                subset = v3_data[v3_data['order'] == order].sort_values('procs')
                if len(subset) >= 2:
                    try:
                        fig, _ = scaling.efficiency_plot(subset)
                        scaling.save(f'scaling_efficiency_v3_G{order_int}')
                        scaling.close()
                    except Exception as e:
                        if verbose:
                            print(f"    Warning: v3 G{order_int} efficiency plot failed: {e}")

        # Weak scaling (if problem_size column exists)
        if 'problem_size' in scaling_data.columns:
            fig, _ = scaling.weak_scaling(scaling_data)
            scaling.save('scaling_weak')
            scaling.close()

        if verbose:
            print("    -> scaling_*.png generated")


def generate_roofline_plot(
    data: BenchmarkData,
    config: PlotConfig,
    peak_gflops: float = 100.0,
    peak_bandwidth: float = 50.0,
    verbose: bool = True
) -> None:
    """Generate Roofline model plot."""
    if data.sequential is None or 'nodes_explored' not in data.sequential.columns:
        if verbose:
            print("  Skipping Roofline model (no nodes_explored data)")
        return

    if verbose:
        print("  Generating Roofline model...")

    roofline = RooflinePlot(
        config,
        peak_gflops=peak_gflops,
        peak_bandwidth_gb=peak_bandwidth
    )

    fig, ax = roofline.from_benchmark_data(data.sequential)
    # Add cache hierarchy
    roofline.add_caches(ax, l1_bandwidth=200, l2_bandwidth=80, l3_bandwidth=50)

    roofline.save('roofline_model')
    roofline.close()

    if verbose:
        print("    -> roofline_model.png generated")


def generate_mpi_timeline(
    data: BenchmarkData,
    config: PlotConfig,
    verbose: bool = True
) -> None:
    """Generate MPI timeline plot."""
    if data.mpi_trace is None:
        if verbose:
            print("  Skipping MPI timeline (no trace data)")
            print("    Run with --trace option: mpirun -np 4 ./build/golomb_v3 10 --trace trace.csv")
        return

    if verbose:
        print("  Generating MPI timeline...")

    timeline = MPITimelinePlot(config)

    # Gantt chart
    fig, _ = timeline.gantt_chart(data.mpi_trace)
    timeline.save('mpi_timeline_gantt')
    timeline.close()

    # Communication matrix
    if 'partner_rank' in data.mpi_trace.columns:
        fig, _ = timeline.communication_matrix(data.mpi_trace)
        timeline.save('mpi_communication_matrix')
        timeline.close()

    # Load balance
    fig, _ = timeline.load_balance_chart(data.mpi_trace)
    timeline.save('mpi_load_balance')
    timeline.close()

    if verbose:
        print("    -> mpi_*.png generated")


def generate_flamegraph(
    config: PlotConfig,
    profiling_data: Optional[Path] = None,
    verbose: bool = True
) -> None:
    """Generate flamegraph/treemap visualization."""
    if profiling_data is None or not profiling_data.exists():
        if verbose:
            print("  Skipping flamegraph (no profiling data)")
            print("    Provide profiling CSV with: --profiling path/to/profiling.csv")
        return

    if verbose:
        print("  Generating flamegraph...")

    import pandas as pd
    flame = FlamegraphPlot(config)
    data = pd.read_csv(profiling_data)

    # Treemap
    fig, _ = flame.treemap(data)
    flame.save('flamegraph_treemap')
    flame.close()

    # Icicle (if data has parent column for hierarchy)
    if 'parent' in data.columns:
        fig, _ = flame.icicle(data)
        flame.save('flamegraph_icicle')
        flame.close()

    if verbose:
        print("    -> flamegraph_*.png generated")


def generate_all_plots(
    data_dir: Path,
    output_dir: Path,
    theme: str = 'light',
    formats: list = ['png'],
    dpi: int = 300,
    profiling_data: Optional[Path] = None,
    verbose: bool = True
) -> None:
    """
    Generate all visualizations from benchmark data.

    Args:
        data_dir: Directory containing benchmark CSV files
        output_dir: Output directory for plots
        theme: Visual theme ('light', 'dark', 'publication')
        formats: Output formats ['png', 'pdf', 'svg']
        dpi: Output resolution
        profiling_data: Optional path to profiling CSV for flamegraph
        verbose: Print progress
    """
    if verbose:
        print(f"Loading data from {data_dir}...")

    # Setup
    apply_style(theme)
    config = PlotConfig(
        output_dir=output_dir,
        formats=formats,
        dpi=dpi,
        theme=theme
    )

    # Load data
    loader = DataLoader(data_dir)
    data = loader.load_all()

    # Check if any data was found
    if data.sequential is None and data.parallel is None:
        print(f"ERROR: No CSV data found in {data_dir}")
        print()
        print("Expected directory structure:")
        print("  results/")
        print("    sequential/  <- v1.csv, v2.csv")
        print("    parallel/    <- v3.csv, v4.csv")
        print()
        print("Run benchmarks first:")
        print("  ./scripts/run_benchmark.sh")
        sys.exit(1)

    if verbose:
        print(f"Platform detected: {data.platform}")
        print(f"Sequential data: {'Yes' if data.sequential is not None else 'No'}")
        print(f"Parallel data: {'Yes' if data.parallel is not None else 'No'}")
        print(f"MPI trace: {'Yes' if data.mpi_trace is not None else 'No'}")
        print()

    # Generate plots
    generate_performance_plots(data, config, verbose)
    generate_scaling_plots(data, config, verbose)
    generate_roofline_plot(data, config, verbose=verbose)
    generate_mpi_timeline(data, config, verbose)
    generate_flamegraph(config, profiling_data, verbose)

    if verbose:
        print(f"\nAll plots saved to {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Generate Golomb Ruler Solver visualizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_all.py --data results/ --output plots/
  python generate_all.py --data results/ --theme publication --formats png,pdf
  python generate_all.py --data results/ --profiling profiling.csv

Prerequisites:
  Run benchmarks first to generate CSV data:
  ./scripts/run_benchmark.sh
        """
    )

    parser.add_argument(
        '--data', '-d',
        type=Path,
        required=True,
        help='Directory containing benchmark CSV files (required)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=Path('output'),
        help='Output directory for plots (default: output/)'
    )
    parser.add_argument(
        '--theme', '-t',
        choices=['light', 'dark', 'publication'],
        default='light',
        help='Visual theme (default: light)'
    )
    parser.add_argument(
        '--formats', '-f',
        type=str,
        default='png',
        help='Output formats, comma-separated (default: png)'
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=300,
        help='Output resolution (default: 300)'
    )
    parser.add_argument(
        '--profiling',
        type=Path,
        default=None,
        help='Path to profiling CSV for flamegraph generation'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Validate data directory
    if not args.data.exists():
        print(f"ERROR: Data directory does not exist: {args.data}")
        print("Run benchmarks first: ./scripts/run_benchmark.sh")
        sys.exit(1)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)

    # Parse formats
    formats = [f.strip() for f in args.formats.split(',')]

    verbose = not args.quiet

    generate_all_plots(
        args.data,
        args.output,
        theme=args.theme,
        formats=formats,
        dpi=args.dpi,
        profiling_data=args.profiling,
        verbose=verbose
    )


if __name__ == '__main__':
    main()
