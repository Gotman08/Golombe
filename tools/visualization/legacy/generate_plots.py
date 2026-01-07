#!/usr/bin/env python3
"""
Golomb Ruler Solver - Visualization Script

Generates comparison plots for sequential and parallel performance analysis.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

# Directories
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = PROJECT_DIR / 'results'
SEQ_DIR = RESULTS_DIR / 'sequential'
PAR_DIR = RESULTS_DIR / 'parallel'
PLOTS_DIR = RESULTS_DIR / 'plots'

# Create plots directory if it doesn't exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_sequential_results():
    """Load all sequential version results into a single DataFrame."""
    dfs = []
    for v in range(1, 6):
        filepath = SEQ_DIR / f'v{v}_results.csv'
        if filepath.exists():
            df = pd.read_csv(filepath)
            dfs.append(df)
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def load_parallel_results():
    """Load all parallel version results."""
    filepath = PAR_DIR / 'benchmark_results.csv'
    if filepath.exists():
        return pd.read_csv(filepath)
    return pd.DataFrame()


def plot_sequential_times(df, save=True):
    """Plot execution time vs order for all sequential versions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    markers = ['o', 's', '^', 'D', 'v']

    for i, v in enumerate(sorted(df['version'].unique())):
        subset = df[df['version'] == v].sort_values('order')
        ax.semilogy(subset['order'], subset['time_ms'],
                    marker=markers[i], color=colors[i],
                    linewidth=2, markersize=8,
                    label=f'v{v}')

    ax.set_xlabel('Ordre de Golomb (n)', fontsize=12)
    ax.set_ylabel('Temps d\'exécution (ms, échelle log)', fontsize=12)
    ax.set_title('Comparaison des temps d\'exécution - Versions séquentielles', fontsize=14)
    ax.legend(title='Version', loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(4, 12))

    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / 'sequential_times.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR / 'sequential_times.png'}")
    plt.close()


def plot_nodes_explored(df, save=True):
    """Plot nodes explored vs order for versions 2-5."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    markers = ['s', '^', 'D', 'v']

    versions = [v for v in sorted(df['version'].unique()) if v >= 2]

    for i, v in enumerate(versions):
        subset = df[df['version'] == v].sort_values('order')
        ax.semilogy(subset['order'], subset['nodes_explored'],
                    marker=markers[i], color=colors[i],
                    linewidth=2, markersize=8,
                    label=f'v{v}')

    ax.set_xlabel('Ordre de Golomb (n)', fontsize=12)
    ax.set_ylabel('Noeuds explorés (échelle log)', fontsize=12)
    ax.set_title('Réduction du nombre de noeuds explorés par version', fontsize=14)
    ax.legend(title='Version', loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(4, 12))

    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / 'nodes_explored.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR / 'nodes_explored.png'}")
    plt.close()


def plot_speedup_bars(df, save=True):
    """Plot speedup of each version compared to v1."""
    # Get orders where v1 exists
    v1_times = df[df['version'] == 1].set_index('order')['time_ms']
    orders = v1_times.index.tolist()

    fig, ax = plt.subplots(figsize=(12, 6))

    bar_width = 0.2
    x = np.arange(len(orders))
    colors = ['#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for i, v in enumerate([2, 3, 4, 5]):
        speedups = []
        for order in orders:
            v_time = df[(df['version'] == v) & (df['order'] == order)]['time_ms']
            if not v_time.empty and v1_times[order] > 0 and v_time.values[0] > 0:
                speedups.append(v1_times[order] / v_time.values[0])
            else:
                speedups.append(1)  # Default to 1x if can't calculate

        ax.bar(x + i * bar_width, speedups, bar_width,
               label=f'v{v}', color=colors[i])

    ax.set_xlabel('Ordre de Golomb', fontsize=12)
    ax.set_ylabel('Speedup (vs v1)', fontsize=12)
    ax.set_title('Accélération de chaque version par rapport à v1 (force brute)', fontsize=14)
    ax.set_xticks(x + bar_width * 1.5)
    ax.set_xticklabels([f'G{o}' for o in orders])
    ax.legend(title='Version')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / 'speedup_vs_v1.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR / 'speedup_vs_v1.png'}")
    plt.close()


def plot_pruning_ratio(df, save=True):
    """Plot pruning ratio for versions that have it."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#4daf4a', '#984ea3', '#ff7f00']
    markers = ['^', 'D', 'v']

    for i, v in enumerate([3, 4, 5]):
        subset = df[df['version'] == v].sort_values('order')
        if 'nodes_pruned' in subset.columns:
            total = subset['nodes_explored'] + subset['nodes_pruned']
            ratio = 100 * subset['nodes_pruned'] / total
            ax.plot(subset['order'], ratio,
                    marker=markers[i], color=colors[i],
                    linewidth=2, markersize=8,
                    label=f'v{v}')

    ax.set_xlabel('Ordre de Golomb (n)', fontsize=12)
    ax.set_ylabel('Ratio d\'élagage (%)', fontsize=12)
    ax.set_title('Efficacité de l\'élagage (Branch & Bound)', fontsize=14)
    ax.legend(title='Version', loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(4, 12))
    ax.set_ylim(0, 100)

    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / 'pruning_ratio.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR / 'pruning_ratio.png'}")
    plt.close()


def plot_version_comparison_table(df, save=True):
    """Create a summary comparison figure comparing all versions at same orders."""
    # Find common orders where all versions have data (for fair comparison)
    # Also show max order achieved by each version

    # Get v1 time for speedup calculation
    v1_data = df[df['version'] == 1].set_index('order')

    summary_data = []
    for v in sorted(df['version'].unique()):
        subset = df[df['version'] == v]
        max_order = subset['order'].max()

        # Use order 7 for comparison (all versions have it)
        comparison_order = 7
        if comparison_order in subset['order'].values:
            row = subset[subset['order'] == comparison_order].iloc[0]
            time_ms = row['time_ms']
            nodes = row['nodes_explored']

            # Calculate speedup vs v1
            if comparison_order in v1_data.index:
                v1_time = v1_data.loc[comparison_order, 'time_ms']
                speedup = v1_time / time_ms if time_ms > 0 else 0
            else:
                speedup = 1.0
        else:
            time_ms = 0
            nodes = 0
            speedup = 0

        summary_data.append({
            'Version': f'v{v}',
            'Max Order': max_order,
            'Time G7 (ms)': f"{time_ms:.2f}",
            'Speedup': f"{speedup:.1f}x",
            'Nodes G7': f"{nodes:,}"
        })

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.axis('off')

    table_data = [[d['Version'], d['Max Order'], d['Time G7 (ms)'], d['Speedup'], d['Nodes G7']]
                  for d in summary_data]
    headers = ['Version', 'Max Ordre', 'Temps G7 (ms)', 'Speedup vs v1', 'Noeuds G7']

    table = ax.table(cellText=table_data, colLabels=headers,
                     loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)

    # Style header
    for i, key in enumerate(headers):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', weight='bold')

    plt.title('Résumé des performances séquentielles (comparaison à G7)', fontsize=14, pad=20)
    plt.tight_layout()

    if save:
        plt.savefig(PLOTS_DIR / 'summary_table.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR / 'summary_table.png'}")
    plt.close()


def plot_parallel_speedup(df, save=True):
    """Plot parallel speedup vs number of processes, grouped by MPI version."""
    if df.empty:
        print("No parallel results to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by (version, order) to separate MPI versions
    versions = sorted(df['version'].unique())
    orders = sorted(df['order'].unique())

    # Color map: different colors for orders, different line styles for versions
    colors = {'8': '#984ea3', '9': '#4daf4a', '10': '#377eb8'}
    markers = {1: 'o', 2: 's', 3: '^'}
    linestyles = {1: '-', 2: '--', 3: ':'}

    for version in versions:
        for order in orders:
            subset = df[(df['version'] == version) & (df['order'] == order)].sort_values('procs')
            if subset.empty:
                continue

            color = colors.get(str(order), '#333333')
            marker = markers.get(version, 'o')
            linestyle = linestyles.get(version, '-')

            ax.plot(subset['procs'], subset['speedup'],
                    marker=marker, color=color, linestyle=linestyle,
                    linewidth=2, markersize=8,
                    label=f'MPI v{version} - G{order}')

    # Ideal speedup line
    procs = sorted(df['procs'].unique())
    ax.plot(procs, procs, 'k--', linewidth=1.5, label='Speedup idéal', alpha=0.7)

    ax.set_xlabel('Nombre de processus', fontsize=12)
    ax.set_ylabel('Speedup', fontsize=12)
    ax.set_title('Speedup parallèle par version MPI', fontsize=14)
    ax.legend(title='Version - Ordre', loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(procs)

    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / 'parallel_speedup.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR / 'parallel_speedup.png'}")
    plt.close()


def plot_parallel_efficiency(df, save=True):
    """Plot parallel efficiency vs number of processes, grouped by MPI version."""
    if df.empty:
        print("No parallel results to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by (version, order) to separate MPI versions
    versions = sorted(df['version'].unique())
    orders = sorted(df['order'].unique())

    # Color map: different colors for orders, different line styles for versions
    colors = {'8': '#984ea3', '9': '#4daf4a', '10': '#377eb8'}
    markers = {1: 'o', 2: 's', 3: '^'}
    linestyles = {1: '-', 2: '--', 3: ':'}

    for version in versions:
        for order in orders:
            subset = df[(df['version'] == version) & (df['order'] == order)].sort_values('procs')
            if subset.empty:
                continue

            color = colors.get(str(order), '#333333')
            marker = markers.get(version, 'o')
            linestyle = linestyles.get(version, '-')

            ax.plot(subset['procs'], subset['efficiency'] * 100,
                    marker=marker, color=color, linestyle=linestyle,
                    linewidth=2, markersize=8,
                    label=f'MPI v{version} - G{order}')

    ax.axhline(y=100, color='k', linestyle='--', linewidth=1.5, label='Efficacité idéale', alpha=0.7)

    ax.set_xlabel('Nombre de processus', fontsize=12)
    ax.set_ylabel('Efficacité (%)', fontsize=12)
    ax.set_title('Efficacité parallèle par version MPI', fontsize=14)
    ax.legend(title='Version - Ordre', loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)

    procs = sorted(df['procs'].unique())
    ax.set_xticks(procs)

    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / 'parallel_efficiency.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR / 'parallel_efficiency.png'}")
    plt.close()


def plot_complexity_analysis(df, save=True):
    """Plot theoretical vs measured complexity analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Theoretical complexity comparison
    orders = np.arange(4, 12)

    # Theoretical complexities (normalized)
    # v1: O(C(L,n)) ~ O(L^n / n!) exponential in both L and n
    # v2: O(n! * L) with pruning
    # v3-v5: O(b^d) where b is effective branching factor, d is depth

    # Simplified theoretical curves (relative)
    v1_theoretical = [math.factorial(o) * 10 for o in orders]
    v2_theoretical = [math.factorial(o) for o in orders]
    v3_theoretical = [2**(o-2) * o for o in orders]

    ax1.semilogy(orders, v1_theoretical, 'r--', linewidth=2, label='v1: O(n!)', alpha=0.7)
    ax1.semilogy(orders, v2_theoretical, 'b--', linewidth=2, label='v2: O(n!/k)', alpha=0.7)
    ax1.semilogy(orders, v3_theoretical, 'g--', linewidth=2, label='v3-v5: O(b^n)', alpha=0.7)

    # Add measured data points for v5
    v5_data = df[df['version'] == 5].sort_values('order')
    if not v5_data.empty:
        # Normalize to fit theoretical curve
        ax1.semilogy(v5_data['order'], v5_data['nodes_explored'],
                    'ko-', linewidth=2, markersize=8, label='v5 mesuré')

    ax1.set_xlabel('Ordre de Golomb (n)', fontsize=12)
    ax1.set_ylabel('Complexité (échelle log)', fontsize=12)
    ax1.set_title('Analyse de complexité: Théorique vs Mesuré', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(orders)

    # Right plot: Growth rate analysis
    v5_data = df[df['version'] == 5].sort_values('order')
    if len(v5_data) > 1:
        times = v5_data['time_ms'].values
        orders_v5 = v5_data['order'].values

        # Calculate growth factors between consecutive orders
        growth_factors = []
        for i in range(1, len(times)):
            if times[i-1] > 0:
                growth_factors.append(times[i] / times[i-1])
            else:
                growth_factors.append(1)

        ax2.bar(range(len(growth_factors)), growth_factors, color='#4472C4', edgecolor='black')
        ax2.set_xlabel('Transition d\'ordre', fontsize=12)
        ax2.set_ylabel('Facteur de croissance (T(n+1)/T(n))', fontsize=12)
        ax2.set_title('Facteur de croissance du temps entre ordres', fontsize=14)
        ax2.set_xticks(range(len(growth_factors)))
        ax2.set_xticklabels([f'G{orders_v5[i]}→G{orders_v5[i+1]}' for i in range(len(growth_factors))],
                          rotation=45, ha='right')
        ax2.axhline(y=np.mean(growth_factors), color='r', linestyle='--',
                   label=f'Moyenne: {np.mean(growth_factors):.1f}x')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / 'complexity_analysis.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR / 'complexity_analysis.png'}")
    plt.close()


def plot_scalability_analysis(seq_df, par_df, save=True):
    """Plot strong scaling analysis with Amdahl's Law reference."""
    if par_df.empty:
        print("No parallel results for scalability analysis")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left plot: Strong scaling (speedup vs procs)
    # Get best MPI version data (v3 for G10)
    g10_data = par_df[(par_df['version'] == 3) & (par_df['order'] == 10)].sort_values('procs')

    if not g10_data.empty:
        procs = g10_data['procs'].values
        speedups = g10_data['speedup'].values

        ax1.plot(procs, speedups, 'bo-', linewidth=2, markersize=10, label='MPI v3 (G10) mesuré')

        # Ideal linear speedup
        ax1.plot(procs, procs, 'k--', linewidth=1.5, label='Speedup idéal (linéaire)', alpha=0.7)

        # Amdahl's Law curves for different parallel fractions
        p_range = np.linspace(1, max(procs), 100)
        for f in [0.9, 0.8, 0.7]:
            amdahl_speedup = 1 / ((1 - f) + f / p_range)
            ax1.plot(p_range, amdahl_speedup, '--', linewidth=1.5,
                    label=f'Amdahl (f={f:.0%})', alpha=0.6)

        ax1.set_xlabel('Nombre de processus', fontsize=12)
        ax1.set_ylabel('Speedup', fontsize=12)
        ax1.set_title('Analyse de scalabilité forte (Strong Scaling)', fontsize=14)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, max(procs) + 1)
        ax1.set_ylim(0, max(procs) + 1)

    # Right plot: Efficiency comparison across orders
    orders_to_plot = [8, 9, 10]
    colors = {'8': '#984ea3', '9': '#4daf4a', '10': '#377eb8'}

    for order in orders_to_plot:
        order_data = par_df[(par_df['version'] == 3) & (par_df['order'] == order)].sort_values('procs')
        if not order_data.empty:
            ax2.plot(order_data['procs'], order_data['efficiency'] * 100,
                    'o-', color=colors[str(order)], linewidth=2, markersize=8,
                    label=f'G{order}')

    ax2.axhline(y=100, color='k', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=50, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    ax2.set_xlabel('Nombre de processus', fontsize=12)
    ax2.set_ylabel('Efficacité parallèle (%)', fontsize=12)
    ax2.set_title('Efficacité vs taille du problème (MPI v3)', fontsize=14)
    ax2.legend(title='Ordre', loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 110)

    # Add annotation
    ax2.annotate('Zone efficace\n(>50%)', xy=(2, 55), fontsize=10, color='green',
                ha='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / 'scalability_analysis.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR / 'scalability_analysis.png'}")
    plt.close()


def plot_optimization_impact(df, save=True):
    """Plot the impact of each optimization technique."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Calculate speedup of each version vs previous version for G7
    order = 7
    versions = [1, 2, 3, 4, 5]
    techniques = ['Baseline\n(Brute Force)', 'Backtracking\n(Early termination)',
                 'Branch & Bound\n(Pruning)', 'Bitset +\nSymmetry', 'Production\n(CLI)']

    times = []
    for v in versions:
        v_data = df[(df['version'] == v) & (df['order'] == order)]
        if not v_data.empty:
            times.append(v_data['time_ms'].values[0])
        else:
            times.append(None)

    # Calculate speedups vs v1
    speedups = []
    for t in times:
        if t and times[0] and t > 0:
            speedups.append(times[0] / t)
        else:
            speedups.append(1)

    # Create bar chart
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    bars = ax.bar(range(len(versions)), speedups, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        height = bar.get_height()
        ax.annotate(f'{speedup:.0f}x',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Version et technique d\'optimisation', fontsize=12)
    ax.set_ylabel('Speedup vs v1 (échelle log)', fontsize=12)
    ax.set_title(f'Impact des optimisations sur G{order}', fontsize=14)
    ax.set_xticks(range(len(versions)))
    ax.set_xticklabels(techniques)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, axis='y')

    # Add annotations for key improvements
    if len(speedups) > 2:
        ax.annotate('', xy=(2, speedups[2]), xytext=(1, speedups[1]),
                   arrowprops=dict(arrowstyle='->', color='red', lw=2))
        improvement = speedups[2] / speedups[1] if speedups[1] > 0 else 1
        ax.text(1.5, (speedups[1] + speedups[2]) / 2, f'+{improvement:.0f}x',
               fontsize=10, color='red', fontweight='bold')

    plt.tight_layout()
    if save:
        plt.savefig(PLOTS_DIR / 'optimization_impact.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR / 'optimization_impact.png'}")
    plt.close()


def main():
    print("=" * 60)
    print("Golomb Ruler Solver - Generating Plots")
    print("=" * 60)

    # Load data
    seq_df = load_sequential_results()
    par_df = load_parallel_results()

    print(f"\nSequential results: {len(seq_df)} rows")
    print(f"Parallel results: {len(par_df)} rows")

    if not seq_df.empty:
        print("\n--- Sequential Plots ---")
        plot_sequential_times(seq_df)
        plot_nodes_explored(seq_df)
        plot_speedup_bars(seq_df)
        plot_pruning_ratio(seq_df)
        plot_version_comparison_table(seq_df)
        plot_complexity_analysis(seq_df)
        plot_optimization_impact(seq_df)

    if not par_df.empty:
        print("\n--- Parallel Plots ---")
        plot_parallel_speedup(par_df)
        plot_parallel_efficiency(par_df)

    if not seq_df.empty and not par_df.empty:
        print("\n--- Scalability Analysis ---")
        plot_scalability_analysis(seq_df, par_df)

    print("\n" + "=" * 60)
    print(f"All plots saved to: {PLOTS_DIR}")
    print("=" * 60)


if __name__ == '__main__':
    main()
