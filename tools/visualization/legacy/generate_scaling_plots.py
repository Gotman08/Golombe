#!/usr/bin/env python3
"""
Generate Strong and Weak Scaling plots for Golomb Ruler benchmarks.

Usage:
    python3 generate_scaling_plots.py [--strong FILE] [--weak FILE] [--output DIR]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_strong_scaling(csv_file: str, output_dir: str):
    """Generate Strong Scaling plot with speedup and efficiency."""
    df = pd.read_csv(csv_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    workers = df['total_workers'].values
    speedup = df['speedup'].values
    efficiency = df['efficiency'].values

    # Speedup plot
    ax1.plot(workers, speedup, 'bo-', linewidth=2, markersize=8, label='Actual')
    ax1.plot(workers, workers, 'k--', linewidth=1, alpha=0.7, label='Ideal (linear)')
    ax1.set_xlabel('Number of Workers', fontsize=12)
    ax1.set_ylabel('Speedup', fontsize=12)
    ax1.set_title('Strong Scaling: Speedup vs Workers', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log', base=2)

    # Efficiency plot
    ax2.bar(range(len(workers)), efficiency, color='steelblue', alpha=0.8)
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Ideal (100%)')
    ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Good (80%)')
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Poor (50%)')
    ax2.set_xticks(range(len(workers)))
    ax2.set_xticklabels([f'{w}' for w in workers])
    ax2.set_xlabel('Number of Workers', fontsize=12)
    ax2.set_ylabel('Efficiency (%)', fontsize=12)
    ax2.set_title('Strong Scaling: Parallel Efficiency', fontsize=14)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 110)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'strong_scaling.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Strong scaling plot saved: {output_file}")
    plt.close()


def plot_weak_scaling(csv_file: str, output_dir: str):
    """Generate Weak Scaling plot."""
    df = pd.read_csv(csv_file)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    nodes = df['nodes'].values
    orders = df['order'].values
    times = df['time_ms'].values
    efficiency = df['relative_efficiency'].values

    # Time vs nodes (ideal = constant)
    ax1.plot(nodes, times, 'go-', linewidth=2, markersize=8, label='Actual time')
    ax1.axhline(y=times[0], color='k', linestyle='--', alpha=0.7, label='Ideal (constant)')

    # Add order labels
    for i, (n, t, o) in enumerate(zip(nodes, times, orders)):
        ax1.annotate(f'G{o}', (n, t), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    ax1.set_xlabel('Number of Nodes', fontsize=12)
    ax1.set_ylabel('Time (ms)', fontsize=12)
    ax1.set_title('Weak Scaling: Time vs Resources', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)

    # Efficiency plot
    colors = ['green' if e >= 80 else 'orange' if e >= 50 else 'red' for e in efficiency]
    ax2.bar(range(len(nodes)), efficiency, color=colors, alpha=0.8)
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.7, label='Ideal (100%)')
    ax2.axhline(y=80, color='orange', linestyle='--', alpha=0.5)
    ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5)

    # Add order labels on bars
    for i, (e, o) in enumerate(zip(efficiency, orders)):
        ax2.text(i, e + 2, f'G{o}', ha='center', fontsize=9)

    ax2.set_xticks(range(len(nodes)))
    ax2.set_xticklabels([f'{n} node{"s" if n > 1 else ""}' for n in nodes])
    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.set_ylabel('Relative Efficiency (%)', fontsize=12)
    ax2.set_title('Weak Scaling: Efficiency', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 120)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'weak_scaling.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Weak scaling plot saved: {output_file}")
    plt.close()


def plot_combined_scaling(strong_file: str, weak_file: str, output_dir: str):
    """Generate combined scaling summary."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Strong scaling data
    if os.path.exists(strong_file):
        df_strong = pd.read_csv(strong_file)
        workers = df_strong['total_workers'].values
        speedup = df_strong['speedup'].values
        eff_strong = df_strong['efficiency'].values

        # Speedup
        axes[0, 0].plot(workers, speedup, 'bo-', linewidth=2, markersize=8, label='Actual')
        axes[0, 0].plot(workers, workers, 'k--', alpha=0.5, label='Ideal')
        axes[0, 0].set_xlabel('Workers')
        axes[0, 0].set_ylabel('Speedup')
        axes[0, 0].set_title('Strong Scaling: Speedup')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Strong efficiency
        axes[0, 1].bar(range(len(workers)), eff_strong, color='steelblue', alpha=0.8)
        axes[0, 1].axhline(y=100, color='green', linestyle='--', alpha=0.7)
        axes[0, 1].set_xticks(range(len(workers)))
        axes[0, 1].set_xticklabels([str(w) for w in workers])
        axes[0, 1].set_xlabel('Workers')
        axes[0, 1].set_ylabel('Efficiency (%)')
        axes[0, 1].set_title('Strong Scaling: Efficiency')
        axes[0, 1].set_ylim(0, 110)
        axes[0, 1].grid(True, alpha=0.3, axis='y')

    # Weak scaling data
    if os.path.exists(weak_file):
        df_weak = pd.read_csv(weak_file)
        nodes = df_weak['nodes'].values
        times = df_weak['time_ms'].values
        eff_weak = df_weak['relative_efficiency'].values
        orders = df_weak['order'].values

        # Time
        axes[1, 0].plot(nodes, times, 'go-', linewidth=2, markersize=8)
        axes[1, 0].axhline(y=times[0], color='k', linestyle='--', alpha=0.5, label='Ideal')
        for i, (n, t, o) in enumerate(zip(nodes, times, orders)):
            axes[1, 0].annotate(f'G{o}', (n, t), xytext=(0, 8), textcoords='offset points', ha='center')
        axes[1, 0].set_xlabel('Nodes')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].set_title('Weak Scaling: Time')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Weak efficiency
        colors = ['green' if e >= 80 else 'orange' if e >= 50 else 'red' for e in eff_weak]
        axes[1, 1].bar(range(len(nodes)), eff_weak, color=colors, alpha=0.8)
        axes[1, 1].axhline(y=100, color='green', linestyle='--', alpha=0.7)
        axes[1, 1].set_xticks(range(len(nodes)))
        axes[1, 1].set_xticklabels([f'{n}N' for n in nodes])
        axes[1, 1].set_xlabel('Nodes')
        axes[1, 1].set_ylabel('Efficiency (%)')
        axes[1, 1].set_title('Weak Scaling: Efficiency')
        axes[1, 1].set_ylim(0, 120)
        axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Golomb Ruler v7: Scaling Analysis', fontsize=16, y=1.02)
    plt.tight_layout()

    output_file = os.path.join(output_dir, 'scaling_summary.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Combined scaling plot saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate scaling plots')
    parser.add_argument('--strong', default='results/parallel/strong_scaling_G12.csv',
                        help='Strong scaling CSV file')
    parser.add_argument('--weak', default='results/parallel/weak_scaling.csv',
                        help='Weak scaling CSV file')
    parser.add_argument('--output', default='results/plots',
                        help='Output directory for plots')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    if os.path.exists(args.strong):
        plot_strong_scaling(args.strong, args.output)
    else:
        print(f"Strong scaling file not found: {args.strong}")

    if os.path.exists(args.weak):
        plot_weak_scaling(args.weak, args.output)
    else:
        print(f"Weak scaling file not found: {args.weak}")

    if os.path.exists(args.strong) and os.path.exists(args.weak):
        plot_combined_scaling(args.strong, args.weak, args.output)


if __name__ == '__main__':
    main()
