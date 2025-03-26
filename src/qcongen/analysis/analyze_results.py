"""Module for analyzing existing comparison results."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger('qcongen')
def plot_trend_with_quartiles(df: pd.DataFrame, output_dir: Path) -> None:
    """Create trend plot with quartiles for non-zero solutions."""
    
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    
    dark_blue = '#1f77b4'  # Dark blue
    orange = '#ff7f0e'     # Orange
    
    
    df_nonzero = df[df[['QAOA_Percentage', 'CG_Percentage']].ne(0).all(axis=1)]
    
    if df_nonzero.empty:
        logger.warning("No non-zero data points found for trend analysis")
        return
    
    
    window = 5
    qaoa_rolling = df_nonzero['QAOA_Percentage'].rolling(window=window, center=True)
    cg_rolling = df_nonzero['CG_Percentage'].rolling(window=window, center=True)
    
    
    qaoa_mean = qaoa_rolling.mean()
    qaoa_q1 = qaoa_rolling.quantile(0.25)
    qaoa_q3 = qaoa_rolling.quantile(0.75)
    
    cg_mean = cg_rolling.mean()
    cg_q1 = cg_rolling.quantile(0.25)
    cg_q3 = cg_rolling.quantile(0.75)
    
    
    plt.figure(figsize=(10, 6))
    
    
    plt.plot(df_nonzero.index, qaoa_mean, color=dark_blue, label='QAOA trend', linewidth=2)
    plt.fill_between(df_nonzero.index, qaoa_q1, qaoa_q3, color=dark_blue, alpha=0.2)
    
    plt.plot(df_nonzero.index, cg_mean, color=orange, label='Constraint gen trend', linewidth=2)
    plt.fill_between(df_nonzero.index, cg_q1, cg_q3, color=orange, alpha=0.2)
    
    plt.axhline(y=100, color='g', linestyle='--', label='Classical reference')
    
    plt.xlabel('Instance')
    plt.ylabel('Relative performance (%)')
    plt.title('Performance over instances (non-zero solutions)')
    plt.legend()
    plt.grid(True)
    
    
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'performance_trend_nonzero.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'performance_trend_nonzero.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.close()
def plot_averages(df: pd.DataFrame, output_dir: Path) -> None:
    """Create two average performance plots: one with all data, one with non-zero only."""
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    dark_blue = '#1f77b4'  # Dark blue
    orange = '#ff7f0e'     # Orange
    
    
    qaoa_stats = df['QAOA_Percentage'].describe()
    cg_stats = df['CG_Percentage'].describe()
    
    
    df_nonzero = df[df[['QAOA_Percentage', 'CG_Percentage']].ne(0).all(axis=1)]
    qaoa_nonzero_stats = df_nonzero['QAOA_Percentage'].describe()
    cg_nonzero_stats = df_nonzero['CG_Percentage'].describe()
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    
    methods = ['QAOA reference', 'Constraint generation']
    means = [qaoa_stats['mean'], cg_stats['mean']]
    
    bars1 = ax1.bar(methods, means, color=[dark_blue, orange])
    ax1.axhline(y=100, color='g', linestyle='--', label='Classical reference')
    
    
    ax1.vlines([0], qaoa_stats['25%'], qaoa_stats['75%'], color=dark_blue)
    ax1.vlines([1], cg_stats['25%'], cg_stats['75%'], color=orange)
    
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    ax1.set_ylabel('Average performance (%)')
    ax1.set_title('Average performance (all instances)')
    ax1.grid(True, axis='y')
    
    
    means_nonzero = [qaoa_nonzero_stats['mean'], cg_nonzero_stats['mean']]
    
    bars2 = ax2.bar(methods, means_nonzero, color=[dark_blue, orange])
    ax2.axhline(y=100, color='g', linestyle='--', label='Classical reference')
    
    
    ax2.vlines([0], qaoa_nonzero_stats['25%'], qaoa_nonzero_stats['75%'], color=dark_blue)
    ax2.vlines([1], cg_nonzero_stats['25%'], cg_nonzero_stats['75%'], color=orange)
    
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    ax2.set_ylabel('Average performance (%)')
    ax2.set_title('Average performance (non-zero instances)')
    ax2.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'average_performance_comparison.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'average_performance_comparison.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.close()
def plot_zero_solutions_percentage(df: pd.DataFrame, output_dir: Path) -> None:
    """Create plot showing percentage of zero solutions."""
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    dark_blue = '#1f77b4'
    orange = '#ff7f0e'
    
    
    total_instances = len(df)
    qaoa_zeros = (df['QAOA_Percentage'] == 0).sum() / total_instances * 100
    cg_zeros = (df['CG_Percentage'] == 0).sum() / total_instances * 100
    
    plt.figure(figsize=(8, 6))
    
    methods = ['QAOA reference', 'Constraint generation']
    percentages = [qaoa_zeros, cg_zeros]
    
    bars = plt.bar(methods, percentages, color=[dark_blue, orange])
    
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
    
    plt.ylabel('Percentage of zero solutions (%)')
    plt.title('Percentage of instances with no solution')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'zero_solutions_percentage.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'zero_solutions_percentage.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.close()
def analyze_results(results_dir: Path) -> None:
    """Analyze results from a previous comparison run.
    
    Args:
        results_dir: Directory containing comparison_data.csv
    """
    csv_path = results_dir / 'comparison_data.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"No comparison data found in {results_dir}")
    
    
    df = pd.read_csv(csv_path)
    
    
    plot_trend_with_quartiles(df, results_dir)
    plot_averages(df, results_dir)
    plot_zero_solutions_percentage(df, results_dir)
    
    logger.info(f"Analysis completed. Results saved in: {results_dir}/plots") 