"""Module for creating sorted double plots from comparison data."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import glob
import os

logger = logging.getLogger('qcongen')
def create_sorted_plots(results_dir: Path) -> None:
    """Create sorted double plots for QAOA reference and constraint generation.
    
    Args:
        results_dir: Directory containing comparison_data.csv
    """
    csv_path = results_dir / 'comparison_data.csv'
    if not csv_path.exists():
        raise FileNotFoundError(f"No comparison data found in {results_dir}")
    
    
    plots_dir = results_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    
    df = pd.read_csv(csv_path)
    
    
    dark_blue = '#1f77b4'
    orange = '#ff7f0e'
    
    
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 22,
        'axes.labelsize': 20,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
    })
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    
    qaoa_data = df['QAOA_Percentage'].copy()
    qaoa_sorted = np.sort(qaoa_data)  
    
    
    cg_data = df['CG_Percentage'].copy()
    cg_sorted = np.sort(cg_data)  
    
    
    x_qaoa = np.arange(len(qaoa_sorted))
    x_cg = np.arange(len(cg_sorted))
    
    
    ax1.plot(x_qaoa, qaoa_sorted, marker='o', linestyle='-', color=dark_blue, linewidth=2, markersize=8)
    ax1.axhline(y=100, color='g', linestyle='--', label='Classical reference', linewidth=2)
    ax1.set_xlabel('Instance (sorted)')
    ax1.set_ylabel('Relative solution value (%)')
    ax1.set_title('QAOA Reference')
    ax1.grid(True)
    ax1.legend(loc='best', frameon=True, fancybox=True)
    
    
    ax2.plot(x_cg, cg_sorted, marker='o', linestyle='-', color=orange, linewidth=2, markersize=8)
    ax2.axhline(y=100, color='g', linestyle='--', label='Classical reference', linewidth=2)
    ax2.set_xlabel('Instance (sorted)')
    ax2.set_ylabel('Relative solution value (%)')
    ax2.set_title('Constraint Generation')
    ax2.grid(True)
    ax2.legend(loc='best', frameon=True, fancybox=True)
    
    
    ax1.set_ylim(bottom=0)
    ax2.set_ylim(bottom=0)
    
    
    ax1_max = max(110, np.ceil(qaoa_sorted.max() * 1.1) if len(qaoa_sorted) > 0 and qaoa_sorted.max() > 0 else 110)
    ax2_max = max(110, np.ceil(cg_sorted.max() * 1.1) if len(cg_sorted) > 0 and cg_sorted.max() > 0 else 110)
    
    ax1.set_ylim(top=ax1_max)
    ax2.set_ylim(top=ax2_max)
    
    
    plt.subplots_adjust(wspace=0.3)
    
    plt.tight_layout()
    
    
    plt.savefig(plots_dir / 'sorted.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'sorted.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Sorted plots created and saved in: {plots_dir}")
def process_all_results_folders() -> None:
    """Process all results folders in the results_plots directory."""
    
    base_dir = Path.cwd() / 'results_plots'
    
    if not base_dir.exists():
        logger.error(f"Results directory not found: {base_dir}")
        return
    
    
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        logger.error(f"No result folders found in {base_dir}")
        return
    
    logger.info(f"Found {len(subdirs)} result folders to process")
    
    
    for subdir in subdirs:
        try:
            logger.info(f"Processing {subdir.name}...")
            create_sorted_plots(subdir)
        except Exception as e:
            logger.error(f"Error processing {subdir.name}: {str(e)}")
    
    logger.info("All folders processed successfully")
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    process_all_results_folders() 