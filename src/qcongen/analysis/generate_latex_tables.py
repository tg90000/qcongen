"""Module for generating LaTeX tables from comparison data."""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import os
from typing import Dict, List

logger = logging.getLogger('qcongen')
def generate_latex_tables(output_dir: Path) -> None:
    """Generate LaTeX tables for feasible solutions percentage and average solution value.
    
    Args:
        output_dir: Directory to save the LaTeX tables
    """
    
    output_dir.mkdir(exist_ok=True)
    
    
    base_dir = Path.cwd() / 'results_plots'
    
    if not base_dir.exists():
        logger.error(f"Results directory not found: {base_dir}")
        return
    
    
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        logger.error(f"No result folders found in {base_dir}")
        return
    
    
    subdirs = sorted(subdirs, key=lambda x: x.name)
    
    
    problem_names: List[str] = []
    feasible_solutions: Dict[str, List[float]] = {'QAOA Reference': [], 'Constraint Generation': []}
    average_values: Dict[str, List[float]] = {'QAOA Reference': [], 'Constraint Generation': []}
    
    
    for i, subdir in enumerate(subdirs, 1):
        problem_name = f"Problem {i}"
        problem_names.append(problem_name)
        
        csv_path = subdir / 'comparison_data.csv'
        if not csv_path.exists():
            logger.error(f"No comparison data found in {subdir}")
            continue
        
        
        df = pd.read_csv(csv_path)
        
        
        total_instances = len(df)
        qaoa_feasible = (df['QAOA_Percentage'] > 0).sum() / total_instances * 100
        cg_feasible = (df['CG_Percentage'] > 0).sum() / total_instances * 100
        
        feasible_solutions['QAOA Reference'].append(qaoa_feasible)
        feasible_solutions['Constraint Generation'].append(cg_feasible)
        
        
        qaoa_avg = df['QAOA_Percentage'].mean()
        cg_avg = df['CG_Percentage'].mean()
        
        average_values['QAOA Reference'].append(qaoa_avg)
        average_values['Constraint Generation'].append(cg_avg)
    
    
    feasible_table = "\\begin{table}[h]\n"
    feasible_table += "\\centering\n"
    feasible_table += "\\caption{Percentage of Feasible Solutions (\\%)}\n"
    feasible_table += "\\begin{tabular}{l" + "c" * len(problem_names) + "}\n"
    feasible_table += "\\hline\n"
    feasible_table += "Method & " + " & ".join(problem_names) + " \\\\\n"
    feasible_table += "\\hline\n"
    
    for method, values in feasible_solutions.items():
        formatted_values = [f"{value:.2f}" for value in values]
        feasible_table += f"{method} & " + " & ".join(formatted_values) + " \\\\\n"
    
    feasible_table += "\\hline\n"
    feasible_table += "\\end{tabular}\n"
    feasible_table += "\\end{table}"
    
    
    average_table = "\\begin{table}[h]\n"
    average_table += "\\centering\n"
    average_table += "\\caption{Average Solution Value (\\%)}\n"
    average_table += "\\begin{tabular}{l" + "c" * len(problem_names) + "}\n"
    average_table += "\\hline\n"
    average_table += "Method & " + " & ".join(problem_names) + " \\\\\n"
    average_table += "\\hline\n"
    
    for method, values in average_values.items():
        formatted_values = [f"{value:.2f}" for value in values]
        average_table += f"{method} & " + " & ".join(formatted_values) + " \\\\\n"
    
    average_table += "\\hline\n"
    average_table += "\\end{tabular}\n"
    average_table += "\\end{table}"
    
    
    with open(output_dir / 'feasible_solutions_table.tex', 'w') as f:
        f.write(feasible_table)
    
    with open(output_dir / 'average_solution_value_table.tex', 'w') as f:
        f.write(average_table)
    
    logger.info(f"LaTeX tables generated and saved in: {output_dir}")
if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)
    
    
    output_dir = Path.cwd() / 'results_plots'
    generate_latex_tables(output_dir) 