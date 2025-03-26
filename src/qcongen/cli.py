"""Command line interface for QConGen."""

import argparse
import logging
import sys
from pathlib import Path
from typing import NoReturn

from qcongen.config import initialize_config
from qcongen.io.config_reader import read_config, BatchConfig
from qcongen.io.config_reader import setup_batch_run
from qcongen.engine.runner import run_single_instance
from qcongen.analysis.run_comparison import main as run_comparison_main
from qcongen.analysis.analyze_results import analyze_results
from qcongen.analysis.plot_sorted_comparison import process_all_results_folders, create_sorted_plots
from qcongen.analysis.generate_latex_tables import generate_latex_tables

logger = logging.getLogger('qcongen')

def main() -> NoReturn:
    """Main entry point for QConGen CLI."""
    parser = argparse.ArgumentParser(description="QConGen: Quantum Constraint Generation Algorithm")
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    run_parser = subparsers.add_parser('run', help='Run QConGen algorithm')
    run_parser.add_argument("config_file", type=Path, help="Path to configuration JSON file")
    run_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.0,
        help="Threshold for adding constraints (default: 0.0)",
    )
    run_parser.add_argument(
        "--max-iters",
        "-m",
        type=int,
        default=1000,
        help="Maximum number of iterations (default: 1000)",
    )
    run_parser.add_argument(
        "--ibm-token",
        type=str,
        help="IBM Quantum token (overrides IBM_TOKEN environment variable)",
    )
    run_parser.add_argument(
        "--ref",
        action="store_true",
        help="Use classical reference solver (OR-Tools) instead of quantum algorithm",
    )
    run_parser.add_argument(
        "--compare-ref",
        "-c",
        action="store_true",
        default=True,
        help="Compare with QAOA reference solution using all constraints (default: True)",
    )
    
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing results')
    analyze_parser.add_argument("results_dir", type=Path, help="Path to results directory containing comparison_data.csv")

    plot_sorted_parser = subparsers.add_parser('plot-sorted', help='Generate sorted double plots for all results')
    plot_sorted_parser.add_argument(
        "--results-dir", 
        type=Path, 
        help="Path to specific results directory (if not provided, all directories in results_plots will be processed)"
    )
    
    latex_tables_parser = subparsers.add_parser('latex-tables', help='Generate LaTeX tables for feasible solutions and average solution values')
    latex_tables_parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path.cwd() / 'results_plots',
        help="Directory to save the LaTeX tables (default: results_plots)"
    )

    args = parser.parse_args()
    
    if args.command == 'analyze':
        try:
            analyze_results(args.results_dir)
            sys.exit(0)
        except Exception as e:
            print(f"Error analyzing results: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    if args.command == 'plot-sorted':
        try:
            if args.results_dir:
                create_sorted_plots(args.results_dir)
                print(f"Sorted plots created for {args.results_dir}")
            else:
                process_all_results_folders()
                print("Sorted plots created for all results folders")
            sys.exit(0)
        except Exception as e:
            print(f"Error creating sorted plots: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    if args.command == 'latex-tables':
        try:
            generate_latex_tables(args.output_dir)
            print(f"LaTeX tables generated and saved in: {args.output_dir}")
            sys.exit(0)
        except Exception as e:
            print(f"Error generating LaTeX tables: {str(e)}", file=sys.stderr)
            sys.exit(1)
    
    try:
        with open(args.config_file) as f:
            import json
            config_data = json.load(f)
            
        if "n_instances" in config_data and "base_instance" in config_data:
            run_comparison_main(str(args.config_file))
            sys.exit(0)
            
        config = read_config(args.config_file)

        initialize_config(ibm_token=args.ibm_token)

        if isinstance(config, BatchConfig):
            batch_dir = setup_batch_run(config)
            logger.info(f"\nStarting batch run in {batch_dir}")

            for i, run_config in enumerate(config.configs, 1):
                run_dir = batch_dir / str(i)
                logger.info(f"\nRunning instance {i}/{len(config.configs)}")
                
                success, solution, value = run_single_instance(
                    config=run_config,
                    t=args.t,
                    max_iters=args.max_iters,
                    use_ref=args.ref,
                    compare_ref=args.compare_ref,
                    output_dir=run_dir,
                )

                if success:
                    print(f"\nInstance {i}: Found solution with value: {value}")
                    print(f"Solution: {solution}")
                else:
                    print(f"\nInstance {i}: No feasible solution found")

            print(f"\nBatch run completed. Results saved in: {batch_dir}")
        else:
            success, solution, value = run_single_instance(
                config=config,
                t=args.t,
                max_iters=args.max_iters,
                use_ref=args.ref,
                compare_ref=args.compare_ref,
            )

            if success:
                print(f"\nFound solution with value: {value}")
                print(f"Solution: {solution}")
            else:
                print("\nNo feasible solution found")

        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
