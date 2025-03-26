"""Script for running multiple random instances and collecting comparison data."""
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
from qcongen.io.config_reader import QConGenConfig, RandomInstanceConfig
from qcongen.engine.runner import run_solver
from qcongen.io.output_writer import create_output_directory
from qcongen.utils.generators import generate_set_partition_instance

logger = logging.getLogger('qcongen')
@dataclass
class ComparisonResult:
    """Results from a single instance comparison."""
    instance_id: int
    classical_value: float
    qaoa_value: float
    constraint_gen_value: float
    
    @property
    def qaoa_percentage(self) -> float:
        """Calculate how close QAOA is to classical (classical/qaoa * 100)."""
        if self.qaoa_value == 0 or self.classical_value == 0:
            return 0.0
        return (self.classical_value / self.qaoa_value) * 100
    
    @property
    def constraint_gen_percentage(self) -> float:
        """Calculate how close constraint generation is to classical (classical/constraint_gen * 100)."""
        if self.constraint_gen_value == 0 or self.classical_value == 0:
            return 0.0
        return (self.classical_value / self.constraint_gen_value) * 100
def read_comparison_config(config_path: str | Path) -> dict[str, Any]:
    """Read the comparison configuration file.
    
    Args:
        config_path: Path to the comparison config file
        
    Returns:
        dict containing the configuration
    """
    with open(config_path) as f:
        config = json.load(f)
    
    
    required_fields = ["n_instances", "base_instance"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")
            
    
    base_instance = config["base_instance"]
    if "input_type" not in base_instance:
        raise ValueError("Missing required field: input_type in base_instance")
    if "sample_size" not in base_instance:
        raise ValueError("Missing required field: sample_size in base_instance")
        
    return config
def run_comparison(
    n_instances: int,
    base_config: QConGenConfig,
    t: float,
    max_iters: int,
    output_dir: Path | None = None,
) -> list[ComparisonResult]:
    """Run multiple random instances and collect comparison data."""
    if output_dir is None:
        output_dir = create_output_directory()
        
    results: list[ComparisonResult] = []
    
    for i in range(n_instances):
        instance_dir = output_dir / f"instance_{i+1}"
        instance_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\nRunning instance {i+1}/{n_instances}")
        
        
        instance_config = QConGenConfig(
            input_type=base_config.input_type,
            sample_size=base_config.sample_size,
            random_instance=RandomInstanceConfig(
                n_sets=base_config.random_instance.n_sets,
                n_elements=base_config.random_instance.n_elements,
                min_set_size=base_config.random_instance.min_set_size,
                max_set_size=base_config.random_instance.max_set_size,
                min_cost=base_config.random_instance.min_cost,
                max_cost=base_config.random_instance.max_cost,
                instance_name=f"random_{i+1}"
            )
        )
        
        
        blp = generate_set_partition_instance(
            n_sets=instance_config.random_instance.n_sets,
            n_elements=instance_config.random_instance.n_elements,
            min_set_size=instance_config.random_instance.min_set_size,
            max_set_size=instance_config.random_instance.max_set_size,
            min_cost=instance_config.random_instance.min_cost,
            max_cost=instance_config.random_instance.max_cost,
            instance_name=instance_config.random_instance.instance_name,
            output_dir=instance_dir,
        )
        instance_path = instance_dir / f"{instance_config.random_instance.instance_name}.mps"
        
        
        ref_success, _, ref_value = run_solver(
            solver_type="classical",
            blp=blp,
            config=instance_config,
            instance_path=instance_path,
            output_dir=instance_dir,
        )
        classical_value = ref_value if ref_success else 0.0
        
        
        qaoa_success, _, qaoa_value = run_solver(
            solver_type="qaoa",
            blp=blp,
            config=instance_config,
            output_dir=instance_dir,
        )
        qaoa_value = qaoa_value if qaoa_success else 0.0
        
        
        cg_success, _, cg_value = run_solver(
            solver_type="constraint_gen",
            blp=blp,
            config=instance_config,
            t=t,
            max_iters=max_iters,
            output_dir=instance_dir,
        )
        cg_value = cg_value if cg_success else 0.0
        
        
        results.append(ComparisonResult(
            instance_id=i+1,
            classical_value=classical_value,
            qaoa_value=qaoa_value,
            constraint_gen_value=cg_value,
        ))
        
        
        logger.info(f"\nInstance {i+1} results:")
        logger.info(f"  Classical value: {classical_value}")
        logger.info(f"  QAOA value: {qaoa_value}")
        logger.info(f"  Constraint generation value: {cg_value}")
        
    return results
def plot_comparison(results: list[ComparisonResult], output_dir: Path) -> None:
    """Create plots comparing the performance of different methods."""
    
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    
    dark_blue = '#1f77b4'
    orange = '#ff7f0e'
    
    
    instance_ids = [r.instance_id for r in results]
    qaoa_percentages = [r.qaoa_percentage for r in results]
    cg_percentages = [r.constraint_gen_percentage for r in results]
    
    
    z_qaoa = np.polyfit(instance_ids, qaoa_percentages, 1)
    p_qaoa = np.poly1d(z_qaoa)
    z_cg = np.polyfit(instance_ids, cg_percentages, 1)
    p_cg = np.poly1d(z_cg)
    
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    
    
    plt.scatter(instance_ids, qaoa_percentages, color=dark_blue, alpha=0.3, s=30, label='QAOA reference')
    plt.scatter(instance_ids, cg_percentages, color=orange, alpha=0.3, s=30, label='Constraint generation')
    
    
    plt.plot(instance_ids, p_qaoa(instance_ids), color=dark_blue, linewidth=2, label='QAOA trend')
    plt.plot(instance_ids, p_cg(instance_ids), color=orange, linewidth=2, label='Constraint gen trend')
    plt.axhline(y=100, color='g', linestyle='--', label='Classical reference')
    
    plt.xlabel('Instance')
    plt.ylabel('Relative performance (%)')
    plt.title('Performance over instances')
    plt.legend()
    plt.grid(True)
    
    
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    
    avg_qaoa = np.mean(qaoa_percentages)
    avg_cg = np.mean(cg_percentages)
    
    
    plt.subplot(1, 2, 2)
    methods = ['QAOA reference', 'Constraint generation']
    averages = [avg_qaoa, avg_cg]
    
    bars = plt.bar(methods, averages, color=[dark_blue, orange])
    plt.axhline(y=100, color='g', linestyle='--', label='Classical reference')
    
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.ylabel('Average performance (%)')
    plt.title('Average performance comparison')
    plt.grid(True, axis='y')
    
    
    plt.tight_layout()
    
    
    plt.savefig(plots_dir / 'comparison_combined.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'comparison_combined.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    
    plt.figure(figsize=(8, 6))
    
    
    plt.scatter(instance_ids, qaoa_percentages, color=dark_blue, alpha=0.3, s=30, label='QAOA reference')
    plt.scatter(instance_ids, cg_percentages, color=orange, alpha=0.3, s=30, label='Constraint generation')
    
    
    plt.plot(instance_ids, p_qaoa(instance_ids), color=dark_blue, linewidth=2, label='QAOA trend')
    plt.plot(instance_ids, p_cg(instance_ids), color=orange, linewidth=2, label='Constraint gen trend')
    plt.axhline(y=100, color='g', linestyle='--', label='Classical reference')
    
    plt.xlabel('Instance')
    plt.ylabel('Relative performance (%)')
    plt.title('Performance over instances')
    plt.legend()
    plt.grid(True)
    
    
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'performance_over_instances.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'performance_over_instances.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(methods, averages, color=[dark_blue, orange])
    plt.axhline(y=100, color='g', linestyle='--', label='Classical reference')
    
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom')
    
    plt.ylabel('Average performance (%)')
    plt.title('Average performance comparison')
    plt.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig(plots_dir / 'average_performance.eps', format='eps', dpi=300, bbox_inches='tight')
    plt.savefig(plots_dir / 'average_performance.jpg', format='jpg', dpi=300, bbox_inches='tight')
    plt.close()
    
    
    with open(output_dir / 'comparison_data.csv', 'w') as f:
        f.write('Instance,Classical,QAOA,ConstraintGen,QAOA_Percentage,CG_Percentage\n')
        for r in results:
            f.write(f'{r.instance_id},{r.classical_value},{r.qaoa_value},{r.constraint_gen_value},'
                   f'{r.qaoa_percentage},{r.constraint_gen_percentage}\n')
            
    
    with open(output_dir / 'average_results.txt', 'w') as f:
        f.write(f'Average QAOA Performance: {avg_qaoa:.1f}%\n')
        f.write(f'Average Constraint Generation Performance: {avg_cg:.1f}%\n')
def main(config_path: str | Path) -> None:
    """Main entry point for comparison analysis.
    
    Args:
        config_path: Path to the comparison configuration file
    """
    
    config = read_comparison_config(config_path)
    
    
    base_instance = config["base_instance"]
    base_config = QConGenConfig(
        input_type=base_instance["input_type"],
        sample_size=base_instance["sample_size"],
    )
    
    
    if "random_instance" in base_instance:
        random_config = base_instance["random_instance"]
        base_config.random_instance = RandomInstanceConfig(
            n_sets=random_config.get("n_sets", 15),
            n_elements=random_config.get("n_elements", 25),
            min_set_size=random_config.get("min_set_size", 1),
            max_set_size=random_config.get("max_set_size", 10),
            min_cost=random_config.get("min_cost", 1),
            max_cost=random_config.get("max_cost", 100),
            instance_name=random_config.get("instance_name", "random")
        )
    
    
    results = run_comparison(
        n_instances=config["n_instances"],
        base_config=base_config,
        t=config.get("t", 0.0),
        max_iters=config.get("max_iters", 1000),
    )
    
    
    output_dir = create_output_directory()
    plot_comparison(results, output_dir)
    
    logger.info(f"\nComparison analysis completed. Results saved in: {output_dir}")
if __name__ == "__main__":
    import sys
    main(sys.argv[1]) 