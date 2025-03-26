# QConGen

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![NumPy](https://img.shields.io/badge/numpy-2.2.2-green.svg)](https://numpy.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/badge/ruff-0.9.4-red.svg)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Quantum-based constraint generation optimization algorithm for solving binary linear programming problems, with a focus on set partitioning problems.

## Installation

Install from source:

```sh
git clone https://github.com/yourusername/qcongen.git
cd qcongen
pip install -e .
```

## Usage

QConGen can solve set partitioning problems using either a quantum algorithm (default) or a classical solver (OR-Tools). You can use either pre-defined MPS files or generate random instances.

### Running with Configuration File

QConGen supports both single instance and batch processing modes:

```sh
# Single instance
qcongen run config.json [options]

# Batch processing
qcongen run batch_config.json [options]

# Analyze existing results
qcongen analyze results_directory

# Generate sorted comparison plots
qcongen plot-sorted [--results-dir specific_directory]

# Generate LaTeX tables
qcongen latex-tables [--output-dir output_directory]
```

#### Configuration File Format

1. Single Instance (config.json):
```json
{
    "input_type": "mps",
    "input_file_path": "input/set_partition_mps/problem15_60.mps",
    "sample_size": 1000
}
```

2. Random Instance:
```json
{
    "input_type": "random",
    "sample_size": 1000,
    "random_instance": {
        "n_sets": 15,          # Number of sets (default: 15)
        "n_elements": 25,      # Number of elements (default: 25)
        "n_constraints": 50,   # Number of constraints (default: 50)
        "min_set_size": 1,     # Minimum set size (default: 1)
        "max_set_size": 10,    # Maximum set size (default: 10)
        "min_cost": 1,         # Minimum cost (default: 1)
        "max_cost": 100,       # Maximum cost (default: 100)
        "instance_name": "random"  # Instance name (default: "random")
    }
}
```

3. Batch Processing (batch_config.json):
```json
{
    "batch_name": "experiment1",  # Optional name for the batch
    "configs": [
        {
            "input_type": "random",
            "sample_size": 1000,
            "random_instance": {
                "n_sets": 15,
                "n_elements": 25,
                "instance_name": "random1"
            }
        },
        {
            "input_type": "mps",
            "sample_size": 2000,
            "input_file_path": "input/set_partition_mps/problem15_60.mps"
        }
    ]
}
```

Required fields:
- `input_type`: Type of input ("mps" or "random")
- `sample_size`: Number of samples for quantum algorithm
- `input_file_path`: Path to MPS file (required for "mps" type)
- `random_instance`: Random instance parameters (optional for "random" type, defaults shown above)
- For batch processing: `configs` array containing multiple configurations

### Example Configuration Files

The repository includes example configuration files:
1. `input/config_example.json`: For using pre-defined MPS files
2. `input/random_config_example.json`: For generating random instances
3. `input/batch_config_example.json`: For running multiple instances in batch

### Running Examples

1. Single Random Instance:
```sh
# Generate and solve a random instance with default parameters
qcongen run input/random_config_example.json

# Generate and solve with quantum algorithm and custom parameters
qcongen run input/random_config_example.json --t 0.1 --max-iters 2000

# Generate and solve with classical solver
qcongen run input/random_config_example.json --ref
```

2. Batch Processing:
```sh
# Run multiple instances in batch
qcongen run input/batch_config_example.json

# Run batch with custom parameters
qcongen run input/batch_config_example.json --t 0.1 --max-iters 2000

# Run batch with classical solver
qcongen run input/batch_config_example.json --ref
```

When using random instances:
1. The instance is generated at runtime
2. The MPS file is saved in your results directory as `instance.mps`
3. The problem is solved using either the quantum algorithm or classical solver
4. All results and logs are saved in the same directory

### Running Comparison Analysis

QConGen includes a comparison analysis tool that can run multiple instances and generate performance statistics and plots. This is useful for comparing the performance of different solution methods (Classical, QAOA, and Constraint Generation).

Create a comparison configuration file (e.g., `input/comparison_config_example.json`):
```json
{
    "n_instances": 100,        # Number of random instances to run
    "t": 0.1,                  # Threshold for adding constraints (optional, default: 0.0)
    "max_iters": 1000,         # Maximum iterations per instance (optional, default: 1000)
    "base_instance": {         # Template for generating instances
        "input_type": "random",
        "sample_size": 200,    # Number of quantum measurements
        "random_instance": {   # Optional for random instances
            "n_sets": 15,      # Number of sets (default: 15)
            "n_elements": 50,   # Number of elements (default: 25)
            "min_set_size": 1,
            "max_set_size": 10,
            "min_cost": 1,
            "max_cost": 100,
            "instance_name": "random"
        }
    }
}
```

Then run the analysis:
```sh
qcongen run input/comparison_config_example.json
```

The analysis will:
1. Run the specified number of random instances
2. For each instance:
   - Run classical reference solver (SCIP)
   - Run QAOA reference solution
   - Run constraint generation algorithm
3. Generate comparison plots and statistics

#### Output Structure

```
results/YYYYMMDD_HHMMSS/
├── plots/
│   ├── performance_trend_nonzero.eps
│   ├── performance_trend_nonzero.jpg
│   ├── average_performance_comparison.eps
│   ├── average_performance_comparison.jpg
│   ├── zero_solutions_percentage.eps
│   └── zero_solutions_percentage.jpg
├── instance_1/
│   ├── run.log
│   └── debug.log
├── instance_2/
│   ├── run.log
│   └── debug.log
...
├── comparison_data.csv
└── average_results.txt
```

The analysis generates:
- Performance trend plot with quartiles (non-zero solutions)
- Average performance comparison (all instances and non-zero instances)
- Percentage of zero solutions plot
- Raw comparison data in CSV format
- Summary of average results
- Detailed logs for each instance

### Analyzing Existing Results

You can analyze results from previous runs using:
```sh
qcongen analyze path/to/results/directory
```

This will generate:
1. Performance trend plot with quartiles for non-zero solutions
2. Average performance comparison plots (all instances and non-zero instances)
3. Percentage of zero solutions plot

### Generating Sorted Comparison Plots

To generate sorted comparison plots for all results or a specific directory:
```sh
# Generate for all results
qcongen plot-sorted

# Generate for specific directory
qcongen plot-sorted --results-dir path/to/directory
```

### Generating LaTeX Tables

To generate LaTeX tables for feasible solutions and average solution values:
```sh
qcongen latex-tables [--output-dir output_directory]
```

### Available Problem Instances

The repository includes several pre-defined set partitioning problem instances:
- `problem10_40.mps`: Small instance (10 variables, 40 constraints)
- `problem15_60.mps`: Medium instance (15 variables, 60 constraints)
- `problem30_100.mps`: Large instance (30 variables, 100 constraints)

You can also generate random instances with custom parameters using the configuration file.

### Command Line Options

| Option               | Default Value         | Description |
|---------------------|----------------------|-------------|
| `config_file`       | (required)           | Path to configuration JSON file |
| `--threshold, -t`   | `0.0`                | Threshold for adding constraints |
| `--max-iters, -m`   | `1000`               | Maximum number of iterations |
| `--ibm-token`       | None                 | IBM Quantum token (overrides IBM_TOKEN environment variable) |
| `--ref`             | False                | Use classical reference solver (OR-Tools) instead of quantum algorithm |
| `--compare-ref`     | True                 | Compare with QAOA reference solution using all constraints |

### Output Structure

For single runs:
```