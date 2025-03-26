"""Configuration reader module for QConGen."""

import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

InputType: TypeAlias = Literal["mps", "random"]

@dataclass
class RandomInstanceConfig:
    """Configuration for random instance generation."""
    n_sets: int = 15
    n_elements: int = 25
    min_set_size: int = 1
    max_set_size: int = 10
    min_cost: int = 1
    max_cost: int = 100
    instance_name: str = "random"

@dataclass
class QConGenConfig:
    """Configuration for QConGen."""
    input_type: InputType
    sample_size: int
    input_file_path: str | None = None
    random_instance: RandomInstanceConfig | None = None

    @property
    def input_file_path_resolved(self) -> Path | None:
        """Get the resolved input file path."""
        return Path(self.input_file_path).resolve() if self.input_file_path else None

@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    configs: list[QConGenConfig]
    batch_name: str | None = None

    @property
    def batch_dir(self) -> str:
        """Get the batch directory name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"batch_{self.batch_name}_{timestamp}" if self.batch_name else f"batch_{timestamp}"

def validate_config(config_data: dict[str, Any]) -> None:
    """Validate the configuration data."""
    if "configs" in config_data:
        if not isinstance(config_data["configs"], list):
            raise ValueError("configs must be a list")
        for cfg in config_data["configs"]:
            validate_single_config(cfg)
        return

    validate_single_config(config_data)

def validate_single_config(config_data: dict[str, Any]) -> None:
    """Validate a single configuration."""
    required_fields = {
        "input_type": str,
        "sample_size": int,
    }
    
    for field, field_type in required_fields.items():
        if field not in config_data:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(config_data[field], field_type):
            raise ValueError(f"Invalid type for {field}. Expected {field_type}")
    
    if config_data["input_type"] not in ["mps", "random"]:
        raise ValueError("input_type must be either 'mps' or 'random'")
    
    if config_data["sample_size"] <= 0:
        raise ValueError("sample_size must be positive")
    
    if config_data["input_type"] == "mps":
        if "input_file_path" not in config_data:
            raise ValueError("input_file_path is required for mps input type")
        if not Path(config_data["input_file_path"]).exists():
            raise FileNotFoundError(f"Input file not found: {config_data['input_file_path']}")
    
    if "random_instance" in config_data:
        ri = config_data["random_instance"]
        if not isinstance(ri, dict):
            raise ValueError("random_instance must be a dictionary")
        
        type_checks = {
            "n_sets": int,
            "n_elements": int,
            "min_set_size": int,
            "max_set_size": int,
            "min_cost": int,
            "max_cost": int,
            "instance_name": str,
        }
        
        for field, field_type in type_checks.items():
            if field in ri and not isinstance(ri[field], field_type):
                raise ValueError(f"Invalid type for random_instance.{field}. Expected {field_type}")

def read_config(config_path: str | Path) -> BatchConfig | QConGenConfig:
    """Read and validate configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        BatchConfig | QConGenConfig: Validated configuration object
        
    Example config.json for single run:
    {
        "input_type": "random",  # or "mps"
        "sample_size": 1000,
        "input_file_path": "input/set_partition_mps/problem.mps",  # required for mps type
        "random_instance": {  # optional, used for random type
            "n_sets": 15,
            "n_elements": 25,
            "instance_name": "random"
        }
    }

    Example config.json for batch run:
    {
        "batch_name": "experiment1",  # optional
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
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path) as f:
        config_data = json.load(f)
    
    validate_config(config_data)
    
    if "configs" in config_data:
        configs = []
        for cfg in config_data["configs"]:
            random_instance = None
            if "random_instance" in cfg:
                random_instance = RandomInstanceConfig(**{
                    **RandomInstanceConfig().__dict__,
                    **cfg["random_instance"]
                })
            
            configs.append(QConGenConfig(
                input_type=cfg["input_type"],
                sample_size=cfg["sample_size"],
                input_file_path=cfg.get("input_file_path"),
                random_instance=random_instance
            ))
        
        return BatchConfig(
            configs=configs,
            batch_name=config_data.get("batch_name")
        )
    
    random_instance = None
    if "random_instance" in config_data:
        random_instance = RandomInstanceConfig(**{
            **RandomInstanceConfig().__dict__,
            **config_data["random_instance"]
        })
    
    return QConGenConfig(
        input_type=config_data["input_type"],
        sample_size=config_data["sample_size"],
        input_file_path=config_data.get("input_file_path"),
        random_instance=random_instance
    )

def setup_batch_run(batch_config: BatchConfig) -> Path:
    """Create directory structure for batch run.
    
    Args:
        batch_config: Batch configuration
        
    Returns:
        Path: Path to the batch directory
    """
    base_dir = Path("results")
    batch_dir = base_dir / batch_config.batch_dir
    batch_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(len(batch_config.configs)):
        run_dir = batch_dir / str(i + 1)
        run_dir.mkdir(exist_ok=True)
        
        config = batch_config.configs[i]
        config_dict = {
            "input_type": config.input_type,
            "sample_size": config.sample_size,
            "input_file_path": config.input_file_path,
            "random_instance": config.random_instance.__dict__ if config.random_instance else None
        }
        with open(run_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=4)
    
    return batch_dir 