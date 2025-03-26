"""Logging utilities for QConGen."""

import logging
from pathlib import Path


def setup_logging(log_dir: Path | None = None) -> Path:
    """Setup logging configuration.
    
    Args:
        log_dir: Directory to store log files (optional)
        
    Returns:
        Path: Path to the log directory
    """
    if log_dir is None:
        from qcongen.io.output_writer import create_output_directory
        log_dir = create_output_directory()
    
    run_formatter = logging.Formatter('%(message)s')
    debug_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger('qcongen')
    logger.setLevel(logging.DEBUG)
    
    logger.handlers.clear()
    
    run_handler = logging.FileHandler(log_dir / 'run.log')
    run_handler.setLevel(logging.INFO)
    run_handler.setFormatter(run_formatter)
    run_handler.addFilter(lambda record: record.levelno in [logging.INFO, logging.WARNING, logging.ERROR])
    logger.addHandler(run_handler)
    
    debug_handler = logging.FileHandler(log_dir / 'debug.log')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(debug_formatter)
    logger.addHandler(debug_handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(debug_formatter)
    logger.addHandler(console_handler)
    
    qiskit_logger = logging.getLogger('qiskit')
    qiskit_logger.setLevel(logging.WARNING)
    
    qiskit_debug_handler = logging.FileHandler(log_dir / 'debug.log')
    qiskit_debug_handler.setLevel(logging.WARNING)
    qiskit_debug_handler.setFormatter(debug_formatter)
    qiskit_logger.addHandler(qiskit_debug_handler)
    
    qiskit_console_handler = logging.StreamHandler()
    qiskit_console_handler.setLevel(logging.WARNING)
    qiskit_console_handler.setFormatter(debug_formatter)
    qiskit_logger.addHandler(qiskit_console_handler)
    
    return log_dir
