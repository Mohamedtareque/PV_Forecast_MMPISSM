"""
Logging utilities for experiments.

Provides:
- setup_logging: Configure logging for experiments
- get_logger: Get a logger instance
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    log_to_console: bool = True
) -> logging.Logger:
    """
    Set up logging for experiments.
    
    Creates a logger that can write to both file and console.
    Useful for tracking experiment progress and debugging.
    
    Args:
        log_file: Path to log file (optional)
        log_level: Logging level (default INFO)
        log_to_console: Whether to print to console (default True)
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = setup_logging(log_file='experiment.log', log_level=logging.INFO)
        >>> logger.info("Starting experiment...")
    """
    # Create logger
    logger = logging.getLogger('physics_residual_mamba')
    logger.setLevel(log_level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = 'physics_residual_mamba') -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (default 'physics_residual_mamba')
        
    Returns:
        Logger instance
        
    Example:
        >>> logger = get_logger()
        >>> logger.info("Message")
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """
    Logger for tracking experiment metrics and results.
    
    Provides structured logging for:
        - Hyperparameters
        - Training progress
        - Evaluation metrics
        - Model parameters
    """
    
    def __init__(self, experiment_name: str, log_file: Optional[str] = None):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of the experiment
            log_file: Optional path to log file
        """
        self.experiment_name = experiment_name
        self.logger = setup_logging(log_file=log_file)
        self.metrics = {}
        
        self.logger.info(f"=== Starting Experiment: {experiment_name} ===")
    
    def log_hyperparams(self, hyperparams: dict):
        """
        Log hyperparameters.
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        self.logger.info("Hyperparameters:")
        for key, value in hyperparams.items():
            self.logger.info(f"  {key}: {value}")
        self.metrics['hyperparams'] = hyperparams
    
    def log_epoch(self, epoch: int, metrics: dict):
        """
        Log training epoch metrics.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics (loss, rmse, mae, etc.)
        """
        msg = f"Epoch {epoch}: "
        msg += ", ".join([f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                           for k, v in metrics.items()])
        self.logger.info(msg)
        
        # Store metrics
        if 'epochs' not in self.metrics:
            self.metrics['epochs'] = []
        self.metrics['epochs'].append({'epoch': epoch, **metrics})
    
    def log_evaluation(self, fold: int, metrics: dict):
        """
        Log evaluation metrics.
        
        Args:
            fold: Fold number
            metrics: Dictionary of evaluation metrics
        """
        self.logger.info(f"Fold {fold} Evaluation:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        # Store metrics
        if 'folds' not in self.metrics:
            self.metrics['folds'] = []
        self.metrics['folds'].append({'fold': fold, **metrics})
    
    def log_final_results(self, results: dict):
        """
        Log final experiment results.
        
        Args:
            results: Dictionary of final results
        """
        self.logger.info("=== Final Results ===")
        for key, value in results.items():
            self.logger.info(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
        self.metrics['final'] = results
        
        self.logger.info("=== Experiment Complete ===")
    
    def get_metrics(self) -> dict:
        """
        Get all logged metrics.
        
        Returns:
            Dictionary with all metrics
        """
        return self.metrics
    
    def save_metrics(self, filepath: str):
        """
        Save metrics to JSON file.
        
        Args:
            filepath: Path to save metrics
        """
        import json
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {filepath}")
