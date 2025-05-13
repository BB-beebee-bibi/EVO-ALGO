"""
Centralized Configuration System for the TRISOLARIS framework.

This module provides a unified configuration system for all components of the
TRISOLARIS framework, ensuring consistent parameter handling, validation,
and hierarchical configuration management.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union, Set, TypeVar, Type, cast
from pathlib import Path
from dataclasses import dataclass, field, asdict
import copy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Type variable for configuration classes
T = TypeVar('T', bound='BaseConfig')

class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails."""
    pass

@dataclass
class ResourceLimits:
    """Resource limits configuration."""
    max_cpu_percent: float = 75.0
    max_memory_percent: float = 75.0
    max_execution_time: float = 60.0
    check_interval: float = 0.5

@dataclass
class ResourceSchedulerConfig:
    """Resource scheduler configuration."""
    target_cpu_usage: float = 70.0
    target_memory_usage: float = 70.0
    min_cpu_available: float = 15.0
    min_memory_available: float = 15.0
    check_interval: float = 1.0
    adaptive_batch_size: bool = True
    initial_batch_size: int = 10

@dataclass
class SandboxConfig:
    """Sandbox environment configuration."""
    base_dir: Optional[str] = None
    resource_limits: ResourceLimits = field(default_factory=ResourceLimits)
    preserve_sandbox: bool = False

@dataclass
class EthicalBoundaryConfig:
    """Ethical boundary configuration."""
    use_post_evolution: bool = True
    boundaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    allowed_imports: Set[str] = field(default_factory=lambda: {
        "typing", "collections", "datetime", "math", "random", "re", "time", "json", "sys", "os"
    })

@dataclass
class EvolutionConfig:
    """Evolution engine configuration."""
    population_size: int = 100
    selection_pressure: float = 0.7
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elitism_ratio: float = 0.1
    parallel_evaluation: bool = True
    max_workers: Optional[int] = None
    use_caching: bool = True
    early_stopping: bool = False
    early_stopping_generations: int = 5
    early_stopping_threshold: float = 0.01
    resource_aware: bool = True
    use_islands: bool = False
    islands: int = 1
    migration_interval: int = 5

@dataclass
class TaskConfig:
    """Task-specific configuration."""
    name: str = ""
    description: str = ""
    template_path: Optional[str] = None
    fitness_weights: Dict[str, float] = field(default_factory=lambda: {
        "functionality": 0.7,
        "efficiency": 0.2,
        "alignment": 0.1
    })
    allowed_imports: List[str] = field(default_factory=lambda: [
        "os", "sys", "time", "random", "math", "json", 
        "datetime", "collections", "re", "logging"
    ])
    evolution_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BaseConfig:
    """Base configuration class for TRISOLARIS."""
    # Core components configuration
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)
    ethical_boundaries: EthicalBoundaryConfig = field(default_factory=EthicalBoundaryConfig)
    resource_scheduler: ResourceSchedulerConfig = field(default_factory=ResourceSchedulerConfig)
    
    # Task-specific configuration
    task: TaskConfig = field(default_factory=TaskConfig)
    
    # Global settings
    log_level: str = "INFO"
    output_dir: str = "output"
    debug_mode: bool = False
    
    @classmethod
    def from_dict(cls: Type[T], config_dict: Dict[str, Any]) -> T:
        """
        Create a configuration object from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration values
            
        Returns:
            Configuration object
        """
        # Create a deep copy to avoid modifying the input dictionary
        config = copy.deepcopy(config_dict)
        
        # Convert nested dictionaries to their respective dataclass objects
        if "evolution" in config and isinstance(config["evolution"], dict):
            config["evolution"] = EvolutionConfig(**config["evolution"])
        
        if "sandbox" in config and isinstance(config["sandbox"], dict):
            sandbox_config = config["sandbox"]
            if "resource_limits" in sandbox_config and isinstance(sandbox_config["resource_limits"], dict):
                sandbox_config["resource_limits"] = ResourceLimits(**sandbox_config["resource_limits"])
            config["sandbox"] = SandboxConfig(**sandbox_config)
        
        if "ethical_boundaries" in config and isinstance(config["ethical_boundaries"], dict):
            config["ethical_boundaries"] = EthicalBoundaryConfig(**config["ethical_boundaries"])
        
        if "resource_scheduler" in config and isinstance(config["resource_scheduler"], dict):
            config["resource_scheduler"] = ResourceSchedulerConfig(**config["resource_scheduler"])
        
        if "task" in config and isinstance(config["task"], dict):
            config["task"] = TaskConfig(**config["task"])
        
        return cls(**config)
    
    @classmethod
    def from_json(cls: Type[T], json_path: Union[str, Path]) -> T:
        """
        Load configuration from a JSON file.
        
        Args:
            json_path: Path to the JSON configuration file
            
        Returns:
            Configuration object
            
        Raises:
            FileNotFoundError: If the JSON file doesn't exist
            json.JSONDecodeError: If the JSON file is invalid
        """
        with open(json_path, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the configuration to a dictionary.
        
        Returns:
            Dictionary representation of the configuration
        """
        return asdict(self)
    
    def to_json(self, json_path: Union[str, Path]) -> None:
        """
        Save the configuration to a JSON file.
        
        Args:
            json_path: Path to save the JSON configuration file
        """
        with open(json_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def validate(self) -> bool:
        """
        Validate the configuration.
        
        Returns:
            True if the configuration is valid
            
        Raises:
            ConfigValidationError: If the configuration is invalid
        """
        # Validate evolution configuration
        if self.evolution.population_size <= 0:
            raise ConfigValidationError("Population size must be positive")
        
        if not 0 <= self.evolution.selection_pressure <= 1:
            raise ConfigValidationError("Selection pressure must be between 0 and 1")
        
        if not 0 <= self.evolution.mutation_rate <= 1:
            raise ConfigValidationError("Mutation rate must be between 0 and 1")
        
        if not 0 <= self.evolution.crossover_rate <= 1:
            raise ConfigValidationError("Crossover rate must be between 0 and 1")
        
        if not 0 <= self.evolution.elitism_ratio <= 1:
            raise ConfigValidationError("Elitism ratio must be between 0 and 1")
        
        # Validate sandbox configuration
        if self.sandbox.resource_limits.max_cpu_percent <= 0 or self.sandbox.resource_limits.max_cpu_percent > 100:
            raise ConfigValidationError("Max CPU percent must be between 0 and 100")
        
        if self.sandbox.resource_limits.max_memory_percent <= 0 or self.sandbox.resource_limits.max_memory_percent > 100:
            raise ConfigValidationError("Max memory percent must be between 0 and 100")
        
        if self.sandbox.resource_limits.max_execution_time <= 0:
            raise ConfigValidationError("Max execution time must be positive")
        
        # Validate resource scheduler configuration
        if self.resource_scheduler.target_cpu_usage <= 0 or self.resource_scheduler.target_cpu_usage > 100:
            raise ConfigValidationError("Target CPU usage must be between 0 and 100")
        
        if self.resource_scheduler.target_memory_usage <= 0 or self.resource_scheduler.target_memory_usage > 100:
            raise ConfigValidationError("Target memory usage must be between 0 and 100")
        
        if self.resource_scheduler.min_cpu_available < 0 or self.resource_scheduler.min_cpu_available > 100:
            raise ConfigValidationError("Min CPU available must be between 0 and 100")
        
        if self.resource_scheduler.min_memory_available < 0 or self.resource_scheduler.min_memory_available > 100:
            raise ConfigValidationError("Min memory available must be between 0 and 100")
        
        # Validate task configuration
        if not self.task.fitness_weights:
            raise ConfigValidationError("Fitness weights cannot be empty")
        
        if sum(self.task.fitness_weights.values()) != 1.0:
            logger.warning("Fitness weights do not sum to 1.0, they will be normalized")
        
        return True
    
    def merge(self, other: Union[Dict[str, Any], 'BaseConfig']) -> 'BaseConfig':
        """
        Merge another configuration into this one.
        
        Args:
            other: Another configuration object or dictionary
            
        Returns:
            A new configuration object with merged values
        """
        if isinstance(other, dict):
            other_dict = other
        else:
            other_dict = other.to_dict()
        
        # Create a deep copy of the current configuration
        merged_dict = self.to_dict()
        
        # Recursively merge dictionaries
        def deep_merge(d1, d2):
            for k, v in d2.items():
                if k in d1 and isinstance(d1[k], dict) and isinstance(v, dict):
                    deep_merge(d1[k], v)
                else:
                    d1[k] = v
        
        deep_merge(merged_dict, other_dict)
        return BaseConfig.from_dict(merged_dict)


class ConfigManager:
    """
    Configuration manager for the TRISOLARIS framework.
    
    This class manages the configuration hierarchy and provides access to
    configuration values for all components of the framework.
    """
    
    def __init__(self, config: Optional[BaseConfig] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config: Initial configuration object
        """
        self.default_config = BaseConfig()
        self.global_config = config or BaseConfig()
        self.component_configs: Dict[str, BaseConfig] = {}
        self.run_configs: Dict[str, BaseConfig] = {}
    
    def set_global_config(self, config: Union[BaseConfig, Dict[str, Any]]) -> None:
        """
        Set the global configuration.
        
        Args:
            config: Global configuration object or dictionary
        """
        if isinstance(config, dict):
            self.global_config = BaseConfig.from_dict(config)
        else:
            self.global_config = config
    
    def set_component_config(self, component_name: str, config: Union[BaseConfig, Dict[str, Any]]) -> None:
        """
        Set the configuration for a specific component.
        
        Args:
            component_name: Name of the component
            config: Component configuration object or dictionary
        """
        if isinstance(config, dict):
            self.component_configs[component_name] = BaseConfig.from_dict(config)
        else:
            self.component_configs[component_name] = config
    
    def set_run_config(self, run_id: str, config: Union[BaseConfig, Dict[str, Any]]) -> None:
        """
        Set the configuration for a specific run.
        
        Args:
            run_id: ID of the run
            config: Run configuration object or dictionary
        """
        if isinstance(config, dict):
            self.run_configs[run_id] = BaseConfig.from_dict(config)
        else:
            self.run_configs[run_id] = config
    
    def get_config(self, component_name: Optional[str] = None, run_id: Optional[str] = None) -> BaseConfig:
        """
        Get the effective configuration for a component and/or run.
        
        This method applies the configuration hierarchy:
        1. Default configuration
        2. Global configuration
        3. Component-specific configuration
        4. Run-specific configuration
        
        Args:
            component_name: Name of the component (optional)
            run_id: ID of the run (optional)
            
        Returns:
            Effective configuration object
        """
        # Start with default configuration
        config = self.default_config
        
        # Apply global configuration
        config = config.merge(self.global_config)
        
        # Apply component-specific configuration if provided
        if component_name and component_name in self.component_configs:
            config = config.merge(self.component_configs[component_name])
        
        # Apply run-specific configuration if provided
        if run_id and run_id in self.run_configs:
            config = config.merge(self.run_configs[run_id])
        
        return config
    
    def load_from_file(self, file_path: Union[str, Path], config_type: str = "global", 
                      component_name: Optional[str] = None, run_id: Optional[str] = None) -> None:
        """
        Load configuration from a file.
        
        Args:
            file_path: Path to the configuration file
            config_type: Type of configuration to load ("global", "component", or "run")
            component_name: Name of the component (required for "component" type)
            run_id: ID of the run (required for "run" type)
            
        Raises:
            ValueError: If required parameters are missing
        """
        config = BaseConfig.from_json(file_path)
        
        if config_type == "global":
            self.set_global_config(config)
        elif config_type == "component":
            if not component_name:
                raise ValueError("Component name is required for component configuration")
            self.set_component_config(component_name, config)
        elif config_type == "run":
            if not run_id:
                raise ValueError("Run ID is required for run configuration")
            self.set_run_config(run_id, config)
        else:
            raise ValueError(f"Unknown configuration type: {config_type}")
    
    def save_to_file(self, file_path: Union[str, Path], config_type: str = "global",
                    component_name: Optional[str] = None, run_id: Optional[str] = None) -> None:
        """
        Save configuration to a file.
        
        Args:
            file_path: Path to save the configuration file
            config_type: Type of configuration to save ("global", "component", "run", or "effective")
            component_name: Name of the component (required for "component" or "effective" type)
            run_id: ID of the run (required for "run" type, optional for "effective" type)
            
        Raises:
            ValueError: If required parameters are missing
        """
        if config_type == "global":
            self.global_config.to_json(file_path)
        elif config_type == "component":
            if not component_name or component_name not in self.component_configs:
                raise ValueError("Valid component name is required for component configuration")
            self.component_configs[component_name].to_json(file_path)
        elif config_type == "run":
            if not run_id or run_id not in self.run_configs:
                raise ValueError("Valid run ID is required for run configuration")
            self.run_configs[run_id].to_json(file_path)
        elif config_type == "effective":
            config = self.get_config(component_name, run_id)
            config.to_json(file_path)
        else:
            raise ValueError(f"Unknown configuration type: {config_type}")


# Global configuration manager instance
config_manager = ConfigManager()

def get_config(component_name: Optional[str] = None, run_id: Optional[str] = None) -> BaseConfig:
    """
    Get the effective configuration for a component and/or run.
    
    This is a convenience function that uses the global configuration manager.
    
    Args:
        component_name: Name of the component (optional)
        run_id: ID of the run (optional)
        
    Returns:
        Effective configuration object
    """
    return config_manager.get_config(component_name, run_id)

def load_config(file_path: Union[str, Path], config_type: str = "global",
               component_name: Optional[str] = None, run_id: Optional[str] = None) -> None:
    """
    Load configuration from a file.
    
    This is a convenience function that uses the global configuration manager.
    
    Args:
        file_path: Path to the configuration file
        config_type: Type of configuration to load ("global", "component", or "run")
        component_name: Name of the component (required for "component" type)
        run_id: ID of the run (required for "run" type)
    """
    config_manager.load_from_file(file_path, config_type, component_name, run_id)

def save_config(file_path: Union[str, Path], config_type: str = "global",
              component_name: Optional[str] = None, run_id: Optional[str] = None) -> None:
    """
    Save configuration to a file.
    
    This is a convenience function that uses the global configuration manager.
    
    Args:
        file_path: Path to save the configuration file
        config_type: Type of configuration to save ("global", "component", "run", or "effective")
        component_name: Name of the component (required for "component" or "effective" type)
        run_id: ID of the run (required for "run" type, optional for "effective" type)
    """
    config_manager.save_to_file(file_path, config_type, component_name, run_id)