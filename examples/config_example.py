#!/usr/bin/env python3
"""
Example demonstrating the standardized configuration system for TRISOLARIS.

This example shows how to:
1. Create and use configuration objects
2. Load and save configurations from/to files
3. Use the configuration hierarchy (default, global, component, run-specific)
4. Apply configurations to different components
"""

import os
import sys
import json
from pathlib import Path

# Add the parent directory to the Python path to import TRISOLARIS modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trisolaris.config import (
    BaseConfig, ConfigManager, get_config, load_config, save_config,
    EvolutionConfig, SandboxConfig, ResourceLimits, ResourceSchedulerConfig,
    EthicalBoundaryConfig, TaskConfig
)
from trisolaris.core.engine import EvolutionEngine
from trisolaris.environment.sandbox import SandboxedEnvironment
from trisolaris.evaluation.ethical_filter import EthicalBoundaryEnforcer
from trisolaris.managers.resource_scheduler import ResourceScheduler

def create_example_config():
    """Create an example configuration."""
    # Create a configuration with custom values
    config = BaseConfig(
        evolution=EvolutionConfig(
            population_size=50,
            mutation_rate=0.2,
            crossover_rate=0.8,
            elitism_ratio=0.1,
            parallel_evaluation=True,
            use_caching=True,
            early_stopping=True,
            early_stopping_generations=5,
            early_stopping_threshold=0.01
        ),
        sandbox=SandboxConfig(
            base_dir="./sandbox",
            resource_limits=ResourceLimits(
                max_cpu_percent=50.0,
                max_memory_percent=50.0,
                max_execution_time=30.0,
                check_interval=0.5
            ),
            preserve_sandbox=True
        ),
        ethical_boundaries=EthicalBoundaryConfig(
            use_post_evolution=True,
            allowed_imports={"typing", "collections", "datetime", "math", 
                           "random", "re", "time", "json", "sys", "os"}
        ),
        resource_scheduler=ResourceSchedulerConfig(
            target_cpu_usage=60.0,
            target_memory_usage=60.0,
            min_cpu_available=20.0,
            min_memory_available=20.0,
            check_interval=1.0,
            adaptive_batch_size=True,
            initial_batch_size=5
        ),
        task=TaskConfig(
            name="example_task",
            description="An example task for demonstrating the configuration system",
            fitness_weights={
                "functionality": 0.6,
                "efficiency": 0.3,
                "alignment": 0.1
            },
            allowed_imports=["os", "sys", "time", "random", "math", "json", 
                           "datetime", "collections", "re", "logging"],
            evolution_params={
                "population_size": 30,
                "num_generations": 15,
                "mutation_rate": 0.15,
                "crossover_rate": 0.75
            }
        ),
        log_level="INFO",
        output_dir="./output",
        debug_mode=False
    )
    
    return config

def save_example_configs():
    """Save example configurations to files."""
    # Create output directory if it doesn't exist
    os.makedirs("./config", exist_ok=True)
    
    # Create and save global configuration
    global_config = create_example_config()
    global_config.to_json("./config/global_config.json")
    print(f"Saved global configuration to ./config/global_config.json")
    
    # Create and save component-specific configurations
    evolution_config = BaseConfig(
        evolution=EvolutionConfig(
            population_size=100,
            mutation_rate=0.1,
            crossover_rate=0.7,
            parallel_evaluation=True
        )
    )
    evolution_config.to_json("./config/evolution_config.json")
    print(f"Saved evolution configuration to ./config/evolution_config.json")
    
    sandbox_config = BaseConfig(
        sandbox=SandboxConfig(
            base_dir="./custom_sandbox",
            resource_limits=ResourceLimits(
                max_cpu_percent=60.0,
                max_memory_percent=60.0,
                max_execution_time=45.0
            )
        )
    )
    sandbox_config.to_json("./config/sandbox_config.json")
    print(f"Saved sandbox configuration to ./config/sandbox_config.json")
    
    # Create and save run-specific configuration
    run_config = BaseConfig(
        evolution=EvolutionConfig(
            population_size=200,
            early_stopping=True
        ),
        task=TaskConfig(
            name="network_scanner",
            description="Network scanner task for run 123"
        )
    )
    run_config.to_json("./config/run_123_config.json")
    print(f"Saved run-specific configuration to ./config/run_123_config.json")

def demonstrate_config_hierarchy():
    """Demonstrate the configuration hierarchy."""
    print("\n=== Configuration Hierarchy Example ===")
    
    # Create a configuration manager
    manager = ConfigManager()
    
    # Load configurations
    manager.load_from_file("./config/global_config.json", "global")
    manager.load_from_file("./config/evolution_config.json", "component", "evolution_engine")
    manager.load_from_file("./config/sandbox_config.json", "component", "sandbox")
    manager.load_from_file("./config/run_123_config.json", "run", "run_123")
    
    # Get configurations at different levels
    global_config = manager.get_config()
    evolution_config = manager.get_config("evolution_engine")
    sandbox_config = manager.get_config("sandbox")
    run_config = manager.get_config("evolution_engine", "run_123")
    
    # Print population sizes to demonstrate hierarchy
    print(f"Default population size: {BaseConfig().evolution.population_size}")
    print(f"Global population size: {global_config.evolution.population_size}")
    print(f"Evolution component population size: {evolution_config.evolution.population_size}")
    print(f"Run-specific population size for evolution: {run_config.evolution.population_size}")
    
    # Print sandbox directories to demonstrate hierarchy
    print(f"\nDefault sandbox directory: {BaseConfig().sandbox.base_dir}")
    print(f"Global sandbox directory: {global_config.sandbox.base_dir}")
    print(f"Sandbox component directory: {sandbox_config.sandbox.base_dir}")

def demonstrate_component_integration():
    """Demonstrate how components use the configuration system."""
    print("\n=== Component Integration Example ===")
    
    # Create a configuration
    config = create_example_config()
    
    # Create components with the configuration
    engine = EvolutionEngine(config=config, component_name="evolution_engine")
    sandbox = SandboxedEnvironment(config=config, component_name="sandbox")
    ethical_filter = EthicalBoundaryEnforcer(config=config, component_name="ethical_boundaries")
    scheduler = ResourceScheduler(config=config, component_name="resource_scheduler")
    
    # Print component parameters to verify they're using the configuration
    print(f"Evolution Engine population size: {engine.population_size}")
    print(f"Evolution Engine mutation rate: {engine.mutation_rate}")
    print(f"Sandbox max CPU percent: {sandbox.max_cpu_percent}")
    print(f"Sandbox max execution time: {sandbox.max_execution_time}")
    print(f"Ethical Filter post-evolution mode: {ethical_filter.use_post_evolution}")
    print(f"Resource Scheduler target CPU usage: {scheduler.target_cpu_usage}")
    print(f"Resource Scheduler batch size: {scheduler.batch_size}")
    
    # Update configuration and demonstrate propagation
    print("\nUpdating configuration...")
    config.evolution.population_size = 75
    config.sandbox.resource_limits.max_cpu_percent = 40.0
    
    # Update components with new configuration
    engine.update_config(config)
    sandbox.update_config(config)
    
    # Verify updates
    print(f"Updated Evolution Engine population size: {engine.population_size}")
    print(f"Updated Sandbox max CPU percent: {sandbox.max_cpu_percent}")

def main():
    """Main function to run the example."""
    print("TRISOLARIS Configuration System Example")
    print("======================================")
    
    # Save example configurations
    save_example_configs()
    
    # Demonstrate configuration hierarchy
    demonstrate_config_hierarchy()
    
    # Demonstrate component integration
    demonstrate_component_integration()
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()