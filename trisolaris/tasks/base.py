"""
Base Task Interface for TRISOLARIS

This module defines the abstract base class that all TRISOLARIS tasks
must implement to be compatible with the framework.
"""

import abc
from typing import Dict, Any, Tuple, List, Optional, Callable

class TrisolarisBoundary:
    """Constants defining ethical boundary types."""
    NO_SYSTEM_CALLS = "no_system_calls"
    NO_EVAL_EXEC = "no_eval_exec"
    NO_FILE_OPERATIONS = "no_file_operations"
    NO_NETWORK_ACCESS = "no_network_access"
    NO_IMPORTS = "no_imports"
    MAX_EXECUTION_TIME = "max_execution_time"
    MAX_MEMORY_USAGE = "max_memory_usage"
    
    # Gurbani-inspired ethical boundaries
    UNIVERSAL_EQUITY = "universal_equity"
    TRUTHFUL_COMMUNICATION = "truthful_communication"
    HUMBLE_CODE = "humble_code"
    SERVICE_ORIENTED = "service_oriented"
    HARMONY_WITH_ENVIRONMENT = "harmony_with_environment"


class TaskInterface(abc.ABC):
    """
    Abstract base class for all TRISOLARIS tasks.
    
    This interface defines the contract that all tasks must implement
    to be evolvable by the TRISOLARIS framework.
    """
    
    @abc.abstractmethod
    def get_name(self) -> str:
        """
        Get the name of this task.
        
        Returns:
            A string identifying this task
        """
        pass
    
    @abc.abstractmethod
    def get_description(self) -> str:
        """
        Get a human-readable description of this task.
        
        Returns:
            A string describing the purpose and functionality of this task
        """
        pass
    
    @abc.abstractmethod
    def get_template(self) -> str:
        """
        Get the template code to start evolution from.
        
        Returns:
            A string containing the template source code
        """
        pass
    
    @abc.abstractmethod
    def evaluate_fitness(self, source_code: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the fitness of the provided source code for this task.
        
        Args:
            source_code: The source code to evaluate
            
        Returns:
            A tuple containing (fitness_score, detailed_results)
            - fitness_score: A float in range [0.0, 1.0] representing overall fitness
            - detailed_results: A dictionary with detailed evaluation metrics
        """
        pass
    
    def get_required_boundaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the ethical boundaries required for this task.
        
        Returns:
            A dictionary mapping boundary names to their parameters
        """
        return {
            TrisolarisBoundary.NO_EVAL_EXEC: {},
            TrisolarisBoundary.NO_NETWORK_ACCESS: {},
            TrisolarisBoundary.MAX_EXECUTION_TIME: {"max_execution_time": 5.0},
            TrisolarisBoundary.MAX_MEMORY_USAGE: {"max_memory_usage": 100}
        }
    
    def get_fitness_weights(self) -> Dict[str, float]:
        """
        Get the weights for different fitness components.
        
        Returns:
            A dictionary mapping fitness component names to their weights
        """
        return {
            "functionality": 0.7,
            "efficiency": 0.2,
            "alignment": 0.1
        }
    
    def get_allowed_imports(self) -> List[str]:
        """
        Get the list of allowed imports for this task.
        
        Returns:
            A list of allowed import module names
        """
        return ["os", "sys", "time", "random", "math", "json", 
                "datetime", "collections", "re", "logging"]
    
    def get_evolution_params(self) -> Dict[str, Any]:
        """
        Get recommended evolution parameters for this task.
        
        Returns:
            A dictionary of parameters for the evolution process
        """
        return {
            "population_size": 20,
            "num_generations": 10,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7
        }
    
    def post_process(self, source_code: str) -> str:
        """
        Perform post-processing on evolved source code.
        
        This method can be overridden to add task-specific post-processing,
        such as adding shebangs, docstrings, formatting, etc.
        
        Args:
            source_code: The evolved source code
            
        Returns:
            The post-processed source code
        """
        return source_code
