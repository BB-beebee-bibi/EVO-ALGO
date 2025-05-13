"""
Debug Utilities for the TRISOLARIS framework.

This module provides comprehensive debug logging functionality to help
track and diagnose issues in the evolutionary process.
"""

import os
import sys
import time
import logging
import inspect
import traceback
import json
import datetime
from typing import Dict, Any, List, Optional, Union, Callable
import threading
import functools

# Configure debug logging
DEBUG_LOGGER = logging.getLogger("trisolaris.debug")
DEBUG_LOGGER.setLevel(logging.DEBUG)

# Create a file handler for debug logs
debug_log_file = "trisolaris_debug.log"
file_handler = logging.FileHandler(debug_log_file)
file_handler.setLevel(logging.DEBUG)

# Create a console handler for important debug messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatters
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
console_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)

# Set formatters
file_handler.setFormatter(detailed_formatter)
console_handler.setFormatter(console_formatter)

# Add handlers
DEBUG_LOGGER.addHandler(file_handler)
DEBUG_LOGGER.addHandler(console_handler)

# Global debug settings
DEBUG_SETTINGS = {
    "enabled": True,
    "log_level": logging.DEBUG,
    "log_to_console": True,
    "log_to_file": True,
    "log_evolution_details": True,
    "log_genome_details": True,
    "log_fitness_details": True,
    "log_resource_usage": True,
    "log_ethical_checks": True,
    "log_performance_metrics": True,
    "capture_exceptions": True,
    "trace_function_calls": False,  # Can be verbose, disabled by default
}

# Performance metrics tracking
PERFORMANCE_METRICS = {
    "function_calls": {},
    "evolution_times": [],
    "fitness_evaluation_times": [],
    "mutation_times": [],
    "crossover_times": [],
    "ethical_check_times": [],
    "start_time": None,
}

def initialize_debug(
    enabled: bool = True,
    log_level: int = logging.DEBUG,
    log_file: str = "trisolaris_debug.log",
    log_to_console: bool = True,
    log_to_file: bool = True,
    **kwargs
) -> None:
    """
    Initialize the debug system with custom settings.
    
    Args:
        enabled: Whether debugging is enabled
        log_level: Logging level (DEBUG, INFO, etc.)
        log_file: Path to the debug log file
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        **kwargs: Additional debug settings
    """
    global DEBUG_SETTINGS, debug_log_file, file_handler
    
    # Update settings
    DEBUG_SETTINGS["enabled"] = enabled
    DEBUG_SETTINGS["log_level"] = log_level
    DEBUG_SETTINGS["log_to_console"] = log_to_console
    DEBUG_SETTINGS["log_to_file"] = log_to_file
    
    # Update other settings if provided
    for key, value in kwargs.items():
        if key in DEBUG_SETTINGS:
            DEBUG_SETTINGS[key] = value
    
    # Update logger level
    DEBUG_LOGGER.setLevel(log_level)
    
    # Update file handler if log file changed
    if log_file != debug_log_file:
        DEBUG_LOGGER.removeHandler(file_handler)
        debug_log_file = log_file
        file_handler = logging.FileHandler(debug_log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        DEBUG_LOGGER.addHandler(file_handler)
    
    # Enable/disable console handler
    if log_to_console:
        console_handler.setLevel(log_level)
    else:
        console_handler.setLevel(logging.CRITICAL)  # Effectively disable
    
    # Reset performance metrics
    reset_performance_metrics()
    
    DEBUG_LOGGER.info(f"Debug system initialized with settings: {DEBUG_SETTINGS}")

def reset_performance_metrics() -> None:
    """Reset all performance metrics."""
    global PERFORMANCE_METRICS
    PERFORMANCE_METRICS = {
        "function_calls": {},
        "evolution_times": [],
        "fitness_evaluation_times": [],
        "mutation_times": [],
        "crossover_times": [],
        "ethical_check_times": [],
        "start_time": datetime.datetime.now(),
    }
    DEBUG_LOGGER.debug("Performance metrics reset")

def debug_log(message: str, level: int = logging.DEBUG, **kwargs) -> None:
    """
    Log a debug message with additional context.
    
    Args:
        message: The message to log
        level: Logging level
        **kwargs: Additional context to include in the log
    """
    if not DEBUG_SETTINGS["enabled"]:
        return
    
    # Get caller information
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    function = frame.f_code.co_name
    
    # Format additional context
    context = ""
    if kwargs:
        context = " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
    
    # Log the message with context
    DEBUG_LOGGER.log(
        level,
        f"[{os.path.basename(filename)}:{lineno} in {function}] {message}{context}"
    )

def debug_exception(e: Exception, context: str = "") -> None:
    """
    Log an exception with detailed traceback.
    
    Args:
        e: The exception to log
        context: Additional context about where the exception occurred
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["capture_exceptions"]:
        return
    
    # Get the full traceback
    tb = traceback.format_exc()
    
    # Log the exception
    DEBUG_LOGGER.error(f"Exception in {context}: {str(e)}\n{tb}")

def debug_decorator(func: Callable) -> Callable:
    """
    Decorator to add debug logging to a function.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with debug logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not DEBUG_SETTINGS["enabled"]:
            return func(*args, **kwargs)
        
        # Get function details
        func_name = func.__qualname__
        
        # Log function entry
        debug_log(f"Entering {func_name}", level=logging.DEBUG)
        
        # Track performance if enabled
        if DEBUG_SETTINGS["log_performance_metrics"]:
            start_time = time.time()
            
            # Update call count
            if func_name not in PERFORMANCE_METRICS["function_calls"]:
                PERFORMANCE_METRICS["function_calls"][func_name] = {
                    "count": 0,
                    "total_time": 0,
                    "min_time": float('inf'),
                    "max_time": 0,
                }
            
            PERFORMANCE_METRICS["function_calls"][func_name]["count"] += 1
        
        # Call the function
        try:
            result = func(*args, **kwargs)
            
            # Log function exit
            debug_log(f"Exiting {func_name}", level=logging.DEBUG)
            
            # Update performance metrics
            if DEBUG_SETTINGS["log_performance_metrics"]:
                elapsed = time.time() - start_time
                metrics = PERFORMANCE_METRICS["function_calls"][func_name]
                metrics["total_time"] += elapsed
                metrics["min_time"] = min(metrics["min_time"], elapsed)
                metrics["max_time"] = max(metrics["max_time"], elapsed)
                
                # Track specific operation types
                if "evaluate" in func_name.lower():
                    PERFORMANCE_METRICS["fitness_evaluation_times"].append(elapsed)
                elif "mutate" in func_name.lower():
                    PERFORMANCE_METRICS["mutation_times"].append(elapsed)
                elif "crossover" in func_name.lower():
                    PERFORMANCE_METRICS["crossover_times"].append(elapsed)
                elif "ethical" in func_name.lower() or "boundary" in func_name.lower():
                    PERFORMANCE_METRICS["ethical_check_times"].append(elapsed)
                elif "evolve" in func_name.lower():
                    PERFORMANCE_METRICS["evolution_times"].append(elapsed)
            
            return result
            
        except Exception as e:
            # Log the exception
            debug_exception(e, context=func_name)
            
            # Re-raise the exception
            raise
    
    return wrapper

def log_genome_details(genome, generation: int = None, fitness: float = None) -> None:
    """
    Log detailed information about a genome.
    
    Args:
        genome: The genome to log
        generation: Optional generation number
        fitness: Optional fitness value
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_genome_details"]:
        return
    
    try:
        # Get genome source code
        if hasattr(genome, 'to_source'):
            source = genome.to_source()
        else:
            source = str(genome)
        
        # Truncate source if too long
        if len(source) > 1000:
            source = source[:500] + "\n...\n" + source[-500:]
        
        # Log basic info
        gen_info = f" (Generation {generation})" if generation is not None else ""
        fit_info = f" (Fitness: {fitness:.4f})" if fitness is not None else ""
        
        DEBUG_LOGGER.debug(f"Genome Details{gen_info}{fit_info}:\n{source}")
        
    except Exception as e:
        debug_exception(e, context="log_genome_details")

def log_fitness_evaluation(genome, fitness: float, details: Dict[str, Any] = None) -> None:
    """
    Log details about a fitness evaluation.
    
    Args:
        genome: The genome that was evaluated
        fitness: The fitness value
        details: Optional detailed evaluation results
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_fitness_details"]:
        return
    
    try:
        # Log basic fitness info
        DEBUG_LOGGER.debug(f"Fitness Evaluation: {fitness:.4f}")
        
        # Log detailed results if available
        if details:
            # Format details for logging
            details_str = json.dumps(details, indent=2)
            DEBUG_LOGGER.debug(f"Fitness Details:\n{details_str}")
        
    except Exception as e:
        debug_exception(e, context="log_fitness_evaluation")

def log_ethical_check(genome, passed: bool, boundaries: Dict[str, bool] = None) -> None:
    """
    Log details about an ethical boundary check.
    
    Args:
        genome: The genome that was checked
        passed: Whether the genome passed all ethical boundaries
        boundaries: Optional dictionary of boundary results
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_ethical_checks"]:
        return
    
    try:
        # Log basic check result
        result = "PASSED" if passed else "FAILED"
        DEBUG_LOGGER.debug(f"Ethical Check: {result}")
        
        # Log detailed boundary results if available
        if boundaries:
            # Format boundaries for logging
            boundaries_str = "\n".join(f"  - {name}: {'PASSED' if passed else 'FAILED'}" 
                                     for name, passed in boundaries.items())
            DEBUG_LOGGER.debug(f"Boundary Results:\n{boundaries_str}")
        
    except Exception as e:
        debug_exception(e, context="log_ethical_check")

def log_resource_usage(resource_monitor) -> None:
    """
    Log current resource usage from a ResourceSteward.
    
    Args:
        resource_monitor: The ResourceSteward instance
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_resource_usage"]:
        return
    
    try:
        # Check if resource monitor is available
        if resource_monitor is None:
            return
        
        # Get current resource status
        status = resource_monitor.check_resources()
        
        # Log resource usage
        DEBUG_LOGGER.debug(
            f"Resource Usage: "
            f"Memory: {(1.0 - status['memory_available']) * 100:.1f}% used, "
            f"CPU: {(1.0 - status['cpu_available']) * 100:.1f}% used, "
            f"Disk: {(1.0 - status['disk_available']) * 100:.1f}% used, "
            f"Throttle Level: {resource_monitor.get_throttle_level()}"
        )
        
    except Exception as e:
        debug_exception(e, context="log_resource_usage")

def log_evolution_progress(generation: int, best_fitness: float, avg_fitness: float, 
                          elapsed_time: float, population_size: int = None) -> None:
    """
    Log progress of the evolutionary process.
    
    Args:
        generation: Current generation number
        best_fitness: Best fitness in the current generation
        avg_fitness: Average fitness in the current generation
        elapsed_time: Time taken for this generation
        population_size: Optional population size
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_evolution_details"]:
        return
    
    try:
        # Format population size info
        pop_info = f", Population: {population_size}" if population_size is not None else ""
        
        # Log progress
        DEBUG_LOGGER.info(
            f"Generation {generation}: "
            f"Best Fitness: {best_fitness:.4f}, "
            f"Avg Fitness: {avg_fitness:.4f}, "
            f"Time: {elapsed_time:.2f}s{pop_info}"
        )
        
    except Exception as e:
        debug_exception(e, context="log_evolution_progress")

def generate_performance_report() -> str:
    """
    Generate a detailed performance report.
    
    Returns:
        A formatted performance report string
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_performance_metrics"]:
        return "Performance reporting disabled"
    
    try:
        # Calculate runtime
        runtime = datetime.datetime.now() - PERFORMANCE_METRICS["start_time"]
        
        # Build the report
        report = []
        report.append("=" * 50)
        report.append("TRISOLARIS PERFORMANCE REPORT")
        report.append("=" * 50)
        report.append(f"Total Runtime: {runtime}")
        
        # Function call statistics
        report.append("\nFunction Call Statistics:")
        report.append("-" * 30)
        
        # Sort functions by total time (descending)
        sorted_funcs = sorted(
            PERFORMANCE_METRICS["function_calls"].items(),
            key=lambda x: x[1]["total_time"],
            reverse=True
        )
        
        for func_name, metrics in sorted_funcs[:20]:  # Show top 20 functions
            count = metrics["count"]
            total = metrics["total_time"]
            avg = total / count if count > 0 else 0
            
            report.append(f"{func_name}:")
            report.append(f"  Calls: {count}")
            report.append(f"  Total Time: {total:.4f}s")
            report.append(f"  Avg Time: {avg:.4f}s")
            report.append(f"  Min Time: {metrics['min_time']:.4f}s")
            report.append(f"  Max Time: {metrics['max_time']:.4f}s")
        
        # Operation type statistics
        report.append("\nOperation Statistics:")
        report.append("-" * 30)
        
        # Evolution times
        evo_times = PERFORMANCE_METRICS["evolution_times"]
        if evo_times:
            avg_evo = sum(evo_times) / len(evo_times)
            report.append(f"Evolution Operations: {len(evo_times)}")
            report.append(f"  Avg Time: {avg_evo:.4f}s")
            report.append(f"  Min Time: {min(evo_times):.4f}s")
            report.append(f"  Max Time: {max(evo_times):.4f}s")
        
        # Fitness evaluation times
        fit_times = PERFORMANCE_METRICS["fitness_evaluation_times"]
        if fit_times:
            avg_fit = sum(fit_times) / len(fit_times)
            report.append(f"Fitness Evaluations: {len(fit_times)}")
            report.append(f"  Avg Time: {avg_fit:.4f}s")
            report.append(f"  Min Time: {min(fit_times):.4f}s")
            report.append(f"  Max Time: {max(fit_times):.4f}s")
        
        # Mutation times
        mut_times = PERFORMANCE_METRICS["mutation_times"]
        if mut_times:
            avg_mut = sum(mut_times) / len(mut_times)
            report.append(f"Mutations: {len(mut_times)}")
            report.append(f"  Avg Time: {avg_mut:.4f}s")
            report.append(f"  Min Time: {min(mut_times):.4f}s")
            report.append(f"  Max Time: {max(mut_times):.4f}s")
        
        # Crossover times
        cross_times = PERFORMANCE_METRICS["crossover_times"]
        if cross_times:
            avg_cross = sum(cross_times) / len(cross_times)
            report.append(f"Crossovers: {len(cross_times)}")
            report.append(f"  Avg Time: {avg_cross:.4f}s")
            report.append(f"  Min Time: {min(cross_times):.4f}s")
            report.append(f"  Max Time: {max(cross_times):.4f}s")
        
        # Ethical check times
        ethical_times = PERFORMANCE_METRICS["ethical_check_times"]
        if ethical_times:
            avg_ethical = sum(ethical_times) / len(ethical_times)
            report.append(f"Ethical Checks: {len(ethical_times)}")
            report.append(f"  Avg Time: {avg_ethical:.4f}s")
            report.append(f"  Min Time: {min(ethical_times):.4f}s")
            report.append(f"  Max Time: {max(ethical_times):.4f}s")
        
        report.append("=" * 50)
        
        return "\n".join(report)
        
    except Exception as e:
        debug_exception(e, context="generate_performance_report")
        return f"Error generating performance report: {str(e)}"

def save_performance_report(output_path: str = "trisolaris_performance_report.txt") -> None:
    """
    Generate and save a performance report to a file.
    
    Args:
        output_path: Path to save the report
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_performance_metrics"]:
        return
    
    try:
        # Generate the report
        report = generate_performance_report()
        
        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        DEBUG_LOGGER.info(f"Performance report saved to {output_path}")
        
    except Exception as e:
        debug_exception(e, context="save_performance_report")

# Thread-safe debug logging for concurrent operations
class ThreadSafeDebugLogger:
    """Thread-safe wrapper for debug logging in concurrent environments."""
    
    def __init__(self):
        """Initialize the thread-safe logger."""
        self.lock = threading.RLock()
    
    def log(self, message: str, level: int = logging.DEBUG, **kwargs) -> None:
        """Thread-safe debug logging."""
        with self.lock:
            debug_log(message, level, **kwargs)
    
    def log_genome(self, genome, generation: int = None, fitness: float = None) -> None:
        """Thread-safe genome logging."""
        with self.lock:
            log_genome_details(genome, generation, fitness)
    
    def log_fitness(self, genome, fitness: float, details: Dict[str, Any] = None) -> None:
        """Thread-safe fitness logging."""
        with self.lock:
            log_fitness_evaluation(genome, fitness, details)
    
    def log_ethical(self, genome, passed: bool, boundaries: Dict[str, bool] = None) -> None:
        """Thread-safe ethical check logging."""
        with self.lock:
            log_ethical_check(genome, passed, boundaries)
    
    def log_resources(self, resource_monitor) -> None:
        """Thread-safe resource usage logging."""
        with self.lock:
            log_resource_usage(resource_monitor)
    
    def log_progress(self, generation: int, best_fitness: float, avg_fitness: float, 
                    elapsed_time: float, population_size: int = None) -> None:
        """Thread-safe evolution progress logging."""
        with self.lock:
            log_evolution_progress(generation, best_fitness, avg_fitness, 
                                 elapsed_time, population_size)
    
    def exception(self, e: Exception, context: str = "") -> None:
        """Thread-safe exception logging."""
        with self.lock:
            debug_exception(e, context)

# Create a global thread-safe logger instance
thread_safe_logger = ThreadSafeDebugLogger()
