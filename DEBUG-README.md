# TRISOLARIS Debug Utilities

This document provides information about the debug utilities added to the TRISOLARIS evolutionary algorithm framework to help with debugging, performance monitoring, and troubleshooting.

## Overview

The debug utilities provide comprehensive logging, performance monitoring, and error tracking capabilities for the TRISOLARIS framework. These tools help identify bottlenecks, track the evolution process in detail, and diagnose issues that may arise during the evolutionary process.

## Debug Components

### 1. Debug Logging Module (`trisolaris/utils/debug.py`)

A comprehensive debug logging module that provides:

- Multi-level logging (minimal, normal, verbose, trace)
- Function call tracing with performance metrics
- Genome content logging
- Fitness evaluation details
- Ethical boundary check results
- Resource usage monitoring
- Evolution progress tracking
- Thread-safe logging for concurrent operations

### 2. Debug Task Runner (`trisolaris_debug_runner.py`)

An enhanced version of the standard task runner with:

- Comprehensive debug logging
- Performance monitoring and reporting
- Detailed progress tracking
- Exception handling and reporting
- Resource usage monitoring

### 3. Task-Specific Debug Scripts

Convenience scripts for running specific tasks with debug capabilities:

- `debug_network_scanner.py` - For network scanner evolution
- `debug_drive_scanner.py` - For drive scanner evolution
- `debug_bluetooth_scanner.py` - For bluetooth scanner evolution

## Usage

### Running a Debug Evolution

To run an evolution with debug capabilities, use one of the task-specific debug scripts:

```bash
# Run network scanner evolution with debug capabilities
./debug_network_scanner.py --gens=5 --pop-size=20 --debug-level=verbose

# Run drive scanner evolution with resource monitoring
./debug_drive_scanner.py --resource-monitoring --debug-level=trace

# Run bluetooth scanner evolution with island model
./debug_bluetooth_scanner.py --use-islands --islands=3 --debug-level=verbose
```

### Debug Levels

The debug utilities support four levels of debugging:

1. **minimal** - Only critical messages and warnings
2. **normal** - Basic information about the evolution process
3. **verbose** - Detailed information including genome details and fitness evaluations
4. **trace** - Comprehensive function call tracing and performance metrics

### Command Line Options

All debug scripts support the following options:

```
--gens=N                 Number of generations to evolve (default: 5)
--pop-size=N             Population size for each generation (default: 20)
--mutation-rate=0.X      Mutation rate (default: 0.2)
--crossover-rate=0.X     Crossover rate (default: 0.7)
--template=FILE          Path to template script
--resource-monitoring    Enable resource monitoring
--output-dir=DIR         Directory to save outputs
--use-islands            Use island model for evolution
--islands=N              Number of islands for island model (default: 3)
--debug-level=LEVEL      Debug logging level (minimal, normal, verbose, trace)
--ethics-level=LEVEL     Ethical filter level (none, basic, full)
```

## Output Files

The debug utilities generate several output files:

1. **trisolaris_debug.log** - Main debug log file with detailed information
2. **trisolaris_evolution.log** - Standard evolution log
3. **performance_report.txt** - Detailed performance metrics report
4. **metadata.json** - Evolution metadata and statistics

## Performance Monitoring

The debug utilities track performance metrics for:

- Function call counts and execution times
- Evolution operation times
- Fitness evaluation times
- Mutation and crossover operation times
- Ethical check times

These metrics are saved in the performance report and can help identify bottlenecks in the evolution process.

## Resource Monitoring

When resource monitoring is enabled, the debug utilities track:

- CPU usage
- Memory usage
- Disk usage

If resource constraints are detected, the evolution process will automatically throttle to prevent system overload.

## Example: Analyzing Debug Output

1. Run an evolution with debug enabled:
   ```bash
   ./debug_network_scanner.py --debug-level=verbose
   ```

2. Check the debug log for detailed information:
   ```bash
   less outputs/network_scanner_debug_*/trisolaris_debug.log
   ```

3. Review the performance report:
   ```bash
   less outputs/network_scanner_debug_*/performance_report.txt
   ```

4. Examine the evolution metadata:
   ```bash
   cat outputs/network_scanner_debug_*/metadata.json
   ```

## Troubleshooting

If you encounter issues during evolution:

1. Increase the debug level to `verbose` or `trace`
2. Enable resource monitoring to check for resource constraints
3. Check the debug log for error messages and exceptions
4. Review the performance report for bottlenecks

## Advanced Usage

### Direct Use of Debug Utilities

You can use the debug utilities directly in your code:

```python
from trisolaris.utils.debug import (
    initialize_debug, debug_log, debug_exception,
    log_genome_details, log_fitness_evaluation
)

# Initialize debug system
initialize_debug(enabled=True, log_level=logging.DEBUG)

# Log a debug message
debug_log("Starting custom evolution", level=logging.INFO)

# Log genome details
log_genome_details(my_genome, generation=5, fitness=0.85)

# Log fitness evaluation
log_fitness_evaluation(my_genome, 0.85, {"functionality": 0.9, "efficiency": 0.8})

# Log an exception
try:
    # Some code that might raise an exception
    pass
except Exception as e:
    debug_exception(e, context="custom_evolution")
```

### Using the Debug Decorator

You can use the debug decorator to automatically log function entry/exit and track performance:

```python
from trisolaris.utils.debug import debug_decorator

@debug_decorator
def my_function(arg1, arg2):
    # Function implementation
    pass
```

## Conclusion

The debug utilities provide powerful tools for understanding, monitoring, and troubleshooting the TRISOLARIS evolutionary process. By using these tools, you can gain insights into the evolution process, identify bottlenecks, and diagnose issues that may arise.
