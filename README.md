# TRISOLARIS Evolutionary Framework

TRISOLARIS (Thoroughly Recursive Iterative System for Organic Learning and Adaptive Resource-Intelligent Solutions) is an advanced evolutionary algorithm framework designed for code evolution with ethical boundaries and resource-aware execution.

## Core Architecture

The TRISOLARIS framework consists of several key components:

1. **Adaptive Landscape Navigator**: Models fitness landscapes for code evolution
2. **Genome Repository**: Git-backed version control for code variants
3. **Ethical Boundary Enforcer**: Ensures generated code follows ethical guidelines
4. **Island Ecosystem Manager**: Maintains multiple subpopulations with different selection pressures
5. **Resource Steward**: Monitors system resources and maintains ≥25% availability
6. **Diversity Guardian**: Tracks population metrics and preserves solution variety
7. **Evolution Monitor & Visualizer**: Records metrics and generates visualizations
8. **Task Interface**: Defines a generic interface for evolvable tasks

## Project Structure

```
EVO-ALGO/
├── run.py                  # Legacy entry point for TRISOLARIS engine
├── trisolaris_task_runner.py  # New entry point for generic task evolution
├── outputs/                # Contains timestamped evolution runs
│   └── run_YYYYMMDD_HHMMSS/
│       └── generation_N/   # Generated solutions by generation
├── minja/                  # Example code for evolution
│   └── minja_usb_scan.py   # USB scanning utility
├── trisolaris/             # Core framework
│   ├── core/               # Evolutionary engine
│   ├── evaluation/         # Fitness and ethical boundaries
│   ├── managers/           # Resource, diversity, and island management
│   ├── tasks/              # Task interfaces and implementations
│   └── utils/              # Configuration and logging utilities
```

## Task-Based Architecture

TRISOLARIS has been enhanced with a task-based architecture that separates the evolution process from specific tasks:

1. **Task Interface**: Defines a common interface that all evolvable tasks must implement
2. **Task Implementations**: Task-specific code that handles fitness evaluation, templates, and evolution parameters
3. **Task Runner**: Generic evolution runner that can evolve any task that implements the TaskInterface

This separation allows for:
- Reusing the same evolutionary engine for different tasks
- Defining task-specific fitness functions and templates
- Applying task-specific post-processing to evolved solutions
- Evolving new tasks without modifying the core engine

## Included Tasks

- **Drive Scanner Task**: A task for evolving programs that scan storage devices and analyze their contents

## Usage

### Evolving a Task

Run the task-based framework with the following command:

```bash
python3 trisolaris/task_runner.py drive_scanner --template drive_scanner.py --pop-size 20 --gens 10 --ethics-level full --resource-monitoring
```

### Command-line Options

- `task`: Name of the task to evolve (e.g., drive_scanner)
- `--template`: Path to a custom template file (optional)
- `--output-dir`: Base directory to save evolved code (default: outputs)
- `--pop-size`: Population size (default: task-specific recommendation)
- `--gens`: Number of generations (default: task-specific recommendation)
- `--mutation-rate`: Mutation rate (default: task-specific recommendation)
- `--crossover-rate`: Crossover rate (default: task-specific recommendation)
- `--ethics-level`: Ethical filter level (none, basic, full) (default: basic)
- `--resource-monitoring`: Enable resource monitoring and throttling
- `--use-git`: Use Git for version control of solution history
- `--use-islands`: Use island model for evolution
- `--islands`: Number of islands when using island model (default: 3)
- `--migration-interval`: Number of generations between migrations (default: 3)
- `--diversity-threshold`: Diversity threshold for injection (default: 0.3)

## Creating New Tasks

To create a new task for TRISOLARIS:

1. Create a new class that implements the `TaskInterface` in `trisolaris/tasks/`
2. Implement all required methods, including:
   - `get_name()`: Return the task name
   - `get_description()`: Return a description of the task
   - `get_template()`: Return template code to start from
   - `evaluate_fitness()`: Evaluate the fitness of a solution
3. Register the task in `TASK_REGISTRY` in `trisolaris/task_runner.py`

## Requirements

- Python 3.6+
- NumPy
- psutil (optional, for resource monitoring)
- Git (optional, for version control)

## Design Principles

The TRISOLARIS framework follows these core design principles:

1. **Safety First**: Evolve code within strict ethical boundaries
2. **Resource Awareness**: Adapt to available system resources
3. **Diversity Preservation**: Maintain genetic diversity to find novel solutions
4. **Version Control**: Track the evolution history of code
5. **Modularity**: Extensible architecture for custom evolutionary strategies
6. **Separation of Concerns**: Generic evolution process separated from task-specific details
