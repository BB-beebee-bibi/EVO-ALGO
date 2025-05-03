# TRISOLARIS

A reference to the science fiction novel "The Three-Body Problem" by Liu Cixin, Trisolaris is an evolutionary computation framework that applies natural selection principles to iteratively improve code, configurations, and designs.

## Overview

Trisolaris is an evolutionary computation framework that applies natural selection principles to iteratively improve code, configurations, and designs. The system generates random "mutations" of code, evaluates them against fitness criteria, selects the most successful variants, and recombines them to create increasingly effective solutions.

## Core Principles

Trisolaris operates on three fundamental priorities, with updated weightings:

1. **Alignment (60%)**: Solutions must adhere to universal principles including:
   - Service to others rather than self-interest
   - Truthful and transparent design
   - Resource harmony and mindfulness
   - Inclusive and respectful language
   - Humble, simple approaches over complexity

2. **Functionality (25%)**: Code must work correctly, satisfying all requirements and test cases.

3. **Efficiency (15%)**: Solutions should minimize resource usage, execution time, and complexity.

## Key Features

- **Adaptive Landscape Navigator**: Mathematical models for efficiently exploring the solution space
- **Ethical Boundary Enforcer**: Hard constraints on solution behavior to ensure alignment
- **Island Ecosystem Model**: Multiple subpopulations evolving with different selection pressures
- **Resource-Aware Processing**: Dynamic throttling to maintain system responsiveness
- **Diversity Guardian**: Techniques to prevent premature convergence
- **Visualization Tools**: Interactive monitoring of evolution progress
- **Git-Based Lineage Tracking**: Complete history of solution development

## Quick Start

```python
from trisolaris import EvolutionEngine
from trisolaris.evaluation import FitnessEvaluator

# Define fitness criteria
evaluator = FitnessEvaluator()
evaluator.add_test_case(input_data, expected_output)
evaluator.add_ethical_boundary("no_system_calls")
evaluator.add_ethical_boundary("universal_equity")
evaluator.add_ethical_boundary("truthful_communication")
evaluator.add_resource_constraint(max_memory_mb=100)

# Create evolution engine
engine = EvolutionEngine(population_size=100, evaluator=evaluator)

# Run evolution
engine.evolve(generations=50)

# Get best solution
best_solution = engine.get_best_solution()
print(best_solution.source_code)
```

## References

See references/ directory for detailed information on the evolutionary algorithms and mathematical models implemented in Trisolaris. 