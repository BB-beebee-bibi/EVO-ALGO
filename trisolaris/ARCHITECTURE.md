# TRISOLARIS Architecture

TRISOLARIS implements a modular architecture inspired by biological evolution while adhering to strict ethical boundaries and resource constraints.

## System Components

![TRISOLARIS Architecture](docs/architecture_diagram.png)

### 1. Adaptive Landscape Navigator

Models the fitness landscape for code evolution, allowing visualization and efficient traversal of the solution space. Implements mathematical models from evolutionary theory to guide the search process.

Key files:
- `core/landscape.py`: Implements the adaptive landscape mathematics
- `core/engine.py`: Main evolutionary loop

### 2. Genome Repository

Provides versioned storage of code genomes using Git integration, with phylogenetic tracking of solution lineages. Maintains archives of both successful and failed variants.

Key files:
- `managers/repository.py`: Git integration
- `utils/phylogeny.py`: Lineage tracking utilities

### 3. Ethical Boundary Enforcer

Implements hard filters for static analysis before fitness evaluation. Enforces inviolable principles as constraints, not just preferences.

Key files:
- `evaluation/ethical_filter.py`: Implements boundary checks
- `evaluation/fitness.py`: Multi-objective fitness evaluation

### 4. Island Ecosystem Manager

Maintains multiple subpopulations with different selection pressures and enables cross-pollination between islands.

Key files:
- `managers/island.py`: Subpopulation management
- `operators/migration.py`: Solution exchange protocols

### 5. Resource Steward

Monitors system resources and maintains ≥25% availability, dynamically adjusting evolution pace based on resource availability.

Key files:
- `managers/resource.py`: Resource monitoring
- `evaluation/surrogate.py`: Efficient approximations for expensive evaluations

### 6. Diversity Guardian

Tracks population metrics and implements strategies to maintain genetic diversity.

Key files:
- `managers/diversity.py`: Diversity maintenance
- `operators/mutation.py`: Variation operators

### 7. Evolution Monitor & Visualizer

Provides tools for tracking evolution progress and visualizing results.

Key files:
- `visualization/monitor.py`: Progress tracking
- `visualization/landscape_viz.py`: Fitness landscape visualization
- `visualization/phylogeny.py`: Lineage visualization

## Data Flow

1. The Evolution Engine initializes a population of candidate solutions
2. Each solution passes through the Ethical Boundary Enforcer
3. Solutions that pass boundary constraints are evaluated by the Fitness Evaluator
4. The Selection Operator chooses solutions for reproduction based on fitness
5. Variation Operators (mutation and crossover) generate new solutions
6. The Resource Steward monitors and throttles the process as needed
7. The Diversity Guardian injects variation if diversity falls too low
8. The Evolution Monitor tracks and visualizes progress
9. The Genome Repository maintains the history of all solutions

## Mathematical Foundation

TRISOLARIS implements key concepts from evolutionary theory:

- Fitness landscapes and adaptive walks
- Selection gradients and differential reproduction
- Mutation-selection balance
- Exploration-exploitation trade-offs
- Population genetics principles

For detailed mathematical models, see the `references/` directory. 