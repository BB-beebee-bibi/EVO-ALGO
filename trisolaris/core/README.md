# TRISOLARIS Core Components

The core module contains the fundamental components of the TRISOLARIS framework:

## Engine (engine.py)

The `EvolutionEngine` class implements the main evolutionary loop. Key features:

- Population initialization
- Generational progression
- Integration with fitness evaluation
- Termination criteria handling

```python
from trisolaris.core import EvolutionEngine

# Create engine with default parameters
engine = EvolutionEngine(population_size=100)

# Configure custom parameters
engine.set_selection_pressure(0.8)
engine.set_mutation_rate(0.1)
engine.set_crossover_rate(0.7)

# Run evolution
best_solution = engine.evolve(generations=50)
```

## Landscape (landscape.py)

The `AdaptiveLandscape` class implements the mathematical foundation of the evolution process:

- Models the fitness landscape for navigating solution space
- Calculates selection gradients to guide evolution
- Provides visualization capabilities

```python
from trisolaris.core import AdaptiveLandscape

# Create landscape based on fitness function
landscape = AdaptiveLandscape(fitness_function=my_fitness_function)

# Calculate gradient at a point
gradient = landscape.get_gradient(solution)

# Visualize landscape
landscape.visualize(population=my_population)
```

## Genome (genome.py)

The `CodeGenome` class represents a single solution in the population:

- Encodes code as manipulable data structure (AST or graph)
- Implements variation operators (mutation, crossover)
- Provides conversion to/from source code

```python
from trisolaris.core import CodeGenome

# Create genome from source code
genome = CodeGenome.from_source("def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)")

# Apply mutation
genome.mutate(rate=0.1)

# Apply crossover with another genome
child = genome.crossover(other_genome)

# Convert to source code
source = genome.to_source()
```

## Extension Points

The core components are designed to be extended for specific use cases:

- Subclass `CodeGenome` to implement domain-specific representations
- Subclass `AdaptiveLandscape` to implement specialized landscape models
- Extend `EvolutionEngine` to customize the evolutionary process 