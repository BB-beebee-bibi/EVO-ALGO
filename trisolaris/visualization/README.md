# TRISOLARIS Visualization Module

This module provides comprehensive visualization capabilities for the TRISOLARIS evolutionary framework, enabling both static and interactive visualizations of evolutionary processes and metrics.

## Features

- **Interactive Visualizations**: Create interactive plots and dashboards using Plotly
- **Static Visualizations**: Generate static plots using Matplotlib (fallback when interactive dependencies are not available)
- **Visualization Dashboard**: Create comprehensive dashboards with multiple visualizations
- **Data Export**: Export visualization data to CSV and JSON formats
- **Population Diversity Metrics**: Track and visualize population diversity metrics
- **Syntax Error Tracking**: Track and visualize syntax error rates and repairs
- **Selection Pressure Visualization**: Visualize selection pressure metrics from Price equation

## Installation

To use the enhanced visualization capabilities, install the required dependencies:

```bash
pip install -r trisolaris/visualization/requirements.txt
```

## Usage Examples

### Creating a Dashboard

```python
from trisolaris.visualization import create_visualization_dashboard

# Create a dashboard with all available visualizations
dashboard_path = create_visualization_dashboard(
    metrics=evolution_metrics,
    output_dir="./output",
    interactive=True
)
```

### Visualizing Fitness Progression

```python
from trisolaris.visualization.dashboard import EvolutionDashboard

# Create a dashboard object
dashboard = EvolutionDashboard(
    metrics=evolution_metrics,
    output_dir="./output",
    interactive=True
)

# Visualize fitness progression
fitness_path = dashboard.visualize_fitness_progression(include_std_dev=True)
```

### Tracking Population Diversity

```python
from trisolaris.visualization import track_diversity

# Track diversity metrics for a population
metrics = track_diversity(
    population=current_population,
    fitness_function=evaluate_fitness,
    generation=current_generation,
    metrics=evolution_metrics
)
```

### Tracking Syntax Errors

```python
from trisolaris.visualization import track_syntax_errors

# Track syntax error metrics for a generation
metrics = track_syntax_errors(
    generation=current_generation,
    population_size=len(current_population),
    error_count=10,
    repair_success_count=8,
    metrics=evolution_metrics
)
```

### Exporting Visualization Data

```python
from trisolaris.visualization import export_visualization_data

# Export visualization data to CSV and JSON formats
exported_files = export_visualization_data(
    metrics=evolution_metrics,
    output_dir="./output",
    formats=["csv", "json"]
)
```

## Module Structure

- `__init__.py`: Main entry point with core visualization functions
- `interactive.py`: Interactive visualization capabilities using Plotly
- `dashboard.py`: Dashboard creation and management
- `diversity.py`: Population diversity metrics and visualization
- `syntax_errors.py`: Syntax error tracking and visualization

## Visualization Types

1. **Fitness Progression**: Visualize best, average, and minimum fitness over generations
2. **Resource Usage**: Visualize CPU, memory, and execution time during evolution
3. **Population Diversity**: Visualize genotypic, phenotypic, and structural diversity
4. **Ethics Evaluation**: Visualize ethics evaluation results by category
5. **Syntax Error Rates**: Visualize syntax error rates and repair success rates
6. **Selection Pressure**: Visualize selection pressure metrics from Price equation

## Graceful Degradation

The visualization module is designed to gracefully degrade when dependencies are not available:

- If Plotly is not available, it falls back to static Matplotlib visualizations
- If optional dependencies like Levenshtein are not available, it uses simpler algorithms

## Data Export Formats

- **CSV**: Tabular data for each metric type
- **JSON**: Complete metrics data in JSON format

## Customization

Most visualization functions accept parameters for customization:

- `figsize`: Figure size as (width, height)
- `interactive`: Whether to create interactive visualizations
- `include_plots`: List of plot types to include in dashboards
- `layout`: Custom layout for dashboards

## Requirements

See `requirements.txt` for the full list of dependencies.