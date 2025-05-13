"""
Interactive visualization module for the TRISOLARIS framework.

This module provides interactive visualizations using Plotly for evolutionary processes,
including fitness progression, population diversity, resource usage, and other metrics.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_fitness_progression_plot(
    metrics: Dict[str, Any],
    include_std_dev: bool = True
) -> go.Figure:
    """
    Create an interactive fitness progression plot.
    
    Args:
        metrics: Dictionary containing evolution metrics
        include_std_dev: Whether to include standard deviation visualization
        
    Returns:
        Plotly figure object
    """
    # Extract data
    generations = [m['generation'] for m in metrics.get('fitness_history', [])]
    best_fitness = [m['best_fitness'] for m in metrics.get('fitness_history', [])]
    avg_fitness = [m['avg_fitness'] for m in metrics.get('fitness_history', [])]
    min_fitness = [m['min_fitness'] for m in metrics.get('fitness_history', [])]
    std_dev = [m['std_dev'] for m in metrics.get('fitness_history', [])]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=generations, 
        y=best_fitness,
        mode='lines+markers',
        name='Best Fitness',
        line=dict(color='blue', width=2),
        hovertemplate='Generation: %{x}<br>Best Fitness: %{y:.4f}'
    ))
    
    fig.add_trace(go.Scatter(
        x=generations, 
        y=avg_fitness,
        mode='lines+markers',
        name='Average Fitness',
        line=dict(color='green', width=2),
        hovertemplate='Generation: %{x}<br>Average Fitness: %{y:.4f}'
    ))
    
    fig.add_trace(go.Scatter(
        x=generations, 
        y=min_fitness,
        mode='lines+markers',
        name='Min Fitness',
        line=dict(color='red', width=2),
        hovertemplate='Generation: %{x}<br>Min Fitness: %{y:.4f}'
    ))
    
    # Add standard deviation as a shaded area if requested
    if include_std_dev:
        upper_bound = [a + s for a, s in zip(avg_fitness, std_dev)]
        lower_bound = [a - s for a, s in zip(avg_fitness, std_dev)]
        
        fig.add_trace(go.Scatter(
            x=generations + generations[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo='skip',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title='Fitness Progression Over Generations',
        xaxis_title='Generation',
        yaxis_title='Fitness',
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white'
    )
    
    return fig

def create_resource_usage_plot(metrics: Dict[str, Any]) -> go.Figure:
    """
    Create an interactive resource usage plot.
    
    Args:
        metrics: Dictionary containing evolution metrics
        
    Returns:
        Plotly figure object
    """
    # Extract data
    generations = [m['generation'] for m in metrics.get('resource_usage', [])]
    cpu_usage = [m.get('cpu_percent', 0) for m in metrics.get('resource_usage', [])]
    memory_usage = [m.get('memory_percent', 0) for m in metrics.get('resource_usage', [])]
    execution_time = [m.get('execution_time', 0) for m in metrics.get('resource_usage', [])]
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=generations, 
            y=cpu_usage,
            mode='lines+markers',
            name='CPU Usage (%)',
            line=dict(color='blue', width=2),
            hovertemplate='Generation: %{x}<br>CPU Usage: %{y:.2f}%'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=generations, 
            y=memory_usage,
            mode='lines+markers',
            name='Memory Usage (%)',
            line=dict(color='green', width=2),
            hovertemplate='Generation: %{x}<br>Memory Usage: %{y:.2f}%'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=generations, 
            y=execution_time,
            mode='lines+markers',
            name='Execution Time (s)',
            line=dict(color='red', width=2),
            hovertemplate='Generation: %{x}<br>Execution Time: %{y:.2f}s'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title='Resource Usage During Evolution',
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white'
    )
    
    # Update axes
    fig.update_xaxes(title_text='Generation')
    fig.update_yaxes(title_text='Usage (%)', secondary_y=False)
    fig.update_yaxes(title_text='Execution Time (s)', secondary_y=True)
    
    return fig

def create_population_diversity_plot(metrics: Dict[str, Any]) -> go.Figure:
    """
    Create an interactive population diversity plot.
    
    Args:
        metrics: Dictionary containing evolution metrics
        
    Returns:
        Plotly figure object
    """
    # Extract data if available, otherwise use placeholder data
    generations = [m['generation'] for m in metrics.get('fitness_history', [])]
    
    # Check if diversity metrics exist, if not create placeholder data
    if 'diversity_metrics' in metrics:
        genotypic_diversity = [m.get('genotypic_diversity', 0) for m in metrics.get('diversity_metrics', [])]
        phenotypic_diversity = [m.get('phenotypic_diversity', 0) for m in metrics.get('diversity_metrics', [])]
        structural_diversity = [m.get('structural_diversity', 0) for m in metrics.get('diversity_metrics', [])]
    else:
        # Calculate a simple diversity metric based on fitness standard deviation
        genotypic_diversity = [m.get('std_dev', 0) for m in metrics.get('fitness_history', [])]
        phenotypic_diversity = [0] * len(generations)  # Placeholder
        structural_diversity = [0] * len(generations)  # Placeholder
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=generations, 
        y=genotypic_diversity,
        mode='lines+markers',
        name='Genotypic Diversity',
        line=dict(color='purple', width=2),
        hovertemplate='Generation: %{x}<br>Genotypic Diversity: %{y:.4f}'
    ))
    
    if any(phenotypic_diversity):
        fig.add_trace(go.Scatter(
            x=generations, 
            y=phenotypic_diversity,
            mode='lines+markers',
            name='Phenotypic Diversity',
            line=dict(color='orange', width=2),
            hovertemplate='Generation: %{x}<br>Phenotypic Diversity: %{y:.4f}'
        ))
    
    if any(structural_diversity):
        fig.add_trace(go.Scatter(
            x=generations, 
            y=structural_diversity,
            mode='lines+markers',
            name='Structural Diversity',
            line=dict(color='brown', width=2),
            hovertemplate='Generation: %{x}<br>Structural Diversity: %{y:.4f}'
        ))
    
    # Update layout
    fig.update_layout(
        title='Population Diversity Over Generations',
        xaxis_title='Generation',
        yaxis_title='Diversity',
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white'
    )
    
    return fig

def create_ethics_evaluation_plot(metrics: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Create an interactive ethics evaluation plot.
    
    Args:
        metrics: Dictionary containing evolution metrics
        
    Returns:
        Plotly figure object or None if no ethics data available
    """
    # Check if ethics evaluations exist
    if not metrics.get('ethics_evaluations'):
        return None
    
    # Get the latest ethics evaluation
    ethics = metrics['ethics_evaluations'][-1]
    
    # Extract category scores if available
    if 'categories' not in ethics:
        return None
    
    categories = list(ethics['categories'].keys())
    scores = [ethics['categories'][cat].get('score', 0) for cat in categories]
    
    # Create figure
    fig = go.Figure()
    
    # Add bar chart
    fig.add_trace(go.Bar(
        x=categories,
        y=scores,
        marker_color='teal',
        hovertemplate='Category: %{x}<br>Score: %{y:.2f}'
    ))
    
    # Update layout
    fig.update_layout(
        title='Ethics Evaluation by Category',
        xaxis_title='Category',
        yaxis_title='Score',
        yaxis=dict(range=[0, 1]),
        template='plotly_white'
    )
    
    return fig

def create_syntax_error_plot(metrics: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Create an interactive syntax error rate plot.
    
    Args:
        metrics: Dictionary containing evolution metrics
        
    Returns:
        Plotly figure object or None if no syntax error data available
    """
    # Check if syntax error metrics exist
    if 'syntax_errors' not in metrics:
        return None
    
    # Extract data
    generations = [m['generation'] for m in metrics.get('syntax_errors', [])]
    error_rates = [m.get('error_rate', 0) for m in metrics.get('syntax_errors', [])]
    repair_success = [m.get('repair_success_rate', 0) for m in metrics.get('syntax_errors', [])]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=generations, 
        y=error_rates,
        mode='lines+markers',
        name='Syntax Error Rate',
        line=dict(color='red', width=2),
        hovertemplate='Generation: %{x}<br>Error Rate: %{y:.2f}'
    ))
    
    fig.add_trace(go.Scatter(
        x=generations, 
        y=repair_success,
        mode='lines+markers',
        name='Repair Success Rate',
        line=dict(color='green', width=2),
        hovertemplate='Generation: %{x}<br>Repair Success: %{y:.2f}'
    ))
    
    # Update layout
    fig.update_layout(
        title='Syntax Error Rates and Repairs',
        xaxis_title='Generation',
        yaxis_title='Rate',
        yaxis=dict(range=[0, 1]),
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white'
    )
    
    return fig

def create_selection_pressure_plot(metrics: Dict[str, Any]) -> Optional[go.Figure]:
    """
    Create an interactive selection pressure plot.
    
    Args:
        metrics: Dictionary containing evolution metrics
        
    Returns:
        Plotly figure object or None if no selection pressure data available
    """
    # Check if price equation data exists
    if 'price_equation' not in metrics:
        return None
    
    # Extract data
    generations = [m['generation'] for m in metrics.get('price_equation', [])]
    total_change = [m.get('total_change', 0) for m in metrics.get('price_equation', [])]
    selection = [m.get('selection_component', 0) for m in metrics.get('price_equation', [])]
    transmission = [m.get('transmission_component', 0) for m in metrics.get('price_equation', [])]
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=generations, 
        y=total_change,
        mode='lines+markers',
        name='Total Change',
        line=dict(color='blue', width=2),
        hovertemplate='Generation: %{x}<br>Total Change: %{y:.4f}'
    ))
    
    fig.add_trace(go.Scatter(
        x=generations, 
        y=selection,
        mode='lines+markers',
        name='Selection Component',
        line=dict(color='green', width=2),
        hovertemplate='Generation: %{x}<br>Selection Component: %{y:.4f}'
    ))
    
    fig.add_trace(go.Scatter(
        x=generations, 
        y=transmission,
        mode='lines+markers',
        name='Transmission Component',
        line=dict(color='red', width=2),
        hovertemplate='Generation: %{x}<br>Transmission Component: %{y:.4f}'
    ))
    
    # Update layout
    fig.update_layout(
        title='Selection Pressure (Price Equation Components)',
        xaxis_title='Generation',
        yaxis_title='Component Value',
        hovermode='closest',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        template='plotly_white'
    )
    
    return fig

def export_visualization_data(metrics: Dict[str, Any], output_path: str) -> Dict[str, str]:
    """
    Export visualization data to various formats.
    
    Args:
        metrics: Dictionary containing evolution metrics
        output_path: Base path for output files
        
    Returns:
        Dictionary mapping data type to file path
    """
    output_files = {}
    output_dir = Path(output_path).parent
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export fitness history as CSV
    if 'fitness_history' in metrics:
        fitness_df = pd.DataFrame(metrics['fitness_history'])
        fitness_csv_path = output_dir / 'fitness_history.csv'
        fitness_df.to_csv(fitness_csv_path, index=False)
        output_files['fitness_history'] = str(fitness_csv_path)
    
    # Export resource usage as CSV
    if 'resource_usage' in metrics:
        resource_df = pd.DataFrame(metrics['resource_usage'])
        resource_csv_path = output_dir / 'resource_usage.csv'
        resource_df.to_csv(resource_csv_path, index=False)
        output_files['resource_usage'] = str(resource_csv_path)
    
    # Export price equation data as CSV
    if 'price_equation' in metrics:
        price_df = pd.DataFrame(metrics['price_equation'])
        price_csv_path = output_dir / 'price_equation.csv'
        price_df.to_csv(price_csv_path, index=False)
        output_files['price_equation'] = str(price_csv_path)
    
    # Export all metrics as JSON
    json_path = output_dir / 'all_metrics.json'
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
    output_files['all_metrics'] = str(json_path)
    
    logger.info(f"Exported visualization data to {output_dir}")
    return output_files

def create_interactive_dashboard(
    metrics: Dict[str, Any],
    output_path: str,
    include_plots: List[str] = None
) -> str:
    """
    Create an interactive dashboard with multiple visualizations.
    
    Args:
        metrics: Dictionary containing evolution metrics
        output_path: Path to save the HTML dashboard
        include_plots: List of plot types to include (default: all available)
        
    Returns:
        Path to the saved HTML file
    """
    # Default to all plots if not specified
    all_plots = [
        'fitness', 'resource', 'diversity', 'ethics', 
        'syntax_errors', 'selection_pressure'
    ]
    include_plots = include_plots or all_plots
    
    # Create subplots based on the number of plots to include
    available_plots = []
    
    # Check which plots are available based on data
    if 'fitness' in include_plots and metrics.get('fitness_history'):
        available_plots.append('fitness')
    
    if 'resource' in include_plots and metrics.get('resource_usage'):
        available_plots.append('resource')
    
    if 'diversity' in include_plots:
        available_plots.append('diversity')
    
    if 'ethics' in include_plots and metrics.get('ethics_evaluations'):
        available_plots.append('ethics')
    
    if 'syntax_errors' in include_plots and metrics.get('syntax_errors'):
        available_plots.append('syntax_errors')
    
    if 'selection_pressure' in include_plots and metrics.get('price_equation'):
        available_plots.append('selection_pressure')
    
    # Calculate grid dimensions
    n_plots = len(available_plots)
    if n_plots == 0:
        logger.warning("No plots available to create dashboard")
        return None
    
    cols = min(2, n_plots)
    rows = (n_plots + cols - 1) // cols  # Ceiling division
    
    # Create subplot figure
    fig = make_subplots(
        rows=rows, 
        cols=cols,
        subplot_titles=[p.replace('_', ' ').title() for p in available_plots]
    )
    
    # Add each plot to the dashboard
    for i, plot_type in enumerate(available_plots):
        row = i // cols + 1
        col = i % cols + 1
        
        if plot_type == 'fitness':
            fitness_fig = create_fitness_progression_plot(metrics, include_std_dev=False)
            for trace in fitness_fig.data:
                fig.add_trace(trace, row=row, col=col)
        
        elif plot_type == 'resource':
            resource_fig = create_resource_usage_plot(metrics)
            for trace in resource_fig.data:
                fig.add_trace(trace, row=row, col=col)
        
        elif plot_type == 'diversity':
            diversity_fig = create_population_diversity_plot(metrics)
            for trace in diversity_fig.data:
                fig.add_trace(trace, row=row, col=col)
        
        elif plot_type == 'ethics':
            ethics_fig = create_ethics_evaluation_plot(metrics)
            if ethics_fig:
                for trace in ethics_fig.data:
                    fig.add_trace(trace, row=row, col=col)
        
        elif plot_type == 'syntax_errors':
            syntax_fig = create_syntax_error_plot(metrics)
            if syntax_fig:
                for trace in syntax_fig.data:
                    fig.add_trace(trace, row=row, col=col)
        
        elif plot_type == 'selection_pressure':
            pressure_fig = create_selection_pressure_plot(metrics)
            if pressure_fig:
                for trace in pressure_fig.data:
                    fig.add_trace(trace, row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title='TRISOLARIS Evolution Dashboard',
        height=300 * rows,
        width=600 * cols,
        template='plotly_white',
        showlegend=True
    )
    
    # Save to HTML
    fig.write_html(
        output_path,
        include_plotlyjs='cdn',
        full_html=True,
        config={
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d', 'select2d']
        }
    )
    
    logger.info(f"Interactive dashboard saved to {output_path}")
    return output_path