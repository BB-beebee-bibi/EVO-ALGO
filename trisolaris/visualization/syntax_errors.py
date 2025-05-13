"""
Syntax error tracking and visualization module for the TRISOLARIS framework.

This module provides functions for tracking and visualizing syntax error rates
and repair success rates during evolutionary processes.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def track_syntax_errors(
    generation: int,
    population_size: int,
    error_count: int,
    repair_success_count: int,
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Track syntax error metrics for a generation.
    
    Args:
        generation: Current generation number
        population_size: Size of the population
        error_count: Number of syntax errors encountered
        repair_success_count: Number of successful repairs
        metrics: Dictionary to update with syntax error metrics
        
    Returns:
        Updated metrics dictionary
    """
    # Initialize syntax_errors list if it doesn't exist
    if 'syntax_errors' not in metrics:
        metrics['syntax_errors'] = []
    
    # Calculate error and repair rates
    error_rate = error_count / population_size if population_size > 0 else 0
    repair_success_rate = repair_success_count / error_count if error_count > 0 else 1.0
    
    # Add metrics for this generation
    metrics['syntax_errors'].append({
        'generation': generation,
        'population_size': population_size,
        'error_count': error_count,
        'repair_success_count': repair_success_count,
        'error_rate': error_rate,
        'repair_success_rate': repair_success_rate
    })
    
    return metrics

def track_syntax_error_types(
    generation: int,
    error_types: Dict[str, int],
    metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Track types of syntax errors encountered.
    
    Args:
        generation: Current generation number
        error_types: Dictionary mapping error types to counts
        metrics: Dictionary to update with syntax error type metrics
        
    Returns:
        Updated metrics dictionary
    """
    # Initialize syntax_error_types list if it doesn't exist
    if 'syntax_error_types' not in metrics:
        metrics['syntax_error_types'] = []
    
    # Add metrics for this generation
    metrics['syntax_error_types'].append({
        'generation': generation,
        'error_types': error_types
    })
    
    return metrics

def visualize_syntax_error_rates(
    metrics: Dict[str, Any],
    output_path: str,
    interactive: bool = False,
    figsize: Tuple[int, int] = (10, 6)
) -> Optional[str]:
    """
    Visualize syntax error rates and repair success rates over generations.
    
    Args:
        metrics: Dictionary containing syntax error metrics
        output_path: Path to save the visualization
        interactive: Whether to create an interactive visualization
        figsize: Figure size as (width, height)
        
    Returns:
        Path to the saved visualization file or None if no data available
    """
    # Check if syntax error metrics exist
    if 'syntax_errors' not in metrics or not metrics['syntax_errors']:
        logger.warning("No syntax error metrics found")
        return None
    
    # Extract data
    generations = [m['generation'] for m in metrics['syntax_errors']]
    error_rates = [m['error_rate'] for m in metrics['syntax_errors']]
    repair_rates = [m['repair_success_rate'] for m in metrics['syntax_errors']]
    
    if interactive:
        try:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add traces
            fig.add_trace(
                go.Scatter(
                    x=generations, 
                    y=error_rates,
                    mode='lines+markers',
                    name='Error Rate',
                    line=dict(color='red', width=2),
                    hovertemplate='Generation: %{x}<br>Error Rate: %{y:.2f}'
                ),
                secondary_y=False
            )
            
            fig.add_trace(
                go.Scatter(
                    x=generations, 
                    y=repair_rates,
                    mode='lines+markers',
                    name='Repair Success Rate',
                    line=dict(color='green', width=2),
                    hovertemplate='Generation: %{x}<br>Repair Success Rate: %{y:.2f}'
                ),
                secondary_y=False
            )
            
            # Add error count as bars
            error_counts = [m['error_count'] for m in metrics['syntax_errors']]
            fig.add_trace(
                go.Bar(
                    x=generations,
                    y=error_counts,
                    name='Error Count',
                    marker_color='rgba(200, 0, 0, 0.3)',
                    hovertemplate='Generation: %{x}<br>Error Count: %{y}'
                ),
                secondary_y=True
            )
            
            # Update layout
            fig.update_layout(
                title='Syntax Error Rates and Repairs Over Generations',
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
            fig.update_yaxes(title_text='Rate (0-1)', secondary_y=False)
            fig.update_yaxes(title_text='Count', secondary_y=True)
            
            html_path = str(Path(output_path).with_suffix('.html'))
            fig.write_html(html_path, include_plotlyjs='cdn')
            return html_path
        except Exception as e:
            logger.warning(f"Error creating interactive plot: {e}")
            interactive = False
    
    # Static plot
    fig, ax1 = plt.subplots(figsize=figsize)
    
    # Plot rates on primary y-axis
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Rate (0-1)')
    ax1.plot(generations, error_rates, 'r-', marker='o', label='Error Rate')
    ax1.plot(generations, repair_rates, 'g-', marker='s', label='Repair Success Rate')
    ax1.set_ylim(0, 1)
    ax1.tick_params(axis='y')
    
    # Plot error count on secondary y-axis
    ax2 = ax1.twinx()
    ax2.set_ylabel('Error Count')
    error_counts = [m['error_count'] for m in metrics['syntax_errors']]
    ax2.bar(generations, error_counts, alpha=0.3, color='red', label='Error Count')
    ax2.tick_params(axis='y')
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('Syntax Error Rates and Repairs Over Generations')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def visualize_error_types(
    metrics: Dict[str, Any],
    output_path: str,
    interactive: bool = False,
    figsize: Tuple[int, int] = (12, 8),
    top_n: int = 10
) -> Optional[str]:
    """
    Visualize types of syntax errors encountered.
    
    Args:
        metrics: Dictionary containing syntax error type metrics
        output_path: Path to save the visualization
        interactive: Whether to create an interactive visualization
        figsize: Figure size as (width, height)
        top_n: Number of top error types to show
        
    Returns:
        Path to the saved visualization file or None if no data available
    """
    # Check if syntax error type metrics exist
    if 'syntax_error_types' not in metrics or not metrics['syntax_error_types']:
        logger.warning("No syntax error type metrics found")
        return None
    
    # Aggregate error types across all generations
    error_type_counts = {}
    for gen_data in metrics['syntax_error_types']:
        for error_type, count in gen_data['error_types'].items():
            if error_type in error_type_counts:
                error_type_counts[error_type] += count
            else:
                error_type_counts[error_type] = count
    
    # Sort error types by count and take top N
    sorted_error_types = sorted(error_type_counts.items(), key=lambda x: x[1], reverse=True)
    top_error_types = sorted_error_types[:top_n]
    
    # Extract data for visualization
    error_names = [e[0] for e in top_error_types]
    error_counts = [e[1] for e in top_error_types]
    
    if interactive:
        try:
            fig = go.Figure()
            
            # Add horizontal bar chart
            fig.add_trace(go.Bar(
                y=error_names,
                x=error_counts,
                orientation='h',
                marker=dict(
                    color='rgba(220, 0, 0, 0.6)',
                    line=dict(color='rgba(220, 0, 0, 1.0)', width=1)
                )
            ))
            
            # Update layout
            fig.update_layout(
                title='Most Common Syntax Error Types',
                xaxis_title='Count',
                yaxis_title='Error Type',
                template='plotly_white',
                height=max(400, 50 * len(error_names)),
                width=figsize[0] * 100
            )
            
            html_path = str(Path(output_path).with_suffix('.html'))
            fig.write_html(html_path, include_plotlyjs='cdn')
            return html_path
        except Exception as e:
            logger.warning(f"Error creating interactive plot: {e}")
            interactive = False
    
    # Static plot
    plt.figure(figsize=figsize)
    y_pos = np.arange(len(error_names))
    plt.barh(y_pos, error_counts, align='center', color='red', alpha=0.6)
    plt.yticks(y_pos, error_names)
    plt.xlabel('Count')
    plt.title('Most Common Syntax Error Types')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def visualize_error_evolution(
    metrics: Dict[str, Any],
    output_path: str,
    interactive: bool = False,
    figsize: Tuple[int, int] = (12, 8),
    top_n: int = 5
) -> Optional[str]:
    """
    Visualize how syntax error types evolve over generations.
    
    Args:
        metrics: Dictionary containing syntax error type metrics
        output_path: Path to save the visualization
        interactive: Whether to create an interactive visualization
        figsize: Figure size as (width, height)
        top_n: Number of top error types to track
        
    Returns:
        Path to the saved visualization file or None if no data available
    """
    # Check if syntax error type metrics exist
    if 'syntax_error_types' not in metrics or not metrics['syntax_error_types']:
        logger.warning("No syntax error type metrics found")
        return None
    
    # Identify top N error types across all generations
    all_error_types = {}
    for gen_data in metrics['syntax_error_types']:
        for error_type, count in gen_data['error_types'].items():
            if error_type in all_error_types:
                all_error_types[error_type] += count
            else:
                all_error_types[error_type] = count
    
    top_error_types = sorted(all_error_types.items(), key=lambda x: x[1], reverse=True)[:top_n]
    top_error_names = [e[0] for e in top_error_types]
    
    # Extract data for each generation
    generations = [m['generation'] for m in metrics['syntax_error_types']]
    error_data = {error_type: [] for error_type in top_error_names}
    
    for gen_data in metrics['syntax_error_types']:
        for error_type in top_error_names:
            count = gen_data['error_types'].get(error_type, 0)
            error_data[error_type].append(count)
    
    if interactive:
        try:
            fig = go.Figure()
            
            # Add a trace for each error type
            for error_type in top_error_names:
                fig.add_trace(go.Scatter(
                    x=generations,
                    y=error_data[error_type],
                    mode='lines+markers',
                    name=error_type,
                    hovertemplate='Generation: %{x}<br>Count: %{y}'
                ))
            
            # Update layout
            fig.update_layout(
                title='Evolution of Syntax Error Types Over Generations',
                xaxis_title='Generation',
                yaxis_title='Count',
                template='plotly_white',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="right",
                    x=0.99
                )
            )
            
            html_path = str(Path(output_path).with_suffix('.html'))
            fig.write_html(html_path, include_plotlyjs='cdn')
            return html_path
        except Exception as e:
            logger.warning(f"Error creating interactive plot: {e}")
            interactive = False
    
    # Static plot
    plt.figure(figsize=figsize)
    for error_type in top_error_names:
        plt.plot(generations, error_data[error_type], marker='o', label=error_type)
    
    plt.title('Evolution of Syntax Error Types Over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Count')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return output_path

def export_syntax_error_data(metrics: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """
    Export syntax error data to CSV and JSON formats.
    
    Args:
        metrics: Dictionary containing syntax error metrics
        output_dir: Directory to save the exported data
        
    Returns:
        Dictionary mapping data type to file path
    """
    output_files = {}
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Export syntax error rates as CSV
    if 'syntax_errors' in metrics and metrics['syntax_errors']:
        df = pd.DataFrame(metrics['syntax_errors'])
        csv_path = output_path / 'syntax_error_rates.csv'
        df.to_csv(csv_path, index=False)
        output_files['syntax_error_rates_csv'] = str(csv_path)
    
    # Export error type data
    if 'syntax_error_types' in metrics and metrics['syntax_error_types']:
        # Create a DataFrame with generations as rows and error types as columns
        generations = [m['generation'] for m in metrics['syntax_error_types']]
        
        # Get all unique error types
        all_types = set()
        for gen_data in metrics['syntax_error_types']:
            all_types.update(gen_data['error_types'].keys())
        
        # Create data for DataFrame
        data = {'generation': generations}
        for error_type in all_types:
            data[error_type] = [
                gen_data['error_types'].get(error_type, 0) 
                for gen_data in metrics['syntax_error_types']
            ]
        
        df = pd.DataFrame(data)
        csv_path = output_path / 'syntax_error_types.csv'
        df.to_csv(csv_path, index=False)
        output_files['syntax_error_types_csv'] = str(csv_path)
    
    return output_files