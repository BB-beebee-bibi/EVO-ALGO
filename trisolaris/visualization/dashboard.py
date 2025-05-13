"""
Dashboard module for the TRISOLARIS framework.

This module provides a unified interface for creating dashboards and visualizations
of evolutionary processes, with support for both static and interactive visualizations.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EvolutionDashboard:
    """
    Dashboard for visualizing evolutionary processes.
    
    This class provides a unified interface for creating dashboards and visualizations
    of evolutionary processes, with support for both static and interactive visualizations.
    """
    
    def __init__(
        self, 
        metrics: Dict[str, Any],
        output_dir: Union[str, Path],
        interactive: bool = True
    ):
        """
        Initialize the dashboard.
        
        Args:
            metrics: Dictionary containing evolution metrics
            output_dir: Directory to save visualizations
            interactive: Whether to use interactive visualizations (requires plotly)
        """
        self.metrics = metrics
        self.output_dir = Path(output_dir)
        self.interactive = interactive
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure metrics are properly formatted
        self._validate_metrics()
    
    def _validate_metrics(self):
        """Validate and normalize metrics format."""
        # Ensure required keys exist
        required_keys = ['fitness_history']
        for key in required_keys:
            if key not in self.metrics:
                logger.warning(f"Required key '{key}' not found in metrics")
                self.metrics[key] = []
    
    def create_dashboard(
        self,
        include_plots: List[str] = None,
        layout: Tuple[int, int] = None
    ) -> str:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            include_plots: List of plot types to include (default: all available)
            layout: Custom layout as (rows, cols) (default: auto-calculated)
            
        Returns:
            Path to the saved dashboard file
        """
        if self.interactive:
            try:
                from trisolaris.visualization.interactive import create_interactive_dashboard
                dashboard_path = str(self.output_dir / "interactive_dashboard.html")
                return create_interactive_dashboard(self.metrics, dashboard_path, include_plots)
            except ImportError:
                logger.warning("Plotly not available, falling back to static dashboard")
                self.interactive = False
        
        # Fall back to static dashboard if interactive is not available or disabled
        from trisolaris.visualization import create_dashboard
        dashboard_path = str(self.output_dir / "dashboard.png")
        create_dashboard(self.metrics, dashboard_path)
        return dashboard_path
    
    def visualize_fitness_progression(self, include_std_dev: bool = True) -> str:
        """
        Create a visualization of fitness progression over generations.
        
        Args:
            include_std_dev: Whether to include standard deviation visualization
            
        Returns:
            Path to the saved visualization file
        """
        if self.interactive:
            try:
                from trisolaris.visualization.interactive import create_fitness_progression_plot
                fig = create_fitness_progression_plot(self.metrics, include_std_dev)
                output_path = str(self.output_dir / "fitness_progression.html")
                fig.write_html(output_path, include_plotlyjs='cdn')
                return output_path
            except ImportError:
                logger.warning("Plotly not available, falling back to static visualization")
                self.interactive = False
        
        # Fall back to static visualization
        from trisolaris.visualization import visualize_evolution_progress
        
        # Extract data
        generations = [m['generation'] for m in self.metrics.get('fitness_history', [])]
        fitness_values = {
            'Best Fitness': [m['best_fitness'] for m in self.metrics.get('fitness_history', [])],
            'Average Fitness': [m['avg_fitness'] for m in self.metrics.get('fitness_history', [])],
            'Min Fitness': [m['min_fitness'] for m in self.metrics.get('fitness_history', [])]
        }
        
        output_path = str(self.output_dir / "fitness_progression.png")
        visualize_evolution_progress(generations, fitness_values, output_path)
        return output_path
    
    def visualize_resource_usage(self) -> str:
        """
        Create a visualization of resource usage during evolution.
        
        Returns:
            Path to the saved visualization file
        """
        if self.interactive:
            try:
                from trisolaris.visualization.interactive import create_resource_usage_plot
                fig = create_resource_usage_plot(self.metrics)
                output_path = str(self.output_dir / "resource_usage.html")
                fig.write_html(output_path, include_plotlyjs='cdn')
                return output_path
            except ImportError:
                logger.warning("Plotly not available, falling back to static visualization")
                self.interactive = False
        
        # Fall back to static visualization
        from trisolaris.visualization import visualize_resource_usage
        
        # Extract data
        generations = [m['generation'] for m in self.metrics.get('resource_usage', [])]
        cpu_usage = [m.get('cpu_percent', 0) for m in self.metrics.get('resource_usage', [])]
        memory_usage = [m.get('memory_percent', 0) for m in self.metrics.get('resource_usage', [])]
        
        output_path = str(self.output_dir / "resource_usage.png")
        visualize_resource_usage(generations, cpu_usage, memory_usage, output_path)
        return output_path
    
    def visualize_ethics_evaluation(self) -> Optional[str]:
        """
        Create a visualization of ethics evaluation results.
        
        Returns:
            Path to the saved visualization file or None if no ethics data available
        """
        # Check if ethics evaluations exist
        if not self.metrics.get('ethics_evaluations'):
            logger.warning("No ethics evaluations found in metrics")
            return None
        
        # Get the latest ethics evaluation
        ethics = self.metrics['ethics_evaluations'][-1]
        
        # Extract category scores if available
        if 'categories' not in ethics:
            logger.warning("No categories found in ethics evaluation")
            return None
        
        categories = list(ethics['categories'].keys())
        scores = [ethics['categories'][cat].get('score', 0) for cat in categories]
        
        if self.interactive:
            try:
                from trisolaris.visualization.interactive import create_ethics_evaluation_plot
                fig = create_ethics_evaluation_plot(self.metrics)
                if fig:
                    output_path = str(self.output_dir / "ethics_evaluation.html")
                    fig.write_html(output_path, include_plotlyjs='cdn')
                    return output_path
            except ImportError:
                logger.warning("Plotly not available, falling back to static visualization")
                self.interactive = False
        
        # Fall back to static visualization
        from trisolaris.visualization import visualize_ethics_evaluation
        output_path = str(self.output_dir / "ethics_evaluation.png")
        visualize_ethics_evaluation(categories, scores, output_path)
        return output_path
    
    def visualize_population_diversity(self) -> str:
        """
        Create a visualization of population diversity metrics.
        
        Returns:
            Path to the saved visualization file
        """
        if self.interactive:
            try:
                from trisolaris.visualization.interactive import create_population_diversity_plot
                fig = create_population_diversity_plot(self.metrics)
                output_path = str(self.output_dir / "population_diversity.html")
                fig.write_html(output_path, include_plotlyjs='cdn')
                return output_path
            except ImportError:
                logger.warning("Plotly not available, falling back to static visualization")
                self.interactive = False
        
        # Fall back to static visualization - create a simple matplotlib plot
        generations = [m['generation'] for m in self.metrics.get('fitness_history', [])]
        
        # Check if diversity metrics exist, if not create placeholder data
        if 'diversity_metrics' in self.metrics:
            genotypic_diversity = [m.get('genotypic_diversity', 0) for m in self.metrics.get('diversity_metrics', [])]
        else:
            # Calculate a simple diversity metric based on fitness standard deviation
            genotypic_diversity = [m.get('std_dev', 0) for m in self.metrics.get('fitness_history', [])]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, genotypic_diversity, 'b-', label='Genotypic Diversity')
        plt.title('Population Diversity')
        plt.xlabel('Generation')
        plt.ylabel('Diversity')
        plt.legend()
        plt.grid(True)
        
        output_path = str(self.output_dir / "population_diversity.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_syntax_errors(self) -> Optional[str]:
        """
        Create a visualization of syntax error rates and repairs.
        
        Returns:
            Path to the saved visualization file or None if no syntax error data available
        """
        # Check if syntax error metrics exist
        if 'syntax_errors' not in self.metrics:
            logger.warning("No syntax error metrics found in metrics")
            return None
        
        if self.interactive:
            try:
                from trisolaris.visualization.interactive import create_syntax_error_plot
                fig = create_syntax_error_plot(self.metrics)
                if fig:
                    output_path = str(self.output_dir / "syntax_errors.html")
                    fig.write_html(output_path, include_plotlyjs='cdn')
                    return output_path
            except ImportError:
                logger.warning("Plotly not available, falling back to static visualization")
                self.interactive = False
        
        # Fall back to static visualization
        generations = [m['generation'] for m in self.metrics.get('syntax_errors', [])]
        error_rates = [m.get('error_rate', 0) for m in self.metrics.get('syntax_errors', [])]
        repair_success = [m.get('repair_success_rate', 0) for m in self.metrics.get('syntax_errors', [])]
        
        plt.figure(figsize=(10, 6))
        plt.plot(generations, error_rates, 'r-', label='Error Rate')
        plt.plot(generations, repair_success, 'g-', label='Repair Success Rate')
        plt.title('Syntax Error Rates and Repairs')
        plt.xlabel('Generation')
        plt.ylabel('Rate')
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(True)
        
        output_path = str(self.output_dir / "syntax_errors.png")
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def visualize_selection_pressure(self) -> Optional[str]:
        """
        Create a visualization of selection pressure metrics.
        
        Returns:
            Path to the saved visualization file or None if no selection pressure data available
        """
        # Check if price equation data exists
        if 'price_equation' not in self.metrics:
            logger.warning("No price equation data found in metrics")
            return None
        
        if self.interactive:
            try:
                from trisolaris.visualization.interactive import create_selection_pressure_plot
                fig = create_selection_pressure_plot(self.metrics)
                if fig:
                    output_path = str(self.output_dir / "selection_pressure.html")
                    fig.write_html(output_path, include_plotlyjs='cdn')
                    return output_path
            except ImportError:
                logger.warning("Plotly not available, falling back to static visualization")
                self.interactive = False
        
        # Fall back to static visualization
        from trisolaris.visualization import visualize_selection_pressure
        
        # Extract data
        generations = [m['generation'] for m in self.metrics.get('price_equation', [])]
        selection_components = [m.get('selection_component', 0) for m in self.metrics.get('price_equation', [])]
        transmission_components = [m.get('transmission_component', 0) for m in self.metrics.get('price_equation', [])]
        
        output_path = str(self.output_dir / "selection_pressure.png")
        visualize_selection_pressure(generations, selection_components, transmission_components, output_path)
        return output_path
    
    def export_data(self, formats: List[str] = None) -> Dict[str, str]:
        """
        Export visualization data to various formats.
        
        Args:
            formats: List of formats to export (default: ['csv', 'json'])
            
        Returns:
            Dictionary mapping data type to file path
        """
        formats = formats or ['csv', 'json']
        output_files = {}
        
        if 'csv' in formats:
            # Export fitness history as CSV
            if 'fitness_history' in self.metrics:
                fitness_df = pd.DataFrame(self.metrics['fitness_history'])
                fitness_csv_path = self.output_dir / 'fitness_history.csv'
                fitness_df.to_csv(fitness_csv_path, index=False)
                output_files['fitness_history_csv'] = str(fitness_csv_path)
            
            # Export resource usage as CSV
            if 'resource_usage' in self.metrics:
                resource_df = pd.DataFrame(self.metrics['resource_usage'])
                resource_csv_path = self.output_dir / 'resource_usage.csv'
                resource_df.to_csv(resource_csv_path, index=False)
                output_files['resource_usage_csv'] = str(resource_csv_path)
            
            # Export price equation data as CSV
            if 'price_equation' in self.metrics:
                price_df = pd.DataFrame(self.metrics['price_equation'])
                price_csv_path = self.output_dir / 'price_equation.csv'
                price_df.to_csv(price_csv_path, index=False)
                output_files['price_equation_csv'] = str(price_csv_path)
        
        if 'json' in formats:
            # Export all metrics as JSON
            json_path = self.output_dir / 'all_metrics.json'
            with open(json_path, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=lambda x: float(x) if isinstance(x, np.float32) else x)
            output_files['all_metrics_json'] = str(json_path)
        
        logger.info(f"Exported visualization data to {self.output_dir}")
        return output_files