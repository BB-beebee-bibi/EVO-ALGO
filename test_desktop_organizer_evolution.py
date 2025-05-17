#!/usr/bin/env python3
"""
Test script to run evolution for the desktop organizer task.
"""
import os
import sys
import logging
from pathlib import Path

from trisolaris.tasks.desktop_organizer_task import DesktopOrganizerTask
from trisolaris.task_runner import TaskRunner

def main():
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("desktop_organizer_test.log")
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Create task instance
    task = DesktopOrganizerTask()
    
    # Create task runner with test parameters
    runner = TaskRunner(
        task=task,
        output_dir="test_outputs",
        num_generations=5,  # Small number for testing
        population_size=20,  # Small population for testing
        mutation_rate=0.1,
        crossover_rate=0.7,
        ethics_level="basic",
        resource_monitoring=False,  # Disable resource monitoring
        show_resource_report=True
    )
    
    # Run evolution
    logger.info("Starting test evolution run...")
    best_genome, stats = runner.run()
    
    # Print results
    logger.info("\nEvolution Results:")
    logger.info(f"Best Fitness: {stats['best_fitness']:.4f}")
    logger.info(f"Duration: {stats['duration_seconds']:.2f} seconds")
    logger.info(f"Output saved to: {stats['output_path']}")
    
    # Print generation metrics
    logger.info("\nGeneration Metrics:")
    for metric in stats['metrics']:
        logger.info(f"Generation {metric['generation']}: "
                   f"Fitness={metric['best_fitness']:.4f}, "
                   f"Time={metric['time_seconds']:.2f}s")

if __name__ == "__main__":
    main() 