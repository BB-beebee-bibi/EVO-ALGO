"""
Resource-Aware Scheduler for the TRISOLARIS framework.

This module provides a scheduler that optimizes resource usage during the evolution process,
dynamically adjusting workloads based on system resource availability.
"""

import time
import logging
import threading
import os
from typing import Dict, Any, List, Optional, Callable, Tuple

from trisolaris.config import get_config, BaseConfig

# Try to import psutil for resource monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not installed. Resource monitoring will be limited.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ResourceScheduler:
    """
    A scheduler that optimizes resource usage during the evolution process.
    
    This class monitors system resources and adjusts the workload accordingly,
    ensuring efficient use of CPU, memory, and other resources.
    """
    
    def __init__(
        self,
        config: Optional[BaseConfig] = None,
        component_name: str = "resource_scheduler",
        run_id: Optional[str] = None,
        target_cpu_usage: Optional[float] = None,
        target_memory_usage: Optional[float] = None,
        min_cpu_available: Optional[float] = None,
        min_memory_available: Optional[float] = None,
        check_interval: Optional[float] = None,
        adaptive_batch_size: Optional[bool] = None,
        initial_batch_size: Optional[int] = None
    ):
        """
        Initialize the resource scheduler.
        
        Args:
            config: Configuration object (if None, will be loaded from global config)
            component_name: Name of this component for configuration lookup
            run_id: Optional run ID for configuration lookup
            target_cpu_usage: Target CPU usage percentage (0-100), overrides config if provided
            target_memory_usage: Target memory usage percentage (0-100), overrides config if provided
            min_cpu_available: Minimum CPU percentage that should remain available, overrides config if provided
            min_memory_available: Minimum memory percentage that should remain available, overrides config if provided
            check_interval: How often to check resource usage (seconds), overrides config if provided
            adaptive_batch_size: Whether to adapt batch size based on resource usage, overrides config if provided
            initial_batch_size: Initial batch size for parallel processing, overrides config if provided
        """
        # Load configuration
        self.config = config or get_config(component_name, run_id)
        self.component_name = component_name
        self.run_id = run_id
        
        # Set parameters from configuration, with override options
        self.target_cpu_usage = target_cpu_usage if target_cpu_usage is not None else self.config.resource_scheduler.target_cpu_usage
        self.target_memory_usage = target_memory_usage if target_memory_usage is not None else self.config.resource_scheduler.target_memory_usage
        self.min_cpu_available = min_cpu_available if min_cpu_available is not None else self.config.resource_scheduler.min_cpu_available
        self.min_memory_available = min_memory_available if min_memory_available is not None else self.config.resource_scheduler.min_memory_available
        self.check_interval = check_interval if check_interval is not None else self.config.resource_scheduler.check_interval
        self.adaptive_batch_size = adaptive_batch_size if adaptive_batch_size is not None else self.config.resource_scheduler.adaptive_batch_size
        self.batch_size = initial_batch_size if initial_batch_size is not None else self.config.resource_scheduler.initial_batch_size
        
        # Resource monitoring state
        self.monitoring_thread = None
        self.monitoring_active = False
        self.resource_usage = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'cpu_available': 100.0,
            'memory_available': 100.0,
            'timestamp': time.time()
        }
        
        # Resource history
        self.resource_history = []
        self.history_max_size = 100  # Keep last 100 measurements
        
        # Process tracking
        self.current_process = None
        if PSUTIL_AVAILABLE:
            self.current_process = psutil.Process()
    
    def start_monitoring(self):
        """Start resource monitoring in a background thread."""
        if not self.monitoring_thread:
            self.monitoring_active = True
            
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=self.check_interval*2)
            self.monitoring_thread = None
            logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Background thread for continuous resource monitoring."""
        while self.monitoring_active:
            try:
                self._check_resources()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
                time.sleep(self.check_interval * 2)  # Back off on error
    
    def _check_resources(self) -> Dict[str, Any]:
        """
        Check current resource usage and update status.
        
        Returns:
            Dictionary with current resource usage
        """
        current_time = time.time()
        
        # Check CPU and memory usage if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                # Get system-wide CPU and memory usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent
                
                # Update resource usage
                self.resource_usage = {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'cpu_available': 100.0 - cpu_percent,
                    'memory_available': 100.0 - memory_percent,
                    'timestamp': current_time
                }
                
                # Add to history
                self.resource_history.append(self.resource_usage.copy())
                
                # Trim history if needed
                if len(self.resource_history) > self.history_max_size:
                    self.resource_history = self.resource_history[-self.history_max_size:]
                
                # Adjust batch size if adaptive batching is enabled
                if self.adaptive_batch_size:
                    self._adjust_batch_size()
                
            except Exception as e:
                logger.error(f"Error checking resource usage: {str(e)}")
        
        return self.resource_usage
    
    def _adjust_batch_size(self):
        """Adjust batch size based on resource usage."""
        cpu_available = self.resource_usage['cpu_available']
        memory_available = self.resource_usage['memory_available']
        
        # Increase batch size if resources are available
        if (cpu_available > self.min_cpu_available * 2 and 
            memory_available > self.min_memory_available * 2):
            self.batch_size = min(self.batch_size + 2, 50)  # Cap at 50
            logger.debug(f"Increased batch size to {self.batch_size}")
        
        # Decrease batch size if resources are constrained
        elif (cpu_available < self.min_cpu_available or 
              memory_available < self.min_memory_available):
            self.batch_size = max(1, self.batch_size - 2)
            logger.debug(f"Decreased batch size to {self.batch_size}")
    
    def get_optimal_batch_size(self) -> int:
        """
        Get the optimal batch size based on current resource usage.
        
        Returns:
            Optimal batch size for parallel processing
        """
        return self.batch_size
    
    def get_optimal_worker_count(self) -> int:
        """
        Get the optimal number of worker processes based on current resource usage.
        
        Returns:
            Optimal number of worker processes
        """
        if not PSUTIL_AVAILABLE:
            return max(1, os.cpu_count() - 1) if os.cpu_count() else 2
        
        # Calculate based on available CPU
        cpu_available = self.resource_usage['cpu_available']
        cpu_count = os.cpu_count() or 4
        
        # If CPU is constrained, use fewer workers
        if cpu_available < self.min_cpu_available:
            return max(1, int(cpu_count * 0.25))
        
        # If CPU is moderately available, use a moderate number of workers
        elif cpu_available < self.min_cpu_available * 2:
            return max(1, int(cpu_count * 0.5))
        
        # If CPU is abundantly available, use more workers
        else:
            return max(1, cpu_count - 1)
    
    def should_throttle(self) -> bool:
        """
        Check if processing should be throttled due to resource constraints.
        
        Returns:
            True if processing should be throttled, False otherwise
        """
        if not PSUTIL_AVAILABLE:
            return False
        
        cpu_available = self.resource_usage['cpu_available']
        memory_available = self.resource_usage['memory_available']
        
        return (cpu_available < self.min_cpu_available or 
                memory_available < self.min_memory_available)
    
    def can_proceed(self) -> bool:
        """
        Check if processing can proceed based on resource availability.
        
        Returns:
            True if processing can proceed, False if resources are constrained
        """
        return not self.should_throttle()
    
    def get_throttle_parameters(self) -> Dict[str, Any]:
        """
        Get parameters for throttling based on current resource usage.
        
        Returns:
            Dictionary with throttling parameters
        """
        cpu_available = self.resource_usage['cpu_available']
        memory_available = self.resource_usage['memory_available']
        
        # Calculate scale factors based on available resources
        cpu_scale = min(1.0, cpu_available / (self.min_cpu_available * 2))
        memory_scale = min(1.0, memory_available / (self.min_memory_available * 2))
        
        # Use the more constrained resource as the limiting factor
        scale_factor = min(cpu_scale, memory_scale)
        
        return {
            'population_scale_factor': scale_factor,
            'worker_count': self.get_optimal_worker_count(),
            'batch_size': self.get_optimal_batch_size(),
            'cpu_available': cpu_available,
            'memory_available': memory_available
        }
    
    def wait_for_resources(self, timeout: float = 60.0) -> bool:
        """
        Wait until sufficient resources are available or timeout is reached.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if resources became available, False if timeout was reached
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if not self.should_throttle():
                return True
            
            # Wait a bit before checking again
            time.sleep(self.check_interval)
        
        return False
    
    def get_resource_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of resource usage measurements.
        
        Returns:
            List of resource usage dictionaries
        """
        return self.resource_history
    
    def get_resource_summary(self) -> Dict[str, Any]:
        """
        Get a summary of resource usage statistics.
        
        Returns:
            Dictionary with resource usage statistics
        """
        if not self.resource_history:
            return {
                'avg_cpu_percent': 0.0,
                'avg_memory_percent': 0.0,
                'max_cpu_percent': 0.0,
                'max_memory_percent': 0.0,
                'min_cpu_available': 100.0,
                'min_memory_available': 100.0
            }
        
        # Calculate statistics
        cpu_percentages = [entry['cpu_percent'] for entry in self.resource_history]
        memory_percentages = [entry['memory_percent'] for entry in self.resource_history]
        cpu_available = [entry['cpu_available'] for entry in self.resource_history]
        memory_available = [entry['memory_available'] for entry in self.resource_history]
        
        return {
            'avg_cpu_percent': sum(cpu_percentages) / len(cpu_percentages),
            'avg_memory_percent': sum(memory_percentages) / len(memory_percentages),
            'max_cpu_percent': max(cpu_percentages),
            'max_memory_percent': max(memory_percentages),
            'min_cpu_available': min(cpu_available),
            'min_memory_available': min(memory_available),
            'current_batch_size': self.batch_size,
            'optimal_worker_count': self.get_optimal_worker_count()
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()
        return False  # Don't suppress exceptions
        
    def update_config(self, config: BaseConfig) -> None:
        """
        Update the scheduler's configuration.
        
        Args:
            config: New configuration object
        """
        self.config = config
        
        # Update parameters from configuration
        self.target_cpu_usage = self.config.resource_scheduler.target_cpu_usage
        self.target_memory_usage = self.config.resource_scheduler.target_memory_usage
        self.min_cpu_available = self.config.resource_scheduler.min_cpu_available
        self.min_memory_available = self.config.resource_scheduler.min_memory_available
        self.check_interval = self.config.resource_scheduler.check_interval
        self.adaptive_batch_size = self.config.resource_scheduler.adaptive_batch_size
        # Don't update batch_size directly as it's dynamically adjusted during runtime

class BatchProcessor:
    """
    A utility class for processing items in resource-aware batches.
    
    This class helps process large collections of items in batches,
    with batch size dynamically adjusted based on resource availability.
    """
    
    def __init__(
        self,
        scheduler: ResourceScheduler,
        items: List[Any],
        process_func: Callable[[Any], Any],
        initial_batch_size: Optional[int] = None,
        config: Optional[BaseConfig] = None
    ):
        """
        Initialize the batch processor.
        
        Args:
            scheduler: ResourceScheduler instance for resource monitoring
            items: List of items to process
            process_func: Function to process each item
            initial_batch_size: Initial batch size (if None, use scheduler's batch size)
            config: Configuration object (if None, will use scheduler's config)
        """
        self.scheduler = scheduler
        self.items = items
        self.process_func = process_func
        self.config = config or scheduler.config
        self.batch_size = initial_batch_size or scheduler.get_optimal_batch_size()
        self.results = []
    
    def process_all(self) -> List[Any]:
        """
        Process all items in resource-aware batches.
        
        Returns:
            List of results from processing all items
        """
        self.results = []
        remaining_items = self.items.copy()
        
        while remaining_items:
            # Get current optimal batch size
            self.batch_size = self.scheduler.get_optimal_batch_size()
            
            # Process a batch
            batch = remaining_items[:self.batch_size]
            batch_results = self._process_batch(batch)
            self.results.extend(batch_results)
            
            # Remove processed items
            remaining_items = remaining_items[self.batch_size:]
            
            # Check if we need to throttle
            if self.scheduler.should_throttle() and remaining_items:
                logger.info("Throttling batch processing due to resource constraints")
                self.scheduler.wait_for_resources(timeout=5.0)
        
        return self.results
    
    def _process_batch(self, batch: List[Any]) -> List[Any]:
        """
        Process a batch of items.
        
        Args:
            batch: List of items to process
            
        Returns:
            List of results from processing the batch
        """
        return [self.process_func(item) for item in batch]