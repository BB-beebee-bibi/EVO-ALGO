"""
Sandboxed Evolution Environment for the TRISOLARIS framework.

This module implements a secure, isolated environment for evolutionary algorithms
to operate within defined resource constraints and security boundaries. It provides
a controlled execution environment that prevents file modification/deletion outside
the sandbox while monitoring resource usage.
"""

import os
import time
import shutil
import logging
import tempfile
import threading
import traceback
from typing import Dict, Any, Optional, List, Callable, Tuple
from pathlib import Path

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

class SandboxViolationError(Exception):
    """Exception raised when a sandbox violation is detected."""
    pass

class ResourceLimitExceededError(Exception):
    """Exception raised when a resource limit is exceeded."""
    pass

class ExecutionTimeoutError(Exception):
    """Exception raised when execution time exceeds the limit."""
    pass

class SandboxedEnvironment:
    """
    Provides an isolated environment for safe evolution of code.
    
    This class creates a sandboxed directory structure where evolutionary processes
    can safely execute without affecting the rest of the system. It monitors resource
    usage and enforces execution time constraints.
    """
    
    def __init__(
        self,
        base_dir: Optional[str] = None,
        config: Optional[BaseConfig] = None,
        component_name: str = "sandbox",
        run_id: Optional[str] = None
    ):
        """
        Initialize the sandboxed environment.
        
        Args:
            base_dir: Base directory for the sandbox (if None, a temporary directory is created)
            config: Configuration object (if None, will be loaded from global config)
            component_name: Name of this component for configuration lookup
            run_id: Optional run ID for configuration lookup
        """
        # Load configuration
        self.config = config or get_config(component_name, run_id)
        self.component_name = component_name
        self.run_id = run_id
        
        # Extract configuration parameters
        self.max_cpu_percent = self.config.sandbox.resource_limits.max_cpu_percent
        self.max_memory_percent = self.config.sandbox.resource_limits.max_memory_percent
        self.max_execution_time = self.config.sandbox.resource_limits.max_execution_time
        self.check_interval = self.config.sandbox.resource_limits.check_interval
        self.preserve_sandbox = self.config.sandbox.preserve_sandbox
        
        # Override base_dir if provided
        if base_dir:
            self.config.sandbox.base_dir = base_dir
        
        # Create sandbox directory
        if self.config.sandbox.base_dir:
            self.base_dir = Path(self.config.sandbox.base_dir)
            os.makedirs(self.base_dir, exist_ok=True)
            self.temp_dir = None
        else:
            self.temp_dir = tempfile.TemporaryDirectory(prefix="trisolaris_sandbox_")
            self.base_dir = Path(self.temp_dir.name)
        
        # Create sandbox structure
        self.sandbox_dirs = {
            'code': self.base_dir / 'code',
            'data': self.base_dir / 'data',
            'output': self.base_dir / 'output',
            'temp': self.base_dir / 'temp',
            'logs': self.base_dir / 'logs'
        }
        
        for dir_path in self.sandbox_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Resource monitoring state
        self.monitoring_thread = None
        self.monitoring_active = False
        self.resource_usage = {
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'execution_time': 0.0,
            'start_time': None,
            'violations': []
        }
        
        # Process tracking
        self.current_process = None
        self.child_processes = []
        
        logger.info(f"Initialized sandbox environment at {self.base_dir}")
    
    def start_monitoring(self):
        """Start resource monitoring in a background thread."""
        if not self.monitoring_thread:
            self.monitoring_active = True
            self.resource_usage['start_time'] = time.time()
            self.resource_usage['violations'] = []
            
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
        # Update execution time
        current_time = time.time()
        if self.resource_usage['start_time']:
            self.resource_usage['execution_time'] = current_time - self.resource_usage['start_time']
        
        # Check if execution time exceeded
        if self.resource_usage['execution_time'] > self.max_execution_time:
            violation = {
                'type': 'execution_time',
                'value': self.resource_usage['execution_time'],
                'limit': self.max_execution_time,
                'timestamp': current_time
            }
            self.resource_usage['violations'].append(violation)
            logger.warning(f"Execution time limit exceeded: {self.resource_usage['execution_time']:.2f}s > {self.max_execution_time}s")
        
        # Check CPU and memory usage if psutil is available
        if PSUTIL_AVAILABLE:
            try:
                # Get current process
                if not self.current_process:
                    self.current_process = psutil.Process()
                
                # Get CPU usage (including children)
                cpu_percent = self.current_process.cpu_percent(interval=0.1)
                for child in self.current_process.children(recursive=True):
                    try:
                        self.child_processes.append(child)
                        cpu_percent += child.cpu_percent(interval=0.1)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                self.resource_usage['cpu_percent'] = cpu_percent
                
                # Check if CPU usage exceeded
                if cpu_percent > self.max_cpu_percent:
                    violation = {
                        'type': 'cpu_percent',
                        'value': cpu_percent,
                        'limit': self.max_cpu_percent,
                        'timestamp': current_time
                    }
                    self.resource_usage['violations'].append(violation)
                    logger.warning(f"CPU usage limit exceeded: {cpu_percent:.2f}% > {self.max_cpu_percent}%")
                
                # Get memory usage (including children)
                memory_info = self.current_process.memory_info()
                memory_percent = memory_info.rss / psutil.virtual_memory().total * 100
                
                for child in self.child_processes:
                    try:
                        if child.is_running():
                            child_memory = child.memory_info().rss
                            memory_percent += child_memory / psutil.virtual_memory().total * 100
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                self.resource_usage['memory_percent'] = memory_percent
                
                # Check if memory usage exceeded
                if memory_percent > self.max_memory_percent:
                    violation = {
                        'type': 'memory_percent',
                        'value': memory_percent,
                        'limit': self.max_memory_percent,
                        'timestamp': current_time
                    }
                    self.resource_usage['violations'].append(violation)
                    logger.warning(f"Memory usage limit exceeded: {memory_percent:.2f}% > {self.max_memory_percent}%")
                
            except Exception as e:
                logger.error(f"Error checking resource usage: {str(e)}")
        
        return self.resource_usage
    
    def execute_in_sandbox(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function within the sandbox environment with resource monitoring.
        
        Args:
            func: Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            Result of the function execution
            
        Raises:
            ResourceLimitExceededError: If resource limits are exceeded
            ExecutionTimeoutError: If execution time limit is exceeded
            SandboxViolationError: If sandbox boundaries are violated
        """
        # Start resource monitoring
        self.start_monitoring()
        
        result = None
        error = None
        
        try:
            # Execute the function
            result = func(*args, **kwargs)
            
        except Exception as e:
            error = e
            logger.error(f"Error in sandboxed execution: {str(e)}")
            logger.debug(traceback.format_exc())
        
        finally:
            # Stop resource monitoring
            self.stop_monitoring()
            
            # Check for violations
            if self.resource_usage['violations']:
                for violation in self.resource_usage['violations']:
                    if violation['type'] == 'execution_time':
                        raise ExecutionTimeoutError(
                            f"Execution time limit exceeded: {violation['value']:.2f}s > {violation['limit']}s"
                        )
                    elif violation['type'] == 'cpu_percent':
                        raise ResourceLimitExceededError(
                            f"CPU usage limit exceeded: {violation['value']:.2f}% > {violation['limit']}%"
                        )
                    elif violation['type'] == 'memory_percent':
                        raise ResourceLimitExceededError(
                            f"Memory usage limit exceeded: {violation['value']:.2f}% > {violation['limit']}%"
                        )
            
            # Re-raise original error if any
            if error:
                raise error
        
        return result
    
    def get_sandbox_path(self, subdir: str = None, filename: str = None) -> Path:
        """
        Get a path within the sandbox.
        
        Args:
            subdir: Subdirectory within the sandbox ('code', 'data', 'output', 'temp', 'logs')
            filename: Optional filename to append to the path
            
        Returns:
            Path object for the requested location
            
        Raises:
            ValueError: If the requested subdirectory doesn't exist
        """
        if subdir and subdir not in self.sandbox_dirs:
            raise ValueError(f"Invalid sandbox subdirectory: {subdir}")
        
        base = self.base_dir if not subdir else self.sandbox_dirs[subdir]
        
        if filename:
            return base / filename
        
        return base
    
    def save_file(self, content: str, subdir: str, filename: str) -> Path:
        """
        Save content to a file within the sandbox.
        
        Args:
            content: Content to save
            subdir: Subdirectory within the sandbox
            filename: Filename to save as
            
        Returns:
            Path to the saved file
            
        Raises:
            ValueError: If the requested subdirectory doesn't exist
        """
        file_path = self.get_sandbox_path(subdir, filename)
        
        with open(file_path, 'w') as f:
            f.write(content)
        
        logger.debug(f"Saved file to {file_path}")
        return file_path
    
    def read_file(self, subdir: str, filename: str) -> str:
        """
        Read content from a file within the sandbox.
        
        Args:
            subdir: Subdirectory within the sandbox
            filename: Filename to read
            
        Returns:
            Content of the file
            
        Raises:
            ValueError: If the requested subdirectory doesn't exist
            FileNotFoundError: If the file doesn't exist
        """
        file_path = self.get_sandbox_path(subdir, filename)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
        
        return content
    
    def list_files(self, subdir: str = None) -> List[Path]:
        """
        List files within a sandbox subdirectory.
        
        Args:
            subdir: Subdirectory within the sandbox (None for base directory)
            
        Returns:
            List of Path objects for files in the directory
            
        Raises:
            ValueError: If the requested subdirectory doesn't exist
        """
        dir_path = self.get_sandbox_path(subdir)
        
        return [f for f in dir_path.iterdir() if f.is_file()]
    
    def cleanup(self):
        """
        Clean up the sandbox environment.
        
        This removes the temporary directory if one was created and not set to be preserved.
        """
        # Stop monitoring if active
        if self.monitoring_active:
            self.stop_monitoring()
        
        # Clean up temporary directory if created and not preserved
        if self.temp_dir and not self.config.sandbox.preserve_sandbox:
            self.temp_dir.cleanup()
            logger.info(f"Cleaned up sandbox environment at {self.base_dir}")
        elif self.config.sandbox.preserve_sandbox:
            logger.info(f"Preserved sandbox environment at {self.base_dir}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False  # Don't suppress exceptions
    
    def get_resource_usage_report(self) -> Dict[str, Any]:
        """
        Get a report of resource usage during execution.
        
        Returns:
            Dictionary with resource usage statistics
        """
        return {
            'cpu_percent': self.resource_usage['cpu_percent'],
            'memory_percent': self.resource_usage['memory_percent'],
            'execution_time': self.resource_usage['execution_time'],
            'violations': self.resource_usage['violations'],
            'limits': {
                'cpu_percent': self.max_cpu_percent,
                'memory_percent': self.max_memory_percent,
                'execution_time': self.max_execution_time
            }
        }
    
    def log_execution(self, operation: str, details: Dict[str, Any] = None):
        """
        Log an execution operation to the sandbox logs.
        
        Args:
            operation: Name of the operation being performed
            details: Additional details to log
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'operation': operation,
            'details': details or {}
        }
        
        # Append to log file
        log_file = self.get_sandbox_path('logs', 'execution.log')
        with open(log_file, 'a') as f:
            f.write(f"{timestamp} - {operation}")
            if details:
                f.write(f": {details}")
            f.write('\n')
        
        logger.debug(f"Logged execution: {operation}")
        
    def update_config(self, config: BaseConfig) -> None:
        """
        Update the sandbox's configuration.
        
        Args:
            config: New configuration object
        """
        self.config = config
        
        # Update parameters from configuration
        self.max_cpu_percent = self.config.sandbox.resource_limits.max_cpu_percent
        self.max_memory_percent = self.config.sandbox.resource_limits.max_memory_percent
        self.max_execution_time = self.config.sandbox.resource_limits.max_execution_time
        self.check_interval = self.config.sandbox.resource_limits.check_interval
        self.preserve_sandbox = self.config.sandbox.preserve_sandbox