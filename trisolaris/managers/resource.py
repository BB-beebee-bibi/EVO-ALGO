"""
Resource Steward for the TRISOLARIS framework.

This module implements resource monitoring and management to ensure the evolutionary process
doesn't overwhelm system resources, maintaining a minimum of 25% resource availability.
"""

import os
import time
import logging
import platform
import threading
from typing import Dict, Optional, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import psutil
except ImportError:
    logger.warning("psutil not installed. Resource monitoring will be limited.")
    psutil = None

class ResourceSteward:
    """
    Monitors and manages system resources during the evolutionary process.
    
    Tracks CPU, memory, and disk usage to ensure the evolutionary process doesn't
    overwhelm the system, maintaining a minimum of 25% resource availability at all times.
    """
    
    def __init__(
        self,
        min_available_memory: float = 0.25,  # 25% memory must remain available
        min_available_cpu: float = 0.25,     # 25% CPU must remain available
        min_available_disk: float = 0.25,    # 25% disk space must remain available
        check_interval: float = 5.0,         # Check resources every 5 seconds
        monitoring_enabled: bool = True,     # Enable continuous monitoring
    ):
        """
        Initialize the Resource Steward.
        
        Args:
            min_available_memory: Minimum fraction of memory that should remain available
            min_available_cpu: Minimum fraction of CPU that should remain available
            min_available_disk: Minimum fraction of disk space that should remain available
            check_interval: How often to check resources (in seconds)
            monitoring_enabled: Whether to enable continuous monitoring in a background thread
        """
        self.min_available_memory = min_available_memory
        self.min_available_cpu = min_available_cpu
        self.min_available_disk = min_available_disk
        self.check_interval = check_interval
        self.monitoring_enabled = monitoring_enabled and psutil is not None
        
        # Resource usage history
        self.history = {
            'timestamp': [],
            'memory_used': [],
            'cpu_used': [],
            'disk_used': []
        }
        
        # Current status
        self.status = {
            'memory_available': 1.0,
            'cpu_available': 1.0,
            'disk_available': 1.0,
            'can_proceed': True,
            'last_check': time.time()
        }
        
        # Throttling state
        self.throttle_level = 0  # 0: none, 1: light, 2: moderate, 3: severe
        
        # Start monitoring thread if enabled
        self.monitoring_thread = None
        if self.monitoring_enabled:
            self.start_monitoring()
    
    def start_monitoring(self):
        """Start the background monitoring thread."""
        if not self.monitoring_thread:
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop the background monitoring thread."""
        self.monitoring_enabled = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=self.check_interval*2)
            self.monitoring_thread = None
            logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Background thread for continuous resource monitoring."""
        while self.monitoring_enabled:
            try:
                self.check_resources()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in resource monitoring: {str(e)}")
                time.sleep(self.check_interval * 2)  # Back off on error
    
    def check_resources(self) -> Dict[str, float]:
        """
        Check current resource usage and update status.
        
        Returns:
            Dictionary with current resource availability fractions
        """
        # Default values if checks fail
        memory_available = 1.0
        cpu_available = 1.0
        disk_available = 1.0
        
        timestamp = time.time()
        
        try:
            if psutil:
                # Memory check
                mem = psutil.virtual_memory()
                memory_available = mem.available / mem.total
                
                # CPU check (average over a short interval)
                cpu_used = psutil.cpu_percent(interval=0.1) / 100.0
                cpu_available = 1.0 - cpu_used
                
                # Disk check for the current working directory
                disk = psutil.disk_usage(os.getcwd())
                disk_available = disk.free / disk.total
                
                # Update history
                self.history['timestamp'].append(timestamp)
                self.history['memory_used'].append(1.0 - memory_available)
                self.history['cpu_used'].append(1.0 - cpu_available)
                self.history['disk_used'].append(1.0 - disk_available)
                
                # Trim history if it gets too long
                if len(self.history['timestamp']) > 1000:
                    for key in self.history:
                        self.history[key] = self.history[key][-1000:]
            else:
                # Fallback to simple checks if psutil not available
                memory_available = 0.5  # Assume 50% available as a safe default
                cpu_available = 0.5
                disk_available = 0.5
        except Exception as e:
            logger.error(f"Resource check failed: {str(e)}")
            # Assume limited resources on failure
            memory_available = 0.3
            cpu_available = 0.3
            disk_available = 0.3
        
        # Update status
        self.status['memory_available'] = memory_available
        self.status['cpu_available'] = cpu_available
        self.status['disk_available'] = disk_available
        self.status['last_check'] = timestamp
        
        # Check if we can proceed
        can_proceed = (
            memory_available >= self.min_available_memory and
            cpu_available >= self.min_available_cpu and
            disk_available >= self.min_available_disk
        )
        
        # Set throttling level
        if can_proceed:
            # No throttling needed
            self.throttle_level = 0
        else:
            # Determine throttling level based on resource constraints
            constraint_level = 0
            if memory_available < self.min_available_memory:
                constraint_level = max(constraint_level, 
                                     int(3 * (1 - memory_available / self.min_available_memory)))
            if cpu_available < self.min_available_cpu:
                constraint_level = max(constraint_level,
                                     int(3 * (1 - cpu_available / self.min_available_cpu)))
            if disk_available < self.min_available_disk:
                constraint_level = max(constraint_level,
                                     int(3 * (1 - disk_available / self.min_available_disk)))
            
            self.throttle_level = min(3, max(1, constraint_level))
        
        self.status['can_proceed'] = can_proceed
        
        return {
            'memory_available': memory_available,
            'cpu_available': cpu_available,
            'disk_available': disk_available,
            'can_proceed': can_proceed,
            'throttle_level': self.throttle_level
        }
    
    def can_proceed(self) -> bool:
        """
        Check if operations can proceed based on resource availability.
        
        Returns:
            True if sufficient resources are available, False otherwise
        """
        # Recheck resources if last check is too old
        if time.time() - self.status['last_check'] > self.check_interval:
            self.check_resources()
        
        return self.status['can_proceed']
    
    def get_throttle_level(self) -> int:
        """
        Get the current throttling level.
        
        Returns:
            0: No throttling
            1: Light throttling (reduce population size, increase delay)
            2: Moderate throttling (significantly reduce operations)
            3: Severe throttling (pause operations)
        """
        return self.throttle_level
    
    def get_throttling_parameters(self) -> Dict[str, float]:
        """
        Get parameters for adjusting operations based on current throttle level.
        
        Returns:
            Dictionary with recommended parameters:
            - population_scale_factor: Factor to multiply population size by
            - delay_factor: Factor to multiply delays by
            - checkpoint_interval: How often to save checkpoints (in generations)
        """
        if self.throttle_level == 0:
            return {
                'population_scale_factor': 1.0,
                'delay_factor': 1.0,
                'checkpoint_interval': 10
            }
        elif self.throttle_level == 1:
            return {
                'population_scale_factor': 0.75,
                'delay_factor': 1.5,
                'checkpoint_interval': 5
            }
        elif self.throttle_level == 2:
            return {
                'population_scale_factor': 0.5,
                'delay_factor': 2.0,
                'checkpoint_interval': 3
            }
        else:  # throttle_level == 3
            return {
                'population_scale_factor': 0.25,
                'delay_factor': 4.0,
                'checkpoint_interval': 1
            }
    
    def wait_for_resources(self, timeout: Optional[float] = None) -> bool:
        """
        Wait until sufficient resources are available.
        
        Args:
            timeout: Maximum time to wait in seconds (None = wait indefinitely)
            
        Returns:
            True if resources became available, False if timeout reached
        """
        start_time = time.time()
        while not self.can_proceed():
            time.sleep(self.check_interval)
            if timeout and (time.time() - start_time > timeout):
                logger.warning(f"Resource wait timeout after {timeout}s")
                return False
        return True
    
    def get_system_info(self) -> Dict[str, str]:
        """
        Get system information for logging and monitoring.
        
        Returns:
            Dictionary with system information
        """
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
        }
        
        if psutil:
            try:
                mem = psutil.virtual_memory()
                info['total_memory'] = f"{mem.total / (1024**3):.2f} GB"
                info['cpu_count'] = str(psutil.cpu_count())
                info['cpu_freq'] = str(psutil.cpu_freq().current) if hasattr(psutil.cpu_freq(), 'current') else 'Unknown'
            except:
                pass
        
        return info
    
    def get_resource_history(self, limit: int = 100) -> Dict[str, List]:
        """
        Get resource usage history for visualization.
        
        Args:
            limit: Maximum number of history points to return
            
        Returns:
            Dictionary with resource usage history
        """
        history = {}
        for key in self.history:
            history[key] = self.history[key][-limit:]
        return history
    
    def generate_report(self) -> str:
        """
        Generate a text report of resource usage.
        
        Returns:
            Resource usage report as a string
        """
        if not psutil:
            return "Resource monitoring disabled (psutil not available)"
        
        status = self.check_resources()
        
        report = []
        report.append("=== RESOURCE USAGE REPORT ===")
        report.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Memory: {(1.0 - status['memory_available']) * 100:.1f}% used, "
                     f"{status['memory_available'] * 100:.1f}% available")
        report.append(f"CPU: {(1.0 - status['cpu_available']) * 100:.1f}% used, "
                     f"{status['cpu_available'] * 100:.1f}% available")
        report.append(f"Disk: {(1.0 - status['disk_available']) * 100:.1f}% used, "
                     f"{status['disk_available'] * 100:.1f}% available")
        report.append(f"Status: {'Sufficient resources' if status['can_proceed'] else 'Resource constrained'}")
        report.append(f"Throttle level: {self.throttle_level}")
        report.append("==========================")
        
        return "\n".join(report)
    
    def __str__(self) -> str:
        """String representation showing current resource status."""
        status = self.status
        return (
            f"ResourceSteward(memory={status['memory_available']*100:.1f}%, "
            f"cpu={status['cpu_available']*100:.1f}%, "
            f"disk={status['disk_available']*100:.1f}%, "
            f"throttle={self.throttle_level})"
        )
