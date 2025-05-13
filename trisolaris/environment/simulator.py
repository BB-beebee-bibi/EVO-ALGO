"""
Resource Simulator for the TRISOLARIS framework.

This module implements simulated resources for the sandboxed evolution environment,
providing controlled access to file system operations, network interfaces, system
resources, and input/output streams.
"""

import os
import re
import time
import json
import random
import logging
import tempfile
from typing import Dict, Any, Optional, List, Union, Tuple, BinaryIO, TextIO
from pathlib import Path
from io import StringIO, BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimulatedFileSystem:
    """
    Simulates a file system within the sandbox environment.
    
    This class provides controlled access to file operations, ensuring that
    all operations are confined to the sandbox directory structure.
    """
    
    def __init__(self, sandbox_dir: Path):
        """
        Initialize the simulated file system.
        
        Args:
            sandbox_dir: Base directory for the sandbox
        """
        self.sandbox_dir = sandbox_dir
        self.open_files: Dict[int, Union[TextIO, BinaryIO]] = {}
        self.file_counter = 0
        
        logger.debug(f"Initialized simulated file system in {sandbox_dir}")
    
    def _validate_path(self, path: Union[str, Path]) -> Path:
        """
        Validate that a path is within the sandbox.
        
        Args:
            path: Path to validate
            
        Returns:
            Resolved Path object
            
        Raises:
            PermissionError: If the path is outside the sandbox
        """
        # Convert to Path object if it's a string
        if isinstance(path, str):
            path = Path(path)
        
        # Resolve to absolute path
        resolved_path = (self.sandbox_dir / path).resolve()
        
        # Check if the path is within the sandbox
        if not str(resolved_path).startswith(str(self.sandbox_dir.resolve())):
            logger.warning(f"Attempted access to path outside sandbox: {path}")
            raise PermissionError(f"Access denied: {path} is outside the sandbox")
        
        return resolved_path
    
    def open(self, path: Union[str, Path], mode: str = 'r', **kwargs) -> int:
        """
        Open a file within the sandbox.
        
        Args:
            path: Path to the file
            mode: File open mode
            **kwargs: Additional arguments for open()
            
        Returns:
            File descriptor (integer)
            
        Raises:
            PermissionError: If the path is outside the sandbox
            FileNotFoundError: If the file doesn't exist (for read modes)
            IOError: If there's an error opening the file
        """
        resolved_path = self._validate_path(path)
        
        # Create parent directories if needed for write modes
        if 'w' in mode or 'a' in mode or '+' in mode:
            os.makedirs(resolved_path.parent, exist_ok=True)
        
        try:
            # Open the file
            file_obj = open(resolved_path, mode, **kwargs)
            
            # Assign a file descriptor
            fd = self.file_counter
            self.file_counter += 1
            
            # Store the file object
            self.open_files[fd] = file_obj
            
            logger.debug(f"Opened file {resolved_path} with fd {fd}")
            return fd
            
        except Exception as e:
            logger.error(f"Error opening file {resolved_path}: {str(e)}")
            raise
    
    def read(self, fd: int, size: int = -1) -> Union[str, bytes]:
        """
        Read from an open file.
        
        Args:
            fd: File descriptor
            size: Number of bytes/chars to read (-1 for all)
            
        Returns:
            Content read from the file
            
        Raises:
            ValueError: If the file descriptor is invalid
            IOError: If there's an error reading the file
        """
        if fd not in self.open_files:
            raise ValueError(f"Invalid file descriptor: {fd}")
        
        try:
            return self.open_files[fd].read(size)
        except Exception as e:
            logger.error(f"Error reading from fd {fd}: {str(e)}")
            raise
    
    def write(self, fd: int, data: Union[str, bytes]) -> int:
        """
        Write to an open file.
        
        Args:
            fd: File descriptor
            data: Data to write
            
        Returns:
            Number of bytes/chars written
            
        Raises:
            ValueError: If the file descriptor is invalid
            IOError: If there's an error writing to the file
        """
        if fd not in self.open_files:
            raise ValueError(f"Invalid file descriptor: {fd}")
        
        try:
            return self.open_files[fd].write(data)
        except Exception as e:
            logger.error(f"Error writing to fd {fd}: {str(e)}")
            raise
    
    def close(self, fd: int) -> None:
        """
        Close an open file.
        
        Args:
            fd: File descriptor
            
        Raises:
            ValueError: If the file descriptor is invalid
        """
        if fd not in self.open_files:
            raise ValueError(f"Invalid file descriptor: {fd}")
        
        try:
            self.open_files[fd].close()
            del self.open_files[fd]
            logger.debug(f"Closed fd {fd}")
        except Exception as e:
            logger.error(f"Error closing fd {fd}: {str(e)}")
            raise
    
    def list_dir(self, path: Union[str, Path] = '.') -> List[str]:
        """
        List contents of a directory within the sandbox.
        
        Args:
            path: Path to the directory
            
        Returns:
            List of filenames in the directory
            
        Raises:
            PermissionError: If the path is outside the sandbox
            FileNotFoundError: If the directory doesn't exist
        """
        resolved_path = self._validate_path(path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not resolved_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        
        return [item.name for item in resolved_path.iterdir()]
    
    def make_dir(self, path: Union[str, Path], mode: int = 0o777, exist_ok: bool = False) -> None:
        """
        Create a directory within the sandbox.
        
        Args:
            path: Path to the directory
            mode: Directory permissions
            exist_ok: Whether to ignore if directory exists
            
        Raises:
            PermissionError: If the path is outside the sandbox
            FileExistsError: If the directory exists and exist_ok is False
        """
        resolved_path = self._validate_path(path)
        
        try:
            os.makedirs(resolved_path, mode=mode, exist_ok=exist_ok)
            logger.debug(f"Created directory {resolved_path}")
        except Exception as e:
            logger.error(f"Error creating directory {resolved_path}: {str(e)}")
            raise
    
    def remove(self, path: Union[str, Path]) -> None:
        """
        Remove a file within the sandbox.
        
        Args:
            path: Path to the file
            
        Raises:
            PermissionError: If the path is outside the sandbox
            FileNotFoundError: If the file doesn't exist
            IsADirectoryError: If the path is a directory
        """
        resolved_path = self._validate_path(path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if resolved_path.is_dir():
            raise IsADirectoryError(f"Is a directory: {path}")
        
        try:
            os.remove(resolved_path)
            logger.debug(f"Removed file {resolved_path}")
        except Exception as e:
            logger.error(f"Error removing file {resolved_path}: {str(e)}")
            raise
    
    def remove_dir(self, path: Union[str, Path]) -> None:
        """
        Remove a directory within the sandbox.
        
        Args:
            path: Path to the directory
            
        Raises:
            PermissionError: If the path is outside the sandbox
            FileNotFoundError: If the directory doesn't exist
            NotADirectoryError: If the path is not a directory
            OSError: If the directory is not empty
        """
        resolved_path = self._validate_path(path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        if not resolved_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {path}")
        
        try:
            os.rmdir(resolved_path)
            logger.debug(f"Removed directory {resolved_path}")
        except Exception as e:
            logger.error(f"Error removing directory {resolved_path}: {str(e)}")
            raise
    
    def exists(self, path: Union[str, Path]) -> bool:
        """
        Check if a path exists within the sandbox.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path exists, False otherwise
            
        Raises:
            PermissionError: If the path is outside the sandbox
        """
        resolved_path = self._validate_path(path)
        return resolved_path.exists()
    
    def is_file(self, path: Union[str, Path]) -> bool:
        """
        Check if a path is a file within the sandbox.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path is a file, False otherwise
            
        Raises:
            PermissionError: If the path is outside the sandbox
        """
        resolved_path = self._validate_path(path)
        return resolved_path.is_file()
    
    def is_dir(self, path: Union[str, Path]) -> bool:
        """
        Check if a path is a directory within the sandbox.
        
        Args:
            path: Path to check
            
        Returns:
            True if the path is a directory, False otherwise
            
        Raises:
            PermissionError: If the path is outside the sandbox
        """
        resolved_path = self._validate_path(path)
        return resolved_path.is_dir()
    
    def get_size(self, path: Union[str, Path]) -> int:
        """
        Get the size of a file within the sandbox.
        
        Args:
            path: Path to the file
            
        Returns:
            Size of the file in bytes
            
        Raises:
            PermissionError: If the path is outside the sandbox
            FileNotFoundError: If the file doesn't exist
            IsADirectoryError: If the path is a directory
        """
        resolved_path = self._validate_path(path)
        
        if not resolved_path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if resolved_path.is_dir():
            raise IsADirectoryError(f"Is a directory: {path}")
        
        return resolved_path.stat().st_size
    
    def cleanup(self) -> None:
        """Close all open files."""
        for fd in list(self.open_files.keys()):
            try:
                self.close(fd)
            except Exception as e:
                logger.error(f"Error closing fd {fd} during cleanup: {str(e)}")


class SimulatedNetwork:
    """
    Simulates network interfaces and connections within the sandbox environment.
    
    This class provides controlled access to network operations, returning
    simulated responses instead of making actual network connections.
    """
    
    def __init__(self):
        """Initialize the simulated network."""
        self.connections = {}
        self.connection_counter = 0
        
        # Predefined responses for common requests
        self.predefined_responses = {
            'GET': {
                r'https?://api\.example\.com/data': {
                    'status': 200,
                    'headers': {'Content-Type': 'application/json'},
                    'content': json.dumps({'data': [1, 2, 3, 4, 5], 'status': 'success'})
                },
                r'https?://api\.example\.com/users': {
                    'status': 200,
                    'headers': {'Content-Type': 'application/json'},
                    'content': json.dumps({'users': [{'id': 1, 'name': 'User 1'}, {'id': 2, 'name': 'User 2'}]})
                },
                r'https?://.*': {
                    'status': 200,
                    'headers': {'Content-Type': 'text/html'},
                    'content': '<html><body><h1>Simulated Response</h1><p>This is a simulated HTTP response.</p></body></html>'
                }
            },
            'POST': {
                r'https?://api\.example\.com/data': {
                    'status': 201,
                    'headers': {'Content-Type': 'application/json'},
                    'content': json.dumps({'status': 'created', 'id': 12345})
                },
                r'https?://.*': {
                    'status': 200,
                    'headers': {'Content-Type': 'application/json'},
                    'content': json.dumps({'status': 'success'})
                }
            }
        }
        
        logger.debug("Initialized simulated network")
    
    def connect(self, host: str, port: int) -> int:
        """
        Simulate connecting to a network host.
        
        Args:
            host: Hostname or IP address
            port: Port number
            
        Returns:
            Connection ID
            
        Raises:
            ConnectionRefusedError: If the connection is refused
            TimeoutError: If the connection times out
        """
        # Simulate connection failures for certain hosts/ports
        if host in ['localhost', '127.0.0.1'] and port in [22, 3306, 5432]:
            raise ConnectionRefusedError(f"Connection refused to {host}:{port}")
        
        # Simulate timeouts for certain hosts
        if host in ['slow.example.com', '10.0.0.1']:
            raise TimeoutError(f"Connection timed out to {host}:{port}")
        
        # Create a new connection
        conn_id = self.connection_counter
        self.connection_counter += 1
        
        self.connections[conn_id] = {
            'host': host,
            'port': port,
            'status': 'connected',
            'created_at': time.time()
        }
        
        logger.debug(f"Simulated connection to {host}:{port} with ID {conn_id}")
        return conn_id
    
    def send(self, conn_id: int, data: Union[str, bytes]) -> int:
        """
        Simulate sending data over a connection.
        
        Args:
            conn_id: Connection ID
            data: Data to send
            
        Returns:
            Number of bytes sent
            
        Raises:
            ValueError: If the connection ID is invalid
            ConnectionError: If the connection is closed
        """
        if conn_id not in self.connections:
            raise ValueError(f"Invalid connection ID: {conn_id}")
        
        if self.connections[conn_id]['status'] != 'connected':
            raise ConnectionError(f"Connection {conn_id} is not connected")
        
        # Convert string to bytes if needed
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Simulate sending data
        logger.debug(f"Simulated sending {len(data)} bytes on connection {conn_id}")
        return len(data)
    
    def recv(self, conn_id: int, size: int = 1024) -> bytes:
        """
        Simulate receiving data from a connection.
        
        Args:
            conn_id: Connection ID
            size: Maximum number of bytes to receive
            
        Returns:
            Received data
            
        Raises:
            ValueError: If the connection ID is invalid
            ConnectionError: If the connection is closed
        """
        if conn_id not in self.connections:
            raise ValueError(f"Invalid connection ID: {conn_id}")
        
        if self.connections[conn_id]['status'] != 'connected':
            raise ConnectionError(f"Connection {conn_id} is not connected")
        
        # Generate simulated response based on connection details
        host = self.connections[conn_id]['host']
        port = self.connections[conn_id]['port']
        
        # Simulate different responses based on host and port
        if port == 80 or port == 443:
            response = f"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n\r\nSimulated response from {host}:{port}"
        else:
            response = f"Simulated data from {host}:{port}"
        
        # Convert to bytes and limit to requested size
        response_bytes = response.encode('utf-8')[:size]
        
        logger.debug(f"Simulated receiving {len(response_bytes)} bytes on connection {conn_id}")
        return response_bytes
    
    def close(self, conn_id: int) -> None:
        """
        Close a simulated connection.
        
        Args:
            conn_id: Connection ID
            
        Raises:
            ValueError: If the connection ID is invalid
        """
        if conn_id not in self.connections:
            raise ValueError(f"Invalid connection ID: {conn_id}")
        
        self.connections[conn_id]['status'] = 'closed'
        logger.debug(f"Closed connection {conn_id}")
    
    def http_request(self, method: str, url: str, headers: Dict[str, str] = None, data: Any = None) -> Dict[str, Any]:
        """
        Simulate an HTTP request.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            headers: Request headers
            data: Request data
            
        Returns:
            Dictionary with response details (status, headers, content)
        """
        method = method.upper()
        headers = headers or {}
        
        logger.debug(f"Simulated HTTP {method} request to {url}")
        
        # Find matching predefined response
        if method in self.predefined_responses:
            for pattern, response in self.predefined_responses[method].items():
                if re.match(pattern, url):
                    logger.debug(f"Using predefined response for {url}")
                    return response
        
        # Default response
        return {
            'status': 200,
            'headers': {'Content-Type': 'text/plain'},
            'content': f"Simulated response for {method} {url}"
        }
    
    def cleanup(self) -> None:
        """Close all open connections."""
        for conn_id in list(self.connections.keys()):
            if self.connections[conn_id]['status'] == 'connected':
                try:
                    self.close(conn_id)
                except Exception as e:
                    logger.error(f"Error closing connection {conn_id} during cleanup: {str(e)}")


class SimulatedResources:
    """
    Simulates system resources within the sandbox environment.
    
    This class provides controlled access to system resources, including
    CPU, memory, and disk space, with configurable limits and simulated usage.
    """
    
    def __init__(self, max_cpu_percent: float = 75.0, max_memory_mb: float = 1024.0, max_disk_mb: float = 1024.0):
        """
        Initialize the simulated resources.
        
        Args:
            max_cpu_percent: Maximum CPU usage allowed (percentage)
            max_memory_mb: Maximum memory usage allowed (MB)
            max_disk_mb: Maximum disk space allowed (MB)
        """
        self.max_cpu_percent = max_cpu_percent
        self.max_memory_mb = max_memory_mb
        self.max_disk_mb = max_disk_mb
        
        # Current usage
        self.cpu_percent = 0.0
        self.memory_mb = 0.0
        self.disk_mb = 0.0
        
        # Usage history
        self.usage_history = {
            'timestamp': [],
            'cpu_percent': [],
            'memory_mb': [],
            'disk_mb': []
        }
        
        logger.debug("Initialized simulated resources")
    
    def allocate_memory(self, size_mb: float) -> bool:
        """
        Simulate allocating memory.
        
        Args:
            size_mb: Amount of memory to allocate (MB)
            
        Returns:
            True if allocation succeeded, False if it would exceed limits
        """
        if self.memory_mb + size_mb > self.max_memory_mb:
            logger.warning(f"Memory allocation of {size_mb}MB would exceed limit of {self.max_memory_mb}MB")
            return False
        
        self.memory_mb += size_mb
        self._update_history()
        
        logger.debug(f"Allocated {size_mb}MB of memory, total: {self.memory_mb}MB")
        return True
    
    def free_memory(self, size_mb: float) -> None:
        """
        Simulate freeing memory.
        
        Args:
            size_mb: Amount of memory to free (MB)
        """
        self.memory_mb = max(0.0, self.memory_mb - size_mb)
        self._update_history()
        
        logger.debug(f"Freed {size_mb}MB of memory, total: {self.memory_mb}MB")
    
    def allocate_disk(self, size_mb: float) -> bool:
        """
        Simulate allocating disk space.
        
        Args:
            size_mb: Amount of disk space to allocate (MB)
            
        Returns:
            True if allocation succeeded, False if it would exceed limits
        """
        if self.disk_mb + size_mb > self.max_disk_mb:
            logger.warning(f"Disk allocation of {size_mb}MB would exceed limit of {self.max_disk_mb}MB")
            return False
        
        self.disk_mb += size_mb
        self._update_history()
        
        logger.debug(f"Allocated {size_mb}MB of disk space, total: {self.disk_mb}MB")
        return True
    
    def free_disk(self, size_mb: float) -> None:
        """
        Simulate freeing disk space.
        
        Args:
            size_mb: Amount of disk space to free (MB)
        """
        self.disk_mb = max(0.0, self.disk_mb - size_mb)
        self._update_history()
        
        logger.debug(f"Freed {size_mb}MB of disk space, total: {self.disk_mb}MB")
    
    def use_cpu(self, percent: float, duration: float) -> bool:
        """
        Simulate CPU usage.
        
        Args:
            percent: CPU usage percentage
            duration: Duration of usage in seconds
            
        Returns:
            True if usage is within limits, False otherwise
        """
        if percent > self.max_cpu_percent:
            logger.warning(f"CPU usage of {percent}% would exceed limit of {self.max_cpu_percent}%")
            return False
        
        self.cpu_percent = percent
        self._update_history()
        
        # Simulate CPU usage by sleeping
        time.sleep(duration * 0.01)  # Scale down to avoid actual long waits
        
        self.cpu_percent = 0.0
        self._update_history()
        
        logger.debug(f"Used {percent}% CPU for {duration}s")
        return True
    
    def get_usage(self) -> Dict[str, float]:
        """
        Get current resource usage.
        
        Returns:
            Dictionary with current usage values
        """
        return {
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'disk_mb': self.disk_mb,
            'memory_percent': (self.memory_mb / self.max_memory_mb) * 100 if self.max_memory_mb > 0 else 0,
            'disk_percent': (self.disk_mb / self.max_disk_mb) * 100 if self.max_disk_mb > 0 else 0
        }
    
    def get_limits(self) -> Dict[str, float]:
        """
        Get resource limits.
        
        Returns:
            Dictionary with resource limits
        """
        return {
            'max_cpu_percent': self.max_cpu_percent,
            'max_memory_mb': self.max_memory_mb,
            'max_disk_mb': self.max_disk_mb
        }
    
    def get_usage_history(self) -> Dict[str, List[float]]:
        """
        Get resource usage history.
        
        Returns:
            Dictionary with usage history
        """
        return self.usage_history
    
    def _update_history(self) -> None:
        """Update the usage history with current values."""
        timestamp = time.time()
        
        self.usage_history['timestamp'].append(timestamp)
        self.usage_history['cpu_percent'].append(self.cpu_percent)
        self.usage_history['memory_mb'].append(self.memory_mb)
        self.usage_history['disk_mb'].append(self.disk_mb)
        
        # Limit history size
        max_history = 1000
        if len(self.usage_history['timestamp']) > max_history:
            for key in self.usage_history:
                self.usage_history[key] = self.usage_history[key][-max_history:]


class SimulatedIO:
    """
    Simulates input/output streams within the sandbox environment.
    
    This class provides controlled access to standard input, output, and error
    streams, capturing output for analysis and providing simulated input.
    """
    
    def __init__(self):
        """Initialize the simulated I/O streams."""
        self.stdin = StringIO()
        self.stdout = StringIO()
        self.stderr = StringIO()
        
        # Predefined inputs for interactive programs
        self.input_queue = []
        
        logger.debug("Initialized simulated I/O streams")
    
    def write_stdout(self, data: str) -> int:
        """
        Write to the simulated stdout.
        
        Args:
            data: Data to write
            
        Returns:
            Number of characters written
        """
        return self.stdout.write(data)
    
    def write_stderr(self, data: str) -> int:
        """
        Write to the simulated stderr.
        
        Args:
            data: Data to write
            
        Returns:
            Number of characters written
        """
        return self.stderr.write(data)
    
    def read_stdin(self, size: int = -1) -> str:
        """
        Read from the simulated stdin.
        
        Args:
            size: Number of characters to read (-1 for all)
            
        Returns:
            Data read from stdin
        """
        # If there's predefined input, use it
        if self.input_queue:
            input_data = self.input_queue.pop(0)
            self.stdin.write(input_data + '\n')
            self.stdin.seek(0)
            data = self.stdin.read(size)
            self.stdin = StringIO()
            return data
        
        # Otherwise, return empty string
        return ""
    
    def add_input(self, data: str) -> None:
        """
        Add predefined input to the queue.
        
        Args:
            data: Input data to add
        """
        self.input_queue.append(data)
        logger.debug(f"Added input to queue: {data}")
    
    def get_stdout(self) -> str:
        """
        Get all data written to stdout.
        
        Returns:
            Content of stdout
        """
        return self.stdout.getvalue()
    
    def get_stderr(self) -> str:
        """
        Get all data written to stderr.
        
        Returns:
            Content of stderr
        """
        return self.stderr.getvalue()
    
    def clear(self) -> None:
        """Clear all I/O streams."""
        self.stdin = StringIO()
        self.stdout = StringIO()
        self.stderr = StringIO()
        self.input_queue = []
        logger.debug("Cleared I/O streams")


class ResourceSimulator:
    """
    Main class for simulating resources within the sandbox environment.
    
    This class integrates all simulated resources (file system, network,
    system resources, I/O) into a unified interface for the sandbox.
    """
    
    def __init__(self, sandbox_dir: Path):
        """
        Initialize the resource simulator.
        
        Args:
            sandbox_dir: Base directory for the sandbox
        """
        self.sandbox_dir = sandbox_dir
        
        # Initialize components
        self.file_system = SimulatedFileSystem(sandbox_dir)
        self.network = SimulatedNetwork()
        self.resources = SimulatedResources()
        self.io = SimulatedIO()
        
        logger.info(f"Initialized resource simulator for sandbox at {sandbox_dir}")
    
    def cleanup(self) -> None:
        """Clean up all simulated resources."""
        self.file_system.cleanup()
        self.network.cleanup()
        self.io.clear()
        
        logger.info("Cleaned up resource simulator")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        return False  # Don't suppress exceptions