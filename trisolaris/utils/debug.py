"""
Debug Utilities for the TRISOLARIS framework.

This module provides comprehensive debug logging functionality to help
track and diagnose issues in the evolutionary process, with a focus on
security, non-leaky logging, and performance monitoring.
"""

import os
import sys
import time
import logging
import inspect
import traceback
import json
import datetime
import hashlib
import re
import base64
from typing import Dict, Any, List, Optional, Union, Callable, Set, Pattern
import threading
import functools
import io
import gzip

# Configure debug logging
DEBUG_LOGGER = logging.getLogger("trisolaris.debug")
DEBUG_LOGGER.setLevel(logging.DEBUG)

# Create a file handler for debug logs
debug_log_file = "trisolaris_debug.log"
file_handler = None  # Will be initialized in initialize_debug

# Create a console handler for important debug messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatters
detailed_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
console_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)

# Set formatters for console handler
console_handler.setFormatter(console_formatter)

# Add console handler
DEBUG_LOGGER.addHandler(console_handler)

# Encryption key for secure logging (will be initialized in initialize_debug)
_encryption_key = None

# Global debug settings
DEBUG_SETTINGS = {
    "enabled": True,
    "log_level": logging.DEBUG,
    "log_to_console": True,
    "log_to_file": True,
    "log_evolution_details": True,
    "log_genome_details": True,
    "log_fitness_details": True,
    "log_resource_usage": True,
    "log_ethical_checks": True,
    "log_performance_metrics": True,
    "capture_exceptions": True,
    "trace_function_calls": False,  # Can be verbose, disabled by default
    "secure_logging": True,         # Enable secure logging by default
    "filter_sensitive_data": True,  # Filter sensitive data by default
    "compress_logs": True,          # Compress logs to save space
    "log_security_events": True,    # Log security-related events
    "hipaa_compliant": True,        # Follow HIPAA compliance guidelines
}

# Patterns for sensitive data that should be filtered
SENSITIVE_DATA_PATTERNS = [
    # IP addresses
    re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
    # MAC addresses
    re.compile(r'\b([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})\b'),
    # Email addresses
    re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    # API keys (common formats)
    re.compile(r'\b[A-Za-z0-9]{32,}\b'),
    re.compile(r'\b[A-Za-z0-9_-]{22,}\.[A-Za-z0-9_-]{43,}\b'),
    # Credit card numbers
    re.compile(r'\b(?:\d{4}[- ]?){3}\d{4}\b'),
    # Social Security Numbers
    re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
    # Authentication tokens
    re.compile(r'(bearer|token|auth|authorization|password|secret|key)[\s:=]+[^\s]+', re.IGNORECASE),
]

# Custom sensitive data patterns (can be updated at runtime)
CUSTOM_SENSITIVE_PATTERNS: List[Pattern] = []

# Performance metrics tracking
PERFORMANCE_METRICS = {
    "function_calls": {},
    "evolution_times": [],
    "fitness_evaluation_times": [],
    "mutation_times": [],
    "crossover_times": [],
    "ethical_check_times": [],
    "security_check_times": [],
    "start_time": None,
}

# Security events tracking
SECURITY_EVENTS = []

def filter_sensitive_data(text: str) -> str:
    """
    Filter sensitive data from text.
    
    Args:
        text: Text to filter
        
    Returns:
        Filtered text with sensitive data replaced by [REDACTED]
    """
    if not DEBUG_SETTINGS["filter_sensitive_data"]:
        return text
    
    # Apply all patterns
    for pattern in SENSITIVE_DATA_PATTERNS + CUSTOM_SENSITIVE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    
    return text

def add_sensitive_data_pattern(pattern: str) -> None:
    """
    Add a custom pattern for sensitive data filtering.
    
    Args:
        pattern: Regular expression pattern as string
    """
    global CUSTOM_SENSITIVE_PATTERNS
    try:
        compiled_pattern = re.compile(pattern)
        CUSTOM_SENSITIVE_PATTERNS.append(compiled_pattern)
        DEBUG_LOGGER.debug(f"Added custom sensitive data pattern")
    except re.error as e:
        DEBUG_LOGGER.error(f"Invalid regex pattern: {e}")

def generate_encryption_key(password: str = None, salt: bytes = None) -> bytes:
    """
    Generate an encryption key for secure logging.
    
    Args:
        password: Optional password for key derivation
        salt: Optional salt for key derivation
        
    Returns:
        Encryption key as bytes
    """
    if not password:
        # Generate a random password if none provided
        password = base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
    
    if not salt:
        # Generate a random salt if none provided
        salt = os.urandom(16)
    
    # Use a key derivation function to derive a key
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode(),
        salt,
        100000,
        dklen=32
    )
    
    # Encode the key in base64
    return base64.urlsafe_b64encode(key)

class SecureFormatter(logging.Formatter):
    """
    A formatter that applies security measures to log records.
    
    This formatter can:
    1. Filter sensitive data from log messages
    2. Encrypt log messages if required
    """
    
    def __init__(self, base_formatter, filter_sensitive=True, encrypt=False):
        """
        Initialize the secure formatter.
        
        Args:
            base_formatter: The base formatter to use
            filter_sensitive: Whether to filter sensitive data
            encrypt: Whether to encrypt log messages
        """
        self.base_formatter = base_formatter
        self.filter_sensitive = filter_sensitive
        self.encrypt = encrypt
    
    def format(self, record):
        """Format the log record with security measures."""
        # First, use the base formatter
        formatted_msg = self.base_formatter.format(record)
        
        # Filter sensitive data if enabled
        if self.filter_sensitive:
            formatted_msg = filter_sensitive_data(formatted_msg)
        
        # Encrypt if enabled
        if self.encrypt and _encryption_key:
            # Simple encryption for demonstration
            # In a real implementation, use a proper encryption library
            formatted_msg = base64.b64encode(formatted_msg.encode()).decode()
        
        return formatted_msg

class CompressedRotatingFileHandler(logging.Handler):
    """
    A handler that writes compressed log files and rotates them when they reach a certain size.
    """
    
    def __init__(self, filename, maxBytes=0, backupCount=0, encoding=None):
        """
        Initialize the handler.
        
        Args:
            filename: Log file name
            maxBytes: Maximum file size before rotation
            backupCount: Number of backup files to keep
            encoding: File encoding
        """
        logging.Handler.__init__(self)
        self.filename = filename
        self.maxBytes = maxBytes
        self.backupCount = backupCount
        self.encoding = encoding
        self.stream = None
        self.current_size = 0
        
        # Check if file exists and get its size
        if os.path.exists(filename):
            self.current_size = os.path.getsize(filename)
    
    def emit(self, record):
        """Emit a record."""
        if self.stream is None:
            self.stream = gzip.open(self.filename, 'at', encoding=self.encoding)
        
        try:
            msg = self.format(record)
            msg_size = len(msg.encode(self.encoding or 'utf-8'))
            
            # Check if rotation is needed
            if self.maxBytes > 0 and self.current_size + msg_size > self.maxBytes:
                self.doRollover()
            
            self.stream.write(msg + '\n')
            self.stream.flush()
            self.current_size += msg_size
            
        except Exception:
            self.handleError(record)
    
    def doRollover(self):
        """Do a rollover."""
        if self.stream:
            self.stream.close()
            self.stream = None
        
        # Rotate the log files
        if self.backupCount > 0:
            for i in range(self.backupCount - 1, 0, -1):
                sfn = f"{self.filename}.{i}.gz"
                dfn = f"{self.filename}.{i+1}.gz"
                if os.path.exists(sfn):
                    if os.path.exists(dfn):
                        os.remove(dfn)
                    os.rename(sfn, dfn)
            
            dfn = f"{self.filename}.1.gz"
            if os.path.exists(dfn):
                os.remove(dfn)
            os.rename(self.filename, dfn)
        
        # Open a new file
        self.stream = gzip.open(self.filename, 'at', encoding=self.encoding)
        self.current_size = 0
    
    def close(self):
        """Close the handler."""
        if self.stream:
            self.stream.close()
            self.stream = None
        logging.Handler.close(self)

def log_security_event(event_type: str, details: Dict[str, Any] = None) -> None:
    """
    Log a security-related event.
    
    Args:
        event_type: Type of security event
        details: Optional details about the event
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_security_events"]:
        return
    
    timestamp = datetime.datetime.now().isoformat()
    
    # Create event record
    event = {
        "timestamp": timestamp,
        "event_type": event_type,
        "details": details or {}
    }
    
    # Add to security events list
    SECURITY_EVENTS.append(event)
    
    # Log the event
    DEBUG_LOGGER.warning(f"Security event: {event_type}", extra={"security_event": True})
    
    # If HIPAA compliance is enabled, add additional context
    if DEBUG_SETTINGS["hipaa_compliant"]:
        # Generate a unique event ID for audit trail
        event_id = hashlib.sha256(f"{timestamp}:{event_type}".encode()).hexdigest()[:12]
        DEBUG_LOGGER.warning(f"HIPAA audit: Event ID {event_id} - {event_type}")

def initialize_debug(
    enabled: bool = True,
    log_level: int = logging.DEBUG,
    log_file: str = "trisolaris_debug.log",
    log_to_console: bool = True,
    log_to_file: bool = True,
    secure_logging: bool = True,
    filter_sensitive_data: bool = True,
    compress_logs: bool = True,
    log_security_events: bool = True,
    hipaa_compliant: bool = True,
    encryption_password: str = None,
    **kwargs
) -> None:
    """
    Initialize the debug system with custom settings and security features.
    
    Args:
        enabled: Whether debugging is enabled
        log_level: Logging level (DEBUG, INFO, etc.)
        log_file: Path to the debug log file
        log_to_console: Whether to log to console
        log_to_file: Whether to log to file
        secure_logging: Whether to use secure logging features
        filter_sensitive_data: Whether to filter sensitive data
        compress_logs: Whether to compress log files
        log_security_events: Whether to log security events
        hipaa_compliant: Whether to follow HIPAA compliance guidelines
        encryption_password: Optional password for log encryption
        **kwargs: Additional debug settings
    """
    global DEBUG_SETTINGS, debug_log_file, file_handler, _encryption_key
    
    # Update settings
    DEBUG_SETTINGS["enabled"] = enabled
    DEBUG_SETTINGS["log_level"] = log_level
    DEBUG_SETTINGS["log_to_console"] = log_to_console
    DEBUG_SETTINGS["log_to_file"] = log_to_file
    DEBUG_SETTINGS["secure_logging"] = secure_logging
    DEBUG_SETTINGS["filter_sensitive_data"] = filter_sensitive_data
    DEBUG_SETTINGS["compress_logs"] = compress_logs
    DEBUG_SETTINGS["log_security_events"] = log_security_events
    DEBUG_SETTINGS["hipaa_compliant"] = hipaa_compliant
    
    # Update other settings if provided
    for key, value in kwargs.items():
        if key in DEBUG_SETTINGS:
            DEBUG_SETTINGS[key] = value
    
    # Update logger level
    DEBUG_LOGGER.setLevel(log_level)
    
    # Remove existing file handler if any
    if file_handler:
        DEBUG_LOGGER.removeHandler(file_handler)
    
    # Create directory for log file if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Update debug log file path
    debug_log_file = log_file
    
    # Generate encryption key if encryption is enabled
    if encryption_password:
        _encryption_key = generate_encryption_key(encryption_password)
    
    # Create appropriate file handler based on settings
    if log_to_file:
        if compress_logs:
            # Use a custom handler for compressed logs
            file_handler = CompressedRotatingFileHandler(
                debug_log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
        else:
            # Use standard file handler
            file_handler = logging.FileHandler(debug_log_file)
        
        file_handler.setLevel(log_level)
        
        # Use secure formatter if secure logging is enabled
        if secure_logging:
            file_handler.setFormatter(SecureFormatter(detailed_formatter, 
                                                     filter_sensitive=filter_sensitive_data,
                                                     encrypt=encryption_password is not None))
        else:
            file_handler.setFormatter(detailed_formatter)
        
        DEBUG_LOGGER.addHandler(file_handler)
    
    # Enable/disable console handler
    if log_to_console:
        console_handler.setLevel(log_level)
        
        # Use secure formatter for console if secure logging is enabled
        if secure_logging and filter_sensitive_data:
            console_handler.setFormatter(SecureFormatter(console_formatter, 
                                                       filter_sensitive=True,
                                                       encrypt=False))
        else:
            console_handler.setFormatter(console_formatter)
    else:
        console_handler.setLevel(logging.CRITICAL)  # Effectively disable
    
    # Reset performance metrics
    reset_performance_metrics()
    
    # Log initialization with filtered settings (don't log encryption key)
    safe_settings = DEBUG_SETTINGS.copy()
    if "encryption_key" in safe_settings:
        safe_settings["encryption_key"] = "[REDACTED]"
    
    DEBUG_LOGGER.info(f"Debug system initialized with settings: {safe_settings}")
    
    # Log security event
    if log_security_events:
        log_security_event("debug_system_initialized", {
            "secure_logging": secure_logging,
            "encrypt_logs": encryption_password is not None,
            "filter_sensitive_data": filter_sensitive_data,
            "hipaa_compliant": hipaa_compliant
        })

def reset_performance_metrics() -> None:
    """Reset all performance metrics."""
    global PERFORMANCE_METRICS, SECURITY_EVENTS
    PERFORMANCE_METRICS = {
        "function_calls": {},
        "evolution_times": [],
        "fitness_evaluation_times": [],
        "mutation_times": [],
        "crossover_times": [],
        "ethical_check_times": [],
        "security_check_times": [],
        "start_time": datetime.datetime.now(),
    }
    SECURITY_EVENTS = []
    DEBUG_LOGGER.debug("Performance metrics and security events reset")

def debug_log(message: str, level: int = logging.DEBUG, **kwargs) -> None:
    """
    Log a debug message with additional context and security features.
    
    Args:
        message: The message to log
        level: Logging level
        **kwargs: Additional context to include in the log
    """
    if not DEBUG_SETTINGS["enabled"]:
        return
    
    # Get caller information
    frame = inspect.currentframe().f_back
    filename = frame.f_code.co_filename
    lineno = frame.f_lineno
    function = frame.f_code.co_name
    
    # Format additional context
    context = ""
    if kwargs:
        # Filter sensitive data from kwargs if enabled
        if DEBUG_SETTINGS["filter_sensitive_data"]:
            filtered_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, str):
                    filtered_kwargs[k] = filter_sensitive_data(v)
                else:
                    filtered_kwargs[k] = v
            context = " | " + " | ".join(f"{k}={v}" for k, v in filtered_kwargs.items())
        else:
            context = " | " + " | ".join(f"{k}={v}" for k, v in kwargs.items())
    
    # Log the message with context
    DEBUG_LOGGER.log(
        level,
        f"[{os.path.basename(filename)}:{lineno} in {function}] {message}{context}"
    )

def debug_exception(e: Exception, context: str = "", security_related: bool = False) -> None:
    """
    Log an exception with detailed traceback and security context.
    
    Args:
        e: The exception to log
        context: Additional context about where the exception occurred
        security_related: Whether this exception is security-related
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["capture_exceptions"]:
        return
    
    # Get the full traceback
    tb = traceback.format_exc()
    
    # Filter sensitive data if enabled
    if DEBUG_SETTINGS["filter_sensitive_data"]:
        tb = filter_sensitive_data(tb)
        context = filter_sensitive_data(context)
        error_msg = filter_sensitive_data(str(e))
    else:
        error_msg = str(e)
    
    # Log the exception
    if security_related and DEBUG_SETTINGS["log_security_events"]:
        # Log as a security event
        DEBUG_LOGGER.error(f"SECURITY EXCEPTION in {context}: {error_msg}\n{tb}")
        log_security_event("security_exception", {
            "context": context,
            "error_type": type(e).__name__,
            "traceback_hash": hashlib.sha256(tb.encode()).hexdigest()
        })
    else:
        # Log as a regular exception
        DEBUG_LOGGER.error(f"Exception in {context}: {error_msg}\n{tb}")

def debug_decorator(func: Callable, security_check: bool = False) -> Callable:
    """
    Decorator to add debug logging to a function with security awareness.
    
    Args:
        func: The function to decorate
        security_check: Whether this function performs security checks
        
    Returns:
        Decorated function with debug logging
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not DEBUG_SETTINGS["enabled"]:
            return func(*args, **kwargs)
        
        # Get function details
        func_name = func.__qualname__
        
        # Filter sensitive data from arguments if enabled
        safe_args = args
        safe_kwargs = kwargs
        
        if DEBUG_SETTINGS["filter_sensitive_data"]:
            # For kwargs, we can filter string values
            safe_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, str):
                    safe_kwargs[k] = filter_sensitive_data(v)
                else:
                    safe_kwargs[k] = v
            
            # For args, we can only filter string values
            safe_args = []
            for arg in args:
                if isinstance(arg, str):
                    safe_args.append(filter_sensitive_data(arg))
                else:
                    safe_args.append(arg)
        
        # Log function entry
        debug_log(f"Entering {func_name}", level=logging.DEBUG)
        
        # Track performance if enabled
        if DEBUG_SETTINGS["log_performance_metrics"]:
            start_time = time.time()
            
            # Update call count
            if func_name not in PERFORMANCE_METRICS["function_calls"]:
                PERFORMANCE_METRICS["function_calls"][func_name] = {
                    "count": 0,
                    "total_time": 0,
                    "min_time": float('inf'),
                    "max_time": 0,
                }
            
            PERFORMANCE_METRICS["function_calls"][func_name]["count"] += 1
        
        # Call the function
        try:
            result = func(*args, **kwargs)
            
            # Log function exit
            debug_log(f"Exiting {func_name}", level=logging.DEBUG)
            
            # Update performance metrics
            if DEBUG_SETTINGS["log_performance_metrics"]:
                elapsed = time.time() - start_time
                metrics = PERFORMANCE_METRICS["function_calls"][func_name]
                metrics["total_time"] += elapsed
                metrics["min_time"] = min(metrics["min_time"], elapsed)
                metrics["max_time"] = max(metrics["max_time"], elapsed)
                
                # Track specific operation types
                if "evaluate" in func_name.lower():
                    PERFORMANCE_METRICS["fitness_evaluation_times"].append(elapsed)
                elif "mutate" in func_name.lower():
                    PERFORMANCE_METRICS["mutation_times"].append(elapsed)
                elif "crossover" in func_name.lower():
                    PERFORMANCE_METRICS["crossover_times"].append(elapsed)
                elif "ethical" in func_name.lower() or "boundary" in func_name.lower():
                    PERFORMANCE_METRICS["ethical_check_times"].append(elapsed)
                elif "security" in func_name.lower() or "secure" in func_name.lower():
                    PERFORMANCE_METRICS["security_check_times"].append(elapsed)
                elif "evolve" in func_name.lower():
                    PERFORMANCE_METRICS["evolution_times"].append(elapsed)
            
            return result
            
        except Exception as e:
            # Log the exception
            debug_exception(e, context=func_name, security_related=security_check)
            
            # Re-raise the exception
            raise
    
    return wrapper

def security_check_decorator(func: Callable) -> Callable:
    """
    Decorator specifically for security-related functions.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with security-aware debug logging
    """
    return debug_decorator(func, security_check=True)

def log_genome_details(genome, generation: int = None, fitness: float = None) -> None:
    """
    Log detailed information about a genome with security awareness.
    
    Args:
        genome: The genome to log
        generation: Optional generation number
        fitness: Optional fitness value
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_genome_details"]:
        return
    
    try:
        # Get genome source code
        if hasattr(genome, 'to_source'):
            source = genome.to_source()
        else:
            source = str(genome)
        
        # Truncate source if too long
        if len(source) > 1000:
            source = source[:500] + "\n...\n" + source[-500:]
        
        # Filter sensitive data if enabled
        if DEBUG_SETTINGS["filter_sensitive_data"]:
            source = filter_sensitive_data(source)
        
        # Log basic info
        gen_info = f" (Generation {generation})" if generation is not None else ""
        fit_info = f" (Fitness: {fitness:.4f})" if fitness is not None else ""
        
        DEBUG_LOGGER.debug(f"Genome Details{gen_info}{fit_info}:\n{source}")
        
    except Exception as e:
        debug_exception(e, context="log_genome_details")

def log_fitness_evaluation(genome, fitness: float, details: Dict[str, Any] = None) -> None:
    """
    Log details about a fitness evaluation with security awareness.
    
    Args:
        genome: The genome that was evaluated
        fitness: The fitness value
        details: Optional detailed evaluation results
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_fitness_details"]:
        return
    
    try:
        # Log basic fitness info
        DEBUG_LOGGER.debug(f"Fitness Evaluation: {fitness:.4f}")
        
        # Log detailed results if available
        if details:
            # Filter sensitive data if enabled
            if DEBUG_SETTINGS["filter_sensitive_data"]:
                filtered_details = {}
                for k, v in details.items():
                    if isinstance(v, str):
                        filtered_details[k] = filter_sensitive_data(v)
                    else:
                        filtered_details[k] = v
                details = filtered_details
            
            # Format details for logging
            details_str = json.dumps(details, indent=2)
            DEBUG_LOGGER.debug(f"Fitness Details:\n{details_str}")
        
    except Exception as e:
        debug_exception(e, context="log_fitness_evaluation")

def log_ethical_check(genome, passed: bool, boundaries: Dict[str, bool] = None) -> None:
    """
    Log details about an ethical boundary check with security awareness.
    
    Args:
        genome: The genome that was checked
        passed: Whether the genome passed all ethical boundaries
        boundaries: Optional dictionary of boundary results
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_ethical_checks"]:
        return
    
    try:
        # Log basic check result
        result = "PASSED" if passed else "FAILED"
        DEBUG_LOGGER.debug(f"Ethical Check: {result}")
        
        # Log detailed boundary results if available
        if boundaries:
            # Format boundaries for logging
            boundaries_str = "\n".join(f"  - {name}: {'PASSED' if passed else 'FAILED'}" 
                                     for name, passed in boundaries.items())
            DEBUG_LOGGER.debug(f"Boundary Results:\n{boundaries_str}")
        
        # Log security event if ethical check failed
        if not passed and DEBUG_SETTINGS["log_security_events"]:
            log_security_event("ethical_boundary_violation", {
                "boundaries_failed": [name for name, passed in (boundaries or {}).items() if not passed],
                "genome_hash": hashlib.sha256(str(genome).encode()).hexdigest()[:12]
            })
        
    except Exception as e:
        debug_exception(e, context="log_ethical_check")

def log_resource_usage(resource_monitor) -> None:
    """
    Log current resource usage from a ResourceSteward with security awareness.
    
    Args:
        resource_monitor: The ResourceSteward instance
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_resource_usage"]:
        return
    
    try:
        # Check if resource monitor is available
        if resource_monitor is None:
            return
        
        # Get current resource status
        status = resource_monitor.check_resources()
        
        # Log resource usage
        DEBUG_LOGGER.debug(
            f"Resource Usage: "
            f"Memory: {(1.0 - status['memory_available']) * 100:.1f}% used, "
            f"CPU: {(1.0 - status['cpu_available']) * 100:.1f}% used, "
            f"Disk: {(1.0 - status['disk_available']) * 100:.1f}% used, "
            f"Throttle Level: {resource_monitor.get_throttle_level()}"
        )
        
        # Log security event if resources are critically low
        if (status['memory_available'] < 0.1 or 
            status['cpu_available'] < 0.1 or 
            status['disk_available'] < 0.1) and DEBUG_SETTINGS["log_security_events"]:
            
            log_security_event("critical_resource_shortage", {
                "memory_available": status['memory_available'],
                "cpu_available": status['cpu_available'],
                "disk_available": status['disk_available'],
                "throttle_level": resource_monitor.get_throttle_level()
            })
        
    except Exception as e:
        debug_exception(e, context="log_resource_usage")

def log_evolution_progress(generation: int, best_fitness: float, avg_fitness: float, 
                          elapsed_time: float, population_size: int = None) -> None:
    """
    Log progress of the evolutionary process with security awareness.
    
    Args:
        generation: Current generation number
        best_fitness: Best fitness in the current generation
        avg_fitness: Average fitness in the current generation
        elapsed_time: Time taken for this generation
        population_size: Optional population size
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_evolution_details"]:
        return
    
    try:
        # Format population size info
        pop_info = f", Population: {population_size}" if population_size is not None else ""
        
        # Log progress
        DEBUG_LOGGER.info(
            f"Generation {generation}: "
            f"Best Fitness: {best_fitness:.4f}, "
            f"Avg Fitness: {avg_fitness:.4f}, "
            f"Time: {elapsed_time:.2f}s{pop_info}"
        )
        
    except Exception as e:
        debug_exception(e, context="log_evolution_progress")

def generate_security_report() -> str:
    """
    Generate a detailed security report.
    
    Returns:
        A formatted security report string
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_security_events"]:
        return "Security reporting disabled"
    
    try:
        # Build the report
        report = []
        report.append("=" * 50)
        report.append("TRISOLARIS SECURITY REPORT")
        report.append("=" * 50)
        
        # Security events
        report.append(f"\nSecurity Events: {len(SECURITY_EVENTS)}")
        report.append("-" * 30)
        
        # Group events by type
        event_types = {}
        for event in SECURITY_EVENTS:
            event_type = event["event_type"]
            if event_type not in event_types:
                event_types[event_type] = []
            event_types[event_type].append(event)
        
        # Report on each event type
        for event_type, events in event_types.items():
            report.append(f"\n{event_type}: {len(events)} occurrences")
            # Show the most recent event of this type
            if events:
                latest = events[-1]
                timestamp = latest.get("timestamp", "Unknown")
                details = latest.get("details", {})
                report.append(f"  Latest: {timestamp}")
                for k, v in details.items():
                    report.append(f"    {k}: {v}")
        
        # Security check performance
        security_times = PERFORMANCE_METRICS["security_check_times"]
        if security_times:
            avg_time = sum(security_times) / len(security_times)
            report.append("\nSecurity Check Performance:")
            report.append(f"  Total checks: {len(security_times)}")
            report.append(f"  Average time: {avg_time:.4f}s")
            report.append(f"  Min time: {min(security_times):.4f}s")
            report.append(f"  Max time: {max(security_times):.4f}s")
        
        report.append("\n" + "=" * 50)
        return "\n".join(report)
        
    except Exception as e:
        debug_exception(e, context="generate_security_report")
        return f"Error generating security report: {str(e)}"

def save_security_report(output_path: str = "trisolaris_security_report.txt") -> None:
    """
    Generate and save a security report to a file.
    
    Args:
        output_path: Path to save the report
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_security_events"]:
        return
    
    try:
        # Generate the report
        report = generate_security_report()
        
        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        DEBUG_LOGGER.info(f"Security report saved to {output_path}")
        
    except Exception as e:
        debug_exception(e, context="save_security_report", security_related=True)

def generate_performance_and_security_report() -> str:
    """
    Generate a combined performance and security report.
    
    Returns:
        A formatted combined report string
    """
    if not DEBUG_SETTINGS["enabled"]:
        return "Reporting disabled"
    
    try:
        # Get individual reports
        performance_report = generate_performance_report()
        security_report = generate_security_report()
        
        # Combine reports
        combined_report = []
        combined_report.append("=" * 50)
        combined_report.append("TRISOLARIS COMBINED REPORT")
        combined_report.append("=" * 50)
        combined_report.append("\n" + performance_report)
        combined_report.append("\n" + security_report)
        
        return "\n".join(combined_report)
        
    except Exception as e:
        debug_exception(e, context="generate_combined_report")
        return f"Error generating combined report: {str(e)}"

def save_combined_report(output_path: str = "trisolaris_combined_report.txt") -> None:
    """
    Generate and save a combined performance and security report to a file.
    
    Args:
        output_path: Path to save the report
    """
    if not DEBUG_SETTINGS["enabled"]:
        return
    
    try:
        # Generate the report
        report = generate_performance_and_security_report()
        
        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        DEBUG_LOGGER.info(f"Combined report saved to {output_path}")
        
    except Exception as e:
        debug_exception(e, context="save_combined_report")

def generate_performance_report() -> str:
    """
    Generate a performance report based on collected metrics.
    
    Returns:
        A formatted performance report string
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_performance_metrics"]:
        return "Performance reporting disabled"
    
    try:
        # Build the report
        report = []
        report.append("=" * 50)
        report.append("TRISOLARIS PERFORMANCE REPORT")
        report.append("=" * 50)
        
        # Add runtime information
        start_time = PERFORMANCE_METRICS["start_time"]
        if start_time:
            elapsed = (datetime.datetime.now() - start_time).total_seconds()
            report.append(f"\nTotal Runtime: {elapsed:.2f} seconds")
        
        # Function call statistics
        report.append("\nFunction Call Statistics:")
        report.append("-" * 30)
        
        # Sort functions by total time
        sorted_funcs = sorted(
            PERFORMANCE_METRICS["function_calls"].items(),
            key=lambda x: x[1]["total_time"],
            reverse=True
        )
        
        for func_name, metrics in sorted_funcs[:20]:  # Show top 20 functions
            count = metrics["count"]
            total_time = metrics["total_time"]
            avg_time = total_time / count if count > 0 else 0
            min_time = metrics["min_time"] if metrics["min_time"] != float('inf') else 0
            max_time = metrics["max_time"]
            
            report.append(f"\n{func_name}:")
            report.append(f"  Calls: {count}")
            report.append(f"  Total Time: {total_time:.4f}s")
            report.append(f"  Avg Time: {avg_time:.4f}s")
            report.append(f"  Min Time: {min_time:.4f}s")
            report.append(f"  Max Time: {max_time:.4f}s")
        
        # Operation statistics
        operations = [
            ("Evolution", "evolution_times"),
            ("Fitness Evaluation", "fitness_evaluation_times"),
            ("Mutation", "mutation_times"),
            ("Crossover", "crossover_times"),
            ("Ethical Check", "ethical_check_times")
        ]
        
        report.append("\nOperation Statistics:")
        report.append("-" * 30)
        
        for op_name, metric_key in operations:
            times = PERFORMANCE_METRICS[metric_key]
            if times:
                avg_time = sum(times) / len(times)
                report.append(f"\n{op_name}:")
                report.append(f"  Count: {len(times)}")
                report.append(f"  Total Time: {sum(times):.4f}s")
                report.append(f"  Avg Time: {avg_time:.4f}s")
                report.append(f"  Min Time: {min(times):.4f}s")
                report.append(f"  Max Time: {max(times):.4f}s")
        
        report.append("\n" + "=" * 50)
        return "\n".join(report)
        
    except Exception as e:
        debug_exception(e, context="generate_performance_report")
        return f"Error generating performance report: {str(e)}"

def save_performance_report(output_path: str = "trisolaris_performance_report.txt") -> None:
    """
    Generate and save a performance report to a file.
    
    Args:
        output_path: Path to save the report
    """
    if not DEBUG_SETTINGS["enabled"] or not DEBUG_SETTINGS["log_performance_metrics"]:
        return
    
    try:
        # Generate the report
        report = generate_performance_report()
        
        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        DEBUG_LOGGER.info(f"Performance report saved to {output_path}")
        
    except Exception as e:
        debug_exception(e, context="save_performance_report")

class ThreadSafeLogger:
    """
    A thread-safe wrapper for logging functions.
    
    This class provides thread-safe versions of logging functions
    to prevent race conditions when logging from multiple threads.
    """
    
    def __init__(self):
        """Initialize the thread-safe logger."""
        self._lock = threading.RLock()
    
    def debug_log(self, message: str, level: int = logging.DEBUG, **kwargs) -> None:
        """Thread-safe version of debug_log."""
        with self._lock:
            debug_log(message, level, **kwargs)
    
    def debug_exception(self, e: Exception, context: str = "", security_related: bool = False) -> None:
        """Thread-safe version of debug_exception."""
        with self._lock:
            debug_exception(e, context, security_related)
    
    def log_genome_details(self, genome, generation: int = None, fitness: float = None) -> None:
        """Thread-safe version of log_genome_details."""
        with self._lock:
            log_genome_details(genome, generation, fitness)
    
    def log_fitness_evaluation(self, genome, fitness: float, details: Dict[str, Any] = None) -> None:
        """Thread-safe version of log_fitness_evaluation."""
        with self._lock:
            log_fitness_evaluation(genome, fitness, details)
    
    def log_ethical_check(self, genome, passed: bool, boundaries: Dict[str, bool] = None) -> None:
        """Thread-safe version of log_ethical_check."""
        with self._lock:
            log_ethical_check(genome, passed, boundaries)
    
    def log_resource_usage(self, resource_monitor) -> None:
        """Thread-safe version of log_resource_usage."""
        with self._lock:
            log_resource_usage(resource_monitor)
    
    def log_evolution_progress(self, generation: int, best_fitness: float, avg_fitness: float,
                              elapsed_time: float, population_size: int = None) -> None:
        """Thread-safe version of log_evolution_progress."""
        with self._lock:
            log_evolution_progress(generation, best_fitness, avg_fitness, elapsed_time, population_size)

# Create a global thread-safe logger instance
thread_safe_logger = ThreadSafeLogger()
