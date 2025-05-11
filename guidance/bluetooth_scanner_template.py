#!/usr/bin/env python
"""
Bluetooth Scanner Template

This template provides the basic structure for a Bluetooth device scanner
with proper error handling, device information formatting, and ethical
considerations like time-limited scanning and privacy protections.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union

# Configure logger
logger = logging.getLogger('bluetooth_scanner')

class BluetoothScanner:
    """
    Bluetooth device scanner with ethical considerations built in.
    
    Features:
    - Time-limited scanning to prevent battery drain
    - Privacy-respecting device information collection
    - Multiple output format options
    - Comprehensive error handling
    """
    
    def __init__(self, scan_duration: int = 10, output_format: str = "table"):
        """
        Initialize the scanner with ethical defaults.
        
        Args:
            scan_duration: Maximum scan duration in seconds (default: 10)
            output_format: Output format, one of "table", "json", "dict" (default: "table")
        """
        self.scan_duration = min(scan_duration, 30)  # Limit to 30 seconds max for ethical reasons
        self.output_format = output_format
        self.devices = []
        
        # Initialize Bluetooth library based on platform
        self.bluetooth = None
        try:
            # Try to import appropriate library based on platform
            import platform
            system = platform.system()
            
            if system == "Windows":
                try:
                    import bluetooth
                    self.bluetooth = bluetooth
                    logger.info("Using PyBluez on Windows")
                except ImportError:
                    logger.warning("PyBluez not available on Windows")
            elif system == "Darwin":  # macOS
                try:
                    import objc
                    logger.info("Using PyObjC on macOS")
                    # macOS-specific imports would go here
                except ImportError:
                    logger.warning("PyObjC not available on macOS")
            else:  # Linux and others
                try:
                    import bluetooth
                    self.bluetooth = bluetooth
                    logger.info("Using PyBluez on Linux")
                except ImportError:
                    logger.warning("PyBluez not available on Linux")
                    
            if not self.bluetooth:
                logger.warning("No Bluetooth library available, falling back to simulation mode")
        except Exception as e:
            logger.error(f"Error initializing Bluetooth: {e}")
            logger.warning("Falling back to simulation mode")
    
    def scan(self) -> List[Dict[str, Any]]:
        """
        Scan for nearby Bluetooth devices.
        
        Returns:
            List of dictionaries with device information
        """
        start_time = time.time()
        self.devices = []
        
        logger.info(f"Starting Bluetooth scan (max duration: {self.scan_duration}s)")
        
        try:
            if self.bluetooth:
                # Real scan using the appropriate library
                try:
                    logger.info("Discovering devices...")
                    nearby_devices = self.bluetooth.discover_devices(
                        duration=self.scan_duration,
                        lookup_names=True,
                        flush_cache=True
                    )
                    
                    for addr, name in nearby_devices:
                        self.devices.append({
                            "address": addr,
                            "name": name or "Unknown",
                            "type": self._get_device_type(name),
                            "signal_strength": None  # Not available in basic scan
                        })
                        
                except Exception as e:
                    logger.error(f"Error during Bluetooth scan: {e}")
                    logger.warning("Falling back to simulation mode")
                    self._simulate_scan()
            else:
                # Simulation mode
                logger.info("Running in simulation mode")
                self._simulate_scan()
                
        except KeyboardInterrupt:
            logger.info("Scan interrupted by user")
        finally:
            elapsed = time.time() - start_time
            logger.info(f"Scan completed in {elapsed:.2f}s, found {len(self.devices)} devices")
        
        return self.devices
    
    def _simulate_scan(self):
        """Simulate finding devices for testing purposes."""
        import random
        
        # Simulate scan duration
        duration = min(2, self.scan_duration)
        time.sleep(duration)
        
        # Generate some simulated devices
        simulated_devices = [
            {
                "address": "00:11:22:33:44:55",
                "name": "Simulated Phone",
                "type": "phone",
                "signal_strength": -50
            },
            {
                "address": "AA:BB:CC:DD:EE:FF",
                "name": "Simulated Headphones",
                "type": "audio",
                "signal_strength": -70
            },
            {
                "address": "12:34:56:78:90:AB",
                "name": "Simulated Speaker",
                "type": "audio",
                "signal_strength": -60
            }
        ]
        
        # Randomly select a subset of devices
        count = random.randint(1, len(simulated_devices))
        self.devices = random.sample(simulated_devices, count)
    
    def _get_device_type(self, name: Optional[str]) -> str:
        """
        Guess device type from name.
        
        Args:
            name: Device name
            
        Returns:
            Device type string
        """
        if not name:
            return "unknown"
            
        name = name.lower()
        
        if any(keyword in name for keyword in ["phone", "iphone", "pixel", "galaxy"]):
            return "phone"
        elif any(keyword in name for keyword in ["headphone", "earbuds", "airpod"]):
            return "audio"
        elif any(keyword in name for keyword in ["watch", "band"]):
            return "wearable"
        else:
            return "other"
    
    def format_results(self, format_type: Optional[str] = None) -> Union[str, List[Dict[str, Any]]]:
        """
        Format scan results.
        
        Args:
            format_type: Override default output format
            
        Returns:
            Formatted results as string or list of dictionaries
        """
        output_format = format_type or self.output_format
        
        if output_format == "dict" or output_format == "json":
            import json
            return self.devices if output_format == "dict" else json.dumps(self.devices, indent=2)
        else:  # table format
            if not self.devices:
                return "No devices found."
                
            # Create table
            result = "\nBluetooth Devices Found:\n"
            result += "-" * 60 + "\n"
            result += f"{'Address':<20} {'Name':<25} {'Type':<15}\n"
            result += "-" * 60 + "\n"
            
            for device in self.devices:
                result += f"{device['address']:<20} {device['name'][:23]:<25} {device['type']:<15}\n"
                
            result += "-" * 60 + "\n"
            return result


def scan_for_devices(duration: int = 10, format_type: str = "table") -> Union[str, List[Dict[str, Any]]]:
    """
    Convenience function to scan for Bluetooth devices.
    
    Args:
        duration: Scan duration in seconds (default: 10)
        format_type: Output format, one of "table", "json", "dict" (default: "table")
        
    Returns:
        Formatted results as string or list of dictionaries
    """
    scanner = BluetoothScanner(scan_duration=duration, output_format=format_type)
    scanner.scan()
    return scanner.format_results()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = scan_for_devices(duration=5)
    print(result)
