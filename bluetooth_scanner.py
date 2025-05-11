#!/usr/bin/env python
"""
Enhanced Bluetooth Scanner Module for the Progremon system.

This module provides specialized Bluetooth scanning functionality with proper
error handling, configurable scanning parameters, and various output formats.
It is designed to be used as a template for Bluetooth scanning tasks in
the Progremon evolution system.
"""

import os
import sys
import json
import time
import datetime
import platform
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

# Configure logger
logger = logging.getLogger('progremon.bluetooth_scanner')

class BluetoothDevice:
    """
    Class representing a discovered Bluetooth device.
    
    This captures device information including address, name, device class,
    signal strength, and other metadata.
    """
    
    def __init__(self, address: str, name: Optional[str] = None, device_class: Optional[str] = None,
                 rssi: Optional[int] = None):
        """
        Initialize a Bluetooth device.
        
        Args:
            address: Device MAC address
            name: Device name (optional)
            device_class: Device class information (optional)
            rssi: Received Signal Strength Indicator (optional)
        """
        self.address = address
        self.name = name or "Unknown"
        self.device_class = device_class or "Unknown"
        self.rssi = rssi
        self.first_seen = datetime.datetime.now()
        self.last_seen = self.first_seen
    
    def update_signal(self, rssi: int) -> None:
        """Update signal strength and last seen time."""
        self.rssi = rssi
        self.last_seen = datetime.datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert device information to a dictionary."""
        return {
            "address": self.address,
            "name": self.name,
            "device_class": self.device_class,
            "rssi": self.rssi,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat()
        }
    
    def __str__(self) -> str:
        """Return string representation of the device."""
        return f"{self.name} ({self.address}) - Signal: {self.rssi} dBm"


class BluetoothScanner:
    """
    Bluetooth device scanner with configurable parameters.
    
    This class provides methods for scanning and discovering Bluetooth devices,
    with support for different scanning durations, output formats, and error handling.
    """
    
    def __init__(self, scan_duration: float = 10.0, output_format: str = "table"):
        """
        Initialize the Bluetooth scanner.
        
        Args:
            scan_duration: Duration of scan in seconds
            output_format: Output format ("table", "json", or "text")
        """
        self.scan_duration = scan_duration
        self.output_format = output_format
        self.devices: Dict[str, BluetoothDevice] = {}
        self.scanner = None
        self.sockets = []
        self._setup_scanner()
    
    def _setup_scanner(self) -> None:
        """
        Set up the appropriate Bluetooth scanner based on the platform.
        
        Raises:
            ImportError: If required Bluetooth libraries are not available
            RuntimeError: If scanner cannot be initialized
        """
        try:
            # Platform-specific setup
            if platform.system() == "Linux":
                # Try to use BlueZ through PyBluez
                try:
                    import bluetooth
                    self.scanner = "pybluez"
                    logger.info("Using PyBluez for Bluetooth scanning")
                except ImportError:
                    logger.warning("PyBluez not available, trying Bleak")
                    import bleak
                    self.scanner = "bleak"
                    logger.info("Using Bleak for Bluetooth scanning")
            
            elif platform.system() == "Windows":
                # Try to use Bleak on Windows
                try:
                    import bleak
                    self.scanner = "bleak"
                    logger.info("Using Bleak for Bluetooth scanning")
                except ImportError:
                    logger.warning("Bleak not available, trying PyBluez")
                    import bluetooth
                    self.scanner = "pybluez"
                    logger.info("Using PyBluez for Bluetooth scanning")
            
            elif platform.system() == "Darwin":  # macOS
                # Use CoreBluetooth through Bleak on macOS
                import bleak
                self.scanner = "bleak"
                logger.info("Using Bleak for Bluetooth scanning on macOS")
            
            else:
                raise RuntimeError(f"Unsupported platform: {platform.system()}")
            
        except ImportError as e:
            logger.error(f"Required Bluetooth libraries not available: {e}")
            logger.info("Please install the appropriate Bluetooth library:")
            logger.info("  - Linux: 'pip install pybluez bleak'")
            logger.info("  - Windows: 'pip install bleak'")
            logger.info("  - macOS: 'pip install bleak'")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Bluetooth scanner: {e}")
            raise RuntimeError(f"Failed to initialize Bluetooth scanner: {e}")
    
    async def _scan_with_bleak(self) -> List[Dict[str, Any]]:
        """Scan for devices using Bleak library."""
        import bleak
        from bleak import BleakScanner
        
        logger.info(f"Scanning for Bluetooth devices for {self.scan_duration} seconds...")
        devices = []
        
        try:
            # Create a scanner
            scanner = BleakScanner()
            
            # Start scanning
            await scanner.start()
            
            # Scan for specified duration
            await asyncio.sleep(self.scan_duration)
            
            # Stop scanning
            await scanner.stop()
            
            # Get discovered devices
            discovered_devices = scanner.discovered_devices
            
            # Convert to our format
            for device in discovered_devices:
                bluetooth_device = BluetoothDevice(
                    address=device.address,
                    name=device.name or "Unknown",
                    rssi=device.rssi
                )
                self.devices[device.address] = bluetooth_device
                devices.append(bluetooth_device.to_dict())
            
            logger.info(f"Found {len(devices)} devices")
            return devices
        
        except Exception as e:
            logger.error(f"Error during Bleak scanning: {e}")
            return [{"error": str(e)}]
    
    def _scan_with_pybluez(self) -> List[Dict[str, Any]]:
        """Scan for devices using PyBluez library."""
        import bluetooth
        
        logger.info(f"Scanning for Bluetooth devices for {self.scan_duration} seconds...")
        devices = []
        
        try:
            # Discover devices
            discovered_devices = bluetooth.discover_devices(
                duration=self.scan_duration,
                lookup_names=True,
                lookup_class=True
            )
            
            # Convert to our format
            for addr, name, dev_class in discovered_devices:
                bluetooth_device = BluetoothDevice(
                    address=addr,
                    name=name or "Unknown",
                    device_class=dev_class
                )
                self.devices[addr] = bluetooth_device
                devices.append(bluetooth_device.to_dict())
            
            logger.info(f"Found {len(devices)} devices")
            return devices
        
        except Exception as e:
            logger.error(f"Error during PyBluez scanning: {e}")
            return [{"error": str(e)}]
    
    def scan_devices(self) -> List[Dict[str, Any]]:
        """
        Scan for nearby Bluetooth devices.
        
        Returns:
            List of dictionaries containing device information
        """
        try:
            if self.scanner == "bleak":
                # Bleak uses async, so we need to run it in an event loop
                import asyncio
                return asyncio.run(self._scan_with_bleak())
            elif self.scanner == "pybluez":
                return self._scan_with_pybluez()
            else:
                return [{"error": "No scanner available"}]
        except Exception as e:
            logger.error(f"Error during scanning: {e}")
            return [{"error": str(e)}]
    
    def get_device_info(self, address: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific Bluetooth device.
        
        Args:
            address: Bluetooth address of the device
            
        Returns:
            Dictionary with device information
        """
        try:
            # Check if we have this device in our cache
            if address in self.devices:
                return self.devices[address].to_dict()
            
            # Try to look up the device
            if self.scanner == "pybluez":
                import bluetooth
                try:
                    name = bluetooth.lookup_name(address, timeout=5)
                    if name:
                        device = BluetoothDevice(address=address, name=name)
                        self.devices[address] = device
                        return device.to_dict()
                except Exception as e:
                    logger.error(f"Error looking up device {address}: {e}")
            
            # Return minimal information if lookup failed
            return {
                "address": address,
                "name": "Unknown",
                "error": "Device not found or not accessible"
            }
        
        except Exception as e:
            logger.error(f"Error getting device info: {e}")
            return {"address": address, "error": str(e)}
    
    def format_output(self, devices: List[Dict[str, Any]]) -> str:
        """
        Format the device list according to the specified output format.
        
        Args:
            devices: List of device dictionaries to format
            
        Returns:
            Formatted string representation of devices
        """
        if self.output_format == "json":
            return json.dumps(devices, indent=2)
        
        elif self.output_format == "table":
            if not devices:
                return "No devices found"
            
            # Create a table header
            header = f"{'Address':<18} | {'Name':<20} | {'Signal':<8} | {'Last Seen':<25}"
            separator = "-" * 75
            
            # Create table rows
            rows = [header, separator]
            for device in devices:
                if "error" in device:
                    rows.append(f"Error: {device['error']}")
                    continue
                
                address = device["address"]
                name = device.get("name", "Unknown")[:20]
                rssi = f"{device.get('rssi', 'N/A')} dBm" if device.get('rssi') is not None else "N/A"
                last_seen = device.get("last_seen", "N/A")
                
                rows.append(f"{address:<18} | {name:<20} | {rssi:<8} | {last_seen:<25}")
            
            return "\n".join(rows)
        
        else:  # text format
            if not devices:
                return "No devices found"
            
            rows = []
            for device in devices:
                if "error" in device:
                    rows.append(f"Error: {device['error']}")
                    continue
                
                address = device["address"]
                name = device.get("name", "Unknown")
                device_class = device.get("device_class", "Unknown")
                rssi = f"{device.get('rssi', 'N/A')} dBm" if device.get('rssi') is not None else "N/A"
                
                rows.append(f"{name} ({address}) - Class: {device_class}, Signal: {rssi}")
            
            return "\n".join(rows)
    
    def close(self) -> None:
        """Release any resources used by the scanner."""
        for socket in self.sockets:
            try:
                socket.close()
            except Exception as e:
                logger.error(f"Error closing socket: {e}")
        
        self.sockets = []
        logger.info("Bluetooth scanner closed")


def scan_bluetooth_devices(scan_duration: float = 10.0, 
                          output_format: str = "table") -> List[Dict[str, Any]]:
    """
    Convenience function to scan for Bluetooth devices.
    
    This is a wrapper around the BluetoothScanner class for easy use in
    evolved code.
    
    Args:
        scan_duration: Duration of scan in seconds
        output_format: Output format ("table", "json", or "text")
        
    Returns:
        List of dictionaries containing device information
    """
    scanner = BluetoothScanner(scan_duration=scan_duration, output_format=output_format)
    try:
        return scanner.scan_devices()
    finally:
        scanner.close()


def main():
    """Main function for standalone testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bluetooth Device Scanner")
    parser.add_argument("--duration", type=float, default=10.0, help="Scan duration in seconds")
    parser.add_argument("--format", choices=["table", "json", "text"], default="table", 
                      help="Output format")
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        scanner = BluetoothScanner(scan_duration=args.duration, output_format=args.format)
        devices = scanner.scan_devices()
        print(scanner.format_output(devices))
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'scanner' in locals():
            scanner.close()


if __name__ == "__main__":
    main()