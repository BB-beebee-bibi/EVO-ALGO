#!/usr/bin/env python3
'''
Bluetooth IoT Scanner

A program that scans for Bluetooth IoT devices like LATCH door locks,
identifies their characteristics, and analyzes potential security vulnerabilities.
'''

import os
import sys
import time
import json
import datetime
import re
import subprocess
import signal
import socket
from typing import Dict, List, Set, Tuple, Any, Optional

# Try to import the Bluetooth libraries
try:
    import bluetooth
    from bluetooth.ble import DiscoveryService
    HAS_BLUETOOTH = True
except ImportError:
    HAS_BLUETOOTH = False
    print("Warning: Bluetooth libraries not found. Install pybluez and gattlib for full functionality.")
    print("You can install them with: pip install pybluez gattlib")

# Known IoT device manufacturers and their identifiers
IOT_DEVICE_MANUFACTURERS = {
    "LATCH": ["LATCH", "LC", "DOOR"],
    "August": ["AUGUST", "AUG", "SMRT"],
    "Schlage": ["SCHLAGE", "SCH"],
    "Yale": ["YALE", "YRD"],
    "Kwikset": ["KWIKSET", "KEVO"],
    "Nest": ["NEST", "GOOG"],
    "Philips": ["HUE", "PHILIPS"],
    "Samsung": ["SAMSUNG", "SMARTTHINGS"],
    "Apple": ["APPLE", "HOMEKIT"],
    "Amazon": ["AMAZON", "ECHO", "RING"],
    "IKEA": ["IKEA", "TRÅDFRI"],
    "Aqara": ["AQARA", "XIAOMI"],
    "Wyze": ["WYZE"],
    "TP-Link": ["TP-LINK", "KASA"],
    "Generic Smart Lock": ["SMARTLOCK", "DOORLOCK", "LOCK"]
}

# Security vulnerability patterns to check for
SECURITY_VULNERABILITIES = {
    "default_pin": {
        "description": "Default PIN codes (like 0000, 1234)",
        "severity": "HIGH",
        "remediation": "Change default PIN codes immediately"
    },
    "weak_encryption": {
        "description": "Weak encryption protocols (e.g., non-BLE devices)",
        "severity": "HIGH",
        "remediation": "Replace with devices supporting modern encryption"
    },
    "no_pairing_protection": {
        "description": "No protection during pairing process",
        "severity": "MEDIUM",
        "remediation": "Only pair devices in secure environments"
    },
    "plaintext_communication": {
        "description": "Data transmitted in plaintext",
        "severity": "HIGH",
        "remediation": "Update device firmware or replace device"
    },
    "replay_attacks": {
        "description": "Vulnerability to replay attacks",
        "severity": "HIGH",
        "remediation": "Update to devices with rolling codes or challenge-response"
    },
    "outdated_firmware": {
        "description": "Outdated firmware with known vulnerabilities",
        "severity": "MEDIUM",
        "remediation": "Update device firmware if available"
    },
    "excessive_permissions": {
        "description": "Excessive permissions requested by companion app",
        "severity": "MEDIUM",
        "remediation": "Check app permissions and consider alternatives"
    },
    "no_mfa": {
        "description": "No multi-factor authentication support",
        "severity": "MEDIUM",
        "remediation": "Use additional security measures"
    }
}

class BluetoothScanner:
    '''Main class for scanning Bluetooth devices and analyzing security'''
    
    def __init__(self):
        '''Initialize the scanner'''
        self.devices = {}
        self.scan_time = datetime.datetime.now()
        self.has_bluetooth = HAS_BLUETOOTH
    
    def check_system_dependencies(self) -> bool:
        '''
        Check if the system has the necessary tools installed.
        
        Returns:
            bool: True if all dependencies are met, False otherwise
        '''
        try:
            # Check if hcitool exists (Linux)
            hcitool_exists = subprocess.call(
                ["which", "hcitool"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            ) == 0
            
            # Check if bluetoothctl exists (Linux)
            bluetoothctl_exists = subprocess.call(
                ["which", "bluetoothctl"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            ) == 0
            
            # Check if we're on Linux
            is_linux = sys.platform.startswith('linux')
            
            # Check if we're on macOS and have the right tools
            is_macos = sys.platform == 'darwin'
            system_profiler_exists = False
            if is_macos:
                system_profiler_exists = subprocess.call(
                    ["which", "system_profiler"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                ) == 0
            
            if (is_linux and (hcitool_exists or bluetoothctl_exists)) or (is_macos and system_profiler_exists):
                return True
            
            print("Warning: Missing Bluetooth command-line tools.")
            if is_linux:
                print("Install bluez tools with: sudo apt-get install bluez")
            elif is_macos:
                print("Make sure system_profiler is available")
            
            return False
            
        except Exception as e:
            print(f"Error checking dependencies: {e}")
            return False
    
    def scan_for_devices(self, duration: int = 10) -> Dict[str, Any]:
        '''
        Scan for Bluetooth devices.
        
        Args:
            duration: Scan duration in seconds
            
        Returns:
            Dict containing discovered devices
        '''
        print(f"Scanning for Bluetooth devices (duration: {duration}s)...")
        self.scan_time = datetime.datetime.now()
        
        # Dictionary to store discovered devices
        discovered_devices = {}
        
        # Try the Python Bluetooth library first
        if HAS_BLUETOOTH:
            try:
                print("Using Python Bluetooth library...")
                # Classic Bluetooth scan
                nearby_devices = bluetooth.discover_devices(
                    duration=duration,
                    lookup_names=True,
                    lookup_class=True,
                    device_id=-1
                )
                
                for addr, name, device_class in nearby_devices:
                    device_type = self._get_device_class_name(device_class)
                    discovered_devices[addr] = {
                        "name": name if name else "Unknown",
                        "address": addr,
                        "type": "Classic Bluetooth",
                        "device_class": device_type,
                        "rssi": None,  # Not available with standard discovery
                        "manufacturer": self._identify_manufacturer(name),
                        "seen_time": datetime.datetime.now().isoformat(),
                        "possible_vulnerabilities": []
                    }
                
                # BLE scan if available
                try:
                    service = DiscoveryService()
                    ble_devices = service.discover(duration)
                    for address, name in ble_devices.items():
                        discovered_devices[address] = {
                            "name": name if name else "Unknown",
                            "address": address,
                            "type": "Bluetooth LE",
                            "device_class": "Low Energy Device",
                            "rssi": None,  # Not directly available through this API
                            "manufacturer": self._identify_manufacturer(name),
                            "seen_time": datetime.datetime.now().isoformat(),
                            "possible_vulnerabilities": []
                        }
                except Exception as e:
                    print(f"BLE scan error: {e}")
            
            except Exception as e:
                print(f"Error using Python Bluetooth library: {e}")
        
        # If that didn't work, try system tools
        if not discovered_devices:
            discovered_devices = self._scan_with_system_tools(duration)
        
        # Update the class's device list
        self.devices.update(discovered_devices)
        
        # Check for security vulnerabilities in each device
        for addr, device in self.devices.items():
            # Identify potential vulnerabilities based on device characteristics
            vulnerabilities = self._check_security_vulnerabilities(device)
            device["possible_vulnerabilities"] = vulnerabilities
        
        return self.devices
    
    def _scan_with_system_tools(self, duration: int = 10) -> Dict[str, Any]:
        '''
        Scan for Bluetooth devices using system tools.
        
        Args:
            duration: Scan duration in seconds
            
        Returns:
            Dict containing discovered devices
        '''
        discovered_devices = {}
        
        try:
            if sys.platform.startswith('linux'):
                # Try bluetoothctl first (newer systems)
                try:
                    print("Using bluetoothctl...")
                    # Start a scan in bluetoothctl
                    scan_process = subprocess.Popen(
                        ["bluetoothctl"],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True
                    )
                    
                    # Send scan commands
                    scan_process.stdin.write("scan on\n")
                    scan_process.stdin.flush()
                    
                    # Wait for scan duration
                    time.sleep(duration)
                    
                    # Stop scanning
                    scan_process.stdin.write("scan off\n")
                    scan_process.stdin.write("devices\n")
                    scan_process.stdin.flush()
                    
                    # Wait a moment to get results
                    time.sleep(1)
                    
                    # Get output and close the process
                    scan_process.stdin.write("quit\n")
                    scan_process.stdin.flush()
                    
                    output, _ = scan_process.communicate(timeout=5)
                    
                    # Parse output
                    for line in output.splitlines():
                        if "Device" in line:
                            parts = line.strip().split(" ", 2)
                            if len(parts) >= 3:
                                addr = parts[1]
                                name = parts[2] if len(parts) > 2 else "Unknown"
                                discovered_devices[addr] = {
                                    "name": name,
                                    "address": addr,
                                    "type": "Unknown (System Tools)",
                                    "device_class": "Unknown",
                                    "rssi": None,
                                    "manufacturer": self._identify_manufacturer(name),
                                    "seen_time": datetime.datetime.now().isoformat(),
                                    "possible_vulnerabilities": []
                                }
                
                except Exception as e:
                    print(f"bluetoothctl scan error: {e}")
                
                # Try hcitool if bluetoothctl didn't work
                if not discovered_devices:
                    try:
                        print("Using hcitool...")
                        # Start hcitool scan
                        output = subprocess.check_output(
                            ["hcitool", "scan"],
                            universal_newlines=True,
                            timeout=duration+5
                        )
                        
                        for line in output.splitlines():
                            if not line.startswith("Scanning") and "	" in line:
                                parts = line.strip().split("	")
                                if len(parts) >= 2:
                                    addr = parts[1]
                                    name = parts[2] if len(parts) > 2 else "Unknown"
                                    discovered_devices[addr] = {
                                        "name": name,
                                        "address": addr,
                                        "type": "Unknown (hcitool)",
                                        "device_class": "Unknown",
                                        "rssi": None,
                                        "manufacturer": self._identify_manufacturer(name),
                                        "seen_time": datetime.datetime.now().isoformat(),
                                        "possible_vulnerabilities": []
                                    }
                    
                    except Exception as e:
                        print(f"hcitool scan error: {e}")
            
            elif sys.platform == 'darwin':  # macOS
                try:
                    print("Using macOS system_profiler...")
                    output = subprocess.check_output(
                        ["system_profiler", "SPBluetoothDataType"],
                        universal_newlines=True
                    )
                    
                    current_device = None
                    device_info = {}
                    
                    for line in output.splitlines():
                        line = line.strip()
                        
                        # Start of a new device
                        if ":" in line and not line.startswith(" "):
                            if current_device and device_info:
                                discovered_devices[device_info.get("address", current_device)] = device_info
                                device_info = {}
                            
                            current_device = line.split(":", 1)[0].strip()
                            device_info = {
                                "name": current_device,
                                "address": "Unknown",
                                "type": "Unknown (macOS)",
                                "device_class": "Unknown",
                                "rssi": None,
                                "manufacturer": self._identify_manufacturer(current_device),
                                "seen_time": datetime.datetime.now().isoformat(),
                                "possible_vulnerabilities": []
                            }
                        
                        # Device address
                        elif "Address:" in line:
                            device_info["address"] = line.split(":", 1)[1].strip()
                        
                        # Device type
                        elif "Minor Type:" in line:
                            device_info["device_class"] = line.split(":", 1)[1].strip()
                    
                    # Add the last device
                    if current_device and device_info:
                        discovered_devices[device_info.get("address", current_device)] = device_info
                
                except Exception as e:
                    print(f"macOS system_profiler scan error: {e}")
            
            elif sys.platform == 'win32':  # Windows
                print("Windows Bluetooth scanning not directly supported.")
                print("Please install the Python Bluetooth libraries.")
        
        except Exception as e:
            print(f"Error scanning with system tools: {e}")
        
        return discovered_devices
    
    def _identify_manufacturer(self, device_name: str) -> str:
        '''
        Identify the manufacturer of a device based on its name.
        
        Args:
            device_name: Name of the device
            
        Returns:
            String identifying the manufacturer
        '''
        if not device_name:
            return "Unknown"
        
        device_name_upper = device_name.upper()
        
        for manufacturer, identifiers in IOT_DEVICE_MANUFACTURERS.items():
            for identifier in identifiers:
                if identifier in device_name_upper:
                    return manufacturer
        
        return "Unknown"
    
    def _get_device_class_name(self, device_class: int) -> str:
        '''
        Get a human-readable name for the device class.
        
        Args:
            device_class: The device class code
            
        Returns:
            Human-readable device class name
        '''
        major_classes = {
            0: "Miscellaneous",
            1: "Computer",
            2: "Phone",
            3: "LAN/Network Access Point",
            4: "Audio/Video",
            5: "Peripheral",
            6: "Imaging",
            7: "Wearable",
            8: "Toy",
            9: "Health",
            31: "Uncategorized"
        }
        
        # Extract major device class (bits 8-12)
        major_class = (device_class >> 8) & 0x1F
        
        return major_classes.get(major_class, "Unknown")
    
    def _check_security_vulnerabilities(self, device: Dict[str, Any]) -> List[Dict[str, Any]]:
        '''
        Check for potential security vulnerabilities in a device.
        
        Args:
            device: Device information dictionary
            
        Returns:
            List of potential security vulnerabilities
        '''
        vulnerabilities = []
        
        # Check device type - older Classic Bluetooth is generally less secure
        if device["type"] == "Classic Bluetooth":
            vulnerabilities.append({
                "type": "weak_encryption",
                **SECURITY_VULNERABILITIES["weak_encryption"]
            })
        
        # Check for LATCH door locks specifically - they've had historical issues
        if self._identify_manufacturer(device["name"]) == "LATCH":
            # Check for older LATCH models which had vulnerability to replay attacks
            if "LC1" in device["name"] or "2017" in device["name"] or "2018" in device["name"]:
                vulnerabilities.append({
                    "type": "replay_attacks",
                    **SECURITY_VULNERABILITIES["replay_attacks"]
                })
        
        # Smart locks typically don't implement MFA via the Bluetooth interface
        if any(keyword in device["name"].upper() for keyword in ["LOCK", "SMART", "DOOR", "LATCH", "YALE", "SCHLAGE"]):
            vulnerabilities.append({
                "type": "no_mfa",
                **SECURITY_VULNERABILITIES["no_mfa"]
            })
        
        # Generic device security checks
        if self._is_low_rssi_device(device):
            # Devices with weak signal might be vulnerable to signal boosting attacks
            vulnerabilities.append({
                "type": "weak_signal",
                "description": "Weak signal strength may indicate vulnerability to signal amplification attacks",
                "severity": "LOW",
                "remediation": "Ensure device is installed in optimal location"
            })
        
        return vulnerabilities
    
    def _is_low_rssi_device(self, device: Dict[str, Any]) -> bool:
        '''
        Check if a device has a low RSSI (signal strength).
        
        Args:
            device: Device information dictionary
            
        Returns:
            True if the device has a weak signal, False otherwise
        '''
        # If RSSI is available and less than -80 dBm, consider it weak
        return device.get("rssi", 0) is not None and device.get("rssi", 0) < -80
    
    def analyze_device_security(self, address: str) -> Dict[str, Any]:
        '''
        Perform a deeper security analysis on a specific device.
        
        Args:
            address: MAC address of the device to analyze
            
        Returns:
            Dictionary with security analysis results
        '''
        if address not in self.devices:
            return {
                "error": "Device not found",
                "address": address
            }
        
        device = self.devices[address]
        security_analysis = {
            "device": device,
            "vulnerabilities": device.get("possible_vulnerabilities", []),
            "recommendations": [],
            "further_tests": []
        }
        
        # Add recommendations based on vulnerabilities
        for vulnerability in device.get("possible_vulnerabilities", []):
            security_analysis["recommendations"].append({
                "issue": vulnerability["description"],
                "action": vulnerability["remediation"]
            })
        
        # Add further tests that could be performed
        if "LATCH" in self._identify_manufacturer(device["name"]):
            security_analysis["further_tests"].append({
                "name": "Connection handshake analysis",
                "description": "Analyze the BLE connection handshake for replay protection",
                "tools": ["Ubertooth", "GATTacker"]
            })
        
        if device["type"] == "Bluetooth LE":
            security_analysis["further_tests"].append({
                "name": "GATT service enumeration",
                "description": "Enumerate and check GATT services for unprotected characteristics",
                "tools": ["gatttool", "bleah", "BtleJack"]
            })
        
        return security_analysis
    
    def save_scan_result(self, filename: str = None) -> str:
        '''
        Save scan results to a JSON file.
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to the saved file
        '''
        if not filename:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bluetooth_scan_{timestamp}.json"
        
        data = {
            "scan_time": self.scan_time.isoformat(),
            "scan_duration_seconds": (datetime.datetime.now() - self.scan_time).total_seconds(),
            "device_count": len(self.devices),
            "devices": self.devices,
            "vulnerabilities_summary": self._generate_vulnerabilities_summary()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Scan results saved to {filename}")
        return filename
    
    def _generate_vulnerabilities_summary(self) -> Dict[str, Any]:
        '''
        Generate a summary of vulnerabilities found in devices.
        
        Returns:
            Dictionary with vulnerability statistics
        '''
        summary = {
            "total_vulnerabilities": 0,
            "devices_with_vulnerabilities": 0,
            "vulnerability_types": {},
            "manufacturers_with_vulnerabilities": {},
            "high_severity_count": 0,
            "medium_severity_count": 0,
            "low_severity_count": 0
        }
        
        # Count devices with vulnerabilities
        for device in self.devices.values():
            vulnerabilities = device.get("possible_vulnerabilities", [])
            if vulnerabilities:
                summary["devices_with_vulnerabilities"] += 1
                summary["total_vulnerabilities"] += len(vulnerabilities)
                
                manufacturer = device.get("manufacturer", "Unknown")
                if manufacturer not in summary["manufacturers_with_vulnerabilities"]:
                    summary["manufacturers_with_vulnerabilities"][manufacturer] = 0
                summary["manufacturers_with_vulnerabilities"][manufacturer] += 1
                
                for vuln in vulnerabilities:
                    vuln_type = vuln.get("type", "unknown")
                    if vuln_type not in summary["vulnerability_types"]:
                        summary["vulnerability_types"][vuln_type] = 0
                    summary["vulnerability_types"][vuln_type] += 1
                    
                    severity = vuln.get("severity", "").upper()
                    if severity == "HIGH":
                        summary["high_severity_count"] += 1
                    elif severity == "MEDIUM":
                        summary["medium_severity_count"] += 1
                    elif severity == "LOW":
                        summary["low_severity_count"] += 1
        
        return summary

def main():
    '''Main function to run the Bluetooth IoT scanner.'''
    print("Bluetooth IoT Scanner & Security Analyzer")
    print("=========================================")
    
    # Create scanner
    scanner = BluetoothScanner()
    
    # Check dependencies
    if not scanner.has_bluetooth:
        print("
Python Bluetooth libraries not found.")
        print("The scanner will attempt to use system tools, but functionality may be limited.")
    
    if not scanner.check_system_dependencies():
        print("
Warning: Some system dependencies are missing.")
        print("The scanner will continue with limited functionality.")
    
    # Set scan duration
    scan_duration = 15  # seconds
    print(f"
Scanning for Bluetooth devices for {scan_duration} seconds...")
    
    # Perform the scan
    devices = scanner.scan_for_devices(duration=scan_duration)
    
    if not devices:
        print("
No Bluetooth devices found.")
        return
    
    # Print discovered devices
    print(f"
Discovered {len(devices)} Bluetooth devices:")
    
    for addr, device in devices.items():
        print(f"
Device: {device['name']} ({addr})")
        print(f"  Type: {device['type']}")
        print(f"  Class: {device['device_class']}")
        print(f"  Manufacturer: {device['manufacturer']}")
        
        # Print vulnerabilities
        vulnerabilities = device.get("possible_vulnerabilities", [])
        if vulnerabilities:
            print(f"  Potential Security Issues: {len(vulnerabilities)}")
            for i, vuln in enumerate(vulnerabilities, 1):
                print(f"    {i}. {vuln['description']} (Severity: {vuln['severity']})")
                print(f"       Recommendation: {vuln['remediation']}")
        else:
            print("  No obvious security issues detected")
    
    # Check for IoT locks specifically
    locks = [d for d in devices.values() if any(k in d['name'].upper() for k in ["LOCK", "LATCH", "YALE", "SCHLAGE", "KWIKSET"])]
    if locks:
        print("
Smart Lock Devices Found:")
        for lock in locks:
            print(f"  {lock['name']} ({lock['address']})")
            print(f"  Manufacturer: {lock['manufacturer']}")
            
            # Deeper analysis for locks
            analysis = scanner.analyze_device_security(lock['address'])
            
            if "error" not in analysis:
                print("  Security Analysis:")
                if analysis["vulnerabilities"]:
                    print("    Potential vulnerabilities:")
                    for vuln in analysis["vulnerabilities"]:
                        print(f"      - {vuln['description']} (Severity: {vuln['severity']})")
                
                if analysis["recommendations"]:
                    print("    Recommendations:")
                    for rec in analysis["recommendations"]:
                        print(f"      - {rec['action']}")
    
    # Generate vulnerability summary
    vulns_summary = scanner._generate_vulnerabilities_summary()
    print("
Security Vulnerability Summary:")
    print(f"  Devices with security issues: {vulns_summary['devices_with_vulnerabilities']}/{len(devices)}")
    print(f"  Total vulnerabilities found: {vulns_summary['total_vulnerabilities']}")
    print(f"  High severity issues: {vulns_summary['high_severity_count']}")
    print(f"  Medium severity issues: {vulns_summary['medium_severity_count']}")
    print(f"  Low severity issues: {vulns_summary['low_severity_count']}")
    
    # Save results
    output_file = scanner.save_scan_result()
    print(f"
Scan complete! Full results saved to {output_file}")

if __name__ == "__main__":
    main()
