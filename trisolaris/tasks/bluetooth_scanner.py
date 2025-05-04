"""
Bluetooth IoT Scanner Task for TRISOLARIS

This module implements the Bluetooth IoT Scanner task interface for the TRISOLARIS
evolutionary framework. This task focuses on finding Bluetooth IoT devices like
LATCH door locks and identifying potential security vulnerabilities.
"""

import os
import json
import subprocess
import tempfile
import time
import ast
import re
import datetime
from typing import Dict, Any, Tuple, List, Optional

from trisolaris.tasks.base import TaskInterface, TrisolarisBoundary

class BluetoothScannerTask(TaskInterface):
    """
    Task interface for evolving a Bluetooth IoT device scanner with security analysis.
    
    This implementation evolves a program that can scan for Bluetooth devices,
    focusing on IoT devices like smart locks, and identify potential security
    vulnerabilities in their implementation.
    """
    
    def __init__(self, template_path: str = None):
        """
        Initialize the Bluetooth scanner task.
        
        Args:
            template_path: Path to a template Bluetooth scanner program file
        """
        self.template_path = template_path
    
    def get_name(self) -> str:
        """
        Get the name of this task.
        
        Returns:
            A string identifying this task
        """
        return "bluetooth_scanner"
    
    def get_description(self) -> str:
        """
        Get a human-readable description of this task.
        
        Returns:
            A string describing the purpose and functionality of this task
        """
        return (
            "A program that scans for Bluetooth IoT devices like LATCH door locks, "
            "identifies their characteristics, connection protocols, and potential "
            "security vulnerabilities such as weak encryption, default credentials, "
            "or outdated firmware."
        )
    
    def get_template(self) -> str:
        """
        Get the template code to start evolution from.
        
        Returns:
            A string containing the template source code
        """
        if self.template_path and os.path.exists(self.template_path):
            with open(self.template_path, 'r', encoding='utf-8') as f:
                return f.read()
        else:
            # Return a template as a fallback
            return """#!/usr/bin/env python3
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
                    scan_process.stdin.write("scan on\\n")
                    scan_process.stdin.flush()
                    
                    # Wait for scan duration
                    time.sleep(duration)
                    
                    # Stop scanning
                    scan_process.stdin.write("scan off\\n")
                    scan_process.stdin.write("devices\\n")
                    scan_process.stdin.flush()
                    
                    # Wait a moment to get results
                    time.sleep(1)
                    
                    # Get output and close the process
                    scan_process.stdin.write("quit\\n")
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
                            if not line.startswith("Scanning") and "\t" in line:
                                parts = line.strip().split("\t")
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
        print("\nPython Bluetooth libraries not found.")
        print("The scanner will attempt to use system tools, but functionality may be limited.")
    
    if not scanner.check_system_dependencies():
        print("\nWarning: Some system dependencies are missing.")
        print("The scanner will continue with limited functionality.")
    
    # Set scan duration
    scan_duration = 15  # seconds
    print(f"\nScanning for Bluetooth devices for {scan_duration} seconds...")
    
    # Perform the scan
    devices = scanner.scan_for_devices(duration=scan_duration)
    
    if not devices:
        print("\nNo Bluetooth devices found.")
        return
    
    # Print discovered devices
    print(f"\nDiscovered {len(devices)} Bluetooth devices:")
    
    for addr, device in devices.items():
        print(f"\nDevice: {device['name']} ({addr})")
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
        print("\nSmart Lock Devices Found:")
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
    print("\nSecurity Vulnerability Summary:")
    print(f"  Devices with security issues: {vulns_summary['devices_with_vulnerabilities']}/{len(devices)}")
    print(f"  Total vulnerabilities found: {vulns_summary['total_vulnerabilities']}")
    print(f"  High severity issues: {vulns_summary['high_severity_count']}")
    print(f"  Medium severity issues: {vulns_summary['medium_severity_count']}")
    print(f"  Low severity issues: {vulns_summary['low_severity_count']}")
    
    # Save results
    output_file = scanner.save_scan_result()
    print(f"\nScan complete! Full results saved to {output_file}")

if __name__ == "__main__":
    main()
"""
    
    def evaluate_fitness(self, source_code: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the fitness of the provided source code for this task.
        
        Args:
            source_code: The Bluetooth scanner source code to evaluate
            
        Returns:
            A tuple containing (fitness_score, detailed_results)
        """
        results = {
            "syntax_valid": False,
            "runtime_successful": False,
            "bluetooth_detection": 0,
            "device_discovery": 0,
            "vulnerability_detection": 0,
            "latch_specific": 0,
            "security_analysis": 0,
            "error_handling": 0,
            "user_interface": 0,
            "output_formatting": 0,
            "errors": []
        }
        
        # Check syntax validity
        try:
            ast.parse(source_code)
            results["syntax_valid"] = True
        except SyntaxError as e:
            results["errors"].append(f"Syntax error: {str(e)}")
            # Return early if syntax is invalid
            return 0.0, results
        
        # Create a temporary file for the program
        try:
            with tempfile.NamedTemporaryFile(suffix='.py', delete=False, mode='w') as temp_file:
                temp_filename = temp_file.name
                temp_file.write(source_code)
            
            # Make the script executable
            os.chmod(temp_filename, 0o755)
            
            # Check required functionality through code analysis
            results.update(self._check_required_functionality(source_code))
            
            # We won't execute the Bluetooth scanner directly for testing as it requires hardware
            # Instead, we'll simulate a successful runtime if the code passes checks
            if results["bluetooth_detection"] > 0 and results["device_discovery"] > 0:
                results["runtime_successful"] = True
            
        finally:
            # Clean up the temporary file
            if 'temp_filename' in locals():
                try:
                    os.unlink(temp_filename)
                except:
                    pass
        
        # Calculate the overall fitness score
        weights = {
            "syntax_valid": 0.1,
            "runtime_successful": 0.1,
            "bluetooth_detection": 0.15,
            "device_discovery": 0.15,
            "vulnerability_detection": 0.2,
            "latch_specific": 0.1,
            "security_analysis": 0.1,
            "error_handling": 0.05,
            "user_interface": 0.025,
            "output_formatting": 0.025
        }
        
        fitness_score = 0.0
        for criterion, weight in weights.items():
            score = float(results.get(criterion, 0))
            fitness_score += score * weight
        
        return fitness_score, results
    
    def _check_required_functionality(self, code: str) -> Dict[str, Any]:
        """
        Check if the code includes required functionality.
        
        Args:
            code: Source code to check
            
        Returns:
            Dictionary with functionality scores
        """
        results = {
            "bluetooth_detection": 0,
            "device_discovery": 0,
            "vulnerability_detection": 0,
            "latch_specific": 0,
            "security_analysis": 0,
            "error_handling": 0,
            "user_interface": 0,
            "output_formatting": 0
        }
        
        # Check for Bluetooth detection functionality
        bluetooth_detection_patterns = [
            r'import\s+bluetooth',
            r'DiscoveryService',
            r'hcitool',
            r'bluetoothctl',
            r'system_profiler\s+SPBluetoothDataType'
        ]
        detection_score = sum(0.2 for pattern in bluetooth_detection_patterns if re.search(pattern, code, re.IGNORECASE))
        results["bluetooth_detection"] = min(1.0, detection_score)
        
        # Check for device discovery functionality
        device_discovery_patterns = [
            r'discover_devices',
            r'scan_for_devices',
            r'scanForDevices',
            r'bluetooth\.discover',
            r'hcitool\s+scan'
        ]
        discovery_score = sum(0.2 for pattern in device_discovery_patterns if re.search(pattern, code, re.IGNORECASE))
        results["device_discovery"] = min(1.0, discovery_score)
        
        # Check for vulnerability detection
        vulnerability_patterns = [
            r'vulnerability',
            r'security',
            r'check_security',
            r'analyze_security',
            r'vulnerability_detection',
            r'weak_encryption',
            r'default_pin',
            r'replay_attack'
        ]
        vulnerability_score = sum(0.125 for pattern in vulnerability_patterns if re.search(pattern, code, re.IGNORECASE))
        results["vulnerability_detection"] = min(1.0, vulnerability_score)
        
        # Check for LATCH door lock specific handling
        latch_patterns = [
            r'LATCH',
            r'door\s*lock',
            r'smart\s*lock',
            r'IOT_DEVICE_MANUFACTURERS',
            r'"LATCH"'
        ]
        latch_score = sum(0.2 for pattern in latch_patterns if re.search(pattern, code, re.IGNORECASE))
        results["latch_specific"] = min(1.0, latch_score)
        
        # Check for security analysis
        security_analysis_patterns = [
            r'security_analysis',
            r'analyzeDevice',
            r'analyze_device',
            r'generateReport',
            r'generate_report',
            r'vulnerabilit(y|ies)_summary'
        ]
        analysis_score = sum(0.2 for pattern in security_analysis_patterns if re.search(pattern, code, re.IGNORECASE))
        results["security_analysis"] = min(1.0, analysis_score)
        
        # Check for error handling
        error_handling_patterns = [
            r'try\s*:.*except',
            r'error',
            r'exception',
            r'if\s+not\s+HAS_BLUETOOTH'
        ]
        if any(re.search(pattern, code, re.IGNORECASE | re.DOTALL) for pattern in error_handling_patterns):
            results["error_handling"] = 1.0
        
        # Check for user interface elements
        ui_patterns = [
            r'print\s*\(',
            r'prompt',
            r'input\s*\(',
            r'display',
            r'user.*friendly'
        ]
        ui_score = sum(0.2 for pattern in ui_patterns if re.search(pattern, code, re.IGNORECASE))
        results["user_interface"] = min(1.0, ui_score)
        
        # Check for output formatting
        output_patterns = [
            r'json\.dump',
            r'save_result',
            r'save_scan_result',
            r'output_file',
            r'\.json'
        ]
        output_score = sum(0.2 for pattern in output_patterns if re.search(pattern, code, re.IGNORECASE))
        results["output_formatting"] = min(1.0, output_score)
        
        return results
    
    def get_required_boundaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the ethical boundaries required for this task.
        
        Returns:
            A dictionary mapping boundary names to their parameters
        """
        return {
            TrisolarisBoundary.NO_EVAL_EXEC: {},
            TrisolarisBoundary.MAX_EXECUTION_TIME: {"max_execution_time": 30.0},  # Bluetooth scans can take time
            TrisolarisBoundary.MAX_MEMORY_USAGE: {"max_memory_usage": 200}        # Generous memory allocation
        }
    
    def get_fitness_weights(self) -> Dict[str, float]:
        """
        Get the weights for different fitness components.
        
        Returns:
            A dictionary mapping fitness component names to their weights
        """
        return {
            "functionality": 0.7,  # Highest priority for functionality
            "efficiency": 0.1,     # Lower priority for efficiency (Bluetooth operations are inherently slow)
            "alignment": 0.2       # Higher priority for ethical alignment due to security implications
        }
    
    def get_allowed_imports(self) -> List[str]:
        """
        Get the list of allowed imports for this task.
        
        Returns:
            A list of allowed import module names
        """
        return [
            "os", "sys", "time", "json", "datetime", "re", "socket",
            "subprocess", "signal", "bluetooth", "threading", "concurrent"
        ]
    
    def get_evolution_params(self) -> Dict[str, Any]:
        """
        Get recommended evolution parameters for this task.
        
        Returns:
            A dictionary of parameters for the evolution process
        """
        return {
            "population_size": 50,     # Larger population for diverse solutions
            "num_generations": 20,     # More generations to improve quality
            "mutation_rate": 0.1,
            "crossover_rate": 0.7
        }
    
    def post_process(self, source_code: str) -> str:
        """
        Perform post-processing on evolved source code.
        
        Args:
            source_code: The evolved source code
            
        Returns:
            The post-processed source code
        """
        # Ensure there's a proper shebang
        if not source_code.startswith("#!/"):
            source_code = "#!/usr/bin/env python3\n" + source_code
        
        # Add a timestamp in a comment
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        source_code = source_code.replace(
            "#!/usr/bin/env python3\n",
            f"#!/usr/bin/env python3\n# Generated by TRISOLARIS on {timestamp}\n"
        )
        
        # Add installation instructions for bluetooth libraries if not present
        if "import bluetooth" in source_code and "pip install" not in source_code:
            instruction = """
# Note: This script requires the Bluetooth libraries.
# You can install them with:
#   pip install pybluez gattlib
# On Linux, you may also need:
#   sudo apt-get install bluetooth libbluetooth-dev
"""
            # Find a good place to insert the instructions - after imports or after docstring
            import_match = re.search(r'(import.*?)$', source_code, re.MULTILINE)
            if import_match:
                pos = import_match.end()
                source_code = source_code[:pos] + instruction + source_code[pos:]
        
        return source_code 