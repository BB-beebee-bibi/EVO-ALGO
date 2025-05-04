"""
Network Scanner Task for TRISOLARIS

This module implements the Network Scanner task interface for the TRISOLARIS
evolutionary framework.
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

class NetworkScannerTask(TaskInterface):
    """
    Task interface for evolving a network scanner program.
    
    This implementation evolves a program that can scan local networks,
    discover devices, and report detailed information about them.
    """
    
    def __init__(self, template_path: str = None):
        """
        Initialize the network scanner task.
        
        Args:
            template_path: Path to a template network scanner program file
        """
        self.template_path = template_path
    
    def get_name(self) -> str:
        """
        Get the name of this task.
        
        Returns:
            A string identifying this task
        """
        return "network_scanner"
    
    def get_description(self) -> str:
        """
        Get a human-readable description of this task.
        
        Returns:
            A string describing the purpose and functionality of this task
        """
        return (
            "A program that scans local networks, identifies all connected devices "
            "including IoT devices like Nest thermostats, and provides detailed information "
            "about them including IP addresses, MAC addresses, device types, and open ports."
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
            # Return a minimal template as a fallback
            return """#!/usr/bin/env python3
'''
Network Scanner

A program that scans local networks, identifies all connected devices including IoT devices,
and provides detailed information about them.
'''

import os
import sys
import socket
import subprocess
import json
import time
import datetime
import re
from typing import Dict, List, Any, Optional

def get_local_ip() -> str:
    '''
    Get the local IP address of the machine.
    
    Returns:
        The local IP address
    '''
    try:
        # Create a socket to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception as e:
        print(f"Error getting local IP: {e}")
        return "127.0.0.1"

def get_network_range(local_ip: str) -> str:
    '''
    Get the network range based on the local IP.
    
    Args:
        local_ip: The local IP address
        
    Returns:
        The network range (e.g., 192.168.1.0/24)
    '''
    # Simple approach: assume 24-bit mask for local networks
    ip_parts = local_ip.split('.')
    return f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.0/24"

def scan_network(network_range: str) -> List[Dict[str, Any]]:
    '''
    Scan the network for active hosts using ping.
    
    Args:
        network_range: The network range to scan
        
    Returns:
        List of dictionaries with information about discovered hosts
    '''
    print(f"Scanning network {network_range}...")
    hosts = []
    
    # Extract base network for ping
    base_network = network_range.split('/')[0].rsplit('.', 1)[0]
    
    # Ping each host in the subnet (1-254)
    for i in range(1, 255):
        ip = f"{base_network}.{i}"
        
        try:
            # Use ping to check if host is active
            response = subprocess.run(
                ["ping", "-c", "1", "-W", "1", ip],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            if response.returncode == 0:
                print(f"Discovered host: {ip}")
                hosts.append({
                    "ip": ip,
                    "status": "active",
                    "mac_address": get_mac_address(ip),
                    "hostname": get_hostname(ip),
                    "open_ports": [],
                    "device_type": "unknown"
                })
        except Exception as e:
            print(f"Error pinging {ip}: {e}")
    
    return hosts

def get_mac_address(ip: str) -> Optional[str]:
    '''
    Get the MAC address for an IP using the ARP table.
    
    Args:
        ip: The IP address
        
    Returns:
        The MAC address or None if not found
    '''
    try:
        # Use the arp command to get the MAC address
        output = subprocess.check_output(["arp", "-n", ip], universal_newlines=True)
        
        # Parse the output to extract the MAC address
        mac_match = re.search(r'([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})', output)
        if mac_match:
            return mac_match.group(0)
    except Exception as e:
        print(f"Error getting MAC address for {ip}: {e}")
    
    return None

def get_hostname(ip: str) -> Optional[str]:
    '''
    Get the hostname for an IP address.
    
    Args:
        ip: The IP address
        
    Returns:
        The hostname or None if not resolvable
    '''
    try:
        return socket.gethostbyaddr(ip)[0]
    except (socket.herror, socket.gaierror):
        return None

def scan_ports(host: Dict[str, Any], ports: List[int]) -> Dict[str, Any]:
    '''
    Scan for open ports on a host.
    
    Args:
        host: Host information dictionary
        ports: List of ports to scan
        
    Returns:
        Updated host dictionary with open ports information
    '''
    open_ports = []
    
    for port in ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex((host["ip"], port))
            if result == 0:
                service = get_service_name(port)
                open_ports.append({
                    "port": port,
                    "service": service
                })
                print(f"  - Open port {port}: {service}")
            sock.close()
        except Exception as e:
            print(f"Error scanning port {port} on {host['ip']}: {e}")
    
    host["open_ports"] = open_ports
    return host

def get_service_name(port: int) -> str:
    '''
    Get the name of a service for a standard port.
    
    Args:
        port: The port number
        
    Returns:
        A string describing the service
    '''
    common_ports = {
        21: "FTP",
        22: "SSH",
        23: "Telnet",
        25: "SMTP",
        53: "DNS",
        80: "HTTP",
        110: "POP3",
        123: "NTP",
        143: "IMAP",
        443: "HTTPS",
        445: "SMB",
        548: "AFP",
        631: "IPP (Printing)",
        993: "IMAPS",
        995: "POP3S",
        1883: "MQTT (IoT)",
        2323: "Telnet Alt",
        3000: "Development Server",
        3306: "MySQL",
        3389: "RDP",
        5000: "UPnP",
        5353: "mDNS",
        5683: "CoAP (IoT)",
        8000: "HTTP Alt",
        8080: "HTTP Proxy",
        8443: "HTTPS Alt",
        8883: "MQTT Secure",
        9100: "Printer"
    }
    return common_ports.get(port, "Unknown")

def identify_device_type(host: Dict[str, Any]) -> Dict[str, Any]:
    '''
    Attempt to identify the device type based on open ports, MAC address, and other info.
    
    Args:
        host: Host information dictionary
        
    Returns:
        Updated host dictionary with device type information
    '''
    # Check if host has open ports that indicate specific device types
    open_port_numbers = [p["port"] for p in host["open_ports"]]
    
    # Check MAC address vendor (first 3 bytes) if available
    mac_vendor = ""
    if host["mac_address"]:
        mac_prefix = host["mac_address"].replace(":", "").replace("-", "").upper()[:6]
        # In a real implementation, this would check against a MAC vendor database
        # For now, just use a simple example
        if mac_prefix.startswith("B0F"):
            mac_vendor = "Google/Nest"
        elif mac_prefix.startswith("CC"):
            mac_vendor = "Amazon"
        elif mac_prefix.startswith("ECDA"):
            mac_vendor = "Samsung"
    
    # Determine device type based on collected information
    device_type = "Unknown"
    
    # Check for specific devices by port signatures
    if 1883 in open_port_numbers or 8883 in open_port_numbers:
        device_type = "IoT Device (MQTT)"
    elif 5683 in open_port_numbers:
        device_type = "IoT Device (CoAP)"
    elif 80 in open_port_numbers and 8080 in open_port_numbers:
        device_type = "Router/Gateway"
    elif 8080 in open_port_numbers:
        device_type = "IP Camera or Media Device"
    elif 548 in open_port_numbers or 445 in open_port_numbers:
        device_type = "Network Storage (NAS)"
    elif 22 in open_port_numbers and len(open_port_numbers) < 3:
        device_type = "Raspberry Pi or Linux Server"
    
    # Override with vendor-specific info if available
    if mac_vendor == "Google/Nest":
        device_type = "Nest Device"
    
    # Further refinement based on hostname
    if host["hostname"]:
        hostname = host["hostname"].lower()
        if "printer" in hostname or "print" in hostname:
            device_type = "Printer"
        elif "thermo" in hostname or "nest" in hostname:
            device_type = "Nest Thermostat"
        elif "cam" in hostname or "camera" in hostname:
            device_type = "Security Camera"
        elif "tv" in hostname or "roku" in hostname or "apple-tv" in hostname:
            device_type = "Smart TV/Media Device"
        elif "google" in hostname or "home" in hostname:
            device_type = "Google Home/Nest Device"
        elif "alexa" in hostname or "echo" in hostname:
            device_type = "Amazon Echo/Alexa Device"
    
    host["device_type"] = device_type
    return host

def save_scan_result(network_info: Dict[str, Any], filename: str) -> None:
    '''
    Save scan result to a JSON file.
    
    Args:
        network_info: Network scan information
        filename: Target filename
    '''
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(network_info, f, indent=2)
        print(f"Scan result saved to {filename}")
    except Exception as e:
        print(f"Error saving scan result: {e}")

def main():
    '''Main function to run the network scanner.'''
    print("Network Scanner")
    print("===============")
    
    # Get local IP and network range
    local_ip = get_local_ip()
    print(f"Local IP: {local_ip}")
    
    network_range = get_network_range(local_ip)
    
    # Scan the network
    scan_start = datetime.datetime.now()
    hosts = scan_network(network_range)
    
    if not hosts:
        print("No hosts discovered.")
        return
    
    print(f"Discovered {len(hosts)} hosts. Scanning ports and identifying devices...")
    
    # Common ports to scan
    common_ports = [21, 22, 23, 25, 53, 80, 110, 123, 143, 443, 445, 548, 631, 993, 995,
                   1883, 2323, 3000, 3306, 3389, 5000, 5353, 5683, 8000, 8080, 8443, 8883, 9100]
    
    # Scan ports and identify each host
    for host in hosts:
        print(f"Scanning {host['ip']}...")
        scan_ports(host, common_ports)
        identify_device_type(host)
    
    # Prepare the result
    scan_duration = (datetime.datetime.now() - scan_start).total_seconds()
    
    network_info = {
        "scan_time": datetime.datetime.now().isoformat(),
        "scan_duration_seconds": scan_duration,
        "local_ip": local_ip,
        "network_range": network_range,
        "hosts": hosts
    }
    
    # Display summary
    print("\nScan Results:")
    print(f"Scan completed in {scan_duration:.2f} seconds")
    print(f"Found {len(hosts)} hosts on network {network_range}")
    
    # Group devices by type
    device_types = {}
    for host in hosts:
        device_type = host["device_type"]
        if device_type not in device_types:
            device_types[device_type] = []
        device_types[device_type].append(host)
    
    print("\nDevice Types Found:")
    for device_type, devices in device_types.items():
        print(f"  {device_type}: {len(devices)} devices")
    
    # Check if any Nest devices were found
    nest_devices = [h for h in hosts if "nest" in h["device_type"].lower()]
    if nest_devices:
        print("\nNest Devices Found:")
        for device in nest_devices:
            print(f"  {device['ip']} - {device['device_type']}")
    
    # Save result to file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"network_scan_{timestamp}.json"
    save_scan_result(network_info, filename)
    
    print(f"\nScan complete! Full results saved to {filename}")

if __name__ == "__main__":
    main()
"""
    
    def evaluate_fitness(self, source_code: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the fitness of the provided source code for this task.
        
        Args:
            source_code: The network scanner source code to evaluate
            
        Returns:
            A tuple containing (fitness_score, detailed_results)
        """
        results = {
            "syntax_valid": False,
            "runtime_successful": False,
            "execution_time": 0,
            "network_detection": 0,
            "device_discovery": 0,
            "port_scanning": 0,
            "device_identification": 0,
            "error_handling": 0,
            "resource_efficiency": 0,
            "user_interface": 0,
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
            
            # Check required functionality
            results.update(self._check_required_functionality(source_code))
            
            # We won't execute the network scanner directly for testing as it might affect the network
            # Instead, we'll simulate a successful runtime if the code passes checks
            if results["network_detection"] > 0 and results["device_discovery"] > 0:
                results["runtime_successful"] = 1.0
            
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
            "network_detection": 0.15,
            "device_discovery": 0.15,
            "port_scanning": 0.15,
            "device_identification": 0.15,
            "error_handling": 0.1,
            "resource_efficiency": 0.05,
            "user_interface": 0.05
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
            "network_detection": 0,
            "device_discovery": 0,
            "port_scanning": 0,
            "device_identification": 0,
            "error_handling": 0,
            "resource_efficiency": 0,
            "user_interface": 0
        }
        
        # Check for network detection functionality
        network_detection_patterns = [
            r'get_local_ip',
            r'subnet',
            r'network_range',
            r'gateway'
        ]
        if any(re.search(pattern, code, re.IGNORECASE) for pattern in network_detection_patterns):
            results["network_detection"] = 1.0
        
        # Check for device discovery functionality
        device_discovery_patterns = [
            r'ping',
            r'scan_network',
            r'discover',
            r'arp',
            r'nmap'
        ]
        discovery_score = sum(0.2 for pattern in device_discovery_patterns if re.search(pattern, code, re.IGNORECASE))
        results["device_discovery"] = min(1.0, discovery_score)
        
        # Check for port scanning functionality
        port_scanning_patterns = [
            r'scan_ports',
            r'socket\.connect',
            r'connect_ex',
            r'port.*open',
            r'service.*port'
        ]
        port_score = sum(0.2 for pattern in port_scanning_patterns if re.search(pattern, code, re.IGNORECASE))
        results["port_scanning"] = min(1.0, port_score)
        
        # Check for device identification
        identification_patterns = [
            r'identify_device',
            r'device_type',
            r'classify',
            r'nest',
            r'thermostat',
            r'IoT',
            r'mac_address'
        ]
        identification_score = sum(0.15 for pattern in identification_patterns if re.search(pattern, code, re.IGNORECASE))
        results["device_identification"] = min(1.0, identification_score)
        
        # Check for error handling
        error_handling_patterns = [
            r'try\s*:.*except',
            r'error',
            r'exception'
        ]
        if any(re.search(pattern, code, re.IGNORECASE | re.DOTALL) for pattern in error_handling_patterns):
            results["error_handling"] = 1.0
        
        # Check for resource efficiency
        resource_efficiency_patterns = [
            r'timeout',
            r'throttle',
            r'sleep',
            r'concurrent'
        ]
        resource_score = sum(0.25 for pattern in resource_efficiency_patterns if re.search(pattern, code, re.IGNORECASE))
        results["resource_efficiency"] = min(1.0, resource_score)
        
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
        
        return results
    
    def get_required_boundaries(self) -> Dict[str, Dict[str, Any]]:
        """
        Get the ethical boundaries required for this task.
        
        Returns:
            A dictionary mapping boundary names to their parameters
        """
        return {
            TrisolarisBoundary.NO_EVAL_EXEC: {},
            TrisolarisBoundary.MAX_EXECUTION_TIME: {"max_execution_time": 30.0},  # Network scans can take longer
            TrisolarisBoundary.MAX_MEMORY_USAGE: {"max_memory_usage": 200}        # Network scans might use more memory
        }
    
    def get_fitness_weights(self) -> Dict[str, float]:
        """
        Get the weights for different fitness components.
        
        Returns:
            A dictionary mapping fitness component names to their weights
        """
        return {
            "functionality": 0.7,  # Highest priority for functionality
            "efficiency": 0.2,     # Second priority is efficiency
            "alignment": 0.1       # Ethical alignment is still important
        }
    
    def get_allowed_imports(self) -> List[str]:
        """
        Get the list of allowed imports for this task.
        
        Returns:
            A list of allowed import module names
        """
        return [
            "os", "sys", "socket", "subprocess", "json", "datetime", "re", 
            "time", "logging", "collections", "ipaddress", "threading", "concurrent"
        ]
    
    def get_evolution_params(self) -> Dict[str, Any]:
        """
        Get recommended evolution parameters for this task.
        
        Returns:
            A dictionary of parameters for the evolution process
        """
        return {
            "population_size": 20,
            "num_generations": 15,
            "mutation_rate": 0.1,
            "crossover_rate": 0.7
        }
    
    def post_process(self, source_code: str) -> str:
        """
        Perform post-processing on evolved source code.
        
        Adds a shebang line and ensures the file has proper
        executable permissions.
        
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
        
        return source_code 