#!/usr/bin/env python3
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