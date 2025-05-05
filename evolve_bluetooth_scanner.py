#!/usr/bin/env python3
"""
Evolution script for a Bluetooth scanner with enhanced vulnerability detection.

This script uses the TRISOLARIS evolutionary framework to evolve a Bluetooth scanner
that can detect and analyze security vulnerabilities in IoT devices.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bluetooth_evolution.log")
    ]
)
logger = logging.getLogger(__name__)

def create_custom_task_template() -> str:
    """
    Create a custom template for the Bluetooth scanner task with enhanced vulnerability detection.
    
    Returns:
        A string containing the template source code
    """
    # Base template from the task
    from trisolaris.tasks import BluetoothScannerTask
    task = BluetoothScannerTask()
    template = task.get_template()
    
    # Enhance template with vulnerability detection capabilities
    enhanced_template = template.replace(
        "# Known IoT device manufacturers and their identifiers",
        """# Path to CVE database file
CVE_DATABASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                'trisolaris/tasks/bluetooth_cve_database.json')

# Known IoT device manufacturers and their identifiers"""
    )
    
    # Add CVE database loading function
    after_class_definition = enhanced_template.find("class BluetoothScanner:")
    if after_class_definition != -1:
        insert_position = enhanced_template.find("def __init__", after_class_definition)
        
        cve_loader_function = """    def load_cve_database(self) -> Dict[str, Any]:
        '''
        Load the CVE database for Bluetooth vulnerabilities.
        
        Returns:
            Dictionary containing known Bluetooth vulnerabilities
        '''
        try:
            if os.path.exists(CVE_DATABASE_PATH):
                with open(CVE_DATABASE_PATH, 'r') as f:
                    return json.load(f)
            else:
                print(f"Warning: CVE database file not found at {CVE_DATABASE_PATH}")
                print("Using built-in vulnerability database...")
                
                # Return a minimal built-in database
                return {
                    "CVE-2017-0785": {
                        "name": "BlueBorne SDP Information Leak",
                        "description": "The BlueBorne vulnerability in SDP servers allows an attacker to get information about memory layout.",
                        "affected_versions": ["Bluetooth 2.1 - 5.0"],
                        "severity": "HIGH",
                        "recommendation": "Update device firmware"
                    },
                    "DEFAULT_PIN": {
                        "name": "Default PIN",
                        "description": "Device uses a default PIN code (like 0000, 1234) which is easily guessable.",
                        "severity": "HIGH",
                        "recommendation": "Change default PIN code"
                    },
                    "PLAINTEXT_DATA": {
                        "name": "Plaintext Data Transmission",
                        "description": "Device transmits sensitive data without encryption.",
                        "severity": "HIGH",
                        "recommendation": "Replace with device that implements encryption"
                    }
                }
        except Exception as e:
            print(f"Error loading CVE database: {e}")
            return {}
    
    def check_device_vulnerabilities(self, device: Dict[str, Any]) -> List[Dict[str, Any]]:
        '''
        Check a device for known vulnerabilities based on its characteristics.
        
        Args:
            device: Device information dictionary
            
        Returns:
            List of vulnerability dictionaries
        '''
        vulnerabilities = []
        cve_database = self.load_cve_database()
        
        # Check for known CVEs based on device characteristics
        device_type = device.get("type", "")
        device_class = device.get("device_class", "")
        device_name = device.get("name", "").upper()
        
        # Check for default PINs in IoT devices
        if any(identifier in device_name for identifier in ["LOCK", "DOOR", "SMART", "IOT"]):
            vulnerabilities.append(cve_database.get("DEFAULT_PIN", {
                "cve_id": "DEFAULT_PIN",
                "name": "Default PIN",
                "description": "Device may use default PIN codes (0000, 1234)",
                "severity": "HIGH",
                "recommendation": "Change default PIN code"
            }))
        
        # Check for BlueBorne vulnerability in older devices
        if device_type == "Classic Bluetooth" or "BR/EDR" in device_type:
            vulnerabilities.append(cve_database.get("CVE-2017-0785", {
                "cve_id": "CVE-2017-0785",
                "name": "BlueBorne SDP Information Leak",
                "description": "Device may be vulnerable to BlueBorne attacks",
                "severity": "HIGH",
                "recommendation": "Update device firmware"
            }))
        
        # Check for plaintext data in IoT devices
        if "LATCH" in device_name or "LOCK" in device_name:
            vulnerabilities.append(cve_database.get("PLAINTEXT_DATA", {
                "cve_id": "PLAINTEXT_DATA",
                "name": "Plaintext Data Transmission",
                "description": "Device may transmit sensitive data without encryption",
                "severity": "HIGH",
                "recommendation": "Update firmware or replace device"
            }))
        
        # Check for KNOB attack vulnerability
        if device_type == "Classic Bluetooth" and "KNOB" not in device_name:
            vulnerabilities.append(cve_database.get("CVE-2019-9506", {
                "cve_id": "CVE-2019-9506",
                "name": "KNOB Attack",
                "description": "Device may be vulnerable to the Key Negotiation of Bluetooth attack",
                "severity": "HIGH",
                "recommendation": "Apply vendor security patches"
            }))
            
        # Check for BIAS attack in older devices
        if device_type == "Classic Bluetooth" and "4.2" not in device_name:
            vulnerabilities.append(cve_database.get("CVE-2020-10135", {
                "cve_id": "CVE-2020-10135",
                "name": "BIAS Attack",
                "description": "Device may be vulnerable to Bluetooth Impersonation Attacks",
                "severity": "HIGH",
                "recommendation": "Apply vendor security patches if available"
            }))
        
        return vulnerabilities
    
    def generate_security_report(self, output_file: str = None) -> Dict[str, Any]:
        '''
        Generate a comprehensive security report for all scanned devices.
        
        Args:
            output_file: Optional file to save the report
            
        Returns:
            Dictionary containing the security report
        '''
        if not self.devices:
            return {"error": "No devices scanned yet"}
        
        report = {
            "scan_time": self.scan_time.isoformat(),
            "devices": self.devices,
            "vulnerabilities": {},
            "summary": {
                "total_devices": len(self.devices),
                "vulnerable_devices": 0,
                "total_vulnerabilities": 0,
                "high_severity": 0,
                "medium_severity": 0,
                "low_severity": 0
            },
            "recommendations": []
        }
        
        all_recommendations = set()
        
        # Check vulnerabilities for each device
        for addr, device in self.devices.items():
            vulnerabilities = self.check_device_vulnerabilities(device)
            
            if vulnerabilities:
                report["vulnerabilities"][addr] = vulnerabilities
                report["summary"]["vulnerable_devices"] += 1
                report["summary"]["total_vulnerabilities"] += len(vulnerabilities)
                
                # Count by severity
                for vuln in vulnerabilities:
                    severity = vuln.get("severity", "").upper()
                    if severity == "HIGH" or severity == "CRITICAL":
                        report["summary"]["high_severity"] += 1
                    elif severity == "MEDIUM":
                        report["summary"]["medium_severity"] += 1
                    else:
                        report["summary"]["low_severity"] += 1
                    
                    # Add recommendation
                    if "recommendation" in vuln:
                        all_recommendations.add(vuln["recommendation"])
        
        # Add recommendations to the report
        report["recommendations"] = list(all_recommendations)
        
        # Save report if requested
        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
            print(f"Security report saved to {output_file}")
        
        return report

"""
        
        enhanced_template = enhanced_template[:insert_position] + cve_loader_function + enhanced_template[insert_position:]
    
    # Add vulnerability scanning to the scan_for_devices method
    scan_for_devices_end = enhanced_template.find("return self.devices", enhanced_template.find("def scan_for_devices"))
    if scan_for_devices_end != -1:
        vulnerability_check = """
        # Check for security vulnerabilities in each device
        for addr, device in self.devices.items():
            # Identify potential vulnerabilities based on device characteristics
            vulnerabilities = self.check_device_vulnerabilities(device)
            device["possible_vulnerabilities"] = vulnerabilities
        
        """
        enhanced_template = enhanced_template[:scan_for_devices_end] + vulnerability_check + enhanced_template[scan_for_devices_end:]
    
    # Enhance main function with additional options and reports
    main_function_end = enhanced_template.find("if __name__ == \"__main__\":")
    if main_function_end != -1:
        enhanced_main_function = """def main():
    '''Main function to run the Bluetooth IoT scanner.'''
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Bluetooth IoT Scanner & Security Analyzer")
    parser.add_argument("--scan", action="store_true", help="Scan for Bluetooth devices")
    parser.add_argument("--duration", type=int, default=15, help="Scan duration in seconds (default: 15)")
    parser.add_argument("--check-vulnerabilities", action="store_true", help="Check for vulnerabilities in detected devices")
    parser.add_argument("--device", help="Check vulnerabilities for a specific device address")
    parser.add_argument("--generate-report", action="store_true", help="Generate a security report")
    parser.add_argument("--output", help="Output file for scan results or security report")
    parser.add_argument("--no-scan", action="store_true", help="Skip scanning (use with --check-vulnerabilities)")
    
    args = parser.parse_args()
    
    print("Bluetooth IoT Scanner & Security Analyzer")
    print("=========================================")
    
    # Create scanner
    scanner = BluetoothScanner()
    
    # Check dependencies
    if not scanner.has_bluetooth:
        print("\\nPython Bluetooth libraries not found.")
        print("The scanner will attempt to use system tools, but functionality may be limited.")
    
    if not scanner.check_system_dependencies():
        print("\\nWarning: Some system dependencies are missing.")
        print("The scanner will continue with limited functionality.")
    
    # Perform the scan if requested
    if args.scan or not args.no_scan:
        scan_duration = args.duration
        print(f"\\nScanning for Bluetooth devices for {scan_duration} seconds...")
        
        # Perform the scan
        devices = scanner.scan_for_devices(duration=scan_duration)
        
        if not devices:
            print("\\nNo Bluetooth devices found.")
            return
        
        # Print discovered devices
        print(f"\\nDiscovered {len(devices)} Bluetooth devices:")
        
        for addr, device in devices.items():
            print(f"\\nDevice: {device['name']} ({addr})")
            print(f"  Type: {device['type']}")
            print(f"  Class: {device['device_class']}")
            print(f"  Manufacturer: {device['manufacturer']}")
            
            # Print vulnerabilities
            vulnerabilities = device.get("possible_vulnerabilities", [])
            if vulnerabilities:
                print(f"  Potential Security Issues: {len(vulnerabilities)}")
                for i, vuln in enumerate(vulnerabilities, 1):
                    print(f"    {i}. {vuln['name']} (Severity: {vuln['severity']})")
                    print(f"       Recommendation: {vuln['recommendation']}")
            else:
                print("  No obvious security issues detected")
    
    # Check vulnerabilities if requested
    if args.check_vulnerabilities:
        if args.device:
            # Check vulnerabilities for a specific device
            if args.device in scanner.devices:
                device = scanner.devices[args.device]
                print(f"\\nChecking vulnerabilities for {device['name']} ({args.device})...")
                vulnerabilities = scanner.check_device_vulnerabilities(device)
                
                if vulnerabilities:
                    print(f"Found {len(vulnerabilities)} potential vulnerabilities:")
                    for i, vuln in enumerate(vulnerabilities, 1):
                        print(f"  {i}. {vuln['name']} (Severity: {vuln['severity']})")
                        print(f"     Description: {vuln['description']}")
                        print(f"     Recommendation: {vuln['recommendation']}")
                else:
                    print("No obvious vulnerabilities detected.")
            else:
                print(f"Device {args.device} not found in scan results.")
        else:
            # Check vulnerabilities for all devices
            print("\\nChecking vulnerabilities for all devices...")
            vulnerable_devices = 0
            total_vulnerabilities = 0
            
            for addr, device in scanner.devices.items():
                vulnerabilities = scanner.check_device_vulnerabilities(device)
                scanner.devices[addr]["possible_vulnerabilities"] = vulnerabilities
                
                if vulnerabilities:
                    vulnerable_devices += 1
                    total_vulnerabilities += len(vulnerabilities)
            
            print(f"Found {total_vulnerabilities} potential vulnerabilities in {vulnerable_devices} devices.")
    
    # Generate security report if requested
    if args.generate_report:
        output_file = args.output if args.output else "bluetooth_security_report.json"
        print("\\nGenerating security report...")
        report = scanner.generate_security_report(output_file)
        
        # Print report summary
        print("\\nSecurity Report Summary:")
        print(f"  Total devices: {report['summary']['total_devices']}")
        print(f"  Vulnerable devices: {report['summary']['vulnerable_devices']}")
        print(f"  Total vulnerabilities: {report['summary']['total_vulnerabilities']}")
        print(f"  High severity issues: {report['summary']['high_severity']}")
        print(f"  Medium severity issues: {report['summary']['medium_severity']}")
        print(f"  Low severity issues: {report['summary']['low_severity']}")
        
        print("\\nRecommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    # Save scan results if not already saved
    if args.scan and args.output and not args.generate_report:
        scanner.save_scan_result(args.output)
    elif args.scan and not args.generate_report:
        scanner.save_scan_result()
    
    print("\\nScan complete!")
"""
        enhanced_template = enhanced_template[:main_function_end] + enhanced_main_function + enhanced_template[main_function_end:]
    
    return enhanced_template

def main():
    """Main function to evolve a Bluetooth scanner with vulnerability detection."""
    parser = argparse.ArgumentParser(
        description="Evolve a Bluetooth scanner with enhanced vulnerability detection"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="outputs",
        help="Directory to save evolved code (default: outputs)"
    )
    parser.add_argument(
        "--pop-size",
        "-p",
        type=int,
        default=50,
        help="Population size (default: 50)"
    )
    parser.add_argument(
        "--gens",
        "-g",
        type=int,
        default=20,
        help="Number of generations (default: 20)"
    )
    parser.add_argument(
        "--custom-template",
        action="store_true",
        help="Use a custom enhanced template instead of the default"
    )
    parser.add_argument(
        "--save-template",
        action="store_true",
        help="Save the custom template to a file"
    )
    
    args = parser.parse_args()
    
    if args.custom_template:
        # Create a custom enhanced template
        template = create_custom_task_template()
        
        if args.save_template:
            # Save template to a file
            template_file = "enhanced_bluetooth_scanner_template.py"
            with open(template_file, "w", encoding="utf-8") as f:
                f.write(template)
            logger.info(f"Custom template saved to {template_file}")
        
        # Create a temporary template file for the task
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(suffix='.py', mode='w', delete=False) as f:
            temp_file = f.name
            f.write(template)
        
        try:
            # Import and run the trisolaris task runner
            from trisolaris.task_runner import main as run_task
            
            # Construct arguments for the task runner
            task_args = [
                "bluetooth_scanner",
                "--template", temp_file,
                "--output-dir", args.output_dir,
                "--pop-size", str(args.pop_size),
                "--gens", str(args.gens),
                "--ethics-level", "full",
                "--resource-monitoring",
                "--use-islands"
            ]
            
            logger.info(f"Running evolution with custom template and {args.pop_size} population size for {args.gens} generations")
            
            # Set sys.argv for the task runner
            old_argv = sys.argv
            sys.argv = ["trisolaris_task_runner.py"] + task_args
            
            # Run the task
            run_task()
            
            # Restore sys.argv
            sys.argv = old_argv
            
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_file)
            except:
                pass
    else:
        # Just use the task runner directly
        from trisolaris.task_runner import main as run_task
        
        task_args = [
            "bluetooth_scanner",
            "--output-dir", args.output_dir,
            "--pop-size", str(args.pop_size),
            "--gens", str(args.gens),
            "--ethics-level", "full", 
            "--resource-monitoring",
            "--use-islands"
        ]
        
        logger.info(f"Running evolution with default template and {args.pop_size} population size for {args.gens} generations")
        
        # Set sys.argv for the task runner
        old_argv = sys.argv
        sys.argv = ["trisolaris_task_runner.py"] + task_args
        
        # Run the task
        run_task()
        
        # Restore sys.argv
        sys.argv = old_argv
    
    logger.info("Evolution complete!")


if __name__ == "__main__":
    main() 