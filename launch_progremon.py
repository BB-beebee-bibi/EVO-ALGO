#!/usr/bin/env python3
"""
Progrémon Launcher
A simple script to launch the Progrémon CLI interface.
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from trisolaris.progremon import ProgrémonCLI

def main():
    """Launch Progrémon."""
    try:
        cli = ProgrémonCLI()
        cli.start()
    except KeyboardInterrupt:
        print("\nProgrémon terminated by user.")
    except Exception as e:
        print(f"\nError launching Progrémon: {str(e)}")
        input("\nPress Enter to exit...")

if __name__ == "__main__":
    main() 