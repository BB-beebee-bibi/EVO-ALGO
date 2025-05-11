# 🚀 Progremon Enhanced - Quick Start Guide

This guide will help you get started with the enhanced Progremon system quickly and easily. Follow these steps to install, configure, and run your first evolution.

## 📋 Contents

1. [Prerequisites](#1-prerequisites)
2. [Installation](#2-installation)
3. [Basic Usage](#3-basic-usage)
4. [Example Tasks](#4-example-tasks)
5. [Troubleshooting](#5-troubleshooting)

## 1. Prerequisites

Before installing Progremon Enhanced, make sure you have:

- Python 3.8 or higher installed
- Git installed (for version control)
- Basic knowledge of command line operations
- Approximately 200MB of free disk space

## 2. Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/username/progremon-enhanced.git
cd progremon-enhanced
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

For Bluetooth functionality, install the platform-specific packages:

- Linux:
  ```bash
  pip install pybluez
  ```

- Windows/macOS:
  ```bash
  pip install bleak
  ```

### Step 3: Verify Installation

Run the test script to verify everything is installed correctly:

```bash
python run_enhanced.py --test
```

You should see all tests passing with a message:
```
🎉 All tests passed! The enhanced Progremon system is working correctly.
```

## 3. Basic Usage

### Interactive Mode

The easiest way to use Progremon is in interactive mode:

```bash
python run_enhanced.py
```

This will start the interactive interface where you can:

1. Enter a description of the program you want to evolve
2. Customize evolution settings if needed
3. Watch the evolution process
4. Get your evolved solution

### Command-line Mode

For automated or scripted usage, use the command-line mode:

```bash
python run_enhanced.py --task general --description "Create a function that calculates the factorial of a number"
```

Available options:
- `--task`: Task type (general, bluetooth_scan)
- `--description`: Natural language description of what to evolve
- `--output-dir`: Directory for storing evolved code

## 4. Example Tasks

### General Purpose Code

```bash
python run_enhanced.py --task general --description "Create a function that sorts a list of numbers using the quicksort algorithm"
```

### Bluetooth Scanner

```bash
python run_enhanced.py --task bluetooth_scan --description "Scan for nearby Bluetooth devices and display their names and signal strength"
```

### Custom Configuration

For more control over the evolution process, use interactive mode and customize the settings when prompted:

```bash
python run_enhanced.py
```

Then enter your task description and adjust settings like:
- Population size
- Number of generations
- Mutation rate
- Crossover rate
- Ethics level

## 5. Troubleshooting

### Common Issues

#### No Bluetooth Devices Found

If the evolved Bluetooth scanner doesn't find any devices:

1. Check that your device has Bluetooth enabled
2. Make sure you have the correct Bluetooth library installed for your platform
3. Try increasing the scan duration in the settings

#### Evolution Seems Stuck

If the evolution process seems to get stuck:

1. Try reducing the population size
2. Check if there are permission issues with file writing
3. Make sure you have enough available memory

#### Missing Dependencies

If you see import errors:

```
pip install -r requirements.txt
```

If a specific module is still missing, install it directly:

```
pip install module_name
```

### Getting Help

For more detailed help, run:

```bash
python run_enhanced.py --help
```

## 🎉 Next Steps

- Check out the full documentation in the `ENHANCED_README.md` file
- Explore the evolved code in the `evolved_output` directory
- Try evolving different types of programs
- Examine the templates in the `guidance` directory to see how they work

Happy evolving! 🧬