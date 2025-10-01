#!/usr/bin/env python3
"""
Setup script for DataGuard AI
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd):
    """Run shell command and check result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"Error running: {cmd}")
        sys.exit(1)


def setup_project():
    """Setup the complete DataGuard AI project"""

    print("🚀 Setting up DataGuard AI...")

    # Create directory structure
    directories = [
        "data/raw",
        "data/processed",
        "notebooks",
        "src/models",
        "src/utils",
        "src/api",
        "models",
        "results/visualizations"
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

    # Create __init__.py files
    init_files = [
        "src/__init__.py",
        "src/models/__init__.py",
        "src/utils/__init__.py",
        "src/api/__init__.py"
    ]

    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('"""Package initialization"""\n')
        print(f"Created: {init_file}")

    # Install requirements
    print("Installing dependencies...")
    run_command("pip install -r requirements.txt")

    # Download spaCy model
    print("Downloading spaCy model...")
    run_command("python -m spacy download en_core_web_sm")

    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Add your training data to data/raw/")
    print("2. Run: python src/train.py --model all")
    print("3. Start API: python src/api/main.py")
    print("4. Launch UI: python app.py")


if __name__ == "__main__":
    setup_project()