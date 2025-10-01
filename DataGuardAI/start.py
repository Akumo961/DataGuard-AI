#!/usr/bin/env python3
"""
Quick start script for DataGuard AI
"""

import os
import sys
import subprocess
import time
from pathlib import Path


def check_requirements():
    """Check if all requirements are installed"""
    try:
        import torch
        import transformers
        import fastapi
        import gradio
        print("✅ All major dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False


def start_api():
    """Start the FastAPI server"""
    print("🚀 Starting DataGuard AI API...")
    try:
        # Check if API file exists
        if not Path("src/api/main.py").exists():
            print("❌ API file not found. Run setup.py first.")
            return False

        # Start the API server
        process = subprocess.Popen([
            sys.executable, "src/api/main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print("✅ API server started on http://localhost:8000")
        print("   Docs: http://localhost:8000/docs")
        return process

    except Exception as e:
        print(f"❌ Failed to start API: {e}")
        return None


def start_ui():
    """Start the Gradio UI"""
    print("🎨 Starting DataGuard AI Web Interface...")
    try:
        if not Path("app.py").exists():
            print("❌ app.py not found")
            return None

        # Give API time to start
        time.sleep(2)

        process = subprocess.Popen([
            sys.executable, "app.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        print("✅ Web interface started")
        print("   URL: http://localhost:7860")
        return process

    except Exception as e:
        print(f"❌ Failed to start UI: {e}")
        return None


def main():
    """Main startup function"""
    print("🛡️ DataGuard AI - Quick Start")
    print("=" * 50)

    # Check if setup is complete
    if not Path("src/api/main.py").exists():
        print("❌ Project not set up. Please run: python setup.py")
        return

    if not check_requirements():
        print("❌ Please install requirements first: pip install -r requirements.txt")
        return

    # Start services
    api_process = start_api()
    if not api_process:
        return

    ui_process = start_ui()

    print("\n🎯 Services running:")
    if api_process:
        print("   📡 API: http://localhost:8000")
    if ui_process:
        print("   🎨 UI:  http://localhost:7860")

    print("\n⏹️  Press Ctrl+C to stop all services")

    try:
        # Keep running until interrupted
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        if api_process:
            api_process.terminate()
        if ui_process:
            ui_process.terminate()
        print("✅ All services stopped")


if __name__ == "__main__":
    main()