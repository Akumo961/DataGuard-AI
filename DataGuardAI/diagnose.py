#!/usr/bin/env python3
"""
Diagnostic script for DataGuard AI server issues
"""

import socket
import subprocess
import sys
import time
import requests
from pathlib import Path


def check_port(port=8000):
    """Check if a port is available"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('localhost', port))
            return True, f"Port {port} is available"
        except OSError as e:
            return False, f"Port {port} is in use: {e}"


def check_api_file():
    """Check if API file exists and is valid"""
    api_file = Path("src/api/main.py")
    if not api_file.exists():
        return False, "API file not found at src/api/main.py"

    # Check if file has basic FastAPI structure
    content = api_file.read_text()
    if "FastAPI" not in content:
        return False, "API file doesn't contain FastAPI code"

    return True, "API file looks good"


def test_fastapi_import():
    """Test if FastAPI can be imported"""
    try:
        from fastapi import FastAPI
        return True, "FastAPI imports correctly"
    except ImportError as e:
        return False, f"FastAPI import failed: {e}"


def start_server_and_test():
    """Start the server and test if it's responding"""
    print("🚀 Starting server for testing...")

    # Start the server process
    process = subprocess.Popen(
        [sys.executable, "src/api/main.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Give it time to start
    time.sleep(5)

    # Test the connection
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        if response.status_code == 200:
            return True, "Server is running and responding correctly", process
        else:
            return False, f"Server responded with status {response.status_code}", process
    except requests.exceptions.ConnectionError:
        return False, "Server is not accepting connections", process
    except Exception as e:
        return False, f"Error testing server: {e}", process


def main():
    print("🔍 DataGuard AI Server Diagnostics")
    print("=" * 50)

    # Check 1: Port availability
    print("1. Checking port 8000...")
    port_available, port_msg = check_port()
    print(f"   {'✅' if port_available else '❌'} {port_msg}")

    # Check 2: API file
    print("2. Checking API file...")
    api_ok, api_msg = check_api_file()
    print(f"   {'✅' if api_ok else '❌'} {api_msg}")

    # Check 3: FastAPI import
    print("3. Checking dependencies...")
    import_ok, import_msg = test_fastapi_import()
    print(f"   {'✅' if import_ok else '❌'} {import_msg}")

    if not all([port_available, api_ok, import_ok]):
        print("\n❌ Prerequisites failed. Please fix above issues first.")
        return

    # Check 4: Start server and test
    print("4. Testing server startup...")
    success, message, process = start_server_and_test()
    print(f"   {'✅' if success else '❌'} {message}")

    # Print server output if it failed
    if not success:
        print("\n📋 Server output:")
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stdout:
                print("STDOUT:", stdout)
            if stderr:
                print("STDERR:", stderr)
        except:
            pass

    # Cleanup
    if process:
        process.terminate()

    print("\n" + "=" * 50)
    if success:
        print("✅ All diagnostics passed! Your server should work.")
        print("\n🎯 Try running: python src/api/main.py")
    else:
        print("❌ Diagnostics failed. See recommendations below.")


if __name__ == "__main__":
    main()