#!/usr/bin/env python3
"""
Launcher script for DataGuard AI API - fixes path issues
"""

import sys
import os
from pathlib import Path


def setup_environment():
    """Setup Python path and environment"""
    # Get the project root directory
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))

    print(f"🏠 Project root: {project_root}")
    print(f"🐍 Python path: {sys.path[0]}")

    # Verify we can import the modules
    try:
        from src.api.main import app
        print("✅ All imports successful!")
        return app
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Try running: python setup.py")
        sys.exit(1)


def main():
    """Main launcher function"""
    print("🚀 DataGuard AI API Launcher")
    print("=" * 40)

    # Setup environment
    app = setup_environment()

    # Start the server
    import uvicorn
    print("\n🎯 Starting server...")
    print("📍 Local URL: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print("⏹️  Press Ctrl+C to stop the server\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()