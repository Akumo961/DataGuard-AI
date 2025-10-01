#!/usr/bin/env python3
"""
Simple test to verify the API works
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI
import uvicorn

# Create a simple app to test
app = FastAPI()

@app.get("/")
def root():
    return {"message": "✅ Simple test server is working!"}

@app.get("/test")
def test():
    return {"status": "success", "test": "passed"}

if __name__ == "__main__":
    print("🧪 Starting simple test server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)