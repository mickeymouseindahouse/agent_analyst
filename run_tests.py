#!/usr/bin/env python3
import os
import sys
import subprocess

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Read the Nebius API key from file
try:
    with open("nebius.key", "r") as key_file:
        nebius_api_key = key_file.read().strip()
    # Set the API key as an environment variable
    os.environ["NEBIUS_API_KEY"] = nebius_api_key
    print("Successfully loaded Nebius API key from nebius.key")
except FileNotFoundError:
    print("Warning: nebius.key file not found. Please create this file with your API key.")
    sys.exit(1)
except Exception as e:
    print(f"Error reading API key: {e}")
    sys.exit(1)

# Run the test file
subprocess.run(["streamlit", "run", "test_agents.py"])
