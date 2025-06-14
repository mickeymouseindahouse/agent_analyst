#!/usr/bin/env python3
"""
Main entry point for Streamlit Cloud deployment.
This file is automatically detected by Streamlit Cloud.
"""
import os
import sys

# Add the current directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import necessary modules
from app.app import *

# This file will be automatically run by Streamlit Cloud
