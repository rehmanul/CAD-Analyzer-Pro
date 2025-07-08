#!/usr/bin/env python3
"""
Streamlit Share Deployment Entry Point
Main entry point for Streamlit Share deployment
"""

import sys
import os

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main production app
from main_production_app import ProductionCADAnalyzer

# Create and run the application
if __name__ == "__main__":
    app = ProductionCADAnalyzer()
    app.run()
else:
    # For Streamlit Share deployment
    app = ProductionCADAnalyzer()
    app.run()