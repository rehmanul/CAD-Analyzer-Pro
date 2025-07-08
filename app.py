"""
CAD Analyzer Pro - Main Application Entry Point
"""

import streamlit as st
from main_production_app import ProductionCADAnalyzer

def main():
    """Main entry point for Streamlit app"""
    app = ProductionCADAnalyzer()
    app.run()

if __name__ == "__main__":
    main()