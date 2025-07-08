
"""
CAD Analyzer Pro - Main Application Entry Point
Streamlit Cloud compatible entry point
"""

import streamlit as st
import sys
import os

# Add error handling for missing dependencies
try:
    from main_production_app import ProductionCADAnalyzer
except ImportError as e:
    st.error(f"Import error: {str(e)}")
    st.error("Please ensure all required packages are installed.")
    st.stop()

def main():
    """Main entry point for Streamlit app"""
    try:
        app = ProductionCADAnalyzer()
        app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please check the application logs for more details.")

if __name__ == "__main__":
    main()
