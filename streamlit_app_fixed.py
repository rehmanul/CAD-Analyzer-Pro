"""
CAD Analyzer Pro - Python 3.13 Compatible Entry Point
Fixed PostgreSQL compatibility and graceful fallback handling
"""

import streamlit as st
import sys
import os
import traceback
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if critical dependencies are available with Python 3.13 compatibility"""
    missing_deps = []
    
    # Core dependencies
    try:
        import pandas
        import numpy
        import plotly
        import shapely
        import ezdxf
        import matplotlib
        import streamlit
    except ImportError as e:
        missing_deps.append(f"Core dependency: {e}")
    
    # PostgreSQL with graceful fallback
    try:
        import psycopg2
        postgres_available = True
        logger.info("PostgreSQL support available")
    except ImportError as e:
        postgres_available = False
        logger.warning(f"PostgreSQL not available: {e}. Will use SQLite fallback.")
    
    # Optional dependencies
    try:
        import cv2
        import scipy
        import sklearn
        import networkx
    except ImportError as e:
        logger.warning(f"Optional dependency missing: {e}")
    
    if missing_deps:
        st.error("Missing critical dependencies:")
        for dep in missing_deps:
            st.error(f"• {dep}")
        st.error("Please ensure all required packages are installed.")
        st.stop()
    
    return postgres_available

def main():
    """Main entry point with Python 3.13 compatibility"""
    try:
        # Check dependencies
        postgres_available = check_dependencies()
        
        # Set PostgreSQL availability in environment
        os.environ['POSTGRES_AVAILABLE'] = str(postgres_available)
        
        # Import and run the main application
        from main_production_app import ProductionCADAnalyzer
        
        app = ProductionCADAnalyzer()
        
        # Override PostgreSQL setting based on availability
        if not postgres_available:
            app.use_postgres = False
            st.sidebar.info("Using SQLite database (PostgreSQL not available)")
        
        app.run()
        
    except Exception as e:
        st.error(f"Application startup failed: {str(e)}")
        st.error("Debug Information")
        st.code(f"Python version: {sys.version}")
        st.code(f"Python path: {sys.path}")
        st.code(f"Current directory: {os.getcwd()}")
        st.code(f"Files in current directory: {os.listdir('.')}")
        
        # Show traceback for debugging
        st.error("Full traceback:")
        st.code(traceback.format_exc())
        
        # Try to find the main class
        try:
            from main_production_app import ProductionCADAnalyzer
            st.success("ProductionCADAnalyzer class found in file")
        except Exception as import_error:
            st.error(f"Import error: {import_error}")

if __name__ == "__main__":
    main()