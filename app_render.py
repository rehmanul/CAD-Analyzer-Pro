#!/usr/bin/env python3
"""
CAD Analyzer Pro - Render Deployment Entry Point
Optimized for Render.com deployment with enhanced performance
"""

import os
import sys
import streamlit as st

# Render-specific configuration
os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
os.environ['STREAMLIT_SERVER_PORT'] = os.environ.get('PORT', '10000')
os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

# Set up paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def configure_render_environment():
    """Configure environment for Render deployment"""
    
    # Set Streamlit configuration
    st.set_page_config(
        page_title="CAD Analyzer Pro",
        page_icon="🏨",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': None,
            'Report a bug': None,
            'About': "# CAD Analyzer Pro\nProfessional Floor Plan Analysis Tool"
        }
    )
    
    # Performance optimizations for Render
    if 'render_optimized' not in st.session_state:
        st.session_state.render_optimized = True
        st.session_state.max_file_size = 200  # MB
        st.session_state.cache_enabled = True

def main():
    """Main entry point for Render deployment"""
    
    # Configure environment
    configure_render_environment()
    
    # Import and run main application
    try:
        from main_production_app import ProductionCADAnalyzer
        
        # Initialize and run the application
        app = ProductionCADAnalyzer()
        app.run()
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.error("Please check the application logs for more details.")
        
        # Show debug information in development
        if os.environ.get('RENDER_ENV') != 'production':
            st.exception(e)

if __name__ == "__main__":
    main()