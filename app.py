
"""
CAD Analyzer Pro - Main Application Entry Point
Streamlit Cloud compatible entry point
Updated: Fresh deployment with psutil fix
"""

import streamlit as st
import sys
import os

def main():
    """Main entry point for Streamlit app"""
    try:
        # Import the main application
        from main_production_app import ProductionCADAnalyzer
        
        # Create and run the application
        app = ProductionCADAnalyzer()
        app.run()
        
    except ImportError as e:
        st.error(f"Import error: {str(e)}")
        st.error("Please ensure all required packages are installed.")
        
        # Show debug information
        st.subheader("Debug Information")
        st.write(f"Python path: {sys.path}")
        st.write(f"Current directory: {os.getcwd()}")
        st.write(f"Files in current directory: {os.listdir('.')}")
        
        # Try to show the actual error from the main file
        try:
            with open('main_production_app.py', 'r') as f:
                content = f.read()
                if 'class ProductionCADAnalyzer' in content:
                    st.success("ProductionCADAnalyzer class found in file")
                else:
                    st.error("ProductionCADAnalyzer class not found in file")
        except Exception as file_err:
            st.error(f"Could not read main_production_app.py: {file_err}")
        
        st.stop()
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please check the application logs for more details.")
        
        # Show traceback for debugging
        import traceback
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
