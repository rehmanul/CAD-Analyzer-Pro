#!/usr/bin/env python3
"""
CAD Analyzer Pro - Universal Deployment Entry Point
Compatible with Streamlit Share, Render, and Replit
Updated: 2025-01-08 - Production-ready with Render optimization
"""

import sys
import os
import streamlit as st

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure for Render deployment
if os.environ.get('RENDER'):
    # Render-specific configuration
    os.environ.setdefault('STREAMLIT_SERVER_HEADLESS', 'true')
    os.environ.setdefault('STREAMLIT_SERVER_PORT', os.environ.get('PORT', '10000'))
    os.environ.setdefault('STREAMLIT_SERVER_ADDRESS', '0.0.0.0')
    os.environ.setdefault('STREAMLIT_BROWSER_GATHER_USAGE_STATS', 'false')

def check_dependencies():
    """Check if critical dependencies are available"""
    missing_deps = []

    # Check for psutil with graceful fallback
    try:
        import psutil
        os.environ['PSUTIL_AVAILABLE'] = 'true'
    except ImportError:
        os.environ['PSUTIL_AVAILABLE'] = 'false'
        # Don't treat psutil as a critical dependency
        pass

    # Check critical dependencies
    critical_deps = ['streamlit', 'plotly', 'pandas', 'numpy']
    for dep in critical_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)

    return missing_deps

def main():
    """Main entry point"""
    # Check dependencies first
    missing_deps = check_dependencies()

    if missing_deps:
        st.error(f"Missing critical dependencies: {missing_deps}")
        st.info("Please install the required packages and restart the application.")
        
        # Show debug information
        st.subheader("Debug Information")
        st.write(f"Python path: {sys.path}")
        st.write(f"Current directory: {os.getcwd()}")
        st.write(f"Files in current directory: {os.listdir('.')}")

        # Show requirements.txt content
        try:
            with open('requirements.txt', 'r') as f:
                st.subheader("Requirements.txt content:")
                st.code(f.read())
        except:
            st.error("Could not read requirements.txt")
        
        return

    # Show warning for non-critical dependencies like psutil
    if missing_deps and "psutil" in missing_deps:
        st.warning("⚠️ psutil not available - memory monitoring disabled")
        st.info("The app will continue without memory monitoring features.")

    # Import and run the main production app
    try:
        from main_production_app import ProductionCADAnalyzer

        # Create and run the application
        app = ProductionCADAnalyzer()
        app.run()

    except ImportError:
        # Fallback to enhanced app
        try:
            from enhanced_professional_app import main as enhanced_main
            enhanced_main()
        except ImportError:
            st.error("Unable to load application modules. Please check the installation.")
            return

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please check the application logs for more details.")

        # Show traceback for debugging
        import traceback
        st.text(traceback.format_exc())

if __name__ == "__main__":
    main()
else:
    # For Streamlit Share deployment
    main()