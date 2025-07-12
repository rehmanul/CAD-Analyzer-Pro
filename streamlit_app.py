
"""
CAD Analyzer Pro - Main Streamlit Application
Fixed entry point for Render deployment
"""

import streamlit as st
import sys
import os

# Set page config first
try:
    st.set_page_config(
        page_title="CAD Analyzer Pro - Ultimate Edition",
        page_icon="🏗️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
except:
    pass  # Already configured

def main():
    """Main application entry point"""
    try:
        # Add utils to path
        if 'utils' not in sys.path:
            sys.path.append('utils')
        
        # Import and run the advanced app
        from utils.advanced_streamlit_app import AdvancedStreamlitApp
        
        # Create and run the application
        app = AdvancedStreamlitApp()
        app.run()
        
    except ImportError as e:
        st.error(f"Import error: {str(e)}")
        st.error("Loading fallback interface...")
        
        # Fallback simple interface
        st.title("🏗️ CAD Analyzer Pro")
        st.write("**Status**: Loading application components...")
        
        # Show debug info
        with st.expander("Debug Information"):
            st.write(f"Python path: {sys.path}")
            st.write(f"Current directory: {os.getcwd()}")
            st.write(f"Available files: {os.listdir('.')}")
            if os.path.exists('utils'):
                st.write(f"Utils files: {os.listdir('utils')}")
        
        st.info("Please check the logs for more details.")
        
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.title("🏗️ CAD Analyzer Pro - Safe Mode")
        st.write("Running in safe mode due to initialization error.")

if __name__ == "__main__":
    main()
