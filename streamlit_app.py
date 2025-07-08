
#!/usr/bin/env python3
"""
Streamlit Share Deployment Entry Point
Main entry point for Streamlit Share deployment
"""

import sys
import os
import streamlit as st

# Add the current directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_dependencies():
    """Check if critical dependencies are available"""
    missing_deps = []
    
    try:
        import psutil
    except ImportError:
        missing_deps.append("psutil")
    
    try:
        import streamlit
    except ImportError:
        missing_deps.append("streamlit")
    
    return missing_deps

def main():
    """Main entry point"""
    # Check dependencies first
    missing_deps = check_dependencies()
    
    if missing_deps:
        st.error(f"Missing dependencies: {', '.join(missing_deps)}")
        st.error("Please ensure all required packages are installed.")
        
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
        
        st.stop()
        return
    
    try:
        # Import and run the main production app
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
else:
    # For Streamlit Share deployment
    main()
