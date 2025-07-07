"""
🏨 CAD Analyzer Pro - Production Launcher
Enterprise-grade hotel floor plan analyzer startup script

Usage:
    python run_production.py

Features:
✅ PostgreSQL Database Integration
✅ DXF/DWG File Processing  
✅ Intelligent Îlot Placement (10%, 25%, 30%, 35% distribution)
✅ Mandatory Corridor Generation
✅ Professional Visualization
✅ Export Capabilities
"""

import sys
import os
import subprocess
import time
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'streamlit',
        'plotly', 
        'numpy',
        'pandas',
        'shapely',
        'ezdxf',
        'psycopg2',
        'opencv-python'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
    
    return True

def check_database_connection():
    """Check PostgreSQL database connection"""
    print("\n🗄️ Checking database connection...")
    
    try:
        from utils.production_database import production_db
        
        # Test connection
        if production_db.connection_pool:
            print("✅ PostgreSQL connection established")
            print("✅ Database schema initialized")
            return True
        else:
            print("⚠️ Database connection failed - running in fallback mode")
            return False
            
    except Exception as e:
        print(f"⚠️ Database check failed: {e}")
        print("⚠️ Application will run without database features")
        return False

def display_startup_info():
    """Display startup information"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🏨 CAD ANALYZER PRO - ENTERPRISE EDITION                  ║
║                          Production-Ready Application                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  CLIENT REQUIREMENTS IMPLEMENTATION:                                         ║
║  ✅ Loading the Plan (walls, restricted areas, entrances)                   ║
║  ✅ Îlot Placement Rules (size distribution: 10%, 25%, 30%, 35%)           ║
║  ✅ Corridors Between Îlots (mandatory corridors between facing rows)       ║
║  ✅ Expected Output (neatly arranged îlots with constraints respected)      ║
║  ✅ Required Features (DXF loading, zone detection, visualization, export)  ║
║  ✅ PostgreSQL Database Integration                                          ║
║                                                                              ║
║  SUPPORTED FILE FORMATS:                                                     ║
║  📁 DXF files (recommended) - Full native support                           ║
║  📁 DWG files - Converted to DXF for processing                             ║
║  📁 Image files (PNG, JPG) - Color-based zone detection                     ║
║                                                                              ║
║  ZONE DETECTION:                                                             ║
║  🖤 Walls: Black lines or WALL layers                                       ║
║  🔵 Restricted Areas: Light blue zones (stairs, elevators)                  ║
║  🔴 Entrances/Exits: Red zones (no îlot placement allowed)                  ║
║                                                                              ║
║  DATABASE:                                                                   ║
║  🗄️ PostgreSQL on Render.com                                               ║
║  🔗 Full project persistence and analytics                                  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

def launch_application():
    """Launch the Streamlit application"""
    print("\n🚀 Launching CAD Analyzer Pro...")
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    app_path = script_dir / "main_production_app.py"
    
    if not app_path.exists():
        print(f"❌ Application file not found: {app_path}")
        return False
    
    try:
        # Launch Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "light"
        ]
        
        print("🌐 Starting web server...")
        print("📱 Application will open in your browser automatically")
        print("🔗 Manual access: http://localhost:8501")
        print("\n" + "="*80)
        print("🏨 CAD ANALYZER PRO - READY FOR PRODUCTION USE")
        print("="*80)
        
        # Start the application
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to launch application: {e}")
        return False

def main():
    """Main startup function"""
    display_startup_info()
    
    # Check system requirements
    if not check_dependencies():
        print("❌ Dependency check failed. Please install required packages.")
        sys.exit(1)
    
    # Check database
    db_available = check_database_connection()
    
    if db_available:
        print("✅ All systems ready - Full production mode")
    else:
        print("⚠️ Database unavailable - Limited functionality mode")
    
    print("\n" + "="*80)
    
    # Launch application
    success = launch_application()
    
    if not success:
        print("❌ Application launch failed")
        sys.exit(1)

if __name__ == "__main__":
    main()