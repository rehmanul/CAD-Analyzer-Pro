import os
import shutil
import zipfile
from pathlib import Path

def create_portable_package():
    """Create portable package without cx_Freeze"""
    
    # Create portable directory
    portable_dir = Path("CAD_Analyzer_Pro_Portable")
    if portable_dir.exists():
        shutil.rmtree(portable_dir)
    
    portable_dir.mkdir()
    
    # Copy application files
    app_files = [
        "../desktop_app/main_production.py",
        "../desktop_app/backend.py"
    ]
    
    for file in app_files:
        if os.path.exists(file):
            shutil.copy2(file, portable_dir)
    
    # Copy utils directory
    utils_src = Path("../utils")
    utils_dst = portable_dir / "utils"
    if utils_src.exists():
        shutil.copytree(utils_src, utils_dst)
    
    # Copy sample files
    samples_src = Path("../sample_files")
    samples_dst = portable_dir / "sample_files"
    if samples_src.exists():
        shutil.copytree(samples_src, samples_dst)
    
    # Create run script
    run_script = portable_dir / "run_cad_analyzer.bat"
    with open(run_script, 'w') as f:
        f.write("""@echo off
echo Starting CAD Analyzer Pro...
python main_production.py
pause
""")
    
    # Create requirements file
    req_file = portable_dir / "requirements.txt"
    with open(req_file, 'w') as f:
        f.write("""PyQt5==5.15.10
matplotlib==3.7.2
numpy==1.24.3
shapely==2.0.1
ezdxf==1.0.3
Pillow==10.0.0
""")
    
    # Create setup instructions
    setup_file = portable_dir / "SETUP.txt"
    with open(setup_file, 'w') as f:
        f.write("""🏨 CAD Analyzer Pro - Portable Version Setup

REQUIREMENTS:
- Python 3.8+ installed on system
- Internet connection for first-time setup

SETUP STEPS:
1. Install Python dependencies:
   pip install -r requirements.txt

2. Run the application:
   Double-click run_cad_analyzer.bat
   OR
   python main_production.py

FEATURES:
✅ Real DXF/DWG processing
✅ Îlot placement with size distribution  
✅ Corridor generation
✅ Professional visualization
✅ Export capabilities

For executable version without Python requirement,
use the installer package instead.
""")
    
    # Create ZIP package
    zip_path = "CAD_Analyzer_Pro_Portable.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(portable_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arc_path = os.path.relpath(file_path, portable_dir.parent)
                zipf.write(file_path, arc_path)
    
    print(f"✅ Portable package created: {zip_path}")
    print(f"✅ Portable folder created: {portable_dir}")

if __name__ == "__main__":
    create_portable_package()