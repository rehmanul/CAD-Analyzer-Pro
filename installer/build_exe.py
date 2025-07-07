import PyInstaller.__main__
import os
import shutil
from pathlib import Path

def build_executable():
    """Build standalone executable with PyInstaller"""
    
    # Clean previous builds
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")
    
    # PyInstaller arguments
    args = [
        "../desktop_app/main_production.py",
        "--name=CAD_Analyzer_Pro",
        "--windowed",
        "--onedir",
        "--add-data=../utils;utils",
        "--add-data=../sample_files;sample_files",
        "--add-data=../desktop_app/backend.py;.",
        "--hidden-import=PyQt5",
        "--hidden-import=matplotlib",
        "--hidden-import=numpy",
        "--hidden-import=shapely",
        "--hidden-import=ezdxf",
        "--hidden-import=concurrent.futures",
        "--collect-all=PyQt5",
        "--collect-all=matplotlib",
        "--noconfirm"
    ]
    
    print("Building executable with PyInstaller...")
    PyInstaller.__main__.run(args)
    
    # Create installer structure
    installer_dir = Path("CAD_Analyzer_Pro_Installer")
    if installer_dir.exists():
        shutil.rmtree(installer_dir)
    
    installer_dir.mkdir()
    
    # Copy executable
    dist_dir = Path("dist/CAD_Analyzer_Pro")
    if dist_dir.exists():
        shutil.copytree(dist_dir, installer_dir / "CAD_Analyzer_Pro")
    
    # Create installer script
    installer_script = installer_dir / "install.bat"
    with open(installer_script, 'w') as f:
        f.write("""@echo off
echo Installing CAD Analyzer Pro...

set INSTALL_DIR=%ProgramFiles%\\CAD Analyzer Pro

echo Creating installation directory...
mkdir "%INSTALL_DIR%" 2>nul

echo Copying files...
xcopy /E /I /Y "CAD_Analyzer_Pro" "%INSTALL_DIR%"

echo Creating desktop shortcut...
powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\\Desktop\\CAD Analyzer Pro.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\\CAD_Analyzer_Pro.exe'; $Shortcut.Save()"

echo Creating start menu shortcut...
mkdir "%ProgramData%\\Microsoft\\Windows\\Start Menu\\Programs\\CAD Analyzer Pro" 2>nul
powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%ProgramData%\\Microsoft\\Windows\\Start Menu\\Programs\\CAD Analyzer Pro\\CAD Analyzer Pro.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\\CAD_Analyzer_Pro.exe'; $Shortcut.Save()"

echo.
echo ✅ CAD Analyzer Pro installed successfully!
echo Desktop shortcut created
echo Start menu shortcut created
echo.
pause
""")
    
    # Create uninstaller
    uninstaller_script = installer_dir / "CAD_Analyzer_Pro" / "uninstall.bat"
    with open(uninstaller_script, 'w') as f:
        f.write("""@echo off
echo Uninstalling CAD Analyzer Pro...

set INSTALL_DIR=%ProgramFiles%\\CAD Analyzer Pro

echo Removing shortcuts...
del "%USERPROFILE%\\Desktop\\CAD Analyzer Pro.lnk" 2>nul
rmdir /S /Q "%ProgramData%\\Microsoft\\Windows\\Start Menu\\Programs\\CAD Analyzer Pro" 2>nul

echo Removing installation directory...
cd /D "%TEMP%"
rmdir /S /Q "%INSTALL_DIR%" 2>nul

echo ✅ CAD Analyzer Pro uninstalled successfully!
pause
""")
    
    # Create README
    readme_file = installer_dir / "README.txt"
    with open(readme_file, 'w') as f:
        f.write("""🏨 CAD Analyzer Pro - Professional Installation Package

INSTALLATION:
1. Right-click "install.bat" and select "Run as administrator"
2. Follow the installation prompts
3. Launch from Desktop shortcut or Start Menu

FEATURES:
✅ Professional CAD file analysis (DXF/DWG)
✅ Intelligent îlot placement with size distribution
✅ Mandatory corridor generation between facing rows
✅ Advanced visualization and export capabilities
✅ No Python installation required

SYSTEM REQUIREMENTS:
- Windows 10/11 (64-bit)
- 4GB RAM minimum
- 1GB free disk space

UNINSTALLATION:
Run uninstall.bat from the installation directory

© 2024 CAD Analyzer Pro - Enterprise Edition
""")
    
    print(f"✅ Installer package created: {installer_dir}")
    print("To install: Right-click install.bat and 'Run as administrator'")

if __name__ == "__main__":
    build_executable()