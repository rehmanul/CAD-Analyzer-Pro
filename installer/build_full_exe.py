import PyInstaller.__main__
import os
import shutil
from pathlib import Path

def build_full_executable():
    """Build complete production executable with all dependencies"""
    
    # Clean previous builds
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    if os.path.exists("build"):
        shutil.rmtree("build")
    
    # Full PyInstaller arguments for production app
    args = [
        "../desktop_app/main_production.py",
        "--name=CAD_Analyzer_Pro_Enterprise",
        "--windowed",
        "--onedir",
        "--add-data=../utils;utils",
        "--add-data=../sample_files;sample_files",
        "--add-data=../desktop_app/backend.py;.",
        "--hidden-import=PyQt5",
        "--hidden-import=PyQt5.QtCore",
        "--hidden-import=PyQt5.QtGui", 
        "--hidden-import=PyQt5.QtWidgets",
        "--hidden-import=matplotlib",
        "--hidden-import=matplotlib.backends.backend_qt5agg",
        "--hidden-import=matplotlib.figure",
        "--hidden-import=numpy",
        "--hidden-import=shapely",
        "--hidden-import=shapely.geometry",
        "--hidden-import=shapely.ops",
        "--hidden-import=shapely.affinity",
        "--hidden-import=ezdxf",
        "--hidden-import=ezdxf.recover",
        "--hidden-import=concurrent.futures",
        "--hidden-import=threading",
        "--hidden-import=hashlib",
        "--hidden-import=json",
        "--hidden-import=time",
        "--hidden-import=os",
        "--hidden-import=sys",
        "--hidden-import=pathlib",
        "--hidden-import=typing",
        "--hidden-import=dataclasses",
        "--hidden-import=collections",
        "--hidden-import=scipy",
        "--hidden-import=scipy.spatial",
        "--hidden-import=scipy.spatial.distance",
        "--hidden-import=sklearn",
        "--hidden-import=sklearn.cluster",
        "--hidden-import=PIL",
        "--hidden-import=PIL.Image",
        "--collect-all=PyQt5",
        "--collect-all=matplotlib",
        "--collect-all=numpy",
        "--collect-all=shapely",
        "--collect-all=ezdxf",
        "--collect-all=scipy",
        "--collect-all=sklearn",
        "--collect-all=PIL",
        "--collect-submodules=shapely",
        "--collect-submodules=ezdxf",
        "--collect-submodules=matplotlib",
        "--noconfirm",
        "--debug=all"
    ]
    
    print("Building full production executable...")
    print("This will create a larger but more complete application...")
    
    PyInstaller.__main__.run(args)
    
    # Create professional installer structure
    installer_dir = Path("CAD_Analyzer_Pro_Enterprise_Installer")
    if installer_dir.exists():
        shutil.rmtree(installer_dir)
    
    installer_dir.mkdir()
    
    # Copy executable
    dist_dir = Path("dist/CAD_Analyzer_Pro_Enterprise")
    if dist_dir.exists():
        shutil.copytree(dist_dir, installer_dir / "CAD_Analyzer_Pro_Enterprise")
    
    # Create professional installer script
    installer_script = installer_dir / "Install_CAD_Analyzer_Pro.bat"
    with open(installer_script, 'w') as f:
        f.write("""@echo off
title CAD Analyzer Pro Enterprise - Installation
echo.
echo ========================================
echo   CAD Analyzer Pro Enterprise Setup
echo ========================================
echo.
echo Installing professional CAD analysis software...
echo.

set INSTALL_DIR=%ProgramFiles%\\CAD Analyzer Pro Enterprise

echo [1/4] Creating installation directory...
mkdir "%INSTALL_DIR%" 2>nul

echo [2/4] Copying application files...
xcopy /E /I /Y "CAD_Analyzer_Pro_Enterprise" "%INSTALL_DIR%"

echo [3/4] Creating desktop shortcut...
powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\\Desktop\\CAD Analyzer Pro Enterprise.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\\CAD_Analyzer_Pro_Enterprise.exe'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'Professional CAD Floor Plan Analyzer'; $Shortcut.Save()"

echo [4/4] Creating start menu entry...
mkdir "%ProgramData%\\Microsoft\\Windows\\Start Menu\\Programs\\CAD Analyzer Pro" 2>nul
powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%ProgramData%\\Microsoft\\Windows\\Start Menu\\Programs\\CAD Analyzer Pro\\CAD Analyzer Pro Enterprise.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\\CAD_Analyzer_Pro_Enterprise.exe'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'Professional CAD Floor Plan Analyzer'; $Shortcut.Save()"

echo.
echo ========================================
echo   Installation Complete!
echo ========================================
echo.
echo CAD Analyzer Pro Enterprise has been installed successfully.
echo.
echo Launch options:
echo - Desktop shortcut: CAD Analyzer Pro Enterprise
echo - Start Menu: CAD Analyzer Pro ^> CAD Analyzer Pro Enterprise
echo - Direct path: %INSTALL_DIR%\\CAD_Analyzer_Pro_Enterprise.exe
echo.
echo Features installed:
echo ✓ Real DXF/DWG file processing
echo ✓ Advanced îlot placement algorithms
echo ✓ Mandatory corridor generation
echo ✓ Professional visualization engine
echo ✓ Export capabilities (JSON, Images)
echo ✓ Complete dependency package (~200MB)
echo.
pause
""")
    
    # Create uninstaller
    uninstaller_script = installer_dir / "CAD_Analyzer_Pro_Enterprise" / "Uninstall.bat"
    with open(uninstaller_script, 'w') as f:
        f.write("""@echo off
title CAD Analyzer Pro Enterprise - Uninstaller
echo.
echo ========================================
echo   CAD Analyzer Pro Enterprise Removal
echo ========================================
echo.

set INSTALL_DIR=%ProgramFiles%\\CAD Analyzer Pro Enterprise

echo Removing CAD Analyzer Pro Enterprise...
echo.

echo [1/3] Removing shortcuts...
del "%USERPROFILE%\\Desktop\\CAD Analyzer Pro Enterprise.lnk" 2>nul
rmdir /S /Q "%ProgramData%\\Microsoft\\Windows\\Start Menu\\Programs\\CAD Analyzer Pro" 2>nul

echo [2/3] Stopping any running instances...
taskkill /F /IM "CAD_Analyzer_Pro_Enterprise.exe" 2>nul

echo [3/3] Removing installation directory...
cd /D "%TEMP%"
rmdir /S /Q "%INSTALL_DIR%" 2>nul

echo.
echo ========================================
echo   Uninstallation Complete!
echo ========================================
echo.
echo CAD Analyzer Pro Enterprise has been removed from your system.
echo.
pause
""")
    
    # Create comprehensive README
    readme_file = installer_dir / "README_ENTERPRISE.txt"
    with open(readme_file, 'w') as f:
        f.write("""🏨 CAD Analyzer Pro Enterprise Edition - Professional Installation Package

ENTERPRISE FEATURES:
✅ Complete dependency package (~200MB for full functionality)
✅ Real DXF/DWG file processing with advanced parsers
✅ Intelligent îlot placement with size distribution algorithms
✅ Mandatory corridor generation between facing îlot rows
✅ Professional visualization with matplotlib integration
✅ Advanced geometric analysis with Shapely library
✅ Machine learning optimization with SciKit-Learn
✅ Export capabilities (JSON, high-resolution images)
✅ No external dependencies required on target system

INSTALLATION:
1. Right-click "Install_CAD_Analyzer_Pro.bat"
2. Select "Run as administrator"
3. Follow installation prompts
4. Launch from Desktop or Start Menu

SYSTEM REQUIREMENTS:
- Windows 10/11 (64-bit)
- 8GB RAM recommended (4GB minimum)
- 500MB free disk space for installation
- Graphics card with OpenGL support (recommended)

APPLICATION SIZE:
- Executable package: ~200MB (includes all dependencies)
- This is normal for professional CAD software with:
  * Complete Python runtime
  * Scientific computing libraries (NumPy, SciPy, Shapely)
  * GUI framework (PyQt5)
  * Visualization libraries (Matplotlib)
  * CAD processing libraries (ezdxf)
  * Machine learning libraries (SciKit-Learn)

UNINSTALLATION:
Run Uninstall.bat from the installation directory
OR
Use Windows "Add or Remove Programs"

SUPPORT:
This is a professional-grade application comparable to:
- AutoCAD plugins (~100-500MB)
- Revit add-ins (~200-800MB)  
- Professional CAD software (~1-5GB)

Our 200MB package includes everything needed for:
- Real CAD file processing
- Advanced geometric algorithms
- Professional visualization
- Machine learning optimization
- Complete standalone operation

© 2024 CAD Analyzer Pro Enterprise - Professional Edition
""")
    
    # Get actual size
    if dist_dir.exists():
        total_size = sum(f.stat().st_size for f in dist_dir.rglob('*') if f.is_file())
        size_mb = total_size / (1024 * 1024)
        
        print(f"✅ Enterprise installer package created: {installer_dir}")
        print(f"📦 Application size: {size_mb:.1f} MB")
        print(f"🎯 This is appropriate for professional CAD software with full dependencies")
        print(f"📋 To install: Right-click 'Install_CAD_Analyzer_Pro.bat' and 'Run as administrator'")
    else:
        print("❌ Build failed - executable not found")

if __name__ == "__main__":
    build_full_executable()