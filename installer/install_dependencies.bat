@echo off
echo Installing CAD Analyzer Pro Dependencies...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Install dependencies
echo Installing required packages...
pip install PyQt5==5.15.10
pip install matplotlib==3.7.2
pip install numpy==1.24.3
pip install shapely==2.0.1
pip install ezdxf==1.0.3
pip install Pillow==10.0.0

echo.
echo ✅ All dependencies installed successfully!
echo You can now run CAD Analyzer Pro
pause