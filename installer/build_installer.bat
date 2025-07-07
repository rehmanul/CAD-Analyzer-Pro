@echo off
echo Building CAD Analyzer Pro Installer...

REM Install required packages
pip install cx_Freeze PyQt5 matplotlib numpy shapely ezdxf

REM Build executable
python setup.py build

REM Create installer directory structure
mkdir "CAD_Analyzer_Pro_Installer"
xcopy /E /I "build\exe.win-amd64-3.*" "CAD_Analyzer_Pro_Installer\CAD_Analyzer_Pro"

REM Copy additional files
copy "..\README.md" "CAD_Analyzer_Pro_Installer\"
copy "install_instructions.txt" "CAD_Analyzer_Pro_Installer\"

echo.
echo ✅ Installer created in CAD_Analyzer_Pro_Installer folder
echo Double-click CAD_Analyzer_Pro.exe to run the application
pause