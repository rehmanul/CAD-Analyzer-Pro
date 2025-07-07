@echo off
echo Uninstalling CAD Analyzer Pro...

set INSTALL_DIR=%ProgramFiles%\CAD Analyzer Pro

echo Removing shortcuts...
del "%USERPROFILE%\Desktop\CAD Analyzer Pro.lnk" 2>nul
rmdir /S /Q "%ProgramData%\Microsoft\Windows\Start Menu\Programs\CAD Analyzer Pro" 2>nul

echo Removing installation directory...
cd /D "%TEMP%"
rmdir /S /Q "%INSTALL_DIR%" 2>nul

echo ✅ CAD Analyzer Pro uninstalled successfully!
pause
