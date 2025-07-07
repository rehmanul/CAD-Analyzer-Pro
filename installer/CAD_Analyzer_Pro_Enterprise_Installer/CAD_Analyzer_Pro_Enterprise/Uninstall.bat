@echo off
title CAD Analyzer Pro Enterprise - Uninstaller
echo.
echo ========================================
echo   CAD Analyzer Pro Enterprise Removal
echo ========================================
echo.

set INSTALL_DIR=%ProgramFiles%\CAD Analyzer Pro Enterprise

echo Removing CAD Analyzer Pro Enterprise...
echo.

echo [1/3] Removing shortcuts...
del "%USERPROFILE%\Desktop\CAD Analyzer Pro Enterprise.lnk" 2>nul
rmdir /S /Q "%ProgramData%\Microsoft\Windows\Start Menu\Programs\CAD Analyzer Pro" 2>nul

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
