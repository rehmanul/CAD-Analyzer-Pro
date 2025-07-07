@echo off
echo 🏨 CAD Analyzer Pro - Quick Start
echo.
echo Choose an option:
echo 1. Install dependencies (first time only)
echo 2. Run CAD Analyzer Pro
echo.
set /p choice="Enter choice (1 or 2): "

if "%choice%"=="1" (
    call install_dependencies.bat
) else if "%choice%"=="2" (
    call run_cad_analyzer.bat
) else (
    echo Invalid choice. Please run again.
    pause
)