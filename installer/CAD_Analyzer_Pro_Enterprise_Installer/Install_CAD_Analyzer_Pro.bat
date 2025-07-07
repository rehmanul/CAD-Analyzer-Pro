@echo off
title CAD Analyzer Pro Enterprise - Installation
echo.
echo ========================================
echo   CAD Analyzer Pro Enterprise Setup
echo ========================================
echo.
echo Installing professional CAD analysis software...
echo.

set INSTALL_DIR=%ProgramFiles%\CAD Analyzer Pro Enterprise

echo [1/4] Creating installation directory...
mkdir "%INSTALL_DIR%" 2>nul

echo [2/4] Copying application files...
xcopy /E /I /Y "CAD_Analyzer_Pro_Enterprise" "%INSTALL_DIR%"

echo [3/4] Creating desktop shortcut...
powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\CAD Analyzer Pro Enterprise.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\CAD_Analyzer_Pro_Enterprise.exe'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'Professional CAD Floor Plan Analyzer'; $Shortcut.Save()"

echo [4/4] Creating start menu entry...
mkdir "%ProgramData%\Microsoft\Windows\Start Menu\Programs\CAD Analyzer Pro" 2>nul
powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%ProgramData%\Microsoft\Windows\Start Menu\Programs\CAD Analyzer Pro\CAD Analyzer Pro Enterprise.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\CAD_Analyzer_Pro_Enterprise.exe'; $Shortcut.WorkingDirectory = '%INSTALL_DIR%'; $Shortcut.Description = 'Professional CAD Floor Plan Analyzer'; $Shortcut.Save()"

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
echo - Direct path: %INSTALL_DIR%\CAD_Analyzer_Pro_Enterprise.exe
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
