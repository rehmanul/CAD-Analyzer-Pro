@echo off
echo Installing CAD Analyzer Pro...

set INSTALL_DIR=%ProgramFiles%\CAD Analyzer Pro

echo Creating installation directory...
mkdir "%INSTALL_DIR%" 2>nul

echo Copying files...
xcopy /E /I /Y "CAD_Analyzer_Pro" "%INSTALL_DIR%"

echo Creating desktop shortcut...
powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%USERPROFILE%\Desktop\CAD Analyzer Pro.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\CAD_Analyzer_Pro.exe'; $Shortcut.Save()"

echo Creating start menu shortcut...
mkdir "%ProgramData%\Microsoft\Windows\Start Menu\Programs\CAD Analyzer Pro" 2>nul
powershell "$WshShell = New-Object -comObject WScript.Shell; $Shortcut = $WshShell.CreateShortcut('%ProgramData%\Microsoft\Windows\Start Menu\Programs\CAD Analyzer Pro\CAD Analyzer Pro.lnk'); $Shortcut.TargetPath = '%INSTALL_DIR%\CAD_Analyzer_Pro.exe'; $Shortcut.Save()"

echo.
echo ✅ CAD Analyzer Pro installed successfully!
echo Desktop shortcut created
echo Start menu shortcut created
echo.
pause
