import os
import sys
from pathlib import Path

def create_wix_installer():
    """Create Windows MSI installer using WiX Toolset"""
    
    # WiX XML configuration
    wix_xml = """<?xml version="1.0" encoding="UTF-8"?>
<Wix xmlns="http://schemas.microsoft.com/wix/2006/wi">
  <Product Id="*" Name="CAD Analyzer Pro" Language="1033" Version="1.0.0.0" 
           Manufacturer="CAD Analyzer Pro Team" UpgradeCode="12345678-1234-1234-1234-123456789012">
    
    <Package InstallerVersion="200" Compressed="yes" InstallScope="perMachine" />
    
    <MajorUpgrade DowngradeErrorMessage="A newer version is already installed." />
    
    <MediaTemplate EmbedCab="yes" />
    
    <Feature Id="ProductFeature" Title="CAD Analyzer Pro" Level="1">
      <ComponentGroupRef Id="ProductComponents" />
    </Feature>
    
    <Directory Id="TARGETDIR" Name="SourceDir">
      <Directory Id="ProgramFilesFolder">
        <Directory Id="INSTALLFOLDER" Name="CAD Analyzer Pro" />
      </Directory>
      <Directory Id="ProgramMenuFolder">
        <Directory Id="ApplicationProgramsFolder" Name="CAD Analyzer Pro"/>
      </Directory>
      <Directory Id="DesktopFolder" Name="Desktop" />
    </Directory>
    
    <ComponentGroup Id="ProductComponents" Directory="INSTALLFOLDER">
      <Component Id="MainExecutable" Guid="*">
        <File Id="MainExe" Source="CAD_Analyzer_Pro.exe" KeyPath="yes">
          <Shortcut Id="ApplicationStartMenuShortcut" Directory="ApplicationProgramsFolder" 
                    Name="CAD Analyzer Pro" WorkingDirectory="INSTALLFOLDER" Icon="AppIcon.exe" IconIndex="0" Advertise="yes" />
          <Shortcut Id="ApplicationDesktopShortcut" Directory="DesktopFolder" 
                    Name="CAD Analyzer Pro" WorkingDirectory="INSTALLFOLDER" Icon="AppIcon.exe" IconIndex="0" Advertise="yes" />
        </File>
      </Component>
      
      <Component Id="SupportFiles" Guid="*">
        <File Id="BackendPy" Source="backend.py" KeyPath="yes" />
        <File Id="RequirementsTxt" Source="requirements.txt" KeyPath="no" />
        <File Id="SetupTxt" Source="SETUP.txt" KeyPath="no" />
      </Component>
      
      <Component Id="UtilsFolder" Guid="*">
        <CreateFolder />
      </Component>
    </ComponentGroup>
    
    <Icon Id="AppIcon.exe" SourceFile="CAD_Analyzer_Pro.exe" />
    
    <Property Id="ARPPRODUCTICON" Value="AppIcon.exe" />
    
  </Product>
</Wix>"""
    
    # Write WiX file
    with open("CADAnalyzerPro.wxs", "w") as f:
        f.write(wix_xml)
    
    print("✅ WiX installer configuration created")
    print("To build MSI installer:")
    print("1. Install WiX Toolset from https://wixtoolset.org/")
    print("2. Run: candle CADAnalyzerPro.wxs")
    print("3. Run: light CADAnalyzerPro.wixobj")

if __name__ == "__main__":
    create_wix_installer()