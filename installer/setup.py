import sys
from cx_Freeze import setup, Executable
import os

# Dependencies
build_exe_options = {
    "packages": [
        "PyQt5", "matplotlib", "numpy", "shapely", "ezdxf", 
        "concurrent.futures", "threading", "hashlib", "json", "time"
    ],
    "include_files": [
        ("../utils/", "utils/"),
        ("../desktop_app/backend.py", "backend.py"),
        ("../sample_files/", "sample_files/")
    ],
    "excludes": ["tkinter", "unittest"],
    "zip_include_packages": ["*"],
    "zip_exclude_packages": []
}

# Base for Windows GUI
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="CAD Analyzer Pro",
    version="1.0",
    description="Professional CAD floor plan analyzer with îlot placement",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "../desktop_app/main_production.py",
            base=base,
            target_name="CAD_Analyzer_Pro.exe"
        )
    ]
)