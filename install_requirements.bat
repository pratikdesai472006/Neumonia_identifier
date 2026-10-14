@echo off
title PneumoVision - Library Installation & Verification
cd /d "%~dp0"

echo ===============================================================================
echo   PneumoVision: Install All Required Dependencies
echo ===============================================================================
echo.
echo Terminal command equivalent:
echo   pip install -r requirements.txt
echo.
echo -------------------------------------------------------------------------------

set PY_EXE=python
if exist ".\.venv\Scripts\python.exe" (
    set PY_EXE=.\.venv\Scripts\python.exe
    echo Using project virtual environment: .\.venv
) else (
    echo Using system Python environment: %PY_EXE%
)
echo -------------------------------------------------------------------------------
echo.

echo [1/3] Upgrading pip...
%PY_EXE% -m pip install --upgrade pip

echo.
echo [2/3] Installing all required libraries from requirements.txt ...
%PY_EXE% -m pip install -r requirements.txt

echo.
echo [3/3] Verifying dependency installations...
%PY_EXE% src\utils\dependency_checker.py

echo.
echo ===============================================================================
echo Installation and verification complete!
echo.
echo Quickstart options:
echo   1. Launch Web Portal:    Double-click 'run_pneumovision.bat'
echo   2. Run Standalone Script: Double-click 'pneumonia_all_in_one\run_pipeline.bat'
echo   3. Run Jupyter Notebook: Open 'pneumonia_all_in_one\pneumonia_complete_pipeline.ipynb'
echo ===============================================================================
pause
