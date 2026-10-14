@echo off
title PneumoVision - Standalone Pipeline
cd /d "%~dp0"

echo ===============================================================================
echo Starting PneumoVision Standalone Python Pipeline...
echo ===============================================================================
echo.

set PY_EXE=python
if exist ".venv\Scripts\python.exe" (
    set PY_EXE=.venv\Scripts\python.exe
) else if exist "..\.venv\Scripts\python.exe" (
    set PY_EXE=..\.venv\Scripts\python.exe
)

echo [1/2] Checking Python environment and dependencies (%PY_EXE%)...
if exist "..\src\utils\dependency_checker.py" (
    %PY_EXE% ..\src\utils\dependency_checker.py
)

echo.
echo [2/2] Running end-to-end pneumonia pipeline...
%PY_EXE% pneumonia_complete_pipeline.py

echo.
echo ===============================================================================
echo Pipeline finished. All results saved to pneumonia_all_in_one\results\
echo ===============================================================================
pause
