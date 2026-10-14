@echo off
title PneumoVision Screening (Localhost Server)
cd /d "%~dp0"

echo ===============================================================================
echo   PneumoVision: 100%% Offline Local Server & AI Radiograph Screening
echo ===============================================================================
echo.

set PY_EXE=python
if exist ".\.venv\Scripts\python.exe" (
    set PY_EXE=.\.venv\Scripts\python.exe
)

echo [1/2] Verifying all required libraries...
%PY_EXE% src\utils\dependency_checker.py
if %ERRORLEVEL% NEQ 0 (
    echo [!] Some libraries were missing. Installing via requirements.txt ...
    %PY_EXE% -m pip install -r requirements.txt
)

echo.
echo [2/2] Launching web interface at http://localhost:8501 ...
echo Keep this terminal window open while using the application.
echo Press Ctrl+C in this window to stop the server.
echo.

%PY_EXE% -m streamlit run app/app.py --browser.gatherUsageStats false
pause
