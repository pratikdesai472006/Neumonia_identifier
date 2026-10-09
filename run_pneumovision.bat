@echo off
title PneumoVision Screening
cd /d "%~dp0"
echo ===================================================
echo Starting PneumoVision (100%% Offline Local Server)
echo ===================================================
echo Opening web interface at http://localhost:8501 ...
echo Keep this window open while using the application.
echo Press Ctrl+C to stop.
echo.

.\.venv\Scripts\python.exe -m streamlit run app/app.py --browser.gatherUsageStats false
pause
