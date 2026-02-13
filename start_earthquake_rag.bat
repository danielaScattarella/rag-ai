@echo off
setlocal enabledelayedexpansion

echo.
echo ======================================================
echo   EARTHQUAKE RAG SYSTEM - STREAMLIT LAUNCHER
echo   Directory: %~dp0
echo   Port: 8501
echo   Using local venv python
echo ======================================================
echo.

REM Go to script directory
cd /d "%~dp0"

REM Check if venv exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] No Python virtual environment found at .venv
    echo Please run: python -m venv .venv
    echo And then install dependencies inside it.
    pause
    exit /b
)

REM Activate venv python
set PYTHON_EXEC=.venv\Scripts\python.exe

echo [i] Using venv Python: %PYTHON_EXEC%
echo.

echo [>] Starting Streamlit...
"%PYTHON_EXEC%" -m streamlit run app.py --server.port 8501

echo.
echo [i] Streamlit stopped. Press any key to exit.
pause >nul
endlocal