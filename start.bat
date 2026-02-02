@echo off
setlocal
title VIBE Image Editor - Launcher

:: --- CONFIGURATION ---
:: Define the path to your python executable within the virtual environment
set PYTHON_EXE=venv\Scripts\python.exe

echo ==================================================
echo       VIBE IMAGE EDITOR - STARTUP CHECK
echo ==================================================

:: 1. Check: Does the virtual environment exist?
echo [INFO] Checking for virtual environment...
if not exist %PYTHON_EXE% (
    echo [ERROR] Virtual environment 'venv' not found!
    echo Please ensure the 'venv' folder exists in the project directory.
    pause
    exit /b
)

:: 2. Check: Is CUDA (GPU) ready?
echo [INFO] Checking GPU (CUDA) status...
%PYTHON_EXE% -c "import torch; exit(0 if torch.cuda.is_available() else 1)"
if %errorlevel% neq 0 (
    echo [ERROR] CUDA is not available! 
    echo This app requires an NVIDIA GPU and properly installed drivers.
    pause
    exit /b
)

echo [OK] CUDA detected.
echo [OK] Starting Web Interface...
echo ==================================================

:: Run the application
%PYTHON_EXE% app.py

:: If the app crashes, keep the window open for debugging
if %errorlevel% neq 0 (
    echo.
    echo [WARNING] Application exited unexpectedly.
    pause
)