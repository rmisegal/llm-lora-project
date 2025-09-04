@echo off
echo ========================================
echo LLM LoRA Project - Environment Setup
echo ========================================
echo.

REM Check if conda is installed
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda is not installed or not in PATH!
    echo Please install Anaconda or Miniconda first.
    echo Download from: https://www.anaconda.com/products/distribution
    pause
    exit /b 1
)

echo [1/6] Checking conda installation...
conda --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda command failed!
    pause
    exit /b 1
)
echo ✓ Conda is available

echo.
echo [2/6] Creating conda environment 'llm-lora-env'...
conda env create -f environment.yml
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Environment creation failed, trying to update existing environment...
    conda env update -f environment.yml
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to create or update environment!
        pause
        exit /b 1
    )
)
echo ✓ Environment created/updated successfully

echo.
echo [3/6] Activating environment...
call conda activate llm-lora-env
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate environment!
    pause
    exit /b 1
)
echo ✓ Environment activated

echo.
echo [4/6] Verifying Python installation...
python --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Python not available in environment!
    pause
    exit /b 1
)
echo ✓ Python is available

echo.
echo [5/6] Installing additional packages via pip...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some packages may have failed to install
    echo This might be normal for some optional dependencies
)
echo ✓ Package installation completed

echo.
echo [6/6] Testing the installation...
python -c "import torch, transformers, peft; print('✓ Core packages imported successfully')"
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Core packages are not working properly!
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ SETUP COMPLETED SUCCESSFULLY!
echo ========================================
echo.
echo To use the project:
echo 1. Open a new Command Prompt or PowerShell
echo 2. Navigate to this project directory
echo 3. Run: conda activate llm-lora-env
echo 4. Run: python main.py
echo.
echo Or use the run.bat script for convenience!
echo.
pause

