@echo off
echo ========================================
echo LLM LoRA Project - Quick Start
echo ========================================
echo.

REM Check if environment exists
conda info --envs | findstr "llm-lora-env" >nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Environment 'llm-lora-env' not found!
    echo Please run setup.bat first to create the environment.
    echo.
    pause
    exit /b 1
)

echo Activating conda environment...
call conda.bat activate llm-lora-env
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate environment!
    echo Please run setup.bat to fix the environment.
    pause
    exit /b 1
)

echo ✓ Environment activated
echo.
echo Starting LLM LoRA Project...
echo.

python main.py

echo.
echo Program ended. Press any key to close...
pause >nul

