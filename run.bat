@echo off
echo ========================================
echo LLM LoRA Project - Smart Run
echo ========================================
echo.

REM Check if environment exists
conda info --envs | findstr "llm-lora-env" >nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Environment 'llm-lora-env' not found!
    echo Please run setup-smart.bat first to create the environment.
    echo.
    pause
    exit /b 1
)

echo Activating conda environment...

REM Try different activation methods
call conda activate llm-lora-env >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Environment activated successfully
    goto :run_program
)

REM If first method fails, try alternative
echo Trying alternative activation method...
call conda.bat activate llm-lora-env >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Environment activated successfully
    goto :run_program
)

REM If both fail, show error
echo ERROR: Failed to activate environment!
echo.
echo This might happen if conda is not properly initialized.
echo Please try:
echo 1. conda init cmd.exe
echo 2. Close and restart Command Prompt
echo 3. Run this script again
echo.
echo Or run setup-smart.bat to fix conda initialization automatically.
pause
exit /b 1

:run_program
echo ✓ Environment: llm-lora-env activated
echo.
echo Starting LLM LoRA Project...
echo.

python main.py

echo.
echo Program ended. Press any key to close...
pause >nul

