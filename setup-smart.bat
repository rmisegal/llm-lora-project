@echo off
echo ========================================
echo LLM LoRA Project - Smart Setup
echo ========================================
echo Automatically detects and configures conda for your shell
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

echo [1/9] Checking conda installation...
conda --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda command failed!
    pause
    exit /b 1
)
echo ✓ Conda is available

echo.
echo [2/9] Detecting shell environment...
set "SHELL_TYPE=unknown"
if defined COMSPEC (
    echo ✓ Detected: Windows Command Prompt (cmd.exe)
    set "SHELL_TYPE=cmd"
) else if defined PSModulePath (
    echo ✓ Detected: PowerShell
    set "SHELL_TYPE=powershell"
) else (
    echo ✓ Detected: Generic Windows shell
    set "SHELL_TYPE=cmd"
)

echo.
echo [3/9] Testing conda activation capability...
call conda activate base >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo ✓ Conda activation is already working
    call conda deactivate >nul 2>&1
    set "CONDA_READY=true"
) else (
    echo ⚠ Conda activation needs initialization
    set "CONDA_READY=false"
)

if "%CONDA_READY%"=="false" (
    echo.
    echo [4/9] Initializing conda for %SHELL_TYPE%...
    if "%SHELL_TYPE%"=="cmd" (
        conda init cmd.exe
    ) else if "%SHELL_TYPE%"=="powershell" (
        conda init powershell
    ) else (
        conda init cmd.exe
    )
    
    if %ERRORLEVEL% NEQ 0 (
        echo ERROR: Failed to initialize conda!
        echo Please run manually: conda init %SHELL_TYPE%
        pause
        exit /b 1
    )
    
    echo ✓ Conda initialized successfully
    echo.
    echo ⚠ IMPORTANT: This script will now restart to apply conda initialization
    echo Press any key to restart the setup with conda properly configured...
    pause >nul
    
    REM Restart the script in a new shell session
    start cmd /k "cd /d %~dp0 && %~nx0"
    exit /b 0
) else (
    echo.
    echo [4/9] Conda activation ready - skipping initialization
)

echo.
echo [5/9] Creating conda environment...
conda create -n llm-lora-env python=3.9 -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create environment!
    pause
    exit /b 1
)
echo ✓ Environment created successfully

echo.
echo [6/9] Activating environment...
call conda activate llm-lora-env
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate environment!
    echo This might happen if conda was just initialized.
    echo Please close this window and run the script again.
    pause
    exit /b 1
)
echo ✓ Environment activated

echo.
echo [7/9] Installing core packages...
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: PyTorch installation had issues, continuing...
)
echo ✓ Core packages installed

echo.
echo [8/9] Installing ML and data science packages...
pip install transformers peft datasets numpy pandas scikit-learn matplotlib seaborn jupyter pytest tensorboard tqdm accelerate evaluate
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some packages may have failed, continuing...
)
echo ✓ ML packages installed

echo.
echo [9/9] Testing installation...
python -c "import torch; print('✓ PyTorch:', torch.__version__)" 2>nul || echo "⚠ PyTorch test failed"
python -c "import transformers; print('✓ Transformers:', transformers.__version__)" 2>nul || echo "⚠ Transformers test failed"
python -c "import peft; print('✓ PEFT: Available')" 2>nul || echo "⚠ PEFT test failed"
python -c "import numpy; print('✓ NumPy:', numpy.__version__)" 2>nul || echo "⚠ NumPy test failed"

echo.
echo ========================================
echo ✅ SMART SETUP COMPLETED!
echo ========================================
echo.
echo Environment: llm-lora-env
echo Shell: %SHELL_TYPE%
echo Conda: Ready
echo.
echo To use the project:
echo 1. conda activate llm-lora-env
echo 2. python main.py
echo.
echo Or use: run.bat
echo.
pause

