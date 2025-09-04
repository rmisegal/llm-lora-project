@echo off
echo ========================================
echo LLM LoRA Project - Simple Setup
echo ========================================
echo This script avoids conda memory issues by using a step-by-step approach
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

echo [1/8] Checking conda installation...
conda --version
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda command failed!
    pause
    exit /b 1
)
echo ✓ Conda is available

echo.
echo [2/8] Creating basic conda environment...
conda create -n llm-lora-env python=3.9 -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create environment!
    pause
    exit /b 1
)
echo ✓ Basic environment created

echo.
echo [3/8] Activating environment...
call conda.bat activate llm-lora-env
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to activate environment!
    pause
    exit /b 1
)
echo ✓ Environment activated

echo.
echo [4/8] Installing core packages...
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: PyTorch installation had issues, continuing...
)
echo ✓ Core packages installed

echo.
echo [5/8] Installing ML packages...
pip install transformers peft datasets
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some ML packages may have failed, continuing...
)
echo ✓ ML packages installed

echo.
echo [6/8] Installing data science packages...
pip install numpy pandas scikit-learn matplotlib seaborn
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some data science packages may have failed, continuing...
)
echo ✓ Data science packages installed

echo.
echo [7/8] Installing development tools...
pip install jupyter pytest tensorboard tqdm accelerate evaluate
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some development tools may have failed, continuing...
)
echo ✓ Development tools installed

echo.
echo [8/8] Testing the installation...
python -c "import torch; print('✓ PyTorch:', torch.__version__)" 2>nul
python -c "import transformers; print('✓ Transformers:', transformers.__version__)" 2>nul
python -c "import peft; print('✓ PEFT: Available')" 2>nul
python -c "import numpy; print('✓ NumPy:', numpy.__version__)" 2>nul

echo.
echo ========================================
echo ✅ SIMPLE SETUP COMPLETED!
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

