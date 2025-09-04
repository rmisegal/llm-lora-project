@echo off
echo ========================================
echo LLM LoRA Project - Manual Setup Guide
echo ========================================
echo.

REM Check if conda is installed
where conda >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Conda is not installed or not in PATH!
    echo Please install Anaconda or Miniconda first.
    pause
    exit /b 1
)

echo ✓ Conda is available (version: 
conda --version
echo )

echo.
echo [STEP 1] Creating conda environment...
conda create -n llm-lora-env python=3.9 -y
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to create environment!
    pause
    exit /b 1
)
echo ✓ Environment 'llm-lora-env' created successfully

echo.
echo ========================================
echo ✅ ENVIRONMENT CREATED!
echo ========================================
echo.
echo 🔸 NEXT STEPS (run these commands manually):
echo.
echo 1. Activate the environment:
echo    conda activate llm-lora-env
echo.
echo 2. Install PyTorch:
echo    pip install torch --index-url https://download.pytorch.org/whl/cpu
echo.
echo 3. Install ML packages:
echo    pip install transformers peft datasets
echo.
echo 4. Install data science packages:
echo    pip install numpy pandas scikit-learn matplotlib seaborn
echo.
echo 5. Install development tools:
echo    pip install jupyter pytest tensorboard tqdm accelerate evaluate
echo.
echo 6. Test the installation:
echo    python -c "import torch; print('PyTorch:', torch.__version__)"
echo    python -c "import transformers; print('Transformers:', transformers.__version__)"
echo.
echo 7. Run the project:
echo    python main.py
echo.
echo ========================================
echo Copy and paste these commands one by one!
echo ========================================
pause

