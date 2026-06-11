@echo off
chcp 65001 >nul
REM ═══════════════════════════════════════════════════════════════════════════════
REM TT100K Small Target Detection — Windows One-Click Training
REM ═══════════════════════════════════════════════════════════════════════════════
REM Drag a dataset folder onto this .bat, or run:
REM   setup_server.bat D:\data\TT100K
REM ═══════════════════════════════════════════════════════════════════════════════

setlocal enabledelayedexpansion

cd /d "%~dp0\.."

REM --- Detect dataset path ---
if "%~1"=="" (
    set "DATASET_PATH=D:\data\TT100K"
) else (
    set "DATASET_PATH=%~1"
)
echo [%time%] Dataset path: %DATASET_PATH%

REM --- Step 1: Find and extract tt100k_2021.zip ---
set "RAW_DIR=data\TT100K_raw"

if exist "data\processed\dataset.yaml" (
    echo [%time%] Dataset already prepared, skipping.
    goto :train
)

echo [%time%] Step 1/3: Extracting dataset...

REM Find tt100k_2021.zip
for /f "delims=" %%f in ('powershell -Command "Get-ChildItem -Path '%DATASET_PATH%' -Recurse -Filter 'tt100k_2021.zip' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName"') do set "ZIP_FILE=%%f"

REM Fallback: find any zip
if "!ZIP_FILE!"=="" (
    for /f "delims=" %%f in ('powershell -Command "Get-ChildItem -Path '%DATASET_PATH%' -Recurse -Filter '*.zip' -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName"') do set "ZIP_FILE=%%f"
)

if not "!ZIP_FILE!"=="" (
    echo [%time%]   Found: !ZIP_FILE!
    if exist "!RAW_DIR!" rmdir /s /q "!RAW_DIR!"
    mkdir "!RAW_DIR!"
    echo [%time%]   Extracting...
    powershell -Command "Expand-Archive -Path '!ZIP_FILE!' -DestinationPath '!RAW_DIR!' -Force"
    echo [%time%]   Done.
) else (
    echo [%time%]   No ZIP found. Using dataset path as source.
    mkdir "!RAW_DIR!\tt100k_2021" 2>nul
    mklink /J "!RAW_DIR!\tt100k_2021" "%DATASET_PATH%" 2>nul
)

REM --- Step 2: Convert to YOLO format ---
echo [%time%] Step 2/3: Preparing YOLO format...

REM Find actual data root inside RAW_DIR (unzip may create subfolder)
set "SRC=!RAW_DIR!"
for /f "delims=" %%d in ('powershell -Command "Get-ChildItem -Path '!RAW_DIR!' -Directory | Select-Object -First 1 -ExpandProperty FullName"') do set "SRC=%%d"

echo [%time%]   Source: !SRC!

python data/prepare_dataset.py --data_dir "!SRC!" --output_dir "data\processed"
if %errorlevel% neq 0 (
    echo [ERROR] Dataset preparation failed!
    pause
    exit /b 1
)

:train
REM --- Step 3: Train ---
echo [%time%] Step 3/3: Training...

REM Auto-detect device
for /f "delims=" %%d in ('python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')"') do set "DEVICE=%%d"
echo [%time%]   Device: %DEVICE%

python scripts/train.py --config configs/baseline.yaml --epochs 100 --batch 8 --imgsz 1280 --device %DEVICE% --project runs/stage1_baseline --name train

echo.
echo [%time%] Done! Best model: experiments\stage1_baseline\train\weights\best.pt
pause
