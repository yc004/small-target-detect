@echo off
REM ────────────────────────────────────────────────────────────────────────────
REM COCO Evaluation — One-click script (Windows)
REM ────────────────────────────────────────────────────────────────────────────
REM Usage:
REM   scripts\eval_coco.bat stage1_baseline          evaluate one stage
REM   scripts\eval_coco.bat                          evaluate ALL completed stages
REM ────────────────────────────────────────────────────────────────────────────
setlocal enabledelayedexpansion

set PROJECT_ROOT=%~dp0..
set IMGSZ=1280
set SPLIT=test
set CONF=0.001
set BATCH=8

REM Auto-detect device
python -c "import torch; print('0' if torch.cuda.is_available() else 'cpu')" > %TEMP%\device.txt 2>nul
set /p DEVICE=<%TEMP%\device.txt
del %TEMP%\device.txt 2>nul
if "%DEVICE%"=="" set DEVICE=0

if "%~1"=="" goto :eval_all

:eval_loop
if "%~1"=="" goto :end
set STAGE=%~1
set WEIGHTS=experiments\%STAGE%\train\weights\best.pt
if not exist "%WEIGHTS%" (
    echo ⚠  Skipping %STAGE%: no best.pt at %WEIGHTS%
    shift
    goto :eval_loop
)

echo.
echo ════════════════════════════════════════════════════
echo   Evaluating: %STAGE%
echo   Weights:    %WEIGHTS%
echo   Device:     %DEVICE%
echo   Image size: %IMGSZ%
echo ════════════════════════════════════════════════════

python scripts\eval.py ^
    --weights %WEIGHTS% ^
    --data data\processed\dataset.yaml ^
    --split %SPLIT% ^
    --imgsz %IMGSZ% ^
    --batch %BATCH% ^
    --device %DEVICE% ^
    --conf %CONF% ^
    --analyze_sizes ^
    --save_dir experiments\%STAGE%\eval

echo   ✅ %STAGE% done
echo      Metrics: experiments\%STAGE%\eval\metrics.json
shift
goto :eval_loop

:eval_all
echo Scanning for trained stages...
set FOUND=0
for /d %%d in (experiments\*) do (
    set STAGE=%%~nxd
    if not "!STAGE!"=="eval_results" (
        if exist "%%d\train\weights\best.pt" (
            set /a FOUND+=1
            call :eval_one "%%d" "!STAGE!"
        )
    )
)
if %FOUND%==0 (
    echo No trained stages found.
    exit /b 1
)
goto :end

:eval_one
echo.
echo ════════════════════════════════════════════════════
echo   Evaluating: %~2
echo   Weights:    %~1\train\weights\best.pt
echo   Device:     %DEVICE%
echo   Image size: %IMGSZ%
echo ════════════════════════════════════════════════════

python scripts\eval.py ^
    --weights %~1\train\weights\best.pt ^
    --data data\processed\dataset.yaml ^
    --split %SPLIT% ^
    --imgsz %IMGSZ% ^
    --batch %BATCH% ^
    --device %DEVICE% ^
    --conf %CONF% ^
    --analyze_sizes ^
    --save_dir experiments\%~2\eval

echo   ✅ %~2 done
exit /b

:end
echo.
echo ════════════════════════════════════════════════════
echo   All evaluations complete.
echo   Device used: %DEVICE%
echo ════════════════════════════════════════════════════
