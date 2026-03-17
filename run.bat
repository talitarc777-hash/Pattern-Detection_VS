@echo off
setlocal
cd /d "%~dp0"

if not exist ".tmp" mkdir .tmp
set "TMP=%CD%\.tmp"
set "TEMP=%CD%\.tmp"

if not exist ".venv\Scripts\python.exe" (
    echo Creating virtual environment...
    py -3 -m venv .venv 2>nul
    if errorlevel 1 (
        python -m venv .venv
    )
)

call ".venv\Scripts\activate.bat"
if errorlevel 1 (
    echo Failed to activate virtual environment.
    exit /b 1
)

echo Installing dependencies...
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo Launching Pattern Markup Counter...
python -m app.main

endlocal
