@echo off
setlocal
cd /d "%~dp0"

if not exist ".tmp" mkdir .tmp
set "TMP=%CD%\.tmp"
set "TEMP=%CD%\.tmp"

echo Installing dependencies...
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

python -m pip install -r requirements.txt
if errorlevel 1 exit /b 1

echo Launching Pattern Markup Counter...
python -m app.main

endlocal
