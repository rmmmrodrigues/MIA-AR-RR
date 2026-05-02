@echo off
setlocal

cd /d "%~dp0"

python run_kbandits.py %*

endlocal
