@echo off
setlocal

cd /d "%~dp0"
python run_lawn_mower.py %*

endlocal
