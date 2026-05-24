@echo off
setlocal

cd /d "%~dp0"
python run_lawn_mower_monte_carlo.py %*

endlocal
