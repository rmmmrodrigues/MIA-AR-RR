@echo off
setlocal

cd /d "%~dp0"
python run_lawn_mower_n_step_sarsa.py %*

endlocal
