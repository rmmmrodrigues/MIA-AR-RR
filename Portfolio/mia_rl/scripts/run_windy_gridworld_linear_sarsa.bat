@echo off
setlocal

cd /d "%~dp0"
python run_windy_gridworld_linear_sarsa.py %*

endlocal
