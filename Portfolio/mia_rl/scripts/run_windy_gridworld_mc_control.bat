@echo off
setlocal

cd /d "%~dp0"
python run_windy_gridworld_mc_control.py %*

endlocal
