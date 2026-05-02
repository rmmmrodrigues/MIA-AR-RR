@echo off
setlocal

cd /d "%~dp0"
python run_windy_gridworld_n_step_sarsa.py %*

endlocal
