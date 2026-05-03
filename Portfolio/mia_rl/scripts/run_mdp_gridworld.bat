@echo off
setlocal

cd /d "%~dp0"
python run_mdp_gridworld.py %*

endlocal
