@echo off
setlocal

cd /d "%~dp0"

python run_tictactoe_demo.py %*

endlocal
