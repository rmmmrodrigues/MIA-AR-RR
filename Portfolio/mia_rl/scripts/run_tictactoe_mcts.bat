@echo off
setlocal

cd /d "%~dp0"

python run_tictactoe_mcts.py %*

endlocal
