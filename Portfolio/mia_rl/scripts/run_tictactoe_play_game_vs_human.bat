@echo off
setlocal

cd /d "%~dp0"

python run_tictactoe_play_game_vs_human.py %*

endlocal
