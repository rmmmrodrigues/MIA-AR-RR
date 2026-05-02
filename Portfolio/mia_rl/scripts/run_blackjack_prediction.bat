@echo off
setlocal

cd /d "%~dp0"
python run_blackjack_prediction.py %*

endlocal
