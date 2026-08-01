@echo off
cd /d "%~dp0"
echo ===== %date% %time% ===== >> logs\daily_predict.log
".venv\Scripts\python.exe" -X utf8 scripts\daily_predict.py >> logs\daily_predict.log 2>&1
