@echo off
cd /d "%~dp0"
echo ===== %date% %time% ===== >> logs\daily_report.log
".venv\Scripts\python.exe" -X utf8 scripts\daily_report.py >> logs\daily_report.log 2>&1
