@echo off
cd /d "%~dp0"

REM This runs when the user logs on to Windows. It shows the prediction
REM output in a visible console window (kept open with /k) while also
REM appending it to the normal log file, using PowerShell Tee-Object
REM (cmd has no built-in tee).
REM
REM A short delay avoids firing before the desktop session is fully ready
REM (same reasoning as run_daily_predict.bat's sleep-wake workaround).
ping -n 16 127.0.0.1 > nul

echo ===== %date% %time% (on-logon run) ===== >> logs\daily_predict.log

start "Boatrace Predictor - Today's Picks" cmd /k powershell -NoLogo -NoProfile -Command ^
    "& '.venv\Scripts\python.exe' -X utf8 scripts\daily_predict.py 2>&1 | Tee-Object -FilePath 'logs\daily_predict.log' -Append"
