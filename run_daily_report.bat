@echo off
cd /d "%~dp0"

REM Comments must stay ASCII-only (see run_daily_predict.bat for the reason).
REM Same sleep-wake workaround: delay the start, then retry on failure.
REM ping is used for the delay because timeout.exe fails when stdin is redirected.
ping -n 31 127.0.0.1 > nul

echo ===== %date% %time% ===== >> logs\daily_report.log

for /L %%i in (1,1,3) do (
    ".venv\Scripts\python.exe" -X utf8 scripts\daily_report.py >> logs\daily_report.log 2>&1
    if not errorlevel 1 goto :done
    echo [retry %%i] failed, retrying in 60s >> logs\daily_report.log
    ping -n 61 127.0.0.1 > nul
)

:done
