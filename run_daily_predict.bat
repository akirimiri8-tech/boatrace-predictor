@echo off
cd /d "%~dp0"

REM NOTE: Comments here must stay ASCII-only. cmd.exe reads .bat files with the
REM system ANSI codepage (CP932), so UTF-8 Japanese text breaks parsing.
REM
REM When the task fires right after the PC wakes from sleep, the interactive
REM session may not be ready yet and the process gets killed (exit -1073741510).
REM Delaying the start and retrying works around it. The proper fix is to change
REM the task LogonType to S4U, which requires administrator rights.
REM
REM ping is used instead of timeout.exe for the delay: timeout.exe aborts with
REM "input redirection is not supported" when stdin is redirected, which happens
REM in non-interactive contexts such as Task Scheduler.
ping -n 31 127.0.0.1 > nul

echo ===== %date% %time% ===== >> logs\daily_predict.log

for /L %%i in (1,1,3) do (
    ".venv\Scripts\python.exe" -X utf8 scripts\daily_predict.py >> logs\daily_predict.log 2>&1
    if not errorlevel 1 goto :done
    echo [retry %%i] failed, retrying in 60s >> logs\daily_predict.log
    ping -n 61 127.0.0.1 > nul
)

:done
