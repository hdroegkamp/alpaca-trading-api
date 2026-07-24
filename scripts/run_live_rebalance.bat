@echo off
REM Wrapper for Windows Task Scheduler — see TRADING_README.md "Live Paper Trading" section.
REM Runs the daily rebalance from the repo root regardless of the task's working directory,
REM and appends console output to logs\live_rebalance_task.log for auditing scheduled runs.
cd /d "%~dp0.."
".venv\Scripts\python.exe" "scripts\run_live_rebalance.py" --universe starter --live >> "logs\live_rebalance_task.log" 2>&1
