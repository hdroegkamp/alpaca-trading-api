@echo off
REM Wrapper for Windows Task Scheduler — see TRADING_README.md "Live Paper Trading" section.
REM Runs the daily rebalance from the repo root regardless of the task's working directory,
REM and appends console output to logs\live_rebalance_task.log for auditing scheduled runs.
REM
REM run_live_rebalance.py exits nonzero whenever the circuit breaker halts a run (either
REM because it was already tripped from a prior run, or because today's run just tripped
REM it) — the marker below makes that impossible to miss when skimming the log, since the
REM halt itself is already logged in detail by the python process above this line.
cd /d "%~dp0.."
".venv\Scripts\python.exe" "scripts\run_live_rebalance.py" --universe starter --live >> "logs\live_rebalance_task.log" 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [%date% %time%] HALTED — run_live_rebalance.py exited %ERRORLEVEL%. Review logs\run_live_rebalance.log before --reset-circuit-breaker. >> "logs\live_rebalance_task.log"
)
