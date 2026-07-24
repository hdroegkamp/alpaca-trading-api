@echo off
REM Weekly headless Claude Code review of the live trading logs — see
REM TRADING_README.md "Live Paper Trading" > "Weekly performance review".
REM Unlike run_live_rebalance.bat, this does not touch Alpaca or submit
REM orders — it only reads the two local CSV logs and writes a plain-English
REM summary. Requires the standalone Claude Code CLI (not just this VSCode
REM extension) and a one-time `claude setup-token` on this machine first.
cd /d "%~dp0.."
echo.>> "logs\weekly_review.log"
echo ==== %date% %time% ==== >> "logs\weekly_review.log"
claude -p "Read data/live/equity_log.csv and data/live/orders_log.csv in this repo. Summarize the paper trading bot's performance in plain English: how equity has moved across runs, the mix of long vs short positions over time, any orders skipped or rejected and why, and anything that looks off (e.g. a sharp equity drop, repeated rejections for the same symbol). Keep it under 200 words. Read-only — do not modify any files." --allowedTools "Read" >> "logs\weekly_review.log" 2>&1
