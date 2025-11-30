# Alpaca Trading API - Python environment

This project contains a Python virtual environment and dependencies for working with Alpaca's Trading API.

Setup (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip freeze > requirements.txt
```

Verify:

```powershell
.\.venv\Scripts\Activate.ps1
pip show alpaca-py
```

Next steps:
- Add Alpaca API keys to environment variables.
- Create a sample script to test authentication and market data.
- Add tests and linting as needed.
