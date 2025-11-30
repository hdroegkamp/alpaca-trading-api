#!/usr/bin/env python3
"""Small script to verify Alpaca SDK can be imported and can fetch account info.

Usage:
  - Set environment variables APCA_API_KEY_ID and APCA_API_SECRET_KEY (and optionally APCA_API_BASE_URL).
  - Run: .venv\Scripts\python.exe scripts\check_alpaca.py

The script will exit with code 2 if keys are missing, 1 on API errors, 0 on success.
"""
import os
import sys
from typing import Optional

try:
    from alpaca.trading.client import TradingClient
except Exception as e:
    print("Failed to import alpaca-py SDK:", str(e))
    print("Make sure you installed alpaca-py in your virtual environment:")
    print("  .\\.venv\\Scripts\\python.exe -m pip install alpaca-py")
    sys.exit(1)


def main():
    # If python-dotenv is installed, load `.env` automatically into os.environ.
    # This is optional: if it's not installed, the script will still read using
    # os.getenv() and rely on environment variables set externally.
    try:
        # local import to keep the dependency optional at import-time
        from dotenv import load_dotenv  # type: ignore

        # load .env from project root if present
        load_dotenv()
    except Exception:
        # python-dotenv not installed or failed to load; continue using os.getenv
        pass

    key = os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("APCA_API_SECRET_KEY")
    base = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

    if not key or not secret:
        print("APCA_API_KEY_ID and APCA_API_SECRET_KEY are not set in the environment.")
        print("Set them and re-run. Example (PowerShell):")
        print("  $env:APCA_API_KEY_ID = 'your_key_here'")
        print("  $env:APCA_API_SECRET_KEY = 'your_secret_here'")
        print("  $env:APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'")
        return 2

    print(
        "Found API keys in environment — attempting to create client and fetch account info (this will contact Alpaca)..."
    )
    try:
        client = TradingClient(key, secret, base_url=base)
        account = client.get_account()
        print("Successfully retrieved account:")
        print(account)
        return 0
    except Exception as e:
        print("Error when contacting Alpaca API:", str(e))
        print(
            "If you are using paper trading, ensure APCA_API_BASE_URL is set to https://paper-api.alpaca.markets"
        )
        return 1


if __name__ == "__main__":
    rc = main()
    sys.exit(rc)
