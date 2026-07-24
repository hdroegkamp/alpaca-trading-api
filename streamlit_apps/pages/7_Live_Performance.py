"""Live Performance — tracking the paper account run by run_live_rebalance.py.

Reads the two append-only logs that script writes on every run (dry or
live) — data/live/equity_log.csv and data/live/orders_log.csv — and shows
an equity curve, drawdown, summary metrics, and recent order history. This
is the read-only tracking/evaluation view; it never talks to Alpaca itself.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.trading.backtest.metrics import PerformanceMetrics

st.title("Live Performance — Paper Account")
st.markdown(
    "Tracks the account `scripts/run_live_rebalance.py` rebalances. Runs in dry-run "
    "mode still append here, tagged `mode=DRY RUN`, so you can sanity-check behavior "
    "before flipping `--live`."
)

with st.sidebar:
    st.header("Configuration")
    log_dir = st.text_input("Live log directory", value="data/live", key="live_log_dir")

equity_path = Path(log_dir) / "equity_log.csv"
orders_path = Path(log_dir) / "orders_log.csv"

if not equity_path.exists():
    st.info(
        f"No equity log found at `{equity_path}` yet. Run "
        "`python scripts/run_live_rebalance.py --universe starter` at least once "
        "(dry run is fine) to populate this page."
    )
    st.stop()

equity_log = pd.read_csv(equity_path, parse_dates=["timestamp"])
equity_log = equity_log.sort_values("timestamp").reset_index(drop=True)

live_only = st.checkbox("Show only --live runs (hide dry runs)", value=False)
if live_only:
    equity_log = equity_log[equity_log["mode"] == "LIVE"].reset_index(drop=True)

if equity_log.empty:
    st.warning("No rows match the current filter.")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Latest Equity", f"${equity_log['equity'].iloc[-1]:,.2f}")
m2.metric("Runs Logged", len(equity_log))
m3.metric("Long Positions (latest)", int(equity_log["n_long"].iloc[-1]))
m4.metric("Short Positions (latest)", int(equity_log["n_short"].iloc[-1]))

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=equity_log["timestamp"], y=equity_log["equity"], mode="lines+markers", name="Equity"
    )
)
fig.update_layout(title="Equity Curve", xaxis_title="Run Time", yaxis_title="Equity ($)", height=350)
st.plotly_chart(fig, width="stretch")

if len(equity_log) > 1:
    equity_log["returns"] = equity_log["equity"].pct_change()
    non_null_returns = equity_log["returns"].dropna()

    cummax = equity_log["equity"].cummax()
    drawdown = (equity_log["equity"] - cummax) / cummax
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(x=equity_log["timestamp"], y=drawdown * 100, fill="tozeroy", name="Drawdown")
    )
    fig_dd.update_layout(
        title="Drawdown", xaxis_title="Run Time", yaxis_title="Drawdown (%)", height=250
    )
    st.plotly_chart(fig_dd, width="stretch")

    if len(non_null_returns) >= 2:
        st.subheader("Summary Metrics")
        st.caption(
            "Computed per rebalance run, not per calendar day — annualization (252/yr) "
            "is only meaningful once runs are roughly daily and this log has real history."
        )
        summary = PerformanceMetrics.calculate_all(non_null_returns, equity_log["equity"])
        st.dataframe(
            pd.DataFrame([summary]).T.rename(columns={0: "value"}).style.format("{:.4f}"),
            width="stretch",
        )
else:
    st.info("Need at least two logged runs to compute returns/drawdown.")

st.subheader("Recent Orders")
if orders_path.exists():
    orders_log = pd.read_csv(orders_path, parse_dates=["timestamp"]).sort_values(
        "timestamp", ascending=False
    )
    status_filter = st.multiselect(
        "Status", options=sorted(orders_log["status"].unique()), default=None
    )
    if status_filter:
        orders_log = orders_log[orders_log["status"].isin(status_filter)]
    st.dataframe(orders_log.head(200), width="stretch", hide_index=True)
else:
    st.info(f"No orders log found at `{orders_path}` yet.")
