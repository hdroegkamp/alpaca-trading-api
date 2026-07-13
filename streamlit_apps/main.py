"""Main Streamlit App for Trading Analysis and Exploration.

This is the main entry point for the trading analysis suite.

Run with:
    streamlit run streamlit_apps/main.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Page configuration - called once here; individual page files must NOT repeat it
st.set_page_config(
    page_title="Trading Analysis Suite",
    page_icon=":money_with_wings:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Define pages and run navigation
pg = st.navigation(
    [
        st.Page("pages/home.py", title="Home", default=True),
        st.Page("pages/1_Data_Explorer.py", title="Data Explorer"),
        st.Page("pages/2_Strategy_Analyzer.py", title="Strategy Analyzer"),
        st.Page("pages/3_Technical_Analysis.py", title="Technical Analysis"),
        st.Page("pages/4_Data_Quality.py", title="Data Quality"),
        st.Page("pages/5_ML_Analysis.py", title="ML Analysis"),
        st.Page("pages/6_Portfolio_ML.py", title="Portfolio ML"),
    ]
)
pg.run()
