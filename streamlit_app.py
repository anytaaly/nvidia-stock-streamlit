import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="NVIDIA Stock Tracker", layout="wide")

st.title(" NVIDIA (NVDA) Stock Price Viewer")

# Sidebar options
period = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=4)
interval = st.sidebar.selectbox("Select Interval", ["1d", "1wk", "1mo"], index=0)

# Download the data
df = yf.download("NVDA", period=period, interval=interval)

# Clean up columns if MultiIndex
if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

# Reset index
df.reset_index(inplace=True)

# Check if dataframe is empty
if df.empty or 'Close' not in df.columns:
    st.warning("No data available for the selected period and interval. Try changing the options.")
else:
    # Plotting
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))

    fig.update_layout(
        title=f"NVIDIA Stock Price - Period: {period}, Interval: {interval}",
        xaxis_title='Date',
        yaxis_title='Price',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)
    st.subheader(" Raw Data")
    st.dataframe(df.tail(10))  # Show last few rows
