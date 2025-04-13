import streamlit as st
import yfinance as yf
import plotly.graph_objects as go

st.set_page_config(page_title="NVIDIA Stock Tracker", layout="wide")

st.title("📈 NVIDIA (NVDA) Stock Price Viewer")

# Sidebar options
period = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=3)
interval = st.sidebar.selectbox("Select Interval", ["1d", "1wk", "1mo"], index=0)

# Fetch data
df = yf.download("NVDA", period=period, interval=interval)
df.reset_index(inplace=True)

# Plotly chart
fig = go.Figure()
fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))

fig.update_layout(
    title=f"NVIDIA Stock Price - Period: {period}, Interval: {interval}",
    xaxis_title='Date',
    yaxis_title='Price',
    height=500
)

st.plotly_chart(fig, use_container_width=True)
