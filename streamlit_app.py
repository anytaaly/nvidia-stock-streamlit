import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd

# ----------------------
# Helper Function
# ----------------------
def get_us_president(date):
    if date >= pd.Timestamp("2021-01-20"):
        return "Joe Biden"
    elif date >= pd.Timestamp("2017-01-20"):
        return "Donald Trump"
    elif date >= pd.Timestamp("2009-01-20"):
        return "Barack Obama"
    elif date >= pd.Timestamp("2001-01-20"):
        return "George W. Bush"
    elif date >= pd.Timestamp("1993-01-20"):
        return "Bill Clinton"
    else:
        return "Unknown"

# ----------------------
# Streamlit App Config
# ----------------------
st.set_page_config(page_title="NVIDIA Stock Tracker", layout="wide")
st.title("\U0001F4C8 NVIDIA (NVDA) Stock Price Viewer")


def get_president_image(president):
    images = {
        "Joe Biden": "https://upload.wikimedia.org/wikipedia/commons/6/68/Joe_Biden_presidential_portrait.jpg",
        "Donald Trump": "https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg",
        "Barack Obama": "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg",
        "George W. Bush": "https://upload.wikimedia.org/wikipedia/commons/d/d4/George-W-Bush.jpeg",
        "Bill Clinton": "https://upload.wikimedia.org/wikipedia/commons/d/d3/Bill_Clinton.jpg",
    }
    return images.get(president, None)


# ----------------------
# Sidebar Inputs
# ----------------------
period = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y"], index=4)
interval = st.sidebar.selectbox("Select Interval", ["1d", "1wk", "1mo"], index=0)
st.sidebar.markdown("### President During Peak")
president_image_url = get_president_image(pres_max)  # or pres_min if you prefer
if president_image_url:
    st.sidebar.image(president_image_url, width=150, caption=pres_max)
    
# ----------------------
# Download and Prepare Data
# ----------------------
df = yf.download("NVDA", period=period, interval=interval)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

df.reset_index(inplace=True)

if df.empty or 'Close' not in df.columns:
    st.warning("No data available for the selected period and interval. Try changing the options.")
else:
    # ----------------------
    # Find High and Low Points
    # ----------------------
    max_idx = df['Close'].idxmax()
    min_idx = df['Close'].idxmin()

    max_price = df.loc[max_idx, 'Close']
    min_price = df.loc[min_idx, 'Close']
    max_date = df.loc[max_idx, 'Date']
    min_date = df.loc[min_idx, 'Date']

    pres_max = get_us_president(max_date)
    pres_min = get_us_president(min_date)

    # ----------------------
    # Plotting
    # ----------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price', line=dict(color='royalblue')))

    # High Marker
    fig.add_trace(go.Scatter(
        x=[max_date], y=[max_price],
        mode='markers+text',
        name='\U0001F4C8 High',
        text=[f'High: {max_price:.2f} ({pres_max})'],
        textposition='top center',
        marker=dict(color='green', size=10)
    ))

    # Low Marker
    fig.add_trace(go.Scatter(
        x=[min_date], y=[min_price],
        mode='markers+text',
        name='\U0001F4C9 Low',
        text=[f'Low: {min_price:.2f} ({pres_min})'],
        textposition='bottom center',
        marker=dict(color='red', size=10)
    ))

    fig.update_layout(
        title=f"NVIDIA Stock Price - Period: {period}, Interval: {interval}",
        xaxis_title='Date',
        yaxis_title='Price',
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------
    # Display President Info
    # ----------------------
    st.info(f"\U0001F4C8 Highest Close: {max_price:.2f} on {max_date.date()} (President: {pres_max})")
    st.info(f"\U0001F4C9 Lowest Close: {min_price:.2f} on {min_date.date()} (President: {pres_min})")

    # Optional: Show Raw Data
    with st.expander("Show Raw Data"):
        st.dataframe(df.tail(20))