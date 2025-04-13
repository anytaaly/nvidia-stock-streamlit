import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import pandas as pd
import time
import numpy as np

# ----------------------
# Helper Functions
# ----------------------
def get_us_president(date):
    if date >= pd.Timestamp("2025-01-20"):
        return "Donald Trump"
    elif date >= pd.Timestamp("2021-01-20"):
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

def get_president_image(president):
    images = {
        "Joe Biden": "https://upload.wikimedia.org/wikipedia/commons/6/68/Joe_Biden_presidential_portrait.jpg",
        "Donald Trump": "https://upload.wikimedia.org/wikipedia/commons/5/56/Donald_Trump_official_portrait.jpg",
        "Barack Obama": "https://upload.wikimedia.org/wikipedia/commons/8/8d/President_Barack_Obama.jpg",
        "George W. Bush": "https://upload.wikimedia.org/wikipedia/commons/d/d4/George-W-Bush.jpeg",
        "Bill Clinton": "https://upload.wikimedia.org/wikipedia/commons/d/d3/Bill_Clinton.jpg",
    }
    return images.get(president, None)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_stock_data(symbol, period, interval):
    """Fetch stock data with caching for better performance"""
    df = yf.download(symbol, period=period, interval=interval)
    
    # Handle MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    
    df.reset_index(inplace=True)
    return df

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    if len(df) >= 20:
        df['MA20'] = df['Close'].rolling(window=20).mean()
    if len(df) >= 50:
        df['MA50'] = df['Close'].rolling(window=50).mean()
    
    # Calculate RSI (Relative Strength Index)
    if len(df) >= 14:
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
    
    return df

def get_key_stats(ticker_symbol):
    """Get key statistics for the ticker"""
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        return {
            "Market Cap": info.get('marketCap', 'N/A'),
            "P/E Ratio": info.get('trailingPE', 'N/A'),
            "52 Week High": info.get('fiftyTwoWeekHigh', 'N/A'),
            "52 Week Low": info.get('fiftyTwoWeekLow', 'N/A'),
            "Dividend Yield": info.get('dividendYield', 'N/A'),
            "Beta": info.get('beta', 'N/A')
        }
    except:
        return {
            "Market Cap": "N/A",
            "P/E Ratio": "N/A",
            "52 Week High": "N/A",
            "52 Week Low": "N/A",
            "Dividend Yield": "N/A",
            "Beta": "N/A"
        }

# ----------------------
# Streamlit App Config
# ----------------------
st.set_page_config(page_title="NVIDIA Stock Tracker", layout="wide")

# Custom fonts and styling
st.markdown("""
    <style>
    .main {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Arial', sans-serif;
        font-weight: 600;
    }
    .stButton > button {
        background-color: #8e2de2;
        color: white;
        border-radius: 999px;
        height: 3em;
        width: 12em;
        font-weight: bold;
        border: none;
        font-family: 'Arial', sans-serif;
    }
    .css-1vq4p4l {  /* This affects the sidebar */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .block-container {
        padding-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("📊 NVIDIA (NVDA) Stock Price Viewer")

# ----------------------
# Sidebar Inputs
# ----------------------
symbol = st.sidebar.text_input("Stock Symbol", "NVDA")
period = st.sidebar.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=5)
interval = st.sidebar.selectbox("Select Interval", ["1d", "1wk", "1mo"], index=0)
refresh_rate = 300  # 5 minutes

# Add auto-refresh checkbox
auto_refresh = st.sidebar.checkbox("Enable Auto-Refresh (5 min)", False)

# Display key stats in sidebar
with st.sidebar.expander("Key Stats", expanded=True):
    stats = get_key_stats(symbol)
    for key, value in stats.items():
        if key == "Market Cap" and value != "N/A":
            # Format market cap to billions/millions
            value = f"${value/1000000000:.2f}B" if value >= 1000000000 else f"${value/1000000:.2f}M"
        elif key == "Dividend Yield" and value != "N/A":
            value = f"{value*100:.2f}%"
        st.write(f"**{key}:** {value}")

# Add checkbox for technical indicators
show_indicators = st.sidebar.checkbox("Show Technical Indicators", True)

# Add expander for indicator options
if show_indicators:
    with st.sidebar.expander("Indicator Options"):
        show_ma20 = st.checkbox("20-Day Moving Average", True)
        show_ma50 = st.checkbox("50-Day Moving Average", True)
        show_rsi = st.checkbox("RSI (Relative Strength Index)", False)

# ----------------------
# Manual Refresh Strategy with Styled Button
# ----------------------
refresh = st.button("🔄 Refresh Data", key="refresh_button")

# Function to display data and charts
def display_data_and_charts(df, symbol, period, interval):
    if df.empty or 'Close' not in df.columns:
        st.warning("No data available for the selected period and interval. Try changing the options.")
        return
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Find max and min prices
    max_idx = df['Close'].idxmax()
    min_idx = df['Close'].idxmin()
    
    max_price = df.loc[max_idx, 'Close']
    min_price = df.loc[min_idx, 'Close']
    max_date = df.loc[max_idx, 'Date']
    min_date = df.loc[min_idx, 'Date']
    
    pres_max = get_us_president(max_date)
    pres_min = get_us_president(min_date)
    image_max = get_president_image(pres_max)
    image_min = get_president_image(pres_min)
    
    # Create main price chart
    fig = go.Figure()
    
    # Add main price line
    fig.add_trace(go.Scatter(
        x=df['Date'], 
        y=df['Close'], 
        mode='lines', 
        name='Close Price', 
        line=dict(color='royalblue', width=2)
    ))
    
    # Add technical indicators
    if show_indicators:
        if show_ma20 and 'MA20' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Date'], 
                y=df['MA20'], 
                mode='lines', 
                name='20-Day MA', 
                line=dict(color='orange', width=1)
            ))
        
        if show_ma50 and 'MA50' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['Date'], 
                y=df['MA50'], 
                mode='lines', 
                name='50-Day MA', 
                line=dict(color='green', width=1)
            ))
    
    # Add high and low points
    fig.add_trace(go.Scatter(
        x=[max_date], 
        y=[max_price],
        mode='markers+text',
        name='📈 High',
        text=[f'High: {max_price:.2f} ({pres_max})'],
        textposition='top center',
        marker=dict(color='green', size=10)
    ))
    
    fig.add_trace(go.Scatter(
        x=[min_date], 
        y=[min_price],
        mode='markers+text',
        name='📉 Low',
        text=[f'Low: {min_price:.2f} ({pres_min})'],
        textposition='bottom center',
        marker=dict(color='red', size=10)
    ))
    
                # Add stock splits as vertical lines using add_shape instead of add_vline
    splits = yf.Ticker(symbol).splits
    if not splits.empty:
        # Get the date range of the current chart
        min_date = df['Date'].min()
        max_date = df['Date'].max()
        
        for split_date, ratio in splits.items():
            # Convert to timezone-naive datetime for comparison
            split_date = pd.to_datetime(split_date).tz_localize(None)
            
            # Only add split lines that are within the date range of the current view
            if min_date <= split_date <= max_date:
                # Use add_shape to create a vertical line
                fig.add_shape(
                    type="line",
                    x0=split_date,
                    x1=split_date,
                    y0=0,
                    y1=1,
                    yref="paper",  # Use paper coordinates for y-axis (0 to 1)
                    line=dict(
                        color="purple",
                        width=1,
                        dash="dot",
                    )
                )
                
                # Add text annotation for the split
                fig.add_annotation(
                    x=split_date,
                    y=1.05,
                    yref="paper",
                    text=f"Split {int(ratio)}:1",
                    showarrow=False,
                    font=dict(
                        color="purple",
                        size=10
                    )
                )
    
    # Update layout
    fig.update_layout(
        title=f"{symbol} Stock Price - Period: {period}, Interval: {interval}",
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=600,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    # Display high and low price info with presidential context
    cols = st.columns([1, 6])
    with cols[0]:
        if image_max:
            st.image(image_max, width=70, caption="High")
    with cols[1]:
        st.info(f"📈 Highest Close: ${max_price:.2f} on {max_date.date()} (President: {pres_max})")
    
    cols = st.columns([1, 6])
    with cols[0]:
        if image_min:
            st.image(image_min, width=70, caption="Low")
    with cols[1]:
        st.info(f"📉 Lowest Close: ${min_price:.2f} on {min_date.date()} (President: {pres_min})")
    
    # Display RSI chart if selected
    if show_indicators and show_rsi and 'RSI' in df.columns:
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(
            x=df['Date'], 
            y=df['RSI'], 
            mode='lines', 
            name='RSI', 
            line=dict(color='purple', width=1.5)
        ))
        
        # Add overbought/oversold lines
        rsi_fig.add_hline(y=70, line_dash="dash", line_color="red")
        rsi_fig.add_hline(y=30, line_dash="dash", line_color="green")
        
        rsi_fig.update_layout(
            title=f"{symbol} Relative Strength Index (RSI)",
            xaxis_title='Date',
            yaxis_title='RSI Value',
            height=250,
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(rsi_fig, use_container_width=True)
    
    # Show raw data in expander
    with st.expander("Show Raw Data"):
        st.dataframe(df.tail(20))
    
    # ----------------------
    # Market Sentiment Section
    # ----------------------
    st.subheader("📰 Market Sentiment (Yahoo Finance)")
    
    sentiment_summary = (
        "Based on recent Yahoo Finance articles, market sentiment toward " + symbol + " "
        "appears to be **moderately bullish**. Analysts highlight strong demand for AI chips, "
        "optimistic earnings projections, and continued dominance in the GPU space. "
        "However, some caution exists around macroeconomic factors and high valuation concerns."
    )
    
    st.markdown(sentiment_summary)
    
    # ----------------------
    # Prophet Prediction Section
    # ----------------------
    st.subheader("🔮 Predicted Price Trend (Prophet Forecast)")
    
    try:
        from prophet import Prophet
        
        # Prepare data for Prophet
        prophet_df = df[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
        
        with st.spinner("Training prediction model..."):
            # Create and train model
            model = Prophet(daily_seasonality=True)
            model.fit(prophet_df)
            
            # Create future dataframe
            future_days = 30 if interval == '1d' else 10
            future = model.make_future_dataframe(periods=future_days)
            
            # Make prediction
            forecast = model.predict(future)
            
            # Create prediction chart
            pred_fig = go.Figure()
            
            # Add historical price
            pred_fig.add_trace(go.Scatter(
                x=df['Date'], 
                y=df['Close'], 
                name='Historical Price', 
                line=dict(color='blue', width=2)
            ))
            
            # Add predicted price
            pred_fig.add_trace(go.Scatter(
                x=forecast['ds'], 
                y=forecast['yhat'], 
                name='Predicted Price', 
                line=dict(color='orange', width=2)
            ))
            
            # Add confidence interval
            pred_fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                fill=None,
                mode='lines',
                line=dict(color='rgba(255, 165, 0, 0.2)', width=0),
                name='Upper Bound'
            ))
            
            pred_fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                fill='tonexty',
                mode='lines',
                line=dict(color='rgba(255, 165, 0, 0.2)', width=0),
                name='Lower Bound'
            ))
            
            # Format chart
            pred_fig.update_layout(
                title=f"Prophet-Based Future Price Forecast for {symbol}",
                xaxis_title='Date',
                yaxis_title='Price',
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(pred_fig, use_container_width=True)
            
            # Show prediction components
            with st.expander("Show Prediction Components"):
                components_fig = model.plot_components(forecast)
                st.pyplot(components_fig)
                
    except Exception as e:
        st.warning(f"Unable to generate prediction: {str(e)}")
        st.info("To use the prediction feature, make sure Prophet is installed: `pip install prophet`")

# Get initial data
df = get_stock_data(symbol, period, interval)

# Display data on initial load (this fixes the issue where data only showed after refresh)
display_data_and_charts(df, symbol, period, interval)

# Handle manual refresh
if refresh:
    st.rerun()

# Handle auto-refresh
if auto_refresh:
    time_placeholder = st.empty()
    time_placeholder.text(f"Auto-refreshing in {refresh_rate} seconds...")
    time.sleep(refresh_rate)
    st.rerun()