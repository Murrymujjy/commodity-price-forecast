import streamlit as st
import requests
import pandas as pd
import json
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

st.set_page_config(layout="wide")
st.title("Forecast Dashboard")
st.markdown("Predict future commodity price indices and get instant insights.")

# URL of the Flask API
API_URL = 'http://127.0.0.1:5000/predict'

# --- Section 1: Forecasting Inputs ---
st.header("1. Make a Forecast")
col1, col2 = st.columns(2)

with col1:
    forecast_date_str = st.text_input("Enter a forecast date (YYYY-MM-DD):", "2025-04-01")
    try:
        forecast_date = datetime.strptime(forecast_date_str, '%Y-%m-%d')
    except ValueError:
        st.error("Invalid date format. Please use YYYY-MM-DD.")
        st.stop()
        
with col2:
    commodity_group = st.text_input("Enter Commodity Group:", "All commodity Group Import Price Index")

if st.button("Get Forecast"):
    with st.spinner("Analyzing data and generating forecast..."):
        try:
            # Prepare data for API request
            request_data = {
                "date": forecast_date_str,
                "commodity_group": commodity_group
            }
            response = requests.post(API_URL, json=request_data)
            response.raise_for_status() # Raises an HTTPError if the status is 4xx or 5xx
            forecast_data = response.json()
            
            st.session_state['forecast_data'] = forecast_data
            
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Please ensure your Flask API is running.")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred while making the API request: {e}")
            st.info(f"Response from API: {response.text}")

# --- Section 2: Insights and Visualizations ---
if 'forecast_data' in st.session_state:
    forecast_data = st.session_state['forecast_data']
    st.header("2. Insights and Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        forecasted_price = forecast_data.get('forecasted_price')
        historical_data = pd.DataFrame(forecast_data.get('historical_data', []))

        if not historical_data.empty:
            historical_data['date'] = pd.to_datetime(historical_data['date'])
            historical_data.set_index('date', inplace=True)
            historical_mean = historical_data['price'].mean()
            
            st.subheader("Price Detector")
            
            # Simple logic to determine if the price is "expensive"
            if forecasted_price > (historical_mean * 1.05):
                st.warning(f"⚠️ **Warning:** The forecasted price of **{forecasted_price:.2f}** suggests this commodity may be becoming more expensive than the historical average.")
            else:
                st.info(f"✅ **Stable:** The forecasted price of **{forecasted_price:.2f}** appears stable relative to the historical average.")
            
    with col4:
        st.subheader("Forecast Summary")
        st.metric(
            label=f"Forecasted Price Index for {forecast_date_str}",
            value=f"{forecasted_price:.2f}"
        )

        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px;">
            This forecast is based on an ensemble of sophisticated machine learning models. The prediction of **{forecasted_price:.2f}** suggests the price index is moving in line with recent historical trends. For a more detailed analysis, see the visualization below.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # --- Section 3: Visualization ---
    st.header("3. Visualization")

    # Create a DataFrame for plotting
    plot_df = historical_data.copy()
    future_df = pd.DataFrame({
        'date': [forecast_date], 
        'price': [forecasted_price]
    }).set_index('date')

    # Combine historical and forecasted data for the plot
    full_df = pd.concat([plot_df, future_df]).sort_index()

    # Create a Plotly figure
    fig = go.Figure()

    # Add historical data trace
    fig.add_trace(go.Scatter(
        x=plot_df.index,
        y=plot_df['price'],
        mode='lines+markers',
        name='Historical Data',
        marker=dict(color='royalblue', size=8),
        line=dict(width=2)
    ))

    # Add forecast data trace
    fig.add_trace(go.Scatter(
        x=future_df.index,
        y=future_df['price'],
        mode='markers',
        name='Forecasted Price',
        marker=dict(color='darkorange', size=12, symbol='star')
    ))

    # Add a line connecting the last historical point to the forecast
    last_hist_point = plot_df.iloc[-1]
    fig.add_trace(go.Scatter(
        x=[last_hist_point.name, future_df.index[0]],
        y=[last_hist_point['price'], future_df.iloc[0]['price']],
        mode='lines',
        name='Forecast Line',
        line=dict(color='darkorange', width=2, dash='dash')
    ))

    # Update layout for a professional look
    fig.update_layout(
        title="Commodity Price Index: Historical Data & Forecast",
        xaxis_title="Date",
        yaxis_title="Price Index",
        hovermode="x unified",
        template="plotly_white",
        font=dict(family="Arial", size=14),
        legend=dict(x=0, y=1, traceorder="normal", orientation="h"),
    )

    st.plotly_chart(fig, use_container_width=True)
