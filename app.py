import streamlit as st
import requests
import pandas as pd
import json
import plotly.graph_objects as go
from datetime import datetime
import numpy as np

# --- Configuration for the LLM API call
LLM_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key="
LLM_API_KEY = "" # Leave as-is, will be populated by Canvas
FORECAST_API_URL = 'http://127.0.0.1:5000/predict'

# --- Main App Configuration ---
st.set_page_config(
    page_title="Commodity Price Forecast App",
    page_icon="📈",
    layout="wide",
)

st.title("Commodity Price Forecast")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Go to", ["Forecast Dashboard", "Chat with Analyst Bot"])

# --- Function to call the LLM API (for the chatbot) ---
def get_llm_response(prompt_text):
    """Calls the LLM API with the given prompt and returns the text response."""
    if 'forecast_data' not in st.session_state or not st.session_state.forecast_data:
        return "Sorry, I can't provide a data-driven analysis. Please go to the **Forecast Dashboard** page and generate a forecast first."

    forecast_data = st.session_state.forecast_data
    forecasted_price = forecast_data.get('forecasted_price')
    historical_data = pd.DataFrame(forecast_data.get('historical_data', []))

    # Clean up the historical data
    if not historical_data.empty:
        historical_data['date'] = pd.to_datetime(historical_data['date'])
        historical_data.set_index('date', inplace=True)
    
    system_prompt = f"""You are a world-class financial analyst and commodity market expert. Your goal is to provide insightful and conversational analysis based on provided data.
    The current context is a commodity price index forecast.
    The most recent historical data points are:
    - January 2018: {historical_data[historical_data.index == '2018-01-01']['price'].iloc[0]:.2f}
    - January 2025: {historical_data[historical_data.index == '2025-01-01']['price'].iloc[0]:.2f}
    - February 2025: {historical_data[historical_data.index == '2025-02-01']['price'].iloc[0]:.2f}
    - March 2025: {historical_data[historical_data.index == '2025-03-01']['price'].iloc[0]:.2f}
    
    The forecasted price index for April 2025 is: {forecasted_price:.2f}.
    
    Based on this information, provide a concise and helpful response to the user's question. Do not make up data or information not provided here. Do not explicitly mention that you are an AI or bot.
    """

    full_payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "tools": [{"google_search": {}}]
    }

    try:
        response = requests.post(
            LLM_API_URL + LLM_API_KEY,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(full_payload)
        )
        response.raise_for_status()
        response_json = response.json()
        
        if 'candidates' in response_json and response_json['candidates']:
            candidate = response_json['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                return candidate['content']['parts'][0]['text']
    except Exception as e:
        return f"An error occurred while getting a response: {e}"
    
    return "I'm sorry, I couldn't generate a response. Please try again."

# --- Page Rendering Logic ---
if page == "Forecast Dashboard":
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
                request_data = {
                    "date": forecast_date_str,
                    "commodity_group": commodity_group
                }
                response = requests.post(FORECAST_API_URL, json=request_data)
                response.raise_for_status()
                forecast_data = response.json()
                st.session_state['forecast_data'] = forecast_data
            except requests.exceptions.ConnectionError:
                st.error("Connection Error: Please ensure your Flask API is running.")
            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred while making the API request: {e}")
                st.info(f"Response from API: {response.text}")

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
                This forecast is based on an ensemble of sophisticated machine learning models. The prediction of **{forecasted_price:.2f}** suggests the price index is moving in line with recent historical trends.
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        st.header("3. Visualization")
        plot_df = historical_data.copy()
        future_df = pd.DataFrame({'date': [forecast_date], 'price': [forecasted_price]}).set_index('date')
        full_df = pd.concat([plot_df, future_df]).sort_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['price'], mode='lines+markers', name='Historical Data', marker=dict(color='royalblue', size=8), line=dict(width=2)))
        fig.add_trace(go.Scatter(x=future_df.index, y=future_df['price'], mode='markers', name='Forecasted Price', marker=dict(color='darkorange', size=12, symbol='star')))
        last_hist_point = plot_df.iloc[-1]
        fig.add_trace(go.Scatter(x=[last_hist_point.name, future_df.index[0]], y=[last_hist_point['price'], future_df.iloc[0]['price']], mode='lines', name='Forecast Line', line=dict(color='darkorange', width=2, dash='dash')))

        fig.update_layout(title="Commodity Price Index: Historical Data & Forecast", xaxis_title="Date", yaxis_title="Price Index", hovermode="x unified", template="plotly_white", font=dict(family="Arial", size=14), legend=dict(x=0, y=1, traceorder="normal", orientation="h"))
        st.plotly_chart(fig, use_container_width=True)

elif page == "Chat with Analyst Bot":
    st.header("Chat with the Analyst Bot")
    st.markdown("Ask the model questions about the commodity price forecast.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the forecast..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                llm_response = get_llm_response(prompt)
                st.markdown(llm_response)
        
        st.session_state.messages.append({"role": "assistant", "content": llm_response})
