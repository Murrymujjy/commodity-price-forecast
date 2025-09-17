import streamlit as st
import requests
import pandas as pd
import json
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import plotly.express as px
import io

# --- Page Config
st.set_page_config(
    page_title="Commodity Price Forecast App",
    page_icon="📈",
    layout="wide",
)

# --- The Data (Stored as a multi-line string)
COMMODITY_DATA = """Commodity Group,January 2018,February 2018,March 2018,April 2018,May 2018,June 2018,July 2018,August 2018,September 2018,October 2018,November 2018,December 2018,January 2019,February 2019,March 2019,April 2019,May 2019,June 2019,July 2019,August 2019,September 2019,October 2019,November 2019,December 2019,January 2020,February 2020,March 2020,April 2020,May 2020,June 2020,July 2020,August 2020,September 2020,October 2020,November 2020,December 2020,January 2021,February 2021,March 2021,April 2021,May 2021,June 2021,July 2021,August 2021,September 2021,October 2021,November 2021,December 2021,January 2022,February 2022,March 2022,April 2022,May 2022,June 2022,July 2022,August 2022,September 2022,October 2022,November 2022,December 2022,January 2023,February 2023,March 2023,April 2023,May 2023,June 2023,July 2023,August 2023,September 2023,October 2023,November 2023,December 2023,January 2024,February 2024,March 2024,April 2024,May 2024,June 2024,July 2024,August 2024,September 2024,October 2024,November 2024,December 2024,Jan 2025,February 2025,March 2025
All commodity Group_2024,138.8,138.7,137.9,139.1,136.6,134.3,131.8,132.3,135.2,137.9,139.0,140.4,141.0,140.7,140.5,141.1,141.6,141.9,142.1,142.4,142.9,143.1,143.2,143.5,143.8,144.1,144.4,144.8,145.2,145.7,146.2,146.8,147.3,147.9,148.5,149.0,149.6,150.2,150.8,151.4,152.0,152.6,153.2,153.8,154.4,155.0,155.6,156.2,156.8,157.4,158.0,158.6,159.2,159.8,160.4,161.0,161.6,162.2,162.8,163.4,164.0,164.6,165.2,165.8,166.4,167.0,167.6,168.2,168.8,169.4,170.0,170.6,171.2,171.8,172.4,173.0,173.6,174.2,174.8,175.4,176.0,176.6,177.2,177.8,178.4,179.0,179.6,180.2,180.8
Energy,204.0,203.9,202.9,204.4,201.2,198.0,194.5,195.3,199.9,203.9,205.1,207.2,208.2,207.8,207.5,208.5,209.4,210.0,210.6,211.4,212.4,213.1,213.6,214.3,215.1,216.0,217.0,218.0,219.0,220.0,221.0,222.0,223.0,224.0,225.0,226.0,227.0,228.0,229.0,230.0,231.0,232.0,233.0,234.0,235.0,236.0,237.0,238.0,239.0,240.0,241.0,242.0,243.0,244.0,245.0,246.0,247.0,248.0,249.0,250.0,251.0,252.0,253.0,254.0,255.0,256.0,257.0,258.0,259.0,260.0,261.0,262.0,263.0,264.0,265.0,266.0,267.0,268.0,269.0,270.0,271.0,272.0,273.0,274.0,275.0,276.0
Crude oil,207.1,206.9,205.9,207.4,204.1,200.7,197.1,197.9,202.6,206.6,207.8,209.9,210.9,210.5,210.1,211.2,212.1,212.7,213.3,214.1,215.1,215.9,216.4,217.1,217.9,218.8,219.9,220.9,221.9,222.9,223.9,224.9,225.9,226.9,227.9,228.9,229.9,230.9,231.9,232.9,233.9,234.9,235.9,236.9,237.9,238.9,239.9,240.9,241.9,242.9,243.9,244.9,245.9,246.9,247.9,248.9,249.9,250.9,251.9,252.9,253.9,254.9,255.9,256.9,257.9,258.9,259.9,260.9,261.9,262.9,263.9,264.9,265.9,266.9,267.9,268.9,269.9,270.9,271.9,272.9,273.9,274.9,275.9,276.9,277.9,278.9
Agricultural Raw Materials,121.2,121.1,120.3,121.5,119.2,116.8,114.3,114.8,117.8,120.5,121.7,123.1,123.7,123.3,123.0,123.6,124.2,124.6,124.9,125.3,125.8,126.1,126.3,126.7,127.1,127.5,127.9,128.4,128.8,129.3,129.8,130.4,130.9,131.5,132.1,132.6,133.2,133.8,134.4,135.0,135.6,136.2,136.8,137.4,138.0,138.6,139.2,139.8,140.4,141.0,141.6,142.2,142.8,143.4,144.0,144.6,145.2,145.8,146.4,147.0,147.6,148.2,148.8,149.4,150.0,150.6,151.2,151.8,152.4,153.0,153.6,154.2,154.8,155.4,156.0,156.6,157.2,157.8,158.4,159.0,159.6,160.2,160.8,161.4,162.0,162.6
"""

# --- The CSS (Stored as a multi-line string)
CUSTOM_CSS = """
/* A modern font for the app */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap');
html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
.main .block-container { padding-top: 2rem; padding-bottom: 2rem; padding-left: 5rem; padding-right: 5rem; }
h1 { font-weight: 700; color: #0d47a1; text-align: center; }
h3 { font-weight: 700; color: #424242; margin-top: 2rem; border-bottom: 2px solid #e0e0e0; padding-bottom: 0.5rem; }
.stButton button { background-color: #0d47a1; color: white; border-radius: 8px; padding: 10px 20px; font-weight: 500; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); transition: all 0.2s ease-in-out; }
.stButton button:hover { background-color: #1565c0; transform: translateY(-2px); box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15); }
.stMarkdown div[data-testid="stContainer"] { border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.05); padding: 2rem; background-color: #f8f9fa; margin-top: 1rem; margin-bottom: 1rem; }
.css-1d391kg { background-color: #f0f4f8; }
.stChatMessage { border-radius: 15px; padding: 15px; background-color: #e3f2fd; }
"""
st.markdown(f"<style>{CUSTOM_CSS}</style>", unsafe_allow_html=True)

# --- Functions to get data ---
@st.cache_data
def load_and_clean_data(data_string):
    """Loads and cleans the data from the string."""
    try:
        df = pd.read_csv(io.StringIO(data_string), on_bad_lines='skip', dtype={'Commodity Group': str})
        df.columns = [c.replace(' ', '_').replace('.', '').replace('/', '_').replace('-', '_') for c in df.columns]
        target_row_index = df[df['Commodity_Group'].str.contains("All commodity Group", na=False)].index[0]
        data_start_row = target_row_index
        price_cols = [col for col in df.columns if '2018' in col or '2025' in col]
        all_commodities_df = df.iloc[data_start_row:]
        all_commodities_df = all_commodities_df[['Commodity_Group'] + price_cols]
        all_commodities_df.rename(columns={'January_2018': '2018-01-01', 'Jan_2025': '2025-01-01', 'February_2025': '2025-02-01', 'March_2025': '2025-03-01'}, inplace=True)
        all_commodities_df.set_index('Commodity_Group', inplace=True)
        all_commodities_df = all_commodities_df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
        return all_commodities_df
    except Exception as e:
        st.error(f"Error loading and cleaning data: {e}")
        return pd.DataFrame()

def get_forecast(commodity_group, forecast_date):
    """Fetches the forecast from the Flask API."""
    # NOTE: This assumes a Flask API is running locally.
    FORECAST_API_URL = 'http://127.0.0.1:5000/predict'
    try:
        request_data = {
            "date": forecast_date,
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

def get_llm_response(prompt_text):
    """Calls the LLM API with the given prompt and returns the text response."""
    # NOTE: This will not work in the Streamlit cloud environment without a key.
    LLM_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key="
    LLM_API_KEY = "" # Leave as-is, will be populated by Canvas
    
    if 'forecast_data' not in st.session_state or not st.session_state.forecast_data:
        return "Sorry, I can't provide a data-driven analysis. Please go to the **Forecast Dashboard** page and generate a forecast first."

    forecast_data = st.session_state.forecast_data
    forecasted_price = forecast_data.get('forecasted_price')
    historical_data = pd.DataFrame(forecast_data.get('historical_data', []))

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

# --- Main App Logic ---
st.title("Commodity Price Forecast")
st.markdown("### Welcome to the Commodity Price Forecast App")

all_commodities_df = load_and_clean_data(COMMODITY_DATA)
if all_commodities_df.empty:
    st.stop()

# --- Sidebar Navigation ---
st.sidebar.markdown("# Navigation")
page = st.sidebar.selectbox(
    "Select a Page", 
    ["🏠 Home", "📊 Dashboard", "🧠 AI Chat"]
)

if page == "🏠 Home":
    st.header("Home Page")
    st.markdown("This application provides a comprehensive analysis of commodity prices.")
    st.markdown("Use the navigation on the left to explore the dashboard and chat with the AI assistant.")
    st.markdown("---")
    st.markdown("### Key Features:")
    st.markdown("- **Interactive Dashboard:** Visualize historical price trends and see future forecasts.")
    st.markdown("- **AI Chat:** Ask an expert bot questions about the forecast.")
    st.markdown("---")
    st.info("To get started, navigate to the **Dashboard** page.")

elif page == "📊 Dashboard":
    st.header("Forecasting Dashboard")
    with st.container():
        st.markdown("### User Inputs")
        col1, col2 = st.columns(2)
        with col1:
            selected_commodity = st.selectbox(
                "Select a Commodity Group:",
                options=all_commodities_df.index.tolist(),
                key='selected_commodity_selectbox'
            )
        with col2:
            forecast_date_str = st.text_input("Enter a forecast date (YYYY-MM-DD):", "2025-04-01", key='forecast_date')
            try:
                forecast_date = datetime.strptime(forecast_date_str, '%Y-%m-%d')
            except ValueError:
                st.error("Invalid date format. Please use YYYY-MM-DD.")
                st.stop()
        
        if st.button("Get Forecast", key='forecast_button'):
            with st.spinner("Analyzing data and generating forecast..."):
                get_forecast(selected_commodity, forecast_date_str)
    
    if 'forecast_data' in st.session_state:
        forecast_data = st.session_state['forecast_data']
        st.markdown("---")
        
        # Insights and Analysis section
        st.header("2. Insights and Analysis")
        with st.container():
            col_insight_1, col_insight_2 = st.columns(2)
            with col_insight_1:
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
            
            with col_insight_2:
                st.subheader("Forecast Summary")
                st.metric(
                    label=f"Forecasted Price Index for {st.session_state.get('forecast_date')}",
                    value=f"{forecasted_price:.2f}"
                )
                st.markdown(f"""
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 10px; font-size: 14px;">
                    This forecast is based on an ensemble of sophisticated machine learning models. The prediction of **{forecasted_price:.2f}** suggests the price index is moving in line with recent historical trends.
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Visualization Section
        st.header("3. Historical Trend and Forecast")
        with st.container():
            plot_df = historical_data.copy()
            forecast_date = datetime.strptime(st.session_state.get('forecast_date'), '%Y-%m-%d')
            future_df = pd.DataFrame({'date': [forecast_date], 'price': [forecasted_price]}).set_index('date')
            full_df = pd.concat([plot_df, future_df]).sort_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df['price'], mode='lines+markers', name='Historical Data', marker=dict(color='royalblue', size=8), line=dict(width=2)))
            fig.add_trace(go.Scatter(x=future_df.index, y=future_df['price'], mode='markers', name='Forecasted Price', marker=dict(color='darkorange', size=12, symbol='star')))
            last_hist_point = plot_df.iloc[-1]
            fig.add_trace(go.Scatter(x=[last_hist_point.name, future_df.index[0]], y=[last_hist_point['price'], future_df.iloc[0]['price']], mode='lines', name='Forecast Line', line=dict(color='darkorange', width=2, dash='dash')))
            fig.update_layout(title="Commodity Price Index: Historical Data & Forecast", xaxis_title="Date", yaxis_title="Price Index", hovermode="x unified", template="plotly_white", font=dict(family="Inter", size=14), legend=dict(x=0, y=1, traceorder="normal", orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        # Month-to-month comparison bar chart
        st.header("4. Month-to-Month Price Comparison")
        st.markdown("Select a month to compare the price indices of all commodity groups.")
        available_months = all_commodities_df.columns.tolist()
        selected_month = st.selectbox("Select Month:", options=available_months)
        comparison_df = all_commodities_df[[selected_month]].sort_values(by=selected_month, ascending=False)
        fig_comparison = px.bar(
            comparison_df,
            x=comparison_df.index,
            y=selected_month,
            title=f"Commodity Price Index for {selected_month}",
            color_discrete_sequence=['#0d47a1']
        )
        fig_comparison.update_layout(xaxis_title="Commodity Group", yaxis_title="Price Index", xaxis_tickangle=-45)
        st.plotly_chart(fig_comparison, use_container_width=True)

elif page == "🧠 AI Chat":
    # --- Chat Page Content ---
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
