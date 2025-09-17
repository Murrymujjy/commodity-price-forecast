import streamlit as st
import requests
import json
import pandas as pd

# --- 1. App Title and Description ---
st.set_page_config(page_title="Commodity Price Forecast", layout="centered")
st.title("Commodity Price Index Forecast")
st.markdown("Enter a date and get a price index forecast powered by a fine-tuned ensemble model.")

# --- 2. User Input ---
# Use a date input widget for user to select a date
forecast_date = st.date_input(
    "Select a date for your forecast:",
    pd.to_datetime("2025-05-01"),
    min_value=pd.to_datetime("2025-04-01")
)

# --- 3. Prediction Button and Logic ---
if st.button("Get Forecast"):
    # Convert the date object to a string in 'YYYY-MM-DD' format
    date_to_predict = forecast_date.strftime('%Y-%m-%d')
    
    # --- Make API Call ---
    # The URL of your locally running Flask API
    api_url = 'http://127.0.0.1:5000/predict'
    
    data = {
        "date": date_to_predict
    }
    
    with st.spinner(f"Getting forecast for {date_to_predict}..."):
        try:
            # Send a POST request to the API with the date
            response = requests.post(api_url, json=data, timeout=10) # 10 seconds timeout
            
            # Check if the request was successful
            if response.status_code == 200:
                result = response.json()
                forecasted_price = result['forecasted_price']
                
                # Display the result
                st.success("Forecast successful!")
                st.markdown(f"**Forecasted Price Index for {date_to_predict}:**")
                st.markdown(f"### **{forecasted_price}**")
                
            else:
                # Handle API errors
                st.error(f"Error from API: Status Code {response.status_code}")
                st.warning("Please ensure your 'api.py' script is running in another terminal.")
                st.json(response.json())
        
        except requests.exceptions.ConnectionError:
            st.error("Connection Error: Could not connect to the API server.")
            st.warning("Please ensure your 'api.py' script is running and that you have a stable internet connection.")
            
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
