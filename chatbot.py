import streamlit as st
import json
import requests
import time
from datetime import datetime

# --- Configuration for the LLM API call
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key="
API_KEY = "" # Leave as-is, will be populated by Canvas

# --- Set page config and title
st.set_page_config(layout="wide")
st.title("Chat with the Analyst Bot")
st.markdown("Ask the model questions about the commodity price forecast.")

# --- Helper function to call the LLM API
def get_llm_response(prompt_text):
    """Calls the LLM API with the given prompt and returns the text response."""
    # Ensure forecast data is available
    if 'forecast_data' not in st.session_state or not st.session_state.forecast_data:
        return "Sorry, I can't provide a data-driven analysis. Please go to the **Forecast Dashboard** page and generate a forecast first."

    forecast_data = st.session_state.forecast_data
    forecasted_price = forecast_data.get('forecasted_price')
    historical_data = pd.DataFrame(forecast_data.get('historical_data', []))

    # Construct the system and user prompts
    system_prompt = f"""You are a world-class financial analyst and commodity market expert. Your goal is to provide insightful and conversational analysis based on provided data.
    The current context is a commodity price index forecast.
    The most recent historical data points are:
    - January 2018: {historical_data[historical_data['date'] == '2018-01-01']['price'].iloc[0]:.2f}
    - January 2025: {historical_data[historical_data['date'] == '2025-01-01']['price'].iloc[0]:.2f}
    - February 2025: {historical_data[historical_data['date'] == '2025-02-01']['price'].iloc[0]:.2f}
    - March 2025: {historical_data[historical_data['date'] == '2025-03-01']['price'].iloc[0]:.2f}
    
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
            API_URL + API_KEY,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(full_payload)
        )
        response.raise_for_status()
        
        response_json = response.json()
        
        # Check for candidates and content parts
        if 'candidates' in response_json and response_json['candidates']:
            candidate = response_json['candidates'][0]
            if 'content' in candidate and 'parts' in candidate['content']:
                return candidate['content']['parts'][0]['text']

    except Exception as e:
        return f"An error occurred while getting a response: {e}"
    
    return "I'm sorry, I couldn't generate a response. Please try again."

# --- Initialize chat history and prompt if not already done
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main chat input loop
if prompt := st.chat_input("Ask a question about the forecast..."):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get a response from the LLM
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            llm_response = get_llm_response(prompt)
            st.markdown(llm_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": llm_response})
