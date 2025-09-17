import streamlit as st
import time

st.set_page_config(layout="wide")
st.title("Chat with the Model")
st.markdown("Ask questions about the commodity price forecast to get instant insights.")

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
    st.session_message(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # --- Simulate a response from a language model (replace with actual API call) ---
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Placeholder for your LLM call (e.g., Gemini API)
            # You would replace this with an actual API call to your LLM
            response = f"I'm an expert analyst. You asked: '{prompt}'. Based on the data, I can provide you with an analysis. Please connect me to your forecasting model."
            time.sleep(2) # Simulate delay
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
