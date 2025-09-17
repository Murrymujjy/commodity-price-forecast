import streamlit as st
from huggingface_hub import InferenceClient

def show():
    st.header("💬 Chatbot Assistant")

    query = st.text_input("Ask me anything about commodities 📊")
    if "HF_TOKEN" in st.secrets:
        client = InferenceClient(token=st.secrets["HF_TOKEN"])
    else:
        client = None

    if st.button("Send"):
        if client:
            try:
                response = client.text_generation("HuggingFaceH4/zephyr-7b-beta", query, max_new_tokens=128)
                st.success(response)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
        else:
            st.info("🤖 (Local Bot) This is a placeholder chatbot. Add your HuggingFace token in `st.secrets` for AI responses.")
            st.write(f"Echo: {query}")
