import streamlit as st
from streamlit_option_menu import option_menu

# Import all pages
import prediction_page
import insights_page
import explainability_page
import forecasting_page
import chatbot_page

st.set_page_config(page_title="📊 Commodity Dashboard", layout="wide")

# --- Custom CSS ---
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    choice = option_menu(
        "Navigation",
        ["Home", "Prediction", "Insights", "Explainability", "Forecasting", "Chatbot"],
        icons=["house", "graph-up", "bar-chart", "lightbulb", "calendar", "chat-dots"],
        menu_icon="cast",
        default_index=0,
    )

# --- Routing ---
if choice == "Home":
    st.title("📊 Commodity Price Dashboard")
    st.write("Welcome! Navigate using the sidebar to explore insights, forecasts, predictions, and explanations.")
elif choice == "Prediction":
    prediction_page.show()
elif choice == "Insights":
    insights_page.show()
elif choice == "Explainability":
    explainability_page.show()
elif choice == "Forecasting":
    forecasting_page.show()
elif choice == "Chatbot":
    chatbot_page.show()
