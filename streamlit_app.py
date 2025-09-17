# streamlit_app.py
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# Load your dataset
@st.cache_data
def load_data():
    return pd.read_csv("commodity_data.csv")

df = load_data()

st.set_page_config(page_title="📊 Commodity Dashboard", layout="wide")

with st.sidebar:
    choice = option_menu(
        "Navigation",
        ["Home", "Prediction", "Insights", "Explainability", "Forecasting", "Chatbot"],
        icons=["house", "graph-up", "bar-chart", "lightbulb", "calendar", "chat-dots"],
        menu_icon="cast",
        default_index=0,
    )

# ---------------- HOME ----------------
if choice == "Home":
    st.title("📊 Commodity Price Dashboard")
    st.write("Welcome! Navigate using the sidebar to explore insights, forecasts, predictions, and explanations.")
    
    # st.title("📊 Commodity Price Dashboard")
    st.write("Explore commodity trends, correlations, and performance insights.")

    # ---------------- LINE PLOTS ----------------
    st.subheader("📈 Commodity Price Index Trends (Jan–Mar 2025)")
    fig, ax = plt.subplots(figsize=(12, 6))
    for commodity in df["Commodity"].unique():
        subset = df[df["Commodity"] == commodity]
        ax.plot(["Jan_2025", "Feb_2025", "Mar_2025"],
                subset[["Jan_2025", "Feb_2025", "Mar_2025"]].values.flatten(),
                marker="o", label=commodity)
    ax.set_title("Commodity Price Trends")
    ax.set_xlabel("Month")
    ax.set_ylabel("Index Value")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig)

    # ---------------- BAR PLOTS ----------------
    st.subheader("📊 Percentage Change: Jan → Feb 2025")
    fig, ax = plt.subplots(figsize=(12, 6))
    df_sorted = df.sort_values("Change_Feb", ascending=False)
    sns.barplot(x="Change_Feb", y="Commodity", data=df_sorted, palette="viridis", ax=ax)
    st.pyplot(fig)

    st.subheader("📊 Percentage Change: Feb → Mar 2025")
    fig, ax = plt.subplots(figsize=(12, 6))
    df_sorted = df.sort_values("Change_Mar", ascending=False)
    sns.barplot(x="Change_Mar", y="Commodity", data=df_sorted, palette="magma", ax=ax)
    st.pyplot(fig)

    # ---------------- CORRELATION HEATMAP ----------------
    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.drop(columns=["Commodity"]).corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ---------------- CONFUSION MATRIX (Example) ----------------
    st.subheader("📉 Confusion Matrix (Example)")
    st.markdown("This is a sample confusion matrix (replace with real model outputs later).")

    y_true = np.random.choice([0, 1], size=20)
    y_pred = np.random.choice([0, 1], size=20)
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

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
