# streamlit_app.py
import streamlit as st
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    try:
        # Try reading CSV with headers
        df = pd.read_csv("commodity_data.csv", header=0)
        # Detect if all column names are Unnamed -> no header
        if all(col.startswith("Unnamed") for col in df.columns):
            # Assign proper column names
            column_names = ["Commodity", "Jan_2025", "Feb_2025", "Mar_2025", "Change_Feb", "Change_Mar"]
            df = pd.read_csv("commodity_data.csv", header=None, names=column_names)
        return df
    except FileNotFoundError:
        st.error("CSV file not found! Upload 'commodity_data.csv' in the app folder.")
        return pd.DataFrame()  # empty DataFrame to prevent further errors

df = load_data()

# ---------------- STREAMLIT CONFIG ----------------
st.set_page_config(page_title="📊 Commodity Dashboard", layout="wide")

# ---------------- SIDEBAR ----------------
with st.sidebar:
    choice = option_menu(
        "Navigation",
        ["Home", "Prediction", "Insights", "Explainability", "Forecasting", "Chatbot"],
        icons=["house", "graph-up", "bar-chart", "lightbulb", "calendar", "chat-dots"],
        menu_icon="cast",
        default_index=0,
    )

# ---------------- HOME PAGE ----------------
# ---------------- HOME PAGE ----------------
if choice == "Home":
    st.title("📊 Commodity Price Dashboard")
    st.write("Welcome! Navigate using the sidebar to explore insights, forecasts, predictions, and explanations.")

    if df.empty:
        st.warning("No data to display. Please check your CSV file.")
    else:
        # Clean column names
        df.columns = df.columns.str.strip().str.replace(" ", "_")
        st.write("Columns detected:", df.columns.tolist())
        st.dataframe(df.head())

        required_cols = ["Commodity", "Jan_2025", "Feb_2025", "Mar_2025", "Change_Feb", "Change_Mar"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns in CSV: {missing_cols}")
        else:
            # ---------------- LINE PLOT ----------------
            st.subheader("📈 Commodity Price Index Trends (Jan–Mar 2025)")
            fig, ax = plt.subplots(figsize=(12, 6))
            for commodity in df["Commodity"].unique():
                subset = df[df["Commodity"] == commodity]
                if subset.empty:
                    continue
                # Take first row values to match x-axis
                y_values = subset[["Jan_2025", "Feb_2025", "Mar_2025"]].iloc[0].values
                ax.plot(
                    ["Jan_2025", "Feb_2025", "Mar_2025"],
                    y_values,
                    marker="o",
                    label=commodity
                )
            ax.set_xlabel("Month")
            ax.set_ylabel("Index Value")
            ax.set_title("Commodity Price Index Trends (Jan–Mar 2025)")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            st.pyplot(fig)

            # ---------------- AGGREGATED BAR PLOTS ----------------
st.subheader("📊 Percentage Change: Jan → Feb 2025")
# Convert to numeric
for col in ["Change_Feb", "Change_Mar"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
# Drop rows where both changes are NaN
df = df.dropna(subset=["Change_Feb", "Change_Mar"], how="all")

# Aggregate multiple rows per commodity
df_bar = df.groupby("Commodity")[["Change_Feb", "Change_Mar"]].mean().reset_index()

fig, ax = plt.subplots(figsize=(12, 6))
df_sorted = df_bar.sort_values("Change_Feb", ascending=False)
sns.barplot(x="Change_Feb", y="Commodity", data=df_sorted, palette="viridis", ax=ax)
ax.set_xlabel("Change (%)")
ax.set_ylabel("Commodity")
st.pyplot(fig)

st.subheader("📊 Percentage Change: Feb → Mar 2025")
fig, ax = plt.subplots(figsize=(12, 6))
df_sorted = df_bar.sort_values("Change_Mar", ascending=False)
sns.barplot(x="Change_Mar", y="Commodity", data=df_sorted, palette="magma", ax=ax)
ax.set_xlabel("Change (%)")
ax.set_ylabel("Commodity")
st.pyplot(fig)

            # ---------------- CORRELATION HEATMAP ----------------
            st.subheader("🔗 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(8, 6))
            # Drop Commodity column for correlation
            sns.heatmap(df.drop(columns=["Commodity"]).corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

            # ---------------- CONFUSION MATRIX EXAMPLE ----------------
            st.subheader("📉 Confusion Matrix (Example)")
            y_true = np.random.choice([0, 1], size=20)
            y_pred = np.random.choice([0, 1], size=20)
            cm = confusion_matrix(y_true, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_title("Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)


# ---------------- OTHER PAGES PLACEHOLDERS ----------------
elif choice == "Prediction":
    st.info("Prediction page coming soon!")
elif choice == "Insights":
    st.info("Insights page coming soon!")
elif choice == "Explainability":
    st.info("Explainability page coming soon!")
elif choice == "Forecasting":
    st.info("Forecasting page coming soon!")
elif choice == "Chatbot":
    st.info("Chatbot page coming soon!")
