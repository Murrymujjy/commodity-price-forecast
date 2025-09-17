import streamlit as st
import pandas as pd
import plotly.express as px

DATA_PATH = "commodity_data.csv"

def show():
    st.header("📅 Forecasting Page")

    try:
        df = pd.read_csv(DATA_PATH)
    except:
        st.error("❌ Data file not found.")
        return

    st.write("📊 Forecast March 2025 vs February 2025")
    df["Forecast_Mar"] = df["Feb_2025"] * (1 + df["Change_Feb"]/100)

    fig = px.bar(df, x="Commodity", y=["Feb_2025", "Forecast_Mar"],
                 barmode="group", title="Forecasted vs Actual")
    st.plotly_chart(fig, use_container_width=True)
