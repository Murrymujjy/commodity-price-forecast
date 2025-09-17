import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

DATA_PATH = "commodity_data.csv"

def show():
    st.header("📊 Insights Page")

    try:
        df = pd.read_csv(DATA_PATH)
    except:
        st.error("❌ Data file not found. Please upload `commodity_data.csv`.")
        return

    # Line chart
    st.subheader("📈 Trend Over Time")
    fig = px.line(df, x="Commodity", y="Jan_2025", title="Index in Jan_2025")
    st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("📊 Correlation Heatmap")
    corr = df[["Jan_2025", "Feb_2025", "Mar_2025", "Change_Feb", "Change_Mar"]].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
