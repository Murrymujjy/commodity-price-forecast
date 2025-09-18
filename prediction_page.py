import streamlit as st
import pandas as pd
import joblib

# MODEL_PATH = "xgb_commodity_model.joblib"

model = joblib.load("xgb_commodity_model.joblib")
print(model.feature_names_in_)


def show():
    st.header("🔮 Prediction Page")

    try:
        model = joblib.load(model)
    except:
        st.error("❌ Model not found. Please upload `xgb_commodity_model.joblib`.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        jan = st.number_input("Jan_2025", value=110.0)
    with col2:
        feb = st.number_input("Feb_2025", value=112.0)
    with col3:
        change_feb = st.number_input("Change_Feb", value=0.15)

    if st.button("Predict March 2025"):
        df = pd.DataFrame([[jan, feb, change_feb]], columns=["Jan_2025", "Feb_2025", "Change_Feb"])
        pred = model.predict(df)[0]
        st.success(f"📈 Predicted Mar_2025 Value: **{pred:.2f}**")
