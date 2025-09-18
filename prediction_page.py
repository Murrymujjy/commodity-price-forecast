import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = "xgb_commodity_model.joblib"

def show():
    st.header("🔮 Prediction Page")

    try:
        model = joblib.load(MODEL_PATH)
    except:
        st.error("❌ Model not found. Please upload `xgb_commodity_model.joblib`.")
        return

    # Inputs
    commodity = st.text_input("Commodity", "Vegetable products")
    base = st.number_input("Base", value=100.0)
    jan = st.number_input("Jan_2025", value=110.0)
    feb = st.number_input("Feb_2025", value=112.0)
    change_feb = st.number_input("Change_Feb", value=0.15)

    if st.button("Predict March 2025"):
        df = pd.DataFrame(
            [[commodity, base, jan, feb, change_feb]],
            columns=["Commodity", "Base", "Jan_2025", "Feb_2025", "Change_Feb"]
        )

        # Convert Commodity to category dtype (important!)
        df["Commodity"] = df["Commodity"].astype("category")

        try:
            pred = model.predict(df)[0]
            st.success(f"📈 Predicted Mar_2025 Value: **{pred:.2f}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
