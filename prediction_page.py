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
    col0, col1, col2, col3 = st.columns(4)
    with col0:
        commodity = st.text_input("Commodity", value="Gold")  # change default if needed
    with col1:
        jan = st.number_input("Jan_2025", value=110.0)
    with col2:
        feb = st.number_input("Feb_2025", value=112.0)
    with col3:
        change_feb = st.number_input("Change_Feb", value=0.15)

    if st.button("Predict March 2025"):
        # Must include ALL features exactly as training
        df = pd.DataFrame(
            [[commodity, jan, feb, change_feb]],
            columns=["Commodity", "Jan_2025", "Feb_2025", "Change_Feb"]
        )

        try:
            pred = model.predict(df)[0]
            st.success(f"📈 Predicted Mar_2025 Value: **{pred:.2f}**")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
