import streamlit as st
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

MODEL_PATH = "xgb_commodity_model.joblib"

def show():
    st.header("💡 Explainability Page")

    try:
        model = joblib.load(MODEL_PATH)
    except:
        st.error("❌ Model not found.")
        return

    # Feature importance
    st.subheader("📌 Feature Importance")
    importances = model.feature_importances_
    features = ["Jan_2025", "Feb_2025", "Change_Feb"]

    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=features, palette="viridis", ax=ax)
    ax.set_title("XGBoost Feature Importances")
    st.pyplot(fig)
