# streamlit_app.py
"""
Complete Streamlit app (single-file) for your commodity price project.
Features:
- Sidebar navigation menu (streamlit_option_menu)
- Pages: Prediction, Insights, Explainability, Forecasting
- Main page with all charts (interactive Plotly)
- Load saved model (joblib), load cleaned CSV
- Simple Chatbot panel (Hugging Face InferenceClient if token provided)
- Custom CSS for colorful, professional look

USAGE:
1. Place this file in your project folder alongside:
   - the cleaned CSV (same filename used below) or update the path
   - the saved model file: 'xgb_commodity_model.joblib' (or change name)
2. Install requirements:
   pip install streamlit streamlit-option-menu plotly seaborn scikit-learn xgboost joblib
   (also: huggingface-hub if you want chatbot via Hugging Face)
3. Run: streamlit run streamlit_app.py

Make sure to populate st.secrets with your Hugging Face token if you want the chatbot to call the HF inference API:

[secrets]
HF_TOKEN = "your_huggingface_api_token_here"

"""

import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from io import StringIO
import base64

# Optional: Hugging Face chatbot client
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# ------------------ App config & CSS ------------------
st.set_page_config(page_title="Commodity Insights", page_icon="📊", layout="wide")

# Custom CSS for colorful professional style
st.markdown("""
<style>
/* body */
.main-svg {background: linear-gradient(135deg, #f6f9ff 0%, #ffffff 100%);} 
[data-testid="stSidebar"] {background: linear-gradient(180deg, #0f172a, #0b1220); color: white}
[data-testid="stHeader"] {background: linear-gradient(90deg,#7c3aed,#06b6d4);} 
.css-1aumxhk {padding-top: 1rem;} /* small top padding */
h1, h2, h3 {font-family: 'Segoe UI', Roboto, sans-serif;}
.stButton>button {border-radius:10px; padding: 8px 16px}
.card {background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(250,250,255,0.9)); border-radius:12px; padding:16px; box-shadow: 0 6px 18px rgba(0,0,0,0.08);} 
.small-muted {color: #6b7280; font-size: 0.9rem}
</style>
""", unsafe_allow_html=True)

# ------------------ Helper functions ------------------
@st.cache_data
def load_data(path):
    # load and ensure numeric types
    df_raw = pd.read_csv(path, skiprows=4)
    # drop unnamed junk if present
    drop_cols = [c for c in df_raw.columns if c.startswith('Unnamed: 0') or c.strip()=='' ]
    df_raw = df_raw.drop(columns=drop_cols, errors='ignore')
    # rename columns if present
    # Typical structure: ['Commodity','Base','Jan_2025','Feb_2025','Mar_2025','Change_Feb','Change_Mar']
    # If header already set from file, we try best-effort mapping
    cols = list(df_raw.columns)
    # if first row is header label leftover, drop it
    if str(df_raw.iloc[0,0]).strip().lower().startswith('commodity'):
        df_raw = df_raw.drop(index=0).reset_index(drop=True)
        cols = list(df_raw.columns)
    # attempt rename based on length
    if len(cols) == 8 and cols[0].startswith('Unnamed'):
        df_raw = df_raw.drop(columns=[cols[0]])
        cols = list(df_raw.columns)
    # final rename if matches 7
    if len(cols) == 7:
        df_raw.columns = ['Commodity','Base','Jan_2025','Feb_2025','Mar_2025','Change_Feb','Change_Mar']
    elif len(cols) == 6:
        df_raw.columns = ['Commodity','Jan_2025','Feb_2025','Mar_2025','Change_Feb','Change_Mar']

    # drop Base if present
    if 'Base' in df_raw.columns:
        df_raw = df_raw.drop(columns=['Base'])

    # convert numeric columns
    num_cols = [c for c in df_raw.columns if c not in ['Commodity']]
    df_raw[num_cols] = df_raw[num_cols].apply(pd.to_numeric, errors='coerce')
    df_raw = df_raw.dropna().reset_index(drop=True)
    return df_raw

@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        return None

# Small utility to create download link for dataframe
def get_table_download_link(df, filename="data.csv"):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href

# Chatbot helper (uses Hugging Face InferenceClient if token available)
def call_hf_chat(prompt: str):
    token = st.secrets.get('HF_TOKEN') if 'HF_TOKEN' in st.secrets else None
    if not token:
        return "Hugging Face token not found in st.secrets. Please add HF_TOKEN to use the chatbot."
    if not HF_AVAILABLE:
        return "huggingface_hub not installed. Install huggingface_hub to enable API chatbot."
    try:
        client = InferenceClient(token=token)
        # use a text-generation model or conversational model identifier
        resp = client.text_generation(prompt, max_new_tokens=200)
        # response may be a list/dict - be tolerant
        if isinstance(resp, (list,tuple)):
            return resp[0].get('generated_text', str(resp[0]))
        if isinstance(resp, dict):
            return resp.get('generated_text', str(resp))
        return str(resp)
    except Exception as e:
        return f"Error calling HF inference: {e}"

# ------------------ Load resources ------------------
DATA_PATH = "Q1_2025_Commodity_Price_Indices_and_Terms_of_Trade_Tables_23062025.csv"
MODEL_PATH = "xgb_commodity_model.joblib"

try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Data file not found at {DATA_PATH}. Upload your CSV or update DATA_PATH in the script.")
    df = pd.DataFrame()

model = load_model(MODEL_PATH)
if model is None:
    st.warning("Model not found or failed to load. Prediction functionality will be disabled until you save your model to 'xgb_commodity_model.joblib'.")

# Prepare long format for plots
@st.cache_data
def to_long(df_local):
    long = df_local.melt(id_vars=['Commodity'], value_vars=['Jan_2025','Feb_2025','Mar_2025'], var_name='Month', value_name='Index_Value')
    # clean month labels
    long['Month_clean'] = long['Month'].str.replace('_2025','')
    return long

df_long = to_long(df) if not df.empty else pd.DataFrame()

# ------------------ Sidebar Navigation ------------------
with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=["Home", "Prediction", "Insights", "Explainability", "Forecasting", "Chatbot"],
        icons=["house", "play", "bar-chart-steps", "info-circle", "calendar2-week", "chat-dots"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#0b1220"},
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px 0px"},
            "nav-link-selected": {"background-color": "#0f172a"},
        }
    )

# ------------------ Page: HOME (All graphs) ------------------
if selected == "Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.title("📊 Commodity Price Dashboard")
    st.markdown("""
    #### Overview
    This dashboard shows Jan–Mar 2025 import price indices for commodity groups. Use the side menu to explore prediction, insights, explainability and forecasting features.
    """)

    # Top KPIs
    col1, col2, col3 = st.columns([1,1,1])
    if not df.empty:
        avg_change_feb = (df['Change_Feb'].mean())
        avg_change_mar = (df['Change_Mar'].mean())
        most_volatile = df.set_index('Commodity')[['Change_Feb','Change_Mar']].abs().sum(axis=1).idxmax()
        col1.metric("Average % change Jan→Feb", f"{avg_change_feb:.3f}")
        col2.metric("Average % change Feb→Mar", f"{avg_change_mar:.3f}")
        col3.metric("Most volatile commodity", most_volatile)

    st.markdown("</div>", unsafe_allow_html=True)

    # Layout: left plots and right controls
    left, right = st.columns((3,1))

    with left:
        st.subheader("Trend Lines — All Commodities")
        if not df_long.empty:
            fig = px.line(df_long, x='Month_clean', y='Index_Value', color='Commodity', markers=True, title='Commodity Price Index Trends (Jan–Mar 2025)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info('No data to show')

        st.subheader("Heatmap — Index Values")
        if not df_long.empty:
            pivot = df_long.pivot(index='Commodity', columns='Month_clean', values='Index_Value')
            fig2 = px.imshow(pivot, text_auto='.1f', aspect='auto', color_continuous_scale='YlGnBu', title='Commodity × Month Heatmap')
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Bar: Percent Changes")
        if not df.empty:
            change_df = df[['Commodity','Change_Feb','Change_Mar']].set_index('Commodity')
            change_df = change_df.reset_index().melt(id_vars='Commodity', var_name='Period', value_name='Change')
            fig3 = px.bar(change_df, x='Change', y='Commodity', color='Period', barmode='group', title='Percent changes by commodity')
            st.plotly_chart(fig3, use_container_width=True)

    with right:
        st.subheader('Controls')
        st.markdown('Filter commodities:')
        commodities = df['Commodity'].unique().tolist()
        sel = st.multiselect('Select commodities', options=commodities, default=commodities[:6])
        if sel:
            filtered = df_long[df_long['Commodity'].isin(sel)]
            st.write(f"Showing {len(sel)} commodities")
            figf = px.line(filtered, x='Month_clean', y='Index_Value', color='Commodity', markers=True)
            st.plotly_chart(figf, use_container_width=True)
        st.markdown(get_table_download_link(df, filename='cleaned_commodity_data.csv'), unsafe_allow_html=True)

# ------------------ Page: PREDICTION ------------------
if selected == "Prediction":
    st.title("🔮 Predict March 2025 Index")
    st.markdown("Use the controls to enter Jan/Feb values and select commodity. The model will predict Mar_2025.")

    with st.form('predict_form'):
        col1, col2, col3 = st.columns(3)
        commodity_choice = col1.selectbox('Commodity', options=df['Commodity'].unique())
        jan_val = col2.number_input('Jan_2025', value=float(df[df['Commodity']==commodity_choice]['Jan_2025'].iloc[0]))
        feb_val = col3.number_input('Feb_2025', value=float(df[df['Commodity']==commodity_choice]['Feb_2025'].iloc[0]))
        change_feb_val = col1.number_input('Change_Feb (optional)', value=float(df[df['Commodity']==commodity_choice]['Change_Feb'].iloc[0]))
        submitted = st.form_submit_button('Predict')

    if submitted:
        if model is None:
            st.error('Model not available. Ensure xgb_commodity_model.joblib exists in app directory.')
        else:
            # Prepare input (encode commodity similarly to training: use categorical codes)
            # We'll create a small helper mapping
            cat_mapping = {v:i for i,v in enumerate(df['Commodity'].astype('category').cat.categories)}
            try:
                cat_code = df['Commodity'].astype('category').cat.codes[df['Commodity']==commodity_choice].iloc[0]
            except Exception:
                # fallback mapping
                cat_code = 0
            X_new = pd.DataFrame([[cat_code, jan_val, feb_val, change_feb_val]], columns=['Commodity','Jan_2025','Feb_2025','Change_Feb'])
            pred = model.predict(X_new)[0]
            st.metric('Predicted Mar_2025', f"{pred:.3f}")

# ------------------ Page: INSIGHTS ------------------
if selected == "Insights":
    st.title('📈 Insights')
    st.markdown('Data-driven insights and summary statistics')

    if not df.empty:
        st.subheader('Top movers (Jan→Mar)')
        df['TotalChange'] = (df['Change_Feb'].abs() + df['Change_Mar'].abs())
        movers = df.sort_values('TotalChange', ascending=False).head(6)[['Commodity','Change_Feb','Change_Mar']]
        st.table(movers)

        st.subheader('Distribution of Index Values')
        fig = px.box(df_long, x='Month_clean', y='Index_Value', title='Index distribution by month')
        st.plotly_chart(fig, use_container_width=True)

# ------------------ Page: EXPLAINABILITY ------------------
if selected == "Explainability":
    st.title('🧠 Explainability')
    st.markdown('Model feature importance and (if available) SHAP explanations')

    if model is None:
        st.info('No trained model loaded — save a model as xgb_commodity_model.joblib to view explainability.')
    else:
        # Feature importance (Plotly)
        try:
            feat_names = ['Commodity','Jan_2025','Feb_2025','Change_Feb']
            importances = model.feature_importances_
            imp_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances}).sort_values('Importance', ascending=True)
            fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title='Model Feature Importance')
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f'Could not compute feature importance: {e}')

        # Optional: SHAP (only if shap installed and model supports it)
        try:
            import shap
            explainer = shap.Explainer(model)
            # build small sample
            sample = df.sample(min(10, len(df))).copy()
            sample_encoded = sample.copy()
            sample_encoded['Commodity'] = sample_encoded['Commodity'].astype('category').cat.codes
            X_sample = sample_encoded[['Commodity','Jan_2025','Feb_2025','Change_Feb']]
            shap_values = explainer(X_sample)
            st.subheader('SHAP summary (sample)')
            st.pyplot(shap.plots.beeswarm(shap_values, show=False))
        except Exception as e:
            st.info('SHAP not available or failed: ' + str(e))

# ------------------ Page: FORECASTING ------------------
if selected == "Forecasting":
    st.title('📅 Forecasting / What-if Analysis')
    st.markdown('Simple what-if experiments: change Jan/Feb and see predicted Mar index for a commodity')

    commodity_choice = st.selectbox('Pick commodity for forecasting', df['Commodity'].unique())
    jan_slider = st.slider('Jan_2025 value', float(df['Jan_2025'].min()), float(df['Jan_2025'].max()), float(df[df['Commodity']==commodity_choice]['Jan_2025'].iloc[0]))
    feb_slider = st.slider('Feb_2025 value', float(df['Feb_2025'].min()), float(df['Feb_2025'].max()), float(df[df['Commodity']==commodity_choice]['Feb_2025'].iloc[0]))
    change_input = st.number_input('Change_Feb', value=float(df[df['Commodity']==commodity_choice]['Change_Feb'].iloc[0]))

    if st.button('Run What-if'):
        if model is None:
            st.error('No model loaded')
        else:
            try:
                cat_code = df['Commodity'].astype('category').cat.codes[df['Commodity']==commodity_choice].iloc[0]
            except Exception:
                cat_code = 0
            X_new = pd.DataFrame([[cat_code, jan_slider, feb_slider, change_input]], columns=['Commodity','Jan_2025','Feb_2025','Change_Feb'])
            pred = model.predict(X_new)[0]
            st.success(f'Predicted Mar_2025: {pred:.3f}')

# ------------------ Page: CHATBOT ------------------
if selected == "Chatbot":
    st.title('💬 Chatbot')
    st.markdown('Ask questions about the data, the model, or commodity trends.')

    chat_input = st.text_area('Enter your question', value='What commodity had the highest percent increase from Jan to Mar?')
    if st.button('Ask'):
        if HF_AVAILABLE and 'HF_TOKEN' in st.secrets:
            with st.spinner('Calling Hugging Face...'):
                answer = call_hf_chat(chat_input)
                st.write('**Bot:**', answer)
        else:
            # Simple local fallback - basic answers from the dataset
            q = chat_input.lower()
            if 'highest' in q and 'increase' in q:
                idx = (df['Change_Feb'] + df['Change_Mar']).idxmax()
                st.write(f"The highest total percent change (Jan->Mar) was {df.loc[idx,'Commodity']} with total change {df.loc[idx,'Change_Feb'] + df.loc[idx,'Change_Mar']:.3f}")
            else:
                st.write('Sorry — chatbot backend not configured. Add HF_TOKEN to st.secrets for a smarter bot.')

# ------------------ Footer ------------------
st.markdown("---")
st.markdown('Made with ❤️ by your data assistant — Streamlit | XGBoost | Plotly')
