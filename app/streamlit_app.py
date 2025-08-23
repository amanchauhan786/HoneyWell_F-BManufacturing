%%writefile app/streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb

# --- Page Configuration ---
st.set_page_config(
    page_title="F&B Anomaly Prediction Dashboard",
    page_icon="🍺",
    layout="wide"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1f77b4;
        margin-bottom: 1.5rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #e0e0e0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .parameter-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stContainer {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1.5rem;
        background-color: white;
    }
    .normal-status {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin-bottom: 1rem;
    }
    .anomaly-status {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin-bottom: 1rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# --- Caching: Load All Models and Data ---
@st.cache_resource
def load_artifacts():
    """
    Loads all specified predictive models, the test data, and creates
    a dictionary of SHAP explainers for compatible models.
    """
    artifacts = {'models': {}, 'explainers': {}}
    try:
        # CHANGED: Use relative paths for GitHub deployment
        models_path = './models/'
        data_path = './data/features/test_dataset.csv'

        # --- FIX: Define all models to load and their display names ---
        models_to_load = {
            "XGBoost": "xgboost.joblib",
            "LightGBM": "lightgbm.joblib",
            "Logistic Regression": "logistic_regression.joblib"
        }

        for display_name, file_name in models_to_load.items():
            model_file = os.path.join(models_path, file_name)
            if os.path.exists(model_file):
                model = joblib.load(model_file)
                artifacts['models'][display_name] = model

                # Only create explainers for tree-based models
                if isinstance(model, (RandomForestClassifier, xgb.XGBClassifier, lgb.LGBMClassifier)):
                    artifacts['explainers'][display_name] = shap.TreeExplainer(model)
            else:
                st.warning(f"Model file not found: {file_name}")

        artifacts['test_df'] = pd.read_csv(data_path)
        return artifacts
    except Exception as e:
        st.error(f"Error during artifact loading: {e}")
        return None

# --- Main Application Logic ---
artifacts = load_artifacts()

st.markdown('<h1 class="main-header">🏭 F&B Process Anomaly Prediction Dashboard</h1>', unsafe_allow_html=True)

if not artifacts or not artifacts['models']:
    st.error("No models were loaded. Please check your file paths and names.")
    st.stop()

# Unpack artifacts
models = artifacts['models']
explainers = artifacts['explainers']
test_df = artifacts['test_df']

# --- Sidebar for User Input ---
st.sidebar.markdown("""
<div style="background-color: #1f77b4; padding: 1rem; border-radius: 10px; margin-bottom: 1.5rem;">
    <h2 style="color: white; margin: 0;">⚙️ Controls</h2>
</div>
""", unsafe_allow_html=True)

model_names = sorted(models.keys())
selected_model_name = st.sidebar.selectbox(
    "Select a Model:",
    model_names
)
selected_batch_id = st.sidebar.selectbox(
    "Select a Batch ID:",
    test_df['Batch_ID'].unique()
)

# Add some info to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**ℹ️ About**")
st.sidebar.info("This dashboard helps predict anomalies in F&B production processes using machine learning models.")

# --- Data and Model Selection ---
model = models[selected_model_name]
explainer = explainers.get(selected_model_name)

batch_data = test_df[test_df['Batch_ID'] == selected_batch_id]
features = [col for col in test_df.columns if col not in ['Batch_ID', 'Brew_Date', 'Quality_Score', 'Quality_Label']]
X_batch = batch_data[features].copy()

bool_cols = X_batch.select_dtypes(include='bool').columns
if not bool_cols.empty:
    X_batch[bool_cols] = X_batch[bool_cols].astype(int)

# --- Model Prediction ---
prediction_proba = model.predict_proba(X_batch)[0]
prediction = model.predict(X_batch)[0]
anomaly_probability = prediction_proba[1]

# --- Dashboard Layout ---
st.markdown(f'<h2 class="section-header">🔍 Analysis for Batch `{selected_batch_id}`</h2>', unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns([1, 1, 1.5])

with col1:
    with st.container():
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #1f77b4;">📈 Predicted Status</h3>', unsafe_allow_html=True)
        if prediction == 0:
            st.markdown('<div class="normal-status"><h4>✅ NORMAL</h4></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="anomaly-status"><h4>🚨 ANOMALY DETECTED</h4></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-card"><h4>Anomaly Confidence Score</h4><h3>{anomaly_probability:.2%}</h3></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col2:
    with st.container():
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #1f77b4;">📋 Key Parameters</h3>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="parameter-card">
        <p><strong>Temperature:</strong> {batch_data['Temperature'].iloc[0]:.2f} °C</p>
        <p><strong>pH Level:</strong> {batch_data['pH_Level'].iloc[0]:.2f}</p>
        <p><strong>Fermentation Time:</strong> {batch_data['Fermentation_Time'].iloc[0]} days</p>
        <p><strong>Efficiency:</strong> {batch_data['Brewhouse_Efficiency'].iloc[0]:.2f} %</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

with col3:
    with st.container():
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #1f77b4;">📊 Quality Score Context</h3>', unsafe_allow_html=True)
        fig, ax = plt.subplots()
        sns.histplot(test_df['Quality_Score'], kde=True, ax=ax, color='skyblue', label='All Batches')
        ax.axvline(batch_data['Quality_Score'].iloc[0], color='red', linestyle='--', lw=2, label=f'Selected Batch ({batch_data["Quality_Score"].iloc[0]:.2f})')
        ax.set_title("Selected Batch vs. Overall Quality Distribution")
        ax.set_xlabel("Quality Score")
        ax.legend()
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown('<h2 class="section-header">🧠 Model Explainability and Feature Analysis</h2>', unsafe_allow_html=True)
exp_col1, exp_col2 = st.columns(2)

with exp_col1:
    with st.container():
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #1f77b4;">❓ Why This Prediction?</h3>', unsafe_allow_html=True)
        if explainer:
            shap_values = explainer.shap_values(X_batch)
            if isinstance(shap_values, list):
                shap_values_for_plot = shap_values[1]
                expected_value_for_plot = explainer.expected_value[1]
            else:
                shap_values_for_plot = shap_values
                expected_value_for_plot = explainer.expected_value

            explanation = shap.Explanation(
                values=shap_values_for_plot[0],
                base_values=expected_value_for_plot,
                data=X_batch.iloc[0],
                feature_names=X_batch.columns.tolist()
            )
            fig, ax = plt.subplots()
            shap.waterfall_plot(explanation, max_display=8, show=False)
            plt.title(f"Feature Impact - {selected_model_name}", fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info(f"SHAP explainability is not available for the '{selected_model_name}' model.")
        st.markdown('</div>', unsafe_allow_html=True)

with exp_col2:
    with st.container():
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)
        st.markdown('<h3 style="color: #1f77b4;">🛰️ Batch Parameter Radar</h3>', unsafe_allow_html=True)
        radar_features = ['Temperature', 'pH_Level', 'Fermentation_Time', 'Brewhouse_Efficiency', 'Alcohol_Content']
        normal_avg = test_df[test_df['Quality_Label'] == 'OK'][radar_features].mean()
        anomaly_avg = test_df[test_df['Quality_Label'] != 'OK'][radar_features].mean()
        batch_values = batch_data[radar_features].iloc[0]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=normal_avg.values, theta=radar_features, fill='toself', name='Avg. Normal Batch', opacity=0.7))
        fig.add_trace(go.Scatterpolar(r=anomaly_avg.values, theta=radar_features, fill='toself', name='Avg. Anomaly Batch', opacity=0.7))
        fig.add_trace(go.Scatterpolar(r=batch_values.values, theta=radar_features, fill='toself', name=f'Batch {selected_batch_id}', opacity=0.9))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True, 
            title="Batch vs. Averages",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Add a footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; padding: 20px;">
        <p>F&B Process Anomaly Prediction Dashboard • Powered by Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)
