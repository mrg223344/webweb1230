import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- Page Configuration ---
st.set_page_config(
    page_title="Pediatric Myopia Risk Predictor",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional medical interface
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #007bff;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
        border-radius: 8px;
        height: 3.5em;
        font-weight: bold;
        font-size: 1.1em;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #0056b3;
        border-color: #0056b3;
    }
    [data-testid="stTable"] {
        background-color: white;
        border-radius: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_all_assets():
    # Load model components
    model = joblib.load("best_random_forest.pkl")
    scaler = joblib.load("scaler.pkl")
    pt = joblib.load("power_transformer.pkl")
    feature_map = joblib.load("best_feature_map.pkl")
    return model, scaler, pt, feature_map

try:
    best_rf, scaler, pt, best_feature_map = load_all_assets()
except Exception as e:
    st.error(f"⚠️ **Error loading model components:** {e}")
    st.stop()

# --- Header ---
st.title("👁️ Pediatric Myopia Screening System")
st.markdown("##### AI-Assisted Risk Assessment for Children (Ages 6-13)")
st.divider()

# --- Sidebar: Input Features ---
with st.sidebar:
    st.header("📋 Clinical Input")
    st.write("Enter the child's measurements below:")
    
    # Demographics
    age = st.slider("Patient Age", 6, 13, 8)
    grade = st.selectbox("School Grade", options=[1, 2, 3, 4, 5, 6], index=1)
    
    st.markdown("---")
    # Biometric Data
    al = st.number_input("Axial Length (AL) - mm", value=23.50, min_value=15.0, max_value=35.0, step=0.01)
    cr = st.number_input("Corneal Radius (CR) - mm", value=7.80, min_value=5.0, max_value=10.0, step=0.01)
    
    # Calculation
    al_cr_calc = round(al / cr, 4) if cr != 0 else 0
    st.info(f"🔢 **Calculated AL/CR:** `{al_cr_calc}`")
    
    ast = st.number_input("AST (Astigmatism)", value=0.0, step=0.25)
    
    st.markdown("---")
    predict_btn = st.button("RUN DIAGNOSIS")

# --- Main Dashboard ---
col_preview, col_result = st.columns([1, 2], gap="large")

with col_preview:
    st.subheader("📝 Input Summary")
    # Table presentation
    display_df = pd.DataFrame({
        "Parameter": ["Age", "Grade", "AL (mm)", "CR (mm)", "AL/CR Ratio", "AST"],
        "Value": [age, grade, al, cr, al_cr_calc, ast]
    })
    st.table(display_df)
    
    

with col_result:
    if predict_btn:
        with st.spinner('Calculating prediction...'):
            try:
                # Prepare data for model
                raw_input_df = pd.DataFrame({
                    'AL': [al], 
                    'AL/CR': [al_cr_calc], 
                    'age': [age], 
                    'grade': [grade], 
                    'AST': [ast]
                })

                # --- Pre-processing & Yeo-Johnson ---
                final_features_data = {}
                for original_col, target_col_name in best_feature_map.items():
                    val = raw_input_df[[original_col]].values
                    if "_yeojohnson" in target_col_name:
                        transformed_val = pt.transform(val) 
                        final_features_data[target_col_name] = transformed_val.flatten()
                    else:
                        final_features_data[target_col_name] = val.flatten()
                
                selected_features = [best_feature_map[col] for col in ['AL', 'AL/CR', "age", "grade", "AST"]]
                X_input = pd.DataFrame(final_features_data)[selected_features]
                
                # Predict
                X_scaled = scaler.transform(X_input)
                prediction = best_rf.predict(X_scaled)[0]
                proba = best_rf.predict_proba(X_scaled)[0][1]
                
                # --- Result Display ---
                st.subheader("📊 Diagnostic Outcome")
                
                # Result Banner
                if prediction == 1:
                    st.error("### 🚩 Status: Myopia Risk Detected")
                else:
                    st.success("### ✅ Status: Low Myopia Risk")

                # Probability Display
                st.metric("Myopia Risk Probability", f"{proba:.1%}")

                # Visual Progress Bar
                st.write("**Risk Analysis Gradient**")
                st.progress(float(proba))
                
            except Exception as e:
                st.error(f"Prediction Error: {e}")
    else:
        st.info("💡 **Ready to Predict:** Please verify the measurements in the sidebar and click 'RUN DIAGNOSIS' to see the results.")

# --- Footer ---
st.markdown("---")
