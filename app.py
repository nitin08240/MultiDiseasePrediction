import os
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Health Risk Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff0000;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #00ff00;
    }
    .risk-medium {
        background-color: #ffffcc;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffaa00;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Getting the working directory
try:
    working_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    working_dir = os.getcwd()

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = {
        'diabetes': None,
        'heart': None,
        'parkinsons': None,
        'breast': None
    }

if 'user_inputs' not in st.session_state:
    st.session_state.user_inputs = {
        'diabetes': None,
        'heart': None,
        'parkinsons': None,
        'breast': None
    }

# =====================================================================
# MODEL LOADING FUNCTIONS
# =====================================================================

@st.cache_resource
def load_models():
    """Load all pre-trained models from saved_models directory"""
    models = {}

    # Diabetes
    try:
        models['diabetes'] = pickle.load(open(f'{working_dir}/saved_models/diabetes_model.sav', 'rb'))
    except FileNotFoundError:
        st.error("⚠️ Diabetes model file not found in saved_models directory.")
        models['diabetes'] = None
    except Exception as e:
        st.error(f"⚠️ Error loading diabetes model: {str(e)}")
        models['diabetes'] = None

    # Heart
    try:
        models['heart'] = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))
    except FileNotFoundError:
        st.error("⚠️ Heart disease model file not found in saved_models directory.")
        models['heart'] = None
    except Exception as e:
        st.error(f"⚠️ Error loading heart disease model: {str(e)}")
        models['heart'] = None

    # Parkinson's
    try:
        models['parkinsons'] = pickle.load(open(f'{working_dir}/saved_models/parkinsons_model.sav', 'rb'))
    except FileNotFoundError:
        st.error("⚠️ Parkinson's model file not found in saved_models directory.")
        models['parkinsons'] = None
    except Exception as e:
        st.error(f"⚠️ Error loading Parkinson's model: {str(e)}")
        models['parkinsons'] = None

    # Breast Cancer
    try:
        models['breast'] = pickle.load(open(f'{working_dir}/saved_models/breast_model.sav', 'rb'))
    except FileNotFoundError:
        st.error("⚠️ Breast cancer model file not found in saved_models directory.")
        models['breast'] = None
    except Exception as e:
        st.error(f"⚠️ Error loading breast cancer model: {str(e)}")
        models['breast'] = None

    return models

# Load models at startup
MODELS = load_models()

# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def get_risk_category(probability):
    """Convert probability to risk category"""
    if probability < 0.3:
        return "Low Risk", "risk-low", "success"
    elif probability < 0.7:
        return "Medium Risk", "risk-medium", "warning"
    else:
        return "High Risk", "risk-high", "error"


def display_prediction_result(disease_name, prediction, probability, risk_factors=None):
    """Display prediction results in a formatted way"""
    risk_category, css_class, alert_type = get_risk_category(probability)

    st.markdown(f"### 🔍 Prediction Results for {disease_name}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Prediction", "Positive" if prediction == 1 else "Negative")

    with col2:
        st.metric("Risk Probability", f"{probability*100:.1f}%")

    with col3:
        st.metric("Risk Category", risk_category)

    # Display alert based on risk
    if alert_type == "error":
        st.error(f"⚠️ **{risk_category}** - The model indicates elevated risk for {disease_name}. Please consult a healthcare professional.")
    elif alert_type == "warning":
        st.warning(f"⚡ **{risk_category}** - The model shows moderate risk. Consider lifestyle modifications and regular checkups.")
    else:
        st.success(f"✅ **{risk_category}** - The model indicates low risk, but maintain healthy habits.")

    # Risk factors summary
    if risk_factors:
        st.markdown("#### 📋 Key Risk Factors Identified:")
        for factor, value, status in risk_factors:
            if status == "High":
                st.markdown(f"- **{factor}**: {value} ⚠️ *({status})*")
            elif status == "Elevated":
                st.markdown(f"- **{factor}**: {value} ⚡ *({status})*")
            else:
                st.markdown(f"- **{factor}**: {value} ✅ *({status})*")


def create_feature_importance_chart(model, feature_names, title):
    """Create feature importance visualization"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]

        fig = go.Figure(data=[
            go.Bar(
                x=[importances[i] for i in indices],
                y=[feature_names[i] for i in indices],
                orientation='h',
                marker=dict(color='lightblue')
            )
        ])

        fig.update_layout(
            title=title,
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=400,
            yaxis={'categoryorder': 'total ascending'}
        )

        return fig
    return None

# =====================================================================
# HOME / DASHBOARD PAGE
# =====================================================================

def home_page():
    """Main dashboard showing overview and combined risk assessment"""
    st.markdown('<p class="main-header">🏥 AI Health Risk Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Multi-Disease Prediction System using Advanced Machine Learning</p>', unsafe_allow_html=True)

    st.markdown("---")

    # Introduction
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 👋 Welcome to Your Personal Health Risk Assessment Platform
        
        This intelligent system uses state-of-the-art machine learning models to assess your risk for multiple diseases:
        
        - **🩺 Diabetes**  
        - **❤️ Heart Disease**  
        - **🧠 Parkinson's Disease**  
        - **🎗️ Breast Cancer**
        
        #### 🎯 Key Features:
        - Real-time risk prediction with confidence scores  
        - Interactive "What-If" analysis to explore risk reduction strategies  
        - Comprehensive health reports with personalized recommendations  
        - Model explainability and feature importance visualization  
        """)

    with col2:
        st.info("""
        **📊 Example Model Performance (Demo)**  
        - Diabetes: 78% accuracy  
        - Heart Disease: 85% accuracy  
        - Parkinson's: 87% accuracy  
        - Breast Cancer: 95% accuracy (based on classical dataset)
        """)

    st.markdown("---")

    # Combined Risk Dashboard
    st.markdown("### 📊 Your Combined Risk Dashboard")

    disease_keys = ['diabetes', 'heart', 'parkinsons', 'breast']

    predictions_made = any(
        st.session_state.predictions[disease] is not None
        for disease in disease_keys
    )

    disease_labels = {
        'diabetes': 'Diabetes',
        'heart': 'Heart Disease',
        'parkinsons': "Parkinson's",
        'breast': "Breast Cancer"
    }

    if predictions_made:
        diseases = []
        risks = []
        colors = []

        for disease, label in disease_labels.items():
            if st.session_state.predictions[disease] is not None:
                diseases.append(label)
                prob = st.session_state.predictions[disease]['probability']
                risks.append(prob * 100)

                if prob < 0.3:
                    colors.append('#00ff00')
                elif prob < 0.7:
                    colors.append('#ffaa00')
                else:
                    colors.append('#ff0000')

        fig = go.Figure(data=[
            go.Bar(
                x=diseases,
                y=risks,
                marker=dict(color=colors),
                text=[f"{r:.1f}%" for r in risks],
                textposition='auto',
            )
        ])

        fig.update_layout(
            title="Risk Probability by Disease (%)",
            xaxis_title="Disease",
            yaxis_title="Risk Probability (%)",
            yaxis=dict(range=[0, 100]),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📋 Risk Summary")
        cols = st.columns(len(disease_labels))

        for idx, (disease, label) in enumerate(disease_labels.items()):
            with cols[idx]:
                if st.session_state.predictions[disease] is not None:
                    pred_data = st.session_state.predictions[disease]
                    risk_cat, _, _ = get_risk_category(pred_data['probability'])

                    st.markdown(f"""
                    <div class="info-box">
                    <h4>{label}</h4>
                    <p><strong>Risk:</strong> {risk_cat}</p>
                    <p><strong>Probability:</strong> {pred_data['probability']*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info(f"No prediction yet for {label}")
    else:
        st.info("👈 Start by selecting a disease from the sidebar to make your first prediction!")

    st.markdown("---")

    st.warning("""
    **⚠️ Medical Disclaimer**: This application provides educational risk assessments based on machine learning models. 
    It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare 
    professionals for medical decisions.
    """)

# =====================================================================
# DIABETES PREDICTION PAGE
# =====================================================================

def diabetes_page():
    """Diabetes risk prediction page"""
    st.markdown("## 🩺 Diabetes Risk Prediction")
    st.markdown("Assess your risk of Type 2 Diabetes based on clinical parameters")

    if MODELS['diabetes'] is None:
        st.error("Diabetes model not loaded. Please check model file.")
        return

    st.markdown("---")

    with st.form("diabetes_form"):
        st.markdown("### 📝 Enter Your Health Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.number_input(
                "Number of Pregnancies",
                min_value=0, max_value=20, value=1,
                help="Number of times pregnant (0 for males or non-applicable)"
            )

            Glucose = st.number_input(
                "Glucose Level (mg/dL)",
                min_value=0, max_value=300, value=120,
                help="Plasma glucose concentration (Normal: 70-100 fasting)"
            )
            
            BloodPressure = st.number_input(
                "Blood Pressure (mm Hg)",
                min_value=0, max_value=200, value=80,
                help="Diastolic blood pressure (Normal: 60-80)"
            )
                        

        with col2:
                       
            SkinThickness = st.number_input(
                "Skin Thickness (mm)",
                min_value=0, max_value=100, value=20,
                help="Triceps skin fold thickness"
            )
            
            Insulin = st.number_input(
                "Insulin Level (μU/ml)",
                min_value=0, max_value=900, value=80,
                help="2-Hour serum insulin"
            )
            
            BMI = st.number_input(
                "BMI (kg/m²)",
                min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                help="Body Mass Index (Normal: 18.5-24.9)"
            )

          
            

        with col3:
            DiabetesPedigreeFunction = st.number_input(
                "Diabetes Pedigree Function",
                min_value=0.0, max_value=2.5, value=0.5, step=0.01,
                help="Family history indicator (higher = more diabetes in family)"
            )
            
            Age = st.number_input(
                "Age (years)",
                min_value=18, max_value=120, value=30,
                help="Your current age"
            )

        submitted = st.form_submit_button("🔍 Predict Diabetes Risk", use_container_width=True)

    if submitted:
        user_input = [
            Pregnancies, Glucose, BloodPressure, SkinThickness,
            Insulin, BMI, DiabetesPedigreeFunction, Age
        ]

        user_input = [float(x) for x in user_input]

        diab_prediction = MODELS['diabetes'].predict([user_input])[0]

        if hasattr(MODELS['diabetes'], 'predict_proba'):
            probability = MODELS['diabetes'].predict_proba([user_input])[0][1]
        else:
            probability = float(diab_prediction)

        risk_factors = []
        if Glucose > 140:
            risk_factors.append(("Glucose", f"{Glucose} mg/dL", "High"))
        elif Glucose > 100:
            risk_factors.append(("Glucose", f"{Glucose} mg/dL", "Elevated"))
        else:
            risk_factors.append(("Glucose", f"{Glucose} mg/dL", "Normal"))

        if BMI > 30:
            risk_factors.append(("BMI", f"{BMI:.1f}", "High"))
        elif BMI > 25:
            risk_factors.append(("BMI", f"{BMI:.1f}", "Elevated"))
        else:
            risk_factors.append(("BMI", f"{BMI:.1f}", "Normal"))

        if Age > 45:
            risk_factors.append(("Age", f"{Age} years", "Elevated"))
        else:
            risk_factors.append(("Age", f"{Age} years", "Normal"))

        st.session_state.predictions['diabetes'] = {
            'prediction': diab_prediction,
            'probability': probability,
            'timestamp': datetime.now()
        }

        st.session_state.user_inputs['diabetes'] = {
            'Pregnancies': Pregnancies,
            'Glucose': Glucose,
            'BloodPressure': BloodPressure,
            'SkinThickness': SkinThickness,
            'Insulin': Insulin,
            'BMI': BMI,
            'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
            'Age': Age
        }

        display_prediction_result("Diabetes", diab_prediction, probability, risk_factors)

        st.markdown("---")
        st.markdown("### 🔄 What-If Analysis: Explore Risk Reduction")
        st.markdown("Adjust key parameters to see how lifestyle changes could affect your risk")

        col1, col2 = st.columns(2)

        with col1:
            new_glucose = st.slider(
                "Adjust Glucose Level (mg/dL)",
                min_value=70, max_value=200, value=int(Glucose)
            )

            new_bmi = st.slider(
                "Adjust BMI",
                min_value=15.0, max_value=45.0, value=float(BMI), step=0.1
            )

        with col2:
            new_input = [
                Pregnancies, new_glucose, BloodPressure, SkinThickness,
                Insulin, new_bmi, DiabetesPedigreeFunction, Age
            ]
            new_input = [float(x) for x in new_input]

            new_prediction = MODELS['diabetes'].predict([new_input])[0]
            if hasattr(MODELS['diabetes'], 'predict_proba'):
                new_probability = MODELS['diabetes'].predict_proba([new_input])[0][1]
            else:
                new_probability = float(new_prediction)

            st.markdown("#### Adjusted Prediction:")
            risk_cat, _, _ = get_risk_category(new_probability)

            delta = (new_probability - probability) * 100

            st.metric(
                "New Risk Probability",
                f"{new_probability*100:.1f}%",
                f"{delta:+.1f}%"
            )

            st.metric("New Risk Category", risk_cat)

            if delta < -5:
                st.success(
                    f"✅ Great! Reducing glucose to {new_glucose} mg/dL and BMI to {new_bmi:.1f} "
                    f"lowers your risk by {abs(delta):.1f}%"
                )
            elif delta > 5:
                st.warning(f"⚠️ These values increase your risk by {delta:.1f}%")
            else:
                st.info("Minimal change in risk with these adjustments")

# =====================================================================
# HEART DISEASE PREDICTION PAGE
# =====================================================================

def heart_page():
    """Heart disease risk prediction page"""
    st.markdown("## ❤️ Heart Disease Risk Prediction")
    st.markdown("Evaluate your cardiovascular risk using clinical parameters")

    if MODELS['heart'] is None:
        st.error("Heart disease model not loaded. Please check model file.")
        return

    st.markdown("---")

    with st.form("heart_form"):
        st.markdown("### 📝 Enter Your Cardiovascular Parameters")

        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age (years)", min_value=18, max_value=120, value=50)

            sex = st.selectbox(
                "Sex",
                options=[1, 0],
                # format_func=lambda x: "Male" if x == 1 else "Female"
            )

            cp = st.selectbox(
                "Chest Pain Type",
                options=[0, 1, 2, 3],
                # format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"][x],
                help="Type of chest pain experienced"
            )

            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=80, max_value=200, value=120,
                help="Blood pressure at rest (Normal: 120/80)"
            )

            chol = st.number_input(
                "Serum Cholesterol (mg/dL)",
                min_value=100, max_value=600, value=200,
                help="Serum cholesterol (Desirable: <200)"
            )

        with col2:
            fbs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                options=[0, 1],
                # format_func=lambda x: "Yes" if x == 1 else "No"
            )

            restecg = st.selectbox(
                "Resting ECG Results",
                options=[0, 1, 2],
                # format_func=lambda x: ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"][x]
            )

            thalach = st.number_input(
                "Max Heart Rate Achieved",
                min_value=60, max_value=220, value=150,
                help="Maximum heart rate during exercise"
            )

            exang = st.selectbox(
                "Exercise Induced Angina",
                options=[0, 1],
                # format_func=lambda x: "Yes" if x == 1 else "No"
            )

        with col3:
            oldpeak = st.number_input(
                "ST Depression (Oldpeak)",
                min_value=0.0, max_value=10.0, value=1.0, step=0.1,
                help="ST depression induced by exercise"
            )

            slope = st.selectbox(
                "Slope of Peak Exercise ST Segment",
                options=[0, 1, 2],
                # format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x]
            )

            ca = st.selectbox(
                "Number of Major Vessels (0-3)",
                options=[0, 1, 2, 3],
                help="Number of major vessels colored by fluoroscopy"
            )

            thal = st.selectbox(
                "Thalassemia",
                options=[0, 1, 2, 3],
                # format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect", "Unknown"][x]
            )

        submitted = st.form_submit_button("🔍 Predict Heart Disease Risk", use_container_width=True)

    if submitted:
        user_input = [
            age, sex, cp, trestbps, chol, fbs, restecg,
            thalach, exang, oldpeak, slope, ca, thal
        ]

        user_input = [float(x) for x in user_input]

        heart_prediction = MODELS['heart'].predict([user_input])[0]

        if hasattr(MODELS['heart'], 'predict_proba'):
            probability = MODELS['heart'].predict_proba([user_input])[0][1]
        else:
            probability = float(heart_prediction)

        risk_factors = []
        if age > 55:
            risk_factors.append(("Age", f"{age} years", "Elevated"))
        else:
            risk_factors.append(("Age", f"{age} years", "Normal"))

        if chol > 240:
            risk_factors.append(("Cholesterol", f"{chol} mg/dL", "High"))
        elif chol > 200:
            risk_factors.append(("Cholesterol", f"{chol} mg/dL", "Elevated"))
        else:
            risk_factors.append(("Cholesterol", f"{chol} mg/dL", "Normal"))

        if trestbps > 140:
            risk_factors.append(("Blood Pressure", f"{trestbps} mm Hg", "High"))
        elif trestbps > 120:
            risk_factors.append(("Blood Pressure", f"{trestbps} mm Hg", "Elevated"))
        else:
            risk_factors.append(("Blood Pressure", f"{trestbps} mm Hg", "Normal"))

        st.session_state.predictions['heart'] = {
            'prediction': heart_prediction,
            'probability': probability,
            'timestamp': datetime.now()
        }

        st.session_state.user_inputs['heart'] = {
            'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps,
            'chol': chol, 'fbs': fbs, 'restecg': restecg, 'thalach': thalach,
            'exang': exang, 'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
        }

        display_prediction_result("Heart Disease", heart_prediction, probability, risk_factors)

        st.markdown("---")
        st.markdown("### 🔄 What-If Analysis: Lifestyle Modifications")

        col1, col2 = st.columns(2)

        with col1:
            new_chol = st.slider(
                "Adjust Cholesterol (mg/dL)",
                min_value=150, max_value=350, value=int(chol)
            )

            new_trestbps = st.slider(
                "Adjust Resting Blood Pressure (mm Hg)",
                min_value=90, max_value=180, value=int(trestbps)
            )

        with col2:
            new_input = [
                age, sex, cp, new_trestbps, new_chol, fbs, restecg,
                thalach, exang, oldpeak, slope, ca, thal
            ]
            new_input = [float(x) for x in new_input]

            new_prediction = MODELS['heart'].predict([new_input])[0]
            if hasattr(MODELS['heart'], 'predict_proba'):
                new_probability = MODELS['heart'].predict_proba([new_input])[0][1]
            else:
                new_probability = float(new_prediction)

            st.markdown("#### Adjusted Prediction:")
            risk_cat, _, _ = get_risk_category(new_probability)

            delta = (new_probability - probability) * 100

            st.metric(
                "New Risk Probability",
                f"{new_probability*100:.1f}%",
                f"{delta:+.1f}%"
            )

            st.metric("New Risk Category", risk_cat)

            if delta < -5:
                st.success(
                    f"✅ Excellent! Lowering cholesterol to {new_chol} mg/dL and BP to {new_trestbps} mm Hg "
                    f"reduces risk by {abs(delta):.1f}%"
                )
            elif delta > 5:
                st.warning(f"⚠️ These values increase your risk by {delta:.1f}%")
            else:
                st.info("Minimal change in risk with these adjustments")

# =====================================================================
# PARKINSON'S PREDICTION PAGE
# =====================================================================

def parkinsons_page():
    """Parkinson's disease prediction page"""
    st.markdown("## 🧠 Parkinson's Disease Prediction")
    st.markdown("Early detection using voice analysis features")

    if MODELS['parkinsons'] is None:
        st.error("Parkinson's model not loaded. Please check model file.")
        return

    st.info("""
    **About Parkinson's Voice Analysis**: This model uses 22 voice measurements to detect early signs of Parkinson's disease.
    These features are typically extracted from sustained phonation of vowel sounds.
    """)

    st.markdown("---")

    with st.form("parkinsons_form"):
        st.markdown("### 📝 Enter Voice Analysis Parameters")

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.markdown("#### Frequency Features")
            fo = st.number_input(
                "MDVP:Fo(Hz)",
                min_value=80.0, max_value=300.0, value=150.0, step=0.1,
                help="Average vocal fundamental frequency"
            )
            
            fhi = st.number_input(
                "MDVP:Fhi(Hz)",
                min_value=100.0, max_value=400.0, value=200.0, step=0.1,
                help="Maximum vocal fundamental frequency"
            )
            
            flo = st.number_input(
                "MDVP:Flo(Hz)",
                min_value=80.0, max_value=300.0, value=100.0, step=0.1,
                help="Minimum vocal fundamental frequency"
            )

        with col2:
            st.markdown("#### Jitter Features")
            Jitter_percent = st.number_input(
                "MDVP:Jitter(%)",
                min_value=0.0, max_value=1.0, value=0.01, step=0.0001, format="%.4f",
                help="Jitter percentage"
            )
            
            Jitter_Abs = st.number_input(
                "MDVP:Jitter(Abs)",
                min_value=0.0, max_value=0.001, value=0.00005, step=0.00001, format="%.5f",
                help="Absolute jitter"
            )
            
            RAP = st.number_input(
                "MDVP:RAP",
                min_value=0.0, max_value=0.05, value=0.003, step=0.0001, format="%.4f",
                help="Relative Average Perturbation"
            )
            
            PPQ = st.number_input(
                "MDVP:PPQ",
                min_value=0.0, max_value=0.05, value=0.003, step=0.0001, format="%.4f",
                help="Pitch Period Perturbation Quotient"
            )
            
            DDP = st.number_input(
                "Jitter:DDP",
                min_value=0.0, max_value=0.1, value=0.008, step=0.0001, format="%.4f",
                help="Delta Delta Period"
            )

        with col3:
            st.markdown("#### Shimmer Features")
            Shimmer = st.number_input(
                "MDVP:Shimmer",
                min_value=0.0, max_value=0.2, value=0.03, step=0.001, format="%.3f",
                help="Shimmer measurement"
            )
            
            Shimmer_dB = st.number_input(
                "MDVP:Shimmer(dB)",
                min_value=0.0, max_value=5.0, value=0.3, step=0.1,
                help="Shimmer in dB"
            )
            
            APQ3 = st.number_input(
                "Shimmer:APQ3",
                min_value=0.0, max_value=0.1, value=0.015, step=0.001, format="%.3f",
                help="Amplitude Perturbation Quotient (3-period)"
            )
            
            APQ5 = st.number_input(
                "Shimmer:APQ5",
                min_value=0.0, max_value=0.15, value=0.018, step=0.001, format="%.3f",
                help="Amplitude Perturbation Quotient (5-period)"
            )
            
            APQ = st.number_input(
                "MDVP:APQ",
                min_value=0.0, max_value=0.15, value=0.02, step=0.001, format="%.3f",
                help="Amplitude Perturbation Quotient"
            )
            
            DDA = st.number_input(
                "Shimmer:DDA",
                min_value=0.0, max_value=0.1, value=0.025, step=0.001, format="%.3f",
                help="Delta Delta Amplitude"
            )

        with col4:
            st.markdown("#### Noise & Nonlinear")
            NHR = st.number_input(
                "NHR",
                min_value=0.0, max_value=0.5, value=0.02, step=0.001, format="%.3f",
                help="Noise-to-Harmonics Ratio"
            )
            
            HNR = st.number_input(
                "HNR",
                min_value=0.0, max_value=40.0, value=21.0, step=0.1,
                help="Harmonics-to-Noise Ratio"
            )
            
            RPDE = st.number_input(
                "RPDE",
                min_value=0.0, max_value=1.0, value=0.5, step=0.001, format="%.3f",
                help="Recurrence Period Density Entropy"
            )
            
            DFA = st.number_input(
                "DFA",
                min_value=0.0, max_value=2.0, value=0.7, step=0.01,
                help="Detrended Fluctuation Analysis"
            )

        with col5:
            st.markdown("#### Complexity Features")
            spread1 = st.number_input(
                "spread1",
                min_value=-10.0, max_value=0.0, value=-4.0, step=0.1,
                help="Nonlinear measure 1"
            )
            
            spread2 = st.number_input(
                "spread2",
                min_value=0.0, max_value=10.0, value=3.0, step=0.1,
                help="Nonlinear measure 2"
            )
            
            D2 = st.number_input(
                "D2",
                min_value=0.0, max_value=5.0, value=2.0, step=0.1,
                help="Correlation Dimension"
            )
            
            PPE = st.number_input(
                "PPE",
                min_value=0.0, max_value=1.0, value=0.2, step=0.01,
                help="Pitch Period Entropy"
            )

        submitted = st.form_submit_button("🔍 Predict Parkinson's Risk", use_container_width=True)

    if submitted:
        user_input = [
            fo, fhi, flo,
            Jitter_percent, Jitter_Abs,
            RAP, PPQ, DDP,
            Shimmer, Shimmer_dB,
            APQ3, APQ5,
            APQ, DDA,
            NHR, HNR,
            RPDE, DFA,
            spread1, spread2,
            D2, PPE
        ]

        user_input = [float(x) for x in user_input]

        parkinsons_prediction = MODELS['parkinsons'].predict([user_input])[0]

        if hasattr(MODELS['parkinsons'], 'predict_proba'):
            probability = MODELS['parkinsons'].predict_proba([user_input])[0][1]
        else:
            probability = float(parkinsons_prediction)

        risk_factors = []
        if HNR < 20:
            risk_factors.append(("HNR", f"{HNR:.1f}", "High"))
        elif HNR < 25:
            risk_factors.append(("HNR", f"{HNR:.1f}", "Elevated"))
        else:
            risk_factors.append(("HNR", f"{HNR:.1f}", "Normal"))

        if RPDE > 0.6:
            risk_factors.append(("RPDE", f"{RPDE:.3f}", "High"))
        elif RPDE > 0.5:
            risk_factors.append(("RPDE", f"{RPDE:.3f}", "Elevated"))
        else:
            risk_factors.append(("RPDE", f"{RPDE:.3f}", "Normal"))

        if PPE > 0.4:
            risk_factors.append(("PPE", f"{PPE:.2f}", "High"))
        elif PPE > 0.3:
            risk_factors.append(("PPE", f"{PPE:.2f}", "Elevated"))
        else:
            risk_factors.append(("PPE", f"{PPE:.2f}", "Normal"))

        st.session_state.predictions['parkinsons'] = {
            'prediction': parkinsons_prediction,
            'probability': probability,
            'timestamp': datetime.now()
        }

        st.session_state.user_inputs['parkinsons'] = {
            'fo': fo, 'fhi': fhi, 'flo': flo,
            'Jitter_percent': Jitter_percent,
            'Jitter_Abs': Jitter_Abs,
            'RAP': RAP, 'PPQ': PPQ, 'DDP': DDP,
            'Shimmer': Shimmer, 'Shimmer_dB': Shimmer_dB,
            'APQ3': APQ3, 'APQ5': APQ5, 'APQ': APQ, 'DDA': DDA,
            'NHR': NHR, 'HNR': HNR,
            'RPDE': RPDE, 'DFA': DFA,
            'spread1': spread1, 'spread2': spread2,
            'D2': D2, 'PPE': PPE
        }

        display_prediction_result("Parkinson's Disease", parkinsons_prediction, probability, risk_factors)

        st.markdown("---")
        st.markdown("### 🔄 What-If Analysis: Voice Stability")

        col1, col2 = st.columns(2)

        with col1:
            new_HNR = st.slider(
                "Adjust HNR (higher = cleaner voice)",
                min_value=0.0, max_value=40.0, value=float(HNR), step=0.5
            )

            new_RPDE = st.slider(
                "Adjust RPDE (0 - 1)",
                min_value=0.0, max_value=1.0, value=float(RPDE), step=0.01
            )

        with col2:
            new_input = [
                fo, fhi, flo,
                Jitter_percent, Jitter_Abs,
                RAP, PPQ, DDP,
                Shimmer, Shimmer_dB,
                APQ3, APQ5, APQ, DDA,
                NHR, new_HNR,
                new_RPDE, DFA,
                spread1, spread2,
                D2, PPE
            ]
            new_input = [float(x) for x in new_input]

            new_prediction = MODELS['parkinsons'].predict([new_input])[0]
            if hasattr(MODELS['parkinsons'], 'predict_proba'):
                new_probability = MODELS['parkinsons'].predict_proba([new_input])[0][1]
            else:
                new_probability = float(new_prediction)

            st.markdown("#### Adjusted Prediction:")
            risk_cat, _, _ = get_risk_category(new_probability)

            delta = (new_probability - probability) * 100

            st.metric(
                "New Risk Probability",
                f"{new_probability*100:.1f}%",
                f"{delta:+.1f}%"
            )

            st.metric("New Risk Category", risk_cat)

            if delta < -5:
                st.success(
                    f"✅ Improvement! Increasing HNR to {new_HNR:.1f} and adjusting RPDE to {new_RPDE:.2f} "
                    f"reduces the predicted risk by {abs(delta):.1f}%"
                )
            elif delta > 5:
                st.warning(f"⚠️ These values increase the predicted risk by {delta:.1f}%")
            else:
                st.info("Minimal change in risk with these adjustments")

# =====================================================================
# BREAST CANCER PREDICTION PAGE
# =====================================================================

def breast_cancer_page():
    """Breast cancer prediction page"""
    st.markdown("## 🎗️ Breast Cancer Prediction")
    st.markdown("Predict the likelihood of a breast tumor being malignant (cancer) or benign.")

    if MODELS['breast'] is None:
        st.error("Breast cancer model not loaded. Please check model file.")
        return

    st.info("""
    This model is trained on breast tumor cell nucleus measurements (e.g., radius, texture, concavity).
    The prediction is based on the **mean**, **error**, and **worst** (largest) values of these measurements.
    """)

    st.markdown("---")

    with st.form("breast_form"):
        st.markdown("### 📝 Enter Tumor Cell Nucleus Features")

        # ----------------- Mean Values -----------------
        st.markdown("#### 🔹 Mean Values")
        c1, c2, c3 = st.columns(3)

        with c1:
            mean_radius = st.number_input("Mean Radius", min_value=0.0, max_value=40.0, value=14.0, step=0.1)
            mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, max_value=400.0, value=90.0, step=0.5)
            mean_concavity = st.number_input("Mean Concavity", min_value=0.0, max_value=1.0, value=0.05, step=0.005)
            mean_symmetry = st.number_input("Mean Symmetry", min_value=0.0, max_value=1.0, value=0.18, step=0.005)

        with c2:
            mean_texture = st.number_input("Mean Texture", min_value=0.0, max_value=50.0, value=19.0, step=0.1)
            mean_area = st.number_input("Mean Area", min_value=0.0, max_value=3000.0, value=600.0, step=10.0)
            mean_concave_points = st.number_input("Mean Concave Points", min_value=0.0, max_value=1.0, value=0.03, step=0.005)
            mean_fractal_dimension = st.number_input("Mean Fractal Dimension", min_value=0.0, max_value=0.2, value=0.06, step=0.001)

        with c3:
            mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, max_value=1.0, value=0.1, step=0.005)
            mean_compactness = st.number_input("Mean Compactness", min_value=0.0, max_value=1.0, value=0.15, step=0.01)

        # ----------------- Error Values -----------------
        st.markdown("#### 🔹 Error Values")
        e1, e2, e3 = st.columns(3)

        with e1:
            radius_error = st.number_input("Radius Error", min_value=0.0, max_value=10.0, value=0.5, step=0.01)
            perimeter_error = st.number_input("Perimeter Error", min_value=0.0, max_value=100.0, value=3.0, step=0.1)
            compactness_error = st.number_input("Compactness Error", min_value=0.0, max_value=1.0, value=0.02, step=0.001)
            concave_points_error = st.number_input("Concave Points Error", min_value=0.0, max_value=1.0, value=0.01, step=0.001)

        with e2:
            texture_error = st.number_input("Texture Error", min_value=0.0, max_value=10.0, value=1.0, step=0.05)
            area_error = st.number_input("Area Error", min_value=0.0, max_value=2000.0, value=40.0, step=1.0)
            concavity_error = st.number_input("Concavity Error", min_value=0.0, max_value=1.0, value=0.02, step=0.001)
            symmetry_error = st.number_input("Symmetry Error", min_value=0.0, max_value=1.0, value=0.02, step=0.001)

        with e3:
            smoothness_error = st.number_input("Smoothness Error", min_value=0.0, max_value=1.0, value=0.008, step=0.001)
            fractal_dimension_error = st.number_input("Fractal Dimension Error", min_value=0.0, max_value=0.1, value=0.003, step=0.0005)

        # ----------------- Worst Values -----------------
        st.markdown("#### 🔹 Worst (Largest) Values")
        w1, w2, w3 = st.columns(3)

        with w1:
            worst_radius = st.number_input("Worst Radius", min_value=0.0, max_value=50.0, value=16.0, step=0.1)
            worst_perimeter = st.number_input("Worst Perimeter", min_value=0.0, max_value=500.0, value=110.0, step=0.5)
            worst_concavity = st.number_input("Worst Concavity", min_value=0.0, max_value=1.5, value=0.15, step=0.01)
            worst_symmetry = st.number_input("Worst Symmetry", min_value=0.0, max_value=1.0, value=0.25, step=0.01)

        with w2:
            worst_texture = st.number_input("Worst Texture", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            worst_area = st.number_input("Worst Area", min_value=0.0, max_value=10000.0, value=800.0, step=10.0)
            worst_concave_points = st.number_input("Worst Concave Points", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
            worst_fractal_dimension = st.number_input("Worst Fractal Dimension", min_value=0.0, max_value=0.3, value=0.08, step=0.001)

        with w3:
            worst_smoothness = st.number_input("Worst Smoothness", min_value=0.0, max_value=1.0, value=0.15, step=0.01)
            worst_compactness = st.number_input("Worst Compactness", min_value=0.0, max_value=2.0, value=0.3, step=0.01)

        submitted = st.form_submit_button("🔍 Predict Breast Cancer Risk", use_container_width=True)

    if submitted:
        # IMPORTANT: keep feature order same as training:
        # ['mean radius','mean texture','mean perimeter','mean area',
        #  'mean smoothness','mean compactness','mean concavity','mean concave points',
        #  'mean symmetry','mean fractal dimension',
        #  'radius error','texture error','perimeter error','area error',
        #  'smoothness error','compactness error','concavity error',
        #  'concave points error','symmetry error','fractal dimension error',
        #  'worst radius','worst texture','worst perimeter','worst area',
        #  'worst smoothness','worst compactness','worst concavity',
        #  'worst concave points','worst symmetry','worst fractal dimension']

        user_input = [
            mean_radius, mean_texture, mean_perimeter, mean_area,
            mean_smoothness, mean_compactness, mean_concavity, mean_concave_points,
            mean_symmetry, mean_fractal_dimension,
            radius_error, texture_error, perimeter_error, area_error,
            smoothness_error, compactness_error, concavity_error,
            concave_points_error, symmetry_error, fractal_dimension_error,
            worst_radius, worst_texture, worst_perimeter, worst_area,
            worst_smoothness, worst_compactness, worst_concavity,
            worst_concave_points, worst_symmetry, worst_fractal_dimension
        ]

        user_input = [float(x) for x in user_input]

        breast_prediction = MODELS['breast'].predict([user_input])[0]

        # Assume class 1 = Malignant (Positive), 0 = Benign (Negative)
        if hasattr(MODELS['breast'], 'predict_proba'):
            probability = MODELS['breast'].predict_proba([user_input])[0][1]
        else:
            probability = float(breast_prediction)

        # Simple heuristic risk factors based on "worst" measurements
        risk_factors = []

        if worst_radius > 20:
            risk_factors.append(("Worst Radius", f"{worst_radius:.1f}", "High"))
        elif worst_radius > 15:
            risk_factors.append(("Worst Radius", f"{worst_radius:.1f}", "Elevated"))
        else:
            risk_factors.append(("Worst Radius", f"{worst_radius:.1f}", "Normal"))

        if worst_area > 1000:
            risk_factors.append(("Worst Area", f"{worst_area:.1f}", "High"))
        elif worst_area > 800:
            risk_factors.append(("Worst Area", f"{worst_area:.1f}", "Elevated"))
        else:
            risk_factors.append(("Worst Area", f"{worst_area:.1f}", "Normal"))

        if worst_concave_points > 0.15:
            risk_factors.append(("Worst Concave Points", f"{worst_concave_points:.3f}", "High"))
        elif worst_concave_points > 0.08:
            risk_factors.append(("Worst Concave Points", f"{worst_concave_points:.3f}", "Elevated"))
        else:
            risk_factors.append(("Worst Concave Points", f"{worst_concave_points:.3f}", "Normal"))

        st.session_state.predictions['breast'] = {
            'prediction': breast_prediction,
            'probability': probability,
            'timestamp': datetime.now()
        }

        st.session_state.user_inputs['breast'] = {
            'mean_radius': mean_radius,
            'mean_texture': mean_texture,
            'mean_perimeter': mean_perimeter,
            'mean_area': mean_area,
            'mean_smoothness': mean_smoothness,
            'mean_compactness': mean_compactness,
            'mean_concavity': mean_concavity,
            'mean_concave_points': mean_concave_points,
            'mean_symmetry': mean_symmetry,
            'mean_fractal_dimension': mean_fractal_dimension,
            'radius_error': radius_error,
            'texture_error': texture_error,
            'perimeter_error': perimeter_error,
            'area_error': area_error,
            'smoothness_error': smoothness_error,
            'compactness_error': compactness_error,
            'concavity_error': concavity_error,
            'concave_points_error': concave_points_error,
            'symmetry_error': symmetry_error,
            'fractal_dimension_error': fractal_dimension_error,
            'worst_radius': worst_radius,
            'worst_texture': worst_texture,
            'worst_perimeter': worst_perimeter,
            'worst_area': worst_area,
            'worst_smoothness': worst_smoothness,
            'worst_compactness': worst_compactness,
            'worst_concavity': worst_concavity,
            'worst_concave_points': worst_concave_points,
            'worst_symmetry': worst_symmetry,
            'worst_fractal_dimension': worst_fractal_dimension
        }

        display_prediction_result("Breast Cancer (Malignant vs Benign)", breast_prediction, probability, risk_factors)

        st.markdown("---")
        st.markdown("### 🔄 What-If Analysis: Tumor Size & Shape")

        col1, col2 = st.columns(2)

        with col1:
            new_worst_radius = st.slider(
                "Adjust Worst Radius",
                min_value=0.0, max_value=40.0, value=float(worst_radius), step=0.1
            )

            new_worst_concave_points = st.slider(
                "Adjust Worst Concave Points",
                min_value=0.0, max_value=0.3, value=float(worst_concave_points), step=0.005
            )

        with col2:
            new_input = [
                mean_radius, mean_texture, mean_perimeter, mean_area,
                mean_smoothness, mean_compactness, mean_concavity, mean_concave_points,
                mean_symmetry, mean_fractal_dimension,
                radius_error, texture_error, perimeter_error, area_error,
                smoothness_error, compactness_error, concavity_error,
                concave_points_error, symmetry_error, fractal_dimension_error,
                new_worst_radius, worst_texture, worst_perimeter, worst_area,
                worst_smoothness, worst_compactness, worst_concavity,
                new_worst_concave_points, worst_symmetry, worst_fractal_dimension
            ]
            new_input = [float(x) for x in new_input]

            new_prediction = MODELS['breast'].predict([new_input])[0]
            if hasattr(MODELS['breast'], 'predict_proba'):
                new_probability = MODELS['breast'].predict_proba([new_input])[0][1]
            else:
                new_probability = float(new_prediction)

            st.markdown("#### Adjusted Prediction:")
            risk_cat, _, _ = get_risk_category(new_probability)

            delta = (new_probability - probability) * 100

            st.metric(
                "New Risk Probability",
                f"{new_probability*100:.1f}%",
                f"{delta:+.1f}%"
            )

            st.metric("New Risk Category", risk_cat)

            if delta < -5:
                st.success(
                    f"✅ If the tumor's worst radius reduces to {new_worst_radius:.1f} "
                    f"and worst concave points to {new_worst_concave_points:.3f}, "
                    f"the predicted risk decreases by {abs(delta):.1f}%."
                )
            elif delta > 5:
                st.warning(
                    f"⚠️ Larger size/irregular shape (higher worst radius or concave points) "
                    f"increases predicted risk by {delta:.1f}%."
                )
            else:
                st.info("Only a small change in predicted risk with these adjustments.")

# =====================================================================
# MODEL INSIGHTS PAGE (simple stub if you don't already have one)
# =====================================================================

def model_insights_page():
    st.markdown("## 🧩 Model Insights & Explainability")
    st.info(
        "You can extend this page to show feature importance charts, ROC curves, "
        "or any other explainability plots for your models."
    )

# =====================================================================
# REPORT DOWNLOAD PAGE  (your new version, extended to include breast)
# =====================================================================

def report_page():
    st.markdown("## 📄 Download Health Report")
    st.markdown("Generate a consolidated report of your latest predictions.")

    st.markdown("---")

    predictions_made = any(
        st.session_state.predictions[disease] is not None
        for disease in ['diabetes', 'heart', 'parkinsons', 'breast']
    )

    if not predictions_made:
        st.info("No predictions available yet. Please run at least one prediction first.")
        return

    rows = []
    for disease_key, disease_name in {
        'diabetes': 'Diabetes',
        'heart': 'Heart Disease',
        'parkinsons': "Parkinson's",
        'breast': "Breast Cancer"
    }.items():
        pred = st.session_state.predictions[disease_key]
        if pred is not None:
            risk_cat, _, _ = get_risk_category(pred['probability'])
            rows.append({
                "Disease": disease_name,
                "Prediction": "Positive" if pred['prediction'] == 1 else "Negative",
                "Risk Category": risk_cat,
                "Risk Probability (%)": round(pred['probability'] * 100, 1),
                "Timestamp": pred['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            })

    df_report = pd.DataFrame(rows)
    st.dataframe(df_report, use_container_width=True)

    # Build text report
    report_lines = []
    report_lines.append("AI Health Risk Analyzer - Summary Report")
    report_lines.append("=" * 50)
    report_lines.append(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")

    for row in rows:
        report_lines.append(f"Disease         : {row['Disease']}")
        report_lines.append(f"Prediction      : {row['Prediction']}")
        report_lines.append(f"Risk Category   : {row['Risk Category']}")
        report_lines.append(f"Risk Probability: {row['Risk Probability (%)']}%")
        report_lines.append(f"Timestamp       : {row['Timestamp']}")
        report_lines.append("-" * 40)

    report_lines.append("")
    report_lines.append("Disclaimer: This report is generated by a machine learning-based risk")
    report_lines.append("assessment system and is meant for educational purposes only. It is not")
    report_lines.append("a substitute for professional medical advice or diagnosis.")

    report_text = "\n".join(report_lines)

    st.download_button(
        label="📥 Download Report as TXT",
        data=report_text,
        file_name="health_risk_report.txt",
        mime="text/plain"
    )

# =====================================================================
# MAIN APP
# =====================================================================

def main():
    with st.sidebar:
        st.title("🏥 AI Health Risk Analyzer")
        st.markdown("A multi-disease ML-based risk prediction system.")

        page = st.radio(
            "Navigate to",
            [
                "Home / Dashboard",
                "Diabetes Prediction",
                "Heart Disease Prediction",
                "Parkinson's Prediction",
                "Breast Cancer Prediction",
                "Model Insights & Explainability",
                "Download Report"
            ]
        )

        st.markdown("---")
        st.caption("Built with ❤️ using Streamlit & Machine Learning")

    if page == "Home / Dashboard":
        home_page()
    elif page == "Diabetes Prediction":
        diabetes_page()
    elif page == "Heart Disease Prediction":
        heart_page()
    elif page == "Parkinson's Prediction":
        parkinsons_page()
    elif page == "Breast Cancer Prediction":
        breast_cancer_page()
    elif page == "Model Insights & Explainability":
        model_insights_page()
    elif page == "Download Report":
        report_page()


if __name__ == "__main__":
    main()
