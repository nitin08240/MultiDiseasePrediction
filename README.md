# 🏥 Multi-Disease Prediction System

A comprehensive web application built with **Streamlit** that uses advanced machine learning models to predict the risk of multiple diseases. This intelligent system provides real-time health risk assessments with detailed explanations and personalized recommendations.

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Machine Learning Models](#machine-learning-models)
- [Streamlit Components](#streamlit-components)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Model Performance](#model-performance)

---

## 🎯 Project Overview

The **Multi-Disease Prediction System** is an AI-powered health analytics platform that predicts the likelihood of four major diseases:

1. **Diabetes** - Early detection for better management
2. **Heart Disease** - Cardiovascular risk assessment
3. **Parkinson's Disease** - Neurological disorder prediction
4. **Breast Cancer** - Oncological risk evaluation

The application provides interactive prediction capabilities with risk categorization, feature importance analysis, and "What-If" scenario testing to help users understand modifiable risk factors.

---

## ✨ Features

### 🔍 **Core Prediction Features**
- Real-time disease risk prediction using pre-trained ML models
- Probability scores with risk categorization (Low/Medium/High)
- Interactive forms for easy data input
- Instant feedback and recommendations

### 📊 **Data Visualization**
- Feature importance charts for model explainability
- Risk probability meters
- Comparative analysis between different risk factors
- Historical prediction tracking

### 🎛️ **Interactive Analysis**
- "What-If" scenario analysis to explore risk reduction
- Adjustable input parameters to see impact on predictions
- Confidence intervals for predictions
- Model performance metrics

### 📱 **User Interface**
- Clean, intuitive navigation using Streamlit Option Menu
- Responsive layout with wide display
- Color-coded risk indicators (green/yellow/red)
- Custom CSS styling for enhanced UX

### 💾 **Session Management**
- Persistent session state for user inputs and predictions
- History tracking of previous assessments
- Save and export prediction reports

---

## 🤖 Machine Learning Models

### **1. Diabetes Prediction Model**
- **Algorithm**: Random Forest / Decision Tree Classifier
- **Training Dataset**: `dataset/diabetes.csv`
- **Input Features**: 8 health parameters (glucose, blood pressure, BMI, etc.)
- **Target**: Binary classification (Diabetic / Non-Diabetic)
- **Accuracy**: ~78%
- **Model File**: `saved_models/diabetes_model.sav`

**Key Features Used**:
- Pregnancies
- Glucose Level
- Blood Pressure
- Skin Thickness
- Insulin
- BMI (Body Mass Index)
- Diabetes Pedigree Function
- Age

### **2. Heart Disease Prediction Model**
- **Algorithm**: Support Vector Classifier (SVC) / Random Forest
- **Training Dataset**: `dataset/heart.csv`
- **Input Features**: 13 cardiovascular health metrics
- **Target**: Binary classification (Disease / No Disease)
- **Accuracy**: ~85%
- **Model File**: `saved_models/heart_disease_model.sav`

**Key Features Used**:
- Age
- Sex
- Chest Pain Type
- Resting Blood Pressure
- Serum Cholesterol
- Fasting Blood Sugar
- Resting ECG Results
- Maximum Heart Rate
- ST Depression
- ST Slope
- And more...

### **3. Parkinson's Disease Prediction Model**
- **Algorithm**: Support Vector Classifier (SVC) / Decision Tree
- **Training Dataset**: `dataset/parkinsons.csv`
- **Input Features**: 22 voice/speech signal characteristics (MFCC features)
- **Target**: Binary classification (Parkinson's / Healthy)
- **Accuracy**: ~87%
- **Model File**: `saved_models/parkinsons_model.sav`

**Key Features Used**:
- MDVP Frequency Parameters
- Fundamental Frequency
- Jitter Measurements
- Shimmer Measurements
- Noise-to-Harmonics Ratio
- Recurrence Quantification Analysis (RQA) Features

### **4. Breast Cancer Prediction Model**
- **Algorithm**: Random Forest Classifier / SVC
- **Training Dataset**: UCI Breast Cancer Dataset (preprocessed)
- **Input Features**: 30 morphological features (radius, texture, perimeter, etc.)
- **Target**: Binary classification (Malignant / Benign)
- **Accuracy**: ~95%
- **Model File**: `saved_models/breast_model.sav`

**Key Features Used**:
- Radius Mean
- Texture Mean
- Perimeter Mean
- Area Mean
- Smoothness Mean
- Compactness Mean
- Concavity Mean
- And derivative/worst case statistics...

---

## 🎨 Streamlit Components Used

### **1. Page Configuration**
```python
st.set_page_config(
    page_title="AI Health Risk Analyzer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)
```
- Sets custom title, icon, and wide layout for better space utilization

### **2. Navigation Menu** (Streamlit Option Menu)
```python
from streamlit_option_menu import option_menu

selected = option_menu(
    menu_title="Navigation",
    options=["Home", "Diabetes", "Heart Disease", "Parkinson's", "Breast Cancer"],
    menu_icon="hospital-fill"
)
```
- Interactive sidebar navigation with custom icons
- Smooth page transitions between disease prediction sections

### **3. Layout Components**
- **`st.columns()`** - Multi-column layouts for side-by-side content display
- **`st.expander()`** - Collapsible sections for detailed information
- **`st.container()`** - Logical grouping of elements
- **`st.markdown()`** - Custom HTML/CSS styling for rich formatting

### **4. Input Components**
- **`st.number_input()`** - For entering numerical health metrics
- **`st.slider()`** - For selecting values in a range (age, BMI, etc.)
- **`st.selectbox()`** - Dropdown selections for categorical data
- **`st.radio()`** - Radio buttons for binary choices

### **5. Display Components**
- **`st.metric()`** - Key metrics display (prediction, probability, risk level)
- **`st.success()`, `st.warning()`, `st.error()`** - Status messages with color coding
- **`st.info()`** - Information boxes
- **`st.dataframe()`** - Tabular data display

### **6. Visualization Components**
- **`st.plotly_chart()`** - Interactive Plotly charts for feature importance
- **`st.bar_chart()`, `st.line_chart()`** - Simple chart rendering
- **`st.progress()`** - Progress bars for risk levels

### **7. State Management**
```python
st.session_state - Maintains:
- User inputs for each disease prediction
- Previous predictions and history
- User preferences and settings
- Model caching for performance
```

### **8. Caching**
```python
@st.cache_resource - Caches model loading
@st.cache_data - Caches computation-heavy operations
```
- Improves app performance by avoiding redundant computations

### **9. Custom Styling**
```python
st.markdown("""<style>
    .risk-high { background-color: #ffcccc; }
    .risk-low { background-color: #ccffcc; }
    .risk-medium { background-color: #ffffcc; }
</style>""", unsafe_allow_html=True)
```
- Custom CSS for risk indicators and visual hierarchy

---

## 🚀 Installation & Setup

### **Prerequisites**
- Python 3.8+
- pip (Python package manager)
- Git

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/nitin08240/MultiDiseasePrediction.git
cd MultiDiseasePrediction
```

### **Step 2: Create Virtual Environment**
```bash
python -m venv venv
```

### **Step 3: Activate Virtual Environment**

**On Windows (PowerShell):**
```powershell
.\venv\Scripts\Activate.ps1
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### **Step 4: Install Dependencies**
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install streamlit streamlit-option-menu scikit-learn numpy pandas plotly imbalanced-learn
```

### **Step 5: Run the Application**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📖 Usage

### **1. Home/Dashboard Page**
- Overview of all disease prediction capabilities
- Quick statistics and model performance metrics
- Links to individual disease prediction pages

### **2. Disease Prediction Pages**
Each disease page follows this workflow:

1. **Input Health Parameters** - Fill in the required health metrics using interactive forms
2. **Generate Prediction** - Click "Predict" button to run the ML model
3. **View Results** - See prediction (Positive/Negative), probability score, and risk category
4. **Analyze Risk Factors** - Identify which parameters contribute most to the risk
5. **Explore "What-If" Scenarios** - Adjust parameters to see impact on prediction

### **3. Example: Diabetes Prediction**
1. Enter values for: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age
2. Click "Predict Diabetes Risk"
3. View results with color-coded risk level
4. Check feature importance to understand key risk factors
5. Use "What-If" slider to adjust values and see updated predictions

---

## 📁 Project Structure

```
MultiDiseasePrediction/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── .gitignore                      # Git ignore file
│
├── dataset/                        # Training datasets
│   ├── diabetes.csv               # Diabetes dataset
│   ├── heart.csv                  # Heart disease dataset
│   └── parkinsons.csv             # Parkinson's dataset
│
├── saved_models/                   # Pre-trained ML models (pickle files)
│   ├── diabetes_model.sav
│   ├── heart_disease_model.sav
│   ├── parkinsons_model.sav
│   └── breast_model.sav
│
└── venv/                          # Virtual environment (not tracked)
```

---

## 🛠️ Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Framework** | Streamlit | Web application framework |
| **Navigation** | Streamlit Option Menu | Custom menu navigation |
| **ML Algorithms** | Scikit-Learn | Machine learning models |
| **Data Processing** | Pandas, NumPy | Data manipulation and computation |
| **Visualization** | Plotly | Interactive charts and graphs |
| **Model Serialization** | Pickle | Saving/loading ML models |
| **Version Control** | Git | Code versioning and collaboration |

### **Python Libraries**
```
streamlit==1.51.0
streamlit-option-menu==0.4.0
scikit-learn==1.6.1
numpy==2.3.5
pandas==2.3.3
plotly==5.x.x
imbalanced-learn==0.14.0
```

---

## 📊 Model Performance

| Disease | Algorithm | Accuracy | Recall | Precision | File |
|---------|-----------|----------|--------|-----------|------|
| **Diabetes** | Random Forest / Decision Tree | ~81% | ~75% | ~80% | `diabetes_model.sav` |
| **Heart Disease** | SVC / Random Forest | ~85% | ~83% | ~87% | `heart_disease_model.sav` |
| **Parkinson's** | SVC / Decision Tree | ~87% | ~85% | ~89% | `parkinsons_model.sav` |
| **Breast Cancer** | Random Forest | ~95% | ~94% | ~96% | `breast_model.sav` |

**Note**: These metrics are based on the training datasets. Actual performance may vary based on new data characteristics.

---

## ⚠️ Disclaimer

This application is for **educational and informational purposes only**. It should NOT be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for:

- Accurate disease diagnosis
- Treatment recommendations
- Medical decision-making
- Emergency medical situations

The models are trained on historical data and may not account for individual variations. Use predictions as a guide for further medical consultation, not as definitive diagnosis.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Author

**Nitin** - [@nitin08240](https://github.com/nitin08240)

---

## 🙏 Acknowledgments

- Datasets sourced from UCI Machine Learning Repository
- Streamlit for the amazing web framework
- Scikit-Learn for robust ML algorithms
- Open-source community for libraries and tools

---

## 📞 Support

For issues, questions, or suggestions:
- Create an [Issue](https://github.com/nitin08240/MultiDiseasePrediction/issues) on GitHub
- Email: nitin84341944660@gmail.com

---

**Last Updated**: November 28, 2025
