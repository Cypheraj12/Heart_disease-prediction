# ❤️ Heart Disease Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://heartdisease-prediction123.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML%20Modeling-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)

An interactive machine learning web application that analyzes clinical patient health data to estimate the risk of cardiovascular disease in real time.

---

## 🚀 Live Demo

Access the live interactive application on Streamlit Cloud:  
👉 **[HEART DISEASE PREDICTOR · Streamlit](https://heartdisease-prediction123.streamlit.app/)**

---

## 📌 Overview

Heart Disease Predictor is a machine learning web application built with Streamlit that estimates the likelihood of heart disease based on patient health information. The application uses a classification pipeline trained on the UCI Heart Disease dataset and provides real-time predictions through an intuitive, interactive interface.

The goal of this project is to demonstrate how predictive machine learning models can be applied to healthcare data to support early risk assessment and proactive awareness.

---

## ✨ Features

- **Live Streamlit Interface**: Clean, responsive layout with intuitive input sliders and selectors.
- **Real-Time Heart Disease Risk Prediction**: Immediate risk assessment calculation upon input submission.
- **Trained ML Classification Pipeline**: Built on clinical patient parameters with missing value imputation, one-hot encoding, and feature scaling via `StandardScaler`.
- **Confidence Scoring & Probability Indicators**: Visual probability gauges and risk indicators to explain the model's confidence.
- **Data Preprocessing & Validation**: Input sanitization and automated category alignment.

---

## 📊 Dataset

This project utilizes the **UCI Heart Disease dataset**, containing patient demographic and clinical diagnostic indicators:

| Feature | Description |
| :--- | :--- |
| **Age** | Age in years |
| **Sex** | Biological sex (Male / Female) |
| **Chest Pain Type (cp)** | Typical angina, atypical angina, non-anginal pain, asymptomatic |
| **Resting BP (trestbps)** | Resting blood pressure in mm Hg |
| **Cholesterol (chol)** | Serum cholesterol in mg/dl |
| **Fasting Blood Sugar (fbs)**| Fasting blood sugar > 120 mg/dl |
| **Resting ECG (restecg)** | Normal, ST-T wave abnormality, left ventricular hypertrophy |
| **Max Heart Rate (thalach)**| Maximum heart rate achieved during stress test |
| **Exercise Angina (exang)** | Exercise-induced angina (Yes / No) |
| **Oldpeak** | ST depression induced by exercise relative to rest |
| **Slope** | Slope of the peak exercise ST segment |
| **Major Vessels (ca)** | Number of major vessels (0-3) colored by flourosopy |
| **Thalassemia (thal)** | Normal, fixed defect, reversible defect |

---

## 🛠️ Machine Learning Pipeline

1. **Exploratory Data Analysis**: Carried out in `heart_pred.ipynb`.
2. **Data Preprocessing**: Handling missing values, outlier checks, and distribution analysis.
3. **Feature Transformation**: One-hot encoding of categorical attributes and standardization using `StandardScaler`.
4. **Model Architecture**: Logistic Regression / Classification models evaluated on cross-validated accuracy, precision, and recall.
5. **Interactive Deployment**: Packaged with Streamlit for zero-friction cloud deployment.

---

## 💻 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Cypheraj12/Heart_disease-prediction.git
cd Heart_disease-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

The application will launch on your local host (usually `http://localhost:8501`).

---

## 📂 Project Structure

```text
Heart_disease-prediction/
├── app.py                  # Streamlit web app & inference pipeline
├── heart_disease_uci.csv   # UCI Heart Disease benchmark dataset
├── heart_pred.ipynb        # Data science & model training notebook
├── requirements.txt        # Python dependency manifest
├── runtime.txt             # Python runtime specification
└── README.md               # Documentation
```

---

## ⚠️ Disclaimer

This application is developed strictly for educational and demonstration purposes. Model predictions do not constitute professional medical advice, clinical diagnosis, or treatment recommendations. Always consult a qualified medical professional for health-related evaluations.

---

## 👤 Author

**Anant Joshi**  
- GitHub: [@Cypheraj12](https://github.com/Cypheraj12)
- Portfolio: [my-portfolio](https://my-portfolio-psi-liart-71.vercel.app/)
