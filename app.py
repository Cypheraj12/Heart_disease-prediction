
import streamlit as st
import pandas as pd
import numpy as np
import base64
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="HEART DISEASE PREDICTOR",
    page_icon=".",
    layout="wide"
)

st.markdown("""
# ❤️ HEART DISEASE PREDICTOR

### AI-Powered Heart Disease Risk Assessment

  They say follow your heart. Our model says maybe analyze it first...
""")

DATA_PATH = "heart_disease_uci.csv"

@st.cache_resource
def train_model():

    df = pd.read_csv(DATA_PATH)

    total_records = len(df)

    df["num"] = (df["num"] > 0).astype(int)

    numeric_cols = ["trestbps","chol","thalch","oldpeak"]

    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    for col in ["fbs","restecg","exang"]:
        df[col] = df[col].fillna(df[col].mode()[0])

    df["slope"] = df["slope"].fillna("Missing")
    df["thal"] = df["thal"].fillna("Missing")
    df["ca"] = df["ca"].fillna(-1)

    df["ca_missing"] = (df["ca"] == -1).astype(int)
    df["thal_missing"] = (df["thal"] == "Missing").astype(int)
    df["age_oldpeak"] = df["age"] * df["oldpeak"]

    X = df.drop(columns=["num"])

    if "id" in X.columns:
        X = X.drop(columns=["id"])

    y = df["num"]

    X = pd.get_dummies(
        X,
        drop_first=True,
        dtype=int
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=59,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=3000,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, pred)

    return (
        model,
        scaler,
        X.columns.tolist(),
        accuracy,
        total_records
    )


try:

    model, scaler, feature_columns, accuracy, total_records = train_model()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Training Data", "~1K Records")
    c2.metric("Model", "Logistic Regression")
    c3.metric("Features", len(feature_columns))
    c4.metric("Accuracy", f"{accuracy*100:.2f}%")

    st.caption(
    "Model trained on approximately 1,000 patient records with feature engineering, encoding and scaling.")

    st.markdown("---")

    st.subheader("Patient Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider(
            "Age (years) — Patient age",
            20,
            90,
            50
        )
        sex = st.selectbox(
            "Sex — Biological sex of the patient",
            ["Male", "Female"]
        )

        cp = st.selectbox(
            "Chest Pain Type — Type of chest pain experienced",
            [
                "typical angina",
                "atypical angina",
                "non-anginal",
                "asymptomatic"
            ]
        )

        trestbps = st.slider(
            "Resting Blood Pressure (mmHg) — BP measured while resting",
            80,
            220,
            130
        )

        chol = st.slider(
            "Cholesterol (mg/dL) — Blood cholesterol level",
            100,
            700,
            240
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar — Above 120 mg/dL?",
            [False, True]
        )

        restecg = st.selectbox(
            "Resting ECG — Heart electrical activity at rest",
            [
                "normal",
                "st-t abnormality",
                "lv hypertrophy"
            ]
        )

    with col2:
        thalch = st.slider(
            "Maximum Heart Rate — Highest heart rate achieved",
            60,
            220,
            150
        )

        exang = st.selectbox(
            "Exercise Induced Angina — Chest pain during exercise?",
            [False, True]
        )

        oldpeak = st.slider(
            "Oldpeak — ECG depression caused by exercise",
            0.0,
            7.0,
            1.0,
            0.1
        )

        slope = st.selectbox(
            "ST Segment Slope — ECG pattern during peak exercise",
            [
                "upsloping",
                "flat",
                "downsloping"
            ]
        )

        ca = st.selectbox(
            "Major Vessels — Number of visible major blood vessels",
            [0, 1, 2, 3]
        )

        thal = st.selectbox(
            "Thalassemia Test — Heart blood-flow defect result",
            [
                "normal",
                "fixed defect",
                "reversable defect"
            ]
        )

    if st.button("Predict Heart Disease Risk", use_container_width=True):

        user = pd.DataFrame({
            "age":[age],
            "sex":[sex],
            "cp":[cp],
            "trestbps":[trestbps],
            "chol":[chol],
            "fbs":[fbs],
            "restecg":[restecg],
            "thalch":[thalch],
            "exang":[exang],
            "oldpeak":[oldpeak],
            "slope":[slope],
            "ca":[ca],
            "thal":[thal]
        })

        user["ca_missing"] = (user["ca"] == -1).astype(int)
        user["thal_missing"] = (user["thal"] == "Missing").astype(int)
        user["age_oldpeak"] = user["age"] * user["oldpeak"]

        user = pd.get_dummies(
            user,
            drop_first=True,
            dtype=int
        )

        user = user.reindex(
            columns=feature_columns,
            fill_value=0
        )

        user_scaled = scaler.transform(user)

        prediction = model.predict(user_scaled)[0]
        probability = model.predict_proba(user_scaled)[0]

        disease_prob = probability[1] * 100
        healthy_prob = probability[0] * 100

        st.markdown("---")

        if prediction == 1:

            audio_path = "alert.mpeg"

            if os.path.exists(audio_path):

                with open(audio_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()

                audio_base64 = base64.b64encode(audio_bytes).decode()

                st.markdown(
                    f"""
                    <audio autoplay>
                        <source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg">
                    </audio>
                    """,
                    unsafe_allow_html=True
                )

            st.error(
                f"""
                ⚠️ HEART DISEASE RISK DETECTED

                Estimated Risk: {disease_prob:.2f}%
                """
            )

            st.metric(
                "Heart Disease Risk",
                f"{disease_prob:.2f}%"
            )

            st.progress(
                min(int(disease_prob), 100)
            )

        else:

            st.balloons()

            st.success(
                f"""
                ✅ HEART APPEARS HEALTHY

                Confidence Score: {healthy_prob:.2f}%
                """
            )

            st.metric(
                "Healthy Heart Confidence",
                f"{healthy_prob:.2f}%"
            )

            st.progress(
                min(int(healthy_prob), 100)
            )

        st.subheader("Prediction Probabilities")

        st.dataframe(
            pd.DataFrame({
                "Outcome":[
                    "No Heart Disease",
                    "Heart Disease"
                ],
                "Probability (%)":[
                    round(healthy_prob,2),
                    round(disease_prob,2)
                ]
            }),
            use_container_width=True
        )

except Exception as e:
    st.error(f"Error: {e}")
