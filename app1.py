import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🎗️",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "breast_cancer_model.pkl"
    )

    scaler = joblib.load(
        "scaler.pkl"
    )

    return model, scaler


model, scaler = load_model()


# ============================================================
# TITLE
# ============================================================

st.title("🎗️ Breast Cancer Classification")

st.write(
    "Enter the tumor measurements below to predict "
    "whether the tumor is **Benign** or **Malignant**."
)

st.warning(
    "⚠️ This application is for educational purposes only "
    "and should not be used as a medical diagnosis."
)


# ============================================================
# INPUT FEATURES
# ============================================================

st.header("🔬 Tumor Measurements")


col1, col2, col3 = st.columns(3)


with col1:

    radius_mean = st.number_input(
        "Radius Mean",
        min_value=0.0,
        value=14.0,
        step=0.01
    )

    texture_mean = st.number_input(
        "Texture Mean",
        min_value=0.0,
        value=19.0,
        step=0.01
    )

    perimeter_mean = st.number_input(
        "Perimeter Mean",
        min_value=0.0,
        value=90.0,
        step=0.01
    )

    area_mean = st.number_input(
        "Area Mean",
        min_value=0.0,
        value=650.0,
        step=0.01
    )

    smoothness_mean = st.number_input(
        "Smoothness Mean",
        min_value=0.0,
        value=0.096,
        step=0.0001
    )

    compactness_mean = st.number_input(
        "Compactness Mean",
        min_value=0.0,
        value=0.10,
        step=0.001
    )

    concavity_mean = st.number_input(
        "Concavity Mean",
        min_value=0.0,
        value=0.09,
        step=0.001
    )

    concave_points_mean = st.number_input(
        "Concave Points Mean",
        min_value=0.0,
        value=0.05,
        step=0.001
    )

    symmetry_mean = st.number_input(
        "Symmetry Mean",
        min_value=0.0,
        value=0.18,
        step=0.001
    )

    fractal_dimension_mean = st.number_input(
        "Fractal Dimension Mean",
        min_value=0.0,
        value=0.063,
        step=0.001
    )


with col2:

    radius_se = st.number_input(
        "Radius SE",
        min_value=0.0,
        value=0.4,
        step=0.01
    )

    texture_se = st.number_input(
        "Texture SE",
        min_value=0.0,
        value=1.2,
        step=0.01
    )

    perimeter_se = st.number_input(
        "Perimeter SE",
        min_value=0.0,
        value=2.8,
        step=0.01
    )

    area_se = st.number_input(
        "Area SE",
        min_value=0.0,
        value=40.0,
        step=0.1
    )

    smoothness_se = st.number_input(
        "Smoothness SE",
        min_value=0.0,
        value=0.007,
        step=0.0001
    )

    compactness_se = st.number_input(
        "Compactness SE",
        min_value=0.0,
        value=0.02,
        step=0.001
    )

    concavity_se = st.number_input(
        "Concavity SE",
        min_value=0.0,
        value=0.03,
        step=0.001
    )

    concave_points_se = st.number_input(
        "Concave Points SE",
        min_value=0.0,
        value=0.01,
        step=0.001
    )

    symmetry_se = st.number_input(
        "Symmetry SE",
        min_value=0.0,
        value=0.02,
        step=0.001
    )

    fractal_dimension_se = st.number_input(
        "Fractal Dimension SE",
        min_value=0.0,
        value=0.003,
        step=0.0001
    )


with col3:

    radius_worst = st.number_input(
        "Radius Worst",
        min_value=0.0,
        value=16.0,
        step=0.01
    )

    texture_worst = st.number_input(
        "Texture Worst",
        min_value=0.0,
        value=25.0,
        step=0.01
    )

    perimeter_worst = st.number_input(
        "Perimeter Worst",
        min_value=0.0,
        value=105.0,
        step=0.01
    )

    area_worst = st.number_input(
        "Area Worst",
        min_value=0.0,
        value=850.0,
        step=0.01
    )

    smoothness_worst = st.number_input(
        "Smoothness Worst",
        min_value=0.0,
        value=0.13,
        step=0.001
    )

    compactness_worst = st.number_input(
        "Compactness Worst",
        min_value=0.0,
        value=0.25,
        step=0.001
    )

    concavity_worst = st.number_input(
        "Concavity Worst",
        min_value=0.0,
        value=0.25,
        step=0.001
    )

    concave_points_worst = st.number_input(
        "Concave Points Worst",
        min_value=0.0,
        value=0.11,
        step=0.001
    )

    symmetry_worst = st.number_input(
        "Symmetry Worst",
        min_value=0.0,
        value=0.29,
        step=0.001
    )

    fractal_dimension_worst = st.number_input(
        "Fractal Dimension Worst",
        min_value=0.0,
        value=0.08,
        step=0.001
    )


# ============================================================
# PREDICTION
# ============================================================

st.divider()

predict_button = st.button(
    "🔍 Predict",
    use_container_width=True
)


if predict_button:

    # Create input dataframe
    input_data = pd.DataFrame(
        [[
            radius_mean,
            texture_mean,
            perimeter_mean,
            area_mean,
            smoothness_mean,
            compactness_mean,
            concavity_mean,
            concave_points_mean,
            symmetry_mean,
            fractal_dimension_mean,

            radius_se,
            texture_se,
            perimeter_se,
            area_se,
            smoothness_se,
            compactness_se,
            concavity_se,
            concave_points_se,
            symmetry_se,
            fractal_dimension_se,

            radius_worst,
            texture_worst,
            perimeter_worst,
            area_worst,
            smoothness_worst,
            compactness_worst,
            concavity_worst,
            concave_points_worst,
            symmetry_worst,
            fractal_dimension_worst
        ]],
        columns=[
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",

            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave points_se",
            "symmetry_se",
            "fractal_dimension_se",

            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave points_worst",
            "symmetry_worst",
            "fractal_dimension_worst"
        ]
    )


    # Scale input
    input_scaled = scaler.transform(
        input_data
    )


    # Make prediction
    prediction = model.predict(
        input_scaled
    )[0]


    # Prediction probability
    probabilities = model.predict_proba(
        input_scaled
    )[0]

    benign_probability = probabilities[0]
    malignant_probability = probabilities[1]


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.subheader("📊 Prediction Result")


    if prediction == 0:

        st.success(
            "🟢 Prediction: BENIGN"
        )

    else:

        st.error(
            "🔴 Prediction: MALIGNANT"
        )


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Benign Probability",
            f"{benign_probability * 100:.2f}%"
        )


    with col2:

        st.metric(
            "Malignant Probability",
            f"{malignant_probability * 100:.2f}%"
        )


    st.progress(
        float(malignant_probability)
    )


    st.caption(
        "The probability shown is the model's prediction confidence, "
        "not a medical diagnosis."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Breast Cancer Classification | Machine Learning Project"
)