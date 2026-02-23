import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("Agg")

# Load model, features, threshold
model = joblib.load("credit_risk_model.pkl")
features = joblib.load("model_features.pkl")
threshold = joblib.load("model_threshold.pkl")

# Create SHAP explainer
explainer = shap.TreeExplainer(model)

st.set_page_config(
    page_title="AI Credit Risk Scoring System",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 AI Credit Risk Scoring System")
st.write("Enter applicant details to generate Credit Score and Risk Explanation.")

st.divider()

# Dynamic input fields
input_data = {}

for feature in features:
    input_data[feature] = st.number_input(
        label=feature,
        value=0.0,
        step=1.0
    )

st.divider()

if st.button("🔍 Generate Credit Score"):

    input_df = pd.DataFrame([input_data])

    # Predict probability
    prob = model.predict_proba(input_df)[0][1]

    # Convert to Credit Score
    credit_score = int(850 - (prob * 550))

    st.subheader("📊 Scoring Results")

    st.metric("💳 Credit Score", credit_score)
    st.write(f"**Probability of Default:** {prob * 100:.2f}%")

    st.progress(float(prob))

    # Risk band
    if credit_score >= 750:
        st.success("🟢 Excellent Credit")
    elif credit_score >= 650:
        st.success("🟢 Good Credit")
    elif credit_score >= 550:
        st.warning("🟡 Fair Credit - Review Required")
    else:
        st.error("🔴 Poor Credit - High Risk")

    # Decision
    st.divider()
    st.write("### 🔎 Loan Decision")

    if prob >= threshold:
        st.error("❌ Reject Application")
    else:
        st.success("✅ Approve Application")

    # ============================
    # SHAP EXPLAINABILITY
    # ============================

    st.divider()
    st.subheader("🧠 Why This Decision Was Made")

    shap_values = explainer.shap_values(input_df)

    # Convert SHAP values to DataFrame
    shap_df = pd.DataFrame({
        "Feature": features,
        "SHAP Value": shap_values[0]
    })

    shap_df["Impact"] = np.where(shap_df["SHAP Value"] > 0,
                                 "Increases Default Risk",
                                 "Decreases Default Risk")

    shap_df = shap_df.reindex(
        shap_df["SHAP Value"].abs().sort_values(ascending=False).index
    )

    st.write("### 🔥 Top 10 Most Influential Features")
    st.dataframe(shap_df.head(10))

    # SHAP Bar Plot
    st.write("### 📊 SHAP Feature Impact Visualization")

    fig, ax = plt.subplots()
    shap.plots._waterfall.waterfall_legacy(
        explainer.expected_value,
        shap_values[0],
        feature_names=features,
        max_display=10,
        show=False
    )
    st.pyplot(fig)

    st.caption("Positive values increase default risk. Negative values reduce risk.")

