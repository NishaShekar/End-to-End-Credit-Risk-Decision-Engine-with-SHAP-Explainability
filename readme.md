# 📊 End-to-End-Credit-Risk-Decision-Engine-with-SHAP-Explainability
### Business-Aligned Loan Approval Decision Engine with SHAP Explainability

---

## 🔍 Overview

This project is an end-to-end **Credit Risk Scoring and Loan Approval System** built using Machine Learning.

The system predicts **Probability of Default (PD)**, converts it into a **Credit Score (300–850 scale)**, categorizes borrower risk, and generates an approval decision based on a defined threshold.

It also provides **SHAP-based explainability** to justify individual predictions — aligning with modern banking and regulatory standards.

---

## 🚀 Key Features

- XGBoost-based Probability of Default model  
- Credit score transformation (300–850 scale)  
- Business-aligned loan approval logic  
- Risk segmentation (Excellent / Good / Fair / High Risk)  
- SHAP explainability for individual predictions  
- Interactive Streamlit dashboard  
- Threshold optimization (F1-score focused on defaulters)

---

## 🧠 Model Performance

- ROC-AUC: ~0.75+  
- Optimized for minority class (defaulters) detection  
- Decision threshold tuned for business risk appetite  
- Individual-level explainability using SHAP  

----

## 📊 Business Logic

### Probability of Default → Credit Score

The model transforms predicted PD into a credit score:

- Low PD → Higher Credit Score  
- High PD → Lower Credit Score  

Score Range: **300 – 850**

---

### Risk Segmentation

| Credit Score | Risk Level |
|--------------|------------|
| 750+         | Excellent  |
| 700–749      | Good       |
| 650–699      | Fair       |
| < 650        | High Risk  |

---

### Loan Decision Rule

If:
 PD < Decision Threshold → Approve
 PD ≥ Decision Threshold → Reject

Threshold optimized based on performance metrics and business tolerance.

---

## 📈 Explainability (SHAP)

The system uses SHAP (SHapley Additive exPlanations) to:

- Identify top contributing features  
- Show feature-level impact on default probability  
- Provide transparency for each decision  
- Support regulatory explainability requirements  

Positive SHAP values → Increase default risk  
Negative SHAP values → Reduce default risk  

---

## 🛠 Tech Stack

- Python  
- XGBoost  
- Scikit-learn  
- SHAP  
- Streamlit  
- Pandas  
- NumPy  
- Matplotlib  
- Joblib  

---

## 📁 Project Structure

Multi_Agent_System/
│
├── app.py
├── model/
│ └── xgboost_model.pkl
├── requirements.txt
├── README.md
└── screenshots/

---

## 🌍 Live Deployment

(After deployment, update this section)

Live App: https://credit-risk-decision-engine-with-shap.streamlit.app/

---

## 🔮 Future Enhancements

- Model monitoring dashboard (Data Drift & PSI)  
- Debt-to-Income ratio feature engineering  
- Risk band segmentation (A/B/C/D grading)  
- Logistic regression scorecard version  
- CI/CD-based deployment pipeline  

---

## ⭐ Business Impact

This project demonstrates:

- Credit Risk Modeling  
- Probability of Default estimation  
- Business Decision Engine Design  
- Model Explainability  
- End-to-End ML Deployment  

Suitable for roles in:

- Credit Risk Analytics  
- Risk Modeling  
- FinTech Analytics  
- Model Validation  
- Decision Science  

