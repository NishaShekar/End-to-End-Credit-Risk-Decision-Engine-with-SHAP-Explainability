# ==============================================
# Multi-Agent AI Credit Risk System (Production Ready)
# ==============================================

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ==================================
# Agent 1: Data Ingestion Agent
# ==================================
class DataIngestionAgent:
    def load_data(self, path):
        print("📥 Loading dataset...")
        data = pd.read_csv(path)
        print("✅ Dataset Loaded Successfully")
        print("Shape:", data.shape)
        return data

# ==================================
# Agent 2: Data Cleaning Agent
# ==================================
class DataCleaningAgent:
    def clean_data(self, data):
        print("\n🧹 Cleaning Data...")

        threshold = len(data) * 0.4
        data = data.dropna(thresh=threshold, axis=1)

        numeric_cols = data.select_dtypes(include=np.number).columns
        for col in numeric_cols:
            data[col].fillna(data[col].median(), inplace=True)

        categorical_cols = data.select_dtypes(include='object').columns
        for col in categorical_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)

        print("✅ Data Cleaning Completed")
        return data

# ==================================
# Agent 3: Feature Engineering Agent
# ==================================
class FeatureEngineeringAgent:
    def transform_data(self, data):
        print("\n⚙️ Encoding Categorical Variables...")

        for col in data.select_dtypes(include='object').columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

        print("✅ Encoding Completed")
        return data

# ==================================
# Agent 4: Model Training Agent
# ==================================
class ModelTrainingAgent:
    def train_model(self, X, y):
        print("\n🤖 Training Industry-Level Model...")

        # Cross-validation model
        cv_model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=(len(y) - sum(y)) / sum(y),
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cv_scores = cross_val_score(
            cv_model,
            X,
            y,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        print("\n📊 Cross-Validated ROC-AUC:", round(cv_scores.mean(), 4))

        # Train/Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Final model
        model = XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

        print("✅ Model Training Completed")

        # Evaluation
        y_prob = model.predict_proba(X_test)[:, 1]

        thresholds = np.arange(0.1, 0.9, 0.05)
        best_f1 = 0
        best_threshold = 0.5

        for t in thresholds:
            y_temp = (y_prob >= t).astype(int)
            report = classification_report(y_test, y_temp, output_dict=True)
            f1_class1 = report["1"]["f1-score"]

            if f1_class1 > best_f1:
                best_f1 = f1_class1
                best_threshold = t

        print(f"\n🔥 Best Threshold: {best_threshold}")
        print(f"🔥 Best F1 for Defaulters: {round(best_f1,4)}")

        y_pred = (y_prob >= best_threshold).astype(int)

        print("\n📊 Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        print("\n📊 Classification Report:")
        print(classification_report(y_test, y_pred))

        roc = roc_auc_score(y_test, y_prob)
        print("🔥 ROC-AUC Score:", round(roc,4))

        # Feature Importance
        importances = model.feature_importances_
        feature_names = X.columns

        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        print("\n📈 Top 15 Features:")
        print(importance_df.head(15))

        # Save model, features, threshold
        joblib.dump(model, "credit_risk_model.pkl")
        joblib.dump(X.columns.tolist(), "model_features.pkl")
        joblib.dump(best_threshold, "model_threshold.pkl")

        print("\n💾 Model saved as credit_risk_model.pkl")
        print("💾 Features saved as model_features.pkl")
        print("💾 Threshold saved as model_threshold.pkl")

        return model

# ==================================
# Orchestrator
# ==================================
class Orchestrator:
    def run_pipeline(self):

        ingestion_agent = DataIngestionAgent()
        data = ingestion_agent.load_data("data/application_train.csv")

        cleaning_agent = DataCleaningAgent()
        data = cleaning_agent.clean_data(data)

        feature_agent = FeatureEngineeringAgent()
        data = feature_agent.transform_data(data)

        print("\n📌 Preparing Features...")
        X = data.drop(["TARGET", "SK_ID_CURR"], axis=1)
        y = data["TARGET"]

        model_agent = ModelTrainingAgent()
        model_agent.train_model(X, y)

        print("\n🎯 Multi-Agent Credit Risk System Completed Successfully!")

if __name__ == "__main__":
    orchestrator = Orchestrator()
    orchestrator.run_pipeline()
