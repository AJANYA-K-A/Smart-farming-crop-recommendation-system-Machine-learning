"""
Smart Farming Crop Recommendation System - Model Training Script
Trains a RandomForestClassifier on soil and environmental features to recommend optimal crops.
"""

import os
import sys

# Ensure UTF-8 stdout encoding on Windows
if sys.platform == 'win32' and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Base directory for relative file paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Crop_recommendation.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'crop_model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'label_encoder.pkl')


def train_and_evaluate():
    print("=" * 60)
    print("[*] Training Smart Farming Crop Recommendation Model...")
    print("=" * 60)

    # 1. Load dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"[*] Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Features and target
    feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    X = df[feature_cols]
    y = df['label']

    # 3. Encode crop labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    print(f"[*] Number of unique crop classes: {num_classes}")
    print(f"    Classes: {', '.join(label_encoder.classes_)}")

    # 4. Train-test split (80-20 stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"[*] Train set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # 5. Model training
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # 6. Evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[+] Model Evaluation Accuracy: {acc * 100:.2f}%")

    print("\n[+] Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # 7. Save model artifacts
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoder, ENCODER_PATH)
    print(f"[+] Model saved to: {MODEL_PATH}")
    print(f"[+] Label encoder saved to: {ENCODER_PATH}")
    print("=" * 60)
    print("[+] Training complete and model ready for deployment!")
    print("=" * 60)

    return model, label_encoder, acc


if __name__ == '__main__':
    train_and_evaluate()