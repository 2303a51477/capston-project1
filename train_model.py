"""
train_model.py
Machine Learning Model Training Pipeline for AI-Based Student Performance Prediction System
Trains a Random Forest Regressor on student academic metrics and saves the model.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')
DATA_PATH = os.path.join(DATA_DIR, 'student_performance.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'student_model.joblib')
METRICS_PATH = os.path.join(MODEL_DIR, 'model_metrics.json')


def generate_realistic_dataset(num_samples: int = 1500, filepath: str = DATA_PATH) -> pd.DataFrame:
    """
    Generates a realistic student academic performance dataset based on
    validated educational performance factors and correlations.
    """
    np.random.seed(42)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # 1. Attendance percentage (50% - 100%, centered around 76%)
    attendance = np.clip(np.random.normal(loc=76.0, scale=12.0, size=num_samples), 45.0, 100.0)

    # 2. Daily Study Hours (0.5 to 10.0 hours, gamma distribution skewed toward 3-5 hrs)
    study_hours = np.clip(np.random.gamma(shape=3.5, scale=1.1, size=num_samples), 0.5, 10.0)

    # 3. Previous Semester Marks (35 - 98%)
    previous_semester_marks = np.clip(np.random.normal(loc=68.0, scale=13.0, size=num_samples), 35.0, 99.0)

    # 4. Internal Assessment Marks (correlated with previous marks + study hours + attendance)
    internal_base = (
        0.45 * previous_semester_marks +
        0.25 * (attendance * 0.9) +
        0.20 * (study_hours * 8.5) +
        np.random.normal(loc=0.0, scale=4.0, size=num_samples)
    )
    internal_marks = np.clip(internal_base, 30.0, 100.0)

    # 5. Assignment Marks (correlated with attendance & effort)
    assignment_base = (
        0.35 * attendance +
        0.30 * internal_marks +
        0.25 * (study_hours * 8.0) +
        np.random.normal(loc=5.0, scale=5.0, size=num_samples)
    )
    assignment_marks = np.clip(assignment_base, 40.0, 100.0)

    # 6. Final Exam Marks (Ground truth with educational weights + non-linear attendance effect)
    # Weights: Internal Marks: 30%, Previous Semester: 25%, Study Hours: 18%, Attendance: 15%, Assignments: 12%
    attendance_factor = np.where(attendance < 65.0, -6.0, np.where(attendance >= 85.0, 3.0, 0.0))
    study_diminishing = np.log1p(study_hours) * 11.5  # diminishing returns after ~6 hours

    final_marks_raw = (
        0.32 * internal_marks +
        0.24 * previous_semester_marks +
        0.18 * assignment_marks +
        0.14 * attendance +
        0.90 * study_diminishing +
        attendance_factor +
        np.random.normal(loc=-1.0, scale=3.2, size=num_samples)
    )
    final_marks = np.clip(np.round(final_marks_raw, 1), 25.0, 100.0)

    df = pd.DataFrame({
        'attendance': np.round(attendance, 1),
        'study_hours': np.round(study_hours, 1),
        'internal_marks': np.round(internal_marks, 1),
        'assignment_marks': np.round(assignment_marks, 1),
        'previous_semester_marks': np.round(previous_semester_marks, 1),
        'final_marks': final_marks
    })

    df.to_csv(filepath, index=False)
    print(f"Generated realistic student dataset saved to {filepath} ({len(df)} records)")
    return df


def train_and_evaluate(dataset_path: str = DATA_PATH):
    """
    Executes the complete ML workflow:
    1. Loads dataset
    2. Cleans & validates
    3. Splits into train & test
    4. Trains Random Forest Regressor
    5. Evaluates with MAE, MSE, RMSE, R2
    6. Saves trained model and evaluation report
    """
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}. Generating dataset now...")
        df = generate_realistic_dataset(1500, dataset_path)
    else:
        df = pd.read_csv(dataset_path)
        print(f"Loaded existing dataset from {dataset_path} ({len(df)} records)")

    # 1. Data Cleaning & Validation
    df = df.dropna()
    print("Dataset Summary:")
    print(df.describe().round(2))

    # 2. Features and Target
    feature_cols = [
        'attendance',
        'study_hours',
        'internal_marks',
        'assignment_marks',
        'previous_semester_marks'
    ]
    target_col = 'final_marks'

    X = df[feature_cols]
    y = df[target_col]

    # 3. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    print(f"\nTraining set size: {len(X_train)} samples")
    print(f"Testing set size:  {len(X_test)} samples")

    # 4. Train Random Forest Regressor
    print("\nTraining Random Forest Regressor...")
    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. Evaluate Model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\n================ MODEL EVALUATION ================")
    print(f"Mean Absolute Error (MAE): {mae:.3f} marks")
    print(f"Mean Squared Error (MSE):  {mse:.3f}")
    print(f"Root Mean Squared Error:   {rmse:.3f} marks")
    print(f"R-squared Score (R²):      {r2:.4f} ({r2*100:.2f}% variance explained)")
    print("==================================================")

    # Feature Importances
    importances = dict(zip(feature_cols, [round(float(v), 4) for v in model.feature_importances_]))
    print("\nFeature Importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {feat:25s}: {imp * 100:.2f}%")

    # 6. Save Model and Metadata
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nTrained Random Forest model successfully saved to: {MODEL_PATH}")

    metrics = {
        'mae': round(float(mae), 3),
        'mse': round(float(mse), 3),
        'rmse': round(float(rmse), 3),
        'r2_score': round(float(r2), 4),
        'feature_importances': importances,
        'features': feature_cols,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'model_type': 'RandomForestRegressor'
    }

    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)
    print(f"Evaluation metrics saved to: {METRICS_PATH}")

    return model, metrics


if __name__ == '__main__':
    train_and_evaluate()

