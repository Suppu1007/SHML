# =====================================================
# train_stroke_models.py
# Enhanced version: All models trained with SMOTE, tuning, and calibration.
# Includes Ensemble stacking and improved FNN (Adam LR=0.001, Dropout=0.4).
# =====================================================

import os
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# =====================================================
# Configuration
# =====================================================
MODEL_DIR = "models"
DATA_PATH = "data/Stroke.csv"
os.makedirs(MODEL_DIR, exist_ok=True)

FINAL_FEATURES = [
    'age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi',
    'gender_Male', 'smoking_status_formerly smoked', 'smoking_status_smokes'
]

# =====================================================
# Load & Preprocess Data
# =====================================================
try:
    df = pd.read_csv(DATA_PATH)
    df = df[df['gender'] != 'Other']

    X = df[['age', 'hypertension', 'heart_disease',
            'avg_glucose_level', 'bmi', 'gender', 'smoking_status']].copy()
    y = df['stroke']

    X['bmi'] = X['bmi'].fillna(X['bmi'].mean())

    X = pd.get_dummies(X, columns=['gender', 'smoking_status'],
                       drop_first=True, dtype=int)

    for col in FINAL_FEATURES:
        if col not in X.columns:
            X[col] = 0

except FileNotFoundError:
    print(f"⚠️ {DATA_PATH} not found. Generating synthetic dataset for testing.")
    X = pd.DataFrame({
        'age': np.random.randint(20, 80, 500),
        'hypertension': np.random.randint(0, 2, 500),
        'heart_disease': np.random.randint(0, 2, 500),
        'avg_glucose_level': np.random.uniform(70, 200, 500),
        'bmi': np.random.uniform(15, 40, 500),
        'gender_Male': np.random.randint(0, 2, 500),
        'smoking_status_formerly smoked': np.random.randint(0, 2, 500),
        'smoking_status_smokes': np.random.randint(0, 2, 500),
    })
    y = pd.Series(np.random.randint(0, 2, 500))

X = X.reindex(columns=FINAL_FEATURES, fill_value=0)
joblib.dump(FINAL_FEATURES, os.path.join(MODEL_DIR, "stroke_feature_names.joblib"))

# Split
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "stroke_scaler.joblib"))

# =====================================================
# Apply SMOTE
# =====================================================
print("\n--- Applying SMOTE ---")
print(f"Original training set: {Counter(y_train_raw)}")

smote = SMOTE(random_state=42)
X_train_scaled_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train_raw)
X_train_raw_smote, _ = smote.fit_resample(X_train_raw, y_train_raw)

print(f"SMOTE training set: {Counter(y_train_smote)}")

# =====================================================
# Define Models
# =====================================================

def create_fnn_model(input_dim):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(64, activation='relu'),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


models = {
    "xgb": XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, random_state=42, eval_metric='logloss'),
    
    "logistic": LogisticRegression(
        max_iter=5000, C=10.0, solver='liblinear', random_state=42, class_weight='balanced'),
    
    "mlp": MLPClassifier(
        hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
        alpha=0.0005, learning_rate_init=0.001, max_iter=1000,
        early_stopping=True, random_state=42),
    
    "svm": SVC(
        kernel='rbf', probability=True, C=10.0, gamma='scale', class_weight='balanced', random_state=42),
    
    "rf": RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=2, class_weight='balanced', random_state=42),
    
    "fnn": create_fnn_model(X_train_scaled_smote.shape[1])
}

# =====================================================
# Train & Save Models
# =====================================================
for name, model in models.items():
    print(f"\n--- Training {name.upper()} ---")
    
    if name == 'fnn':
        model.fit(X_train_scaled_smote, y_train_smote,
                  epochs=80, batch_size=32, verbose=0,
                  validation_split=0.2)
        model.save(os.path.join(MODEL_DIR, f"stroke_{name}.h5"))
        print(f"✅ Saved FNN model (.h5)")
    
    elif name in ['logistic', 'mlp', 'svm']:
        model.fit(X_train_scaled_smote, y_train_smote)
        calibrated = CalibratedClassifierCV(model, cv=5)
        calibrated.fit(X_train_scaled_smote, y_train_smote)
        joblib.dump(calibrated, os.path.join(MODEL_DIR, f"stroke_{name}.joblib"))
        print(f"✅ Saved {name.upper()} model with calibration")
    
    else:  # RF and XGB
        model.fit(X_train_raw_smote, y_train_smote)
        calibrated = CalibratedClassifierCV(model, cv=5)
        calibrated.fit(X_train_raw_smote, y_train_smote)
        joblib.dump(calibrated, os.path.join(MODEL_DIR, f"stroke_{name}.joblib"))
        print(f"✅ Saved {name.upper()} model with calibration")

# =====================================================
# Evaluate Models
# =====================================================
print("\n--- Model Evaluation (on test set) ---")
for name, model in models.items():
    X_eval = X_test_scaled if name in ['logistic', 'mlp', 'svm', 'fnn'] else X_test
    
    if name == 'fnn':
        y_prob = model.predict(X_eval).flatten()
    else:
        try:
            loaded = joblib.load(os.path.join(MODEL_DIR, f"stroke_{name}.joblib"))
            y_prob = loaded.predict_proba(X_eval)[:, 1]
        except Exception:
            print(f"⚠️ Skipping {name.upper()} (not probability model).")
            continue

    y_pred = (y_prob > 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f"{name.upper():<8} → Accuracy: {acc:.3f}, AUC: {auc:.3f}")

# =====================================================
# Ensemble Model (Optional)
# =====================================================
print("\n--- Training Ensemble (Stacking) ---")
base_estimators = [
    ('rf', joblib.load(os.path.join(MODEL_DIR, "stroke_rf.joblib"))),
    ('xgb', joblib.load(os.path.join(MODEL_DIR, "stroke_xgb.joblib"))),
    ('svm', joblib.load(os.path.join(MODEL_DIR, "stroke_svm.joblib"))),
    ('logistic', joblib.load(os.path.join(MODEL_DIR, "stroke_logistic.joblib"))),
]

meta_model = LogisticRegression(max_iter=5000, solver='liblinear', random_state=42)
stack = StackingClassifier(estimators=base_estimators, final_estimator=meta_model, cv=5, n_jobs=-1)
stack.fit(X_train_scaled_smote, y_train_smote)
joblib.dump(stack, os.path.join(MODEL_DIR, "stroke_ensemble.joblib"))

# Evaluate ensemble
y_prob_ens = stack.predict_proba(X_test_scaled)[:, 1]
y_pred_ens = (y_prob_ens > 0.5).astype(int)
acc_ens = accuracy_score(y_test, y_pred_ens)
auc_ens = roc_auc_score(y_test, y_prob_ens)
print(f"\nENSEMBLE → Accuracy: {acc_ens:.3f}, AUC: {auc_ens:.3f}")
print("✅ All models trained, calibrated, and saved successfully.")
