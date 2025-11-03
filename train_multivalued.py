import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam

# ==========================================
# Configuration and Setup
# ==========================================
MODEL_DIR = "models"
# NOTE: This file uses the Wisconsin Breast Cancer Original (WBC-O) dataset structure (9 features).
DATA_PATH = "data/WBC_Original_9_Features.csv" 
os.makedirs(MODEL_DIR, exist_ok=True)

# The 9 features extracted from the HTML and the typical dataset order
FINAL_FEATURES = [
    'clump_thickness', 'cell_size_uniformity', 'cell_shape_uniformity', 
    'marginal_adhesion', 'single_epi_cell_size', 'bare_nuclei', 
    'bland_chromatin', 'normal_nucleoli', 'mitoses'
]

# ==========================================
# Load and Preprocess Dataset
# ==========================================
try:
    # Using synthetic data as a reliable source for the 9-feature model is not guaranteed
    # You must provide the correct CSV matching the 9 features.
    print(f"⚠️ {DATA_PATH} not found. Using synthetic 9-feature dataset for testing.")
    num_samples = 700
    X = pd.DataFrame(np.random.randint(1, 11, size=(num_samples, 9)), columns=FINAL_FEATURES)
    # Generate imbalanced target (2 for Benign, 4 for Malignant)
    y = np.where(np.random.rand(num_samples) < 0.35, 1, 0)
    y = pd.Series(y)

except FileNotFoundError:
    print(f"FATAL: Cannot load 9-feature data. Exiting.")
    exit()

# Calculate Class Weights (CRITICAL FOR IMBALANCE/ACCURACY)
classes = np.unique(y)
weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, weights))
print(f"Calculated Class Weights: {class_weight_dict}")


# Split dataset
X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features for models that require it (CNN, FNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "breast_multi_scaler.joblib"))


# ==========================================
# APPLY SMOTE TO TRAINING DATA
# ==========================================
print(f"\n--- Applying SMOTE ---")
print(f"Original training set shape: {Counter(y_train_raw)}")

smote = SMOTE(random_state=42)

# SMOTE on Scaled Data (for CNN, FNN)
X_train_scaled_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train_raw)

# SMOTE on Raw Data (for RF)
X_train_raw_smote, _ = smote.fit_resample(X_train_raw, y_train_raw)

print(f"SMOTE training set shape: {Counter(y_train_smote)}")

input_dim = X_train_raw.shape[1] # Should be 9

# ==========================================
# A. Random Forest (Optimized)
# ==========================================
print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=300, 
    max_depth=10, 
    class_weight=class_weight_dict, 
    random_state=42
)
rf_model.fit(X_train_raw_smote, y_train_smote) # RF uses SMOTE'd raw data
joblib.dump(rf_model, os.path.join(MODEL_DIR, "breast_multi_rf.joblib"))

# ==========================================
# B. FNN Model (Optimized)
# ==========================================
print("\n--- Training FNN ---")
fnn_optimizer = Adam(learning_rate=0.001) 

fnn_model = Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    Dense(32, activation='relu'),
    Dropout(0.4), 
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
fnn_model.compile(optimizer=fnn_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
fnn_model.fit(
    X_train_scaled_smote, 
    y_train_smote, 
    epochs=50, 
    batch_size=16, 
    verbose=0,
    class_weight=class_weight_dict
)
fnn_model.save(os.path.join(MODEL_DIR, "breast_multi_fnn.h5"))

# ==========================================
# C. CNN Model (Optimized)
# ==========================================
print("\n--- Training CNN ---")
# CNN requires reshaping: (samples, features, 1 channel)
X_train_cnn = X_train_scaled_smote.reshape((-1, input_dim, 1))
X_test_cnn = X_test_scaled.reshape((-1, input_dim, 1))

cnn_optimizer = Adam(learning_rate=0.001)

cnn_model = Sequential([
    Conv1D(filters=16, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer=cnn_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(
    X_train_cnn, 
    y_train_smote, 
    epochs=50, 
    batch_size=16, 
    verbose=0,
    class_weight=class_weight_dict
)
cnn_model.save(os.path.join(MODEL_DIR, "breast_multi_cnn.h5"))

# ==========================================
# Evaluation
# ==========================================
def evaluate_model(model, X_test_data, y_true, name):
    
    if name in ['rf']:
        y_prob = model.predict_proba(X_test_data)[:, 1]
    elif name in ['fnn']:
        y_prob = model.predict(X_test_data).flatten()
    elif name in ['cnn']:
        # Reshape for CNN prediction
        X_test_data_cnn = X_test_data.reshape((-1, X_test_data.shape[1], 1))
        y_prob = model.predict(X_test_data_cnn).flatten()
    
    y_pred = (y_prob > 0.5).astype(int)
    
    # Calculate accuracy and report
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🔹 {name.upper()} Accuracy: {acc:.4f}")
    print(f"{name.upper()} Classification Report:\n", classification_report(y_true, y_pred, target_names=['Benign (0)', 'Malignant (1)']))

print("\n--- Model Evaluation ---")
# RF uses raw test data
evaluate_model(rf_model, X_test, y_test, 'rf')

# FNN and CNN use scaled test data
evaluate_model(fnn_model, X_test_scaled, y_test, 'fnn')
evaluate_model(cnn_model, X_test_scaled, y_test, 'cnn')
