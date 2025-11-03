import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight # Import for class weighting
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

# ==========================================
# Create models directory if not exists
# ==========================================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ==========================================
# Load and Preprocess the Custom Dataset
# ==========================================
FILE_PATH = "data/Breast_cancer_dataset.csv" # Adjusted to relative path

try:
    df = pd.read_csv(FILE_PATH)
except FileNotFoundError:
    print(f"Error: The file {FILE_PATH} was not found. Please check your data directory.")
    exit()

# 1. Drop irrelevant columns: 'id' and 'Unnamed: 32' (all NaN)
df = df.drop(columns=['id', 'Unnamed: 32'], axis=1)

# 2. Encode the target variable 'diagnosis': M -> 1 (Malignant), B -> 0 (Benign)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# 3. Define Features (X) and Target (y)
X = df.drop('diagnosis', axis=1).values 
y = df['diagnosis'].values           

print(f"Dataset loaded from CSV: {X.shape[0]} samples, {X.shape[1]} features.")
print(f"Target count: Benign (0): {np.sum(y == 0)}, Malignant (1): {np.sum(y == 1)}")

# Calculate Class Weights (CRITICAL FOR IMBALANCE/ACCURACY)
# This assigns higher penalty to the majority class (Benign=0) to focus training on Malignant cases.
classes = np.unique(y)
weights = compute_class_weight('balanced', classes=classes, y=y)
class_weight_dict = dict(zip(classes, weights))
print(f"Calculated Class Weights: {class_weight_dict}")


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(MODEL_DIR, "breast_scaler.joblib"))

# Feature dimension (30 features)
input_dim = X_train.shape[1]

# ==========================================
# A. Random Forest (Optimized)
# ==========================================
print("\n--- Training Random Forest ---")
rf_model = RandomForestClassifier(
    n_estimators=300, # Increased estimators
    max_depth=10, 
    class_weight=class_weight_dict, # Use calculated weights
    random_state=42
)
rf_model.fit(X_train, y_train) # RF uses raw data
joblib.dump(rf_model, os.path.join(MODEL_DIR, "breast_rf.joblib"))

# ==========================================
# B. XGBoost (Optimized)
# ==========================================
print("\n--- Training XGBoost ---")
# Calculate scale_pos_weight for XGBoost (Malignant / Benign)
ratio = np.sum(y == 0) / np.sum(y == 1) 

xgb_model = XGBClassifier(
    n_estimators=300, 
    max_depth=5,
    scale_pos_weight=ratio, # Specific imbalance parameter for XGBoost
    use_label_encoder=False, 
    eval_metric='logloss', 
    random_state=42
)
xgb_model.fit(X_train, y_train) # XGBoost uses raw data
joblib.dump(xgb_model, os.path.join(MODEL_DIR, "breast_xgb.joblib"))


# ==========================================
# C. FNN Model (Optimized)
# ==========================================
print("\n--- Training FNN ---")
# FNN Optimization: Custom Adam optimizer, L2 regularization, increased Dropout
fnn_optimizer = Adam(learning_rate=0.0005) # Lower LR for better convergence

fnn_model = Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    Dense(128, activation='relu', kernel_regularizer=l2(0.001)), # L2 Regularization
    Dropout(0.4), # Increased Dropout
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
fnn_model.compile(optimizer=fnn_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
fnn_model.fit(
    X_train_scaled, 
    y_train, 
    validation_split=0.1, 
    epochs=50, 
    batch_size=32, 
    verbose=0,
    class_weight=class_weight_dict # Use calculated weights
)
fnn_model.save(os.path.join(MODEL_DIR, "breast_fnn.h5"))

# ==========================================
# D. CNN Model (Optimized)
# ==========================================
print("\n--- Training CNN ---")
# CNN expects (samples, features, channels)
X_train_cnn = X_train_scaled.reshape((-1, input_dim, 1))
X_test_cnn = X_test_scaled.reshape((-1, input_dim, 1))

cnn_optimizer = Adam(learning_rate=0.0005) # Consistent lower LR

cnn_model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(input_dim, 1), kernel_regularizer=l2(0.001)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
cnn_model.compile(optimizer=cnn_optimizer, loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(
    X_train_cnn, 
    y_train, 
    validation_split=0.1, 
    epochs=50, 
    batch_size=32, 
    verbose=0,
    class_weight=class_weight_dict # Use calculated weights
)
cnn_model.save(os.path.join(MODEL_DIR, "breast_cnn.h5"))

# ==========================================
# Evaluation
# ==========================================
def evaluate_model(model, X_test_data, y_true, name):
    if name in ['cnn']:
        # Reshape CNN test data
        X_test_data = X_test_data.reshape((-1, X_test_data.shape[1], 1))
        
    if name in ['rf', 'xgb']:
        # Tree models use raw data
        y_prob = model.predict_proba(X_test_data)[:, 1]
    elif name in ['fnn', 'cnn']:
        # Keras models use scaled data
        y_prob = model.predict(X_test_data).flatten()
    else:
        # Fallback (shouldn't happen with these models)
        y_prob = model.predict_proba(X_test_data)[:, 1]
        
    y_pred = (y_prob > 0.5).astype(int)
    
    # Calculate accuracy and report
    acc = accuracy_score(y_true, y_pred)
    print(f"\n🔹 {name.upper()} Accuracy:", acc)
    print(f"{name.upper()} Classification Report:\n", classification_report(y_true, y_pred, target_names=['Benign (0)', 'Malignant (1)']))

print("\n--- Model Evaluation ---")
# RF and XGBoost use raw test data
evaluate_model(rf_model, X_test, y_test, 'rf')
evaluate_model(xgb_model, X_test, y_test, 'xgb')

# FNN and CNN use scaled test data
evaluate_model(fnn_model, X_test_scaled, y_test, 'fnn')
evaluate_model(cnn_model, X_test_scaled, y_test, 'cnn')
