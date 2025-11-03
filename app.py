import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model 

# --- AUTHENTICATION IMPORTS AND CONFIGURATION ---
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
# Import the database functions from database.py (REQUIRED FILE)
from database import get_user_by_email, create_user 
# --- END AUTH IMPORTS ---

# =====================================================
# Configuration
# =====================================================
MODELS_DIR = "models"
BREAST_CANCER_30_FEATURES = 30 # For the initial 30-feature dashboard
BREAST_CANCER_9_FEATURES = 9  # For the new 9-feature dashboard

# =====================================================
# Flask Initialization
# =====================================================
app = Flask(__name__)
CORS(app)

# IMPORTANT: Set a secret key for session management (change this to a strong, random value)
# Reads from environment variable or uses a default if not found
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'a_highly_secret_and_complex_key_12345')

# Set session lifespan for the authentication system
app.permanent_session_lifetime = timedelta(minutes=30)


# =====================================================
# Global Model Variables
# =====================================================
# Stroke models (Retained)
STROKE_MODELS = {'logistic': None, 'mlp': None, 'svm': None, 'rf': None, 'xgb': None, 'fnn': None}
STROKE_SCALER = None
STROKE_FEATURE_NAMES = None

# 30-Feature Breast Cancer models (Retained)
BREAST_MODELS_30 = {'rf': None, 'xgb': None, 'cnn': None, 'fnn': None}
BREAST_SCALER_30 = None

# NEW: 9-Feature Breast Cancer models
BREAST_MULTI_MODELS = {'rf': None, 'cnn': None, 'fnn': None}
BREAST_MULTI_SCALER = None

# =====================================================
# Model Loading
# =====================================================
def load_models():
    global STROKE_MODELS, STROKE_SCALER, STROKE_FEATURE_NAMES
    global BREAST_MODELS_30, BREAST_SCALER_30
    global BREAST_MULTI_MODELS, BREAST_MULTI_SCALER
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

    # --- Load Stroke Models ---
    print("--- Loading Stroke Models ---")
    try:
        STROKE_SCALER = joblib.load(os.path.join(MODELS_DIR, "stroke_scaler.joblib"))
        STROKE_FEATURE_NAMES = joblib.load(os.path.join(MODELS_DIR, "stroke_feature_names.joblib"))
        
        for name in STROKE_MODELS.keys():
            if name == 'fnn':
                STROKE_MODELS[name] = load_model(os.path.join(MODELS_DIR, f"stroke_{name}.h5"))
            else:
                STROKE_MODELS[name] = joblib.load(os.path.join(MODELS_DIR, f"stroke_{name}.joblib"))
        print("✅ Stroke Models Loaded")
    except Exception as e:
        print(f"❌ Error loading stroke models. Error: {e}")
        STROKE_MODELS = {k: None for k in STROKE_MODELS}

    # --- Load 30-Feature Breast Cancer Models ---
    print("--- Loading 30-Feature Breast Cancer Models ---")
    try:
        BREAST_SCALER_30 = joblib.load(os.path.join(MODELS_DIR, "breast_scaler.joblib"))
        
        for name in BREAST_MODELS_30.keys():
            if name in ['cnn', 'fnn']:
                BREAST_MODELS_30[name] = load_model(os.path.join(MODELS_DIR, f"breast_{name}.h5"))
            else:
                BREAST_MODELS_30[name] = joblib.load(os.path.join(MODELS_DIR, f"breast_{name}.joblib"))
        print("✅ 30-Feature Models Loaded")
    except Exception as e:
        print(f"⚠️ 30-Feature models not fully loaded. Error: {e}")
        BREAST_MODELS_30 = {k: None for k in BREAST_MODELS_30}

    # --- NEW: Load 9-Feature Breast Cancer Models ---
    print("--- Loading Multi-Value (9-Feature) Breast Cancer Models ---")
    try:
        BREAST_MULTI_SCALER = joblib.load(os.path.join(MODELS_DIR, "breast_multi_scaler.joblib"))
        
        for name in BREAST_MULTI_MODELS.keys():
            if name in ['cnn', 'fnn']:
                # Load Keras models (FNN, CNN)
                BREAST_MULTI_MODELS[name] = load_model(os.path.join(MODELS_DIR, f"breast_multi_{name}.h5"))
            else:
                # Load joblib models (RF)
                BREAST_MULTI_MODELS[name] = joblib.load(os.path.join(MODELS_DIR, f"breast_multi_{name}.joblib"))
        print("✅ 9-Feature Multi-Value Models Loaded")
    except Exception as e:
        print(f"⚠️ 9-Feature Multi-Value models not fully loaded. Error: {e}")
        BREAST_MULTI_MODELS = {k: None for k in BREAST_MULTI_MODELS}


with app.app_context():
    load_models()

# =====================================================
# Utility Functions (Stroke Prediction - Unchanged)
# =====================================================
# =====================================================
# Utility Functions (Stroke Prediction - Improved)
# =====================================================
from sklearn.calibration import CalibratedClassifierCV

def calibrate_model_if_needed(model, X_sample, y_sample):
    """Wraps RF, XGB, SVM, Logistic with calibration if not already done."""
    try:
        if hasattr(model, "predict_proba"):
            calibrated = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
            calibrated.fit(X_sample, y_sample)
            return calibrated
    except Exception as e:
        print(f"⚠️ Calibration failed: {e}")
    return model

# 1️⃣ Map frontend codes
def map_frontend_to_model(gender_code, smoking_code):
    gender_map = {0: 'Female', 1: 'Male', 2: 'Other'}
    smoking_map = {0: 'never smoked', 1: 'formerly smoked', 2: 'smokes'}
    gender_str = gender_map.get(int(gender_code), 'Female')
    smoking_str = smoking_map.get(int(smoking_code), 'never smoked')
    return gender_str, smoking_str

# 2️⃣ Preprocess stroke input
def preprocess_stroke_input(data):
    gender_str, smoking_str = map_frontend_to_model(
        data.get('gender', 0), data.get('smoking_status', 0)
    )
    df = pd.DataFrame([{
        'age': data.get('age',0),
        'hypertension': data.get('hypertension',0),
        'heart_disease': data.get('heart_disease',0),
        'avg_glucose_level': data.get('avg_glucose',0),
        'bmi': data.get('bmi',0),
        'gender': gender_str, 
        'smoking_status': smoking_str 
    }])
    df['bmi'] = df['bmi'].fillna(0)
    df_encoded = pd.get_dummies(df, columns=['gender','smoking_status'], drop_first=True, dtype=int)
    final_df = pd.DataFrame(0, index=[0], columns=STROKE_FEATURE_NAMES)
    for col in df_encoded.columns:
        if col in STROKE_FEATURE_NAMES:
            final_df[col] = df_encoded[col]
    return final_df.reindex(columns=STROKE_FEATURE_NAMES, fill_value=0)

# 3️⃣ Mock feature contributions
def get_mock_feature_contributions(X_input, prob, model_type):
    age = X_input['age'].iloc[0]
    hypertension = X_input['hypertension'].iloc[0]
    heart_disease = X_input['heart_disease'].iloc[0]
    avg_glucose_level = X_input['avg_glucose_level'].iloc[0]
    bmi = X_input['bmi'].iloc[0]
    is_male = X_input.get('gender_Male', pd.Series([0])).iloc[0]
    is_former_smoker = X_input.get('smoking_status_formerly smoked', pd.Series([0])).iloc[0]
    is_smoker = X_input.get('smoking_status_smokes', pd.Series([0])).iloc[0]
    
    gender_str = 'Male' if is_male == 1 else 'Female'
    smoking_str = 'Smokes' if is_smoker == 1 else ('Formerly Smoked' if is_former_smoker == 1 else 'Never Smoked')

    risk_points = {
        'Age': round(max(0, age - 50) * 0.2 + (prob/100)*15, 1),
        'Hypertension': round(hypertension * 7.0 + (prob/100)*10, 1),
        'Heart Disease': round(heart_disease * 8.0 + (prob/100)*10, 1),
        'Avg. Glucose': round(max(0, avg_glucose_level - 120) * 0.2 + (prob/100)*5, 1),
        'BMI': round(max(0, bmi - 30) * 0.5 + (prob/100)*2, 1),
        'Gender': round(is_male * 1.5 + (prob/100)*1, 1),
        'Smoking Status': round((is_smoker * 5.0) + (is_former_smoker * 3.0) + (prob/100)*5, 1),
    }

    factors = {
        'Age': f"{age:.0f} years (Risk Score: {risk_points['Age']:.1f})",
        'Hypertension': f"{'Yes' if hypertension == 1 else 'No'} (Risk Score: {risk_points['Hypertension']:.1f})",
        'Heart Disease': f"{'Yes' if heart_disease == 1 else 'No'} (Risk Score: {risk_points['Heart Disease']:.1f})",
        'Avg. Glucose': f"{avg_glucose_level:.1f} mg/dL (Risk Score: {risk_points['Avg. Glucose']:.1f})",
        'BMI': f"{bmi:.1f} (Risk Score: {risk_points['BMI']:.1f})",
        'Gender': f"{gender_str} (Risk Score: {risk_points['Gender']:.1f})",
        'Smoking Status': f"{smoking_str} (Risk Score: {risk_points['Smoking Status']:.1f})",
    }
    return factors

# 4️⃣ Predict stroke
def predict_stroke(data, model_type):
    model_type = model_type.lower()
    model = STROKE_MODELS.get(model_type)
    
    if model is None:
        raise ValueError(f"Stroke model '{model_type}' not found or loaded.")

    X_input = preprocess_stroke_input(data)
    is_scaled_model = model_type in ['logistic','mlp','svm', 'fnn'] 

    if is_scaled_model:
        X_scaled = STROKE_SCALER.transform(X_input)
        if model_type == 'fnn':
            prob = model.predict(X_scaled)[0][0] 
        elif hasattr(model, "predict_proba"):
            prob = model.predict_proba(X_scaled)[0][1]
        else:
            prob = model.decision_function(X_scaled)[0] 
            prob = 1 / (1 + np.exp(-prob))
    else:
        prob = model.predict_proba(X_input)[0][1]

    risk = "Low Risk" if prob < 0.1 else ("Moderate Risk" if prob < 0.3 else "High Risk")
    factors = get_mock_feature_contributions(X_input, prob, model_type)
    
    return round(prob*100,2), risk, factors



# =====================================================
# Utility Functions (Breast Cancer Prediction - 30 Features)
# =====================================================
def predict_breast_30(features, model_type='rf'):
    """Predicts breast cancer malignancy using the 30-feature models."""
    model_type = model_type.lower()
    model = BREAST_MODELS_30.get(model_type)
    
    if model is None:
        raise ValueError(f"30-Feature Breast model '{model_type}' not found.")
    if len(features) != BREAST_CANCER_30_FEATURES:
        raise ValueError(f"Expected {BREAST_CANCER_30_FEATURES} features, got {len(features)}.")

    x_input = np.array([features], dtype=np.float32)

    # Models using raw input: RF, XGB
    if model_type in ['rf', 'xgb']:
        prob = model.predict_proba(x_input)[0][1]
    
    # Deep Learning models using scaled input: FNN, CNN
    else: 
        x_scaled = BREAST_SCALER_30.transform(x_input)
        
        if model_type == 'cnn':
            # CNN requires reshaping: (1, 30, 1)
            x_scaled = x_scaled.reshape((1, BREAST_CANCER_30_FEATURES, 1))
        
        prob = model.predict(x_scaled)[0][0]

    label = "Malignant" if prob >= 0.5 else "Benign"
    confidence = prob*100 if label=="Malignant" else (100-prob*100)
    
    return label, round(confidence,2), model_type.upper()

# =====================================================
# NEW Utility Functions (Breast Cancer Prediction - 9 Features)
# =====================================================
def predict_breast_multi(features, model_type='rf'):
    """Predicts breast cancer malignancy using the 9-feature models."""
    model_type = model_type.lower()
    model = BREAST_MULTI_MODELS.get(model_type)
    
    if model is None:
        raise ValueError(f"9-Feature Breast model '{model_type}' not found.")
    if len(features) != BREAST_CANCER_9_FEATURES:
        raise ValueError(f"Expected {BREAST_CANCER_9_FEATURES} features, got {len(features)}.")

    x_input = np.array([features], dtype=np.float32)

    # Models using raw input: RF (joblib model)
    if model_type in ['rf']:
        prob = model.predict_proba(x_input)[0][1]
    
    # Deep Learning models using scaled input: FNN, CNN (Keras models)
    else: 
        x_scaled = BREAST_MULTI_SCALER.transform(x_input)
        
        if model_type == 'cnn':
            # CNN requires reshaping: (1, 9, 1)
            x_scaled = x_scaled.reshape((1, BREAST_CANCER_9_FEATURES, 1))
        
        prob = model.predict(x_scaled)[0][0]

    label = "Malignant" if prob >= 0.5 else "Benign"
    confidence = prob*100 if label=="Malignant" else (100-prob*100)
    
    return label, round(confidence,2), model_type.upper()


# =====================================================
# API Routes (ML Prediction Endpoints)
# =====================================================
@app.route("/predict_stroke", methods=["POST"])
def stroke_route():
    data = request.get_json()
    model_type = data.get("model_type","xgb") 
    
    try:
        prob, risk_label, factors = predict_stroke(data, model_type)
        
        return jsonify({
            "stroke_risk": f"{prob:.2f}%",
            "risk_label": risk_label,
            "model_type": model_type.upper(),
            "factors": factors
        })
    except Exception as e:
        print(f"Stroke prediction failed: {e}")
        return jsonify({"error": "Prediction service failed. Check Python console.", "details": str(e)}), 500

# Endpoint for the existing 30-feature dashboard
@app.route("/predict_breast", methods=["POST"])
def breast_route_30():
    data = request.get_json()
    features = data.get("features", [])
    model_type = data.get("model_type","rf")
    try:
        label, confidence, model_used = predict_breast_30(features, model_type)
        return jsonify({
            "result": label,
            "confidence": f"{confidence:.2f}%",
            "model_used": model_used
        })
    except Exception as e:
        print(f"30-feature breast cancer prediction failed: {e}")
        return jsonify({"error": str(e)}), 500

# NEW Endpoint for the 9-feature dashboard
@app.route("/predict_breast_multi", methods=["POST"])
def breast_route_multi():
    data = request.get_json()
    features = data.get("features", [])
    model_type = data.get("model_type","rf")
    try:
        label, confidence, model_used = predict_breast_multi(features, model_type)
        return jsonify({
            "result": label,
            "confidence": f"{confidence:.2f}%",
            "model_used": model_used
        })
    except Exception as e:
        print(f"9-feature multi-value breast cancer prediction failed: {e}")
        return jsonify({"error": str(e)}), 500


# =====================================================
# Authentication & Template Routes (Functional Login/Register)
# =====================================================
@app.route("/")
def index():
    """Landing page."""
    return render_template("index.html")

# ------------------- Registration -------------------

@app.route('/Register', methods=['GET', 'POST'])
def register():
    """Handles user registration."""
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Basic validation
        if not email or not password:
            flash("Email and Password are required.", "danger")
            return redirect(url_for('register'))

        try:
            # Check if user already exists using the imported DB function
            if get_user_by_email(email):
                flash("An account with that email already exists.", "danger")
                return redirect(url_for('register'))
            
            # Hash the password for security
            hashed_password = generate_password_hash(password)
            
            # Store user in MongoDB using the imported DB function
            # The create_user function saves the hash under the key 'password'
            create_user(email, hashed_password) 
            
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))
        except Exception as e:
            print(f"Database error during registration: {e}")
            flash("A server error occurred during registration. Please try again.", "danger")
            return redirect(url_for('register'))
        
    return render_template('Register.html')

@app.route('/Login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        # *** MODIFICATION: Get JSON data from the AJAX request ***
        data = request.get_json()
        
        # Check if the request is JSON and contains data
        if not data:
             return jsonify({"error": "Request body must be valid JSON."}), 400

        email = data.get('email')
        password = data.get('password')

        if not email or not password:
            return jsonify({"error": "Email and Password fields are required."}), 400

        try:
            user = get_user_by_email(email)
        except Exception as e:
            # Catch database connection or query errors
            print(f"Database error during login query: {e}")
            return jsonify({"error": "Connection error: Could not verify credentials. Please try again."}), 500

        # The MongoDB function returns the hash under the 'password' key
        if user and check_password_hash(user['password'], password):
            # Login Success
            session['logged_in'] = True
            session['email'] = email
            session.permanent = True
            
            # *** MODIFICATION: Return JSON Success for AJAX ***
            # The client-side JS handles the redirect to 'home'
            return jsonify({"message": "Login successful!", "redirect_to": url_for('home')}), 200
        else:
            # Login Failure
            # *** MODIFICATION: Return JSON Failure (401 Unauthorized) for AJAX ***
            return jsonify({"error": "Invalid email or password."}), 401

    # For a GET request, just render the template as before
    return render_template('Login.html')


# ------------------- Logout -------------------
@app.route('/logout')
def logout():
    """
    Logout Route
    ------------
    Logs out the current user by clearing session data,
    flashes a confirmation message, and redirects to the index page.
    """
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for('index'))


# ------------------- Dashboard -------------------
@app.route('/home')
def home():
    if 'logged_in' not in session:
        flash("Please log in to access the dashboard.", "warning")
        return redirect(url_for('login'))
    
    user_email = session.get('email', 'Guest')
    return render_template('home.html', email=user_email)

# ------------------- Profile -------------------

@app.route('/profile')
def user_profile():
    """
    User Profile Route
    ------------------
    Renders the logged-in user's profile page.
    Includes session validation, user data retrieval, and flash message handling.
    """
    # 1️⃣ Check login session
    user_email = session.get('email')
    if not user_email or not session.get('logged_in'):
        flash("Please log in to view your profile.", "warning")
        return redirect(url_for('login'))

    # 2️⃣ Retrieve user details from the database
    user_data = get_user_by_email(user_email)
    if not user_data:
        flash("Profile data not found. Please log in again.", "danger")
        session.pop('logged_in', None)
        session.pop('email', None)
        return redirect(url_for('login'))

    # 3️⃣ Render profile template with user data
    return render_template('profile.html', user=user_data)

# ------------------- Dashboards -------------------
@app.route("/breast_cancer") 
def breast_cancer_30(): 
    if 'logged_in' not in session: 
        flash("Please log in to access the dashboard.", "warning") 
        return redirect(url_for('login')) # Assuming this route serves the original 30-feature dashboard 
    return render_template("breast_dashboard.html") 
        
@app.route("/breast_multi") 
def breast_cancer_multi(): 
    if 'logged_in' not in session: 
        flash("Please log in to access the dashboard.", "warning") 
        return redirect(url_for('login')) # NEW route for the 9-feature dashboard 
    return render_template("breast_multi_dashboard.html")

# ------------------- Other Pages -------------------
@app.route("/usybrainmptoms")
def usybrainmptoms():
    if 'logged_in' not in session:
        flash("Please log in to view this page.", "warning")
        return redirect(url_for('login'))
    return render_template("usybrainmptoms.html")

@app.route("/modelperformance")
def modelperformance():
    if 'logged_in' not in session:
        flash("Please log in to view model performance.", "warning")
        return redirect(url_for('login'))
    return render_template("modelperformance.html")

@app.route("/report")
def report():
    if 'logged_in' not in session:
        flash("Please log in to view reports.", "warning")
        return redirect(url_for('login'))
    return render_template("report.html")
# =====================================================
# Run Flask App
# =====================================================
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # NOTE: Run with 'flask run' or use the app.run() below.
    app.run(debug=True, host='127.0.0.1', port=5000)