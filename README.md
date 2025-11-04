# 🧠 Smart Health Prediction Using Machine Learning

This repository contains the full implementation of a **Smart Health Prediction System** built using **Flask**, **TensorFlow/Keras**, **Scikit-Learn**, and **MongoDB**. The project applies **Machine Learning** and **Deep Learning** models to predict two major health conditions — **Brain Stroke** and **Breast Cancer** — based on user-provided health metrics.

---

## 🚀 Overview

The **Smart Health Prediction** project is a web-based application that uses trained ML/DL models to forecast disease likelihood based on clinical or lifestyle inputs. Users can log in to the system, enter their health details, and instantly receive predictions. Admins can manage user accounts and monitor overall model performance.

---

## 🧩 Features

* **User Authentication**: Secure registration and login with hashed passwords (Flask sessions + Werkzeug security).
* **Disease Prediction Dashboards**:

  * **Brain Stroke Prediction** – Logistic Regression, SVM, Random Forest, XGBoost, and Neural Network models.
  * **Breast Cancer Prediction** – Two variants:

    * *30-feature dataset*: RF, XGBoost, CNN, FNN.
    * *9-feature multi-model dataset*: RF, CNN, FNN.
* **Scalable API Endpoints** (`/predict_stroke`, `/predict_breast`, `/predict_breast_multi`) for integrating predictions.
* **Flask + TensorFlow Integration** for model serving.
* **Model Calibration** for probabilistic accuracy using `CalibratedClassifierCV`.
* **Dynamic Frontend** using HTML, CSS, JS templates for interaction.
* **Session-based Dashboards** for authenticated users.
* **Visualization-ready structure** for performance and risk factor display.

---

## 🧠 Machine Learning Models

| Disease                     | Models Used                                                | Frameworks               |
| --------------------------- | ---------------------------------------------------------- | ------------------------ |
| Stroke Prediction           | Logistic Regression, MLP, SVM, Random Forest, XGBoost, FNN | Scikit-learn, Keras      |
| Breast Cancer (30 features) | Random Forest, XGBoost, CNN, FNN                           | Scikit-learn, TensorFlow |
| Breast Cancer (9 features)  | Random Forest, CNN, FNN                                    | Scikit-learn, TensorFlow |

All models are pre-trained and stored in the `/models/` directory.

---

## 🧰 Tech Stack

* **Backend:** Flask (Python)
* **Frontend:** HTML, CSS, JavaScript
* **ML/DL Frameworks:** Scikit-learn, TensorFlow/Keras, XGBoost
* **Database:** MongoDB  (for user management)
* **Utilities:** Pandas, NumPy, Joblib
* **Deployment:** Gunicorn / Render / Localhost

---

## 📁 Repository Structure

```
├── app.py                     # Main Flask backend
├── database.py                # MongoDB/PostgreSQL user management functions
├── models/                    # Trained models (.joblib and .h5)
├── templates/                 # HTML templates (Login, Dashboard, etc.)
├── static/                    # CSS, JS, and assets
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/Smart-Health-Prediction-ML.git
cd Smart-Health-Prediction-ML
pip install -r requirements.txt
python app.py
```

Then open **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** in your browser.

---

## 🧪 Datasets

* **Stroke Dataset:** Kaggle Health Datasets
* **Breast Cancer Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
* Preprocessed and standardized with `joblib` scalers.

---

## 📊 Results

* Stroke Prediction Accuracy: ~94%
* Breast Cancer Prediction (30 features): ~98%
* Breast Cancer Prediction (9 features): ~96%

---

## 🧑‍💻 Contributors
-Mathyam Supriya
---

## 📚 References

1. Shubham Salunke et al., *Smart Health Prediction Using ML*, IJRAR, 2020.
2. Gupta A. et al., *Heart Disease Prediction Using Naive Bayes*, IC4S 2019.
3. D. Dahiwade et al., *Disease Prediction Model Using ML*, ICCMC 2019.
4. A. Rajkomar et al., *Deep Learning with Electronic Health Records*, NPJ Digital Medicine, 2018.
5. Min Chen et al., *Disease Prediction by ML over Big Data from Healthcare Communities*, 2017.

---

## 🏁 Conclusion

The system demonstrates how predictive modeling and AI can empower early diagnosis and healthcare decision-making. By integrating ML/DL algorithms into a web-based platform, users can receive fast, data-driven insights for proactive health management.

---

**License:** MIT License
**Keywords:** Flask, Machine Learning, Healthcare AI, Stroke Prediction, Breast Cancer Detection
