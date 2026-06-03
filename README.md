# Autoimmune Disease Prediction System

A machine learning–based clinical decision support system for predicting autoimmune diseases using laboratory biomarkers, antibody profiles, and patient symptoms.

The project includes:

* Data preprocessing and feature engineering
* Multiple ML models with hyperparameter tuning
* Probability calibration
* Clinical threshold logic for uncertainty handling
* PCA dimensionality reduction
* SHAP explainability
* Interactive Streamlit web application

---

# Features

## Machine Learning Models

The system trains and evaluates multiple classifiers:

* Decision Tree
* K-Nearest Neighbors (KNN)
* Random Forest
* Stacking Ensemble

Each model is additionally calibrated using:

* `CalibratedClassifierCV`

---

## Advanced Feature Engineering

The pipeline automatically creates engineered features such as:

### Symptom Count

Combines multiple symptom indicators into a single score:

* Low-grade fever
* Fatigue or chronic tiredness
* Rashes and skin lesions
* Brittle hair or hair loss
* Dry eyes and/or mouth
* General unwell feeling

---

### Autoantibody Correlation Feature

Combines important antibody markers:

* Anti_dsDNA
* Anti_RNP
* Anti_Ro_SSA
* Anti_La_SSB
* Anti_Sm

---

### PCA Combined Biomarker Feature

A PCA-generated feature based on:

* CRP
* ESR
* ANA
* Sickness Duration

---

## Probability Calibration

Model probabilities are calibrated using sigmoid calibration to improve clinical confidence estimation.

---

## Clinical Decision Thresholds

The project introduces medical-style confidence thresholds:

| Condition               | Interpretation                    |
| ----------------------- | --------------------------------- |
| High Normal Probability | Likely healthy                    |
| Medium Probability      | Uncertain / follow-up recommended |
| Low Normal Probability  | Non-normal likely                 |

This improves safety and interpretability in clinical environments.

---

## SHAP Explainability

The application supports SHAP-based explainability:

* Waterfall plots
* PCA contribution analysis
* Original feature mapping from principal components

---

# Tech Stack

## Backend / ML

* Python
* NumPy
* Pandas
* Scikit-learn
* SHAP
* Joblib

## Frontend

* Streamlit
* Matplotlib

---

# Project Structure

```bash
.
├── Autoimmune_Dataset.csv
├── train.py
├── app.py
│
├── model_rf_calibrated.pkl
├── model_knn_calibrated.pkl
├── model_dt_calibrated.pkl
├── model_stacking_calibrated.pkl
│
├── imputer.pkl
├── scaler.pkl
├── pca.pkl
├── combined_imputer.pkl
├── combined_pca.pkl
│
├── label_encoder.pkl
├── feature_names.pkl
├── raw_input_feature_names.pkl
├── pca_feature_names.pkl
│
├── pca_pc_to_features.json
├── thresholds_by_model.json
├── shap_background.npy
│
└── README.md
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Required Libraries

```txt
numpy
pandas
scikit-learn
matplotlib
streamlit
joblib
shap
```

---

# Training the Models

Run:

```bash
python train.py
```

This will:

* preprocess the dataset
* engineer features
* train all models
* calibrate probabilities
* generate PCA artifacts
* save trained models
* save SHAP resources
* save threshold configurations

Generated artifacts:

```bash
*.pkl
*.json
*.npy
```

---

# Running the Streamlit App

```bash
streamlit run app.py
```

---

# Streamlit Application Features

## Model Selection

Users can switch between:

* Stacking
* RandomForest
* KNN
* DecisionTree

---

## Manual Patient Input

Users can enter:

* biomarker values
* antibody values
* symptoms
* laboratory measurements

Unknown values can be left empty.

---

## Random Dataset Testing

The app can load a random patient sample from the dataset for quick testing.

---

## Probability Visualization

Displays:

* top prediction
* second prediction
* class probabilities
* prediction margins

---

## Clinical Confidence Interpretation

The app automatically classifies predictions into:

### High Confidence Normal

Healthy prediction with strong confidence.

### Uncertain

Possible overlap between healthy and disease classes.

### Non-Normal Likely

Suggests further medical evaluation.

---

## SHAP Explainability Dashboard

Optional SHAP explanations include:

* PCA-level contribution plots
* Waterfall visualizations
* Principal component interpretation
* Original feature contribution mapping

---

# Machine Learning Pipeline

## Data Preprocessing

* Missing value imputation
* MinMax scaling
* PCA dimensionality reduction

---

## Hyperparameter Optimization

### Decision Tree

Uses:

```python
GridSearchCV
```

### KNN

Uses:

```python
GridSearchCV
```

### Random Forest

Uses:

```python
RandomizedSearchCV
```

---

## Ensemble Learning

Final ensemble uses:

```python
StackingClassifier
```

with:

```python
LogisticRegression
```

as the meta-classifier.

---

# Explainability

The system uses SHAP for interpretable AI.

Because PCA is applied before prediction:

1. SHAP explains PCA components
2. PCA components are mapped back to original features
3. The app shows influential biomarkers contributing to predictions

---

# Clinical Safety Logic

The system does not rely only on the highest predicted class.

It additionally evaluates:

* probability of the Normal class
* confidence margins
* ambiguity between classes

This reduces overconfident predictions and improves uncertainty handling.

---

# Example Workflow

1. User enters patient biomarkers
2. Data preprocessing is applied
3. Engineered features are generated
4. PCA transformation occurs
5. Selected model predicts disease probabilities
6. Clinical thresholds interpret prediction confidence
7. SHAP explains the prediction

---

# Author

Kimia Mazloomifar
