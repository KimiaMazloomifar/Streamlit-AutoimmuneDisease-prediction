# Autoimmune Disease Prediction System

A machine learning-based clinical decision support system for predicting autoimmune diseases using laboratory biomarkers, antibody profiles, and patient symptoms.

The project includes:

* Data preprocessing and feature engineering
* Multiple machine learning models with hyperparameter optimization
* Probability calibration
* Clinical threshold logic for uncertainty handling
* PCA dimensionality reduction
* SHAP explainability
* Interactive Streamlit web application
* Dockerized deployment
* Docker Hub image distribution

---

## Live Technologies

### Machine Learning

* Scikit-Learn
* PCA
* SHAP
* Probability Calibration
* Ensemble Learning

---

## Docker Image

Docker Hub Repository:

https://hub.docker.com/r/kimiamzf/autoimmune-disease-prediction

Pull the latest image:

```bash
docker pull kimiamzf/autoimmune-disease-prediction:latest
```

Run the application:

```bash
docker run -p 8501:8501 kimiamzf/autoimmune-disease-prediction:latest
```

Open:

```text
http://localhost:8501
```

---

## Docker Compose

Run with Docker Compose:

```bash
docker compose up -d
```

Stop containers:

```bash
docker compose down
```

---

## Project Architecture

```text
User Input
     │
     ▼
Feature Engineering
     │
     ▼
Imputation
     │
     ▼
Scaling
     │
     ▼
PCA Transformation
     │
     ▼
Machine Learning Model
     │
     ▼
Probability Calibration
     │
     ▼
Clinical Threshold Logic
     │
     ▼
Prediction & Explainability
     │
     ▼
Streamlit Dashboard
```

---

## Key Features

### Machine Learning Models

* Decision Tree
* K-Nearest Neighbors (KNN)
* Random Forest
* Stacking Ensemble

Each model is calibrated using:

```python
CalibratedClassifierCV
```

---

### Advanced Feature Engineering

#### Symptom Count

Generated from:

* Low-grade fever
* Fatigue or chronic tiredness
* Rashes and skin lesions
* Brittle hair or hair loss
* Dry eyes and/or mouth
* General unwell feeling

#### Autoantibody Correlation Feature

Generated from:

* Anti_dsDNA
* Anti_RNP
* Anti_Ro_SSA
* Anti_La_SSB
* Anti_Sm

#### PCA Biomarker Feature

Generated using:

* CRP
* ESR
* ANA
* Sickness Duration Months

---

### Explainable AI

SHAP-based explanations provide:

* Waterfall plots
* PCA contribution analysis
* Principal component interpretation
* Original feature mapping

---

## Project Structure

```bash
.
├── app.py
├── train.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── Autoimmune_Dataset.csv
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

## Installation

### Clone Repository

```bash
git clone https://github.com/KimiaMazloomifar/Streamlit-AutoimmuneDisease-prediction.git

cd Streamlit-AutoimmuneDisease-prediction
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Streamlit

```bash
streamlit run app.py
```

---

## Training

Train and generate all artifacts:

```bash
python Autoimmune_Disease.py
```

Generated artifacts:

```text
*.pkl
*.json
*.npy
```

---

## Clinical Safety Logic

The system evaluates:

* Predicted disease probabilities
* Normal-class confidence
* Confidence margins
* Clinical uncertainty thresholds

This reduces overconfident predictions and improves interpretability.

---

## Author

Kimia Mazloomifar
