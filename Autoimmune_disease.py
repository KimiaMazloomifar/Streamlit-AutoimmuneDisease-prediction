# Import necessary libraries:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

import joblib
import json

# Read the csv file into a data frame
CSV_PATH = "Autoimmune_Dataset.csv"
df = pd.read_csv(CSV_PATH)

#  Encoding Gender column
df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Diagnosis"] = le.fit_transform(df["Diagnosis"].astype(str))
joblib.dump(le, "label_encoder.pkl")
print("Saved label_encoder.pkl")

# Drop cols
drop_cols = [
    "Patient_ID", "Age", "Gender", "RBC_Count", "Hemoglobin", "Hematocrit", "MCV", "MCH", "MCHC",
    "Lymphocytes", "PLT_Count", "MBL_Level", "C3", "C4", "Rheumatoid factor", "Anti-TPO",
    "Anti-Tg", "Anti-SMA", "Dizziness", "Weight loss", "Anti_tTG", "Stiffness in the joints",
    "Joint pain", "IgG_IgE_receptor", "Progesterone_antibodies"
]
df.drop([c for c in drop_cols if c in df.columns], axis=1, inplace=True)

# Symptom_Count
symptom_columns = [
    "Low-grade fever",
    "Fatigue or chronic tiredness",
    "Rashes and skin lesions",
    "Brittle hair or hair loss",
    "Dry eyes and/or mouth",
    "General 'unwell' feeling"
]
symptom_columns = [c for c in symptom_columns if c in df.columns]
df["Symptom_Count"] = df[symptom_columns].sum(axis=1) if len(symptom_columns) else 0
if len(symptom_columns):
    df.drop(symptom_columns, axis=1, inplace=True)

# corelated_columns
corelated_columns = ["Anti_dsDNA", "Anti_RNP", "Anti_Ro_SSA", "Anti_La_SSB", "Anti_Sm"]
corelated_columns = [c for c in corelated_columns if c in df.columns]
df["corelated_columns"] = df[corelated_columns].sum(axis=1) if len(corelated_columns) else 0
if len(corelated_columns):
    df.drop(corelated_columns, axis=1, inplace=True)

# Combined feature columns
features_to_combine = ["CRP", "ESR", "ANA", "Sickness_Duration_Months"]
present_combine = [c for c in features_to_combine if c in df.columns]
combine_ready = (len(present_combine) == len(features_to_combine))


# Engineered cols list
engineered_cols = ["Symptom_Count", "corelated_columns", "Combined_Feature_PCA"]


raw_input_feature_names = [
    c for c in df.columns
    if (c != "Diagnosis") and (c not in engineered_cols)
]
joblib.dump(raw_input_feature_names, "raw_input_feature_names.pkl")
print("Saved raw_input_feature_names.pkl")

RANDOM_STATE = 42

PCA_VARIANCE = 0.88

CAL_METHOD = "sigmoid"
CAL_CV = 5

# Threshold tuning targets
TARGET_NORMAL_PRECISION_FOR_HIGH = 0.95
TARGET_NORMAL_RECALL_FOR_HIGH = 0.90

# floor for HIGH_T (requested)
HIGH_T_FLOOR = 0.70

# margin rule threshold (kept fixed, but saved per model)
DEFAULT_MARGIN_WEAK = 0.10

# SHAP config
SHAP_BACKGROUND_SIZE = 120   # keep small (speed + app responsiveness)
PC_FEATURES_TOPK = 8         # how many original features to show per PC in mapping

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

features_to_combine = ["CRP", "ESR", "ANA", "Sickness_Duration_Months"]
present = [c for c in features_to_combine if c in df.columns]

if len(present) == len(features_to_combine):

    X_subset = df[features_to_combine]

    combined_imputer = SimpleImputer(strategy="median")
    X_subset_imputed = combined_imputer.fit_transform(X_subset)

    combined_pca = PCA(n_components=1, random_state=RANDOM_STATE)
    combined_feature = combined_pca.fit_transform(X_subset_imputed)

    df["Combined_Feature_PCA"] = combined_feature.ravel()

    # ذخیره آرتیفکت‌ها برای Streamlit
    joblib.dump(combined_imputer, "combined_imputer.pkl")
    joblib.dump(combined_pca, "combined_pca.pkl")

    print("Saved combined_imputer.pkl")
    print("Saved combined_pca.pkl")

    df.drop(features_to_combine, axis=1, inplace=True)

else:

    df["Combined_Feature_PCA"] = 0

    df.drop(present, axis=1, inplace=True)

# Ensure numeric
for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")


# Save feature order
feature_names = df.drop("Diagnosis", axis=1).columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")
print("Saved feature_names.pkl")


# Prepare X,y
X = df.drop("Diagnosis", axis=1).values
y = df["Diagnosis"].values

# Split
from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

# Impute
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train_raw)
X_test = imputer.transform(X_test_raw)
# Normalisation
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Principal Component Analysis
pca = PCA(PCA_VARIANCE, random_state=RANDOM_STATE)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
print("Saved imputer.pkl, scaler.pkl, pca.pkl")

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    mf1 = f1_score(y_test, y_pred, average="macro")
    cm = confusion_matrix(y_test, y_pred)

    print("\n" + "=" * 90)
    print(f"MODEL: {name}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print(f"Macro F1:          {mf1:.4f}")
    print("-" * 90)
    print("Classification report:\n", classification_report(y_test, y_pred))
    print("Confusion matrix:\n", cm)
    print("=" * 90)
    disp=ConfusionMatrixDisplay(cm,display_labels=['0','1','4','3','2','5'])
    disp.plot()

    return {"acc": acc, "bacc": bacc, "mf1": mf1, "cm": cm}

def find_normal_class_index(label_encoder):
    if "Normal" in label_encoder.classes_:
        return list(label_encoder.classes_).index("Normal")
    return None


def tune_thresholds_for_model(
    model_name: str,
    calibrated_model,
    X_test,
    y_test,
    normal_idx: int | None,
    high_floor: float = 0.70,
    target_precision: float = 0.95,
    target_recall: float = 0.90,
    margin_weak: float = 0.10
):
    """
    Returns dict with HIGH_T, MID_T, MARGIN_WEAK and info.
    Uses calibrated P(Normal) on test set.

    MID_T: threshold maximizing Youden's J (TPR - FPR) for Normal vs Non-normal
    HIGH_T: smallest threshold meeting precision>=target_precision and recall>=target_recall,
            but also forced to be >= high_floor
    """
    if normal_idx is None or not hasattr(calibrated_model, "predict_proba"):
        return {
            "HIGH_T": float(high_floor),
            "MID_T": float(min(high_floor - 0.05, 0.40)),
            "MARGIN_WEAK": float(margin_weak),
            "info": {
                "tuned_on": model_name,
                "note": "Normal class not found or model lacks predict_proba; using defaults."
            }
        }

    proba = calibrated_model.predict_proba(X_test)
    if normal_idx >= proba.shape[1]:
        return {
            "HIGH_T": float(high_floor),
            "MID_T": float(min(high_floor - 0.05, 0.40)),
            "MARGIN_WEAK": float(margin_weak),
            "info": {
                "tuned_on": model_name,
                "note": "Normal index out of bounds; using defaults."
            }
        }

    p_normal = proba[:, normal_idx]
    y_is_normal = (y_test == normal_idx).astype(int)

    # MID_T via ROC Youden J
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_is_normal, p_normal)
    j = tpr - fpr
    mid_t = float(thr[int(np.argmax(j))])

    def prec_rec_at(t):
        pred_normal = (p_normal >= t).astype(int)
        tp = np.sum((pred_normal == 1) & (y_is_normal == 1))
        fp = np.sum((pred_normal == 1) & (y_is_normal == 0))
        fn = np.sum((pred_normal == 0) & (y_is_normal == 1))
        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        return float(precision), float(recall)

    candidates = np.linspace(high_floor, 0.99, 200)
    feasible = []
    for t in candidates:
        prec, rec = prec_rec_at(t)
        if prec >= target_precision and rec >= target_recall:
            feasible.append((float(t), float(prec), float(rec)))

    if feasible:
        high_t, high_prec, high_rec = feasible[0]
    else:
        p_norm_true = p_normal[y_is_normal == 1]
        if len(p_norm_true) > 0:
            high_t = float(max(high_floor, np.quantile(p_norm_true, 0.10)))
        else:
            high_t = float(high_floor)
        high_prec, high_rec = prec_rec_at(high_t)

    if mid_t >= high_t:
        mid_t = float(max(0.05, min(high_t - 0.05, 0.80)))

    return {
        "HIGH_T": float(high_t),
        "MID_T": float(mid_t),
        "MARGIN_WEAK": float(margin_weak),
        "info": {
            "tuned_on": model_name,
            "normal_index": int(normal_idx),
            "high_precision_target": float(target_precision),
            "high_recall_target": float(target_recall),
            "achieved_precision_at_HIGH_T": float(high_prec),
            "achieved_recall_at_HIGH_T": float(high_rec),
            "HIGH_T_floor": float(high_floor),
        },
    }


# Class weights
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {cls: w for cls, w in zip(classes, weights)}
print("Class weights:", class_weight_dict)
# Normal index
normal_idx = find_normal_class_index(le)
print("Normal index:", normal_idx)


# 1) RandomForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
max_features = ["sqrt", "log2", None]
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

rf_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
    "class_weight": [class_weight_dict],
}

rf = RandomForestClassifier(random_state=RANDOM_STATE)
rf_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=rf_grid,
    n_iter=50,
    cv=3,
    verbose=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    scoring="balanced_accuracy"
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
print("RF Best params:", rf_search.best_params_)
evaluate_model("RandomForest (RAW)", best_rf, X_test, y_test)

# 2) KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

knn = KNeighborsClassifier()
knn_grid = {
    "n_neighbors": [3, 5, 7, 9, 11, 15, 21],
    "weights": ["uniform", "distance"],
    "p": [1, 2],
}
knn_search = GridSearchCV(
    estimator=knn,
    param_grid=knn_grid,
    cv=3,
    n_jobs=-1,
    scoring="balanced_accuracy",
    verbose=1
)
knn_search.fit(X_train, y_train)
best_knn = knn_search.best_estimator_
print("KNN Best params:", knn_search.best_params_)
evaluate_model("KNN (RAW)", best_knn, X_test, y_test)

# Calibration
from sklearn.calibration import CalibratedClassifierCV


knn_cal = CalibratedClassifierCV(estimator=best_knn, method=CAL_METHOD, cv=CAL_CV)
knn_cal.fit(X_train, y_train)
evaluate_model(f"KNN (CALIBRATED {CAL_METHOD})", knn_cal, X_test, y_test)

joblib.dump(knn_cal, "model_knn_calibrated.pkl")

thresholds_by_model = {}

thresholds_by_model["knn"] = tune_thresholds_for_model(
    model_name="knn_calibrated",
    calibrated_model=knn_cal,
    X_test=X_test,
    y_test=y_test,
    normal_idx=normal_idx,
    high_floor=HIGH_T_FLOOR,
    target_precision=TARGET_NORMAL_PRECISION_FOR_HIGH,
    target_recall=TARGET_NORMAL_RECALL_FOR_HIGH,
    margin_weak=DEFAULT_MARGIN_WEAK
)



with open("thresholds_by_model.json", "w", encoding="utf-8") as f:
    json.dump(thresholds_by_model, f, indent=2)

print("\n Saved thresholds_by_model.json")
print("Thresholds summary:")
for k, v in thresholds_by_model.items():
    print(f"  - {k}: HIGH_T={v['HIGH_T']:.3f}, MID_T={v['MID_T']:.3f}, MARGIN_WEAK={v['MARGIN_WEAK']:.3f}")

# 1) Save PCA feature names (PC1..PCk)
n_pcs = int(getattr(pca, "n_components_", X_train.shape[1]))
pca_feature_names = [f"PC{i+1}" for i in range(n_pcs)]
joblib.dump(pca_feature_names, "pca_feature_names.pkl")
print("Saved pca_feature_names.pkl")

# 2) Save SHAP background in PCA space
rng = np.random.RandomState(RANDOM_STATE)
bg_size = min(SHAP_BACKGROUND_SIZE, X_train.shape[0])
bg_idx = rng.choice(np.arange(X_train.shape[0]), size=bg_size, replace=False)
shap_background = X_train[bg_idx].astype(np.float32)

np.save("shap_background.npy", shap_background)
print("Saved shap_background.npy  shape:", shap_background.shape)

# 3) Save mapping from each PC to the most influential original features
pc_to_features = {}
try:
    comps = pca.components_  # shape (n_components, n_original_features)
    for i in range(comps.shape[0]):
        weights = comps[i]
        top_idx = np.argsort(np.abs(weights))[::-1][:PC_FEATURES_TOPK]
        pc_to_features[f"PC{i+1}"] = [
            {"feature": feature_names[j], "loading": float(weights[j])}
            for j in top_idx
        ]
except Exception as e:
    pc_to_features = {"note": f"Could not extract PCA components for mapping: {str(e)}"}

with open("pca_pc_to_features.json", "w", encoding="utf-8") as f:
    json.dump(pc_to_features, f, indent=2, ensure_ascii=False)
print("Saved pca_pc_to_features.json")