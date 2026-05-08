import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import joblib
import json

# LOAD DATA
CSV_PATH = "Autoimmune_Dataset.csv"
df = pd.read_csv(CSV_PATH)

# Encoding Gender
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Encoding Diagnosis
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

# Symptom columns
symptom_columns = [
    "Low-grade fever",
    "Fatigue or chronic tiredness",
    "Rashes and skin lesions",
    "Brittle hair or hair loss",
    "Dry eyes and/or mouth",
    "General 'unwell' feeling"
]
symptom_columns = [c for c in symptom_columns if c in df.columns]

# Corelated columns
corelated_columns = ["Anti_dsDNA", "Anti_RNP", "Anti_Ro_SSA", "Anti_La_SSB", "Anti_Sm"]
corelated_columns = [c for c in corelated_columns if c in df.columns]

# Combined feature columns
features_to_combine = ["CRP", "ESR", "ANA", "Sickness_Duration_Months"]
present_combine = [c for c in features_to_combine if c in df.columns]
combine_ready = (len(present_combine) == len(features_to_combine))

raw_input_feature_names = [
    c for c in df.columns
    if (c != "Diagnosis") and (c not in engineered_cols)
]

# Engineered cols list
engineered_cols = ["Symptom_Count", "corelated_columns", "Combined_Feature_PCA"]
joblib.dump(raw_input_feature_names, "raw_input_feature_names.pkl")
print("Saved raw_input_feature_names.pkl")

# TRAIN/TEST SPLIT
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
X_df = df.drop("Diagnosis", axis=1)
y = df["Diagnosis"].values

X_train_df, X_test_df, y_train, y_test = train_test_split(
    X_df, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

# FEATURE ENGINEERING
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

def add_sum_features(df_part: pd.DataFrame) -> pd.DataFrame:
    df_part = df_part.copy()

    # Symptom_Count
    if len(symptom_columns) > 0:
        df_part["Symptom_Count"] = df_part[symptom_columns].sum(axis=1)
    else:
        df_part["Symptom_Count"] = 0

    # corelated_columns
    if len(corelated_columns) > 0:
        df_part["corelated_columns"] = df_part[corelated_columns].sum(axis=1)
    else:
        df_part["corelated_columns"] = 0

    return df_part

X_train_df = add_sum_features(X_train_df)
X_test_df = add_sum_features(X_test_df)

# Prepare X arrays
X_train_raw = X_train_df.values
X_test_raw = X_test_df.values

# Impute -> Scale -> PCA
imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train_raw)
X_test = imputer.transform(X_test_raw)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Combined_Feature_PCA
if combine_ready:
    subset_imputer = SimpleImputer(strategy="median")

    train_subset = subset_imputer.fit_transform(X_train_df[features_to_combine])
    test_subset = subset_imputer.transform(X_test_df[features_to_combine])

    pca_1 = PCA(n_components=1, random_state=RANDOM_STATE)
    train_combined = pca_1.fit_transform(train_subset)
    test_combined = pca_1.transform(test_subset)

    X_train_df["Combined_Feature_PCA"] = train_combined
    X_test_df["Combined_Feature_PCA"] = test_combined

    # Save artifacts so Streamlit 
    joblib.dump(subset_imputer, "combined_imputer.pkl")
    joblib.dump(pca_1, "combined_pca.pkl")
    print("Saved combined_imputer.pkl and combined_pca.pkl")
else:
    X_train_df["Combined_Feature_PCA"] = 0
    X_test_df["Combined_Feature_PCA"] = 0

# Drop raw columns
def drop_engineering_sources(df_part: pd.DataFrame) -> pd.DataFrame:
    df_part = df_part.copy()

    # Drop raw symptom columns
    if len(symptom_columns) > 0:
        df_part.drop(symptom_columns, axis=1, inplace=True, errors="ignore")

    # Drop raw corelated columns
    if len(corelated_columns) > 0:
        df_part.drop(corelated_columns, axis=1, inplace=True, errors="ignore")

    # Drop combine source columns
    if len(present_combine) > 0:
        df_part.drop(present_combine, axis=1, inplace=True, errors="ignore")

    return df_part

X_train_df = drop_engineering_sources(X_train_df)
X_test_df = drop_engineering_sources(X_test_df)

# Save feature order
for c in X_train_df.columns:
    X_train_df[c] = pd.to_numeric(X_train_df[c], errors="coerce")
for c in X_test_df.columns:
    X_test_df[c] = pd.to_numeric(X_test_df[c], errors="coerce")

X_test_df = X_test_df.reindex(columns=X_train_df.columns)

feature_names = X_train_df.columns.tolist()
joblib.dump(feature_names, "feature_names.pkl")
print("Saved feature_names.pkl")

# # Prepare X arrays
# X_train_raw = X_train_df.values
# X_test_raw = X_test_df.values

# # Impute -> Scale -> PCA
# imputer = SimpleImputer(strategy="median")
# X_train = imputer.fit_transform(X_train_raw)
# X_test = imputer.transform(X_test_raw)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

pca = PCA(PCA_VARIANCE = 88, random_state=RANDOM_STATE)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca, "pca.pkl")
print("Saved imputer.pkl, scaler.pkl, pca.pkl")

# Class weights
from sklearn.utils.class_weight import compute_class_weight
classes = np.unique(y_train)
weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
class_weight_dict = {cls: w for cls, w in zip(classes, weights)}
print("Class weights:", class_weight_dict)

def find_normal_class_index(label_encoder):
    if "Normal" in label_encoder.classes_:
        return list(label_encoder.classes_).index("Normal")
    return None

normal_idx = find_normal_class_index(le)
print("Normal index:", normal_idx)

# Evaluate models
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix
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

    return {"acc": acc, "bacc": bacc, "mf1": mf1, "cm": cm}


# Decision tree (GridSearchCV)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

dt = DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight=class_weight_dict)
dt_grid = {
    "criterion": ["gini", "entropy"],
    "max_depth": [None, 5, 10, 20, 40, 60],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 4, 8],
}
dt_search = GridSearchCV(
    estimator=dt,
    param_grid=dt_grid,
    cv=3,
    n_jobs=-1,
    scoring="balanced_accuracy",
    verbose=1
)
dt_search.fit(X_train, y_train)
best_dt = dt_search.best_estimator_
print("DT Best params:", dt_search.best_params_)
evaluate_model("DecisionTree (RAW)", best_dt, X_test, y_test)


# KNN (GridSearchCV)
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


#RANDOM FOREST (RandomizedSearchCV)
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
evaluate_model("RandomForest (RAW best)", best_rf, X_test, y_test)

# STACKING
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

final_estimator = LogisticRegression(
    max_iter=5000,
    multi_class="auto",
    class_weight="balanced"
)

stacking = StackingClassifier(
    estimators=[("rf", best_rf), ("knn", best_knn), ("dt", best_dt)],
    final_estimator=final_estimator,
    stack_method="predict_proba",
    n_jobs=-1,
    passthrough=False,
    cv=5
)
stacking.fit(X_train, y_train)
evaluate_model("Stacking (RAW)", stacking, X_test, y_test)

# CALIBRATION
from sklearn.calibration import CalibratedClassifierCV

CAL_METHOD = "sigmoid"
CAL_CV = 5

rf_cal = CalibratedClassifierCV(estimator=best_rf, method=CAL_METHOD, cv=CAL_CV)
rf_cal.fit(X_train, y_train)
evaluate_model(f"RandomForest (CALIBRATED {CAL_METHOD})", rf_cal, X_test, y_test)

knn_cal = CalibratedClassifierCV(estimator=best_knn, method=CAL_METHOD, cv=CAL_CV)
knn_cal.fit(X_train, y_train)
evaluate_model(f"KNN (CALIBRATED {CAL_METHOD})", knn_cal, X_test, y_test)

dt_cal = CalibratedClassifierCV(estimator=best_dt, method=CAL_METHOD, cv=CAL_CV)
dt_cal.fit(X_train, y_train)
evaluate_model(f"DecisionTree (CALIBRATED {CAL_METHOD})", dt_cal, X_test, y_test)

stack_cal = CalibratedClassifierCV(estimator=stacking, method=CAL_METHOD, cv=CAL_CV)
stack_cal.fit(X_train, y_train)
evaluate_model(f"Stacking (CALIBRATED {CAL_METHOD})", stack_cal, X_test, y_test)

# Save calibrated models
joblib.dump(rf_cal, "model_rf_calibrated.pkl")
joblib.dump(knn_cal, "model_knn_calibrated.pkl")
joblib.dump(dt_cal, "model_dt_calibrated.pkl")
joblib.dump(stack_cal, "model_stacking_calibrated.pkl")


# Threshold per model
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

# Threshold tuning targets
TARGET_NORMAL_PRECISION_FOR_HIGH = 0.95
TARGET_NORMAL_RECALL_FOR_HIGH = 0.90

# floor for HIGH_T
HIGH_T_FLOOR = 0.70

# margin rule threshold
DEFAULT_MARGIN_WEAK = 0.10

thresholds_by_model = {}

thresholds_by_model["stacking"] = tune_thresholds_for_model(
    model_name="stacking_calibrated",
    calibrated_model=stack_cal,
    X_test=X_test,
    y_test=y_test,
    normal_idx=normal_idx,
    high_floor=HIGH_T_FLOOR,
    target_precision=TARGET_NORMAL_PRECISION_FOR_HIGH,
    target_recall=TARGET_NORMAL_RECALL_FOR_HIGH,
    margin_weak=DEFAULT_MARGIN_WEAK
)

thresholds_by_model["rf"] = tune_thresholds_for_model(
    model_name="rf_calibrated",
    calibrated_model=rf_cal,
    X_test=X_test,
    y_test=y_test,
    normal_idx=normal_idx,
    high_floor=HIGH_T_FLOOR,
    target_precision=TARGET_NORMAL_PRECISION_FOR_HIGH,
    target_recall=TARGET_NORMAL_RECALL_FOR_HIGH,
    margin_weak=DEFAULT_MARGIN_WEAK
)

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

thresholds_by_model["dt"] = tune_thresholds_for_model(
    model_name="dt_calibrated",
    calibrated_model=dt_cal,
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

print("Thresholds summary:")
for k, v in thresholds_by_model.items():
    print(f"  - {k}: HIGH_T={v['HIGH_T']:.3f}, MID_T={v['MID_T']:.3f}, MARGIN_WEAK={v['MARGIN_WEAK']:.3f}")


# SHAP ARTIFACTS
SHAP_BACKGROUND_SIZE = 120
PC_FEATURES_TOPK = 8

# 1) Save PCA feature names
n_pcs = int(getattr(pca, "n_components_", X_train.shape[1]))
pca_feature_names = [f"PC{i+1}" for i in range(n_pcs)]
joblib.dump(pca_feature_names, "pca_feature_names.pkl")

# 2) Save SHAP background in PCA space
rng = np.random.RandomState(RANDOM_STATE)
bg_size = min(SHAP_BACKGROUND_SIZE, X_train.shape[0])
bg_idx = rng.choice(np.arange(X_train.shape[0]), size=bg_size, replace=False)
shap_background = X_train[bg_idx].astype(np.float32)

np.save("shap_background.npy", shap_background)

# 3) Save mapping from each PC to the most influential original features
pc_to_features = {}
try:
    comps = pca.components_
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


