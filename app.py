import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Autoimmune Disease Prediction", layout="wide")


# Load artifacts
@st.cache_resource
def load_artifacts():
    base = Path(".")
    required = [
        "imputer.pkl",
        "scaler.pkl",
        "pca.pkl",
        "feature_names.pkl",
        "label_encoder.pkl",
        "pca_feature_names.pkl",
        "pca_pc_to_features.json",
        "shap_background.npy",
        "model_knn_calibrated.pkl",
        "thresholds_by_model.json",
        "raw_input_feature_names.pkl",
        "combined_imputer.pkl",
        "combined_pca.pkl",
    ]
    missing = [f for f in required if not (base / f).exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    thresholds_by_model = json.loads((base / "thresholds_by_model.json").read_text(encoding="utf-8"))
    pc_to_features = json.loads((base / "pca_pc_to_features.json").read_text(encoding="utf-8"))
    shap_background = np.load(base / "shap_background.npy")

    return {
        "imputer": joblib.load(base / "imputer.pkl"),
        "scaler": joblib.load(base / "scaler.pkl"),
        "pca": joblib.load(base / "pca.pkl"),
        "feature_names": joblib.load(base / "feature_names.pkl"),
        "pca_feature_names": joblib.load(base / "pca_feature_names.pkl"),
        "label_encoder": joblib.load(base / "label_encoder.pkl"),

        # NEW
        "raw_input_feature_names": joblib.load(base / "raw_input_feature_names.pkl"),
        "combined_imputer": joblib.load(base / "combined_imputer.pkl"),
        "combined_pca": joblib.load(base / "combined_pca.pkl"),

        "models": {
     "knn": joblib.load(base / "model_knn_calibrated.pkl"),
},
        "thresholds_by_model": thresholds_by_model,
        "pc_to_features": pc_to_features,
        "shap_background": shap_background,
    }


@st.cache_data
def load_dataset(csv_path: str):
    return pd.read_csv(csv_path)



# Helpers
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def find_normal_class_index(label_encoder):
    try:
        classes = list(label_encoder.classes_)
    except Exception:
        return None

    if "Normal" in classes:
        return classes.index("Normal")

    return None


def is_normal_label(label: str) -> bool:
    return str(label).strip() == "Normal"


def preprocess_user_input(user_dict, feature_names):
    row = {f: np.nan for f in feature_names}
    for k, v in user_dict.items():
        if k in row:
            row[k] = v
    df_row = pd.DataFrame([row], columns=feature_names)
    for c in df_row.columns:
        df_row[c] = pd.to_numeric(df_row[c], errors="coerce")
    return df_row


def transform_to_pca(df_row, imputer, scaler, pca):
    X = df_row.values
    X = imputer.transform(X)
    X = scaler.transform(X)
    X = pca.transform(X)
    return X


def predict_pipeline(df_row, imputer, scaler, pca, model):
    X = transform_to_pca(df_row, imputer, scaler, pca)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    return pred, proba, X


def to_feature_dict_from_dataset_row(raw_row: pd.Series, feature_names: list):
    """
    (kept as-is) - old behavior: fills model feature_names and attempts to compute Symptom_Count/corelated_columns
    This may not be used anymore for the form, but we keep it to not remove anything.
    """
    user = {f: np.nan for f in feature_names}

    for f in feature_names:
        if f in raw_row.index:
            user[f] = raw_row[f]

    # Symptom_Count
    if "Symptom_Count" in feature_names and pd.isna(user.get("Symptom_Count")):
        symptom_cols = [
            "Low-grade fever",
            "Fatigue or chronic tiredness",
            "Rashes and skin lesions",
            "Brittle hair or hair loss",
            "Dry eyes and/or mouth",
            "General 'unwell' feeling",
        ]
        present = [c for c in symptom_cols if c in raw_row.index]
        if present:
            vals = pd.to_numeric(pd.Series({c: raw_row[c] for c in present}), errors="coerce").fillna(0.0)
            user["Symptom_Count"] = float(vals.sum())

    # corelated_columns
    if "corelated_columns" in feature_names and pd.isna(user.get("corelated_columns")):
        corr_cols = ["Anti_dsDNA", "Anti_RNP", "Anti_Ro_SSA", "Anti_La_SSB", "Anti_Sm"]
        present = [c for c in corr_cols if c in raw_row.index]
        if present:
            vals = pd.to_numeric(pd.Series({c: raw_row[c] for c in present}), errors="coerce").fillna(0.0)
            user["corelated_columns"] = float(vals.sum())

    return user


def to_raw_feature_dict_from_dataset_row(raw_row: pd.Series, raw_input_feature_names: list):
    """
    NEW: Fills ONLY raw inputs (what user should type in the form).
    """
    user = {f: np.nan for f in raw_input_feature_names}
    for f in raw_input_feature_names:
        if f in raw_row.index:
            user[f] = raw_row[f]
    return user


# =========================
# NEW: Feature engineering at inference (Streamlit)
# =========================
def apply_feature_engineering(df_raw: pd.DataFrame, combined_imputer, combined_pca) -> pd.DataFrame:
    df = df_raw.copy()

    # ensure numeric for everything present
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Symptom_Count
    symptom_cols = [
        "Low-grade fever",
        "Fatigue or chronic tiredness",
        "Rashes and skin lesions",
        "Brittle hair or hair loss",
        "Dry eyes and/or mouth",
        "General 'unwell' feeling",
    ]
    present_sym = [c for c in symptom_cols if c in df.columns]
    if present_sym:
        df["Symptom_Count"] = df[present_sym].fillna(0.0).sum(axis=1)
    else:
        df["Symptom_Count"] = 0.0

    # corelated_columns
    corr_cols = ["Anti_dsDNA", "Anti_RNP", "Anti_Ro_SSA", "Anti_La_SSB", "Anti_Sm"]
    present_corr = [c for c in corr_cols if c in df.columns]
    if present_corr:
        df["corelated_columns"] = df[present_corr].fillna(0.0).sum(axis=1)
    else:
        df["corelated_columns"] = 0.0

    # Combined_Feature_PCA (use saved artifacts, NOT fit again)
    combine_cols = ["CRP", "ESR", "ANA", "Sickness_Duration_Months"]
    if all(c in df.columns for c in combine_cols):
        subset = df[combine_cols]
        subset_imputed = combined_imputer.transform(subset)
        combined = combined_pca.transform(subset_imputed)

        df["Combined_Feature_PCA"] = combined.ravel()
    else:
        df["Combined_Feature_PCA"] = 0.0

    return df


def drop_raw_engineering_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop raw columns that were used to create engineered features in training.
    """
    drop_cols = [
        # symptoms
        "Low-grade fever",
        "Fatigue or chronic tiredness",
        "Rashes and skin lesions",
        "Brittle hair or hair loss",
        "Dry eyes and/or mouth",
        "General 'unwell' feeling",

        # correlated antibodies
        "Anti_dsDNA", "Anti_RNP", "Anti_Ro_SSA", "Anti_La_SSB", "Anti_Sm",

        # combined PCA inputs
        "CRP", "ESR", "ANA", "Sickness_Duration_Months"
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


# SHAP utils 
@st.cache_resource
def get_shap_objects():
    try:
        import shap
        return shap
    except Exception:
        return None


@st.cache_resource
def build_shap_explainer(model_key: str):
    shap = get_shap_objects()
    if shap is None:
        return None

    models = st.session_state.get("_models_for_shap", None)
    bg_all = st.session_state.get("_shap_background", None)

    if models is None or bg_all is None:
        return None
    if model_key not in models:
        return None

    model = models[model_key]

    bg = bg_all
    if isinstance(bg, np.ndarray) and bg.shape[0] > 60:
        bg = bg[:60]

    def f(X):
        return model.predict_proba(X)

    masker = shap.maskers.Independent(bg)
    explainer = shap.Explainer(f, masker)
    return explainer


def render_shap_for_instance(
    explainer,
    x_instance_pca: np.ndarray,
    class_index: int,
    pca_feature_names: list,
    max_display: int = 14
):
    shap = get_shap_objects()
    if shap is None or explainer is None:
        st.warning("SHAP is not available.")
        return None

    x1 = np.asarray(x_instance_pca, dtype=np.float32)
    if x1.ndim == 2:
        x1 = x1[0]

    with st.spinner("Computing SHAP explanation ..."):
        sv = explainer(x1.reshape(1, -1))

    values = sv.values
    base = sv.base_values

    if values.ndim == 3:
        v = values[0, :, class_index]
        b = base[0][class_index] if np.ndim(base[0]) > 0 else float(base[0])
    else:
        v = values[0, :]
        b = float(base[0]) if np.ndim(base) > 0 else float(base)

    exp = shap.Explanation(
        values=v,
        base_values=b,
        data=x1,
        feature_names=pca_feature_names
    )

    fig = plt.figure()
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    st.pyplot(fig, clear_figure=True)

    # st.markdown("**Top contributors for this prediction**")
    # abs_df = pd.DataFrame({
    #     "PC": pca_feature_names,
    #     "abs_shap": np.abs(v),
    #     "shap": v
    # }).sort_values("abs_shap", ascending=False).head(max_display)
    # st.dataframe(abs_df, use_container_width=True)

    # return v


def pc_to_original_feature_hint(pc_to_features: dict, pc_name: str):
    items = pc_to_features.get(pc_name, None)
    if not items or not isinstance(items, list):
        return None
    return items


def pc_hint_to_df(hint_list, max_items: int = 10) -> pd.DataFrame:
    if not hint_list or not isinstance(hint_list, list):
        return pd.DataFrame(columns=["Feature", "Loading", "abs(Loading)"])

    rows = []
    for h in hint_list[:max_items]:
        try:
            feat = str(h.get("feature", "")).strip()
            loading = float(h.get("loading", np.nan))
            if feat == "" or np.isnan(loading):
                continue
            rows.append({"Feature": feat, "Loading": loading, "abs(Loading)": abs(loading)})
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=["Feature", "Loading", "abs(Loading)"])

    df = df.sort_values("abs(Loading)", ascending=False).reset_index(drop=True)
    return df


def pc_hint_to_text(hint_list, max_items: int = 6) -> str:
    df = pc_hint_to_df(hint_list, max_items=max_items)
    if df.empty:
        return ""
    return ", ".join([f"{r.Feature} ({r.Loading:+.2f})" for r in df.itertuples(index=False)])


# App start
st.title("Autoimmune Disease Prediction")

try:
    A = load_artifacts()
except Exception as e:
    st.error("Failed to load model artifacts.")
    st.code(str(e))
    st.stop()

models = A["models"]
thresholds_by_model = A["thresholds_by_model"]

imputer = A["imputer"]
scaler = A["scaler"]
pca = A["pca"]
feature_names = A["feature_names"]
pca_feature_names = A["pca_feature_names"]
pc_to_features = A["pc_to_features"]
shap_background = A["shap_background"]

# NEW
raw_input_feature_names = A["raw_input_feature_names"]
combined_imputer = A["combined_imputer"]
combined_pca = A["combined_pca"]

label_encoder = A["label_encoder"]
normal_idx = find_normal_class_index(label_encoder)

# Make SHAP resources accessible without hashing them
st.session_state["_models_for_shap"] = models
st.session_state["_shap_background"] = shap_background



# Sidebar
st.sidebar.header("Model selection")

MODEL_OPTIONS = {
    "Use KNN": "knn",
}


def apply_thresholds_for_model_key(model_key: str):
    t = thresholds_by_model.get(model_key, None)
    if not isinstance(t, dict):
        st.session_state["HIGH_T"] = 0.70
        st.session_state["MID_T"] = 0.40
        st.session_state["MARGIN_WEAK"] = 0.10
        return

    st.session_state["HIGH_T"] = float(t.get("HIGH_T", 0.70))
    st.session_state["MID_T"] = float(t.get("MID_T", 0.40))
    st.session_state["MARGIN_WEAK"] = float(t.get("MARGIN_WEAK", 0.10))


if "model_choice_label" not in st.session_state:
    st.session_state["model_choice_label"] = "Use KNN"

if "HIGH_T" not in st.session_state or "MID_T" not in st.session_state or "MARGIN_WEAK" not in st.session_state:
    apply_thresholds_for_model_key(MODEL_OPTIONS[st.session_state["model_choice_label"]])


def on_model_change():
    label = st.session_state["model_choice_label"]
    key = MODEL_OPTIONS[label]
    apply_thresholds_for_model_key(key)


model_choice_label = st.sidebar.selectbox(
    "Choose model",
    list(MODEL_OPTIONS.keys()),
    key="model_choice_label",
    on_change=on_model_change
)

model_key = MODEL_OPTIONS[model_choice_label]
model = models[model_key]

st.sidebar.markdown("---")
show_all_probs = st.sidebar.checkbox("Show all class probabilities", value=True)
show_debug = st.sidebar.checkbox("Show input row", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset quick test")
default_dataset_name = "Autoimmune_Dataset.csv"
dataset_path = st.sidebar.text_input("CSV filename", value=default_dataset_name)

st.sidebar.markdown("---")
st.sidebar.subheader("Clinical thresholds")

HIGH_T = st.sidebar.slider("High confidence Normal (>=)", 0.50, 0.99, float(st.session_state["HIGH_T"]), 0.01, key="HIGH_T_slider")
MID_T  = st.sidebar.slider("Uncertain lower bound (>=)", 0.05, 0.95, float(st.session_state["MID_T"]), 0.01, key="MID_T_slider")
MARGIN_WEAK = st.sidebar.slider("Weak margin (<) => uncertainty", 0.01, 0.30, float(st.session_state["MARGIN_WEAK"]), 0.01, key="MARGIN_slider")

st.session_state["HIGH_T"] = float(HIGH_T)
st.session_state["MID_T"] = float(MID_T)
st.session_state["MARGIN_WEAK"] = float(MARGIN_WEAK)

if MID_T >= HIGH_T:
    st.sidebar.error("MID_T must be smaller than HIGH_T.")

with st.sidebar.expander("Threshold info for selected model", expanded=False):
    st.json(thresholds_by_model.get(model_key, {}))

st.sidebar.markdown("---")
st.sidebar.subheader("Explainability")
enable_shap = st.sidebar.checkbox("Enable SHAP explanation", value=False)
shap_max_display = st.sidebar.slider("SHAP max PCs to display", 5, 25, 14, 1)
show_pc_mapping = st.sidebar.checkbox("Original Feature mapping", value=True)
pc_hint_max_items = st.sidebar.slider("PC mapping: max original features per PC", 3, 20, 12, 1)


# Session state for form values (UPDATED: raw_input_feature_names)
if "form_values" not in st.session_state:
    st.session_state.form_values = {f: "" for f in raw_input_feature_names}
if "loaded_row_info" not in st.session_state:
    st.session_state.loaded_row_info = None

# Buttons
colA, colB = st.columns([1, 1])

with colA:
    if st.button("Load random sample from dataset", use_container_width=True):
        try:
            df_data = load_dataset(dataset_path)
            if len(df_data) == 0:
                st.warning("The dataset is empty.")
            else:
                idx = int(np.random.randint(0, len(df_data)))
                raw_row = df_data.iloc[idx]

                # UPDATED: load raw inputs only
                user_dict = to_raw_feature_dict_from_dataset_row(raw_row, raw_input_feature_names)

                for f in raw_input_feature_names:
                    v = user_dict.get(f, np.nan)
                    st.session_state.form_values[f] = "" if pd.isna(v) else str(v)

                true_diag = raw_row.get("Diagnosis", None) if "Diagnosis" in df_data.columns else None
                st.session_state.loaded_row_info = {"row_index": idx, "true_diagnosis_raw": true_diag}
                st.success(f"Loaded row #{idx} into the form.")
        except Exception as e:
            st.error("Failed to load dataset row.")
            st.code(str(e))

with colB:
    if st.button("Clear inputs", use_container_width=True):
        st.session_state.form_values = {f: "" for f in raw_input_feature_names}
        st.session_state.loaded_row_info = None
        st.success("All inputs cleared.")


# Inputs UI (UPDATED: raw only)
st.markdown("### Enter patient features ")

cols = st.columns(3)
user_values = {}

for i, feat in enumerate(raw_input_feature_names):
    col = cols[i % 3]
    with col:
        raw = st.text_input(feat, value=st.session_state.form_values.get(feat, ""), help="Leave empty if unknown.")
        user_values[feat] = safe_float(raw, default=np.nan)

if st.session_state.loaded_row_info is not None:
    info = st.session_state.loaded_row_info
    msg = f"Loaded dataset row index: **{info['row_index']}**"
    if info.get("true_diagnosis_raw") is not None:
        msg += f" | dataset Diagnosis: **{info['true_diagnosis_raw']}**"
    st.info(msg)

st.markdown("---")


# Predict (UPDATED PIPELINE)
if st.button("Predict", use_container_width=True):

    # 1) RAW DF from user inputs
    df_raw = pd.DataFrame([user_values])

    # 2) Feature engineering inside Streamlit (consistent with training artifacts)
    df_fe = apply_feature_engineering(df_raw, combined_imputer, combined_pca)

    # 3) Drop raw columns used to create engineered ones (training dropped them)
    df_fe = drop_raw_engineering_columns(df_fe)

    # 4) Build model input row exactly in feature_names order (fill missing with NaN)
    row_dict = {f: np.nan for f in feature_names}
    for f in feature_names:
        if f in df_fe.columns:
            row_dict[f] = df_fe.iloc[0][f]

    df_row = pd.DataFrame([row_dict], columns=feature_names)
    for c in df_row.columns:
        df_row[c] = pd.to_numeric(df_row[c], errors="coerce")

    try:
        pred_idx, proba, X_pca = predict_pipeline(df_row, imputer, scaler, pca, model)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
        st.stop()

    st.subheader("Prediction result")
    st.caption(f"Model used: **{model_choice_label}**")

    if proba is None:
        st.success(f"Predicted Diagnosis: **{pred_label}**")
        st.warning("This model does not provide probabilities.")
        st.stop()

    order = np.argsort(proba)[::-1]
    top1_i = int(order[0])
    top2_i = int(order[1]) if len(order) > 1 else int(order[0])

    p1 = float(proba[top1_i])
    p2 = float(proba[top2_i])
    margin = p1 - p2

    top1_label = label_encoder.inverse_transform([top1_i])[0]
    top2_label = label_encoder.inverse_transform([top2_i])[0]

    st.success(f"Predicted Diagnosis: **{pred_label}**")

    m1, m2, m3 = st.columns(3)
    m1.metric("Top-1", f"{top1_label} ({p1:.3f})")
    m2.metric("Top-2", f"{top2_label} ({p2:.3f})")
    m3.metric("Margin (P1 - P2)", f"{margin:.3f}")

    # Clinical messaging based on calibrated P(Normal)
    if normal_idx is not None and normal_idx < len(proba):
        p_normal = float(proba[normal_idx])

        top1_is_normal = is_normal_label(top1_label)
        top2_is_normal = is_normal_label(top2_label)

        if p_normal >= HIGH_T:
            if top1_is_normal and (not top2_is_normal) and (margin < MARGIN_WEAK):
                st.warning(
                    f"🟡 Uncertain: Normal vs disease overlap "
                    f"(P(Normal)={p_normal:.3f} ≥ {HIGH_T:.2f}, margin={margin:.3f} < {MARGIN_WEAK:.2f})."
                )
                st.caption("Recommendation: consider follow-up tests or clinical review.")
            else:
                st.success(
                    f"🟢 Normal likely (high confidence) "
                    f"(P(Normal)={p_normal:.3f} ≥ {HIGH_T:.2f})."
                )

        elif p_normal >= MID_T:
            st.warning(
                f"🟡 Uncertain: follow-up recommended "
                f"(P(Normal)={p_normal:.3f} between {MID_T:.2f} and {HIGH_T:.2f})."
            )
            if margin < MARGIN_WEAK:
                st.caption(f"Low separation between top classes: {top1_label} vs {top2_label}.")
        else:
            if (not top1_is_normal) and (not top2_is_normal) and (margin < MARGIN_WEAK):
                st.error(
                    f"🔴 Non-normal likely, but disease type is ambiguous "
                    f"(P(Normal)={p_normal:.3f} < {MID_T:.2f}, margin={margin:.3f} < {MARGIN_WEAK:.2f})."
                )
                st.caption("Recommendation: prioritize specialist evaluation and confirmatory testing.")
            else:
                st.error(
                    f"🔴 Non-normal likely "
                    f"(P(Normal)={p_normal:.3f} < {MID_T:.2f})."
                )
                st.caption(f"Most likely: {top1_label} (P={p1:.3f}); Second: {top2_label} (P={p2:.3f}).")
    else:
        st.warning("A 'Normal' class was not found in label_encoder.pkl.")

    # Probabilities table
    classes = list(label_encoder.classes_)
    prob_df = pd.DataFrame({"Class": classes, "Probability": proba})
    prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)

    if show_all_probs:
        st.markdown("### Class probabilities")
        st.dataframe(prob_df, use_container_width=True)

    if show_debug:
        st.markdown("###input row")
        st.dataframe(df_row, use_container_width=True)
        st.markdown("###PCA-transformed vector")
        st.dataframe(pd.DataFrame(X_pca, columns=pca_feature_names), use_container_width=True)

    # SHAP EXPLANATION
    if enable_shap:
        shap = get_shap_objects()
        if shap is None:
            st.warning("SHAP is not installed.")
        else:
            st.markdown("---")
            st.subheader("SHAP Explanation")

            explainer = build_shap_explainer(model_key)
            if explainer is None:
                st.warning("Could not build SHAP explainer.")
            else:
                class_to_explain = st.selectbox(
                    "Explain which class?",
                    options=[(top1_label, top1_i), (top2_label, top2_i)],
                    format_func=lambda x: f"{x[0]}",
                    index=0
                )
                chosen_label, chosen_i = class_to_explain

                st.markdown(f"**Explaining class:** `{chosen_label}`")

                shap_vals = render_shap_for_instance(
                    explainer=explainer,
                    x_instance_pca=X_pca,
                    class_index=int(chosen_i),
                    pca_feature_names=pca_feature_names,
                    max_display=int(shap_max_display)
                )

                if show_pc_mapping and shap_vals is not None:
                    # Top PCs 
                    v = shap_vals
                    top_k = min(int(shap_max_display), len(pca_feature_names))
                    idxs = np.argsort(np.abs(v))[::-1][:top_k]

                    # Summary table
                    summary_rows = []
                    details = []
                    for j in idxs:
                        pc_name = pca_feature_names[j]
                        hint = pc_to_original_feature_hint(pc_to_features, pc_name)

                        hint_text = pc_hint_to_text(hint, max_items=6)
                        hint_df = pc_hint_to_df(hint, max_items=int(pc_hint_max_items))

                        summary_rows.append({
                            "PC": pc_name,
                            "SHAP": float(v[j]),
                            "Top original features": hint_text
                        })
                        details.append((pc_name, float(v[j]), hint_df))

                    # st.markdown("#### Summary")
                    # st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                    # st.markdown("#### Details")
                    # for pc_name, shap_val, hint_df in details:
                    #     with st.expander(f"{pc_name}  |  SHAP = {shap_val:+.4f}", expanded=False):
                    #         if hint_df is None or hint_df.empty:
                    #             st.info("No PCA loading info found for this PC.")
                    #         else:
                    #             st.dataframe(hint_df, use_container_width=True)

st.markdown("---")

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt

st.set_page_config(page_title="Autoimmune Disease Prediction", layout="wide")


# Load artifacts
@st.cache_resource
def load_artifacts():
    base = Path(".")
    required = [
        "imputer.pkl", "scaler.pkl", "pca.pkl",
        "feature_names.pkl", "label_encoder.pkl",
        "pca_feature_names.pkl",
        "pca_pc_to_features.json",
        "shap_background.npy",
        "model_stacking_calibrated.pkl",
        "model_rf_calibrated.pkl",
        "model_knn_calibrated.pkl",
        "model_dt_calibrated.pkl",
        "thresholds_by_model.json",
        "raw_input_feature_names.pkl",
        "combined_imputer.pkl",
        "combined_pca.pkl",
    ]
    missing = [f for f in required if not (base / f).exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    thresholds_by_model = json.loads((base / "thresholds_by_model.json").read_text(encoding="utf-8"))
    pc_to_features = json.loads((base / "pca_pc_to_features.json").read_text(encoding="utf-8"))
    shap_background = np.load(base / "shap_background.npy")

    return {
        "imputer": joblib.load(base / "imputer.pkl"),
        "scaler": joblib.load(base / "scaler.pkl"),
        "pca": joblib.load(base / "pca.pkl"),
        "feature_names": joblib.load(base / "feature_names.pkl"),
        "pca_feature_names": joblib.load(base / "pca_feature_names.pkl"),
        "label_encoder": joblib.load(base / "label_encoder.pkl"),

        # NEW
        "raw_input_feature_names": joblib.load(base / "raw_input_feature_names.pkl"),
        "combined_imputer": joblib.load(base / "combined_imputer.pkl"),
        "combined_pca": joblib.load(base / "combined_pca.pkl"),

        "models": {
            "stacking": joblib.load(base / "model_stacking_calibrated.pkl"),
            "rf": joblib.load(base / "model_rf_calibrated.pkl"),
            "knn": joblib.load(base / "model_knn_calibrated.pkl"),
            "dt": joblib.load(base / "model_dt_calibrated.pkl"),
        },
        "thresholds_by_model": thresholds_by_model,
        "pc_to_features": pc_to_features,
        "shap_background": shap_background,
    }


@st.cache_data
def load_dataset(csv_path: str):
    return pd.read_csv(csv_path)



# Helpers
def safe_float(x, default=np.nan):
    try:
        if x is None:
            return default
        if isinstance(x, str) and x.strip() == "":
            return default
        return float(x)
    except Exception:
        return default


def find_normal_class_index(label_encoder):
    try:
        classes = list(label_encoder.classes_)
    except Exception:
        return None

    if "Normal" in classes:
        return classes.index("Normal")

    return None


def is_normal_label(label: str) -> bool:
    return str(label).strip() == "Normal"


def preprocess_user_input(user_dict, feature_names):
    row = {f: np.nan for f in feature_names}
    for k, v in user_dict.items():
        if k in row:
            row[k] = v
    df_row = pd.DataFrame([row], columns=feature_names)
    for c in df_row.columns:
        df_row[c] = pd.to_numeric(df_row[c], errors="coerce")
    return df_row


def transform_to_pca(df_row, imputer, scaler, pca):
    X = df_row.values
    X = imputer.transform(X)
    X = scaler.transform(X)
    X = pca.transform(X)
    return X


def predict_pipeline(df_row, imputer, scaler, pca, model):
    X = transform_to_pca(df_row, imputer, scaler, pca)
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    return pred, proba, X


def to_feature_dict_from_dataset_row(raw_row: pd.Series, feature_names: list):
    """
    (kept as-is) - old behavior: fills model feature_names and attempts to compute Symptom_Count/corelated_columns
    This may not be used anymore for the form, but we keep it to not remove anything.
    """
    user = {f: np.nan for f in feature_names}

    for f in feature_names:
        if f in raw_row.index:
            user[f] = raw_row[f]

    # Symptom_Count
    if "Symptom_Count" in feature_names and pd.isna(user.get("Symptom_Count")):
        symptom_cols = [
            "Low-grade fever",
            "Fatigue or chronic tiredness",
            "Rashes and skin lesions",
            "Brittle hair or hair loss",
            "Dry eyes and/or mouth",
            "General 'unwell' feeling",
        ]
        present = [c for c in symptom_cols if c in raw_row.index]
        if present:
            vals = pd.to_numeric(pd.Series({c: raw_row[c] for c in present}), errors="coerce").fillna(0.0)
            user["Symptom_Count"] = float(vals.sum())

    # corelated_columns
    if "corelated_columns" in feature_names and pd.isna(user.get("corelated_columns")):
        corr_cols = ["Anti_dsDNA", "Anti_RNP", "Anti_Ro_SSA", "Anti_La_SSB", "Anti_Sm"]
        present = [c for c in corr_cols if c in raw_row.index]
        if present:
            vals = pd.to_numeric(pd.Series({c: raw_row[c] for c in present}), errors="coerce").fillna(0.0)
            user["corelated_columns"] = float(vals.sum())

    return user


def to_raw_feature_dict_from_dataset_row(raw_row: pd.Series, raw_input_feature_names: list):
    """
    NEW: Fills ONLY raw inputs (what user should type in the form).
    """
    user = {f: np.nan for f in raw_input_feature_names}
    for f in raw_input_feature_names:
        if f in raw_row.index:
            user[f] = raw_row[f]
    return user


# =========================
# NEW: Feature engineering at inference (Streamlit)
# =========================
def apply_feature_engineering(df_raw: pd.DataFrame, combined_imputer, combined_pca) -> pd.DataFrame:
    df = df_raw.copy()

    # ensure numeric for everything present
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Symptom_Count
    symptom_cols = [
        "Low-grade fever",
        "Fatigue or chronic tiredness",
        "Rashes and skin lesions",
        "Brittle hair or hair loss",
        "Dry eyes and/or mouth",
        "General 'unwell' feeling",
    ]
    present_sym = [c for c in symptom_cols if c in df.columns]
    if present_sym:
        df["Symptom_Count"] = df[present_sym].fillna(0.0).sum(axis=1)
    else:
        df["Symptom_Count"] = 0.0

    # corelated_columns
    corr_cols = ["Anti_dsDNA", "Anti_RNP", "Anti_Ro_SSA", "Anti_La_SSB", "Anti_Sm"]
    present_corr = [c for c in corr_cols if c in df.columns]
    if present_corr:
        df["corelated_columns"] = df[present_corr].fillna(0.0).sum(axis=1)
    else:
        df["corelated_columns"] = 0.0

    # Combined_Feature_PCA (use saved artifacts, NOT fit again)
    combine_cols = ["CRP", "ESR", "ANA", "Sickness_Duration_Months"]
    if all(c in df.columns for c in combine_cols):
        subset = df[combine_cols]
        subset_imputed = combined_imputer.transform(subset)
        combined = combined_pca.transform(subset_imputed)
        df["Combined_Feature_PCA"] = combined
    else:
        df["Combined_Feature_PCA"] = 0.0

    return df


def drop_raw_engineering_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop raw columns that were used to create engineered features in training.
    """
    drop_cols = [
        # symptoms
        "Low-grade fever",
        "Fatigue or chronic tiredness",
        "Rashes and skin lesions",
        "Brittle hair or hair loss",
        "Dry eyes and/or mouth",
        "General 'unwell' feeling",

        # correlated antibodies
        "Anti_dsDNA", "Anti_RNP", "Anti_Ro_SSA", "Anti_La_SSB", "Anti_Sm",

        # combined PCA inputs
        "CRP", "ESR", "ANA", "Sickness_Duration_Months"
    ]
    return df.drop(columns=[c for c in drop_cols if c in df.columns])


# SHAP utils 
@st.cache_resource
def get_shap_objects():
    try:
        import shap
        return shap
    except Exception:
        return None


@st.cache_resource
def build_shap_explainer(model_key: str):
    shap = get_shap_objects()
    if shap is None:
        return None

    models = st.session_state.get("_models_for_shap", None)
    bg_all = st.session_state.get("_shap_background", None)

    if models is None or bg_all is None:
        return None
    if model_key not in models:
        return None

    model = models[model_key]

    bg = bg_all
    if isinstance(bg, np.ndarray) and bg.shape[0] > 60:
        bg = bg[:60]

    def f(X):
        return model.predict_proba(X)

    masker = shap.maskers.Independent(bg)
    explainer = shap.Explainer(f, masker)
    return explainer


def render_shap_for_instance(
    explainer,
    x_instance_pca: np.ndarray,
    class_index: int,
    pca_feature_names: list,
    max_display: int = 14
):
    shap = get_shap_objects()
    if shap is None or explainer is None:
        st.warning("SHAP is not available.")
        return None

    x1 = np.asarray(x_instance_pca, dtype=np.float32)
    if x1.ndim == 2:
        x1 = x1[0]

    with st.spinner("Computing SHAP explanation ..."):
        sv = explainer(x1.reshape(1, -1))

    values = sv.values
    base = sv.base_values

    if values.ndim == 3:
        v = values[0, :, class_index]
        b = base[0][class_index] if np.ndim(base[0]) > 0 else float(base[0])
    else:
        v = values[0, :]
        b = float(base[0]) if np.ndim(base) > 0 else float(base)

    exp = shap.Explanation(
        values=v,
        base_values=b,
        data=x1,
        feature_names=pca_feature_names
    )

    fig = plt.figure()
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    st.pyplot(fig, clear_figure=True)

    # st.markdown("**Top contributors for this prediction**")
    # abs_df = pd.DataFrame({
    #     "PC": pca_feature_names,
    #     "abs_shap": np.abs(v),
    #     "shap": v
    # }).sort_values("abs_shap", ascending=False).head(max_display)
    # st.dataframe(abs_df, use_container_width=True)

    # return v


def pc_to_original_feature_hint(pc_to_features: dict, pc_name: str):
    items = pc_to_features.get(pc_name, None)
    if not items or not isinstance(items, list):
        return None
    return items


def pc_hint_to_df(hint_list, max_items: int = 10) -> pd.DataFrame:
    if not hint_list or not isinstance(hint_list, list):
        return pd.DataFrame(columns=["Feature", "Loading", "abs(Loading)"])

    rows = []
    for h in hint_list[:max_items]:
        try:
            feat = str(h.get("feature", "")).strip()
            loading = float(h.get("loading", np.nan))
            if feat == "" or np.isnan(loading):
                continue
            rows.append({"Feature": feat, "Loading": loading, "abs(Loading)": abs(loading)})
        except Exception:
            continue

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return pd.DataFrame(columns=["Feature", "Loading", "abs(Loading)"])

    df = df.sort_values("abs(Loading)", ascending=False).reset_index(drop=True)
    return df


def pc_hint_to_text(hint_list, max_items: int = 6) -> str:
    df = pc_hint_to_df(hint_list, max_items=max_items)
    if df.empty:
        return ""
    return ", ".join([f"{r.Feature} ({r.Loading:+.2f})" for r in df.itertuples(index=False)])


# App start
st.title("Autoimmune Disease Prediction")

try:
    A = load_artifacts()
except Exception as e:
    st.error("Failed to load model artifacts.")
    st.code(str(e))
    st.stop()

models = A["models"]
thresholds_by_model = A["thresholds_by_model"]

imputer = A["imputer"]
scaler = A["scaler"]
pca = A["pca"]
feature_names = A["feature_names"]
pca_feature_names = A["pca_feature_names"]
pc_to_features = A["pc_to_features"]
shap_background = A["shap_background"]

# NEW
raw_input_feature_names = A["raw_input_feature_names"]
combined_imputer = A["combined_imputer"]
combined_pca = A["combined_pca"]

label_encoder = A["label_encoder"]
normal_idx = find_normal_class_index(label_encoder)

# Make SHAP resources accessible without hashing them
st.session_state["_models_for_shap"] = models
st.session_state["_shap_background"] = shap_background



# Sidebar
st.sidebar.header("Model selection")

MODEL_OPTIONS = {
    "Use Stacking": "stacking",
    "Use RandomForest": "rf",
    "Use KNN": "knn",
    "Use DecisionTree": "dt",
}


def apply_thresholds_for_model_key(model_key: str):
    t = thresholds_by_model.get(model_key, None)
    if not isinstance(t, dict):
        st.session_state["HIGH_T"] = 0.70
        st.session_state["MID_T"] = 0.40
        st.session_state["MARGIN_WEAK"] = 0.10
        return

    st.session_state["HIGH_T"] = float(t.get("HIGH_T", 0.70))
    st.session_state["MID_T"] = float(t.get("MID_T", 0.40))
    st.session_state["MARGIN_WEAK"] = float(t.get("MARGIN_WEAK", 0.10))


if "model_choice_label" not in st.session_state:
    st.session_state["model_choice_label"] = "Use Stacking"

if "HIGH_T" not in st.session_state or "MID_T" not in st.session_state or "MARGIN_WEAK" not in st.session_state:
    apply_thresholds_for_model_key(MODEL_OPTIONS[st.session_state["model_choice_label"]])


def on_model_change():
    label = st.session_state["model_choice_label"]
    key = MODEL_OPTIONS[label]
    apply_thresholds_for_model_key(key)


model_choice_label = st.sidebar.selectbox(
    "Choose model",
    list(MODEL_OPTIONS.keys()),
    key="model_choice_label",
    on_change=on_model_change
)

model_key = MODEL_OPTIONS[model_choice_label]
model = models[model_key]

st.sidebar.markdown("---")
show_all_probs = st.sidebar.checkbox("Show all class probabilities", value=True)
show_debug = st.sidebar.checkbox("Show input row", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset quick test")
default_dataset_name = "Autoimmune_Dataset.csv"
dataset_path = st.sidebar.text_input("CSV filename", value=default_dataset_name)

st.sidebar.markdown("---")
st.sidebar.subheader("Clinical thresholds")

HIGH_T = st.sidebar.slider("High confidence Normal (>=)", 0.50, 0.99, float(st.session_state["HIGH_T"]), 0.01, key="HIGH_T_slider")
MID_T  = st.sidebar.slider("Uncertain lower bound (>=)", 0.05, 0.95, float(st.session_state["MID_T"]), 0.01, key="MID_T_slider")
MARGIN_WEAK = st.sidebar.slider("Weak margin (<) => uncertainty", 0.01, 0.30, float(st.session_state["MARGIN_WEAK"]), 0.01, key="MARGIN_slider")

st.session_state["HIGH_T"] = float(HIGH_T)
st.session_state["MID_T"] = float(MID_T)
st.session_state["MARGIN_WEAK"] = float(MARGIN_WEAK)

if MID_T >= HIGH_T:
    st.sidebar.error("MID_T must be smaller than HIGH_T.")

with st.sidebar.expander("Threshold info for selected model", expanded=False):
    st.json(thresholds_by_model.get(model_key, {}))

st.sidebar.markdown("---")
st.sidebar.subheader("Explainability")
enable_shap = st.sidebar.checkbox("Enable SHAP explanation", value=False)
shap_max_display = st.sidebar.slider("SHAP max PCs to display", 5, 25, 14, 1)
show_pc_mapping = st.sidebar.checkbox("Original Feature mapping", value=True)
pc_hint_max_items = st.sidebar.slider("PC mapping: max original features per PC", 3, 20, 12, 1)


# Session state for form values (UPDATED: raw_input_feature_names)
if "form_values" not in st.session_state:
    st.session_state.form_values = {f: "" for f in raw_input_feature_names}
if "loaded_row_info" not in st.session_state:
    st.session_state.loaded_row_info = None

# Buttons
colA, colB = st.columns([1, 1])

with colA:
    if st.button("Load random sample from dataset", use_container_width=True):
        try:
            df_data = load_dataset(dataset_path)
            if len(df_data) == 0:
                st.warning("The dataset is empty.")
            else:
                idx = int(np.random.randint(0, len(df_data)))
                raw_row = df_data.iloc[idx]

                # UPDATED: load raw inputs only
                user_dict = to_raw_feature_dict_from_dataset_row(raw_row, raw_input_feature_names)

                for f in raw_input_feature_names:
                    v = user_dict.get(f, np.nan)
                    st.session_state.form_values[f] = "" if pd.isna(v) else str(v)

                true_diag = raw_row.get("Diagnosis", None) if "Diagnosis" in df_data.columns else None
                st.session_state.loaded_row_info = {"row_index": idx, "true_diagnosis_raw": true_diag}
                st.success(f"Loaded row #{idx} into the form.")
        except Exception as e:
            st.error("Failed to load dataset row.")
            st.code(str(e))

with colB:
    if st.button("Clear inputs", use_container_width=True):
        st.session_state.form_values = {f: "" for f in raw_input_feature_names}
        st.session_state.loaded_row_info = None
        st.success("All inputs cleared.")


# Inputs UI (UPDATED: raw only)
st.markdown("### Enter patient features ")

cols = st.columns(3)
user_values = {}

for i, feat in enumerate(raw_input_feature_names):
    col = cols[i % 3]
    with col:
        raw = st.text_input(feat, value=st.session_state.form_values.get(feat, ""), help="Leave empty if unknown.")
        user_values[feat] = safe_float(raw, default=np.nan)

if st.session_state.loaded_row_info is not None:
    info = st.session_state.loaded_row_info
    msg = f"Loaded dataset row index: **{info['row_index']}**"
    if info.get("true_diagnosis_raw") is not None:
        msg += f" | dataset Diagnosis: **{info['true_diagnosis_raw']}**"
    st.info(msg)

st.markdown("---")


# Predict (UPDATED PIPELINE)
if st.button("Predict", use_container_width=True):

    # 1) RAW DF from user inputs
    df_raw = pd.DataFrame([user_values])

    # 2) Feature engineering inside Streamlit (consistent with training artifacts)
    df_fe = apply_feature_engineering(df_raw, combined_imputer, combined_pca)

    # 3) Drop raw columns used to create engineered ones (training dropped them)
    df_fe = drop_raw_engineering_columns(df_fe)

    # 4) Build model input row exactly in feature_names order (fill missing with NaN)
    row_dict = {f: np.nan for f in feature_names}
    for f in feature_names:
        if f in df_fe.columns:
            row_dict[f] = df_fe.iloc[0][f]

    df_row = pd.DataFrame([row_dict], columns=feature_names)
    for c in df_row.columns:
        df_row[c] = pd.to_numeric(df_row[c], errors="coerce")

    try:
        pred_idx, proba, X_pca = predict_pipeline(df_row, imputer, scaler, pca, model)
        pred_label = label_encoder.inverse_transform([pred_idx])[0]
    except Exception as e:
        st.error("Prediction failed.")
        st.code(str(e))
        st.stop()

    st.subheader("Prediction result")
    st.caption(f"Model used: **{model_choice_label}**")

    if proba is None:
        st.success(f"Predicted Diagnosis: **{pred_label}**")
        st.warning("This model does not provide probabilities.")
        st.stop()

    order = np.argsort(proba)[::-1]
    top1_i = int(order[0])
    top2_i = int(order[1]) if len(order) > 1 else int(order[0])

    p1 = float(proba[top1_i])
    p2 = float(proba[top2_i])
    margin = p1 - p2

    top1_label = label_encoder.inverse_transform([top1_i])[0]
    top2_label = label_encoder.inverse_transform([top2_i])[0]

    st.success(f"Predicted Diagnosis: **{pred_label}**")

    m1, m2, m3 = st.columns(3)
    m1.metric("Top-1", f"{top1_label} ({p1:.3f})")
    m2.metric("Top-2", f"{top2_label} ({p2:.3f})")
    m3.metric("Margin (P1 - P2)", f"{margin:.3f}")

    # Clinical messaging based on calibrated P(Normal)
    if normal_idx is not None and normal_idx < len(proba):
        p_normal = float(proba[normal_idx])

        top1_is_normal = is_normal_label(top1_label)
        top2_is_normal = is_normal_label(top2_label)

        if p_normal >= HIGH_T:
            if top1_is_normal and (not top2_is_normal) and (margin < MARGIN_WEAK):
                st.warning(
                    f"🟡 Uncertain: Normal vs disease overlap "
                    f"(P(Normal)={p_normal:.3f} ≥ {HIGH_T:.2f}, margin={margin:.3f} < {MARGIN_WEAK:.2f})."
                )
                st.caption("Recommendation: consider follow-up tests or clinical review.")
            else:
                st.success(
                    f"🟢 Normal likely (high confidence) "
                    f"(P(Normal)={p_normal:.3f} ≥ {HIGH_T:.2f})."
                )

        elif p_normal >= MID_T:
            st.warning(
                f"🟡 Uncertain: follow-up recommended "
                f"(P(Normal)={p_normal:.3f} between {MID_T:.2f} and {HIGH_T:.2f})."
            )
            if margin < MARGIN_WEAK:
                st.caption(f"Low separation between top classes: {top1_label} vs {top2_label}.")
        else:
            if (not top1_is_normal) and (not top2_is_normal) and (margin < MARGIN_WEAK):
                st.error(
                    f"🔴 Non-normal likely, but disease type is ambiguous "
                    f"(P(Normal)={p_normal:.3f} < {MID_T:.2f}, margin={margin:.3f} < {MARGIN_WEAK:.2f})."
                )
                st.caption("Recommendation: prioritize specialist evaluation and confirmatory testing.")
            else:
                st.error(
                    f"🔴 Non-normal likely "
                    f"(P(Normal)={p_normal:.3f} < {MID_T:.2f})."
                )
                st.caption(f"Most likely: {top1_label} (P={p1:.3f}); Second: {top2_label} (P={p2:.3f}).")
    else:
        st.warning("A 'Normal' class was not found in label_encoder.pkl.")

    # Probabilities table
    classes = list(label_encoder.classes_)
    prob_df = pd.DataFrame({"Class": classes, "Probability": proba})
    prob_df = prob_df.sort_values("Probability", ascending=False).reset_index(drop=True)

    if show_all_probs:
        st.markdown("### Class probabilities")
        st.dataframe(prob_df, use_container_width=True)

    if show_debug:
        st.markdown("###input row")
        st.dataframe(df_row, use_container_width=True)
        st.markdown("###PCA-transformed vector")
        st.dataframe(pd.DataFrame(X_pca, columns=pca_feature_names), use_container_width=True)

    # SHAP EXPLANATION
    if enable_shap:
        shap = get_shap_objects()
        if shap is None:
            st.warning("SHAP is not installed.")
        else:
            st.markdown("---")
            st.subheader("SHAP Explanation")

            explainer = build_shap_explainer(model_key)
            if explainer is None:
                st.warning("Could not build SHAP explainer.")
            else:
                class_to_explain = st.selectbox(
                    "Explain which class?",
                    options=[(top1_label, top1_i), (top2_label, top2_i)],
                    format_func=lambda x: f"{x[0]}",
                    index=0
                )
                chosen_label, chosen_i = class_to_explain

                st.markdown(f"**Explaining class:** `{chosen_label}`")

                shap_vals = render_shap_for_instance(
                    explainer=explainer,
                    x_instance_pca=X_pca,
                    class_index=int(chosen_i),
                    pca_feature_names=pca_feature_names,
                    max_display=int(shap_max_display)
                )

                if show_pc_mapping and shap_vals is not None:
                    # Top PCs 
                    v = shap_vals
                    top_k = min(int(shap_max_display), len(pca_feature_names))
                    idxs = np.argsort(np.abs(v))[::-1][:top_k]

                    # Summary table
                    summary_rows = []
                    details = []
                    for j in idxs:
                        pc_name = pca_feature_names[j]
                        hint = pc_to_original_feature_hint(pc_to_features, pc_name)

                        hint_text = pc_hint_to_text(hint, max_items=6)
                        hint_df = pc_hint_to_df(hint, max_items=int(pc_hint_max_items))

                        summary_rows.append({
                            "PC": pc_name,
                            "SHAP": float(v[j]),
                            "Top original features": hint_text
                        })
                        details.append((pc_name, float(v[j]), hint_df))

                    # st.markdown("#### Summary")
                    # st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

                    # st.markdown("#### Details")
                    # for pc_name, shap_val, hint_df in details:
                    #     with st.expander(f"{pc_name}  |  SHAP = {shap_val:+.4f}", expanded=False):
                    #         if hint_df is None or hint_df.empty:
                    #             st.info("No PCA loading info found for this PC.")
                    #         else:
                    #             st.dataframe(hint_df, use_container_width=True)

st.markdown("---")

