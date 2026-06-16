"""
train_model.py
ValSafe ML — Incident Prediction Model Trainer
Trains 3 Random Forest classifiers to predict:
  1. severity_level  (low / medium / high)
  2. type            (unsafe_act / near_miss / incident / ...)
  3. status          (Pending / Resolved / Escalated)

Run:   python train_model.py
Output: models/severity_model.pkl
        models/type_model.pkl
        models/status_model.pkl
        models/encoders.pkl
        models/feature_columns.pkl
        models/evaluation_report.txt
"""

import os
import joblib
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for VS Code)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)
os.makedirs("data",   exist_ok=True)

# ── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  ValSafe ML — Model Training")
print("=" * 60)

DATA_PATH = "data/incidents.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"'{DATA_PATH}' not found. Run generate_data.py first."
    )

df = pd.read_csv(DATA_PATH)
print(f"\n[1/6] Loaded {len(df):,} records with {df.shape[1]} columns.")

# ── 2. FEATURE ENGINEERING ───────────────────────────────────────────────────
print("\n[2/6] Engineering features...")

# Encode categorical columns the model will use as INPUT features
CATEGORICAL_FEATURES = ["cat_id", "root_cause", "dept_id", "site_id", "media_status"]
TARGET_COLUMNS       = ["severity_level", "type", "status"]

encoders = {}

for col in CATEGORICAL_FEATURES + TARGET_COLUMNS:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"   Encoded '{col}' → {list(le.classes_)}")

# Final feature set (what the model sees as input)
FEATURE_COLS = [
    "cat_id_enc",
    "root_cause_enc",
    "dept_id_enc",
    "site_id_enc",
    "media_status_enc",
    "hour_of_day",
    "day_of_week",
    "month",
    "days_since_occurred",
    "resolved_by_admin",
]

# Convert boolean to int
df["resolved_by_admin"] = df["resolved_by_admin"].astype(int)

X = df[FEATURE_COLS]
print(f"\n   Feature matrix shape: {X.shape}")
print(f"   Features used: {FEATURE_COLS}")

# ── 3. TRAIN MODELS ──────────────────────────────────────────────────────────
print("\n[3/6] Training models...")

TARGETS = {
    "severity_level": "severity_model",
    "type":           "type_model",
    "status":         "status_model",
}

trained_models  = {}
eval_results    = {}
report_lines    = []

report_lines.append("=" * 60)
report_lines.append("  ValSafe ML — Evaluation Report")
report_lines.append(f"  Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
report_lines.append("=" * 60)

for target_col, model_name in TARGETS.items():
    print(f"\n   ── Training: {target_col} ──")

    y = df[target_col + "_enc"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",   # handles uneven class sizes
        random_state=42,
        n_jobs=-1,                 # use all CPU cores
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy", n_jobs=-1)

    label_names = encoders[target_col].classes_
    report      = classification_report(y_test, y_pred, target_names=label_names)
    cm          = confusion_matrix(y_test, y_pred)

    print(f"   Accuracy : {accuracy:.4f}")
    print(f"   CV mean  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"\n{report}")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True, fmt="d",
        xticklabels=label_names,
        yticklabels=label_names,
        cmap="Blues", ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"Confusion Matrix — {target_col}", fontsize=11, pad=10)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual", fontsize=9)
    plt.tight_layout()
    cm_path = f"models/{model_name}_confusion_matrix.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    print(f"   Confusion matrix saved → {cm_path}")

    # Feature importance plot
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values()
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    importances.plot(kind="barh", ax=ax2, color="#3B8BD4")
    ax2.set_title(f"Feature Importance — {target_col}", fontsize=11, pad=10)
    ax2.set_xlabel("Importance score", fontsize=9)
    plt.tight_layout()
    fi_path = f"models/{model_name}_feature_importance.png"
    fig2.savefig(fi_path, dpi=120)
    plt.close(fig2)
    print(f"   Feature importance saved → {fi_path}")

    # Store everything
    trained_models[model_name] = model
    eval_results[target_col] = {
        "accuracy":  accuracy,
        "cv_mean":   cv_scores.mean(),
        "cv_std":    cv_scores.std(),
        "report":    report,
        "labels":    list(label_names),
    }

    # Write to report
    report_lines.append(f"\n{'─'*60}")
    report_lines.append(f"  Target: {target_col}")
    report_lines.append(f"  Accuracy : {accuracy:.4f}")
    report_lines.append(f"  CV Score : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    report_lines.append(f"\n{report}")

# ── 4. SAVE MODELS ───────────────────────────────────────────────────────────
print("\n[4/6] Saving models to models/ ...")

for model_name, model in trained_models.items():
    path = f"models/{model_name}.pkl"
    joblib.dump(model, path)
    print(f"   Saved {path}")

joblib.dump(encoders,    "models/encoders.pkl")
joblib.dump(FEATURE_COLS, "models/feature_columns.pkl")
print("   Saved models/encoders.pkl")
print("   Saved models/feature_columns.pkl")

# ── 5. SAVE EVALUATION REPORT ────────────────────────────────────────────────
print("\n[5/6] Saving evaluation report...")

report_path = "models/evaluation_report.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))
print(f"   Saved {report_path}")

# ── 6. FINAL SUMMARY ─────────────────────────────────────────────────────────
print("\n[6/6] Done!\n")
print("=" * 60)
print("  SUMMARY")
print("=" * 60)
for target_col, res in eval_results.items():
    print(f"  {target_col:<20} accuracy={res['accuracy']:.4f}   cv={res['cv_mean']:.4f} ± {res['cv_std']:.4f}")

print("\n  Files saved:")
print("    models/severity_model.pkl")
print("    models/type_model.pkl")
print("    models/status_model.pkl")
print("    models/encoders.pkl")
print("    models/feature_columns.pkl")
print("    models/evaluation_report.txt")
print("    models/*_confusion_matrix.png")
print("    models/*_feature_importance.png")
print("\n  Ready for Phase 4 → python app.py (Streamlit dashboard)")
print("=" * 60)