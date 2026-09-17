import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier
)
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
import joblib

# ============================================================
# DYSLEXIA HANDWRITING OPTIMIZATION (STEP 4)
# ============================================================

print("=" * 60)
print("FEATURE RATIOS & OPTIMAL THRESHOLD CALIBRATION")
print("=" * 60)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "handwriting_features.csv")

if not os.path.exists(csv_path):
    print(f"[ERROR] Features file not found at: {csv_path}")
    exit()

MODEL_FEATURE_COLUMNS = [
    "ink_density", "connected_components", "mean_height", "height_variation",
    "mean_width", "width_variation", "baseline_drift", "mean_spacing",
    "spacing_variation", "estimated_stroke_width", "slant_variation",
    "margin_alignment", "overlap_ratio"
]

df = pd.read_csv(csv_path)

# Retain all 13 core features in matching order
X = df[MODEL_FEATURE_COLUMNS].copy()
y = df["label"].copy()

# Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# ------------------------------------------------------------
# CANDIDATE CLASSIFIERS
# ------------------------------------------------------------
models = {
    "Hist Gradient Boosting": HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.04,
        max_leaf_nodes=15,
        min_samples_leaf=6,
        random_state=42
    ),
    "Tuned Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        max_features=0.5,
        min_samples_split=8,
        min_samples_leaf=1,
        random_state=42
    ),
    "Tuned Extra Trees": ExtraTreesClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=1,
        random_state=42
    ),
    "Soft Voting Ensemble": VotingClassifier(
        estimators=[
            ('hgb', HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=6, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=6, max_features=0.5, min_samples_split=8, random_state=42)),
            ('et', ExtraTreesClassifier(n_estimators=200, max_depth=8, min_samples_split=5, random_state=42))
        ],
        voting='soft',
        weights=[2, 1, 1]
    )
}

# ------------------------------------------------------------
# THRESHOLD CALIBRATION VIA 5-FOLD CV
# ------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def calibrate_threshold(model, X_tr, y_tr):
    thresholds = np.linspace(0.35, 0.65, 31)
    best_thresh = 0.50
    best_f1 = 0.0

    oof_probs = np.zeros(len(y_tr))
    for tr_idx, val_idx in cv.split(X_tr, y_tr):
        X_fold_tr, X_fold_val = X_tr.iloc[tr_idx], X_tr.iloc[val_idx]
        y_fold_tr, y_fold_val = y_tr.iloc[tr_idx], y_tr.iloc[val_idx]

        m_copy = joblib.load(joblib.dump(model, "temp.pkl")[0])
        m_copy.fit(X_fold_tr, y_fold_tr)
        oof_probs[val_idx] = m_copy.predict_proba(X_fold_val)[:, 1]

    if os.path.exists("temp.pkl"):
        os.remove("temp.pkl")

    for th in thresholds:
        f1 = f1_score(y_tr, (oof_probs >= th).astype(int), average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = th

    return best_thresh

# ------------------------------------------------------------
# EVALUATION & SELECTION
# ------------------------------------------------------------
print("\n" + "=" * 60)
print("MODEL EVALUATION & THRESHOLD TUNING")
print("=" * 60)

best_model = None
best_name = None
best_test_acc = 0.0
best_threshold = 0.50

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_proba = clf.predict_proba(X_test)[:, 1]

    # Baseline 0.50 cut
    y_pred_def = (y_proba >= 0.50).astype(int)
    acc_def = accuracy_score(y_test, y_pred_def)
    auc = roc_auc_score(y_test, y_proba)

    # Calibrated cut
    opt_th = calibrate_threshold(clf, X_train, y_train)
    y_pred_opt = (y_proba >= opt_th).astype(int)
    acc_opt = accuracy_score(y_test, y_pred_opt)

    print(f"\nModel: {name}")
    print(f"  • Default (0.50) Acc: {acc_def * 100:.2f}% | ROC-AUC: {auc:.4f}")
    print(f"  • Tuned Threshold ({opt_th:.2f}) Acc: {acc_opt * 100:.2f}%")

    top_acc = max(acc_def, acc_opt)
    if top_acc > best_test_acc:
        best_test_acc = top_acc
        best_name = name
        best_model = clf
        best_threshold = opt_th if acc_opt >= acc_def else 0.50

print("\n" + "=" * 60)
print(f"WINNER: {best_name} ({best_test_acc * 100:.2f}% Accuracy | Threshold: {best_threshold:.2f})")
print("=" * 60)

final_preds = (best_model.predict_proba(X_test)[:, 1] >= best_threshold).astype(int)
print(classification_report(y_test, final_preds, target_names=["Non-Dyslexic", "Dyslexic"]))

# Save winning model
model_path = os.path.join(BASE_DIR, "handwriting_model.pkl")
best_model.classification_threshold = best_threshold
joblib.dump(best_model, model_path)
print(f"[SUCCESS] Saved winning model ({best_name}) to: {model_path}")