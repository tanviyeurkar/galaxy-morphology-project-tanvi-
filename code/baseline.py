"""
baseline.py
------------
Baseline morphology classifier using ONLY the 5 hand-crafted features from
features.py (mean_brightness, size_px, ellipticity, axis_ratio, concentration)
-- no raw pixels. Trains both logistic regression and a random forest and
reports whichever is stronger. The CNN (cnn.py) has to beat this number on
the same test split to be worth anything.

Usage:
    python baseline.py --data-dir ../data
"""
import argparse
import os

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

from data import load_raw, GALAXY10_CLASSES
from features import extract_features_batch


def get_or_compute_features(data_dir, images):
    feat_path = os.path.join(data_dir, "handcrafted_features.csv")
    if os.path.exists(feat_path):
        print(f"Loading cached features from {feat_path}")
        return pd.read_csv(feat_path)
    feats = extract_features_batch(images)
    df = pd.DataFrame(feats)
    df.to_csv(feat_path, index=False)
    print(f"Saved features -> {feat_path}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    args = ap.parse_args()

    images, labels = load_raw(args.data_dir)
    split = np.load(os.path.join(args.data_dir, "split_indices.npz"))
    train_idx, val_idx, test_idx = split["train_idx"], split["val_idx"], split["test_idx"]

    df = get_or_compute_features(args.data_dir, images)
    X = df.values
    y = labels

    scaler = StandardScaler().fit(X[train_idx])
    X_train, X_val, X_test = scaler.transform(X[train_idx]), scaler.transform(X[val_idx]), scaler.transform(X[test_idx])
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    results = {}

    # Note: scikit-learn >=1.5 removed the multi_class= argument; lbfgs now
    # always uses a multinomial (softmax) fit for multi-class problems.
    logreg = LogisticRegression(max_iter=2000, C=1.0)
    logreg.fit(X_train, y_train)
    acc_lr = accuracy_score(y_test, logreg.predict(X_test))
    bal_lr = balanced_accuracy_score(y_test, logreg.predict(X_test))
    results["logreg"] = (logreg, acc_lr, bal_lr)
    print(f"Logistic Regression:  acc={acc_lr:.4f}  balanced_acc={bal_lr:.4f}")

    rf = RandomForestClassifier(n_estimators=400, max_depth=None, n_jobs=-1, random_state=42)
    rf.fit(X_train, y_train)
    acc_rf = accuracy_score(y_test, rf.predict(X_test))
    bal_rf = balanced_accuracy_score(y_test, rf.predict(X_test))
    results["random_forest"] = (rf, acc_rf, bal_rf)
    print(f"Random Forest:        acc={acc_rf:.4f}  balanced_acc={bal_rf:.4f}")

    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc, best_bal = results[best_name]
    print(f"\nBest baseline: {best_name}  acc={best_acc:.4f}  balanced_acc={best_bal:.4f}")
    print("\nClassification report (best baseline):")
    print(
        classification_report(
            y_test,
            best_model.predict(X_test),
            target_names=[GALAXY10_CLASSES[c] for c in sorted(GALAXY10_CLASSES)],
        )
    )

    if best_name == "random_forest":
        importances = pd.Series(rf.feature_importances_, index=df.columns).sort_values(ascending=False)
        print("\nRandom forest feature importances:")
        print(importances)

    np.savez(
        os.path.join(args.data_dir, "baseline_results.npz"),
        best_name=best_name,
        best_acc=best_acc,
        best_bal_acc=best_bal,
        test_pred=best_model.predict(X_test),
        test_true=y_test,
    )
    print(f"\nSaved baseline predictions -> {os.path.join(args.data_dir, 'baseline_results.npz')}")


if __name__ == "__main__":
    main()
