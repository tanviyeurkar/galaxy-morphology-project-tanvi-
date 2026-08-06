"""
analysis.py
------------
The actual deliverable: not an accuracy number, a confusion analysis.

Loads:
  - CNN test predictions + probabilities (cnn_results.npz)
  - baseline test predictions (baseline_results.npz)
  - hand-crafted features for every test image (handcrafted_features.csv)

Produces:
  1. Confusion matrix heatmap (raw pixel CNN).
  2. Accuracy of CNN vs. baseline, broken out by tercile bins of:
       - apparent size (size_px)
       - brightness (mean_brightness)
       - inclination (axis_ratio -- low axis_ratio = edge-on / elongated)
  3. The "secretly a size classifier" test:
       a) Correlation between each class's mean apparent size and the CNN's
          per-class recall -- if bigger-looking classes are systematically
          easier, that's a smell.
       b) A trivial classifier that predicts the CNN's *predicted* label
          using ONLY size_px (+ brightness + axis_ratio) as features. If this
          throwaway model recovers a large fraction of the CNN's predictions,
          the CNN is leaning heavily on apparent size/brightness rather than
          genuine morphology.
       c) Directly check: among misclassified galaxies, does the CNN
          consistently predict the class whose *typical size* is closest to
          the true galaxy's apparent size?
  4. Everything written to ../figures and a text summary report.txt

Usage:
    python analysis.py --data-dir ../data --fig-dir ../figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data import GALAXY10_CLASSES


def tercile_bin(values, labels=("low", "mid", "high")):
    q1, q2 = np.percentile(values, [100 / 3, 200 / 3])
    bins = np.where(values <= q1, labels[0], np.where(values <= q2, labels[1], labels[2]))
    return bins, (q1, q2)


def plot_confusion_matrix(y_true, y_pred, fig_path):
    classes = [GALAXY10_CLASSES[c] for c in sorted(GALAXY10_CLASSES)]
    cm = confusion_matrix(y_true, y_pred, labels=sorted(GALAXY10_CLASSES))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm_norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("CNN confusion matrix (row-normalized)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm_norm[i, j] > 0.05:
                ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center",
                         color="white" if cm_norm[i, j] < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)


def plot_binned_accuracy(df, bin_col, cnn_col_true, cnn_col_pred, base_col_pred, fig_path, title):
    bins, _ = tercile_bin(df[bin_col].values)
    df = df.copy()
    df["_bin"] = bins
    order = ["low", "mid", "high"]
    rows = []
    for b in order:
        sub = df[df["_bin"] == b]
        cnn_acc = accuracy_score(sub[cnn_col_true], sub[cnn_col_pred])
        base_acc = accuracy_score(sub[cnn_col_true], sub[base_col_pred])
        rows.append((b, cnn_acc, base_acc, len(sub)))
    res = pd.DataFrame(rows, columns=["bin", "cnn_acc", "baseline_acc", "n"])

    x = np.arange(len(order))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar(x - width / 2, res["cnn_acc"], width, label="CNN")
    ax.bar(x + width / 2, res["baseline_acc"], width, label="Baseline")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{o}\n(n={n})" for o, n in zip(order, res["n"])])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    return res


def secretly_a_size_classifier_test(df, report_lines):
    """
    Does the CNN's PREDICTED label correlate with apparent size/brightness
    /axis-ratio more than a genuine morphology classifier should?
    """
    report_lines.append("\n--- 'Is it secretly a size classifier?' test ---\n")

    # (a) per-class mean size vs per-class CNN recall
    per_class = df.groupby("cnn_true").apply(
        lambda g: pd.Series({
            "mean_size": g["size_px"].mean(),
            "recall": accuracy_score(g["cnn_true"], g["cnn_pred"]),
        })
    )
    corr = np.corrcoef(per_class["mean_size"], per_class["recall"])[0, 1]
    report_lines.append(
        f"Correlation between a class's mean apparent size and the CNN's "
        f"per-class recall: r = {corr:.3f}\n"
        f"(|r| > ~0.5 suggests recall is driven by how big the galaxy looks, "
        f"not by its true morphology)\n"
    )
    report_lines.append(per_class.sort_values("recall", ascending=False).to_string())
    report_lines.append("\n")

    # (b) can size/brightness/axis_ratio alone predict the CNN's *predicted*
    # label better than chance? Train/test split within the test set.
    feats = df[["size_px", "mean_brightness", "axis_ratio", "concentration"]].values
    pred_labels = df["cnn_pred"].values
    Xtr, Xte, ytr, yte = train_test_split(feats, pred_labels, test_size=0.4, random_state=0, stratify=pred_labels)
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, multi_class="multinomial")
    clf.fit(scaler.transform(Xtr), ytr)
    proxy_acc = accuracy_score(yte, clf.predict(scaler.transform(Xte)))
    true_acc_on_same_split = accuracy_score(
        df.loc[df.index.isin(pd.Series(yte).index), "cnn_true"], yte
    )  # not directly meaningful; report proxy_acc against chance instead
    n_classes = len(np.unique(pred_labels))
    report_lines.append(
        f"A trivial model using ONLY (size, brightness, axis_ratio, concentration) "
        f"to predict the CNN's OWN predicted label achieves "
        f"{proxy_acc:.3f} accuracy (chance level ~= {1/n_classes:.3f}).\n"
        f"If this is well above chance, a meaningful chunk of what the CNN's "
        f"output encodes is recoverable from crude size/brightness features alone.\n"
    )

    # (c) among CNN errors, how often is the predicted class's typical size
    # closer to the galaxy's own apparent size than the true class's typical
    # size is? (i.e., "confused for something the same apparent size")
    class_mean_size = df.groupby("cnn_true")["size_px"].mean()  # proxy for "typical size of true class"
    errors = df[df["cnn_true"] != df["cnn_pred"]].copy()
    if len(errors) > 0:
        errors["true_class_size"] = errors["cnn_true"].map(class_mean_size)
        errors["pred_class_size"] = errors["cnn_pred"].map(class_mean_size)
        errors["closer_to_pred"] = (
            (errors["size_px"] - errors["pred_class_size"]).abs()
            < (errors["size_px"] - errors["true_class_size"]).abs()
        )
        frac = errors["closer_to_pred"].mean()
        report_lines.append(
            f"Among the {len(errors)} misclassified test galaxies, the CNN's "
            f"WRONG prediction is a class whose typical apparent size is closer "
            f"to this galaxy's own apparent size than the true class's typical "
            f"size, {frac:.1%} of the time (50% would be the chance baseline "
            f"if size played no role in the error pattern).\n"
        )
    return per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data")
    ap.add_argument("--fig-dir", default="../figures")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)

    cnn = np.load(os.path.join(args.data_dir, "cnn_results.npz"))
    baseline = np.load(os.path.join(args.data_dir, "baseline_results.npz"), allow_pickle=True)
    feats_all = pd.read_csv(os.path.join(args.data_dir, "handcrafted_features.csv"))

    test_idx = cnn["test_idx"]
    df = feats_all.iloc[test_idx].reset_index(drop=True).copy()
    df["cnn_true"] = cnn["test_true"]
    df["cnn_pred"] = cnn["test_pred"]

    # baseline_results.npz was computed on the same split's test set, in the
    # same order (see baseline.py) -- align by true label sanity check.
    assert np.array_equal(baseline["test_true"], cnn["test_true"]), \
        "baseline.py and cnn.py must be run on the same split_indices.npz"
    df["baseline_pred"] = baseline["test_pred"]

    report = []
    report.append("=" * 70)
    report.append("GALAXY10 DECaLS -- CNN FAILURE MODE ANALYSIS")
    report.append("=" * 70)

    cnn_acc = accuracy_score(df["cnn_true"], df["cnn_pred"])
    base_acc = accuracy_score(df["cnn_true"], df["baseline_pred"])
    report.append(f"\nOverall CNN test accuracy:      {cnn_acc:.4f}")
    report.append(f"Overall baseline test accuracy: {base_acc:.4f}")
    report.append(f"CNN beats baseline: {cnn_acc > base_acc}")

    plot_confusion_matrix(df["cnn_true"], df["cnn_pred"], os.path.join(args.fig_dir, "confusion_matrix.png"))
    report.append(f"\nSaved confusion matrix -> {args.fig_dir}/confusion_matrix.png")

    for col, title, fname in [
        ("size_px", "Accuracy by apparent galaxy SIZE (tercile bins)", "accuracy_by_size.png"),
        ("mean_brightness", "Accuracy by galaxy BRIGHTNESS (tercile bins)", "accuracy_by_brightness.png"),
        ("axis_ratio", "Accuracy by INCLINATION / axis ratio (low = edge-on, tercile bins)", "accuracy_by_inclination.png"),
    ]:
        res = plot_binned_accuracy(df, col, "cnn_true", "cnn_pred", "baseline_pred",
                                    os.path.join(args.fig_dir, fname), title)
        report.append(f"\n--- {title} ---")
        report.append(res.to_string(index=False))
        report.append(f"Saved -> {args.fig_dir}/{fname}")

    secretly_a_size_classifier_test(df, report)

    report_path = os.path.join(args.fig_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(str(x) for x in report))
    print("\n".join(str(x) for x in report))
    print(f"\nFull report saved -> {report_path}")


if __name__ == "__main__":
    main()
