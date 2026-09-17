"""ROC curve plotting for image classification evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def plot_roc_curves(metrics: dict[str, Any], output_path: str | Path) -> Path | None:
    """Render ROC curve(s) from computed metrics and save as PNG.

    Binary tasks get a single curve; multiclass tasks get one curve per class (one-vs-rest)
    plus a micro-average curve. Returns None if metrics carry no curve data (e.g. labels-only
    predictions with no class probabilities).
    """
    curves = metrics.get("roc_curves")
    if not curves:
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    class_curves = {label: curve for label, curve in curves.items() if label != "micro"}
    auc_lookup = {row["label"]: row["roc_auc"] for row in metrics.get("per_class_roc_auc", [])}
    if len(class_curves) == 1 and not auc_lookup:
        only_label = next(iter(class_curves))
        auc_lookup[only_label] = metrics.get("roc_auc")

    fig, ax = plt.subplots(figsize=(7, 6))

    for label, curve in class_curves.items():
        auc = auc_lookup.get(label)
        label_text = f"{label} (AUC = {auc:.3f})" if auc is not None else label
        ax.plot(curve["fpr"], curve["tpr"], linewidth=1.5, label=label_text)

    micro_curve = curves.get("micro")
    if micro_curve is not None:
        micro_auc = metrics.get("roc_auc_micro")
        micro_label = f"micro-average (AUC = {micro_auc:.3f})" if micro_auc is not None else "micro-average"
        ax.plot(micro_curve["fpr"], micro_curve["tpr"], linestyle="--", linewidth=2.0, color="black", label=micro_label)

    ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1.0, label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve" + (" (One-vs-Rest)" if len(class_curves) > 1 else ""))
    ax.legend(loc="lower right", fontsize="small")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
