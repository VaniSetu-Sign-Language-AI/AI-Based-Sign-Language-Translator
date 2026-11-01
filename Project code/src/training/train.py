"""Two-stage training: (1) CNN+LSTM backbone pre-training → (2) RandomForest head.

Stage 1 — backbone.fit with softmax head (EarlyStopping / ReduceLROnPlateau /
           ModelCheckpoint), curves saved to models/training_curves.png.
Stage 2 — freeze backbone, extract embeddings, fit RandomForestClassifier,
           save models/randomforest.pkl, print RF report + confusion matrix.

Re-running training after adding new signs re-fits both stages end-to-end.
`--rf-only` skips Stage 1 and re-fits just the RF head on the saved backbone
(fast path when only the label set changed).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model

from src.config import load_config
from src.data.dataset import build_datasets
from src.logger import get_logger
from src.models.hybrid_model import build_embedder, build_hybrid

log = get_logger("train")
app_cfg, _, train_cfg, _, raw = load_config()


def _callbacks(ckpt: Path):
    return [
        EarlyStopping(monitor="val_loss", patience=train_cfg.early_stopping_patience,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=train_cfg.reduce_lr_patience, verbose=1),
        ModelCheckpoint(str(ckpt), monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]


def _save_curves(hist, mdir: Path) -> None:
    """Bold on-palette curves + raw history.json (re-plottable without retraining)."""
    ORANGE, DARK, GRID = "#F94C00", "#6B3200", "#F3B27A"
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), facecolor="#FFFAFA")
    ax[0].plot(hist.history["accuracy"], label="train", color=ORANGE, linewidth=2.8)
    ax[0].plot(hist.history["val_accuracy"], label="val", color=DARK,
               linewidth=2.8, linestyle="--")
    ax[0].set_title("Backbone accuracy", fontsize=12, fontweight="bold")
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("accuracy")
    ax[0].legend(frameon=False)
    ax[0].grid(True, color=GRID, alpha=0.8)
    ax[1].plot(hist.history["loss"], label="train", color=ORANGE, linewidth=2.8)
    ax[1].plot(hist.history["val_loss"], label="val", color=DARK,
               linewidth=2.8, linestyle="--")
    ax[1].set_title("Backbone loss", fontsize=12, fontweight="bold")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("loss")
    ax[1].legend(frameon=False)
    ax[1].grid(True, color=GRID, alpha=0.8)
    for a in ax:
        a.set_facecolor("#FFFAFA")
        a.tick_params(labelsize=10)
    fig.tight_layout()
    fig.savefig(mdir / "training_curves.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    with open(mdir / "history.json", "w", encoding="utf-8") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in hist.history.items()}, f)


def _fit_rf(Etr: np.ndarray, ytr: np.ndarray):
    rf = RandomForestClassifier(
        n_estimators=train_cfg.rf_n_estimators,
        max_depth=train_cfg.rf_max_depth,
        n_jobs=train_cfg.rf_n_jobs,
        random_state=app_cfg.random_state,
        class_weight="balanced",
    )
    rf.fit(Etr, ytr)
    return rf


def _save_open_set(backbone, le, mdir: Path) -> Path:
    """Persist per-class embedding centroids + accept/reject distance limits.

    Unknown-hand rejection at serve time: a frame whose embedding lands
    farther than its predicted class limit is reported as "no known sign"
    instead of being forced into the nearest trained label (e.g. a plain
    palm must not come back as FOOD at 88%).

    Calibrated on single frames tiled ×T — exactly what inference embeds
    (every Flask request is one snapshot tiled to a sequence) — not on
    multi-frame motion windows, whose embeddings live elsewhere.
    Limit = 95th percentile of that class's train distances × open_set_margin.
    """
    from src.data.dataset import load_frame_csv

    out = mdir / train_cfg.open_set_name
    margin = float(train_cfg.open_set_margin or 1.25)
    T = int(app_cfg.sequence_length)
    df = load_frame_csv(app_cfg.raw_csv)
    Fmat = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    y = le.transform(df["label"].to_numpy())
    scaler = joblib.load(mdir / train_cfg.scaler_name)
    embedder = build_embedder(backbone)

    # Tile each frame ×T and embed in batches (matches predictor._window/_scale).
    E_parts: list[np.ndarray] = []
    for s in range(0, len(Fmat), 2048):
        B = Fmat[s:s + 2048]
        Xs = np.tile(B[:, None, :], (1, T, 1)).astype(np.float32)
        f = Xs.shape[2]
        Xs = scaler.transform(Xs.reshape(-1, f)).reshape(len(B), T, f).astype(np.float32)
        E_parts.append(np.asarray(embedder.predict(Xs, verbose=0), dtype=float))
    E = np.vstack(E_parts)

    labels = list(le.classes_)
    centroids: dict[str, list[float]] = {}
    thresholds: dict[str, float] = {}
    for i, lbl in enumerate(labels):
        Ei = E[y == i]
        if len(Ei) == 0:
            continue
        c = Ei.mean(axis=0)
        d = np.linalg.norm(Ei - c, axis=1)
        centroids[lbl] = [float(v) for v in c]
        thresholds[lbl] = float(np.percentile(d, 95)) * margin + 1e-6
    out.write_text(json.dumps({
        "embedding_dim": int(E.shape[1]),
        "margin": margin,
        "calibration": "single-frame-tiled",
        "centroids": centroids,
        "thresholds": thresholds,
    }, indent=2), encoding="utf-8")
    log.info("Saved open-set rejection to %s (%d classes)", out, len(centroids))
    return out


def _save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           labels: list[str], path: Path) -> None:
    """Annotated confusion-matrix heatmap (orange palette, publication-ready)."""
    cm = confusion_matrix(y_true, y_pred)
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(6.0, n * 0.8), max(5.0, n * 0.7)))
    im = ax.imshow(cm, cmap="Oranges")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n), labels, rotation=45, ha="right")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix — RandomForest head")
    thresh = cm.max() / 2 if cm.max() else 0.5
    for r in range(n):
        for c_ in range(n):
            ax.text(c_, r, str(cm[r, c_]), ha="center", va="center",
                    color="white" if cm[r, c_] > thresh else "black", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _write_metrics(path: Path, *, backbone_accuracy: float, rf_accuracy: float,
                   y_true: np.ndarray, y_pred: np.ndarray, labels: list[str],
                   test_samples: int, ckpt: Path, rf_path: Path) -> dict:
    """Persist the full performance matrix (accuracy + per-class P/R/F1)."""
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    per_class = {
        lbl: {k: round(float(report[lbl][k]), 4) for k in ("precision", "recall", "f1-score")}
        | {"support": int(report[lbl]["support"])}
        for lbl in labels
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "CNN+LSTM backbone + RandomForest head",
        "pipeline": ["cnn_lstm_backbone", "randomforest"],
        "backbone_accuracy": round(backbone_accuracy, 4),
        "rf_accuracy": round(rf_accuracy, 4),
        "test_samples": int(test_samples),
        "classes": labels,
        "per_class": per_class,
        "macro_avg": {k: round(float(report["macro avg"][k]), 4)
                      for k in ("precision", "recall", "f1-score")},
        "weighted_avg": {k: round(float(report["weighted avg"][k]), 4)
                         for k in ("precision", "recall", "f1-score")},
        "artifacts": {
            "backbone": ckpt.name,
            "randomforest": rf_path.name,
            "curves": "training_curves.png",
            "confusion": "confusion_matrix.png",
            "history": "history.json",
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def train(epochs: int | None = None, batch_size: int | None = None,
          rf_only: bool = False, extra_callbacks: list | None = None) -> dict:
    epochs = epochs or train_cfg.epochs
    batch_size = batch_size or train_cfg.batch_size

    (Xtr, Xte, ytr, yte), le, _ = build_datasets()
    n_classes = len(le.classes_)

    mdir = Path(train_cfg.model_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    ckpt = mdir / train_cfg.model_name
    rf_path = mdir / train_cfg.rf_name

    # ── Stage 1 : backbone ──
    if rf_only and ckpt.exists():
        log.info("--rf-only: reusing backbone %s", ckpt)
        backbone = load_model(ckpt)
        hist = None
    else:
        backbone = build_hybrid((Xtr.shape[1], Xtr.shape[2]), n_classes)
        backbone.summary(print_fn=log.info, line_length=120)
        cbs = _callbacks(ckpt) + list(extra_callbacks or [])
        hist = backbone.fit(Xtr, ytr, validation_data=(Xte, yte),
                            epochs=epochs, batch_size=batch_size,
                            callbacks=cbs, verbose=1)
        _save_curves(hist, mdir)
        backbone = load_model(ckpt)  # best checkpoint, not last epoch
    bb_pred = np.argmax(backbone.predict(Xte, verbose=0), axis=1)
    bb_acc = float((bb_pred == yte).mean())
    log.info("Stage-1 backbone test accuracy: %.4f", bb_acc)

    # ── Stage 2 : RandomForest on frozen embeddings ──
    embedder = build_embedder(backbone)
    log.info("Extracting %d-D embeddings …", embedder.output_shape[-1])
    Etr = embedder.predict(Xtr, verbose=0)
    Ete = embedder.predict(Xte, verbose=0)

    rf = _fit_rf(Etr, ytr)
    joblib.dump(rf, rf_path)

    pred = rf.predict(Ete)
    acc = float((pred == yte).mean())
    log.info("Stage-2 RandomForest test accuracy: %.4f", acc)
    print(classification_report(yte, pred, target_names=list(le.classes_)))
    log.info("Confusion matrix:\n%s", confusion_matrix(yte, pred))

    # ── Graphs + performance matrix for the Metrics page ──
    _save_confusion_matrix(yte, pred, list(le.classes_), mdir / "confusion_matrix.png")
    open_set_p = _save_open_set(backbone, le, mdir)
    metrics = _write_metrics(
        mdir / "metrics.json", backbone_accuracy=bb_acc, rf_accuracy=acc,
        y_true=yte, y_pred=pred, labels=list(le.classes_),
        test_samples=len(yte), ckpt=ckpt, rf_path=rf_path)
    log.info("Saved backbone → %s | RF head → %s | metrics → %s",
             ckpt, rf_path, mdir / "metrics.json")

    return {"rf_accuracy": acc, "backbone_accuracy": bb_acc,
            "backbone": str(ckpt), "rf": str(rf_path),
            "metrics": str(mdir / "metrics.json"),
            "open_set": str(open_set_p),
            "classes": list(le.classes_)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Train CNN+LSTM backbone + RandomForest head")
    ap.add_argument("--epochs", type=int, default=train_cfg.epochs)
    ap.add_argument("--batch-size", type=int, default=train_cfg.batch_size)
    ap.add_argument("--rf-only", action="store_true",
                    help="skip backbone training; re-fit only the RF head")
    a = ap.parse_args()
    train(a.epochs, a.batch_size, a.rf_only)


if __name__ == "__main__":
    main()
