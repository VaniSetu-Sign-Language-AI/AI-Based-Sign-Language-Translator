"""Dataset builder: frame-level CSV → normalised sliding-window sequences.

Why sequences? The hybrid CNN+LSTM+CNN consumes temporal context
  X: (n_samples, SEQUENCE_LENGTH, F), y: (n_samples,)
where F=126 for two-hand landmarks (h0_63 | h1_63) or 63 for legacy CSVs.
Single still images are handled at inference by tiling one frame × T.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import load_config
from src.logger import get_logger

log = get_logger("dataset")
app_cfg, _, train_cfg, _, _ = load_config()


def load_frame_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("CSV must contain a 'label' column.")
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df.dropna().reset_index(drop=True)
    log.info("Loaded %d frames, %d classes: %s", len(df), df.label.nunique(), sorted(df.label.unique()))
    return df


def make_sequences(df: pd.DataFrame, seq_len: int, stride: int):
    """Sliding window *within each contiguous label run* (avoids cross-class windows)."""
    feats = df.drop(columns=["label"]).to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy()
    Xs, ys = [], []
    i = 0
    while i < len(df):
        j = i
        while j < len(df) and labels[j] == labels[i]:
            j += 1
        run = feats[i:j]
        for s in range(0, len(run) - seq_len + 1, stride):
            Xs.append(run[s:s + seq_len])
            ys.append(labels[i])
        # A short run (< seq_len): tile last frame to fill — keeps rare classes usable
        if len(run) < seq_len and len(run) > 0:
            pad = np.tile(run[-1:], (seq_len - len(run), 1))
            Xs.append(np.vstack([run, pad]))
            ys.append(labels[i])
        i = j
    return np.array(Xs, dtype=np.float32), np.array(ys)


def build_datasets(csv_path: str | None = None, seq_len: int | None = None,
                   stride: int | None = None, save_artifacts: bool = True):
    csv_path = csv_path or app_cfg.raw_csv
    seq_len = seq_len or app_cfg.sequence_length
    stride = stride or app_cfg.stride

    df = load_frame_csv(csv_path)
    X, y_str = make_sequences(df, seq_len, stride)
    if len(X) == 0:
        raise ValueError("No sequences built — collect more frames (need ≥ sequence_length per class).")

    le = LabelEncoder().fit(y_str)
    y = le.transform(y_str)

    # Scale per-feature over (samples*timesteps, features) — fit on train only
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=app_cfg.test_size, random_state=app_cfg.random_state, stratify=y)
    scaler = StandardScaler()
    s = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, s[2])).reshape(s)
    X_test = scaler.transform(X_test.reshape(-1, s[2])).reshape(X_test.shape)

    log.info("Sequences: train=%s test=%s classes=%s", X_train.shape, X_test.shape, list(le.classes_))

    if save_artifacts:
        mdir = Path(train_cfg.model_dir)
        mdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(le, mdir / train_cfg.encoder_name)
        joblib.dump(scaler, mdir / train_cfg.scaler_name)
        log.info("Saved encoder + scaler → %s", mdir)
    return (X_train, X_test, y_train, y_test), le, scaler
