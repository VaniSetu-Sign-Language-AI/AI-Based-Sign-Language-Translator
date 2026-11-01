"""Thread-safe inference: sequence → embedding → RandomForest → (label, confidence).

Pipeline:  (T,F) landmarks → scaler → CNN+LSTM embedder → RF predict_proba.
* F=126 two-hand (h0_63 | h1_63, missing hand zero-padded); legacy F=63
  single-hand models still load (vectors auto-adapted).
* Single still image → tiled ×T so the sequence backbone works as-is.
* Video/sequence   → last T frames used directly.
* Majority-vote smoothing over recent predictions stabilises realtime output.
* Always returns the top-1 class — no unknown-sign rejection.
* Legacy fallback: a pre-RF `hybrid_cnn_lstm_cnn.keras` softmax model is used
  directly if the new backbone+RF artifacts are not present yet.
"""
from __future__ import annotations

import threading
from collections import Counter, deque
from pathlib import Path

import joblib
import numpy as np
from tensorflow.keras.models import load_model

from src.config import ROOT, load_config
from src.logger import get_logger
from src.models.hybrid_model import build_embedder

log = get_logger("infer")
app_cfg, _, train_cfg, infer_cfg, _ = load_config()

LEGACY_MODEL = "hybrid_cnn_lstm_cnn.keras"


def _load_backbone_compat(path: Path):
    """Load a backbone saved by older TF/Keras.

    Current env is TF 2.21 + Keras 3, which expects `.keras` = zip.
    Existing artifact is HDF5 content misnamed `.keras` + LSTM config
    containing removed `time_major` arg. Handle both transparently.
    """
    from tensorflow.keras.models import load_model as _load

    try:
        return _load(path)
    except ValueError as e:
        msg = str(e)
        needs_h5_fallback = ("File not found" in msg and path.exists()) \
            or ("time_major" in msg) or ("not a zip file" in msg.lower())
        if not needs_h5_fallback:
            raise
    # Fallback: copy HDF5 bytes to a .h5 temp file and strip time_major.
    import shutil
    import tempfile

    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != b"\x89HDF":
        raise ValueError(
            f"File not found: filepath={path}. Please ensure the file "
            f"is an accessible `.keras` zip file.")
    tmp = Path(tempfile.gettempdir()) / f"bb_compat_{path.stem}.h5"
    shutil.copyfile(path, tmp)
    try:
        from keras.src.layers.rnn.lstm import LSTM as _LSTM
        _orig = _LSTM.from_config.__func__

        def _stripped(cls, config: dict):
            if isinstance(config, dict):
                config = {k: v for k, v in config.items() if k != "time_major"}
            return _orig(cls, config)

        _LSTM.from_config = classmethod(_stripped)
        try:
            return _load(tmp, compile=False)
        finally:
            _LSTM.from_config = classmethod(_orig)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass


class SignPredictor:
    _lock = threading.Lock()

    def __init__(self, model_path: str | None = None):
        mdir = Path(train_cfg.model_dir)
        if not mdir.is_absolute():
            mdir = ROOT / mdir
        self.le = joblib.load(mdir / train_cfg.encoder_name)
        self.scaler = joblib.load(mdir / train_cfg.scaler_name)
        self.T = infer_cfg.sequence_length
        # Feature dim from fitted scaler (126 two-hand, 63 legacy single-hand).
        try:
            self.F = int(getattr(self.scaler, "n_features_in_", 126))
        except Exception:
            self.F = 126
        self.history: deque[str] = deque(maxlen=infer_cfg.smoothing_window)

        backbone_p = Path(model_path or mdir / train_cfg.model_name)
        rf_p = mdir / train_cfg.rf_name
        legacy_p = mdir / LEGACY_MODEL

        if backbone_p.exists() and rf_p.exists():
            self.mode = "cnn_lstm_rf"
            backbone = _load_backbone_compat(backbone_p)
            self.embedder = build_embedder(backbone)
            self.rf = joblib.load(rf_p)
            self.model_path = backbone_p
            log.info("Predictor ready [CNN+LSTM->RF]: %s + %s | classes=%s",
                     backbone_p, rf_p, list(self.le.classes_))
        elif legacy_p.exists():
            self.mode = "legacy_softmax"
            self.model = _load_backbone_compat(legacy_p)
            self.model_path = legacy_p
            log.info("Predictor ready [legacy softmax]: %s | classes=%s. "
                     "Re-train to switch to CNN+LSTM+RF.", legacy_p, list(self.le.classes_))
        else:
            raise FileNotFoundError(
                f"No model found in {mdir}. Train first: python -m src.training.train")

    def _adapt(self, vec: np.ndarray) -> np.ndarray:
        """Adapt 63/126-D vectors to the model's feature dim."""
        v = np.asarray(vec, dtype=np.float32).flatten()
        if v.shape[0] == self.F:
            return v
        if v.shape[0] == 63 and self.F == 126:
            # Legacy single-hand → pad missing second hand with zeros.
            return np.concatenate([v, np.zeros(63, dtype=np.float32)])
        if v.shape[0] == 126 and self.F == 63:
            # Two-hand vector on legacy model → use primary hand (h0).
            return v[:63]
        raise ValueError(f"Expected {self.F}-D vector, got {v.shape[0]}-D")

    def _window(self, seq: np.ndarray) -> np.ndarray:
        seq = np.asarray(seq, dtype=np.float32)
        if seq.ndim != 2:
            raise ValueError(f"Expected (Ti,F), got {seq.shape}")
        if seq.shape[1] != self.F:
            # Adapt each frame (e.g. 63-D input on 126-D model).
            seq = np.stack([self._adapt(r) for r in seq]).astype(np.float32)
        if len(seq) >= self.T:
            return seq[-self.T:]
        if len(seq):
            return np.tile(seq[-1:], (self.T, 1))
        return np.zeros((self.T, self.F), np.float32)

    def _scale(self, seq: np.ndarray) -> np.ndarray:
        f = seq.shape[1]
        return self.scaler.transform(seq.reshape(-1, f)).reshape(1, self.T, f).astype(np.float32)

    def _probs(self, x: np.ndarray) -> np.ndarray:
        probs, _ = self._predict_with_embedding(x)
        return probs

    def _predict_with_embedding(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        with self._lock:
            if self.mode == "cnn_lstm_rf":
                emb = self.embedder.predict(x, verbose=0)
                return np.asarray(self.rf.predict_proba(emb)[0], dtype=float), np.asarray(emb[0], dtype=float)
            return np.asarray(self.model.predict(x, verbose=0)[0], dtype=float), None

    def predict_sequence(self, seq: np.ndarray) -> tuple[str, float, dict]:
        """seq: (Ti, F). Uses last T frames (pads by tiling if short).

        Always returns the top-1 class.
        """
        probs, _ = self._predict_with_embedding(self._scale(self._window(seq)))
        idx = int(np.argmax(probs))
        label = str(self.le.classes_[idx])
        conf = float(probs[idx])
        dist = {str(k): float(v) for k, v in zip(self.le.classes_, probs)}
        if conf >= infer_cfg.confidence_threshold:
            self.history.append(label)
        return label, conf, dist

    def predict_single(self, vec63: np.ndarray) -> tuple[str, float, dict]:
        v = self._adapt(vec63)
        seq = np.tile(v.reshape(1, self.F), (self.T, 1))
        return self.predict_sequence(seq)

    def smoothed(self) -> str | None:
        if not self.history:
            return None
        return Counter(self.history).most_common(1)[0][0]

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.le.classes_]
