"""MediaPipe helpers — single place for all landmark logic."""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

N_HANDS = 2
N_LANDMARKS = 21
N_PER_HAND = N_LANDMARKS * 3  # x,y,z per landmark per hand
N_FEATURES = N_HANDS * N_PER_HAND  # 126-D: two hands concatenated

HEADER = ["label"] + [
    f"h{h}_{axis}{i}"
    for h in range(N_HANDS)
    for i in range(N_LANDMARKS)
    for axis in ("x", "y", "z")
]


def build_hands(min_det: float = 0.5, min_track: float = 0.5):
    return mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=N_HANDS,
        min_detection_confidence=min_det,
        min_tracking_confidence=min_track,
    )


def build_hands_static(min_det: float = 0.5):
    return mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=N_HANDS,
        min_detection_confidence=min_det,
    )


def extract_normalised(frame_bgr: np.ndarray, hands) -> np.ndarray | None:
    """Return 126-D normalised two-hand vector or None (no hand detected).

    Layout: [h0_63 | h1_63], left-to-right by wrist x. Missing hand = zeros.
    Single-hand gestures work as before (second half zeros).
    """
    vec, _ = extract_with_raw(frame_bgr, hands)
    return vec


def extract_with_raw(frame_bgr: np.ndarray, hands) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(normalised 126-vector, raw (2,21,3) image-space matrix) or (None, None).

    Two-hand behaviour:
      * Hands sorted left-to-right by wrist x → stable h0/h1 ordering.
      * Origin = wrist of h0 (leftmost). Scale = max distance from origin
        across all present landmarks. Preserves inter-hand offset while
        keeping single-hand case identical to the old 63-D logic.
      * Missing second hand → zeros in second half of vec and raw[1].
    Raw coordinates stay in MediaPipe image space (x/y in 0..1) so callers
    can map fingertips back onto the displayed video for cursor control.
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None, None
    raws = [
        np.array([[p.x, p.y, p.z] for p in h.landmark], dtype=np.float32)
        for h in res.multi_hand_landmarks
    ]
    # Stable ordering: left-to-right by wrist x.
    raws.sort(key=lambda r: float(r[0][0]))
    raws = raws[:N_HANDS]
    n_found = len(raws)

    origin = raws[0][0].copy()  # wrist of leftmost hand
    rel = [r - origin for r in raws]
    scale = float(max(float(np.max(np.linalg.norm(r, axis=1))) for r in rel)) + 1e-8

    normed: list[np.ndarray] = [(r / scale).astype(np.float32) for r in rel]
    while len(normed) < N_HANDS:
        normed.append(np.zeros((N_LANDMARKS, 3), dtype=np.float32))
    vec = np.concatenate([m.flatten() for m in normed]).astype(np.float32)

    padded = np.zeros((N_HANDS, N_LANDMARKS, 3), dtype=np.float32)
    for i, r in enumerate(raws):
        padded[i] = r
    # n_found available via non-zero check; callers needing exact count
    # can use count_hands(padded) below.
    return vec, padded


def count_hands(raw_2x21x3: np.ndarray | None) -> int:
    """Count present hands in a padded (2,21,3) raw matrix (or legacy (21,3))."""
    if raw_2x21x3 is None:
        return 0
    a = np.asarray(raw_2x21x3, dtype=float)
    if a.shape == (N_LANDMARKS, 3):
        return 1 if np.abs(a).sum() > 1e-9 else 0
    if a.ndim == 3 and a.shape[1:] == (N_LANDMARKS, 3):
        return int(sum(float(np.abs(a[i]).sum()) > 1e-9 for i in range(a.shape[0])))
    return 0


def extract_raw(frame_bgr: np.ndarray, hands) -> np.ndarray | None:
    """Return 126-D zero-padded raw vector (image space) or None."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if not res.multi_hand_landmarks:
        return None
    raws = [
        np.array([[p.x, p.y, p.z] for p in h.landmark], dtype=np.float32)
        for h in res.multi_hand_landmarks
    ]
    raws.sort(key=lambda r: float(r[0][0]))
    raws = raws[:N_HANDS]
    while len(raws) < N_HANDS:
        raws.append(np.zeros((N_LANDMARKS, 3), dtype=np.float32))
    return np.concatenate([r.flatten() for r in raws]).astype(np.float32)


def draw_landmarks(frame: np.ndarray, hands) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        for h in res.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
    return frame
