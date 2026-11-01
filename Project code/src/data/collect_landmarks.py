"""Data collection: images folder and/or webcam → frame-level landmarks CSV.

Enterprise behaviour:
  * --mode image   : walk data/raw_images/<LABEL>/*.jpg|png → 1 CSV row per image
  * --mode webcam  : press A-Z/0-9 to record frames, TAB for a free-typed
                     custom sign (e.g. HELLO), ESC to quit, overlay live
  * --label NAME   : record a custom sign straight from CLI, no picker
                     (any folder name also works as a custom label in image mode)
  * --mode both    : run image pass then webcam pass
  * Idempotent header handling, skips images with no hand detected (logged).
  * Two-hand support: up to 2 hands per frame (left-to-right h0/h1);
    single-hand frames zero-pad the missing hand.

CSV schema: label, h0_x0,h0_y0,h0_z0, ..., h1_x20,h1_y20,h1_z20 (126 features)
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
from tqdm import tqdm

from src.config import load_config
from src.logger import get_logger
from src.utils.mediapipe_utils import (
    HEADER,
    build_hands,
    build_hands_static,
    draw_landmarks,
    extract_normalised,
)

log = get_logger("collect")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

app_cfg, _, _, _, _ = load_config()


def _ensure_header(csv_path: Path) -> bool:
    """Return True if header was just written."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(HEADER)
        return True
    return False


def collect_from_images(image_dir: str, csv_path: str) -> dict:
    root = Path(image_dir)
    csv_p = Path(csv_path)
    _ensure_header(csv_p)
    hands = build_hands_static(app_cfg.min_detection_confidence)
    stats = {"written": 0, "skipped": 0, "labels": set()}

    if not root.exists():
        log.warning("Image dir %s missing — skipping image pass.", root)
        return stats

    imgs = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        log.warning("No images under %s. Expected <label>/*.jpg layout.", root)
        return stats

    with open(csv_p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for p in tqdm(imgs, desc="Extracting landmarks"):
            label = p.parent.name.strip().upper()
            img = cv2.imread(str(p))
            if img is None:
                stats["skipped"] += 1
                continue
            vec = extract_normalised(img, hands)
            if vec is None:
                stats["skipped"] += 1
                log.warning("No hand in %s — skipped", p)
                continue
            w.writerow([label, *vec.tolist()])
            stats["written"] += 1
            stats["labels"].add(label)
    hands.close()
    log.info("Image pass done: %s", {k: (len(v) if isinstance(v, set) else v) for k, v in stats.items()})
    return stats


def _valid_label(raw: str) -> str | None:
    """Normalise free-typed labels: letters/digits/spaces, upper-cased."""
    label = re.sub(r"\s+", " ", raw.strip().upper())
    if not label or len(label) > 30 or not re.fullmatch(r"[A-Z0-9 _-]+", label):
        return None
    return label


def _record_label(cap, hands, writer, f, label: str, frames: int,
                  mirror_fix: bool = False) -> int:
    """Record `frames` rows for one label. Returns rows written."""
    print(f"Recording {frames} frames for '{label}' ... (Q aborts)")
    written = 0
    for _ in tqdm(range(frames), desc=label):
        ok, fr = cap.read()
        if not ok:
            break
        if mirror_fix:
            # Correct driver-mirrored cameras so saved data is true-camera,
            # matching Flask inference with mirrorFix OFF.
            fr = cv2.flip(fr, 1)
        # Default (mirror_fix=False): true camera, matches Flask inference.
        vec = extract_normalised(fr, hands)
        if vec is not None:
            writer.writerow([label, *vec.tolist()])
            written += 1
            f.flush()
        cv2.imshow("Sign Data Collection", fr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    print(f"Done '{label}': {written}/{frames} frames kept.")
    return written


def collect_from_webcam(csv_path: str, frames_per_class: int = 120,
                        label: str | None = None, mirror_fix: bool = False) -> None:
    csv_p = Path(csv_path)
    _ensure_header(csv_p)
    hands = build_hands(app_cfg.min_detection_confidence, app_cfg.min_tracking_confidence)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    with open(csv_p, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)

        # ── Non-interactive: custom label straight from CLI ──
        if label:
            ok, _ = cap.read()
            if ok:
                _record_label(cap, hands, w, f, label, frames_per_class, mirror_fix)
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            return

        # ── Interactive: quick keys + free-typed custom labels ──
        print("\n[WEBCAM] A-Z/0-9 = record that sign | N = custom sign name | Q = quit")
        print("Two-hand supported: show 1 or 2 hands; missing hand is zero-padded.")
        print(f"Recording {frames_per_class} frames per pick.\n")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if mirror_fix:
                frame = cv2.flip(frame, 1)
            # Preview shows exactly what gets saved (mirror-corrected when enabled).
            vis = draw_landmarks(frame.copy(), hands)
            cv2.putText(vis, "A-Z/0-9: Record | TAB: Custom (1-2 hands) | ESC: Quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Sign Data Collection", vis)
            key = cv2.waitKey(1) & 0xFF
            # 27 = ESC key, 'q' handled below if ESC isn't used
            if key == 27:  # ESC to quit
                break
            if key == 9:   # TAB key for custom multi-letter sign name (e.g. HELLO, ALERT)
                raw = input("Custom sign label (A-Z, 0-9, spaces), empty cancels: ")
                custom = _valid_label(raw)
                if custom is None:
                    print("Invalid/empty — cancelled." if raw.strip() else "Cancelled.")
                    continue
                _record_label(cap, hands, w, f, custom, frames_per_class, mirror_fix)
            elif key != 255:
                ch = chr(key).upper()
                if ch.isalnum():
                    _record_label(cap, hands, w, f, ch, frames_per_class, mirror_fix)
    cap.release()
    cv2.destroyAllWindows()
    hands.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect hand landmarks to CSV")
    ap.add_argument("--mode", choices=["image", "webcam", "both"], default="both")
    ap.add_argument("--image-dir", default=app_cfg.image_dir)
    ap.add_argument("--csv", default=app_cfg.raw_csv)
    ap.add_argument("--frames-per-class", type=int, default=120)
    ap.add_argument("--label", default=None,
                    help="custom sign name to record directly (e.g. --label 'HELLO'). "
                         "Skips the interactive picker; any A-Z/0-9/spaces name allowed.")
    ap.add_argument("--mirror-fix", action="store_true",
                    help="correct a driver-mirrored webcam (flip frames) so saved data "
                         "is true-camera, matching Flask inference with mirrorFix OFF.")
    a = ap.parse_args()
    if a.label is not None:
        custom = _valid_label(a.label)
        if custom is None:
            raise SystemExit("Invalid --label. Use letters, digits, spaces (max 30 chars).")
        a.label = custom
    if a.mode in ("image", "both"):
        collect_from_images(a.image_dir, a.csv)
    if a.mode in ("webcam", "both"):
        collect_from_webcam(a.csv, a.frames_per_class, a.label, a.mirror_fix)


if __name__ == "__main__":
    main()
