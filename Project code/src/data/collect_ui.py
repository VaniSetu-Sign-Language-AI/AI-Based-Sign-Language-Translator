"""CustomTkinter data-collection UI — custom signs with typed label input.

Run:  python -m src.data.collect_ui

Features:
  * Label entry for custom signs (HELLO, THANK YOU, ...) with live validation.
  * Frames-per-class slider, CSV / image-dir entries.
  * [Record] captures 1-2 hands per frame via webcam (OpenCV preview window).
    Missing second hand is zero-padded to 126-D automatically.
  * Quick A-Z / 0-9 buttons for fingerspelling collection.
  * [Image folder pass] runs data/raw_images/<LABEL>/*.jpg through MediaPipe.
  * Live progress bar, status line, log box + dataset counts.

Threading: Tk mainloop stays responsive; capture runs in a worker thread.
Stop button or 'q' in the preview window aborts a recording.
"""
from __future__ import annotations

import csv
import queue
import sys
import threading
from collections import Counter
from pathlib import Path
from tkinter import messagebox

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import cv2
import customtkinter as ctk

from src.config import load_config
from src.data.collect_landmarks import (
    _ensure_header,
    _valid_label,
    collect_from_images,
)
from src.utils.mediapipe_utils import build_hands, extract_normalised

app_cfg, _, _, _, _ = load_config()

BRAND = "#F94C00"
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class CollectUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sign Collection — Custom Signs (1-2 hands)")
        self.geometry("860x780")

        self.csv_path = ctk.StringVar(value=app_cfg.raw_csv)
        self.image_dir = ctk.StringVar(value=app_cfg.image_dir)
        self.label_var = ctk.StringVar(value="")
        self.frames_var = ctk.IntVar(value=120)
        self.mirror_var = ctk.BooleanVar(value=False)
        self.status_var = ctk.StringVar(value="Idle — type a label, then Record.")
        self.counts_var = ctk.StringVar(value="Dataset: —")

        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._busy = False
        self._ui_queue: queue.Queue = queue.Queue()

        self._build()
        self.refresh_counts()
        self.after(120, self._pump_queue)

    # ── layout ──
    def _build(self) -> None:
        root = ctk.CTkFrame(self)
        root.pack(fill="both", expand=True, padx=12, pady=12)
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        # Left: controls
        left = ctk.CTkFrame(root)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)

        ctk.CTkLabel(left, text="Custom sign label",
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
        self.label_entry = ctk.CTkEntry(left, textvariable=self.label_var,
                                        placeholder_text="e.g. HELLO  (A-Z, 0-9, spaces)",
                                        height=38, font=("Segoe UI", 14))
        self.label_entry.pack(fill="x", padx=12, pady=4)
        self.label_entry.bind("<KeyRelease>", lambda _e: self._show_validation())
        self.valid_lbl = ctk.CTkLabel(left, text="", font=("Segoe UI", 12))
        self.valid_lbl.pack(anchor="w", padx=12)

        ctk.CTkLabel(left, text="Frames per sign", font=("Segoe UI", 13, "bold")).pack(
            anchor="w", padx=12, pady=(10, 0))
        slider_row = ctk.CTkFrame(left, fg_color="transparent")
        slider_row.pack(fill="x", padx=12, pady=4)
        self.slider = ctk.CTkSlider(slider_row, from_=30, to=300, number_of_steps=27,
                                    command=lambda v: self.frames_var.set(int(float(v))))
        self.slider.set(120)
        self.slider.pack(side="left", fill="x", expand=True)
        self.frames_lbl = ctk.CTkLabel(slider_row, text="120", width=44)
        self.frames_lbl.pack(side="left", padx=8)
        self.frames_var.trace_add("write",
                                  lambda *_: self.frames_lbl.configure(text=str(self.frames_var.get())))

        self.record_btn = ctk.CTkButton(left, text="●  Record custom sign (webcam)",
                                        fg_color=BRAND, hover_color="#D63E00",
                                        height=42, font=("Segoe UI", 14, "bold"),
                                        command=self.on_record_custom)
        self.record_btn.pack(fill="x", padx=12, pady=(10, 6))
        self.stop_btn = ctk.CTkButton(left, text="■  Stop", height=34,
                                      fg_color="#444444", hover_color="#666666",
                                      command=self.on_stop)
        self.stop_btn.pack(fill="x", padx=12, pady=2)
        self.img_btn = ctk.CTkButton(left, text="Run image-folder pass",
                                     height=34, command=self.on_image_pass)
        self.img_btn.pack(fill="x", padx=12, pady=2)
        self.mirror_check = ctk.CTkCheckBox(
            left, text="Camera looks mirrored — correct it",
            variable=self.mirror_var)
        self.mirror_check.pack(anchor="w", padx=12, pady=(2, 4))

        ctk.CTkLabel(left, text="CSV", font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=(8, 0))
        ctk.CTkEntry(left, textvariable=self.csv_path, height=28).pack(fill="x", padx=12)
        ctk.CTkLabel(left, text="Image dir", font=("Segoe UI", 11)).pack(anchor="w", padx=12, pady=(4, 0))
        ctk.CTkEntry(left, textvariable=self.image_dir, height=28).pack(fill="x", padx=12, pady=(0, 10))

        ctk.CTkLabel(left, text="Quick A-Z / 0-9", font=("Segoe UI", 13, "bold")).pack(
            anchor="w", padx=12, pady=(4, 2))
        grid = ctk.CTkScrollableFrame(left, height=150)
        grid.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        chars = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [str(d) for d in range(10)]
        for i, ch in enumerate(chars):
            b = ctk.CTkButton(grid, text=ch, width=44, height=30,
                              command=lambda c=ch: self.on_record_single(c))
            b.grid(row=i // 6, column=i % 6, padx=3, pady=3)

        # Right: progress + log
        right = ctk.CTkFrame(root)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        ctk.CTkLabel(right, text="Status", font=("Segoe UI", 13, "bold")).pack(
            anchor="w", padx=12, pady=(12, 0))
        ctk.CTkLabel(right, textvariable=self.status_var, wraplength=360,
                     font=("Segoe UI", 12)).pack(anchor="w", padx=12, pady=2)
        self.progress = ctk.CTkProgressBar(right, progress_color=BRAND)
        self.progress.pack(fill="x", padx=12, pady=6)
        self.progress.set(0)
        ctk.CTkLabel(right, textvariable=self.counts_var, wraplength=360,
                     font=("Segoe UI", 12)).pack(anchor="w", padx=12, pady=2)
        ctk.CTkButton(right, text="Refresh counts", height=30,
                      command=self.refresh_counts).pack(anchor="w", padx=12, pady=4)
        ctk.CTkLabel(right, text="Collected signs — ✕ deletes one sign's rows",
                     font=("Segoe UI", 13, "bold")).pack(anchor="w", padx=12, pady=(6, 0))
        self.signs_frame = ctk.CTkScrollableFrame(right, height=110)
        self.signs_frame.pack(fill="x", padx=12, pady=4)
        ctk.CTkLabel(right, text="Log", font=("Segoe UI", 13, "bold")).pack(
            anchor="w", padx=12, pady=(8, 0))
        self.logbox = ctk.CTkTextbox(right, height=190, font=("Consolas", 11))
        self.logbox.pack(fill="both", expand=True, padx=12, pady=(2, 12))
        self.logbox.configure(state="disabled")

    # ── helpers ──
    def log(self, msg: str) -> None:
        self.logbox.configure(state="normal")
        self.logbox.insert("end", msg + "\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def _show_validation(self) -> str | None:
        custom = _valid_label(self.label_var.get())
        if custom:
            self.valid_lbl.configure(text=f"✓  '{custom}' — ready", text_color="#4ADE80")
        else:
            self.valid_lbl.configure(
                text="Use letters/digits/spaces, max 30 chars" if self.label_var.get().strip()
                else "Type the sign name you are about to perform",
                text_color="#FBBF24")
        return custom

    def refresh_counts(self) -> None:
        p = Path(self.csv_path.get())
        if not p.exists() or p.stat().st_size == 0:
            self.counts_var.set("Dataset: empty — collect your first sign.")
            self._counts: Counter = Counter()
            self._rebuild_sign_list()
            return
        try:
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if len(rows) < 2:
                self.counts_var.set("Dataset: header only — no rows yet.")
                self._counts = Counter()
                self._rebuild_sign_list()
                return
            c = Counter(r[0] for r in rows[1:] if r)
            self._counts = c
            total = sum(c.values())
            top = ", ".join(f"{k}:{v}" for k, v in sorted(c.items())[:12])
            more = f" +{len(c) - 12} more" if len(c) > 12 else ""
            self.counts_var.set(f"Dataset: {total} rows, {len(c)} labels — {top}{more}")
            self._rebuild_sign_list()
        except Exception as e:
            self.counts_var.set(f"Dataset: unreadable ({e})")

    def _rebuild_sign_list(self) -> None:
        for w in self.signs_frame.winfo_children():
            w.destroy()
        counts = getattr(self, "_counts", Counter())
        if not counts:
            ctk.CTkLabel(self.signs_frame, text="No signs collected yet.",
                         font=("Segoe UI", 11)).pack(padx=8, pady=6)
            return
        for lbl, n in sorted(counts.items()):
            row = ctk.CTkFrame(self.signs_frame, fg_color="transparent")
            row.pack(fill="x", padx=4, pady=1)
            ctk.CTkLabel(row, text=lbl, font=("Segoe UI", 12, "bold"),
                         width=170, anchor="w").pack(side="left", padx=(4, 2))
            ctk.CTkLabel(row, text=f"{n} rows",
                         font=("Segoe UI", 11), width=70).pack(side="left")
            ctk.CTkButton(row, text="✕", width=32, height=24, fg_color="#7F1D1D",
                          hover_color="#B91C1C", font=("Segoe UI", 12, "bold"),
                          command=lambda l=lbl: self.on_delete_label(l)).pack(
                side="right", padx=4)

    def on_delete_label(self, label: str) -> None:
        """Delete every CSV row for one sign (keeps header). Retrain afterwards."""
        if self._busy:
            self.status_var.set("Busy — Stop recording first, then delete.")
            return
        n = getattr(self, "_counts", Counter()).get(label, 0)
        if not messagebox.askyesno(
                "Delete sign data",
                f"Delete all {n} rows for '{label}'?\n"
                "This cannot be undone. Retrain afterwards."):
            return
        p = Path(self.csv_path.get())
        try:
            with open(p, newline="", encoding="utf-8") as f:
                rows = list(csv.reader(f))
            if not rows:
                raise ValueError("CSV is empty.")
            kept = [r for r in rows[1:] if r and r[0] != label]
            removed = len(rows) - 1 - len(kept)
            with open(p, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(rows[0])
                w.writerows(kept)
            self.log(f"[delete] '{label}': removed {removed} rows.")
            self.status_var.set(f"Deleted '{label}' ({removed} rows). Retrain to update the model.")
        except Exception as e:
            self.log(f"[delete] FAILED: {e}")
            self.status_var.set(f"Delete failed: {e}")
        self.refresh_counts()

    def _pump_queue(self) -> None:
        try:
            while True:
                kind, payload = self._ui_queue.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "progress":
                    done, total = payload
                    self.progress.set(done / max(total, 1))
                    self.status_var.set(f"Recording… {done}/{total} frames")
                elif kind == "done":
                    self._busy = False
                    self.progress.set(1.0 if payload else 0.0)
                    self.status_var.set(payload if isinstance(payload, str) else "Done.")
                    self.refresh_counts()
        except queue.Empty:
            pass
        self.after(120, self._pump_queue)

    def _q(self, kind: str, payload) -> None:
        self._ui_queue.put((kind, payload))

    # ── actions ──
    def on_record_custom(self) -> None:
        custom = self._show_validation()
        if custom is None:
            self.status_var.set("Invalid label — fix it first.")
            return
        self._start_record(custom, int(self.frames_var.get()))

    def on_record_single(self, ch: str) -> None:
        self.label_var.set(ch)
        self._show_validation()
        self._start_record(ch, int(self.frames_var.get()))

    def on_stop(self) -> None:
        self._stop.set()
        self._q("log", "[UI] stop requested… (or press Q in preview)")

    def on_image_pass(self) -> None:
        if self._busy:
            self.status_var.set("Busy — wait for current job.")
            return
        self._busy = True
        img_dir, csv_path = self.image_dir.get(), self.csv_path.get()

        def job() -> None:
            self._q("status", f"Image pass: {img_dir} …")
            try:
                stats = collect_from_images(img_dir, csv_path)
                self._q("log", f"[image] written={stats.get('written', 0)} "
                               f"skipped={stats.get('skipped', 0)} "
                               f"labels={sorted(stats.get('labels', set()))}")
                self._q("done", "Image pass done.")
            except Exception as e:
                self._q("log", f"[image] FAILED: {e}")
                self._q("done", f"Image pass failed: {e}")

        threading.Thread(target=job, daemon=True).start()

    def _start_record(self, label: str, frames: int) -> None:
        if self._busy:
            self.status_var.set("Busy — Stop first, then record again.")
            return
        self._busy = True
        self._stop.clear()
        self.progress.set(0)
        csv_path = self.csv_path.get()
        mirror_fix = bool(self.mirror_var.get())
        t = threading.Thread(target=self._record_worker,
                             args=(label, frames, csv_path, mirror_fix), daemon=True)
        self._worker = t
        t.start()

    # ── worker (background thread; all cv2 calls live here) ──
    def _record_worker(self, label: str, frames: int, csv_path: str,
                       mirror_fix: bool = False) -> None:
        self._q("log", f"[webcam] '{label}' — {frames} frames, show 1-2 hands. "
                       f"mirror_fix={mirror_fix}. Q / Stop aborts.")
        self._q("status", f"Recording '{label}'…")
        try:
            _ensure_header(Path(csv_path))
            hands = build_hands(app_cfg.min_detection_confidence,
                                app_cfg.min_tracking_confidence)
        except Exception as e:
            self._q("log", f"[webcam] init FAILED: {e}")
            self._q("done", f"Init failed: {e}")
            return
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self._q("log", "[webcam] Could not open webcam (index 0).")
            self._q("done", "No webcam found.")
            hands.close()
            return
        kept = 0
        try:
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                for i in range(frames):
                    if self._stop.is_set():
                        self._q("log", "[webcam] stopped by user.")
                        break
                    ok, fr = cap.read()
                    if not ok:
                        self._q("log", "[webcam] frame grab failed.")
                        break
                    if mirror_fix:
                        # Correct driver-mirrored cameras; preview shows saved bytes.
                        fr = cv2.flip(fr, 1)
                    # Default: true camera, matches Flask inference with mirrorFix OFF.
                    vec = extract_normalised(fr, hands)
                    if vec is not None:
                        w.writerow([label, *vec.tolist()])
                        kept += 1
                        if kept % 10 == 0:
                            f.flush()
                    cv2.putText(fr, f"{label}  {i + 1}/{frames}  kept:{kept}  (Q=abort)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                    cv2.imshow("Sign Data Collection (UI) — Q aborts", fr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._q("log", "[webcam] aborted with Q.")
                        break
                    self._q("progress", (i + 1, frames))
                f.flush()
        except Exception as e:
            self._q("log", f"[webcam] FAILED: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
        self._q("log", f"[webcam] Done '{label}': {kept}/{frames} frames kept.")
        self._q("done", f"'{label}': {kept}/{frames} kept.")


def main() -> None:
    ui = CollectUI()
    ui.mainloop()


if __name__ == "__main__":
    main()
