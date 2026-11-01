"""CustomTkinter training UI — one-click train with live progress.

Run:  python -m src.training.train_ui   (or scripts/train_ui.bat)

  * Epochs slider, batch-size dropdown, RF-only checkbox.
  * Dataset check (rows / classes) before training.
  * Live epoch progress bar + log box; Stop button aborts gracefully.
  * Shows final backbone/RF accuracy + artifact paths on completion.

Threading: Tk mainloop stays responsive; Keras fit runs in a worker thread.
Uses src.training.train(train(..., extra_callbacks=[UICallback])).
"""
from __future__ import annotations

import csv
import queue
import sys
import threading
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import customtkinter as ctk
from tensorflow.keras.callbacks import Callback

from src.config import load_config

app_cfg, _, train_cfg, _, _ = load_config()

BRAND = "#F94C00"
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class UICallback(Callback):
    """Push epoch updates to the UI queue; honour stop_event."""

    def __init__(self, q: queue.Queue, total_epochs: int, stop_event: threading.Event):
        super().__init__()
        self._q = q
        self._total = total_epochs
        self._stop = stop_event

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self._q.put(("progress", (epoch + 1, self._total)))
        self._q.put(("log", f"epoch {epoch + 1}/{self._total} — "
                            f"loss={logs.get('loss', 0):.4f} acc={logs.get('accuracy', 0):.4f} "
                            f"val_loss={logs.get('val_loss', 0):.4f} "
                            f"val_acc={logs.get('val_accuracy', 0):.4f}"))
        if self._stop.is_set():
            self.model.stop_training = True
            self._q.put(("log", "[train] stop requested — finishing epoch…"))


class TrainUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Sign Training — CNN+LSTM + RF")
        self.geometry("760x620")

        self.epochs_var = ctk.IntVar(value=train_cfg.epochs)
        self.batch_var = ctk.StringVar(value=str(train_cfg.batch_size))
        self.rf_only_var = ctk.BooleanVar(value=False)
        self.status_var = ctk.StringVar(value="Idle — check dataset, then Start.")
        self.data_var = ctk.StringVar(value="Dataset: —")
        self.model_var = ctk.StringVar(value="Models: —")

        self._busy = False
        self._stop = threading.Event()
        self._q: queue.Queue = queue.Queue()

        self._build()
        self.refresh_status()
        self.after(150, self._pump)

    # ── layout ──
    def _build(self) -> None:
        root = ctk.CTkFrame(self)
        root.pack(fill="both", expand=True, padx=12, pady=12)

        ctk.CTkLabel(root, text="Training — hybrid CNN+LSTM backbone + RF head",
                     font=("Segoe UI", 15, "bold")).pack(anchor="w", padx=12, pady=(10, 2))
        ctk.CTkLabel(root, textvariable=self.data_var, wraplength=700,
                     font=("Segoe UI", 12)).pack(anchor="w", padx=12)
        ctk.CTkLabel(root, textvariable=self.model_var, wraplength=700,
                     font=("Segoe UI", 12)).pack(anchor="w", padx=12, pady=(0, 6))

        ctrl = ctk.CTkFrame(root, fg_color="transparent")
        ctrl.pack(fill="x", padx=12, pady=4)
        ctrl.grid_columnconfigure((0, 1), weight=1)

        left = ctk.CTkFrame(ctrl)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ctk.CTkLabel(left, text="Epochs", font=("Segoe UI", 13, "bold")).pack(
            anchor="w", padx=12, pady=(10, 0))
        row = ctk.CTkFrame(left, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=4)
        self.epoch_slider = ctk.CTkSlider(row, from_=5, to=200, number_of_steps=39,
                                          command=lambda v: self.epochs_var.set(int(float(v))))
        self.epoch_slider.set(self.epochs_var.get())
        self.epoch_slider.pack(side="left", fill="x", expand=True)
        self.epoch_lbl = ctk.CTkLabel(row, text=str(self.epochs_var.get()), width=44)
        self.epoch_lbl.pack(side="left", padx=8)
        self.epochs_var.trace_add("write",
                                  lambda *_: self.epoch_lbl.configure(text=str(self.epochs_var.get())))

        ctk.CTkLabel(left, text="Batch size", font=("Segoe UI", 13, "bold")).pack(
            anchor="w", padx=12, pady=(6, 0))
        ctk.CTkOptionMenu(left, variable=self.batch_var,
                          values=["16", "32", "64", "128"]).pack(
            anchor="w", padx=12, pady=4)
        self.rf_check = ctk.CTkCheckBox(left, text="RF-only (reuse backbone, fast)",
                                        variable=self.rf_only_var)
        self.rf_check.pack(anchor="w", padx=12, pady=8)

        right = ctk.CTkFrame(ctrl)
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self.start_btn = ctk.CTkButton(right, text="▶  Start training",
                                       fg_color=BRAND, hover_color="#D63E00",
                                       height=44, font=("Segoe UI", 14, "bold"),
                                       command=self.on_start)
        self.start_btn.pack(fill="x", padx=12, pady=(12, 6))
        self.stop_btn = ctk.CTkButton(right, text="■  Stop", height=34,
                                      fg_color="#444444", hover_color="#666666",
                                      command=self.on_stop)
        self.stop_btn.pack(fill="x", padx=12, pady=2)
        ctk.CTkButton(right, text="Refresh status", height=30,
                      command=self.refresh_status).pack(fill="x", padx=12, pady=(6, 12))

        ctk.CTkLabel(root, textvariable=self.status_var, wraplength=700,
                     font=("Segoe UI", 12)).pack(anchor="w", padx=12, pady=(6, 0))
        self.progress = ctk.CTkProgressBar(root, progress_color=BRAND)
        self.progress.pack(fill="x", padx=12, pady=6)
        self.progress.set(0)

        ctk.CTkLabel(root, text="Log", font=("Segoe UI", 13, "bold")).pack(
            anchor="w", padx=12)
        self.logbox = ctk.CTkTextbox(root, height=240, font=("Consolas", 11))
        self.logbox.pack(fill="both", expand=True, padx=12, pady=(2, 12))
        self.logbox.configure(state="disabled")

    # ── helpers ──
    def log(self, msg: str) -> None:
        self.logbox.configure(state="normal")
        self.logbox.insert("end", msg + "\n")
        self.logbox.see("end")
        self.logbox.configure(state="disabled")

    def refresh_status(self) -> None:
        p = Path(app_cfg.raw_csv)
        if not p.exists() or p.stat().st_size == 0:
            self.data_var.set("Dataset: empty — collect data first "
                              "(python -m src.data.collect_ui).")
        else:
            try:
                with open(p, newline="", encoding="utf-8") as f:
                    rows = list(csv.reader(f))
                ncols = len(rows[0]) - 1 if rows else 0
                c = Counter(r[0] for r in rows[1:] if r)
                self.data_var.set(
                    f"Dataset: {sum(c.values())} rows, {len(c)} labels, "
                    f"{ncols}-D features ({'two-hand 126' if ncols == 126 else 'legacy 63' if ncols == 63 else 'unexpected'}).")
            except Exception as e:
                self.data_var.set(f"Dataset: unreadable ({e})")
        mdir = Path(train_cfg.model_dir)
        arts = [train_cfg.model_name, train_cfg.rf_name,
                train_cfg.encoder_name, train_cfg.scaler_name, "metrics.json"]
        have = [a for a in arts if (mdir / a).exists()]
        missing = [a for a in arts if a not in have]
        self.model_var.set(f"Models: have [{', '.join(have)}]" if have else "Models: none yet."
                           + (f"  missing [{', '.join(missing)}]" if missing else ""))

    def _pump(self) -> None:
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "log":
                    self.log(payload)
                elif kind == "status":
                    self.status_var.set(payload)
                elif kind == "progress":
                    done, total = payload
                    self.progress.set(done / max(total, 1))
                    self.status_var.set(f"Training… epoch {done}/{total}")
                elif kind == "done":
                    self._busy = False
                    self.progress.set(1.0)
                    self.status_var.set(payload if isinstance(payload, str) else "Done.")
                    self.refresh_status()
        except queue.Empty:
            pass
        self.after(150, self._pump)

    # ── actions ──
    def on_start(self) -> None:
        if self._busy:
            self.status_var.set("Busy — Stop first.")
            return
        self._busy = True
        self._stop.clear()
        self.progress.set(0)
        epochs = int(self.epochs_var.get())
        batch = int(self.batch_var.get())
        rf_only = bool(self.rf_only_var.get())
        threading.Thread(target=self._worker, args=(epochs, batch, rf_only),
                         daemon=True).start()

    def on_stop(self) -> None:
        self._stop.set()
        self._q.put(("log", "[UI] stop requested…"))

    def _worker(self, epochs: int, batch: int, rf_only: bool) -> None:
        from src.training.train import train as run_train

        self._q.put(("status", f"Training: epochs={epochs} batch={batch} rf_only={rf_only}…"))
        self._q.put(("log", f"[train] start — epochs={epochs} batch={batch} rf_only={rf_only}"))
        try:
            cb = UICallback(self._q, epochs, self._stop)
            res = run_train(epochs, batch, rf_only, extra_callbacks=[cb])
            self._q.put(("log", f"[train] backbone_acc={res.get('backbone_accuracy')} "
                                f"rf_acc={res.get('rf_accuracy')} "
                                f"classes={res.get('classes')}"))
            self._q.put(("done", f"Done — RF acc {res.get('rf_accuracy', 0):.4f}, "
                                 f"backbone {res.get('backbone_accuracy', 0):.4f}."))
        except Exception as e:
            self._q.put(("log", f"[train] FAILED: {e}"))
            self._q.put(("done", f"Training failed: {e}"))


def main() -> None:
    TrainUI().mainloop()


if __name__ == "__main__":
    main()
