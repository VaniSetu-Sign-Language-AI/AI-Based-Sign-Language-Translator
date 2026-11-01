"""Flask serving layer — enterprise structure (factory + blueprints-lite).

Endpoints:
  GET  /                      → UI (Sign→Text + Text→Sign)
  GET  /api/health            → service + model status
  GET  /api/classes           → trained label vocabulary
  POST /api/predict           → {image: base64 JPEG} → {label, confidence, smoothed, distribution}
  POST /api/text-to-sign      → {text: str} → [{char, sign_image, available}]  (per-character cards)
  GET  /api/sign-image/<CHAR> → reference sign image (user-supplied or generated placeholder)

Text→Sign asset convention: app/static/signs/<CHAR>.png  (A-Z, 0-9).
  Drop real ASL reference images there; missing ones auto-render as styled placeholders.
"""
from __future__ import annotations

import base64
import io
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

# Allow `python app/app.py` from project root (adds D:\mc0177 to sys.path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import joblib
import numpy as np
from flask import Flask, jsonify, request, render_template, send_file
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont

from src.config import load_config
from src.logger import get_logger
from src.utils.mediapipe_utils import build_hands_static, extract_with_raw
from src.utils.email_alert import load_smtp_config, save_smtp_config, send_alert_email

log = get_logger("api")
app_cfg, _, train_cfg, _, srv = load_config()

BASE = Path(__file__).resolve().parent
SIGNS_DIR = BASE / "static" / "signs"
SIGNS_DIR.mkdir(parents=True, exist_ok=True)
# Absolute: Flask's send_file resolves relative paths against the app folder,
# so model artifacts must never rely on the process working directory.
MODEL_DIR = BASE.parent / train_cfg.model_dir

_predictor = None  # lazy singleton — keeps `flask --app` import light without a model


def get_predictor():
    global _predictor
    if _predictor is None:
        from src.inference.predictor import SignPredictor
        _predictor = SignPredictor()
    return _predictor


def _placeholder(char: str) -> Image.Image:
    """High-contrast fallback tile: solid brand field, bold snow glyph + ring."""
    img = Image.new("RGB", (240, 240), (249, 76, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([10, 10, 230, 230], radius=18, outline=(255, 250, 250), width=5)
    try:
        font = ImageFont.truetype("arial.ttf", 120)
    except OSError:
        font = ImageFont.load_default()
    d.text((120, 120), char, fill=(255, 250, 250), font=font, anchor="mm",
           stroke_width=2, stroke_fill=(255, 250, 250))
    return img


def create_app() -> Flask:
    app = Flask(__name__, template_folder="templates", static_folder="static")
    CORS(app)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/health")
    def health():
        # Fast, TF-free: report artifact presence + vocabulary from the encoder.
        # (The heavy model loads lazily on first /api/predict call.)
        mdir = MODEL_DIR
        ready = ((mdir / train_cfg.model_name).exists()
                 and (mdir / train_cfg.rf_name).exists()) \
            or (mdir / "hybrid_cnn_lstm_cnn.keras").exists()
        classes: list[str] = []
        enc = mdir / train_cfg.encoder_name
        if ready and enc.exists():
            try:
                classes = [str(c) for c in joblib.load(enc).classes_]
            except Exception:
                log.warning("Encoder unreadable", exc_info=True)
        return jsonify(status="ok", model_loaded=_predictor is not None,
                       model_exists=ready, classes=classes)

    @app.get("/api/classes")
    def classes():
        try:
            return jsonify(classes=get_predictor().classes)
        except FileNotFoundError as e:
            return jsonify(error=str(e)), 404

    @app.post("/api/predict")
    def predict():
        """Body: {image: 'data:image/jpeg;base64,...' | raw base64}"""
        try:
            pred = get_predictor()
        except FileNotFoundError as e:
            return jsonify(error=str(e)), 404
        data = request.get_json(force=True, silent=True) or {}
        b64 = data.get("image", "")
        b64 = re.sub(r"^data:image/\w+;base64,", "", b64)
        if not b64:
            return jsonify(error="Missing 'image' (base64 JPEG)"), 400
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        except Exception:
            return jsonify(error="Invalid base64 image"), 400
        hands = build_hands_static()
        try:
            vec, _raw = extract_with_raw(frame, hands)
        finally:
            hands.close()
        if vec is None:
            return jsonify(error="No hand detected — show palm clearly to camera"), 422
        try:
            label, conf, dist = pred.predict_single(vec)
        except ValueError as e:
            # Feature-dim mismatch.
            return jsonify(error=str(e)), 422
        return jsonify(label=label, confidence=round(conf, 4),
                        smoothed=pred.smoothed(), distribution=dist)

    @app.post("/api/text-to-sign")
    def text_to_sign():
        """Whole-word custom signs win; anything else is fingerspelled per character."""
        data = request.get_json(force=True, silent=True) or {}
        text = str(data.get("text", ""))[:120]
        by_label = {m["label"]: s for s, m in _load_custom().items()}
        words = text.upper().split(" ")
        maxw = max([len(lbl.split()) for lbl in by_label] or [1])
        seq = []
        i = 0
        while i < len(words):
            if i:
                seq.append({"char": " ", "sign_image": None, "available": False,
                            "word_gap": True, "custom": False})
            if not words[i]:
                i += 1
                continue
            match = next((phrase for k in range(min(maxw, len(words) - i), 0, -1)
                          for phrase in [" ".join(words[i:i + k])]
                          if phrase in by_label), None)
            if match is not None:
                seq.append({"char": match,
                            "sign_image": f"/api/custom-signs/{by_label[match]}/image",
                            "available": True, "word_gap": False, "custom": True})
                i += len(match.split())
            else:
                for ch in words[i]:
                    if not ch.isalnum():
                        continue
                    p = SIGNS_DIR / f"{ch}.png"
                    seq.append({"char": ch, "sign_image": f"/api/sign-image/{ch}",
                                "available": p.exists(), "word_gap": False, "custom": False})
                i += 1
        return jsonify(text=text, sequence=seq,
                       hint="Add real ASL images at app/static/signs/<CHAR>.png to replace placeholders.")

    @app.get("/api/sign-image/<char>")
    def sign_image(char: str):
        c = char.upper()[:1]
        p = SIGNS_DIR / f"{c}.png"
        if p.exists():
            return send_file(p, mimetype="image/png")
        buf = io.BytesIO()
        _placeholder(c).save(buf, format="PNG")
        buf.seek(0)
        return send_file(buf, mimetype="image/png")

    # ── Characters (vocabulary artwork management) ──
    ALPHABET = [chr(c) for c in range(ord("A"), ord("Z") + 1)] + [str(d) for d in range(10)]

    def _trained_set() -> set:
        try:
            return set(get_predictor().classes)
        except FileNotFoundError:
            return set()

    @app.get("/api/vocabulary")
    def vocabulary():
        trained = _trained_set()
        items = [{
            "char": ch,
            "sign_image": f"/api/sign-image/{ch}",
            "available": (SIGNS_DIR / f"{ch}.png").exists(),
            "trained": ch in trained,
        } for ch in ALPHABET]
        return jsonify(characters=items, count=len(items),
                       trained_count=sum(1 for i in items if i["trained"]))

    @app.post("/api/characters")
    def add_character():
        """Upload reference artwork: {char: 'C', image: base64} → signs/C.png"""
        data = request.get_json(force=True, silent=True) or {}
        ch = str(data.get("char", "")).strip().upper()
        if len(ch) != 1 or not ch.isalnum():
            return jsonify(error="Field 'char' must be a single A-Z / 0-9 character"), 400
        b64 = re.sub(r"^data:image/\w+;base64,", "", data.get("image", ""))
        if not b64:
            return jsonify(error="Missing 'image' (base64)"), 400
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").resize((240, 240))
        except Exception:
            return jsonify(error="Invalid base64 image"), 400
        img.save(SIGNS_DIR / f"{ch}.png")
        log.info("Character artwork saved: %s", ch)
        return jsonify(char=ch, sign_image=f"/api/sign-image/{ch}", available=True), 201

    @app.delete("/api/characters/<char>")
    def delete_character(char: str):
        """Removes custom artwork only — trained classes are untouched."""
        c = char.upper()[:1]
        p = SIGNS_DIR / f"{c}.png"
        if not p.exists():
            return jsonify(error=f"No custom artwork stored for '{c}'"), 404
        p.unlink()
        log.info("Character artwork deleted: %s", c)
        return jsonify(deleted=c)

    # ── Custom signs (user-defined words/phrases) ──
    CUSTOM_REG = SIGNS_DIR / "custom.json"

    def _load_custom() -> dict:
        if CUSTOM_REG.exists():
            try:
                return json.loads(CUSTOM_REG.read_text(encoding="utf-8"))
            except ValueError:
                return {}
        return {}

    def _save_custom(reg: dict) -> None:
        CUSTOM_REG.write_text(json.dumps(reg, indent=2), encoding="utf-8")

    def _slugify(label: str) -> str:
        return re.sub(r"[^A-Z0-9]+", "_", label.strip().upper()).strip("_")[:30]

    def _custom_item(slug: str, meta: dict) -> dict:
        return {"id": slug, "label": meta["label"],
                "sign_image": f"/api/custom-signs/{slug}/image"}

    @app.get("/api/custom-signs")
    def list_custom():
        reg = _load_custom()
        return jsonify(signs=[_custom_item(s, m) for s, m in sorted(reg.items())],
                       count=len(reg))

    @app.post("/api/custom-signs")
    def add_custom():
        """Add a word-level sign: {label: 'Thank you', image: base64}."""
        data = request.get_json(force=True, silent=True) or {}
        label = str(data.get("label", "")).strip()
        if not (1 <= len(label) <= 30) or not re.fullmatch(r"[A-Za-z0-9 ]+", label):
            return jsonify(error="Field 'label' must be 1–30 letters/digits/spaces"), 400
        slug = _slugify(label)
        if not slug:
            return jsonify(error="Label has no usable characters"), 400
        b64 = re.sub(r"^data:image/\w+;base64,", "", data.get("image", ""))
        if not b64:
            return jsonify(error="Missing 'image' (base64)"), 400
        try:
            img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB").resize((240, 240))
        except Exception:
            return jsonify(error="Invalid base64 image"), 400
        img.save(SIGNS_DIR / f"custom_{slug}.png")
        reg = _load_custom()
        reg[slug] = {"label": re.sub(r"\s+", " ", label.upper()), "file": f"custom_{slug}.png"}
        _save_custom(reg)
        log.info("Custom sign saved: %s", reg[slug]["label"])
        return jsonify(_custom_item(slug, reg[slug])), 201

    @app.delete("/api/custom-signs/<slug>")
    def delete_custom(slug: str):
        reg = _load_custom()
        meta = reg.pop(slug.upper(), None)
        if meta is None:
            return jsonify(error=f"No custom sign '{slug}'"), 404
        (SIGNS_DIR / meta["file"]).unlink(missing_ok=True)
        _save_custom(reg)
        log.info("Custom sign deleted: %s", slug.upper())
        return jsonify(deleted=slug.upper())

    @app.get("/api/custom-signs/<slug>/image")
    def custom_image(slug: str):
        meta = _load_custom().get(slug.upper())
        if not meta:
            return jsonify(error="Not found"), 404
        return send_file(SIGNS_DIR / meta["file"], mimetype="image/png")

    @app.get("/metrics")
    def metrics_page():
        return render_template("metrics.html")

    # ── AI chat (sign → local Ollama LLM, no typing required) ──
    LLM = srv.get("llm", {}) or {}

    def _ollama_generate(prompt: str) -> str:
        url = str(LLM.get("url", "http://localhost:11434")).rstrip("/")
        body = json.dumps({
            "model": LLM.get("model", "qwen2.5-coder:1.5b"),
            "prompt": prompt,
            "stream": False,
        }).encode("utf-8")
        req = urllib.request.Request(
            url + "/api/generate", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=int(LLM.get("timeout", 120))) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            raise ConnectionError(
                "Ollama is not reachable at %s — start it with `ollama serve` "
                "and pull the model (`ollama pull %s`). [%s]"
                % (url, LLM.get("model", "qwen2.5-coder:1.5b"), e))
        text = (data.get("response") or "").strip()
        if not text:
            raise ValueError("Empty reply from model: %s" % (data.get("error") or "unknown"))
        return text

    @app.get("/chat")
    def chat_page():
        return render_template("chat.html",
                               llm_model=LLM.get("model", "qwen2.5-coder:1.5b"))

    @app.get("/guide")
    def guide_page():
        return render_template("guide.html")

    @app.post("/api/chat")
    def chat_api():
        data = request.get_json(force=True, silent=True) or {}
        message = str(data.get("message", "")).strip()[:2000]
        if not message:
            return jsonify(error="Missing 'message'"), 400
        history = data.get("history", []) or []
        turns = []
        for h in history[-6:]:
            if isinstance(h, dict) and h.get("role") in ("user", "assistant") and h.get("text"):
                who = "User" if h["role"] == "user" else "Assistant"
                turns.append("%s: %s" % (who, str(h["text"])[:1000]))
        prompt = "%s\n\n%s\nUser: %s\nAssistant:" % (
            LLM.get("system_prompt", "You are a helpful assistant."),
            "\n".join(turns), message)
        try:
            reply = _ollama_generate(prompt)
        except ConnectionError as e:
            return jsonify(error=str(e)), 502
        except Exception as e:
            log.warning("LLM request failed", exc_info=True)
            return jsonify(error="Model request failed: %s" % e), 502
        return jsonify(reply=reply, model=LLM.get("model", "qwen2.5-coder:1.5b"))

    # ── Translation (English → Hindi / Marathi, server-side) ──
    # Browser previously called MyMemory directly: rate-limited, no error
    # handling, and MyMemory crowdsourced entries are wrong (e.g. "Hello how
    # are you" → unrelated Hindi). Chain: Google gtx (best quality, retried)
    # → MyMemory (last resort). Results cached in memory.
    _TR_CACHE: dict = {}
    _TR_ORDER: list = []

    def _tr_cache_get(key):
        return _TR_CACHE.get(key)

    def _tr_cache_put(key, val: str) -> None:
        if key in _TR_CACHE:
            return
        _TR_CACHE[key] = val
        _TR_ORDER.append(key)
        if len(_TR_ORDER) > 500:  # bounded memory
            _TR_CACHE.pop(_TR_ORDER.pop(0), None)

    def _tr_google(text: str, target: str) -> str:
        import time as _time
        params = urllib.parse.urlencode(
            {"client": "gtx", "sl": "en", "tl": target, "dt": "t", "q": text})
        last: Exception | None = None
        for attempt in range(2):  # retries on 429 rate-limit
            try:
                req = urllib.request.Request(
                    "https://translate.googleapis.com/translate_a/single?" + params,
                    headers={"User-Agent": "Mozilla/5.0"}, method="GET")
                with urllib.request.urlopen(req, timeout=6) as r:
                    data = json.loads(r.read().decode("utf-8"))
                # data[0] = [[translated, original, ...], ...]
                parts = [(seg[0] or "") for seg in (data[0] or []) if seg]
                out = "".join(parts).strip()
                if not out:
                    raise ValueError("Empty reply from Google Translate")
                return out
            except Exception as e:
                last = e
                _time.sleep(2)
        raise ValueError("Google Translate: %s" % last)

    def _tr_mymemory(text: str, target: str) -> str:
        params = urllib.parse.urlencode(
            {"q": text, "langpair": "en|" + target})
        req = urllib.request.Request(
            "https://api.mymemory.translated.net/get?" + params,
            headers={"User-Agent": "Mozilla/5.0"}, method="GET")
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read().decode("utf-8"))
        out = ((data.get("responseData") or {}).get("translatedText") or "").strip()
        bad = ("MYMEMORY WARNING" in out.upper() or "QUERY LENGTH LIMIT" in out.upper()
               or "INVALID" in out.upper() and len(out) > len(text) * 2)
        if not out or bad:
            raise ValueError("MyMemory refused: %s" % (out[:120] or "empty"))
        return out

    @app.post("/api/translate")
    def translate_api():
        data = request.get_json(force=True, silent=True) or {}
        text = str(data.get("text", "")).strip()[:500]
        target = str(data.get("target", "")).strip().lower()[:5]
        if target not in ("hi", "mr"):
            return jsonify(error="Field 'target' must be 'hi' or 'mr'"), 400
        if not text:
            return jsonify(error="Missing 'text'"), 400
        key = (text, target)
        hit = _tr_cache_get(key)
        if hit:
            return jsonify(text=text, target=target, translated=hit, cached=True)
        errors = []
        for fn in (_tr_google, _tr_mymemory):
            try:
                out = fn(text, target)
                _tr_cache_put(key, out)
                return jsonify(text=text, target=target, translated=out, cached=False,
                               provider=fn.__name__)
            except Exception as e:
                errors.append("%s: %s" % (fn.__name__, e))
        log.warning("Translate failed hi/mr: %s", " | ".join(errors))
        return jsonify(error="Translation service unavailable — check internet and retry.",
                       detail=" | ".join(errors)[:300]), 502

    @app.get("/api/metrics")
    def metrics_api():
        p = MODEL_DIR / "metrics.json"
        if not p.exists():
            return jsonify(error="No training metrics yet — train the model first."), 404
        try:
            return jsonify(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            log.warning("metrics.json unreadable", exc_info=True)
            return jsonify(error="Training metrics file is corrupt."), 500

    @app.get("/api/plots/<name>")
    def plots(name: str):
        allowed = {"training_curves.png", "confusion_matrix.png"}
        if name not in allowed:
            return jsonify(error="Unknown plot."), 404
        p = MODEL_DIR / name
        if not p.exists():
            return jsonify(error="Plot not generated yet — train the model first."), 404
        return send_file(p, mimetype="image/png", max_age=60)

    # ── SMTP Email Alert Endpoints ──
    @app.get("/api/smtp-config")
    def get_smtp_config():
        cfg = load_smtp_config()
        return jsonify({
            "smtp_host": cfg.get("smtp_host", "smtp.gmail.com"),
            "smtp_port": cfg.get("smtp_port", 587),
            "use_tls": cfg.get("use_tls", True),
            "sender_email": cfg.get("sender_email", ""),
            "default_recipient": cfg.get("default_recipient", ""),
            "password_configured": bool((cfg.get("sender_password") or "").strip())
        })

    @app.post("/api/smtp-config")
    def update_smtp_config():
        data = request.get_json(force=True, silent=True) or {}
        updated = save_smtp_config(data)
        log.info("SMTP configuration updated")
        return jsonify({
            "status": "ok",
            "message": "SMTP configuration saved successfully",
            "config": {
                "smtp_host": updated.get("smtp_host"),
                "smtp_port": updated.get("smtp_port"),
                "use_tls": updated.get("use_tls"),
                "sender_email": updated.get("sender_email"),
                "default_recipient": updated.get("default_recipient"),
                "password_configured": bool((updated.get("sender_password") or "").strip())
            }
        })

    @app.post("/api/send-alert")
    def send_alert():
        data = request.get_json(force=True, silent=True) or {}
        text = str(data.get("text", "")).strip()
        recipient = str(data.get("recipient", "")).strip() or None
        subject = str(data.get("subject", "")).strip() or None

        if not text:
            return jsonify(error="Alert text content is empty. Sign some text or enter a message first."), 400

        try:
            res = send_alert_email(text_content=text, recipient=recipient, subject=subject)
            log.info("Alert email sent to %s", res.get("recipient"))
            return jsonify(res)
        except ValueError as e:
            return jsonify(error=str(e)), 400
        except Exception as e:
            log.error("Failed to send alert email", exc_info=True)
            return jsonify(error=f"SMTP Send Failed: {str(e)}"), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host=srv["server"]["host"], port=srv["server"]["port"], debug=srv["server"]["debug"])
