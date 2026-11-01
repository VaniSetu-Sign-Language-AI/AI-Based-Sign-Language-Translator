# 🤟 Sign Language Enterprise — Hybrid CNN + LSTM + CNN

End-to-end **Sign ⇄ Text** system built like a production ML service:

| Stage | Implementation |
|---|---|
| **1. Data collection** | Webcam / image-folder → MediaPipe Hands (21 pts × xyz = 63-D, wrist-relative + scale-normalised) → `data/landmarks.csv` |
| **2. Dataset** | Frame CSV → sliding-window sequences `(30, 63)`, stratified split, `StandardScaler` + `LabelEncoder` persisted |
| **3. Model** | **Hybrid CNN → LSTM → RandomForest**: `Conv1D×2+Pool` (local features) → `BiLSTM+LSTM` (temporal embedding) → `RandomForest` head on frozen embeddings |
| **4. Serving** | Flask: Sign→Text (`/api/predict` from webcam snapshot/upload) + Text→Sign (per-character sign cards) |

```
D:\Project Code
├── app/                  # Flask service
│   ├── app.py            # factory, /api/* endpoints
│   ├── templates/index.html
│   └── static/{css,js,signs/}   # put real ASL refs as signs/<CHAR>.png
├── configs/config.yaml   # all hyperparams / paths (single source of truth)
├── src/
│   ├── config.py  logger.py
│   ├── utils/mediapipe_utils.py
│   ├── data/{collect_landmarks.py, dataset.py}
│   ├── models/hybrid_model.py
│   ├── training/train.py
│   └── inference/predictor.py
├── data/  models/  logs/
└── requirements.txt
```

## 1. Setup

```powershell
python -m venv venv; .\venv\Scripts\Activate
pip install -r requirements.txt
```

## 2. Collect data

Option A — organised image folders (recommended for bootstrap):
```
data/raw_images/A/*.jpg  data/raw_images/B/*.jpg  ...
```
Option B — live webcam recording.

```powershell
# images only
python -m src.data.collect_landmarks --mode image
# webcam only (press A-Z/0-9 to record, N for a custom sign name, Q to quit)
python -m src.data.collect_landmarks --mode webcam
# custom sign straight from CLI — any name (words like HELLO work, not just letters)
python -m src.data.collect_landmarks --mode webcam --label HELLO --frames-per-class 150
# both
python -m src.data.collect_landmarks --mode both
```

Verify: `data/landmarks.csv` → `label,x0,y0,z0,...,x20,y20,z20` (~≥60 frames/class minimum, 120+ recommended).

## 3. Train (hybrid CNN+LSTM+RandomForest)

```powershell
python -m src.training.train --epochs 60 --batch-size 32
# Stage 1: backbone pre-training → models/hybrid_cnn_lstm_backbone.keras
# Stage 2: RF head on frozen embeddings → models/randomforest.pkl (+ encoder/scaler, training_curves.png)

# Fast path — only re-fit the RF head (e.g. after adding new signs, backbone unchanged):
python -m src.training.train --rf-only
```
Every run also writes `models/metrics.json` + `training_curves.png` + `confusion_matrix.png`
— open the **Metrics** nav page in the app for accuracy, macro-F1, per-class P/R/F1 and both graphs.

Tune in `configs/config.yaml` (`sequence_length`, filters, LSTM units, dropout, LR).

## 4. Run Flask (prediction + text-to-sign)

```powershell
python app/app.py
# open http://localhost:5000
```

* **Sign→Text**: Start camera → show sign → Capture & Predict (or Upload). High-confidence preds append to the sentence strip; 🔊 speaks it.
* **Text→Sign**: type `HELLO` → card strip `H E L L O`, each served from `/api/sign-image/<CHAR>`. Drop real reference PNGs into `app/static/signs/` to replace placeholders.

API quick test:
```powershell
curl http://localhost:5000/api/health
curl http://localhost:5000/api/classes
```

## 5. Production notes (from 5-yr AI-engineer lens)

* Landmarks (not raw pixels) → lighting/skin-tone robust, tiny model, CPU-servable.
* Normalisation (wrist-relative / scale-invariant) is what makes cross-user accuracy hold.
* Single stills are tiled ×30 at inference so the sequence model works for static alphabets *and* dynamic words without code change.
* Majority-vote smoothing (`smoothing_window: 5`) removes realtime flicker.
* Swap `build_hybrid()` for TFLite/ONNX export when moving to edge/mobile — API contract (`predict_sequence`) stays unchanged.
