"""Hybrid CNN → LSTM → RandomForest for hand-landmark sequences.

Two-stage enterprise design:
  Stage 1 (deep): CNN front-end + stacked LSTM backbone trained with a
                  temporary softmax head → learns temporal embeddings.
  Stage 2 (ML)  : RandomForestClassifier fitted on frozen backbone embeddings.

Why RF as head: robust on small per-class datasets, calibrated
predict_proba, no gradient tuning at serve time, trivial to re-fit when
new signs are added (just re-run Stage 2).

Input : (T=sequence_length, F=126 two-hand | 63 legacy single-hand)
Stage-1 output: P(class)   |   Embedding: `embedding` layer vector
Stage-2 output: P(class) via RF
"""
from __future__ import annotations

from tensorflow.keras import layers, models

from src.config import load_config

_, model_cfg, _, _, _ = load_config()

EMBEDDING_LAYER = "embedding"


def build_hybrid(input_shape: tuple[int, int], n_classes: int):
    """Stage-1 backbone: CNN feature extractor → LSTM temporal → embedding → softmax.

    Name kept as `build_hybrid` so existing training entry-points are unaffected.
    """
    c = model_cfg
    inp = layers.Input(shape=input_shape, name="landmarks")

    # ── CNN front-end: local spatial/motion features ──
    x = layers.Conv1D(c.conv1_filters, 3, padding="same", activation="relu", name="cnn1a")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(c.conv2_filters, 3, padding="same", activation="relu", name="cnn1b")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2, name="pool1")(x)
    x = layers.Dropout(c.dropout)(x)

    # ── LSTM temporal core: sequence → fixed vector ──
    x = layers.Bidirectional(layers.LSTM(c.lstm1_units, return_sequences=True), name="bilstm1")(x)
    x = layers.Dropout(c.dropout)(x)
    x = layers.LSTM(c.lstm2_units, name="lstm2")(x)
    x = layers.Dropout(c.dropout)(x)

    # ── Embedding (frozen later for the RF head) ──
    x = layers.Dense(c.embedding_dim, activation="relu", name=EMBEDDING_LAYER)(x)
    x = layers.BatchNormalization(name="emb_bn")(x)
    x = layers.Dropout(c.dropout, name="emb_drop")(x)

    # ── Temporary softmax head (Stage-1 pre-training only) ──
    out = layers.Dense(n_classes, activation="softmax", name="logits")(x)

    model = models.Model(inp, out, name="Hybrid_CNN_LSTM_backbone")
    model.compile(
        optimizer=__import__("tensorflow").keras.optimizers.Adam(c.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_embedder(trained_model) -> models.Model:
    """Freeze a trained backbone and expose its embedding vector."""
    emb = trained_model.get_layer(EMBEDDING_LAYER).output
    m = models.Model(trained_model.input, emb, name="CNN_LSTM_embedder")
    m.trainable = False
    return m
