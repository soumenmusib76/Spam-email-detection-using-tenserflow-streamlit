from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_dataset(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)

    if "text" not in df.columns:
        raise ValueError("Dataset must contain a 'text' column containing email messages.")

    if "label_num" in df.columns:
        labels = df["label_num"].astype(int)
    elif "label" in df.columns:
        labels = df["label"].astype(str).str.lower().map({"ham": 0, "not spam": 0, "spam": 1})
        if labels.isna().any():
            raise ValueError("Could not convert the 'label' column to 0/1 values.")
    else:
        raise ValueError("Dataset must contain either 'label_num' or 'label' column.")

    texts = df["text"].fillna("").astype(str)

    valid_rows = texts.str.strip().ne("")
    texts = texts[valid_rows].to_numpy()
    labels = labels[valid_rows].to_numpy()

    return texts, labels


def build_model(max_words: int, max_len: int, embedding_dim: int, lstm_units: int, dense_units: int) -> tf.keras.Model:
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(max_len,)),
            tf.keras.layers.Embedding(input_dim=max_words, output_dim=embedding_dim),
            tf.keras.layers.LSTM(lstm_units),
            tf.keras.layers.Dense(dense_units, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    return model


def main() -> None:
    cfg = load_config()

    data_path = PROJECT_ROOT / cfg["data_path"]
    model_path = PROJECT_ROOT / cfg["model_path"]
    tokenizer_path = PROJECT_ROOT / cfg["tokenizer_path"]
    metrics_path = PROJECT_ROOT / cfg["metrics_path"]

    model_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from:", data_path)
    texts, labels = load_dataset(data_path)

    print(f"Total samples: {len(texts)}")
    print(f"Ham samples: {(labels == 0).sum()}")
    print(f"Spam samples: {(labels == 1).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=labels,
    )

    tokenizer = Tokenizer(num_words=cfg["max_words"], oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(
        X_train_seq,
        maxlen=cfg["max_len"],
        padding="post",
        truncating="post",
    )
    X_test_pad = pad_sequences(
        X_test_seq,
        maxlen=cfg["max_len"],
        padding="post",
        truncating="post",
    )

    model = build_model(
        max_words=cfg["max_words"],
        max_len=cfg["max_len"],
        embedding_dim=cfg["embedding_dim"],
        lstm_units=cfg["lstm_units"],
        dense_units=cfg["dense_units"],
    )

    model.summary()

    print("\nTraining model...")
    history = model.fit(
        X_train_pad,
        y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_data=(X_test_pad, y_test),
        verbose=1,
    )

    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test_pad, y_test, verbose=0)
    y_prob = model.predict(X_test_pad, verbose=0).reshape(-1)
    y_pred = (y_prob >= cfg["threshold"]).astype(int)

    report = classification_report(y_test, y_pred, target_names=["ham", "spam"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "test_loss": float(loss),
        "test_accuracy": float(accuracy),
        "sklearn_accuracy": float(accuracy_score(y_test, y_pred)),
        "confusion_matrix": cm,
        "classification_report": report,
        "epochs": cfg["epochs"],
        "max_len": cfg["max_len"],
        "max_words": cfg["max_words"],
    }

    print(f"Test accuracy: {accuracy:.4f}")
    print("Confusion matrix:", cm)

    print("\nSaving model and tokenizer...")
    model.save(model_path)

    with open(tokenizer_path, "wb") as file:
        pickle.dump(tokenizer, file)

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("Saved model to:", model_path)
    print("Saved tokenizer to:", tokenizer_path)
    print("Saved metrics to:", metrics_path)
    print("\nDone. Now run: streamlit run app.py")


if __name__ == "__main__":
    main()