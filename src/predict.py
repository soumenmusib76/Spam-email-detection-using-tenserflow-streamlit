from __future__ import annotations

import json
import pickle
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("config.json")


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    cfg = load_config()

    model_path = PROJECT_ROOT / cfg["model_path"]
    tokenizer_path = PROJECT_ROOT / cfg["tokenizer_path"]

    if not model_path.exists() or not tokenizer_path.exists():
        raise FileNotFoundError("Model files not found. Run: python src/train_model.py")

    model = tf.keras.models.load_model(model_path)
    with open(tokenizer_path, "rb") as file:
        tokenizer = pickle.load(file)

    print("Type an email message and press Enter.")
    text = input("Email text: ").strip()

    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(
        sequence,
        maxlen=cfg["max_len"],
        padding="post",
        truncating="post",
    )

    spam_probability = float(model.predict(padded, verbose=0)[0][0])
    label = "SPAM" if spam_probability >= cfg["threshold"] else "NOT SPAM"

    print("Prediction:", label)
    print(f"Spam probability: {spam_probability:.4f}")


if __name__ == "__main__":
    main()