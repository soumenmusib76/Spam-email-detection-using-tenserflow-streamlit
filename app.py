from __future__ import annotations

import json
import pickle
from pathlib import Path

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "src" / "config.json"


def load_config() -> dict:
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


cfg = load_config()
MAX_LEN = cfg["max_len"]
THRESHOLD = cfg["threshold"]
MODEL_PATH = PROJECT_ROOT / cfg["model_path"]
TOKENIZER_PATH = PROJECT_ROOT / cfg["tokenizer_path"]
METRICS_PATH = PROJECT_ROOT / cfg["metrics_path"]


st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="📧",
    layout="centered",
)

st.title("📧 Email Spam Detection using TensorFlow")
st.write(
    "Paste an email or SMS-style message below. "
    "The trained LSTM model will classify it as spam or not spam."
)


@st.cache_resource
def load_model_and_tokenizer():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as file:
        tokenizer = pickle.load(file)
    return model, tokenizer


if not MODEL_PATH.exists() or not TOKENIZER_PATH.exists():
    st.error("Model files are missing.")
    st.write("First train the model by running this command in the project folder:")
    st.code("python src/train_model.py", language="bash")
    st.stop()


model, tokenizer = load_model_and_tokenizer()

example = "Congratulations! You have won a free prize. Click here to claim now."

email_text = st.text_area(
    "Enter email text:",
    value="",
    placeholder=example,
    height=180,
)

if st.button("Check Spam"):
    if not email_text.strip():
        st.warning("Please enter an email message first.")
    else:
        sequence = tokenizer.texts_to_sequences([email_text])
        padded = pad_sequences(
            sequence,
            maxlen=MAX_LEN,
            padding="post",
            truncating="post",
        )

        spam_probability = float(model.predict(padded, verbose=0)[0][0])

        st.subheader("Prediction Result")

        if spam_probability >= THRESHOLD:
            st.error("🚨 Spam")
        else:
            st.success("✅ Not Spam")

        st.write(f"Spam probability: **{spam_probability:.4f}**")
        st.progress(spam_probability)

with st.expander("Model information"):
    st.write("Model file:", str(MODEL_PATH))
    st.write("Tokenizer file:", str(TOKENIZER_PATH))
    st.write("Maximum sequence length:", MAX_LEN)
    st.write("Decision threshold:", THRESHOLD)

    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as file:
            metrics = json.load(file)
        st.write("Test accuracy:", round(metrics.get("test_accuracy", 0), 4))
        st.json(metrics)
    else:
        st.info("Metrics file will appear after training.")