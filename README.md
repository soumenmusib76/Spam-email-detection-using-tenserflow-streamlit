# Email Spam Detection using TensorFlow

A beginner-friendly machine learning project that detects whether an email message is **Spam** or **Not Spam** using TensorFlow/Keras and an LSTM neural network.

The project includes:

- Text preprocessing
- Tokenization
- Sequence padding
- LSTM-based TensorFlow model
- Saved model and tokenizer generation
- Streamlit web app for interactive prediction
- CLI prediction script
- Setup checker script

---

## Project Structure

```text
email-spam-detection-tensorflow-complete/
├── app.py
├── data/
│   └── email.csv
├── docs/
├── models/
│   └── README.md
├── notebooks/
│   └── email_spam_detection_tensorflow.ipynb
├── src/
│   ├── check_setup.py
│   ├── config.json
│   ├── predict.py
│   └── train_model.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## Step 1: Install dependencies

Open terminal in the main project folder and run:

```bash
pip install -r requirements.txt
```

---

## Step 2: Train the model

Run:

```bash
python src/train_model.py
```

This will create:

```text
models/spam_classifier.keras
models/tokenizer.pkl
models/metrics.json
```

---

## Step 3: Run the Streamlit app

After training is complete, run:

```bash
streamlit run app.py
```

Your browser will open an interactive app where you can paste an email message and check whether it is spam.

---

## Step 4: Test from terminal

You can also test prediction without Streamlit:

```bash
python src/predict.py
```

---

## Step 5: Check setup

After installing dependencies, you can check the project setup:

```bash
python src/check_setup.py
```

---

## How the model works

The text message is converted into numbers using a tokenizer.

Then the model processes the sequence:

```text
Input text
↓
Tokenizer
↓
Padding
↓
Embedding layer
↓
LSTM layer
↓
Dense layer
↓
Sigmoid output
↓
Spam probability
```

If the spam probability is greater than or equal to `0.5`, the app predicts **Spam**.

---

## Important files

### `models/spam_classifier.keras`

This is the trained TensorFlow model. It is created after running:

```bash
python src/train_model.py
```

### `models/tokenizer.pkl`

This stores the tokenizer vocabulary. It is also created after training.

### `app.py`

This is the Streamlit app. It loads the saved model and tokenizer and allows interactive spam checking.

---

## GitHub and Streamlit Cloud

For Streamlit Community Cloud:

1. Upload this project to GitHub.
2. Train the model locally or in Google Colab.
3. Make sure these files exist in the GitHub repo:

```text
models/spam_classifier.keras
models/tokenizer.pkl
```

4. In Streamlit Community Cloud, set:

```text
Main file path: app.py
```

---

## Notes

TensorFlow GPU may not work on native Windows for newer TensorFlow versions. CPU training is fine for this project.
