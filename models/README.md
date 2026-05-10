# Models Folder

This folder stores the trained model files.

After running:

```bash
python src/train_model.py
```

the following files will be created:

```text
models/spam_classifier.keras
models/tokenizer.pkl
models/metrics.json
```

Meaning:

- `spam_classifier.keras` - trained TensorFlow/Keras LSTM model
- `tokenizer.pkl` - saved tokenizer/dictionary used to convert email text into numbers
- `metrics.json` - test accuracy, confusion matrix, and classification report

The Streamlit app needs `spam_classifier.keras` and `tokenizer.pkl` to run predictions.
