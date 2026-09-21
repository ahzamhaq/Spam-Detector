# SMS Spam Detector

A simple Streamlit web app that classifies an SMS message as **spam** or **not spam** using a Naive Bayes classifier.

## How it works

- Trained on the [SMS Spam Collection dataset](spam.csv) (5,572 labeled messages).
- Text is vectorized with `CountVectorizer` (bag-of-words).
- Classified with a `MultinomialNB` model (~97.8% test accuracy).
- Model and vectorizer are pickled (`spam.pkl`, `vectorizer.pkl`) and loaded by the app for inference.

See `Spam Detector.ipynb` for the full training pipeline.

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Enter a message in the text box and click **Predict** to see whether it's spam.

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit app for inference |
| `Spam Detector.ipynb` | Notebook used to train and export the model |
| `spam.csv` | Training dataset |
| `spam.pkl` | Pickled trained model |
| `vectorizer.pkl` | Pickled fitted `CountVectorizer` |
| `requirements.txt` | Python dependencies |
