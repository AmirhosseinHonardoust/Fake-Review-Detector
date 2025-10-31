# Fake Review Detector (NLP + Streamlit)

A machine learning project that detects **fake vs real** product reviews using **TF-IDF vectorization**, **Logistic Regression**, and **behavioral text features** such as exclamation count, sentiment, and repeated promotional phrases.  
It also includes a sleek **Streamlit app** for interactive real-time predictions.

---

## Features
- Text cleaning and normalization pipeline  
- Hybrid feature extraction:
  - TF-IDF (1–2 grams)
  - Numeric sentiment & behavioral features  
- Interpretable Logistic Regression model  
- Evaluation metrics: Confusion Matrix, ROC, and PR curves  
- Interactive Streamlit app with adjustable decision threshold  

---

## Folder Structure
```
fake-review-detector/
├── app/
│   └── streamlit_app.py
├── src/
│   ├── clean_text.py
│   ├── features.py
│   ├── train.py
│   └── predict.py
├── data/
│   └── reviews_sample.csv
├── outputs/
│   ├── pipeline.joblib
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── pr_curve.png
├── requirements.txt
└── README.md
```

---

## How It Works

1. **Data Input:** CSV containing `text` and `label` columns.  
2. **Preprocessing:** URL, punctuation, and HTML removal + lowercasing.  
3. **Feature Engineering:**  
   - Sentiment score  
   - Exclamation & ALL-CAPS detection  
   - Fake-review clichés (e.g., “best product ever”)  
4. **Modeling:** Logistic Regression trained on combined features.  
5. **Prediction:** Threshold-tunable classification for FAKE vs REAL.  

---

## Streamlit Interface

Below is a preview of the web app UI built with Streamlit:

<img width="628" height="556" alt="Screenshot 2025-10-30 at 11-15-11 Fake Review Detector" src="https://github.com/user-attachments/assets/dbfe14df-f69b-4079-9bbc-9d89c12e903a" />

---

### Highlights
- Paste or type any review text.  
- Adjust decision threshold for sensitivity.  
- Get immediate prediction with fake probability.  
- Built-in tips to help interpret the model.

---

## Model Evaluation

### Confusion Matrix
<img width="640" height="480" alt="confusion_matrix" src="https://github.com/user-attachments/assets/1aae55a8-9d54-4280-bc0b-60c0ebd8d577" />

---

### Precision-Recall Curve
<img width="640" height="480" alt="pr_curve" src="https://github.com/user-attachments/assets/b4d4f65d-31df-428b-b9ea-75e456bc151d" />

---

### ROC Curve
<img width="640" height="480" alt="roc_curve" src="https://github.com/user-attachments/assets/01cfd7ad-ac1e-4a1b-ac52-a165067c4039" />

The model achieves **AUC ≈ 1.00** and **AP ≈ 1.00** on sample data (balanced, synthetic).

---

## Setup & Usage

```bash
python -m venv .venv
# Activate
.venv\Scripts\activate  # (Windows)
# source .venv/bin/activate  # (macOS/Linux)

pip install -r requirements.txt

# Train the model
python src/train.py --csv data/reviews_sample.csv --outdir outputs

# Predict a single review
python src/predict.py --pipeline outputs/pipeline.joblib --text "I got this for free, best product ever!!!"

# Launch the app
streamlit run app/streamlit_app.py
```

---

## Insights
- Excessive punctuation, emotional exaggeration, or ALL-CAPS usage strongly correlates with fake reviews.  
- Real reviews tend to include neutral tone and product-specific feedback.  
- The combination of linguistic + behavioral features improves reliability over text-only models.

---

## Future Improvements
- Integrate a larger, real-world labeled dataset.  
- Replace TF-IDF with contextual embeddings (BERT/SentenceTransformer).  
- Deploy via Streamlit Cloud or Hugging Face Spaces.  
- Add explainability (SHAP/LIME) for feature-level insights.
