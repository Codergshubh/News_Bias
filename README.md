# 📰 News Bias Detection using RoBERTa

A transformer-based NLP project for detecting contextual bias in short news headlines using **RoBERTa-base**. The model is fine-tuned on a manually labeled dataset and evaluated for performance and interpretability.


## 📌 Overview

News headlines often contain subtle contextual bias through emotionally charged or ideologically suggestive wording.  
This project builds a binary classifier to detect such bias automatically.

**Classes:**
- `0` → Neutral  
- `1` → Biased  

The model captures deep semantic context rather than relying on simple keyword frequency.


## 🚀 Results

- **Accuracy:** 95%
- **Macro-F1 Score:** 0.95
- Balanced precision and recall across both classes
- Minimal overfitting through regularization
- LIME-based explanations confirm context-driven predictions

## 🛠 Tech Stack

- Python
- PyTorch
- HuggingFace Transformers
- Scikit-learn
- LIME (Explainability)
- Google Colab (GPU Training)


## 📂 Project Structure

├── code.py # Model training & evaluation script
├── train.csv # Training dataset (100 samples)
├── test.csv # Test dataset (20 samples)
└── README.md


## 📊 Dataset

- Total Samples: 120 headlines  
- Training Set: 100  
- Test Set: 20  
- Balanced distribution (Neutral / Biased)
