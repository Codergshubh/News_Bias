!pip install --upgrade transformers

from google.colab import drive
drive.mount('/content/drive')

!pip install --quiet transformers datasets torch scikit-learn lime shap pandas matplotlib seaborn

import os, random, numpy as np, pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt, seaborn as sns

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

train_path = "/content/drive/MyDrive/news_bias_project/train_news_bias.csv"
test_path  = "/content/drive/MyDrive/news_bias_project/test_news_bias.csv"

train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)

print("Train shape:", train.shape)
print("Test shape:", test.shape)
display(train.head())
display(test.head())
print("Train label distribution:\n", train['label'].value_counts(normalize=True))

import re
def clean_text(s):
    s = str(s).strip()
    s = re.sub(r'\s+', ' ', s)
    return s

train['text'] = train['headline'].apply(clean_text)
test['text']  = test['headline'].apply(clean_text)

from transformers import RobertaTokenizerFast
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

import torch
class BiasDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.texts = df['text'].tolist()
        self.labels = df['label'].astype(int).tolist()
        self.tokenizer = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = BiasDataset(train, tokenizer)
test_ds  = BiasDataset(test, tokenizer)

from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import numpy as np
import sklearn

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2).to(DEVICE)
data_collator = DataCollatorWithPadding(tokenizer)

training_args = TrainingArguments(
    output_dir="/content/drive/MyDrive/news_bias_project/bias_model",
    eval_strategy="epoch",
    save_strategy="epoch", # Added this line to match eval_strategy
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    learning_rate=2e-5,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro", # Changed from 'eval_f1' to 'eval_f1_macro'
    logging_steps=10,
    seed=SEED
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1);
    f1 = sklearn.metrics.f1_score(labels, preds, average='macro', zero_division=0)
    acc = (preds==labels).mean()
    return {"accuracy": acc, "f1_macro": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model("/content/drive/MyDrive/news_bias_project/bias_model/best")
tokenizer.save_pretrained("/content/drive/MyDrive/news_bias_project/bias_model/best")

pred_out = trainer.predict(test_ds)
logits = pred_out.predictions
probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
preds = np.argmax(logits, axis=1)
y_true = test['label'].values

print(classification_report(y_true, preds, digits=4))
cm = confusion_matrix(y_true, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues'); plt.xlabel('Pred'); plt.ylabel('True'); plt.show()

out = test.copy()
out['pred'] = preds
out['prob_neutral'] = probs[:,0]
out['prob_biased']  = probs[:,1]
out_path = "/content/drive/MyDrive/news_bias_project/predictions.csv"
out.to_csv(out_path, index=False)
print("Saved predictions to:", out_path)

from lime.lime_text import LimeTextExplainer
explainer = LimeTextExplainer(class_names=['neutral','biased'])

def predict_proba(texts):
    enc = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='pt').to(DEVICE)
    with torch.no_grad():
        logits = model(**enc).logits.cpu().numpy()
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return probs

lime_results = []
for i in range(min(5, len(test))):
    text = test.iloc[i].text
    exp = explainer.explain_instance(text, predict_proba, num_features=6)
    print("TEXT:", text)
    print("LIME:", exp.as_list(), "\n")
    lime_results.append({"id": int(test.iloc[i].id), "text": text, "lime": exp.as_list()}) # Convert numpy.int64 to int

import json
with open("/content/drive/MyDrive/news_bias_project/lime_results.json", "w") as f:
    json.dump(lime_results, f, ensure_ascii=False, indent=2)
print("Saved LIME results to Drive.")

!zip -r /content/drive/MyDrive/news_bias_project/submission.zip /content/drive/MyDrive/news_bias_project
print("Zipped project to Drive.")