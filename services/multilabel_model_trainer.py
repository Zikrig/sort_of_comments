import pandas as pd
import joblib
import os
import string
import re
from datetime import datetime
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Текстовый препроцессор
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return text

class MultiLabelTrainer:
    def __init__(self, dataset_path: str, output_dir: str = "models"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, text_column="text", label_columns: List[str] = None, test_size=0.2):
        df = pd.read_csv(self.dataset_path)
        if label_columns is None:
            label_columns = [c for c in df.columns if c.startswith("id_")]

        X = df[text_column].astype(str).tolist()
        y = df[label_columns].astype(int).values  # numpy array

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), preprocessor=clean_text)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        models = {}
        print("Начало обучения по классам...")
        for i, col in enumerate(label_columns):
            print(f"Обучаем класс {col} ({i+1}/{len(label_columns)})...")
            clf = RandomForestClassifier(n_estimators=200, random_state=42)#, class_weight="balanced")
            clf.fit(X_train_vec, y_train[:, i])
            models[col] = clf

            y_pred = clf.predict(X_test_vec)
            print(classification_report(y_test[:, i], y_pred, zero_division=0))

        # Сохраняем модель и векторизатор
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.output_dir, f"multilabel_{timestamp}")
        os.makedirs(save_path, exist_ok=True)

        joblib.dump(models, os.path.join(save_path, "models.pkl"))
        joblib.dump(vectorizer, os.path.join(save_path, "vectorizer.pkl"))
        print(f"Модели сохранены в {save_path}")
        return save_path
