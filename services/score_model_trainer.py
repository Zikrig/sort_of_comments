import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Препроцессор
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return text

class ScoreModelTrainer:
    def __init__(self, dataset_path: str, output_dir: str = "models"):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def train(self, text_column="text", target_column="score", test_size=0.2, n_estimators=60, batch_size=20):
        df = pd.read_csv(self.dataset_path)
        X = df[text_column].astype(str).tolist()
        y = df[target_column].astype(float).tolist()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

        vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), preprocessor=clean_text)
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = RandomForestRegressor(n_estimators=0, warm_start=True, random_state=42)

        # Обучаем пакетами деревьев и выводим прогресс
        for i in range(batch_size, n_estimators+1, batch_size):
            model.n_estimators = i
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)
            mae = np.mean(np.abs(np.array(y_test) - y_pred))
            print(f"Обучено деревьев: {i}/{n_estimators}, MAE: {mae:.4f}")

        # Сохраняем
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(self.output_dir, f"score_model_{timestamp}")
        os.makedirs(save_path, exist_ok=True)
        joblib.dump(model, os.path.join(save_path, "model.pkl"))
        joblib.dump(vectorizer, os.path.join(save_path, "vectorizer.pkl"))

        print(f"Модель сохранена в {save_path}")
        return save_path
