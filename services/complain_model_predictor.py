import joblib
import pandas as pd
from typing import List, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

import string
import re

# Тот же препроцессор, что и при обучении
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return text

class TextModelPredictor:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.model: RandomForestClassifier = joblib.load(model_path)
        self.vectorizer: TfidfVectorizer = joblib.load(vectorizer_path)

    def predict_text(self, text: str) -> int:
        """Предсказание одного текста"""
        X_vec = self.vectorizer.transform([clean_text(text)])
        return int(self.model.predict(X_vec)[0])

    def predict_proba(self, text: str) -> float:
        """Вероятность класса 1"""
        X_vec = self.vectorizer.transform([clean_text(text)])
        return float(self.model.predict_proba(X_vec)[0][1])

    def predict_batch(self, texts: List[str]) -> List[int]:
        """Предсказание списка текстов"""
        texts_clean = [clean_text(t) for t in texts]
        X_vec = self.vectorizer.transform(texts_clean)
        return self.model.predict(X_vec).tolist()

    def predict_dataframe(self, df: pd.DataFrame, text_field: str, output_field: str = "pred") -> pd.DataFrame:
        """Предсказание для DataFrame, добавляет столбец с предсказанием"""
        df = df.copy()
        texts_clean = df[text_field].astype(str).map(clean_text).tolist()
        X_vec = self.vectorizer.transform(texts_clean)
        df[output_field] = self.model.predict(X_vec)
        return df
