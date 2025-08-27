import pandas as pd
import joblib
import os
import string
import re
import math

# Препроцессор
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    if len(text) < 100:
        text *= 3
    return text

class ScorePredictor:
    def __init__(self, model_path: str, vectorizer_path: str):
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)

    def _custom_round(self, value: float) -> int:
        """Кастомное округление: до 0.75 вниз, от 0.75 вверх"""
        fractional = value - math.floor(value)
        if fractional < 0.75:
            return math.floor(value)
        else:
            return math.ceil(value)
        
    def predict_text(self, text: str) -> int:
        """Предсказание score для одной строки"""
        vec = self.vectorizer.transform([text])
        pred = self.model.predict(vec)
        # Ограничение диапазона 0-5
        # print(pred[0])
        pred_score = max(0, min(5, int(self._custom_round(pred[0]))))
        return pred_score

    def predict_dataframe(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        """Добавляет колонки score и pred_score, строки с расхождением наверх"""
        df = df.copy()
        # Предсказания
        texts = df[text_column].astype(str).tolist()
        vec = self.vectorizer.transform(texts)
        preds = [max(0, min(5, int(round(p)))) for p in self.model.predict(vec)]

        # Добавляем колонки score (если нет — создаём как NaN) и pred_score
        if "score" not in df.columns:
            df.insert(0, "score", pd.NA)
        df.insert(1, "pred_score", preds)

        # Сортировка: несовпадения наверх
        df["_mismatch"] = df["score"] != df["pred_score"]
        df = df.sort_values("_mismatch", ascending=False).drop(columns="_mismatch").reset_index(drop=True)

        return df
